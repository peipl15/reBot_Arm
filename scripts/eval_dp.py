#!/usr/bin/env python3
"""Real-arm evaluation of a trained LeRobot DiffusionPolicy.

Loop:
  1. Read C922 + D435 RGB frames
  2. Read follower joint state (7-dim) via ArmController
  3. Build observation dict matching the dataset features
  4. Call policy.select_action(obs) → 7-dim target
  5. Send target to follower via arm.move_joint (sign-flip + soft limits + watchdog all from our wrapper)

Run from the dp_maniskill venv (where lerobot + torch live):

  source ~/code/dp_maniskill/.venv/bin/activate
  export LD_LIBRARY_PATH=$HOME/.cache/motorbridge/libdeps
  python3 ~/code/rebot_arm/scripts/eval_dp.py \\
      --ckpt outputs/training_runs/dp_single_cam/checkpoints/last/pretrained_model \\
      --no-wrist

Keys (on the OpenCV preview window):
  SPACE     reset arm to home and start/stop an eval episode
  R         reset arm to home (without starting an episode)
  E or ESC  EMERGENCY STOP — immediately disable all motors and halt the
            episode. Re-arm with SPACE or R.
  Q         quit

Safety:
  - vlim 0.5 rad/s on arm joints, 2.0 rad/s on the gripper (same as record_demo.py)
  - Soft limits enforced by ArmController.move_joint (auto-clamp)
  - Torque watchdog every 1s — abort if any joint exceeds 0.80 * tmax_fw
  - Ctrl-C disables all motors in finally
  - E / ESC: instant disable_all() with no further motion until re-arm
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

if "libdeps" not in os.environ.get("LD_LIBRARY_PATH", ""):
    print("ERROR: export LD_LIBRARY_PATH=$HOME/.cache/motorbridge/libdeps:$LD_LIBRARY_PATH first")
    sys.exit(1)

# We run this script from the dp_maniskill venv (which has torch+lerobot)
# but the `arm` package is editable-installed only in the rebot_arm venv.
# Add the repo root so `from arm import ...` works regardless of venv.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

from arm import ArmController, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = REPO_ROOT / "configs" / "joint_config.yaml"

GRIPPER_JOINT = 7


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True,
                   help="path to trained policy directory or .safetensors file")
    p.add_argument("--policy-type", default="diffusion",
                   help="lerobot policy type — currently we only test diffusion")
    p.add_argument("--device", default="cuda",
                   help="torch device for inference (cuda / cpu)")
    p.add_argument("--c922-index", type=int, default=-1,
                   help="/dev/videoN index; -1 = auto-find C922 by device name")
    p.add_argument("--c922-width", type=int, default=640)
    p.add_argument("--c922-height", type=int, default=480)
    p.add_argument("--rs-width", type=int, default=640)
    p.add_argument("--rs-height", type=int, default=480)
    p.add_argument("--no-wrist", action="store_true",
                   help="don't feed the wrist camera to the policy "
                        "(for single-camera ablation)")
    p.add_argument("--rate", type=float, default=10.0,
                   help="control loop frequency (Hz). Lower = smoother motion "
                        "because each move_joint has time to complete before "
                        "the next target lands. Was 15 Hz, dropped to 10.")
    p.add_argument("--vlim", type=float, default=0.3,
                   help="arm joint velocity limit (rad/s). Lower = slower & "
                        "smoother. Was 0.5, dropped to 0.3.")
    p.add_argument("--gripper-vlim", type=float, default=2.0)
    p.add_argument("--smooth", type=float, default=0.6,
                   help="EMA factor for the policy output: "
                        "target_filt = smooth*target_prev + (1-smooth)*target_now. "
                        "0 = no smoothing (raw policy), 0.6 = decent damping. "
                        "Reduces stepping/jerkiness between DP replans.")
    p.add_argument("--torque-abort-ratio", type=float, default=0.80)
    p.add_argument("--home-pos", type=str, default="0,0,0,0,0,0,0",
                   help="comma-separated 7 joint positions (rad) that we treat "
                        "as 'home' for arm.move_joint resets")
    p.add_argument("--home-order", type=str, default="4,1,2,3,5,6,7",
                   help="order in which home_arm moves joints. Defaults to "
                        "j4 first because folding/unfolding j4 last can collide "
                        "with j2/j3 in some configurations.")
    p.add_argument("--task-prompt", default="pick_tape",
                   help="language prompt for VLA policies (SmolVLA / PI05). "
                        "Should match what the policy was fine-tuned on.")
    return p.parse_args()


def load_policy(ckpt_path: Path, device: str):
    """Load any LeRobot policy (DP / SmolVLA / PI0 / PI05 / PI0Fast) by
    inspecting the ckpt's config.json. Returns (policy, preproc, postproc).

    lerobot 0.5 standardised the pipeline: ALL policy types (DP included)
    factor out normalization into separate `policy_preprocessor.json` /
    `policy_postprocessor.json` files. The preproc normalizes state +
    images (and for VLAs also tokenizes the task string); the postproc
    UN-normalizes the action chunk back to physical joint space. Without
    postproc, select_action returns a value in [-1, 1] MIN_MAX space which
    looks like a small rad target on some joints (j2/j3 near home) but a
    far one on others (j1 with wide range) → asymmetric drift, exactly
    what we saw on first eval.

    For lerobot 0.3 ckpts (DP only) those files don't exist; we return
    None for preproc/postproc and the main loop skips them."""
    import json
    cfg = json.load(open(ckpt_path / "config.json"))
    ptype = cfg.get("type", "diffusion")
    print(f"Loading {ptype} from {ckpt_path}...")

    if ptype == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        policy = DiffusionPolicy.from_pretrained(str(ckpt_path))
        preproc = postproc = None
        # Fall through to the unified normalizer-detection block below.
    else:
        # VLA models need a PolicyPreprocessor (tokenizes task string into
        # observation.language.tokens, normalizes images, etc.).
        # IMPORTANT: import the policy module FIRST so the @ProcessorStepRegistry
        # decorators run and register the smolvla/pi0/pi05-specific processor
        # steps. Otherwise DataProcessorPipeline.from_pretrained throws
        # KeyError on the step name.
        # Also: alias `relative_actions_processor` → `delta_actions_processor`
        # so pi05 ckpts (which reference the former name) load on lerobot
        # 0.5.1 (which only registers the same class under the latter name).
        try:
            from lerobot.processor import ProcessorStepRegistry
            _reg = ProcessorStepRegistry._registry
            if ("relative_actions_processor" not in _reg
                    and "delta_actions_processor" in _reg):
                ProcessorStepRegistry.register("relative_actions_processor")(
                    _reg["delta_actions_processor"])
        except Exception:
            pass
        if ptype == "smolvla":
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.policies.smolvla import processor_smolvla  # registers steps
        elif ptype in ("pi0", "pi05"):
            from importlib import import_module
            mod = "pi0" if ptype == "pi0" else "pi05"
            import_module(f"lerobot.policies.{mod}.modeling_{mod}")
            try:
                import_module(f"lerobot.policies.{mod}.processor_{mod}")
            except ImportError:
                pass
            # pi05 patch for transformers 4.55+ SigLIP API change.
            if ptype == "pi05":
                from lerobot.policies.pi05.modeling_pi05 import PaliGemmaWithExpertModel
                if not getattr(PaliGemmaWithExpertModel.embed_image,
                               "_pi05_patched", False):
                    import torch as _torch
                    def _patched_embed(self, image):
                        out_dtype = image.dtype
                        if image.dtype != _torch.float32:
                            image = image.to(_torch.float32)
                        vision = self.paligemma.model.vision_tower(image)
                        feats = vision.last_hidden_state
                        feats = self.paligemma.model.multi_modal_projector(feats)
                        if feats.dtype != out_dtype:
                            feats = feats.to(out_dtype)
                        return feats
                    _patched_embed._pi05_patched = True
                    PaliGemmaWithExpertModel.embed_image = _patched_embed
                # transformers 4.55+ renamed create_causal_mask's `inputs_embeds`
                # kwarg to `input_embeds`. Wrap to accept both.
                import lerobot.policies.pi_gemma as _pi_gemma_mod
                _orig_create_causal = _pi_gemma_mod.create_causal_mask
                if not getattr(_orig_create_causal, "_pi05_patched", False):
                    def _shimmed_create_causal_mask(*args, **kwargs):
                        if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
                            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
                        return _orig_create_causal(*args, **kwargs)
                    _shimmed_create_causal_mask._pi05_patched = True
                    _pi_gemma_mod.create_causal_mask = _shimmed_create_causal_mask
        elif ptype in ("pi0fast", "pi0_fast"):
            try:
                from lerobot.policies.pi0_fast import modeling_pi0_fast as _
                try:
                    from lerobot.policies.pi0_fast import processor_pi0_fast as _
                except ImportError:
                    pass
            except ImportError:
                from lerobot.policies.pi0fast import modeling_pi0fast as _
        from lerobot.processor import DataProcessorPipeline
        preproc = DataProcessorPipeline.from_pretrained(
            str(ckpt_path), config_filename="policy_preprocessor.json"
        )
        postproc = DataProcessorPipeline.from_pretrained(
            str(ckpt_path), config_filename="policy_postprocessor.json"
        )
        # LoRA detection: adapter_config.json present → ckpt is just the
        # adapter (~36MB safetensors), need to load base HF model first
        # and then wrap with PEFT.
        is_lora = (ckpt_path / "adapter_config.json").exists()
        if ptype == "smolvla":
            base_repo = "lerobot/smolvla_base"
            PCls = SmolVLAPolicy
        elif ptype in ("pi0", "pi05"):
            mod_name = "pi0" if ptype == "pi0" else "pi05"
            from importlib import import_module
            m = import_module(f"lerobot.policies.{mod_name}.modeling_{mod_name}")
            PCls = getattr(m, "PI0Policy" if ptype == "pi0" else "PI05Policy")
            base_repo = "lerobot/pi0_base" if ptype == "pi0" else "lerobot/pi05_base"
        elif ptype in ("pi0fast", "pi0_fast"):
            try:
                from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy as PCls
            except ImportError:
                from lerobot.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy as PCls
            base_repo = "lerobot/pi0fast-base"
        else:
            raise ValueError(f"unknown policy type {ptype!r} in {ckpt_path}/config.json")

        if is_lora:
            print(f"  LoRA ckpt detected — loading base {base_repo} + adapter")
            # pi05 fp32 ~12 GB OOMs the local 12 GB RTX 3500 Ada. PI05Config
            # has a `dtype` field that accepts "bfloat16"; load base config,
            # mutate dtype, then load policy with that config. A40 has 48 GB
            # so this branch only triggers on small GPUs.
            import torch as _torch
            if (ptype == "pi05" and _torch.cuda.is_available()
                    and _torch.cuda.mem_get_info()[1] < 24 * 1024**3):
                from lerobot.configs.policies import PreTrainedConfig
                base_cfg = PreTrainedConfig.from_pretrained(base_repo)
                base_cfg.dtype = "bfloat16"
                print(f"  GPU only {_torch.cuda.mem_get_info()[1]/1024**3:.1f} GB; "
                      f"loading pi05 in bf16 via PI05Config.dtype")
                policy = PCls.from_pretrained(base_repo, config=base_cfg)
            else:
                policy = PCls.from_pretrained(base_repo)
            # PreTrainedPolicy.wrap_with_peft only builds a fresh adapter
            # from a PeftConfig — it doesn't load saved weights. To load
            # a saved adapter we use PEFT's own API.
            from peft import PeftModel
            policy = PeftModel.from_pretrained(policy, str(ckpt_path))
            # The base ckpt's `input_features` references pi05's original
            # training data names (base_0_rgb, left_wrist_0_rgb, ...).
            # Our trained ckpt's config.json has the actual feature names
            # (observation.images.third_person / wrist) — overwrite the
            # loaded policy's config with those so select_action /
            # _preprocess_images sees the right keys.
            from lerobot.configs.policies import PreTrainedConfig
            local_cfg = PreTrainedConfig.from_pretrained(str(ckpt_path))
            for obj in (policy,
                        getattr(policy, "base_model", None),
                        getattr(getattr(policy, "base_model", None), "model", None)):
                if obj is None or not hasattr(obj, "config"):
                    continue
                try:
                    obj.config.input_features = local_cfg.input_features
                    obj.config.output_features = local_cfg.output_features
                except (AttributeError, TypeError):
                    pass
        else:
            policy = PCls.from_pretrained(str(ckpt_path))
    # lerobot 0.5 ckpts ALWAYS have policy_preprocessor.json / policy_postprocessor.json,
    # even for DP. If preproc/postproc weren't set in the type-specific branch
    # above (DP path), load them now if the files are present.
    pp_pre = ckpt_path / "policy_preprocessor.json"
    pp_post = ckpt_path / "policy_postprocessor.json"
    if preproc is None and pp_pre.exists():
        from lerobot.processor import DataProcessorPipeline
        preproc = DataProcessorPipeline.from_pretrained(
            str(ckpt_path), config_filename="policy_preprocessor.json"
        )
        print(f"  loaded preproc from {pp_pre.name} (lerobot 0.5 ckpt)")
    if postproc is None and pp_post.exists():
        from lerobot.processor import DataProcessorPipeline
        postproc = DataProcessorPipeline.from_pretrained(
            str(ckpt_path), config_filename="policy_postprocessor.json"
        )
        print(f"  loaded postproc from {pp_post.name}")

    policy.to(device)
    policy.eval()
    return policy, preproc, postproc


def prep_image_for_policy(bgr_frame: np.ndarray, target_h: int, target_w: int,
                          device: str) -> torch.Tensor:
    """BGR uint8 (H,W,3) → torch float32 (1, 3, H, W) in [0,1] on device."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (target_h, target_w):
        rgb = cv2.resize(rgb, (target_w, target_h))
    t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    return t.unsqueeze(0).to(device)


def home_arm(arm: ArmController, target_pos: list[float], vlim: float,
             order: list[int] | None = None):
    """Move 7 joints to target_pos in the given order. Tracking watchdog off."""
    if order is None:
        order = list(range(1, 8))
    print(f"  homing follower to {target_pos} (order={order})...")
    for jidx in order:
        cur = arm.read_joint(jidx, settle_s=0.05)
        if cur is None:
            continue
        dist = abs(target_pos[jidx - 1] - cur["pos_user"])
        if dist < 0.02:
            continue
        hold = dist / vlim + 1.0
        try:
            arm.move_joint(jidx, target_pos[jidx - 1], vlim=vlim,
                           hold_s=hold, check_tracking_error=False)
        except Exception as e:
            print(f"    j{jidx} home error: {e}")
    print("  ...home complete")


def build_display(c922, rs_rgb, episode_active, t_in_episode, action, fps,
                  emergency_stop=False):
    tiles = []
    if c922 is not None:
        tiles.append(cv2.resize(c922, (480, 360)))
    if rs_rgb is not None:
        tiles.append(cv2.resize(rs_rgb, (480, 360)))
    if not tiles:
        return np.zeros((360, 480, 3), dtype=np.uint8)
    display = np.hstack(tiles)
    if emergency_stop:
        # Red overlay banner across the top half
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], 200),
                      (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)
        cv2.putText(display, "EMERGENCY STOP",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
        cv2.putText(display, "motors disabled - press SPACE or R to re-arm",
                    (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
    else:
        color = (0, 0, 255) if episode_active else (200, 200, 200)
        label = (f"EVAL  t={t_in_episode:5.2f}s" if episode_active
                 else "IDLE  SPACE=start  R=home  E/ESC=STOP  Q=quit")
        cv2.putText(display, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, f"fps {fps:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)
        if action is not None:
            act_str = "  ".join(f"{a:+.2f}" for a in action[:7])
            cv2.putText(display, f"action: {act_str}",
                        (10, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return display


def main():
    args = parse_args()
    cfg = load_config(CFG_PATH)
    home_pos = [float(x) for x in args.home_pos.split(",")]
    if len(home_pos) != 7:
        print(f"ERROR: --home-pos must have 7 floats, got {len(home_pos)}")
        sys.exit(1)
    home_order = [int(x) for x in args.home_order.split(",")]
    if sorted(home_order) != list(range(1, 8)):
        print(f"ERROR: --home-order must be a permutation of 1..7, got {home_order}")
        sys.exit(1)

    # 1) Policy
    policy, preproc, postproc = load_policy(Path(args.ckpt), args.device)
    if preproc is not None:
        # VLA preprocessor needs a task string per sample. Use the dataset
        # task name (we recorded with "pick_tape"; this is the same prompt
        # the policy saw during fine-tuning).
        print(f"  preproc loaded; task prompt = {args.task_prompt!r}")
        print(f"  postproc loaded ({type(postproc).__name__}) — will unnormalize action")

    # The policy's input feature shapes are baked into the checkpoint; we
    # match the resolution we feed it to the dataset that produced the ckpt.
    # For now we trust the user/dataset to have used (480, 640) C922 + (480, 640) wrist.
    img_h, img_w = args.c922_height, args.c922_width

    # 2) Cameras
    if args.c922_index < 0:
        from camera_utils import find_camera_index
        idx = find_camera_index("C922")
        if idx is None:
            raise RuntimeError("C922 not found by name (is it plugged in?)")
        args.c922_index = idx
        print(f"Auto-detected C922 at /dev/video{idx}")
    print(f"\nOpening C922 on /dev/video{args.c922_index}...")
    cap = cv2.VideoCapture(args.c922_index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.c922_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.c922_height)
    if not cap.isOpened():
        raise RuntimeError(f"could not open C922 (/dev/video{args.c922_index})")

    print("Opening RealSense D435 (with hardware_reset)...")
    ctx = rs.context()
    for d in ctx.query_devices():
        d.hardware_reset()
    time.sleep(3.0)
    rs_pipe = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_stream(rs.stream.color, args.rs_width, args.rs_height, rs.format.bgr8, 30)
    rs_cfg.enable_stream(rs.stream.depth, args.rs_width, args.rs_height, rs.format.z16, 30)
    rs_pipe.start(rs_cfg)
    for _ in range(10):
        rs_pipe.wait_for_frames(timeout_ms=2000)

    # 3) Follower
    print("Opening follower...")
    arm = ArmController(cfg)
    arm.open()
    arm.enable_all()
    time.sleep(0.3)

    print("\n=== READY ===")
    print("  SPACE = home arm + start eval episode (press again to stop)")
    print("  R     = home arm to --home-pos (does NOT start an episode)")
    print("  Q     = quit")
    print()

    interval = 1.0 / args.rate
    next_tick = time.monotonic()
    episode_active = False
    emergency_stop = False
    episode_start_t = 0.0
    fps_window = []
    follower_motors = [arm._motors[i] for i in range(1, 8)]
    follower_signs = [cfg.joint(i).sign for i in range(1, 8)]
    last_action = None
    # EMA-filtered target sent to the motors. Initialised on first action.
    smoothed_target = None

    # Terminal-side emergency stop: SIGINT handler that flips the flag and
    # disables all motors immediately. We still let the main loop wind down
    # cleanly so the cv2 window can be dismissed.
    import signal
    def _sigint(_sig, _frame):
        nonlocal emergency_stop, episode_active
        if not emergency_stop:
            print("\n[SIGINT] EMERGENCY STOP — disabling all motors")
            try:
                arm.disable_all()
            except Exception as e:
                print(f"  disable_all error: {e}")
            emergency_stop = True
            episode_active = False
        else:
            print("\n[SIGINT 2x] forcing exit")
            raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _sigint)

    try:
        while True:
            now = time.monotonic()
            wall_t = time.time()

            # Read sensors
            ok, c922_frame = cap.read()
            if not ok:
                c922_frame = None
            try:
                fs = rs_pipe.wait_for_frames(timeout_ms=100)
                color = fs.get_color_frame()
                rs_rgb = np.asanyarray(color.get_data()) if color else None
            except Exception:
                rs_rgb = None

            # Follower state
            state_arr = []
            for m, sgn in zip(follower_motors, follower_signs):
                s = m.get_state()
                state_arr.append(s.pos * sgn if s else 0.0)
            state_t = torch.tensor(state_arr, dtype=torch.float32).unsqueeze(0).to(args.device)

            if episode_active and not emergency_stop and c922_frame is not None and (rs_rgb is not None or args.no_wrist):
                # Build observation for the policy
                third_t = prep_image_for_policy(c922_frame, img_h, img_w, args.device)
                obs = {
                    "observation.state": state_t,
                    "observation.images.third_person": third_t,
                }
                if not args.no_wrist:
                    wrist_t = prep_image_for_policy(rs_rgb, img_h, img_w, args.device)
                    obs["observation.images.wrist"] = wrist_t

                # VLA preprocessor: needs "task" key + may run tokenizer
                if preproc is not None:
                    obs["task"] = [args.task_prompt]
                    obs = preproc(obs)

                # Inference
                with torch.inference_mode():
                    action_tensor = policy.select_action(obs)

                # VLA returns NORMALIZED action; postproc unnormalizes back to
                # physical joint angles. Without this, j1 etc. wander far.
                if postproc is not None:
                    out = postproc({"action": action_tensor})
                    action_tensor = out["action"] if isinstance(out, dict) else out

                action = action_tensor.squeeze(0).cpu().numpy().tolist()
                last_action = action

                # EMA smoothing: target_filt = a*prev + (1-a)*new
                # Reduces stepping between DP replans → arm moves more
                # continuously and overshoots less.
                a = args.smooth
                if smoothed_target is None or a <= 0.0:
                    smoothed_target = list(action[:7])
                else:
                    smoothed_target = [a * smoothed_target[i] + (1 - a) * float(action[i])
                                       for i in range(7)]

                # Send to follower (sign-flip + soft limits + watchdog auto)
                for jidx in range(1, 8):
                    target = float(smoothed_target[jidx - 1])
                    vlim = args.gripper_vlim if jidx == GRIPPER_JOINT else args.vlim
                    arm.move_joint(jidx, target, vlim=vlim, hold_s=0.0)

            # Display
            fps_window.append(time.monotonic())
            if len(fps_window) > 30:
                fps_window.pop(0)
            fps = (len(fps_window) - 1) / (fps_window[-1] - fps_window[0]) \
                if len(fps_window) >= 2 else 0.0
            t_in_ep = (wall_t - episode_start_t) if episode_active else 0.0
            display = build_display(c922_frame, rs_rgb, episode_active, t_in_ep,
                                    last_action, fps, emergency_stop=emergency_stop)
            cv2.imshow("eval_dp", display)
            key = cv2.waitKey(1) & 0xFF

            # Emergency stop: E or ESC (key 27). Do this FIRST so it takes
            # priority over every other action.
            if key == ord("e") or key == ord("E") or key == 27:
                if not emergency_stop:
                    print(f"\n[EMERGENCY STOP] disabling all motors immediately")
                    try:
                        arm.disable_all()
                    except Exception as e:
                        print(f"  disable_all error: {e}")
                    emergency_stop = True
                    episode_active = False
                    last_action = None
            elif key == ord(" "):
                if emergency_stop:
                    # Re-arm: re-enable motors before starting a new episode
                    print(f"[RE-ARM] re-enabling motors after E-stop")
                    try:
                        arm.enable_all()
                        time.sleep(0.3)
                        emergency_stop = False
                    except Exception as e:
                        print(f"  enable_all error: {e}")
                        continue
                if not episode_active:
                    home_arm(arm, home_pos, args.vlim, order=home_order)
                    policy.reset()
                    episode_active = True
                    episode_start_t = wall_t
                    last_action = None
                    smoothed_target = None  # clear EMA so first action lands raw
                    print(f"[EVAL START]  episode t=0")
                else:
                    episode_active = False
                    print(f"[EVAL STOP]   episode lasted {t_in_ep:.1f}s")
            elif key == ord("r") or key == ord("R"):
                if emergency_stop:
                    print(f"[RE-ARM] re-enabling motors after E-stop")
                    try:
                        arm.enable_all()
                        time.sleep(0.3)
                        emergency_stop = False
                    except Exception as e:
                        print(f"  enable_all error: {e}")
                        continue
                if episode_active:
                    print("ignoring R while episode active — press SPACE first")
                else:
                    home_arm(arm, home_pos, args.vlim, order=home_order)
            elif key == ord("q") or key == ord("Q"):
                break

            next_tick += interval
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()

    except KeyboardInterrupt:
        print("\n[Ctrl-C]")
    finally:
        print("Disabling follower + closing cameras...")
        try:
            arm.disable_all()
        except Exception:
            pass
        arm.close()
        try:
            rs_pipe.stop()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
