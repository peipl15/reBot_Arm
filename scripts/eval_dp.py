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
  SPACE  reset arm to home and start/stop an eval episode
  R      reset arm to home (without starting an episode)
  Q      quit

Safety:
  - vlim 0.5 rad/s on arm joints, 2.0 rad/s on the gripper (same as record_demo.py)
  - Soft limits enforced by ArmController.move_joint (auto-clamp)
  - Torque watchdog every 1s — abort if any joint exceeds 0.80 * tmax_fw
  - Ctrl-C disables all motors in finally
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
    p.add_argument("--c922-index", type=int, default=8)
    p.add_argument("--c922-width", type=int, default=640)
    p.add_argument("--c922-height", type=int, default=480)
    p.add_argument("--rs-width", type=int, default=640)
    p.add_argument("--rs-height", type=int, default=480)
    p.add_argument("--no-wrist", action="store_true",
                   help="don't feed the wrist camera to the policy "
                        "(for single-camera ablation)")
    p.add_argument("--rate", type=float, default=15.0,
                   help="control loop frequency (Hz). DP inference is slower "
                        "than teleop, so 15 Hz is a sane default.")
    p.add_argument("--vlim", type=float, default=0.5)
    p.add_argument("--gripper-vlim", type=float, default=2.0)
    p.add_argument("--torque-abort-ratio", type=float, default=0.80)
    p.add_argument("--home-pos", type=str, default="0,0,0,0,0,0,0",
                   help="comma-separated 7 joint positions (rad) that we treat "
                        "as 'home' for arm.move_joint resets")
    return p.parse_args()


def load_policy(ckpt_path: Path, device: str):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    print(f"Loading DiffusionPolicy from {ckpt_path}...")
    policy = DiffusionPolicy.from_pretrained(str(ckpt_path))
    policy.to(device)
    policy.eval()
    return policy


def prep_image_for_policy(bgr_frame: np.ndarray, target_h: int, target_w: int,
                          device: str) -> torch.Tensor:
    """BGR uint8 (H,W,3) → torch float32 (1, 3, H, W) in [0,1] on device."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (target_h, target_w):
        rgb = cv2.resize(rgb, (target_w, target_h))
    t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    return t.unsqueeze(0).to(device)


def home_arm(arm: ArmController, target_pos: list[float], vlim: float):
    """Move all 7 joints to target_pos. Long traversal: tracking watchdog off."""
    print(f"  homing follower to {target_pos}...")
    for jidx in range(1, 8):
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


def build_display(c922, rs_rgb, episode_active, t_in_episode, action, fps):
    tiles = []
    if c922 is not None:
        tiles.append(cv2.resize(c922, (480, 360)))
    if rs_rgb is not None:
        tiles.append(cv2.resize(rs_rgb, (480, 360)))
    if not tiles:
        return np.zeros((360, 480, 3), dtype=np.uint8)
    display = np.hstack(tiles)
    color = (0, 0, 255) if episode_active else (200, 200, 200)
    label = (f"EVAL  t={t_in_episode:5.2f}s" if episode_active
             else "IDLE  (SPACE=start, R=reset home, Q=quit)")
    cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(display, f"fps {fps:.1f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
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

    # 1) Policy
    policy = load_policy(Path(args.ckpt), args.device)

    # The policy's input feature shapes are baked into the checkpoint; we
    # match the resolution we feed it to the dataset that produced the ckpt.
    # For now we trust the user/dataset to have used (480, 640) C922 + (480, 640) wrist.
    img_h, img_w = args.c922_height, args.c922_width

    # 2) Cameras
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
    episode_start_t = 0.0
    fps_window = []
    follower_motors = [arm._motors[i] for i in range(1, 8)]
    follower_signs = [cfg.joint(i).sign for i in range(1, 8)]
    last_action = None

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

            if episode_active and c922_frame is not None and (rs_rgb is not None or args.no_wrist):
                # Build observation for the policy
                third_t = prep_image_for_policy(c922_frame, img_h, img_w, args.device)
                obs = {
                    "observation.state": state_t,
                    "observation.images.third_person": third_t,
                }
                if not args.no_wrist:
                    wrist_t = prep_image_for_policy(rs_rgb, img_h, img_w, args.device)
                    obs["observation.images.wrist"] = wrist_t

                # Inference
                with torch.inference_mode():
                    action_tensor = policy.select_action(obs)
                action = action_tensor.squeeze(0).cpu().numpy().tolist()
                last_action = action

                # Send to follower (sign-flip + soft limits + watchdog auto)
                for jidx in range(1, 8):
                    target = float(action[jidx - 1])
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
                                    last_action, fps)
            cv2.imshow("eval_dp", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if not episode_active:
                    home_arm(arm, home_pos, args.vlim)
                    policy.reset()
                    episode_active = True
                    episode_start_t = wall_t
                    last_action = None
                    print(f"[EVAL START]  episode t=0")
                else:
                    episode_active = False
                    print(f"[EVAL STOP]   episode lasted {t_in_ep:.1f}s")
            elif key == ord("r") or key == ord("R"):
                if episode_active:
                    print("ignoring R while episode active — press SPACE first")
                else:
                    home_arm(arm, home_pos, args.vlim)
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
