#!/usr/bin/env python3
"""Visualize where SmolVLA / DP / pi0 is "looking at" during select_action.

Uses input-gradient saliency: take ∂||action||/∂image_pixels and overlay
the gradient magnitude as a heatmap on the original RGB. High values =
pixels the model actually used to decide the action; low values = pixels
it ignored. If the model is doing its job, the heatmap should concentrate
on the target object (the black tape) and the gripper, not random
background.

Sources of frames:
  --episode <HDF5>     read a frame from a recorded demo
  --frame-idx N        which frame within the HDF5 (default 10)
  --episode-range a,b  visualize a..b range (composes a single grid PNG)

For real-time use during eval_dp.py, see the --record-frames flag there
(planned). This script is for offline inspection of any saved trajectory.

Examples:
  # one frame
  .venv_le05/bin/python scripts/visualize_attention.py \\
      --ckpt outputs/training_runs/test2_SMOLVLA/checkpoints/030000/pretrained_model \\
      --episode outputs/demos/pick_tape/episode_000005.hdf5 \\
      --frame-idx 60 \\
      --output outputs/saliency_smolvla_ep5_t60.png

  # grid of evenly-spaced frames across one episode
  .venv_le05/bin/python scripts/visualize_attention.py \\
      --ckpt outputs/training_runs/test2_SMOLVLA/checkpoints/030000/pretrained_model \\
      --episode outputs/demos/pick_tape/episode_000005.hdf5 \\
      --grid 5 \\
      --output outputs/saliency_smolvla_ep5_grid.png

The heatmap is informative when it covers a small bright region around the
tape / silver box. Noisy uniform red = model isn't really "looking" at
anything → suggests perception isn't grounded.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_policy(ckpt_path: Path, device: str):
    spec = importlib.util.spec_from_file_location(
        "eval_dp", str(REPO_ROOT / "scripts" / "eval_dp.py"))
    ed = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(ed)
    except SystemExit:
        pass
    return ed.load_policy(ckpt_path, device)


def _to_tensor_rgb(bgr: np.ndarray, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return t.to(device)


def saliency(policy, preproc, postproc, third_bgr, wrist_bgr, state, task_prompt, device):
    """Run select_action with gradient tracking; return per-pixel saliency
    for the two image inputs and the action vector. Saliency = mean over RGB
    channels of |gradient|."""
    third_t = _to_tensor_rgb(third_bgr, device).requires_grad_(True)
    wrist_t = _to_tensor_rgb(wrist_bgr, device).requires_grad_(True)

    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    obs = {
        "observation.state": state_t,
        "observation.images.third_person": third_t,
        "observation.images.wrist": wrist_t,
        "task": [task_prompt],
    }
    if preproc is not None:
        obs = preproc(obs)

    policy.reset()
    # select_action is decorated with @torch.no_grad() in lerobot 0.5, which
    # blocks gradient flow. Force-enable autograd inside, and unwrap any
    # parameters that may have been moved off the grad graph.
    third_t.retain_grad()
    wrist_t.retain_grad()
    with torch.enable_grad():
        action = policy.select_action(obs)
    if postproc is not None:
        out = postproc({"action": action})
        action = out["action"] if isinstance(out, dict) else out

    if action.grad_fn is None:
        # Fallback for DP/SmolVLA: select_action breaks the grad chain.
        # Use forward() with a zero action target so the loss = MSE(pred, 0).
        # DP also expects time dimension on observations (n_obs_steps=2),
        # so tile the single frame to satisfy that assertion.
        chunk_size = getattr(policy.config, "chunk_size",
                             getattr(policy.config, "horizon", 8))
        n_obs = getattr(policy.config, "n_obs_steps", 1)
        action_dim = 7
        zero_action = torch.zeros((1, chunk_size, action_dim), device=device)
        obs2 = dict(obs)
        # Tile state and images to T=n_obs_steps. Images shape becomes
        # (B, T, C, H, W); state becomes (B, T, D).
        # We retain grad through the tiles by reconstructing from the
        # leaf third_t / wrist_t tensors with .repeat or .expand.
        def _tile_t(x):
            if x.dim() == 4:   # (B, C, H, W) -> (B, T, C, H, W)
                return x.unsqueeze(1).expand(-1, n_obs, -1, -1, -1)
            if x.dim() == 2:   # (B, D) -> (B, T, D)
                return x.unsqueeze(1).expand(-1, n_obs, -1)
            return x
        for k in list(obs2.keys()):
            if k.startswith("observation.images.") or k == "observation.state":
                obs2[k] = _tile_t(obs2[k])
        obs2["action"] = zero_action
        obs2["action_is_pad"] = torch.zeros((1, chunk_size), dtype=torch.bool,
                                            device=device)
        with torch.enable_grad():
            out = policy.forward(obs2)
        loss = (out["loss"] if isinstance(out, dict)
                else (out[0] if isinstance(out, tuple) else out))
        target = loss
    else:
        # |action|_1 — uniform attribution across all 7 joints. Could weight
        # by joint magnitude or pick a single dim (e.g. j7 for grasp attention).
        target = action.abs().sum()
    target.backward()

    third_sal = third_t.grad.abs().mean(dim=1).squeeze(0).cpu().numpy()
    wrist_sal = (wrist_t.grad.abs().mean(dim=1).squeeze(0).cpu().numpy()
                 if wrist_t.grad is not None else None)
    return third_sal, wrist_sal, action.detach().squeeze().cpu().numpy()


def overlay_heatmap(rgb: np.ndarray, heatmap: np.ndarray | None,
                    blur_ksize: int = 25) -> np.ndarray:
    if heatmap is None:
        return rgb.copy()
    # Smooth to suppress pixel-level noise, then normalize to [0, 255].
    h = cv2.GaussianBlur(heatmap, (blur_ksize, blur_ksize), 0)
    h = h - h.min()
    h = h / max(h.max(), 1e-9) * 255.0
    h = h.astype(np.uint8)
    cmap = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    cmap_rgb = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 0.55, cmap_rgb, 0.45, 0)


def load_frame(hdf: h5py.File, idx: int):
    third_buf = np.frombuffer(hdf["third_person"][idx], dtype=np.uint8)
    wrist_buf = np.frombuffer(hdf["wrist_rgb"][idx], dtype=np.uint8)
    return (cv2.imdecode(third_buf, cv2.IMREAD_COLOR),
            cv2.imdecode(wrist_buf, cv2.IMREAD_COLOR),
            hdf["follower_pos"][idx])


def make_panel(third_bgr, wrist_bgr, third_sal, wrist_sal, action, label: str):
    """Build a single 2-row × 2-col panel: top row = raw RGB, bottom row =
    saliency overlay. Returns RGB image."""
    third_rgb = cv2.cvtColor(third_bgr, cv2.COLOR_BGR2RGB)
    wrist_rgb = cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB)
    third_overlay = overlay_heatmap(third_rgb, third_sal)
    wrist_overlay = overlay_heatmap(wrist_rgb, wrist_sal) if wrist_sal is not None else wrist_rgb

    gap = 6
    H, W = third_rgb.shape[:2]
    spacer_h = np.zeros((H, gap, 3), dtype=np.uint8)
    spacer_v = np.zeros((gap, W * 2 + gap, 3), dtype=np.uint8)
    top = np.hstack([third_rgb, spacer_h, wrist_rgb])
    bot = np.hstack([third_overlay, spacer_h, wrist_overlay])
    panel = np.vstack([top, spacer_v, bot])

    # label band on top
    band = np.zeros((40, panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(band, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    act_str = "action: " + " ".join(f"{a:+.2f}" for a in action[:7])
    cv2.putText(panel, act_str, (10, panel.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return np.vstack([band, panel])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episode", required=True,
                   help="HDF5 demo to read frames from")
    p.add_argument("--frame-idx", type=int, default=None,
                   help="single frame index (overrides --grid)")
    p.add_argument("--grid", type=int, default=5,
                   help="if --frame-idx not given, sample N evenly-spaced "
                        "frames and stack them vertically (default 5)")
    p.add_argument("--task-prompt", default="pick_tape")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print(f"Loading policy from {args.ckpt}...")
    policy, preproc, postproc = load_policy(Path(args.ckpt), args.device)
    print(f"  preproc: {type(preproc).__name__ if preproc else None}")
    print(f"  postproc: {type(postproc).__name__ if postproc else None}")

    with h5py.File(args.episode, "r") as f:
        n = int(f.attrs["n_frames"])
        if args.frame_idx is not None:
            indices = [args.frame_idx]
        else:
            indices = [int(i) for i in np.linspace(0, n - 1, args.grid)]
        print(f"  episode {Path(args.episode).name}: n_frames={n}, indices={indices}")

        panels = []
        for i in indices:
            third_bgr, wrist_bgr, state = load_frame(f, i)
            third_sal, wrist_sal, action = saliency(
                policy, preproc, postproc, third_bgr, wrist_bgr,
                state, args.task_prompt, args.device)
            label = f"frame {i}/{n}  ({i/n*100:.0f}%)"
            panels.append(make_panel(third_bgr, wrist_bgr,
                                     third_sal, wrist_sal, action, label))
            wmax = wrist_sal.max() if wrist_sal is not None else 0.0
            print(f"  frame {i}: third_max={third_sal.max():.4f}  "
                  f"wrist_max={wmax:.4f}  "
                  f"action_l1={np.abs(action).sum():.3f}")

    spacer = np.zeros((10, panels[0].shape[1], 3), dtype=np.uint8)
    out = panels[0]
    for pa in panels[1:]:
        out = np.vstack([out, spacer, pa])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"\nwrote {args.output}  ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
