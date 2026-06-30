#!/usr/bin/env python3
"""Post-hoc relabel of pick_tape demos with a `grasp_event_frame` attribute.

EXPERIMENTS.md Test #1, finding D: 0/51 episodes recorded any contact-event
signal even though all 51 were successful grasps. The torque-based detector
in record_demo.py was only used for the firmware-side overpress lock, never
written to disk. The downstream policy has no idea when "I have grasped"
happens — it has to infer it from gripper position alone, which is
degenerate at the grasp boundary.

Detection heuristic (no model, just data):
  A frame `t` is the grasp event iff
    (a) the gripper action target at t is commanding further close
        (i.e. the leader is still asking for more closure), AND
    (b) the follower gripper position has stalled —
        |follower_pos[t+1..t+stall_win, 6] - follower_pos[t, 6]| < stall_eps
        for `stall_win` consecutive frames, AND
    (c) the gripper torque is above a contact threshold at one of those
        frames (loose; we don't want to count "open then idle" as a grasp).

This is conservative: we require evidence from action, kinematics, AND
torque. If none of the three fires, the episode is reported without a
grasp event and we surface it to the user — most likely an actual failed
grasp that snuck into the dataset.

Usage:
  python3 scripts/relabel_grasp.py --task pick_tape
  python3 scripts/relabel_grasp.py --task pick_tape --dry-run   # no writes
  python3 scripts/relabel_grasp.py --task pick_tape --limit 5   # first N only

Each HDF5 gets new attributes:
  grasp_event_frame: int (or -1 if not detected)
  grasp_event_torque: float (the j7 torque at the event frame)
  grasp_relabel_version: str (so a future re-run can tell if labels are stale)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# v2: recomputed on DM-scale-corrected torque (scripts/fix_dm_scale.py).
# v1 ran on j4-j7 torque that was 2.8x inflated (DM-J4340 mislabel); its 0.5 Nm
# threshold was therefore an effective ~0.18 Nm in true units. v2 applies that
# same true-unit threshold (--torque-threshold 0.18) uniformly to all episodes.
RELABEL_VERSION = "v2"

# Leader gripper sits at column 6 in `leader_deg` (SERVO_IDS=[0..6], gripper
# maps via JOINT_MAP[6]). Follower gripper sits at column 6 in
# `follower_pos / action / follower_torq` (joints stored in joint-index order
# j1..j7, with j7 being the gripper).
LEADER_GRIPPER_COL = 6
FOLLOWER_GRIPPER_COL = 6


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="pick_tape")
    p.add_argument("--demos-dir", default=None,
                   help="default outputs/demos/<task>/")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="report only; do not write back to HDF5")

    # Detection knobs. Defaults tuned for our 20 Hz / 1.0 Nm contact regime.
    p.add_argument("--stall-win", type=int, default=10,
                   help="# of consecutive frames the follower gripper must be stalled (default 10 = 0.5 s @ 20 Hz).")
    p.add_argument("--stall-eps", type=float, default=0.005,
                   help="rad. Follower gripper movement smaller than this counts as stalled (0.005 rad ≈ 0.3°).")
    p.add_argument("--commanded-close-eps", type=float, default=0.005,
                   help="rad. The action target must be more negative than current follower pos by at least this (gripper closes in -rad direction).")
    p.add_argument("--torque-threshold", type=float, default=0.15,
                   help="Nm. Min |follower_torq[j7]| at the stall window to call it "
                        "'real contact'. v2 default 0.15: tape grasps are light "
                        "(median contact ~0.20 Nm, noise floor ~0.08 after the "
                        "DM-J4310 scale correction). 0.15 catches the canonical "
                        "ep0 (true grasp torque 0.164 Nm) and sits ~2x the floor.")
    p.add_argument("--min-ratio", type=float, default=0.10,
                   help="reject detections in the first N%% of the episode "
                        "(grasp can't physically happen in the approach phase; "
                        "0.10 = grasp must be after frame n_frames*0.10).")
    p.add_argument("--mode", default="auto",
                   choices=["stall", "force", "auto"],
                   help="Detection mode:\n"
                        "  stall: position stalls for stall_win frames with tau>thresh (original v1/v2)\n"
                        "  force: sustained torque >= force_threshold for force_win frames\n"
                        "         (works on short recovery demos that don't have a hold tail)\n"
                        "  auto:  try stall first, fall back to force if not detected.")
    p.add_argument("--force-threshold", type=float, default=0.30,
                   help="Nm. For mode=force: minimum |tau_j7| during the sustained window.")
    p.add_argument("--force-win", type=int, default=10,
                   help="# of consecutive frames tau must stay >= force_threshold "
                        "(default 10 = 0.5s @ 20Hz).")
    return p.parse_args()


def _detect_stall(action, follower_pos, follower_torq, args):
    """Original v1/v2 detector: position stall + commanded_close + tau threshold.
    Works well on full demos that have a hold tail after grasp."""
    n = len(action)
    g_act = action[:, FOLLOWER_GRIPPER_COL]
    g_pos = follower_pos[:, FOLLOWER_GRIPPER_COL]
    g_tau = follower_torq[:, FOLLOWER_GRIPPER_COL] if follower_torq is not None else np.zeros(n)

    commanded_close = (g_pos - g_act) > args.commanded_close_eps
    t_min = int(n * args.min_ratio)
    for t in range(t_min, n - args.stall_win):
        if not commanded_close[t]:
            continue
        window = g_pos[t : t + args.stall_win]
        if np.max(np.abs(window - g_pos[t])) > args.stall_eps:
            continue
        tau_max = float(np.max(np.abs(g_tau[t : t + args.stall_win])))
        if tau_max < args.torque_threshold:
            continue
        return t, tau_max
    return -1, 0.0


def _detect_force(action, follower_pos, follower_torq, args):
    """Force-plateau detector: tau stays above force_threshold for force_win
    consecutive frames. Works on short recovery demos that end right after
    contact (no hold tail). Triggers at the FIRST frame where the sustained
    high-torque region starts."""
    n = len(action)
    if follower_torq is None:
        return -1, 0.0
    g_tau = np.abs(follower_torq[:, FOLLOWER_GRIPPER_COL])
    t_min = int(n * args.min_ratio)
    for t in range(t_min, n - args.force_win):
        window = g_tau[t : t + args.force_win]
        if window.min() < args.force_threshold:
            continue
        return t, float(window.mean())
    return -1, 0.0


def detect_grasp(action, follower_pos, follower_torq, args):
    """Return (grasp_frame, tau_at_event) or (-1, 0.0).
    Dispatches based on --mode."""
    if args.mode == "stall":
        return _detect_stall(action, follower_pos, follower_torq, args)
    if args.mode == "force":
        return _detect_force(action, follower_pos, follower_torq, args)
    # auto: try stall first, fall back to force
    t, tau = _detect_stall(action, follower_pos, follower_torq, args)
    if t >= 0:
        return t, tau
    return _detect_force(action, follower_pos, follower_torq, args)


def process_one(fp: Path, args) -> dict:
    with h5py.File(fp, "r") as f:
        action = f["action"][:]
        follower_pos = f["follower_pos"][:]
        follower_torq = f["follower_torq"][:] if "follower_torq" in f else None
        n_frames = int(f.attrs["n_frames"])

    grasp_t, tau = detect_grasp(action, follower_pos, follower_torq, args)

    if not args.dry_run:
        with h5py.File(fp, "a") as f:
            f.attrs["grasp_event_frame"] = int(grasp_t)
            f.attrs["grasp_event_torque"] = float(tau)
            f.attrs["grasp_relabel_version"] = RELABEL_VERSION

    return {
        "episode": fp.name,
        "n_frames": n_frames,
        "grasp_frame": grasp_t,
        "tau_at_event": tau,
        "found": grasp_t >= 0,
        # frame_index/n_frames ratio — early grasp = good fast demo
        "ratio": (grasp_t / n_frames) if grasp_t >= 0 else None,
    }


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    demos_dir = Path(args.demos_dir) if args.demos_dir else (
        repo_root / "outputs" / "demos" / args.task
    )
    files = sorted(demos_dir.glob("episode_*.hdf5"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        print(f"no episodes found in {demos_dir}")
        sys.exit(1)

    print(f"Scanning {len(files)} episodes in {demos_dir} "
          f"(stall_win={args.stall_win}, stall_eps={args.stall_eps}, "
          f"torque_thresh={args.torque_threshold})...")
    if args.dry_run:
        print("  [DRY RUN] no writes will be made")

    results = []
    for fp in files:
        r = process_one(fp, args)
        results.append(r)
        if r["found"]:
            print(f"  ✓ {r['episode']:25s}  n={r['n_frames']:4d}  "
                  f"grasp@{r['grasp_frame']:4d}  ratio={r['ratio']:.2f}  "
                  f"tau={r['tau_at_event']:5.2f}")
        else:
            print(f"  ✗ {r['episode']:25s}  n={r['n_frames']:4d}  NO GRASP DETECTED")

    n_found = sum(r["found"] for r in results)
    print(f"\nDetected grasp in {n_found}/{len(results)} episodes.")
    if n_found < len(results):
        misses = [r['episode'] for r in results if not r['found']]
        print(f"No grasp found in: {misses}")
        print("Consider lowering --torque-threshold or --stall-eps, "
              "or trimming these from training data.")

    if n_found > 0:
        ratios = [r["ratio"] for r in results if r["found"]]
        print(f"grasp_frame / n_frames: median={np.median(ratios):.2f}  "
              f"p25={np.percentile(ratios,25):.2f}  p75={np.percentile(ratios,75):.2f}")


if __name__ == "__main__":
    main()
