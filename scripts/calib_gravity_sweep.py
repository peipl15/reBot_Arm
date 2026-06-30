#!/usr/bin/env python3
"""Collect quasi-static gravity-calibration data for the follower arm.

Slowly sweeps selected joints (default shoulder j2 + elbow j3 — they carry the
most gravity load) back and forth at constant low velocity, logging every
joint's angle / velocity / torque. Sweeping in BOTH directions lets the offline
fitter cancel Coulomb friction:  gravity ≈ (torque_+dir + torque_-dir)/2  at the
same angle, and friction ≈ (torque_+dir − torque_-dir)/2.

  >>> THIS MOVES THE ARM AUTONOMOUSLY. <<<
Before starting, put the arm in a safe, open pose (the NON-swept joints are held
wherever they are at start). Keep the workspace clear and a hand on the power
switch / Ctrl-C. Supervise the whole run — do not walk away.

Reads torque via ArmController (DM feedback). Make sure the j4–j7 model fix
(DM4310) is in configs/joint_config.yaml so torque is correctly scaled.

Usage:
  cd ~/code/rebot_arm && source .venv/bin/activate && \
  export LD_LIBRARY_PATH=$HOME/.cache/motorbridge/libdeps && \
  python3 scripts/calib_gravity_sweep.py                 # sweeps j2 then j3
  python3 scripts/calib_gravity_sweep.py --joints 2 --vlim 0.12 --span 0.6
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

if "libdeps" not in os.environ.get("LD_LIBRARY_PATH", ""):
    print("ERROR: export LD_LIBRARY_PATH=$HOME/.cache/motorbridge/libdeps first")
    sys.exit(1)

from arm import ArmController, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = REPO_ROOT / "configs" / "joint_config.yaml"
ALL_JOINTS = [1, 2, 3, 4, 5, 6, 7]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--joints", default="2,3,4,5",
                   help="comma list of joints to sweep. Default 2,3,4,5 = the "
                        "gravity-relevant joints (shoulder/elbow/wrist_flex/wrist_yaw); "
                        "j1 (vertical axis) and j6 (roll) carry ~no gravity. Solving "
                        "all four lets the fitter remove cross-coupling.")
    p.add_argument("--vlim", type=float, default=0.2,
                   help="sweep velocity (rad/s), kept low so inertia≈0 (default 0.2)")
    p.add_argument("--span", type=float, default=0.8,
                   help="sweep ± this many rad around the start angle, clamped to "
                        "soft limits with --margin (default 0.8)")
    p.add_argument("--margin", type=float, default=0.12,
                   help="stay this far inside soft limits at sweep endpoints (rad)")
    p.add_argument("--lo", type=float, default=None,
                   help="explicit sweep LOW endpoint (rad, user frame) for the swept "
                        "joint(s), overriding --span. Clamped to soft_min+margin.")
    p.add_argument("--hi", type=float, default=None,
                   help="explicit sweep HIGH endpoint (rad). Use --lo/--hi to cover "
                        "the FULL operating range so the model doesn't extrapolate.")
    p.add_argument("--repeats", type=int, default=2,
                   help="back-and-forth sweeps per joint (default 2)")
    p.add_argument("--rate", type=float, default=25.0, help="sample rate (Hz)")
    p.add_argument("--settle", type=float, default=0.5,
                   help="pause after reaching an endpoint (s)")
    p.add_argument("--torque-abort-ratio", type=float, default=0.80,
                   help="abort if any joint |torq| > ratio * tmax_fw")
    p.add_argument("--out", default=None,
                   help="CSV path (default outputs/gravity_calib_<timestamp>.csv)")
    p.add_argument("--no-wait", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(CFG_PATH)
    sweep_joints = [int(x) for x in args.joints.split(",") if x.strip()]
    interval = 1.0 / args.rate

    out_path = (Path(args.out) if args.out else
                REPO_ROOT / "outputs" / f"gravity_calib_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arm = ArmController(cfg)
    arm.open()
    arm.enable_all()
    time.sleep(0.3)

    # Read start pose — non-swept joints are held here for the whole run.
    hold = {}
    for i in ALL_JOINTS:
        s = arm.read_joint(i, settle_s=0.02)
        hold[i] = s["pos_user"] if s else 0.0
    print("Start pose (rad):  " + "  ".join(f"j{i}={hold[i]:+.3f}" for i in ALL_JOINTS))

    # Build per-joint sweep endpoints, clamped to soft limits.
    plan = {}
    for j in sweep_joints:
        jc = cfg.joint(j)
        lo = (max(jc.soft_min + args.margin, args.lo) if args.lo is not None
              else max(jc.soft_min + args.margin, hold[j] - args.span))
        hi = (min(jc.soft_max - args.margin, args.hi) if args.hi is not None
              else min(jc.soft_max - args.margin, hold[j] + args.span))
        plan[j] = (lo, hi)
        print(f"  sweep j{j} ({jc.name}): {lo:+.3f} .. {hi:+.3f} rad  "
              f"(soft [{jc.soft_min:+.2f},{jc.soft_max:+.2f}])")

    if not args.no_wait:
        print("\n=== READY ===  Confirm: workspace clear, arm in safe open pose, "
              "hand on power switch.")
        try:
            input("Press ENTER to start the sweep (Ctrl-C anytime to stop): ")
        except (KeyboardInterrupt, EOFError):
            arm.disable_all(); arm.close(); print("\naborted before start"); return

    fh = open(out_path, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["t_s", "phase", "swept_joint", "direction"]
               + [f"q{i}" for i in ALL_JOINTS]
               + [f"v{i}" for i in ALL_JOINTS]
               + [f"tau{i}" for i in ALL_JOINTS]
               + [f"st{i}" for i in ALL_JOINTS])
    print(f"Logging → {out_path}\n")

    t0 = time.monotonic()
    n_rows = 0

    def sample_and_command(active_j, target_active, direction, phase):
        """Command all joints (active→target, others→hold), read + log one row,
        run the watchdog. Returns the active joint's current position."""
        nonlocal n_rows
        for i in ALL_JOINTS:
            tgt = target_active if i == active_j else hold[i]
            arm.move_joint(i, tgt, vlim=args.vlim, hold_s=0.0)
        q, v, tau, st = {}, {}, {}, {}
        for i in ALL_JOINTS:
            s = arm.read_joint(i, settle_s=0.0)
            q[i] = s["pos_user"] if s else float("nan")
            v[i] = s["vel"] if s else float("nan")
            tau[i] = s["torq"] if s else float("nan")
            st[i] = s["status_code"] if s else -1
            jc = cfg.joint(i)
            if st[i] != 1:
                raise RuntimeError(f"j{i} status={st[i]}")
            if abs(tau[i]) > args.torque_abort_ratio * jc.tmax_fw:
                raise RuntimeError(f"j{i} torq={tau[i]:.2f} > "
                                   f"{args.torque_abort_ratio*jc.tmax_fw:.2f}")
        w.writerow([f"{time.monotonic()-t0:.4f}", phase, active_j, direction]
                   + [f"{q[i]:.5f}" for i in ALL_JOINTS]
                   + [f"{v[i]:.4f}" for i in ALL_JOINTS]
                   + [f"{tau[i]:.4f}" for i in ALL_JOINTS]
                   + [st[i] for i in ALL_JOINTS])
        n_rows += 1
        return q[active_j]

    def drive_to(active_j, target, direction, phase, tol=0.02, timeout=30.0):
        t_end = time.monotonic() + timeout
        nxt = time.monotonic()
        while time.monotonic() < t_end:
            cur = sample_and_command(active_j, target, direction, phase)
            if abs(cur - target) < tol:
                return
            nxt += interval
            slp = nxt - time.monotonic()
            if slp > 0:
                time.sleep(slp)
            else:
                nxt = time.monotonic()
        print(f"  [warn] j{active_j} didn't reach {target:+.3f} in {timeout}s "
              f"(cur {cur:+.3f}) — continuing")

    try:
        for j in sweep_joints:
            lo, hi = plan[j]
            print(f"--- sweeping j{j} ({cfg.joint(j).name}) ---")
            drive_to(j, lo, "to_lo", "approach")            # go to low endpoint
            time.sleep(args.settle)
            for r in range(args.repeats):
                print(f"  pass {r+1}/{args.repeats}: {lo:+.3f} → {hi:+.3f} → {lo:+.3f}")
                drive_to(j, hi, "+up", "sweep"); time.sleep(args.settle)
                drive_to(j, lo, "-down", "sweep"); time.sleep(args.settle)
            drive_to(j, hold[j], "return", "approach")      # back to start angle
            print(f"  j{j} done.")
        print(f"\nSweep complete. {n_rows} samples written.")
    except KeyboardInterrupt:
        print("\n[Ctrl-C] stopping")
    except RuntimeError as e:
        print(f"\n[abort] watchdog: {e}")
    except Exception as e:
        print(f"\n[unexpected] {type(e).__name__}: {e}")
    finally:
        print("Disabling follower...")
        try:
            arm.disable_all()
        except Exception:
            pass
        arm.close()
        fh.close()
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
