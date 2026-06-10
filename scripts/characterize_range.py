#!/usr/bin/env python3
"""Single-joint range characterization (Phase 2 ramp test).

Walks one joint outward from its current position in small steps, in the
requested direction(s), until any of:

  - watchdog trips (ArmSafetyError: torque / status / tracking)
  - reach diverges from target by more than `--reach-tol` rad
  - configured `--max-deg` excursion is reached

The last position the motor actually reached (in user frame) is reported as
the joint's empirical limit on that side. Subtract a safety buffer (e.g. 5°)
before writing it back to joint_config.yaml as soft_min / soft_max — that's
the next script's job, not this one.

Usage:

    python3 scripts/characterize_range.py --joint 7 --direction both
    python3 scripts/characterize_range.py --joint 7 --direction pos --step-deg 0.5 --vlim 0.15

Safety:
  - vlim defaults to 0.2 rad/s (conservative). DM-J4340 motors will trip
    on torque/status before causing damage if a true mechanical end-stop
    is hit, but USER MUST PHYSICALLY SUPPORT THE ARM regardless.
  - On any unexpected condition (watchdog, Ctrl-C, exception),
    the controller's __exit__ disables ALL motors. The arm WILL FALL
    UNDER GRAVITY without support.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

if "libdeps" not in os.environ.get("LD_LIBRARY_PATH", ""):
    print("ERROR: export LD_LIBRARY_PATH=$HOME/.cache/motorbridge/libdeps:$LD_LIBRARY_PATH first")
    sys.exit(1)

from arm import ArmController, ArmSafetyError, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "joint_config.yaml"
OUTPUT_DIR = REPO_ROOT / "outputs"


def deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def rad(deg_: float) -> float:
    return deg_ * math.pi / 180.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--joint", "-j", type=int, required=True,
                   help="joint index 1..7")
    p.add_argument("--direction", choices=["pos", "neg", "both"], default="both",
                   help="which direction to ramp (default: both)")
    p.add_argument("--step-deg", type=float, default=1.0,
                   help="step size in degrees (default 1.0)")
    p.add_argument("--vlim", type=float, default=0.2,
                   help="velocity limit rad/s (default 0.2)")
    p.add_argument("--hold-s", type=float, default=1.0,
                   help="hold time per step in seconds (default 1.0)")
    p.add_argument("--max-deg", type=float, default=120.0,
                   help="hard upper bound on excursion in degrees (default 120 — stop before then)")
    p.add_argument("--reach-tol-deg", type=float, default=1.5,
                   help="if |target-actual| exceeds this for a step, treat as blocked (default 1.5deg)")
    p.add_argument("--inter-step-s", type=float, default=0.1,
                   help="extra settle time between steps")
    p.add_argument("--home-first", action="store_true",
                   help="move joint to 0 (tracking watchdog disabled) before ramping. "
                        "Use when the joint is far from 0 — e.g. after a previous test "
                        "that left it at the edge.")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and exit without enabling motors")
    return p.parse_args()


def ramp_one_direction(arm, joint_index, sign_step, args, csv_writer):
    """Walk one direction step by step until a stop condition is hit.

    sign_step: +1 for positive, -1 for negative direction.
    Returns the last successfully reached actual_user position (rad).
    """
    j = arm.cfg.joint(joint_index)
    step_rad = rad(args.step_deg) * sign_step
    max_rad = rad(args.max_deg)
    tol_rad = rad(args.reach_tol_deg)

    # Read starting position (in user frame).
    initial = arm.read_joint(joint_index, settle_s=0.05)
    if initial is None:
        raise RuntimeError(f"joint {joint_index}: no initial feedback")
    start_user = initial["pos_user"]
    print(f"  start pos = {start_user:+.4f} rad ({deg(start_user):+.2f}°)")

    last_reached = start_user
    step_idx = 0
    while True:
        step_idx += 1
        target_user = start_user + step_idx * step_rad

        # cap at the user-requested max excursion
        excursion = abs(target_user - start_user)
        if excursion > max_rad:
            print(f"  reached --max-deg ({args.max_deg}°) — stopping for safety")
            return last_reached, "max_deg_hit"

        # raise_on_clamp=True: if our target is outside the YAML's soft_min/soft_max,
        # we deliberately want an explicit signal, because Phase 2's whole point is
        # discovering the real range. Bypass the placeholder yaml limits by reading
        # the joint cfg's pmax_fw as the conceptual ceiling.
        # But move_joint clamps to soft_min/soft_max — those are ±0.1 rad placeholders!
        # We need to compare against pmax_fw, not soft limits, during characterization.

        # Approach: bypass the wrapper's clamp by checking against pmax_fw ourselves
        # and sending pos-vel directly through the underlying motor handle when target
        # exceeds the placeholder soft limit. But that defeats sign-flip / watchdog.

        # Cleaner: this script REQUIRES temporarily wide soft limits. Either patch the
        # cfg in-memory or read ground-truth from pmax_fw.
        # We'll patch in-memory (only this script's instance) so the wrapper's
        # clamp + watchdog still apply, just with a wider envelope.
        pass  # handled in main() via override_soft_limits()

        print(f"  step {step_idx:2d}: target={target_user:+.4f} ({deg(target_user):+6.2f}°)",
              end="  ", flush=True)
        ts = time.time()
        try:
            arm.move_joint(joint_index, target_user,
                           vlim=args.vlim, hold_s=args.hold_s)
        except ArmSafetyError as e:
            print(f"WATCHDOG TRIP: {e.event.kind}")
            csv_writer.writerow([ts, step_idx, joint_index, target_user,
                                 "", "", "", "", f"trip:{e.event.kind}"])
            return last_reached, f"watchdog_{e.event.kind}"

        time.sleep(args.inter_step_s)
        state = arm.read_joint(joint_index, settle_s=0.05)
        if state is None:
            print("(no feedback)")
            csv_writer.writerow([ts, step_idx, joint_index, target_user,
                                 "", "", "", "", "no_feedback"])
            return last_reached, "no_feedback"

        actual = state["pos_user"]
        err = target_user - actual
        print(f"actual={actual:+.4f} err={err:+.4f}  torq={state['torq']:+.2f}  st={state['status_code']}")
        csv_writer.writerow([ts, step_idx, joint_index, target_user, actual,
                             state["vel"], state["torq"], state["status_code"], "ok"])

        if abs(err) > tol_rad:
            print(f"  reach tolerance exceeded (|err|={abs(err):.4f} > {tol_rad:.4f}) — blocked")
            return last_reached, "reach_tolerance"

        last_reached = actual


def override_soft_limits(cfg, joint_index: int, half_width_rad: float):
    """Return a new ArmConfig where joint_index's soft limits are set to
    ±half_width_rad around 0 — needed only during characterization, since
    the YAML's placeholder limits are tighter than what we'll explore.
    """
    from dataclasses import replace
    new_joints = []
    for j in cfg.joints:
        if j.index == joint_index:
            j = replace(j, soft_min=-half_width_rad, soft_max=+half_width_rad)
        new_joints.append(j)
    return replace(cfg, joints=tuple(new_joints))


def recover(arm, joint_index: int) -> bool:
    """Re-enable a joint after ArmController.disable_all() was triggered by a
    watchdog trip. Idempotent — harmless if the joint is still enabled.
    """
    try:
        arm.enable(joint_index)
        time.sleep(0.3)
        return True
    except Exception as e:
        print(f"  recover failed for joint {joint_index}: {e}")
        return False


def safe_return_to_zero(arm, joint_index, args):
    """Walk the joint back toward 0. Disables only the tracking watchdog —
    torque + status remain active — because a large pre-existing |target-actual|
    at the start of the move is normal and not a fault. Re-enables after any
    trip so the next phase can still run.
    """
    print(f"\n  returning joint {joint_index} to 0...")
    cur = arm.read_joint(joint_index, settle_s=0.05)
    if cur is None:
        print("  no feedback before return")
        return
    distance = abs(cur["pos_user"])
    # transit time = distance / vlim + 1s margin
    hold = distance / max(args.vlim, 1e-3) + 1.5
    print(f"  current={cur['pos_user']:+.4f}  estimated transit {hold:.1f}s at vlim={args.vlim}")
    try:
        arm.move_joint(joint_index, 0.0, vlim=args.vlim,
                       hold_s=hold, check_tracking_error=False)
    except ArmSafetyError as e:
        print(f"  return-to-zero tripped: {e}")
        recover(arm, joint_index)
    state = arm.read_joint(joint_index, settle_s=0.1)
    if state:
        print(f"  rest pos = {state['pos_user']:+.4f} ({deg(state['pos_user']):+.2f}°)")


def main():
    args = parse_args()
    cfg = load_config(CONFIG_PATH)
    try:
        target_joint = cfg.joint(args.joint)
    except KeyError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    half_width = rad(args.max_deg) + rad(args.step_deg) * 5  # comfortable headroom
    cfg = override_soft_limits(cfg, args.joint, half_width)
    print(f"Characterizing joint {args.joint} ({target_joint.name})")
    print(f"  step={args.step_deg}°  vlim={args.vlim} rad/s  hold={args.hold_s}s "
          f"max={args.max_deg}°  reach_tol={args.reach_tol_deg}°")
    print(f"  override soft limits for this run only: ±{deg(half_width):.1f}°\n")

    if args.dry_run:
        print("(--dry-run: not opening hardware)")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"range_j{args.joint}_{ts}.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    arm = ArmController(cfg)
    results = {}
    try:
        arm.open()
        arm.enable(args.joint)
        time.sleep(0.3)

        if args.home_first:
            cur = arm.read_joint(args.joint, settle_s=0.05)
            if cur is None:
                print("  home-first: no feedback, skipping home")
            else:
                dist = abs(cur["pos_user"])
                if dist < rad(args.step_deg):
                    print(f"  already near 0 ({cur['pos_user']:+.4f}), no home needed")
                else:
                    hold = dist / max(args.vlim, 1e-3) + 1.5
                    print(f"  homing joint {args.joint} from {cur['pos_user']:+.4f} "
                          f"to 0 (est {hold:.1f}s, tracking watchdog off)")
                    try:
                        arm.move_joint(args.joint, 0.0, vlim=args.vlim,
                                       hold_s=hold, check_tracking_error=False)
                    except ArmSafetyError as e:
                        print(f"  home tripped: {e}")
                        recover(arm, args.joint)
                    after = arm.read_joint(args.joint, settle_s=0.1)
                    if after:
                        print(f"  homed to {after['pos_user']:+.4f}\n")

        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts", "step_idx", "joint_index", "target_user",
                        "actual_user", "vel", "torq", "status_code", "result"])

            if args.direction in ("pos", "both"):
                print("[+] walking POSITIVE direction")
                reached, why = ramp_one_direction(arm, args.joint, +1, args, w)
                results["pos"] = (reached, why)
                print(f"  POS reached {reached:+.4f} ({deg(reached):+.2f}°), reason: {why}")
                if why.startswith("watchdog_") or why == "no_feedback":
                    recover(arm, args.joint)
                if args.direction == "both":
                    safe_return_to_zero(arm, args.joint, args)
                    recover(arm, args.joint)  # idempotent

            if args.direction in ("neg", "both"):
                print("\n[-] walking NEGATIVE direction")
                reached, why = ramp_one_direction(arm, args.joint, -1, args, w)
                results["neg"] = (reached, why)
                print(f"  NEG reached {reached:+.4f} ({deg(reached):+.2f}°), reason: {why}")
                if why.startswith("watchdog_") or why == "no_feedback":
                    recover(arm, args.joint)

            safe_return_to_zero(arm, args.joint, args)
    except KeyboardInterrupt:
        print("\n[!] Ctrl-C caught — disabling and exiting")
    except Exception as e:
        print(f"\n[!] unexpected error: {type(e).__name__}: {e}")
        raise
    finally:
        try:
            arm.disable(args.joint)
        except Exception:
            pass
        arm.close()

    print(f"\nCSV: {csv_path}")
    print("Summary:")
    for direction, (reached, why) in results.items():
        print(f"  {direction}: reached {reached:+.4f} rad ({deg(reached):+.2f}°)  reason: {why}")


if __name__ == "__main__":
    main()
