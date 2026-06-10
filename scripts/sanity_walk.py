#!/usr/bin/env python3
"""Sanity walk: drive each joint to 50% of its soft limit on each side and back.

Verifies that the calibrated soft limits + sign-flip + watchdog + return-to-zero
all play together without surprises. Skips joints whose soft limit on a side is
~0 (e.g. j2/j3/j4 have soft_min=0 → no negative excursion).

Excursion capped at ±CAP_RAD (default ~60°) per side so wide joints (j6, j7)
don't take forever to traverse. The goal is to verify the wrapper, not
re-characterize.

Order: distal → proximal (j7, j6, j5, j4, j3, j2, j1). Lighter joints first
so any trouble surfaces before driving the heavy gravity joints + base sweep.

Tracking watchdog is ON for short moves (<0.3 rad) and OFF for long ones
(would otherwise false-trip on transit) — torque + status watchdog always on.
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

if "libdeps" not in os.environ.get("LD_LIBRARY_PATH", ""):
    print("ERROR: export LD_LIBRARY_PATH=$HOME/.cache/motorbridge/libdeps:$LD_LIBRARY_PATH first")
    sys.exit(1)

from arm import ArmController, ArmSafetyError, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = REPO_ROOT / "configs" / "joint_config.yaml"

CAP_RAD = 1.05           # ~60° cap per excursion
HALF_FRAC = 0.5          # walk to half of soft limit
JOINT_ORDER = [7, 6, 5, 4, 3, 2, 1]  # distal → proximal
TINY = 0.02              # below this absolute value we treat target as zero/skip


def cap_target(soft_limit: float) -> float:
    """Cap target_user to ±CAP_RAD magnitude."""
    if soft_limit >= 0:
        return min(soft_limit * HALF_FRAC, CAP_RAD)
    return max(soft_limit * HALF_FRAC, -CAP_RAD)


def recover_joint(arm: ArmController, joint: int) -> None:
    """Re-enable a joint after ArmController.disable_all() following a trip."""
    try:
        arm.enable(joint)
        time.sleep(0.3)
    except Exception as e:
        print(f"    recover failed: {e}")


def go_to(arm: ArmController, joint: int, target_user: float, label: str) -> bool:
    """Walk joint to target_user. Returns True on success, False on trip or no-feedback."""
    j = arm.cfg.joint(joint)
    cur = arm.read_joint(joint, settle_s=0.05)
    if cur is None:
        print(f"    [{label}] no feedback before move")
        return False
    distance = abs(target_user - cur["pos_user"])
    vlim = min(j.vlim_default * 0.6, 0.3)
    hold = distance / max(vlim, 1e-3) + 1.2
    # Tracking watchdog only when motor can plausibly reach target within
    # its timeout window. Otherwise (long traversal) the watchdog will fire
    # before the motor has had time to arrive.
    tracking_timeout = arm.cfg.watchdog.tracking_error_timeout_s
    use_tracking = distance < vlim * tracking_timeout
    print(f"    [{label}] target={target_user:+.3f}  cur={cur['pos_user']:+.3f}  "
          f"dist={distance:.3f}  hold={hold:.1f}s  vlim={vlim:.2f}  track_wd={use_tracking}")
    try:
        arm.move_joint(joint, target_user, vlim=vlim, hold_s=hold,
                       check_tracking_error=use_tracking)
    except ArmSafetyError as e:
        print(f"    [{label}] !! TRIP: {e}")
        recover_joint(arm, joint)
        return False
    after = arm.read_joint(joint, settle_s=0.1)
    if after is not None:
        err = abs(target_user - after["pos_user"])
        flag = "  OK" if err < 0.03 else f"  WARN err={err:.3f}"
        print(f"    [{label}] reached {after['pos_user']:+.3f}  err={err:.3f}{flag}")
    return True


def sanity_joint(arm: ArmController, j_idx: int) -> dict:
    j = arm.cfg.joint(j_idx)
    print(f"\n=== Joint {j_idx} ({j.name})  soft=[{j.soft_min:+.3f}, {j.soft_max:+.3f}] ===")

    pos_target = cap_target(j.soft_max)
    neg_target = cap_target(j.soft_min)
    print(f"  plan: pos→{pos_target:+.3f}  neg→{neg_target:+.3f}")

    result = {"pos": "skipped", "neg": "skipped"}
    try:
        arm.enable(j_idx)
        time.sleep(0.3)

        if pos_target > TINY:
            result["pos"] = "ok" if go_to(arm, j_idx, pos_target, "pos half") else "trip"
            go_to(arm, j_idx, 0.0, "return-A")
        else:
            print(f"    pos_target {pos_target:+.3f} below TINY, skip positive walk")

        if neg_target < -TINY:
            result["neg"] = "ok" if go_to(arm, j_idx, neg_target, "neg half") else "trip"
            go_to(arm, j_idx, 0.0, "return-B")
        else:
            print(f"    neg_target {neg_target:+.3f} above -TINY, skip negative walk")
    finally:
        try:
            arm.disable(j_idx)
        except Exception:
            pass
        time.sleep(0.15)

    return result


def main():
    cfg = load_config(CFG_PATH)
    arm = ArmController(cfg)
    summary = {}
    try:
        arm.open()
        for j in JOINT_ORDER:
            try:
                summary[j] = sanity_joint(arm, j)
            except Exception as e:
                print(f"[!!] Joint {j} unexpected error: {type(e).__name__}: {e}")
                summary[j] = {"pos": f"error:{type(e).__name__}", "neg": "error"}
                try:
                    arm.disable(j)
                except Exception:
                    pass
    except KeyboardInterrupt:
        print("\n[!] Ctrl-C — disabling all and exiting")
    finally:
        try:
            arm.disable_all()
        except Exception:
            pass
        arm.close()

    print("\n=== Sanity walk summary ===")
    for j in JOINT_ORDER:
        if j in summary:
            r = summary[j]
            name = cfg.joint(j).name
            print(f"  j{j} {name:<11}  pos={r['pos']:<10}  neg={r['neg']:<10}")
    print()


if __name__ == "__main__":
    main()
