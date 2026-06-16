#!/usr/bin/env python3
"""Reset the follower arm to a known state:

  - j1 (base): set current position as the new encoder zero, persisted to flash
    (use when j1 has been physically rotated and you want that as the new home)
  - j2..j7: walk back to encoder 0 (tracking watchdog disabled for the long traversal)

Reads + reports current pos for all 7 joints first so the user can sanity-check
which joints will actually move.

Safety:
  - Joints are enabled one at a time, moved, then disabled. The arm must be
    physically supported in case j2/j3 swing under gravity.
  - vlim 0.2 rad/s for heavy joints, 0.3 rad/s for wrist/gripper.
"""
from __future__ import annotations

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

J1_REZERO = True                   # set j1 current pos as new zero
J1_PERSIST = True                  # write to motor flash
HOME_JOINTS = [2, 3, 4, 5, 6, 7]   # joints to move to 0


def read_all(arm: ArmController) -> dict[int, dict]:
    out = {}
    for j in arm.cfg.joints:
        s = arm.read_joint(j.index, settle_s=0.06)
        out[j.index] = s
    return out


def home_one(arm: ArmController, joint_idx: int) -> None:
    j = arm.cfg.joint(joint_idx)
    s = arm.read_joint(joint_idx, settle_s=0.05)
    if s is None:
        print(f"  j{joint_idx} {j.name}: no feedback, skip")
        return
    cur = s["pos_user"]
    if abs(cur) < 0.02:
        print(f"  j{joint_idx} {j.name}: already at {cur:+.4f}, skip")
        return
    # Heavy joints get lower vlim, wrist joints faster
    vlim = 0.2 if j.tmax_fw >= 20.0 else 0.3
    hold = abs(cur) / vlim + 1.5
    print(f"  j{joint_idx} {j.name}: {cur:+.4f} -> 0  (est {hold:.1f}s @ vlim={vlim})")
    try:
        arm.enable(joint_idx)
        time.sleep(0.25)
        arm.move_joint(joint_idx, 0.0, vlim=vlim, hold_s=hold,
                       check_tracking_error=False)
        after = arm.read_joint(joint_idx, settle_s=0.1)
        if after:
            print(f"    rest={after['pos_user']:+.4f}")
    except ArmSafetyError as e:
        print(f"    !! TRIP: {e}")
    finally:
        try:
            arm.disable(joint_idx)
        except Exception:
            pass
        time.sleep(0.12)


def rezero_j1(arm: ArmController) -> None:
    """Re-zero j1 at current position, persist to flash. Motor stays disabled."""
    motor = arm._motors[1]
    motor.request_feedback()
    time.sleep(0.1)
    s = motor.get_state()
    before = s.pos if s else None
    print(f"\n  j1 BEFORE set_zero: motor_pos={before}")
    motor.set_zero_position()
    time.sleep(0.3)
    if J1_PERSIST:
        print("  persisting to flash (store_parameters)...")
        motor.store_parameters()
        time.sleep(0.8)
    motor.request_feedback()
    time.sleep(0.1)
    s2 = motor.get_state()
    after = s2.pos if s2 else None
    print(f"  j1 AFTER set_zero:  motor_pos={after}  (target=0)")
    if after is not None and abs(after) < 0.01:
        print("  [OK] j1 now at zero")
    else:
        print("  [!!] j1 didn't snap to 0; check motor was disabled")


def main():
    cfg = load_config(CFG_PATH)
    arm = ArmController(cfg)
    try:
        arm.open()

        print("=== Current joint positions (user frame) ===")
        states = read_all(arm)
        for idx in sorted(states):
            j = cfg.joint(idx)
            s = states[idx]
            if s is None:
                print(f"  j{idx} {j.name:<11} no feedback")
            else:
                print(f"  j{idx} {j.name:<11} pos_user={s['pos_user']:+.4f}  "
                      f"vel={s['vel']:+.3f}  torq={s['torq']:+.3f}  "
                      f"status={s['status_code']}")

        if J1_REZERO:
            print("\n=== Re-zeroing j1 at current position ===")
            rezero_j1(arm)

        print("\n=== Homing j2..j7 to 0 ===")
        for idx in HOME_JOINTS:
            home_one(arm, idx)

        print("\n=== Final positions ===")
        states = read_all(arm)
        for idx in sorted(states):
            j = cfg.joint(idx)
            s = states[idx]
            if s is None:
                continue
            print(f"  j{idx} {j.name:<11} pos_user={s['pos_user']:+.4f}")

    except KeyboardInterrupt:
        print("\n[!] Ctrl-C, disabling all")
    finally:
        try:
            arm.disable_all()
        except Exception:
            pass
        arm.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
