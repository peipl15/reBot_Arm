#!/usr/bin/env python3
"""Set the current position of one joint as encoder zero, persisted to motor flash.

Use after manually positioning the joint (e.g. hand-pushed a gripper to physically
closed when motor/jaw calibration drifted). Motor MUST be disabled when running
this — we do not enable it, so an incorrect new-zero won't cause sudden motion.

Sequence:
  1. open controller (does NOT enable any motor)
  2. read & print encoder pos before zero
  3. send set_zero_position
  4. (default) send store_parameters → writes to flash, survives power-cycle
  5. read & print encoder pos after  (should be ~0)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

if "libdeps" not in os.environ.get("LD_LIBRARY_PATH", ""):
    print("ERROR: export LD_LIBRARY_PATH=$HOME/.cache/motorbridge/libdeps:$LD_LIBRARY_PATH first")
    sys.exit(1)

from arm import ArmController, load_config

CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "joint_config.yaml"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--joint", "-j", type=int, required=True, help="joint index 1..7")
    p.add_argument("--no-persist", action="store_true",
                   help="set zero in volatile RAM only; do NOT call store_parameters "
                        "(lost on power-cycle). Default: persist to flash.")
    p.add_argument("--yes", action="store_true",
                   help="skip the confirm prompt")
    args = p.parse_args()

    cfg = load_config(CFG_PATH)
    j = cfg.joint(args.joint)
    print(f"Will set encoder zero on joint {args.joint} ({j.name}, id=0x{j.id:02x}).")
    if not args.no_persist:
        print("Persisting to motor flash (survives power-cycle).")
    else:
        print("Volatile only (will revert on power-cycle).")

    if not args.yes:
        confirm = input("\nProceed? [y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            print("aborted")
            return

    arm = ArmController(cfg)
    try:
        arm.open()
        motor = arm._motors[args.joint]

        # Read pre
        motor.request_feedback()
        time.sleep(0.1)
        s = motor.get_state()
        before_pos = s.pos if s else None
        before_pos_user = (before_pos * j.sign) if before_pos is not None else None
        print(f"\nbefore: motor_pos={before_pos}  pos_user={before_pos_user}")

        # Set zero
        print("sending set_zero_position...")
        motor.set_zero_position()
        time.sleep(0.3)

        if not args.no_persist:
            print("sending store_parameters (flash write)...")
            motor.store_parameters()
            time.sleep(0.8)  # flash write takes time

        # Read post
        motor.request_feedback()
        time.sleep(0.1)
        s2 = motor.get_state()
        after_pos = s2.pos if s2 else None
        after_pos_user = (after_pos * j.sign) if after_pos is not None else None
        print(f"\nafter:  motor_pos={after_pos}  pos_user={after_pos_user}")

        if after_pos is not None and abs(after_pos) < 0.01:
            print("\n[OK] encoder reads ~0 — re-zero successful.")
        else:
            print("\n[!] encoder did not snap to 0 — check that motor was disabled "
                  "and command actually reached it.")

    finally:
        arm.close()


if __name__ == "__main__":
    main()
