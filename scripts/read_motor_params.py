#!/usr/bin/env python3
"""Read PMAX/VMAX/TMAX and ID registers from all 7 motors.

Opens DM_Device once, iterates motors 0x01..0x07, reads:
  rid=21 (PMAX, f32) — position mapping range
  rid=22 (VMAX, f32) — velocity mapping range
  rid=23 (TMAX, f32) — torque mapping range
  rid=7  (MST_ID, u32) — feedback (host) ID
  rid=8  (ESC_ID, u32) — command (device) ID

Writes both stdout and outputs/motor_params_<timestamp>.txt.
Does not enable motors — reads only.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

if "libdeps" not in os.environ.get("LD_LIBRARY_PATH", ""):
    print("ERROR: export LD_LIBRARY_PATH=$HOME/.cache/motorbridge/libdeps:$LD_LIBRARY_PATH first")
    sys.exit(1)

from motorbridge import Controller

MOTOR_IDS = list(range(0x01, 0x08))
FEEDBACK_OFFSET = 0x10
MODEL = "4340"
PARAMS_F32 = [
    (21, "PMAX", "position mapping range (rad)"),
    (22, "VMAX", "velocity mapping range (rad/s)"),
    (23, "TMAX", "torque mapping range (Nm)"),
]
PARAMS_U32 = [
    (7, "MST_ID", "feedback / host ID"),
    (8, "ESC_ID", "command / device ID"),
]


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).resolve().parent.parent / "outputs" / f"motor_params_{ts}.txt"

    print(f"Reading motor params via dm-device (single channel 0), MODEL={MODEL}")
    print(f"Saving to: {out_path}\n")

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log(f"# motor_params dump  ts={ts}  model={MODEL}")
    log("# motor_id, feedback_id, MST_ID, ESC_ID, PMAX(rad), VMAX(rad/s), TMAX(Nm)")

    ctrl = Controller.from_dm_device("usb2canfd-dual", "0")
    try:
        for mid in MOTOR_IDS:
            fbid = mid + FEEDBACK_OFFSET
            row = [f"0x{mid:02x}", f"0x{fbid:02x}"]
            try:
                m = ctrl.add_damiao_motor(mid, fbid, MODEL)
                # u32 first (IDs)
                for rid, _, _ in PARAMS_U32:
                    try:
                        v = m.get_register_u32(rid, 800)
                        row.append(f"0x{v:02x}")
                    except Exception as e:
                        row.append(f"ERR({type(e).__name__})")
                # f32 (PMAX/VMAX/TMAX)
                for rid, _, _ in PARAMS_F32:
                    try:
                        v = m.get_register_f32(rid, 800)
                        row.append(f"{v:+.4f}")
                    except Exception as e:
                        row.append(f"ERR({type(e).__name__})")
            except Exception as e:
                row.append(f"add_motor ERR: {e}")
            log(", ".join(row))
            time.sleep(0.05)
    finally:
        try:
            ctrl.disable_all()
        except Exception:
            pass
        ctrl.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
