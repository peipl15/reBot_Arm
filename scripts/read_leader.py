#!/usr/bin/env python3
"""One-shot read of all 7 leader-arm servo angles.

Counterpart to scripts/read_motor_params.py (which reads the follower
motors). Quick check that the leader is alive and producing stable
encoder values before running teleop.

Usage:
    python3 scripts/read_leader.py
    python3 scripts/read_leader.py --port /dev/ttyUSB1 --samples 10 --interval 0.1
"""
from __future__ import annotations

import argparse
import time

from fashionstar_uart_sdk.uart_pocket_handler import PortHandler


JOINT_NAMES = {
    0: "shoulder_pan",
    1: "shoulder_lift",
    2: "elbow_flex",
    3: "wrist_flex",
    4: "wrist_yaw",
    5: "wrist_roll",
    6: "gripper",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="/dev/ttyUSB0",
                   help="serial port the leader is on (default /dev/ttyUSB0)")
    p.add_argument("--baudrate", type=int, default=1_000_000,
                   help="serial baudrate (default 1 Mbps)")
    p.add_argument("--samples", type=int, default=5,
                   help="number of samples to read (default 5)")
    p.add_argument("--interval", type=float, default=0.2,
                   help="seconds between samples (default 0.2)")
    return p.parse_args()


def main():
    args = parse_args()
    ids = list(JOINT_NAMES.keys())

    h = PortHandler(args.port, args.baudrate)
    h.openPort()

    # Ping each servo first — easier to spot a missing one than a missing reading.
    print(f"Pinging servos on {args.port} @ {args.baudrate}:")
    missing = []
    for sid in ids:
        ok = h.ping(sid)
        print(f"  servo {sid:>2} ({JOINT_NAMES[sid]:<13}): {'OK' if ok else 'NO RESPONSE'}")
        if not ok:
            missing.append(sid)
    if missing:
        print(f"\nWARNING: servos {missing} did not respond. Check power/wiring.")
    print()

    # Read N samples and tabulate.
    header = f"{'servo':<14} {'id':>3} | " + "  ".join(f"sample{i+1}" for i in range(args.samples))
    print(header)
    print("-" * len(header))

    samples = []
    for _ in range(args.samples):
        data = h.sync_read["Monitor"](ids)
        row = {sid: getattr(data.get(sid), "current_position", None) for sid in ids}
        samples.append(row)
        time.sleep(args.interval)

    for sid in ids:
        vals = [s[sid] for s in samples]
        vstr = "  ".join(f"{v:+7.2f}" if v is not None else "    nan" for v in vals)
        print(f"{JOINT_NAMES[sid]:<14} {sid:>3} | {vstr}")


if __name__ == "__main__":
    main()
