#!/usr/bin/env python3
"""Sanity check: load config, list joints, instantiate ArmController without opening.

Does NOT touch hardware. Safe to run anytime.
"""
from __future__ import annotations

from pathlib import Path

from arm import ArmController, load_config

CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "joint_config.yaml"


def main() -> None:
    print(f"loading {CFG_PATH}")
    cfg = load_config(CFG_PATH)
    print(f"\ntransport: {cfg.transport} / {cfg.dm_device_type} ch={cfg.dm_channel}")
    print(f"watchdog: torque_abort={cfg.watchdog.torque_abort_ratio:.0%}, "
          f"tracking_err={cfg.watchdog.tracking_error_rad}rad / "
          f"timeout={cfg.watchdog.tracking_error_timeout_s}s")
    print("\n  idx  id    fb    name      model  sign  pmax   vmax   tmax  soft           vlim")
    print("  ---  ----  ----  --------  -----  ----  ----   ----   ----  --------------  ----")
    for j in cfg.joints:
        soft = f"[{j.soft_min:+.2f},{j.soft_max:+.2f}]"
        print(f"  {j.index}    0x{j.id:02x}  0x{j.feedback_id:02x}  "
              f"{j.name:<8}  {j.model:<5}  {j.sign:+d}    "
              f"{j.pmax_fw:4.1f}  {j.vmax_fw:5.1f}  {j.tmax_fw:4.1f}  {soft:14s}  {j.vlim_default:.2f}")

    # Instantiate but don't open — proves the class loads cleanly
    arm = ArmController(cfg)
    print(f"\nArmController instantiated ok: {arm!r}")
    print("(open() not called — no hardware contact)")


if __name__ == "__main__":
    main()
