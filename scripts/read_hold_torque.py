#!/usr/bin/env python3
"""SAFE read-only check: hold the arm at its CURRENT pose (POS_VEL, no motion)
and report the holding torque each joint produces. Compare to URDF gravity g(q)
to determine the sign/convention relationship before ever commanding MIT torque.

Run with the arm already at the pose of interest (e.g. mechanical zero).
No motion is commanded beyond holding where it already is.

  cd ~/code/rebot_arm && source .venv/bin/activate && \
  export LD_LIBRARY_PATH=$HOME/.cache/motorbridge/libdeps && \
  python3 scripts/read_hold_torque.py
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

if "libdeps" not in os.environ.get("LD_LIBRARY_PATH", ""):
    print("ERROR: export LD_LIBRARY_PATH=$HOME/.cache/motorbridge/libdeps first"); sys.exit(1)

from arm import ArmController, load_config

REPO = Path(__file__).resolve().parent.parent
cfg = load_config(REPO / "configs" / "joint_config.yaml")
arm = ArmController(cfg); arm.open()
try:
    arm.enable_all(); time.sleep(0.3)
    q0 = {j: arm.read_joint(j, settle_s=0.02)["pos_user"] for j in range(1, 8)}
    print("Holding at current pose (rad): " + "  ".join(f"j{j}={q0[j]:+.3f}" for j in range(1, 8)))
    # hold in place for ~1.5 s, then read steady torque
    t_end = time.monotonic() + 1.5
    while time.monotonic() < t_end:
        for j in range(1, 8):
            arm.move_joint(j, q0[j], vlim=0.3, hold_s=0.0)
        time.sleep(0.02)
    meas, q = {}, {}
    for j in range(1, 8):
        s = arm.read_joint(j, settle_s=0.03)
        meas[j] = s["torq"]; q[j] = s["pos_user"]

    # URDF gravity at the CURRENT q (our q fed directly: offset=0, sign=+1)
    import numpy as np, pinocchio as pin
    U = ("/home/yinzi/code/reBotArm_control_py/urdf/"
         "reBot-DevArm_fixend_description/urdf/reBot-DevArm_fixend.urdf")
    m = pin.buildModelFromUrdf(U); d = m.createData()
    qv = np.array([q[j] for j in range(1, 7)])
    g = pin.computeGeneralizedGravity(m, d, qv)

    print("\n  joint        q(rad)   MEAS torq   URDF g(our_q)   meas/urdf")
    print("  " + "-" * 58)
    for j in range(1, 7):
        gu = g[j - 1]
        ratio = (meas[j] / gu) if abs(gu) > 0.2 else float("nan")
        print(f"  j{j} {cfg.joint(j).name:<11} {q[j]:+6.3f}   {meas[j]:+7.2f}    {gu:+7.2f}        {ratio:+.2f}")
    print(f"  j7 {cfg.joint(7).name:<11} {q[7]:+6.3f}   {meas[7]:+7.2f}    (gripper, n/a)")
finally:
    arm.disable_all(); arm.close()
    print("\n判读: ratio≈+1 → 符号+量级都对(直接用URDF g);  ≈-1 → 仅符号反;")
    print("      ratio 量级偏离1很多/各关节不一致 → 我们零位≠URDF home(需对齐).")
