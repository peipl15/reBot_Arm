#!/usr/bin/env python3
"""Validate the fitted planar gravity model on the arm via MIT feedforward.

Puts the vertical-plane joints (j2,j3,j4) in MIT mode with:
    tau = kp·(pos_target − q) + kd·(0 − v) + tau_gravity(q)
where tau_gravity comes from configs/gravity_planar.json. The other joints
(j1,j5,j6,j7) are held rigidly in POS_VEL.

  >>> FIRST USE OF TORQUE CONTROL — REAL RISK. <<<
SUPPORT THE ARM BY HAND before/while starting. If the gravity model is off, the
arm can drift or push — be ready to Ctrl-C (disables instantly).

Staged use (recommended):
  1) default (compliant HOLD toward the start pose): arm should hold its pose
     with LOW measured torque (the feedforward is doing the lifting). Safe.
  2) once it holds cleanly, add --float: pos target follows q ⇒ true weightless
     float (push it anywhere, it should stay). Riskier — keep a hand on it.

Usage:
  cd ~/code/rebot_arm && source .venv/bin/activate && \
  export LD_LIBRARY_PATH=$HOME/.cache/motorbridge/libdeps && \
  python3 scripts/gravity_float_test.py            # compliant hold (safe)
  python3 scripts/gravity_float_test.py --float --kp 2 --kd 1
"""
from __future__ import annotations

import argparse
import json
import math
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
MODEL_PATH = REPO_ROOT / "configs" / "zero_calib.json"
MIT_JOINTS = [2, 3, 4]
HOLD_JOINTS = [1, 5, 6, 7]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(MODEL_PATH))
    p.add_argument("--kp", type=float, default=3.0, help="MIT position stiffness")
    p.add_argument("--kd", type=float, default=1.0, help="MIT damping")
    p.add_argument("--float", action="store_true",
                   help="pos target follows q (true float). Default: hold start pose.")
    p.add_argument("--hold-on-stop", action="store_true",
                   help="pos follows q WHILE moving, FREEZES when you stop "
                        "(|v|<vthresh) → compliant to push, holds where you leave it. "
                        "Best 'move-anywhere-it-stays' despite the ~2Nm gravity floor. "
                        "Use a higher --kp (e.g. 15) so the hold is firm.")
    p.add_argument("--vthresh", type=float, default=0.08,
                   help="speed (rad/s) below which a joint counts as 'stopped'")
    p.add_argument("--ki", type=float, default=4.0,
                   help="integral gain (Nm per rad·s) — cancels steady-state gravity "
                        "residual when holding (the official lock-mode trick). Active "
                        "only in --hold-on-stop when a joint is stopped.")
    p.add_argument("--i-clamp", type=float, default=1.0,
                   help="anti-windup clamp on the integral accumulator (rad·s)")
    p.add_argument("--clamp-ratio", type=float, default=0.7,
                   help="clamp |tau_g| to ratio*tmax_fw per joint (default 0.7)")
    p.add_argument("--rate", type=float, default=100.0)
    p.add_argument("--abort-ratio", type=float, default=0.85,
                   help="disable if any measured |torq| > ratio*tmax_fw")
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--wrist", action="store_true",
                   help="also gravity-comp j5,j6 (free the wrist orientation). Their "
                        "gravity is small (j6=0, j5<0.6Nm). NOTE: j5/j6 are on the "
                        "distal CAN segment — only enable once that wiring is solid.")
    p.add_argument("--wrist-kp", type=float, default=0.5,
                   help="MIT stiffness for j5/j6 only. Small by default: the wrist "
                        "carries ~no gravity, so a soft kp keeps it easy to hand-deflect "
                        "(high kp fights you). It stays put via its own friction.")
    p.add_argument("--wrist-kd", type=float, default=0.3,
                   help="MIT damping for j5/j6 only (low → easy to move).")
    p.add_argument("--wrist-vthresh", type=float, default=0.03,
                   help="speed below which j5/j6 count as 'stopped'. Lower than the "
                        "arm's --vthresh because the wrist is hand-moved slowly; the "
                        "integral then learns to cancel its small gravity so it stays "
                        "where left instead of creeping back to center.")
    return p.parse_args()


def make_predictor(calib):
    """Pinocchio gravity feedforward from configs/zero_calib.json.

    tau_ff[j] = scale[j] * SIGN[j] * g_urdf( SIGN .* (q_user - zero) )[j]
    q_user is read_joint pos_user; q (dict) must carry j1..j6 (gravity on j2/j3/j4
    depends on the distal j5/j6 config too). scale absorbs the DM-J4340P decode
    inflation so tau_ff is already in motor/decoded units → command directly.
    """
    import numpy as np
    import pinocchio as pin
    model = pin.buildModelFromUrdf(calib["urdf"])
    data = model.createData()
    SIGN = {int(k): float(v) for k, v in calib["sign"].items()}
    ZERO = {int(k): float(v) for k, v in calib["zero"].items()}
    SCALE = {int(k): float(v) for k, v in calib.get("torque_scale", {}).items()}

    def predict(q):  # q: dict joint->rad, needs j1..j6
        qv = np.array([SIGN[k] * (q.get(k, 0.0) - ZERO.get(k, 0.0)) for k in [1, 2, 3, 4, 5, 6]])
        qv[0] = 0.0  # j1 vertical axis: gravity-irrelevant
        g = pin.computeGeneralizedGravity(model, data, qv)
        return {j: SCALE.get(j, 1.0) * SIGN[j] * g[j - 1] for j in MIT_JOINTS}

    return predict


def main():
    args = parse_args()
    if args.wrist:
        global MIT_JOINTS, HOLD_JOINTS
        MIT_JOINTS = [2, 3, 4, 5, 6]   # add wrist yaw/roll to gravity comp
        HOLD_JOINTS = [1, 7]           # j7 gripper stays rigid (not a float joint)
    cfg = load_config(CFG_PATH)
    model = json.load(open(args.model))
    predict = make_predictor(model)
    from motorbridge.models import Mode

    arm = ArmController(cfg)
    arm.open()

    # Hold non-plane joints rigidly in POS_VEL.
    for j in HOLD_JOINTS:
        arm.enable(j)
    time.sleep(0.2)
    start = {}
    for j in [1, 2, 3, 4, 5, 6, 7]:
        s = arm.read_joint(j, settle_s=0.02)
        start[j] = s["pos_user"] if s else 0.0

    # Predicted gravity at start (sanity print). predict needs the full j1..j6 pose.
    q0 = {j: start[j] for j in range(1, 7)}
    tg0 = predict(q0)
    print("Start pose (rad):  " + "  ".join(f"j{j}={start[j]:+.3f}" for j in MIT_JOINTS))
    print("Predicted gravity ff at start (Nm):  "
          + "  ".join(f"j{j}={tg0[j]:+.2f}" for j in MIT_JOINTS))
    mode = ("HOLD-ON-STOP + integral" if args.hold_on_stop else
            "TRUE FLOAT (pos follows q)" if args.float else "compliant HOLD (pos=start)")
    print(f"Mode: {mode}   kp={args.kp} kd={args.kd}"
          + (f" ki={args.ki}" if args.hold_on_stop else ""))

    if not args.no_wait:
        print("\n>>> SUPPORT THE ARM BY HAND. Ctrl-C disables instantly. <<<")
        try:
            input("Press ENTER to enable MIT gravity comp on j2,j3,j4: ")
        except (KeyboardInterrupt, EOFError):
            arm.disable_all(); arm.close(); print("\naborted"); return

    # Switch plane joints to MIT and enable.
    for j in MIT_JOINTS:
        m = arm._motors[j]
        m.ensure_mode(Mode.MIT, 1000)
        m.enable()
    time.sleep(0.2)

    interval = 1.0 / args.rate
    nxt = time.monotonic()
    last_print = 0.0
    pos_target = {j: start[j] for j in MIT_JOINTS}   # for --hold-on-stop
    integ = {j: 0.0 for j in MIT_JOINTS}             # integral accumulator (hold-on-stop)
    print("\n=== Gravity comp running. Ctrl-C to stop. ===\n")
    try:
        while True:
            # read plane-joint state
            q, v, meas = {}, {}, {}
            for j in MIT_JOINTS:
                s = arm.read_joint(j, settle_s=0.0)
                q[j] = s["pos_user"] if s else start[j]
                v[j] = s["vel"] if s else 0.0
                meas[j] = s["torq"] if s else 0.0
                if s and s["status_code"] != 1:
                    raise RuntimeError(f"j{j} status={s['status_code']}")
                if s and abs(meas[j]) > args.abort_ratio * cfg.joint(j).tmax_fw:
                    raise RuntimeError(f"j{j} |torq|={meas[j]:.1f} over limit")
                # MIT bypasses soft limits — abort if a joint sags/drives past them.
                jc = cfg.joint(j)
                if q[j] < jc.soft_min - 0.05 or q[j] > jc.soft_max + 0.05:
                    raise RuntimeError(f"j{j} q={q[j]:.2f} outside soft limits "
                                       f"[{jc.soft_min:.2f},{jc.soft_max:.2f}]")
            # gravity on j2/j3/j4 depends on the FULL pose incl. the wrist. Always
            # read j5/j6 LIVE (even when held) so deflecting them updates j2/j3/j4 ff.
            q[1] = start[1]               # j1 vertical: irrelevant to gravity anyway
            for j in (5, 6):
                if j in q:
                    continue              # already read live (in MIT_JOINTS)
                s = arm.read_joint(j, settle_s=0.0)
                q[j] = s["pos_user"] if s else start[j]

            tg = predict(q)
            for j in MIT_JOINTS:
                lim = args.clamp_ratio * cfg.joint(j).tmax_fw
                if j == 6:
                    # j6 (wrist roll) has EXACTLY zero gravity → pure soft float: target
                    # follows q, no ff. Stays where left by its own friction.
                    pos_t = q[j]
                    tau_cmd = 0.0
                else:
                    # j2,j3,j4 AND j5 now all have a calibrated gravity ff (j5 zero/scale
                    # fitted from the j4-pitched sweep) → normal gravity-comp path.
                    tau_g = max(-lim, min(lim, tg[j]))
                    if args.hold_on_stop:
                        if abs(v[j]) > args.vthresh:   # being pushed → follow, reset integral
                            pos_target[j] = q[j]; integ[j] = 0.0
                        else:                          # stopped → integrate residual to zero it
                            integ[j] += (pos_target[j] - q[j]) * interval
                            integ[j] = max(-args.i_clamp, min(args.i_clamp, integ[j]))
                        pos_t = pos_target[j]
                        tau_cmd = tau_g + args.ki * integ[j]
                    elif args.float:
                        pos_t = q[j]; tau_cmd = tau_g
                    else:
                        pos_t = start[j]; tau_cmd = tau_g
                    tau_cmd = max(-lim, min(lim, tau_cmd))
                # wrist joints get a soft kp/kd so they stay easy to hand-deflect
                kp = args.wrist_kp if j in (5, 6) else args.kp
                kd = args.wrist_kd if j in (5, 6) else args.kd
                arm._motors[j].send_mit(pos=pos_t, vel=0.0, kp=kp, kd=kd, tau=tau_cmd)

            for j in HOLD_JOINTS:        # keep the rest rigid
                arm.move_joint(j, start[j], vlim=0.5, hold_s=0.0)

            now = time.monotonic()
            if now - last_print > 1.0:
                print("  " + "  ".join(
                    f"j{j} q={q[j]:+.2f} ff={tg[j]:+5.1f} meas={meas[j]:+5.1f}"
                    for j in MIT_JOINTS))
                last_print = now
            nxt += interval
            slp = nxt - time.monotonic()
            if slp > 0: time.sleep(slp)
            else: nxt = time.monotonic()
    except KeyboardInterrupt:
        print("\n[Ctrl-C] stopping")
    except RuntimeError as e:
        print(f"\n[abort] {e}")
    except Exception as e:
        print(f"\n[unexpected] {type(e).__name__}: {e}")
    finally:
        print("Disabling...")
        try:
            arm.disable_all()
        except Exception:
            pass
        arm.close()
        print("Done.")


if __name__ == "__main__":
    main()
