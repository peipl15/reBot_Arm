#!/usr/bin/env python3
"""Collect STATIC multi-pose holding-torque data for zero-offset + mass calibration.

Path A of the gravity-comp fix: our encoder zero != URDF q=0. We hand-guide the
arm to a handful of well-spread static poses; at each pose the held joint torque
equals the gravity torque g(q) (v=0, a=0 -> no Coriolis/inertia). fit_zero_mass.py
then solves the per-joint zero offsets (and per-link density scales) so Pinocchio's
g(q_urdf) matches these measurements.

Stiction handling: a single static reading is g(q) +/- Coulomb friction (the joint
can rest anywhere in its stiction band). For each gravity joint we therefore read
the holding torque at q0+delta AND q0-delta and average -- Coulomb friction flips
sign with the approach direction, so the mean cancels it and leaves clean g(q0).

Usage:
  cd ~/code/rebot_arm && source .venv/bin/activate && \
  export LD_LIBRARY_PATH=$HOME/.cache/motorbridge/libdeps && \
  python3 scripts/calib_zero_poses.py --out outputs/zero_poses.csv

Pose advice (aim for 10-15, spread the gravity joints):
  - shoulder j2 at several angles (arm high / horizontal / dropped)
  - elbow j3 bent at a few angles with the upper arm both horizontal and raised
  - wrist-pitch j4 pitched up/down with the forearm horizontal
  - a couple poses twisting j5 with the wrist pitched (gives j5 some load)
Avoid resting against a hard end-stop (that adds contact torque, not gravity).
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
from motorbridge.errors import CallError

GRAV_JOINTS = [2, 3, 4, 5]   # joints we bracket-average (carry gravity load)


def enable_all_robust(arm, tries=6, pause=0.2):
    """enable_all() that survives the intermittent Damiao 'register write ack
    not received within 50ms' on the mode-set. Re-enabling already-enabled
    motors is idempotent, so a whole-arm retry is safe; clear_error each pass."""
    for k in range(tries):
        try:
            arm.enable_all()
            return
        except CallError as e:
            for j in arm.cfg.joints:
                try:
                    arm._motor(j.index).clear_error()
                except Exception:
                    pass
            if k == tries - 1:
                raise
            print(f"  ⚠ enable 重试 {k+1}/{tries} ({e})")
            time.sleep(pause)


def save_csv(path: Path, poses: list):
    if not poses:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(poses[0].keys()))
        w.writeheader()
        w.writerows(poses)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="outputs/zero_poses.csv")
    p.add_argument("--dither", type=float, default=0.02,
                   help="rad. +/- bracket amplitude per gravity joint (default 0.02 ~ 1.1 deg).")
    p.add_argument("--settle", type=float, default=0.35, help="s to settle before reading torque.")
    p.add_argument("--vlim", type=float, default=0.3, help="rad/s for the small bracket moves.")
    return p.parse_args()


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parent.parent
    cfg = load_config(repo / "configs" / "joint_config.yaml")
    arm = ArmController(cfg)
    arm.open()
    out_path = (repo / args.out) if not os.path.isabs(args.out) else Path(args.out)
    # Resume: keep any poses already collected (crash-safe; we rewrite per pose).
    poses = list(csv.DictReader(open(out_path))) if out_path.exists() else []
    if poses:
        print(f"续采:已有 {len(poses)} 个姿态({out_path}),从 idx {len(poses)} 继续。")
    try:
        # Start LIMP so the arm can be hand-guided. Motors are only energized
        # briefly (POS_VEL lock) at each pose to read holding torque, then
        # released again. The arm DROOPS under gravity whenever it is limp --
        # keep a hand on it.
        arm.disable_all()
        time.sleep(0.2)
        print("\n=== 手摆 + POS 锁定采集 ===")
        print("电机现已卸力(可手动掰动)。每个姿态:")
        print("  1) 把臂搬到姿态并【扶住】 -> 回车")
        print("  2) 脚本锁定当前位置,提示后【松手】,自动测力矩")
        print("  3) 再次卸力,臂会下垂【请扶住】,搬向下一姿态")
        print("输入 q 回车结束。\n")
        idx = len(poses)
        while True:
            cmd = input(f"[pose {idx}] 扶住姿态后回车采集 / q 结束: ").strip().lower()
            if cmd == "q":
                break
            # 1) lock current physical pose in POS_VEL
            enable_all_robust(arm)
            time.sleep(0.1)
            q0 = {j: arm.read_joint(j, settle_s=0.03)["pos_user"] for j in range(1, 8)}
            for _ in range(int(0.4 / 0.02)):           # clamp to where the user is holding
                for j in range(1, 8):
                    arm.move_joint(j, q0[j], vlim=args.vlim, hold_s=0.0)
                time.sleep(0.02)
            print("  已锁定 -> 请【松手】... ", end="", flush=True)
            time.sleep(1.5)                             # user releases; motor holds
            print("测量中")
            # 2) bracket-read each gravity joint: avg of +delta and -delta holding torque
            tau = {}
            for j in range(1, 8):
                tau[j] = arm.read_joint(j, settle_s=0.03)["torq"]   # steady default
            for j in GRAV_JOINTS:
                arm.move_joint(j, q0[j] + args.dither, vlim=args.vlim, hold_s=0.0)
                time.sleep(args.settle)
                tp = arm.read_joint(j, settle_s=0.03)["torq"]
                arm.move_joint(j, q0[j] - args.dither, vlim=args.vlim, hold_s=0.0)
                time.sleep(args.settle)
                tm = arm.read_joint(j, settle_s=0.03)["torq"]
                arm.move_joint(j, q0[j], vlim=args.vlim, hold_s=0.0)  # restore
                tau[j] = 0.5 * (tp + tm)
            rec = {"idx": idx}
            rec.update({f"q{j}": q0[j] for j in range(1, 7)})
            rec.update({f"tau{j}": tau[j] for j in range(1, 7)})
            poses.append(rec)
            save_csv(out_path, poses)                    # crash-safe: persist every pose
            print("  q(rad): " + " ".join(f"j{j}={q0[j]:+.3f}" for j in range(1, 7)))
            print("  tau   : " + " ".join(f"j{j}={tau[j]:+.2f}" for j in range(1, 7))
                  + f"   (j2-5 已±{args.dither:.2f}消摩擦)")
            idx += 1
            # release for the next pose -- ARM WILL DROOP, keep a hand on it
            print("  >>> 即将卸力,臂会下垂,请【扶住】 <<<")
            time.sleep(0.8)
            arm.disable_all()
            time.sleep(0.2)
    finally:
        arm.disable_all()
        arm.close()
        if poses:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(poses[0].keys()))
                w.writeheader()
                w.writerows(poses)
            print(f"\n已保存 {len(poses)} 个姿态 -> {out_path}")
        else:
            print("\n未采集任何姿态。")


if __name__ == "__main__":
    main()
