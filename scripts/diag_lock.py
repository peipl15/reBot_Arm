#!/usr/bin/env python3
"""Diagnose why j2/j3 (DM-J4340P, high gravity load) sometimes fail to LOCK after
a disable -> re-enable cycle in the static-pose collector.

Reproduces the exact path: disable -> hand-hold a loaded pose -> enable_all_robust
-> POS_VEL lock -> watch each joint's status_code + position drift for a few
seconds. Release your hand after it locks and see whether j2/j3 sag.

Verdict per joint:
  status != 1            -> not actually enabled / comms (mode-set ack likely lost)
  status == 1, big drift -> enabled but not holding (mode didn't take POS_VEL, or effort)
  status == 1, no drift  -> healthy lock

  cd ~/code/rebot_arm && source .venv/bin/activate && \
  export LD_LIBRARY_PATH=$HOME/.cache/motorbridge/libdeps && \
  python3 scripts/diag_lock.py
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

if "libdeps" not in os.environ.get("LD_LIBRARY_PATH", ""):
    print("ERROR: export LD_LIBRARY_PATH=$HOME/.cache/motorbridge/libdeps first"); sys.exit(1)

from arm import ArmController, load_config
from motorbridge.errors import CallError


def enable_all_robust(arm, tries=6, pause=0.2):
    for k in range(tries):
        try:
            arm.enable_all(); return
        except CallError as e:
            for j in arm.cfg.joints:
                try: arm._motor(j.index).clear_error()
                except Exception: pass
            if k == tries - 1: raise
            print(f"  enable 重试 {k+1}/{tries} ({e})"); time.sleep(pause)


def main():
    repo = Path(__file__).resolve().parent.parent
    cfg = load_config(repo / "configs" / "joint_config.yaml")
    arm = ArmController(cfg); arm.open()
    for j in cfg.joints:
        try: arm._motor(j.index).set_can_timeout_ms(250)
        except Exception: pass
    try:
        arm.disable_all(); time.sleep(0.2)
        print("把臂扶到一个 j2/j3 受载的姿态(肩平举 / 肘抬起),扶住后回车。")
        input()
        # immediately after enable, report status BEFORE any motion command
        enable_all_robust(arm)
        time.sleep(0.15)
        st0 = {j: arm.read_joint(j, settle_s=0.03) for j in range(1, 8)}
        q0 = {j: st0[j]["pos_user"] for j in range(1, 8)}
        print("\n使能后(未发命令)各关节 status_code:")
        print("  " + "  ".join(f"j{j}={st0[j]['status_code']}" for j in range(1, 8)))
        print("\n开始 POS_VEL 锁定,1.5s 后请【松手】,观察 4s...")
        drift = {j: 0.0 for j in range(1, 8)}
        tq = {j: 0.0 for j in range(1, 8)}
        stat = {j: st0[j]["status_code"] for j in range(1, 8)}
        t_end = time.monotonic() + 4.0
        i = 0
        while time.monotonic() < t_end:
            for j in range(1, 8):
                arm.move_joint(j, q0[j], vlim=0.3, hold_s=0.0)
            if i % 5 == 0:
                for j in range(1, 8):
                    s = arm.read_joint(j, settle_s=0.0)
                    if s is None: continue
                    drift[j] = max(drift[j], abs(s["pos_user"] - q0[j]))
                    tq[j] = max(tq[j], abs(s["torq"]))
                    stat[j] = s["status_code"]
            time.sleep(0.02); i += 1
        print("\n  joint  status  max|drift|(rad)  max|torq|   判读")
        for j in range(1, 8):
            if stat[j] != 1:
                verdict = "❌ 未正常使能/通信(模式应答可能丢失)"
            elif drift[j] > 0.08:
                verdict = "⚠ 使能了但没锁住(模式没进POS_VEL/顶不住)"
            else:
                verdict = "✅ 锁定正常"
            mark = " <<<" if j in (2, 3) else ""
            print(f"   j{j}     {stat[j]:>2}      {drift[j]:6.3f}        {tq[j]:6.2f}   {verdict}{mark}")
        print("\n如果 j2/j3 是 ❌ status!=1 -> 重新使能时模式寄存器没设上(应在 enable 后回读校验/重设)。")
        print("如果 j2/j3 是 ⚠ drift 大但 status=1 -> POS_VEL 未生效或扭矩不足。")
    finally:
        arm.disable_all(); arm.close()
        print("\n已卸力。")


if __name__ == "__main__":
    main()
