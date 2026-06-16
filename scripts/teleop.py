#!/usr/bin/env python3
"""Joint-space teleoperation: reBot Arm 102 leader → reBot Arm B601 DM follower.

Same kinematic structure on both sides → 1:1 joint mapping, no IK. Per-joint
leader→follower sign accounts for the two arms having opposite rotation
direction on j3 (elbow) and j7 (gripper) based on Seeed's joint_ranges in
config_rebot_arm_102_leader.py:

  Seeed leader joint_ranges:
    shoulder_pan   (-150,  +150)  symmetric
    shoulder_lift  ( -1,   +170)  + = drop with gravity (matches follower +)
    elbow_flex     (-200,   +1)   + = nearly extended, leader bends in -
    wrist_flex     ( -80,   +90)
    wrist_yaw      ( -90,   +90)
    wrist_roll     ( -90,   +90)
    gripper        ( 0,    +270)  + = open

  Follower (Phase 2, after sign flip):
    j1 base        soft  [-1.50, +2.05]
    j2 shoulder    soft  [ 0.00, +1.57]  + = drop with gravity
    j3 elbow       soft  [ 0.00, +1.57]  + = bend with gravity (sign -1 → leader -X)
    j4 wrist_flex  soft  [ 0.00, +1.57]
    j5 wrist_yaw   soft  [-1.48, +1.48]
    j6 wrist_roll  soft  [-3.05, +3.05]
    j7 gripper     soft  [-6.00, +0.04]  + = close (sign -1 → leader +X opens)

Safety:
  - vlim 0.5 rad/s on all joints first (override via --vlim)
  - move_joint clamps target_user to soft limits automatically
  - Ctrl-C / KeyboardInterrupt → disable_all in finally
  - State watchdog every 1s: aborts if any joint status_code != 1 or
    |torque| > 0.8 * tmax
"""
from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import time
from pathlib import Path

if "libdeps" not in os.environ.get("LD_LIBRARY_PATH", ""):
    print("ERROR: export LD_LIBRARY_PATH=$HOME/.cache/motorbridge/libdeps:$LD_LIBRARY_PATH first")
    sys.exit(1)

from fashionstar_uart_sdk.uart_pocket_handler import PortHandler

from arm import ArmController, ArmSafetyError, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = REPO_ROOT / "configs" / "joint_config.yaml"

# Leader servo id → (follower joint index, leader→follower sign, scale)
#
# scale ≠ 1.0 means leader degrees don't map 1:1 to follower radians at
# the deg2rad ratio. Currently only j7 needs this:
#
# Empirical leader gripper max measured on 2026-06-16: +54.60° (NOT the
# +270° declared in Seeed's joint_ranges config — hardware actual range is
# shorter). Follower j7 max open is -6.00 rad. To make leader full-open
# hit follower full-open:
#   scale = 6.00 / (54.60° × π/180) ≈ 6.30
# At leader 0° follower target = 0 (≈ closed). At leader 54.60° follower
# target = -6.00 (fully open).
#
# Sign discovered empirically on 2026-06-16 first teleop test:
#   - j1 base and j4 wrist_flex went the wrong direction with my initial
#     guess of +1 based on the Seeed joint_ranges symmetry — flipped to -1.
#   - j3 elbow and j7 gripper were already -1 from joint_ranges analysis
#     (leader's negative range matches follower's positive bend; leader +
#     opens gripper which corresponds to follower -).
JOINT_MAP = {
    0: (1, -1, 1.000),   # shoulder_pan → base
    1: (2, +1, 1.000),   # shoulder_lift → shoulder
    2: (3, -1, 1.000),   # elbow_flex → elbow
    3: (4, -1, 1.000),   # wrist_flex → wrist_flex
    4: (5, +1, 1.000),   # wrist_yaw → wrist_yaw
    5: (6, +1, 1.000),   # wrist_roll → wrist_roll
    6: (7, -1, 6.30),    # gripper: leader 0..+54.6° → follower 0..-6.00 rad
}
SERVO_IDS = list(JOINT_MAP.keys())

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--leader-port", default="/dev/ttyUSB0")
    p.add_argument("--leader-baud", type=int, default=1_000_000)
    p.add_argument("--vlim", type=float, default=0.5,
                   help="velocity limit applied to non-gripper joints (rad/s)")
    p.add_argument("--gripper-vlim", type=float, default=2.0,
                   help="velocity limit for the gripper j7 (rad/s). Faster "
                        "than arm joints so close/open intent tracks tightly.")
    p.add_argument("--gripper-contact-torque", type=float, default=1.0,
                   help="|torq| (Nm) on j7 above which we consider the gripper "
                        "to have contacted an object.")
    p.add_argument("--gripper-overpress-deg", type=float, default=2.0,
                   help="additional close motion allowed after contact (deg).")
    p.add_argument("--rate", type=float, default=50.0,
                   help="control loop frequency (Hz)")
    p.add_argument("--print-every", type=float, default=1.0,
                   help="print follow status every N seconds")
    p.add_argument("--torque-abort-ratio", type=float, default=0.80,
                   help="abort if any joint |torq| > ratio * tmax_fw")
    p.add_argument("--no-wait", action="store_true",
                   help="skip 'press Enter to start' confirmation")
    return p.parse_args()


def read_leader_degrees(handler: PortHandler) -> dict[int, float] | None:
    """Read all 7 leader servo angles. Returns dict {servo_id: degrees} or None."""
    try:
        data = handler.sync_read["Monitor"](SERVO_IDS)
    except Exception as e:
        print(f"  [leader read error] {e}")
        return None
    out = {}
    for sid in SERVO_IDS:
        state = data.get(sid)
        if state is None or state.current_position is None:
            return None
        out[sid] = float(state.current_position)
    return out


def main():
    args = parse_args()
    cfg = load_config(CFG_PATH)

    # Open leader
    print(f"Opening leader on {args.leader_port} @ {args.leader_baud}...")
    handler = PortHandler(args.leader_port, args.leader_baud)
    handler.openPort()
    for sid in SERVO_IDS:
        if not handler.ping(sid):
            print(f"  WARNING: servo {sid} did not ping")

    # Initial leader read
    initial = read_leader_degrees(handler)
    if initial is None:
        print("ERROR: could not read all leader joints, aborting")
        sys.exit(1)
    print("\nInitial leader pose (deg):")
    for sid in SERVO_IDS:
        joint_idx, sign, scale = JOINT_MAP[sid]
        leader_deg = initial[sid]
        target_user = sign * leader_deg * DEG2RAD * scale
        jcfg = cfg.joint(joint_idx)
        clamped = max(jcfg.soft_min, min(jcfg.soft_max, target_user))
        clamp_note = "" if clamped == target_user else f"  [CLAMPED → {clamped:+.3f}]"
        print(f"  servo {sid} {jcfg.name:<12} = {leader_deg:+6.1f}°  "
              f"→ j{joint_idx} target_user={target_user:+.3f} rad{clamp_note}")

    # Open follower
    print("\nOpening follower (motorbridge dm-device)...")
    arm = ArmController(cfg)
    arm.open()

    # Enable all 7 joints
    print("Enabling all 7 follower joints...")
    arm.enable_all()
    time.sleep(0.3)

    if not args.no_wait:
        print(f"\n=== READY ===  vlim={args.vlim} rad/s  rate={args.rate} Hz")
        print("    Confirm:")
        print("    - workspace clear")
        print("    - leader near 'home pose' (small deltas only)")
        print("    - hand near power switch / Ctrl-C")
        try:
            input("    Press ENTER to start teleop (Ctrl-C anytime to stop): ")
        except (KeyboardInterrupt, EOFError):
            print("\n  aborted before start")
            arm.disable_all()
            arm.close()
            return

    print("\n=== Teleop running. Ctrl-C to stop. ===\n")

    interval = 1.0 / args.rate
    next_tick = time.monotonic()
    next_print = time.monotonic()
    iteration = 0
    last_leader_deg: dict[int, float] = {}

    # Gripper contact-aware closing: once j7 torque exceeds the threshold
    # (gripper has pinched something), latch the contact position and
    # forbid commanding more than overpress_rad beyond it. Reset when leader
    # opens past the contact point.
    GRIPPER_JOINT = 7
    overpress_rad = args.gripper_overpress_deg * DEG2RAD
    contact_pos: float | None = None      # follower pos_user where contact was detected
    j7_motor = arm._motors[GRIPPER_JOINT]  # for fast torque polling

    def j7_state_fast():
        """Get j7 state without the 30ms settle that read_joint uses."""
        try:
            j7_motor.request_feedback()
        except Exception:
            return None
        st = j7_motor.get_state()
        if st is None:
            return None
        j = cfg.joint(GRIPPER_JOINT)
        return {
            "pos_user": st.pos * j.sign,
            "torq": st.torq,
            "status_code": st.status_code,
        }

    try:
        while True:
            now = time.monotonic()

            # 1) Read leader
            leader_deg = read_leader_degrees(handler)
            if leader_deg is None:
                # transient read miss — keep previous targets, just delay
                time.sleep(0.01)
                continue
            last_leader_deg = leader_deg

            # 2) Send commands to all 7 follower joints
            #    Special handling for j7:
            #    - separate vlim
            #    - contact-aware overpress clamp
            for sid in SERVO_IDS:
                joint_idx, sign, scale = JOINT_MAP[sid]
                target_user = sign * leader_deg[sid] * DEG2RAD * scale

                if joint_idx == GRIPPER_JOINT:
                    # Contact detection / release
                    st = j7_state_fast()
                    actual = st["pos_user"] if st else None
                    torq = st["torq"] if st else 0.0

                    if contact_pos is None:
                        # No active contact. If torque crosses threshold AND
                        # we were trying to close (target > actual), latch.
                        if (
                            actual is not None
                            and abs(torq) >= args.gripper_contact_torque
                            and target_user > actual
                        ):
                            contact_pos = actual
                    else:
                        # Active contact. Release if leader is now commanding
                        # to open well below the contact position.
                        if target_user < contact_pos - 2.0 * DEG2RAD:
                            contact_pos = None

                    if contact_pos is not None:
                        # Clamp: don't allow closing past contact + overpress.
                        max_close = contact_pos + overpress_rad
                        if target_user > max_close:
                            target_user = max_close

                    arm.move_joint(joint_idx, target_user,
                                   vlim=args.gripper_vlim, hold_s=0.0)
                else:
                    arm.move_joint(joint_idx, target_user,
                                   vlim=args.vlim, hold_s=0.0)
            iteration += 1

            # 3) Periodic status print + torque/status watchdog
            if now >= next_print:
                bad = []
                rows = []
                for sid in SERVO_IDS:
                    joint_idx, sign, scale = JOINT_MAP[sid]
                    target_user = sign * leader_deg[sid] * DEG2RAD * scale
                    s = arm.read_joint(joint_idx, settle_s=0.0)  # no extra delay
                    jcfg = cfg.joint(joint_idx)
                    actual = s["pos_user"] if s else float("nan")
                    torq = s["torq"] if s else float("nan")
                    status = s["status_code"] if s else -1
                    err = abs(target_user - actual) if s else float("nan")
                    rows.append(f"j{joint_idx:<2} L={leader_deg[sid]:+6.1f}°  "
                                f"tgt={target_user:+6.3f}  act={actual:+6.3f}  "
                                f"err={err:.3f}  torq={torq:+5.1f}  st={status}")
                    # Watchdog
                    if status != 1:
                        bad.append(f"j{joint_idx} status={status}")
                    if abs(torq) > args.torque_abort_ratio * jcfg.tmax_fw:
                        bad.append(f"j{joint_idx} torq={torq:.1f} > "
                                   f"{args.torque_abort_ratio*jcfg.tmax_fw:.1f}")
                contact_str = ""
                if contact_pos is not None:
                    contact_str = (f"  [j7 CONTACT @ {contact_pos:+.3f}rad, "
                                   f"max_close={contact_pos+overpress_rad:+.3f}]")
                print(f"[iter {iteration:>5}]{contact_str}")
                for r in rows:
                    print(f"  {r}")
                if bad:
                    print(f"!! WATCHDOG TRIP: {', '.join(bad)}")
                    raise RuntimeError("watchdog")
                next_print = now + args.print_every

            # 4) Sleep to maintain rate
            next_tick += interval
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # We're falling behind; reset
                next_tick = time.monotonic()

    except KeyboardInterrupt:
        print("\n[Ctrl-C] stopping")
    except RuntimeError as e:
        print(f"\n[abort] {e}")
    except Exception as e:
        print(f"\n[unexpected] {type(e).__name__}: {e}")
    finally:
        print("Disabling follower...")
        try:
            arm.disable_all()
        except Exception:
            pass
        arm.close()
        print("Done.")


if __name__ == "__main__":
    main()
