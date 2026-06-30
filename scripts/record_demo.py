#!/usr/bin/env python3
"""Demo recording: teleop + sync capture of arm state + 3 camera streams.

Runs the same leader → follower teleop loop as scripts/teleop.py, but every
loop iteration also captures:
  - leader joint angles (7 servos, deg)
  - follower joint state (7 motors: pos_user/vel/torq, rad)
  - commanded action (what we sent to follower this step)
  - third-person C922 RGB frame (BGR uint8)
  - wrist RealSense D435 RGB frame (BGR uint8)
  - wrist RealSense D435 depth frame (uint16, mm)
  - wall-clock + episode-relative timestamps

Keyboard (on the OpenCV preview window):
  SPACE  start / stop recording an episode (buffer in RAM, saved at stop)
  D      discard the in-progress episode (do not save)
  Q      quit

Each saved episode goes to:
  outputs/demos/<task>/episode_NNNNNN.hdf5

Storage:
  Per frame at default resolution (640x480 all three streams):
    C922 BGR        :  0.92 MB raw, ~0.3 MB gzipped
    D435 BGR        :  0.92 MB raw, ~0.3 MB gzipped
    D435 depth u16  :  0.61 MB raw, ~0.25 MB gzipped
    arm state       :  ~1 KB
  At 20 Hz × 30 s = 600 frames ≈ 1.5 GB RAM during recording, ≈ 0.5 GB on disk.
  Buffer is in RAM until the episode ends, so don't make episodes too long.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

if "libdeps" not in os.environ.get("LD_LIBRARY_PATH", ""):
    print("ERROR: export LD_LIBRARY_PATH=$HOME/.cache/motorbridge/libdeps:$LD_LIBRARY_PATH first")
    sys.exit(1)

import cv2
import h5py
import numpy as np
import pyrealsense2 as rs

from fashionstar_uart_sdk.uart_pocket_handler import PortHandler

from arm import ArmController, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = REPO_ROOT / "configs" / "joint_config.yaml"
DEMOS_DIR = REPO_ROOT / "outputs" / "demos"

# Same leader → follower mapping as scripts/teleop.py (see that file for context).
JOINT_MAP = {
    0: (1, -1, 1.000),
    1: (2, +1, 1.000),
    2: (3, -1, 1.000),
    3: (4, -1, 1.000),
    4: (5, +1, 1.000),
    5: (6, -1, 1.000),   # wrist_roll: flipped on 2026-06-17 during record_demo test
    6: (7, -1, 6.30),
}
SERVO_IDS = list(JOINT_MAP.keys())
DEG2RAD = math.pi / 180.0
GRIPPER_JOINT = 7


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True,
                   help="task name; episodes saved to outputs/demos/<task>/")
    p.add_argument("--leader-port", default="/dev/ttyUSB0")
    p.add_argument("--leader-baud", type=int, default=1_000_000)
    p.add_argument("--c922-index", type=int, default=-1,
                   help="/dev/videoN index for the C922 webcam; -1 = auto-find by name")
    p.add_argument("--c922-width", type=int, default=640)
    p.add_argument("--c922-height", type=int, default=480)
    p.add_argument("--c922-rotate", type=int, default=0, choices=[0, 90, 180, 270],
                   help="rotate C922 frames before recording (degrees clockwise). "
                        "Use when the camera is physically mounted in portrait "
                        "orientation. Rotation is applied AFTER capture but BEFORE "
                        "save, so the HDF5 stores the upright image and downstream "
                        "training code never has to know.")
    p.add_argument("--rs-width", type=int, default=640)
    p.add_argument("--rs-height", type=int, default=480)
    p.add_argument("--rate", type=float, default=20.0,
                   help="recording loop frequency (Hz). 20 is a sane default for DP.")
    p.add_argument("--vlim", type=float, default=0.5,
                   help="vlim for arm joints (rad/s)")
    p.add_argument("--gripper-vlim", type=float, default=2.0)
    p.add_argument("--gripper-contact-torque", type=float, default=1.0)
    p.add_argument("--gripper-overpress-deg", type=float, default=2.0)
    p.add_argument("--no-display", action="store_true",
                   help="disable cv2 preview window (still records via stdin r/s/d/q)")
    p.add_argument("--save-depth", action="store_true",
                   help="save D435 depth stream to HDF5 (gzip-compressed uint16). "
                        "Off by default — vanilla DP only uses RGB and depth is "
                        "60+ MB per 40s episode.")
    p.add_argument("--jpeg-quality", type=int, default=90,
                   help="JPEG quality (1-100) for the RGB streams. 90 is a sane "
                        "default — barely visible compression artifacts, ~10x "
                        "smaller than gzip-on-raw, and encode is also faster.")
    return p.parse_args()


def build_display(c922, rs_rgb, rs_depth, recording, n_frames, iter_n, fps,
                  show_depth):
    """Build a side-by-side preview with status overlay."""
    tiles = []
    if c922 is not None:
        tiles.append(cv2.resize(c922, (480, 360)))
    if rs_rgb is not None:
        tiles.append(cv2.resize(rs_rgb, (480, 360)))
    if show_depth and rs_depth is not None:
        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(rs_depth, alpha=0.03), cv2.COLORMAP_JET
        )
        tiles.append(cv2.resize(depth_vis, (480, 360)))
    if not tiles:
        return np.zeros((360, 480, 3), dtype=np.uint8)
    display = np.hstack(tiles)
    color = (0, 0, 255) if recording else (200, 200, 200)
    label = f"REC  n={n_frames}" if recording else "IDLE"
    cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(display, f"iter {iter_n}  fps {fps:.1f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(display, "SPACE: start/stop  D: discard  Q: quit",
                (10, display.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return display


def save_episode(task_dir: Path, idx: int, args, buffer):
    """Write the buffered frames out to HDF5.

    RGB images stored as JPEG-encoded variable-length byte arrays (much
    smaller AND faster than gzip-on-raw). Depth, if enabled, stored as
    gzip-compressed uint16. Joint timeseries stored as float arrays.

    To load an RGB frame at index i:
        with h5py.File(path) as f:
            buf = np.frombuffer(f['third_person'][i], dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    """
    path = task_dir / f"episode_{idx:06d}.hdf5"
    n = len(buffer)
    if n == 0:
        return

    with h5py.File(path, "w") as f:
        f.attrs["task"] = args.task
        f.attrs["episode_idx"] = idx
        f.attrs["rate_hz"] = args.rate
        f.attrs["n_frames"] = n
        f.attrs["recorded_at"] = datetime.now().isoformat()
        f.attrs["c922_resolution"] = f"{args.c922_width}x{args.c922_height}"
        f.attrs["rs_resolution"] = f"{args.rs_width}x{args.rs_height}"
        f.attrs["rgb_format"] = "jpeg"
        f.attrs["jpeg_quality"] = args.jpeg_quality
        f.attrs["has_depth"] = bool(args.save_depth)

        # scalar / 7-channel timeseries
        f.create_dataset("ts", data=np.array([b["ts"] for b in buffer], dtype=np.float64))
        f.create_dataset("leader_deg",
                         data=np.array([b["leader_deg"] for b in buffer], dtype=np.float32))
        f.create_dataset("follower_pos",
                         data=np.array([b["follower_pos"] for b in buffer], dtype=np.float32))
        f.create_dataset("follower_vel",
                         data=np.array([b["follower_vel"] for b in buffer], dtype=np.float32))
        f.create_dataset("follower_torq",
                         data=np.array([b["follower_torq"] for b in buffer], dtype=np.float32))
        f.create_dataset("action",
                         data=np.array([b["action"] for b in buffer], dtype=np.float32))

        # JPEG-encoded RGB streams as variable-length uint8 arrays.
        # h5py.vlen_dtype is the current API; special_dtype is the deprecated alias.
        vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
        jpg_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]

        def jpeg_encode_stack(key):
            ds = f.create_dataset(key, (n,), dtype=vlen_uint8)
            for i, b in enumerate(buffer):
                frame = b[key]
                if frame is None:
                    ds[i] = np.array([], dtype=np.uint8)
                else:
                    ok, jpg = cv2.imencode(".jpg", frame, jpg_params)
                    ds[i] = np.frombuffer(jpg.tobytes(), dtype=np.uint8) if ok else \
                            np.array([], dtype=np.uint8)

        jpeg_encode_stack("third_person")
        jpeg_encode_stack("wrist_rgb")

        # Depth (optional, still gzip)
        if args.save_depth:
            first_depth = next((b["wrist_depth"] for b in buffer
                                if b["wrist_depth"] is not None), None)
            if first_depth is not None:
                shape = (n,) + first_depth.shape
                arr = np.zeros(shape, dtype=np.uint16)
                for i, b in enumerate(buffer):
                    if b["wrist_depth"] is not None:
                        arr[i] = b["wrist_depth"]
                f.create_dataset("wrist_depth", data=arr,
                                 compression="gzip", compression_opts=4,
                                 chunks=(1,) + first_depth.shape)

    size_mb = path.stat().st_size / 1e6
    print(f"  saved {path.name} ({size_mb:.1f} MB)")


def main():
    args = parse_args()
    cfg = load_config(CFG_PATH)

    task_dir = DEMOS_DIR / args.task
    task_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(task_dir.glob("episode_*.hdf5"))
    next_idx = int(existing[-1].stem.split("_")[1]) + 1 if existing else 0
    print(f"Task: {args.task}; next episode idx: {next_idx} "
          f"({len(existing)} already in {task_dir})")

    # Leader
    print(f"\nOpening leader on {args.leader_port}...")
    leader = PortHandler(args.leader_port, args.leader_baud)
    leader.openPort()
    missing = []
    for sid in SERVO_IDS:
        if not leader.ping(sid):
            missing.append(sid)
    if missing:
        print(f"  WARNING: leader servos {missing} did not ping")

    # Follower
    print("Opening follower (motorbridge dm-device)...")
    arm = ArmController(cfg)
    arm.open()
    arm.enable_all()
    time.sleep(0.3)

    # C922
    if args.c922_index < 0:
        from camera_utils import find_camera_index
        idx = find_camera_index("C922")
        if idx is None:
            raise RuntimeError("C922 not found by name (is it plugged in?)")
        args.c922_index = idx
        print(f"Auto-detected C922 at /dev/video{idx}")
    print(f"Opening C922 on /dev/video{args.c922_index} at "
          f"{args.c922_width}x{args.c922_height}...")
    cap = cv2.VideoCapture(args.c922_index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.c922_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.c922_height)
    if not cap.isOpened():
        raise RuntimeError(f"could not open C922 (/dev/video{args.c922_index})")

    # RealSense — hardware_reset first to clear any stuck pipeline state from
    # a previous Python session that didn't shut down cleanly. Without this
    # the next wait_for_frames will time out indefinitely.
    print(f"Opening RealSense D435 at {args.rs_width}x{args.rs_height} BGR+depth...")
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        raise RuntimeError("no RealSense device found")
    print(f"  hardware_reset() on {devs[0].get_info(rs.camera_info.name)} "
          f"SN={devs[0].get_info(rs.camera_info.serial_number)} ...")
    for d in devs:
        d.hardware_reset()
    time.sleep(3.0)  # wait for re-enumeration

    rs_pipe = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_stream(rs.stream.color, args.rs_width, args.rs_height, rs.format.bgr8, 30)
    rs_cfg.enable_stream(rs.stream.depth, args.rs_width, args.rs_height, rs.format.z16, 30)
    rs_pipe.start(rs_cfg)
    # warmup (auto-exposure)
    for _ in range(10):
        rs_pipe.wait_for_frames(timeout_ms=2000)

    print("\n=== READY ===")
    print("  SPACE = start/stop recording an episode")
    print("  D     = discard the in-progress episode")
    print("  Q     = quit")
    print()

    interval = 1.0 / args.rate
    next_tick = time.monotonic()
    iteration = 0
    recording = False
    buffer: list[dict] = []
    episode_start_t: float | None = None

    overpress_rad = args.gripper_overpress_deg * DEG2RAD
    contact_pos: float | None = None

    j7_motor = arm._motors[GRIPPER_JOINT]
    follower_motors = [arm._motors[i] for i in range(1, 8)]
    follower_signs = [cfg.joint(i).sign for i in range(1, 8)]

    fps_window = []  # short rolling for display fps

    try:
        while True:
            loop_t0 = time.monotonic()
            wall_t = time.time()

            # 1) Leader (degrees)
            try:
                ld = leader.sync_read["Monitor"](SERVO_IDS)
                leader_deg = {sid: float(ld.get(sid).current_position)
                              for sid in SERVO_IDS
                              if ld.get(sid) is not None
                              and ld.get(sid).current_position is not None}
                if len(leader_deg) != len(SERVO_IDS):
                    raise RuntimeError("partial leader read")
            except Exception:
                time.sleep(0.005)
                continue

            # 2) Cameras (read in parallel-ish — both return latest)
            ok, c922_frame = cap.read()
            if not ok:
                c922_frame = None
            elif args.c922_rotate == 90:
                c922_frame = cv2.rotate(c922_frame, cv2.ROTATE_90_CLOCKWISE)
            elif args.c922_rotate == 180:
                c922_frame = cv2.rotate(c922_frame, cv2.ROTATE_180)
            elif args.c922_rotate == 270:
                c922_frame = cv2.rotate(c922_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            try:
                fs = rs_pipe.wait_for_frames(timeout_ms=100)
                color = fs.get_color_frame()
                depth = fs.get_depth_frame()
                rs_rgb = np.asanyarray(color.get_data()) if color else None
                rs_depth = np.asanyarray(depth.get_data()) if depth else None
            except Exception:
                rs_rgb, rs_depth = None, None

            # 3) Compute and send 7 follower commands
            action = [0.0] * 7
            for sid in SERVO_IDS:
                joint_idx, sign, scale = JOINT_MAP[sid]
                target = sign * leader_deg[sid] * DEG2RAD * scale

                if joint_idx == GRIPPER_JOINT:
                    # contact-aware closing (mirrors scripts/teleop.py)
                    j7_motor.request_feedback()
                    s = j7_motor.get_state()
                    if s is not None:
                        actual = s.pos * cfg.joint(GRIPPER_JOINT).sign
                        torq = s.torq
                        if contact_pos is None:
                            if abs(torq) >= args.gripper_contact_torque and target > actual:
                                contact_pos = actual
                        else:
                            if target < contact_pos - 2.0 * DEG2RAD:
                                contact_pos = None
                        if contact_pos is not None:
                            max_close = contact_pos + overpress_rad
                            if target > max_close:
                                target = max_close
                    arm.move_joint(joint_idx, target,
                                   vlim=args.gripper_vlim, hold_s=0.0)
                else:
                    arm.move_joint(joint_idx, target,
                                   vlim=args.vlim, hold_s=0.0)
                action[joint_idx - 1] = target

            # 4) Read follower state (auto-broadcast feedback after each move)
            follower_pos = []
            follower_vel = []
            follower_torq = []
            for m, sgn in zip(follower_motors, follower_signs):
                s = m.get_state()
                if s is None:
                    follower_pos.append(float("nan"))
                    follower_vel.append(float("nan"))
                    follower_torq.append(float("nan"))
                else:
                    follower_pos.append(s.pos * sgn)
                    follower_vel.append(s.vel)
                    follower_torq.append(s.torq)

            # 5) Buffer if recording. Depth only buffered if --save-depth set.
            if recording:
                frame = {
                    "ts": wall_t - episode_start_t,
                    "leader_deg": [leader_deg[sid] for sid in SERVO_IDS],
                    "follower_pos": follower_pos,
                    "follower_vel": follower_vel,
                    "follower_torq": follower_torq,
                    "action": action,
                    "third_person": c922_frame.copy() if c922_frame is not None else None,
                    "wrist_rgb": rs_rgb.copy() if rs_rgb is not None else None,
                }
                if args.save_depth:
                    frame["wrist_depth"] = rs_depth.copy() if rs_depth is not None else None
                else:
                    frame["wrist_depth"] = None  # buffered as None, not saved
                buffer.append(frame)

            iteration += 1

            # 6) Display + keyboard
            if not args.no_display:
                # Track FPS over last ~30 iterations
                fps_window.append(time.monotonic())
                if len(fps_window) > 30:
                    fps_window.pop(0)
                if len(fps_window) >= 2:
                    fps = (len(fps_window) - 1) / (fps_window[-1] - fps_window[0])
                else:
                    fps = 0.0
                display = build_display(c922_frame, rs_rgb, rs_depth,
                                        recording, len(buffer), iteration, fps,
                                        show_depth=args.save_depth)
                cv2.imshow("record_demo", display)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 0xFF

            if key == ord(" "):
                if not recording:
                    recording = True
                    buffer = []
                    episode_start_t = wall_t
                    contact_pos = None
                    print(f"[REC START] episode {next_idx}")
                else:
                    print(f"[REC STOP]  saving episode {next_idx} "
                          f"({len(buffer)} frames)")
                    save_episode(task_dir, next_idx, args, buffer)
                    next_idx += 1
                    buffer = []
                    recording = False
            elif key == ord("d"):
                if recording:
                    print(f"[REC DISCARD] threw away {len(buffer)} frames")
                    recording = False
                    buffer = []
            elif key == ord("q"):
                if recording:
                    print(f"[REC DISCARD on quit] threw away {len(buffer)} frames")
                break

            # 7) Maintain rate
            next_tick += interval
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # falling behind
                next_tick = time.monotonic()

    except KeyboardInterrupt:
        print("\n[Ctrl-C] stopping")
        if recording:
            print(f"[REC DISCARD on Ctrl-C] threw away {len(buffer)} frames")
    except Exception as e:
        print(f"\n[unexpected] {type(e).__name__}: {e}")
        raise
    finally:
        print("Disabling follower + closing cameras...")
        try:
            arm.disable_all()
        except Exception:
            pass
        arm.close()
        try:
            rs_pipe.stop()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()
        print(f"Done. {next_idx} episodes saved this session "
              f"({next_idx - (int(existing[-1].stem.split('_')[1]) + 1 if existing else 0)} new).")


if __name__ == "__main__":
    main()
