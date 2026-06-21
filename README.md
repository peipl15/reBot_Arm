# rebot-arm

Python wrapper for the Seeed reBot 7-DOF arm with Damiao DM motors, on top of
[motorbridge](https://motorbridge.seeedstudio.com/) using the DM-USB2FDCAN
adapter via `dm-device` transport.

## Hardware

### Follower (the arm we control)
- 7 Damiao DM motors, IDs `0x01..0x07`, feedback IDs `0x11..0x17` — two models:
  - **j1–j3** (base, shoulder, elbow): **DM-J4340** — T_MAX 28 Nm, V_MAX 10 rad/s
  - **j4–j7** (wrist_flex, wrist_yaw, wrist_roll, gripper): **DM-J4310** — T_MAX 10 Nm, V_MAX 30 rad/s
  - The per-joint `model` in `configs/joint_config.yaml` MUST match the real model
    (4340 vs 4310): motorbridge decodes torque/velocity feedback against the
    model's T_MAX/V_MAX. Tagging a J4310 as "4340" inflates its torque readings
    2.8× (28/10) and mis-scales velocity — POS_VEL command speed is unaffected.
- DaMiao DM-USB2FDCAN adapter (single-channel CAN-FD, motor bus at 5 Mbps data phase)
- Mounted as the reBot Arm B601 DM follower (Seeed-Projects / kit-miao)

### Leader (Star Arm 102 / reBot Arm 102 leader)
- 7x Fashion Star smart servos, IDs `0..6`, mapped to follower joints j1..j7
- USB-UART (CH340 or similar) at 1 Mbps, exposed as `/dev/ttyUSB0`
- Used for joint-space teleoperation by `scripts/teleop.py`
- SDK: `fashionstar-uart-sdk` + `motorbridge-smart-servo` (declared in
  `pyproject.toml`)

### Joint-space leader → follower mapping

Same kinematic structure on both sides → 1:1 joint mapping, no IK. Per-joint
sign and scale discovered empirically on 2026-06-16 first teleop test:

| Leader servo | Follower joint | sign | scale | notes |
| --- | --- | --- | --- | --- |
| 0 (shoulder_pan)   | j1 base       | -1 | 1.000 | sign flipped on test |
| 1 (shoulder_lift)  | j2 shoulder   | +1 | 1.000 | |
| 2 (elbow_flex)     | j3 elbow      | -1 | 1.000 | from Seeed joint_ranges (leader -, follower +) |
| 3 (wrist_flex)     | j4 wrist_flex | -1 | 1.000 | sign flipped on test |
| 4 (wrist_yaw)      | j5 wrist_yaw  | +1 | 1.000 | |
| 5 (wrist_roll)     | j6 wrist_roll | +1 | 1.000 | |
| 6 (gripper)        | j7 gripper    | -1 | 6.30  | leader 0..+54.6° → follower 0..-6.00 rad (max open) |

The gripper's empirical leader range is +54.6°, NOT the +270° declared in
Seeed's `config_rebot_arm_102_leader.py` joint_ranges — that's the servo's
internal range, not what this specific arm's mechanical linkage allows.

## Setup (Ubuntu 22.04)

The motorbridge runtime needs `libstdc++` with GLIBCXX_3.4.32 and
libusb 1.0.27+, both newer than what jammy ships. The full recipe lives
in the user's memory (`project_motorbridge_damiao_setup.md`). Short version:

```bash
# Once per machine: SDK runtime + isolated libdeps
motorbridge-install-dm-device --download
# (extract conda-forge libstdcxx + libusb into ~/.cache/motorbridge/libdeps/)
# (write /etc/udev/rules.d/99-damiao-usb2canfd.rules for raw USB R/W)

# This repo
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# Every run: LD_LIBRARY_PATH for the isolated libs
export LD_LIBRARY_PATH=$HOME/.cache/motorbridge/libdeps${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

## Layout

```
arm/         wrapper code (Controller, JointConfig, safety/watchdog)
configs/     joint_config.yaml (per-joint id, sign, soft limits, vlim)
scripts/     standalone runnable scripts (characterize_range, sanity_walk, ...)
tests/       pytest unit tests (mocked motorbridge)
outputs/     calibration CSVs and run logs (gitignored)
```

## Running tests

ROS2 Humble (`/opt/ros/humble`) is on `PYTHONPATH` by default in this shell and
its `launch_testing` pytest plugin entry-point gets discovered, but its `lark`
dep is missing from this venv — pytest then fails to start. Clear PYTHONPATH
for test runs:

```bash
env -u PYTHONPATH python3 -m pytest tests/ -v
```

## Running the dry-run sanity check

```bash
python3 scripts/dry_run.py
```

No hardware contact; prints the loaded joint config in a table.

## Operational scripts (touch hardware)

All need `LD_LIBRARY_PATH` set and the .venv activated.

| Script | What it does |
| --- | --- |
| `scripts/read_motor_params.py` | One-shot dump of PMAX/VMAX/TMAX/ESC_ID/MST_ID for all 7 follower motors. Read-only, no motion. |
| `scripts/read_leader.py` | Ping all 7 leader-arm servos + tabulate angles over N samples. Quick alive-check / posture probe; the leader counterpart of `read_motor_params.py`. |
| `scripts/set_joint_zero.py` | Re-zero one follower joint at its current physical position; persists to motor flash by default. Use when a joint has been physically rotated and you want the new pose as encoder zero. |
| `scripts/characterize_range.py` | Phase 2 ramp test for a single joint — walks step-by-step until torque/status/tracking watchdog trips. Writes CSV to outputs/. |
| `scripts/sanity_walk.py` | Walks each joint to 50% of its soft limit per side and back — verifies wrapper safety + sign-flip + limits without re-characterizing. |
| `scripts/reset_arm.py` | Reads all 7 joint positions, optionally re-zeroes j1 at current, and moves j2..j7 back to encoder zero. Useful after manual adjustment / power-cycle drift. |
| `scripts/teleop.py` | Joint-space leader → follower teleop with gripper contact-aware closing. See above for sign/scale mapping. |

### Teleop quick start

```bash
# Follower (DM-USB2FDCAN) and leader (CH340 /dev/ttyUSB0) both plugged in.
python3 scripts/teleop.py
# At "Press ENTER to start" — workspace clear, leader near home — press Enter.
# Ctrl-C to disable + exit.
```

Defaults: 50 Hz loop, arm-joint vlim 0.5 rad/s, gripper vlim 2.0 rad/s.
Gripper contact: torque ≥ 1.0 Nm latches the current position; afterward
the gripper is allowed at most +2° more closing motion until the leader
opens past the contact point.
