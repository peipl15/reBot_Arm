# rebot-arm

Python wrapper for the Seeed reBot 7-DOF arm with Damiao DM motors, on top of
[motorbridge](https://motorbridge.seeedstudio.com/) using the DM-USB2FDCAN
adapter via `dm-device` transport.

## Hardware

- 7x Damiao DM-J4340 motors, IDs `0x01..0x07`, feedback IDs `0x11..0x17`
- DaMiao DM-USB2FDCAN adapter (single-channel CAN-FD, motor bus at 5 Mbps data phase)

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
