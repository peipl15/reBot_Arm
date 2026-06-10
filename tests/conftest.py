"""Shared pytest fixtures."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from arm.config import ArmConfig, JointConfig, WatchdogConfig


@pytest.fixture
def watchdog_cfg() -> WatchdogConfig:
    return WatchdogConfig(
        torque_warn_ratio=0.6,
        torque_abort_ratio=0.8,
        tracking_error_rad=0.05,
        tracking_error_timeout_s=0.5,
        status_must_be=1,
    )


@pytest.fixture
def j2_config() -> JointConfig:
    """Joint 2 — reversed sign, big motor (28 Nm)."""
    return JointConfig(
        index=2, id=0x02, feedback_id=0x12, name="shoulder", model="4340",
        sign=-1, pmax_fw=12.5, vmax_fw=10.0, tmax_fw=28.0,
        soft_min=-0.5, soft_max=+0.5, vlim_default=0.5,
    )


@pytest.fixture
def j5_config() -> JointConfig:
    """Joint 5 — normal sign, small motor (10 Nm)."""
    return JointConfig(
        index=5, id=0x05, feedback_id=0x15, name="wrist_1", model="4340",
        sign=+1, pmax_fw=12.5, vmax_fw=30.0, tmax_fw=10.0,
        soft_min=-1.0, soft_max=+1.0, vlim_default=1.0,
    )


@pytest.fixture
def sample_cfg(watchdog_cfg, j2_config, j5_config) -> ArmConfig:
    return ArmConfig(
        transport="dm-device",
        dm_device_type="usb2canfd-dual",
        dm_channel="0",
        feedback_dt_ms=20,
        watchdog=watchdog_cfg,
        joints=(j2_config, j5_config),
    )


def make_state(pos: float = 0.0, vel: float = 0.0,
               torq: float = 0.0, status_code: int = 1) -> SimpleNamespace:
    """Build a minimal feedback-state stand-in."""
    return SimpleNamespace(pos=pos, vel=vel, torq=torq, status_code=status_code)


@pytest.fixture
def make_state_fn():
    return make_state


@pytest.fixture
def mock_mb_ctrl():
    """Build a Mock motorbridge.Controller.

    add_damiao_motor returns a fresh Mock motor each call with
    .send_pos_vel, .request_feedback, .get_state, .enable, .disable,
    .ensure_mode (all MagicMock); get_state() defaults to a healthy state
    (pos=0, vel=0, torq=0, status=1) — override per test with
    `mock_motor.get_state.return_value = make_state(...)`.
    """
    mb_ctrl = MagicMock(name="mb_ctrl")

    def _add_motor(motor_id, fb_id, model):
        m = MagicMock(name=f"motor_0x{motor_id:02x}")
        m.get_state.return_value = make_state()
        return m

    mb_ctrl.add_damiao_motor.side_effect = _add_motor
    return mb_ctrl
