from __future__ import annotations

import pytest

from arm.controller import ArmController, ArmSafetyError
from tests.conftest import make_state


# ---------- open / register ----------

def test_open_registers_all_joints(sample_cfg, mock_mb_ctrl):
    arm = ArmController(sample_cfg)
    arm.open(mock_mb_ctrl)
    # add_damiao_motor called once per joint, with correct (id, fb, model)
    calls = mock_mb_ctrl.add_damiao_motor.call_args_list
    assert len(calls) == 2  # j2 + j5
    ids_called = sorted(c.args[0] for c in calls)
    assert ids_called == [0x02, 0x05]
    arm.close()


def test_motor_lookup_before_open_raises(sample_cfg):
    arm = ArmController(sample_cfg)
    with pytest.raises(RuntimeError, match="not opened"):
        arm.read_joint(2)


# ---------- sign flip ----------

def test_move_joint_sign_flip_j2(sample_cfg, mock_mb_ctrl):
    """j2 has sign=-1. move_joint(2, +0.1) → motor.send_pos_vel(-0.1, ...)"""
    arm = ArmController(sample_cfg).open(mock_mb_ctrl)
    arm.move_joint(2, +0.1)  # within soft [-0.5, +0.5]
    motor2 = arm._motors[2]
    assert motor2.send_pos_vel.call_count == 1
    args, kwargs = motor2.send_pos_vel.call_args
    assert args[0] == pytest.approx(-0.1)


def test_move_joint_sign_flip_j5_unchanged(sample_cfg, mock_mb_ctrl):
    """j5 has sign=+1. move_joint(5, +0.1) → motor.send_pos_vel(+0.1, ...)"""
    arm = ArmController(sample_cfg).open(mock_mb_ctrl)
    arm.move_joint(5, +0.1)
    motor5 = arm._motors[5]
    args, _ = motor5.send_pos_vel.call_args
    assert args[0] == pytest.approx(+0.1)


def test_read_joint_sign_flip(sample_cfg, mock_mb_ctrl):
    """j2 has sign=-1. motor reports pos=-0.1 → read_joint returns pos_user=+0.1."""
    arm = ArmController(sample_cfg).open(mock_mb_ctrl)
    arm._motors[2].get_state.return_value = make_state(pos=-0.1)
    result = arm.read_joint(2, settle_s=0)
    assert result["pos_user"] == pytest.approx(+0.1)
    assert result["pos_motor"] == pytest.approx(-0.1)


def test_read_joint_returns_none_if_no_state(sample_cfg, mock_mb_ctrl):
    arm = ArmController(sample_cfg).open(mock_mb_ctrl)
    arm._motors[2].get_state.return_value = None
    assert arm.read_joint(2, settle_s=0) is None


# ---------- soft limits ----------

def test_clamp_inside_limits(sample_cfg, mock_mb_ctrl):
    arm = ArmController(sample_cfg).open(mock_mb_ctrl)
    assert arm.clamp_target(2, 0.0) == 0.0
    assert arm.clamp_target(2, 0.3) == 0.3


def test_clamp_above_max(sample_cfg, mock_mb_ctrl):
    arm = ArmController(sample_cfg).open(mock_mb_ctrl)
    # j2 soft_max=+0.5
    assert arm.clamp_target(2, +5.0) == pytest.approx(+0.5)


def test_clamp_below_min(sample_cfg, mock_mb_ctrl):
    arm = ArmController(sample_cfg).open(mock_mb_ctrl)
    assert arm.clamp_target(2, -5.0) == pytest.approx(-0.5)


def test_move_joint_silent_clamp(sample_cfg, mock_mb_ctrl):
    """Default behavior: clamp silently, no exception."""
    arm = ArmController(sample_cfg).open(mock_mb_ctrl)
    arm.move_joint(2, +5.0)  # way out of bounds
    args, _ = arm._motors[2].send_pos_vel.call_args
    # +0.5 clamped → sign=-1 → -0.5 sent to motor
    assert args[0] == pytest.approx(-0.5)


def test_move_joint_raise_on_clamp(sample_cfg, mock_mb_ctrl):
    arm = ArmController(sample_cfg).open(mock_mb_ctrl)
    with pytest.raises(ValueError, match="soft limit"):
        arm.move_joint(2, +5.0, raise_on_clamp=True)


def test_move_joint_inside_bounds_no_raise_even_with_flag(sample_cfg, mock_mb_ctrl):
    arm = ArmController(sample_cfg).open(mock_mb_ctrl)
    arm.move_joint(2, +0.3, raise_on_clamp=True)  # within [-0.5, +0.5]


# ---------- vlim ----------

def test_move_joint_default_vlim(sample_cfg, mock_mb_ctrl):
    arm = ArmController(sample_cfg).open(mock_mb_ctrl)
    arm.move_joint(2, 0.1)
    args, _ = arm._motors[2].send_pos_vel.call_args
    # j2 vlim_default=0.5
    assert args[1] == pytest.approx(0.5)


def test_move_joint_explicit_vlim(sample_cfg, mock_mb_ctrl):
    arm = ArmController(sample_cfg).open(mock_mb_ctrl)
    arm.move_joint(2, 0.1, vlim=0.2)
    args, _ = arm._motors[2].send_pos_vel.call_args
    assert args[1] == pytest.approx(0.2)


def test_move_joint_vlim_above_vmax_raises(sample_cfg, mock_mb_ctrl):
    arm = ArmController(sample_cfg).open(mock_mb_ctrl)
    # j2 vmax_fw=10
    with pytest.raises(ValueError, match="vlim"):
        arm.move_joint(2, 0.1, vlim=99.0)


def test_move_joint_vlim_zero_raises(sample_cfg, mock_mb_ctrl):
    arm = ArmController(sample_cfg).open(mock_mb_ctrl)
    with pytest.raises(ValueError, match="vlim"):
        arm.move_joint(2, 0.1, vlim=0)


# ---------- watchdog integration ----------

def test_move_with_hold_torque_trip_disables_all(sample_cfg, mock_mb_ctrl):
    """During hold, if torque exceeds threshold, all motors disabled + raise."""
    arm = ArmController(sample_cfg).open(mock_mb_ctrl)
    # j5 reports torq above abort threshold (0.8 * 10 = 8.0)
    arm._motors[5].get_state.return_value = make_state(torq=9.0)

    with pytest.raises(ArmSafetyError) as excinfo:
        arm.move_joint(5, 0.1, vlim=0.5, hold_s=0.1, loop_dt_s=0.01)
    assert excinfo.value.event.kind == "torque"
    assert excinfo.value.event.joint_index == 5
    # disable was called on both motors
    arm._motors[2].disable.assert_called()
    arm._motors[5].disable.assert_called()


def test_move_with_hold_status_trip(sample_cfg, mock_mb_ctrl):
    arm = ArmController(sample_cfg).open(mock_mb_ctrl)
    arm._motors[5].get_state.return_value = make_state(status_code=5)

    with pytest.raises(ArmSafetyError) as excinfo:
        arm.move_joint(5, 0.1, vlim=0.5, hold_s=0.1, loop_dt_s=0.01)
    assert excinfo.value.event.kind == "status"


def test_move_with_hold_healthy_completes(sample_cfg, mock_mb_ctrl):
    """No watchdog trips → loop completes normally, no exception."""
    arm = ArmController(sample_cfg).open(mock_mb_ctrl)
    arm._motors[5].get_state.return_value = make_state(pos=0.1, torq=0.0, status_code=1)
    arm.move_joint(5, 0.1, vlim=0.5, hold_s=0.05, loop_dt_s=0.01)
    # send_pos_vel called multiple times during hold
    assert arm._motors[5].send_pos_vel.call_count >= 2


# ---------- enable / disable lifecycle ----------

def test_disable_all_swallows_per_motor_errors(sample_cfg, mock_mb_ctrl):
    arm = ArmController(sample_cfg).open(mock_mb_ctrl)
    arm._motors[2].disable.side_effect = RuntimeError("simulated comms drop")
    # Should not raise — and j5 still gets disabled
    arm.disable_all()
    arm._motors[5].disable.assert_called()


def test_exit_disables_all_and_closes(sample_cfg, mock_mb_ctrl):
    """__exit__ must disable every joint and call mb_ctrl.close()."""
    arm = ArmController(sample_cfg).open(mock_mb_ctrl)
    motors_before = dict(arm._motors)
    arm.__exit__(None, None, None)
    for m in motors_before.values():
        m.disable.assert_called()
    mock_mb_ctrl.close.assert_called()


def test_close_clears_state(sample_cfg, mock_mb_ctrl):
    arm = ArmController(sample_cfg).open(mock_mb_ctrl)
    arm.close()
    assert arm._mb_ctrl is None
    assert arm._motors == {}
