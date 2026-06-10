from __future__ import annotations

from pathlib import Path

import pytest

from arm.config import ArmConfig, JointConfig, WatchdogConfig, load_config


CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "joint_config.yaml"


def test_load_real_config_has_7_joints():
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg, ArmConfig)
    assert len(cfg.joints) == 7
    assert [j.index for j in cfg.joints] == [1, 2, 3, 4, 5, 6, 7]


def test_real_config_j2_j3_are_sign_minus_1():
    cfg = load_config(CONFIG_PATH)
    assert cfg.joint(2).sign == -1, "joint 2 should be reversed per 2026-06-09 observation"
    assert cfg.joint(3).sign == -1, "joint 3 should be reversed per 2026-06-09 observation"
    assert cfg.joint(1).sign == +1
    assert cfg.joint(7).sign == +1


def test_real_config_motor_classes():
    cfg = load_config(CONFIG_PATH)
    # j1-j3: 28 Nm "big" motors
    for idx in (1, 2, 3):
        assert cfg.joint(idx).tmax_fw == 28.0
        assert cfg.joint(idx).vmax_fw == 10.0
    # j4-j7: 10 Nm "small" motors
    for idx in (4, 5, 6, 7):
        assert cfg.joint(idx).tmax_fw == 10.0
        assert cfg.joint(idx).vmax_fw == 30.0


def test_joint_lookup_by_index_and_name():
    cfg = load_config(CONFIG_PATH)
    j2 = cfg.joint(2)
    assert j2.name == "shoulder"
    assert cfg.joint_by_name("shoulder") is j2


def test_joint_lookup_missing_raises():
    cfg = load_config(CONFIG_PATH)
    with pytest.raises(KeyError):
        cfg.joint(99)
    with pytest.raises(KeyError):
        cfg.joint_by_name("nonexistent")


def test_watchdog_loaded():
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg.watchdog, WatchdogConfig)
    assert cfg.watchdog.torque_abort_ratio == 0.80
    assert cfg.watchdog.status_must_be == 1


def test_joint_config_invalid_sign():
    with pytest.raises(ValueError, match="sign"):
        JointConfig(
            index=1, id=1, feedback_id=0x11, name="x", model="4340",
            sign=0, pmax_fw=12.5, vmax_fw=10, tmax_fw=10,
            soft_min=-0.1, soft_max=+0.1, vlim_default=0.5,
        )


def test_joint_config_invalid_soft_range():
    with pytest.raises(ValueError, match="soft_min"):
        JointConfig(
            index=1, id=1, feedback_id=0x11, name="x", model="4340",
            sign=+1, pmax_fw=12.5, vmax_fw=10, tmax_fw=10,
            soft_min=+0.1, soft_max=-0.1, vlim_default=0.5,
        )


def test_joint_config_vlim_above_vmax():
    with pytest.raises(ValueError, match="vlim_default"):
        JointConfig(
            index=1, id=1, feedback_id=0x11, name="x", model="4340",
            sign=+1, pmax_fw=12.5, vmax_fw=1.0, tmax_fw=10,
            soft_min=-0.1, soft_max=+0.1, vlim_default=99.0,
        )
