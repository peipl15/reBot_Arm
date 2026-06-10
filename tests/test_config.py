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


def test_real_config_all_signs_plus_1_after_phase_2():
    """After Phase 2 (2026-06-10), j2 and j3 were flipped to sign=+1 so that
    user-frame positive = bend-with-gravity (intuitive). All 7 joints end up +1
    in the final config."""
    cfg = load_config(CONFIG_PATH)
    for j in cfg.joints:
        assert j.sign == +1, f"joint {j.index} ({j.name}) sign should be +1, got {j.sign}"


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


def test_real_config_semantic_names_match_phase_2():
    """Semantic names refined per user on 2026-06-10."""
    cfg = load_config(CONFIG_PATH)
    assert cfg.joint(1).name == "base"
    assert cfg.joint(2).name == "shoulder"
    assert cfg.joint(3).name == "elbow"
    assert cfg.joint(4).name == "wrist_flex"
    assert cfg.joint(5).name == "wrist_yaw"
    assert cfg.joint(6).name == "wrist_roll"
    assert cfg.joint(7).name == "gripper"


def test_real_config_soft_limits_backfilled():
    """Soft limits should reflect Phase 2 measurements, not the original
    ±0.1 placeholders."""
    cfg = load_config(CONFIG_PATH)
    # Each joint's full range should exceed 0.5 rad (>>0.2 of placeholder)
    for j in cfg.joints:
        span = j.soft_max - j.soft_min
        assert span > 0.5, f"joint {j.index} ({j.name}) span {span} looks like a placeholder"


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
