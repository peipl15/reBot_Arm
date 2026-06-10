"""Joint and arm configuration: dataclasses + YAML loader.

Coordinate convention: all user-facing positions (soft_min/soft_max, move_joint
arguments, read_joint return value) are in the URDF/user frame. The motor's
native encoder frame is converted via `sign`:

    user_pos  = sign * motor_pos
    motor_cmd = sign * user_cmd

`sign` is +1 or -1. Set per joint after empirical direction check.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class WatchdogConfig:
    torque_warn_ratio: float
    torque_abort_ratio: float
    tracking_error_rad: float
    tracking_error_timeout_s: float
    status_must_be: int


@dataclass(frozen=True)
class JointConfig:
    index: int          # 1-based, 1..7
    id: int             # CAN command ID, e.g. 0x01
    feedback_id: int    # CAN feedback ID, e.g. 0x11
    name: str           # semantic name, e.g. "shoulder"
    model: str          # motorbridge model string, e.g. "4340"
    sign: int           # +1 or -1
    pmax_fw: float      # firmware PMAX (MIT mode position mapping)
    vmax_fw: float      # firmware VMAX (rad/s)
    tmax_fw: float      # firmware TMAX (Nm)
    soft_min: float     # user-frame soft lower limit
    soft_max: float     # user-frame soft upper limit
    vlim_default: float # rad/s, default vlim for move_joint

    def __post_init__(self) -> None:
        if self.sign not in (-1, +1):
            raise ValueError(f"joint {self.index}: sign must be +1 or -1, got {self.sign}")
        if self.soft_min >= self.soft_max:
            raise ValueError(
                f"joint {self.index}: soft_min ({self.soft_min}) must be < soft_max ({self.soft_max})"
            )
        if self.vlim_default <= 0 or self.vlim_default > self.vmax_fw:
            raise ValueError(
                f"joint {self.index}: vlim_default ({self.vlim_default}) must be in (0, vmax_fw={self.vmax_fw}]"
            )


@dataclass(frozen=True)
class ArmConfig:
    transport: str
    dm_device_type: str
    dm_channel: str
    feedback_dt_ms: int
    watchdog: WatchdogConfig
    joints: tuple[JointConfig, ...]

    def joint(self, index: int) -> JointConfig:
        for j in self.joints:
            if j.index == index:
                return j
        raise KeyError(f"joint index {index} not in config (have {[j.index for j in self.joints]})")

    def joint_by_name(self, name: str) -> JointConfig:
        for j in self.joints:
            if j.name == name:
                return j
        raise KeyError(f"joint name {name!r} not in config (have {[j.name for j in self.joints]})")


def load_config(path: str | Path) -> ArmConfig:
    """Load and validate joint_config.yaml."""
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)

    defaults = raw["defaults"]
    wd_raw = defaults["watchdog"]
    watchdog = WatchdogConfig(
        torque_warn_ratio=float(wd_raw["torque_warn_ratio"]),
        torque_abort_ratio=float(wd_raw["torque_abort_ratio"]),
        tracking_error_rad=float(wd_raw["tracking_error_rad"]),
        tracking_error_timeout_s=float(wd_raw["tracking_error_timeout_s"]),
        status_must_be=int(wd_raw["status_must_be"]),
    )

    joints: list[JointConfig] = []
    for j in raw["joints"]:
        joints.append(JointConfig(
            index=int(j["index"]),
            id=int(j["id"]),
            feedback_id=int(j["feedback_id"]),
            name=str(j["name"]),
            model=str(j["model"]),
            sign=int(j["sign"]),
            pmax_fw=float(j["pmax_fw"]),
            vmax_fw=float(j["vmax_fw"]),
            tmax_fw=float(j["tmax_fw"]),
            soft_min=float(j["soft_min"]),
            soft_max=float(j["soft_max"]),
            vlim_default=float(j["vlim_default"]),
        ))

    if len({j.index for j in joints}) != len(joints):
        raise ValueError("duplicate joint index in config")
    if len({j.id for j in joints}) != len(joints):
        raise ValueError("duplicate motor id in config")

    return ArmConfig(
        transport=str(defaults["transport"]),
        dm_device_type=str(defaults["dm_device_type"]),
        dm_channel=str(defaults["dm_channel"]),
        feedback_dt_ms=int(defaults["feedback_dt_ms"]),
        watchdog=watchdog,
        joints=tuple(sorted(joints, key=lambda j: j.index)),
    )
