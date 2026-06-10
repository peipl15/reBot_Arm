"""ArmController: wraps motorbridge.Controller with sign flip, soft limits,
vlim defaults, and watchdog-protected motion.

Usage:

    cfg = load_config("configs/joint_config.yaml")
    with ArmController(cfg) as arm:
        arm.enable(2)
        arm.move_joint(2, target_user=-0.05, hold_s=1.0)
        print(arm.read_joint(2))
        arm.disable(2)
"""
from __future__ import annotations

import time
from typing import Any

from arm.config import ArmConfig, JointConfig
from arm.safety import Watchdog, WatchdogEvent


class ArmSafetyError(RuntimeError):
    """Raised when a watchdog trips during commanded motion."""

    def __init__(self, event: WatchdogEvent) -> None:
        super().__init__(str(event))
        self.event = event


class ArmController:
    """Wraps motorbridge.Controller. Open via context manager or .open()."""

    def __init__(self, cfg: ArmConfig) -> None:
        self.cfg = cfg
        self._mb_ctrl: Any | None = None
        self._motors: dict[int, Any] = {}

    # ---- lifecycle ----

    def open(self, mb_ctrl: Any | None = None) -> "ArmController":
        """Open underlying controller and register motors.

        If mb_ctrl is None, create a real motorbridge.Controller from cfg.
        Pass an explicit mock/stub for tests.
        """
        if mb_ctrl is None:
            from motorbridge import Controller as MBController
            mb_ctrl = MBController.from_dm_device(
                self.cfg.dm_device_type, self.cfg.dm_channel,
            )
        self._mb_ctrl = mb_ctrl
        for j in self.cfg.joints:
            m = mb_ctrl.add_damiao_motor(j.id, j.feedback_id, j.model)
            self._motors[j.index] = m
        return self

    def close(self) -> None:
        if self._mb_ctrl is not None:
            try:
                self._mb_ctrl.close()
            except Exception:
                pass
            self._mb_ctrl = None
        self._motors.clear()

    def __enter__(self) -> "ArmController":
        return self.open()

    def __exit__(self, *exc_info) -> None:
        try:
            self.disable_all()
        finally:
            self.close()

    # ---- internals ----

    def _joint_cfg(self, index: int) -> JointConfig:
        return self.cfg.joint(index)

    def _motor(self, index: int) -> Any:
        if index not in self._motors:
            raise RuntimeError(f"joint {index} not opened (call .open() first)")
        return self._motors[index]

    # ---- enable / disable ----

    def enable(self, index: int) -> None:
        from motorbridge.models import Mode
        m = self._motor(index)
        m.ensure_mode(Mode.POS_VEL, 1000)
        m.enable()

    def disable(self, index: int) -> None:
        try:
            self._motor(index).disable()
        except Exception:
            pass

    def enable_all(self) -> None:
        for j in self.cfg.joints:
            self.enable(j.index)

    def disable_all(self) -> None:
        for j in self.cfg.joints:
            self.disable(j.index)

    # ---- read ----

    def read_joint(self, index: int, settle_s: float = 0.03):
        """Return (pos_user, vel_motor, torq, status_code) or None.

        Position is converted to user frame via sign. Velocity and torque are
        left in motor units (sign-flipping them adds confusion in the watchdog
        output; users wanting URDF velocity can multiply themselves).
        """
        j = self._joint_cfg(index)
        m = self._motor(index)
        m.request_feedback()
        time.sleep(settle_s)
        state = m.get_state()
        if state is None:
            return None
        return {
            "pos_user": state.pos * j.sign,
            "pos_motor": state.pos,
            "vel": state.vel,
            "torq": state.torq,
            "status_code": state.status_code,
        }

    # ---- move ----

    def clamp_target(self, index: int, target_user: float) -> float:
        """Clamp target_user to [soft_min, soft_max]. Returns clamped value."""
        j = self._joint_cfg(index)
        return max(j.soft_min, min(j.soft_max, target_user))

    def move_joint(
        self,
        index: int,
        target_user: float,
        vlim: float | None = None,
        hold_s: float = 0.0,
        raise_on_clamp: bool = False,
        loop_dt_s: float = 0.02,
    ) -> None:
        """Send a pos-vel command in user frame.

        target_user is in URDF/user frame; sign flip happens internally.
        Target is clamped to [soft_min, soft_max]; if raise_on_clamp and the
        original value was outside, ValueError is raised before any command.

        If hold_s > 0, the command is re-issued every loop_dt_s for hold_s
        seconds while a Watchdog checks each feedback sample. Any watchdog
        trip immediately disables all motors and raises ArmSafetyError.
        """
        j = self._joint_cfg(index)
        m = self._motor(index)

        clamped = self.clamp_target(index, target_user)
        if clamped != target_user and raise_on_clamp:
            raise ValueError(
                f"joint {j.index}: target {target_user:+.3f} outside "
                f"soft limits [{j.soft_min:+.3f}, {j.soft_max:+.3f}]"
            )

        if vlim is None:
            vlim = j.vlim_default
        if vlim <= 0:
            raise ValueError(f"vlim must be > 0, got {vlim}")
        if vlim > j.vmax_fw:
            raise ValueError(f"joint {j.index}: vlim {vlim} > vmax_fw {j.vmax_fw}")

        target_motor = clamped * j.sign

        # single command path
        if hold_s <= 0:
            m.send_pos_vel(target_motor, vlim)
            return

        # held command with watchdog
        wd = Watchdog(j, self.cfg.watchdog)
        t_end = time.monotonic() + hold_s
        while time.monotonic() < t_end:
            m.send_pos_vel(target_motor, vlim)
            time.sleep(loop_dt_s)
            m.request_feedback()
            # very short settle to read feedback that crossed since last loop
            time.sleep(0.005)
            state = m.get_state()
            if state is None:
                continue
            event = wd.check(clamped, state)
            if event is not None:
                self.disable_all()
                raise ArmSafetyError(event)
