"""Watchdog: monitors a joint's motor state during commanded motion.

Three trip conditions:

  1. torque  - |state.torq| > torque_abort_ratio * tmax_fw  (instant abort)
  2. status  - state.status_code != status_must_be          (instant abort)
  3. tracking - |target_user - actual_user| > tracking_error_rad
                persisted for > tracking_error_timeout_s    (timed abort)

Watchdog is stateful only for #3 (it remembers when the violation started so it
can time out). Reuse a single Watchdog instance across one move_joint call;
create a fresh one per move.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from arm.config import JointConfig, WatchdogConfig


class _StateLike(Protocol):
    pos: float
    vel: float
    torq: float
    status_code: int


@dataclass(frozen=True)
class WatchdogEvent:
    kind: str            # "torque" | "status" | "tracking"
    joint_index: int
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] joint {self.joint_index}: {self.detail}"


class Watchdog:
    """Stateful checker for a single joint.

    Inject `clock` for deterministic testing (defaults to time.monotonic).
    """

    def __init__(
        self,
        joint_cfg: JointConfig,
        watchdog_cfg: WatchdogConfig,
        clock=time.monotonic,
    ) -> None:
        self.j = joint_cfg
        self.w = watchdog_cfg
        self._clock = clock
        self._tracking_violation_started: float | None = None

    def reset(self) -> None:
        self._tracking_violation_started = None

    def check(self, target_user: float, state: _StateLike) -> WatchdogEvent | None:
        """Inspect one feedback sample. Return event if a trip condition is met."""
        # 1. status (instant)
        if state.status_code != self.w.status_must_be:
            return WatchdogEvent(
                kind="status",
                joint_index=self.j.index,
                detail=f"status_code={state.status_code}, expected {self.w.status_must_be}",
            )

        # 2. torque (instant)
        abort_thresh = self.w.torque_abort_ratio * self.j.tmax_fw
        if abs(state.torq) > abort_thresh:
            return WatchdogEvent(
                kind="torque",
                joint_index=self.j.index,
                detail=(
                    f"|torq|={abs(state.torq):.2f} Nm > {abort_thresh:.2f} "
                    f"(= {self.w.torque_abort_ratio:.0%} of tmax_fw={self.j.tmax_fw})"
                ),
            )

        # 3. tracking error (persistent)
        actual_user = state.pos * self.j.sign
        err = abs(target_user - actual_user)
        now = self._clock()
        if err > self.w.tracking_error_rad:
            if self._tracking_violation_started is None:
                self._tracking_violation_started = now
            elif now - self._tracking_violation_started > self.w.tracking_error_timeout_s:
                return WatchdogEvent(
                    kind="tracking",
                    joint_index=self.j.index,
                    detail=(
                        f"|target={target_user:+.3f} - actual_user={actual_user:+.3f}|"
                        f" = {err:.3f} rad > {self.w.tracking_error_rad} for "
                        f"{now - self._tracking_violation_started:.2f}s "
                        f"(timeout {self.w.tracking_error_timeout_s}s)"
                    ),
                )
        else:
            self._tracking_violation_started = None

        return None
