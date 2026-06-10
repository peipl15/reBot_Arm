from __future__ import annotations

from arm.safety import Watchdog
from tests.conftest import make_state


class FakeClock:
    """Deterministic monotonic clock for testing the tracking-error timer."""

    def __init__(self, t0: float = 0.0) -> None:
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# ---------- torque ----------

def test_torque_below_threshold_no_trip(j5_config, watchdog_cfg):
    wd = Watchdog(j5_config, watchdog_cfg)
    # j5 tmax=10, abort_ratio=0.8, threshold=8.0
    state = make_state(torq=7.9)
    assert wd.check(target_user=0.0, state=state) is None


def test_torque_above_threshold_trips(j5_config, watchdog_cfg):
    wd = Watchdog(j5_config, watchdog_cfg)
    state = make_state(torq=8.5)  # > 0.8 * 10
    event = wd.check(target_user=0.0, state=state)
    assert event is not None
    assert event.kind == "torque"
    assert event.joint_index == 5


def test_torque_negative_also_trips(j5_config, watchdog_cfg):
    wd = Watchdog(j5_config, watchdog_cfg)
    state = make_state(torq=-8.5)
    event = wd.check(0.0, state)
    assert event is not None and event.kind == "torque"


# ---------- status ----------

def test_status_ok_no_trip(j5_config, watchdog_cfg):
    wd = Watchdog(j5_config, watchdog_cfg)
    assert wd.check(0.0, make_state(status_code=1)) is None


def test_status_nonzero_trips(j5_config, watchdog_cfg):
    wd = Watchdog(j5_config, watchdog_cfg)
    state = make_state(status_code=7)  # any non-expected code
    event = wd.check(0.0, state)
    assert event is not None
    assert event.kind == "status"
    assert "status_code=7" in event.detail


def test_status_zero_trips(j5_config, watchdog_cfg):
    wd = Watchdog(j5_config, watchdog_cfg)
    event = wd.check(0.0, make_state(status_code=0))
    assert event is not None and event.kind == "status"


# ---------- tracking ----------

def test_tracking_within_threshold_no_trip(j5_config, watchdog_cfg):
    clock = FakeClock()
    wd = Watchdog(j5_config, watchdog_cfg, clock=clock)
    # target=0.1, actual_user=0.07 (pos=0.07 because sign=+1), err=0.03 < 0.05
    state = make_state(pos=0.07)
    assert wd.check(0.1, state) is None


def test_tracking_exceeds_but_within_timeout_no_trip(j5_config, watchdog_cfg):
    clock = FakeClock()
    wd = Watchdog(j5_config, watchdog_cfg, clock=clock)
    # First sample: err > 0.05, starts timer
    assert wd.check(0.5, make_state(pos=0.0)) is None
    clock.advance(0.3)
    # Still over but only 0.3s in (timeout is 0.5s)
    assert wd.check(0.5, make_state(pos=0.0)) is None


def test_tracking_exceeds_past_timeout_trips(j5_config, watchdog_cfg):
    clock = FakeClock()
    wd = Watchdog(j5_config, watchdog_cfg, clock=clock)
    assert wd.check(0.5, make_state(pos=0.0)) is None
    clock.advance(0.6)  # > 0.5s timeout
    event = wd.check(0.5, make_state(pos=0.0))
    assert event is not None
    assert event.kind == "tracking"


def test_tracking_recovers_resets_timer(j5_config, watchdog_cfg):
    clock = FakeClock()
    wd = Watchdog(j5_config, watchdog_cfg, clock=clock)
    # Violate
    assert wd.check(0.5, make_state(pos=0.0)) is None
    clock.advance(0.3)
    # Come back into range
    assert wd.check(0.5, make_state(pos=0.49)) is None
    clock.advance(0.5)  # would have tripped if timer still running
    assert wd.check(0.5, make_state(pos=0.49)) is None


def test_tracking_sign_flip(j2_config, watchdog_cfg):
    """j2 has sign=-1. motor pos=-0.1 → user_pos=+0.1."""
    clock = FakeClock()
    wd = Watchdog(j2_config, watchdog_cfg, clock=clock)
    # target_user=+0.1, motor_pos=-0.1 → actual_user=+0.1, err=0 — no trip
    assert wd.check(target_user=+0.1, state=make_state(pos=-0.1)) is None
    # target_user=+0.1, motor_pos=+0.1 → actual_user=-0.1, err=0.2 → eventual trip
    assert wd.check(target_user=+0.1, state=make_state(pos=+0.1)) is None  # timer starts
    clock.advance(0.6)
    event = wd.check(target_user=+0.1, state=make_state(pos=+0.1))
    assert event is not None and event.kind == "tracking"


def test_reset_clears_tracking_state(j5_config, watchdog_cfg):
    clock = FakeClock()
    wd = Watchdog(j5_config, watchdog_cfg, clock=clock)
    wd.check(0.5, make_state(pos=0.0))
    wd.reset()
    clock.advance(0.6)
    # Even after timeout would have fired, fresh first call starts new timer
    assert wd.check(0.5, make_state(pos=0.0)) is None


# ---------- priority ----------

def test_status_takes_priority_over_torque(j5_config, watchdog_cfg):
    wd = Watchdog(j5_config, watchdog_cfg)
    # Both bad: status code is checked first
    event = wd.check(0.0, make_state(torq=99, status_code=7))
    assert event.kind == "status"
