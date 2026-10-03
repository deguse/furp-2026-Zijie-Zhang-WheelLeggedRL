"""Per-side make-before-break support-transfer primitives.

The module is simulator-independent and deliberately owns no actuator command.
It converts per-side riser and vertical-load observations into a conservative
support-transfer phase plus role flags.  A fast one-sample riser edge may enter
``LEAD_GUARD`` because that phase only preserves support; declaring a new
support requires Schmitt hysteresis and consecutive confirmation samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import IntEnum

from .stair_dynamic import LeadSide


class SupportTransferPhase(IntEnum):
    """Contact roles for one make-before-break stair transfer."""

    IDLE = 0
    LEAD_GUARD = 1
    TRAIL_TRANSFER = 2
    BOTH_SUPPORTED = 3
    DONE = 4
    ABORT = 5


@dataclass(frozen=True)
class SupportTransferConfig:
    """Thresholds and timing for the support-state estimator.

    These defaults are development values, not promotion thresholds.  The
    5-newton fast edge is the zero-flat-fire value observed by the matched R0c
    trigger probe.  Lower release thresholds implement Schmitt hysteresis.
    """

    control_dt_s: float = 0.005
    riser_on_force_n: float = 5.0
    riser_off_force_n: float = 2.5
    support_on_force_n: float = 5.0
    support_off_force_n: float = 1.0
    support_confirm_steps: int = 2
    both_confirm_steps: int = 3
    minimum_guard_travel_m: float = 0.05
    lead_timeout_s: float = 1.0
    trail_timeout_s: float = 1.0
    completion_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        positive = {
            "control_dt_s": self.control_dt_s,
            "riser_on_force_n": self.riser_on_force_n,
            "support_on_force_n": self.support_on_force_n,
            "lead_timeout_s": self.lead_timeout_s,
            "trail_timeout_s": self.trail_timeout_s,
            "completion_timeout_s": self.completion_timeout_s,
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in positive.values()):
            raise ValueError(
                "Support-transfer rates, on-thresholds, and timeouts must be positive."
            )
        releases = {
            "riser_off_force_n": self.riser_off_force_n,
            "support_off_force_n": self.support_off_force_n,
            "minimum_guard_travel_m": self.minimum_guard_travel_m,
        }
        if any(not math.isfinite(value) or value < 0.0 for value in releases.values()):
            raise ValueError(
                "Support-transfer release thresholds must be finite and non-negative."
            )
        if self.riser_off_force_n >= self.riser_on_force_n:
            raise ValueError("Riser release threshold must be below its on threshold.")
        if self.support_off_force_n >= self.support_on_force_n:
            raise ValueError(
                "Support release threshold must be below its on threshold."
            )
        if self.support_confirm_steps < 1 or self.both_confirm_steps < 1:
            raise ValueError("Support-transfer confirmation windows must be positive.")
        for name, timeout in (
            ("lead_timeout_s", self.lead_timeout_s),
            ("trail_timeout_s", self.trail_timeout_s),
            ("completion_timeout_s", self.completion_timeout_s),
        ):
            steps = timeout / self.control_dt_s
            if not math.isclose(steps, round(steps), rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError(f"{name} must be an integer multiple of control_dt_s.")

    @property
    def lead_timeout_steps(self) -> int:
        return round(self.lead_timeout_s / self.control_dt_s)

    @property
    def trail_timeout_steps(self) -> int:
        return round(self.trail_timeout_s / self.control_dt_s)

    @property
    def completion_timeout_steps(self) -> int:
        return round(self.completion_timeout_s / self.control_dt_s)


@dataclass(frozen=True)
class SupportTransferSensors:
    """One synchronized per-side support observation."""

    left_riser_force_n: float
    right_riser_force_n: float
    left_vertical_force_n: float
    right_vertical_force_n: float
    forward_increment_m: float = 0.0
    unsafe: bool = False

    def validate(self) -> None:
        values = (
            self.left_riser_force_n,
            self.right_riser_force_n,
            self.left_vertical_force_n,
            self.right_vertical_force_n,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "Support-transfer force observations must be finite and non-negative."
            )
        if not math.isfinite(self.forward_increment_m):
            raise ValueError("Support-transfer forward increment must be finite.")


@dataclass(frozen=True)
class SupportTransferState:
    """Scalar state for one robot."""

    phase: SupportTransferPhase = SupportTransferPhase.IDLE
    phase_steps: int = 0
    lead_side: LeadSide = LeadSide.NONE
    preferred_side: LeadSide = LeadSide.LEFT
    left_riser_active: bool = False
    right_riser_active: bool = False
    left_support_active: bool = False
    right_support_active: bool = False
    left_riser_seen: bool = False
    right_riser_seen: bool = False
    confirm_streak: int = 0
    travel_since_trigger_m: float = 0.0
    abort_reason: str | None = None


@dataclass(frozen=True)
class SupportTransferTargets:
    """Contact roles consumed by a later bounded actuator safety filter."""

    phase: SupportTransferPhase
    lead_side: LeadSide
    hold_support_side: LeadSide = LeadSide.NONE
    compliant_side: LeadSide = LeadSide.NONE
    active: bool = False
    abort: bool = False


def _schmitt(value: float, active: bool, *, on: float, off: float) -> bool:
    if active:
        return value > off
    return value >= on


def _opposite(side: LeadSide) -> LeadSide:
    if side == LeadSide.LEFT:
        return LeadSide.RIGHT
    if side == LeadSide.RIGHT:
        return LeadSide.LEFT
    return LeadSide.NONE


def _choose_lead(
    *,
    left_force_n: float,
    right_force_n: float,
    preferred_side: LeadSide,
) -> LeadSide:
    left = left_force_n >= 0.0
    right = right_force_n >= 0.0
    if not left and not right:
        return LeadSide.NONE
    if left_force_n > right_force_n:
        return LeadSide.LEFT
    if right_force_n > left_force_n:
        return LeadSide.RIGHT
    if preferred_side not in (LeadSide.LEFT, LeadSide.RIGHT):
        raise ValueError("Support-transfer preferred side must be LEFT or RIGHT.")
    return preferred_side


def _roles(state: SupportTransferState) -> SupportTransferTargets:
    if state.phase == SupportTransferPhase.LEAD_GUARD:
        return SupportTransferTargets(
            phase=state.phase,
            lead_side=state.lead_side,
            hold_support_side=_opposite(state.lead_side),
            compliant_side=state.lead_side,
            active=True,
        )
    if state.phase == SupportTransferPhase.TRAIL_TRANSFER:
        return SupportTransferTargets(
            phase=state.phase,
            lead_side=state.lead_side,
            hold_support_side=state.lead_side,
            compliant_side=_opposite(state.lead_side),
            active=True,
        )
    return SupportTransferTargets(
        phase=state.phase,
        lead_side=state.lead_side,
        active=state.phase == SupportTransferPhase.BOTH_SUPPORTED,
        abort=state.phase == SupportTransferPhase.ABORT,
    )


def support_transfer_step(
    config: SupportTransferConfig,
    state: SupportTransferState,
    sensors: SupportTransferSensors,
    *,
    stair_request: bool,
) -> tuple[SupportTransferTargets, SupportTransferState]:
    """Advance one scalar make-before-break contact-state sample."""

    sensors.validate()
    if not stair_request:
        reset = SupportTransferState(preferred_side=state.preferred_side)
        return _roles(reset), reset
    if state.preferred_side not in (LeadSide.LEFT, LeadSide.RIGHT):
        raise ValueError("Support-transfer preferred side must be LEFT or RIGHT.")
    if sensors.unsafe:
        aborted = replace(
            state,
            phase=SupportTransferPhase.ABORT,
            phase_steps=0,
            confirm_streak=0,
            abort_reason="unsafe",
        )
        return _roles(aborted), aborted
    if state.phase == SupportTransferPhase.ABORT:
        return _roles(state), state

    left_riser = _schmitt(
        sensors.left_riser_force_n,
        state.left_riser_active,
        on=config.riser_on_force_n,
        off=config.riser_off_force_n,
    )
    right_riser = _schmitt(
        sensors.right_riser_force_n,
        state.right_riser_active,
        on=config.riser_on_force_n,
        off=config.riser_off_force_n,
    )
    left_support = _schmitt(
        sensors.left_vertical_force_n,
        state.left_support_active,
        on=config.support_on_force_n,
        off=config.support_off_force_n,
    )
    right_support = _schmitt(
        sensors.right_vertical_force_n,
        state.right_support_active,
        on=config.support_on_force_n,
        off=config.support_off_force_n,
    )
    state = replace(
        state,
        left_riser_active=left_riser,
        right_riser_active=right_riser,
        left_support_active=left_support,
        right_support_active=right_support,
        left_riser_seen=state.left_riser_seen or left_riser,
        right_riser_seen=state.right_riser_seen or right_riser,
    )
    if state.phase not in (
        SupportTransferPhase.IDLE,
        SupportTransferPhase.DONE,
        SupportTransferPhase.ABORT,
    ):
        state = replace(
            state,
            travel_since_trigger_m=max(
                0.0,
                state.travel_since_trigger_m + sensors.forward_increment_m,
            ),
        )

    if state.phase == SupportTransferPhase.DONE:
        return _roles(state), state

    if state.phase == SupportTransferPhase.BOTH_SUPPORTED:
        both_stable = (
            left_support and right_support and not left_riser and not right_riser
        )
        streak = state.confirm_streak + 1 if both_stable else 0
        elapsed = state.phase_steps + 1
        release_ready = (
            state.travel_since_trigger_m >= config.minimum_guard_travel_m
            and streak >= config.both_confirm_steps
        )
        if release_ready:
            done = replace(
                state,
                phase=SupportTransferPhase.DONE,
                phase_steps=0,
                confirm_streak=streak,
            )
            return _roles(done), done
        if elapsed >= config.completion_timeout_steps:
            aborted = replace(
                state,
                phase=SupportTransferPhase.ABORT,
                phase_steps=0,
                confirm_streak=0,
                abort_reason="completion_timeout",
            )
            return _roles(aborted), aborted
        waiting = replace(state, phase_steps=elapsed, confirm_streak=streak)
        return _roles(waiting), waiting

    if state.phase == SupportTransferPhase.IDLE:
        left_hit = sensors.left_riser_force_n >= config.riser_on_force_n
        right_hit = sensors.right_riser_force_n >= config.riser_on_force_n
        if not left_hit and not right_hit:
            idle = replace(state, phase_steps=0, confirm_streak=0)
            return _roles(idle), idle
        lead = _choose_lead(
            left_force_n=sensors.left_riser_force_n if left_hit else -1.0,
            right_force_n=sensors.right_riser_force_n if right_hit else -1.0,
            preferred_side=state.preferred_side,
        )
        guarded = replace(
            state,
            phase=SupportTransferPhase.LEAD_GUARD,
            phase_steps=0,
            lead_side=lead,
            confirm_streak=0,
            travel_since_trigger_m=0.0,
        )
        return _roles(guarded), guarded

    if state.phase == SupportTransferPhase.LEAD_GUARD:
        lead_left = state.lead_side == LeadSide.LEFT
        lead_ready = (
            state.left_riser_seen and not left_riser and left_support
            if lead_left
            else state.right_riser_seen and not right_riser and right_support
        )
        streak = state.confirm_streak + 1 if lead_ready else 0
        elapsed = state.phase_steps + 1
        if streak >= config.support_confirm_steps:
            transferred = replace(
                state,
                phase=SupportTransferPhase.TRAIL_TRANSFER,
                phase_steps=0,
                confirm_streak=0,
            )
            return _roles(transferred), transferred
        if elapsed >= config.lead_timeout_steps:
            aborted = replace(
                state,
                phase=SupportTransferPhase.ABORT,
                phase_steps=0,
                confirm_streak=0,
                abort_reason="lead_support_timeout",
            )
            return _roles(aborted), aborted
        waiting = replace(state, phase_steps=elapsed, confirm_streak=streak)
        return _roles(waiting), waiting

    if state.phase == SupportTransferPhase.TRAIL_TRANSFER:
        trail_left = state.lead_side == LeadSide.RIGHT
        trail_ready = (
            state.left_riser_seen and not left_riser and left_support
            if trail_left
            else state.right_riser_seen and not right_riser and right_support
        )
        streak = state.confirm_streak + 1 if trail_ready else 0
        elapsed = state.phase_steps + 1
        if streak >= config.support_confirm_steps:
            both = replace(
                state,
                phase=SupportTransferPhase.BOTH_SUPPORTED,
                phase_steps=0,
                confirm_streak=0,
            )
            return _roles(both), both
        if elapsed >= config.trail_timeout_steps:
            aborted = replace(
                state,
                phase=SupportTransferPhase.ABORT,
                phase_steps=0,
                confirm_streak=0,
                abort_reason="trail_support_timeout",
            )
            return _roles(aborted), aborted
        waiting = replace(state, phase_steps=elapsed, confirm_streak=streak)
        return _roles(waiting), waiting

    raise ValueError(f"Unsupported support-transfer phase: {state.phase!r}")
