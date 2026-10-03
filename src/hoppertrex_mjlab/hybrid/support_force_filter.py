"""Rejected development ablation for per-side support-load correction.

This module is retained only to reproduce the negative exact-reset experiment.
It is not wired into the formal controller: even sub-0.5 mrad position offsets
degraded both strict support and geometric traversal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hoppertrex_mjlab.assets.HopperTrex_CFG import (
    LEG_POSITION_DAMPING,
    LEG_POSITION_STIFFNESS,
)

from .roll_feedback import ROLL_FEEDBACK_MAX_IDENTIFIED_AMPLITUDE_RAD
from .stair_dynamic import LEFT_LIFT_BASIS, RIGHT_LIFT_BASIS, LeadSide
from .support_transfer import SupportTransferTargets


@dataclass(frozen=True)
class SupportForceFilterConfig:
    """Development authority for a position-equivalent damping correction."""

    control_dt_s: float = 0.005
    load_filter_time_constant_s: float = 0.020
    reference_total_load_n: float = 150.0
    hold_load_fraction: float = 0.60
    force_to_velocity_gain_rad_per_n_s: float = 5.0e-4
    velocity_equivalent_time_s: float = LEG_POSITION_DAMPING / LEG_POSITION_STIFFNESS
    max_amplitude_rad: float = 5.0e-4

    def __post_init__(self) -> None:
        positive = {
            "control_dt_s": self.control_dt_s,
            "load_filter_time_constant_s": self.load_filter_time_constant_s,
            "reference_total_load_n": self.reference_total_load_n,
            "force_to_velocity_gain_rad_per_n_s": self.force_to_velocity_gain_rad_per_n_s,
            "velocity_equivalent_time_s": self.velocity_equivalent_time_s,
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in positive.values()):
            raise ValueError(
                "Support-force filter rates, loads, and gains must be positive."
            )
        if not 0.5 <= self.hold_load_fraction < 1.0:
            raise ValueError("Support hold-load fraction must stay within [0.5, 1.0).")
        if (
            not math.isfinite(self.max_amplitude_rad)
            or self.max_amplitude_rad < 0.0
            or self.max_amplitude_rad > ROLL_FEEDBACK_MAX_IDENTIFIED_AMPLITUDE_RAD
        ):
            raise ValueError(
                "Support-force amplitude exceeds the identified leg bracket."
            )

    @property
    def filter_alpha(self) -> float:
        return self.control_dt_s / (
            self.load_filter_time_constant_s + self.control_dt_s
        )


@dataclass(frozen=True)
class SupportForceFilterState:
    """Low-pass contact loads; the correction itself never integrates."""

    initialized: bool = False
    filtered_left_force_n: float = 0.0
    filtered_right_force_n: float = 0.0
    left_amplitude_rad: float = 0.0
    right_amplitude_rad: float = 0.0


@dataclass(frozen=True)
class SupportForceFilterOutput:
    left_target_force_n: float
    right_target_force_n: float
    left_amplitude_rad: float
    right_amplitude_rad: float
    leg_offsets_rad: tuple[float, float, float, float]


def support_force_leg_offsets(
    left_amplitude_rad: float,
    right_amplitude_rad: float,
) -> tuple[float, float, float, float]:
    """Map positive per-side lift coordinates into four joint offsets."""

    values = (left_amplitude_rad, right_amplitude_rad)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Support-force leg amplitudes must be finite.")
    return (
        left_amplitude_rad * LEFT_LIFT_BASIS[0],
        right_amplitude_rad * RIGHT_LIFT_BASIS[0],
        left_amplitude_rad * LEFT_LIFT_BASIS[1],
        right_amplitude_rad * RIGHT_LIFT_BASIS[1],
    )


def _force_targets(
    config: SupportForceFilterConfig,
    roles: SupportTransferTargets,
) -> tuple[float, float]:
    half = 0.5 * config.reference_total_load_n
    hold = config.hold_load_fraction * config.reference_total_load_n
    compliant = config.reference_total_load_n - hold
    if roles.hold_support_side == LeadSide.LEFT:
        return hold, compliant
    if roles.hold_support_side == LeadSide.RIGHT:
        return compliant, hold
    return half, half


def support_force_filter_step(
    config: SupportForceFilterConfig,
    state: SupportForceFilterState,
    roles: SupportTransferTargets,
    *,
    left_vertical_force_n: float,
    right_vertical_force_n: float,
) -> tuple[SupportForceFilterOutput, SupportForceFilterState]:
    """Compute one bounded force-proportional leg-position correction.

    The existing position actuator obeys ``kp * position_error - kd * velocity``.
    ``velocity_equivalent_time_s = kd/kp`` maps a desired force-feedback velocity
    into the position offset producing the same incremental joint torque.  The
    correction is recomputed from filtered force every sample and never integrates.
    """

    measured = (left_vertical_force_n, right_vertical_force_n)
    if any(not math.isfinite(value) or value < 0.0 for value in measured):
        raise ValueError("Support-force observations must be finite and non-negative.")
    if state.initialized:
        alpha = config.filter_alpha
        filtered_left = state.filtered_left_force_n + alpha * (
            left_vertical_force_n - state.filtered_left_force_n
        )
        filtered_right = state.filtered_right_force_n + alpha * (
            right_vertical_force_n - state.filtered_right_force_n
        )
    else:
        filtered_left, filtered_right = measured

    target_left, target_right = _force_targets(config, roles)
    if roles.active:
        scale = (
            config.velocity_equivalent_time_s
            * config.force_to_velocity_gain_rad_per_n_s
        )
        left = scale * (filtered_left - target_left)
        right = scale * (filtered_right - target_right)
        left = max(-config.max_amplitude_rad, min(config.max_amplitude_rad, left))
        right = max(-config.max_amplitude_rad, min(config.max_amplitude_rad, right))
    else:
        left = 0.0
        right = 0.0

    next_state = SupportForceFilterState(
        initialized=True,
        filtered_left_force_n=filtered_left,
        filtered_right_force_n=filtered_right,
        left_amplitude_rad=left,
        right_amplitude_rad=right,
    )
    output = SupportForceFilterOutput(
        left_target_force_n=target_left,
        right_target_force_n=target_right,
        left_amplitude_rad=left,
        right_amplitude_rad=right,
        leg_offsets_rad=support_force_leg_offsets(left, right),
    )
    return output, next_state
