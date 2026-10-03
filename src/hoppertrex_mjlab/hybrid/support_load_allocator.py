"""Bounded two-side support-load allocation for contact-aware roll control.

The allocator is a pure development kernel. It solves the convex two-variable
QP over left/right vertical wheel loads without changing any simulator or HAL
interface. Contact state, desired total support load, and desired body roll
moment are explicit inputs; minimum positive load is therefore a constraint,
not an indirect position-target heuristic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SupportLoadAllocatorConfig:
    track_width_m: float
    minimum_contact_load_n: float
    maximum_contact_load_n: float
    total_load_weight: float = 1.0
    roll_moment_weight: float = 1.0
    regularization_weight: float = 0.0

    def __post_init__(self) -> None:
        values = {
            "track_width_m": self.track_width_m,
            "minimum_contact_load_n": self.minimum_contact_load_n,
            "maximum_contact_load_n": self.maximum_contact_load_n,
            "total_load_weight": self.total_load_weight,
            "roll_moment_weight": self.roll_moment_weight,
            "regularization_weight": self.regularization_weight,
        }
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("Support-load allocator parameters must be finite.")
        if self.track_width_m <= 0.0:
            raise ValueError("Support-load track width must be positive.")
        if self.minimum_contact_load_n <= 0.0:
            raise ValueError("Minimum contact load must be positive.")
        if self.maximum_contact_load_n < self.minimum_contact_load_n:
            raise ValueError("Maximum contact load must cover the minimum.")
        if self.total_load_weight <= 0.0 or self.roll_moment_weight <= 0.0:
            raise ValueError("Support-load objective weights must be positive.")
        if self.regularization_weight < 0.0:
            raise ValueError("Support-load regularization cannot be negative.")


@dataclass(frozen=True)
class SupportLoadAllocation:
    left_load_n: float
    right_load_n: float
    achieved_total_load_n: float
    achieved_roll_moment_nm: float
    total_load_residual_n: float
    roll_moment_residual_nm: float
    objective: float
    contact_mode: str
    left_at_lower_bound: bool
    left_at_upper_bound: bool
    right_at_lower_bound: bool
    right_at_upper_bound: bool
    bilateral_minimum_satisfied: bool


def _objective(
    left: float,
    right: float,
    *,
    desired_total: float,
    desired_moment: float,
    half_track: float,
    total_weight: float,
    moment_weight: float,
    regularization: float,
    reference_left: float,
    reference_right: float,
) -> float:
    total_error = left + right - desired_total
    moment_error = half_track * (left - right) - desired_moment
    return (
        total_weight * total_error * total_error
        + moment_weight * moment_error * moment_error
        + regularization
        * (
            (left - reference_left) * (left - reference_left)
            + (right - reference_right) * (right - reference_right)
        )
    )


def _bounds(
    config: SupportLoadAllocatorConfig,
    in_contact: bool,
) -> tuple[float, float]:
    if not in_contact:
        return 0.0, 0.0
    return config.minimum_contact_load_n, config.maximum_contact_load_n


def allocate_support_loads(
    config: SupportLoadAllocatorConfig,
    *,
    desired_total_load_n: float,
    desired_roll_moment_nm: float,
    left_in_contact: bool,
    right_in_contact: bool,
    reference_left_load_n: float | None = None,
    reference_right_load_n: float | None = None,
) -> SupportLoadAllocation:
    """Solve the bounded two-load QP by complete active-set enumeration."""

    scalars = (desired_total_load_n, desired_roll_moment_nm)
    if any(not math.isfinite(value) for value in scalars):
        raise ValueError("Desired support load and roll moment must be finite.")
    if desired_total_load_n < 0.0:
        raise ValueError("Desired total support load cannot be negative.")
    if not isinstance(left_in_contact, bool) or not isinstance(right_in_contact, bool):
        raise TypeError("Support contact flags must be bool values.")

    left_bounds = _bounds(config, left_in_contact)
    right_bounds = _bounds(config, right_in_contact)
    nominal_half = 0.5 * desired_total_load_n
    reference_left = (
        nominal_half
        if reference_left_load_n is None
        else float(reference_left_load_n)
    )
    reference_right = (
        nominal_half
        if reference_right_load_n is None
        else float(reference_right_load_n)
    )
    if not math.isfinite(reference_left) or not math.isfinite(reference_right):
        raise ValueError("Support-load references must be finite.")

    half_track = 0.5 * config.track_width_m
    total_weight = config.total_load_weight
    moment_weight = config.roll_moment_weight
    regularization = config.regularization_weight
    diagonal = total_weight + moment_weight * half_track**2 + regularization
    off_diagonal = total_weight - moment_weight * half_track**2
    rhs_left = (
        total_weight * desired_total_load_n
        + moment_weight * half_track * desired_roll_moment_nm
        + regularization * reference_left
    )
    rhs_right = (
        total_weight * desired_total_load_n
        - moment_weight * half_track * desired_roll_moment_nm
        + regularization * reference_right
    )
    determinant = diagonal * diagonal - off_diagonal * off_diagonal
    if determinant <= 0.0 or not math.isfinite(determinant):
        raise RuntimeError("Support-load QP Hessian is not positive definite.")

    candidates: list[tuple[float, float]] = [
        (
            (diagonal * rhs_left - off_diagonal * rhs_right) / determinant,
            (diagonal * rhs_right - off_diagonal * rhs_left) / determinant,
        )
    ]
    one_dimensional_denominator = diagonal
    for fixed_left in set(left_bounds):
        optimal_right = (
            total_weight * (desired_total_load_n - fixed_left)
            + moment_weight
            * half_track
            * (half_track * fixed_left - desired_roll_moment_nm)
            + regularization * reference_right
        ) / one_dimensional_denominator
        candidates.append((fixed_left, optimal_right))
    for fixed_right in set(right_bounds):
        optimal_left = (
            total_weight * (desired_total_load_n - fixed_right)
            + moment_weight
            * half_track
            * (half_track * fixed_right + desired_roll_moment_nm)
            + regularization * reference_left
        ) / one_dimensional_denominator
        candidates.append((optimal_left, fixed_right))
    candidates.extend(
        (left, right)
        for left in set(left_bounds)
        for right in set(right_bounds)
    )

    tolerance = 1.0e-9
    feasible = []
    for left, right in candidates:
        if (
            left_bounds[0] - tolerance <= left <= left_bounds[1] + tolerance
            and right_bounds[0] - tolerance <= right <= right_bounds[1] + tolerance
        ):
            clipped_left = min(max(left, left_bounds[0]), left_bounds[1])
            clipped_right = min(max(right, right_bounds[0]), right_bounds[1])
            cost = _objective(
                clipped_left,
                clipped_right,
                desired_total=desired_total_load_n,
                desired_moment=desired_roll_moment_nm,
                half_track=half_track,
                total_weight=total_weight,
                moment_weight=moment_weight,
                regularization=regularization,
                reference_left=reference_left,
                reference_right=reference_right,
            )
            feasible.append((cost, clipped_left, clipped_right))
    if not feasible:
        raise RuntimeError("Support-load active-set enumeration found no solution.")
    objective, left, right = min(feasible)
    achieved_total = left + right
    achieved_moment = half_track * (left - right)
    if left_in_contact and right_in_contact:
        mode = "bilateral"
    elif left_in_contact:
        mode = "left_only"
    elif right_in_contact:
        mode = "right_only"
    else:
        mode = "unsupported"

    def at(value: float, bound: float) -> bool:
        return math.isclose(value, bound, rel_tol=0.0, abs_tol=1.0e-8)

    return SupportLoadAllocation(
        left_load_n=left,
        right_load_n=right,
        achieved_total_load_n=achieved_total,
        achieved_roll_moment_nm=achieved_moment,
        total_load_residual_n=achieved_total - desired_total_load_n,
        roll_moment_residual_nm=achieved_moment - desired_roll_moment_nm,
        objective=objective,
        contact_mode=mode,
        left_at_lower_bound=at(left, left_bounds[0]),
        left_at_upper_bound=at(left, left_bounds[1]),
        right_at_lower_bound=at(right, right_bounds[0]),
        right_at_upper_bound=at(right, right_bounds[1]),
        bilateral_minimum_satisfied=(
            left_in_contact
            and right_in_contact
            and left >= config.minimum_contact_load_n
            and right >= config.minimum_contact_load_n
        ),
    )


__all__ = [
    "SupportLoadAllocation",
    "SupportLoadAllocatorConfig",
    "allocate_support_loads",
]
