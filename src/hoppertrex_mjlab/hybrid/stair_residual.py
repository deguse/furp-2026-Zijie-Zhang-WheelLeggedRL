"""Residual-PPO contract layered on a frozen classical stair controller."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hoppertrex_mjlab.hybrid.config import DEFAULT_ACTION_SCALES
from hoppertrex_mjlab.hybrid.stair_classical import PHASE_COUNT, StairPhase

STAIR_RESIDUAL_ACTION_SCALES = DEFAULT_ACTION_SCALES
ACTOR_FIELD_NAMES = (
    "proprioception",
    "phase_one_hot",
    "classical_wheel_baseline",
    "nominal_leg_targets",
    "classical_errors",
    "previous_residual",
    "stair_mode",
)
PRIVILEGED_FIELD_NAMES = (
    "step_height",
    "distance_to_riser",
    "contact_force",
    "friction",
    "randomization_parameters",
)


def build_stair_residual_observations(
    *,
    proprioception: Sequence[float],
    phase: StairPhase,
    classical_wheel_baseline: Sequence[float],
    nominal_leg_targets: Sequence[float],
    classical_errors: Sequence[float],
    previous_residual: Sequence[float],
    stair_mode: bool,
    privileged: Mapping[str, Sequence[float] | float] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    phase_one_hot = np.zeros(PHASE_COUNT, dtype=np.float32)
    phase_one_hot[int(phase)] = 1.0
    actor = np.concatenate(
        (
            np.asarray(proprioception, dtype=np.float32).reshape(-1),
            phase_one_hot,
            np.asarray(classical_wheel_baseline, dtype=np.float32).reshape(2),
            np.asarray(nominal_leg_targets, dtype=np.float32).reshape(4),
            np.asarray(classical_errors, dtype=np.float32).reshape(-1),
            np.asarray(previous_residual, dtype=np.float32).reshape(6),
            np.asarray([float(stair_mode)], dtype=np.float32),
        )
    )
    critic_parts = [actor]
    for name in PRIVILEGED_FIELD_NAMES:
        if privileged is None or name not in privileged:
            critic_parts.append(np.zeros(1, dtype=np.float32))
        else:
            critic_parts.append(
                np.asarray(privileged[name], dtype=np.float32).reshape(-1)
            )
    return actor, np.concatenate(critic_parts)


@dataclass(frozen=True)
class StairCurriculumState:
    lower_height_m: float
    upper_height_m: float
    consecutive_ready_evaluations: int = 0


def update_stair_curriculum(
    state: StairCurriculumState,
    *,
    success_rate: float,
    maximum_height_m: float = 0.15,
) -> StairCurriculumState:
    ready = state.consecutive_ready_evaluations + 1 if success_rate >= 0.80 else 0
    upper = state.upper_height_m
    if ready >= 3:
        upper = min(maximum_height_m, round(upper + 0.01, 10))
        ready = 0
    return StairCurriculumState(
        lower_height_m=state.lower_height_m,
        upper_height_m=upper,
        consecutive_ready_evaluations=ready,
    )


def highest_contiguous_passing_height(
    rows: Sequence[Mapping[str, float | int]],
) -> float | None:
    highest: float | None = None
    for row in sorted(rows, key=lambda item: float(item["height_m"])):
        passed = (
            float(row["success_rate"]) >= 0.90
            and int(row["terminations"]) == 0
            and int(row["non_wheel_contacts"]) == 0
        )
        if not passed:
            break
        highest = float(row["height_m"])
    return highest


def residual_promotion_decision(
    *,
    classical_rows: Sequence[Mapping[str, float | int]],
    residual_rows: Sequence[Mapping[str, float | int]],
    flat_gate_passed: bool,
    standing_gate_passed: bool,
    velocity_gate_passed: bool,
    stage5_gate_passed: bool,
    ablations_complete: bool,
) -> dict[str, object]:
    classical_height = highest_contiguous_passing_height(classical_rows)
    residual_height = highest_contiguous_passing_height(residual_rows)
    improved = (
        classical_height is not None
        and residual_height is not None
        and residual_height >= classical_height + 0.01 - 1.0e-12
    )
    regressions_clear = all(
        (
            flat_gate_passed,
            standing_gate_passed,
            velocity_gate_passed,
            stage5_gate_passed,
        )
    )
    passed = improved and regressions_clear and ablations_complete
    return {
        "promotion_eligible": passed,
        "classification": (
            "RESIDUAL_PPO_EXTENDS_CLASSICAL_BOUNDARY" if passed else "STOP_NO_PROMOTION"
        ),
        "classical_height_m": classical_height,
        "residual_height_m": residual_height,
        "boundary_extension_m": (
            None
            if classical_height is None or residual_height is None
            else residual_height - classical_height
        ),
        "regression_gates_passed": regressions_clear,
        "ablations_complete": ablations_complete,
    }
