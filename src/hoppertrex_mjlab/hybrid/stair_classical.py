"""Deployable stair-mode classical controller and offline CEM utilities."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

MANEUVER_ARTIFACT_TYPE = "classical_stair_maneuver"
CONTROL_DT_S = 0.02
ARM_DISTANCE_M = 0.25
WHEEL_VELOCITY_LIMIT = 12.0
WHEEL_SLEW_LIMIT = 6.0
NOMINAL_WHEEL_RADIUS_M = 0.1
PHASE_COUNT = 9


class StairPhase(IntEnum):
    IDLE = 0
    APPROACH = 1
    PRELOAD = 2
    CONTACT_WAIT = 3
    CLIMB = 4
    CREST = 5
    RECOVER = 6
    DONE = 7
    ABORT = 8


@dataclass(frozen=True)
class ContactDetectorCfg:
    pitch_rate_delta: float
    wheel_speed_error: float
    body_deceleration: float
    consecutive_ticks: int = 2

    def __post_init__(self) -> None:
        values = (
            self.pitch_rate_delta,
            self.wheel_speed_error,
            self.body_deceleration,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Contact detector thresholds must be finite and positive.")
        if self.consecutive_ticks < 1:
            raise ValueError("Contact detector consecutive_ticks must be positive.")


@dataclass(frozen=True)
class ContactDetectorState:
    previous_pitch_rate: float = 0.0
    previous_body_vx: float = 0.0
    consecutive_hits: int = 0


def contact_detector_step(
    cfg: ContactDetectorCfg,
    state: ContactDetectorState,
    *,
    pitch_rate: float,
    wheel_speed_error: float,
    body_vx: float,
    dt: float = CONTROL_DT_S,
) -> tuple[bool, ContactDetectorState, tuple[bool, bool, bool]]:
    if dt <= 0.0:
        raise ValueError("Contact detector dt must be positive.")
    pitch_hit = abs(pitch_rate - state.previous_pitch_rate) >= cfg.pitch_rate_delta
    wheel_hit = abs(wheel_speed_error) >= cfg.wheel_speed_error
    deceleration = max(0.0, (state.previous_body_vx - body_vx) / dt)
    decel_hit = deceleration >= cfg.body_deceleration
    votes = (pitch_hit, wheel_hit, decel_hit)
    hits = state.consecutive_hits + 1 if sum(votes) >= 2 else 0
    next_state = ContactDetectorState(
        previous_pitch_rate=float(pitch_rate),
        previous_body_vx=float(body_vx),
        consecutive_hits=hits,
    )
    return hits >= cfg.consecutive_ticks, next_state, votes


def qualify_contact_detector(
    cfg: ContactDetectorCfg,
    *,
    flat_sequences: Sequence[Sequence[tuple[float, float, float]]],
    stair_sequences: Sequence[Sequence[tuple[float, float, float]]],
    impact_indices: Sequence[int],
    max_delay_ticks: int = 3,
) -> dict[str, Any]:
    if len(stair_sequences) != len(impact_indices):
        raise ValueError("Each stair sequence requires one impact index.")

    def detections(sequence: Sequence[tuple[float, float, float]]) -> list[int]:
        state = ContactDetectorState()
        found: list[int] = []
        for index, (pitch_rate, wheel_error, body_vx) in enumerate(sequence):
            detected, state, _ = contact_detector_step(
                cfg,
                state,
                pitch_rate=pitch_rate,
                wheel_speed_error=wheel_error,
                body_vx=body_vx,
            )
            if detected:
                found.append(index)
        return found

    flat_false_positives = sum(
        bool(detections(sequence)) for sequence in flat_sequences
    )
    timely = 0
    delays: list[int] = []
    for sequence, impact in zip(stair_sequences, impact_indices, strict=True):
        after = [index for index in detections(sequence) if index >= impact]
        if not after:
            continue
        delay = after[0] - impact
        delays.append(delay)
        timely += int(delay <= max_delay_ticks)
    detection_rate = timely / len(stair_sequences) if stair_sequences else 0.0
    return {
        "qualified": flat_false_positives == 0 and detection_rate >= 0.95,
        "flat_false_positive_sequences": flat_false_positives,
        "stair_sequence_count": len(stair_sequences),
        "timely_detection_count": timely,
        "timely_detection_rate": detection_rate,
        "detection_delays_ticks": delays,
        "max_delay_ticks": max_delay_ticks,
    }


@dataclass(frozen=True)
class StairManeuver:
    approach_vx: float
    preload_trigger_m: float
    preload_duration_s: float
    preload_height_m: float
    preload_pitch_rad: float
    contact_vx: float
    climb_vx: float
    drive_feedforward_radps: float
    climb_height_m: float
    climb_pitch_rad: float
    climb_timeout_s: float
    crest_progress_m: float
    recover_duration_s: float
    detector: ContactDetectorCfg
    maneuver_hash: str
    bindings: dict[str, str]
    source: str = "memory"


@dataclass(frozen=True)
class StairControllerState:
    phase: StairPhase = StairPhase.IDLE
    phase_elapsed_s: float = 0.0
    local_progress_m: float = 0.0
    previous_signed_wheel_speed: float = 0.0
    detector_state: ContactDetectorState = ContactDetectorState()
    abort_reason: str | None = None


@dataclass(frozen=True)
class StairSensors:
    pitch: float
    pitch_rate: float
    body_vx: float
    signed_wheel_speed: float
    wheel_speed_error: float
    non_wheel_contact: bool = False
    actuator_limit: bool = False


@dataclass(frozen=True)
class StairTargets:
    vx: float
    height: float
    pitch: float
    drive_feedforward_radps: float
    phase: StairPhase
    contact_detected: bool
    abort_reason: str | None

    def phase_one_hot(self) -> tuple[float, ...]:
        return tuple(float(index == int(self.phase)) for index in range(PHASE_COUNT))


def _lerp(start: float, end: float, fraction: float) -> float:
    return start + min(max(fraction, 0.0), 1.0) * (end - start)


def stair_controller_step(
    maneuver: StairManeuver,
    state: StairControllerState,
    sensors: StairSensors,
    *,
    stair_mode: bool,
    nominal_height: float,
    nominal_pitch: float,
    dt: float = CONTROL_DT_S,
) -> tuple[StairTargets, StairControllerState]:
    if dt <= 0.0:
        raise ValueError("Stair controller dt must be positive.")
    phase = state.phase
    elapsed = state.phase_elapsed_s + dt
    progress = state.local_progress_m
    if stair_mode and phase == StairPhase.IDLE:
        phase = StairPhase.APPROACH
        elapsed = 0.0
        progress = 0.0
    if not stair_mode and phase not in (StairPhase.IDLE, StairPhase.DONE):
        phase = StairPhase.ABORT

    progress += sensors.signed_wheel_speed * NOMINAL_WHEEL_RADIUS_M * dt
    detected, detector_state, _ = contact_detector_step(
        maneuver.detector,
        state.detector_state,
        pitch_rate=sensors.pitch_rate,
        wheel_speed_error=sensors.wheel_speed_error,
        body_vx=sensors.body_vx,
        dt=dt,
    )
    abort_reason: str | None = state.abort_reason
    if sensors.non_wheel_contact:
        phase, abort_reason = StairPhase.ABORT, "non_wheel_contact"
    elif sensors.actuator_limit:
        phase, abort_reason = StairPhase.ABORT, "actuator_limit"
    elif abs(sensors.pitch) > 0.35:
        phase, abort_reason = StairPhase.ABORT, "pitch_limit"
    elif progress < -0.10 or progress > 1.0:
        phase, abort_reason = StairPhase.ABORT, "odometry_limit"

    if phase == StairPhase.APPROACH and progress >= maneuver.preload_trigger_m:
        phase, elapsed = StairPhase.PRELOAD, 0.0
    elif phase == StairPhase.PRELOAD and elapsed >= maneuver.preload_duration_s:
        phase, elapsed = StairPhase.CONTACT_WAIT, 0.0
    elif phase == StairPhase.CONTACT_WAIT:
        if detected:
            phase, elapsed = StairPhase.CLIMB, 0.0
        elif progress >= ARM_DISTANCE_M + 0.10:
            phase, abort_reason = StairPhase.ABORT, "contact_timeout"
    elif phase == StairPhase.CLIMB:
        if progress >= maneuver.crest_progress_m:
            phase, elapsed = StairPhase.CREST, 0.0
        elif elapsed >= maneuver.climb_timeout_s:
            phase, abort_reason = StairPhase.ABORT, "climb_timeout"
    elif phase == StairPhase.CREST and elapsed >= 0.5:
        phase, elapsed = StairPhase.RECOVER, 0.0
    elif phase == StairPhase.RECOVER and elapsed >= maneuver.recover_duration_s:
        phase, elapsed = StairPhase.DONE, 0.0

    vx = 0.0
    height = nominal_height
    pitch = nominal_pitch
    feedforward = 0.0
    if phase == StairPhase.APPROACH:
        vx = maneuver.approach_vx
    elif phase == StairPhase.PRELOAD:
        fraction = elapsed / maneuver.preload_duration_s
        vx = _lerp(maneuver.approach_vx, maneuver.contact_vx, fraction)
        height = _lerp(nominal_height, maneuver.preload_height_m, fraction)
        pitch = _lerp(nominal_pitch, maneuver.preload_pitch_rad, fraction)
    elif phase == StairPhase.CONTACT_WAIT:
        vx = maneuver.contact_vx
        height = maneuver.preload_height_m
        pitch = maneuver.preload_pitch_rad
    elif phase in (StairPhase.CLIMB, StairPhase.CREST):
        vx = maneuver.climb_vx
        height = maneuver.climb_height_m
        pitch = maneuver.climb_pitch_rad
        feedforward = maneuver.drive_feedforward_radps
    elif phase == StairPhase.RECOVER:
        fraction = elapsed / maneuver.recover_duration_s
        vx = _lerp(maneuver.climb_vx, 0.0, fraction)
        height = _lerp(maneuver.climb_height_m, nominal_height, fraction)
        pitch = _lerp(maneuver.climb_pitch_rad, nominal_pitch, fraction)
        feedforward = _lerp(maneuver.drive_feedforward_radps, 0.0, fraction)
    targets = StairTargets(
        vx=vx,
        height=height,
        pitch=pitch,
        drive_feedforward_radps=feedforward,
        phase=phase,
        contact_detected=detected,
        abort_reason=abort_reason,
    )
    return targets, StairControllerState(
        phase=phase,
        phase_elapsed_s=elapsed,
        local_progress_m=progress,
        previous_signed_wheel_speed=sensors.signed_wheel_speed,
        detector_state=detector_state,
        abort_reason=abort_reason,
    )


def _canonical_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "maneuver_hash"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def parse_stair_maneuver(
    payload: dict[str, Any], *, source: str = "memory"
) -> StairManeuver:
    if payload.get("schema_version") != 1:
        raise ValueError("Stair maneuver schema_version must be 1.")
    if payload.get("artifact_type") != MANEUVER_ARTIFACT_TYPE:
        raise ValueError("Stair maneuver artifact_type is invalid.")
    if payload.get("known_step_height") is not None:
        raise ValueError("Deployable maneuver may not depend on known step height.")
    if float(payload.get("arm_distance_m", math.nan)) != ARM_DISTANCE_M:
        raise ValueError("Stair maneuver arm distance must be 0.25 m.")
    parameters = payload.get("parameters")
    detector = payload.get("contact_detector")
    bindings = payload.get("bindings")
    if not isinstance(parameters, dict) or not isinstance(detector, dict):
        raise TypeError("Stair maneuver parameters and detector are required.")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("Stair maneuver artifact bindings are required.")
    expected = _canonical_hash(payload)
    if payload.get("maneuver_hash") != expected:
        raise ValueError("Stair maneuver hash does not match artifact data.")
    cfg = ContactDetectorCfg(
        pitch_rate_delta=float(detector["pitch_rate_delta"]),
        wheel_speed_error=float(detector["wheel_speed_error"]),
        body_deceleration=float(detector["body_deceleration"]),
        consecutive_ticks=int(detector.get("consecutive_ticks", 2)),
    )
    maneuver = StairManeuver(
        **{key: float(value) for key, value in parameters.items()},
        detector=cfg,
        maneuver_hash=expected,
        bindings={str(key): str(value) for key, value in bindings.items()},
        source=source,
    )
    if not 0.0 < maneuver.preload_trigger_m < ARM_DISTANCE_M:
        raise ValueError("Preload trigger must lie between arming point and riser.")
    if not 0.0 <= maneuver.drive_feedforward_radps <= 2.0:
        raise ValueError("Drive feedforward must stay within [0, 2] rad/s.")
    if maneuver.crest_progress_m <= ARM_DISTANCE_M:
        raise ValueError("Crest progress must lie beyond the first riser.")
    return maneuver


def load_stair_maneuver(path: Path) -> StairManeuver:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Stair maneuver must contain a JSON object.")
    return parse_stair_maneuver(payload, source=str(path.resolve()))


@dataclass(frozen=True)
class CandidateScore:
    safe_successes: int
    median_progress: float
    peak_pitch: float
    energy: float
    target_smoothness: float
    unsafe_trials: int = 0

    def rank(self) -> tuple[float, ...]:
        return (
            float(self.unsafe_trials == 0),
            float(self.safe_successes),
            float(self.median_progress),
            -float(self.peak_pitch),
            -float(self.energy),
            -float(self.target_smoothness),
        )


@dataclass(frozen=True)
class CemResult:
    parameters: NDArray[np.float64]
    score: CandidateScore
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    history: tuple[dict[str, Any], ...]


def classical_plateau_decision(
    rounds: Sequence[dict[str, float]],
) -> dict[str, Any]:
    """Freeze only after two consecutive rounds fail both improvement tests."""

    if len(rounds) < 3:
        return {"freeze": False, "reason": "need_three_rounds"}
    last = rounds[-3:]
    stalled = []
    for previous, current in pairwise(last):
        height_gain = current["highest_contiguous_pass_m"] - previous[
            "highest_contiguous_pass_m"
        ]
        rate_gain = current["first_failure_success_rate"] - previous[
            "first_failure_success_rate"
        ]
        stalled.append(height_gain < 0.01 - 1.0e-12 and rate_gain < 0.05)
    best = max(
        enumerate(rounds),
        key=lambda item: (
            item[1]["highest_contiguous_pass_m"],
            item[1]["first_failure_success_rate"],
            -item[1].get("peak_pitch", math.inf),
            -item[1].get("energy", math.inf),
        ),
    )
    return {
        "freeze": all(stalled),
        "reason": "stable_platform" if all(stalled) else "continue_optimization",
        "selected_round_index": best[0],
    }


def optimize_cem(
    evaluate: Callable[[NDArray[np.float64]], CandidateScore],
    *,
    lower: NDArray[np.floating],
    upper: NDArray[np.floating],
    population: int = 256,
    elite_fraction: float = 0.10,
    iterations: int = 20,
    seed: int = 1,
    smoothing: float = 0.25,
) -> CemResult:
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    if lower_array.shape != upper_array.shape or np.any(lower_array >= upper_array):
        raise ValueError("CEM bounds must have equal shape and lower < upper.")
    if population < 2 or iterations < 1 or not 0.0 < elite_fraction < 1.0:
        raise ValueError("CEM population, iterations, and elite fraction are invalid.")
    elite_count = max(1, math.ceil(population * elite_fraction))
    rng = np.random.default_rng(seed)
    mean = 0.5 * (lower_array + upper_array)
    std = 0.5 * (upper_array - lower_array)
    minimum_std = 0.01 * (upper_array - lower_array)
    best_parameters = mean.copy()
    best_score = evaluate(best_parameters)
    history: list[dict[str, Any]] = []
    for iteration in range(iterations):
        samples = np.clip(
            rng.normal(mean, std, size=(population, mean.size)),
            lower_array,
            upper_array,
        )
        scored = [(evaluate(sample), sample) for sample in samples]
        scored.sort(key=lambda item: item[0].rank(), reverse=True)
        elites = np.stack([sample for _, sample in scored[:elite_count]])
        elite_mean = np.mean(elites, axis=0)
        elite_std = np.std(elites, axis=0)
        mean = (1.0 - smoothing) * mean + smoothing * elite_mean
        std = np.maximum(
            (1.0 - smoothing) * std + smoothing * elite_std,
            minimum_std,
        )
        if scored[0][0].rank() > best_score.rank():
            best_score, best_parameters = scored[0][0], scored[0][1].copy()
        history.append(
            {
                "iteration": iteration,
                "best_rank": list(scored[0][0].rank()),
                "mean": mean.tolist(),
                "std": std.tolist(),
            }
        )
    return CemResult(
        parameters=best_parameters,
        score=best_score,
        mean=mean,
        std=std,
        history=tuple(history),
    )
