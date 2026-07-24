"""Qualified gain-scheduled LQR artifacts for the classical upper bound."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hoppertrex_mjlab.hybrid.identification import (
    CONTROLLER_STATE_NAMES,
    NOMINAL_WHEEL_RADIUS_M,
)

SCHEDULE_ARTIFACT_TYPE = "gain_scheduled_lqr"
SCHEDULE_STATE_DEFINITION = "hybrid_lqr_equilibrium_pitch_v2"
BASE_Q_DIAG = (20.0, 2.0, 4.0, 0.5)
BASE_R_DIAG = (1.0,)
SCALE_GRID = (0.5, 1.0, 2.0)
NRMSE_LIMIT = 0.15
SELECTION_METRICS = (
    "worst_velocity_error",
    "p95_pitch",
    "p99_pitch_rate",
    "wheel_target_rate",
)


def _require_hex_digest(value: Any, length: int, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(
        rf"[0-9a-f]{{{length}}}", value
    ) is None:
        raise ValueError(f"{name} must be a lowercase {length}-character hex digest.")
    return value


def canonical_hash(payload: dict[str, Any], *, hash_field: str) -> str:
    body = {key: value for key, value in payload.items() if key != hash_field}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def select_symmetric_pitch_nodes(
    qualification: dict[float, Iterable[bool]],
    *,
    candidates: tuple[float, ...] = (0.032, 0.024, 0.016),
) -> tuple[float, float, float]:
    """Choose the widest pitch bound that passes at all registered heights."""

    for bound in candidates:
        negative = qualification.get(-bound)
        zero = qualification.get(0.0)
        positive = qualification.get(bound)
        if negative is None or zero is None or positive is None:
            continue
        if all(negative) and all(zero) and all(positive):
            return (-bound, 0.0, bound)
    raise ValueError(
        "No qualified symmetric pitch range contains zero at all height nodes."
    )


def qr_candidate_grid() -> tuple[dict[str, Any], ...]:
    candidates: list[dict[str, Any]] = []
    for balance_scale in SCALE_GRID:
        for drive_scale in SCALE_GRID:
            for r_scale in SCALE_GRID:
                candidates.append(
                    {
                        "balance_scale": balance_scale,
                        "drive_scale": drive_scale,
                        "r_scale": r_scale,
                        "q_diag": [
                            BASE_Q_DIAG[0] * balance_scale,
                            BASE_Q_DIAG[1] * balance_scale,
                            BASE_Q_DIAG[2] * drive_scale,
                            BASE_Q_DIAG[3] * drive_scale,
                        ],
                        "r_diag": [BASE_R_DIAG[0] * r_scale],
                    }
                )
    return tuple(candidates)


def validate_flat_gate_selection(
    selection: Any,
    *,
    selected_q_diag: tuple[float, ...],
    selected_r_diag: tuple[float, ...],
) -> None:
    """Require auditable results for all 27 registered Q/R candidates."""

    if (
        not isinstance(selection, dict)
        or selection.get("status") != "flat_gate_selected"
    ):
        raise ValueError(
            "Controller schedule must be selected by the flat safety gate."
        )
    _require_hex_digest(
        selection.get("evaluation_artifact_sha256"),
        64,
        "Flat safety evaluation artifact SHA256",
    )
    for field in ("git_sha", "mjlab_git_sha"):
        _require_hex_digest(selection.get(field), 40, field)
    evaluated = selection.get("evaluated_candidates")
    if not isinstance(evaluated, list) or len(evaluated) != 27:
        raise ValueError("Flat safety selection must contain 27 evaluated candidates.")
    expected = {
        (tuple(item["q_diag"]), tuple(item["r_diag"]))
        for item in qr_candidate_grid()
    }
    observed: set[tuple[tuple[float, ...], tuple[float, ...]]] = set()
    for candidate in evaluated:
        if not isinstance(candidate, dict):
            raise TypeError("Each flat-gate candidate must be a JSON object.")
        q_diag = tuple(float(value) for value in candidate.get("q_diag", ()))
        r_diag = tuple(float(value) for value in candidate.get("r_diag", ()))
        observed.add((q_diag, r_diag))
        if not isinstance(candidate.get("flat_gate_passed"), bool):
            raise ValueError("Each flat-gate candidate must record pass/fail.")
        for metric in SELECTION_METRICS:
            value = candidate.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"Each flat-gate candidate must record finite {metric}."
                )
            if float(value) < 0.0:
                raise ValueError(f"Flat-gate metric {metric} must be non-negative.")
    if observed != expected:
        raise ValueError(
            "Flat safety selection does not cover the registered Q/R grid."
        )
    selected_index = selection.get("selected_candidate_index")
    if not isinstance(selected_index, int) or not 0 <= selected_index < 27:
        raise ValueError("Flat safety selection requires a valid selected index.")
    selected = evaluated[selected_index]
    if not selected["flat_gate_passed"]:
        raise ValueError("Selected Q/R candidate did not pass the flat safety gate.")
    passed_indices = [
        index
        for index, candidate in enumerate(evaluated)
        if candidate["flat_gate_passed"]
    ]
    if not passed_indices:
        raise ValueError("No Q/R candidate passed the flat safety gate.")
    best_index = min(
        passed_indices,
        key=lambda index: tuple(
            float(evaluated[index][metric]) for metric in SELECTION_METRICS
        ),
    )
    if selected_index != best_index:
        raise ValueError("Selected Q/R candidate is not the registered best candidate.")
    if tuple(float(value) for value in selected["q_diag"]) != selected_q_diag:
        raise ValueError("Selected Q diagonal does not match the schedule artifact.")
    if tuple(float(value) for value in selected["r_diag"]) != selected_r_diag:
        raise ValueError("Selected R diagonal does not match the schedule artifact.")


@dataclass(frozen=True)
class ControllerSchedule:
    height_nodes: tuple[float, ...]
    pitch_nodes: tuple[float, ...]
    gains: NDArray[np.float64]
    equilibrium_pitch: NDArray[np.float64]
    schedule_hash: str
    q_diag: tuple[float, float, float, float]
    r_diag: tuple[float]
    bindings: dict[str, str]
    source: str

    @property
    def qualified(self) -> bool:
        return True

    def interpolate(
        self,
        height: float,
        pitch_command: float,
    ) -> tuple[NDArray[np.float64], float, bool]:
        gain = bilinear_interpolate(
            self.height_nodes,
            self.pitch_nodes,
            self.gains,
            height,
            pitch_command,
        )
        equilibrium = bilinear_interpolate(
            self.height_nodes,
            self.pitch_nodes,
            self.equilibrium_pitch,
            height,
            pitch_command,
        )
        clamped = not (
            self.height_nodes[0] <= height <= self.height_nodes[-1]
            and self.pitch_nodes[0] <= pitch_command <= self.pitch_nodes[-1]
        )
        return np.asarray(gain, dtype=np.float64), float(equilibrium), clamped


def _axis_interval(nodes: tuple[float, ...], value: float) -> tuple[int, int, float]:
    clipped = min(max(float(value), nodes[0]), nodes[-1])
    upper = int(np.searchsorted(np.asarray(nodes), clipped, side="right"))
    upper = min(max(upper, 1), len(nodes) - 1)
    lower = upper - 1
    weight = (clipped - nodes[lower]) / (nodes[upper] - nodes[lower])
    return lower, upper, float(weight)


def bilinear_interpolate(
    height_nodes: tuple[float, ...],
    pitch_nodes: tuple[float, ...],
    values: NDArray[np.floating],
    height: float,
    pitch: float,
) -> NDArray[np.float64] | np.float64:
    h0, h1, hw = _axis_interval(height_nodes, height)
    p0, p1, pw = _axis_interval(pitch_nodes, pitch)
    array = np.asarray(values, dtype=np.float64)
    low = (1.0 - pw) * array[h0, p0] + pw * array[h0, p1]
    high = (1.0 - pw) * array[h1, p0] + pw * array[h1, p1]
    return (1.0 - hw) * low + hw * high


def _finite_vector(
    value: Any, shape: tuple[int, ...], name: str
) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape {shape} and finite values.")
    return array


def parse_controller_schedule(
    payload: dict[str, Any],
    *,
    source: str = "memory",
) -> ControllerSchedule:
    if payload.get("schema_version") != 1:
        raise ValueError("Controller schedule schema_version must be 1.")
    if payload.get("artifact_type") != SCHEDULE_ARTIFACT_TYPE:
        raise ValueError("Controller schedule artifact_type is invalid.")
    if tuple(payload.get("state_names", ())) != CONTROLLER_STATE_NAMES:
        raise ValueError("Controller schedule state_names do not match runtime.")
    state = payload.get("state_construction")
    if not isinstance(state, dict):
        raise TypeError("Controller schedule requires state_construction.")
    if state.get("state_definition_version") != SCHEDULE_STATE_DEFINITION:
        raise ValueError("Controller schedule state definition is incompatible.")
    radius = state.get("wheel_radius")
    if not isinstance(radius, (int, float)) or isinstance(radius, bool):
        raise TypeError("Controller schedule wheel_radius must be numeric.")
    if abs(float(radius) - NOMINAL_WHEEL_RADIUS_M) > 1.0e-9:
        raise ValueError("Controller schedule wheel radius does not match runtime.")

    height_nodes = tuple(float(value) for value in payload.get("height_nodes", ()))
    pitch_nodes = tuple(float(value) for value in payload.get("pitch_nodes", ()))
    if len(height_nodes) != 3 or len(pitch_nodes) != 3:
        raise ValueError("Controller schedule must contain a 3x3 node grid.")
    if not all(math.isfinite(value) for value in (*height_nodes, *pitch_nodes)):
        raise ValueError("Controller schedule nodes must be finite.")
    if tuple(sorted(height_nodes)) != height_nodes or len(set(height_nodes)) != 3:
        raise ValueError("Controller schedule height nodes must strictly increase.")
    if tuple(sorted(pitch_nodes)) != pitch_nodes or len(set(pitch_nodes)) != 3:
        raise ValueError("Controller schedule pitch nodes must strictly increase.")
    if pitch_nodes[1] != 0.0 or not math.isclose(
        pitch_nodes[0], -pitch_nodes[2], abs_tol=1.0e-12
    ):
        raise ValueError(
            "Controller schedule pitch nodes must be symmetric about zero."
        )

    nodes = payload.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 3:
        raise ValueError("Controller schedule nodes must contain three height rows.")
    gains = np.zeros((3, 3, 4), dtype=np.float64)
    equilibria = np.zeros((3, 3), dtype=np.float64)
    for h_index, row in enumerate(nodes):
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("Each controller schedule row must contain three nodes.")
        for p_index, node in enumerate(row):
            if not isinstance(node, dict):
                raise TypeError("Controller schedule node must be a JSON object.")
            if node.get("controller_type") != "lqr":
                raise ValueError("Every schedule node must be a qualified LQR.")
            rank = node.get("controllability_rank")
            if isinstance(rank, bool) or not isinstance(rank, int) or rank != 4:
                raise ValueError(
                    "Every schedule node must have controllability rank four."
                )
            nrmse = node.get("heldout_one_step_nrmse")
            maximum = nrmse.get("maximum") if isinstance(nrmse, dict) else None
            if (
                isinstance(maximum, bool)
                or not isinstance(maximum, (int, float))
                or not math.isfinite(float(maximum))
                or not 0.0 <= float(maximum) <= NRMSE_LIMIT
            ):
                raise ValueError("Every schedule node must satisfy the NRMSE limit.")
            if node.get("fallback_reasons") not in ([], ()):
                raise ValueError("Controller schedule nodes may not use a fallback.")
            model = node.get("model")
            if not isinstance(model, dict):
                raise ValueError(
                    "Controller schedule node must retain its fitted model."
                )
            _finite_vector(model.get("a"), (4, 4), "model.a")
            _finite_vector(model.get("b"), (4, 1), "model.b")
            if not isinstance(node.get("source_npz"), str) or not node["source_npz"]:
                raise ValueError("Controller schedule node must retain source_npz.")
            for hash_field in (
                "controller_file_sha256",
                "source_npz_sha256",
                "source_metadata_sha256",
            ):
                _require_hex_digest(node.get(hash_field), 64, hash_field)
            gains[h_index, p_index] = _finite_vector(node.get("gain"), (4,), "gain")
            equilibrium = node.get("equilibrium_pitch")
            if not isinstance(equilibrium, (int, float)) or not math.isfinite(
                float(equilibrium)
            ):
                raise ValueError("Node equilibrium_pitch must be finite.")
            equilibria[h_index, p_index] = float(equilibrium)

    q_diag = tuple(float(value) for value in payload.get("q_diag", ()))
    r_diag = tuple(float(value) for value in payload.get("r_diag", ()))
    if (
        len(q_diag) != 4
        or len(r_diag) != 1
        or not all(math.isfinite(value) and value > 0.0 for value in (*q_diag, *r_diag))
    ):
        raise ValueError("Controller schedule Q/R diagonals must be positive.")
    protocol = payload.get("collection_protocol")
    expected_protocol = {
        "num_envs": 32,
        "steps": 2500,
        "warmup_steps": 250,
        "hold_steps": 5,
        "heldout_fraction": 0.20,
    }
    if protocol != expected_protocol:
        raise ValueError("Controller schedule collection protocol is not frozen.")
    bindings = payload.get("bindings")
    if not isinstance(bindings, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value
        for key, value in bindings.items()
    ):
        raise ValueError("Controller schedule requires non-empty artifact bindings.")
    required_bindings = {
        "identification_controller_gain_hash",
        "identification_calibration_hash",
        "posture_artifact_hash",
    }
    if not required_bindings.issubset(bindings):
        raise ValueError(
            "Controller schedule is missing identification/posture bindings."
        )
    for key in required_bindings:
        _require_hex_digest(bindings[key], 64, f"bindings.{key}")
    validate_flat_gate_selection(
        payload.get("selection"),
        selected_q_diag=q_diag,
        selected_r_diag=r_diag,
    )
    expected_hash = canonical_hash(payload, hash_field="schedule_hash")
    if payload.get("schedule_hash") != expected_hash:
        raise ValueError("Controller schedule hash does not match artifact data.")
    return ControllerSchedule(
        height_nodes=height_nodes,
        pitch_nodes=pitch_nodes,
        gains=gains,
        equilibrium_pitch=equilibria,
        schedule_hash=expected_hash,
        q_diag=q_diag,  # type: ignore[arg-type]
        r_diag=r_diag,
        bindings=dict(bindings),
        source=source,
    )


def load_controller_schedule(path: Path) -> ControllerSchedule:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Controller schedule must contain a JSON object.")
    return parse_controller_schedule(payload, source=str(path.resolve()))
