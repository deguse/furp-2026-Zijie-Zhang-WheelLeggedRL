"""Qualified gain-scheduled LQR artifacts for the classical upper bound."""

from __future__ import annotations

import hashlib
import json
import math
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
            if int(node.get("controllability_rank", -1)) != 4:
                raise ValueError(
                    "Every schedule node must have controllability rank four."
                )
            nrmse = node.get("heldout_one_step_nrmse")
            maximum = nrmse.get("maximum") if isinstance(nrmse, dict) else None
            if not isinstance(maximum, (int, float)) or float(maximum) > NRMSE_LIMIT:
                raise ValueError("Every schedule node must satisfy the NRMSE limit.")
            if node.get("fallback_reasons") not in ([], ()):
                raise ValueError("Controller schedule nodes may not use a fallback.")
            gains[h_index, p_index] = _finite_vector(node.get("gain"), (4,), "gain")
            equilibrium = node.get("equilibrium_pitch")
            if not isinstance(equilibrium, (int, float)) or not math.isfinite(
                float(equilibrium)
            ):
                raise ValueError("Node equilibrium_pitch must be finite.")
            equilibria[h_index, p_index] = float(equilibrium)

    q_diag = tuple(float(value) for value in payload.get("q_diag", ()))
    r_diag = tuple(float(value) for value in payload.get("r_diag", ()))
    if len(q_diag) != 4 or len(r_diag) != 1 or min(*q_diag, *r_diag) <= 0.0:
        raise ValueError("Controller schedule Q/R diagonals must be positive.")
    bindings = payload.get("bindings")
    if not isinstance(bindings, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value
        for key, value in bindings.items()
    ):
        raise ValueError("Controller schedule requires non-empty artifact bindings.")
    selection = payload.get("selection")
    if (
        not isinstance(selection, dict)
        or selection.get("status") != "flat_gate_selected"
    ):
        raise ValueError(
            "Controller schedule must be selected by the flat safety gate."
        )
    if int(selection.get("candidate_count", -1)) != 27:
        raise ValueError(
            "Controller schedule selection must evaluate 27 Q/R candidates."
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
