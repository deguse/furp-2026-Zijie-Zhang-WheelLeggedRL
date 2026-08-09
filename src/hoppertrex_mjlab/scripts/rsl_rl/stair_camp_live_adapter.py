#!/usr/bin/env python3
"""Real MjLab/RSL-RL live adapter for the registered StairCamp evaluator.

The sibling :mod:`evaluate_stair_camp` module deliberately stays free of
Torch and MjLab. This module owns the opposite side of that seam: it validates
the signed adapter request before importing any heavy dependency, loads the
registered 52-observation StairCamp policy, builds generated terrain variants,
executes the requested rollouts, and returns only aggregate JSON-safe data.

Importing this module is intentionally cheap. Torch, MjLab, RSL-RL, and the
task registry are imported only after :func:`collect` (or the trigger-FP
collector) has accepted the pure request contract.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import contextlib
import copy
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any, Protocol
import uuid


PROJECT_PATH = Path(__file__).resolve().parents[2]
SRC_PATH = Path(__file__).resolve().parents[3]
for _path in (PROJECT_PATH, SRC_PATH):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


LIVE_ADAPTER_SCHEMA_VERSION = 1
TRIGGER_FALSE_POSITIVE_KIND = "stair_camp_trigger_false_positive_check"
TRIGGER_FALSE_POSITIVE_DOMAINS = ("camp_flat_rolling", "stage5_kick")
PRETRAINING_TRIGGER_REQUEST_KIND = "stair_camp_trigger_pretraining_request"
PRETRAINING_ARTIFACT_BINDING_NAMES = (
    "controller_gain_hash",
    "calibration_hash",
    "yaw_calibration_hash",
    "posture_map_hash",
    "posture_artifact_hash",
    "station_calibration_hash",
)
_PRETRAINING_REQUEST_KEYS = frozenset(
    (
        "schema_version",
        "kind",
        "task",
        "evaluation_seed",
        "device",
        "git_sha",
        "contract_sha256",
        "artifact_bindings",
    )
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
TRIGGER_FORCE_N = 18.0
TRIGGER_WINDOW_STEPS = 3
ACTOR_OBSERVATION_WIDTH = 52
CRITIC_OBSERVATION_WIDTH = 55
ACTION_WIDTH = 6
FORMAL_WARMUP_STEPS = 300
FORMAL_WINDOW_STEPS = 800
CONTROL_FREQUENCY_HZ = 50.0
EPISODE_LENGTH_S = 1.0e9

_ADAPTER_CONFIG_KEYS = frozenset(
    (
        "schema_version",
        "kind",
        "task",
        "domain",
        "profile",
        "evaluation_seed",
        "device",
        "evidence_eligible",
        "promotion_evidence_eligible",
        "checkpoint",
        "policy_interface",
        "protocol",
        "gate_bindings",
        "ablation",
        "config_sha256",
    )
)
_TRIAL_KEYS = frozenset(
    (
        "cell",
        "repeat",
        "env_id",
        "success",
        "terminated",
        "non_wheel_contact",
        "stair_mode_false_positive",
        "triggered",
        "pre_impact_triggered",
    )
)
_GATE_OUTCOME_KEYS = frozenset(
    (
        "name",
        "num_envs",
        "steps",
        "scenario_count",
        "kick_events",
        "upstream_gate_passed",
        "terminations",
        "non_wheel_contacts",
        "stair_mode_false_positives",
    )
)
_TRIGGER_OUTCOME_KEYS = frozenset(("events", "stair_mode_false_positives"))


@dataclass(frozen=True, kw_only=True)
class ScanRequest:
    """One exact vectorized stairs or slope scan request."""

    domain: str
    profile: str
    terrain: str
    cell_key: str
    cells: tuple[float, ...]
    num_envs_per_cell: int
    repeats: int
    settle_steps: int
    drive_steps: int
    stable_steps: int
    travel_distance_m: float

    @property
    def events_per_cell(self) -> int:
        return self.num_envs_per_cell * self.repeats

    @property
    def num_envs(self) -> int:
        return len(self.cells) * self.num_envs_per_cell


@dataclass(frozen=True, kw_only=True)
class GateRequest:
    """One gate invocation copied from the signed evaluator config."""

    name: str
    source_suite: str
    terrain: str
    profile: str
    num_envs: int
    steps: int
    scenario_count: int
    commands: tuple[tuple[float, float], ...]
    settle_steps: int
    measure_steps: int
    kick_scale: float | None
    minimum_kick_events: int


@dataclass(frozen=True, kw_only=True)
class TerrainPlan:
    """Pure construction plan used before any MjLab object is allocated."""

    task: str
    domain: str
    profile: str
    terrain: str
    cells: tuple[float, ...]
    num_envs: int
    pushes_enabled: bool
    actor_observation_width: int = ACTOR_OBSERVATION_WIDTH
    critic_observation_width: int = CRITIC_OBSERVATION_WIDTH
    action_width: int = ACTION_WIDTH


class RolloutBackend(Protocol):
    """Small mockable seam around the heavy simulation implementation."""

    def run_scan(self, request: ScanRequest) -> Sequence[Mapping[str, object]]: ...

    def run_gate(self, request: GateRequest) -> Mapping[str, object]: ...

    def run_trigger_false_positive(
        self, domain: str, request: GateRequest
    ) -> Mapping[str, object]: ...

    def metadata(self) -> Mapping[str, object]: ...


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array.")
    return value


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean.")
    return value


def _json_safe(value: object, *, path: str = "payload") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number.")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key.")
            _json_safe(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _json_safe(item, path=f"{path}[{index}]")
        return
    raise ValueError(f"{path} contains non-JSON type {type(value).__name__}.")


def _evaluator() -> Any:
    return importlib.import_module(
        "hoppertrex_mjlab.scripts.rsl_rl.evaluate_stair_camp"
    )


def validate_adapter_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate the evaluator-signed request without importing heavy modules."""

    if set(config) != _ADAPTER_CONFIG_KEYS:
        raise ValueError("StairCamp live-adapter config top-level schema drifted.")
    evaluator = _evaluator()
    validator = getattr(evaluator, "_validate_adapter_config", None)
    if not callable(validator):
        raise RuntimeError("StairCamp evaluator config validator is unavailable.")
    validator(config)
    normalized = dict(config)
    interface = _mapping(normalized["policy_interface"], name="policy_interface")
    if (
        interface.get("actor_observation_width") != ACTOR_OBSERVATION_WIDTH
        or interface.get("critic_observation_width") != CRITIC_OBSERVATION_WIDTH
        or interface.get("action_width") != ACTION_WIDTH
        or interface.get("stage5_actor_adapter_forbidden") is not True
    ):
        raise ValueError("Live adapter accepts only the 52/55 StairCamp interface.")
    return normalized


def validate_pretraining_trigger_request(
    request: Mapping[str, object],
) -> dict[str, object]:
    """Validate checkpoint-free provenance for the two pretraining FP checks."""

    candidate = _mapping(request, name="pretraining_request")
    if set(candidate) != _PRETRAINING_REQUEST_KEYS:
        raise ValueError("Pretraining trigger request top-level schema drifted.")
    if (
        _exact_int(candidate.get("schema_version"), name="schema_version", minimum=1)
        != LIVE_ADAPTER_SCHEMA_VERSION
    ):
        raise ValueError("Pretraining trigger request schema version is unsupported.")
    if candidate.get("kind") != PRETRAINING_TRIGGER_REQUEST_KIND:
        raise ValueError("Pretraining trigger request kind drifted.")

    evaluator = _evaluator()
    task = candidate.get("task")
    if task != evaluator.STAIR_CAMP_TASK_ID:
        raise ValueError("Pretraining trigger request task is not StairCamp.")
    if (
        _exact_int(candidate.get("evaluation_seed"), name="evaluation_seed", minimum=1)
        != evaluator.REGISTERED_EVALUATION_SEED
    ):
        raise ValueError("Pretraining trigger evaluation seed must be exactly 1.")
    device = candidate.get("device")
    if not isinstance(device, str) or not device or device.strip() != device:
        raise ValueError("Pretraining trigger device must be a non-empty exact string.")

    git_sha = candidate.get("git_sha")
    if not isinstance(git_sha, str) or _GIT_SHA_RE.fullmatch(git_sha) is None:
        raise ValueError("Pretraining trigger Git SHA must be lowercase 40-hex.")
    contract_sha256 = candidate.get("contract_sha256")
    if (
        not isinstance(contract_sha256, str)
        or _SHA256_RE.fullmatch(contract_sha256) is None
    ):
        raise ValueError("Pretraining trigger contract SHA256 must be lowercase hex.")
    if contract_sha256 != evaluator.STAIR_CAMP_CANONICAL_CONTRACT_SHA256:
        raise ValueError("Pretraining trigger contract is not canonical.")

    evaluator_artifact_names = tuple(evaluator.STAIR_CAMP_ARTIFACT_BINDING_NAMES)
    if evaluator_artifact_names != PRETRAINING_ARTIFACT_BINDING_NAMES:
        raise RuntimeError("Evaluator artifact binding names drifted.")
    raw_artifacts = _mapping(
        candidate.get("artifact_bindings"), name="artifact_bindings"
    )
    if set(raw_artifacts) != set(PRETRAINING_ARTIFACT_BINDING_NAMES):
        raise ValueError("Pretraining trigger artifact bindings are not the exact six.")
    artifacts: dict[str, str] = {}
    for name in PRETRAINING_ARTIFACT_BINDING_NAMES:
        digest = raw_artifacts.get(name)
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(
                f"Pretraining trigger artifact {name} is not lowercase SHA256."
            )
        artifacts[name] = digest

    normalized: dict[str, object] = {
        "schema_version": LIVE_ADAPTER_SCHEMA_VERSION,
        "kind": PRETRAINING_TRIGGER_REQUEST_KIND,
        "task": task,
        "evaluation_seed": evaluator.REGISTERED_EVALUATION_SEED,
        "device": device,
        "git_sha": git_sha,
        "contract_sha256": contract_sha256,
        "artifact_bindings": artifacts,
    }
    _json_safe(normalized)
    return normalized


def _scan_request(config: Mapping[str, object]) -> ScanRequest:
    protocol = _mapping(config.get("protocol"), name="protocol")
    domain = str(config.get("domain"))
    if domain not in ("stairs", "slope"):
        raise ValueError("Only stairs and slope configs define scan requests.")
    cell_key = protocol.get("cell_key")
    if not isinstance(cell_key, str) or not cell_key:
        raise ValueError("Scan protocol requires a cell key.")
    cells = tuple(
        _finite(value, name=f"protocol.cells[{index}]")
        for index, value in enumerate(
            _sequence(protocol.get("cells"), name="protocol.cells")
        )
    )
    if not cells:
        raise ValueError("Scan protocol requires at least one cell.")
    request = ScanRequest(
        domain=domain,
        profile=str(config.get("profile")),
        terrain=str(protocol.get("terrain")),
        cell_key=cell_key,
        cells=cells,
        num_envs_per_cell=_exact_int(
            protocol.get("num_envs_per_cell"),
            name="protocol.num_envs_per_cell",
            minimum=1,
        ),
        repeats=_exact_int(protocol.get("repeats"), name="protocol.repeats", minimum=1),
        settle_steps=_exact_int(
            protocol.get("settle_steps"), name="protocol.settle_steps", minimum=1
        ),
        drive_steps=_exact_int(
            protocol.get("drive_steps"), name="protocol.drive_steps", minimum=1
        ),
        stable_steps=_exact_int(
            protocol.get("stable_steps"), name="protocol.stable_steps", minimum=1
        ),
        travel_distance_m=_finite(
            protocol.get("travel_distance_m"), name="protocol.travel_distance_m"
        ),
    )
    if request.travel_distance_m <= 0.0:
        raise ValueError("Scan travel distance must be positive.")
    if protocol.get("events_per_cell") != request.events_per_cell:
        raise ValueError("Scan events_per_cell is inconsistent.")
    return request


def _gate_requests(config: Mapping[str, object]) -> tuple[GateRequest, ...]:
    if config.get("domain") != "flat":
        raise ValueError("Gate requests exist only for the flat adapter.")
    raw = _mapping(config.get("gate_bindings"), name="gate_bindings")
    expected_names = (
        "flat_gate_passed",
        "standing_gate_passed",
        "velocity_gate_passed",
        "stage5_gate_passed",
    )
    if tuple(raw) != expected_names:
        raise ValueError("Flat gate binding order or names drifted.")
    result: list[GateRequest] = []
    for name in expected_names:
        binding = _mapping(raw[name], name=f"gate_bindings.{name}")
        commands: list[tuple[float, float]] = []
        for index, command_value in enumerate(
            _sequence(binding.get("commands"), name=f"{name}.commands")
        ):
            command = _sequence(command_value, name=f"{name}.commands[{index}]")
            if len(command) != 2:
                raise ValueError(f"{name}.commands[{index}] must contain two values.")
            commands.append(
                (
                    _finite(command[0], name=f"{name}.commands[{index}][0]"),
                    _finite(command[1], name=f"{name}.commands[{index}][1]"),
                )
            )
        kick_raw = binding.get("kick_scale")
        kick_scale = (
            None if kick_raw is None else _finite(kick_raw, name=f"{name}.kick_scale")
        )
        result.append(
            GateRequest(
                name=name,
                source_suite=str(binding.get("source_suite")),
                terrain=str(binding.get("terrain")),
                profile=str(binding.get("profile")),
                num_envs=_exact_int(
                    binding.get("num_envs"), name=f"{name}.num_envs", minimum=1
                ),
                steps=_exact_int(binding.get("steps"), name=f"{name}.steps", minimum=1),
                scenario_count=_exact_int(
                    binding.get("scenario_count"),
                    name=f"{name}.scenario_count",
                    minimum=1,
                ),
                commands=tuple(commands),
                settle_steps=_exact_int(
                    binding.get("settle_steps"), name=f"{name}.settle_steps"
                ),
                measure_steps=_exact_int(
                    binding.get("measure_steps"), name=f"{name}.measure_steps"
                ),
                kick_scale=kick_scale,
                minimum_kick_events=_exact_int(
                    binding.get("minimum_kick_events"),
                    name=f"{name}.minimum_kick_events",
                ),
            )
        )
    return tuple(result)


def _formal_pretraining_gate_requests() -> dict[str, GateRequest]:
    """Load and pin the two formal FP protocols from the evaluator registry."""

    evaluator = _evaluator()
    bindings = evaluator.gate_bindings_for_profile("formal")
    expected_names = (
        "flat_gate_passed",
        "standing_gate_passed",
        "velocity_gate_passed",
        "stage5_gate_passed",
    )
    if not isinstance(bindings, Mapping) or tuple(bindings) != expected_names:
        raise RuntimeError("Registered formal flat gate bindings drifted.")
    serialized: dict[str, Mapping[str, object]] = {}
    for name in expected_names:
        to_dict = getattr(bindings[name], "to_dict", None)
        if not callable(to_dict):
            raise RuntimeError(f"Registered gate {name} cannot be serialized.")
        serialized[name] = _mapping(to_dict(), name=f"registered gate {name}")
    parsed = {
        request.name: request
        for request in _gate_requests({"domain": "flat", "gate_bindings": serialized})
    }
    expected = {
        "velocity_gate_passed": GateRequest(
            name="velocity_gate_passed",
            source_suite="hybrid_linear_velocity",
            terrain="flat",
            profile="formal",
            num_envs=16,
            steps=3000,
            scenario_count=2,
            commands=((-0.07, 0.0), (0.07, 0.0)),
            settle_steps=0,
            measure_steps=0,
            kick_scale=None,
            minimum_kick_events=0,
        ),
        "stage5_gate_passed": GateRequest(
            name="stage5_gate_passed",
            source_suite="hybrid_robust_stage5_8x",
            terrain="flat",
            profile="formal",
            num_envs=32,
            steps=3000,
            scenario_count=1,
            commands=((0.0, 0.0),),
            settle_steps=0,
            measure_steps=0,
            kick_scale=8.0,
            minimum_kick_events=128,
        ),
    }
    for name, registered in expected.items():
        if parsed.get(name) != registered:
            raise RuntimeError(f"Registered pretraining gate {name} drifted.")
    return expected


def make_terrain_plan(
    config: Mapping[str, object],
    *,
    gate_name: str | None = None,
) -> TerrainPlan:
    """Return the exact terrain/task plan without importing MjLab."""

    normalized = validate_adapter_config(config)
    domain = str(normalized["domain"])
    profile = str(normalized["profile"])
    protocol = _mapping(normalized["protocol"], name="protocol")
    if domain in ("stairs", "slope"):
        if gate_name is not None:
            raise ValueError("Scan terrain plans cannot name a flat gate.")
        scan = _scan_request(normalized)
        cells = scan.cells
        num_envs = scan.num_envs
    else:
        cells = (0.0,)
        if gate_name is None:
            raise ValueError("A flat terrain plan requires one fixed gate name.")
        gates = {request.name: request for request in _gate_requests(normalized)}
        if gate_name not in gates:
            raise ValueError(f"Unknown flat gate name: {gate_name!r}.")
        num_envs = gates[gate_name].num_envs
    pushes = gate_name == "stage5_gate_passed"
    if pushes and domain != "flat":
        raise ValueError("Pushes are legal only in the flat Stage5 gate suite.")
    return TerrainPlan(
        task=str(normalized["task"]),
        domain=domain,
        profile=profile,
        terrain=str(protocol["terrain"]),
        cells=cells,
        num_envs=num_envs,
        pushes_enabled=pushes,
    )


def _normalize_trial(
    value: Mapping[str, object],
    *,
    request: ScanRequest,
    index: int,
) -> dict[str, object]:
    if set(value) != _TRIAL_KEYS:
        raise ValueError(f"Scan trial {index} schema drifted.")
    cell = _finite(value.get("cell"), name=f"trials[{index}].cell")
    matches = [
        candidate for candidate in request.cells if abs(cell - candidate) <= 1.0e-12
    ]
    if len(matches) != 1:
        raise ValueError(f"Scan trial {index} has an unregistered cell.")
    return {
        "cell": float(matches[0]),
        "repeat": _exact_int(
            value.get("repeat"), name=f"trials[{index}].repeat", minimum=1
        ),
        "env_id": _exact_int(value.get("env_id"), name=f"trials[{index}].env_id"),
        "success": _boolean(value.get("success"), name=f"trials[{index}].success"),
        "terminated": _boolean(
            value.get("terminated"), name=f"trials[{index}].terminated"
        ),
        "non_wheel_contact": _boolean(
            value.get("non_wheel_contact"),
            name=f"trials[{index}].non_wheel_contact",
        ),
        "stair_mode_false_positive": _boolean(
            value.get("stair_mode_false_positive"),
            name=f"trials[{index}].stair_mode_false_positive",
        ),
        "triggered": _boolean(
            value.get("triggered"), name=f"trials[{index}].triggered"
        ),
        "pre_impact_triggered": _boolean(
            value.get("pre_impact_triggered"),
            name=f"trials[{index}].pre_impact_triggered",
        ),
    }


def aggregate_scan_trials(
    request: ScanRequest,
    trials: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Strictly reduce one vectorized scan to evaluator row mappings."""

    expected_total = len(request.cells) * request.events_per_cell
    if len(trials) != expected_total:
        raise ValueError(
            f"Scan returned {len(trials)} trials; expected exactly {expected_total}."
        )
    normalized = [
        _normalize_trial(value, request=request, index=index)
        for index, value in enumerate(trials)
    ]
    seen: set[tuple[float, int, int]] = set()
    for index, trial in enumerate(normalized):
        repeat = int(trial["repeat"])
        env_id = int(trial["env_id"])
        if repeat > request.repeats or env_id >= request.num_envs_per_cell:
            raise ValueError(f"Scan trial {index} repeat/env_id is out of range.")
        identity = (float(trial["cell"]), repeat, env_id)
        if identity in seen:
            raise ValueError(f"Duplicate scan trial identity: {identity!r}.")
        seen.add(identity)

    rows: list[dict[str, object]] = []
    for cell in request.cells:
        selected = [trial for trial in normalized if trial["cell"] == cell]
        if len(selected) != request.events_per_cell:
            raise ValueError(f"Scan cell {cell} is incomplete.")
        rows.append(
            {
                request.cell_key: cell,
                "trials": len(selected),
                "successes": sum(bool(trial["success"]) for trial in selected),
                "terminations": sum(bool(trial["terminated"]) for trial in selected),
                "non_wheel_contacts": sum(
                    bool(trial["non_wheel_contact"]) for trial in selected
                ),
                "stair_mode_false_positives": sum(
                    bool(trial["stair_mode_false_positive"]) for trial in selected
                ),
                "trigger_count": sum(bool(trial["triggered"]) for trial in selected),
                "pre_impact_trigger_count": sum(
                    bool(trial["pre_impact_triggered"]) for trial in selected
                ),
            }
        )
    return rows


def _normalize_gate_outcome(
    outcome: Mapping[str, object], request: GateRequest
) -> dict[str, object]:
    if set(outcome) != _GATE_OUTCOME_KEYS:
        raise ValueError(f"Gate {request.name} outcome schema drifted.")
    expected = {
        "name": request.name,
        "num_envs": request.num_envs,
        "steps": request.steps,
        "scenario_count": request.scenario_count,
        "kick_events": request.minimum_kick_events,
    }
    for field, value in expected.items():
        if outcome.get(field) != value:
            raise ValueError(f"Gate {request.name} {field} did not match its binding.")
    return {
        **expected,
        "upstream_gate_passed": _boolean(
            outcome.get("upstream_gate_passed"),
            name=f"{request.name}.upstream_gate_passed",
        ),
        "terminations": _exact_int(
            outcome.get("terminations"), name=f"{request.name}.terminations"
        ),
        "non_wheel_contacts": _exact_int(
            outcome.get("non_wheel_contacts"),
            name=f"{request.name}.non_wheel_contacts",
        ),
        "stair_mode_false_positives": _exact_int(
            outcome.get("stair_mode_false_positives"),
            name=f"{request.name}.stair_mode_false_positives",
        ),
    }


def collect_with_backend(
    config: Mapping[str, object], backend: RolloutBackend
) -> dict[str, object]:
    """Execute a validated config through a supplied (usually real) backend."""

    normalized = validate_adapter_config(config)
    result: dict[str, object] = {
        "config_sha256": normalized["config_sha256"],
        "evaluation_source": "mjlab_rsl_rl_live_adapter",
        "adapter_metadata": dict(_mapping(backend.metadata(), name="backend metadata")),
    }
    if normalized["domain"] == "flat":
        result["gates"] = [
            _normalize_gate_outcome(
                _mapping(backend.run_gate(request), name=f"{request.name} outcome"),
                request,
            )
            for request in _gate_requests(normalized)
        ]
    else:
        request = _scan_request(normalized)
        trials = _sequence(backend.run_scan(request), name="scan trials")
        result["rows"] = aggregate_scan_trials(
            request,
            [
                _mapping(value, name=f"scan trials[{index}]")
                for index, value in enumerate(trials)
            ],
        )
    _json_safe(result)
    return result


def collect(config: Mapping[str, object]) -> Mapping[str, object]:
    """Entry point consumed by ``evaluate_stair_camp.run_live_adapter``."""

    normalized = validate_adapter_config(config)
    dependencies = _load_live_dependencies()
    return collect_with_backend(normalized, _MjLabBackend(normalized, dependencies))


def make_trigger_false_positive_check(
    *,
    domain: str,
    events: int,
    stair_mode_false_positives: int,
) -> dict[str, object]:
    """Build the exact schema consumed by the StairCamp training preflight."""

    if domain not in TRIGGER_FALSE_POSITIVE_DOMAINS:
        raise ValueError(f"Unknown trigger false-positive domain: {domain!r}.")
    event_count = _exact_int(events, name="events", minimum=1)
    false_positives = _exact_int(
        stair_mode_false_positives,
        name="stair_mode_false_positives",
    )
    if false_positives > event_count:
        raise ValueError("Trigger false positives cannot exceed events.")
    return {
        "schema_version": LIVE_ADAPTER_SCHEMA_VERSION,
        "kind": TRIGGER_FALSE_POSITIVE_KIND,
        "domain": domain,
        "threshold_n": TRIGGER_FORCE_N,
        "window_steps": TRIGGER_WINDOW_STEPS,
        "events": event_count,
        "stair_mode_false_positives": false_positives,
        "completed": True,
    }


def collect_trigger_false_positive_with_backend(
    request: Mapping[str, object], backend: RolloutBackend
) -> dict[str, object]:
    """Mockable implementation of the two checkpoint-free live FP checks."""

    candidate = _mapping(request, name="trigger false-positive request")
    if set(candidate) != {"domain", "pretraining_request"}:
        raise ValueError("Trigger false-positive request schema drifted.")
    domain = candidate.get("domain")
    if domain not in TRIGGER_FALSE_POSITIVE_DOMAINS:
        raise ValueError("Trigger false-positive domain is not registered.")
    validate_pretraining_trigger_request(
        _mapping(candidate.get("pretraining_request"), name="pretraining_request")
    )

    gates = _formal_pretraining_gate_requests()
    gate_name = (
        "velocity_gate_passed"
        if domain == "camp_flat_rolling"
        else "stage5_gate_passed"
    )
    gate = gates[gate_name]
    outcome = _mapping(
        backend.run_trigger_false_positive(str(domain), gate),
        name="trigger false-positive backend outcome",
    )
    if set(outcome) != _TRIGGER_OUTCOME_KEYS:
        raise ValueError("Trigger false-positive backend outcome schema drifted.")
    events = _exact_int(outcome.get("events"), name="events", minimum=1)
    expected_events = (
        gate.num_envs * gate.steps * len(gate.commands)
        if domain == "camp_flat_rolling"
        else gate.minimum_kick_events
    )
    if events != expected_events:
        raise ValueError(
            "Trigger false-positive event count is not the formal binding."
        )
    result = make_trigger_false_positive_check(
        domain=str(domain),
        events=events,
        stair_mode_false_positives=_exact_int(
            outcome.get("stair_mode_false_positives"),
            name="stair_mode_false_positives",
        ),
    )
    _json_safe(result)
    return result


def collect_trigger_false_positive_check(
    request: Mapping[str, object],
) -> Mapping[str, object]:
    """Run one provenance-bound, checkpoint-free pretraining trigger FP check.

    The request contains exactly domain and pretraining_request. The nested
    request binds current Git, the canonical contract, six artifacts, task,
    evaluation seed, and device; it deliberately cannot name a checkpoint.
    """

    candidate = _mapping(request, name="trigger false-positive request")
    if set(candidate) != {"domain", "pretraining_request"}:
        raise ValueError("Trigger false-positive request schema drifted.")
    normalized = validate_pretraining_trigger_request(
        _mapping(candidate.get("pretraining_request"), name="pretraining_request")
    )
    dependencies = _load_live_dependencies()
    backend = _PretrainingFpBackend(normalized, dependencies)
    return collect_trigger_false_positive_with_backend(candidate, backend)


@dataclass(frozen=True)
class _LiveDependencies:
    torch: Any
    tasks: Any
    task_module: Any
    contract_module: Any
    runner_module: Any
    env_module: Any
    rl_module: Any
    registry_module: Any
    manager_module: Any
    terrain_module: Any
    terrain_config_module: Any
    torch_utils_module: Any
    balance_task_module: Any
    fixed_command_module: Any
    fixed_yaw_module: Any
    hybrid_evaluator_module: Any
    hybrid_gate_module: Any
    c1_gate_module: Any
    stair_probe_module: Any


def _load_live_dependencies() -> _LiveDependencies:
    """Import every heavy dependency lazily and fail closed if one is absent."""

    modules = {
        "torch": "torch",
        "tasks": "hoppertrex_mjlab.tasks",
        "task_module": "hoppertrex_mjlab.tasks.hoppertrex_hybrid_task",
        "contract_module": "hoppertrex_mjlab.hybrid.stair_camp_contract",
        "runner_module": "hoppertrex_mjlab.hybrid.runner",
        "env_module": "mjlab.envs",
        "rl_module": "mjlab.rl",
        "registry_module": "mjlab.tasks.registry",
        "manager_module": "mjlab.managers",
        "terrain_module": "mjlab.terrains",
        "terrain_config_module": "mjlab.terrains.config",
        "torch_utils_module": "mjlab.utils.torch",
        "balance_task_module": "hoppertrex_mjlab.tasks.hoppertrex_balance_task",
        "fixed_command_module": (
            "hoppertrex_mjlab.scripts.rsl_rl.evaluate_fixed_command"
        ),
        "fixed_yaw_module": "hoppertrex_mjlab.scripts.rsl_rl.evaluate_fixed_yaw",
        "hybrid_evaluator_module": (
            "hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate"
        ),
        "hybrid_gate_module": "hoppertrex_mjlab.scripts.rsl_rl.hybrid_gate",
        "c1_gate_module": "hoppertrex_mjlab.scripts.evaluate_hybrid_c1_flat_gate",
        "stair_probe_module": "hoppertrex_mjlab.scripts.probe_hybrid_stair_height",
    }
    imported: dict[str, Any] = {}
    for name, module_name in modules.items():
        try:
            imported[name] = importlib.import_module(module_name)
        except Exception as exc:
            raise RuntimeError(
                f"Required StairCamp live dependency {module_name!r} is unavailable."
            ) from exc
    return _LiveDependencies(**imported)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _live_step_height_observation(
    env: Any,
    cell_values: Sequence[float],
) -> Any:
    """Return the real fixed-cell height selected by each terrain type."""

    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_types is None:
        raise RuntimeError("Live generated terrain has no terrain type indices.")
    terrain_types = terrain.terrain_types
    lookup = terrain_types.new_tensor(
        tuple(cell_values), dtype=env.scene["robot"].data.root_link_pos_w.dtype
    )
    if terrain_types.numel() and int(terrain_types.max().item()) >= len(cell_values):
        raise RuntimeError("Live terrain type index exceeds its cell-value table.")
    return lookup[terrain_types].unsqueeze(-1)


def _expected_leg_scales(scale: float) -> tuple[float, ...]:
    return (0.0, 0.0, scale, scale, scale, scale)


def apply_environment_ablation(
    action_cfg: Any, descriptor: Mapping[str, object]
) -> None:
    """Apply only registered environment-side ablation mutations."""

    name = descriptor.get("name")
    kind = descriptor.get("kind")
    if name == "baseline" and kind == "baseline":
        return
    if name == "leg-off" and kind == "leg_off":
        return
    if kind == "zero_shot_scale":
        scale = _finite(
            descriptor.get("deployment_leg_scale_rad"),
            name="ablation.deployment_leg_scale_rad",
        )
        if scale not in (0.035, 0.070, 0.100):
            raise ValueError("Zero-shot leg scale is not registered.")
        if tuple(action_cfg.action_scales) != _expected_leg_scales(0.070):
            raise ValueError("Zero-shot scale must start from the 0.070-rad camp.")
        action_cfg.action_scales = _expected_leg_scales(scale)
        return
    if name == "mode-always-on" and kind == "mode_always_on":
        if descriptor.get("force_stair_mode_from_reset") is not True:
            raise ValueError("Mode-always-on descriptor did not force stair mode.")
        action_cfg.stair_trigger_sensor_name = None
        action_cfg.stair_mode_forced = True
        validator = getattr(action_cfg, "__post_init__", None)
        if callable(validator):
            validator()
        return
    raise ValueError(f"Unsupported StairCamp ablation descriptor: {name!r}.")


def apply_policy_ablation(
    policy: Callable[[Any], Any], descriptor: Mapping[str, object]
) -> Callable[[Any], Any]:
    """Wrap the policy for leg-off; all other ablations are env-side."""

    name = descriptor.get("name")
    kind = descriptor.get("kind")
    if name != "leg-off" or kind != "leg_off":
        return policy
    raw_indices = _sequence(
        descriptor.get("zero_action_indices"), name="ablation.zero_action_indices"
    )
    indices = tuple(
        _exact_int(value, name=f"zero_action_indices[{index}]")
        for index, value in enumerate(raw_indices)
    )
    if indices != (2, 3, 4, 5):
        raise ValueError("Leg-off must zero exactly action indices 2..5.")

    def leg_off_policy(observations: Any) -> Any:
        actions = policy(observations).clone()
        if actions.shape[-1] != ACTION_WIDTH:
            raise RuntimeError("StairCamp policy did not emit six actions.")
        actions[..., 2:6] = 0.0
        return actions

    return leg_off_policy


def _observation_group(observations: Any, name: str) -> Any:
    try:
        return observations[name]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Live environment has no {name!r} observation group."
        ) from exc


def assert_policy_interface(
    observations: Any,
    *,
    action_width: int,
) -> None:
    """Fail before rollout unless the real environment is exactly 52/55/6."""

    actor = _observation_group(observations, "actor")
    critic = _observation_group(observations, "critic")
    if len(actor.shape) != 2 or int(actor.shape[-1]) != ACTOR_OBSERVATION_WIDTH:
        raise RuntimeError("Live StairCamp actor observation width is not 52.")
    if len(critic.shape) != 2 or int(critic.shape[-1]) != CRITIC_OBSERVATION_WIDTH:
        raise RuntimeError("Live StairCamp critic observation width is not 55.")
    if action_width != ACTION_WIDTH:
        raise RuntimeError("Live StairCamp action width is not six.")


_LIVE_EVIDENCE_ATTR = "_stair_camp_live_pre_reset_evidence"
_LIVE_EVIDENCE_TERM_NAME = "stair_camp_live_pre_reset_evidence"


def _validated_stage5_push_event(training_cfg: Any, task_module: Any) -> Any:
    """Validate and clone only the canonical Stage5 interval push event."""

    if getattr(training_cfg, "stair_camp_training_contract", None) is not True:
        raise RuntimeError("Canonical StairCamp training marker is not true.")
    events = getattr(training_cfg, "events", None)
    if not isinstance(events, Mapping) or "push_robot" not in events:
        raise RuntimeError("Canonical StairCamp training cfg has no push_robot event.")
    push = events["push_robot"]
    expected_fields = {
        "func",
        "params",
        "mode",
        "interval_range_s",
        "is_global_time",
        "min_step_count_between_reset",
    }
    if set(vars(push)) != expected_fields:
        raise RuntimeError("Canonical push_robot event schema drifted.")
    stage = task_module.HYBRID_STAGES[5]
    expected_velocity = {
        "x": (-float(stage.push_lin_vel_x), float(stage.push_lin_vel_x)),
        "pitch": (-float(stage.push_pitch_rate), float(stage.push_pitch_rate)),
    }
    params = getattr(push, "params", None)
    if not isinstance(params, Mapping) or set(params) != {
        "asset_cfg",
        "velocity_range",
    }:
        raise RuntimeError("Canonical push_robot params drifted.")
    expected_asset = task_module.SceneEntityCfg("robot")
    if params["asset_cfg"] != expected_asset:
        raise RuntimeError("Canonical push_robot asset binding drifted.")
    if params["velocity_range"] != expected_velocity:
        raise RuntimeError("Canonical push_robot velocity range drifted.")
    if (
        push.func is not task_module.envs_mdp.push_by_setting_velocity
        or push.mode != "interval"
        or tuple(push.interval_range_s or ()) != tuple(stage.push_interval_s)
        or push.is_global_time is not False
        or push.min_step_count_between_reset != 0
    ):
        raise RuntimeError("Canonical push_robot scheduling semantics drifted.")
    cloned = copy.deepcopy(push)
    if cloned != push or cloned is push:
        raise RuntimeError("Canonical push_robot event could not be cloned exactly.")
    return cloned


class _LivePreResetEvidenceMetric:
    """Capture safety/trigger evidence after reward and before auto-reset."""

    def __init__(self, cfg: Any, env: Any) -> None:
        params = getattr(cfg, "params", None)
        if not isinstance(params, Mapping):
            raise RuntimeError("Live evidence metric params are unavailable.")
        self._env = env
        self._contact_func = params.get("non_wheel_contact_func")
        self._sensor_name = params.get("sensor_name")
        if not callable(self._contact_func) or not isinstance(self._sensor_name, str):
            raise RuntimeError("Live evidence contact semantics are unavailable.")
        if hasattr(env, _LIVE_EVIDENCE_ATTR):
            raise RuntimeError("Live evidence metric was installed more than once.")
        torch = importlib.import_module("torch")
        self._torch = torch
        self.sequence = 0
        self.events = 0
        self.terminations = 0
        self.non_wheel_contacts = 0
        self.stair_mode_false_positives = 0
        self.previous_mode = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self.previous_contact = torch.zeros_like(self.previous_mode)
        self.last_mode = torch.zeros_like(self.previous_mode)
        self.last_mode_rising = torch.zeros_like(self.previous_mode)
        self.last_contact = torch.zeros_like(self.previous_mode)
        self.last_contact_rising = torch.zeros_like(self.previous_mode)
        self.last_terminated = torch.zeros_like(self.previous_mode)
        self.last_reset = torch.zeros_like(self.previous_mode)
        self.last_root_x = torch.zeros(
            env.num_envs, dtype=torch.float, device=env.device
        )
        setattr(env, _LIVE_EVIDENCE_ATTR, self)

    def _mode(self) -> Any:
        term = self._env.action_manager.get_term("hybrid_wheel_leg")
        mode = getattr(term, "stair_mode", None)
        if mode is None:
            raise RuntimeError("Camp action term exposes no stair_mode state.")
        return mode.bool()

    def reset(self, env_ids: Any = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        # Metrics reset runs after ActionManager.reset, so this captures the
        # true post-reset baseline (False normally, True for mode-always-on).
        self.previous_mode[ids] = self._mode()[ids]
        self.previous_contact[ids] = False
        self.last_mode[ids] = False
        self.last_mode_rising[ids] = False
        self.last_contact[ids] = False
        self.last_contact_rising[ids] = False
        self.last_terminated[ids] = False
        self.last_reset[ids] = False

    def __call__(
        self,
        env: Any,
        non_wheel_contact_func: Callable[..., Any],
        sensor_name: str,
    ) -> Any:
        if (
            env is not self._env
            or non_wheel_contact_func is not self._contact_func
            or sensor_name != self._sensor_name
        ):
            raise RuntimeError("Live evidence metric invocation drifted.")
        mode = self._mode().clone()
        direct_contact = self._contact_func(env, self._sensor_name).bool()
        try:
            termination_contact = env.termination_manager.get_term(
                "non_wheel_ground_contact"
            ).bool()
        except (AttributeError, KeyError, ValueError) as exc:
            raise RuntimeError(
                "Non-wheel contact termination semantics are unavailable."
            ) from exc
        contact = (direct_contact | termination_contact).clone()
        terminated = env.reset_terminated.bool().clone()
        reset = env.reset_buf.bool().clone()
        mode_rising = mode & ~self.previous_mode
        contact_rising = contact & ~self.previous_contact

        self.sequence += 1
        self.events += int(env.num_envs)
        self.terminations += int(terminated.sum().item())
        self.non_wheel_contacts += int(contact_rising.sum().item())
        self.stair_mode_false_positives += int(mode_rising.sum().item())
        self.last_mode.copy_(mode)
        self.last_mode_rising.copy_(mode_rising)
        self.last_contact.copy_(contact)
        self.last_contact_rising.copy_(contact_rising)
        self.last_terminated.copy_(terminated)
        self.last_reset.copy_(reset)
        self.last_root_x.copy_(env.scene["robot"].data.root_link_pos_w[:, 0])
        self.previous_mode.copy_(
            self._torch.where(reset, self._torch.zeros_like(mode), mode)
        )
        self.previous_contact.copy_(
            self._torch.where(reset, self._torch.zeros_like(contact), contact)
        )
        return mode.to(dtype=self._torch.float)

    def counts(self) -> _SafetyCounts:
        return _SafetyCounts(
            events=self.events,
            terminations=self.terminations,
            non_wheel_contacts=self.non_wheel_contacts,
            stair_mode_false_positives=self.stair_mode_false_positives,
        )


def _live_evidence(env: Any) -> _LivePreResetEvidenceMetric:
    evidence = getattr(env, _LIVE_EVIDENCE_ATTR, None)
    if not isinstance(evidence, _LivePreResetEvidenceMetric):
        raise RuntimeError("Evaluation env has no pre-reset live evidence hook.")
    return evidence


@dataclass
class _SafetyCounts:
    events: int = 0
    terminations: int = 0
    non_wheel_contacts: int = 0
    stair_mode_false_positives: int = 0

    def copy(self) -> _SafetyCounts:
        return _SafetyCounts(**asdict(self))

    def delta(self, earlier: _SafetyCounts) -> _SafetyCounts:
        return _SafetyCounts(
            events=self.events - earlier.events,
            terminations=self.terminations - earlier.terminations,
            non_wheel_contacts=self.non_wheel_contacts - earlier.non_wheel_contacts,
            stair_mode_false_positives=(
                self.stair_mode_false_positives - earlier.stair_mode_false_positives
            ),
        )

    def add(self, other: _SafetyCounts) -> None:
        self.events += other.events
        self.terminations += other.terminations
        self.non_wheel_contacts += other.non_wheel_contacts
        self.stair_mode_false_positives += other.stair_mode_false_positives


class _SafetyTrackingWrapper:
    """Vec-env proxy consuming post-reward/pre-reset evidence snapshots."""

    def __init__(self, wrapped: Any, dependencies: _LiveDependencies):
        self._wrapped = wrapped
        self._deps = dependencies
        self.counts = _SafetyCounts()
        evidence = _live_evidence(self.unwrapped)
        self._last_sequence = evidence.sequence
        self._last_totals = evidence.counts()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    @property
    def unwrapped(self) -> Any:
        return self._wrapped.unwrapped

    def reset(self) -> Any:
        result = self._wrapped.reset()
        evidence = _live_evidence(self.unwrapped)
        self._last_sequence = evidence.sequence
        self._last_totals = evidence.counts()
        return result

    def step(self, actions: Any) -> Any:
        evidence = _live_evidence(self.unwrapped)
        before_sequence = evidence.sequence
        before_totals = evidence.counts()
        result = self._wrapped.step(actions)
        evidence = _live_evidence(self.unwrapped)
        if evidence.sequence != before_sequence + 1:
            raise RuntimeError(
                "Pre-reset evidence hook did not run exactly once for env step."
            )
        delta = evidence.counts().delta(before_totals)
        if delta.events != int(self.unwrapped.num_envs):
            raise RuntimeError("Pre-reset evidence event accounting is incomplete.")
        self.counts.add(delta)
        self._last_sequence = evidence.sequence
        self._last_totals = evidence.counts()
        return result


@dataclass
class _LiveSession:
    wrapped: Any
    tracker: _SafetyTrackingWrapper
    env_cfg: Any


class _MjLabBackend:
    """Heavy implementation of the strict rollout backend protocol."""

    def __init__(
        self,
        config: Mapping[str, object],
        dependencies: _LiveDependencies,
    ) -> None:
        self.config = validate_adapter_config(config)
        self.deps = dependencies
        self.device = str(self.config["device"])
        self.evaluation_seed = int(self.config["evaluation_seed"])
        self.descriptor = _mapping(self.config["ablation"], name="ablation")
        self._policy_owner: Any | None = None
        self._base_policy: Callable[[Any], Any] | None = None
        self._policy: Callable[[Any], Any] | None = None
        self._session_records: list[dict[str, object]] = []

    def metadata(self) -> Mapping[str, object]:
        return {
            "schema_version": LIVE_ADAPTER_SCHEMA_VERSION,
            "adapter": "stair_camp_live_adapter",
            "task": self.config["task"],
            "actor_observation_width": ACTOR_OBSERVATION_WIDTH,
            "critic_observation_width": CRITIC_OBSERVATION_WIDTH,
            "action_width": ACTION_WIDTH,
            "stage5_actor_adapter_used": False,
            "ablation": self.descriptor["name"],
            "sessions": list(self._session_records),
        }

    def _registered_configs(self, *, play: bool) -> tuple[Any, Any]:
        registry = self.deps.registry_module
        task = str(self.config["task"])
        env_cfg = registry.load_env_cfg(task, play=play)
        # Provenance is always checked against the canonical training config.
        # Play configs intentionally use fewer envs and omit the metrics hook, so
        # hashing a play config would incorrectly reject every valid checkpoint.
        contract_cfg = env_cfg if not play else registry.load_env_cfg(task, play=False)
        agent_cfg = registry.load_rl_cfg(task)
        if play:
            if getattr(env_cfg, "stair_camp_training_contract", None) is not False:
                raise RuntimeError("Registered StairCamp play marker is not false.")
            if getattr(contract_cfg, "stair_camp_training_contract", None) is not True:
                raise RuntimeError("Canonical StairCamp training marker is not true.")
        elif getattr(env_cfg, "stair_camp_training_contract", None) is not True:
            raise RuntimeError("Registered StairCamp training marker is not true.")
        for candidate in (env_cfg, contract_cfg):
            if getattr(candidate, "stair_camp_task_id", None) != task:
                raise RuntimeError("Registry did not return the StairCamp environment.")
            actor_names = tuple(candidate.observations["actor"].terms)
            critic_names = tuple(candidate.observations["critic"].terms)
            expected_actor = tuple(
                self.deps.contract_module.STAIR_CAMP_EXPECTED_ACTOR_TERMS
            )
            expected_critic = expected_actor + tuple(
                self.deps.contract_module.STAIR_CAMP_EXPECTED_CRITIC_TAIL
            )
            if actor_names != expected_actor or critic_names != expected_critic:
                raise RuntimeError(
                    "Registered StairCamp observation term order drifted."
                )
        training = _mapping(
            _mapping(self.config["checkpoint"], name="checkpoint")["training"],
            name="checkpoint.training",
        )
        agent_cfg.seed = int(training["training_seed"])
        baseline_hash = self.deps.contract_module.stair_camp_contract_hash(
            contract_cfg, agent_cfg
        )
        if baseline_hash != training["contract_sha256"]:
            raise RuntimeError(
                "Live registered config does not match checkpoint contract."
            )
        artifacts = self.deps.contract_module.stair_camp_artifact_bindings(contract_cfg)
        evaluation_artifacts = self.deps.contract_module.stair_camp_artifact_bindings(
            env_cfg
        )
        if (
            artifacts != training["artifact_bindings"]
            or evaluation_artifacts != artifacts
        ):
            raise RuntimeError(
                "Live frozen artifacts do not match checkpoint bindings."
            )
        current_git = self.deps.runner_module.repository_git_sha()
        if current_git != training["git_sha"]:
            raise RuntimeError("Live Git SHA does not match checkpoint training SHA.")
        return env_cfg, agent_cfg

    def _verify_checkpoint_bytes(self) -> Path:
        checkpoint = _mapping(self.config["checkpoint"], name="checkpoint")
        path = Path(str(checkpoint["checkpoint_file"]))
        if not path.is_file():
            raise RuntimeError(f"StairCamp checkpoint does not exist: {path}.")
        if _sha256(path) != checkpoint["checkpoint_file_sha256"]:
            raise RuntimeError("Live checkpoint SHA256 does not match its envelope.")
        return path

    def _load_policy(self) -> Callable[[Any], Any]:
        if self._policy is not None:
            return self._policy
        torch = self.deps.torch
        configure = getattr(self.deps.torch_utils_module, "configure_torch_backends")
        configure()
        torch.manual_seed(self.evaluation_seed)
        env_cfg, agent_cfg = self._registered_configs(play=True)
        # Allocate one physical env, then temporarily restore the registered 256
        # config value while constructing the Hybrid runner. The runner binds the
        # canonical training contract, while RSL-RL uses the wrapper batch of one.
        env_cfg.scene.num_envs = 1
        if env_cfg.scene.terrain is not None:
            env_cfg.scene.terrain.num_envs = 1
        env = self.deps.env_module.ManagerBasedRlEnv(cfg=env_cfg, device=self.device)
        wrapped = self.deps.rl_module.RslRlVecEnvWrapper(
            env, clip_actions=agent_cfg.clip_actions
        )
        try:
            observations = wrapped.get_observations()
            assert_policy_interface(
                observations,
                action_width=int(wrapped.unwrapped.action_manager.total_action_dim),
            )
            runner_cls = self.deps.registry_module.load_runner_cls(
                str(self.config["task"])
            )
            if runner_cls is None:
                raise RuntimeError("StairCamp has no registered Hybrid runner class.")
            registered_num_envs = int(self.deps.contract_module.STAIR_CAMP_NUM_ENVS)
            env_cfg.scene.num_envs = registered_num_envs
            if env_cfg.scene.terrain is not None:
                env_cfg.scene.terrain.num_envs = registered_num_envs
            try:
                runner = runner_cls(wrapped, asdict(agent_cfg), device=self.device)
            finally:
                env_cfg.scene.num_envs = 1
                if env_cfg.scene.terrain is not None:
                    env_cfg.scene.terrain.num_envs = 1
            infos = runner.load(
                str(self._verify_checkpoint_bytes()),
                load_cfg={"actor": True},
                strict=True,
                map_location=self.device,
            )
            actual_training = _mapping(
                _mapping(infos, name="checkpoint infos").get("stair_camp_training"),
                name="checkpoint infos.stair_camp_training",
            )
            normalized_actual = _evaluator().validate_stair_camp_training_info(
                actual_training
            )
            expected_training = _mapping(
                _mapping(self.config["checkpoint"], name="checkpoint")["training"],
                name="checkpoint.training",
            )
            if normalized_actual != expected_training:
                raise RuntimeError(
                    "Loaded checkpoint provenance differs from its evaluator envelope."
                )
            base_policy = runner.get_inference_policy(device=self.device)
            actions = base_policy(observations)
            if len(actions.shape) != 2 or int(actions.shape[-1]) != ACTION_WIDTH:
                raise RuntimeError("Loaded StairCamp actor does not emit six actions.")
            self._policy_owner = runner
            self._base_policy = base_policy
            self._policy = apply_policy_ablation(base_policy, self.descriptor)
        finally:
            wrapped.close()
        return self._policy

    def _policy_for_rollout(self) -> Callable[[Any], Any]:
        """Return an already-bound policy or lazily load the checkpoint actor."""

        if self._policy is not None:
            return self._policy
        return self._load_policy()

    def _terrain_cfg(
        self,
        *,
        domain: str,
        cells: tuple[float, ...],
        num_envs: int,
    ) -> Any:
        task_module = self.deps.task_module
        terrain_config = self.deps.terrain_config_module
        if domain == "stairs":
            sub_terrains = {
                f"stair_{index:02d}": terrain_config.pyramid_stairs(
                    proportion=1.0,
                    step_height_range=(height, height),
                    step_width=task_module.STAIR_CAMP_STEP_WIDTH_M,
                    platform_width=task_module.STAIR_CAMP_PLATFORM_WIDTH_M,
                    border_width=task_module.STAIR_CAMP_TERRAIN_BORDER_WIDTH_M,
                )
                for index, height in enumerate(cells)
            }
        elif domain == "flat":
            if cells != (0.0,):
                raise RuntimeError("Flat adapter requires the single zero-height cell.")
            sub_terrains = {"flat_zero_height": terrain_config.flat(proportion=1.0)}
        elif domain == "slope":
            sub_terrains = {
                f"slope_{index:02d}": terrain_config.hf_pyramid_slope(
                    proportion=1.0,
                    slope_range=(math.tan(math.radians(degrees)),) * 2,
                    platform_width=task_module.STAIR_CAMP_PLATFORM_WIDTH_M,
                    border_width=task_module.STAIR_CAMP_TERRAIN_BORDER_WIDTH_M,
                )
                for index, degrees in enumerate(cells)
            }
        else:
            raise RuntimeError(f"Unknown live terrain domain: {domain!r}.")
        return self.deps.terrain_module.TerrainEntityCfg(
            terrain_type="generator",
            terrain_generator=self.deps.terrain_module.TerrainGeneratorCfg(
                seed=self.evaluation_seed,
                curriculum=True,
                size=task_module.STAIR_CAMP_TERRAIN_SIZE_M,
                num_rows=1,
                num_cols=len(cells),
                difficulty_range=(0.0, 0.0),
                sub_terrains=sub_terrains,
            ),
            max_init_terrain_level=0,
            num_envs=num_envs,
        )

    def _evaluation_env_cfg(
        self,
        *,
        domain: str,
        cells: tuple[float, ...],
        num_envs: int,
        pushes: bool,
    ) -> tuple[Any, Any]:
        if pushes and domain != "flat":
            raise RuntimeError("Pushes are restricted to the flat Stage5 gate suite.")
        # Every evaluator starts from the play surface. Training markers,
        # curriculum, metrics, corruption, and interval events are never inherited.
        env_cfg, agent_cfg = self._registered_configs(play=True)
        if getattr(env_cfg, "stair_camp_training_contract", None) is not False:
            raise RuntimeError("Evaluation env inherited a training-contract marker.")
        events = getattr(env_cfg, "events", None)
        if not isinstance(events, dict) or "push_robot" in events:
            raise RuntimeError("Registered StairCamp play event surface drifted.")
        if pushes:
            training_cfg = self.deps.registry_module.load_env_cfg(
                str(self.config["task"]), play=False
            )
            if getattr(training_cfg, "stair_camp_task_id", None) != self.config["task"]:
                raise RuntimeError("Canonical push source is not the StairCamp task.")
            events["push_robot"] = _validated_stage5_push_event(
                training_cfg, self.deps.task_module
            )
        if ("push_robot" in events) is not pushes:
            raise RuntimeError("Evaluation push event binding is inconsistent.")

        # Set the evaluator marker explicitly even though the registered play cfg
        # already carries False; downstream audits inspect this exact field.
        env_cfg.stair_camp_training_contract = False
        env_cfg.seed = self.evaluation_seed
        env_cfg.scene.num_envs = num_envs
        env_cfg.scene.terrain = self._terrain_cfg(
            domain=domain,
            cells=cells,
            num_envs=num_envs,
        )
        # Evaluation owns fixed-cell resets. The only metric is a post-reward,
        # pre-auto-reset evidence capture; no training curriculum/state survives.
        env_cfg.curriculum = {}
        env_cfg.metrics = {
            _LIVE_EVIDENCE_TERM_NAME: self.deps.manager_module.MetricsTermCfg(
                func=_LivePreResetEvidenceMetric,
                params={
                    "non_wheel_contact_func": (
                        self.deps.balance_task_module.non_wheel_ground_contact
                    ),
                    "sensor_name": (
                        self.deps.balance_task_module.NON_WHEEL_GROUND_SENSOR_NAME
                    ),
                },
                reduce="last",
            )
        }
        env_cfg.episode_length_s = EPISODE_LENGTH_S
        action_cfg = env_cfg.actions["hybrid_wheel_leg"]
        apply_environment_ablation(action_cfg, self.descriptor)
        critic_term = env_cfg.observations["critic"].terms["step_height"]
        critic_term.func = _live_step_height_observation
        critic_term.params = {
            "cell_values": (cells if domain == "stairs" else tuple(0.0 for _ in cells))
        }
        return env_cfg, agent_cfg

    @contextlib.contextmanager
    def _session(
        self,
        *,
        domain: str,
        cells: tuple[float, ...],
        num_envs: int,
        pushes: bool,
        purpose: str,
    ) -> Any:
        self._policy_for_rollout()
        env_cfg, agent_cfg = self._evaluation_env_cfg(
            domain=domain,
            cells=cells,
            num_envs=num_envs,
            pushes=pushes,
        )
        env = self.deps.env_module.ManagerBasedRlEnv(cfg=env_cfg, device=self.device)
        if getattr(env.cfg, "stair_camp_training_contract", None) is not False:
            raise RuntimeError("Constructed evaluation env carries a training marker.")
        wrapped = self.deps.rl_module.RslRlVecEnvWrapper(
            env, clip_actions=agent_cfg.clip_actions
        )
        _live_evidence(wrapped.unwrapped)
        tracker = _SafetyTrackingWrapper(wrapped, self.deps)
        try:
            observations = wrapped.get_observations()
            assert_policy_interface(
                observations,
                action_width=int(wrapped.unwrapped.action_manager.total_action_dim),
            )
            terrain = wrapped.unwrapped.scene.terrain
            if terrain is None or terrain.terrain_types is None:
                raise RuntimeError("Live adapter did not build generated terrain.")
            counts = self.deps.torch.bincount(
                terrain.terrain_types,
                minlength=len(cells),
            )
            if int(counts.sum().item()) != num_envs or bool((counts == 0).any()):
                raise RuntimeError(
                    "Generated terrain did not populate every fixed cell."
                )
            self._session_records.append(
                {
                    "purpose": purpose,
                    "domain": domain,
                    "terrain": (
                        "pyramid_stairs"
                        if domain == "stairs"
                        else "inclined_plane"
                        if domain == "slope"
                        else "flat_generated_zero_height"
                    ),
                    "num_envs": num_envs,
                    "cells": list(cells),
                    "pushes_enabled": pushes,
                    "stair_camp_training_contract": False,
                    "pre_reset_evidence_term": _LIVE_EVIDENCE_TERM_NAME,
                }
            )
            yield _LiveSession(wrapped=wrapped, tracker=tracker, env_cfg=env_cfg)
        finally:
            wrapped.close()

    def _policy_actions(self, wrapped: Any) -> Any:
        policy = self._policy_for_rollout()
        torch = self.deps.torch
        with torch.no_grad():
            observations = wrapped.get_observations()
            actions = policy(observations).detach()
        if int(actions.shape[-1]) != ACTION_WIDTH:
            raise RuntimeError("Live policy output width changed during rollout.")
        return actions

    def _force_commands(
        self,
        wrapped: Any,
        *,
        vx: float,
        yaw: float,
        posture: tuple[float, float],
    ) -> None:
        function = getattr(self.deps.fixed_command_module, "_force_command", None)
        force_posture = getattr(
            self.deps.fixed_command_module, "_force_static_posture", None
        )
        if not callable(function) or not callable(force_posture):
            raise RuntimeError("Registered fixed-command helpers are unavailable.")
        function(wrapped.unwrapped, vx, yaw)
        force_posture(wrapped.unwrapped, posture)

    @staticmethod
    def _posture_center(env_cfg: Any) -> tuple[float, float]:
        posture = env_cfg.commands["posture"]
        return (
            0.5 * sum(float(value) for value in posture.height_range),
            0.5 * sum(float(value) for value in posture.pitch_range),
        )

    def run_scan(self, request: ScanRequest) -> Sequence[Mapping[str, object]]:
        if request.domain != self.config["domain"]:
            raise RuntimeError("Backend scan domain does not match adapter config.")
        torch = self.deps.torch
        with self._session(
            domain=request.domain,
            cells=request.cells,
            num_envs=request.num_envs,
            pushes=False,
            purpose=f"{request.domain}_scan",
        ) as session:
            wrapped = session.wrapped
            env = wrapped.unwrapped
            terrain = env.scene.terrain
            assert terrain is not None and terrain.terrain_types is not None
            terrain_types = terrain.terrain_types.clone()
            counts = torch.bincount(terrain_types, minlength=len(request.cells))
            if not bool((counts == request.num_envs_per_cell).all()):
                raise RuntimeError(
                    "Scan terrain cells do not contain equal env counts."
                )
            slot_ids = torch.empty_like(terrain_types)
            for cell_index in range(len(request.cells)):
                ids = torch.nonzero(
                    terrain_types == cell_index, as_tuple=False
                ).squeeze(-1)
                slot_ids[ids] = torch.arange(len(ids), device=env.device)

            probe = self.deps.stair_probe_module
            card = probe.POSTURE_CARDS[0]
            if card.get("name") != "envelope_center":
                raise RuntimeError(
                    "Frozen stair-probe center reset card is unavailable."
                )
            posture = (float(card["height_m"]), float(card["pitch_rad"]))
            trials: list[dict[str, object]] = []
            for repeat in range(1, request.repeats + 1):
                reset_types, cross_x, _reset_metadata = probe._reset_to_approach(
                    env,
                    root_height=posture[0],
                    card_name="envelope_center",
                    repeat=repeat,
                )
                if not bool(torch.equal(reset_types, terrain_types)):
                    raise RuntimeError("Scan terrain assignment changed across resets.")
                robot = env.scene["robot"]
                start_x = robot.data.root_link_pos_w[:, 0].clone()
                outer_face = cross_x - float(probe.CROSS_DEPTH_M)
                target_x = (
                    cross_x
                    if request.domain == "stairs"
                    else start_x + request.travel_distance_m
                )
                success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                terminated_ever = torch.zeros_like(success)
                contact_ever = torch.zeros_like(success)
                triggered_ever = torch.zeros_like(success)
                pre_impact = torch.zeros_like(success)
                stable = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

                def step_once(vx: float, *, score: bool) -> None:
                    was_active = ~success & ~terminated_ever
                    self._force_commands(wrapped, vx=vx, yaw=0.0, posture=posture)
                    actions = self._policy_actions(wrapped)
                    evidence = _live_evidence(env)
                    before_sequence = evidence.sequence
                    wrapped.step(actions)
                    self._force_commands(wrapped, vx=vx, yaw=0.0, posture=posture)
                    evidence = _live_evidence(env)
                    if evidence.sequence != before_sequence + 1:
                        raise RuntimeError(
                            "Scan pre-reset evidence hook did not run exactly once."
                        )
                    # These tensors were cloned by the metric before MjLab's
                    # automatic reset can clear stair_mode or move the root.
                    terminated = evidence.last_terminated
                    contact = evidence.last_contact
                    mode = evidence.last_mode | evidence.last_mode_rising
                    root_x = evidence.last_root_x
                    terminated_ever.logical_or_(was_active & terminated)
                    contact_ever.logical_or_(was_active & contact)
                    triggered_ever.logical_or_(was_active & mode)
                    if request.domain == "stairs":
                        pre_impact.logical_or_(
                            was_active & mode & (root_x < outer_face)
                        )
                    else:
                        pre_impact.logical_or_(was_active & mode)
                    if not score:
                        return
                    active = was_active & ~terminated & ~contact
                    stable.copy_(
                        torch.where(
                            active & (root_x >= target_x),
                            stable + 1,
                            torch.zeros_like(stable),
                        )
                    )
                    success.logical_or_(active & (stable >= request.stable_steps))

                for _ in range(request.settle_steps):
                    step_once(0.0, score=False)
                for _ in range(request.drive_steps):
                    step_once(
                        float(self.deps.stair_probe_module.COMMAND_VX_MPS), score=True
                    )

                for env_index in range(env.num_envs):
                    cell_index = int(terrain_types[env_index].item())
                    false_positive = (
                        bool(pre_impact[env_index].item())
                        if request.domain == "stairs"
                        else bool(triggered_ever[env_index].item())
                    )
                    trials.append(
                        {
                            "cell": request.cells[cell_index],
                            "repeat": repeat,
                            "env_id": int(slot_ids[env_index].item()),
                            "success": bool(success[env_index].item()),
                            "terminated": bool(terminated_ever[env_index].item()),
                            "non_wheel_contact": bool(contact_ever[env_index].item()),
                            "stair_mode_false_positive": false_positive,
                            "triggered": bool(triggered_ever[env_index].item()),
                            "pre_impact_triggered": bool(pre_impact[env_index].item()),
                        }
                    )
        return trials

    @staticmethod
    def _gate_outcome(
        request: GateRequest,
        *,
        upstream: bool,
        safety: _SafetyCounts,
        kick_events: int | None = None,
    ) -> dict[str, object]:
        actual_kicks = (
            request.minimum_kick_events if kick_events is None else kick_events
        )
        return {
            "name": request.name,
            "num_envs": request.num_envs,
            "steps": request.steps,
            "scenario_count": request.scenario_count,
            "kick_events": actual_kicks,
            "upstream_gate_passed": bool(upstream),
            "terminations": safety.terminations,
            "non_wheel_contacts": safety.non_wheel_contacts,
            "stair_mode_false_positives": safety.stair_mode_false_positives,
        }

    def _run_c1_cell(
        self,
        session: _LiveSession,
        *,
        height: float,
        pitch: float,
        vx: float,
        settle_steps: int,
        measure_steps: int,
    ) -> dict[str, float]:
        torch = self.deps.torch
        wrapped = session.tracker
        wrapped.reset()
        robot = wrapped.unwrapped.scene["robot"]
        action_term = wrapped.unwrapped.action_manager.get_term("hybrid_wheel_leg")
        heights: list[Any] = []
        pitches: list[Any] = []
        pitch_rates: list[Any] = []
        velocities: list[Any] = []
        wheel_rates: list[Any] = []
        previous_targets: Any | None = None
        before = wrapped.counts.copy()
        for step in range(settle_steps + measure_steps):
            self._force_commands(wrapped, vx=vx, yaw=0.0, posture=(height, pitch))
            wrapped.step(self._policy_actions(wrapped))
            self._force_commands(wrapped, vx=vx, yaw=0.0, posture=(height, pitch))
            wheel_targets = action_term.wheel_targets.detach().clone()
            if step >= settle_steps:
                data = robot.data
                gravity = data.projected_gravity_b
                actual_pitch = torch.atan2(
                    gravity[:, 0], torch.clamp(-gravity[:, 2], min=1.0e-6)
                )
                heights.append(data.root_link_pos_w[:, 2].detach().cpu())
                pitches.append(actual_pitch.detach().cpu())
                pitch_rates.append(data.root_link_ang_vel_b[:, 1].abs().detach().cpu())
                velocities.append(data.root_link_lin_vel_b[:, 0].detach().cpu())
                if previous_targets is not None:
                    wheel_rates.append(
                        torch.sum(
                            torch.square(wheel_targets - previous_targets), dim=1
                        ).cpu()
                    )
            previous_targets = wheel_targets
        if not heights or not wheel_rates:
            raise RuntimeError("C1 gate collected no measurable samples.")
        height_error = torch.stack(heights) - height
        pitch_error = torch.stack(pitches) - pitch
        pitch_rate = torch.stack(pitch_rates)
        velocity = torch.stack(velocities)
        wheel_rate = torch.stack(wheel_rates)
        safety = wrapped.counts.delta(before)
        samples = max((settle_steps + measure_steps) * wrapped.num_envs, 1)
        return {
            "target_height": float(height),
            "target_pitch": float(pitch),
            "vx_command": float(vx),
            "height_rmse": float(
                torch.sqrt(torch.mean(torch.square(height_error))).item()
            ),
            "pitch_rmse": float(
                torch.sqrt(torch.mean(torch.square(pitch_error))).item()
            ),
            "pitch_error_abs_p95": float(
                torch.quantile(pitch_error.abs(), 0.95).item()
            ),
            "pitch_rate_abs_p99": float(torch.quantile(pitch_rate, 0.99).item()),
            "mean_actual_lin_x": float(velocity.mean().item()),
            "velocity_error_abs": float(abs(velocity.mean().item() - vx)),
            "wheel_target_rate_rms": float(torch.sqrt(torch.mean(wheel_rate)).item()),
            "non_wheel_contact_rate": safety.non_wheel_contacts / samples,
            "terminated_events": float(safety.terminations),
            "safety_window_steps": float(settle_steps + measure_steps),
        }

    def _run_flat_c1_gate(self, request: GateRequest) -> dict[str, object]:
        if request.source_suite != "c1_affine_full_15_cell_safety":
            raise RuntimeError("Flat gate source-suite binding drifted.")
        module = self.deps.c1_gate_module
        cells = list(module.evaluation_cells(module.FORMAL_VX_CHECK))
        if request.profile == "smoke":
            center_h = module.REGISTERED_HEIGHT_NODES[1]
            center_p = module.REGISTERED_PITCH_NODES[1]
            cells = [(center_h, center_p, 0.0)]
        if len(cells) != request.scenario_count:
            raise RuntimeError("C1 gate scenario count does not match its binding.")
        if request.settle_steps + request.measure_steps != request.steps:
            raise RuntimeError("C1 gate settle/measure cadence does not match steps.")
        with self._session(
            domain="flat",
            cells=(0.0,),
            num_envs=request.num_envs,
            pushes=False,
            purpose="flat_gate_c1_15_cell",
        ) as session:
            before = session.tracker.counts.copy()
            metrics = [
                self._run_c1_cell(
                    session,
                    height=float(height),
                    pitch=float(pitch),
                    vx=float(vx),
                    settle_steps=request.settle_steps,
                    measure_steps=request.measure_steps,
                )
                for height, pitch, vx in cells
            ]
            verdict = module.aggregate_candidate(metrics, dict(module.REGISTERED_CAPS))
            safety = session.tracker.counts.delta(before)
        return self._gate_outcome(
            request,
            upstream=verdict.get("flat_gate_passed") is True,
            safety=safety,
        )

    @staticmethod
    def _rollout_args(request: GateRequest, device: str) -> SimpleNamespace:
        if request.profile == "formal":
            warmup = FORMAL_WARMUP_STEPS
            window = FORMAL_WINDOW_STEPS
        elif request.profile == "smoke":
            warmup = max(0, min(1, request.steps - 2))
            window = max(2, request.steps - warmup)
        else:
            raise RuntimeError("Gate profile is neither formal nor smoke.")
        if warmup >= request.steps:
            raise RuntimeError("Gate warmup leaves no measurement samples.")
        return SimpleNamespace(
            task=None,
            seed=1,
            num_envs=request.num_envs,
            steps=request.steps,
            warmup_steps=warmup,
            window_steps=window,
            progress_interval=0,
            episode_length_s=EPISODE_LENGTH_S,
            device=device,
        )

    def _fixed_command_args(
        self, request: GateRequest, base: SimpleNamespace
    ) -> SimpleNamespace:
        helper = getattr(self.deps.hybrid_evaluator_module, "_fixed_command_args", None)
        if not callable(helper):
            raise RuntimeError("Hybrid fixed-command argument helper is unavailable.")
        return helper(base, str(self.config["task"]))

    def _run_linear_gate(self, request: GateRequest) -> dict[str, object]:
        expected_sources = {
            "standing_gate_passed": "hybrid_linear_standing",
            "velocity_gate_passed": "hybrid_linear_velocity",
        }
        if request.source_suite != expected_sources[request.name]:
            raise RuntimeError(f"{request.name} source-suite binding drifted.")
        commands = request.commands
        if request.profile == "smoke" and request.scenario_count == 1:
            # The pure evaluator keeps the formal command list in its smoke
            # binding but explicitly reduces scenario_count to one. Execute the
            # first registered command rather than inventing a new smoke cell.
            commands = commands[:1]
        if len(commands) != request.scenario_count:
            raise RuntimeError(f"{request.name} command count drifted.")
        base = self._rollout_args(request, self.device)
        with self._session(
            domain="flat",
            cells=(0.0,),
            num_envs=request.num_envs,
            pushes=False,
            purpose=request.source_suite,
        ) as session:
            before = session.tracker.counts.copy()
            posture = self._posture_center(session.env_cfg)
            scenarios: list[dict[str, object]] = []
            for vx, yaw in commands:
                if abs(yaw) > 1.0e-12:
                    raise RuntimeError(
                        "Standing/velocity gate unexpectedly commands yaw."
                    )
                row = self.deps.fixed_command_module._run_fixed_command(
                    wrapped=session.tracker,
                    policy=self._load_policy(),
                    args=self._fixed_command_args(request, base),
                    lin_x_cmd=vx,
                    posture_target=posture,
                )
                scenarios.append(
                    {
                        "name": f"linear_vx_{vx:+.3f}",
                        "kind": "linear",
                        "lin_x": vx,
                        "metrics": row,
                    }
                )
            checks = [
                check
                for scenario in scenarios
                for check in self.deps.hybrid_gate_module.linear_scenario_checks(
                    scenario
                )
            ]
            if not checks:
                raise RuntimeError("Linear gate produced no registered checks.")
            safety = session.tracker.counts.delta(before)
        return self._gate_outcome(
            request,
            upstream=all(bool(check.passed) for check in checks),
            safety=safety,
        )

    def _zero_policy(self, observations: Any) -> Any:
        actor = _observation_group(observations, "actor")
        return self.deps.torch.zeros(
            (actor.shape[0], ACTION_WIDTH), dtype=actor.dtype, device=actor.device
        )

    def _fixed_yaw_args(
        self,
        request: GateRequest,
        base: SimpleNamespace,
        lin_x: float,
    ) -> SimpleNamespace:
        helper = getattr(self.deps.hybrid_evaluator_module, "_fixed_yaw_args", None)
        if not callable(helper):
            raise RuntimeError("Hybrid fixed-yaw argument helper is unavailable.")
        return helper(base, str(self.config["task"]), lin_x)

    def _clean_robust_scenarios(
        self,
        request: GateRequest,
        base: SimpleNamespace,
    ) -> tuple[list[dict[str, object]], _SafetyCounts]:
        evaluator = self.deps.hybrid_evaluator_module
        with self._session(
            domain="flat",
            cells=(0.0,),
            num_envs=request.num_envs,
            pushes=False,
            purpose="stage5_robust_clean_channels",
        ) as session:
            before = session.tracker.counts.copy()
            wrapped = session.tracker
            policy = self._load_policy()
            posture_center = evaluator._posture_targets(wrapped)[0]
            scenarios: list[dict[str, object]] = []
            command_args = self._fixed_command_args(request, base)
            for vx in (-0.07, 0.0, 0.07):
                row = self.deps.fixed_command_module._run_fixed_command(
                    wrapped=wrapped,
                    policy=policy,
                    args=command_args,
                    lin_x_cmd=vx,
                    posture_target=posture_center,
                )
                row["duration_s"] = request.steps / CONTROL_FREQUENCY_HZ
                scenarios.append(
                    {
                        "name": f"linear_vx_{vx:+.3f}",
                        "kind": "linear",
                        "lin_x": vx,
                        "metrics": row,
                    }
                )
            yaw_rows: list[dict[str, float | str]] = []
            combo_rows: list[dict[str, float | str]] = []
            for lin_x in (0.0, -0.07, 0.07):
                yaw_args = self._fixed_yaw_args(request, base, lin_x)
                for yaw in (-0.10, 0.10):
                    row = self.deps.fixed_yaw_module._run_fixed_yaw(
                        wrapped=wrapped,
                        policy=policy,
                        args=yaw_args,
                        yaw_cmd=yaw,
                        posture_target=posture_center,
                    )
                    (yaw_rows if lin_x == 0.0 else combo_rows).append(row)
            scenarios.extend(evaluator._fixed_rows_to_scenarios("yaw", yaw_rows))
            scenarios.extend(evaluator._fixed_rows_to_scenarios("combo", combo_rows))
            for target_height, target_pitch in evaluator._posture_targets(wrapped):
                metrics = evaluator._run_posture_scenario(
                    wrapped=wrapped,
                    policy=policy,
                    args=base,
                    target_height=target_height,
                    target_pitch=target_pitch,
                )
                scenarios.append(
                    {
                        "name": f"posture_h_{target_height:+.4f}_p_{target_pitch:+.4f}",
                        "kind": "posture",
                        "target_height": target_height,
                        "target_pitch": target_pitch,
                        "metrics": metrics,
                    }
                )
            safety = session.tracker.counts.delta(before)
        return scenarios, safety

    def _push_robust_scenarios(
        self,
        request: GateRequest,
        base: SimpleNamespace,
        *,
        policy: Callable[[Any], Any],
        purpose: str,
    ) -> tuple[list[dict[str, object]], dict[str, float], _SafetyCounts]:
        evaluator = self.deps.hybrid_evaluator_module
        with self._session(
            domain="flat",
            cells=(0.0,),
            num_envs=request.num_envs,
            pushes=True,
            purpose=purpose,
        ) as session:
            before = session.tracker.counts.copy()
            random_metrics = evaluator._run_integrated_rollout(
                wrapped=session.tracker,
                policy=policy,
                args=base,
                force_commands=False,
            )
            fixed_metrics = evaluator._run_integrated_rollout(
                wrapped=session.tracker,
                policy=policy,
                args=base,
                force_commands=True,
            )
            safety = session.tracker.counts.delta(before)
        scenarios = [
            {
                "name": "random_integrated",
                "kind": "random",
                "metrics": random_metrics,
            },
            {
                "name": "robust_pushes",
                "kind": "robust",
                "metrics": fixed_metrics,
            },
        ]
        return scenarios, fixed_metrics, safety

    def _recovery_metrics(
        self,
        request: GateRequest,
        base: SimpleNamespace,
        *,
        policy: Callable[[Any], Any],
        purpose: str,
    ) -> tuple[dict[str, float], _SafetyCounts]:
        evaluator = self.deps.hybrid_evaluator_module
        with self._session(
            domain="flat",
            cells=(0.0,),
            num_envs=request.num_envs,
            pushes=False,
            purpose=purpose,
        ) as session:
            before = session.tracker.counts.copy()
            center = evaluator._posture_targets(session.tracker)[0]
            metrics = evaluator._run_recovery_scenario(
                wrapped=session.tracker,
                policy=policy,
                args=base,
                target_height=center[0],
                target_pitch=center[1],
                kick_scale=float(request.kick_scale),
            )
            safety = session.tracker.counts.delta(before)
        return metrics, safety

    def _formal_stage5_gate(self, request: GateRequest) -> dict[str, object]:
        if request.source_suite != "hybrid_robust_stage5_8x":
            raise RuntimeError("Stage5 gate source-suite binding drifted.")
        if request.kick_scale != 8.0 or request.minimum_kick_events != 128:
            raise RuntimeError("Formal Stage5 8x/128-event binding drifted.")
        base = self._rollout_args(request, self.device)
        scenarios, clean_safety = self._clean_robust_scenarios(request, base)
        push_scenarios, _fixed, push_safety = self._push_robust_scenarios(
            request,
            base,
            policy=self._load_policy(),
            purpose="stage5_robust_pushes",
        )
        scenarios.extend(push_scenarios)
        _baseline_scenarios, baseline_fixed, baseline_push_safety = (
            self._push_robust_scenarios(
                request,
                base,
                policy=self._zero_policy,
                purpose="stage5_classical_push_reference",
            )
        )
        candidate, candidate_safety = self._recovery_metrics(
            request,
            base,
            policy=self._load_policy(),
            purpose="stage5_candidate_8x_recovery",
        )
        baseline, baseline_safety = self._recovery_metrics(
            request,
            base,
            policy=self._zero_policy,
            purpose="stage5_classical_8x_recovery",
        )
        merged: dict[str, float] = {}
        for key, value in candidate.items():
            merged[f"candidate_{key}"] = float(value)
        for key, value in baseline.items():
            merged[f"baseline_{key}"] = float(value)
        scenarios.append(
            {
                "name": "stage5_recovery_center_8x",
                "kind": "recovery",
                "kick_scale": request.kick_scale,
                "metrics": merged,
            }
        )
        checks = self.deps.hybrid_gate_module.evaluate_capability_suite(
            "robust",
            scenarios,
            profile="formal",
            stage4_reference={"tracking_error": baseline_fixed["tracking_error"]},
        )
        if not checks:
            raise RuntimeError("Formal Stage5 robust suite produced no checks.")
        kick_events = int(round(float(candidate["kick_event_count"])))
        safety = _SafetyCounts()
        for item in (
            clean_safety,
            push_safety,
            baseline_push_safety,
            candidate_safety,
            baseline_safety,
        ):
            safety.add(item)
        return self._gate_outcome(
            request,
            upstream=all(bool(check.passed) for check in checks),
            safety=safety,
            kick_events=kick_events,
        )

    def _smoke_kick_rollout(
        self,
        request: GateRequest,
        *,
        purpose: str,
        pushes: bool,
    ) -> tuple[_SafetyCounts, int]:
        if request.minimum_kick_events != request.num_envs:
            raise RuntimeError("Smoke kick binding must contain one kick per env.")
        torch = self.deps.torch
        with self._session(
            domain="flat",
            cells=(0.0,),
            num_envs=request.num_envs,
            pushes=pushes,
            purpose=purpose,
        ) as session:
            wrapped = session.tracker
            wrapped.reset()
            before = wrapped.counts.copy()
            posture = self._posture_center(session.env_cfg)
            env_ids = torch.arange(request.num_envs, device=self.device)
            self.deps.hybrid_evaluator_module._apply_scaled_stage1_kick(
                wrapped,
                env_ids,
                kick_index=0,
                scale=float(request.kick_scale),
            )
            for _ in range(request.steps):
                self._force_commands(wrapped, vx=0.0, yaw=0.0, posture=posture)
                wrapped.step(self._policy_actions(wrapped))
            safety = wrapped.counts.delta(before)
        return safety, request.num_envs

    def _run_stage5_gate(self, request: GateRequest) -> dict[str, object]:
        if request.profile == "formal":
            return self._formal_stage5_gate(request)
        safety, events = self._smoke_kick_rollout(
            request, purpose="stage5_smoke_kick", pushes=True
        )
        return self._gate_outcome(
            request,
            upstream=safety.terminations == 0 and safety.non_wheel_contacts == 0,
            safety=safety,
            kick_events=events,
        )

    def run_gate(self, request: GateRequest) -> Mapping[str, object]:
        if self.config["domain"] != "flat":
            raise RuntimeError("Gate backend requires a flat adapter config.")
        if request.terrain != "flat":
            raise RuntimeError("Gate terrain binding is not flat.")
        if request.name == "flat_gate_passed":
            return self._run_flat_c1_gate(request)
        if request.name in ("standing_gate_passed", "velocity_gate_passed"):
            return self._run_linear_gate(request)
        if request.name == "stage5_gate_passed":
            return self._run_stage5_gate(request)
        raise RuntimeError(f"Unknown flat gate request: {request.name!r}.")

    def _run_flat_rolling_fp(self, request: GateRequest) -> dict[str, object]:
        if request.name != "velocity_gate_passed" or not request.commands:
            raise RuntimeError("Flat rolling FP check requires velocity gate commands.")
        with self._session(
            domain="flat",
            cells=(0.0,),
            num_envs=request.num_envs,
            pushes=False,
            purpose="pretraining_camp_flat_rolling_fp",
        ) as session:
            wrapped = session.tracker
            posture = self._posture_center(session.env_cfg)
            before = wrapped.counts.copy()
            for vx, yaw in request.commands:
                wrapped.reset()
                for _ in range(request.steps):
                    self._force_commands(wrapped, vx=vx, yaw=yaw, posture=posture)
                    wrapped.step(self._policy_actions(wrapped))
            safety = wrapped.counts.delta(before)
        expected_events = request.num_envs * request.steps * len(request.commands)
        if safety.events != expected_events:
            raise RuntimeError("Flat rolling FP event accounting is incomplete.")
        return {
            "events": expected_events,
            "stair_mode_false_positives": safety.stair_mode_false_positives,
        }

    def _run_stage5_kick_fp(self, request: GateRequest) -> dict[str, object]:
        if request.name != "stage5_gate_passed" or request.kick_scale != 8.0:
            raise RuntimeError("Stage5 kick FP check requires the registered 8x gate.")
        if request.profile == "smoke":
            safety, events = self._smoke_kick_rollout(
                request, purpose="pretraining_stage5_kick_fp_smoke", pushes=False
            )
        else:
            base = self._rollout_args(request, self.device)
            metrics, safety = self._recovery_metrics(
                request,
                base,
                policy=self._policy_for_rollout(),
                purpose="pretraining_stage5_kick_fp",
            )
            events = int(round(float(metrics["kick_event_count"])))
        if events != request.minimum_kick_events:
            raise RuntimeError("Stage5 kick FP event count does not match its binding.")
        return {
            "events": events,
            "stair_mode_false_positives": safety.stair_mode_false_positives,
        }

    def run_trigger_false_positive(
        self, domain: str, request: GateRequest
    ) -> Mapping[str, object]:
        if self.config["domain"] != "flat":
            raise RuntimeError("Trigger FP backend requires a flat adapter config.")
        if domain == "camp_flat_rolling":
            return self._run_flat_rolling_fp(request)
        if domain == "stage5_kick":
            return self._run_stage5_kick_fp(request)
        raise RuntimeError(f"Unknown trigger FP domain: {domain!r}.")


class _PretrainingFpBackend(_MjLabBackend):
    """Checkpoint-free backend bound to canonical pretraining provenance."""

    def __init__(
        self,
        request: Mapping[str, object],
        dependencies: _LiveDependencies,
    ) -> None:
        self.pretraining_request = validate_pretraining_trigger_request(request)
        self.config: dict[str, object] = {
            "task": self.pretraining_request["task"],
            "domain": "flat",
        }
        self.deps = dependencies
        self.device = str(self.pretraining_request["device"])
        self.evaluation_seed = int(self.pretraining_request["evaluation_seed"])
        self.descriptor: Mapping[str, object] = {
            "name": "baseline",
            "kind": "baseline",
        }
        self._policy_owner: Any | None = None
        self._base_policy: Callable[[Any], Any] | None = self._zero_policy
        self._policy: Callable[[Any], Any] | None = self._zero_policy
        self._session_records: list[dict[str, object]] = []
        # Validate the canonical registry surface before allocating any env.
        self._registered_configs(play=False)

    def _registered_configs(self, *, play: bool) -> tuple[Any, Any]:
        registry = self.deps.registry_module
        task = str(self.pretraining_request["task"])
        env_cfg = registry.load_env_cfg(task, play=play)
        training_cfg = env_cfg if not play else registry.load_env_cfg(task, play=False)
        agent_cfg = registry.load_rl_cfg(task)

        if getattr(training_cfg, "stair_camp_training_contract", None) is not True:
            raise RuntimeError("Canonical StairCamp training marker is not true.")
        if (
            getattr(training_cfg, "stair_camp_zero_initialize_actor_output", None)
            is not True
        ):
            raise RuntimeError("Canonical StairCamp zero-mean marker is not true.")
        if play and getattr(env_cfg, "stair_camp_training_contract", None) is not False:
            raise RuntimeError("Registered StairCamp play marker is not false.")
        for candidate in (training_cfg, env_cfg):
            if getattr(candidate, "stair_camp_task_id", None) != task:
                raise RuntimeError("Registry did not return the StairCamp environment.")
            actor_names = tuple(candidate.observations["actor"].terms)
            critic_names = tuple(candidate.observations["critic"].terms)
            expected_actor = tuple(
                self.deps.contract_module.STAIR_CAMP_EXPECTED_ACTOR_TERMS
            )
            expected_critic = expected_actor + tuple(
                self.deps.contract_module.STAIR_CAMP_EXPECTED_CRITIC_TAIL
            )
            if actor_names != expected_actor or critic_names != expected_critic:
                raise RuntimeError("Registered StairCamp observation terms drifted.")

        requested_contract = str(self.pretraining_request["contract_sha256"])
        module_contract = getattr(
            self.deps.contract_module,
            "STAIR_CAMP_CANONICAL_CONTRACT_SHA256",
            None,
        )
        if module_contract != requested_contract:
            raise RuntimeError("Runtime canonical StairCamp contract constant drifted.")
        actual_contract = self.deps.contract_module.stair_camp_contract_hash(
            training_cfg, agent_cfg
        )
        if actual_contract != requested_contract:
            raise RuntimeError("Canonical registry StairCamp contract drifted.")

        requested_artifacts = dict(
            _mapping(
                self.pretraining_request["artifact_bindings"],
                name="artifact_bindings",
            )
        )
        training_artifacts = self.deps.contract_module.stair_camp_artifact_bindings(
            training_cfg
        )
        evaluation_artifacts = self.deps.contract_module.stair_camp_artifact_bindings(
            env_cfg
        )
        if (
            set(training_artifacts) != set(PRETRAINING_ARTIFACT_BINDING_NAMES)
            or training_artifacts != requested_artifacts
            or evaluation_artifacts != requested_artifacts
        ):
            raise RuntimeError("Canonical registry StairCamp artifacts drifted.")

        current_git = self.deps.runner_module.repository_git_sha()
        if current_git != self.pretraining_request["git_sha"]:
            raise RuntimeError("Current Git SHA differs from pretraining request.")
        return env_cfg, agent_cfg

    def _load_policy(self) -> Callable[[Any], Any]:
        raise RuntimeError("Pretraining trigger checks forbid checkpoint loading.")

    def _policy_for_rollout(self) -> Callable[[Any], Any]:
        return self._zero_policy


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant {value!r} is forbidden.")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} is forbidden.")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(encoding="utf-8-sig"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    return _mapping(payload, name=str(path))


def _deterministic_json(payload: Mapping[str, object]) -> str:
    _json_safe(payload)
    return (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


def _write_output(payload: Mapping[str, object], output: Path | None) -> None:
    encoded = _deterministic_json(payload)
    if output is None:
        sys.stdout.write(encoded)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite live-adapter output: {output}.")
    temporary = output.with_name(f".{output.name}.incomplete.{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and, unlike Path.replace(), fails if
        # a concurrent publisher won the destination-name race.
        os.link(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser(
        "collect", help="Run the raw live adapter collection."
    )
    collect_parser.add_argument("--config", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path)
    trigger_parser = subparsers.add_parser(
        "trigger-fp", help="Run one pretraining trigger false-positive check."
    )
    trigger_parser.add_argument(
        "--domain", choices=TRIGGER_FALSE_POSITIVE_DOMAINS, required=True
    )
    trigger_parser.add_argument("--request", type=Path, required=True)
    trigger_parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "collect":
        result = collect(_read_json_mapping(args.config))
    elif args.command == "trigger-fp":
        result = collect_trigger_false_positive_check(
            {
                "domain": args.domain,
                "pretraining_request": _read_json_mapping(args.request),
            }
        )
    else:  # pragma: no cover - argparse makes this unreachable.
        raise AssertionError(f"Unhandled live-adapter command: {args.command}.")
    _write_output(_mapping(result, name="live adapter result"), args.output)
    return 0


__all__ = [
    "ACTION_WIDTH",
    "ACTOR_OBSERVATION_WIDTH",
    "CRITIC_OBSERVATION_WIDTH",
    "GateRequest",
    "LIVE_ADAPTER_SCHEMA_VERSION",
    "PRETRAINING_ARTIFACT_BINDING_NAMES",
    "PRETRAINING_TRIGGER_REQUEST_KIND",
    "RolloutBackend",
    "ScanRequest",
    "TRIGGER_FALSE_POSITIVE_DOMAINS",
    "TRIGGER_FALSE_POSITIVE_KIND",
    "TerrainPlan",
    "aggregate_scan_trials",
    "apply_environment_ablation",
    "apply_policy_ablation",
    "assert_policy_interface",
    "collect",
    "collect_trigger_false_positive_check",
    "collect_trigger_false_positive_with_backend",
    "collect_with_backend",
    "main",
    "make_terrain_plan",
    "make_trigger_false_positive_check",
    "parse_args",
    "validate_adapter_config",
    "validate_pretraining_trigger_request",
]


if __name__ == "__main__":
    raise SystemExit(main())
