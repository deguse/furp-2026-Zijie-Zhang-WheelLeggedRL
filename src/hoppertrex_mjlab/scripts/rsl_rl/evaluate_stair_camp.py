#!/usr/bin/env python3
# ruff: noqa: TRY004
"""Pure StairCamp evaluation contracts plus a lazy live-adapter sidecar.

This module intentionally does not import MjLab or RSL-RL. Formal simulation
is supplied by a separately-owned ``module:callable`` adapter which receives a
JSON-safe request and returns aggregate observations. The pure layer pins and
validates the registered protocol, checkpoint provenance, ablations, result
shape, and K=3 selection before any output can be used as evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

SRC_PATH = Path(__file__).resolve().parents[3]
if str(SRC_PATH) not in sys.path:
  sys.path.insert(0, str(SRC_PATH))

from hoppertrex_mjlab.hybrid.config import (  # noqa: E402
  STAIR_CAMP_ACTION_MASK,
  STAIR_CAMP_STAGE,
  STAIR_CAMP_TASK_ID,
)

# These scalar/string values mirror public frozen contracts. Importing their
# owning modules would transitively import Torch; keeping the sidecar importable
# in a stdlib-only CPU process is part of this module's integration contract.
STAIR_CAMP_CONTRACT_SCHEMA_VERSION = 1
STAIR_CAMP_CANONICAL_CONTRACT_SHA256 = (
  "ad4007ec9334b27b64eae7bcff96aae3e16b7be298e0531228669f37b46c888f"
)
STAIR_CAMP_TRAINING_INFO_KEY = "stair_camp_training"
STAIR_CAMP_ARTIFACT_BINDING_NAMES = (
  "controller_gain_hash",
  "calibration_hash",
  "yaw_calibration_hash",
  "posture_map_hash",
  "posture_artifact_hash",
  "station_calibration_hash",
)
MIN_STAGE5_KICK_EVENTS = 128
STAIR_CAMP_ACTOR_WIDTH = 52
STAIR_CAMP_CRITIC_WIDTH = 55
STAIR_CAMP_ACTION_WIDTH = 6

EVALUATOR_SCHEMA_VERSION = 1
CHECKPOINT_ENVELOPE_KIND = "stair_camp_checkpoint"
EVALUATION_ENVELOPE_KIND = "stair_camp_evaluation"
K3_SCREEN_KIND = "stair_camp_k3_screen"
K3_SELECTION_KIND = "stair_camp_k3_selection"
REGISTERED_TRAINING_SEEDS = (1, 2, 3)
REGISTERED_EVALUATION_SEED = 1
REGISTERED_BUDGETS = (1000, 3000)
CHECKPOINT_SAVE_INTERVAL = 100

STAIR_HEIGHTS_M = (0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15)
# The classical arm comes from the frozen C0 stair probe, whose height sweep is
# 0.00-0.10 m in 0.01 m steps, so 0.15 m is the one registered height it cannot
# supply. That cell is verdict-irrelevant: the classical contiguous passing
# prefix terminates at 0.01 m (measured 0/48 at every tier from 0.01 m up), so
# `classical_height_m` is 0.00 m with or without it. Requiring it would force
# either an authored number or a re-sweep of a frozen script.
CLASSICAL_HEIGHTS_M = (0.01, 0.02, 0.03, 0.05, 0.07, 0.10)
SLOPE_DEGREES = (5.0, 10.0, 15.0)
ZERO_SHOT_LEG_SCALES_RAD = (0.035, 0.070, 0.100)
OFFICIAL_ENVS_PER_CELL = 16
OFFICIAL_REPEATS = 3
OFFICIAL_EVENTS_PER_CELL = OFFICIAL_ENVS_PER_CELL * OFFICIAL_REPEATS
OFFICIAL_SETTLE_STEPS = 100
OFFICIAL_DRIVE_STEPS = 500
OFFICIAL_STABLE_STEPS = 25
SUCCESS_RATE_LIMIT = 0.90
OFFICIAL_MIN_SUCCESSES = math.ceil(SUCCESS_RATE_LIMIT * OFFICIAL_EVENTS_PER_CELL)
APPROACH_DISTANCE_M = 0.25
CROSS_DEPTH_M = 0.15
SUCCESS_TRAVEL_DISTANCE_M = APPROACH_DISTANCE_M + CROSS_DEPTH_M

# Frozen C1 affine full-gate protocol. Importing its executable module would
# eagerly import MjLab, so the source values are mirrored at the pure/live
# boundary while public Hybrid constants above are imported directly.
FLAT_GATE_SCENARIO_COUNT = 15
FLAT_GATE_NUM_ENVS = 16
FLAT_GATE_SETTLE_STEPS = 100
FLAT_GATE_MEASURE_STEPS = 200
FORMAL_GATE_STEPS = 3000
STAGE5_KICK_SCALE = 8.0

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, kw_only=True)
class DomainProtocol:
  """One immutable terrain protocol exposed to a live adapter."""

  domain: str
  profile: str
  terrain: str
  cell_key: str | None
  cells: tuple[float, ...]
  num_envs_per_cell: int | None
  repeats: int | None
  settle_steps: int | None
  drive_steps: int | None
  stable_steps: int | None
  success_rate_limit: float | None
  travel_distance_m: float | None
  evidence_eligible: bool

  @property
  def events_per_cell(self) -> int | None:
    if self.num_envs_per_cell is None or self.repeats is None:
      return None
    return self.num_envs_per_cell * self.repeats

  def to_dict(self) -> dict[str, object]:
    payload = asdict(self)
    payload["cells"] = list(self.cells)
    payload["events_per_cell"] = self.events_per_cell
    return payload


STAIRS_PROTOCOL = DomainProtocol(
  domain="stairs",
  profile="formal",
  terrain="pyramid_stairs",
  cell_key="height_m",
  cells=STAIR_HEIGHTS_M,
  num_envs_per_cell=OFFICIAL_ENVS_PER_CELL,
  repeats=OFFICIAL_REPEATS,
  settle_steps=OFFICIAL_SETTLE_STEPS,
  drive_steps=OFFICIAL_DRIVE_STEPS,
  stable_steps=OFFICIAL_STABLE_STEPS,
  success_rate_limit=SUCCESS_RATE_LIMIT,
  travel_distance_m=SUCCESS_TRAVEL_DISTANCE_M,
  evidence_eligible=True,
)
FLAT_PROTOCOL = DomainProtocol(
  domain="flat",
  profile="formal",
  terrain="flat",
  cell_key=None,
  cells=(),
  num_envs_per_cell=None,
  repeats=None,
  settle_steps=None,
  drive_steps=None,
  stable_steps=None,
  success_rate_limit=None,
  travel_distance_m=None,
  evidence_eligible=True,
)
SLOPE_PROTOCOL = DomainProtocol(
  domain="slope",
  profile="formal",
  terrain="inclined_plane",
  cell_key="slope_deg",
  cells=SLOPE_DEGREES,
  num_envs_per_cell=OFFICIAL_ENVS_PER_CELL,
  repeats=OFFICIAL_REPEATS,
  settle_steps=OFFICIAL_SETTLE_STEPS,
  drive_steps=OFFICIAL_DRIVE_STEPS,
  stable_steps=OFFICIAL_STABLE_STEPS,
  success_rate_limit=None,
  travel_distance_m=SUCCESS_TRAVEL_DISTANCE_M,
  evidence_eligible=True,
)
K3_SCREEN_PROTOCOL = DomainProtocol(
  domain="stairs",
  profile="screen",
  terrain="pyramid_stairs",
  cell_key="height_m",
  cells=(0.01,),
  num_envs_per_cell=OFFICIAL_ENVS_PER_CELL,
  repeats=1,
  settle_steps=OFFICIAL_SETTLE_STEPS,
  drive_steps=OFFICIAL_DRIVE_STEPS,
  stable_steps=OFFICIAL_STABLE_STEPS,
  success_rate_limit=SUCCESS_RATE_LIMIT,
  travel_distance_m=SUCCESS_TRAVEL_DISTANCE_M,
  evidence_eligible=False,
)
_FORMAL_PROTOCOLS = {
  "stairs": STAIRS_PROTOCOL,
  "flat": FLAT_PROTOCOL,
  "slope": SLOPE_PROTOCOL,
}


@dataclass(frozen=True, kw_only=True)
class GateBinding:
  """Exact meaning of one frozen promotion-function gate boolean."""

  name: str
  source_suite: str
  terrain: str
  profile: str
  num_envs: int
  steps: int
  scenario_count: int
  commands: tuple[tuple[float, float], ...]
  settle_steps: int = 0
  measure_steps: int = 0
  kick_scale: float | None = None
  minimum_kick_events: int = 0
  evidence_eligible: bool = True

  def to_dict(self) -> dict[str, object]:
    payload = asdict(self)
    payload["commands"] = [list(command) for command in self.commands]
    return payload


GATE_BINDINGS = {
  "flat_gate_passed": GateBinding(
    name="flat_gate_passed",
    source_suite="c1_affine_full_15_cell_safety",
    terrain="flat",
    profile="formal",
    num_envs=FLAT_GATE_NUM_ENVS,
    steps=FLAT_GATE_SETTLE_STEPS + FLAT_GATE_MEASURE_STEPS,
    scenario_count=FLAT_GATE_SCENARIO_COUNT,
    commands=(),
    settle_steps=FLAT_GATE_SETTLE_STEPS,
    measure_steps=FLAT_GATE_MEASURE_STEPS,
  ),
  "standing_gate_passed": GateBinding(
    name="standing_gate_passed",
    source_suite="hybrid_linear_standing",
    terrain="flat",
    profile="formal",
    num_envs=16,
    steps=FORMAL_GATE_STEPS,
    scenario_count=1,
    commands=((0.0, 0.0),),
  ),
  "velocity_gate_passed": GateBinding(
    name="velocity_gate_passed",
    source_suite="hybrid_linear_velocity",
    terrain="flat",
    profile="formal",
    num_envs=16,
    steps=FORMAL_GATE_STEPS,
    scenario_count=2,
    commands=((-0.07, 0.0), (0.07, 0.0)),
  ),
  "stage5_gate_passed": GateBinding(
    name="stage5_gate_passed",
    source_suite="hybrid_robust_stage5_8x",
    terrain="flat",
    profile="formal",
    num_envs=32,
    steps=FORMAL_GATE_STEPS,
    scenario_count=1,
    commands=((0.0, 0.0),),
    kick_scale=STAGE5_KICK_SCALE,
    minimum_kick_events=MIN_STAGE5_KICK_EVENTS,
  ),
}
GATE_NAMES = tuple(GATE_BINDINGS)


@dataclass(frozen=True, kw_only=True)
class AblationDescriptor:
  """Evaluation-only manipulation; only ``baseline`` is promotable."""

  name: str
  kind: str
  interpretation: str
  zero_action_indices: tuple[int, ...] = ()
  deployment_leg_scale_rad: float | None = None
  force_stair_mode_from_reset: bool = False
  promotion_evidence_eligible: bool = False
  coupled_factors: tuple[str, ...] = ()

  def to_dict(self) -> dict[str, object]:
    payload = asdict(self)
    payload["zero_action_indices"] = list(self.zero_action_indices)
    payload["coupled_factors"] = list(self.coupled_factors)
    return payload


BASELINE_ABLATION = AblationDescriptor(
  name="baseline",
  kind="baseline",
  promotion_evidence_eligible=True,
  interpretation="Unmodified registered StairCamp policy evaluation.",
)
LEG_OFF_ABLATION = AblationDescriptor(
  name="leg-off",
  kind="leg_off",
  zero_action_indices=(2, 3, 4, 5),
  interpretation=(
    "Zero the four learned leg-residual heads before the action term; the "
    "designed stair-mode effects remain active."
  ),
)
ZERO_SHOT_SCALE_ABLATIONS = tuple(
  AblationDescriptor(
    name=f"zero-shot-scale-{scale:.3f}",
    kind="zero_shot_scale",
    deployment_leg_scale_rad=scale,
    interpretation=(
      "Evaluation-only leg-authority sensitivity; never promotion evidence."
    ),
  )
  for scale in ZERO_SHOT_LEG_SCALES_RAD
)
MODE_ALWAYS_ON_ABLATION = AblationDescriptor(
  name="mode-always-on",
  kind="mode_always_on",
  force_stair_mode_from_reset=True,
  interpretation=(
    "Three-factor composite mode-always-on cost; never attribute it to "
    "trigger timing alone."
  ),
  coupled_factors=(
    "trigger_timing_removed",
    "classical_leg_reference_frozen_from_reset",
    "stair_progress_reward_ungated_from_reset",
  ),
)
ABLATION_DESCRIPTORS = {
  descriptor.name: descriptor
  for descriptor in (
    BASELINE_ABLATION,
    LEG_OFF_ABLATION,
    *ZERO_SHOT_SCALE_ABLATIONS,
    MODE_ALWAYS_ON_ABLATION,
  )
}


ADJUDICATION_ABLATION_NAMES = (
  LEG_OFF_ABLATION.name,
  *(descriptor.name for descriptor in ZERO_SHOT_SCALE_ABLATIONS),
  MODE_ALWAYS_ON_ABLATION.name,
)


@dataclass(frozen=True, kw_only=True)
class CheckpointExpectation:
  """Optional exact bindings supplied by the evaluation launcher."""

  git_sha: str | None = None
  contract_sha256: str | None = None
  artifact_bindings: Mapping[str, str] | None = None
  training_seed: int | None = None
  completed_updates: int | None = None


class LiveAdapter(Protocol):
  """Small integration seam implemented outside this pure sidecar."""

  def collect(self, config: Mapping[str, object]) -> Mapping[str, object]: ...


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
  if not isinstance(value, Mapping):
    raise ValueError(f"{name} must be a JSON object.")
  return value


def _require_sequence(value: object, *, name: str) -> Sequence[object]:
  if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
    raise ValueError(f"{name} must be a JSON array.")
  return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
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


def _sha256(value: object, *, name: str) -> str:
  if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
    raise ValueError(f"{name} must be a 64-character SHA256 hex digest.")
  return value.lower()


def _git_sha(value: object, *, name: str) -> str:
  if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
    raise ValueError(f"{name} must be a full 40-character git SHA.")
  return value.lower()


def _boolean(value: object, *, name: str) -> bool:
  if not isinstance(value, bool):
    raise ValueError(f"{name} must be boolean.")
  return value


def _validate_json_value(value: object, *, path: str = "payload") -> None:
  if value is None or isinstance(value, (str, bool, int)):
    return
  if isinstance(value, float):
    if not math.isfinite(value):
      raise ValueError(f"{path} contains a non-finite float.")
    return
  if isinstance(value, Mapping):
    for key, item in value.items():
      if not isinstance(key, str):
        raise ValueError(f"{path} contains a non-string object key.")
      _validate_json_value(item, path=f"{path}.{key}")
    return
  if isinstance(value, (list, tuple)):
    for index, item in enumerate(value):
      _validate_json_value(item, path=f"{path}[{index}]")
    return
  raise ValueError(f"{path} contains a non-JSON value: {type(value).__name__}.")


def deterministic_json(payload: Mapping[str, object]) -> str:
  """Return strict, stable JSON with exactly one trailing newline."""

  _validate_json_value(payload)
  return json.dumps(
    dict(payload),
    indent=2,
    sort_keys=True,
    ensure_ascii=True,
    allow_nan=False,
  ) + "\n"


def _canonical_sha256(payload: Mapping[str, object]) -> str:
  _validate_json_value(payload)
  encoded = json.dumps(
    dict(payload),
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def protocol_for(domain: str, profile: str = "formal") -> DomainProtocol:
  """Return a formal protocol or a one-cell non-evidential CPU smoke copy."""

  if domain not in _FORMAL_PROTOCOLS:
    raise ValueError(f"Unknown StairCamp evaluation domain: {domain!r}.")
  if profile == "formal":
    return _FORMAL_PROTOCOLS[domain]
  if profile != "smoke":
    raise ValueError("Evaluation profile must be 'formal' or 'smoke'.")
  formal = _FORMAL_PROTOCOLS[domain]
  if domain == "flat":
    return replace(formal, profile="smoke", evidence_eligible=False)
  return replace(
    formal,
    profile="smoke",
    cells=(formal.cells[0],),
    num_envs_per_cell=1,
    repeats=1,
    settle_steps=2,
    drive_steps=5,
    stable_steps=2,
    evidence_eligible=False,
  )


def gate_bindings_for_profile(profile: str = "formal") -> dict[str, GateBinding]:
  if profile == "formal":
    return dict(GATE_BINDINGS)
  if profile != "smoke":
    raise ValueError("Gate profile must be 'formal' or 'smoke'.")
  smoke: dict[str, GateBinding] = {}
  for name, binding in GATE_BINDINGS.items():
    smoke[name] = replace(
      binding,
      profile="smoke",
      num_envs=1,
      steps=7 if name == "flat_gate_passed" else 5,
      scenario_count=1,
      settle_steps=2 if name == "flat_gate_passed" else 0,
      measure_steps=5 if name == "flat_gate_passed" else 0,
      minimum_kick_events=1 if name == "stage5_gate_passed" else 0,
      evidence_eligible=False,
    )
  return smoke


def resolve_ablation(name: str) -> AblationDescriptor:
  try:
    return ABLATION_DESCRIPTORS[name]
  except KeyError as exc:
    raise ValueError(f"Unknown StairCamp ablation: {name!r}.") from exc


def _normalize_artifacts(value: object) -> dict[str, str]:
  artifacts = _require_mapping(value, name="training artifacts")
  if set(artifacts) != set(STAIR_CAMP_ARTIFACT_BINDING_NAMES):
    raise ValueError(
      "StairCamp checkpoint must bind exactly the six frozen artifact fields."
    )
  return {
    key: _sha256(artifacts[key], name=f"artifact {key}")
    for key in STAIR_CAMP_ARTIFACT_BINDING_NAMES
  }


def validate_stair_camp_training_info(
  training: Mapping[str, object],
  *,
  expectation: CheckpointExpectation | None = None,
) -> dict[str, object]:
  """Validate and normalize ``infos['stair_camp_training']`` provenance."""

  expected_fields = {
    "schema_version",
    "task",
    "training_seed",
    "git_sha",
    "contract_sha256",
    "artifact_bindings",
    "action_scales",
    "zero_initialized_deterministic_mean",
    "init_std",
    "completed_updates",
  }
  actual_fields = set(training)
  if actual_fields not in (expected_fields, expected_fields | {"action_mask"}):
    raise ValueError("StairCamp checkpoint training provenance schema drifted.")
  if "action_mask" in training and tuple(
    _require_sequence(training["action_mask"], name="training action_mask")
  ) != STAIR_CAMP_ACTION_MASK:
    raise ValueError("Checkpoint normalized action mask drifted.")
  if training.get("schema_version") != STAIR_CAMP_CONTRACT_SCHEMA_VERSION:
    raise ValueError("StairCamp checkpoint training schema version does not match.")
  if training.get("task") != STAIR_CAMP_TASK_ID:
    raise ValueError("Checkpoint was not trained as the registered StairCamp task.")
  training_seed = _integer(
    training.get("training_seed"), name="training_seed", minimum=1
  )
  if training_seed not in REGISTERED_TRAINING_SEEDS:
    raise ValueError("StairCamp training seed must be one of {1, 2, 3}.")
  git_sha = _git_sha(training.get("git_sha"), name="training git_sha")
  contract_sha256 = _sha256(
    training.get("contract_sha256"), name="training contract_sha256"
  )
  if contract_sha256 != STAIR_CAMP_CANONICAL_CONTRACT_SHA256:
    raise ValueError("Checkpoint contract hash is not the preregistered StairCamp hash.")
  artifacts = _normalize_artifacts(training.get("artifact_bindings"))
  raw_scales = _require_sequence(
    training.get("action_scales"), name="training action_scales"
  )
  action_scales = tuple(
    _finite(value, name=f"action_scales[{index}]")
    for index, value in enumerate(raw_scales)
  )
  expected_scales = tuple(float(value) for value in STAIR_CAMP_STAGE.action_scales)
  if len(action_scales) != 6 or any(
    abs(actual - expected) > 1.0e-12
    for actual, expected in zip(action_scales, expected_scales, strict=True)
  ):
    raise ValueError("Checkpoint action scales do not match StairCamp 0.070 rad.")
  if training.get("zero_initialized_deterministic_mean") is not True:
    raise ValueError(
      "Checkpoint must attest deterministic actor-output zero initialization."
    )
  init_std = _finite(training.get("init_std"), name="training init_std")
  if abs(init_std - 0.6) > 1.0e-12:
    raise ValueError("Checkpoint must attest the frozen exploration init_std=0.6.")
  completed_updates = _integer(
    training.get("completed_updates"), name="completed_updates", minimum=1
  )

  normalized: dict[str, object] = {
    "schema_version": STAIR_CAMP_CONTRACT_SCHEMA_VERSION,
    "task": STAIR_CAMP_TASK_ID,
    "training_seed": training_seed,
    "git_sha": git_sha,
    "contract_sha256": contract_sha256,
    "artifact_bindings": artifacts,
    "action_scales": list(action_scales),
    "action_mask": list(STAIR_CAMP_ACTION_MASK),
    "zero_initialized_deterministic_mean": True,
    "init_std": init_std,
    "completed_updates": completed_updates,
  }
  if expectation is not None:
    if expectation.git_sha is not None and git_sha != _git_sha(
      expectation.git_sha, name="expected git_sha"
    ):
      raise ValueError("Checkpoint and evaluation git SHAs do not match.")
    if expectation.contract_sha256 is not None and contract_sha256 != _sha256(
      expectation.contract_sha256, name="expected contract_sha256"
    ):
      raise ValueError("Checkpoint contract hash does not match evaluation config.")
    if expectation.artifact_bindings is not None:
      expected_artifacts = _normalize_artifacts(expectation.artifact_bindings)
      if artifacts != expected_artifacts:
        raise ValueError("Checkpoint artifact bindings do not match evaluation.")
    if (
      expectation.training_seed is not None
      and training_seed != expectation.training_seed
    ):
      raise ValueError("Checkpoint training seed does not match evaluation.")
    if (
      expectation.completed_updates is not None
      and completed_updates != expectation.completed_updates
    ):
      raise ValueError("Checkpoint completed-update count does not match.")
  return normalized


def validate_stair_camp_checkpoint_envelope(
  envelope: Mapping[str, object],
  *,
  expectation: CheckpointExpectation | None = None,
  verify_file: bool = False,
) -> dict[str, object]:
  """Validate a tensor-free sidecar envelope for one camp checkpoint."""

  if envelope.get("schema_version") != EVALUATOR_SCHEMA_VERSION:
    raise ValueError("StairCamp checkpoint-envelope schema version does not match.")
  if envelope.get("kind") != CHECKPOINT_ENVELOPE_KIND:
    raise ValueError("Input is not a StairCamp checkpoint envelope.")
  checkpoint_file = envelope.get("checkpoint_file")
  if not isinstance(checkpoint_file, str) or not checkpoint_file.strip():
    raise ValueError("checkpoint_file must be a non-empty path string.")
  checkpoint_sha = _sha256(
    envelope.get("checkpoint_file_sha256"), name="checkpoint_file_sha256"
  )
  training = validate_stair_camp_training_info(
    _require_mapping(envelope.get("training"), name="checkpoint training"),
    expectation=expectation,
  )
  iteration_value: int | None = None
  if "checkpoint_iteration" in envelope:
    iteration_value = _integer(
      envelope.get("checkpoint_iteration"),
      name="checkpoint_iteration",
      minimum=0,
    )
    if iteration_value + 1 != int(training["completed_updates"]):
      raise ValueError(
        "Checkpoint zero-based iteration does not match completed updates."
      )
  if verify_file:
    path = Path(checkpoint_file)
    if not path.is_file():
      raise ValueError(f"Checkpoint file does not exist: {path}.")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != checkpoint_sha:
      raise ValueError("Checkpoint file SHA256 does not match its envelope.")
  normalized_envelope: dict[str, object] = {
    "schema_version": EVALUATOR_SCHEMA_VERSION,
    "kind": CHECKPOINT_ENVELOPE_KIND,
    "checkpoint_file": checkpoint_file,
    "checkpoint_file_sha256": checkpoint_sha,
    "checkpoint_file_verified": bool(verify_file),
    "training": training,
  }
  if iteration_value is not None:
    normalized_envelope["checkpoint_iteration"] = iteration_value
  return normalized_envelope


def checkpoint_envelope_from_loaded_checkpoint(
  checkpoint_file: str | Path,
  checkpoint: Mapping[str, object],
  *,
  expectation: CheckpointExpectation | None = None,
) -> dict[str, object]:
  """Create a pure envelope after a caller has loaded a runner checkpoint.

  Loading remains the caller's responsibility, so this helper does not assume
  a PyTorch/RSL-RL API. It only consumes the registered ``infos`` mapping.
  """

  path = Path(checkpoint_file)
  if not path.is_file():
    raise ValueError(f"Checkpoint file does not exist: {path}.")
  infos = _require_mapping(checkpoint.get("infos"), name="checkpoint infos")
  training = _require_mapping(
    infos.get(STAIR_CAMP_TRAINING_INFO_KEY),
    name=f"checkpoint infos.{STAIR_CAMP_TRAINING_INFO_KEY}",
  )
  envelope: dict[str, object] = {
    "schema_version": EVALUATOR_SCHEMA_VERSION,
    "kind": CHECKPOINT_ENVELOPE_KIND,
    "checkpoint_file": str(path.resolve()),
    "checkpoint_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "training": dict(training),
  }
  if "iter" in checkpoint:
    envelope["checkpoint_iteration"] = checkpoint["iter"]
  return validate_stair_camp_checkpoint_envelope(
    envelope,
    expectation=expectation,
    verify_file=True,
  )


def checkpoint_envelope_from_file(
  checkpoint_file: str | Path,
  *,
  expectation: CheckpointExpectation | None = None,
) -> dict[str, object]:
  """Lazily load one Torch checkpoint and emit its verified pure envelope."""

  import torch

  path = Path(checkpoint_file)
  checkpoint = torch.load(path, map_location="cpu", weights_only=False)
  if not isinstance(checkpoint, Mapping):
    raise ValueError("StairCamp checkpoint file must contain a mapping.")
  return checkpoint_envelope_from_loaded_checkpoint(
    path,
    checkpoint,
    expectation=expectation,
  )


def make_adapter_config(
  *,
  domain: str,
  checkpoint_envelope: Mapping[str, object],
  profile: str = "formal",
  ablation: str = "baseline",
  device: str = "cpu",
  expectation: CheckpointExpectation | None = None,
  verify_checkpoint_file: bool = False,
) -> dict[str, object]:
  """Build the complete JSON-safe request passed to a live adapter."""

  if not isinstance(device, str) or not device.strip():
    raise ValueError("Adapter device must be a non-empty string.")
  protocol = protocol_for(domain, profile)
  descriptor = resolve_ablation(ablation)
  if domain != "stairs" and descriptor is not BASELINE_ABLATION:
    raise ValueError("StairCamp ablations are defined only for the stairs scan.")
  checkpoint = validate_stair_camp_checkpoint_envelope(
    checkpoint_envelope,
    expectation=expectation,
    verify_file=verify_checkpoint_file,
  )
  file_verified = bool(checkpoint["checkpoint_file_verified"])
  evidence_eligible = bool(protocol.evidence_eligible and file_verified)
  payload: dict[str, object] = {
    "schema_version": EVALUATOR_SCHEMA_VERSION,
    "kind": "stair_camp_adapter_config",
    "task": STAIR_CAMP_TASK_ID,
    "domain": domain,
    "profile": profile,
    "evaluation_seed": REGISTERED_EVALUATION_SEED,
    "device": device,
    "evidence_eligible": evidence_eligible,
    "promotion_evidence_eligible": bool(
      evidence_eligible
      and domain == "stairs"
      and descriptor.promotion_evidence_eligible
    ),
    "checkpoint": checkpoint,
    "policy_interface": {
      "task": STAIR_CAMP_TASK_ID,
      "actor_observation_width": STAIR_CAMP_ACTOR_WIDTH,
      "critic_observation_width": STAIR_CAMP_CRITIC_WIDTH,
      "action_width": STAIR_CAMP_ACTION_WIDTH,
      "stage5_actor_adapter_forbidden": True,
    },
    "protocol": protocol.to_dict(),
    "gate_bindings": (
      {
        name: binding.to_dict()
        for name, binding in gate_bindings_for_profile(profile).items()
      }
      if domain == "flat"
      else {}
    ),
    "ablation": descriptor.to_dict(),
  }
  payload["config_sha256"] = _canonical_sha256(payload)
  return payload


def _validate_adapter_config(config: Mapping[str, object]) -> None:
  supplied_hash = _sha256(config.get("config_sha256"), name="config_sha256")
  unhashed = dict(config)
  unhashed.pop("config_sha256", None)
  if _canonical_sha256(unhashed) != supplied_hash:
    raise ValueError("Adapter config digest does not match its contents.")
  if config.get("schema_version") != EVALUATOR_SCHEMA_VERSION:
    raise ValueError("Adapter config schema version does not match.")
  if config.get("kind") != "stair_camp_adapter_config":
    raise ValueError("Input is not a StairCamp adapter config.")
  if config.get("task") != STAIR_CAMP_TASK_ID:
    raise ValueError("Adapter config task does not match StairCamp.")
  if config.get("evaluation_seed") != REGISTERED_EVALUATION_SEED:
    raise ValueError("Adapter config evaluation seed must be 1.")
  device = config.get("device")
  if not isinstance(device, str) or not device.strip():
    raise ValueError("Adapter config device must be a non-empty string.")
  domain = config.get("domain")
  profile = config.get("profile")
  if not isinstance(domain, str) or not isinstance(profile, str):
    raise ValueError("Adapter config domain/profile are missing.")
  protocol = protocol_for(domain, profile)
  if config.get("protocol") != protocol.to_dict():
    raise ValueError("Adapter config protocol drifted from the registration.")
  checkpoint = _require_mapping(config.get("checkpoint"), name="checkpoint")
  file_verified = _boolean(
    checkpoint.get("checkpoint_file_verified"),
    name="checkpoint.checkpoint_file_verified",
  )
  normalized_checkpoint = validate_stair_camp_checkpoint_envelope(
    checkpoint, verify_file=file_verified
  )
  if checkpoint != normalized_checkpoint:
    raise ValueError("Adapter checkpoint envelope is not canonical.")
  evidence_eligible = bool(protocol.evidence_eligible and file_verified)
  if config.get("evidence_eligible") is not evidence_eligible:
    raise ValueError("Adapter evidence eligibility is inconsistent.")
  expected_interface = {
    "task": STAIR_CAMP_TASK_ID,
    "actor_observation_width": STAIR_CAMP_ACTOR_WIDTH,
    "critic_observation_width": STAIR_CAMP_CRITIC_WIDTH,
    "action_width": STAIR_CAMP_ACTION_WIDTH,
    "stage5_actor_adapter_forbidden": True,
  }
  if config.get("policy_interface") != expected_interface:
    raise ValueError("Adapter config policy interface is not the 52-D StairCamp actor.")
  ablation = _require_mapping(config.get("ablation"), name="ablation")
  descriptor = resolve_ablation(str(ablation.get("name")))
  if config.get("ablation") != descriptor.to_dict():
    raise ValueError("Adapter config ablation descriptor drifted.")
  if domain != "stairs" and descriptor is not BASELINE_ABLATION:
    raise ValueError("StairCamp ablations are defined only for the stairs scan.")
  expected_promotion_evidence = bool(
    evidence_eligible
    and domain == "stairs"
    and descriptor.promotion_evidence_eligible
  )
  if config.get("promotion_evidence_eligible") is not expected_promotion_evidence:
    raise ValueError("Adapter promotion-evidence eligibility is inconsistent.")
  expected_gates = (
    {
      name: binding.to_dict()
      for name, binding in gate_bindings_for_profile(profile).items()
    }
    if domain == "flat"
    else {}
  )
  if config.get("gate_bindings") != expected_gates:
    raise ValueError("Adapter config gate bindings drifted.")


def _match_cell(value: float, cells: Sequence[float], *, name: str) -> float:
  matches = [cell for cell in cells if abs(value - cell) <= 1.0e-12]
  if len(matches) != 1:
    raise ValueError(f"{name}={value!r} is not a registered protocol cell.")
  return float(matches[0])


def _normalize_scan_rows(
  rows_value: object,
  protocol: DomainProtocol,
) -> list[dict[str, object]]:
  if protocol.cell_key is None or protocol.events_per_cell is None:
    raise ValueError("The flat protocol does not accept scan rows.")
  rows = _require_sequence(rows_value, name="adapter rows")
  by_cell: dict[float, dict[str, object]] = {}
  for index, value in enumerate(rows):
    row = _require_mapping(value, name=f"rows[{index}]")
    cell_value = _match_cell(
      _finite(row.get(protocol.cell_key), name=f"rows[{index}].{protocol.cell_key}"),
      protocol.cells,
      name=protocol.cell_key,
    )
    if cell_value in by_cell:
      raise ValueError(f"Duplicate {protocol.cell_key} row: {cell_value}.")
    trials = _integer(row.get("trials"), name=f"rows[{index}].trials", minimum=1)
    if trials != protocol.events_per_cell:
      raise ValueError(
        f"{protocol.cell_key}={cell_value} requires exactly "
        f"{protocol.events_per_cell} trials, got {trials}."
      )
    successes = _integer(
      row.get("successes"), name=f"rows[{index}].successes", minimum=0
    )
    if successes > trials:
      raise ValueError("Row successes cannot exceed trials.")
    terminations = _integer(
      row.get("terminations"), name=f"rows[{index}].terminations", minimum=0
    )
    contacts = _integer(
      row.get("non_wheel_contacts"),
      name=f"rows[{index}].non_wheel_contacts",
      minimum=0,
    )
    if terminations > trials or contacts > trials:
      raise ValueError("Termination/contact trial counts cannot exceed trials.")
    false_positives = _integer(
      row.get("stair_mode_false_positives", 0),
      name=f"rows[{index}].stair_mode_false_positives",
      minimum=0,
    )
    rate = successes / trials
    if "success_rate" in row and not math.isclose(
      _finite(row["success_rate"], name=f"rows[{index}].success_rate"),
      rate,
      rel_tol=0.0,
      abs_tol=1.0e-12,
    ):
      raise ValueError("Reported success_rate does not match successes/trials.")
    passed = (
      None
      if protocol.success_rate_limit is None
      else bool(
        rate >= protocol.success_rate_limit
        and terminations == 0
        and contacts == 0
      )
    )
    normalized: dict[str, object] = {
      protocol.cell_key: cell_value,
      "trials": trials,
      "successes": successes,
      "success_rate": rate,
      "terminations": terminations,
      "non_wheel_contacts": contacts,
      "stair_mode_false_positives": false_positives,
      "passed": passed,
    }
    for optional_count in ("trigger_count", "pre_impact_trigger_count"):
      if optional_count in row:
        normalized[optional_count] = _integer(
          row[optional_count],
          name=f"rows[{index}].{optional_count}",
          minimum=0,
        )
    by_cell[cell_value] = normalized
  if set(by_cell) != set(protocol.cells):
    missing = sorted(set(protocol.cells) - set(by_cell))
    extra = sorted(set(by_cell) - set(protocol.cells))
    raise ValueError(f"Scan row cells are incomplete (missing={missing}, extra={extra}).")
  return [by_cell[float(cell)] for cell in protocol.cells]


def _highest_contiguous_passing_height(
  rows: Sequence[Mapping[str, object]],
) -> float | None:
  """Mirror the frozen prefix helper without importing its NumPy module."""

  highest: float | None = None
  for row in sorted(rows, key=lambda item: float(item["height_m"])):
    passed = (
      float(row["success_rate"]) >= SUCCESS_RATE_LIMIT
      and int(row["terminations"]) == 0
      and int(row["non_wheel_contacts"]) == 0
    )
    if not passed:
      break
    highest = float(row["height_m"])
  return highest

def _normalize_gate_results(
  value: object,
  *,
  profile: str,
) -> tuple[list[dict[str, object]], dict[str, bool]]:
  rows = _require_sequence(value, name="adapter gates")
  bindings = gate_bindings_for_profile(profile)
  by_name: dict[str, dict[str, object]] = {}
  booleans: dict[str, bool] = {}
  for index, item in enumerate(rows):
    row = _require_mapping(item, name=f"gates[{index}]")
    name = row.get("name")
    if not isinstance(name, str) or name not in bindings:
      raise ValueError(f"Unknown gate result name: {name!r}.")
    if name in by_name:
      raise ValueError(f"Duplicate gate result: {name}.")
    binding = bindings[name]
    expected_counts = {
      "num_envs": binding.num_envs,
      "steps": binding.steps,
      "scenario_count": binding.scenario_count,
      "kick_events": binding.minimum_kick_events,
    }
    normalized_counts: dict[str, int] = {}
    for field, expected in expected_counts.items():
      actual = _integer(row.get(field, 0), name=f"gates[{index}].{field}")
      if actual != expected:
        raise ValueError(
          f"Gate {name} requires {field}={expected}, got {actual}."
        )
      normalized_counts[field] = actual
    upstream = _boolean(
      row.get("upstream_gate_passed"),
      name=f"gates[{index}].upstream_gate_passed",
    )
    safety: dict[str, int] = {}
    for field in (
      "terminations",
      "non_wheel_contacts",
      "stair_mode_false_positives",
    ):
      safety[field] = _integer(
        row.get(field), name=f"gates[{index}].{field}", minimum=0
      )
    passed = upstream and not any(safety.values())
    if "passed" in row and _boolean(
      row["passed"], name=f"gates[{index}].passed"
    ) != passed:
      raise ValueError(f"Gate {name} reported a boolean inconsistent with evidence.")
    normalized = {
      "name": name,
      "binding": binding.to_dict(),
      "upstream_gate_passed": upstream,
      **normalized_counts,
      **safety,
      "passed": passed,
    }
    by_name[name] = normalized
    booleans[name] = passed
  if set(by_name) != set(bindings):
    missing = sorted(set(bindings) - set(by_name))
    extra = sorted(set(by_name) - set(bindings))
    raise ValueError(f"Gate results are incomplete (missing={missing}, extra={extra}).")
  return [by_name[name] for name in GATE_NAMES], {
    name: booleans[name] for name in GATE_NAMES
  }


def finalize_adapter_output(
  config: Mapping[str, object],
  collection: Mapping[str, object],
) -> dict[str, object]:
  """Validate aggregate adapter data and build one machine-readable result."""

  _validate_adapter_config(config)
  config_hash = str(config["config_sha256"])
  if collection.get("config_sha256") != config_hash:
    raise ValueError("Adapter output is not bound to the supplied config digest.")
  source = collection.get("evaluation_source", "live_adapter")
  if not isinstance(source, str) or not source.strip():
    raise ValueError("evaluation_source must be a non-empty string.")
  metadata = collection.get("adapter_metadata", {})
  _validate_json_value(metadata, path="adapter_metadata")
  metadata_map = dict(_require_mapping(metadata, name="adapter_metadata"))
  domain = str(config["domain"])
  profile = str(config["profile"])
  result: dict[str, object] = {
    "schema_version": EVALUATOR_SCHEMA_VERSION,
    "kind": EVALUATION_ENVELOPE_KIND,
    "status": "complete",
    "task": STAIR_CAMP_TASK_ID,
    "domain": domain,
    "profile": profile,
    "evaluation_seed": REGISTERED_EVALUATION_SEED,
    "evaluation_source": source,
    "device": config["device"],
    "evidence_eligible": bool(config["evidence_eligible"]),
    "promotion_evidence_eligible": bool(
      config["promotion_evidence_eligible"]
    ),
    "config_sha256": config_hash,
    "checkpoint": config["checkpoint"],
    "protocol": config["protocol"],
    "ablation": config["ablation"],
    "adapter_metadata": metadata_map,
  }
  if domain == "flat":
    gates, gate_booleans = _normalize_gate_results(
      collection.get("gates"), profile=profile
    )
    result.update(
      {
        "gates": gates,
        "gate_booleans": gate_booleans,
        "all_gates_passed": all(gate_booleans.values()),
        "result_passed": all(gate_booleans.values()),
      }
    )
  else:
    protocol = protocol_for(domain, profile)
    rows = _normalize_scan_rows(collection.get("rows"), protocol)
    result["rows"] = rows
    if domain == "stairs":
      all_cells_passed = all(bool(row["passed"]) for row in rows)
      result["all_cells_passed"] = all_cells_passed
      result["result_passed"] = all_cells_passed
      result["highest_contiguous_passing_height_m"] = (
        _highest_contiguous_passing_height(rows)
      )
    else:
      result["secondary_metric_only"] = True
      result["registered_pass_threshold"] = None
      result["result_passed"] = None
  _validate_json_value(result)
  return result


def run_live_adapter(
  config: Mapping[str, object],
  adapter: LiveAdapter | Callable[[Mapping[str, object]], Mapping[str, object]],
) -> dict[str, object]:
  """Call a lazy adapter and immediately validate everything it returns."""

  _validate_adapter_config(config)
  collector = getattr(adapter, "collect", None)
  if callable(collector):
    collected = collector(config)
  elif callable(adapter):
    collected = adapter(config)
  else:
    raise ValueError("Live adapter must be callable or expose collect(config).")
  return finalize_adapter_output(
    config,
    _require_mapping(collected, name="live adapter result"),
  )


def load_live_adapter(spec: str) -> object:
  """Load ``module:attribute`` lazily; the target owns all MjLab imports."""

  if not isinstance(spec, str) or spec.count(":") != 1:
    raise ValueError("Adapter must use the form 'module:callable'.")
  module_name, attribute = spec.split(":", 1)
  if not module_name or not attribute:
    raise ValueError("Adapter must use the form 'module:callable'.")
  module = importlib.import_module(module_name)
  try:
    adapter = getattr(module, attribute)
  except AttributeError as exc:
    raise ValueError(f"Adapter attribute does not exist: {spec}.") from exc
  if not callable(adapter) and not callable(getattr(adapter, "collect", None)):
    raise ValueError(f"Adapter is not callable: {spec}.")
  return adapter


def _normalize_gate_boolean_map(value: object, *, name: str) -> dict[str, bool]:
  mapping = _require_mapping(value, name=name)
  if set(mapping) != set(GATE_NAMES):
    raise ValueError(f"{name} must contain exactly the four registered gate names.")
  return {gate: _boolean(mapping[gate], name=f"{name}.{gate}") for gate in GATE_NAMES}


def _normalize_gate_count_map(value: object, *, name: str) -> dict[str, int]:
  mapping = _require_mapping(value, name=name)
  if set(mapping) != set(GATE_NAMES):
    raise ValueError(f"{name} must contain exactly the four registered gate names.")
  return {
    gate: _integer(mapping[gate], name=f"{name}.{gate}", minimum=0)
    for gate in GATE_NAMES
  }


def make_k3_screen_candidate(
  *,
  checkpoint_envelope: Mapping[str, object],
  budget_updates: int,
  gate_passes: Mapping[str, object],
  gate_stair_mode_false_positives: Mapping[str, object],
  height_row: Mapping[str, object],
) -> dict[str, object]:
  """Build one rejection-only K=3 screen envelope."""

  checkpoint = validate_stair_camp_checkpoint_envelope(checkpoint_envelope)
  budget = _integer(budget_updates, name="budget_updates", minimum=1)
  if budget not in REGISTERED_BUDGETS:
    raise ValueError("K=3 budget must be the registered 1000 or 3000 updates.")
  gates = _normalize_gate_boolean_map(gate_passes, name="gate_passes")
  false_positives = _normalize_gate_count_map(
    gate_stair_mode_false_positives,
    name="gate_stair_mode_false_positives",
  )
  rows = _normalize_scan_rows([height_row], K3_SCREEN_PROTOCOL)
  screen_passed = bool(
    all(gates.values())
    and not any(false_positives.values())
    and bool(rows[0]["passed"])
    and int(rows[0]["stair_mode_false_positives"]) == 0
  )
  return {
    "schema_version": EVALUATOR_SCHEMA_VERSION,
    "kind": K3_SCREEN_KIND,
    "task": STAIR_CAMP_TASK_ID,
    "profile": "screen",
    "evidence_eligible": False,
    "evaluation_seed": REGISTERED_EVALUATION_SEED,
    "budget_updates": budget,
    "checkpoint": checkpoint,
    "gate_passes": gates,
    "gate_stair_mode_false_positives": false_positives,
    "height_screen": rows[0],
    "screen_passed": screen_passed,
  }


def validate_k3_screen_candidate(candidate: Mapping[str, object]) -> dict[str, object]:
  if candidate.get("schema_version") != EVALUATOR_SCHEMA_VERSION:
    raise ValueError("K=3 candidate schema version does not match.")
  if candidate.get("kind") != K3_SCREEN_KIND:
    raise ValueError("Input is not a StairCamp K=3 screen candidate.")
  if candidate.get("task") != STAIR_CAMP_TASK_ID:
    raise ValueError("K=3 candidate task does not match StairCamp.")
  if (
    candidate.get("profile") != "screen"
    or candidate.get("evidence_eligible") is not False
  ):
    raise ValueError("K=3 candidate must be rejection-only screen evidence.")
  if candidate.get("evaluation_seed") != REGISTERED_EVALUATION_SEED:
    raise ValueError("K=3 candidate evaluation seed must be 1.")
  normalized = make_k3_screen_candidate(
    checkpoint_envelope=_require_mapping(
      candidate.get("checkpoint"), name="K=3 checkpoint"
    ),
    budget_updates=_integer(
      candidate.get("budget_updates"), name="budget_updates", minimum=1
    ),
    gate_passes=_require_mapping(candidate.get("gate_passes"), name="gate_passes"),
    gate_stair_mode_false_positives=_require_mapping(
      candidate.get("gate_stair_mode_false_positives"),
      name="gate_stair_mode_false_positives",
    ),
    height_row=_require_mapping(
      candidate.get("height_screen"), name="height_screen"
    ),
  )
  if candidate.get("screen_passed") is not normalized["screen_passed"]:
    raise ValueError("K=3 candidate screen_passed is inconsistent with evidence.")
  return normalized


def select_newest_passing_checkpoint(
  candidates: Sequence[Mapping[str, object]],
) -> dict[str, object]:
  """Select the newest passer from exactly the latest three checkpoints."""

  if len(candidates) != 3:
    raise ValueError("K=3 selection requires exactly three candidate envelopes.")
  normalized = [validate_k3_screen_candidate(candidate) for candidate in candidates]
  budgets = {int(candidate["budget_updates"]) for candidate in normalized}
  if len(budgets) != 1:
    raise ValueError("K=3 candidates must belong to one budget pool.")
  budget = budgets.pop()
  checkpoints = [
    _require_mapping(candidate["checkpoint"], name="candidate checkpoint")
    for candidate in normalized
  ]
  training = [
    _require_mapping(checkpoint["training"], name="candidate training")
    for checkpoint in checkpoints
  ]
  update_counts = [int(item["completed_updates"]) for item in training]
  # RSL-RL names periodic saves by the zero-based `iter` that just
  # completed. Thus model_800/model_900/model_999 attest 801/901/1000
  # completed updates; the final save supplies the exact budget endpoint.
  expected_updates = {
    budget - 2 * CHECKPOINT_SAVE_INTERVAL + 1,
    budget - CHECKPOINT_SAVE_INTERVAL + 1,
    budget,
  }
  if set(update_counts) != expected_updates or len(set(update_counts)) != 3:
    raise ValueError(
      "K=3 candidates must be exactly the latest three save-interval updates."
    )
  checkpoint_hashes = [str(item["checkpoint_file_sha256"]) for item in checkpoints]
  if len(set(checkpoint_hashes)) != 3:
    raise ValueError("K=3 candidates must reference three distinct checkpoints.")
  for key in (
    "training_seed",
    "git_sha",
    "contract_sha256",
    "artifact_bindings",
    "action_scales",
    "action_mask",
  ):
    first = training[0].get(key)
    if any(item.get(key) != first for item in training[1:]):
      raise ValueError(f"K=3 candidate checkpoint bindings disagree on {key}.")
  ordered = sorted(
    normalized,
    key=lambda candidate: int(
      _require_mapping(
        _require_mapping(candidate["checkpoint"], name="checkpoint")["training"],
        name="training",
      )["completed_updates"]
    ),
    reverse=True,
  )
  selected = next(
    (candidate for candidate in ordered if candidate["screen_passed"] is True),
    None,
  )
  result = {
    "schema_version": EVALUATOR_SCHEMA_VERSION,
    "kind": K3_SELECTION_KIND,
    "task": STAIR_CAMP_TASK_ID,
    "status": "selected" if selected is not None else "no_passing_checkpoint",
    "classification": (
      "STAIR_CAMP_CHECKPOINT_SELECTED"
      if selected is not None
      else "STOP_NO_PROMOTION"
    ),
    "selection_rule": "newest_passing_of_exact_latest_3",
    "budget_updates": budget,
    "training_seed": training[0]["training_seed"],
    "ordered_candidates": [
      {
        "completed_updates": _require_mapping(
          _require_mapping(candidate["checkpoint"], name="checkpoint")["training"],
          name="training",
        )["completed_updates"],
        "checkpoint_file": _require_mapping(
          candidate["checkpoint"], name="checkpoint"
        )["checkpoint_file"],
        "checkpoint_file_sha256": _require_mapping(
          candidate["checkpoint"], name="checkpoint"
        )["checkpoint_file_sha256"],
        "screen_passed": candidate["screen_passed"],
      }
      for candidate in ordered
    ],
    "selected_checkpoint": None if selected is None else selected["checkpoint"],
  }
  _validate_json_value(result)
  return result


def _project_adjudication_rows(
  value: object,
  *,
  name: str,
  grid: tuple[float, ...] = STAIR_HEIGHTS_M,
) -> list[dict[str, float | int]]:
  rows = _require_sequence(value, name=name)
  by_height: dict[float, dict[str, float | int]] = {}
  for index, raw in enumerate(rows):
    row = _require_mapping(raw, name=f"{name}[{index}]")
    height = _match_cell(
      _finite(row.get("height_m"), name=f"{name}[{index}].height_m"),
      grid,
      name="height_m",
    )
    if height in by_height:
      raise ValueError(f"{name} duplicates height {height}.")
    trials = _integer(row.get("trials"), name=f"{name}[{index}].trials", minimum=1)
    if trials != OFFICIAL_EVENTS_PER_CELL:
      raise ValueError(f"{name} requires exactly 48 trials per height.")
    if "success_rate" in row:
      rate = _finite(
        row["success_rate"], name=f"{name}[{index}].success_rate"
      )
    else:
      successes = _integer(
        row.get("successes"), name=f"{name}[{index}].successes", minimum=0
      )
      if successes > trials:
        raise ValueError(f"{name} successes exceed trials.")
      rate = successes / trials
    if not 0.0 <= rate <= 1.0:
      raise ValueError(f"{name} success rate must be in [0, 1].")
    terminations = _integer(
      row.get("terminations"),
      name=f"{name}[{index}].terminations",
      minimum=0,
    )
    contacts = _integer(
      row.get("non_wheel_contacts"),
      name=f"{name}[{index}].non_wheel_contacts",
      minimum=0,
    )
    if terminations > trials or contacts > trials:
      raise ValueError(f"{name} safety counts exceed trials.")
    by_height[height] = {
      "height_m": height,
      "success_rate": rate,
      "terminations": terminations,
      "non_wheel_contacts": contacts,
      "trials": trials,
    }
  if set(by_height) != set(grid):
    raise ValueError(f"{name} does not cover the exact registered height grid.")
  return [by_height[height] for height in grid]


def _validate_formal_result(
  value: object,
  *,
  domain: str,
  ablation_name: str,
) -> dict[str, object]:
  result = _require_mapping(value, name=f"{domain}/{ablation_name} result")
  if (
    result.get("schema_version") != EVALUATOR_SCHEMA_VERSION
    or result.get("kind") != EVALUATION_ENVELOPE_KIND
    or result.get("status") != "complete"
    or result.get("task") != STAIR_CAMP_TASK_ID
    or result.get("domain") != domain
    or result.get("profile") != "formal"
    or result.get("evaluation_seed") != REGISTERED_EVALUATION_SEED
    or result.get("evidence_eligible") is not True
  ):
    raise ValueError(f"{domain}/{ablation_name} is not eligible formal evidence.")
  descriptor = resolve_ablation(ablation_name)
  if result.get("ablation") != descriptor.to_dict():
    raise ValueError(f"{domain}/{ablation_name} ablation descriptor drifted.")
  expected_promotion = (
    descriptor.promotion_evidence_eligible and domain == "stairs"
  )
  if result.get("promotion_evidence_eligible") is not expected_promotion:
    raise ValueError(f"{domain}/{ablation_name} promotion eligibility drifted.")
  protocol = protocol_for(domain, "formal")
  if result.get("protocol") != protocol.to_dict():
    raise ValueError(f"{domain}/{ablation_name} protocol drifted.")
  checkpoint_value = _require_mapping(
    result.get("checkpoint"), name=f"{domain}/{ablation_name} checkpoint"
  )
  if checkpoint_value.get("checkpoint_file_verified") is not True:
    raise ValueError(f"{domain}/{ablation_name} checkpoint was not verified.")
  checkpoint = validate_stair_camp_checkpoint_envelope(
    checkpoint_value,
    verify_file=True,
  )
  normalized: dict[str, object] = {
    "checkpoint": checkpoint,
    "result": dict(result),
  }
  if domain == "stairs":
    rows = _normalize_scan_rows(result.get("rows"), protocol)
    if result.get("rows") != rows:
      raise ValueError(f"{domain}/{ablation_name} rows are not canonical.")
    normalized["rows"] = rows
  elif domain == "flat":
    gates, booleans = _normalize_gate_results(
      result.get("gates"), profile="formal"
    )
    if result.get("gates") != gates or result.get("gate_booleans") != booleans:
      raise ValueError("Flat formal gate evidence is not canonical.")
    normalized["gates"] = gates
    normalized["gate_booleans"] = booleans
  return normalized


def _validate_k3_selection_for_composition(
  value: object,
  *,
  checkpoint: Mapping[str, object],
  budget_iterations: int,
) -> None:
  selection = _require_mapping(value, name="K=3 selection")
  if (
    selection.get("schema_version") != EVALUATOR_SCHEMA_VERSION
    or selection.get("kind") != K3_SELECTION_KIND
    or selection.get("task") != STAIR_CAMP_TASK_ID
    or selection.get("status") != "selected"
    or selection.get("classification") != "STAIR_CAMP_CHECKPOINT_SELECTED"
    or selection.get("selection_rule") != "newest_passing_of_exact_latest_3"
    or selection.get("budget_updates") != budget_iterations
  ):
    raise ValueError("K=3 selection is not the registered selected result.")
  selected_raw = _require_mapping(
    selection.get("selected_checkpoint"), name="selected K=3 checkpoint"
  )
  selected = validate_stair_camp_checkpoint_envelope(
    selected_raw, verify_file=True
  )
  if selected != checkpoint:
    raise ValueError("Formal evaluation checkpoint differs from K=3 selection.")
  training = _require_mapping(checkpoint.get("training"), name="checkpoint training")
  if selection.get("training_seed") != training.get("training_seed"):
    raise ValueError("K=3 selection training seed drifted.")
  ordered = _require_sequence(
    selection.get("ordered_candidates"), name="K=3 ordered candidates"
  )
  if len(ordered) != 3:
    raise ValueError("K=3 selection must archive exactly three candidates.")
  updates: list[int] = []
  hashes: list[str] = []
  passing: list[bool] = []
  for index, candidate_value in enumerate(ordered):
    row = _require_mapping(
      candidate_value, name=f"K=3 ordered_candidates[{index}]"
    )
    if set(row) != {
      "completed_updates",
      "checkpoint_file",
      "checkpoint_file_sha256",
      "screen_passed",
    }:
      raise ValueError("K=3 candidate summary schema drifted.")
    updates.append(
      _integer(row["completed_updates"], name="completed_updates", minimum=1)
    )
    hashes.append(_sha256(row["checkpoint_file_sha256"], name="checkpoint hash"))
    passing.append(_boolean(row["screen_passed"], name="screen_passed"))
  expected_updates = [
    budget_iterations,
    budget_iterations - CHECKPOINT_SAVE_INTERVAL + 1,
    budget_iterations - 2 * CHECKPOINT_SAVE_INTERVAL + 1,
  ]
  if updates != expected_updates or len(set(hashes)) != 3:
    raise ValueError("K=3 ordered candidate cadence drifted.")
  first_passer = next((index for index, passed in enumerate(passing) if passed), None)
  selected_training = _require_mapping(selected["training"], name="selected training")
  if first_passer is None or (
    ordered[first_passer]["checkpoint_file"] != selected["checkpoint_file"]
    or ordered[first_passer]["checkpoint_file_sha256"]
    != selected["checkpoint_file_sha256"]
    or ordered[first_passer]["completed_updates"]
    != selected_training["completed_updates"]
  ):
    raise ValueError("K=3 selected checkpoint is not the newest passer.")


def compose_adjudication_seed_envelope(
  *,
  stairs_result: object,
  flat_result: object,
  classical_rows: object,
  ablation_results: Sequence[object],
  k3_selection: object,
  budget_iterations: int,
) -> dict[str, object]:
  """Compose one strict adjudicator envelope from canonical evaluator outputs."""

  if isinstance(budget_iterations, bool) or budget_iterations not in REGISTERED_BUDGETS:
    raise ValueError("Adjudication budget must be exactly 1000 or 3000.")
  stairs = _validate_formal_result(
    stairs_result, domain="stairs", ablation_name="baseline"
  )
  flat = _validate_formal_result(
    flat_result, domain="flat", ablation_name="baseline"
  )
  checkpoint = _require_mapping(stairs["checkpoint"], name="stairs checkpoint")
  if flat["checkpoint"] != checkpoint:
    raise ValueError("Flat and stairs results use different checkpoints.")
  _validate_k3_selection_for_composition(
    k3_selection,
    checkpoint=checkpoint,
    budget_iterations=budget_iterations,
  )
  if len(ablation_results) != len(ADJUDICATION_ABLATION_NAMES):
    raise ValueError("Every registered StairCamp ablation result is required.")
  completed: list[str] = []
  for expected_name, result in zip(
    ADJUDICATION_ABLATION_NAMES, ablation_results, strict=True
  ):
    validated = _validate_formal_result(
      result, domain="stairs", ablation_name=expected_name
    )
    if validated["checkpoint"] != checkpoint:
      raise ValueError(f"Ablation {expected_name} uses a different checkpoint.")
    completed.append(expected_name)

  training = _require_mapping(checkpoint["training"], name="checkpoint training")
  gates = _require_sequence(flat["gates"], name="flat gates")
  false_positives = {
    str(_require_mapping(row, name="flat gate")["name"]): int(
      _require_mapping(row, name="flat gate")["stair_mode_false_positives"]
    )
    for row in gates
  }
  gate_booleans = _require_mapping(flat["gate_booleans"], name="gate booleans")
  envelope: dict[str, object] = {
    "training_seed": training["training_seed"],
    "evaluation_seed": REGISTERED_EVALUATION_SEED,
    "budget_iterations": budget_iterations,
    "git_sha": training["git_sha"],
    "contract_hash": training["contract_sha256"],
    "artifact_bindings": training["artifact_bindings"],
    "classical_rows": _project_adjudication_rows(
      classical_rows, name="classical_rows", grid=CLASSICAL_HEIGHTS_M
    ),
    "residual_rows": _project_adjudication_rows(
      stairs["rows"], name="residual_rows"
    ),
    **{name: gate_booleans[name] for name in GATE_NAMES},
    "gate_stair_mode_false_positives": false_positives,
    "completed_ablations": completed,
    "ablations_complete": set(completed) == set(ADJUDICATION_ABLATION_NAMES),
    "evidence_eligible": True,
    "checkpoint": checkpoint["checkpoint_file"],
    "checkpoint_file_sha256": checkpoint["checkpoint_file_sha256"],
  }
  _validate_json_value(envelope)
  return envelope


def manifest_payload(profile: str = "formal") -> dict[str, object]:
  """Expose every registered sidecar constant as deterministic JSON."""

  protocols = {
    domain: protocol_for(domain, profile).to_dict()
    for domain in ("stairs", "flat", "slope")
  }
  gates = {
    name: binding.to_dict()
    for name, binding in gate_bindings_for_profile(profile).items()
  }
  return {
    "schema_version": EVALUATOR_SCHEMA_VERSION,
    "kind": "stair_camp_evaluator_manifest",
    "task": STAIR_CAMP_TASK_ID,
    "profile": profile,
    "evaluation_seed": REGISTERED_EVALUATION_SEED,
    "training_seeds": list(REGISTERED_TRAINING_SEEDS),
    "protocols": protocols,
    "gate_bindings": gates,
    "ablations": {
      name: descriptor.to_dict()
      for name, descriptor in ABLATION_DESCRIPTORS.items()
    },
    "adjudication_ablations": list(ADJUDICATION_ABLATION_NAMES),
    "checkpoint_contract": {
      "training_info_key": STAIR_CAMP_TRAINING_INFO_KEY,
      "training_schema_version": STAIR_CAMP_CONTRACT_SCHEMA_VERSION,
      "action_mask": list(STAIR_CAMP_ACTION_MASK),
      "action_scales": list(STAIR_CAMP_STAGE.action_scales),
      "actor_observation_width": STAIR_CAMP_ACTOR_WIDTH,
      "critic_observation_width": STAIR_CAMP_CRITIC_WIDTH,
      "action_width": STAIR_CAMP_ACTION_WIDTH,
    },
    "k3": {
      "pool_size": 3,
      "save_interval_updates": CHECKPOINT_SAVE_INTERVAL,
      "budgets": list(REGISTERED_BUDGETS),
      "selection_rule": "newest_passing_of_exact_latest_3",
      "screen_protocol": K3_SCREEN_PROTOCOL.to_dict(),
    },
    "live_adapter_contract": {
      "entrypoint": "module:callable",
      "input_kind": "stair_camp_adapter_config",
      "output_binding_field": "config_sha256",
      "stairs_or_slope_output": "rows",
      "flat_output": "gates",
      "mjlab_import_policy": "adapter_only",
    },
  }


def write_machine_output(
  payload: Mapping[str, object], output: Path | None,
) -> None:
  """Print strict JSON or atomically create a new result file."""

  encoded = deterministic_json(payload)
  if output is None:
    print(encoded, end="")
    return
  if output.exists():
    raise FileExistsError(f"Refusing to overwrite output: {output}")
  output.parent.mkdir(parents=True, exist_ok=True)
  temporary = output.with_name(f".{output.name}.incomplete.{uuid.uuid4().hex}")
  try:
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
      stream.write(encoded)
      stream.flush()
      os.fsync(stream.fileno())
    # A hard link publishes the fully-fsynced inode atomically and, unlike
    # os.replace(), fails if another process won the output-name race.
    os.link(temporary, output)
  finally:
    if temporary.exists():
      temporary.unlink()


def _read_json_mapping(path: Path, *, name: str) -> Mapping[str, object]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  return _require_mapping(payload, name=name)


def _add_output_argument(parser: argparse.ArgumentParser) -> None:
  parser.add_argument("--output", type=Path, default=None)


def _add_expectation_arguments(parser: argparse.ArgumentParser) -> None:
  parser.add_argument("--expected-git-sha", default=None)
  parser.add_argument(
    "--expected-contract-sha256",
    "--expected-contract-hash",
    dest="expected_contract_sha256",
    default=None,
  )
  parser.add_argument("--expected-training-seed", type=int, default=None)
  parser.add_argument("--verify-checkpoint-file", action="store_true")


def _expectation_from_args(args: argparse.Namespace) -> CheckpointExpectation:
  return CheckpointExpectation(
    git_sha=getattr(args, "expected_git_sha", None),
    contract_sha256=getattr(args, "expected_contract_sha256", None),
    training_seed=getattr(args, "expected_training_seed", None),
  )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  subparsers = parser.add_subparsers(dest="command", required=True)

  manifest = subparsers.add_parser("manifest", help="Print the frozen protocol.")
  manifest.add_argument("--profile", choices=("formal", "smoke"), default="formal")
  _add_output_argument(manifest)

  envelope = subparsers.add_parser(
    "checkpoint-envelope", help="Create a verified envelope from one .pt file."
  )
  envelope.add_argument("--checkpoint-file", type=Path, required=True)
  _add_expectation_arguments(envelope)
  _add_output_argument(envelope)

  validate = subparsers.add_parser(
    "validate-checkpoint", help="Validate one tensor-free checkpoint envelope."
  )
  validate.add_argument("--envelope", type=Path, required=True)
  _add_expectation_arguments(validate)
  _add_output_argument(validate)

  for command in ("finalize", "live"):
    subparser = subparsers.add_parser(command)
    subparser.add_argument(
      "--domain", choices=("stairs", "flat", "slope"), required=True
    )
    subparser.add_argument(
      "--profile", choices=("formal", "smoke"), default="formal"
    )
    subparser.add_argument("--checkpoint-envelope", type=Path, required=True)
    subparser.add_argument(
      "--ablation", choices=tuple(ABLATION_DESCRIPTORS), default="baseline"
    )
    subparser.add_argument("--device", default="cpu")
    _add_expectation_arguments(subparser)
    if command == "finalize":
      subparser.add_argument("--collection", type=Path, required=True)
    else:
      subparser.add_argument("--adapter", required=True)
    _add_output_argument(subparser)

  compose = subparsers.add_parser(
    "compose-seed", help="Compose one strict three-seed adjudicator envelope."
  )
  compose.add_argument("--stairs-result", type=Path, required=True)
  compose.add_argument("--flat-result", type=Path, required=True)
  compose.add_argument("--classical-rows", type=Path, required=True)
  compose.add_argument(
    "--ablation-result",
    type=Path,
    nargs=len(ADJUDICATION_ABLATION_NAMES),
    required=True,
  )
  compose.add_argument("--k3-selection", type=Path, required=True)
  compose.add_argument("--budget-iterations", type=int, required=True)
  _add_output_argument(compose)

  select = subparsers.add_parser(
    "select-k3", help="Select the newest passer from three screen envelopes."
  )
  select.add_argument("--candidate", type=Path, nargs=3, required=True)
  _add_output_argument(select)
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  if args.command == "manifest":
    result = manifest_payload(args.profile)
  elif args.command == "checkpoint-envelope":
    result = checkpoint_envelope_from_file(
      args.checkpoint_file,
      expectation=_expectation_from_args(args),
    )
  elif args.command == "validate-checkpoint":
    checkpoint = validate_stair_camp_checkpoint_envelope(
      _read_json_mapping(args.envelope, name="checkpoint envelope"),
      expectation=_expectation_from_args(args),
      verify_file=args.verify_checkpoint_file,
    )
    result = {
      "schema_version": EVALUATOR_SCHEMA_VERSION,
      "kind": "stair_camp_checkpoint_validation",
      "valid": True,
      "checkpoint": checkpoint,
    }
  elif args.command in ("finalize", "live"):
    config = make_adapter_config(
      domain=args.domain,
      checkpoint_envelope=_read_json_mapping(
        args.checkpoint_envelope, name="checkpoint envelope"
      ),
      profile=args.profile,
      ablation=args.ablation,
      device=args.device,
      expectation=_expectation_from_args(args),
      verify_checkpoint_file=args.verify_checkpoint_file,
    )
    if args.command == "finalize":
      result = finalize_adapter_output(
        config,
        _read_json_mapping(args.collection, name="adapter collection"),
      )
    else:
      result = run_live_adapter(config, load_live_adapter(args.adapter))
  elif args.command == "compose-seed":
    classical_wrapper = _read_json_mapping(
      args.classical_rows, name="classical rows wrapper"
    )
    if set(classical_wrapper) != {"rows"}:
      raise ValueError('Classical rows file must contain only the key "rows".')
    result = compose_adjudication_seed_envelope(
      stairs_result=_read_json_mapping(
        args.stairs_result, name="stairs result"
      ),
      flat_result=_read_json_mapping(args.flat_result, name="flat result"),
      classical_rows=classical_wrapper["rows"],
      ablation_results=[
        _read_json_mapping(path, name=f"ablation result {index}")
        for index, path in enumerate(args.ablation_result, start=1)
      ],
      k3_selection=_read_json_mapping(
        args.k3_selection, name="K=3 selection"
      ),
      budget_iterations=args.budget_iterations,
    )
  elif args.command == "select-k3":
    result = select_newest_passing_checkpoint(
      [
        _read_json_mapping(path, name=f"K=3 candidate {index}")
        for index, path in enumerate(args.candidate, start=1)
      ]
    )
  else:  # pragma: no cover - argparse makes this unreachable.
    raise AssertionError(f"Unhandled command: {args.command}")
  write_machine_output(result, args.output)
  return 0


__all__ = [
  "ABLATION_DESCRIPTORS",
  "ADJUDICATION_ABLATION_NAMES",
  "APPROACH_DISTANCE_M",
  "BASELINE_ABLATION",
  "CHECKPOINT_ENVELOPE_KIND",
  "CHECKPOINT_SAVE_INTERVAL",
  "CROSS_DEPTH_M",
  "EVALUATION_ENVELOPE_KIND",
  "EVALUATOR_SCHEMA_VERSION",
  "FLAT_PROTOCOL",
  "GATE_BINDINGS",
  "GATE_NAMES",
  "K3_SCREEN_KIND",
  "K3_SCREEN_PROTOCOL",
  "K3_SELECTION_KIND",
  "LEG_OFF_ABLATION",
  "MODE_ALWAYS_ON_ABLATION",
  "OFFICIAL_ENVS_PER_CELL",
  "OFFICIAL_EVENTS_PER_CELL",
  "OFFICIAL_MIN_SUCCESSES",
  "OFFICIAL_REPEATS",
  "REGISTERED_BUDGETS",
  "REGISTERED_EVALUATION_SEED",
  "REGISTERED_TRAINING_SEEDS",
  "SLOPE_DEGREES",
  "SLOPE_PROTOCOL",
  "STAIRS_PROTOCOL",
  "STAIR_CAMP_ACTION_WIDTH",
  "STAIR_CAMP_ACTOR_WIDTH",
  "STAIR_CAMP_ARTIFACT_BINDING_NAMES",
  "STAIR_CAMP_CANONICAL_CONTRACT_SHA256",
  "STAIR_CAMP_CRITIC_WIDTH",
  "STAIR_HEIGHTS_M",
  "SUCCESS_RATE_LIMIT",
  "SUCCESS_TRAVEL_DISTANCE_M",
  "ZERO_SHOT_LEG_SCALES_RAD",
  "ZERO_SHOT_SCALE_ABLATIONS",
  "AblationDescriptor",
  "CheckpointExpectation",
  "DomainProtocol",
  "GateBinding",
  "checkpoint_envelope_from_file",
  "checkpoint_envelope_from_loaded_checkpoint",
  "compose_adjudication_seed_envelope",
  "deterministic_json",
  "finalize_adapter_output",
  "gate_bindings_for_profile",
  "load_live_adapter",
  "main",
  "make_adapter_config",
  "make_k3_screen_candidate",
  "manifest_payload",
  "parse_args",
  "protocol_for",
  "resolve_ablation",
  "run_live_adapter",
  "select_newest_passing_checkpoint",
  "validate_k3_screen_candidate",
  "validate_stair_camp_checkpoint_envelope",
  "validate_stair_camp_training_info",
  "write_machine_output",
]


if __name__ == "__main__":
  raise SystemExit(main())
