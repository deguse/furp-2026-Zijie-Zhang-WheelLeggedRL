#!/usr/bin/env python3
# ruff: noqa: E402, TRY004
"""Pure evaluation contracts for HopperTrex Hybrid-v3 StairDynamic.

This module deliberately imports neither MjLab nor RSL-RL. It validates the
v3 checkpoint provenance, signs rollout requests, normalizes per-trial stair
evidence, reuses the frozen StairCamp flat/standing/velocity/Stage5 gate
bindings, validates the six registered ablations, and performs rejection-only
K=3 selection. The sibling stair_dynamic_live_adapter has one narrow job:
consume a signed request and return the JSON-safe collection accepted by
finalize_collection().
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[3]
if str(SRC_PATH) not in sys.path:
  sys.path.insert(0, str(SRC_PATH))

from hoppertrex_mjlab.hybrid.config import HYBRID_ACTION_NAMES, HYBRID_ACTION_STD
from hoppertrex_mjlab.hybrid.stair_dynamic import (
  DYNAMIC_STAIR_CONTRACT_SCHEMA_VERSION,
  DYNAMIC_STAIR_FEEDFORWARD_LIMIT_RAD,
  DYNAMIC_STAIR_PPO_LEG_SCALE_RAD,
  DYNAMIC_STAIR_TASK_ID,
  DynamicLiftMode,
)
from hoppertrex_mjlab.hybrid.stair_dynamic_contract import (
  DYNAMIC_STAIR_ACTION_SCALES,
  DYNAMIC_STAIR_ACTOR_WIDTH,
  DYNAMIC_STAIR_CRITIC_WIDTH,
  DYNAMIC_STAIR_EXTENSION_TOTAL_UPDATES,
  DYNAMIC_STAIR_MIGRATION_INFO_KEY,
  DYNAMIC_STAIR_PROBE_UPDATES,
  DYNAMIC_STAIR_SAVE_INTERVAL,
  DYNAMIC_STAIR_TRAINING_INFO_KEY,
)
from hoppertrex_mjlab.scripts.rsl_rl import (
  evaluate_stair_camp as stair_camp,
)

EVALUATOR_SCHEMA_VERSION = 1
CHECKPOINT_ENVELOPE_KIND = "stair_dynamic_checkpoint"
MIGRATION_CHECKPOINT_ENVELOPE_KIND = "stair_dynamic_migration_checkpoint"
EXTENSION_AUTHORIZATION_KIND = "stair_dynamic_extension_authorization"
EVALUATION_REQUEST_KIND = "stair_dynamic_evaluation_request"
EVALUATION_RESULT_KIND = "stair_dynamic_evaluation"
ABLATION_BUNDLE_KIND = "stair_dynamic_ablation_bundle"
K3_SCREEN_KIND = "stair_dynamic_k3_screen"
K3_SELECTION_KIND = "stair_dynamic_k3_selection"
REGISTERED_TRAINING_SEED = 1
REGISTERED_EVALUATION_SEED = 1
REGISTERED_BUDGETS = (
  DYNAMIC_STAIR_PROBE_UPDATES,
  DYNAMIC_STAIR_EXTENSION_TOTAL_UPDATES,
)
ACTION_WIDTH = 6
FORMAL_HEIGHTS_M = (0.01, 0.02, 0.03)
FORMAL_ENVS_PER_HEIGHT = 16
FORMAL_REPEATS = 3
FORMAL_TRIALS_PER_HEIGHT = FORMAL_ENVS_PER_HEIGHT * FORMAL_REPEATS
FORMAL_MIN_SUCCESSES = 44
FORMAL_STABLE_STEPS = 25
PRIMARY_HEIGHT_M = 0.01
CONTINUOUS_RISERS = 3
K3_ENVS = 16
K3_REPEATS = 1
K3_MIN_SUCCESSES = 15

PHASE_NAMES = (
  "IDLE",
  "APPROACH",
  "PRELOAD",
  "CONTACT_WAIT",
  "LEAD_LIFT",
  "TRAIL_LIFT",
  "RECOVER",
  "DONE",
  "ABORT",
)
TRAVERSAL_MODES = ("ROLL", "DYNAMIC", "ABORT")
LIFT_MODES = tuple(mode.value for mode in DynamicLiftMode)
LEAD_SIDES = ("NONE", "LEFT", "RIGHT")

# Re-export the exact frozen gate protocol instead of creating a v3 copy.
GateBinding = stair_camp.GateBinding
GATE_BINDINGS = stair_camp.GATE_BINDINGS
GATE_NAMES = stair_camp.GATE_NAMES
gate_bindings_for_profile = stair_camp.gate_bindings_for_profile
deterministic_json = stair_camp.deterministic_json
write_machine_output = stair_camp.write_machine_output

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

DYNAMIC_ARTIFACT_BINDING_NAMES = (
  "controller_gain_hash",
  "calibration_hash",
  "yaw_calibration_hash",
  "posture_map_hash",
  "posture_artifact_hash",
  "station_calibration_hash",
  "dynamic_maneuver_hash",
)


@dataclass(frozen=True, kw_only=True)
class StairEvaluationProtocol:
  """One immutable regular-stair rollout matrix."""

  suite: str
  terrain: str
  heights_m: tuple[float, ...]
  risers_per_trial: int
  num_envs_per_height: int
  repeats: int
  stable_steps: int
  minimum_successes: int
  primary_height_m: float
  profile: str = "formal"
  evidence_eligible: bool = True

  @property
  def trials_per_height(self) -> int:
    return self.num_envs_per_height * self.repeats

  def to_dict(self) -> dict[str, object]:
    payload = asdict(self)
    payload["heights_m"] = list(self.heights_m)
    payload["trials_per_height"] = self.trials_per_height
    return payload


SINGLE_RISER_PROTOCOL = StairEvaluationProtocol(
  suite="single-riser",
  terrain="regular_uniform_single_riser",
  heights_m=FORMAL_HEIGHTS_M,
  risers_per_trial=1,
  num_envs_per_height=FORMAL_ENVS_PER_HEIGHT,
  repeats=FORMAL_REPEATS,
  stable_steps=FORMAL_STABLE_STEPS,
  minimum_successes=FORMAL_MIN_SUCCESSES,
  primary_height_m=PRIMARY_HEIGHT_M,
)
CONTINUOUS_STAIRS_PROTOCOL = StairEvaluationProtocol(
  suite="continuous-stairs",
  terrain="regular_uniform_three_riser_stairs",
  heights_m=FORMAL_HEIGHTS_M,
  risers_per_trial=CONTINUOUS_RISERS,
  num_envs_per_height=FORMAL_ENVS_PER_HEIGHT,
  repeats=FORMAL_REPEATS,
  stable_steps=FORMAL_STABLE_STEPS,
  minimum_successes=FORMAL_MIN_SUCCESSES,
  primary_height_m=PRIMARY_HEIGHT_M,
)
STAIR_PROTOCOLS = {
  SINGLE_RISER_PROTOCOL.suite: SINGLE_RISER_PROTOCOL,
  CONTINUOUS_STAIRS_PROTOCOL.suite: CONTINUOUS_STAIRS_PROTOCOL,
}
RETENTION_SUITE = "retention-gates"
EVALUATION_SUITES = (*STAIR_PROTOCOLS, RETENTION_SUITE)

# Reuse StairCamp's mature row schema for the cheap K=3 first-riser screen.
K3_SCREEN_PROTOCOL = stair_camp.DomainProtocol(
  domain="stairs",
  profile="screen",
  terrain="regular_uniform_single_riser",
  cell_key="height_m",
  cells=(PRIMARY_HEIGHT_M,),
  num_envs_per_cell=K3_ENVS,
  repeats=K3_REPEATS,
  settle_steps=0,
  drive_steps=None,
  stable_steps=FORMAL_STABLE_STEPS,
  success_rate_limit=K3_MIN_SUCCESSES / K3_ENVS,
  travel_distance_m=None,
  evidence_eligible=False,
)


@dataclass(frozen=True, kw_only=True)
class AblationDescriptor:
  """Exact evaluation-time control manipulation."""

  name: str
  interpretation: str
  force_stair_request_false: bool = False
  disable_feedforward: bool = False
  zero_action_indices: tuple[int, ...] = ()
  primary_evidence_eligible: bool = False

  def to_dict(self) -> dict[str, object]:
    payload = asdict(self)
    payload["zero_action_indices"] = list(self.zero_action_indices)
    return payload


ROLL_ONLY_ABLATION = AblationDescriptor(
  name="roll-only",
  force_stair_request_false=True,
  disable_feedforward=True,
  interpretation=(
    "Force the v3 stair request false so the FSM stays IDLE; retain the "
    "Stage5-compatible policy path and remove all stair feedforward."
  ),
)
FEEDFORWARD_ONLY_ABLATION = AblationDescriptor(
  name="feedforward-only",
  zero_action_indices=tuple(range(ACTION_WIDTH)),
  interpretation="Run the frozen maneuver with all six PPO residual heads zeroed.",
)
POLICY_ONLY_ABLATION = AblationDescriptor(
  name="policy-only",
  disable_feedforward=True,
  interpretation="Run the v3 policy feedback with stair feedforward zeroed.",
)
FULL_ABLATION = AblationDescriptor(
  name="full",
  interpretation="Run the selected v3 checkpoint and frozen maneuver unchanged.",
  primary_evidence_eligible=True,
)
LEG_PPO_OFF_ABLATION = AblationDescriptor(
  name="leg-PPO-off",
  zero_action_indices=(2, 3, 4, 5),
  interpretation="Zero only the four learned leg-feedback heads.",
)
WHEEL_PPO_OFF_ABLATION = AblationDescriptor(
  name="wheel-PPO-off",
  zero_action_indices=(0, 1),
  interpretation="Zero only the two learned wheel-feedback heads.",
)
ABLATION_ORDER = (
  "roll-only",
  "feedforward-only",
  "policy-only",
  "full",
  "leg-PPO-off",
  "wheel-PPO-off",
)
ABLATION_DESCRIPTORS = {
  descriptor.name: descriptor
  for descriptor in (
    ROLL_ONLY_ABLATION,
    FEEDFORWARD_ONLY_ABLATION,
    POLICY_ONLY_ABLATION,
    FULL_ABLATION,
    LEG_PPO_OFF_ABLATION,
    WHEEL_PPO_OFF_ABLATION,
  )
}


@dataclass(frozen=True, kw_only=True)
class CheckpointExpectation:
  """Launcher-owned exact values needed to bind a formal request."""

  git_sha: str | None = None
  contract_sha256: str | None = None
  artifact_bindings: Mapping[str, str] | None = None
  maneuver_sha256: str | None = None
  source_stage5_checkpoint_sha256: str | None = None
  source_stage5_gate_sha256: str | None = None
  completed_updates: int | None = None

  def require_complete(self) -> None:
    required = {
      "git_sha": self.git_sha,
      "contract_sha256": self.contract_sha256,
      "artifact_bindings": self.artifact_bindings,
      "maneuver_sha256": self.maneuver_sha256,
      "source_stage5_checkpoint_sha256": self.source_stage5_checkpoint_sha256,
      "source_stage5_gate_sha256": self.source_stage5_gate_sha256,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
      raise ValueError(
        "Formal StairDynamic request is missing exact checkpoint expectations: "
        + ", ".join(missing)
      )



def checkpoint_expectation_from_mapping(
  value: Mapping[str, object],
) -> CheckpointExpectation:
  expected = {
    "git_sha",
    "contract_sha256",
    "artifact_bindings",
    "maneuver_sha256",
    "source_stage5_checkpoint_sha256",
    "source_stage5_gate_sha256",
    "completed_updates",
  }
  if set(value) != expected:
    raise ValueError("Checkpoint expectation schema drifted.")
  completed = value.get("completed_updates")
  if completed is not None:
    completed = _integer(completed, name="completed_updates")
  result = CheckpointExpectation(
    git_sha=_git_sha(value.get("git_sha"), name="git_sha"),
    contract_sha256=_sha256(
      value.get("contract_sha256"), name="contract_sha256"
    ),
    artifact_bindings=_normalize_artifacts(value.get("artifact_bindings")),
    maneuver_sha256=_sha256(
      value.get("maneuver_sha256"), name="maneuver_sha256"
    ),
    source_stage5_checkpoint_sha256=_sha256(
      value.get("source_stage5_checkpoint_sha256"),
      name="source_stage5_checkpoint_sha256",
    ),
    source_stage5_gate_sha256=_sha256(
      value.get("source_stage5_gate_sha256"),
      name="source_stage5_gate_sha256",
    ),
    completed_updates=completed,
  )
  result.require_complete()
  return result


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
  if not isinstance(value, Mapping):
    raise ValueError(f"{name} must be a JSON object.")
  return value


def _sequence(value: object, *, name: str) -> Sequence[object]:
  if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
    raise ValueError(f"{name} must be a JSON array.")
  return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
  if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
    raise ValueError(f"{name} must be an integer >= {minimum}.")
  return int(value)


def _finite(value: object, *, name: str, minimum: float | None = None) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"{name} must be numeric.")
  result = float(value)
  if not math.isfinite(result):
    raise ValueError(f"{name} must be finite.")
  if minimum is not None and result < minimum:
    raise ValueError(f"{name} must be >= {minimum}.")
  return result


def _boolean(value: object, *, name: str) -> bool:
  if not isinstance(value, bool):
    raise ValueError(f"{name} must be boolean.")
  return value


def _sha256(value: object, *, name: str) -> str:
  if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
    raise ValueError(f"{name} must be a lowercase SHA256 digest.")
  return value


def _git_sha(value: object, *, name: str) -> str:
  if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
    raise ValueError(f"{name} must be a full lowercase Git SHA.")
  return value


def _canonical_sha256(payload: Mapping[str, object]) -> str:
  stair_camp._validate_json_value(payload)
  encoded = json.dumps(
    dict(payload),
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def _match_height(value: object, heights: Sequence[float], *, name: str) -> float:
  actual = _finite(value, name=name)
  matches = [height for height in heights if math.isclose(actual, height, abs_tol=1e-12)]
  if len(matches) != 1:
    raise ValueError(f"{name}={actual} is outside the registered height grid.")
  return float(matches[0])


def resolve_ablation(name: str) -> AblationDescriptor:
  try:
    return ABLATION_DESCRIPTORS[name]
  except KeyError as exc:
    raise ValueError(f"Unknown StairDynamic ablation: {name!r}.") from exc


def protocol_for(suite: str) -> StairEvaluationProtocol:
  try:
    return STAIR_PROTOCOLS[suite]
  except KeyError as exc:
    raise ValueError(f"Unknown StairDynamic stair suite: {suite!r}.") from exc


def _protocol_binding_payload() -> dict[str, object]:
  return {
    "schema_version": EVALUATOR_SCHEMA_VERSION,
    "task": DYNAMIC_STAIR_TASK_ID,
    "evaluation_seed": REGISTERED_EVALUATION_SEED,
    "single_seed_result": "provisional",
    "protocols": {
      name: protocol.to_dict() for name, protocol in STAIR_PROTOCOLS.items()
    },
    "retention_gate_bindings": {
      name: binding.to_dict() for name, binding in GATE_BINDINGS.items()
    },
    "trial_contract": {
      "traversal_modes": list(TRAVERSAL_MODES),
      "lead_sides": list(LEAD_SIDES),
      "phase_names": list(PHASE_NAMES),
      "feedforward_limit_rad": DYNAMIC_STAIR_FEEDFORWARD_LIMIT_RAD,
      "ppo_leg_limit_rad": DYNAMIC_STAIR_PPO_LEG_SCALE_RAD,
      "stable_steps": FORMAL_STABLE_STEPS,
    },
    "ablations": {
      name: ABLATION_DESCRIPTORS[name].to_dict() for name in ABLATION_ORDER
    },
  }


EVALUATION_PROTOCOL_SHA256 = _canonical_sha256(_protocol_binding_payload())


def _normalize_artifacts(value: object) -> dict[str, str]:
  artifacts = _mapping(value, name="artifact_bindings")
  if set(artifacts) != set(DYNAMIC_ARTIFACT_BINDING_NAMES):
    raise ValueError("StairDynamic checkpoint artifact binding schema drifted.")
  return {
    name: _sha256(artifacts[name], name=f"artifact_bindings.{name}")
    for name in DYNAMIC_ARTIFACT_BINDING_NAMES
  }



def _runtime_binding_from_expectation(
  expectation: CheckpointExpectation,
) -> dict[str, object]:
  """Build the honest zero-update runtime binding for a Stage5 migration."""

  expectation.require_complete()
  if expectation.completed_updates != 0:
    raise ValueError("Zero-update evaluation requires completed_updates=0.")
  return {
    "task": DYNAMIC_STAIR_TASK_ID,
    "training_seed": REGISTERED_TRAINING_SEED,
    "git_sha": _git_sha(expectation.git_sha, name="git_sha"),
    "contract_sha256": _sha256(
      expectation.contract_sha256, name="contract_sha256"
    ),
    "artifact_bindings": _normalize_artifacts(expectation.artifact_bindings),
    "action_scales": [float(value) for value in DYNAMIC_STAIR_ACTION_SCALES],
    "maneuver_sha256": _sha256(
      expectation.maneuver_sha256, name="maneuver_sha256"
    ),
    "source_stage5_checkpoint_sha256": _sha256(
      expectation.source_stage5_checkpoint_sha256,
      name="source_stage5_checkpoint_sha256",
    ),
    "source_stage5_gate_sha256": _sha256(
      expectation.source_stage5_gate_sha256,
      name="source_stage5_gate_sha256",
    ),
    "stage5_prefix_preserved_and_new_columns_zero": True,
    "completed_updates": 0,
  }


def validate_zero_update_runtime_binding(
  value: Mapping[str, object],
  *,
  expectation: CheckpointExpectation | None = None,
) -> dict[str, object]:
  expected_fields = {
    "task",
    "training_seed",
    "git_sha",
    "contract_sha256",
    "artifact_bindings",
    "action_scales",
    "maneuver_sha256",
    "source_stage5_checkpoint_sha256",
    "source_stage5_gate_sha256",
    "stage5_prefix_preserved_and_new_columns_zero",
    "completed_updates",
  }
  if set(value) != expected_fields:
    raise ValueError("StairDynamic zero-update runtime binding schema drifted.")
  if value.get("task") != DYNAMIC_STAIR_TASK_ID:
    raise ValueError("Zero-update runtime task drifted.")
  if _integer(value.get("training_seed"), name="training_seed") != 1:
    raise ValueError("Zero-update runtime seed must remain 1.")
  scales = [
    _finite(item, name=f"action_scales[{index}]")
    for index, item in enumerate(
      _sequence(value.get("action_scales"), name="action_scales")
    )
  ]
  if scales != [float(item) for item in DYNAMIC_STAIR_ACTION_SCALES]:
    raise ValueError("Zero-update runtime action scales drifted.")
  artifacts = _normalize_artifacts(value.get("artifact_bindings"))
  maneuver = _sha256(value.get("maneuver_sha256"), name="maneuver_sha256")
  if artifacts["dynamic_maneuver_hash"] != maneuver:
    raise ValueError("Zero-update runtime maneuver binding disagrees.")
  normalized = {
    "task": DYNAMIC_STAIR_TASK_ID,
    "training_seed": 1,
    "git_sha": _git_sha(value.get("git_sha"), name="git_sha"),
    "contract_sha256": _sha256(
      value.get("contract_sha256"), name="contract_sha256"
    ),
    "artifact_bindings": artifacts,
    "action_scales": scales,
    "maneuver_sha256": maneuver,
    "source_stage5_checkpoint_sha256": _sha256(
      value.get("source_stage5_checkpoint_sha256"),
      name="source_stage5_checkpoint_sha256",
    ),
    "source_stage5_gate_sha256": _sha256(
      value.get("source_stage5_gate_sha256"),
      name="source_stage5_gate_sha256",
    ),
    "stage5_prefix_preserved_and_new_columns_zero": True,
    "completed_updates": 0,
  }
  if value.get("stage5_prefix_preserved_and_new_columns_zero") is not True:
    raise ValueError("Zero-update migration network attestation is missing.")
  if value.get("completed_updates") != 0:
    raise ValueError("Zero-update migration must not claim a training update.")
  if expectation is not None and normalized != _runtime_binding_from_expectation(
    expectation
  ):
    raise ValueError("Zero-update runtime binding differs from launcher expectations.")
  return normalized


def _normalize_migration_info(value: object) -> dict[str, object]:
  migration = _mapping(value, name="migration")
  expected_fields = {
    "source_checkpoint_sha256",
    "source_gate_sha256",
    "source_task",
    "source_seed",
    "source_completed_updates",
    "target_task",
    "source_actor_width",
    "target_actor_width",
    "source_critic_width",
    "target_critic_width",
    "actor_first_layer",
    "critic_first_layer",
    "std_key",
    "source_action_std",
    "target_action_std",
    "collapsed_std_threshold",
    "collapsed_active_indices",
    "collapsed_active_actions",
    "reset_collapsed_active_std",
    "created_at",
  }
  if set(migration) != expected_fields:
    raise ValueError("StairDynamic migration provenance schema drifted.")
  fixed = {
    "source_task": "HopperTrex-Hybrid-v2-Stage5",
    "source_seed": 1,
    "source_completed_updates": 100,
    "target_task": DYNAMIC_STAIR_TASK_ID,
    "source_actor_width": 34,
    "target_actor_width": DYNAMIC_STAIR_ACTOR_WIDTH,
    "source_critic_width": 34,
    "target_critic_width": DYNAMIC_STAIR_CRITIC_WIDTH,
  }
  for name, expected in fixed.items():
    if migration.get(name) != expected:
      raise ValueError(f"StairDynamic migration {name} drifted.")
  actor_layer = migration.get("actor_first_layer")
  critic_layer = migration.get("critic_first_layer")
  if not isinstance(actor_layer, str) or not actor_layer:
    raise ValueError("Migration actor first-layer key is invalid.")
  if not isinstance(critic_layer, str) or not critic_layer:
    raise ValueError("Migration critic first-layer key is invalid.")
  std_key = migration.get("std_key")
  if std_key not in ("distribution.std_param", "distribution.log_std_param"):
    raise ValueError("Migration std parameter key is unsupported.")
  threshold = _finite(
    migration.get("collapsed_std_threshold"), name="collapsed_std_threshold"
  )
  if threshold <= 0.0:
    raise ValueError("Migration collapsed-std threshold must be positive.")
  source_std = [
    _finite(item, name=f"source_action_std[{index}]")
    for index, item in enumerate(
      _sequence(migration.get("source_action_std"), name="source_action_std")
    )
  ]
  target_std = [
    _finite(item, name=f"target_action_std[{index}]")
    for index, item in enumerate(
      _sequence(migration.get("target_action_std"), name="target_action_std")
    )
  ]
  if len(source_std) != ACTION_WIDTH or len(target_std) != ACTION_WIDTH:
    raise ValueError("Migration action std vectors must have six entries.")
  if any(value <= 0.0 for value in source_std + target_std):
    raise ValueError("Migration action std values must be positive.")
  raw_indices = _sequence(
    migration.get("collapsed_active_indices"), name="collapsed_active_indices"
  )
  indices = [
    _integer(item, name=f"collapsed_active_indices[{index}]")
    for index, item in enumerate(raw_indices)
  ]
  if len(set(indices)) != len(indices) or any(index >= ACTION_WIDTH for index in indices):
    raise ValueError("Migration collapsed action indices are invalid.")
  expected_indices = [
    index for index, value in enumerate(source_std) if value < threshold
  ]
  if indices != expected_indices:
    raise ValueError("Migration collapsed action index audit drifted.")
  actions = list(
    _sequence(
      migration.get("collapsed_active_actions"), name="collapsed_active_actions"
    )
  )
  if actions != [HYBRID_ACTION_NAMES[index] for index in indices]:
    raise ValueError("Migration collapsed action name audit drifted.")
  reset = migration.get("reset_collapsed_active_std")
  if not isinstance(reset, bool) or (indices and not reset):
    raise ValueError("Collapsed StairDynamic migration std was not explicitly reset.")
  for index, (source, target) in enumerate(zip(source_std, target_std, strict=True)):
    expected = float(HYBRID_ACTION_STD[index]) if index in indices else source
    if not math.isclose(target, expected, abs_tol=1e-7):
      raise ValueError("Migration target action std audit drifted.")
  created_at = migration.get("created_at")
  if not isinstance(created_at, str) or not created_at.strip():
    raise ValueError("Migration created_at is missing.")
  normalized = dict(migration)
  normalized.update(
    {
      "source_checkpoint_sha256": _sha256(
        migration.get("source_checkpoint_sha256"),
        name="source_checkpoint_sha256",
      ),
      "source_gate_sha256": _sha256(
        migration.get("source_gate_sha256"), name="source_gate_sha256"
      ),
      "source_action_std": source_std,
      "target_action_std": target_std,
      "collapsed_std_threshold": threshold,
      "collapsed_active_indices": indices,
      "collapsed_active_actions": actions,
    }
  )
  return normalized


def validate_training_info(
  training: Mapping[str, object],
  *,
  expectation: CheckpointExpectation | None = None,
) -> dict[str, object]:
  """Validate the runner's StairDynamic training provenance."""

  expected_fields = {
    "schema_version",
    "task",
    "training_seed",
    "git_sha",
    "contract_sha256",
    "artifact_bindings",
    "action_scales",
    "maneuver_sha256",
    "source_stage5_checkpoint_sha256",
    "source_stage5_gate_sha256",
    "stage5_prefix_preserved_and_new_columns_zero",
    "completed_updates",
  }
  if set(training) != expected_fields:
    raise ValueError("StairDynamic checkpoint training provenance schema drifted.")
  if training.get("schema_version") != DYNAMIC_STAIR_CONTRACT_SCHEMA_VERSION:
    raise ValueError("StairDynamic checkpoint schema version is unsupported.")
  if training.get("task") != DYNAMIC_STAIR_TASK_ID:
    raise ValueError("Checkpoint task is not HopperTrex Hybrid-v3 StairDynamic.")
  seed = _integer(training.get("training_seed"), name="training_seed", minimum=1)
  if seed != REGISTERED_TRAINING_SEED:
    raise ValueError("StairDynamic currently accepts training seed 1 only.")
  git_sha = _git_sha(training.get("git_sha"), name="git_sha")
  contract_sha = _sha256(training.get("contract_sha256"), name="contract_sha256")
  artifacts = _normalize_artifacts(training.get("artifact_bindings"))
  scales = tuple(
    _finite(value, name=f"action_scales[{index}]")
    for index, value in enumerate(
      _sequence(training.get("action_scales"), name="action_scales")
    )
  )
  expected_scales = tuple(float(value) for value in DYNAMIC_STAIR_ACTION_SCALES)
  if len(scales) != ACTION_WIDTH or any(
    not math.isclose(actual, expected, abs_tol=1e-12)
    for actual, expected in zip(scales, expected_scales, strict=True)
  ):
    raise ValueError("StairDynamic checkpoint action scales drifted.")
  maneuver_sha = _sha256(training.get("maneuver_sha256"), name="maneuver_sha256")
  if artifacts["dynamic_maneuver_hash"] != maneuver_sha:
    raise ValueError("StairDynamic maneuver and artifact hashes disagree.")
  source_checkpoint_sha = _sha256(
    training.get("source_stage5_checkpoint_sha256"),
    name="source_stage5_checkpoint_sha256",
  )
  source_gate_sha = _sha256(
    training.get("source_stage5_gate_sha256"),
    name="source_stage5_gate_sha256",
  )
  if training.get("stage5_prefix_preserved_and_new_columns_zero") is not True:
    raise ValueError("StairDynamic Stage5-prefix migration attestation is missing.")
  updates = _integer(training.get("completed_updates"), name="completed_updates", minimum=1)
  if updates > DYNAMIC_STAIR_EXTENSION_TOTAL_UPDATES:
    raise ValueError("StairDynamic checkpoint exceeds the registered 500-update cap.")

  normalized: dict[str, object] = {
    "schema_version": DYNAMIC_STAIR_CONTRACT_SCHEMA_VERSION,
    "task": DYNAMIC_STAIR_TASK_ID,
    "training_seed": seed,
    "git_sha": git_sha,
    "contract_sha256": contract_sha,
    "artifact_bindings": artifacts,
    "action_scales": list(scales),
    "maneuver_sha256": maneuver_sha,
    "source_stage5_checkpoint_sha256": source_checkpoint_sha,
    "source_stage5_gate_sha256": source_gate_sha,
    "stage5_prefix_preserved_and_new_columns_zero": True,
    "completed_updates": updates,
  }
  if expectation is not None:
    comparisons = {
      "git_sha": expectation.git_sha,
      "contract_sha256": expectation.contract_sha256,
      "artifact_bindings": (
        None
        if expectation.artifact_bindings is None
        else _normalize_artifacts(expectation.artifact_bindings)
      ),
      "maneuver_sha256": expectation.maneuver_sha256,
      "source_stage5_checkpoint_sha256": expectation.source_stage5_checkpoint_sha256,
      "source_stage5_gate_sha256": expectation.source_stage5_gate_sha256,
      "completed_updates": expectation.completed_updates,
    }
    for name, expected in comparisons.items():
      if expected is not None and normalized[name] != expected:
        raise ValueError(f"Checkpoint {name} does not match the exact expectation.")
  return normalized



def validate_checkpoint_envelope(
  envelope: Mapping[str, object],
  *,
  expectation: CheckpointExpectation | None = None,
  verify_file: bool = False,
) -> dict[str, object]:
  """Validate one tensor-free trained-v3 checkpoint sidecar."""

  allowed = {
    "schema_version",
    "kind",
    "checkpoint_file",
    "checkpoint_file_sha256",
    "checkpoint_file_verified",
    "checkpoint_iteration",
    "training",
  }
  required = allowed - {"checkpoint_file_verified", "checkpoint_iteration"}
  if not required.issubset(envelope) or set(envelope) - allowed:
    raise ValueError("StairDynamic checkpoint-envelope schema drifted.")
  if envelope.get("schema_version") != EVALUATOR_SCHEMA_VERSION:
    raise ValueError("StairDynamic checkpoint-envelope version is unsupported.")
  if envelope.get("kind") != CHECKPOINT_ENVELOPE_KIND:
    raise ValueError("Input is not a StairDynamic checkpoint envelope.")
  checkpoint_file = envelope.get("checkpoint_file")
  if not isinstance(checkpoint_file, str) or not checkpoint_file.strip():
    raise ValueError("checkpoint_file must be a non-empty path string.")
  checkpoint_sha = _sha256(
    envelope.get("checkpoint_file_sha256"), name="checkpoint_file_sha256"
  )
  training = validate_training_info(
    _mapping(envelope.get("training"), name="training"), expectation=expectation
  )
  iteration: int | None = None
  if "checkpoint_iteration" in envelope:
    iteration = _integer(
      envelope.get("checkpoint_iteration"), name="checkpoint_iteration"
    )
    if iteration + 1 != training["completed_updates"]:
      raise ValueError("Checkpoint iteration does not match completed updates.")
  if verify_file:
    path = Path(checkpoint_file)
    if not path.is_file():
      raise ValueError(f"Checkpoint file does not exist: {path}.")
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha != checkpoint_sha:
      raise ValueError("Checkpoint file SHA256 does not match its envelope.")
  normalized: dict[str, object] = {
    "schema_version": EVALUATOR_SCHEMA_VERSION,
    "kind": CHECKPOINT_ENVELOPE_KIND,
    "checkpoint_file": checkpoint_file,
    "checkpoint_file_sha256": checkpoint_sha,
    "checkpoint_file_verified": bool(verify_file),
    "training": training,
  }
  if iteration is not None:
    normalized["checkpoint_iteration"] = iteration
  return normalized



def validate_migration_checkpoint_envelope(
  envelope: Mapping[str, object],
  *,
  expectation: CheckpointExpectation | None = None,
  verify_file: bool = False,
) -> dict[str, object]:
  """Validate a distinct, honest Stage5-to-v3 zero-update checkpoint envelope."""

  allowed = {
    "schema_version",
    "kind",
    "checkpoint_file",
    "checkpoint_file_sha256",
    "checkpoint_file_verified",
    "checkpoint_iteration",
    "migration",
    "runtime_binding",
  }
  required = allowed - {"checkpoint_file_verified"}
  if not required.issubset(envelope) or set(envelope) - allowed:
    raise ValueError("StairDynamic migration-envelope schema drifted.")
  if (
    envelope.get("schema_version") != EVALUATOR_SCHEMA_VERSION
    or envelope.get("kind") != MIGRATION_CHECKPOINT_ENVELOPE_KIND
  ):
    raise ValueError("Input is not a StairDynamic zero-update migration envelope.")
  checkpoint_file = envelope.get("checkpoint_file")
  if not isinstance(checkpoint_file, str) or not checkpoint_file.strip():
    raise ValueError("checkpoint_file must be a non-empty path string.")
  checkpoint_sha = _sha256(
    envelope.get("checkpoint_file_sha256"), name="checkpoint_file_sha256"
  )
  if _integer(
    envelope.get("checkpoint_iteration"), name="checkpoint_iteration"
  ) != 0:
    raise ValueError("Zero-update migration checkpoint iteration must be zero.")
  migration = _normalize_migration_info(envelope.get("migration"))
  runtime = validate_zero_update_runtime_binding(
    _mapping(envelope.get("runtime_binding"), name="runtime_binding"),
    expectation=expectation,
  )
  if (
    migration["source_checkpoint_sha256"]
    != runtime["source_stage5_checkpoint_sha256"]
    or migration["source_gate_sha256"] != runtime["source_stage5_gate_sha256"]
  ):
    raise ValueError("Zero-update migration source differs from runtime binding.")
  if verify_file:
    path = Path(checkpoint_file)
    if not path.is_file():
      raise ValueError(f"Checkpoint file does not exist: {path}.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != checkpoint_sha:
      raise ValueError("Checkpoint file SHA256 does not match its envelope.")
  return {
    "schema_version": EVALUATOR_SCHEMA_VERSION,
    "kind": MIGRATION_CHECKPOINT_ENVELOPE_KIND,
    "checkpoint_file": checkpoint_file,
    "checkpoint_file_sha256": checkpoint_sha,
    "checkpoint_file_verified": bool(verify_file),
    "checkpoint_iteration": 0,
    "migration": migration,
    "runtime_binding": runtime,
  }


def validate_evaluation_checkpoint_envelope(
  envelope: Mapping[str, object],
  *,
  expectation: CheckpointExpectation | None = None,
  verify_file: bool = False,
) -> dict[str, object]:
  """Dispatch without ever recasting a zero-update migration as trained."""

  if envelope.get("kind") == MIGRATION_CHECKPOINT_ENVELOPE_KIND:
    return validate_migration_checkpoint_envelope(
      envelope, expectation=expectation, verify_file=verify_file
    )
  return validate_checkpoint_envelope(
    envelope, expectation=expectation, verify_file=verify_file
  )


def _validate_loaded_zero_update_network(
  checkpoint: Mapping[str, object],
  migration: Mapping[str, object],
) -> None:
  import torch

  if checkpoint.get("iter") != 0:
    raise ValueError("Zero-update migration checkpoint must have iter=0.")
  infos = _mapping(checkpoint.get("infos"), name="checkpoint infos")
  if DYNAMIC_STAIR_TRAINING_INFO_KEY in infos:
    raise ValueError("Zero-update migration must not contain training provenance.")
  if _normalize_migration_info(infos.get(DYNAMIC_STAIR_MIGRATION_INFO_KEY)) != dict(
    migration
  ):
    raise ValueError("Zero-update migration provenance changed while loading.")
  env_state = _mapping(infos.get("env_state"), name="env_state")
  if env_state.get("common_step_counter") != 0:
    raise ValueError("Zero-update migration must reset common_step_counter.")
  optimizer = _mapping(
    checkpoint.get("optimizer_state_dict"), name="optimizer_state_dict"
  )
  state = _mapping(optimizer.get("state"), name="optimizer_state_dict.state")
  if state:
    raise ValueError("Zero-update migration optimizer state must be empty.")
  for label, state_name, layer_field, source_field, target_field in (
    (
      "actor",
      "actor_state_dict",
      "actor_first_layer",
      "source_actor_width",
      "target_actor_width",
    ),
    (
      "critic",
      "critic_state_dict",
      "critic_first_layer",
      "source_critic_width",
      "target_critic_width",
    ),
  ):
    state_dict = _mapping(checkpoint.get(state_name), name=state_name)
    key = str(migration[layer_field])
    tensor = state_dict.get(key)
    target_width = int(migration[target_field])
    source_width = int(migration[source_field])
    if (
      not isinstance(tensor, torch.Tensor)
      or tensor.ndim != 2
      or tensor.shape[1] != target_width
    ):
      raise ValueError(f"Zero-update {label} first layer has the wrong shape.")
    if torch.count_nonzero(tensor[:, source_width:]).item() != 0:
      raise ValueError(f"Zero-update {label} added observation columns are non-zero.")


def migration_checkpoint_envelope_from_loaded_checkpoint(
  checkpoint_file: str | Path,
  checkpoint: Mapping[str, object],
  *,
  expectation: CheckpointExpectation,
) -> dict[str, object]:
  """Create a verified zero-update envelope without fabricating update 1."""

  path = Path(checkpoint_file)
  if not path.is_file():
    raise ValueError(f"Checkpoint file does not exist: {path}.")
  infos = _mapping(checkpoint.get("infos"), name="checkpoint infos")
  migration = _normalize_migration_info(
    infos.get(DYNAMIC_STAIR_MIGRATION_INFO_KEY)
  )
  _validate_loaded_zero_update_network(checkpoint, migration)
  envelope = {
    "schema_version": EVALUATOR_SCHEMA_VERSION,
    "kind": MIGRATION_CHECKPOINT_ENVELOPE_KIND,
    "checkpoint_file": str(path.resolve()),
    "checkpoint_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "checkpoint_iteration": 0,
    "migration": migration,
    "runtime_binding": _runtime_binding_from_expectation(expectation),
  }
  return validate_migration_checkpoint_envelope(
    envelope, expectation=expectation, verify_file=True
  )


def migration_checkpoint_envelope_from_file(
  checkpoint_file: str | Path,
  *,
  expectation: CheckpointExpectation,
) -> dict[str, object]:
  """Load and verify the dedicated zero-update migration checkpoint."""

  import torch

  path = Path(checkpoint_file)
  checkpoint = torch.load(path, map_location="cpu", weights_only=False)
  if not isinstance(checkpoint, Mapping):
    raise ValueError("StairDynamic migration checkpoint must contain a mapping.")
  return migration_checkpoint_envelope_from_loaded_checkpoint(
    path, checkpoint, expectation=expectation
  )


def checkpoint_envelope_from_loaded_checkpoint(
  checkpoint_file: str | Path,
  checkpoint: Mapping[str, object],
  *,
  expectation: CheckpointExpectation | None = None,
) -> dict[str, object]:
  """Create a verified pure envelope after a caller has loaded a checkpoint."""

  path = Path(checkpoint_file)
  if not path.is_file():
    raise ValueError(f"Checkpoint file does not exist: {path}.")
  infos = _mapping(checkpoint.get("infos"), name="checkpoint infos")
  training = _mapping(
    infos.get(DYNAMIC_STAIR_TRAINING_INFO_KEY),
    name=f"checkpoint infos.{DYNAMIC_STAIR_TRAINING_INFO_KEY}",
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
  return validate_checkpoint_envelope(
    envelope, expectation=expectation, verify_file=True
  )


def checkpoint_envelope_from_file(
  checkpoint_file: str | Path,
  *,
  expectation: CheckpointExpectation | None = None,
) -> dict[str, object]:
  """Lazily import Torch; importing this module itself remains simulation-free."""

  import torch

  path = Path(checkpoint_file)
  checkpoint = torch.load(path, map_location="cpu", weights_only=False)
  if not isinstance(checkpoint, Mapping):
    raise ValueError("StairDynamic checkpoint must contain a mapping.")
  return checkpoint_envelope_from_loaded_checkpoint(
    path, checkpoint, expectation=expectation
  )


def _protocol_for_request(suite: str) -> dict[str, object]:
  if suite in STAIR_PROTOCOLS:
    return protocol_for(suite).to_dict()
  if suite == RETENTION_SUITE:
    return {
      "suite": RETENTION_SUITE,
      "profile": "formal",
      "terrain": "flat",
      "gate_bindings": {
        name: binding.to_dict() for name, binding in GATE_BINDINGS.items()
      },
    }
  raise ValueError(f"Unknown StairDynamic evaluation suite: {suite!r}.")


def make_evaluation_request(
  *,
  suite: str,
  checkpoint_envelope: Mapping[str, object],
  expectation: CheckpointExpectation,
  ablation: str = "full",
  device: str = "cuda:0",
) -> dict[str, object]:
  """Build a signed request; formal launchers must supply every exact binding."""

  expectation.require_complete()
  checkpoint = validate_evaluation_checkpoint_envelope(
    checkpoint_envelope, expectation=expectation, verify_file=True
  )
  descriptor = resolve_ablation(ablation)
  if suite == RETENTION_SUITE and descriptor.name != "full":
    raise ValueError("Retention gates are reused unchanged and accept only full control.")
  protocol = _protocol_for_request(suite)
  if not isinstance(device, str) or not device.strip():
    raise ValueError("device must be a non-empty string.")
  payload: dict[str, object] = {
    "schema_version": EVALUATOR_SCHEMA_VERSION,
    "kind": EVALUATION_REQUEST_KIND,
    "task": DYNAMIC_STAIR_TASK_ID,
    "suite": suite,
    "profile": "formal",
    "evaluation_seed": REGISTERED_EVALUATION_SEED,
    "single_seed_status": "provisional",
    "promotion_claim_eligible": False,
    "evaluation_protocol_sha256": EVALUATION_PROTOCOL_SHA256,
    "protocol": protocol,
    "checkpoint": checkpoint,
    "policy_interface": {
      "actor_observation_width": DYNAMIC_STAIR_ACTOR_WIDTH,
      "critic_observation_width": DYNAMIC_STAIR_CRITIC_WIDTH,
      "action_width": ACTION_WIDTH,
      "stage5_34_observation_adapter_forbidden": True,
    },
    "gate_bindings": {
      name: binding.to_dict() for name, binding in GATE_BINDINGS.items()
    },
    "ablation": descriptor.to_dict(),
    "device": device,
  }
  payload["request_sha256"] = _canonical_sha256(payload)
  return payload


def validate_evaluation_request(
  request: Mapping[str, object],
  *,
  expectation: CheckpointExpectation | None = None,
  verify_checkpoint_file: bool = True,
) -> dict[str, object]:
  """Fail closed on any checkpoint, protocol, interface, or ablation drift."""

  expected_fields = {
    "schema_version",
    "kind",
    "task",
    "suite",
    "profile",
    "evaluation_seed",
    "single_seed_status",
    "promotion_claim_eligible",
    "evaluation_protocol_sha256",
    "protocol",
    "checkpoint",
    "policy_interface",
    "gate_bindings",
    "ablation",
    "device",
    "request_sha256",
  }
  if set(request) != expected_fields:
    raise ValueError("StairDynamic evaluation request schema drifted.")
  if (
    request.get("schema_version") != EVALUATOR_SCHEMA_VERSION
    or request.get("kind") != EVALUATION_REQUEST_KIND
    or request.get("task") != DYNAMIC_STAIR_TASK_ID
  ):
    raise ValueError("StairDynamic evaluation request identity drifted.")
  suite = request.get("suite")
  if not isinstance(suite, str) or suite not in EVALUATION_SUITES:
    raise ValueError("StairDynamic evaluation suite is unsupported.")
  if (
    request.get("profile") != "formal"
    or request.get("evaluation_seed") != REGISTERED_EVALUATION_SEED
    or request.get("single_seed_status") != "provisional"
    or request.get("promotion_claim_eligible") is not False
  ):
    raise ValueError("StairDynamic single-seed formal status drifted.")
  if request.get("evaluation_protocol_sha256") != EVALUATION_PROTOCOL_SHA256:
    raise ValueError("StairDynamic evaluation protocol digest drifted.")
  if request.get("protocol") != _protocol_for_request(suite):
    raise ValueError("StairDynamic suite protocol drifted.")
  expected_gates = {
    name: binding.to_dict() for name, binding in GATE_BINDINGS.items()
  }
  if request.get("gate_bindings") != expected_gates:
    raise ValueError("StairDynamic reused gate bindings drifted.")
  interface = _mapping(request.get("policy_interface"), name="policy_interface")
  if interface != {
    "actor_observation_width": DYNAMIC_STAIR_ACTOR_WIDTH,
    "critic_observation_width": DYNAMIC_STAIR_CRITIC_WIDTH,
    "action_width": ACTION_WIDTH,
    "stage5_34_observation_adapter_forbidden": True,
  }:
    raise ValueError("StairDynamic policy interface must remain 52/56/6.")
  descriptor_name = _mapping(request.get("ablation"), name="ablation").get("name")
  if not isinstance(descriptor_name, str):
    raise ValueError("StairDynamic ablation name is missing.")
  descriptor = resolve_ablation(descriptor_name)
  if request.get("ablation") != descriptor.to_dict():
    raise ValueError("StairDynamic ablation descriptor drifted.")
  if suite == RETENTION_SUITE and descriptor.name != "full":
    raise ValueError("Retention gates accept only the unchanged full controller.")
  checkpoint = validate_evaluation_checkpoint_envelope(
    _mapping(request.get("checkpoint"), name="checkpoint"),
    expectation=expectation,
    verify_file=verify_checkpoint_file,
  )
  if request.get("checkpoint") != checkpoint:
    raise ValueError("StairDynamic request checkpoint envelope is not canonical.")
  device = request.get("device")
  if not isinstance(device, str) or not device.strip():
    raise ValueError("StairDynamic request device must be non-empty.")
  without_hash = dict(request)
  supplied_hash = without_hash.pop("request_sha256")
  if supplied_hash != _canonical_sha256(without_hash):
    raise ValueError("StairDynamic request SHA256 does not match its contents.")
  stair_camp._validate_json_value(request)
  return dict(request)


def _optional_time(value: object, *, name: str) -> float | None:
  if value is None:
    return None
  return _finite(value, name=name, minimum=0.0)


def _normalize_trial(
  value: object,
  *,
  index: int,
  protocol: StairEvaluationProtocol,
  ablation: AblationDescriptor,
) -> dict[str, object]:
  trial = _mapping(value, name=f"trials[{index}]")
  expected_fields = {
    "height_m",
    "env_index",
    "repeat_index",
    "success",
    "traversal_mode",
    "lift_mode",
    "lead_side",
    "left_trigger_time_s",
    "right_trigger_time_s",
    "phase_durations_s",
    "wheel_ppo_rms",
    "wheel_ppo_max_abs",
    "leg_ppo_rms",
    "leg_ppo_max_abs",
    "feedforward_max_abs_rad",
    "peak_abs_pitch_rad",
    "peak_abs_roll_rad",
    "steps_completed",
    "step_recovery_times_s",
    "stable_steps",
    "terminated",
    "non_wheel_contact",
    "abort_reason",
  }
  if set(trial) != expected_fields:
    raise ValueError(f"trials[{index}] schema drifted.")
  height = _match_height(
    trial.get("height_m"), protocol.heights_m, name=f"trials[{index}].height_m"
  )
  env_index = _integer(trial.get("env_index"), name=f"trials[{index}].env_index")
  repeat_index = _integer(
    trial.get("repeat_index"), name=f"trials[{index}].repeat_index"
  )
  if env_index >= protocol.num_envs_per_height or repeat_index >= protocol.repeats:
    raise ValueError(f"trials[{index}] env/repeat index is outside the protocol.")
  success = _boolean(trial.get("success"), name=f"trials[{index}].success")
  mode = trial.get("traversal_mode")
  lift_mode = trial.get("lift_mode")
  lead = trial.get("lead_side")
  if (
    mode not in TRAVERSAL_MODES
    or lift_mode not in LIFT_MODES
    or lead not in LEAD_SIDES
  ):
    raise ValueError(
      f"trials[{index}] traversal mode, lift mode, or lead side is invalid."
    )
  left_trigger = _optional_time(
    trial.get("left_trigger_time_s"), name=f"trials[{index}].left_trigger_time_s"
  )
  right_trigger = _optional_time(
    trial.get("right_trigger_time_s"), name=f"trials[{index}].right_trigger_time_s"
  )
  phases = _mapping(trial.get("phase_durations_s"), name="phase_durations_s")
  if set(phases) != set(PHASE_NAMES):
    raise ValueError(f"trials[{index}] phase duration schema drifted.")
  normalized_phases = {
    phase: _finite(phases[phase], name=f"phase_durations_s.{phase}", minimum=0.0)
    for phase in PHASE_NAMES
  }
  wheel_rms = _finite(
    trial.get("wheel_ppo_rms"), name=f"trials[{index}].wheel_ppo_rms", minimum=0.0
  )
  wheel_max = _finite(
    trial.get("wheel_ppo_max_abs"),
    name=f"trials[{index}].wheel_ppo_max_abs",
    minimum=0.0,
  )
  leg_rms = _finite(
    trial.get("leg_ppo_rms"), name=f"trials[{index}].leg_ppo_rms", minimum=0.0
  )
  leg_max = _finite(
    trial.get("leg_ppo_max_abs"),
    name=f"trials[{index}].leg_ppo_max_abs",
    minimum=0.0,
  )
  if wheel_rms > wheel_max + 1e-12 or leg_rms > leg_max + 1e-12:
    raise ValueError("PPO RMS cannot exceed the corresponding absolute maximum.")
  if wheel_max > max(DYNAMIC_STAIR_ACTION_SCALES[:2]) + 1e-12:
    raise ValueError("Wheel PPO feedback exceeds the registered action authority.")
  if leg_max > DYNAMIC_STAIR_PPO_LEG_SCALE_RAD + 1e-12:
    raise ValueError("Leg PPO feedback exceeds 0.035 rad.")
  feedforward = _finite(
    trial.get("feedforward_max_abs_rad"),
    name=f"trials[{index}].feedforward_max_abs_rad",
    minimum=0.0,
  )
  if feedforward > DYNAMIC_STAIR_FEEDFORWARD_LIMIT_RAD + 1e-12:
    raise ValueError("Stair feedforward exceeds 0.070 rad.")
  peak_pitch = _finite(
    trial.get("peak_abs_pitch_rad"),
    name=f"trials[{index}].peak_abs_pitch_rad",
    minimum=0.0,
  )
  peak_roll = _finite(
    trial.get("peak_abs_roll_rad"),
    name=f"trials[{index}].peak_abs_roll_rad",
    minimum=0.0,
  )
  steps_completed = _integer(
    trial.get("steps_completed"), name=f"trials[{index}].steps_completed"
  )
  if steps_completed > protocol.risers_per_trial:
    raise ValueError("steps_completed exceeds the configured riser count.")
  recoveries = tuple(
    _finite(value, name=f"step_recovery_times_s[{recovery}]", minimum=0.0)
    for recovery, value in enumerate(
      _sequence(trial.get("step_recovery_times_s"), name="step_recovery_times_s")
    )
  )
  if len(recoveries) != steps_completed:
    raise ValueError("Recovery-time count must equal steps_completed.")
  stable_steps = _integer(
    trial.get("stable_steps"), name=f"trials[{index}].stable_steps"
  )
  terminated = _boolean(
    trial.get("terminated"), name=f"trials[{index}].terminated"
  )
  non_wheel_contact = _boolean(
    trial.get("non_wheel_contact"), name=f"trials[{index}].non_wheel_contact"
  )
  abort_reason = trial.get("abort_reason")
  if abort_reason is not None and (
    not isinstance(abort_reason, str) or not abort_reason.strip()
  ):
    raise ValueError("abort_reason must be null or a non-empty string.")

  if success and (
    mode == "ABORT"
    or terminated
    or non_wheel_contact
    or steps_completed != protocol.risers_per_trial
    or stable_steps < protocol.stable_steps
  ):
    raise ValueError("Successful trial contradicts traversal/safety/stability evidence.")
  if (terminated or non_wheel_contact) and mode != "ABORT":
    raise ValueError("Dangerous trials must be reported as ABORT.")
  if mode == "ABORT" and abort_reason is None:
    raise ValueError("ABORT trials must report an abort_reason.")
  if mode != "ABORT" and abort_reason is not None:
    raise ValueError("Non-ABORT trials cannot report an abort_reason.")
  if mode == "ROLL" and (
    lead != "NONE" or left_trigger is not None or right_trigger is not None
  ):
    raise ValueError("ROLL means no loaded-contact lift trigger or lead side.")
  if mode == "DYNAMIC":
    if lead == "NONE":
      raise ValueError("DYNAMIC trials require a selected lead side.")
    lead_time, trail_time = (
      (left_trigger, right_trigger) if lead == "LEFT" else (right_trigger, left_trigger)
    )
    if lift_mode == DynamicLiftMode.ALTERNATING.value:
      if lead_time is None or trail_time is None:
        raise ValueError(
          "Alternating DYNAMIC trials require both observed trigger times."
        )
    elif lead_time is None:
      raise ValueError(
        "Synchronized DYNAMIC trials require the observed lead trigger time."
      )
    if trail_time is not None and lead_time > trail_time + 1e-12:
      raise ValueError(
        "DYNAMIC trail trigger cannot precede the selected lead trigger."
      )
  if ablation.force_stair_request_false and mode == "DYNAMIC":
    raise ValueError("roll-only cannot report a DYNAMIC traversal.")
  if ablation.disable_feedforward and feedforward != 0.0:
    raise ValueError(f"{ablation.name} requires zero stair feedforward.")
  zero_indices = set(ablation.zero_action_indices)
  if {0, 1}.issubset(zero_indices) and (wheel_rms != 0.0 or wheel_max != 0.0):
    raise ValueError(f"{ablation.name} requires zero wheel PPO feedback.")
  if {2, 3, 4, 5}.issubset(zero_indices) and (leg_rms != 0.0 or leg_max != 0.0):
    raise ValueError(f"{ablation.name} requires zero leg PPO feedback.")

  return {
    "height_m": height,
    "env_index": env_index,
    "repeat_index": repeat_index,
    "success": success,
    "traversal_mode": mode,
    "lift_mode": lift_mode,
    "lead_side": lead,
    "left_trigger_time_s": left_trigger,
    "right_trigger_time_s": right_trigger,
    "phase_durations_s": normalized_phases,
    "wheel_ppo_rms": wheel_rms,
    "wheel_ppo_max_abs": wheel_max,
    "leg_ppo_rms": leg_rms,
    "leg_ppo_max_abs": leg_max,
    "feedforward_max_abs_rad": feedforward,
    "peak_abs_pitch_rad": peak_pitch,
    "peak_abs_roll_rad": peak_roll,
    "steps_completed": steps_completed,
    "step_recovery_times_s": list(recoveries),
    "stable_steps": stable_steps,
    "terminated": terminated,
    "non_wheel_contact": non_wheel_contact,
    "abort_reason": abort_reason,
  }



def normalize_trials(
  value: object,
  *,
  protocol: StairEvaluationProtocol,
  ablation: AblationDescriptor,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
  """Validate an exact height/env/repeat matrix and aggregate canonical rows."""

  raw_trials = _sequence(value, name="trials")
  expected_total = len(protocol.heights_m) * protocol.trials_per_height
  if len(raw_trials) != expected_total:
    raise ValueError(f"Stair suite requires exactly {expected_total} trials.")
  by_key: dict[tuple[float, int, int], dict[str, object]] = {}
  for index, raw in enumerate(raw_trials):
    trial = _normalize_trial(
      raw, index=index, protocol=protocol, ablation=ablation
    )
    key = (
      float(trial["height_m"]),
      int(trial["repeat_index"]),
      int(trial["env_index"]),
    )
    if key in by_key:
      raise ValueError(f"Duplicate stair trial key: {key}.")
    by_key[key] = trial
  expected_keys = {
    (float(height), repeat, env)
    for height in protocol.heights_m
    for repeat in range(protocol.repeats)
    for env in range(protocol.num_envs_per_height)
  }
  if set(by_key) != expected_keys:
    raise ValueError("Stair trial matrix is incomplete or contains extra cells.")
  ordered = [by_key[key] for key in sorted(expected_keys)]
  rows: list[dict[str, object]] = []
  for height in protocol.heights_m:
    selected = [trial for trial in ordered if trial["height_m"] == height]
    successes = sum(bool(trial["success"]) for trial in selected)
    terminations = sum(bool(trial["terminated"]) for trial in selected)
    contacts = sum(bool(trial["non_wheel_contact"]) for trial in selected)
    mode_counts = {
      mode: sum(trial["traversal_mode"] == mode for trial in selected)
      for mode in TRAVERSAL_MODES
    }
    rows.append(
      {
        "height_m": float(height),
        "risers_per_trial": protocol.risers_per_trial,
        "trials": len(selected),
        "successes": successes,
        "success_rate": successes / len(selected),
        "minimum_successes": protocol.minimum_successes,
        "terminations": terminations,
        "non_wheel_contacts": contacts,
        "mode_counts": mode_counts,
        "passed": bool(
          successes >= protocol.minimum_successes
          and terminations == 0
          and contacts == 0
        ),
      }
    )
  return ordered, rows


def finalize_collection(
  request: Mapping[str, object],
  collection: Mapping[str, object],
) -> dict[str, object]:
  """Validate one live collection and return canonical provisional evidence."""

  normalized_request = validate_evaluation_request(request)
  suite = str(normalized_request["suite"])
  common_fields = {
    "request_sha256",
    "evaluation_source",
    "adapter_metadata",
  }
  expected_fields = common_fields | ({"gates"} if suite == RETENTION_SUITE else {"trials"})
  if set(collection) != expected_fields:
    raise ValueError("StairDynamic adapter collection schema drifted.")
  if collection.get("request_sha256") != normalized_request["request_sha256"]:
    raise ValueError("Adapter collection is not bound to the signed request.")
  source = collection.get("evaluation_source")
  if not isinstance(source, str) or not source.strip():
    raise ValueError("evaluation_source must be a non-empty string.")
  metadata = dict(_mapping(collection.get("adapter_metadata"), name="adapter_metadata"))
  stair_camp._validate_json_value(metadata, path="adapter_metadata")
  result: dict[str, object] = {
    "schema_version": EVALUATOR_SCHEMA_VERSION,
    "kind": EVALUATION_RESULT_KIND,
    "status": "complete",
    "task": DYNAMIC_STAIR_TASK_ID,
    "suite": suite,
    "profile": "formal",
    "evaluation_seed": REGISTERED_EVALUATION_SEED,
    "single_seed_status": "provisional",
    "promotion_claim_eligible": False,
    "request_sha256": normalized_request["request_sha256"],
    "request": normalized_request,
    "evaluation_source": source,
    "adapter_metadata": metadata,
  }
  if suite == RETENTION_SUITE:
    # Reuse the mature, frozen StairCamp normalizer over the same four gates.
    gates, booleans = stair_camp._normalize_gate_results(
      collection.get("gates"), profile="formal"
    )
    result.update(
      {
        "gates": gates,
        "gate_booleans": booleans,
        "all_gates_passed": all(booleans.values()),
        "result_passed": all(booleans.values()),
      }
    )
  else:
    protocol = protocol_for(suite)
    descriptor = resolve_ablation(
      str(_mapping(request["ablation"], name="ablation")["name"])
    )
    trials, rows = normalize_trials(
      collection.get("trials"), protocol=protocol, ablation=descriptor
    )
    primary = next(row for row in rows if row["height_m"] == PRIMARY_HEIGHT_M)
    result.update(
      {
        "trials": trials,
        "rows": rows,
        "primary_height_m": PRIMARY_HEIGHT_M,
        "primary_gate_passed": primary["passed"],
        "capability_extension_heights_m": [0.02, 0.03],
        "result_passed": primary["passed"],
      }
    )
  stair_camp._validate_json_value(result)
  return result


def validate_evaluation_result(
  value: Mapping[str, object],
) -> dict[str, object]:
  """Recompute a result from its embedded request and evidence."""

  if value.get("kind") != EVALUATION_RESULT_KIND:
    raise ValueError("Input is not a StairDynamic evaluation result.")
  request = _mapping(value.get("request"), name="result request")
  suite = request.get("suite")
  collection: dict[str, object] = {
    "request_sha256": value.get("request_sha256"),
    "evaluation_source": value.get("evaluation_source"),
    "adapter_metadata": value.get("adapter_metadata"),
  }
  collection["gates" if suite == RETENTION_SUITE else "trials"] = value.get(
    "gates" if suite == RETENTION_SUITE else "trials"
  )
  canonical = finalize_collection(request, collection)
  if dict(value) != canonical:
    raise ValueError("StairDynamic evaluation result is not canonical.")
  return canonical


def make_ablation_bundle(
  results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
  """Require exactly the six registered ablations for one selected checkpoint."""

  if len(results) != len(ABLATION_ORDER):
    raise ValueError("Ablation bundle requires exactly six results.")
  validated = [validate_evaluation_result(result) for result in results]
  by_name: dict[str, dict[str, object]] = {}
  for result in validated:
    request = _mapping(result["request"], name="result request")
    name = str(_mapping(request["ablation"], name="ablation")["name"])
    if name in by_name:
      raise ValueError(f"Duplicate ablation result: {name}.")
    by_name[name] = result
  if set(by_name) != set(ABLATION_ORDER):
    raise ValueError("Ablation bundle does not contain the registered six modes.")
  first_request = _mapping(by_name[ABLATION_ORDER[0]]["request"], name="request")
  suite = first_request.get("suite")
  if suite not in STAIR_PROTOCOLS:
    raise ValueError("Ablations must target a stair suite, not retention gates.")
  checkpoint_sha = _mapping(first_request["checkpoint"], name="checkpoint")[
    "checkpoint_file_sha256"
  ]
  for name in ABLATION_ORDER:
    request = _mapping(by_name[name]["request"], name=f"{name} request")
    if (
      request.get("suite") != suite
      or request.get("evaluation_protocol_sha256") != EVALUATION_PROTOCOL_SHA256
      or _mapping(request["checkpoint"], name="checkpoint").get(
        "checkpoint_file_sha256"
      )
      != checkpoint_sha
      or request.get("ablation") != ABLATION_DESCRIPTORS[name].to_dict()
    ):
      raise ValueError("Ablation results disagree on suite/checkpoint/protocol.")
  bundle = {
    "schema_version": EVALUATOR_SCHEMA_VERSION,
    "kind": ABLATION_BUNDLE_KIND,
    "status": "complete",
    "task": DYNAMIC_STAIR_TASK_ID,
    "suite": suite,
    "evaluation_seed": REGISTERED_EVALUATION_SEED,
    "single_seed_status": "provisional",
    "promotion_claim_eligible": False,
    "evaluation_protocol_sha256": EVALUATION_PROTOCOL_SHA256,
    "checkpoint_file_sha256": checkpoint_sha,
    "completed_ablations": list(ABLATION_ORDER),
    "results": {
      name: {
        "result_sha256": _canonical_sha256(by_name[name]),
        "primary_gate_passed": by_name[name]["primary_gate_passed"],
      }
      for name in ABLATION_ORDER
    },
  }
  stair_camp._validate_json_value(bundle)
  return bundle


def make_k3_screen_candidate(
  *,
  checkpoint_envelope: Mapping[str, object],
  budget_updates: int,
  gate_passes: Mapping[str, object],
  gate_stair_mode_false_positives: Mapping[str, object],
  height_row: Mapping[str, object],
) -> dict[str, object]:
  """Build one cheap rejection-only Stage5-retention/1 cm screen."""

  checkpoint = validate_checkpoint_envelope(checkpoint_envelope)
  budget = _integer(budget_updates, name="budget_updates", minimum=1)
  if budget not in REGISTERED_BUDGETS:
    raise ValueError("StairDynamic K=3 budget must be 100 or 500 updates.")
  gates = stair_camp._normalize_gate_boolean_map(
    gate_passes, name="gate_passes"
  )
  false_positives = stair_camp._normalize_gate_count_map(
    gate_stair_mode_false_positives,
    name="gate_stair_mode_false_positives",
  )
  rows = stair_camp._normalize_scan_rows(
    [height_row], K3_SCREEN_PROTOCOL
  )
  row = rows[0]
  passed = bool(
    all(gates.values())
    and not any(false_positives.values())
    and row["passed"] is True
    and row["terminations"] == 0
    and row["non_wheel_contacts"] == 0
  )
  return {
    "schema_version": EVALUATOR_SCHEMA_VERSION,
    "kind": K3_SCREEN_KIND,
    "task": DYNAMIC_STAIR_TASK_ID,
    "profile": "screen",
    "evidence_eligible": False,
    "evaluation_seed": REGISTERED_EVALUATION_SEED,
    "budget_updates": budget,
    "checkpoint": checkpoint,
    "gate_passes": gates,
    "gate_stair_mode_false_positives": false_positives,
    "height_screen": row,
    "screen_passed": passed,
  }


def validate_k3_screen_candidate(
  candidate: Mapping[str, object],
) -> dict[str, object]:
  if (
    candidate.get("schema_version") != EVALUATOR_SCHEMA_VERSION
    or candidate.get("kind") != K3_SCREEN_KIND
    or candidate.get("task") != DYNAMIC_STAIR_TASK_ID
    or candidate.get("profile") != "screen"
    or candidate.get("evidence_eligible") is not False
    or candidate.get("evaluation_seed") != REGISTERED_EVALUATION_SEED
  ):
    raise ValueError("StairDynamic K=3 candidate identity drifted.")
  normalized = make_k3_screen_candidate(
    checkpoint_envelope=_mapping(candidate.get("checkpoint"), name="checkpoint"),
    budget_updates=_integer(candidate.get("budget_updates"), name="budget_updates"),
    gate_passes=_mapping(candidate.get("gate_passes"), name="gate_passes"),
    gate_stair_mode_false_positives=_mapping(
      candidate.get("gate_stair_mode_false_positives"),
      name="gate_stair_mode_false_positives",
    ),
    height_row=_mapping(candidate.get("height_screen"), name="height_screen"),
  )
  if dict(candidate) != normalized:
    raise ValueError("StairDynamic K=3 candidate is not canonical.")
  return normalized



def select_newest_passing_checkpoint(
  candidates: Sequence[Mapping[str, object]],
) -> dict[str, object]:
  """Select the newest passer from exactly the latest three saved checkpoints."""

  if len(candidates) != 3:
    raise ValueError("StairDynamic K=3 requires exactly three candidates.")
  normalized = [validate_k3_screen_candidate(candidate) for candidate in candidates]
  budgets = {int(candidate["budget_updates"]) for candidate in normalized}
  if len(budgets) != 1:
    raise ValueError("StairDynamic K=3 candidates must share one budget pool.")
  budget = budgets.pop()
  checkpoints = [
    _mapping(candidate["checkpoint"], name="checkpoint") for candidate in normalized
  ]
  training = [
    _mapping(checkpoint["training"], name="training") for checkpoint in checkpoints
  ]
  updates = [int(record["completed_updates"]) for record in training]
  expected_updates = {
    budget - 2 * DYNAMIC_STAIR_SAVE_INTERVAL + 1,
    budget - DYNAMIC_STAIR_SAVE_INTERVAL + 1,
    budget,
  }
  if set(updates) != expected_updates or len(set(updates)) != 3:
    raise ValueError("K=3 candidates are not the exact latest three save-interval states.")
  checkpoint_hashes = [
    str(checkpoint["checkpoint_file_sha256"]) for checkpoint in checkpoints
  ]
  if len(set(checkpoint_hashes)) != 3:
    raise ValueError("K=3 candidates must reference three distinct checkpoints.")
  binding_fields = (
    "training_seed",
    "git_sha",
    "contract_sha256",
    "artifact_bindings",
    "action_scales",
    "maneuver_sha256",
    "source_stage5_checkpoint_sha256",
    "source_stage5_gate_sha256",
    "stage5_prefix_preserved_and_new_columns_zero",
  )
  for field in binding_fields:
    if any(record.get(field) != training[0].get(field) for record in training[1:]):
      raise ValueError(f"K=3 checkpoint bindings disagree on {field}.")
  ordered = sorted(
    normalized,
    key=lambda candidate: int(
      _mapping(
        _mapping(candidate["checkpoint"], name="checkpoint")["training"],
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
    "task": DYNAMIC_STAIR_TASK_ID,
    "status": "selected" if selected is not None else "no_passing_checkpoint",
    "classification": (
      "STAIR_DYNAMIC_CHECKPOINT_SELECTED"
      if selected is not None
      else "STOP_DYNAMIC_STAIR_UNQUALIFIED"
    ),
    "selection_rule": "newest_passing_of_exact_latest_3",
    "budget_updates": budget,
    "training_seed": REGISTERED_TRAINING_SEED,
    "ordered_candidates": [
      {
        "completed_updates": _mapping(
          _mapping(candidate["checkpoint"], name="checkpoint")["training"],
          name="training",
        )["completed_updates"],
        "checkpoint_file": _mapping(candidate["checkpoint"], name="checkpoint")[
          "checkpoint_file"
        ],
        "checkpoint_file_sha256": _mapping(
          candidate["checkpoint"], name="checkpoint"
        )["checkpoint_file_sha256"],
        "screen_passed": candidate["screen_passed"],
      }
      for candidate in ordered
    ],
    "selected_checkpoint": None if selected is None else selected["checkpoint"],
  }
  stair_camp._validate_json_value(result)
  return result



def _checkpoint_identity(value: Mapping[str, object]) -> dict[str, object]:
  training = _mapping(value.get("training"), name="checkpoint training")
  return {
    "checkpoint_file": value.get("checkpoint_file"),
    "checkpoint_file_sha256": value.get("checkpoint_file_sha256"),
    "training": training,
  }


def validate_k3_selection(
  value: Mapping[str, object],
  *,
  verify_selected_file: bool = False,
) -> dict[str, object]:
  expected_fields = {
    "schema_version",
    "kind",
    "task",
    "status",
    "classification",
    "selection_rule",
    "budget_updates",
    "training_seed",
    "ordered_candidates",
    "selected_checkpoint",
  }
  if set(value) != expected_fields:
    raise ValueError("StairDynamic K=3 selection schema drifted.")
  if (
    value.get("schema_version") != EVALUATOR_SCHEMA_VERSION
    or value.get("kind") != K3_SELECTION_KIND
    or value.get("task") != DYNAMIC_STAIR_TASK_ID
    or value.get("selection_rule") != "newest_passing_of_exact_latest_3"
    or value.get("training_seed") != REGISTERED_TRAINING_SEED
  ):
    raise ValueError("StairDynamic K=3 selection identity drifted.")
  budget = _integer(value.get("budget_updates"), name="budget_updates", minimum=1)
  if budget not in REGISTERED_BUDGETS:
    raise ValueError("StairDynamic K=3 selection budget drifted.")
  raw_ordered = _sequence(value.get("ordered_candidates"), name="ordered_candidates")
  if len(raw_ordered) != 3:
    raise ValueError("StairDynamic K=3 selection must archive three candidates.")
  ordered: list[dict[str, object]] = []
  for index, raw in enumerate(raw_ordered):
    row = _mapping(raw, name=f"ordered_candidates[{index}]")
    if set(row) != {
      "completed_updates",
      "checkpoint_file",
      "checkpoint_file_sha256",
      "screen_passed",
    }:
      raise ValueError("StairDynamic K=3 candidate summary schema drifted.")
    checkpoint_file = row.get("checkpoint_file")
    if not isinstance(checkpoint_file, str) or not checkpoint_file.strip():
      raise ValueError("StairDynamic K=3 candidate checkpoint path is invalid.")
    passed = row.get("screen_passed")
    if not isinstance(passed, bool):
      raise ValueError("StairDynamic K=3 screen result must be boolean.")
    ordered.append(
      {
        "completed_updates": _integer(
          row.get("completed_updates"), name="completed_updates", minimum=1
        ),
        "checkpoint_file": checkpoint_file,
        "checkpoint_file_sha256": _sha256(
          row.get("checkpoint_file_sha256"), name="checkpoint_file_sha256"
        ),
        "screen_passed": passed,
      }
    )
  expected_updates = [
    budget,
    budget - DYNAMIC_STAIR_SAVE_INTERVAL + 1,
    budget - 2 * DYNAMIC_STAIR_SAVE_INTERVAL + 1,
  ]
  if [row["completed_updates"] for row in ordered] != expected_updates:
    raise ValueError("StairDynamic K=3 ordered checkpoint cadence drifted.")
  if len({row["checkpoint_file_sha256"] for row in ordered}) != 3:
    raise ValueError("StairDynamic K=3 candidate hashes must be distinct.")
  first_passer = next(
    (row for row in ordered if row["screen_passed"] is True), None
  )
  selected_raw = value.get("selected_checkpoint")
  if first_passer is None:
    if (
      value.get("status") != "no_passing_checkpoint"
      or value.get("classification") != "STOP_DYNAMIC_STAIR_UNQUALIFIED"
      or selected_raw is not None
    ):
      raise ValueError("StairDynamic K=3 STOP selection drifted.")
    selected = None
  else:
    if (
      value.get("status") != "selected"
      or value.get("classification") != "STAIR_DYNAMIC_CHECKPOINT_SELECTED"
    ):
      raise ValueError("StairDynamic K=3 selected status drifted.")
    selected = validate_checkpoint_envelope(
      _mapping(selected_raw, name="selected_checkpoint"),
      verify_file=verify_selected_file,
    )
    training = _mapping(selected.get("training"), name="selected training")
    if (
      first_passer["checkpoint_file"] != selected["checkpoint_file"]
      or first_passer["checkpoint_file_sha256"]
      != selected["checkpoint_file_sha256"]
      or first_passer["completed_updates"] != training["completed_updates"]
    ):
      raise ValueError("StairDynamic K=3 did not select the newest passer.")
  normalized = {
    "schema_version": EVALUATOR_SCHEMA_VERSION,
    "kind": K3_SELECTION_KIND,
    "task": DYNAMIC_STAIR_TASK_ID,
    "status": value.get("status"),
    "classification": value.get("classification"),
    "selection_rule": "newest_passing_of_exact_latest_3",
    "budget_updates": budget,
    "training_seed": REGISTERED_TRAINING_SEED,
    "ordered_candidates": ordered,
    "selected_checkpoint": selected,
  }
  # Selection generation intentionally does not re-read checkpoint bytes.  When
  # a downstream authorization requests verification, canonicalize the flag to
  # true after the byte/hash check above.
  if not verify_selected_file and dict(value) != normalized:
    raise ValueError("StairDynamic K=3 selection is not canonical.")
  return normalized


def make_extension_authorization(
  *,
  k3_selection: Mapping[str, object],
  retention_result: Mapping[str, object],
  single_riser_result: Mapping[str, object],
) -> dict[str, object]:
  """Authorize 100-pool selected evidence, never an unselected checkpoint."""

  selection = validate_k3_selection(k3_selection, verify_selected_file=True)
  if selection["status"] != "selected" or selection["budget_updates"] != 100:
    raise ValueError("Extension requires a selected 100-update-pool K=3 result.")
  selected = _mapping(selection["selected_checkpoint"], name="selected_checkpoint")
  retention = validate_evaluation_result(retention_result)
  stairs = validate_evaluation_result(single_riser_result)
  for result, suite in ((retention, RETENTION_SUITE), (stairs, "single-riser")):
    request = _mapping(result.get("request"), name=f"{suite} request")
    if request.get("suite") != suite:
      raise ValueError(f"Extension requires the formal {suite} result.")
    if _mapping(request.get("ablation"), name="ablation").get("name") != "full":
      raise ValueError("Extension qualification requires the full controller.")
    checkpoint = _mapping(request.get("checkpoint"), name="result checkpoint")
    if _checkpoint_identity(checkpoint) != _checkpoint_identity(selected):
      raise ValueError("Extension evidence does not use the K=3 selected checkpoint.")
  if retention.get("result_passed") is not True:
    raise ValueError("Extension requires all four formal retention gates.")
  gates = _sequence(retention.get("gates"), name="retention gates")
  if any(
    _mapping(gate, name="retention gate").get("stair_mode_false_positives") != 0
    for gate in gates
  ):
    raise ValueError("Extension retention gates require zero stair false positives.")
  rows = _sequence(stairs.get("rows"), name="single-riser rows")
  primary = next(
    (
      _mapping(row, name="single-riser row")
      for row in rows
      if _mapping(row, name="single-riser row").get("height_m")
      == PRIMARY_HEIGHT_M
    ),
    None,
  )
  if primary is None or (
    primary.get("trials") != FORMAL_TRIALS_PER_HEIGHT
    or _integer(primary.get("successes"), name="successes")
    < FORMAL_MIN_SUCCESSES
    or primary.get("terminations") != 0
    or primary.get("non_wheel_contacts") != 0
    or primary.get("passed") is not True
  ):
    raise ValueError("Extension requires at least 44/48 safe 1 cm successes.")
  training = _mapping(selected.get("training"), name="selected training")
  payload: dict[str, object] = {
    "schema_version": EVALUATOR_SCHEMA_VERSION,
    "kind": EXTENSION_AUTHORIZATION_KIND,
    "status": "authorized",
    "classification": "STAIR_DYNAMIC_EXTENSION_AUTHORIZED",
    "task": DYNAMIC_STAIR_TASK_ID,
    "source_budget_updates": DYNAMIC_STAIR_PROBE_UPDATES,
    "target_total_updates": DYNAMIC_STAIR_EXTENSION_TOTAL_UPDATES,
    "selected_completed_updates": training["completed_updates"],
    "selected_checkpoint_file": selected["checkpoint_file"],
    "selected_checkpoint_sha256": selected["checkpoint_file_sha256"],
    "k3_selection": selection,
    "retention_result": retention,
    "single_riser_result": stairs,
  }
  payload["authorization_sha256"] = _canonical_sha256(payload)
  stair_camp._validate_json_value(payload)
  return payload


def validate_extension_authorization(
  value: Mapping[str, object],
) -> dict[str, object]:
  expected_fields = {
    "schema_version",
    "kind",
    "status",
    "classification",
    "task",
    "source_budget_updates",
    "target_total_updates",
    "selected_completed_updates",
    "selected_checkpoint_file",
    "selected_checkpoint_sha256",
    "k3_selection",
    "retention_result",
    "single_riser_result",
    "authorization_sha256",
  }
  if set(value) != expected_fields:
    raise ValueError("StairDynamic extension authorization schema drifted.")
  canonical = make_extension_authorization(
    k3_selection=_mapping(value.get("k3_selection"), name="k3_selection"),
    retention_result=_mapping(
      value.get("retention_result"), name="retention_result"
    ),
    single_riser_result=_mapping(
      value.get("single_riser_result"), name="single_riser_result"
    ),
  )
  if dict(value) != canonical:
    raise ValueError("StairDynamic extension authorization is not canonical.")
  return canonical


def manifest_payload() -> dict[str, object]:
  """Expose the complete pure evaluator and future live-hook contract."""

  return {
    "schema_version": EVALUATOR_SCHEMA_VERSION,
    "kind": "stair_dynamic_evaluator_manifest",
    "task": DYNAMIC_STAIR_TASK_ID,
    "evaluation_seed": REGISTERED_EVALUATION_SEED,
    "training_seed": REGISTERED_TRAINING_SEED,
    "single_seed_status": "provisional",
    "promotion_claim_eligible": False,
    "evaluation_protocol_sha256": EVALUATION_PROTOCOL_SHA256,
    **_protocol_binding_payload(),
    "checkpoint_contract": {
      "training_info_key": DYNAMIC_STAIR_TRAINING_INFO_KEY,
      "migration_info_key": DYNAMIC_STAIR_MIGRATION_INFO_KEY,
      "trained_envelope_kind": CHECKPOINT_ENVELOPE_KIND,
      "zero_update_envelope_kind": MIGRATION_CHECKPOINT_ENVELOPE_KIND,
      "zero_update_completed_updates": 0,
      "actor_observation_width": DYNAMIC_STAIR_ACTOR_WIDTH,
      "critic_observation_width": DYNAMIC_STAIR_CRITIC_WIDTH,
      "action_width": ACTION_WIDTH,
      "action_scales": list(DYNAMIC_STAIR_ACTION_SCALES),
      "artifact_binding_names": list(DYNAMIC_ARTIFACT_BINDING_NAMES),
    },
    "k3": {
      "pool_size": 3,
      "budgets": list(REGISTERED_BUDGETS),
      "save_interval_updates": DYNAMIC_STAIR_SAVE_INTERVAL,
      "selection_rule": "newest_passing_of_exact_latest_3",
      "screen_protocol": K3_SCREEN_PROTOCOL.to_dict(),
      "extension_authorization_kind": EXTENSION_AUTHORIZATION_KIND,
      "extension_requires": [
        "selected_100_pool_k3_checkpoint",
        "all_four_formal_retention_gates",
        "single_riser_1cm_at_least_44_of_48_safe",
      ],
    },
    "live_hook": {
      "implemented_here": False,
      "implemented": True,
      "module": (
        "hoppertrex_mjlab.scripts.rsl_rl.stair_dynamic_live_adapter"
      ),
      "interface": "collect(request: Mapping[str, object]) -> Mapping[str, object]",
      "stair_collection_keys": [
        "request_sha256",
        "evaluation_source",
        "adapter_metadata",
        "trials",
      ],
      "retention_collection_keys": [
        "request_sha256",
        "evaluation_source",
        "adapter_metadata",
        "gates",
      ],
      "heavy_import_policy": "live_hook_only",
    },
  }


def _read_json(path: Path, *, name: str) -> Mapping[str, object]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  return _mapping(payload, name=name)


def _add_output(parser: argparse.ArgumentParser) -> None:
  parser.add_argument("--output", type=Path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest="command", required=True)
  manifest = commands.add_parser("manifest")
  _add_output(manifest)
  envelope = commands.add_parser("checkpoint-envelope")
  envelope.add_argument("--checkpoint-file", type=Path, required=True)
  _add_output(envelope)
  migration_envelope = commands.add_parser("migration-checkpoint-envelope")
  migration_envelope.add_argument("--checkpoint-file", type=Path, required=True)
  migration_envelope.add_argument("--expectation", type=Path, required=True)
  _add_output(migration_envelope)
  request = commands.add_parser("make-request")
  request.add_argument("--suite", choices=EVALUATION_SUITES, required=True)
  request.add_argument("--checkpoint-envelope", type=Path, required=True)
  request.add_argument("--expectation", type=Path, required=True)
  request.add_argument("--ablation", choices=ABLATION_ORDER, default="full")
  request.add_argument("--device", default="cuda:0")
  _add_output(request)
  validate = commands.add_parser("validate-checkpoint")
  validate.add_argument("--envelope", type=Path, required=True)
  validate.add_argument("--verify-file", action="store_true")
  _add_output(validate)
  finalize = commands.add_parser("finalize")
  finalize.add_argument("--request", type=Path, required=True)
  finalize.add_argument("--collection", type=Path, required=True)
  _add_output(finalize)
  select = commands.add_parser("select-k3")
  select.add_argument("--candidate", type=Path, nargs=3, required=True)
  _add_output(select)
  bundle = commands.add_parser("bundle-ablations")
  bundle.add_argument("--result", type=Path, nargs=6, required=True)
  _add_output(bundle)
  authorize = commands.add_parser("authorize-extension")
  authorize.add_argument("--selection", type=Path, required=True)
  authorize.add_argument("--retention-result", type=Path, required=True)
  authorize.add_argument("--single-riser-result", type=Path, required=True)
  _add_output(authorize)
  validate_authorization = commands.add_parser("validate-extension-authorization")
  validate_authorization.add_argument("--authorization", type=Path, required=True)
  _add_output(validate_authorization)
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  if args.command == "manifest":
    result = manifest_payload()
  elif args.command == "checkpoint-envelope":
    result = checkpoint_envelope_from_file(args.checkpoint_file)
  elif args.command == "migration-checkpoint-envelope":
    expectation = checkpoint_expectation_from_mapping(
      _read_json(args.expectation, name="checkpoint expectation")
    )
    result = migration_checkpoint_envelope_from_file(
      args.checkpoint_file, expectation=expectation
    )
  elif args.command == "make-request":
    expectation = checkpoint_expectation_from_mapping(
      _read_json(args.expectation, name="checkpoint expectation")
    )
    result = make_evaluation_request(
      suite=args.suite,
      checkpoint_envelope=_read_json(
        args.checkpoint_envelope, name="checkpoint envelope"
      ),
      expectation=expectation,
      ablation=args.ablation,
      device=args.device,
    )
  elif args.command == "validate-checkpoint":
    checkpoint = validate_checkpoint_envelope(
      _read_json(args.envelope, name="checkpoint envelope"),
      verify_file=args.verify_file,
    )
    result = {
      "schema_version": EVALUATOR_SCHEMA_VERSION,
      "kind": "stair_dynamic_checkpoint_validation",
      "valid": True,
      "checkpoint": checkpoint,
    }
  elif args.command == "finalize":
    result = finalize_collection(
      _read_json(args.request, name="evaluation request"),
      _read_json(args.collection, name="adapter collection"),
    )
  elif args.command == "select-k3":
    result = select_newest_passing_checkpoint(
      [_read_json(path, name="K=3 candidate") for path in args.candidate]
    )
  elif args.command == "bundle-ablations":
    result = make_ablation_bundle(
      [_read_json(path, name="ablation result") for path in args.result]
    )
  elif args.command == "authorize-extension":
    result = make_extension_authorization(
      k3_selection=_read_json(args.selection, name="K=3 selection"),
      retention_result=_read_json(
        args.retention_result, name="retention result"
      ),
      single_riser_result=_read_json(
        args.single_riser_result, name="single-riser result"
      ),
    )
  elif args.command == "validate-extension-authorization":
    result = validate_extension_authorization(
      _read_json(args.authorization, name="extension authorization")
    )
  else:  # pragma: no cover
    raise AssertionError(f"Unhandled command: {args.command}")
  write_machine_output(result, args.output)
  return 0


__all__ = [
  "ABLATION_DESCRIPTORS",
  "ABLATION_ORDER",
  "CHECKPOINT_ENVELOPE_KIND",
  "CONTINUOUS_STAIRS_PROTOCOL",
  "EVALUATION_PROTOCOL_SHA256",
  "EVALUATION_RESULT_KIND",
  "EVALUATION_SUITES",
  "EXTENSION_AUTHORIZATION_KIND",
  "FORMAL_HEIGHTS_M",
  "FORMAL_MIN_SUCCESSES",
  "GATE_BINDINGS",
  "GATE_NAMES",
  "K3_SCREEN_KIND",
  "K3_SCREEN_PROTOCOL",
  "K3_SELECTION_KIND",
  "MIGRATION_CHECKPOINT_ENVELOPE_KIND",
  "PHASE_NAMES",
  "PRIMARY_HEIGHT_M",
  "RETENTION_SUITE",
  "SINGLE_RISER_PROTOCOL",
  "CheckpointExpectation",
  "StairEvaluationProtocol",
  "checkpoint_envelope_from_file",
  "checkpoint_envelope_from_loaded_checkpoint",
  "checkpoint_expectation_from_mapping",
  "deterministic_json",
  "finalize_collection",
  "gate_bindings_for_profile",
  "main",
  "make_ablation_bundle",
  "make_evaluation_request",
  "make_extension_authorization",
  "make_k3_screen_candidate",
  "manifest_payload",
  "migration_checkpoint_envelope_from_file",
  "migration_checkpoint_envelope_from_loaded_checkpoint",
  "normalize_trials",
  "parse_args",
  "protocol_for",
  "resolve_ablation",
  "select_newest_passing_checkpoint",
  "validate_checkpoint_envelope",
  "validate_evaluation_checkpoint_envelope",
  "validate_evaluation_request",
  "validate_evaluation_result",
  "validate_extension_authorization",
  "validate_k3_screen_candidate",
  "validate_k3_selection",
  "validate_migration_checkpoint_envelope",
  "validate_training_info",
  "write_machine_output",
]


if __name__ == "__main__":
  raise SystemExit(main())
