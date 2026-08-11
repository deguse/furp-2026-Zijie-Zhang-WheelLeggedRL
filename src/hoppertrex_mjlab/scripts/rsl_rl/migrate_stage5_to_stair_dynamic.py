#!/usr/bin/env python3
# ruff: noqa: TRY004
"""Migrate one selected Hybrid-v2 Stage5 checkpoint to StairDynamic v3."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import uuid
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from hoppertrex_mjlab.hybrid.config import (
  HYBRID_ACTION_NAMES,
  HYBRID_ACTION_STD,
)
from hoppertrex_mjlab.hybrid.stair_dynamic import DYNAMIC_STAIR_TASK_ID
from hoppertrex_mjlab.hybrid.stair_dynamic_contract import (
  DYNAMIC_STAIR_ACTOR_WIDTH,
  DYNAMIC_STAIR_CRITIC_WIDTH,
  DYNAMIC_STAIR_MIGRATION_INFO_KEY,
  DYNAMIC_STAIR_STAGE5_ACTOR_WIDTH,
)
from hoppertrex_mjlab.scripts.rsl_rl.hybrid_gate import MIN_STAGE5_KICK_EVENTS
from hoppertrex_mjlab.scripts.rsl_rl.migrate_hybrid_stage import (
  COLLAPSED_ACTION_STD_THRESHOLD,
)

STAGE1_TASK_ID = "HopperTrex-Hybrid-v2-Stage1"
STAGE5_TASK_ID = "HopperTrex-Hybrid-v2-Stage5"
STAGE5_SELECTED_SEED = 1
STAGE5_COMPLETED_UPDATES = 100
STAGE5_GATE_SCHEMA_VERSION = 2
STAGE5_CRITIC_WIDTH = DYNAMIC_STAIR_STAGE5_ACTOR_WIDTH


def _sha256_bytes(payload: bytes) -> str:
  return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
  if not isinstance(value, Mapping):
    raise ValueError(f"{name} must be a mapping.")
  return value


def _exact_int(value: object, *, name: str) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise ValueError(f"{name} must be an integer.")
  return value


def _finite_number(value: object, *, name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"{name} must be a finite number.")
  result = float(value)
  if not math.isfinite(result):
    raise ValueError(f"{name} must be a finite number.")
  return result


def _validate_sha256(value: object, *, name: str) -> str:
  if (
    not isinstance(value, str)
    or len(value) != 64
    or any(character not in "0123456789abcdef" for character in value)
  ):
    raise ValueError(f"{name} must be a lowercase SHA256 digest.")
  return value


def _validate_git_sha(value: object, *, name: str) -> str:
  if (
    not isinstance(value, str)
    or len(value) != 40
    or any(character not in "0123456789abcdef" for character in value)
  ):
    raise ValueError(f"{name} must be a full lowercase Git SHA.")
  return value


def _validate_source_migration(migration: Mapping[str, Any]) -> None:
  if (
    _exact_int(migration.get("source_stage"), name="source_stage") != 4
    or _exact_int(migration.get("target_stage"), name="target_stage") != 5
  ):
    raise ValueError("Source checkpoint must be an adjacent Stage4-to-Stage5 migration.")
  source_std = migration.get("source_action_std")
  if not isinstance(source_std, list) or len(source_std) != len(HYBRID_ACTION_NAMES):
    raise ValueError("Source Stage5 migration is missing its six-action std audit.")
  for index, value in enumerate(source_std):
    if _finite_number(value, name=f"source_action_std[{index}]") <= 0.0:
      raise ValueError("Source Stage5 migration std audit must be positive.")
  collapsed = migration.get("collapsed_active_actions")
  reset = migration.get("reset_collapsed_active_std")
  if not isinstance(collapsed, list) or not isinstance(reset, bool):
    raise ValueError("Source Stage5 migration is missing collapsed-std provenance.")
  if collapsed and not reset:
    raise ValueError("Source Stage5 migration contains unreset collapsed actions.")


def _validate_formal_robust_gate(
  gate: Mapping[str, Any],
  *,
  source_checkpoint_sha256: str,
  source_seed: int,
  source_git_sha: str,
) -> None:
  if (
    _exact_int(gate.get("schema_version"), name="Stage5 gate schema_version")
    != STAGE5_GATE_SCHEMA_VERSION
  ):
    raise ValueError("Stage5 robust gate schema_version does not match.")
  if gate.get("gate_pass") is not True:
    raise ValueError("Stage5 robust gate gate_pass does not match: expected True.")
  if _exact_int(gate.get("seed"), name="Stage5 gate seed") != source_seed:
    raise ValueError("Stage5 robust gate seed does not match the source seed.")
  expected = {
    "suite": "robust",
    "task": STAGE5_TASK_ID,
    "evaluation_profile": "formal",
    "evaluation_source": "live",
    "git_sha": source_git_sha,
    "checkpoint_file_sha256": source_checkpoint_sha256,
  }
  for field, expected_value in expected.items():
    if gate.get(field) != expected_value:
      raise ValueError(
        f"Stage5 robust gate {field} does not match: "
        f"{gate.get(field)!r} != {expected_value!r}."
      )
  _validate_sha256(
    gate.get("checkpoint_file_sha256"),
    name="Stage5 gate checkpoint_file_sha256",
  )
  _validate_git_sha(gate.get("git_sha"), name="Stage5 gate git_sha")
  if not isinstance(gate.get("checkpoint"), str) or not gate["checkpoint"]:
    raise ValueError("Stage5 robust gate is missing its checkpoint path audit.")

  rollout = _mapping(gate.get("rollout"), name="Stage5 robust gate rollout")
  if _exact_int(rollout.get("steps"), name="formal gate steps") < 3000:
    raise ValueError("Stage5 robust formal gate must contain at least 3000 steps.")
  if _exact_int(rollout.get("num_envs"), name="formal gate num_envs") < 32:
    raise ValueError("Stage5 robust formal gate must contain at least 32 environments.")
  if rollout.get("leg_residuals_ablated") is not False:
    raise ValueError("Stage5 migration rejects leg-ablated formal gate evidence.")

  checks = gate.get("checks")
  if not isinstance(checks, list) or not checks:
    raise ValueError("Stage5 robust formal gate must contain check evidence.")
  if any(
    not isinstance(check, Mapping) or check.get("pass") is not True
    for check in checks
  ):
    raise ValueError("Stage5 robust formal gate contains a failed or malformed check.")
  event_checks = [
    check
    for check in checks
    if check.get("name") == "recovery_kick_event_count"
    and check.get("scenario") == "stage5_ablation"
  ]
  if len(event_checks) != 1:
    raise ValueError("Stage5 robust formal gate has no unique recovery event audit.")
  event_count = _finite_number(
    event_checks[0].get("value"), name="Stage5 recovery kick event count"
  )
  if event_count < MIN_STAGE5_KICK_EVENTS:
    raise ValueError(
      "Stage5 robust formal gate has fewer than "
      f"{MIN_STAGE5_KICK_EVENTS} recovery kick events."
    )


def validate_stage5_source(
  checkpoint: Mapping[str, Any],
  gate: Mapping[str, Any],
  *,
  source_checkpoint_sha256: str,
) -> dict[str, Any]:
  """Validate the exact selected Stage5 source and its formal robust gate."""

  source_checkpoint_sha256 = _validate_sha256(
    source_checkpoint_sha256, name="source checkpoint SHA256"
  )
  infos = _mapping(checkpoint.get("infos"), name="Stage5 source infos")
  if "stair_camp_training" in infos:
    raise ValueError("StairCamp checkpoints cannot migrate to StairDynamic v3.")
  if DYNAMIC_STAIR_MIGRATION_INFO_KEY in infos:
    raise ValueError("Checkpoint already contains a StairDynamic migration record.")

  bootstrap = _mapping(
    infos.get("hybrid_stage1_bootstrap"),
    name="Stage5 source bootstrap provenance",
  )
  if (
    bootstrap.get("task") != STAGE1_TASK_ID
    or _exact_int(bootstrap.get("stage"), name="bootstrap stage") != 1
  ):
    raise ValueError("Stage5 source bootstrap task/stage provenance is invalid.")
  if tuple(bootstrap.get("action_order", ())) != HYBRID_ACTION_NAMES:
    raise ValueError("Stage5 source action order does not match Hybrid v2.")
  source_seed = _exact_int(bootstrap.get("seed"), name="source training seed")
  if source_seed != STAGE5_SELECTED_SEED:
    raise ValueError(
      f"Stage5 source seed must be the selected seed {STAGE5_SELECTED_SEED}."
    )

  source_migration = _mapping(
    infos.get("hybrid_stage_migration"),
    name="Stage5 source migration provenance",
  )
  _validate_source_migration(source_migration)

  training = _mapping(
    infos.get("hybrid_training"), name="Stage5 source training provenance"
  )
  source_git_sha = _validate_git_sha(
    training.get("git_sha"), name="Stage5 source training git_sha"
  )
  if training.get("task", STAGE5_TASK_ID) != STAGE5_TASK_ID:
    raise ValueError("Stage5 source training task provenance is invalid.")
  if "training_seed" in training and (
    _exact_int(training.get("training_seed"), name="training_seed") != source_seed
  ):
    raise ValueError("Stage5 source training seed provenance is inconsistent.")

  source_iteration = _exact_int(checkpoint.get("iter"), name="source iteration")
  completed_updates = source_iteration + 1
  if completed_updates != STAGE5_COMPLETED_UPDATES:
    raise ValueError(
      "StairDynamic migration requires the selected Stage5 100-update checkpoint."
    )
  if "completed_updates" in training and (
    _exact_int(training.get("completed_updates"), name="completed_updates")
    != completed_updates
  ):
    raise ValueError("Stage5 source completed-update provenance is inconsistent.")

  _validate_formal_robust_gate(
    gate,
    source_checkpoint_sha256=source_checkpoint_sha256,
    source_seed=source_seed,
    source_git_sha=source_git_sha,
  )
  return {
    "source_task": STAGE5_TASK_ID,
    "source_seed": source_seed,
    "source_completed_updates": completed_updates,
    "source_git_sha": source_git_sha,
  }


def _tensor_state_dict(value: object, *, name: str) -> Mapping[str, torch.Tensor]:
  state = _mapping(value, name=name)
  if any(not isinstance(key, str) for key in state):
    raise ValueError(f"{name} keys must be strings.")
  if any(not isinstance(tensor, torch.Tensor) for tensor in state.values()):
    raise ValueError(f"{name} values must be tensors.")
  return state  # type: ignore[return-value]


def expand_observation_input(
  source_state: Mapping[str, torch.Tensor],
  *,
  source_width: int,
  target_width: int,
  name: str,
) -> tuple[dict[str, torch.Tensor], str]:
  """Append zero columns to the unique first layer and clone all tensors."""

  candidates: list[str] = []
  for key, weight in source_state.items():
    if not key.endswith(".weight") or weight.ndim != 2:
      continue
    if tuple(weight.shape)[1] != source_width:
      continue
    bias_key = f"{key[:-len('.weight')]}.bias"
    bias = source_state.get(bias_key)
    if isinstance(bias, torch.Tensor) and tuple(bias.shape) == (weight.shape[0],):
      candidates.append(key)
  if len(candidates) != 1:
    raise ValueError(
      f"{name} must contain exactly one {source_width}-wide first layer; "
      f"found {len(candidates)}."
    )
  if target_width <= source_width:
    raise ValueError(f"{name} target width must exceed its source width.")

  first_layer_key = candidates[0]
  result = {key: tensor.clone() for key, tensor in source_state.items()}
  source_weight = source_state[first_layer_key]
  expanded = source_weight.new_zeros((source_weight.shape[0], target_width))
  expanded[:, :source_width].copy_(source_weight)
  result[first_layer_key] = expanded
  return result, first_layer_key


def _migrate_actor_std(
  actor_state: dict[str, torch.Tensor],
  *,
  reset_collapsed_active_std: bool,
) -> dict[str, Any]:
  std_key = "distribution.std_param"
  log_std_key = "distribution.log_std_param"
  present = [key for key in (std_key, log_std_key) if key in actor_state]
  if len(present) != 1:
    raise ValueError("Stage5 actor must contain exactly one per-action std parameter.")
  key = present[0]
  parameter = actor_state[key]
  if tuple(parameter.shape) != (len(HYBRID_ACTION_NAMES),):
    raise ValueError(f"{key} must contain six action values.")
  if not parameter.is_floating_point() or not torch.isfinite(parameter).all().item():
    raise ValueError(f"{key} must contain finite floating-point values.")

  source_std_tensor = parameter.clone() if key == std_key else torch.exp(parameter)
  if (
    not torch.isfinite(source_std_tensor).all().item()
    or not torch.all(source_std_tensor > 0.0).item()
  ):
    raise ValueError("Stage5 action std must be finite and positive.")
  collapsed_indices = [
    index
    for index, value in enumerate(source_std_tensor)
    if float(value) < COLLAPSED_ACTION_STD_THRESHOLD
  ]
  if collapsed_indices and not reset_collapsed_active_std:
    actions = ", ".join(HYBRID_ACTION_NAMES[index] for index in collapsed_indices)
    raise ValueError(
      "Stage5 source has collapsed active action std: "
      f"{actions}. Rerun with --reset-collapsed-active-std."
    )

  target_parameter = parameter.clone()
  for index in collapsed_indices:
    value = HYBRID_ACTION_STD[index]
    target_parameter[index] = value if key == std_key else math.log(value)
  actor_state[key] = target_parameter
  target_std_tensor = (
    target_parameter.clone() if key == std_key else torch.exp(target_parameter)
  )
  return {
    "std_key": key,
    "source_action_std": [float(value) for value in source_std_tensor],
    "target_action_std": [float(value) for value in target_std_tensor],
    "collapsed_std_threshold": COLLAPSED_ACTION_STD_THRESHOLD,
    "collapsed_active_indices": collapsed_indices,
    "collapsed_active_actions": [
      HYBRID_ACTION_NAMES[index] for index in collapsed_indices
    ],
    "reset_collapsed_active_std": reset_collapsed_active_std,
  }


def migrate_checkpoint(
  checkpoint: Mapping[str, Any],
  gate: Mapping[str, Any],
  *,
  source_checkpoint_sha256: str,
  source_gate_sha256: str,
  reset_collapsed_active_std: bool = False,
  created_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Build a fresh-optimizer StairDynamic checkpoint without mutating source."""

  source = validate_stage5_source(
    checkpoint,
    gate,
    source_checkpoint_sha256=source_checkpoint_sha256,
  )
  source_gate_sha256 = _validate_sha256(
    source_gate_sha256, name="source gate SHA256"
  )
  actor_source = _tensor_state_dict(
    checkpoint.get("actor_state_dict"), name="Stage5 actor_state_dict"
  )
  critic_source = _tensor_state_dict(
    checkpoint.get("critic_state_dict"), name="Stage5 critic_state_dict"
  )
  actor, actor_first_layer = expand_observation_input(
    actor_source,
    source_width=DYNAMIC_STAIR_STAGE5_ACTOR_WIDTH,
    target_width=DYNAMIC_STAIR_ACTOR_WIDTH,
    name="Stage5 actor",
  )
  critic, critic_first_layer = expand_observation_input(
    critic_source,
    source_width=STAGE5_CRITIC_WIDTH,
    target_width=DYNAMIC_STAIR_CRITIC_WIDTH,
    name="Stage5 critic",
  )
  std_report = _migrate_actor_std(
    actor,
    reset_collapsed_active_std=reset_collapsed_active_std,
  )

  migrated = deepcopy(dict(checkpoint))
  migrated["actor_state_dict"] = actor
  migrated["critic_state_dict"] = critic
  optimizer = migrated.get("optimizer_state_dict")
  if not isinstance(optimizer, dict) or not isinstance(
    optimizer.get("state"), Mapping
  ):
    raise ValueError("Stage5 optimizer_state_dict has no state mapping.")
  optimizer["state"] = {}
  migrated["iter"] = 0

  migration = {
    "source_checkpoint_sha256": source_checkpoint_sha256,
    "source_gate_sha256": source_gate_sha256,
    "source_task": source["source_task"],
    "source_seed": source["source_seed"],
    "source_completed_updates": source["source_completed_updates"],
    "target_task": DYNAMIC_STAIR_TASK_ID,
    "source_actor_width": DYNAMIC_STAIR_STAGE5_ACTOR_WIDTH,
    "target_actor_width": DYNAMIC_STAIR_ACTOR_WIDTH,
    "source_critic_width": STAGE5_CRITIC_WIDTH,
    "target_critic_width": DYNAMIC_STAIR_CRITIC_WIDTH,
    "actor_first_layer": actor_first_layer,
    "critic_first_layer": critic_first_layer,
    **std_report,
    "created_at": created_at or datetime.now(UTC).isoformat(timespec="seconds"),
  }
  infos = migrated.get("infos")
  if not isinstance(infos, dict):
    raise ValueError("Stage5 source infos must be a mutable mapping after copy.")
  infos[DYNAMIC_STAIR_MIGRATION_INFO_KEY] = migration
  # This is a new task, not a Stage5 continuation.  MjLab otherwise restores
  # the source env counter and would immediately skip/cross v3 curriculum
  # boundaries despite ``iter=0``.
  infos["env_state"] = {"common_step_counter": 0}
  return migrated, migration


def atomic_torch_save_no_clobber(payload: object, output: Path) -> None:
  """Publish a fully flushed torch file atomically without replacing a peer."""

  if output.exists():
    raise FileExistsError(f"Refusing to overwrite output checkpoint: {output}")
  output.parent.mkdir(parents=True, exist_ok=True)
  temporary = output.with_name(f".{output.name}.incomplete.{uuid.uuid4().hex}")
  try:
    with temporary.open("xb") as stream:
      torch.save(payload, stream)
      stream.flush()
      os.fsync(stream.fileno())
    os.link(temporary, output)
  finally:
    if temporary.exists():
      temporary.unlink()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source-checkpoint", type=Path, required=True)
  parser.add_argument("--source-gate-json", type=Path, required=True)
  parser.add_argument("--output-checkpoint", type=Path, required=True)
  parser.add_argument(
    "--reset-collapsed-active-std",
    action="store_true",
    help=(
      "Explicitly reset only active Stage5 action std values below the frozen "
      "collapse threshold."
    ),
  )
  return parser.parse_args(argv)


def main() -> None:
  args = parse_args()
  if not args.source_checkpoint.is_file():
    raise FileNotFoundError(
      f"Source checkpoint not found: {args.source_checkpoint}"
    )
  if not args.source_gate_json.is_file():
    raise FileNotFoundError(f"Source gate JSON not found: {args.source_gate_json}")
  if args.output_checkpoint.exists():
    raise FileExistsError(
      f"Refusing to overwrite output checkpoint: {args.output_checkpoint}"
    )

  checkpoint_bytes = args.source_checkpoint.read_bytes()
  gate_bytes = args.source_gate_json.read_bytes()
  checkpoint = torch.load(
    io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False
  )
  if not isinstance(checkpoint, Mapping):
    raise ValueError("Source checkpoint must contain a mapping.")
  gate = json.loads(gate_bytes.decode("utf-8"))
  if not isinstance(gate, Mapping):
    raise ValueError("Source gate JSON must contain an object.")

  migrated, migration = migrate_checkpoint(
    checkpoint,
    gate,
    source_checkpoint_sha256=_sha256_bytes(checkpoint_bytes),
    source_gate_sha256=_sha256_bytes(gate_bytes),
    reset_collapsed_active_std=args.reset_collapsed_active_std,
  )
  atomic_torch_save_no_clobber(migrated, args.output_checkpoint)
  print(f"[OK] Wrote StairDynamic migration: {args.output_checkpoint}")
  print(f"[OK] Source checkpoint SHA256: {migration['source_checkpoint_sha256']}")
  print(f"[OK] Source gate SHA256: {migration['source_gate_sha256']}")
  collapsed = migration["collapsed_active_actions"]
  if collapsed:
    print(f"[OK] Reset collapsed active std: {', '.join(collapsed)}")
  else:
    print("[OK] Preserved all six Stage5 action std values")
  print("[OK] Expanded actor 34->52 and critic 34->56; optimizer cleared; iter=0")


if __name__ == "__main__":
  main()
