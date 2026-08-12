#!/usr/bin/env python3
"""Train HopperTrex policies with MjLab's RSL-RL launcher."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import mjlab
import torch
import tyro

PROJECT_PATH = Path(__file__).resolve().parents[2]
SRC_PATH = Path(__file__).resolve().parents[3]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

try:
  from hoppertrex_mjlab import tasks
except ImportError:
  import tasks  # noqa: F401
from mjlab.scripts.train import TrainConfig, launch_training  # noqa: E402
from mjlab.utils.os import (  # noqa: E402
  get_checkpoint_path,
  get_wandb_checkpoint_path,
)

from hoppertrex_mjlab.hybrid.config import (  # noqa: E402
  HYBRID_ACTION_NAMES,
  HYBRID_STAGES,
  STAIR_CAMP_LQR_ALPHA05_TASK_ID,
  STAIR_CAMP_TASK_ID,
  STAIR_CAMP_TASK_IDS,
)
from hoppertrex_mjlab.hybrid.mismatch import (  # noqa: E402
  STAGE1_MISMATCH_PROFILE_VERSION,
)
from hoppertrex_mjlab.hybrid.roll_assist import (
  ROLL_ASSIST_ACTION_MASK,
  ROLL_ASSIST_CONTROLLER_SCHEDULE_HASH,
  ROLL_ASSIST_INITIAL_UPDATES,
  ROLL_ASSIST_MAX_UPDATES,
  ROLL_ASSIST_SAVE_INTERVAL,
  ROLL_ASSIST_STEPS_PER_UPDATE,
  ROLL_ASSIST_TASK_ID,
  ROLL_ASSIST_TRAINING_INFO_KEY,
  file_sha256,
  validate_extension_authorization,
  validate_roll_assist_training_record,
)
from hoppertrex_mjlab.hybrid.stair_camp_contract import (  # noqa: E402
  STAIR_CAMP_CONTRACT_SCHEMA_VERSION,
  STAIR_CAMP_CURRICULUM_INFO_KEY,
  STAIR_CAMP_EXTENSION_TOTAL_UPDATES,
  STAIR_CAMP_FRESH_UPDATES,
  STAIR_CAMP_PROGRESS_INFO_KEY,
  STAIR_CAMP_TRAINING_INFO_KEY,
  stair_camp_artifact_bindings,
  stair_camp_contract_hash,
  stair_camp_init_std,
  validate_stair_camp_progress_payload,
  validate_stair_camp_training_request,
)
from hoppertrex_mjlab.hybrid.stair_dynamic import (  # noqa: E402
  DYNAMIC_STAIR_TASK_ID,
)
from hoppertrex_mjlab.hybrid.stair_dynamic_contract import (  # noqa: E402
  DYNAMIC_STAIR_CONTRACT_SCHEMA_VERSION,
  DYNAMIC_STAIR_CURRICULUM_INFO_KEY,
  DYNAMIC_STAIR_EXTENSION_TOTAL_UPDATES,
  DYNAMIC_STAIR_MIGRATION_INFO_KEY,
  DYNAMIC_STAIR_PROBE_UPDATES,
  DYNAMIC_STAIR_PROGRESS_INFO_KEY,
  DYNAMIC_STAIR_SAVE_INTERVAL,
  DYNAMIC_STAIR_TRAINING_INFO_KEY,
  dynamic_stair_artifact_bindings,
  dynamic_stair_contract_hash,
  validate_dynamic_stair_progress_payload,
  validate_dynamic_stair_training_request,
)

DEFAULT_TASK = "Mjlab-HopperTrex-Balance-v0"
REPOSITORY_PATH = Path(__file__).resolve().parents[4]
DYNAMIC_STAIR_EXTENSION_AUTHORIZATION_PATH_ENV = (
  "HOPPERTREX_DYNAMIC_STAIR_EXTENSION_AUTHORIZATION_PATH"
)
ROLL_ASSIST_EXTENSION_AUTHORIZATION_PATH_ENV = (
  "HOPPERTREX_ROLL_ASSIST_EXTENSION_AUTHORIZATION_PATH"
)


HYBRID_TASK_PREFIX = 'HopperTrex-Hybrid-v2-Stage'

# The residual stair camp (mainline doc S5B) trains on the frozen Stage5
# classical stack but carries its own task id, which does NOT match
# HYBRID_TASK_PREFIX. Without this entry _hybrid_stage() would return None and
# every artifact guard plus the clean-worktree refusal below would be silently
# skipped for the camp (final audit finding A1-iv/B-v). The camp consumes all
# five frozen artifacts regardless of the legs-only residual mask, so it maps
# to the strictest stage.
HYBRID_NAMED_TASK_STAGES = {
  STAIR_CAMP_TASK_ID: 5,
  STAIR_CAMP_LQR_ALPHA05_TASK_ID: 5,
  DYNAMIC_STAIR_TASK_ID: 5,
  ROLL_ASSIST_TASK_ID: 5,
}


def _hybrid_stage(task: str) -> int | None:
  named_stage = HYBRID_NAMED_TASK_STAGES.get(task)
  if named_stage is not None:
    return named_stage
  if not task.startswith(HYBRID_TASK_PREFIX):
    return None
  stage_text = task.removeprefix(HYBRID_TASK_PREFIX)
  if not stage_text.isdigit() or int(stage_text) not in range(6):
    raise ValueError(f'Unsupported Hybrid v2 training task: {task}')
  return int(stage_text)


def validate_hybrid_repository_status(task: str, status: str) -> None:
  """Reject unreproducible Hybrid training from a dirty checkout."""

  if _hybrid_stage(task) is not None and status.strip():
    raise ValueError(
      "Hybrid training requires a clean git worktree. Launch only after the "
      "validated checkout has been pulled into the training machine."
    )


def _repository_status() -> str:
  completed = subprocess.run(
    ["git", "status", "--porcelain"],
    cwd=REPOSITORY_PATH,
    check=True,
    capture_output=True,
    text=True,
  )
  return completed.stdout


def validate_hybrid_training_artifacts(task: str, env_cfg: object) -> None:
  stage = _hybrid_stage(task)
  if stage is None:
    return
  if stage == 0:
    raise ValueError('Hybrid Stage0 has no PPO training phase.')
  actions = getattr(env_cfg, 'actions', {})
  action = actions.get('hybrid_wheel_leg')
  if action is None or not getattr(action, 'controller_qualified', False):
    raise ValueError(
      'Hybrid Stage1-5 training requires a qualified controller artifact. '
      'Set HOPPERTREX_HYBRID_CONTROLLER_PATH before launching training.'
    )
  if not getattr(action, 'calibration_hash', None):
    raise ValueError(
      'Hybrid Stage1-5 training requires a velocity calibration artifact. '
      'Set HOPPERTREX_HYBRID_CALIBRATION_PATH before launching training.'
    )
  if stage >= 2 and not getattr(action, 'yaw_calibration_qualified', False):
    raise ValueError(
      'Hybrid Stage2-5 training requires a probe-fitted yaw calibration '
      'artifact: the classical layer owns nominal yaw tracking from Stage '
      '2.0 on. Set HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH before launching '
      'training.'
    )
  if stage >= 3 and not getattr(action, 'posture_map_qualified', False):
    raise ValueError(
      'Hybrid Stage3-5 training requires a qualified posture map artifact. '
      'Set HOPPERTREX_HYBRID_POSTURE_MAP_PATH before launching training.'
    )
  if stage >= 3 and not getattr(action, 'station_calibration_qualified', False):
    raise ValueError(
      'Hybrid Stage3-5 training requires a probe-fitted station-keeping '
      'calibration: the classical layer owns posture station keeping from '
      'Stage 3.0 on. Set HOPPERTREX_HYBRID_STATION_CALIBRATION_PATH before '
      'launching training.'
    )
  if (
    task == ROLL_ASSIST_TASK_ID
    and getattr(env_cfg, "roll_assist_qualified", False) is not True
  ):
    raise ValueError("RollAssist training artifacts are not qualified.")
  if task == DYNAMIC_STAIR_TASK_ID:
    maneuver = getattr(action, "dynamic_stair_maneuver", None)
    if (
      getattr(env_cfg, "stair_dynamic_maneuver_qualified", False) is not True
      or maneuver is None
      or not getattr(maneuver, "maneuver_hash", "")
      or not getattr(maneuver, "bindings", None)
    ):
      raise ValueError(
        "StairDynamic training requires the CEM-selected, live-qualified "
        "dynamic_stair_maneuver artifact."
      )


def validate_hybrid_training_checkpoint(
  task: str,
  env_cfg: object,
  checkpoint: Mapping[str, Any],
) -> None:
  """Reject Hybrid training origins that bypass bootstrap or migration."""

  stage = _hybrid_stage(task)
  if stage is None:
    return
  infos = checkpoint.get("infos")
  if not isinstance(infos, Mapping):
    raise ValueError("Hybrid resume checkpoint has no provenance infos.")
  bootstrap = infos.get("hybrid_stage1_bootstrap")
  if not isinstance(bootstrap, Mapping):
    raise ValueError(
      "Hybrid resume checkpoint is missing Stage1 bootstrap provenance."
    )
  actions = getattr(env_cfg, "actions", {})
  action = actions.get("hybrid_wheel_leg")
  expected_controller = getattr(action, "controller_gain_hash", None)
  expected_calibration = getattr(action, "calibration_hash", None)
  if bootstrap.get("controller_gain_hash") != expected_controller:
    raise ValueError(
      "Hybrid checkpoint controller hash does not match the training environment."
    )
  if bootstrap.get("calibration_hash") != expected_calibration:
    raise ValueError(
      "Hybrid checkpoint calibration hash does not match the training environment."
    )
  if tuple(bootstrap.get("action_order", ())) != HYBRID_ACTION_NAMES:
    raise ValueError("Hybrid checkpoint action order does not match Hybrid v2.")
  if stage == 1:
    if bootstrap.get("stage") != 1 or bootstrap.get("task") != task:
      raise ValueError("Stage1 training requires a Stage1 bootstrap checkpoint.")
    expected_profile = getattr(env_cfg, "stage1_profile_version", None)
    extension = infos.get("hybrid_stage1_extension")
    if expected_profile == STAGE1_MISMATCH_PROFILE_VERSION:
      if not isinstance(extension, Mapping):
        raise ValueError(
          "Stage1-B training requires a prepare_hybrid_stage1_extension "
          "checkpoint."
        )
      if extension.get("target_profile_version") != expected_profile:
        raise ValueError(
          "Stage1 extension profile does not match the training environment."
        )
      source_std = extension.get("source_action_std")
      if not isinstance(source_std, list) or len(source_std) != 6:
        raise ValueError("Stage1 extension is missing the six-action std audit.")
      collapsed = extension.get("collapsed_active_actions")
      reset_collapsed = extension.get("reset_collapsed_active_std")
      if not isinstance(collapsed, list):
        raise ValueError("Stage1 extension is missing collapsed-action audit data.")
      if collapsed and reset_collapsed is not True:
        raise ValueError(
          "Stage1 extension contains collapsed exploration that was not reset."
        )
    return

  migration = infos.get("hybrid_stage_migration")
  if not isinstance(migration, Mapping):
    raise ValueError(
      f"Hybrid Stage{stage} requires a migrate_hybrid_stage checkpoint."
    )
  if migration.get("target_stage") != stage:
    raise ValueError(
      "Hybrid migration target does not match the requested training stage."
    )
  expected_action_scales = tuple(
    float(value)
    for value in getattr(
      action, "action_scales", HYBRID_STAGES[stage].action_scales
    )
  )
  recorded_action_scales = migration.get("target_action_scales")
  if recorded_action_scales is None:
    if expected_action_scales != HYBRID_STAGES[stage].action_scales:
      raise ValueError(
        "Hybrid migration has no action-scale provenance for the requested "
        "experimental authority. Regenerate the handoff checkpoint."
      )
  elif tuple(float(value) for value in recorded_action_scales) != (
    expected_action_scales
  ):
    raise ValueError(
      "Hybrid migration action scales do not match the training environment."
    )
  source_stage = migration.get("source_stage")
  if not isinstance(source_stage, int) or source_stage != stage - 1:
    raise ValueError(
      "Hybrid migration must come from the immediately preceding stage."
    )
  source_std = migration.get("source_action_std")
  if not isinstance(source_std, list) or len(source_std) != 6:
    raise ValueError("Hybrid migration is missing the six-action std audit.")
  collapsed = migration.get("collapsed_active_actions")
  reset_collapsed = migration.get("reset_collapsed_active_std")
  if not isinstance(collapsed, list):
    raise ValueError("Hybrid migration is missing collapsed-action audit data.")
  if collapsed and reset_collapsed is not True:
    raise ValueError(
      "Hybrid migration contains collapsed active actions that were not reset."
    )
  if stage == 2:
    required_gate_fields = (
      "source_checkpoint_sha256",
      "source_gate",
      "source_gate_sha256",
    )
    if any(not migration.get(field) for field in required_gate_fields):
      raise ValueError(
        "Hybrid Stage2 migration is missing its Stage1 formal gate audit."
      )
    if (
      migration.get("source_gate_profile") != "formal"
      or migration.get("source_gate_suite") != "residual"
      or migration.get("source_gate_stage1_profile_version")
      != STAGE1_MISMATCH_PROFILE_VERSION
    ):
      raise ValueError(
        "Hybrid Stage2 migration does not reference the current formal "
        "Stage1-B gate profile."
      )
  # The yaw hash lives in the migration record, not the Stage1 bootstrap:
  # frozen Stage1-B checkpoints predate yaw calibration and stay valid, while
  # every stage>=2 training origin must have been migrated against the same
  # yaw artifact the training environment now loads.
  if migration.get("yaw_calibration_hash") != getattr(
    action, "yaw_calibration_hash", None
  ):
    raise ValueError(
      "Hybrid migration yaw calibration hash does not match the training "
      "environment."
    )
  if stage >= 3:
    if migration.get("posture_map_hash") != getattr(
      action, "posture_map_hash", None
    ):
      raise ValueError(
        "Hybrid migration posture map hash does not match the training "
        "environment."
      )
    migration_artifact_hash = migration.get("posture_artifact_hash")
    if migration_artifact_hash is not None and migration_artifact_hash != getattr(
      action, "posture_artifact_hash", None
    ):
      raise ValueError(
        "Hybrid migration posture artifact hash does not match the training "
        "command envelope."
      )
    if migration.get("station_calibration_hash") != getattr(
      action, "station_calibration_hash", None
    ):
      raise ValueError(
        "Hybrid migration station calibration hash does not match the "
        "training environment."
      )


STAIR_CAMP_ENV_MARKERS = (
  "stair_camp_task_id",
  "stair_camp_zero_initialize_actor_output",
  "stair_camp_training_contract",
  "stair_camp_contract_schema_version",
  "stair_camp_contract_sha256",
  "stair_camp_failure_ladder_variant",
)


def restore_stair_camp_markers(task: str, source: Any, parsed: Any) -> None:
  """Re-attach the camp markers that a tyro CLI round-trip drops.

  The markers are set with `setattr` and are therefore NOT dataclass fields,
  while `tyro.cli` reconstructs the config from its fields alone. The parsed
  object is a different instance with every marker missing, so every guard
  keyed on them -- the training-request validator, the contract binding, the
  runner's camp detection -- fails at launch with "StairCamp task marker is
  missing", after Validate has already passed. Measured on the training host
  2026-08-10; the registered command line cannot start without this.

  The values are copied from the pre-parse config built by the registry, so
  this restores exactly what registration declared and cannot introduce a
  value the CLI could have chosen: none of these markers is a CLI flag.
  """

  if task not in STAIR_CAMP_TASK_IDS:
    return
  for name in STAIR_CAMP_ENV_MARKERS:
    if not hasattr(source, name):
      continue
    if hasattr(parsed, name):
      continue
    setattr(parsed, name, getattr(source, name))
  if getattr(parsed, "stair_camp_task_id", None) != task:
    raise ValueError(
      "StairCamp marker restoration did not reproduce the registered task."
    )


ROLL_ASSIST_ENV_MARKERS = (
  "roll_assist_task_id", "roll_assist_training_contract", "roll_assist_qualified",
  "roll_assist_hpass_m", "roll_assist_hnext_m", "roll_assist_flat_env_count",
  "roll_assist_r0_path", "roll_assist_r0_sha256",
  "roll_assist_r0_git_sha", "roll_assist_r0_schedule_hash",
  "roll_assist_reward_calibration_path", "roll_assist_reward_calibration_sha256",
  "roll_assist_reward_calibration_content_sha256", "roll_assist_progress_weight", "roll_assist_success_weight", "roll_assist_settle_steps",
  "roll_assist_zero_initialize_actor_output",
)


def restore_roll_assist_markers(task: str, source: Any, parsed: Any) -> None:
  if task != ROLL_ASSIST_TASK_ID:
    return
  for name in ROLL_ASSIST_ENV_MARKERS:
    if hasattr(source, name) and not hasattr(parsed, name):
      setattr(parsed, name, getattr(source, name))
  if getattr(parsed, "roll_assist_task_id", None) != ROLL_ASSIST_TASK_ID:
    raise ValueError("RollAssist marker restoration failed.")


def validate_roll_assist_training_request(env_cfg: Any, agent_cfg: Any, *, resume: bool) -> None:
  if getattr(env_cfg, "roll_assist_task_id", None) != ROLL_ASSIST_TASK_ID:
    raise ValueError("RollAssist task marker is missing.")
  if getattr(env_cfg, "roll_assist_qualified", False) is not True:
    raise ValueError("RollAssist requires formal R0 and reward-calibration artifacts.")
  if getattr(env_cfg, "roll_assist_r0_sha256", None) is None:
    raise ValueError("RollAssist R0 byte binding is missing.")
  if getattr(env_cfg, "roll_assist_r0_git_sha", None) != _repository_head():
    raise ValueError("RollAssist R0 Git SHA differs from the training checkout.")
  if (
    getattr(env_cfg, "roll_assist_r0_schedule_hash", None)
    != ROLL_ASSIST_CONTROLLER_SCHEDULE_HASH
  ):
    raise ValueError("RollAssist R0 does not bind the frozen C1 schedule.")
  reward_path = Path(str(getattr(env_cfg, "roll_assist_reward_calibration_path", "")))
  if (
    not reward_path.is_file()
    or file_sha256(reward_path)
    != getattr(env_cfg, "roll_assist_reward_calibration_sha256", None)
  ):
    raise ValueError("RollAssist reward-calibration file bytes drifted.")
  reward_payload = json.loads(reward_path.read_text(encoding="utf-8-sig"))
  if (
    not isinstance(reward_payload, Mapping)
    or reward_payload.get("calibration_sha256")
    != getattr(env_cfg, "roll_assist_reward_calibration_content_sha256", None)
  ):
    raise ValueError("RollAssist reward-calibration content binding drifted.")
  action = env_cfg.actions.get("hybrid_wheel_leg")
  if tuple(getattr(action, "action_mask", ())) != ROLL_ASSIST_ACTION_MASK:
    raise ValueError("RollAssist runtime wheel mask drifted.")
  distribution = getattr(agent_cfg.actor, "distribution_cfg", None)
  if not isinstance(distribution, Mapping) or tuple(distribution.get("active_mask", ())) != ROLL_ASSIST_ACTION_MASK:
    raise ValueError("RollAssist PPO active mask drifted.")
  if agent_cfg.seed != 1 or agent_cfg.num_steps_per_env != ROLL_ASSIST_STEPS_PER_UPDATE:
    raise ValueError("RollAssist is pinned to seed1 and 24 steps/update.")
  if agent_cfg.save_interval != ROLL_ASSIST_SAVE_INTERVAL:
    raise ValueError("RollAssist save interval must remain 25 updates.")
  if resume:
    if not 151 <= agent_cfg.max_iterations <= ROLL_ASSIST_MAX_UPDATES:
      raise ValueError(
        "RollAssist resume target must be an authorized selected+100 total up to 500."
      )
  elif agent_cfg.max_iterations != ROLL_ASSIST_INITIAL_UPDATES:
    raise ValueError("Initial RollAssist launch is exactly 100 updates.")


DYNAMIC_STAIR_ENV_MARKERS = (
  "stair_dynamic_task_id",
  "stair_dynamic_training_contract",
  "stair_dynamic_contract_schema_version",
  "stair_dynamic_contract_sha256",
  "stair_dynamic_maneuver_qualified",
  "stair_dynamic_maneuver_bindings",
)


def restore_stair_dynamic_markers(task: str, source: Any, parsed: Any) -> None:
  """Restore non-dataclass v3 markers after tyro reconstructs the env cfg."""

  if task != DYNAMIC_STAIR_TASK_ID:
    return
  for name in DYNAMIC_STAIR_ENV_MARKERS:
    if hasattr(source, name) and not hasattr(parsed, name):
      setattr(parsed, name, getattr(source, name))
  if getattr(parsed, "stair_dynamic_task_id", None) != DYNAMIC_STAIR_TASK_ID:
    raise ValueError("StairDynamic marker restoration failed.")


def _repository_head() -> str:
  completed = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=REPOSITORY_PATH,
    check=True,
    capture_output=True,
    text=True,
  )
  return completed.stdout.strip()


def validate_stair_camp_extension_checkpoint(
  cfg: Any,
  checkpoint: Mapping[str, Any],
) -> None:
  """Accept only the registered camp's own 1000 -> 3000 continuation."""

  def exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
      raise ValueError(f"StairCamp extension {name} must be an integer.")  # noqa: TRY004
    if value < minimum:
      raise ValueError(f"StairCamp extension {name} is invalid.")
    return value

  if exact_int(cfg.agent.max_iterations, name="target") != STAIR_CAMP_EXTENSION_TOTAL_UPDATES:
    raise ValueError("StairCamp extension target must be exactly 3000 iterations.")
  infos = checkpoint.get("infos")
  if not isinstance(infos, Mapping):
    raise ValueError("StairCamp extension checkpoint has no provenance infos.")  # noqa: TRY004
  record = infos.get(STAIR_CAMP_TRAINING_INFO_KEY)
  if not isinstance(record, Mapping):
    raise ValueError("StairCamp extension requires its own camp checkpoint.")  # noqa: TRY004
  expected_record_fields = {
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
  if set(record) != expected_record_fields:
    raise ValueError("StairCamp extension training provenance schema drifted.")
  if exact_int(record.get("schema_version"), name="schema_version") != STAIR_CAMP_CONTRACT_SCHEMA_VERSION:
    raise ValueError("StairCamp extension checkpoint schema is unsupported.")
  if record.get("task") != STAIR_CAMP_TASK_ID:
    raise ValueError("StairCamp extension checkpoint task does not match.")
  source_seed = exact_int(record.get("training_seed"), name="training_seed")
  if source_seed != cfg.agent.seed:
    raise ValueError("StairCamp extension checkpoint seed does not match.")
  current_git_sha = _repository_head()
  if record.get("git_sha") != current_git_sha:
    raise ValueError("StairCamp extension checkpoint Git SHA does not match.")
  expected_contract = stair_camp_contract_hash(cfg.env, cfg.agent)
  if record.get("contract_sha256") != expected_contract:
    raise ValueError("StairCamp extension checkpoint contract does not match.")
  if getattr(cfg.env, "stair_camp_contract_sha256", None) != expected_contract:
    raise ValueError("StairCamp extension environment contract is not bound.")
  expected_artifacts = stair_camp_artifact_bindings(cfg.env)
  if record.get("artifact_bindings") != expected_artifacts:
    raise ValueError("StairCamp extension artifact bindings do not match.")
  actions = getattr(cfg.env, "actions", {})
  action = actions.get("hybrid_wheel_leg") if isinstance(actions, Mapping) else None
  expected_scales = [float(value) for value in getattr(action, "action_scales", ())]
  if record.get("action_scales") != expected_scales:
    raise ValueError("StairCamp extension action scales do not match.")
  init_std = record.get("init_std")
  if (
    record.get("zero_initialized_deterministic_mean") is not True
    or isinstance(init_std, bool)
    or not isinstance(init_std, (int, float))
    or not math.isfinite(float(init_std))
    or float(init_std) != stair_camp_init_std(cfg.agent)
  ):
    raise ValueError("StairCamp extension initialization provenance is invalid.")
  completed_updates = exact_int(
    record.get("completed_updates"), name="completed_updates"
  )
  checkpoint_iteration = exact_int(checkpoint.get("iter"), name="iteration")
  if (
    completed_updates != STAIR_CAMP_FRESH_UPDATES
    or checkpoint_iteration + 1 != completed_updates
  ):
    raise ValueError(
      "StairCamp extension must start from the completed 1000-update checkpoint."
    )
  curriculum = infos.get(STAIR_CAMP_CURRICULUM_INFO_KEY)
  if not isinstance(curriculum, Mapping):
    raise ValueError("StairCamp extension checkpoint has no curriculum state.")  # noqa: TRY004
  progress = infos.get(STAIR_CAMP_PROGRESS_INFO_KEY)
  if not isinstance(progress, Mapping):
    raise ValueError("StairCamp extension checkpoint has no progress snapshot.")  # noqa: TRY004
  validate_stair_camp_progress_payload(progress, curriculum)
  env_state = infos.get("env_state")
  if not isinstance(env_state, Mapping):
    raise ValueError("StairCamp extension checkpoint has no environment state.")  # noqa: TRY004
  exact_int(
    env_state.get("common_step_counter"),
    name="common_step_counter",
  )
  if not isinstance(checkpoint.get("actor_state_dict"), Mapping):
    raise ValueError("StairCamp extension checkpoint has no actor state.")  # noqa: TRY004
  if not isinstance(checkpoint.get("critic_state_dict"), Mapping):
    raise ValueError("StairCamp extension checkpoint has no critic state.")  # noqa: TRY004

def _dynamic_exact_int(
  value: object,
  *,
  name: str,
  minimum: int = 0,
) -> int:
  if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
    raise ValueError(f"StairDynamic checkpoint {name} is invalid.")
  return int(value)


def _validate_dynamic_migration_network(
  checkpoint: Mapping[str, Any],
  migration: Mapping[str, Any],
) -> None:
  actor = checkpoint.get("actor_state_dict")
  critic = checkpoint.get("critic_state_dict")
  if not isinstance(actor, Mapping) or not isinstance(critic, Mapping):
    raise TypeError("StairDynamic migration is missing actor/critic state.")
  actor_key = migration.get("actor_first_layer")
  critic_key = migration.get("critic_first_layer")
  if not isinstance(actor_key, str) or not isinstance(critic_key, str):
    raise TypeError("StairDynamic migration first-layer audit is missing.")
  actor_weight = actor.get(actor_key)
  critic_weight = critic.get(critic_key)
  if not isinstance(actor_weight, torch.Tensor) or actor_weight.ndim != 2:
    raise ValueError("StairDynamic migrated actor first layer is invalid.")
  if not isinstance(critic_weight, torch.Tensor) or critic_weight.ndim != 2:
    raise ValueError("StairDynamic migrated critic first layer is invalid.")
  if actor_weight.shape[1] != 52 or critic_weight.shape[1] != 56:
    raise ValueError("StairDynamic migrated observation widths drifted.")
  if torch.count_nonzero(actor_weight[:, 34:]).item() != 0:
    raise ValueError("StairDynamic migrated actor new columns are not zero.")
  if torch.count_nonzero(critic_weight[:, 34:]).item() != 0:
    raise ValueError("StairDynamic migrated critic new columns are not zero.")
  optimizer = checkpoint.get("optimizer_state_dict")
  if (
    not isinstance(optimizer, Mapping)
    or not isinstance(optimizer.get("state"), Mapping)
    or len(optimizer["state"]) != 0
  ):
    raise ValueError("StairDynamic migration optimizer state was not cleared.")


def validate_stair_dynamic_migration_checkpoint(
  cfg: Any,
  checkpoint: Mapping[str, Any],
) -> None:
  """Accept only the formal Stage5-100 to v3 zero-column migration."""

  if _dynamic_exact_int(cfg.agent.max_iterations, name="target", minimum=1) != DYNAMIC_STAIR_PROBE_UPDATES:
    raise ValueError("Initial StairDynamic training target must be exactly 100 updates.")
  if _dynamic_exact_int(checkpoint.get("iter"), name="iteration") != 0:
    raise ValueError("StairDynamic migration must reset iter to zero.")
  infos = checkpoint.get("infos")
  if not isinstance(infos, Mapping):
    raise TypeError("StairDynamic migration has no provenance infos.")
  if STAIR_CAMP_TRAINING_INFO_KEY in infos:
    raise ValueError("Round1 StairCamp checkpoints cannot initialize v3.")
  migration = infos.get(DYNAMIC_STAIR_MIGRATION_INFO_KEY)
  if not isinstance(migration, Mapping):
    raise TypeError("StairDynamic requires the dedicated Stage5 migration.")
  bindings = getattr(cfg.env, "stair_dynamic_maneuver_bindings", None)
  if not isinstance(bindings, Mapping):
    raise TypeError("StairDynamic environment has no maneuver bindings.")
  if (
    migration.get("source_task") != "HopperTrex-Hybrid-v2-Stage5"
    or migration.get("target_task") != DYNAMIC_STAIR_TASK_ID
    or migration.get("source_seed") != 1
    or migration.get("source_completed_updates") != 100
    or migration.get("source_actor_width") != 34
    or migration.get("target_actor_width") != 52
    or migration.get("source_critic_width") != 34
    or migration.get("target_critic_width") != 56
    or migration.get("source_checkpoint_sha256")
    != bindings.get("stage5_checkpoint_sha256")
    or migration.get("source_gate_sha256")
    != bindings.get("stage5_formal_gate_sha256")
  ):
    raise ValueError("StairDynamic migration does not match the frozen maneuver.")
  if bindings.get("git_sha") != _repository_head():
    raise ValueError("StairDynamic maneuver Git SHA does not match checkout.")
  env_state = infos.get("env_state")
  if (
    not isinstance(env_state, Mapping)
    or env_state.get("common_step_counter") != 0
  ):
    raise ValueError("StairDynamic migration must reset common_step_counter.")
  _validate_dynamic_migration_network(checkpoint, migration)
  # Reuse the mature Stage5 provenance/artifact checks on the preserved infos.
  validate_hybrid_training_checkpoint(
    "HopperTrex-Hybrid-v2-Stage5",
    cfg.env,
    checkpoint,
  )


def validate_stair_dynamic_extension_checkpoint(
  cfg: Any,
  checkpoint: Mapping[str, Any],
) -> None:
  """Accept only an own seed-1 K=3-selected checkpoint for total budget 500."""

  if _dynamic_exact_int(cfg.agent.max_iterations, name="target", minimum=1) != DYNAMIC_STAIR_EXTENSION_TOTAL_UPDATES:
    raise ValueError("StairDynamic extension target must be exactly 500 updates.")
  infos = checkpoint.get("infos")
  if not isinstance(infos, Mapping):
    raise TypeError("StairDynamic extension has no provenance infos.")
  record = infos.get(DYNAMIC_STAIR_TRAINING_INFO_KEY)
  if not isinstance(record, Mapping):
    raise TypeError("StairDynamic extension requires its own v3 checkpoint.")
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
  if set(record) != expected_fields:
    raise ValueError("StairDynamic extension provenance schema drifted.")
  if _dynamic_exact_int(record.get("schema_version"), name="schema_version") != DYNAMIC_STAIR_CONTRACT_SCHEMA_VERSION:
    raise ValueError("StairDynamic extension schema is unsupported.")
  if record.get("task") != DYNAMIC_STAIR_TASK_ID:
    raise ValueError("StairDynamic extension task does not match.")
  if _dynamic_exact_int(record.get("training_seed"), name="training_seed") != 1 or cfg.agent.seed != 1:
    raise ValueError("StairDynamic extension seed must remain 1.")
  if record.get("git_sha") != _repository_head():
    raise ValueError("StairDynamic extension Git SHA does not match.")
  expected_contract = dynamic_stair_contract_hash(cfg.env, cfg.agent)
  if (
    record.get("contract_sha256") != expected_contract
    or getattr(cfg.env, "stair_dynamic_contract_sha256", None)
    != expected_contract
  ):
    raise ValueError("StairDynamic extension contract does not match.")
  if record.get("artifact_bindings") != dynamic_stair_artifact_bindings(cfg.env):
    raise ValueError("StairDynamic extension artifact bindings do not match.")
  action = cfg.env.actions.get("hybrid_wheel_leg")
  maneuver = getattr(action, "dynamic_stair_maneuver", None)
  bindings = getattr(cfg.env, "stair_dynamic_maneuver_bindings", {})
  if (
    record.get("action_scales")
    != [float(value) for value in getattr(action, "action_scales", ())]
    or record.get("maneuver_sha256") != getattr(maneuver, "maneuver_hash", None)
    or record.get("source_stage5_checkpoint_sha256")
    != bindings.get("stage5_checkpoint_sha256")
    or record.get("source_stage5_gate_sha256")
    != bindings.get("stage5_formal_gate_sha256")
    or record.get("stage5_prefix_preserved_and_new_columns_zero") is not True
  ):
    raise ValueError("StairDynamic extension control provenance drifted.")
  completed = _dynamic_exact_int(
    record.get("completed_updates"),
    name="completed_updates",
    minimum=1,
  )
  iteration = _dynamic_exact_int(checkpoint.get("iter"), name="iteration")
  allowed_selected_updates = {
    DYNAMIC_STAIR_PROBE_UPDATES,
    DYNAMIC_STAIR_PROBE_UPDATES - DYNAMIC_STAIR_SAVE_INTERVAL + 1,
    DYNAMIC_STAIR_PROBE_UPDATES - 2 * DYNAMIC_STAIR_SAVE_INTERVAL + 1,
  }
  if completed not in allowed_selected_updates or iteration + 1 != completed:
    raise ValueError(
      "StairDynamic extension must start from the selected 51/76/100-update "
      "K=3 checkpoint."
    )
  curriculum = infos.get(DYNAMIC_STAIR_CURRICULUM_INFO_KEY)
  progress = infos.get(DYNAMIC_STAIR_PROGRESS_INFO_KEY)
  if not isinstance(curriculum, Mapping) or not isinstance(progress, Mapping):
    raise TypeError("StairDynamic extension is missing curriculum/progress state.")
  validate_dynamic_stair_progress_payload(progress, curriculum)
  env_state = infos.get("env_state")
  if not isinstance(env_state, Mapping):
    raise TypeError("StairDynamic extension is missing environment state.")
  _dynamic_exact_int(
    env_state.get("common_step_counter"),
    name="common_step_counter",
  )
  migration = infos.get(DYNAMIC_STAIR_MIGRATION_INFO_KEY)
  if not isinstance(migration, Mapping):
    raise TypeError("StairDynamic extension lost Stage5 migration provenance.")
  if not isinstance(checkpoint.get("actor_state_dict"), Mapping) or not isinstance(
    checkpoint.get("critic_state_dict"), Mapping
  ):
    raise TypeError("StairDynamic extension is missing actor/critic state.")



def validate_stair_dynamic_extension_authorization(
  checkpoint_path: Path,
  checkpoint: Mapping[str, Any],
) -> dict[str, object]:
  """Require K=3 selection plus formal retention and 44/48 evidence."""

  value = os.environ.get(DYNAMIC_STAIR_EXTENSION_AUTHORIZATION_PATH_ENV)
  if value is None or not value.strip():
    raise ValueError(
      "StairDynamic 500-update extension requires "
      f"{DYNAMIC_STAIR_EXTENSION_AUTHORIZATION_PATH_ENV}."
    )
  path = Path(value).expanduser().resolve()
  if not path.is_file():
    raise ValueError(f"StairDynamic extension authorization does not exist: {path}.")
  try:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
  except (OSError, json.JSONDecodeError) as exc:
    raise ValueError("StairDynamic extension authorization is not valid JSON.") from exc
  if not isinstance(raw, Mapping):
    raise TypeError("StairDynamic extension authorization must be a JSON object.")
  from hoppertrex_mjlab.scripts.rsl_rl.evaluate_stair_dynamic import (
    validate_extension_authorization,
  )

  authorization = validate_extension_authorization(raw)
  resolved_checkpoint = checkpoint_path.resolve()
  selected_path = Path(str(authorization["selected_checkpoint_file"])).resolve()
  selected_sha = str(authorization["selected_checkpoint_sha256"])
  actual_sha = hashlib.sha256(resolved_checkpoint.read_bytes()).hexdigest()
  infos = checkpoint.get("infos")
  if not isinstance(infos, Mapping):
    raise TypeError("StairDynamic selected checkpoint has no provenance infos.")
  training = infos.get(DYNAMIC_STAIR_TRAINING_INFO_KEY)
  if not isinstance(training, Mapping):
    raise TypeError("StairDynamic selected checkpoint has no training record.")
  if (
    selected_path != resolved_checkpoint
    or selected_sha != actual_sha
    or authorization["selected_completed_updates"]
    != training.get("completed_updates")
    or authorization["target_total_updates"]
    != DYNAMIC_STAIR_EXTENSION_TOTAL_UPDATES
  ):
    raise ValueError(
      "StairDynamic extension checkpoint differs from the authorized K=3 selection."
    )
  return authorization


def validate_roll_assist_extension_checkpoint(
  cfg: Any,
  checkpoint_path: Path,
  checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
  """Bind one resume checkpoint to one passing 100-update authorization."""

  infos = checkpoint.get("infos")
  if not isinstance(infos, Mapping):
    raise TypeError("RollAssist resume checkpoint has no provenance infos.")
  record = infos.get(ROLL_ASSIST_TRAINING_INFO_KEY)
  if not isinstance(record, Mapping):
    raise TypeError("RollAssist resume checkpoint has no training record.")
  completed = validate_roll_assist_training_record(
    record,
    git_sha=_repository_head(),
    r0_sha256=str(getattr(cfg.env, "roll_assist_r0_sha256", "")),
    reward_calibration_sha256=str(
      getattr(cfg.env, "roll_assist_reward_calibration_sha256", "")
    ),
    action_scales=tuple(
      float(value)
      for value in cfg.env.actions["hybrid_wheel_leg"].action_scales
    ),
  )
  iteration = checkpoint.get("iter")
  if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration + 1 != completed:
    raise ValueError("RollAssist resume checkpoint iteration/update count drifted.")
  curriculum = infos.get("roll_assist_curriculum")
  progress = infos.get("roll_assist_progress")
  env_state = infos.get("env_state")
  if not all(isinstance(value, Mapping) for value in (curriculum, progress, env_state)):
    raise TypeError("RollAssist resume checkpoint lacks restorable runtime state.")
  if (
    curriculum.get("decision_made") is not True
    or float(progress.get("active_height_m", -1.0))
    != float(record["active_height_m"])
    or env_state.get("common_step_counter")
    != completed * ROLL_ASSIST_STEPS_PER_UPDATE
  ):
    raise ValueError("RollAssist resume runtime state drifted from training provenance.")
  value = os.environ.get(ROLL_ASSIST_EXTENSION_AUTHORIZATION_PATH_ENV)
  if value is None or not value.strip():
    raise ValueError(
      "RollAssist resume requires "
      f"{ROLL_ASSIST_EXTENSION_AUTHORIZATION_PATH_ENV}."
    )
  authorization_path = Path(value).expanduser().resolve()
  try:
    raw = json.loads(authorization_path.read_text(encoding="utf-8-sig"))
  except (OSError, json.JSONDecodeError) as exc:
    raise ValueError("RollAssist extension authorization is unreadable.") from exc
  if not isinstance(raw, Mapping):
    raise TypeError("RollAssist extension authorization must be a JSON object.")
  authorization = validate_extension_authorization(raw)
  resolved_checkpoint = checkpoint_path.resolve()
  if (
    Path(str(authorization["selected_checkpoint_file"])).resolve()
    != resolved_checkpoint
    or authorization["selected_checkpoint_sha256"]
    != file_sha256(resolved_checkpoint)
    or authorization["selected_completed_updates"] != completed
    or authorization["target_total_updates"] != cfg.agent.max_iterations
  ):
    raise ValueError(
      "RollAssist resume differs from its selected checkpoint or authorized block."
    )
  return authorization


def resolve_and_validate_hybrid_resume(
  task: str,
  cfg: Any,
) -> Path | None:
  """Resolve the exact Hybrid resume checkpoint and validate its provenance."""

  stage = _hybrid_stage(task)
  if stage is None:
    return None
  resume = cfg.agent.resume
  if task in STAIR_CAMP_TASK_IDS:
    validate_stair_camp_training_request(cfg.env, cfg.agent, resume=resume)
    if task == STAIR_CAMP_LQR_ALPHA05_TASK_ID or resume is False:
      return None
  elif task == DYNAMIC_STAIR_TASK_ID:
    validate_dynamic_stair_training_request(cfg.env, cfg.agent, resume=resume)
  elif task == ROLL_ASSIST_TASK_ID:
    validate_roll_assist_training_request(cfg.env, cfg.agent, resume=resume)
    if not resume:
      return None
  elif not resume:
    raise ValueError(
      f"Hybrid Stage{stage} training must resume from a qualified bootstrap "
      "or migrated checkpoint."
    )
  log_root = (Path(cfg.log_root) / cfg.agent.experiment_name).resolve()
  if cfg.wandb_run_path is None:
    checkpoint_path = get_checkpoint_path(
      log_root,
      cfg.agent.load_run,
      cfg.agent.load_checkpoint,
    )
  else:
    checkpoint_path, _was_cached = get_wandb_checkpoint_path(
      log_root,
      Path(cfg.wandb_run_path),
      cfg.wandb_checkpoint_name,
    )
  checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
  )
  if not isinstance(checkpoint, Mapping):
    raise ValueError("Hybrid resume checkpoint must contain a mapping.")
  if task == STAIR_CAMP_TASK_ID:
    validate_stair_camp_extension_checkpoint(cfg, checkpoint)
  elif task == ROLL_ASSIST_TASK_ID:
    validate_roll_assist_extension_checkpoint(cfg, checkpoint_path, checkpoint)
  elif task == DYNAMIC_STAIR_TASK_ID:
    infos = checkpoint.get("infos")
    if not isinstance(infos, Mapping):
      raise ValueError("StairDynamic resume checkpoint has no provenance infos.")
    if DYNAMIC_STAIR_TRAINING_INFO_KEY in infos:
      validate_stair_dynamic_extension_checkpoint(cfg, checkpoint)
      validate_stair_dynamic_extension_authorization(
        checkpoint_path, checkpoint
      )
    else:
      validate_stair_dynamic_migration_checkpoint(cfg, checkpoint)
  else:
    validate_hybrid_training_checkpoint(task, cfg.env, checkpoint)
  return checkpoint_path


def _normalize_argv() -> tuple[str, list[str]]:
  args = sys.argv[1:]
  task = DEFAULT_TASK
  if "--task" in args:
    idx = args.index("--task")
    task = args[idx + 1]
    args = args[:idx] + args[idx + 2 :]
  elif args and not args[0].startswith("-"):
    task = args[0]
    args = args[1:]
  return task, args


def main() -> None:
  task, remaining = _normalize_argv()
  if (
    _hybrid_stage(task) is not None
    and not any(arg in ("-h", "--help") for arg in remaining)
  ):
    validate_hybrid_repository_status(task, _repository_status())
  default_cfg = replace(
    TrainConfig.from_task(task),
    log_root=str(PROJECT_PATH / "logs" / "rsl_rl"),
  )

  cfg = tyro.cli(
    TrainConfig,
    args=remaining,
    default=default_cfg,
    prog=f"{sys.argv[0]} {task}",
    config=mjlab.TYRO_FLAGS,
  )
  restore_stair_camp_markers(task, default_cfg.env, cfg.env)
  restore_stair_dynamic_markers(task, default_cfg.env, cfg.env)
  restore_roll_assist_markers(task, default_cfg.env, cfg.env)
  validate_hybrid_training_artifacts(task, cfg.env)
  resume_path = resolve_and_validate_hybrid_resume(task, cfg)
  if resume_path is not None:
    print(f"[PASS] Hybrid resume preflight: {resume_path}")
  launch_training(task_id=task, args=cfg)


if __name__ == "__main__":
  main()
