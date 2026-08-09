#!/usr/bin/env python3
"""Train HopperTrex policies with MjLab's RSL-RL launcher."""

from __future__ import annotations

import math
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

DEFAULT_TASK = "Mjlab-HopperTrex-Balance-v0"
REPOSITORY_PATH = Path(__file__).resolve().parents[4]


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
  validate_hybrid_training_artifacts(task, cfg.env)
  resume_path = resolve_and_validate_hybrid_resume(task, cfg)
  if resume_path is not None:
    print(f"[PASS] Hybrid resume preflight: {resume_path}")
  launch_training(task_id=task, args=cfg)


if __name__ == "__main__":
  main()
