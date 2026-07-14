#!/usr/bin/env python3
"""Train HopperTrex policies with MjLab's RSL-RL launcher."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
import subprocess
from typing import Any, Mapping

import mjlab
import torch
import tyro

PROJECT_PATH = Path(__file__).resolve().parents[2]
SRC_PATH = Path(__file__).resolve().parents[3]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

try:
  import hoppertrex_mjlab.tasks as tasks  # noqa: F401
except ImportError:
  import tasks  # noqa: F401
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.utils.os import get_checkpoint_path, get_wandb_checkpoint_path

from hoppertrex_mjlab.hybrid.config import HYBRID_ACTION_NAMES

DEFAULT_TASK = "Mjlab-HopperTrex-Balance-v0"
REPOSITORY_PATH = Path(__file__).resolve().parents[4]


HYBRID_TASK_PREFIX = 'HopperTrex-Hybrid-v2-Stage'


def _hybrid_stage(task: str) -> int | None:
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
      "Hybrid training requires a clean git worktree. Commit and push local "
      "code changes before launching the machine-room run."
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
  if stage >= 3 and not getattr(action, 'posture_map_qualified', False):
    raise ValueError(
      'Hybrid Stage3-5 training requires a qualified posture map artifact. '
      'Set HOPPERTREX_HYBRID_POSTURE_MAP_PATH before launching training.'
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


def resolve_and_validate_hybrid_resume(
  task: str,
  cfg: Any,
) -> Path | None:
  """Resolve the exact Hybrid resume checkpoint and validate its provenance."""

  stage = _hybrid_stage(task)
  if stage is None:
    return None
  if not cfg.agent.resume:
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
