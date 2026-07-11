#!/usr/bin/env python3
"""Train HopperTrex policies with MjLab's RSL-RL launcher."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import mjlab
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
from mjlab.tasks.registry import load_env_cfg

DEFAULT_TASK = "Mjlab-HopperTrex-Balance-v0"


HYBRID_TASK_PREFIX = 'HopperTrex-Hybrid-v2-Stage'


def validate_hybrid_training_artifacts(task: str, env_cfg: object) -> None:
  if not task.startswith(HYBRID_TASK_PREFIX):
    return
  stage_text = task.removeprefix(HYBRID_TASK_PREFIX)
  if not stage_text.isdigit() or int(stage_text) not in range(6):
    raise ValueError(f'Unsupported Hybrid v2 training task: {task}')
  stage = int(stage_text)
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
  validate_hybrid_training_artifacts(task, load_env_cfg(task, play=False))
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
  launch_training(task_id=task, args=cfg)


if __name__ == "__main__":
  main()
