#!/usr/bin/env python3
"""Train HopperTrex policies with MjLab's RSL-RL launcher."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import mjlab
import tyro

PROJECT_PATH = Path(__file__).resolve().parents[2]
if str(PROJECT_PATH) not in sys.path:
  sys.path.insert(0, str(PROJECT_PATH))

import tasks  # noqa: F401
from mjlab.scripts.train import TrainConfig, launch_training

DEFAULT_TASK = "Mjlab-HopperTrex-Balance-v0"


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
