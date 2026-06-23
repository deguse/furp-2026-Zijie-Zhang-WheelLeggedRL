#!/usr/bin/env python3
"""Play HopperTrex policies with MjLab's RSL-RL launcher."""

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
from mjlab.scripts.play import PlayConfig, run_play

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
    PlayConfig(),
    agent="zero",
    log_root=str(PROJECT_PATH / "logs" / "rsl_rl"),
  )
  cfg = tyro.cli(
    PlayConfig,
    args=remaining,
    default=default_cfg,
    prog=f"{sys.argv[0]} {task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_play(task_id=task, cfg=cfg)


if __name__ == "__main__":
  main()

