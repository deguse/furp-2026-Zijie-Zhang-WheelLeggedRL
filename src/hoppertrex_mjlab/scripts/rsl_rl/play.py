#!/usr/bin/env python3
"""Play HopperTrex policies with MjLab's RSL-RL launcher."""

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
  from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
    HYBRID_TASK_IDS,
    hybrid_provenance_lines,
  )
except ImportError:
  import tasks  # noqa: F401
  from tasks.hoppertrex_hybrid_task import (  # type: ignore[no-redef]
    HYBRID_TASK_IDS,
    hybrid_provenance_lines,
  )
from mjlab.scripts.play import PlayConfig, run_play  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

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
  if task in HYBRID_TASK_IDS:
    for line in hybrid_provenance_lines(load_env_cfg(task, play=True)):
      print(line)
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
