#!/usr/bin/env python3
"""Warm-start a fixed-leg slow-speed policy from a trained 1D wheel policy."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from migrate_balance_1d_to_turn_2d import (
  DEFAULT_EXPERIMENT_NAME,
  PROJECT_PATH,
  _create_target_checkpoint,
  _find_source_checkpoint,
  _migrate_state_dict,
  _normalize_checkpoint_state_dicts,
)


DEFAULT_TARGET_TASK = "Mjlab-HopperTrex-Balance-SlowSpeed-Easy-LinSign-v0"
DEFAULT_SOURCE_RUN_PATTERNS = (
  "robust_l2_seed{seed}",
  "push_l3_seed{seed}",
  "robust_init_seed{seed}",
  "clean_wheel_seed{seed}",
)


def _validate_expected_shapes(
  source: dict[str, Any],
  target: dict[str, Any],
  *,
  allow_unexpected: bool,
) -> None:
  checks = [
    ("source actor obs", source["actor_state_dict"]["mlp.0.weight"].shape[1], 25),
    ("source critic obs", source["critic_state_dict"]["mlp.0.weight"].shape[1], 25),
    ("source actor action", source["actor_state_dict"]["mlp.4.weight"].shape[0], 1),
    ("target actor obs", target["actor_state_dict"]["mlp.0.weight"].shape[1], 25),
    ("target critic obs", target["critic_state_dict"]["mlp.0.weight"].shape[1], 25),
    ("target actor action", target["actor_state_dict"]["mlp.4.weight"].shape[0], 1),
  ]
  failures = [
    f"{label}: got {actual}, expected {expected}"
    for label, actual, expected in checks
    if actual != expected
  ]
  if failures and not allow_unexpected:
    raise ValueError(
      "Unexpected checkpoint shapes for 1D -> fixed-leg slow-speed migration:\n"
      + "\n".join(failures)
      + "\nPass --allow-unexpected-shapes only if you have verified the task dimensions."
    )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--target-task", default=DEFAULT_TARGET_TASK)
  parser.add_argument("--output-run", default=None)
  parser.add_argument("--device", default="cpu")
  parser.add_argument(
    "--log-dir",
    type=Path,
    default=PROJECT_PATH / "logs" / "rsl_rl" / DEFAULT_EXPERIMENT_NAME,
  )
  parser.add_argument("--source-checkpoint", type=Path, default=None)
  parser.add_argument(
    "--source-run",
    default=None,
    help="Optional source run-name substring. Supports {seed}.",
  )
  parser.add_argument("--force", action="store_true")
  parser.add_argument("--allow-unexpected-shapes", action="store_true")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if args.output_run is None:
    args.output_run = f"migrated_robust_l2_to_slow_speed_easy_linsign_seed{args.seed}"

  log_dir = args.log_dir.resolve()
  output_dir = log_dir / args.output_run
  if output_dir.exists():
    if not args.force:
      raise FileExistsError(
        f"Output run already exists: {output_dir}. Use --force to overwrite."
      )
    shutil.rmtree(output_dir)

  source_run = args.source_run
  if source_run is None:
    source_run = DEFAULT_SOURCE_RUN_PATTERNS[0]
  source_checkpoint, source_run_name = _find_source_checkpoint(
    log_dir=log_dir,
    seed=args.seed,
    source_checkpoint=args.source_checkpoint,
    source_run=source_run,
  )
  source_checkpoint = source_checkpoint.resolve()
  print(f"[INFO] Source run: {source_run_name}")
  print(f"[INFO] Source checkpoint: {source_checkpoint}")

  source = torch.load(source_checkpoint, map_location=args.device, weights_only=False)
  _normalize_checkpoint_state_dicts(source)
  target, output_dir = _create_target_checkpoint(
    target_task=args.target_task,
    log_dir=log_dir,
    output_run=args.output_run,
    device=args.device,
    seed=args.seed,
  )

  try:
    _validate_expected_shapes(
      source,
      target,
      allow_unexpected=args.allow_unexpected_shapes,
    )
  except Exception:
    if output_dir.exists():
      shutil.rmtree(output_dir)
    raise

  target["actor_state_dict"], actor_report = _migrate_state_dict(
    source["actor_state_dict"],
    target["actor_state_dict"],
    "actor",
  )
  target["critic_state_dict"], critic_report = _migrate_state_dict(
    source["critic_state_dict"],
    target["critic_state_dict"],
    "critic",
  )
  target["iter"] = 0
  target["infos"] = {
    "migration": {
      "created_at": datetime.now().isoformat(timespec="seconds"),
      "type": "1d_wheel_to_fixed_leg_slow_speed_linsign",
      "target_task": args.target_task,
      "source_run": source_run_name,
      "source_checkpoint": str(source_checkpoint),
      "seed": args.seed,
      "notes": "same 1D wheel output; target adds slow-speed command/reward only",
    }
  }

  output_checkpoint = output_dir / "model_0.pt"
  torch.save(target, output_checkpoint)

  report_path = output_dir / "migration_report.txt"
  report_path.write_text(
    "\n".join(
      [
        f"target_task={args.target_task}",
        f"source_run={source_run_name}",
        f"source_checkpoint={source_checkpoint}",
        f"output_checkpoint={output_checkpoint}",
        "",
        "[actor]",
        *actor_report,
        "",
        "[critic]",
        *critic_report,
        "",
      ]
    ),
    encoding="utf-8",
  )

  print(f"[OK] Wrote migrated checkpoint: {output_checkpoint}")
  print(f"[OK] Wrote migration report: {report_path}")
  print("[NEXT] Use --agent.resume True with:")
  print(f'       --agent.load-run "{args.output_run}"')
  print('       --agent.load-checkpoint "model_0.pt"')


if __name__ == "__main__":
  main()
