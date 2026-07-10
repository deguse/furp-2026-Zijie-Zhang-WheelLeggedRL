#!/usr/bin/env python3
"""Migrate a fixed-6D Hybrid v2 checkpoint to a later capability stage."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import math
from pathlib import Path
from typing import Any

import torch

from hoppertrex_mjlab.hybrid.config import (
  HYBRID_ACTION_NAMES,
  HYBRID_ACTION_STD,
  HYBRID_STAGES,
)


def _action_head_keys(actor_state: dict[str, torch.Tensor]) -> tuple[str, str]:
  candidates: list[tuple[str, str]] = []
  for key, value in actor_state.items():
    if not key.endswith(".weight") or value.ndim != 2 or value.shape[0] != 6:
      continue
    bias_key = f"{key[:-len('.weight')]}.bias"
    bias = actor_state.get(bias_key)
    if bias is not None and bias.shape == (6,):
      candidates.append((key, bias_key))
  if len(candidates) != 1:
    raise ValueError(
      "Expected exactly one six-row actor output head, "
      f"found {len(candidates)} candidates."
    )
  return candidates[0]


def _validate_transition(source_stage: int, target_stage: int) -> list[int]:
  if source_stage not in HYBRID_STAGES or target_stage not in HYBRID_STAGES:
    raise ValueError("Hybrid source and target stages must be in the range 0..5.")
  if target_stage <= source_stage:
    raise ValueError("Hybrid checkpoint migration must be a forward stage transition.")
  source_mask = HYBRID_STAGES[source_stage].action_mask
  target_mask = HYBRID_STAGES[target_stage].action_mask
  if any(source and not target for source, target in zip(source_mask, target_mask)):
    raise ValueError("Hybrid stage transition must not disable an active action.")
  return [
    index
    for index, (source, target) in enumerate(zip(source_mask, target_mask))
    if target and not source
  ]


def migrate_hybrid_actor_state(
  source_actor: dict[str, torch.Tensor],
  *,
  source_stage: int,
  target_stage: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
  """Zero newly activated outputs and restore only their exploration std."""

  activated = _validate_transition(source_stage, target_stage)
  migrated = {key: value.clone() for key, value in source_actor.items()}
  weight_key, bias_key = _action_head_keys(migrated)
  for index in activated:
    migrated[weight_key][index, :] = 0.0
    migrated[bias_key][index] = 0.0

  std_key = "distribution.std_param"
  log_std_key = "distribution.log_std_param"
  if std_key in migrated:
    if migrated[std_key].shape != (6,):
      raise ValueError(f"{std_key} must contain six action values.")
    for index in activated:
      migrated[std_key][index] = HYBRID_ACTION_STD[index]
    changed_std_key = std_key
  elif log_std_key in migrated:
    if migrated[log_std_key].shape != (6,):
      raise ValueError(f"{log_std_key} must contain six action values.")
    for index in activated:
      migrated[log_std_key][index] = math.log(HYBRID_ACTION_STD[index])
    changed_std_key = log_std_key
  else:
    raise ValueError("Actor state has no per-action std parameter.")

  report = {
    "source_stage": source_stage,
    "target_stage": target_stage,
    "activated_indices": activated,
    "activated_actions": [HYBRID_ACTION_NAMES[index] for index in activated],
    "std_key": changed_std_key,
  }
  return migrated, report


def migrate_checkpoint(
  checkpoint: dict[str, Any],
  *,
  source_stage: int,
  target_stage: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Return a migrated checkpoint with fresh optimizer moments."""

  if "actor_state_dict" not in checkpoint:
    raise ValueError("Checkpoint has no actor_state_dict.")
  migrated = deepcopy(checkpoint)
  migrated["actor_state_dict"], report = migrate_hybrid_actor_state(
    checkpoint["actor_state_dict"],
    source_stage=source_stage,
    target_stage=target_stage,
  )
  optimizer = migrated.get("optimizer_state_dict")
  if isinstance(optimizer, dict) and "state" in optimizer:
    optimizer["state"] = {}
    report["optimizer_state"] = "cleared"
  else:
    report["optimizer_state"] = "not present"
  migrated["iter"] = 0
  infos = migrated.setdefault("infos", {})
  infos["hybrid_stage_migration"] = {
    **report,
    "created_at": datetime.now().isoformat(timespec="seconds"),
  }
  return migrated, report


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source-checkpoint", type=Path, required=True)
  parser.add_argument("--output-checkpoint", type=Path, required=True)
  parser.add_argument("--source-stage", type=int, required=True, choices=range(6))
  parser.add_argument("--target-stage", type=int, required=True, choices=range(6))
  parser.add_argument("--force", action="store_true")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if not args.source_checkpoint.is_file():
    raise FileNotFoundError(f"Source checkpoint not found: {args.source_checkpoint}")
  if args.output_checkpoint.exists() and not args.force:
    raise FileExistsError(
      f"Output checkpoint already exists: {args.output_checkpoint}. Use --force."
    )
  checkpoint = torch.load(
    args.source_checkpoint,
    map_location="cpu",
    weights_only=False,
  )
  migrated, report = migrate_checkpoint(
    checkpoint,
    source_stage=args.source_stage,
    target_stage=args.target_stage,
  )
  args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
  torch.save(migrated, args.output_checkpoint)
  print(f"[OK] Wrote migrated checkpoint: {args.output_checkpoint}")
  print(f"[OK] Activated actions: {', '.join(report['activated_actions'])}")
  print("[OK] Cleared optimizer state and reset checkpoint iteration to 0")


if __name__ == "__main__":
  main()
