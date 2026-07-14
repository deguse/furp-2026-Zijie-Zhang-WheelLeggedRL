#!/usr/bin/env python3
"""Prepare a screened Stage1-A checkpoint for the Stage1-B curriculum."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from hoppertrex_mjlab.hybrid.config import HYBRID_ACTION_NAMES, HYBRID_ACTION_STD
from hoppertrex_mjlab.hybrid.mismatch import (
  STAGE1_MISMATCH_PROFILE_VERSION,
  stage1_mismatch_spec,
)
from hoppertrex_mjlab.scripts.rsl_rl.migrate_hybrid_stage import (
  COLLAPSED_ACTION_STD_THRESHOLD,
)


STAGE1_TASK = "HopperTrex-Hybrid-v2-Stage1"
STAGE1_SOURCE_PROFILE_VERSION = "stage1a_core_v1"


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _action_std(actor_state: Mapping[str, torch.Tensor]) -> tuple[str, torch.Tensor]:
  scalar = actor_state.get("distribution.std_param")
  if scalar is not None:
    if scalar.shape != (6,):
      raise ValueError("distribution.std_param must contain six values.")
    return "distribution.std_param", scalar.clone()
  log = actor_state.get("distribution.log_std_param")
  if log is not None:
    if log.shape != (6,):
      raise ValueError("distribution.log_std_param must contain six values.")
    return "distribution.log_std_param", torch.exp(log.clone())
  raise ValueError("Stage1 source actor has no per-action std parameter.")


def validate_stage1_source(
  checkpoint: Mapping[str, Any],
  gate: Mapping[str, Any],
  *,
  source_sha256: str,
) -> None:
  infos = checkpoint.get("infos")
  if not isinstance(infos, Mapping):
    raise ValueError("Stage1 source checkpoint has no provenance infos.")
  bootstrap = infos.get("hybrid_stage1_bootstrap")
  if not isinstance(bootstrap, Mapping):
    raise ValueError("Stage1 source is missing bootstrap provenance.")
  if bootstrap.get("stage") != 1 or bootstrap.get("task") != STAGE1_TASK:
    raise ValueError("Stage1 source bootstrap task/stage is invalid.")
  if tuple(bootstrap.get("action_order", ())) != HYBRID_ACTION_NAMES:
    raise ValueError("Stage1 source action order does not match Hybrid v2.")
  training = infos.get("hybrid_training")
  if not isinstance(training, Mapping) or not training.get("git_sha"):
    raise ValueError("Stage1 source is missing training git provenance.")
  if gate.get("suite") != "residual" or gate.get("gate_pass") is not True:
    raise ValueError("Stage1 source gate must be a passing residual gate.")
  gate_checkpoint_sha = gate.get("checkpoint_file_sha256")
  if gate_checkpoint_sha != source_sha256:
    raise ValueError(
      "Stage1 source gate checkpoint SHA256 does not match the source file."
    )


def prepare_stage1_extension_checkpoint(
  checkpoint: Mapping[str, Any],
  gate: Mapping[str, Any],
  *,
  source_checkpoint: str,
  source_checkpoint_sha256: str,
  source_gate: str,
  source_gate_sha256: str,
  collapsed_std_threshold: float = COLLAPSED_ACTION_STD_THRESHOLD,
  reset_collapsed_active_std: bool = False,
  created_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Return a same-stage handoff with fresh optimizer state and provenance."""

  validate_stage1_source(
    checkpoint,
    gate,
    source_sha256=source_checkpoint_sha256,
  )
  if collapsed_std_threshold <= 0.0:
    raise ValueError("collapsed_std_threshold must be positive.")
  actor_state = checkpoint.get("actor_state_dict")
  if not isinstance(actor_state, Mapping):
    raise ValueError("Stage1 source has no actor_state_dict.")
  std_key, source_std = _action_std(actor_state)
  collapsed = float(source_std[0]) < collapsed_std_threshold
  if collapsed and not reset_collapsed_active_std:
    raise ValueError(
      "Stage1 balance action std is collapsed. Rerun with "
      "--reset-collapsed-active-std to explicitly restore exploration."
    )

  result = deepcopy(dict(checkpoint))
  target_actor = result["actor_state_dict"]
  if collapsed:
    value = HYBRID_ACTION_STD[0]
    if std_key == "distribution.std_param":
      target_actor[std_key][0] = value
    else:
      target_actor[std_key][0] = math.log(value)
  optimizer = result.get("optimizer_state_dict")
  if not isinstance(optimizer, dict) or "state" not in optimizer:
    raise ValueError("Stage1 source optimizer_state_dict has no state mapping.")
  optimizer["state"] = {}
  source_iteration = int(result.get("iter", -1))
  result["iter"] = 0
  infos = result.setdefault("infos", {})
  infos.pop("hybrid_training", None)
  provenance = {
    "created_at": created_at,
    "task": STAGE1_TASK,
    "stage": 1,
    "source_profile_version": STAGE1_SOURCE_PROFILE_VERSION,
    "target_profile_version": STAGE1_MISMATCH_PROFILE_VERSION,
    "source_checkpoint": source_checkpoint,
    "source_checkpoint_sha256": source_checkpoint_sha256,
    "source_gate": source_gate,
    "source_gate_sha256": source_gate_sha256,
    "source_iteration": source_iteration,
    "source_action_std": [float(value) for value in source_std],
    "collapsed_std_threshold": collapsed_std_threshold,
    "collapsed_active_actions": (
      [HYBRID_ACTION_NAMES[0]] if collapsed else []
    ),
    "reset_collapsed_active_std": reset_collapsed_active_std,
    "optimizer_state": "cleared",
    "mismatch": stage1_mismatch_spec(),
  }
  infos["hybrid_stage1_extension"] = provenance
  return result, provenance


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source-checkpoint", type=Path, required=True)
  parser.add_argument("--source-gate-json", type=Path, required=True)
  parser.add_argument("--output-checkpoint", type=Path, required=True)
  parser.add_argument("--force", action="store_true")
  parser.add_argument(
    "--collapsed-std-threshold",
    type=float,
    default=COLLAPSED_ACTION_STD_THRESHOLD,
  )
  parser.add_argument("--reset-collapsed-active-std", action="store_true")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if not args.source_checkpoint.is_file():
    raise FileNotFoundError(
      f"Source checkpoint not found: {args.source_checkpoint}"
    )
  if not args.source_gate_json.is_file():
    raise FileNotFoundError(f"Source gate JSON not found: {args.source_gate_json}")
  if args.output_checkpoint.exists() and not args.force:
    raise FileExistsError(
      f"Output checkpoint already exists: {args.output_checkpoint}. Use --force."
    )
  checkpoint = torch.load(
    args.source_checkpoint,
    map_location="cpu",
    weights_only=False,
  )
  if not isinstance(checkpoint, Mapping):
    raise ValueError("Stage1 source checkpoint must contain a mapping.")
  gate = json.loads(args.source_gate_json.read_text(encoding="utf-8"))
  if not isinstance(gate, Mapping):
    raise ValueError("Stage1 source gate JSON must contain an object.")
  prepared, provenance = prepare_stage1_extension_checkpoint(
    checkpoint,
    gate,
    source_checkpoint=str(args.source_checkpoint.resolve()),
    source_checkpoint_sha256=_sha256(args.source_checkpoint),
    source_gate=str(args.source_gate_json.resolve()),
    source_gate_sha256=_sha256(args.source_gate_json),
    collapsed_std_threshold=args.collapsed_std_threshold,
    reset_collapsed_active_std=args.reset_collapsed_active_std,
    created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
  )
  args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
  torch.save(prepared, args.output_checkpoint)
  print(f"[OK] Wrote Stage1-B handoff: {args.output_checkpoint}")
  print(f"[OK] Source iteration: {provenance['source_iteration']}")
  print(f"[OK] Source action std: {provenance['source_action_std']}")
  print(f"[OK] Target profile: {provenance['target_profile_version']}")
  print("[OK] Cleared optimizer state and reset checkpoint iteration to 0")


if __name__ == "__main__":
  main()
