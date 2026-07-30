"""Non-evidence C2 flat-only detector feature-floor calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv

from hoppertrex_mjlab.scripts import probe_hybrid_c2_paired_capture_v1 as c2
from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import hybrid_provenance_lines


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args(argv)


def summarize_or_invalid(
  attempts: list[dict[str, Any]],
) -> dict[str, Any]:
  """Convert malformed or unhealthy floor data into the frozen invalid branch."""

  try:
    return c2.flat_floor_summary(attempts)
  except (KeyError, TypeError, ValueError) as exc:
    return {
      "classification": "INVALID_FLOOR_CAPTURE",
      "invalid_reason": str(exc),
      "cells": {},
      "overall": {"features": {}},
    }


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  if args.output.exists():
    raise FileExistsError(f"Output already exists: {args.output}")
  if args.device != "cuda:0" or not torch.cuda.is_available():
    raise RuntimeError("C2 flat-floor calibration is pinned to cuda:0.")
  protocol = c2.flat_floor_protocol(args.device)
  cfg = c2.make_causal_env_cfg(c2.DIAGNOSTIC_HEIGHTS_M, c2.OFFICIAL_ENVS_PER_HEIGHT)
  cfg.seed = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  all_trials: list[dict[str, object]] = []
  all_attempts: list[dict[str, object]] = []
  try:
    action_term = env.action_manager.get_term("hybrid_wheel_leg")
    schedule_hash = c2._require_schedule_hash(action_term)
    provenance = c2._capture_provenance(cfg, args.device)
    for line in hybrid_provenance_lines(cfg):
      print(line)
    for cell in protocol["command_cells"]:
      trials, _captures, flat_attempts = c2.run_cell(
        env,
        heights=c2.DIAGNOSTIC_HEIGHTS_M,
        cell=cell,
        protocol=protocol,
        flat_only=True,
      )
      all_trials.extend(
        {
          "cell_name": trial["cell_name"],
          "flat_envs": trial["flat_envs"],
          "flat_terminated": trial["flat_terminated"],
          "flat_non_wheel_contact": trial["flat_non_wheel_contact"],
          "recorded_drive_steps": trial["recorded_drive_steps"],
        }
        for trial in trials
      )
      all_attempts.extend(flat_attempts)
  finally:
    env.close()

  floor = summarize_or_invalid(all_attempts)
  payload = {
    "schema_version": 1,
    "probe": "hybrid_c2_flat_floor_v1",
    "classification": floor["classification"],
    "evidence_eligible": False,
    "promotion_eligible": False,
    "training_eligible": False,
    "detector_fit_eligible": False,
    "task": c2.stair.TASK,
    "seed": 1,
    "device": args.device,
    **provenance,
    "controller_schedule_hash": schedule_hash,
    "protocol": protocol,
    "trials": all_trials,
    "flat_attempt_count": len(all_attempts),
    "floor": floor,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
    json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
  )
  print(f"[c2-flat-floor] classification={payload['classification']}")


if __name__ == "__main__":
  main()
