#!/usr/bin/env python3
"""Build a hash-bound RollAssist reward calibration from a safe R0 stall."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hoppertrex_mjlab.hybrid.roll_assist import (
  build_reward_calibration,
  file_sha256,
  load_roll_boundary_verdict,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--roll-boundary", type=Path, required=True)
  parser.add_argument("--stall", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  return parser.parse_args(argv)


def positive_reward_rate_from_stall(payload: Mapping[str, Any]) -> float:
  """Read B only from a valid final-3-second zero-residual stall window."""

  if payload.get("kind") != "roll_assist_zero_residual_stall":
    raise ValueError("Reward calibration requires the dedicated zero-residual stall artifact.")
  if payload.get("evidence_eligible") is not True:
    raise ValueError("Smoke stall output cannot calibrate rewards.")
  protocol = payload.get("protocol")
  safety = payload.get("safety")
  measurement = payload.get("measurement")
  if not all(isinstance(value, Mapping) for value in (protocol, safety, measurement)):
    raise TypeError("Stall artifact schema is incomplete.")
  if protocol.get("policy_action") != [0.0] * 6 or protocol.get("wheel_residual_exact_zero") is not True:
    raise ValueError("Stall calibration was not zero residual.")
  if protocol.get("measurement_window_s") != 3.0 or protocol.get("height_role") != "Hnext":
    raise ValueError("Stall calibration must use the final 3 seconds at Hnext.")
  if safety.get("scope") != "final_3s_measurement_window":
    raise ValueError("Stall safety must cover the final 3-second B window.")
  if any(int(safety.get(name, -1)) != 0 for name in (
    "terminations", "non_wheel_contacts", "bilateral_airborne",
  )):
    raise ValueError("No safe stall window exists; RollAssist training is forbidden.")
  full_safety = payload.get("full_rollout_safety")
  if (
    not isinstance(full_safety, Mapping)
    or full_safety.get("scope") != "post_reset_settle_and_drive"
  ):
    raise ValueError("Stall artifact does not report full-rollout safety.")
  if any(int(full_safety.get(name, -1)) != 0 for name in (
    "terminations", "non_wheel_contacts", "bilateral_airborne",
  )):
    raise ValueError("Unsafe Hnext stall rollout forbids RollAssist training.")
  value = measurement.get("inherited_positive_reward_rate")
  if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0.0:
    raise ValueError("Stall artifact contains no positive inherited reward rate B.")
  return float(value)


def _git_sha() -> str:
  repository = Path(__file__).resolve().parents[3]
  return subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
    capture_output=True, text=True,
  ).stdout.strip()


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite reward calibration: {args.output}")
  verdict = load_roll_boundary_verdict(
    args.roll_boundary, expected_git_sha=_git_sha()
  )
  stall = json.loads(args.stall.read_text(encoding="utf-8-sig"))
  if not isinstance(stall, Mapping):
    raise TypeError("Stall artifact must be a JSON object.")
  if float(stall.get("stair_height_m", -1.0)) != verdict["hnext_m"]:
    raise ValueError("Stall artifact is not measured at the R0 Hnext.")
  artifact = build_reward_calibration(
    baseline_positive_reward_rate=positive_reward_rate_from_stall(stall),
    source_stall_sha256=file_sha256(args.stall.resolve()),
    roll_boundary_sha256=verdict["file_sha256"],
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  temporary = args.output.with_name(f".{args.output.name}.incomplete")
  try:
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
  finally:
    if temporary.exists():
      temporary.unlink()
  print(f"[roll-assist] reward_calibration={args.output}")
  print(f"[roll-assist] calibration_sha256={artifact['calibration_sha256']}")


if __name__ == "__main__":
  main()
