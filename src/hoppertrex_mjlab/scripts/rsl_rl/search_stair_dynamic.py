#!/usr/bin/env python3
# ruff: noqa: TRY004
"""Screen and CEM-tune the frozen Hybrid-v3 dynamic stair feedforward."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from hoppertrex_mjlab.hybrid.stair_classical import CandidateScore, optimize_cem
from hoppertrex_mjlab.hybrid.stair_dynamic import (
  DYNAMIC_STAIR_CEM_ITERATIONS,
  DYNAMIC_STAIR_CEM_LOWER,
  DYNAMIC_STAIR_CEM_POPULATION,
  DYNAMIC_STAIR_CEM_REPLICATES,
  DYNAMIC_STAIR_CEM_SEED,
  DYNAMIC_STAIR_CEM_UPPER,
  DynamicLiftMode,
  DynamicStairManeuver,
  dynamic_maneuver_payload,
  validate_dynamic_maneuver_bindings,
)

SEARCH_SCHEMA_VERSION = 1
Adapter = Callable[[Mapping[str, object]], Mapping[str, object]]


def _score(value: object, *, name: str) -> CandidateScore:
  if not isinstance(value, Mapping):
    raise ValueError(f"{name} must be a score mapping.")
  expected = {
    "safe_successes",
    "median_progress",
    "peak_pitch",
    "energy",
    "target_smoothness",
    "unsafe_trials",
  }
  if set(value) != expected:
    raise ValueError(f"{name} score schema drifted.")
  integers: dict[str, int] = {}
  for field in ("safe_successes", "unsafe_trials"):
    item = value[field]
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
      raise ValueError(f"{name}.{field} must be a non-negative integer.")
    integers[field] = int(item)
  floats: dict[str, float] = {}
  for field in ("median_progress", "peak_pitch", "energy", "target_smoothness"):
    item = value[field]
    if isinstance(item, bool) or not isinstance(item, (int, float)):
      raise ValueError(f"{name}.{field} must be finite.")
    result = float(item)
    if not math.isfinite(result) or (field != "median_progress" and result < 0.0):
      raise ValueError(f"{name}.{field} must be finite and physically valid.")
    floats[field] = result
  if integers["safe_successes"] + integers["unsafe_trials"] > DYNAMIC_STAIR_CEM_REPLICATES:
    raise ValueError(f"{name} counts exceed eight registered replicas.")
  return CandidateScore(
    safe_successes=integers["safe_successes"],
    median_progress=floats["median_progress"],
    peak_pitch=floats["peak_pitch"],
    energy=floats["energy"],
    target_smoothness=floats["target_smoothness"],
    unsafe_trials=integers["unsafe_trials"],
  )


def score_payload(score: CandidateScore) -> dict[str, float | int]:
  return {
    "safe_successes": score.safe_successes,
    "median_progress": score.median_progress,
    "peak_pitch": score.peak_pitch,
    "energy": score.energy,
    "target_smoothness": score.target_smoothness,
    "unsafe_trials": score.unsafe_trials,
  }


def validate_trigger_qualification(value: object) -> dict[str, object]:
  if not isinstance(value, Mapping):
    raise ValueError("Per-wheel trigger qualification is missing.")
  expected = {
    "metric",
    "threshold_n",
    "window",
    "left_sensor_identity",
    "right_sensor_identity",
    "left_live_detected",
    "right_live_detected",
    "flat_false_positives",
    "kick_false_positives",
    "evidence_sha256",
  }
  if set(value) != expected:
    raise ValueError("Per-wheel trigger qualification schema drifted.")
  digest = value["evidence_sha256"]
  if (
    value["metric"] != "abs(F0*nx)"
    or float(value["threshold_n"]) != 18.0
    or value["window"] != 3
    or value["left_sensor_identity"] is not True
    or value["right_sensor_identity"] is not True
    or value["left_live_detected"] is not True
    or value["right_live_detected"] is not True
    or value["flat_false_positives"] != 0
    or value["kick_false_positives"] != 0
    or not isinstance(digest, str)
    or len(digest) != 64
    or any(char not in "0123456789abcdef" for char in digest)
  ):
    raise ValueError("Per-wheel 18 N x 3 live qualification failed.")
  return dict(value)


def select_feedforward_family(
  *,
  roll_only: CandidateScore,
  synchronized: CandidateScore,
  alternating: CandidateScore,
) -> DynamicLiftMode | None:
  """Select safely by CandidateScore; never assume alternating is superior."""

  eligible = [
    (mode, score)
    for mode, score in (
      (DynamicLiftMode.SYNCHRONIZED, synchronized),
      (DynamicLiftMode.ALTERNATING, alternating),
    )
    if score.unsafe_trials == 0
    and score.safe_successes >= 1
    and score.median_progress > roll_only.median_progress
  ]
  if not eligible:
    return None
  return max(eligible, key=lambda item: item[1].rank())[0]


def _load_adapter(spec: str) -> Adapter:
  module_name, separator, function_name = spec.partition(":")
  if not separator or not module_name or not function_name:
    raise ValueError("Adapter must be written as module:function.")
  function = getattr(importlib.import_module(module_name), function_name, None)
  if not callable(function):
    raise ValueError(f"Search adapter {spec!r} is not callable.")
  return function


def _atomic_json_no_clobber(payload: object, output: Path) -> None:
  if output.exists():
    raise FileExistsError(f"Refusing to overwrite {output}")
  output.parent.mkdir(parents=True, exist_ok=True)
  temporary = output.with_name(f".{output.name}.incomplete.{uuid.uuid4().hex}")
  try:
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
      json.dump(payload, stream, indent=2, sort_keys=True)
      stream.write("\n")
      stream.flush()
      os.fsync(stream.fileno())
    os.link(temporary, output)
  finally:
    temporary.unlink(missing_ok=True)


def run_search(
  adapter: Adapter,
  *,
  bindings: Mapping[str, str],
  device: str = "cuda:0",
) -> tuple[dict[str, object], dict[str, object] | None]:
  if not isinstance(device, str) or not device.strip():
    raise ValueError("device must be a non-empty string.")
  validated_bindings = validate_dynamic_maneuver_bindings(bindings)
  screen_response = adapter(
    {
      "schema_version": SEARCH_SCHEMA_VERSION,
      "kind": "family_screen",
      "height_m": 0.01,
      "replicates": DYNAMIC_STAIR_CEM_REPLICATES,
      "families": [
        "roll_only",
        DynamicLiftMode.SYNCHRONIZED.value,
        DynamicLiftMode.ALTERNATING.value,
      ],
      "approach_vx_mps": 0.07,
      "device": device,
      "expected_trigger_qualification_sha256": validated_bindings[
        "per_wheel_trigger_qualification_sha256"
      ],
      "expected_stage5_checkpoint_sha256": validated_bindings[
        "stage5_checkpoint_sha256"
      ],
    }
  )
  if not isinstance(screen_response, Mapping):
    raise ValueError("Family-screen adapter response must be a mapping.")
  if set(screen_response) != {"scores", "trigger_qualification"}:
    raise ValueError("Family-screen adapter response schema drifted.")
  qualification = validate_trigger_qualification(
    screen_response["trigger_qualification"]
  )
  if (
    qualification["evidence_sha256"]
    != validated_bindings["per_wheel_trigger_qualification_sha256"]
  ):
    raise ValueError("Trigger qualification binding does not match evidence.")
  raw_scores = screen_response["scores"]
  if not isinstance(raw_scores, Mapping) or set(raw_scores) != {
    "roll_only",
    DynamicLiftMode.SYNCHRONIZED.value,
    DynamicLiftMode.ALTERNATING.value,
  }:
    raise ValueError("Family-screen scores are missing or extra.")
  roll = _score(raw_scores["roll_only"], name="roll_only")
  synchronized = _score(
    raw_scores[DynamicLiftMode.SYNCHRONIZED.value], name="synchronized"
  )
  alternating = _score(
    raw_scores[DynamicLiftMode.ALTERNATING.value], name="alternating"
  )
  family = select_feedforward_family(
    roll_only=roll,
    synchronized=synchronized,
    alternating=alternating,
  )
  report: dict[str, object] = {
    "schema_version": SEARCH_SCHEMA_VERSION,
    "task": "HopperTrex-Hybrid-v3-StairDynamic",
    "protocol": {
      "population": DYNAMIC_STAIR_CEM_POPULATION,
      "iterations": DYNAMIC_STAIR_CEM_ITERATIONS,
      "seed": DYNAMIC_STAIR_CEM_SEED,
      "replicates": DYNAMIC_STAIR_CEM_REPLICATES,
      "lower": list(DYNAMIC_STAIR_CEM_LOWER),
      "upper": list(DYNAMIC_STAIR_CEM_UPPER),
      "feedback_policy": "stage5_seed1_100_selected_deterministic_mean",
      "observation_adapter": "34_to_52_zero_appended_columns",
    },
    "screen": {
      "roll_only": score_payload(roll),
      "synchronized": score_payload(synchronized),
      "alternating": score_payload(alternating),
      "selected_family": None if family is None else family.value,
    },
    "trigger_qualification": qualification,
    "classification": "STOP_DYNAMIC_STAIR_UNQUALIFIED",
  }
  if family is None:
    return report, None

  def batch(candidates: NDArray[np.float64]) -> Sequence[CandidateScore]:
    response = adapter(
      {
        "schema_version": SEARCH_SCHEMA_VERSION,
        "kind": "cem_batch",
        "family": family.value,
        "height_m": 0.01,
        "replicates": DYNAMIC_STAIR_CEM_REPLICATES,
        "candidates": candidates.tolist(),
        "device": device,
        "expected_stage5_checkpoint_sha256": validated_bindings[
          "stage5_checkpoint_sha256"
        ],
      }
    )
    if not isinstance(response, Mapping) or set(response) != {"scores"}:
      raise ValueError("CEM adapter response schema drifted.")
    values = response["scores"]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
      raise ValueError("CEM adapter scores must be a sequence.")
    return tuple(_score(value, name=f"candidate[{index}]") for index, value in enumerate(values))

  def single(candidate: NDArray[np.float64]) -> CandidateScore:
    scores = batch(candidate.reshape(1, -1))
    if len(scores) != 1:
      raise ValueError("Single CEM candidate adapter returned wrong count.")
    return scores[0]

  result = optimize_cem(
    single,
    lower=np.asarray(DYNAMIC_STAIR_CEM_LOWER, dtype=np.float64),
    upper=np.asarray(DYNAMIC_STAIR_CEM_UPPER, dtype=np.float64),
    population=DYNAMIC_STAIR_CEM_POPULATION,
    iterations=DYNAMIC_STAIR_CEM_ITERATIONS,
    seed=DYNAMIC_STAIR_CEM_SEED,
    evaluate_batch=batch,
  )
  qualified = (
    result.score.unsafe_trials == 0
    and result.score.safe_successes >= 1
    and result.score.median_progress > roll.median_progress
  )
  report["cem"] = {
    "best_parameters": result.parameters.tolist(),
    "best_score": score_payload(result.score),
    "final_mean": result.mean.tolist(),
    "final_std": result.std.tolist(),
    "history": list(result.history),
  }
  if not qualified:
    return report, None
  maneuver = DynamicStairManeuver(
    lift_mode=family,
    split_amplitude_rad=float(result.parameters[0]),
    lift_amplitude_rad=float(result.parameters[1]),
    trailing_delay_s=float(result.parameters[2]),
    drive_feedforward_radps=float(result.parameters[3]),
  )
  artifact = dynamic_maneuver_payload(maneuver, bindings=validated_bindings)
  report["classification"] = "DYNAMIC_STAIR_MANEUVER_QUALIFIED"
  report["maneuver_sha256"] = artifact["maneuver_hash"]
  return report, artifact


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--adapter", required=True, help="module:function live adapter")
  parser.add_argument("--bindings-json", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--report", type=Path, required=True)
  parser.add_argument("--maneuver-output", type=Path, required=True)
  return parser


def main() -> None:
  args = _parser().parse_args()
  bindings = json.loads(args.bindings_json.read_text(encoding="utf-8"))
  if not isinstance(bindings, Mapping):
    raise ValueError("Bindings JSON must contain an object.")
  report, artifact = run_search(
    _load_adapter(args.adapter), bindings=bindings, device=args.device
  )
  _atomic_json_no_clobber(report, args.report)
  if artifact is not None:
    _atomic_json_no_clobber(artifact, args.maneuver_output)
    print(f"[PASS] qualified maneuver: {args.maneuver_output}")
  else:
    print("[STOP] no safe feedforward family/candidate qualified; PPO is blocked")


if __name__ == "__main__":
  main()
