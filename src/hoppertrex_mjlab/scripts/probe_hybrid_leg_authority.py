#!/usr/bin/env python3
"""Measure Stage5 leg-residual authority against repeated zero residual."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch

from mjlab.tasks.registry import load_env_cfg

import hoppertrex_mjlab.tasks  # noqa: F401
from hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate import (
  HYBRID_STAGE_TASKS,
  _policy_session,
  _posture_targets_from_cfg,
  _run_recovery_scenario,
  validate_hybrid_evaluation_checkpoint,
)
from hoppertrex_mjlab.scripts.rsl_rl.hybrid_gate import to_deterministic_json
from hoppertrex_mjlab.hybrid.config import DEFAULT_ACTION_SCALES
from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import LEG_RESIDUAL_SCALE_ENV


REPOSITORY_PATH = Path(__file__).resolve().parents[3]
TASK = HYBRID_STAGE_TASKS[5]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint-file", type=Path, default=None)
  parser.add_argument(
    "--baseline-json",
    type=Path,
    default=None,
    help="Optional output from a prior zero-residual repeat measurement.",
  )
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--warmup-steps", type=int, default=300)
  parser.add_argument("--episode-length-s", type=float, default=1.0e9)
  parser.add_argument("--output", type=Path, default=None)
  parser.add_argument("--preflight-only", action="store_true")
  return parser.parse_args(argv)


def validate_authority_registration() -> None:
  configured_scale = float(
    os.environ.get(LEG_RESIDUAL_SCALE_ENV, DEFAULT_ACTION_SCALES[2])
  )
  for stage in range(6):
    cfg = load_env_cfg(f"HopperTrex-Hybrid-v2-Stage{stage}", play=True)
    actual = tuple(cfg.actions["hybrid_wheel_leg"].action_scales)
    expected = list(DEFAULT_ACTION_SCALES)
    if stage == 5:
      expected[2:6] = [configured_scale] * 4
    if actual != tuple(expected):
      raise RuntimeError(
        f"Stage{stage} action scales {actual} != {tuple(expected)}"
      )


def _git_sha() -> str:
  return subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=REPOSITORY_PATH,
    check=True,
    capture_output=True,
    text=True,
  ).stdout.strip()


def _run_mode(
  args: argparse.Namespace,
  *,
  checkpoint: Path | None,
  ablate_legs: bool,
) -> dict[str, float]:
  args.ablate_leg_residuals = ablate_legs
  center = _posture_targets_from_cfg(TASK)
  with _policy_session(
    task=TASK,
    checkpoint=checkpoint,
    args=args,
    play=True,
  ) as (wrapped, policy, _env_cfg):
    return _run_recovery_scenario(
      wrapped=wrapped,
      policy=policy,
      args=args,
      target_height=center[0],
      target_pitch=center[1],
    )


def _baseline_repeats(args: argparse.Namespace) -> list[dict[str, float]]:
  return [
    _run_mode(args, checkpoint=None, ablate_legs=False),
    _run_mode(args, checkpoint=None, ablate_legs=False),
  ]


def _validated_baseline(path: Path) -> list[dict[str, float]]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict) or payload.get("probe") != (
    "hybrid_leg_authority"
  ):
    raise ValueError("Baseline JSON is not a leg-authority probe artifact.")
  repeats = payload.get("baseline_repeats")
  if not isinstance(repeats, list) or len(repeats) != 2:
    raise ValueError("Baseline JSON must contain exactly two repeats.")
  return [dict(row) for row in repeats]


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  if args.preflight_only:
    validate_authority_registration()
    print("[PASS] Stage0-5 registration and authority preflight")
    return
  if args.output is None:
    raise ValueError("--output is required unless --preflight-only is set.")
  if args.num_envs != 32:
    raise ValueError("P1.2 pre-registration requires exactly 32 envs.")
  checkpoint = (
    None
    if args.checkpoint_file is None
    else args.checkpoint_file.expanduser().resolve()
  )
  if checkpoint is not None and not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
  git_sha = _git_sha()
  env_cfg = load_env_cfg(TASK, play=True)
  action_scales = tuple(
    float(value)
    for value in env_cfg.actions["hybrid_wheel_leg"].action_scales
  )
  if checkpoint is not None:
    checkpoint_payload = torch.load(
      checkpoint, map_location="cpu", weights_only=False
    )
    validate_hybrid_evaluation_checkpoint(
      TASK,
      env_cfg,
      checkpoint_payload,
      evaluation_git_sha=git_sha,
    )

  baseline_repeats = (
    _validated_baseline(args.baseline_json.resolve())
    if args.baseline_json is not None
    else _baseline_repeats(args)
  )
  baseline_times = [float(row["recovery_time_s"]) for row in baseline_repeats]
  baseline_mean = sum(baseline_times) / len(baseline_times)
  run_noise = abs(baseline_times[1] - baseline_times[0])
  result: dict[str, object] = {
    "schema_version": 1,
    "probe": "hybrid_leg_authority",
    "git_sha": git_sha,
    "task": TASK,
    "seed": int(args.seed),
    "device": args.device,
    "num_envs": int(args.num_envs),
    "kick_scale": 8.0,
    "kick_events_per_run": 4 * int(args.num_envs),
    "center_posture": {
      "height": _posture_targets_from_cfg(TASK)[0],
      "pitch": _posture_targets_from_cfg(TASK)[1],
    },
    "action_scales": list(action_scales),
    "leg_residual_scale": action_scales[2],
    "baseline_repeats": baseline_repeats,
    "baseline_recovery_mean_s": baseline_mean,
    "baseline_run_noise_s": run_noise,
    "ten_x_run_noise_s": 10.0 * run_noise,
    "baseline_safety_pass": all(
      float(row["terminated_event_rate"]) == 0.0
      and float(row["non_wheel_contact_rate"]) == 0.0
      for row in baseline_repeats
    ),
  }
  if checkpoint is not None:
    candidate = _run_mode(args, checkpoint=checkpoint, ablate_legs=False)
    ablated = _run_mode(args, checkpoint=checkpoint, ablate_legs=True)
    candidate_improvement = baseline_mean - candidate["recovery_time_s"]
    ablated_improvement = baseline_mean - ablated["recovery_time_s"]
    result.update(
      {
        "checkpoint": str(checkpoint),
        "checkpoint_file_sha256": hashlib.sha256(
          checkpoint.read_bytes()
        ).hexdigest(),
        "candidate": candidate,
        "legs_ablated": ablated,
        "candidate_improvement_s": candidate_improvement,
        "candidate_fractional_improvement": (
          candidate_improvement / baseline_mean if baseline_mean > 0 else 0.0
        ),
        "legs_ablated_improvement_s": ablated_improvement,
        "legs_ablated_fractional_improvement": (
          ablated_improvement / baseline_mean if baseline_mean > 0 else 0.0
        ),
        "safety_pass": all(
          float(row["terminated_event_rate"]) == 0.0
          and float(row["non_wheel_contact_rate"]) == 0.0
          for row in (candidate, ablated)
        ),
      }
    )

  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(to_deterministic_json(result), encoding="utf-8")
  print(
    f"[authority] baseline={baseline_mean:.4f}s "
    f"noise={run_noise:.4f}s scale={action_scales[2]:.3f}"
  )
  if checkpoint is not None:
    print(
      "[authority] candidate="
      f"{result['candidate_fractional_improvement']:+.2%} "
      "legs-ablated="
      f"{result['legs_ablated_fractional_improvement']:+.2%} "
      f"safety={result['safety_pass']}"
    )
  print(f"[authority] wrote {args.output}")


if __name__ == "__main__":
  main()
