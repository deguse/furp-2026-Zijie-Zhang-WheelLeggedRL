#!/usr/bin/env python3
"""Recovery-improvement magnitude curve: is the Stage5 win a single point?

The Stage5 robust formal proved the pre-registered center@8x recovery
improvement (12.42% at 2fb98ee) and the leg ablation attributed ~2/3 of
it to the leg residual heads. This evaluator hardens that result into a
curve: for each kick magnitude in multiples of the exact Stage1 kick it
measures center-posture recovery for three policies on the same
mechanics - the trained candidate, the candidate with leg heads zeroed
(attribution), and the zero-residual classical stack (baseline) - and
reports the fractional improvement per magnitude. A monotone or broad
improvement band is far stronger evidence than the single pre-registered
point; a curve that collapses off 8x bounds the claim honestly.

Observational hardening only: the pre-registered adjudication stays the
formal gate. This probe writes no gate verdicts.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parents[1]
SRC_PATH = Path(__file__).resolve().parents[2]
REPOSITORY_PATH = Path(__file__).resolve().parents[3]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

try:
  import hoppertrex_mjlab.tasks as tasks  # noqa: E402,F401
  from mjlab.tasks.registry import load_env_cfg
  from hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate import (
    HYBRID_STAGE_TASKS,
    STAGE5_RECOVERY_KICKS,
    _policy_session,
    _posture_targets_from_cfg,
    _run_recovery_scenario,
  )
  from hoppertrex_mjlab.scripts.rsl_rl.hybrid_gate import (
    to_deterministic_json,
  )
  from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
    hybrid_provenance_lines,
  )
except ImportError:
  import tasks  # noqa: E402,F401
  from mjlab.tasks.registry import load_env_cfg  # type: ignore[no-redef]
  from scripts.rsl_rl.evaluate_hybrid_gate import (  # type: ignore[no-redef]
    HYBRID_STAGE_TASKS,
    STAGE5_RECOVERY_KICKS,
    _policy_session,
    _posture_targets_from_cfg,
    _run_recovery_scenario,
  )
  from scripts.rsl_rl.hybrid_gate import (  # type: ignore[no-redef]
    to_deterministic_json,
  )
  from tasks.hoppertrex_hybrid_task import (  # type: ignore[no-redef]
    hybrid_provenance_lines,
  )

DEFAULT_SCALES = (2.0, 4.0, 6.0, 8.0)
POLICY_MODES = ("candidate", "legs_ablated", "zero_residual")


def _git_sha() -> str:
  return subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=REPOSITORY_PATH,
    text=True,
  ).strip()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint-file", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--warmup-steps", type=int, default=300)
  parser.add_argument(
    "--episode-length-s", type=float, default=1000000000.0
  )
  parser.add_argument(
    "--kick-scales",
    type=float,
    nargs="+",
    default=list(DEFAULT_SCALES),
    help="Kick magnitudes in multiples of the exact Stage1 kick.",
  )
  parser.add_argument("--output", type=Path, required=True)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  checkpoint = args.checkpoint_file.expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
  task = HYBRID_STAGE_TASKS[5]
  for line in hybrid_provenance_lines(load_env_cfg(task, play=True)):
    print(line)
  center = _posture_targets_from_cfg(task)
  print(f"[curve] center posture: height={center[0]:+.4f} pitch={center[1]:+.4f}")

  cells: list[dict[str, object]] = []
  for mode in POLICY_MODES:
    mode_checkpoint = None if mode == "zero_residual" else checkpoint
    args.ablate_leg_residuals = mode == "legs_ablated"
    with _policy_session(
      task=task,
      checkpoint=mode_checkpoint,
      args=args,
      play=True,
    ) as (wrapped, policy, _env_cfg):
      for scale in args.kick_scales:
        metrics = _run_recovery_scenario(
          wrapped=wrapped,
          policy=policy,
          args=args,
          target_height=center[0],
          target_pitch=center[1],
          kick_scale=float(scale),
        )
        cells.append(
          {
            "policy": mode,
            "kick_scale": float(scale),
            "metrics": metrics,
          }
        )
        print(
          f"[curve] {mode} @ {scale:.0f}x: "
          f"recovery={metrics['recovery_time_s']:.4f}s "
          f"terminated={metrics['terminated_event_rate']:.4f} "
          f"contact={metrics['non_wheel_contact_rate']:.4f}"
        )

  by_mode: dict[str, dict[float, dict[str, float]]] = {}
  for cell in cells:
    by_mode.setdefault(str(cell["policy"]), {})[
      float(cell["kick_scale"])  # type: ignore[arg-type]
    ] = cell["metrics"]  # type: ignore[assignment]
  curve = []
  for scale in args.kick_scales:
    scale = float(scale)
    baseline = by_mode["zero_residual"][scale]["recovery_time_s"]
    candidate = by_mode["candidate"][scale]["recovery_time_s"]
    ablated = by_mode["legs_ablated"][scale]["recovery_time_s"]
    improvement = (baseline - candidate) / baseline if baseline > 0 else 0.0
    ablated_improvement = (
      (baseline - ablated) / baseline if baseline > 0 else 0.0
    )
    curve.append(
      {
        "kick_scale": scale,
        "baseline_recovery_time_s": baseline,
        "candidate_recovery_time_s": candidate,
        "legs_ablated_recovery_time_s": ablated,
        "fractional_improvement": improvement,
        "legs_ablated_fractional_improvement": ablated_improvement,
        "leg_contribution_fraction": (
          (improvement - ablated_improvement) / improvement
          if improvement > 0
          else 0.0
        ),
      }
    )
    print(
      f"[curve] {scale:.0f}x: improvement {improvement:+.2%} "
      f"(ablated {ablated_improvement:+.2%}, "
      f"legs carry {curve[-1]['leg_contribution_fraction']:.0%})"
    )

  payload = {
    "schema_version": 1,
    "probe": "stage5_recovery_magnitude_curve",
    "task": task,
    "git_sha": _git_sha(),
    "seed": int(args.seed),
    "device": args.device,
    "num_envs": int(args.num_envs),
    "warmup_steps": int(args.warmup_steps),
    "kicks_per_cell": int(STAGE5_RECOVERY_KICKS),
    "kick_event_count_per_cell": int(STAGE5_RECOVERY_KICKS * args.num_envs),
    "checkpoint": str(checkpoint),
    "checkpoint_file_sha256": hashlib.sha256(
      checkpoint.read_bytes()
    ).hexdigest(),
    "center_posture": {"height": center[0], "pitch": center[1]},
    "policy_modes": list(POLICY_MODES),
    "kick_scales": [float(scale) for scale in args.kick_scales],
    "cells": cells,
    "curve": curve,
    "note": (
      "Observational hardening of the pre-registered center@8x formal "
      "adjudication; not a gate verdict."
    ),
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(to_deterministic_json(payload), encoding="utf-8")
  print(f"[curve] wrote {args.output}")


if __name__ == "__main__":
  main()
