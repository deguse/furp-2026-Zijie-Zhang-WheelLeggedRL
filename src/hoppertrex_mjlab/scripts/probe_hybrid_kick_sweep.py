#!/usr/bin/env python3
"""Kick-magnitude sweep: where does the classical stack start to degrade?

Stage 3.0 closed the tested regime classically: statics at the noise
floor, shaped transitions below the healthy band, and Stage1-magnitude
kicks (0.04 m/s) recovered in under 0.5 s at every posture - with a
cross-posture recovery window of the same order as the run-to-run noise.
A pre-registered PPO improvement claim needs a regime with real headroom,
so this probe sweeps the kick impulse in multiples of the exact Stage1
kick at three envelope postures (center, the measured weak corner
tall+nose-up, and its diagonal opposite) and reports recovery time,
survival, and terminations per magnitude. The degradation knee - if one
exists - is where Stage3 PPO pre-registers; a flat response to the top of
the sweep closes Stage3 as classically sufficient.

Caveat recorded in the payload: a terminated env auto-resets to the
nominal pose mid-window, so recovery times are only clean below the
termination knee; at and above it, ``terminated_events`` is the verdict.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_PATH = Path(__file__).resolve().parents[1]
SRC_PATH = Path(__file__).resolve().parents[2]
REPOSITORY_PATH = Path(__file__).resolve().parents[3]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

try:
  import hoppertrex_mjlab.tasks as tasks  # noqa: E402,F401
  from hoppertrex_mjlab.scripts.probe_hybrid_posture_transition import (
    run_kick_cell,
  )
  from hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate import (
    STAGE1_KICK_LIN_X,
    STAGE1_KICK_PITCH_RATE,
  )
  from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
    hybrid_provenance_lines,
  )
except ImportError:
  import tasks  # noqa: E402,F401
  from scripts.probe_hybrid_posture_transition import (  # type: ignore[no-redef]
    run_kick_cell,
  )
  from scripts.rsl_rl.evaluate_hybrid_gate import (  # type: ignore[no-redef]
    STAGE1_KICK_LIN_X,
    STAGE1_KICK_PITCH_RATE,
  )
  from tasks.hoppertrex_hybrid_task import (  # type: ignore[no-redef]
    hybrid_provenance_lines,
  )
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default="HopperTrex-Hybrid-v2-Stage3")
  parser.add_argument(
    "--device",
    default="cuda:0" if torch.cuda.is_available() else "cpu",
  )
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--settle-steps", type=int, default=150)
  parser.add_argument("--kicks-per-cell", type=int, default=4)
  parser.add_argument(
    "--kick-interval",
    type=int,
    default=300,
    help="Steps between kicks; 6 s so degraded recoveries stay separable.",
  )
  parser.add_argument(
    "--kick-scales",
    type=float,
    nargs="+",
    default=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0),
    help="Multiples of the Stage1 kick impulse (0.04 m/s + 0.06 rad/s).",
  )
  parser.add_argument(
    "--fit-output",
    type=Path,
    default=None,
    help="Write the kick-sweep qualification JSON to this path.",
  )
  return parser.parse_args(argv)


def sweep_postures(
  height_range: tuple[float, float],
  pitch_range: tuple[float, float],
) -> list[tuple[float, float]]:
  """Center, the measured weak corner (tall+nose-up), and its opposite."""

  h_lo, h_hi = (float(height_range[0]), float(height_range[1]))
  p_lo, p_hi = (float(pitch_range[0]), float(pitch_range[1]))
  return [
    (0.5 * (h_lo + h_hi), 0.5 * (p_lo + p_hi)),
    (h_hi, p_hi),
    (h_lo, p_lo),
  ]


def sweep_payload(
  *,
  cells: list[dict[str, float]],
  controller_gain_hash: str | None,
  controller_qualified: bool,
  posture_map_hash: str | None,
  posture_map_qualified: bool,
  station_calibration_hash: str | None,
  station_calibration_qualified: bool,
  source_probe: dict[str, object],
) -> dict[str, object]:
  if not cells:
    raise ValueError("Kick sweep requires at least one measured cell.")
  postures = sorted(
    {(cell["target_height"], cell["target_pitch"]) for cell in cells}
  )
  per_posture: dict[str, object] = {}
  for height, pitch in postures:
    rows = sorted(
      (cell for cell in cells
       if (cell["target_height"], cell["target_pitch"]) == (height, pitch)),
      key=lambda cell: cell["kick_scale"],
    )
    knee = next(
      (row["kick_scale"] for row in rows if row["terminated_events"] > 0.0),
      None,
    )
    per_posture[f"({height:.4f},{pitch:+.4f})"] = {
      "baseline_recovery_time_s": rows[0]["recovery_time_s"],
      "max_recovery_time_s": max(row["recovery_time_s"] for row in rows),
      "termination_knee_scale": knee,
      "max_survived_scale": max(
        (row["kick_scale"] for row in rows
         if row["terminated_events"] == 0.0),
        default=None,
      ),
    }
  return {
    "schema_version": 1,
    "kind": "kick_magnitude_sweep_qualification",
    "controller_gain_hash": controller_gain_hash,
    "controller_qualified": bool(controller_qualified),
    "posture_map_hash": posture_map_hash,
    "posture_map_qualified": bool(posture_map_qualified),
    "station_calibration_hash": station_calibration_hash,
    "station_calibration_qualified": bool(station_calibration_qualified),
    "cells": cells,
    "summary": {
      "cell_count": len(cells),
      "terminated_events_total": sum(
        cell["terminated_events"] for cell in cells
      ),
      "per_posture": per_posture,
    },
    "source_probe": dict(source_probe),
  }


def _git_sha() -> str:
  completed = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=REPOSITORY_PATH,
    check=True,
    capture_output=True,
    text=True,
  )
  return completed.stdout.strip()


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = args.num_envs
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.num_envs = args.num_envs
  for line in hybrid_provenance_lines(cfg):
    print(line)
  action_cfg = cfg.actions["hybrid_wheel_leg"]
  if not getattr(action_cfg, "station_calibration_qualified", False):
    print(
      "[probe][WARN] No station calibration active: held postures will "
      "drift and pollute the recovery estimate. Official sweep data "
      "requires the station artifact."
    )
  posture_command = cfg.commands["posture"]
  postures = sweep_postures(
    tuple(posture_command.height_range),
    tuple(posture_command.pitch_range),
  )

  cells: list[dict[str, float]] = []
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    print(
      f"{'kick@(h,p)':>18} {'scale':>6} {'lin_x':>6} {'recovery_s':>10} "
      f"{'|vx|max':>8} {'term':>5}"
    )
    for height, pitch in postures:
      for scale in args.kick_scales:
        cell = run_kick_cell(
          env,
          height=height,
          pitch=pitch,
          kicks=args.kicks_per_cell,
          kick_interval=args.kick_interval,
          settle_steps=args.settle_steps,
          kick_scale=float(scale),
        )
        cells.append(cell)
        print(
          f"({cell['target_height']:.3f},{cell['target_pitch']:+.3f})"
          f" {cell['kick_scale']:>6.1f} {cell['kick_lin_x']:>6.2f}"
          f" {cell['recovery_time_s']:>10.3f}"
          f" {cell['post_kick_lin_x_abs_max']:>8.4f}"
          f" {cell['terminated_events']:>5.0f}"
        )
  finally:
    env.close()

  if args.fit_output is None:
    return
  payload = sweep_payload(
    cells=cells,
    controller_gain_hash=action_cfg.controller_gain_hash,
    controller_qualified=bool(action_cfg.controller_qualified),
    posture_map_hash=action_cfg.posture_map_hash,
    posture_map_qualified=bool(action_cfg.posture_map_qualified),
    station_calibration_hash=getattr(
      action_cfg, "station_calibration_hash", None
    ),
    station_calibration_qualified=bool(
      getattr(action_cfg, "station_calibration_qualified", False)
    ),
    source_probe={
      "git_sha": _git_sha(),
      "task": args.task,
      "device": args.device,
      "num_envs": args.num_envs,
      "settle_steps": args.settle_steps,
      "kicks_per_cell": args.kicks_per_cell,
      "kick_interval": args.kick_interval,
      "kick_scales": [float(scale) for scale in args.kick_scales],
      "stage1_kick_lin_x": float(STAGE1_KICK_LIN_X),
      "stage1_kick_pitch_rate": float(STAGE1_KICK_PITCH_RATE),
    },
  )
  args.fit_output.parent.mkdir(parents=True, exist_ok=True)
  args.fit_output.write_text(
    json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
  )
  print(f"[probe] kick-sweep qualification written: {args.fit_output}")
  print(f"[probe] summary={json.dumps(payload['summary'], indent=1)}")


if __name__ == "__main__":
  main()
