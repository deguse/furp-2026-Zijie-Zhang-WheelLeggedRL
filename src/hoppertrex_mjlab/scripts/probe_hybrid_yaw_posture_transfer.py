#!/usr/bin/env python3
"""Yaw transfer across postures: does the Stage 2.0 feedforward drift?

The yaw feedforward was probe-fitted at the NOMINAL leg posture. Stage4
turns while holding commanded postures, and the wheel-differential to
body-yaw-rate transfer depends on geometry the posture map moves (height,
CoM, contact state). This Stage 4.0 qualification drives the yaw residual
head with fixed differentials (raw plant methodology: probe scale
override, feedforward zeroed) at the envelope center, the weak corner
(tall+nose-up), and its opposite, and compares the measured transfer
curves across postures.

Pre-registered rule (Q3 pattern): a working-point (|action| 0.55)
transfer deviation above 20% versus the center posture triggers a
posture-scheduled feedforward batch; below it the global yaw calibration
stands for Stage4/5. The half-sum differential measurement is immune to
the balance and station-keeping common-mode terms, so the station
artifact stays active and postures hold position during the sweep.
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
  from hoppertrex_mjlab.scripts.probe_hybrid_kick_sweep import (
    sweep_postures,
  )
  from hoppertrex_mjlab.scripts.probe_hybrid_posture_balance import (
    _force_commands,
    _pitch,
  )
  from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
    hybrid_provenance_lines,
  )
except ImportError:
  import tasks  # noqa: E402,F401
  from scripts.probe_hybrid_kick_sweep import (  # type: ignore[no-redef]
    sweep_postures,
  )
  from scripts.probe_hybrid_posture_balance import (  # type: ignore[no-redef]
    _force_commands,
    _pitch,
  )
  from tasks.hoppertrex_hybrid_task import (  # type: ignore[no-redef]
    hybrid_provenance_lines,
  )
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

DEFAULT_YAW_ACTIONS = (
  -0.75, -0.55, -0.35, -0.15,
  0.15, 0.35, 0.55, 0.75,
)
WORKING_POINT_ACTION = 0.55
DEVIATION_LIMIT = 0.20


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default="HopperTrex-Hybrid-v2-Stage4")
  parser.add_argument(
    "--device",
    default="cuda:0" if torch.cuda.is_available() else "cpu",
  )
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--settle-steps", type=int, default=50)
  parser.add_argument("--measure-steps", type=int, default=150)
  parser.add_argument(
    "--yaw-actions",
    type=float,
    nargs="+",
    default=DEFAULT_YAW_ACTIONS,
    help="Yaw-head action values in [-1, 1] to hold fixed per posture.",
  )
  parser.add_argument(
    "--probe-yaw-scale",
    type=float,
    default=1.0,
    help="Probe-env yaw action scale (raw plant, decoupled from training).",
  )
  parser.add_argument(
    "--fit-output",
    type=Path,
    default=None,
    help="Write the cross-posture transfer qualification JSON here.",
  )
  return parser.parse_args(argv)


def run_transfer_cell(
  env: ManagerBasedRlEnv,
  *,
  height: float,
  pitch: float,
  value: float,
  settle_steps: int,
  measure_steps: int,
) -> dict[str, float]:
  """Hold one posture, drive one fixed yaw differential, measure transfer."""

  env.reset()
  actions = torch.zeros(
    (env.num_envs, env.action_space.shape[-1]),
    device=env.device,
  )
  actions[:, 1] = value
  term = env.action_manager.get_term("hybrid_wheel_leg")
  robot = env.scene["robot"]

  yaw_rates: list[torch.Tensor] = []
  mapped_yaws: list[torch.Tensor] = []
  lin_x: list[torch.Tensor] = []
  pitch_errors: list[torch.Tensor] = []
  terminated_total = 0
  for step in range(settle_steps + measure_steps):
    _force_commands(env, vx=0.0, height=height, pitch=pitch)
    _obs, _rewards, terminated, _time_outs, _extras = env.step(actions)
    _force_commands(env, vx=0.0, height=height, pitch=pitch)
    terminated_total += int(terminated.sum().item())
    if step < settle_steps:
      continue
    data = robot.data
    yaw_rates.append(data.root_link_ang_vel_b[:, 2].detach().cpu())
    wheel = term.wheel_targets.detach().cpu()
    # The half-sum isolates the driven differential: the balance term and
    # the station-keeping common mode cancel between the two wheels.
    mapped_yaws.append(0.5 * (wheel[:, 0] + wheel[:, 1]))
    lin_x.append(data.root_link_lin_vel_b[:, 0].detach().cpu())
    pitch_errors.append((_pitch(data) - pitch).abs().detach().cpu())

  yaw = torch.stack(yaw_rates)
  mapped = torch.stack(mapped_yaws)
  lin = torch.stack(lin_x)
  pitch_error = torch.stack(pitch_errors)
  mean_yaw = float(yaw.mean().item())
  mean_mapped = float(mapped.mean().item())
  transfer = (
    mean_yaw / mean_mapped if abs(mean_mapped) > 1.0e-9 else float("nan")
  )
  return {
    "target_height": float(height),
    "target_pitch": float(pitch),
    "yaw_action": float(value),
    "mean_mapped_yaw": mean_mapped,
    "mean_body_yaw": mean_yaw,
    "transfer": transfer,
    "lin_x_abs_mean": float(lin.abs().mean().item()),
    "pitch_error_abs_p95": float(torch.quantile(pitch_error, 0.95).item()),
    "terminated_events": float(terminated_total),
  }


def transfer_deviation_summary(
  cells: list[dict[str, float]],
  *,
  center: tuple[float, float],
  working_point_action: float = WORKING_POINT_ACTION,
) -> dict[str, object]:
  """Relative transfer deviation of every posture against the center."""

  center_key = (round(center[0], 6), round(center[1], 6))
  center_transfer: dict[float, float] = {}
  for cell in cells:
    key = (round(cell["target_height"], 6), round(cell["target_pitch"], 6))
    if key == center_key:
      center_transfer[cell["yaw_action"]] = cell["transfer"]
  if not center_transfer:
    raise ValueError("Transfer cells do not include the center posture.")

  deviations: list[dict[str, float]] = []
  for cell in cells:
    key = (round(cell["target_height"], 6), round(cell["target_pitch"], 6))
    if key == center_key:
      continue
    reference = center_transfer.get(cell["yaw_action"])
    if reference is None or abs(reference) < 1.0e-9:
      raise ValueError(
        f"No center reference transfer for action {cell['yaw_action']}."
      )
    deviations.append(
      {
        "target_height": cell["target_height"],
        "target_pitch": cell["target_pitch"],
        "yaw_action": cell["yaw_action"],
        "deviation": abs(cell["transfer"] - reference) / abs(reference),
      }
    )
  worst = max(deviations, key=lambda item: item["deviation"], default=None)
  working = [
    item for item in deviations
    if abs(item["yaw_action"]) == abs(working_point_action)
  ]
  worst_working = max(
    working, key=lambda item: item["deviation"], default=None
  )
  return {
    "deviations": deviations,
    "worst_deviation": worst,
    "worst_working_point_deviation": worst_working,
    "working_point_action": abs(working_point_action),
    "deviation_limit": DEVIATION_LIMIT,
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
  action_cfg = cfg.actions["hybrid_wheel_leg"]
  # Raw plant methodology (mirrors probe_hybrid_yaw_transfer --fit-output):
  # drive the yaw head at probe scale with the feedforward zeroed, so the
  # measured differential is exactly the commanded one.
  scales = list(action_cfg.action_scales)
  scales[1] = float(args.probe_yaw_scale)
  action_cfg.action_scales = tuple(scales)
  action_cfg.yaw_feedforward_breakpoints = (
    (-1.0, 0.0), (0.0, 0.0), (1.0, 0.0),
  )
  action_cfg.yaw_calibration_qualified = False
  for line in hybrid_provenance_lines(cfg):
    print(line)
  if not getattr(action_cfg, "station_calibration_qualified", False):
    print(
      "[probe][WARN] No station calibration active: held postures drift "
      "during the sweep and contaminate the transfer. Official data "
      "requires the station artifact."
    )
  posture_command = cfg.commands["posture"]
  postures = sweep_postures(
    tuple(posture_command.height_range),
    tuple(posture_command.pitch_range),
  )
  center = postures[0]

  cells: list[dict[str, float]] = []
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    print(
      f"{'posture(h,p)':>18} {'action':>7} {'mapped':>8} {'body_wz':>8} "
      f"{'transfer':>9} {'|lin_x|':>8} {'p95|dp|':>8} {'term':>5}"
    )
    for height, pitch in postures:
      for value in args.yaw_actions:
        cell = run_transfer_cell(
          env,
          height=height,
          pitch=pitch,
          value=float(value),
          settle_steps=args.settle_steps,
          measure_steps=args.measure_steps,
        )
        cells.append(cell)
        print(
          f"({cell['target_height']:.3f},{cell['target_pitch']:+.3f})"
          f" {cell['yaw_action']:>+7.2f} {cell['mean_mapped_yaw']:>+8.4f}"
          f" {cell['mean_body_yaw']:>+8.4f} {cell['transfer']:>9.4f}"
          f" {cell['lin_x_abs_mean']:>8.4f}"
          f" {cell['pitch_error_abs_p95']:>8.4f}"
          f" {cell['terminated_events']:>5.0f}"
        )
  finally:
    env.close()

  summary = transfer_deviation_summary(cells, center=center)
  worst_working = summary["worst_working_point_deviation"]
  if worst_working is not None:
    verdict = (
      "posture-scheduled feedforward batch required"
      if worst_working["deviation"] > DEVIATION_LIMIT
      else "global yaw calibration stands"
    )
    print(
      f"[probe] worst working-point deviation: "
      f"{worst_working['deviation']:.1%} at "
      f"({worst_working['target_height']:.3f},"
      f"{worst_working['target_pitch']:+.3f}) -> {verdict}"
    )

  if args.fit_output is None:
    return
  payload = {
    "schema_version": 1,
    "kind": "yaw_posture_transfer_qualification",
    "controller_gain_hash": action_cfg.controller_gain_hash,
    "controller_qualified": bool(action_cfg.controller_qualified),
    "posture_map_hash": action_cfg.posture_map_hash,
    "posture_map_qualified": bool(action_cfg.posture_map_qualified),
    "station_calibration_hash": getattr(
      action_cfg, "station_calibration_hash", None
    ),
    "station_calibration_qualified": bool(
      getattr(action_cfg, "station_calibration_qualified", False)
    ),
    "cells": cells,
    "summary": summary,
    "source_probe": {
      "git_sha": _git_sha(),
      "task": args.task,
      "device": args.device,
      "num_envs": args.num_envs,
      "settle_steps": args.settle_steps,
      "measure_steps": args.measure_steps,
      "yaw_actions": [float(value) for value in args.yaw_actions],
      "probe_yaw_scale": float(args.probe_yaw_scale),
    },
  }
  args.fit_output.parent.mkdir(parents=True, exist_ok=True)
  args.fit_output.write_text(
    json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
  )
  print(f"[probe] yaw-posture transfer qualification written: {args.fit_output}")


if __name__ == "__main__":
  main()
