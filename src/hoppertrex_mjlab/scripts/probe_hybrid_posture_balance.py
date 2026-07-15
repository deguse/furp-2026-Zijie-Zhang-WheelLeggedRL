#!/usr/bin/env python3
"""Grid-scan posture x balance qualification for the Hybrid Stage 3.0 probe.

Two gaps make this the hard precondition before any Stage3 PPO step (k.0 of
the Hybrid v3 schema): the posture map artifact has never been produced in
the machine room, and the posture envelope was only ever verified as
kinematically feasible — never as balance-controllable under the Stage0 LQR,
which was identified at a single nominal leg posture.

The probe drives a Stage3 play env with zero residual actions (classical
layer only), sweeps a height x pitch grid over the posture envelope, and
measures per cell: terminations, pitch-deviation p95 and pitch-rate p99
(balance controllability), height/pitch RMSE steady-state floors (the
POSTURE_RULES calibration input), forward drift, and non-wheel contact. It
also spot-checks small vx commands at a few postures to expose velocity
calibration drift across postures. ``--fit-output`` writes the
qualification JSON with the controller/posture-map binding for the 3.1
manifest calibration and the envelope verdict.
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
  from hoppertrex_mjlab.tasks.hoppertrex_balance_task import (
    NON_WHEEL_GROUND_SENSOR_NAME,
    non_wheel_ground_contact,
  )
  from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
    hybrid_provenance_lines,
  )
except ImportError:
  import tasks  # noqa: E402,F401
  from tasks.hoppertrex_balance_task import (  # type: ignore[no-redef]
    NON_WHEEL_GROUND_SENSOR_NAME,
    non_wheel_ground_contact,
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
  parser.add_argument("--settle-steps", type=int, default=100)
  parser.add_argument("--measure-steps", type=int, default=200)
  parser.add_argument("--height-points", type=int, default=5)
  parser.add_argument("--pitch-points", type=int, default=5)
  parser.add_argument(
    "--vx-check",
    type=float,
    default=0.05,
    help="Absolute vx command for the cross-posture velocity spot check.",
  )
  parser.add_argument(
    "--fit-output",
    type=Path,
    default=None,
    help="Write the posture-balance qualification JSON to this path.",
  )
  return parser.parse_args(argv)


def build_grid(
  height_range: tuple[float, float],
  pitch_range: tuple[float, float],
  height_points: int,
  pitch_points: int,
) -> list[tuple[float, float]]:
  """Evenly cover the envelope rectangle, always including center and corners."""

  if height_points < 1 or pitch_points < 1:
    raise ValueError("Grid needs at least one point per axis.")

  def _axis(bounds: tuple[float, float], count: int) -> list[float]:
    low, high = (float(bounds[0]), float(bounds[1]))
    if high < low:
      raise ValueError("Grid axis bounds must be ordered.")
    if count == 1 or high - low <= 1.0e-12:
      return [0.5 * (low + high)]
    return [
      low + (high - low) * index / (count - 1) for index in range(count)
    ]

  return [
    (height, pitch)
    for height in _axis(height_range, height_points)
    for pitch in _axis(pitch_range, pitch_points)
  ]


def vx_check_postures(
  grid: list[tuple[float, float]],
) -> list[tuple[float, float]]:
  """Center plus the two extreme corners: where LQR drift would show first."""

  ordered = sorted(set(grid))
  center = ordered[len(ordered) // 2]
  seen: list[tuple[float, float]] = []
  for posture in (center, ordered[0], ordered[-1]):
    if posture not in seen:
      seen.append(posture)
  return seen


def _force_commands(
  env: ManagerBasedRlEnv,
  *,
  vx: float,
  height: float,
  pitch: float,
) -> None:
  twist = env.command_manager.get_term("twist")
  for attribute in ("vel_command_b", "vel_command_w"):
    command = getattr(twist, attribute)
    command[:, :] = 0.0
    command[:, 0] = vx
  for attribute in (
    "is_standing_env",
    "is_heading_env",
    "is_world_env",
    "is_forward_env",
  ):
    value = getattr(twist, attribute, None)
    if value is not None:
      value[:] = False
  posture = env.command_manager.get_term("posture")
  command = getattr(posture, "_command", None)
  if command is None:
    raise AttributeError("Posture command term does not expose _command.")
  command[:, 0] = height
  command[:, 1] = pitch


def _pitch(robot_data: object) -> torch.Tensor:
  gravity = robot_data.projected_gravity_b
  return torch.atan2(
    gravity[:, 0],
    torch.clamp(-gravity[:, 2], min=1.0e-6),
  )


def run_cell(
  env: ManagerBasedRlEnv,
  *,
  height: float,
  pitch: float,
  vx: float,
  settle_steps: int,
  measure_steps: int,
) -> dict[str, float]:
  """Hold one posture (and optional vx) under zero residual and measure it."""

  env.reset()
  actions = torch.zeros(
    (env.num_envs, env.action_space.shape[-1]),
    device=env.device,
  )
  robot = env.scene["robot"]
  heights: list[torch.Tensor] = []
  pitches: list[torch.Tensor] = []
  pitch_rates: list[torch.Tensor] = []
  lin_x: list[torch.Tensor] = []
  contacts: list[torch.Tensor] = []
  terminated_total = 0
  for step in range(settle_steps + measure_steps):
    _force_commands(env, vx=vx, height=height, pitch=pitch)
    _obs, _rewards, terminated, _time_outs, _extras = env.step(actions)
    _force_commands(env, vx=vx, height=height, pitch=pitch)
    terminated_total += int(terminated.sum().item())
    if step < settle_steps:
      continue
    data = robot.data
    heights.append(data.root_link_pos_w[:, 2].detach().cpu())
    pitches.append(_pitch(data).detach().cpu())
    pitch_rates.append(data.root_link_ang_vel_b[:, 1].abs().detach().cpu())
    lin_x.append(data.root_link_lin_vel_b[:, 0].detach().cpu())
    contacts.append(
      non_wheel_ground_contact(env, NON_WHEEL_GROUND_SENSOR_NAME)
      .detach()
      .cpu()
    )

  height_error = torch.stack(heights) - height
  pitch_error = torch.stack(pitches) - pitch
  pitch_rate_abs = torch.stack(pitch_rates)
  lin = torch.stack(lin_x)
  contact = torch.stack(contacts).float()
  return {
    "target_height": float(height),
    "target_pitch": float(pitch),
    "vx_command": float(vx),
    "height_rmse": float(
      torch.sqrt(torch.mean(torch.square(height_error))).item()
    ),
    "pitch_rmse": float(
      torch.sqrt(torch.mean(torch.square(pitch_error))).item()
    ),
    "pitch_error_abs_p95": float(
      torch.quantile(pitch_error.abs(), 0.95).item()
    ),
    "pitch_rate_abs_p99": float(torch.quantile(pitch_rate_abs, 0.99).item()),
    "mean_actual_lin_x": float(lin.mean().item()),
    "lin_x_abs_mean": float(lin.abs().mean().item()),
    "non_wheel_contact_rate": float(contact.mean().item()),
    "terminated_events": float(terminated_total),
  }


def qualification_payload(
  *,
  grid_cells: list[dict[str, float]],
  vx_cells: list[dict[str, float]],
  controller_gain_hash: str | None,
  controller_qualified: bool,
  posture_map_hash: str | None,
  posture_map_qualified: bool,
  calibration_hash: str | None,
  station_calibration_hash: str | None,
  station_calibration_qualified: bool,
  source_probe: dict[str, object],
) -> dict[str, object]:
  """Assemble the qualification JSON with its double artifact binding."""

  if not grid_cells:
    raise ValueError("Qualification requires at least one measured grid cell.")
  worst_height = max(cell["height_rmse"] for cell in grid_cells)
  worst_pitch = max(cell["pitch_rmse"] for cell in grid_cells)
  worst_pitch_rate = max(cell["pitch_rate_abs_p99"] for cell in grid_cells)
  # vx=0 cells should station-keep; the worst absolute drift is the direct
  # retest verdict for the Stage 3.0 station-keeping compensation.
  worst_drift = max(abs(cell["mean_actual_lin_x"]) for cell in grid_cells)
  terminated = sum(cell["terminated_events"] for cell in grid_cells)
  return {
    "schema_version": 1,
    "kind": "posture_balance_qualification",
    "controller_gain_hash": controller_gain_hash,
    "controller_qualified": bool(controller_qualified),
    "posture_map_hash": posture_map_hash,
    "posture_map_qualified": bool(posture_map_qualified),
    "calibration_hash": calibration_hash,
    "station_calibration_hash": station_calibration_hash,
    "station_calibration_qualified": bool(station_calibration_qualified),
    "grid_cells": grid_cells,
    "vx_checks": vx_cells,
    "summary": {
      "cells": len(grid_cells),
      "terminated_events": terminated,
      "worst_height_rmse": worst_height,
      "worst_pitch_rmse": worst_pitch,
      "worst_pitch_rate_abs_p99": worst_pitch_rate,
      "worst_abs_station_drift": worst_drift,
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
  posture_command = cfg.commands["posture"]
  if not action_cfg.posture_map_qualified:
    print(
      "[probe][WARN] Default (unqualified) posture artifact: the envelope "
      "collapses to the nominal posture, so this run only checks probe "
      "mechanics. Official data requires the machine-room posture map."
    )
  grid = build_grid(
    tuple(posture_command.height_range),
    tuple(posture_command.pitch_range),
    args.height_points,
    args.pitch_points,
  )
  vx_postures = vx_check_postures(grid)

  grid_cells: list[dict[str, float]] = []
  vx_cells: list[dict[str, float]] = []
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    header = (
      f"{'height':>7} {'pitch':>7} {'vx':>6} {'h_rmse':>8} {'p_rmse':>8} "
      f"{'p95|dp|':>8} {'p99|pr|':>8} {'lin_x':>7} {'contact':>8} {'term':>5}"
    )
    print(header)

    def _print_cell(cell: dict[str, float]) -> None:
      print(
        f"{cell['target_height']:>7.3f} {cell['target_pitch']:>+7.3f} "
        f"{cell['vx_command']:>+6.2f} {cell['height_rmse']:>8.4f} "
        f"{cell['pitch_rmse']:>8.4f} {cell['pitch_error_abs_p95']:>8.4f} "
        f"{cell['pitch_rate_abs_p99']:>8.4f} "
        f"{cell['mean_actual_lin_x']:>+7.4f} "
        f"{cell['non_wheel_contact_rate']:>8.4f} "
        f"{cell['terminated_events']:>5.0f}"
      )

    for height, pitch in grid:
      cell = run_cell(
        env,
        height=height,
        pitch=pitch,
        vx=0.0,
        settle_steps=args.settle_steps,
        measure_steps=args.measure_steps,
      )
      grid_cells.append(cell)
      _print_cell(cell)
    for height, pitch in vx_postures:
      for sign in (1.0, -1.0):
        cell = run_cell(
          env,
          height=height,
          pitch=pitch,
          vx=sign * args.vx_check,
          settle_steps=args.settle_steps,
          measure_steps=args.measure_steps,
        )
        vx_cells.append(cell)
        _print_cell(cell)
  finally:
    env.close()

  if args.fit_output is None:
    return
  payload = qualification_payload(
    grid_cells=grid_cells,
    vx_cells=vx_cells,
    controller_gain_hash=action_cfg.controller_gain_hash,
    controller_qualified=bool(action_cfg.controller_qualified),
    posture_map_hash=action_cfg.posture_map_hash,
    posture_map_qualified=bool(action_cfg.posture_map_qualified),
    calibration_hash=action_cfg.calibration_hash,
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
      "measure_steps": args.measure_steps,
      "height_points": args.height_points,
      "pitch_points": args.pitch_points,
      "vx_check": float(args.vx_check),
    },
  )
  args.fit_output.parent.mkdir(parents=True, exist_ok=True)
  args.fit_output.write_text(
    json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
  )
  print(f"[probe] posture-balance qualification written: {args.fit_output}")
  print(f"[probe] summary={payload['summary']}")


if __name__ == "__main__":
  main()
