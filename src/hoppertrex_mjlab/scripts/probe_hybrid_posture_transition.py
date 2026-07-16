#!/usr/bin/env python3
"""Posture transition and cross-posture disturbance probe (Stage 3.0 k.0).

The static qualification probe certified balance, tracking floors, and
(after the station-keeping feedforward) station keeping across the posture
envelope. What it never measured is the DYNAMIC residue - the domain where
the classical stack is thin by construction and where any Stage3 PPO value
claim must therefore be tested:

* posture TRANSITIONS: the posture map is a static map and the commands
  step; nothing classical owns the transient between two postures;
* DISTURBANCE recovery away from the nominal posture: the LQR was
  identified at one posture, and its recovery behavior across the envelope
  is unverified.

The probe drives a Stage3 play env with zero residual actions (classical
layer only, station compensation active), steps the posture command through
center/corner/axis legs, and kicks the plant at the envelope waypoints with
the exact Stage1 kick (same impulse, same healthy bands, same settling-time
estimator imported from the gate) so cross-posture recovery times are
directly comparable to the frozen Stage1 baselines. ``--fit-output`` writes
the qualification JSON that pre-registers where - if anywhere - PPO has
measured headroom.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_PATH = Path(__file__).resolve().parents[1]
SRC_PATH = Path(__file__).resolve().parents[2]
REPOSITORY_PATH = Path(__file__).resolve().parents[3]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

try:
  import hoppertrex_mjlab.tasks as tasks  # noqa: E402,F401
  from hoppertrex_mjlab.scripts.probe_hybrid_posture_balance import (
    _force_commands,
    _pitch,
  )
  from hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate import (
    _apply_stage1_kick,
    _settling_time_s,
    CONTROL_FREQUENCY_HZ,
  )
  from hoppertrex_mjlab.tasks.hoppertrex_balance_task import (
    NON_WHEEL_GROUND_SENSOR_NAME,
    non_wheel_ground_contact,
  )
  from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
    hybrid_provenance_lines,
  )
except ImportError:
  import tasks  # noqa: E402,F401
  from scripts.probe_hybrid_posture_balance import (  # type: ignore[no-redef]
    _force_commands,
    _pitch,
  )
  from scripts.rsl_rl.evaluate_hybrid_gate import (  # type: ignore[no-redef]
    _apply_stage1_kick,
    _settling_time_s,
    CONTROL_FREQUENCY_HZ,
  )
  from tasks.hoppertrex_balance_task import (  # type: ignore[no-redef]
    NON_WHEEL_GROUND_SENSOR_NAME,
    non_wheel_ground_contact,
  )
  from tasks.hoppertrex_hybrid_task import (  # type: ignore[no-redef]
    hybrid_provenance_lines,
  )
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

# Same healthy bands as the gate's posture suite (lin/yaw/height/pitch), so
# "recovered" means the same thing here as in every recorded gate number.
HEALTHY_LIN_BAND = 0.06
HEALTHY_YAW_BAND = 0.08
HEALTHY_HEIGHT_BAND = 0.015
HEALTHY_PITCH_BAND = 0.04


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default="HopperTrex-Hybrid-v2-Stage3")
  parser.add_argument(
    "--device",
    default="cuda:0" if torch.cuda.is_available() else "cpu",
  )
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--settle-steps", type=int, default=150)
  parser.add_argument("--measure-steps", type=int, default=300)
  parser.add_argument("--kicks-per-posture", type=int, default=4)
  parser.add_argument("--kick-interval", type=int, default=200)
  parser.add_argument(
    "--height-band",
    type=float,
    default=0.005,
    help="Transition settled band on height error (m); static floor 0.0004.",
  )
  parser.add_argument(
    "--pitch-band",
    type=float,
    default=0.015,
    help="Transition settled band on pitch error (rad); static floor 0.005.",
  )
  parser.add_argument(
    "--height-slew-rate",
    type=float,
    default=None,
    help=(
      "Override the posture command height slew rate (m/s) for the shaping "
      "matrix; 0 disables shaping (legacy step commands)."
    ),
  )
  parser.add_argument(
    "--pitch-slew-rate",
    type=float,
    default=None,
    help=(
      "Override the posture command pitch slew rate (rad/s) for the shaping "
      "matrix; 0 disables shaping (legacy step commands)."
    ),
  )
  parser.add_argument(
    "--fit-output",
    type=Path,
    default=None,
    help="Write the transition/disturbance qualification JSON to this path.",
  )
  return parser.parse_args(argv)


def transition_legs(
  height_range: tuple[float, float],
  pitch_range: tuple[float, float],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
  """Center<->corner legs plus single-axis legs across the envelope."""

  h_lo, h_hi = (float(height_range[0]), float(height_range[1]))
  p_lo, p_hi = (float(pitch_range[0]), float(pitch_range[1]))
  center = (0.5 * (h_lo + h_hi), 0.5 * (p_lo + p_hi))
  corners = [(h_lo, p_lo), (h_lo, p_hi), (h_hi, p_lo), (h_hi, p_hi)]
  legs: list[tuple[tuple[float, float], tuple[float, float]]] = []
  for corner in corners:
    legs.append((center, corner))
    legs.append((corner, center))
  legs.append(((h_lo, center[1]), (h_hi, center[1])))
  legs.append(((h_hi, center[1]), (h_lo, center[1])))
  legs.append(((center[0], p_lo), (center[0], p_hi)))
  legs.append(((center[0], p_hi), (center[0], p_lo)))
  return legs


def kick_postures(
  height_range: tuple[float, float],
  pitch_range: tuple[float, float],
) -> list[tuple[float, float]]:
  h_lo, h_hi = (float(height_range[0]), float(height_range[1]))
  p_lo, p_hi = (float(pitch_range[0]), float(pitch_range[1]))
  return [
    (0.5 * (h_lo + h_hi), 0.5 * (p_lo + p_hi)),
    (h_lo, p_lo),
    (h_lo, p_hi),
    (h_hi, p_lo),
    (h_hi, p_hi),
  ]


def _force_transition_target(
  env: ManagerBasedRlEnv,
  *,
  height: float,
  pitch: float,
) -> None:
  """Zero the twist and move only the RAW posture target.

  Unlike the static ``_force_commands`` snap, this leaves the shaped
  command alone so the reference-shaping slew (if configured) generates
  the transient under measurement.
  """

  twist = env.command_manager.get_term("twist")
  for attribute in ("vel_command_b", "vel_command_w"):
    command = getattr(twist, attribute)
    command[:, :] = 0.0
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
  target = getattr(posture, "_target", None)
  if target is None:
    raise AttributeError("Posture command term does not expose _target.")
  target[:, 0] = height
  target[:, 1] = pitch


def _directional_overshoot(
  values: torch.Tensor,
  start: float,
  target: float,
) -> float:
  """Worst excursion beyond the target along the direction of travel."""

  travel = target - start
  if abs(travel) < 1.0e-9:
    return 0.0
  direction = 1.0 if travel > 0.0 else -1.0
  overshoot = (values - target) * direction
  return float(torch.clamp(overshoot, min=0.0).max().item())


def run_transition(
  env: ManagerBasedRlEnv,
  *,
  start: tuple[float, float],
  target: tuple[float, float],
  settle_steps: int,
  measure_steps: int,
  height_band: float,
  pitch_band: float,
) -> dict[str, float]:
  """Settle at ``start``, step the command to ``target``, measure the leg."""

  env.reset()
  actions = torch.zeros(
    (env.num_envs, env.action_space.shape[-1]),
    device=env.device,
  )
  robot = env.scene["robot"]
  for _ in range(settle_steps):
    _force_commands(env, vx=0.0, height=start[0], pitch=start[1])
    env.step(actions)
    _force_commands(env, vx=0.0, height=start[0], pitch=start[1])

  heights: list[torch.Tensor] = []
  pitches: list[torch.Tensor] = []
  pitch_rates: list[torch.Tensor] = []
  lin_x: list[torch.Tensor] = []
  contacts: list[torch.Tensor] = []
  terminated_total = 0
  for _ in range(measure_steps):
    _force_transition_target(env, height=target[0], pitch=target[1])
    _obs, _rewards, terminated, _time_outs, _extras = env.step(actions)
    _force_transition_target(env, height=target[0], pitch=target[1])
    terminated_total += int(terminated.sum().item())
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

  height_series = torch.stack(heights)
  pitch_series = torch.stack(pitches)
  pitch_rate_abs = torch.stack(pitch_rates)
  lin = torch.stack(lin_x)
  contact = torch.stack(contacts).float()
  in_band = (
    ((height_series - target[0]).abs() <= height_band)
    & ((pitch_series - target[1]).abs() <= pitch_band)
  )
  return {
    "start_height": float(start[0]),
    "start_pitch": float(start[1]),
    "target_height": float(target[0]),
    "target_pitch": float(target[1]),
    "settling_time_s": float(_settling_time_s(in_band, [0])),
    "height_overshoot": _directional_overshoot(
      height_series, start[0], target[0]
    ),
    "pitch_overshoot": _directional_overshoot(
      pitch_series, start[1], target[1]
    ),
    "pitch_rate_abs_max": float(pitch_rate_abs.max().item()),
    "pitch_rate_abs_p99": float(torch.quantile(pitch_rate_abs, 0.99).item()),
    "lin_x_abs_max": float(lin.abs().max().item()),
    "lin_x_abs_mean": float(lin.abs().mean().item()),
    "non_wheel_contact_rate": float(contact.mean().item()),
    "terminated_events": float(terminated_total),
  }


def run_kick_cell(
  env: ManagerBasedRlEnv,
  *,
  height: float,
  pitch: float,
  kicks: int,
  kick_interval: int,
  settle_steps: int,
) -> dict[str, float]:
  """Hold one posture and kick it with the exact Stage1 impulse."""

  env.reset()
  actions = torch.zeros(
    (env.num_envs, env.action_space.shape[-1]),
    device=env.device,
  )
  robot = env.scene["robot"]
  shim = SimpleNamespace(unwrapped=env)
  env_ids = torch.arange(env.num_envs, device=env.device)
  kick_steps = {
    settle_steps + index * kick_interval: index for index in range(kicks)
  }
  total_steps = settle_steps + kicks * kick_interval

  healthy_rows: list[torch.Tensor] = []
  lin_rows: list[torch.Tensor] = []
  contacts: list[torch.Tensor] = []
  terminated_total = 0
  for step in range(total_steps):
    _force_commands(env, vx=0.0, height=height, pitch=pitch)
    if step in kick_steps:
      _apply_stage1_kick(shim, env_ids, kick_index=kick_steps[step])
    _obs, _rewards, terminated, _time_outs, _extras = env.step(actions)
    _force_commands(env, vx=0.0, height=height, pitch=pitch)
    terminated_total += int(terminated.sum().item())
    if step < settle_steps:
      continue
    data = robot.data
    lin_error = data.root_link_lin_vel_b[:, 0]
    yaw_error = data.root_link_ang_vel_b[:, 2]
    height_error = data.root_link_pos_w[:, 2] - height
    pitch_error = _pitch(data) - pitch
    healthy = (
      (lin_error.abs() <= HEALTHY_LIN_BAND)
      & (yaw_error.abs() <= HEALTHY_YAW_BAND)
      & (height_error.abs() <= HEALTHY_HEIGHT_BAND)
      & (pitch_error.abs() <= HEALTHY_PITCH_BAND)
    )
    healthy_rows.append(healthy.detach().cpu())
    lin_rows.append(lin_error.detach().cpu())
    contacts.append(
      non_wheel_ground_contact(env, NON_WHEEL_GROUND_SENSOR_NAME)
      .detach()
      .cpu()
    )

  healthy_series = torch.stack(healthy_rows)
  lin = torch.stack(lin_rows)
  contact = torch.stack(contacts).float()
  kick_relative = [step - settle_steps for step in sorted(kick_steps)]
  return {
    "target_height": float(height),
    "target_pitch": float(pitch),
    "recovery_time_s": float(
      _settling_time_s(healthy_series, kick_relative)
    ),
    "kick_event_count": float(kicks * env.num_envs),
    "post_kick_lin_x_abs_max": float(lin.abs().max().item()),
    "non_wheel_contact_rate": float(contact.mean().item()),
    "terminated_events": float(terminated_total),
  }


def qualification_payload(
  *,
  transitions: list[dict[str, float]],
  kick_cells: list[dict[str, float]],
  controller_gain_hash: str | None,
  controller_qualified: bool,
  posture_map_hash: str | None,
  posture_map_qualified: bool,
  station_calibration_hash: str | None,
  station_calibration_qualified: bool,
  source_probe: dict[str, object],
) -> dict[str, object]:
  if not transitions or not kick_cells:
    raise ValueError(
      "Qualification requires measured transitions and kick cells."
    )
  terminated = sum(
    cell["terminated_events"] for cell in transitions + kick_cells
  )
  return {
    "schema_version": 1,
    "kind": "posture_transition_qualification",
    "controller_gain_hash": controller_gain_hash,
    "controller_qualified": bool(controller_qualified),
    "posture_map_hash": posture_map_hash,
    "posture_map_qualified": bool(posture_map_qualified),
    "station_calibration_hash": station_calibration_hash,
    "station_calibration_qualified": bool(station_calibration_qualified),
    "transitions": transitions,
    "kick_cells": kick_cells,
    "summary": {
      "transition_count": len(transitions),
      "kick_cell_count": len(kick_cells),
      "kick_event_count_total": sum(
        cell["kick_event_count"] for cell in kick_cells
      ),
      "terminated_events": terminated,
      "worst_settling_time_s": max(
        cell["settling_time_s"] for cell in transitions
      ),
      "worst_height_overshoot": max(
        cell["height_overshoot"] for cell in transitions
      ),
      "worst_pitch_overshoot": max(
        cell["pitch_overshoot"] for cell in transitions
      ),
      "worst_transition_pitch_rate_abs_max": max(
        cell["pitch_rate_abs_max"] for cell in transitions
      ),
      "worst_transition_lin_x_abs_max": max(
        cell["lin_x_abs_max"] for cell in transitions
      ),
      "worst_recovery_time_s": max(
        cell["recovery_time_s"] for cell in kick_cells
      ),
      "best_recovery_time_s": min(
        cell["recovery_time_s"] for cell in kick_cells
      ),
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
      "[probe][WARN] No station calibration active: transition and kick "
      "windows will ride on the uncompensated posture drift. Official "
      "Stage 3.0 data requires the station artifact."
    )
  posture_command = cfg.commands["posture"]
  if args.height_slew_rate is not None:
    posture_command.height_slew_rate = (
      None if args.height_slew_rate <= 0.0 else float(args.height_slew_rate)
    )
  if args.pitch_slew_rate is not None:
    posture_command.pitch_slew_rate = (
      None if args.pitch_slew_rate <= 0.0 else float(args.pitch_slew_rate)
    )
  print(
    f"[probe] posture slew rates: height={posture_command.height_slew_rate} "
    f"pitch={posture_command.pitch_slew_rate} (None = legacy step)"
  )
  legs = transition_legs(
    tuple(posture_command.height_range),
    tuple(posture_command.pitch_range),
  )
  postures = kick_postures(
    tuple(posture_command.height_range),
    tuple(posture_command.pitch_range),
  )

  transitions: list[dict[str, float]] = []
  kick_cells: list[dict[str, float]] = []
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    print(
      f"{'from(h,p)':>18} {'to(h,p)':>18} {'settle_s':>8} {'h_over':>7} "
      f"{'p_over':>7} {'pr_max':>7} {'|vx|max':>8} {'term':>5}"
    )
    for start, target in legs:
      cell = run_transition(
        env,
        start=start,
        target=target,
        settle_steps=args.settle_steps,
        measure_steps=args.measure_steps,
        height_band=args.height_band,
        pitch_band=args.pitch_band,
      )
      transitions.append(cell)
      print(
        f"({cell['start_height']:.3f},{cell['start_pitch']:+.3f})"
        f" -> ({cell['target_height']:.3f},{cell['target_pitch']:+.3f})"
        f" {cell['settling_time_s']:>8.3f} {cell['height_overshoot']:>7.4f}"
        f" {cell['pitch_overshoot']:>7.4f} {cell['pitch_rate_abs_max']:>7.3f}"
        f" {cell['lin_x_abs_max']:>8.4f} {cell['terminated_events']:>5.0f}"
      )
    print(
      f"{'kick@(h,p)':>18} {'recovery_s':>10} {'events':>7} {'|vx|max':>8} "
      f"{'term':>5}"
    )
    for height, pitch in postures:
      cell = run_kick_cell(
        env,
        height=height,
        pitch=pitch,
        kicks=args.kicks_per_posture,
        kick_interval=args.kick_interval,
        settle_steps=args.settle_steps,
      )
      kick_cells.append(cell)
      print(
        f"({cell['target_height']:.3f},{cell['target_pitch']:+.3f})"
        f" {cell['recovery_time_s']:>10.3f} {cell['kick_event_count']:>7.0f}"
        f" {cell['post_kick_lin_x_abs_max']:>8.4f}"
        f" {cell['terminated_events']:>5.0f}"
      )
  finally:
    env.close()

  if args.fit_output is None:
    return
  payload = qualification_payload(
    transitions=transitions,
    kick_cells=kick_cells,
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
      "measure_steps": args.measure_steps,
      "kicks_per_posture": args.kicks_per_posture,
      "kick_interval": args.kick_interval,
      "height_band": float(args.height_band),
      "pitch_band": float(args.pitch_band),
      "height_slew_rate": posture_command.height_slew_rate,
      "pitch_slew_rate": posture_command.pitch_slew_rate,
      "control_frequency_hz": float(CONTROL_FREQUENCY_HZ),
    },
  )
  args.fit_output.parent.mkdir(parents=True, exist_ok=True)
  args.fit_output.write_text(
    json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
  )
  print(f"[probe] posture-transition qualification written: {args.fit_output}")
  print(f"[probe] summary={payload['summary']}")


if __name__ == "__main__":
  main()
