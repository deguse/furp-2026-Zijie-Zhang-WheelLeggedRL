#!/usr/bin/env python3
"""Diagnose WHY the classical stack stalls at the 1 cm stair (P2 k.0 result).

The k.0 sweep (9edb8b7) measured a hard cliff: 48/48 success on flat,
0/48 at every height from 0.01 m up, with zero falls — the stack balances
against the riser indefinitely without climbing. Quasi-static edge-pivot
torque at 1 cm is ~3.6 N·m per wheel against a 5.8 N·m peak, so raw
torque should suffice; this probe discriminates the remaining stall
mechanisms and measures the two unplayed classical cards:

- torque saturation vs wheel spin (friction) vs drive-target collapse
  (the balance loop pulling the wheel command back), via a stall-window
  channel capture at the pre-registered k.0 operating point;
- lean-in posture feedforward (negative pitch commands, outside the
  qualified envelope — diagnostic only) and higher approach speed
  (0.10 m/s, inside the calibration domain, outside the Stage5 task
  range), as success/progress deltas against the k.0 baseline cell.

Observational only: never a gate, never a training entrypoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_PATH = Path(__file__).resolve().parents[1]
SRC_PATH = Path(__file__).resolve().parents[2]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

try:
  from hoppertrex_mjlab.assets.HopperTrex_CFG import (
    RMD_L_9025_35T_PEAK_TORQUE,
    WHEEL_VELOCITY_DAMPING,
  )
  from hoppertrex_mjlab.hybrid.identification import (
    NOMINAL_WHEEL_RADIUS_M,
  )
  from hoppertrex_mjlab.scripts import (
    probe_hybrid_stair_height as stair,
  )
  from hoppertrex_mjlab.tasks.hoppertrex_balance_task import (
    NON_WHEEL_GROUND_SENSOR_NAME,
    non_wheel_ground_contact,
  )
  from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
    hybrid_provenance_lines,
  )
except ImportError:
  from assets.HopperTrex_CFG import (  # type: ignore[no-redef]
    RMD_L_9025_35T_PEAK_TORQUE,
    WHEEL_VELOCITY_DAMPING,
  )
  from hybrid.identification import (  # type: ignore[no-redef]
    NOMINAL_WHEEL_RADIUS_M,
  )
  from tasks.hoppertrex_balance_task import (  # type: ignore[no-redef]
    NON_WHEEL_GROUND_SENSOR_NAME,
    non_wheel_ground_contact,
  )
  from tasks.hoppertrex_hybrid_task import (  # type: ignore[no-redef]
    hybrid_provenance_lines,
  )

  from scripts import (  # type: ignore[no-redef]
    probe_hybrid_stair_height as stair,
  )

import mjlab
from mjlab.envs import ManagerBasedRlEnv

DIAGNOSTIC_HEIGHTS_M = (0.0, 0.01)
OFFICIAL_ENVS_PER_HEIGHT = 16
OFFICIAL_SETTLE_STEPS = 200
OFFICIAL_DRIVE_STEPS = 500
OFFICIAL_STABLE_STEPS = 25
OFFICIAL_STALL_WINDOW_STEPS = 150
SMOKE_ENVS_PER_HEIGHT = 1
SMOKE_SETTLE_STEPS = 5
SMOKE_DRIVE_STEPS = 12
SMOKE_STABLE_STEPS = 3
SMOKE_STALL_WINDOW_STEPS = 6

# Diagnostic card: the k.0 envelope-center height. Pitch is swept per cell.
CARD_NAME = "envelope_center"
CARD_HEIGHT_M = 0.3092089487
# Qualified envelope pitch range at the P1 floored envelope; negative
# pitch (lean-in) is deliberately OUTSIDE it — diagnostic only.
QUALIFIED_PITCH_RANGE = (0.0, 0.032)
# Stage5 task command range vs the velocity-calibration fit domain.
STAGE5_VX_LIMIT_MPS = 0.07
CALIBRATION_VX_LIMIT_MPS = 0.10

COMMAND_CELLS = (
  {"name": "pitch_up_0p032", "pitch_rad": 0.032, "vx_mps": 0.07},
  {"name": "pitch_up_0p016", "pitch_rad": 0.016, "vx_mps": 0.07},
  {"name": "pitch_zero", "pitch_rad": 0.0, "vx_mps": 0.07},
  {"name": "lean_in_0p016", "pitch_rad": -0.016, "vx_mps": 0.07},
  {"name": "lean_in_0p032", "pitch_rad": -0.032, "vx_mps": 0.07},
  {"name": "lean_in_0p048", "pitch_rad": -0.048, "vx_mps": 0.07},
  {"name": "fast_pitch_zero", "pitch_rad": 0.0, "vx_mps": 0.10},
  {"name": "fast_lean_0p032", "pitch_rad": -0.032, "vx_mps": 0.10},
)
BASELINE_CELL_NAME = "pitch_zero"
# Smoke exercises the full runtime path on two representative cells only
# (one qualified, one lean-in) to keep the CPU suite affordable.
SMOKE_CELL_NAMES = ("pitch_zero", "lean_in_0p032")

CLASSIFICATIONS = (
  "CLASSICAL_CARD_CANDIDATE_FOUND",
  "WHEEL_SPIN_FRICTION_LIMITED",
  "TORQUE_SATURATED_STALL",
  "DRIVE_TARGET_COLLAPSED",
  "MIXED_STALL_MECHANISM",
  "INVALID_FLAT_CONTROL_STOP",
)
SUCCESS_CANDIDATE_RATE = 0.5
FLAT_CONTROL_SUCCESS_RATE = 0.9
SATURATION_FRAC_LIMIT = 0.9
SLIP_SPIN_LIMIT_MPS = 0.02
TARGET_COLLAPSE_FRACTION = 0.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--smoke",
    action="store_true",
    help="CPU mechanics smoke: tiny rollout, never evidence eligible.",
  )
  return parser.parse_args(argv)


def protocol_for_mode(smoke: bool, device: str) -> dict[str, Any]:
  cells = (
    tuple(
      cell
      for cell in COMMAND_CELLS
      if cell["name"] in SMOKE_CELL_NAMES
    )
    if smoke
    else COMMAND_CELLS
  )
  return {
    "heights_m": list(DIAGNOSTIC_HEIGHTS_M),
    "command_cells": cells,
    "envs_per_height": (
      SMOKE_ENVS_PER_HEIGHT if smoke else OFFICIAL_ENVS_PER_HEIGHT
    ),
    "settle_steps": SMOKE_SETTLE_STEPS if smoke else OFFICIAL_SETTLE_STEPS,
    "drive_steps": SMOKE_DRIVE_STEPS if smoke else OFFICIAL_DRIVE_STEPS,
    "stable_steps": SMOKE_STABLE_STEPS if smoke else OFFICIAL_STABLE_STEPS,
    "stall_window_steps": (
      SMOKE_STALL_WINDOW_STEPS if smoke else OFFICIAL_STALL_WINDOW_STEPS
    ),
    "evidence_eligible": (not smoke) and device.startswith("cuda"),
  }


def model_wheel_torque(
  target_radps: torch.Tensor,
  actual_radps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Velocity-actuator law: force = damping * (ctrl - qvel), clipped.

  Returns (clipped torque, saturated flag). This is the configured
  actuator model (BuiltinVelocityActuatorCfg, damping 200, forcerange
  +/-5.8), not a sensor read, and is exact for that model.
  """

  raw = WHEEL_VELOCITY_DAMPING * (target_radps - actual_radps)
  clipped = torch.clamp(
    raw, -RMD_L_9025_35T_PEAK_TORQUE, RMD_L_9025_35T_PEAK_TORQUE
  )
  saturated = raw.abs() >= RMD_L_9025_35T_PEAK_TORQUE
  return clipped, saturated


def signed_balance_channel(wheel_values: torch.Tensor) -> torch.Tensor:
  """Project left/right wheel values onto the forward balance channel.

  HopperTrex uses opposite wheel signs for forward motion. A plain mean
  projects onto yaw and would erase the drive target at commanded yaw=0.
  """

  if wheel_values.ndim != 2 or wheel_values.shape[1] != 2:
    raise ValueError("wheel_values must have shape (num_envs, 2).")
  return 0.5 * (wheel_values[:, 1] - wheel_values[:, 0])


def _pitch_from_gravity(robot: Any) -> torch.Tensor:
  gravity = robot.data.projected_gravity_b
  return torch.atan2(
    gravity[:, 0], torch.clamp(-gravity[:, 2], min=1.0e-6)
  )


def cell_flags(cell: dict[str, Any]) -> dict[str, bool]:
  pitch = float(cell["pitch_rad"])
  vx = float(cell["vx_mps"])
  return {
    "pitch_in_qualified_envelope": (
      QUALIFIED_PITCH_RANGE[0] <= pitch <= QUALIFIED_PITCH_RANGE[1]
    ),
    "vx_in_stage5_range": abs(vx) <= STAGE5_VX_LIMIT_MPS + 1.0e-12,
    "vx_in_calibration_domain": abs(vx) <= CALIBRATION_VX_LIMIT_MPS + 1.0e-12,
  }


def run_cell(
  env: ManagerBasedRlEnv,
  *,
  heights: tuple[float, ...],
  cell: dict[str, Any],
  protocol: dict[str, Any],
) -> list[dict[str, Any]]:
  """One command cell over all stair columns, with a stall-window capture.

  Resets are PAIRED across cells (repeat=1 for every cell) so that
  cell-to-cell deltas are same-initial-condition comparisons.
  """

  terrain_types, cross_x, reset_metadata = stair._reset_to_approach(
    env,
    root_height=CARD_HEIGHT_M,
    card_name=CARD_NAME,
    repeat=1,
  )
  if int(terrain_types.max().item()) >= len(heights):
    raise RuntimeError("Terrain type index exceeds the diagnostic heights.")

  robot = env.scene["robot"]
  term = env.action_manager.get_term("hybrid_wheel_leg")
  wheel_ids = term._wheel_ids
  action_dim = env.action_space.shape[-1]
  actions = torch.zeros((env.num_envs, action_dim), device=env.device)
  terminated_ever = torch.zeros(
    env.num_envs, dtype=torch.bool, device=env.device
  )
  contact_ever = torch.zeros_like(terminated_ever)
  success = torch.zeros_like(terminated_ever)
  stable = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  max_progress = reset_metadata["x_relative_to_face_m"].clone()
  pitch_cmd = float(cell["pitch_rad"])
  vx_cmd = float(cell["vx_mps"])
  drive_steps = int(protocol["drive_steps"])
  window_steps = int(protocol["stall_window_steps"])
  window_start = drive_steps - window_steps
  window: dict[str, list[torch.Tensor]] = {
    "wheel_target_radps": [],
    "wheel_speed_radps": [],
    "torque_abs_nm": [],
    "saturated": [],
    "pitch_rad": [],
    "body_vx_mps": [],
  }
  window_progress_start: torch.Tensor | None = None

  def _step(vx: float, drive_index: int | None) -> None:
    nonlocal window_progress_start
    was_active = ~success & ~terminated_ever
    stair._force_commands(env, vx=vx, height=CARD_HEIGHT_M, pitch=pitch_cmd)
    _obs, _reward, terminated, _timeouts, _extras = env.step(actions)
    stair._force_commands(env, vx=vx, height=CARD_HEIGHT_M, pitch=pitch_cmd)
    terminated_ever.logical_or_(was_active & terminated)
    active = was_active & ~terminated
    direct_contact = non_wheel_ground_contact(
      env, NON_WHEEL_GROUND_SENSOR_NAME
    ).bool()
    termination_contact = env.termination_manager.get_term(
      "non_wheel_ground_contact"
    )
    contact = stair.merge_contact_observations(
      direct_contact, termination_contact
    )
    contact_ever.copy_(
      stair.update_contact_history(contact_ever, contact, was_active)
    )
    progress = robot.data.root_link_pos_w[:, 0] - (
      cross_x - stair.CROSS_DEPTH_M
    )
    max_progress.copy_(
      stair.update_valid_max_progress(max_progress, progress, active)
    )
    if drive_index is None:
      return
    stable.copy_(
      torch.where(
        active
        & ~contact
        & (robot.data.root_link_pos_w[:, 0] >= cross_x),
        stable + 1,
        torch.zeros_like(stable),
      )
    )
    newly_successful = (
      active & ~contact & (stable >= int(protocol["stable_steps"])) & ~success
    )
    success.logical_or_(newly_successful)
    if drive_index >= window_start:
      if window_progress_start is None:
        window_progress_start = max_progress.clone()
      wheel_speed = signed_balance_channel(
        robot.data.joint_vel[:, wheel_ids]
      )
      wheel_target = signed_balance_channel(term.wheel_targets)
      torque, saturated = model_wheel_torque(
        term.wheel_targets, robot.data.joint_vel[:, wheel_ids]
      )
      window["wheel_target_radps"].append(wheel_target.detach())
      window["wheel_speed_radps"].append(wheel_speed.detach())
      window["torque_abs_nm"].append(torque.abs().mean(dim=-1).detach())
      window["saturated"].append(saturated.any(dim=-1).detach())
      window["pitch_rad"].append(_pitch_from_gravity(robot).detach())
      window["body_vx_mps"].append(
        robot.data.root_link_lin_vel_b[:, 0].detach()
      )

  for _ in range(int(protocol["settle_steps"])):
    _step(0.0, None)
  for drive_index in range(drive_steps):
    _step(vx_cmd, drive_index)

  stacked = {key: torch.stack(values) for key, values in window.items()}
  if window_progress_start is None:
    raise RuntimeError("Stall window never opened; drive_steps too small.")
  window_duration_s = window_steps / stair.CONTROL_FREQUENCY_HZ
  progress_rate = (max_progress - window_progress_start) / window_duration_s
  wheel_speed_mps = stacked["wheel_speed_radps"] * NOMINAL_WHEEL_RADIUS_M
  slip = (wheel_speed_mps - stacked["body_vx_mps"]).abs()

  rows: list[dict[str, Any]] = []
  terrain_types_cpu = terrain_types.detach().cpu().tolist()
  for env_id, terrain_type in enumerate(terrain_types_cpu):
    rows.append({
      "cell": str(cell["name"]),
      "pitch_rad": pitch_cmd,
      "vx_mps": vx_cmd,
      "stair_height_m": float(heights[terrain_type]),
      "env_id": env_id,
      "success": bool(success[env_id].item()),
      "terminated": bool(terminated_ever[env_id].item()),
      "non_wheel_contact": bool(contact_ever[env_id].item()),
      "max_progress_past_face_m": float(max_progress[env_id].item()),
      "stall_window": {
        "wheel_target_radps_mean": float(
          stacked["wheel_target_radps"][:, env_id].mean().item()
        ),
        "wheel_speed_radps_mean": float(
          stacked["wheel_speed_radps"][:, env_id].mean().item()
        ),
        "model_torque_abs_nm_mean": float(
          stacked["torque_abs_nm"][:, env_id].mean().item()
        ),
        "model_torque_abs_nm_p95": float(
          stacked["torque_abs_nm"][:, env_id].quantile(0.95).item()
        ),
        "torque_saturation_frac": float(
          stacked["saturated"][:, env_id].float().mean().item()
        ),
        "wheel_slip_mps_mean": float(slip[:, env_id].mean().item()),
        "pitch_rad_mean": float(
          stacked["pitch_rad"][:, env_id].mean().item()
        ),
        "pitch_error_rad_mean": float(
          (stacked["pitch_rad"][:, env_id] - pitch_cmd).mean().item()
        ),
        "body_vx_mps_mean": float(
          stacked["body_vx_mps"][:, env_id].mean().item()
        ),
        "progress_rate_mps": float(progress_rate[env_id].item()),
      },
    })
  return rows


def aggregate_cells(
  rows: list[dict[str, Any]],
  *,
  command_cells: tuple[dict[str, Any], ...],
  heights: tuple[float, ...],
  expected_trials: int | None = None,
) -> list[dict[str, Any]]:
  cells: list[dict[str, Any]] = []
  for cell in command_cells:
    for height in heights:
      members = [
        row
        for row in rows
        if row["cell"] == cell["name"]
        and abs(row["stair_height_m"] - height) < 1.0e-9
      ]
      if not members:
        raise ValueError(
          f"Cell {cell['name']} at {height} has no trials."
        )
      if expected_trials is not None and len(members) != expected_trials:
        raise ValueError(
          f"Cell {cell['name']} at {height} has {len(members)} trials; "
          f"expected {expected_trials}."
        )
      env_ids = [int(row["env_id"]) for row in members]
      if len(set(env_ids)) != len(env_ids):
        raise ValueError(
          f"Cell {cell['name']} at {height} has duplicate env_ids."
        )
      window_keys = members[0]["stall_window"].keys()
      cells.append({
        "cell": str(cell["name"]),
        "pitch_rad": float(cell["pitch_rad"]),
        "vx_mps": float(cell["vx_mps"]),
        **cell_flags(cell),
        "stair_height_m": float(height),
        "trials": len(members),
        "successes": sum(bool(row["success"]) for row in members),
        "success_rate": (
          sum(bool(row["success"]) for row in members) / len(members)
        ),
        "terminated_trials": sum(
          bool(row["terminated"]) for row in members
        ),
        "non_wheel_contact_trials": sum(
          bool(row["non_wheel_contact"]) for row in members
        ),
        "max_progress_past_face_p50_m": float(
          sorted(row["max_progress_past_face_m"] for row in members)[
            len(members) // 2
          ]
        ),
        "stall_window": {
          key: (
            sum(row["stall_window"][key] for row in members) / len(members)
          )
          for key in window_keys
        },
      })
  return cells


def classify_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
  """Pre-registered observational triage of the stall mechanism."""

  flat_cells = {
    str(cell["cell"]): cell
    for cell in cells
    if abs(float(cell["stair_height_m"])) <= 1.0e-12
  }

  def flat_control_pass(cell: dict[str, Any] | None) -> bool:
    return bool(
      cell is not None
      and float(cell.get("success_rate", 0.0)) >= FLAT_CONTROL_SUCCESS_RATE
      and int(cell.get("terminated_trials", 0)) == 0
      and int(cell.get("non_wheel_contact_trials", 0)) == 0
    )

  invalid_flat_cells = sorted(
    name for name, cell in flat_cells.items() if not flat_control_pass(cell)
  )
  baseline_flat = flat_cells.get(BASELINE_CELL_NAME)
  if not flat_control_pass(baseline_flat):
    return {
      "classification": "INVALID_FLAT_CONTROL_STOP",
      "candidate_cells": [],
      "best_cell": None,
      "invalid_flat_cells": invalid_flat_cells,
    }

  stair_cells = [c for c in cells if c["stair_height_m"] > 0.0]
  candidates = [
    c
    for c in stair_cells
    if c["success_rate"] >= SUCCESS_CANDIDATE_RATE
    and flat_control_pass(flat_cells.get(str(c["cell"])))
  ]
  if candidates:
    best = max(candidates, key=lambda c: c["success_rate"])
    return {
      "classification": "CLASSICAL_CARD_CANDIDATE_FOUND",
      "candidate_cells": [c["cell"] for c in candidates],
      "best_cell": best["cell"],
      "invalid_flat_cells": invalid_flat_cells,
    }
  baseline = next(
    c
    for c in stair_cells
    if c["cell"] == BASELINE_CELL_NAME
  )
  window = baseline["stall_window"]
  tracking_target_radps = (
    float(baseline["vx_mps"]) / NOMINAL_WHEEL_RADIUS_M
  )
  if window["torque_saturation_frac"] >= SATURATION_FRAC_LIMIT:
    if window["wheel_slip_mps_mean"] > SLIP_SPIN_LIMIT_MPS:
      classification = "WHEEL_SPIN_FRICTION_LIMITED"
    else:
      classification = "TORQUE_SATURATED_STALL"
  elif (
    abs(window["wheel_target_radps_mean"])
    < TARGET_COLLAPSE_FRACTION * tracking_target_radps
  ):
    classification = "DRIVE_TARGET_COLLAPSED"
  else:
    classification = "MIXED_STALL_MECHANISM"
  return {
    "classification": classification,
    "candidate_cells": [],
    "best_cell": None,
    "invalid_flat_cells": invalid_flat_cells,
  }


def build_payload(
  *,
  rows: list[dict[str, Any]],
  cells: list[dict[str, Any]],
  verdict: dict[str, Any] | None,
  action_cfg: Any,
  protocol: dict[str, Any],
  device: str,
) -> dict[str, Any]:
  mjlab_root = Path(mjlab.__file__).resolve().parents[2]
  return {
    "schema_version": 1,
    "probe": "hybrid_p2_stall_mechanism_diagnostic",
    "evidence_eligible": bool(protocol["evidence_eligible"]),
    "promotion_eligible": False,
    "training_eligible": False,
    "classification": (
      None if verdict is None else verdict["classification"]
    ),
    "candidate_cells": (
      [] if verdict is None else verdict["candidate_cells"]
    ),
    "best_cell": None if verdict is None else verdict["best_cell"],
    "task": stair.TASK,
    "seed": stair.SEED,
    "git_sha": stair._git_sha(stair.REPOSITORY_PATH),
    "mjlab_git_sha": stair._git_sha(mjlab_root),
    "device": device,
    "runtime": stair._runtime_metadata(device),
    "checkpoint": None,
    "checkpoint_file_sha256": None,
    "controller_gain_hash": action_cfg.controller_gain_hash,
    "calibration_hash": action_cfg.calibration_hash,
    "yaw_calibration_hash": action_cfg.yaw_calibration_hash,
    "posture_map_hash": action_cfg.posture_map_hash,
    "posture_artifact_hash": action_cfg.posture_artifact_hash,
    "station_calibration_hash": action_cfg.station_calibration_hash,
    "action_scales": list(action_cfg.action_scales),
    "protocol": {
      "diagnostic_card": {
        "name": CARD_NAME,
        "height_m": CARD_HEIGHT_M,
      },
      "command_cells": [
        dict(cell) for cell in protocol["command_cells"]
      ],
      "baseline_cell": BASELINE_CELL_NAME,
      "paired_resets_across_cells": True,
      "environment_seed": stair.SEED,
      "terrain": "pyramid_stairs",
      "step_width_m": stair.STEP_WIDTH_M,
      "heights_m": list(protocol["heights_m"]),
      "envs_per_height": int(protocol["envs_per_height"]),
      "settle_steps": int(protocol["settle_steps"]),
      "drive_steps": int(protocol["drive_steps"]),
      "stable_steps": int(protocol["stable_steps"]),
      "stall_window_steps": int(protocol["stall_window_steps"]),
      "qualified_pitch_range_rad": list(QUALIFIED_PITCH_RANGE),
      "stage5_vx_limit_mps": STAGE5_VX_LIMIT_MPS,
      "calibration_vx_limit_mps": CALIBRATION_VX_LIMIT_MPS,
      "commanded_yaw_rate": 0.0,
      "policy_action": [0.0] * 6,
      "wheel_model": {
        "radius_m": NOMINAL_WHEEL_RADIUS_M,
        "peak_torque_nm": RMD_L_9025_35T_PEAK_TORQUE,
        "velocity_damping": WHEEL_VELOCITY_DAMPING,
        "forward_channel": "0.5 * (right - left)",
        "torque_source": (
          "actuator model damping*(target-actual) clipped to peak; "
          "exact for BuiltinVelocityActuatorCfg, not a sensor read"
        ),
      },
      "design_context_estimates": {
        "quasi_static_edge_pivot_torque_nm_per_wheel": {
          "0.01": 3.59,
          "0.02": 4.94,
          "0.03": 5.88,
        },
        "note": (
          "(m/2)*g*sqrt(2Rh-h^2) with m=16.77 kg, R=0.1 m; analytic "
          "estimate for context, not a measurement"
        ),
      },
      "classifications": list(CLASSIFICATIONS),
      "classification_rules": {
        "success_candidate_rate": SUCCESS_CANDIDATE_RATE,
        "flat_control_success_rate": FLAT_CONTROL_SUCCESS_RATE,
        "saturation_frac_limit": SATURATION_FRAC_LIMIT,
        "slip_spin_limit_mps": SLIP_SPIN_LIMIT_MPS,
        "target_collapse_fraction": TARGET_COLLAPSE_FRACTION,
      },
    },
    "cells": cells,
    "trials": rows,
  }


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite output: {args.output}")
  protocol = protocol_for_mode(args.smoke, args.device)
  heights = tuple(float(value) for value in protocol["heights_m"])
  cfg = stair.make_stair_env_cfg(
    heights, int(protocol["envs_per_height"])
  )
  for line in hybrid_provenance_lines(cfg):
    print(line)
  action_cfg = cfg.actions["hybrid_wheel_leg"]
  if protocol["evidence_eligible"]:
    required = {
      "controller": action_cfg.controller_qualified,
      "velocity calibration": action_cfg.calibration_hash,
      "posture map": action_cfg.posture_map_qualified,
      "posture artifact hash": action_cfg.posture_artifact_hash,
      "station calibration": action_cfg.station_calibration_qualified,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
      raise ValueError(
        "Official stall diagnostic lacks: " + ", ".join(missing)
      )
    if action_cfg.yaw_calibration_hash is not None:
      raise ValueError(
        "Official zero-yaw diagnostic must not load a yaw artifact."
      )

  rows: list[dict[str, Any]] = []
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    for cell in protocol["command_cells"]:
      print(f"[stall] cell={cell['name']}")
      rows.extend(run_cell(
        env,
        heights=heights,
        cell=cell,
        protocol=protocol,
      ))
  finally:
    env.close()

  cells = aggregate_cells(
    rows,
    command_cells=tuple(protocol["command_cells"]),
    heights=heights,
    expected_trials=int(protocol["envs_per_height"]),
  )
  verdict = (
    classify_cells(cells) if protocol["evidence_eligible"] else None
  )
  payload = build_payload(
    rows=rows,
    cells=cells,
    verdict=verdict,
    action_cfg=action_cfg,
    protocol=protocol,
    device=args.device,
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
    encoding="utf-8",
  )
  print(f"[stall] output={args.output}")
  print(f"[stall] classification={payload['classification']}")


if __name__ == "__main__":
  main()
