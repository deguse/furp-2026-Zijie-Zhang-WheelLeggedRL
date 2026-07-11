#!/usr/bin/env python3
"""Collect a static two-leg posture sweep for the Hybrid v2 posture map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch

PROJECT_PATH = Path(__file__).resolve().parents[1]
SRC_PATH = Path(__file__).resolve().parents[2]
REPOSITORY_PATH = Path(__file__).resolve().parents[3]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))


POSTURE_REQUIRED_ARRAY_NAMES = (
  "heights",
  "pitches",
  "joint_positions",
  "non_wheel_contact",
  "joint_lower",
  "joint_upper",
  "actuator_load_fraction",
)
POSTURE_DIAGNOSTIC_ARRAY_NAMES = (
  "target_joint_positions",
  "hip_offsets",
  "knee_offsets",
  "invalid",
)
HIP_DIRECTION = np.array([-1.0, 1.0, 0.0, 0.0])
KNEE_DIRECTION = np.array([0.0, 0.0, -1.0, 1.0])


def posture_sweep_grid(
  *,
  hip_range: tuple[float, float],
  knee_range: tuple[float, float],
  hip_points: int,
  knee_points: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Return deterministic hip-major coordinates for a symmetric joint sweep."""

  if hip_points < 2 or knee_points < 2:
    raise ValueError("Posture sweep requires at least two points per coordinate.")
  if hip_range[0] >= hip_range[1] or knee_range[0] >= knee_range[1]:
    raise ValueError("Posture sweep ranges must be strictly increasing.")
  hip_grid, knee_grid = np.meshgrid(
    np.linspace(*hip_range, hip_points),
    np.linspace(*knee_range, knee_points),
    indexing="ij",
  )
  return hip_grid.ravel(), knee_grid.ravel()


def build_symmetric_leg_targets(
  initial_joint_positions: ArrayLike,
  *,
  hip_offsets: ArrayLike,
  knee_offsets: ArrayLike,
) -> NDArray[np.float64]:
  """Map two coordinates to mirrored targets for two legs and four joints."""

  initial = np.asarray(initial_joint_positions, dtype=np.float64)
  hip = np.asarray(hip_offsets, dtype=np.float64)
  knee = np.asarray(knee_offsets, dtype=np.float64)
  if initial.shape != (4,):
    raise ValueError("initial_joint_positions must contain four values.")
  if hip.ndim != 1 or knee.shape != hip.shape or hip.size < 1:
    raise ValueError("hip_offsets and knee_offsets must be equal non-empty vectors.")
  if not all(np.all(np.isfinite(value)) for value in (initial, hip, knee)):
    raise ValueError("Posture target inputs must contain only finite values.")
  return (
    initial[None, :]
    + hip[:, None] * HIP_DIRECTION[None, :]
    + knee[:, None] * KNEE_DIRECTION[None, :]
  )


def summarize_posture_samples(
  *,
  heights: ArrayLike,
  pitches: ArrayLike,
  joint_positions: ArrayLike,
  non_wheel_contact: ArrayLike,
  actuator_load_fraction: ArrayLike,
  invalid: ArrayLike,
) -> dict[str, NDArray[np.generic]]:
  """Reduce a static sample window conservatively for each sweep point."""

  height_array = np.asarray(heights, dtype=np.float64)
  pitch_array = np.asarray(pitches, dtype=np.float64)
  joint_array = np.asarray(joint_positions, dtype=np.float64)
  contact_array = np.asarray(non_wheel_contact, dtype=bool)
  load_array = np.asarray(actuator_load_fraction, dtype=np.float64)
  invalid_array = np.asarray(invalid, dtype=bool)
  if height_array.ndim != 2 or height_array.shape[0] < 1:
    raise ValueError("heights must have shape (sample_steps, num_points).")
  if pitch_array.shape != height_array.shape:
    raise ValueError("pitches must have the same shape as heights.")
  expected_joint_shape = (*height_array.shape, 4)
  if joint_array.shape != expected_joint_shape:
    raise ValueError("joint_positions must have shape (sample_steps, num_points, 4).")
  if contact_array.shape != height_array.shape:
    raise ValueError("non_wheel_contact must have the same shape as heights.")
  if load_array.shape != expected_joint_shape:
    raise ValueError(
      "actuator_load_fraction must have shape (sample_steps, num_points, 4)."
    )
  if invalid_array.shape != (height_array.shape[1],):
    raise ValueError("invalid must contain one value per sweep point.")
  if not all(
    np.all(np.isfinite(value))
    for value in (height_array, pitch_array, joint_array, load_array)
  ):
    raise ValueError("Posture samples must contain only finite values.")
  return {
    "heights": height_array.mean(axis=0),
    "pitches": pitch_array.mean(axis=0),
    "joint_positions": joint_array.mean(axis=0),
    "non_wheel_contact": contact_array.any(axis=0) | invalid_array,
    "actuator_load_fraction": load_array.max(axis=0),
  }


def write_posture_sweep_dataset(
  output: Path,
  arrays: Mapping[str, ArrayLike],
  metadata: Mapping[str, object],
) -> Path:
  """Write fitter inputs, diagnostic targets, and an auditable JSON sidecar."""

  expected = (*POSTURE_REQUIRED_ARRAY_NAMES, *POSTURE_DIAGNOSTIC_ARRAY_NAMES)
  missing = [name for name in expected if name not in arrays]
  if missing:
    raise ValueError(f"Posture sweep arrays missing: {', '.join(missing)}")
  output = output.resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez(output, **{name: np.asarray(arrays[name]) for name in expected})
  metadata_path = output.with_suffix(".json")
  metadata_path.write_text(
    json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  return metadata_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--controller-path", type=Path, default=None)
  parser.add_argument("--calibration-path", type=Path, default=None)
  parser.add_argument(
    "--allow-unqualified-controller",
    action="store_true",
    help="Permit the local fallback PD for smoke tests only.",
  )
  parser.add_argument("--device", default="cpu")
  parser.add_argument(
    "--hip-range",
    type=float,
    nargs=2,
    default=(-0.18, 0.18),
    metavar=("MIN", "MAX"),
  )
  parser.add_argument(
    "--knee-range",
    type=float,
    nargs=2,
    default=(-0.18, 0.18),
    metavar=("MIN", "MAX"),
  )
  parser.add_argument("--hip-points", type=int, default=7)
  parser.add_argument("--knee-points", type=int, default=7)
  parser.add_argument("--ramp-steps", type=int, default=100)
  parser.add_argument("--settle-steps", type=int, default=200)
  parser.add_argument("--sample-steps", type=int, default=100)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--progress-interval", type=int, default=100)
  return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
  if args.ramp_steps < 0 or args.settle_steps < 0:
    raise ValueError("--ramp-steps and --settle-steps must be non-negative.")
  if args.sample_steps <= 0:
    raise ValueError("--sample-steps must be positive.")
  if args.progress_interval <= 0:
    raise ValueError("--progress-interval must be positive.")


def _git_sha() -> str:
  completed = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=REPOSITORY_PATH,
    check=True,
    capture_output=True,
    text=True,
  )
  return completed.stdout.strip()


def load_runtime_dependencies() -> dict[str, object]:
  """Load simulator dependencies through one task-package namespace."""

  from assets.HopperTrex_CFG import (
    DM_J6248P_PEAK_TORQUE,
    INIT_JOINT_POS,
  )
  from mjlab.envs import ManagerBasedRlEnv
  from hoppertrex_mjlab.tasks.hoppertrex_balance_task import (
    NON_WHEEL_GROUND_SENSOR_NAME,
    non_wheel_ground_contact,
  )
  from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
    make_hoppertrex_hybrid_env_cfg,
  )

  return {
    "peak_torque": DM_J6248P_PEAK_TORQUE,
    "initial_joint_pos": INIT_JOINT_POS,
    "env_class": ManagerBasedRlEnv,
    "non_wheel_sensor_name": NON_WHEEL_GROUND_SENSOR_NAME,
    "non_wheel_ground_contact": non_wheel_ground_contact,
    "make_env_cfg": make_hoppertrex_hybrid_env_cfg,
  }


def _force_zero_velocity_command(env: object) -> None:
  term = env.command_manager.get_term("twist")
  for attribute in ("vel_command_b", "vel_command_w"):
    command = getattr(term, attribute)
    command[:, :] = 0.0
  for attribute in (
    "is_standing_env",
    "is_heading_env",
    "is_world_env",
    "is_forward_env",
  ):
    value = getattr(term, attribute, None)
    if value is not None:
      value[:] = False


def _force_sweep_coordinates(
  env: object,
  hip_offsets: torch.Tensor,
  knee_offsets: torch.Tensor,
  *,
  fraction: float,
) -> None:
  term = env.command_manager.get_term("posture")
  command = getattr(term, "_command", None)
  if command is None:
    raise AttributeError("Posture command term does not expose _command.")
  command[:, 0] = fraction * hip_offsets
  command[:, 1] = fraction * knee_offsets


def _pitch(projected_gravity: torch.Tensor) -> torch.Tensor:
  return torch.atan2(
    projected_gravity[:, 0],
    torch.clamp(-projected_gravity[:, 2], min=1.0e-6),
  )


def collect(args: argparse.Namespace) -> tuple[dict[str, NDArray[np.generic]], dict[str, object]]:
  """Run one parallel static sweep and return fitter-compatible arrays."""

  _validate_args(args)
  dependencies = load_runtime_dependencies()
  peak_torque = float(dependencies["peak_torque"])
  initial_joint_pos = dependencies["initial_joint_pos"]
  env_class = dependencies["env_class"]
  non_wheel_sensor_name = str(dependencies["non_wheel_sensor_name"])
  non_wheel_ground_contact = dependencies["non_wheel_ground_contact"]
  make_env_cfg = dependencies["make_env_cfg"]

  hip_offsets, knee_offsets = posture_sweep_grid(
    hip_range=tuple(args.hip_range),
    knee_range=tuple(args.knee_range),
    hip_points=args.hip_points,
    knee_points=args.knee_points,
  )
  point_count = hip_offsets.size
  cfg = make_env_cfg(
    stage=0,
    play=True,
    controller_path=args.controller_path,
    calibration_path=args.calibration_path,
  )
  action_cfg = cfg.actions["hybrid_wheel_leg"]
  if not action_cfg.controller_qualified and not args.allow_unqualified_controller:
    raise ValueError(
      "A qualified --controller-path is required for a posture artifact; "
      "use --allow-unqualified-controller only for a local smoke test."
    )
  if not action_cfg.calibration_hash and not args.allow_unqualified_controller:
    raise ValueError(
      "A qualified --calibration-path is required for a formal posture artifact."
    )
  initial = np.array(
    [initial_joint_pos[name] for name in action_cfg.leg_joint_names],
    dtype=np.float64,
  )
  target_joint_positions = build_symmetric_leg_targets(
    initial,
    hip_offsets=hip_offsets,
    knee_offsets=knee_offsets,
  )
  action_cfg.posture_coefficients = tuple(
    tuple(float(value) for value in row)
    for row in np.stack((initial, HIP_DIRECTION, KNEE_DIRECTION))
  )
  cfg.scene.num_envs = point_count
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.num_envs = point_count
  cfg.episode_length_s = 1.0e9
  for name in (
    "bad_orientation",
    "root_too_low",
    "non_wheel_ground_contact",
  ):
    cfg.terminations.pop(name, None)
  if hasattr(cfg, "seed"):
    cfg.seed = args.seed
  torch.manual_seed(args.seed)

  env = env_class(cfg=cfg, device=args.device)
  try:
    env.reset()
    robot = env.scene["robot"]
    leg_ids, leg_names = robot.find_joints(
      action_cfg.leg_joint_names,
      preserve_order=True,
    )
    if tuple(leg_names) != action_cfg.leg_joint_names:
      raise ValueError(
        f"Expected leg joints {action_cfg.leg_joint_names}, got {leg_names}."
      )
    leg_id_tensor = torch.tensor(leg_ids, device=env.device, dtype=torch.long)
    joint_limits = robot.data.joint_pos_limits[:, leg_id_tensor, :]
    joint_lower = joint_limits[0, :, 0].detach().cpu().numpy()
    joint_upper = joint_limits[0, :, 1].detach().cpu().numpy()
    target_outside_limits = np.any(
      (target_joint_positions < joint_lower)
      | (target_joint_positions > joint_upper),
      axis=1,
    )
    invalid = torch.as_tensor(
      target_outside_limits,
      device=env.device,
      dtype=torch.bool,
    )
    hip_tensor = torch.as_tensor(
      hip_offsets,
      device=env.device,
      dtype=torch.float,
    )
    knee_tensor = torch.as_tensor(
      knee_offsets,
      device=env.device,
      dtype=torch.float,
    )
    zero_action = torch.zeros(
      (point_count, action_cfg.action_dim),
      device=env.device,
    )
    total_steps = args.ramp_steps + args.settle_steps + args.sample_steps
    height_samples: list[NDArray[np.float64]] = []
    pitch_samples: list[NDArray[np.float64]] = []
    joint_samples: list[NDArray[np.float64]] = []
    contact_samples: list[NDArray[np.bool_]] = []
    load_samples: list[NDArray[np.float64]] = []

    for step in range(total_steps):
      if args.ramp_steps > 0 and step < args.ramp_steps:
        fraction = (step + 1) / args.ramp_steps
      else:
        fraction = 1.0
      _force_zero_velocity_command(env)
      _force_sweep_coordinates(
        env,
        hip_tensor,
        knee_tensor,
        fraction=fraction,
      )
      _, _, terminated, truncated, _ = env.step(zero_action)
      invalid |= terminated | truncated
      _force_zero_velocity_command(env)
      _force_sweep_coordinates(
        env,
        hip_tensor,
        knee_tensor,
        fraction=1.0,
      )
      if step >= args.ramp_steps + args.settle_steps:
        robot_data = robot.data
        height_samples.append(
          robot_data.root_link_pos_w[:, 2].detach().cpu().numpy()
        )
        pitch_samples.append(
          _pitch(robot_data.projected_gravity_b).detach().cpu().numpy()
        )
        joint_samples.append(
          robot_data.joint_pos[:, leg_id_tensor].detach().cpu().numpy()
        )
        contact_samples.append(
          non_wheel_ground_contact(
            env,
            non_wheel_sensor_name,
          ).to(dtype=torch.bool).detach().cpu().numpy()
        )
        load_samples.append(
          (
            torch.abs(robot_data.qfrc_actuator[:, leg_id_tensor])
            / peak_torque
          ).detach().cpu().numpy()
        )
      if (step + 1) % args.progress_interval == 0:
        print(f"[sweep] {step + 1}/{total_steps} steps")

    invalid_array = invalid.detach().cpu().numpy()
    summary = summarize_posture_samples(
      heights=np.stack(height_samples),
      pitches=np.stack(pitch_samples),
      joint_positions=np.stack(joint_samples),
      non_wheel_contact=np.stack(contact_samples),
      actuator_load_fraction=np.stack(load_samples),
      invalid=invalid_array,
    )
    arrays: dict[str, NDArray[np.generic]] = {
      **summary,
      "joint_lower": joint_lower,
      "joint_upper": joint_upper,
      "target_joint_positions": target_joint_positions,
      "hip_offsets": hip_offsets,
      "knee_offsets": knee_offsets,
      "invalid": invalid_array,
    }
    metadata: dict[str, object] = {
      "schema_version": 1,
      "git_sha": _git_sha(),
      "seed": args.seed,
      "device": str(args.device),
      "point_count": int(point_count),
      "grid_shape": [args.hip_points, args.knee_points],
      "hip_range": list(args.hip_range),
      "knee_range": list(args.knee_range),
      "ramp_steps": args.ramp_steps,
      "settle_steps": args.settle_steps,
      "sample_steps": args.sample_steps,
      "joint_names": list(action_cfg.leg_joint_names),
      "target_parameterization": {
        "bias": initial.tolist(),
        "hip_direction": HIP_DIRECTION.tolist(),
        "knee_direction": KNEE_DIRECTION.tolist(),
      },
      "controller": {
        "type": action_cfg.controller_type,
        "qualified": action_cfg.controller_qualified,
        "source": action_cfg.controller_source,
        "gain_hash": action_cfg.controller_gain_hash,
      },
      "calibration": {
        "hash": action_cfg.calibration_hash,
        "velocity_command_scale": action_cfg.velocity_command_scale,
        "velocity_command_bias": action_cfg.velocity_command_bias,
      },
      "invalid_point_count": int(np.count_nonzero(invalid_array)),
      "non_wheel_contact_point_count": int(
        np.count_nonzero(summary["non_wheel_contact"])
      ),
    }
    return arrays, metadata
  finally:
    env.close()


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  arrays, metadata = collect(args)
  metadata_path = write_posture_sweep_dataset(args.output, arrays, metadata)
  print(f"Wrote two-leg posture sweep: {args.output.resolve()}")
  print(f"Wrote posture sweep metadata: {metadata_path}")


if __name__ == "__main__":
  main()
