#!/usr/bin/env python3
"""Collect Hybrid v2 wheel-balance identification transitions on CPU or GPU."""

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

from hoppertrex_mjlab.hybrid.identification import (  # noqa: E402
  CONTROLLER_STATE_NAMES,
  STATE_DEFINITION_VERSION,
)
from hoppertrex_mjlab.hybrid.controller_schedule import (  # noqa: E402
  SCHEDULE_STATE_DEFINITION,
)


IDENTIFICATION_ARRAY_NAMES = (
  "states",
  "inputs",
  "next_states",
  "heldout_states",
  "heldout_inputs",
  "heldout_next_states",
)


def build_controller_state(
  *,
  projected_gravity: torch.Tensor,
  pitch_rate: torch.Tensor,
  root_vx: torch.Tensor,
  commanded_vx: torch.Tensor,
  wheel_velocity: torch.Tensor,
  wheel_radius: float,
) -> torch.Tensor:
  """Build ``[pitch, pitch_rate, vx_error, signed_wheel_speed_error]``."""

  if projected_gravity.ndim != 2 or projected_gravity.shape[1] != 3:
    raise ValueError("projected_gravity must have shape (num_envs, 3).")
  num_envs = projected_gravity.shape[0]
  for name, value in (
    ("pitch_rate", pitch_rate),
    ("root_vx", root_vx),
    ("commanded_vx", commanded_vx),
  ):
    if value.shape != (num_envs,):
      raise ValueError(f"{name} must have shape (num_envs,).")
  if wheel_velocity.shape != (num_envs, 2):
    raise ValueError("wheel_velocity must have shape (num_envs, 2).")
  if wheel_radius <= 0.0:
    raise ValueError("wheel_radius must be positive.")

  pitch = torch.atan2(
    projected_gravity[:, 0],
    torch.clamp(-projected_gravity[:, 2], min=1.0e-6),
  )
  vx_error = root_vx - commanded_vx
  signed_wheel_speed = 0.5 * (
    wheel_velocity[:, 1] - wheel_velocity[:, 0]
  )
  desired_wheel_speed = commanded_vx / wheel_radius
  wheel_speed_error = signed_wheel_speed - desired_wheel_speed
  return torch.stack(
    (pitch, pitch_rate, vx_error, wheel_speed_error),
    dim=1,
  )


def signed_balance_input(wheel_targets: torch.Tensor) -> torch.Tensor:
  """Return the actual signed balance target after slew and saturation."""

  if wheel_targets.ndim != 2 or wheel_targets.shape[1] != 2:
    raise ValueError("wheel_targets must have shape (num_envs, 2).")
  return (
    0.5 * (wheel_targets[:, 1] - wheel_targets[:, 0])
  ).unsqueeze(1)


def _transition_arrays(
  states: ArrayLike,
  inputs: ArrayLike,
  next_states: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
  state_array = np.asarray(states, dtype=np.float64)
  input_array = np.asarray(inputs, dtype=np.float64)
  next_state_array = np.asarray(next_states, dtype=np.float64)
  if (
    state_array.ndim != 2
    or state_array.shape[1] != len(CONTROLLER_STATE_NAMES)
  ):
    raise ValueError("states must have shape (num_samples, 4).")
  if input_array.shape != (state_array.shape[0], 1):
    raise ValueError("inputs must have shape (num_samples, 1).")
  if next_state_array.shape != state_array.shape:
    raise ValueError("next_states must have the same shape as states.")
  if not all(
    np.all(np.isfinite(array))
    for array in (state_array, input_array, next_state_array)
  ):
    raise ValueError("Identification transitions must contain only finite values.")
  return state_array, input_array, next_state_array


def filter_valid_transitions(
  states: ArrayLike,
  inputs: ArrayLike,
  next_states: ArrayLike,
  *,
  terminated: ArrayLike,
  truncated: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
  """Drop transitions whose next state belongs to an automatic reset."""

  state_array, input_array, next_state_array = _transition_arrays(
    states,
    inputs,
    next_states,
  )
  terminated_array = np.asarray(terminated, dtype=bool)
  truncated_array = np.asarray(truncated, dtype=bool)
  expected_shape = (state_array.shape[0],)
  if terminated_array.shape != expected_shape or truncated_array.shape != expected_shape:
    raise ValueError("Termination flags must contain one value per transition.")
  valid = ~(terminated_array | truncated_array)
  return state_array[valid], input_array[valid], next_state_array[valid]


def deterministic_transition_split(
  states: ArrayLike,
  inputs: ArrayLike,
  next_states: ArrayLike,
  *,
  heldout_fraction: float,
  seed: int,
) -> dict[str, NDArray[np.float64]]:
  """Shuffle once with a fixed seed and return fitter-compatible arrays."""

  state_array, input_array, next_state_array = _transition_arrays(
    states,
    inputs,
    next_states,
  )
  sample_count = state_array.shape[0]
  if sample_count < 2:
    raise ValueError("At least two valid transitions are required.")
  if not 0.0 < heldout_fraction < 1.0:
    raise ValueError("heldout_fraction must be in (0, 1).")
  heldout_count = int(round(sample_count * heldout_fraction))
  heldout_count = min(max(heldout_count, 1), sample_count - 1)
  permutation = np.random.default_rng(seed).permutation(sample_count)
  heldout_indices = permutation[:heldout_count]
  train_indices = permutation[heldout_count:]
  return {
    "states": state_array[train_indices],
    "inputs": input_array[train_indices],
    "next_states": next_state_array[train_indices],
    "heldout_states": state_array[heldout_indices],
    "heldout_inputs": input_array[heldout_indices],
    "heldout_next_states": next_state_array[heldout_indices],
  }


def write_identification_dataset(
  output: Path,
  arrays: Mapping[str, ArrayLike],
  metadata: Mapping[str, object],
) -> Path:
  """Write the exact identification NPZ schema and an auditable JSON sidecar."""

  missing = [name for name in IDENTIFICATION_ARRAY_NAMES if name not in arrays]
  if missing:
    raise ValueError(f"Identification arrays missing: {', '.join(missing)}")
  output = output.resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    output,
    **{
      name: np.asarray(arrays[name], dtype=np.float64)
      for name in IDENTIFICATION_ARRAY_NAMES
    },
  )
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
  parser.add_argument("--posture-map-path", type=Path, default=None)
  parser.add_argument("--station-calibration-path", type=Path, default=None)
  parser.add_argument("--height-command", type=float, default=None)
  parser.add_argument("--pitch-command", type=float, default=None)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--steps", type=int, default=2500)
  parser.add_argument("--warmup-steps", type=int, default=250)
  parser.add_argument("--hold-steps", type=int, default=5)
  parser.add_argument(
    "--balance-amplitude",
    type=float,
    default=0.35,
    help="Normalized balance-head PRBS amplitude in [0, 1].",
  )
  parser.add_argument("--heldout-fraction", type=float, default=0.20)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--progress-interval", type=int, default=250)
  return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
  if args.num_envs <= 0:
    raise ValueError("--num-envs must be positive.")
  if args.steps <= 0:
    raise ValueError("--steps must be positive.")
  if args.warmup_steps < 0:
    raise ValueError("--warmup-steps must be non-negative.")
  if args.hold_steps <= 0:
    raise ValueError("--hold-steps must be positive.")
  if not 0.0 < args.balance_amplitude <= 1.0:
    raise ValueError("--balance-amplitude must be in (0, 1].")
  if not 0.0 < args.heldout_fraction < 1.0:
    raise ValueError("--heldout-fraction must be in (0, 1).")
  if args.progress_interval <= 0:
    raise ValueError("--progress-interval must be positive.")
  posture_values = (args.height_command, args.pitch_command)
  if any(value is not None for value in posture_values):
    if not all(value is not None for value in posture_values):
      raise ValueError("Set both --height-command and --pitch-command.")
    if args.calibration_path is None or args.posture_map_path is None:
      raise ValueError(
        "Scheduled identification requires calibration and posture artifacts."
      )


def _git_sha() -> str:
  completed = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=REPOSITORY_PATH,
    check=True,
    capture_output=True,
    text=True,
  )
  return completed.stdout.strip()


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


def _force_posture_command(env: object, height: float, pitch: float) -> None:
  term = env.command_manager.get_term("posture")
  command = getattr(term, "_command")
  target = getattr(term, "_target", None)
  command[:, 0] = height
  command[:, 1] = pitch
  if target is not None:
    target[:, 0] = height
    target[:, 1] = pitch


def calibrated_velocity_reference(
  commanded_vx: torch.Tensor,
  posture_pitch: torch.Tensor,
  *,
  scale: float,
  bias: float,
  station_breakpoints: tuple[tuple[float, float], ...],
) -> torch.Tensor:
  xp = torch.tensor(
    [point[0] for point in station_breakpoints],
    device=commanded_vx.device,
    dtype=commanded_vx.dtype,
  )
  fp = torch.tensor(
    [point[1] for point in station_breakpoints],
    device=commanded_vx.device,
    dtype=commanded_vx.dtype,
  )
  clamped = torch.clamp(posture_pitch, min=xp[0], max=xp[-1])
  upper = torch.clamp(
    torch.bucketize(clamped, xp, right=True), min=1, max=xp.numel() - 1
  )
  lower = upper - 1
  weight = (clamped - xp[lower]) / (xp[upper] - xp[lower])
  station_drift = fp[lower] + weight * (fp[upper] - fp[lower])
  return float(scale) * commanded_vx + float(bias) - station_drift


def _state_from_env(
  env: object,
  *,
  wheel_ids: torch.Tensor,
  action_cfg: object,
  equilibrium_pitch: float = 0.0,
) -> torch.Tensor:
  robot = env.scene["robot"]
  velocity_command = env.command_manager.get_command("twist")
  posture_command = env.command_manager.get_command("posture")
  calibrated_vx = calibrated_velocity_reference(
    velocity_command[:, 0],
    posture_command[:, 1],
    scale=action_cfg.velocity_command_scale,
    bias=action_cfg.velocity_command_bias,
    station_breakpoints=action_cfg.station_drift_breakpoints,
  )
  state = build_controller_state(
    projected_gravity=robot.data.projected_gravity_b,
    pitch_rate=robot.data.root_link_ang_vel_b[:, 1],
    root_vx=robot.data.root_link_lin_vel_b[:, 0],
    commanded_vx=calibrated_vx,
    wheel_velocity=robot.data.joint_vel[:, wheel_ids],
    wheel_radius=action_cfg.wheel_radius,
  )
  state[:, 0] -= equilibrium_pitch
  return state


def collect(args: argparse.Namespace) -> tuple[dict[str, NDArray[np.float64]], dict[str, object]]:
  """Run the deterministic excitation rollout and return split transitions."""

  _validate_args(args)
  from mjlab.envs import ManagerBasedRlEnv
  from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
    make_hoppertrex_hybrid_env_cfg,
  )

  torch.manual_seed(args.seed)
  rng = np.random.default_rng(args.seed)
  scheduled = args.height_command is not None
  cfg = make_hoppertrex_hybrid_env_cfg(
    stage=3 if scheduled else 0,
    play=True,
    controller_path=args.controller_path,
    calibration_path=args.calibration_path,
    posture_map_path=args.posture_map_path,
    station_calibration_path=args.station_calibration_path,
  )
  cfg.scene.num_envs = args.num_envs
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.num_envs = args.num_envs
  cfg.episode_length_s = 1.0e9
  action_cfg = cfg.actions["hybrid_wheel_leg"]
  action_cfg.action_mask = (True, False, False, False, False, False)

  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    env.reset()
    _force_zero_velocity_command(env)
    if scheduled:
      _force_posture_command(env, args.height_command, args.pitch_command)
    action_term = env.action_manager.get_term("hybrid_wheel_leg")
    robot = env.scene["robot"]
    wheel_ids, wheel_names = robot.find_joints(
      action_cfg.wheel_joint_names,
      preserve_order=True,
    )
    if tuple(wheel_names) != action_cfg.wheel_joint_names:
      raise ValueError(
        f"Expected wheel joints {action_cfg.wheel_joint_names}, got {wheel_names}."
      )
    wheel_id_tensor = torch.tensor(
      wheel_ids,
      device=env.device,
      dtype=torch.long,
    )
    zero_action = torch.zeros(
      (args.num_envs, action_cfg.action_dim),
      device=env.device,
    )
    for _ in range(args.warmup_steps):
      _force_zero_velocity_command(env)
      if scheduled:
        _force_posture_command(env, args.height_command, args.pitch_command)
      env.step(zero_action)

    projected = robot.data.projected_gravity_b
    equilibrium_pitch = float(
      torch.atan2(
        projected[:, 0], torch.clamp(-projected[:, 2], min=1.0e-6)
      ).mean().item()
    )

    state_rows: list[NDArray[np.float64]] = []
    input_rows: list[NDArray[np.float64]] = []
    next_state_rows: list[NDArray[np.float64]] = []
    terminated_rows: list[NDArray[np.bool_]] = []
    truncated_rows: list[NDArray[np.bool_]] = []
    action = zero_action.clone()
    for step in range(args.steps):
      if step % args.hold_steps == 0:
        signs = rng.choice((-1.0, 1.0), size=args.num_envs)
        action.zero_()
        action[:, 0] = torch.as_tensor(
          signs * args.balance_amplitude,
          device=env.device,
          dtype=action.dtype,
        )
      _force_zero_velocity_command(env)
      if scheduled:
        _force_posture_command(env, args.height_command, args.pitch_command)
      states = _state_from_env(
        env,
        wheel_ids=wheel_id_tensor,
        action_cfg=action_cfg,
        equilibrium_pitch=equilibrium_pitch,
      )
      _, _, terminated, truncated, _ = env.step(action)
      inputs = signed_balance_input(action_term.wheel_targets)
      _force_zero_velocity_command(env)
      if scheduled:
        _force_posture_command(env, args.height_command, args.pitch_command)
      next_states = _state_from_env(
        env,
        wheel_ids=wheel_id_tensor,
        action_cfg=action_cfg,
        equilibrium_pitch=equilibrium_pitch,
      )
      state_rows.append(states.detach().cpu().numpy())
      input_rows.append(inputs.detach().cpu().numpy())
      next_state_rows.append(next_states.detach().cpu().numpy())
      terminated_rows.append(terminated.detach().cpu().numpy())
      truncated_rows.append(truncated.detach().cpu().numpy())
      if (step + 1) % args.progress_interval == 0:
        print(f"[collect] {step + 1}/{args.steps} steps")

    valid = filter_valid_transitions(
      np.concatenate(state_rows),
      np.concatenate(input_rows),
      np.concatenate(next_state_rows),
      terminated=np.concatenate(terminated_rows),
      truncated=np.concatenate(truncated_rows),
    )
    arrays = deterministic_transition_split(
      *valid,
      heldout_fraction=args.heldout_fraction,
      seed=args.seed,
    )
    metadata: dict[str, object] = {
      "schema_version": 1,
      "git_sha": _git_sha(),
      "seed": args.seed,
      "device": str(args.device),
      "num_envs": args.num_envs,
      "steps": args.steps,
      "warmup_steps": args.warmup_steps,
      "hold_steps": args.hold_steps,
      "balance_amplitude": args.balance_amplitude,
      "heldout_fraction": args.heldout_fraction,
      "height_command": args.height_command,
      "pitch_command": args.pitch_command,
      "equilibrium_pitch": equilibrium_pitch,
      "state_names": list(CONTROLLER_STATE_NAMES),
      "state_definition_version": (
        SCHEDULE_STATE_DEFINITION if scheduled else STATE_DEFINITION_VERSION
      ),
      "wheel_radius": float(action_cfg.wheel_radius),
      "input_name": "actual_signed_balance_wheel_velocity_target",
      "valid_sample_count": int(valid[0].shape[0]),
      "discarded_sample_count": int(args.steps * args.num_envs - valid[0].shape[0]),
      "controller": {
        "type": action_cfg.controller_type,
        "qualified": action_cfg.controller_qualified,
        "source": action_cfg.controller_source,
        "gain_hash": action_cfg.controller_gain_hash,
      },
      "calibration_hash": action_cfg.calibration_hash,
      "posture_artifact_hash": action_cfg.posture_artifact_hash,
      "station_calibration_hash": action_cfg.station_calibration_hash,
    }
    return arrays, metadata
  finally:
    env.close()


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  arrays, metadata = collect(args)
  metadata_path = write_identification_dataset(args.output, arrays, metadata)
  print(f"Wrote identification data: {args.output.resolve()}")
  print(f"Wrote identification metadata: {metadata_path}")


if __name__ == "__main__":
  main()
