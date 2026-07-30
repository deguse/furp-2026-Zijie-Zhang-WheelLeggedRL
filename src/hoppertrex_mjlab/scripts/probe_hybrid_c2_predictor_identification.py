"""Collect and fit the preregistered C2-j1 flat-only innovation predictor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

PROJECT_PATH = Path(__file__).resolve().parents[1]
SRC_PATH = Path(__file__).resolve().parents[2]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from hoppertrex_mjlab.hybrid.classical_stack import (  # noqa: E402
  ClassicalCommands,
  ClassicalSensors,
  ClassicalStackConfig,
  classical_step,
  reset_state,
)
from hoppertrex_mjlab.hybrid.controller_schedule import canonical_hash  # noqa: E402
from hoppertrex_mjlab.hybrid.innovation_detector import (  # noqa: E402
  GRID_SHA256,
  OFFICIAL_IDENTIFICATION_PROTOCOL,
  PREDICTOR_ARTIFACT_TYPE,
  PREDICTOR_STATE_NAMES,
  REGISTERED_HEIGHT_NODES,
  REGISTERED_PITCH_NODES,
  fit_predictor_node,
  parse_innovation_predictor,
  velocity_prbs,
)
from hoppertrex_mjlab.scripts import probe_hybrid_stair_height as stair  # noqa: E402
from hoppertrex_mjlab.tasks.hoppertrex_balance_task import (  # noqa: E402
  NON_WHEEL_GROUND_SENSOR_NAME,
  non_wheel_ground_contact,
)
from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (  # noqa: E402
  make_hoppertrex_hybrid_env_cfg,
)

PROBE_NAME = "hybrid_c2_predictor_identification_v1"
SCHEDULE_HASH = "8fe8548bca85978c164bbd7de39d2d6463cdfd8d7ab91796cf57696b0f64e203"
IDENTIFICATION_CONTROLLER_GAIN_HASH = (
  "8fee25a0339dd1e99127cbed912941dc3ad8ef2030ce49a0d310d1563cb87d98"
)
VELOCITY_CALIBRATION_HASH = (
  "f62648b57bd17a3503bcbdbf58f349f91fcd8de8ef0cf04551c200401233ed01"
)
POSTURE_ARTIFACT_HASH = (
  "3b96fd3dae66ad781b5b875c74184db101c42da02c53dfcc40a5137a6b5de11a"
)
STATION_CALIBRATION_HASH = (
  "c00e859b3093b4812d54799253accdaeb99171a2cf4028b08bc39e68eaaa7d8a"
)
OFFICIAL_ENVS = 32
OFFICIAL_WARMUP = 250
OFFICIAL_STEPS = 2500
FIT_ENVS = 24
CONTROL_DT_S = 0.02
PORTABLE_EQUIVALENCE_ATOL = 2.0e-5
POSTURE_CAPTURE_ATOL = 1.0e-7


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--controller-path", type=Path, required=True)
  parser.add_argument("--calibration-path", type=Path, required=True)
  parser.add_argument("--posture-map-path", type=Path, required=True)
  parser.add_argument("--station-calibration-path", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--smoke", action="store_true")
  args = parser.parse_args(argv)
  if not args.smoke and args.device != "cuda:0":
    parser.error("The official C2-j1 protocol is pinned to --device cuda:0.")
  return args


def protocol(smoke: bool, device: str) -> dict[str, Any]:
  result = {
    "probe": PROBE_NAME,
    "seed": 1,
    "device": device,
    "height_nodes": list(REGISTERED_HEIGHT_NODES),
    "pitch_nodes": list(REGISTERED_PITCH_NODES),
    "grid_sha256": GRID_SHA256,
    "num_envs": 4 if smoke else OFFICIAL_ENVS,
    "fit_envs": [0, 1, 2] if smoke else list(range(FIT_ENVS)),
    "heldout_envs": [3] if smoke else list(range(FIT_ENVS, OFFICIAL_ENVS)),
    "warmup_steps": 2 if smoke else OFFICIAL_WARMUP,
    "collection_steps": 8 if smoke else OFFICIAL_STEPS,
    "control_dt_s": CONTROL_DT_S,
    "prbs": {
      "bit_generator": "numpy.PCG64",
      "seed_formula": "1+1000*node_index+env_index",
      "draw_count": 550,
      "dtype": "uint8",
      "levels_vx_mps": [0.0, 0.10],
      "hold_ticks": 5,
      "collection_stream_ticks": [250, 2749],
    },
    "residual_action": [0.0] * 6,
    "yaw_command": 0.0,
    "evidence_eligible": False,
    "detector_fit_eligible": False,
    "promotion_eligible": False,
    "training_eligible": False,
  }
  if not smoke and device == "cuda:0" and result != OFFICIAL_IDENTIFICATION_PROTOCOL:
    raise RuntimeError("C2-j1 producer protocol drifted from the runtime parser.")
  return result


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _force_commands(
  env: Any, vx: np.ndarray, *, height: float, pitch: float
) -> None:
  twist = env.command_manager.get_term("twist")
  values = torch.as_tensor(vx, device=env.device, dtype=torch.float)
  for attribute in ("vel_command_b", "vel_command_w"):
    command = getattr(twist, attribute)
    command[:, :] = 0.0
    command[:, 0] = values
  for attribute in (
    "is_standing_env", "is_heading_env", "is_world_env", "is_forward_env"
  ):
    value = getattr(twist, attribute, None)
    if value is not None:
      value[:] = False
  posture = env.command_manager.get_term("posture")
  posture._command[:, 0] = height
  posture._command[:, 1] = pitch
  posture._target[:, 0] = height
  posture._target[:, 1] = pitch


def _sensor_state(robot: Any, wheel_ids: torch.Tensor) -> torch.Tensor:
  wheel = robot.data.joint_vel[:, wheel_ids]
  signed = 0.5 * (wheel[:, 1] - wheel[:, 0])
  return torch.stack((robot.data.root_link_ang_vel_b[:, 1], signed), dim=1)


def validate_shaped_posture(
  posture: np.ndarray, *, height: float, pitch: float
) -> None:
  expected = np.asarray([height, pitch], dtype=np.float64)
  if (
    posture.ndim != 2
    or posture.shape[1] != 2
    or not np.all(np.isfinite(posture))
    or not np.allclose(posture, expected, rtol=0.0, atol=POSTURE_CAPTURE_ATOL)
  ):
    raise RuntimeError("C2-j1 shaped posture drifted from its registered node.")


def _portable_config(action_term: Any, robot: Any) -> ClassicalStackConfig:
  limits = robot.data.soft_joint_pos_limits[0, action_term._leg_ids]
  return ClassicalStackConfig(
    controller_gain=tuple(float(value) for value in action_term.cfg.controller_gain),
    velocity_command_scale=float(action_term.cfg.velocity_command_scale),
    velocity_command_bias=float(action_term.cfg.velocity_command_bias),
    yaw_feedforward_breakpoints=action_term.cfg.yaw_feedforward_breakpoints,
    station_drift_breakpoints=action_term.cfg.station_drift_breakpoints,
    posture_coefficients=action_term.cfg.posture_coefficients,
    action_mask=action_term.cfg.action_mask,
    action_scales=action_term.cfg.action_scales,
    leg_position_lower=tuple(float(value) for value in limits[:, 0].tolist()),
    leg_position_upper=tuple(float(value) for value in limits[:, 1].tolist()),
    wheel_radius=float(action_term.cfg.wheel_radius),
    wheel_velocity_limit=float(action_term.cfg.wheel_velocity_limit),
    wheel_slew_limit=float(action_term.cfg.wheel_slew_limit),
    controller_schedule=action_term.cfg.controller_schedule,
  )


def _portable_targets(
  config: ClassicalStackConfig,
  states: list[Any],
  robot: Any,
  wheel_ids: torch.Tensor,
  vx: np.ndarray,
  height: float,
  pitch: float,
) -> tuple[np.ndarray, list[Any]]:
  gravity = robot.data.projected_gravity_b.detach().cpu().numpy()
  pitch_rate = robot.data.root_link_ang_vel_b[:, 1].detach().cpu().numpy()
  root_vx = robot.data.root_link_lin_vel_b[:, 0].detach().cpu().numpy()
  wheel = robot.data.joint_vel[:, wheel_ids].detach().cpu().numpy()
  targets = np.empty((len(states), 2), dtype=np.float64)
  next_states = []
  for env_index, state in enumerate(states):
    pitch_sensor = float(np.arctan2(gravity[env_index, 0], max(-gravity[env_index, 2], 1e-6)))
    output, _legs, next_state = classical_step(
      config,
      state,
      ClassicalSensors(
        pitch=pitch_sensor,
        pitch_rate=float(pitch_rate[env_index]),
        vx=float(root_vx[env_index]),
        body_deceleration=0.0,
        wheel_vel_left=float(wheel[env_index, 0]),
        wheel_vel_right=float(wheel[env_index, 1]),
      ),
      ClassicalCommands(vx=float(vx[env_index]), height=height, pitch=pitch),
      np.zeros(6, dtype=np.float32),
    )
    targets[env_index] = output
    next_states.append(next_state)
  return targets, next_states


def _env_major(values: np.ndarray, env_ids: list[int]) -> np.ndarray:
  return values[:, env_ids, ...].transpose(1, 0, *range(2, values.ndim)).reshape(
    -1, *values.shape[2:]
  )


def collect_node(
  *,
  node_index: int,
  height: float,
  pitch: float,
  args: argparse.Namespace,
  node_path: Path,
) -> dict[str, Any]:
  run_protocol = protocol(args.smoke, args.device)
  num_envs = int(run_protocol["num_envs"])
  warmup = int(run_protocol["warmup_steps"])
  steps = int(run_protocol["collection_steps"])
  cfg = make_hoppertrex_hybrid_env_cfg(
    stage=3,
    play=True,
    controller_path=args.controller_path,
    calibration_path=args.calibration_path,
    posture_map_path=args.posture_map_path,
    station_calibration_path=args.station_calibration_path,
  )
  cfg.seed = 1
  cfg.scene.num_envs = num_envs
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.num_envs = num_envs
  cfg.episode_length_s = 1.0e9

  from mjlab.envs import ManagerBasedRlEnv

  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    env.reset()
    action_term = env.action_manager.get_term("hybrid_wheel_leg")
    robot = env.scene["robot"]
    wheel_ids = action_term._wheel_ids
    if action_term.cfg.yaw_calibration_hash is not None:
      raise RuntimeError("C2-j1 must run with yaw calibration unset.")
    if action_term.cfg.controller_gain_hash != SCHEDULE_HASH:
      raise RuntimeError("C2-j1 did not load the frozen C1 schedule.")
    if (
      action_term.cfg.calibration_hash != VELOCITY_CALIBRATION_HASH
      or action_term.cfg.posture_artifact_hash != POSTURE_ARTIFACT_HASH
      or action_term.cfg.station_calibration_hash != STATION_CALIBRATION_HASH
    ):
      raise RuntimeError("C2-j1 runtime artifact bindings drifted.")
    schedule = action_term.cfg.controller_schedule
    if schedule is None or schedule.bindings != {
      "identification_controller_gain_hash": IDENTIFICATION_CONTROLLER_GAIN_HASH,
      "identification_calibration_hash": VELOCITY_CALIBRATION_HASH,
      "posture_artifact_hash": POSTURE_ARTIFACT_HASH,
    }:
      raise RuntimeError("C2-j1 schedule identification bindings drifted.")
    portable_config = _portable_config(action_term, robot)
    portable_states = [reset_state(height, pitch) for _ in range(num_envs)]
    streams = np.stack([velocity_prbs(node_index, env) for env in range(num_envs)], axis=1)
    zero_action = torch.zeros((num_envs, 6), device=env.device)
    z_rows: list[np.ndarray] = []
    u_rows: list[np.ndarray] = []
    next_z_rows: list[np.ndarray] = []
    posture_rows: list[np.ndarray] = []
    portable_max_error = 0.0
    terminated_count = 0
    timeout_count = 0
    contact_count = 0
    for stream_tick in range(warmup + steps):
      vx = streams[stream_tick]
      _force_commands(env, vx, height=height, pitch=pitch)
      z = _sensor_state(robot, wheel_ids).detach().cpu().numpy().astype(np.float64)
      shaped_posture = (
        env.command_manager.get_command("posture")
        .detach().cpu().numpy().astype(np.float64)
      )
      validate_shaped_posture(shaped_posture, height=height, pitch=pitch)
      portable, portable_states = _portable_targets(
        portable_config, portable_states, robot, wheel_ids, vx, height, pitch
      )
      _obs, _reward, terminated, timeout, _extras = env.step(zero_action)
      actual_targets = action_term.wheel_targets.detach().cpu().numpy().astype(np.float64)
      error = float(np.max(np.abs(actual_targets - portable)))
      portable_max_error = max(portable_max_error, error)
      if error > PORTABLE_EQUIVALENCE_ATOL:
        raise RuntimeError("MjLab wheel targets diverged from portable classical_step.")
      _force_commands(env, vx, height=height, pitch=pitch)
      next_z = _sensor_state(robot, wheel_ids).detach().cpu().numpy().astype(np.float64)
      terminated_count += int(terminated.sum().item())
      timeout_count += int(timeout.sum().item())
      contact_count += int(non_wheel_ground_contact(
        env, NON_WHEEL_GROUND_SENSOR_NAME
      ).sum().item())
      if stream_tick >= warmup:
        z_rows.append(z)
        u_rows.append(
          (0.5 * (actual_targets[:, 1] - actual_targets[:, 0]))[:, None]
        )
        next_z_rows.append(next_z)
        posture_rows.append(shaped_posture)
    z_all = np.stack(z_rows)
    u_all = np.stack(u_rows)
    next_z_all = np.stack(next_z_rows)
    posture_all = np.stack(posture_rows)
    np.savez(
      node_path,
      z=z_all,
      u=u_all,
      next_z=next_z_all,
      shaped_posture=posture_all,
    )
    fit_envs = list(run_protocol["fit_envs"])
    heldout_envs = list(run_protocol["heldout_envs"])
    fitted = fit_predictor_node(
      _env_major(z_all, fit_envs), _env_major(u_all, fit_envs),
      _env_major(next_z_all, fit_envs), _env_major(z_all, heldout_envs),
      _env_major(u_all, heldout_envs), _env_major(next_z_all, heldout_envs),
    )
    return fitted | {
      "node_index": node_index,
      "height_m": height,
      "pitch_rad": pitch,
      "raw_file": node_path.name,
      "raw_sha256": _sha256(node_path),
      "raw_shape": [steps, num_envs],
      "termination_count": terminated_count,
      "timeout_count": timeout_count,
      "non_wheel_contact_count": contact_count,
      "portable_max_abs_target_error_radps": portable_max_error,
    }
  finally:
    env.close()


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  if args.output_dir.exists() and any(args.output_dir.iterdir()):
    raise FileExistsError("C2-j1 output directory must be absent or empty.")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  nodes = []
  selected_nodes = [(1, 1)] if args.smoke else [
    (h, p) for h in range(3) for p in range(3)
  ]
  for height_index, pitch_index in selected_nodes:
    node_index = 3 * height_index + pitch_index
    print(f"[C2-j1] node {node_index}/8")
    nodes.append(collect_node(
      node_index=node_index,
      height=REGISTERED_HEIGHT_NODES[height_index],
      pitch=REGISTERED_PITCH_NODES[pitch_index],
      args=args,
      node_path=args.output_dir / f"node_h{height_index}_p{pitch_index}.npz",
    ))
  healthy = all(
    node["regression_rank"] == 4
    and max(node["heldout_nrmse"]) <= 0.15
    and node["termination_count"] == 0
    and node["timeout_count"] == 0
    and node["non_wheel_contact_count"] == 0
    for node in nodes
  )
  payload: dict[str, Any] = {
    "schema_version": 1,
    "artifact_type": PREDICTOR_ARTIFACT_TYPE,
    "probe": PROBE_NAME,
    "classification": (
      "SMOKE_COMPLETE" if args.smoke else
      ("PREDICTOR_IDENTIFICATION_QUALIFIED" if healthy else "PREDICTOR_IDENTIFICATION_INVALID_STOP")
    ),
    "git_sha": stair._git_sha(stair.REPOSITORY_PATH),
    "mjlab_git_sha": stair._git_sha(Path(stair.mjlab.__file__).resolve().parents[2]),
    "state_names": list(PREDICTOR_STATE_NAMES),
    "height_nodes": list(REGISTERED_HEIGHT_NODES),
    "pitch_nodes": list(REGISTERED_PITCH_NODES),
    "grid_sha256": GRID_SHA256,
    "protocol": protocol(args.smoke, args.device),
    "bindings": {
      "controller_schedule_hash": SCHEDULE_HASH,
      "identification_controller_gain_hash": IDENTIFICATION_CONTROLLER_GAIN_HASH,
      "velocity_calibration_hash": VELOCITY_CALIBRATION_HASH,
      "posture_artifact_hash": POSTURE_ARTIFACT_HASH,
      "station_calibration_hash": STATION_CALIBRATION_HASH,
      "yaw_calibration_hash": None,
    },
    "nodes": nodes,
    "evidence_eligible": False,
    "detector_fit_eligible": False,
    "promotion_eligible": False,
    "training_eligible": False,
    "checkpoint": None,
  }
  if not args.smoke and healthy and len(nodes) == 9:
    payload["predictor_hash"] = canonical_hash(payload, hash_field="predictor_hash")
    parse_innovation_predictor(payload)
  output = args.output_dir / "c2_innovation_predictor.json"
  output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(f"CLASSIFICATION={payload['classification']}")
  print(f"RESULT={output.resolve()}")


if __name__ == "__main__":
  main()
