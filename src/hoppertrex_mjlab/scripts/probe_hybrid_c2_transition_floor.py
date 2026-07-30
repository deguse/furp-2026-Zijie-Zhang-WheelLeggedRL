"""Run the preregistered seed-2 flat transition innovation floor."""

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

from hoppertrex_mjlab.hybrid.controller_schedule import canonical_hash  # noqa: E402
from hoppertrex_mjlab.hybrid.innovation_detector import (  # noqa: E402
  DOMAIN_TOLERANCE_RADPS,
  FEATURE_NAMES,
  FLOOR_ARTIFACT_TYPE,
  OFFICIAL_TRANSITION_FLOOR_PROTOCOL,
  TRANSITION_CENTER,
  InnovationPredictor,
  parse_innovation_predictor,
  parse_transition_floor,
  threshold_table,
  threshold_table_hash,
  transition_floor_cells,
)
from hoppertrex_mjlab.scripts import probe_hybrid_c2_paired_capture_v1 as capture  # noqa: E402
from hoppertrex_mjlab.scripts import probe_hybrid_stair_height as stair  # noqa: E402
from hoppertrex_mjlab.tasks.hoppertrex_balance_task import (  # noqa: E402
  NON_WHEEL_GROUND_SENSOR_NAME,
  non_wheel_ground_contact,
)

PROBE_NAME = "hybrid_c2_transition_floor_v1"
PREDICTOR_HASH = "d1374e4c0c071777bdb3e964e644cad3ba854df4f9976dab016bf9a8d861232d"
SCHEDULE_HASH = "8fe8548bca85978c164bbd7de39d2d6463cdfd8d7ab91796cf57696b0f64e203"
IDENTIFICATION_GAIN_HASH = "8fee25a0339dd1e99127cbed912941dc3ad8ef2030ce49a0d310d1563cb87d98"
CALIBRATION_HASH = "f62648b57bd17a3503bcbdbf58f349f91fcd8de8ef0cf04551c200401233ed01"
POSTURE_HASH = "3b96fd3dae66ad781b5b875c74184db101c42da02c53dfcc40a5137a6b5de11a"
STATION_HASH = "c00e859b3093b4812d54799253accdaeb99171a2cf4028b08bc39e68eaaa7d8a"
CENTER = TRANSITION_CENTER
OFFICIAL_ENVS = 16
OFFICIAL_SETTLE = 200
OFFICIAL_DRIVE = 500
POSTURE_HEIGHT_SLEW_RATE = 0.01215
POSTURE_PITCH_SLEW_RATE = 0.07755
WHEEL_SLEW_RADPS_PER_TICK = 6.0
CONTROL_DT_S = 0.02


def transition_cells() -> list[dict[str, Any]]:
  return transition_floor_cells()


def raw_command(cell: dict[str, Any], tick: int) -> tuple[float, float, float]:
  if not 0 <= tick < OFFICIAL_DRIVE:
    raise ValueError("Transition-floor tick is outside [0, 499].")
  target = np.asarray(cell["target"], dtype=np.float64)
  center = np.asarray(CENTER, dtype=np.float64)
  if cell["kind"] == "constant":
    return tuple(float(value) for value in target)
  if tick < 80:
    value = center + ((tick + 1) / 80.0) * (target - center)
  elif tick < 420:
    value = target
  else:
    value = target + ((tick - 419) / 80.0) * (center - target)
  return tuple(float(item) for item in value)


def protocol(smoke: bool, device: str) -> dict[str, Any]:
  result = {
    "probe": PROBE_NAME,
    "seed": 2,
    "device": device,
    "cells": transition_cells()[:1] if smoke else transition_cells(),
    "envs_per_cell": 2 if smoke else OFFICIAL_ENVS,
    "settle_steps": 2 if smoke else OFFICIAL_SETTLE,
    "drive_steps": 8 if smoke else OFFICIAL_DRIVE,
    "settle_raw_command": list(CENTER),
    "transition_schedule": {
      "outward_ticks": [0, 79],
      "hold_ticks": [80, 419],
      "return_ticks": [420, 499],
    },
    "height_slew_rate_mps": POSTURE_HEIGHT_SLEW_RATE,
    "pitch_slew_rate_radps": POSTURE_PITCH_SLEW_RATE,
    "wheel_slew_radps_per_tick": WHEEL_SLEW_RADPS_PER_TICK,
    "activation": "integrated_signed_wheel_odometry_lt_0p35m",
    "first_tick_no_vote": True,
    "threshold_factors": [1.05, 1.25, 1.5, 2.0, 3.0],
    "evidence_eligible": False,
    "detector_fit_eligible": False,
    "promotion_eligible": False,
    "training_eligible": False,
  }
  if not smoke and device == "cuda:0" and result != OFFICIAL_TRANSITION_FLOOR_PROTOCOL:
    raise RuntimeError("Transition-floor producer protocol drifted from parser.")
  return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--predictor", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--smoke", action="store_true")
  args = parser.parse_args(argv)
  if not args.smoke and args.device != "cuda:0":
    parser.error("The official transition floor is pinned to --device cuda:0.")
  return args


def _sha256(path: Path) -> str:
  digest = hashlib.sha256(path.read_bytes()).hexdigest()
  return digest


def _is_finite_domain_violation(
  predictor: InnovationPredictor,
  state: np.ndarray,
  u: float,
  height: float,
  pitch: float,
) -> bool:
  if not (
    np.all(np.isfinite(state))
    and np.isfinite(u)
    and np.isfinite(height)
    and np.isfinite(pitch)
  ):
    return False
  minimum, maximum = predictor.input_domain(height, pitch)
  return not (
    minimum - DOMAIN_TOLERANCE_RADPS
    <= u
    <= maximum + DOMAIN_TOLERANCE_RADPS
  )


def _initialize_commands(env: Any) -> None:
  stair._force_commands(env, vx=CENTER[2], height=CENTER[0], pitch=CENTER[1])


def _assert_runtime_stack(env: Any) -> None:
  action = env.action_manager.get_term("hybrid_wheel_leg")
  posture = env.command_manager.get_term("posture")
  if (
    action.cfg.controller_gain_hash != SCHEDULE_HASH
    or action.cfg.calibration_hash != CALIBRATION_HASH
    or action.cfg.posture_artifact_hash != POSTURE_HASH
    or action.cfg.station_calibration_hash != STATION_HASH
    or action.cfg.yaw_calibration_hash is not None
  ):
    raise RuntimeError("C2-j2 runtime artifact bindings drifted.")
  schedule = action.cfg.controller_schedule
  if schedule is None or schedule.bindings != {
    "identification_controller_gain_hash": IDENTIFICATION_GAIN_HASH,
    "identification_calibration_hash": CALIBRATION_HASH,
    "posture_artifact_hash": POSTURE_HASH,
  }:
    raise RuntimeError("C2-j2 schedule identification bindings drifted.")
  if (
    float(action.cfg.wheel_slew_limit) != WHEEL_SLEW_RADPS_PER_TICK
    or float(posture.cfg.height_slew_rate) != POSTURE_HEIGHT_SLEW_RATE
    or float(posture.cfg.pitch_slew_rate) != POSTURE_PITCH_SLEW_RATE
  ):
    raise RuntimeError("C2-j2 deployed slew limits drifted.")


def _set_deployed_commands(
  env: Any, *, vx: float, height: float, pitch: float
) -> np.ndarray:
  twist = env.command_manager.get_term("twist")
  for attribute in ("vel_command_b", "vel_command_w"):
    command = getattr(twist, attribute)
    command[:, :] = 0.0
    command[:, 0] = vx
  for attribute in (
    "is_standing_env", "is_heading_env", "is_world_env", "is_forward_env"
  ):
    value = getattr(twist, attribute, None)
    if value is not None:
      value[:] = False
  posture = env.command_manager.get_term("posture")
  current = posture._command
  raw = torch.tensor([height, pitch], device=env.device, dtype=current.dtype)
  limits = torch.tensor(
    [
      POSTURE_HEIGHT_SLEW_RATE * CONTROL_DT_S,
      POSTURE_PITCH_SLEW_RATE * CONTROL_DT_S,
    ],
    device=env.device,
    dtype=current.dtype,
  )
  shaped = current + torch.clamp(raw - current, min=-limits, max=limits)
  # Freeze the already-shaped value so CommandManager.compute cannot apply a
  # second slew after physics; the next control tick performs the next step.
  posture._command[:] = shaped
  posture._target[:] = shaped
  return shaped.detach().cpu().numpy().astype(np.float64)


def _restore_deployed_commands(
  env: Any, *, vx: float, shaped_posture: np.ndarray
) -> None:
  """Undo any post-step command resample without applying another slew step."""

  twist = env.command_manager.get_term("twist")
  for attribute in ("vel_command_b", "vel_command_w"):
    command = getattr(twist, attribute)
    command[:, :] = 0.0
    command[:, 0] = vx
  posture = env.command_manager.get_term("posture")
  shaped = torch.as_tensor(
    shaped_posture, device=env.device, dtype=posture._command.dtype
  )
  posture._command[:] = shaped
  posture._target[:] = shaped


def _z(robot: Any, wheel_ids: torch.Tensor) -> np.ndarray:
  wheels = robot.data.joint_vel[:, wheel_ids]
  signed = 0.5 * (wheels[:, 1] - wheels[:, 0])
  return torch.stack((robot.data.root_link_ang_vel_b[:, 1], signed), dim=1).detach().cpu().numpy().astype(np.float64)


def run_cell(
  env: Any,
  predictor: InnovationPredictor,
  cell: dict[str, Any],
  run_protocol: dict[str, Any],
  raw_path: Path,
) -> dict[str, Any]:
  env.reset()
  _initialize_commands(env)
  action_term = env.action_manager.get_term("hybrid_wheel_leg")
  robot = env.scene["robot"]
  wheel_ids = action_term._wheel_ids
  accelerometer = env.scene.sensors[capture.IMU_ACCELEROMETER_SCENE_NAME]
  actions = torch.zeros((env.num_envs, 6), device=env.device)
  terminated_count = 0
  timeout_count = 0
  contact_count = 0
  for _ in range(int(run_protocol["settle_steps"])):
    shaped = _set_deployed_commands(
      env, vx=CENTER[2], height=CENTER[0], pitch=CENTER[1]
    )
    _obs, _reward, terminated, timeout, _extras = env.step(actions)
    _restore_deployed_commands(
      env, vx=CENTER[2], shaped_posture=shaped
    )
    terminated_count += int(terminated.sum().item())
    timeout_count += int(timeout.sum().item())
    contact_count += int(
      non_wheel_ground_contact(env, NON_WHEEL_GROUND_SENSOR_NAME).sum().item()
    )

  progress = np.zeros(env.num_envs, dtype=np.float64)
  active_latched = np.ones(env.num_envs, dtype=bool)
  rows: dict[str, list[np.ndarray]] = {
    key: [] for key in (
      "z", "u", "next_z", "shaped_posture", "innovation",
      "accelerometer_specific_force_x", "projected_gravity_x",
      "forward_deceleration", "active", "raw_command"
    )
  }
  domain_violations = 0
  drive_steps = int(run_protocol["drive_steps"])
  for tick in range(drive_steps):
    height, pitch, vx = raw_command(cell, tick)
    shaped = _set_deployed_commands(
      env, vx=vx, height=height, pitch=pitch
    )
    state = _z(robot, wheel_ids)
    _obs, _reward, terminated, timeout, _extras = env.step(actions)
    _restore_deployed_commands(env, vx=vx, shaped_posture=shaped)
    targets = action_term.wheel_targets.detach().cpu().numpy().astype(np.float64)
    u = (0.5 * (targets[:, 1] - targets[:, 0]))[:, None]
    next_state = _z(robot, wheel_ids)
    progress += next_state[:, 1] * float(action_term.cfg.wheel_radius) * CONTROL_DT_S
    active_latched &= progress < 0.35
    current_active = active_latched.copy()
    predicted = np.empty_like(next_state)
    for env_index in range(env.num_envs):
      try:
        predicted[env_index] = predictor.predict(
          state[env_index], float(u[env_index, 0]),
          float(shaped[env_index, 0]), float(shaped[env_index, 1]),
        )
      except ValueError as error:
        if _is_finite_domain_violation(
          predictor, state[env_index], float(u[env_index, 0]),
          float(shaped[env_index, 0]), float(shaped[env_index, 1]),
        ):
          domain_violations += int(current_active[env_index])
        elif not (
          "outside the fitted domain" in str(error)
          or "must be finite" in str(error)
        ):
          raise
        predicted[env_index] = np.nan
    innovation = np.abs(next_state - predicted)
    specific_force_x = accelerometer.data[:, 0]
    projected_gravity_x = robot.data.projected_gravity_b[:, 0]
    deceleration = capture.body_forward_deceleration(
      specific_force_x, projected_gravity_x
    ).detach().cpu().numpy().astype(np.float64)
    terminated_count += int(terminated.sum().item())
    timeout_count += int(timeout.sum().item())
    contact_count += int(non_wheel_ground_contact(env, NON_WHEEL_GROUND_SENSOR_NAME).sum().item())
    rows["z"].append(state)
    rows["u"].append(u)
    rows["next_z"].append(next_state)
    rows["shaped_posture"].append(shaped)
    rows["innovation"].append(innovation)
    rows["accelerometer_specific_force_x"].append(
      specific_force_x.detach().cpu().numpy().astype(np.float64)[:, None]
    )
    rows["projected_gravity_x"].append(
      projected_gravity_x.detach().cpu().numpy().astype(np.float64)[:, None]
    )
    rows["forward_deceleration"].append(deceleration[:, None])
    rows["active"].append(current_active[:, None])
    rows["raw_command"].append(np.tile([height, pitch, vx], (env.num_envs, 1)))
  arrays = {key: np.stack(value) for key, value in rows.items()}
  np.savez(raw_path, **arrays)
  voting = arrays["active"][:, :, 0].copy()
  voting[0, :] = False
  voting_per_env = np.count_nonzero(voting, axis=0)
  features = np.concatenate(
    (arrays["innovation"], arrays["forward_deceleration"]), axis=2
  )
  maxima = [
    float(np.max(features[:, :, index][voting])) if np.any(voting) else float("nan")
    for index in range(3)
  ]
  return {
    "cell_index": int(raw_path.stem.split("_")[-1]),
    "name": cell["name"],
    "kind": cell["kind"],
    "target": cell["target"],
    "raw_file": raw_path.name,
    "raw_sha256": _sha256(raw_path),
    "raw_shape": [drive_steps, env.num_envs],
    "active_voting_ticks": int(voting.sum()),
    "active_voting_ticks_per_env": voting_per_env.astype(int).tolist(),
    "feature_maxima": dict(zip(FEATURE_NAMES, maxima, strict=True)),
    "domain_violation_count": domain_violations,
    "termination_count": terminated_count,
    "timeout_count": timeout_count,
    "non_wheel_contact_count": contact_count,
  }


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  if args.output_dir.exists() and any(args.output_dir.iterdir()):
    raise FileExistsError("Transition-floor output directory must be absent or empty.")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  predictor_payload = json.loads(args.predictor.read_text(encoding="utf-8"))
  predictor = parse_innovation_predictor(predictor_payload)
  if predictor.predictor_hash != PREDICTOR_HASH:
    raise ValueError("Transition floor requires the frozen C2-j1 predictor.")
  run_protocol = protocol(args.smoke, args.device)
  cfg = capture.make_causal_env_cfg((0.0,), int(run_protocol["envs_per_cell"]))
  cfg.seed = 2
  from mjlab.envs import ManagerBasedRlEnv

  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    _assert_runtime_stack(env)
    cells = []
    for index, cell in enumerate(run_protocol["cells"]):
      print(f"[C2-j2] cell {index}/{len(run_protocol['cells']) - 1}: {cell['name']}")
      cells.append(run_cell(
        env, predictor, cell, run_protocol,
        args.output_dir / f"cell_{index:02d}.npz",
      ))
  finally:
    env.close()
  invalid = any(
    cell["termination_count"] or cell["timeout_count"]
    or cell["non_wheel_contact_count"]
    or any(value <= 0 for value in cell["active_voting_ticks_per_env"])
    or any(not np.isfinite(value) or value <= 0.0 for value in cell["feature_maxima"].values())
    for cell in cells
  )
  domain_uncovered = any(cell["domain_violation_count"] for cell in cells)
  pooled = np.asarray([
    max(cell["feature_maxima"][name] for cell in cells) for name in FEATURE_NAMES
  ])
  classification = (
    "SMOKE_COMPLETE" if args.smoke else
    ("PREDICTOR_DOMAIN_UNCOVERED_STOP" if domain_uncovered else
     ("INVALID_INNOVATION_FLOOR" if invalid else "INNOVATION_FLOOR_QUALIFIED"))
  )
  pooled_mapping = dict(zip(FEATURE_NAMES, pooled.tolist(), strict=True))
  if classification != "INNOVATION_FLOOR_QUALIFIED":
    for cell in cells:
      cell["feature_maxima"] = {
        name: (float(value) if np.isfinite(value) else None)
        for name, value in cell["feature_maxima"].items()
      }
    pooled_mapping = {
      name: (float(value) if np.isfinite(value) else None)
      for name, value in pooled_mapping.items()
    }
  payload: dict[str, Any] = {
    "schema_version": 1,
    "artifact_type": FLOOR_ARTIFACT_TYPE,
    "probe": PROBE_NAME,
    "classification": classification,
    "git_sha": stair._git_sha(stair.REPOSITORY_PATH),
    "mjlab_git_sha": stair._git_sha(Path(stair.mjlab.__file__).resolve().parents[2]),
    "predictor_hash": predictor.predictor_hash,
    "bindings": predictor.bindings,
    "protocol": run_protocol,
    "cells": cells,
    "pooled_feature_maxima": pooled_mapping,
    "evidence_eligible": False,
    "detector_fit_eligible": False,
    "promotion_eligible": False,
    "training_eligible": False,
    "checkpoint": None,
  }
  if classification == "INNOVATION_FLOOR_QUALIFIED":
    table = threshold_table(pooled)
    payload["threshold_table"] = table
    payload["threshold_table_hash"] = threshold_table_hash(table)
    payload["floor_hash"] = canonical_hash(payload, hash_field="floor_hash")
    parse_transition_floor(payload, predictor_hash=PREDICTOR_HASH)
  output = args.output_dir / "c2_innovation_floor.json"
  output.write_text(
    json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  print(f"CLASSIFICATION={classification}")
  print(f"RESULT={output.resolve()}")


if __name__ == "__main__":
  main()
