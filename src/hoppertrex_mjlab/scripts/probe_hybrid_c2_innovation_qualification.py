"""Run the preregistered one-shot C2-j3 innovation qualification."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_PATH = Path(__file__).resolve().parents[1]
SRC_PATH = Path(__file__).resolve().parents[2]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))


_EARLY_ARTIFACT_FLAGS = {
  "--controller-path": "HOPPERTREX_HYBRID_CONTROLLER_PATH",
  "--calibration-path": "HOPPERTREX_HYBRID_CALIBRATION_PATH",
  "--posture-map-path": "HOPPERTREX_HYBRID_POSTURE_MAP_PATH",
  "--station-calibration-path": "HOPPERTREX_HYBRID_STATION_CALIBRATION_PATH",
}


def _preconfigure_artifact_environment(argv: list[str]) -> None:
  """Set artifact paths before task-config modules read their environment."""

  present = [flag for flag in _EARLY_ARTIFACT_FLAGS if flag in argv]
  if not present:
    return
  if len(present) != len(_EARLY_ARTIFACT_FLAGS):
    raise RuntimeError("C2-j3 direct invocation requires all four artifact paths.")
  for flag, variable in _EARLY_ARTIFACT_FLAGS.items():
    if argv.count(flag) != 1:
      raise RuntimeError(f"C2-j3 direct invocation requires one {flag} value.")
    index = argv.index(flag)
    if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
      raise RuntimeError(f"C2-j3 direct invocation is missing the {flag} value.")
    os.environ[variable] = str(Path(argv[index + 1]).resolve())
  os.environ.pop("HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH", None)


if __name__ == "__main__":
  _preconfigure_artifact_environment(sys.argv[1:])


from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_from_euler_xyz,
)

from hoppertrex_mjlab.hybrid.classical_stack import reset_state
from hoppertrex_mjlab.hybrid.controller_schedule import canonical_hash
from hoppertrex_mjlab.hybrid.innovation_detector import (
  EXPECTED_BINDINGS,
  OFFICIAL_QUALIFICATION_PROTOCOL,
  QUALIFICATION_ARTIFACT_TYPE,
  QUALIFICATION_DRIVE_STEPS,
  QUALIFICATION_GEOMETRY_WRITE_ATOL,
  QUALIFICATION_OUTER_FACE_OFFSET_FROM_TERRAIN_ORIGIN_M,
  QUALIFICATION_PAIRS_PER_CELL,
  QUALIFICATION_PORTABLE_EQUIVALENCE_ATOL,
  QUALIFICATION_POST_IMPACT_STEPS,
  QUALIFICATION_POSTURE_CAPTURE_ATOL,
  QUALIFICATION_POSTURE_HEIGHT_SLEW_RATE_MPS,
  QUALIFICATION_POSTURE_PITCH_SLEW_RATE_RADPS,
  QUALIFICATION_PRE_IMPACT_STEPS,
  QUALIFICATION_RESET_WRITE_ATOL,
  QUALIFICATION_SCHEMA_VERSION,
  QUALIFICATION_WHEEL_SLEW_RADPS_PER_TICK,
  QUALIFICATION_WHEEL_VELOCITY_LIMIT_RADPS,
  RESET_PERTURBATION_BOUNDS,
  InnovationPredictor,
  evaluate_qualification_candidate,
  parse_innovation_detector_qualification,
  parse_innovation_predictor,
  parse_transition_floor,
  qualification_cells,
  qualification_selection,
  select_qualification_candidate,
)
from hoppertrex_mjlab.scripts import (
  probe_hybrid_c2_paired_capture_v1 as capture,
)
from hoppertrex_mjlab.scripts import (
  probe_hybrid_c2_predictor_identification as identification,
)
from hoppertrex_mjlab.scripts import (
  probe_hybrid_c2_transition_floor as transition,
)
from hoppertrex_mjlab.scripts import probe_hybrid_stair_height as stair
from hoppertrex_mjlab.tasks.hoppertrex_balance_task import (
  NON_WHEEL_GROUND_SENSOR_NAME,
  non_wheel_ground_contact,
)

PROBE_NAME = "hybrid_c2_innovation_qualification_v1"
PREDICTOR_HASH = "d1374e4c0c071777bdb3e964e644cad3ba854df4f9976dab016bf9a8d861232d"
FLOOR_HASH = "1692f8e6a3ff9d82b22ee5ac579b48d832a852b8bcfccb88fb02d85b360e4e58"
THRESHOLD_TABLE_HASH = "098888c153e60d5539e98e85c7e523a5a27c0848f6628d191c79f0613d3566fc"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--predictor", type=Path, required=True)
  parser.add_argument("--transition-floor", type=Path, required=True)
  parser.add_argument("--controller-path", type=Path, required=True)
  parser.add_argument("--calibration-path", type=Path, required=True)
  parser.add_argument("--posture-map-path", type=Path, required=True)
  parser.add_argument("--station-calibration-path", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--smoke",
    action="store_true",
    help="Run the full 18-cell structure on CPU; never research evidence.",
  )
  args = parser.parse_args(argv)
  if not args.smoke and args.device != "cuda:0":
    parser.error("The official C2-j3 protocol is pinned to --device cuda:0.")
  return args


def protocol(smoke: bool, device: str) -> dict[str, Any]:
  result = copy.deepcopy(OFFICIAL_QUALIFICATION_PROTOCOL)
  result["device"] = device
  if smoke:
    result["evidence_eligible"] = False
    result["smoke"] = True
  elif result != OFFICIAL_QUALIFICATION_PROTOCOL:
    raise RuntimeError("C2-j3 producer protocol drifted from registration.")
  return result


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def reset_perturbations(cell_index: int, slots: int) -> torch.Tensor:
  if not 0 <= cell_index < 18 or slots != QUALIFICATION_PAIRS_PER_CELL:
    raise ValueError("C2-j3 reset perturbations require one registered 16-slot cell.")
  generator = torch.Generator(device="cpu")
  generator.manual_seed(30_000 + cell_index)
  unit = 2.0 * torch.rand((slots, 4), generator=generator) - 1.0
  bounds = torch.tensor(RESET_PERTURBATION_BOUNDS, dtype=torch.float32)
  return unit * bounds


def _configure_artifact_environment(args: argparse.Namespace) -> None:
  paths = {
    "HOPPERTREX_HYBRID_CONTROLLER_PATH": args.controller_path,
    "HOPPERTREX_HYBRID_CALIBRATION_PATH": args.calibration_path,
    "HOPPERTREX_HYBRID_POSTURE_MAP_PATH": args.posture_map_path,
    "HOPPERTREX_HYBRID_STATION_CALIBRATION_PATH": args.station_calibration_path,
  }
  for name, path in paths.items():
    if not path.is_file():
      raise FileNotFoundError(f"Required C2-j3 artifact is missing: {path}")
    os.environ[name] = str(path.resolve())
  os.environ.pop("HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH", None)


def _paired_reset(
  env: Any,
  *,
  cell: dict[str, Any],
) -> dict[str, Any]:
  env.reset()
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_types is None:
    raise RuntimeError("C2-j3 requires generated flat/stair terrain types.")
  terrain_types = terrain.terrain_types.clone()
  pairs = capture.paired_environment_ids(terrain_types)
  if len(pairs) != QUALIFICATION_PAIRS_PER_CELL:
    raise RuntimeError("C2-j3 reset did not produce 16 flat/stair pairs.")
  origins = env.scene.env_origins
  geometry = stair.approach_geometry(0.0)
  outer_face_x = origins[:, 0] + geometry["outer_face_x"]
  cross_x = origins[:, 0] + geometry["cross_x"]
  perturbations = reset_perturbations(
    int(cell["cell_index"]), QUALIFICATION_PAIRS_PER_CELL
  ).to(env.device)
  slot_ids = torch.empty_like(terrain_types)
  for pair in pairs:
    slot_ids[pair["flat_env_id"]] = pair["slot"]
    slot_ids[pair["stair_env_id"]] = pair["slot"]
  robot = env.scene["robot"]
  root_states = robot.data.default_root_state.clone()
  canonical_by_slot = torch.zeros(
    (QUALIFICATION_PAIRS_PER_CELL, 13),
    device=env.device,
    dtype=root_states.dtype,
  )
  canonical_by_slot[:, 0] = -0.25 + perturbations[:, 0]
  canonical_by_slot[:, 1] = perturbations[:, 1]
  canonical_by_slot[:, 2] = float(cell["height_m"])
  slot_zero = torch.zeros(
    QUALIFICATION_PAIRS_PER_CELL,
    device=env.device,
    dtype=root_states.dtype,
  )
  slot_pitch = torch.full_like(slot_zero, float(cell["pitch_rad"]))
  canonical_by_slot[:, 3:7] = quat_from_euler_xyz(
    slot_zero, slot_pitch, slot_zero
  )
  canonical_by_slot[:, 7] = perturbations[:, 2]
  canonical_by_slot[:, 11] = perturbations[:, 3]
  canonical_by_env = canonical_by_slot[slot_ids]
  root_states[:, 0] = outer_face_x + canonical_by_env[:, 0]
  root_states[:, 1] = origins[:, 1] + canonical_by_env[:, 1]
  root_states[:, 2:13] = canonical_by_env[:, 2:13]
  robot.write_root_state_to_sim(root_states)
  env.sim.forward()
  env.sim.sense()

  roll, recovered_pitch, yaw = euler_xyz_from_quat(root_states[:, 3:7])
  pitch_error = float(
    torch.max(torch.abs(recovered_pitch - float(cell["pitch_rad"]))).item()
  )
  other_velocity = torch.cat((
    root_states[:, 8:11],
    root_states[:, 12:13],
  ), dim=1)
  written_relative_root_states = root_states.clone()
  written_relative_root_states[:, 0] -= outer_face_x
  written_relative_root_states[:, 1] -= origins[:, 1]
  written_reset_error = float(
    torch.max(
      torch.abs(written_relative_root_states - canonical_by_env)
    ).item()
  )

  paired_error = 0.0
  written_paired_error = 0.0
  for pair in pairs:
    flat_id = pair["flat_env_id"]
    stair_id = pair["stair_env_id"]
    paired_error = max(
      paired_error,
      float(
        torch.max(
          torch.abs(canonical_by_env[flat_id] - canonical_by_env[stair_id])
        ).item()
      ),
    )
    written_paired_error = max(
      written_paired_error,
      float(
        torch.max(
          torch.abs(
            written_relative_root_states[flat_id]
            - written_relative_root_states[stair_id]
          )
        ).item()
      ),
    )
  return {
    "terrain_types": terrain_types,
    "pairs": pairs,
    "terrain_origin_x": origins[:, 0],
    "outer_face_x": outer_face_x,
    "cross_x": cross_x,
    "root_states": root_states,
    "relative_root_states": canonical_by_env,
    "written_relative_root_states": written_relative_root_states,
    "perturbations": perturbations,
    "paired_reset_max_abs_error": paired_error,
    "written_reset_max_abs_error": written_reset_error,
    "written_paired_reset_max_abs_error": written_paired_error,
    "root_pitch_max_abs_error_rad": pitch_error,
    "root_roll_yaw_max_abs_rad": float(
      torch.max(torch.abs(torch.stack((roll, yaw), dim=1))).item()
    ),
    "other_root_velocity_max_abs": float(torch.max(torch.abs(other_velocity)).item()),
  }


def _assert_runtime_stack(env: Any) -> None:
  transition._assert_runtime_stack(env)
  action = env.action_manager.get_term("hybrid_wheel_leg")
  posture = env.command_manager.get_term("posture")
  noise_fields = (
    "sensor_noise_pitch_std",
    "sensor_noise_pitch_rate_std",
    "sensor_noise_vx_std",
    "sensor_noise_wheel_vel_std",
  )
  if int(action.cfg.action_delay_steps) != 0:
    raise RuntimeError("C2-j3 requires zero action delay for applied-u alignment.")
  if any(float(getattr(action.cfg, name)) != 0.0 for name in noise_fields):
    raise RuntimeError("C2-j3 requires zero runtime sensor noise.")
  if (
    float(action.cfg.wheel_velocity_limit)
    != QUALIFICATION_WHEEL_VELOCITY_LIMIT_RADPS
    or float(action.cfg.wheel_slew_limit)
    != QUALIFICATION_WHEEL_SLEW_RADPS_PER_TICK
    or float(posture.cfg.height_slew_rate)
    != QUALIFICATION_POSTURE_HEIGHT_SLEW_RATE_MPS
    or float(posture.cfg.pitch_slew_rate)
    != QUALIFICATION_POSTURE_PITCH_SLEW_RATE_RADPS
  ):
    raise RuntimeError("C2-j3 deployed actuator/reference limits drifted.")


def _split(array: np.ndarray, env_ids: list[int]) -> np.ndarray:
  return array[:, env_ids, ...]


_STAIR_CONTACT_RAW_KEYS = {
  "stair_contact_found",
  "stair_contact_force_contact_frame",
  "stair_contact_pos_global",
  "stair_contact_normal_global",
  "stair_outer_face_x",
  "stair_terrain_origin_x",
}


def _stair_contact_raw_arrays(
  sensor_stacked: dict[str, torch.Tensor],
  stair_ids_tensor: torch.Tensor,
  outer_face_x: torch.Tensor,
  terrain_origin_x: torch.Tensor,
) -> dict[str, np.ndarray]:
  """Extract the tick-major raw contacts needed to replay riser truth."""

  if (
    stair_ids_tensor.ndim != 1
    or stair_ids_tensor.dtype != torch.long
    or stair_ids_tensor.numel() != QUALIFICATION_PAIRS_PER_CELL
  ):
    raise ValueError(
      "C2-j3 stair ids must be one int64 index per registered stair trial."
    )
  if (
    bool((stair_ids_tensor < 0).any().item())
    or torch.unique(stair_ids_tensor).numel() != QUALIFICATION_PAIRS_PER_CELL
  ):
    raise ValueError("C2-j3 stair ids must be unique nonnegative indices.")
  maximum_stair_id = int(stair_ids_tensor.max().item())
  source_fields = {
    "found": "stair_contact_found",
    "force": "stair_contact_force_contact_frame",
    "pos": "stair_contact_pos_global",
    "normal": "stair_contact_normal_global",
  }
  selected: dict[str, torch.Tensor] = {}
  for source_name, output_name in source_fields.items():
    if source_name not in sensor_stacked:
      raise ValueError(f"C2-j3 contact history is missing {source_name!r}.")
    value = sensor_stacked[source_name]
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
      raise ValueError(
        f"C2-j3 raw contact field {source_name!r} must be torch.float32."
      )
    if value.device != stair_ids_tensor.device:
      raise ValueError(
        f"C2-j3 raw contact field {source_name!r} and stair ids must share a device."
      )
    if value.ndim < 2 or value.shape[1] <= maximum_stair_id:
      raise ValueError(
        f"C2-j3 raw contact field {source_name!r} does not contain every stair env."
      )
    selected[output_name] = value[:, stair_ids_tensor]

  found = selected["stair_contact_found"]
  expected_prefix = (
    QUALIFICATION_DRIVE_STEPS,
    QUALIFICATION_PAIRS_PER_CELL,
  )
  if found.ndim != 3 or tuple(found.shape[:2]) != expected_prefix:
    raise ValueError(
      "C2-j3 raw found contacts must have shape "
      f"({QUALIFICATION_DRIVE_STEPS}, {QUALIFICATION_PAIRS_PER_CELL}, slots)."
    )
  if found.shape[2] <= 0:
    raise ValueError("C2-j3 raw contact history must contain contact slots.")
  expected_vector_shape = (*found.shape, 3)
  for name in (
    "stair_contact_force_contact_frame",
    "stair_contact_pos_global",
    "stair_contact_normal_global",
  ):
    if tuple(selected[name].shape) != expected_vector_shape:
      raise ValueError(f"{name} must have shape {expected_vector_shape}.")

  for name, value in (
    ("outer_face_x", outer_face_x),
    ("terrain_origin_x", terrain_origin_x),
  ):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
      raise ValueError(f"C2-j3 {name} must be torch.float32.")
    if (
      value.device != stair_ids_tensor.device
      or value.ndim != 1
      or value.shape[0] <= maximum_stair_id
    ):
      raise ValueError(
        f"C2-j3 {name} must contain every stair env on the capture device."
      )
  selected_outer_face_x = outer_face_x[stair_ids_tensor]
  selected_terrain_origin_x = terrain_origin_x[stair_ids_tensor]
  if tuple(selected_outer_face_x.shape) != (QUALIFICATION_PAIRS_PER_CELL,):
    raise ValueError(
      "C2-j3 stair_outer_face_x must have shape "
      f"({QUALIFICATION_PAIRS_PER_CELL},)."
    )
  if tuple(selected_terrain_origin_x.shape) != (QUALIFICATION_PAIRS_PER_CELL,):
    raise ValueError(
      "C2-j3 stair_terrain_origin_x must have shape "
      f"({QUALIFICATION_PAIRS_PER_CELL},)."
    )

  raw = {
    name: value.detach().cpu().numpy().copy()
    for name, value in selected.items()
  }
  raw["stair_outer_face_x"] = (
    selected_outer_face_x.detach().cpu().numpy().copy()
  )
  raw["stair_terrain_origin_x"] = (
    selected_terrain_origin_x.detach().cpu().numpy().copy()
  )
  return raw


def _contact_raw_health(contact_raw: dict[str, np.ndarray]) -> dict[str, int]:
  """Return invalid-capture counts while reserving schema drift for errors."""

  found = contact_raw["stair_contact_found"]
  finite_found = found[np.isfinite(found)]
  if np.any(finite_found < 0.0) or not np.all(
    finite_found == np.floor(finite_found)
  ):
    raise ValueError(
      "stair_contact_found must contain nonnegative integer match counts when finite."
    )
  outer_face_x = contact_raw["stair_outer_face_x"]
  terrain_origin_x = contact_raw["stair_terrain_origin_x"]
  finite_geometry = np.isfinite(outer_face_x) & np.isfinite(terrain_origin_x)
  expected_outer_face_x = (
    terrain_origin_x + QUALIFICATION_OUTER_FACE_OFFSET_FROM_TERRAIN_ORIGIN_M
  )
  return {
    "nonfinite_sample_count": sum(
      int(np.count_nonzero(~np.isfinite(value)))
      for value in contact_raw.values()
    ),
    "outer_face_binding_violation_count": int(np.count_nonzero(
      finite_geometry
      & (
        np.abs(outer_face_x - expected_outer_face_x)
        > QUALIFICATION_GEOMETRY_WRITE_ATOL
      )
    )),
  }


def _recompute_first_riser_truth(
  contact_raw: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
  """Recompute impact steps and the tick-major mask from archived raw data."""

  if set(contact_raw) != _STAIR_CONTACT_RAW_KEYS:
    raise ValueError("C2-j3 raw contact arrays have missing or unexpected keys.")
  for name, value in contact_raw.items():
    if not isinstance(value, np.ndarray) or value.dtype != np.float32:
      raise ValueError(f"{name} must be a numpy.float32 array.")

  found = contact_raw["stair_contact_found"]
  expected_prefix = (
    QUALIFICATION_DRIVE_STEPS,
    QUALIFICATION_PAIRS_PER_CELL,
  )
  if found.ndim != 3 or tuple(found.shape[:2]) != expected_prefix:
    raise ValueError(
      "stair_contact_found must have shape "
      f"({QUALIFICATION_DRIVE_STEPS}, {QUALIFICATION_PAIRS_PER_CELL}, slots)."
    )
  if found.shape[2] <= 0:
    raise ValueError("stair_contact_found must contain contact slots.")
  expected_vector_shape = (*found.shape, 3)
  for name in (
    "stair_contact_force_contact_frame",
    "stair_contact_pos_global",
    "stair_contact_normal_global",
  ):
    if contact_raw[name].shape != expected_vector_shape:
      raise ValueError(f"{name} must have shape {expected_vector_shape}.")
  outer_face_x = contact_raw["stair_outer_face_x"]
  if outer_face_x.shape != (QUALIFICATION_PAIRS_PER_CELL,):
    raise ValueError(
      "stair_outer_face_x must have shape "
      f"({QUALIFICATION_PAIRS_PER_CELL},)."
    )
  terrain_origin_x = contact_raw["stair_terrain_origin_x"]
  if terrain_origin_x.shape != (QUALIFICATION_PAIRS_PER_CELL,):
    raise ValueError(
      "stair_terrain_origin_x must have shape "
      f"({QUALIFICATION_PAIRS_PER_CELL},)."
    )
  _contact_raw_health(contact_raw)

  def _env_major(name: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(contact_raw[name])).transpose(0, 1)

  riser_by_env = capture.riser_contact_mask_over_time(
    found=_env_major("stair_contact_found"),
    force_contact_frame=_env_major("stair_contact_force_contact_frame"),
    pos_global=_env_major("stair_contact_pos_global"),
    normal_global=_env_major("stair_contact_normal_global"),
    outer_face_x=torch.from_numpy(np.ascontiguousarray(outer_face_x)),
  ).any(dim=-1)
  has_impact = riser_by_env.any(dim=-1)
  first_impact = torch.full(
    (QUALIFICATION_PAIRS_PER_CELL,), -1, dtype=torch.long
  )
  first_impact[has_impact] = (
    riser_by_env[has_impact].to(torch.long).argmax(dim=-1)
  )
  return (
    first_impact.numpy().astype(np.int64, copy=False),
    riser_by_env.transpose(0, 1).numpy().astype(np.bool_, copy=False),
  )


def _health_counts(values: torch.Tensor, env_ids: list[int]) -> int:
  index = torch.tensor(env_ids, device=values.device, dtype=torch.long)
  return int(values[index].sum().item())


def run_cell(
  env: Any,
  predictor: InnovationPredictor,
  *,
  cell: dict[str, Any],
  raw_path: Path,
  run_protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
  reset = _paired_reset(env, cell=cell)
  pairs = reset["pairs"]
  flat_ids = [pair["flat_env_id"] for pair in pairs]
  stair_ids = [pair["stair_env_id"] for pair in pairs]
  action = env.action_manager.get_term("hybrid_wheel_leg")
  robot = env.scene["robot"]
  wheel_ids = action._wheel_ids
  if len(wheel_ids) != 2:
    raise RuntimeError("C2-j3 requires exactly two wheel joints.")
  accelerometer = env.scene.sensors[capture.IMU_ACCELEROMETER_SCENE_NAME]
  contact_sensor = env.scene.sensors[capture.DIAGNOSTIC_SENSOR_NAME]
  portable_config = identification._portable_config(action, robot)
  portable_states = [
    reset_state(float(cell["height_m"]), float(cell["pitch_rad"]))
    for _ in range(env.num_envs)
  ]
  actions = torch.zeros((env.num_envs, 6), device=env.device)
  terminated_ever = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  timeout_ever = torch.zeros_like(terminated_ever)
  contact_ever = torch.zeros_like(terminated_ever)
  settle_riser_ever = torch.zeros_like(terminated_ever)
  rows: dict[str, list[np.ndarray]] = {
    name: [] for name in (
      "z", "u", "next_z", "shaped_posture", "features", "active",
      "wheel_targets", "portable_targets", "specific_force_x",
      "projected_gravity_x",
    )
  }
  sensor_history: dict[str, list[torch.Tensor]] = {
    field: [] for field in capture.DIAGNOSTIC_SENSOR_FIELDS
  }
  domain_violations = 0
  predictor_evaluation_errors = 0
  vx_values = np.full(env.num_envs, float(cell["vx_mps"]), dtype=np.float64)
  settle_vx = np.zeros(env.num_envs, dtype=np.float64)

  def step(vx: np.ndarray, *, record: bool) -> None:
    nonlocal portable_states, domain_violations, predictor_evaluation_errors
    identification._force_commands(
      env, vx, height=float(cell["height_m"]), pitch=float(cell["pitch_rad"])
    )
    z = identification._sensor_state(robot, wheel_ids).detach().cpu().numpy().astype(np.float64)
    shaped = (
      env.command_manager.get_command("posture")
      .detach().cpu().numpy().astype(np.float64)
    )
    portable, portable_states = identification._portable_targets(
      portable_config,
      portable_states,
      robot,
      wheel_ids,
      vx,
      float(cell["height_m"]),
      float(cell["pitch_rad"]),
    )
    _obs, _reward, terminated, timeout, _extras = env.step(actions)
    identification._force_commands(
      env, vx, height=float(cell["height_m"]), pitch=float(cell["pitch_rad"])
    )
    actual_targets = action.wheel_targets.detach().cpu().numpy().astype(np.float64)
    next_z = identification._sensor_state(robot, wheel_ids).detach().cpu().numpy().astype(np.float64)
    direct_contact = non_wheel_ground_contact(
      env, NON_WHEEL_GROUND_SENSOR_NAME
    ).bool()
    termination_contact = env.termination_manager.get_term(
      "non_wheel_ground_contact"
    ).bool()
    contact = stair.merge_contact_observations(direct_contact, termination_contact)
    terminated_ever.logical_or_(terminated.bool())
    timeout_ever.logical_or_(timeout.bool())
    contact_ever.logical_or_(contact)

    found = contact_sensor.data.found
    riser = capture.riser_contact_mask(
      found=found,
      force_contact_frame=contact_sensor.data.force,
      pos_global=contact_sensor.data.pos,
      normal_global=contact_sensor.data.normal,
      outer_face_x=reset["outer_face_x"],
    ).any(dim=-1)
    if not record:
      settle_riser_ever.logical_or_(riser)
      return

    u = (0.5 * (actual_targets[:, 1] - actual_targets[:, 0]))[:, None]
    predicted = np.full_like(next_z, np.nan)
    for env_index in range(env.num_envs):
      state = z[env_index]
      value = float(u[env_index, 0])
      try:
        predicted[env_index] = predictor.predict(
          state,
          value,
          float(shaped[env_index, 0]),
          float(shaped[env_index, 1]),
        )
      except ValueError as error:
        if "outside the fitted domain" in str(error):
          domain_violations += 1
        elif "outside the registered rectangle" in str(error):
          pass
        else:
          predictor_evaluation_errors += 1
    innovation = np.abs(next_z - predicted)
    specific_force_x = accelerometer.data[:, 0]
    projected_gravity_x = robot.data.projected_gravity_b[:, 0]
    deceleration = capture.body_forward_deceleration(
      specific_force_x, projected_gravity_x
    ).detach().cpu().numpy().astype(np.float64)
    features = np.column_stack((innovation, deceleration))
    rows["z"].append(z)
    rows["u"].append(u)
    rows["next_z"].append(next_z)
    rows["shaped_posture"].append(shaped)
    rows["features"].append(features)
    rows["active"].append(np.ones((env.num_envs, 1), dtype=bool))
    rows["wheel_targets"].append(actual_targets)
    rows["portable_targets"].append(portable)
    rows["specific_force_x"].append(
      specific_force_x.detach().cpu().numpy().astype(np.float64)[:, None]
    )
    rows["projected_gravity_x"].append(
      projected_gravity_x.detach().cpu().numpy().astype(np.float64)[:, None]
    )
    for field in capture.DIAGNOSTIC_SENSOR_FIELDS:
      sensor_history[field].append(getattr(contact_sensor.data, field).clone())

  for _ in range(int(run_protocol["settle_steps"])):
    step(settle_vx, record=False)
  drive_start_past_face_count = int(
    (robot.data.root_link_pos_w[:, 0] >= reset["outer_face_x"]).sum().item()
  )
  drive_start_past_face = (
    robot.data.root_link_pos_w[:, 0] >= reset["outer_face_x"]
  ).detach().cpu().numpy().astype(bool)
  for _ in range(int(run_protocol["drive_steps"])):
    step(vx_values, record=True)

  arrays = {name: np.stack(values) for name, values in rows.items()}
  expected_posture = np.asarray(
    [cell["height_m"], cell["pitch_rad"]], dtype=np.float64
  )
  posture_error = np.max(
    np.abs(arrays["shaped_posture"] - expected_posture), axis=2
  )
  posture_violations = int(
    np.count_nonzero(posture_error > QUALIFICATION_POSTURE_CAPTURE_ATOL)
  )
  portable_delta = np.abs(
    arrays["wheel_targets"] - arrays["portable_targets"]
  )
  portable_finite = np.all(np.isfinite(portable_delta), axis=2)
  portable_max_error = (
    float(np.max(portable_delta)) if np.all(portable_finite) else None
  )
  portable_target_violations = int(np.count_nonzero(
    portable_finite
    & (np.max(np.where(np.isfinite(portable_delta), portable_delta, 0.0), axis=2)
       > QUALIFICATION_PORTABLE_EQUIVALENCE_ATOL)
  ))
  raw_numeric = (
    arrays["z"],
    arrays["u"],
    arrays["next_z"],
    arrays["shaped_posture"],
    arrays["features"],
    arrays["wheel_targets"],
    arrays["portable_targets"],
    arrays["specific_force_x"],
    arrays["projected_gravity_x"],
  )
  nonfinite_samples = sum(
    int(np.count_nonzero(~np.isfinite(value))) for value in raw_numeric
  )
  negative_feature_samples = int(
    np.count_nonzero(arrays["features"] < 0.0)
  )
  sensor_stacked = capture._stack_samples(sensor_history)
  stair_ids_tensor = torch.tensor(stair_ids, device=env.device, dtype=torch.long)
  contact_raw = _stair_contact_raw_arrays(
    sensor_stacked,
    stair_ids_tensor,
    reset["outer_face_x"],
    reset["terrain_origin_x"],
  )
  contact_health = _contact_raw_health(contact_raw)
  nonfinite_samples += contact_health["nonfinite_sample_count"]
  first_impact, riser_mask = _recompute_first_riser_truth(contact_raw)
  window_valid = (
    (first_impact >= QUALIFICATION_PRE_IMPACT_STEPS)
    & (
      first_impact + QUALIFICATION_POST_IMPACT_STEPS
      < QUALIFICATION_DRIVE_STEPS
    )
  )
  flat_features = _split(arrays["features"], flat_ids)
  stair_features = _split(arrays["features"], stair_ids)
  flat_active = _split(arrays["active"], flat_ids)[:, :, 0]
  stair_active = _split(arrays["active"], stair_ids)[:, :, 0]
  np.savez(
    raw_path,
    flat_z=_split(arrays["z"], flat_ids),
    stair_z=_split(arrays["z"], stair_ids),
    flat_u=_split(arrays["u"], flat_ids),
    stair_u=_split(arrays["u"], stair_ids),
    flat_next_z=_split(arrays["next_z"], flat_ids),
    stair_next_z=_split(arrays["next_z"], stair_ids),
    flat_shaped_posture=_split(arrays["shaped_posture"], flat_ids),
    stair_shaped_posture=_split(arrays["shaped_posture"], stair_ids),
    flat_features=flat_features,
    stair_features=stair_features,
    flat_active=flat_active,
    stair_active=stair_active,
    stair_riser_contact=riser_mask,
    impact_steps=first_impact,
    reset_perturbations=reset["perturbations"].detach().cpu().numpy(),
    flat_reset_relative=_split(
      reset["relative_root_states"].detach().cpu().numpy()[None, ...], flat_ids
    )[0],
    stair_reset_relative=_split(
      reset["relative_root_states"].detach().cpu().numpy()[None, ...], stair_ids
    )[0],
    flat_written_reset_relative=_split(
      reset["written_relative_root_states"].detach().cpu().numpy()[None, ...],
      flat_ids,
    )[0],
    stair_written_reset_relative=_split(
      reset["written_relative_root_states"].detach().cpu().numpy()[None, ...],
      stair_ids,
    )[0],
    flat_wheel_targets=_split(arrays["wheel_targets"], flat_ids),
    stair_wheel_targets=_split(arrays["wheel_targets"], stair_ids),
    flat_portable_targets=_split(arrays["portable_targets"], flat_ids),
    stair_portable_targets=_split(arrays["portable_targets"], stair_ids),
    flat_specific_force_x=_split(arrays["specific_force_x"], flat_ids),
    stair_specific_force_x=_split(arrays["specific_force_x"], stair_ids),
    flat_projected_gravity_x=_split(arrays["projected_gravity_x"], flat_ids),
    stair_projected_gravity_x=_split(arrays["projected_gravity_x"], stair_ids),
    flat_terminated=terminated_ever[flat_ids].detach().cpu().numpy().astype(bool),
    stair_terminated=terminated_ever[stair_ids].detach().cpu().numpy().astype(bool),
    flat_timeout=timeout_ever[flat_ids].detach().cpu().numpy().astype(bool),
    stair_timeout=timeout_ever[stair_ids].detach().cpu().numpy().astype(bool),
    flat_non_wheel_contact=contact_ever[flat_ids].detach().cpu().numpy().astype(bool),
    stair_non_wheel_contact=contact_ever[stair_ids].detach().cpu().numpy().astype(bool),
    flat_settle_riser_contact=settle_riser_ever[flat_ids].detach().cpu().numpy().astype(bool),
    stair_settle_riser_contact=settle_riser_ever[stair_ids].detach().cpu().numpy().astype(bool),
    flat_drive_start_past_face=drive_start_past_face[flat_ids],
    stair_drive_start_past_face=drive_start_past_face[stair_ids],
    **contact_raw,
  )
  health = {
    "flat_termination_count": _health_counts(terminated_ever, flat_ids),
    "stair_termination_count": _health_counts(terminated_ever, stair_ids),
    "flat_timeout_count": _health_counts(timeout_ever, flat_ids),
    "stair_timeout_count": _health_counts(timeout_ever, stair_ids),
    "flat_non_wheel_contact_count": _health_counts(contact_ever, flat_ids),
    "stair_non_wheel_contact_count": _health_counts(contact_ever, stair_ids),
    "settle_riser_contact_count": int(settle_riser_ever.sum().item()),
    "drive_start_past_face_count": drive_start_past_face_count,
    "missing_impact_count": int(np.count_nonzero(first_impact < 0)),
    "invalid_window_count": int(np.count_nonzero(~window_valid)),
    "predictor_domain_violation_count": domain_violations,
    "posture_violation_count": posture_violations,
    "predictor_evaluation_error_count": predictor_evaluation_errors,
    "nonfinite_sample_count": nonfinite_samples,
    "negative_feature_sample_count": negative_feature_samples,
    "portable_target_violation_count": portable_target_violations,
    "outer_face_binding_violation_count": contact_health[
      "outer_face_binding_violation_count"
    ],
  }
  summary = {
    "cell": cell,
    "raw_file": raw_path.name,
    "raw_sha256": _sha256(raw_path),
    "raw_shape": [int(run_protocol["drive_steps"]), len(flat_ids)],
    "impact_steps": first_impact.tolist(),
    "diagnostic_windows": [
      {
        "slot": slot,
        "start_tick": int(impact - QUALIFICATION_PRE_IMPACT_STEPS),
        "impact_tick": int(impact),
        "end_tick": int(impact + QUALIFICATION_POST_IMPACT_STEPS),
      }
      for slot, impact in enumerate(first_impact)
      if impact >= 0
    ],
    "paired_reset_max_abs_error": reset["paired_reset_max_abs_error"],
    "written_reset_max_abs_error": reset["written_reset_max_abs_error"],
    "written_paired_reset_max_abs_error": reset[
      "written_paired_reset_max_abs_error"
    ],
    "root_pitch_max_abs_error_rad": reset["root_pitch_max_abs_error_rad"],
    "root_roll_yaw_max_abs_rad": reset["root_roll_yaw_max_abs_rad"],
    "other_root_velocity_max_abs": reset["other_root_velocity_max_abs"],
    "portable_max_abs_target_error_radps": portable_max_error,
    "health": health,
  }
  evaluation = {
    "cell": cell,
    "flat_features": flat_features,
    "stair_features": stair_features,
    "flat_active": flat_active,
    "stair_active": stair_active,
    "impact_steps": first_impact,
  }
  return summary, evaluation


def _cell_invalid(summary: dict[str, Any]) -> bool:
  health = summary["health"]
  return (
    summary["raw_shape"]
    != [QUALIFICATION_DRIVE_STEPS, QUALIFICATION_PAIRS_PER_CELL]
    or summary["paired_reset_max_abs_error"] != 0.0
    or summary["written_reset_max_abs_error"] > QUALIFICATION_RESET_WRITE_ATOL
    or summary["written_paired_reset_max_abs_error"]
    > QUALIFICATION_RESET_WRITE_ATOL
    or summary["root_pitch_max_abs_error_rad"] > 1.0e-7
    or summary["root_roll_yaw_max_abs_rad"] > 1.0e-7
    or summary["other_root_velocity_max_abs"] != 0.0
    or any(int(value) != 0 for value in health.values())
    or len(summary["impact_steps"]) != QUALIFICATION_PAIRS_PER_CELL
    or len(summary["diagnostic_windows"]) != QUALIFICATION_PAIRS_PER_CELL
  )


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  if args.output_dir.exists() and any(args.output_dir.iterdir()):
    raise FileExistsError("C2-j3 output directory must be absent or empty.")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  _configure_artifact_environment(args)
  predictor_payload = json.loads(args.predictor.read_text(encoding="utf-8-sig"))
  floor_payload = json.loads(args.transition_floor.read_text(encoding="utf-8-sig"))
  predictor = parse_innovation_predictor(predictor_payload)
  floor = parse_transition_floor(floor_payload, predictor_hash=predictor.predictor_hash)
  if predictor.predictor_hash != PREDICTOR_HASH:
    raise ValueError("C2-j3 requires the frozen C2-j1 predictor.")
  if floor["floor_hash"] != FLOOR_HASH or floor["threshold_table_hash"] != THRESHOLD_TABLE_HASH:
    raise ValueError("C2-j3 requires the frozen C2-j2 floor and threshold table.")
  run_protocol = protocol(args.smoke, args.device)
  cfg = capture.make_causal_env_cfg((0.0, 0.01), QUALIFICATION_PAIRS_PER_CELL)
  cfg.seed = 3
  from mjlab.envs import ManagerBasedRlEnv

  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    _assert_runtime_stack(env)
    cell_summaries: list[dict[str, Any]] = []
    evaluation_cells: list[dict[str, Any]] = []
    for cell in qualification_cells():
      index = int(cell["cell_index"])
      print(f"[C2-j3] cell {index}/17")
      summary, evaluation = run_cell(
        env,
        predictor,
        cell=cell,
        raw_path=args.output_dir / f"cell_{index:02d}.npz",
        run_protocol=run_protocol,
      )
      cell_summaries.append(summary)
      evaluation_cells.append(evaluation)
  finally:
    env.close()

  invalid = any(_cell_invalid(summary) for summary in cell_summaries)
  candidates: list[dict[str, Any]] = []
  selected: dict[str, Any] | None = None
  if not invalid:
    table = floor["threshold_table"]
    for index, row in enumerate(table):
      if index % 10 == 0 or index == len(table) - 1:
        print(f"[C2-j3] replay candidate {index}/{len(table) - 1}")
      candidates.append(evaluate_qualification_candidate(row, evaluation_cells))
    selected = select_qualification_candidate(candidates)
  classification = (
    "SMOKE_COMPLETE"
    if args.smoke and not invalid
    else "SMOKE_INVALID"
    if args.smoke
    else "INVALID_INNOVATION_CAPTURE"
    if invalid
    else "INNOVATION_DETECTOR_QUALIFIED"
    if selected is not None
    else "C2_INNOVATION_DETECTOR_UNQUALIFIED_STOP"
  )
  evidence_eligible = (
    not args.smoke
    and classification
    in (
      "INNOVATION_DETECTOR_QUALIFIED",
      "C2_INNOVATION_DETECTOR_UNQUALIFIED_STOP",
    )
  )
  next_step = {
    "INNOVATION_DETECTOR_QUALIFIED": "FREEZE_AND_INDEPENDENT_AUDIT_BEFORE_C3",
    "C2_INNOVATION_DETECTOR_UNQUALIFIED_STOP": "STOP_FOR_USER_ROUTE_DECISION",
    "INVALID_INNOVATION_CAPTURE": "INDEPENDENT_IMPLEMENTATION_DIAGNOSIS_ONLY",
    "SMOKE_COMPLETE": "NONE_CPU_MECHANICS_ONLY",
    "SMOKE_INVALID": "REPAIR_CPU_MECHANICS_BEFORE_FORMAL_RUN",
  }[classification]
  payload: dict[str, Any] = {
    "schema_version": QUALIFICATION_SCHEMA_VERSION,
    "artifact_type": QUALIFICATION_ARTIFACT_TYPE,
    "probe": PROBE_NAME,
    "classification": classification,
    "git_sha": stair._git_sha(stair.REPOSITORY_PATH),
    "mjlab_git_sha": stair._git_sha(Path(stair.mjlab.__file__).resolve().parents[2]),
    "predictor_hash": predictor.predictor_hash,
    "floor_hash": floor["floor_hash"],
    "threshold_table_hash": floor["threshold_table_hash"],
    "bindings": dict(EXPECTED_BINDINGS),
    "protocol": run_protocol,
    "cells": cell_summaries,
    "completed_cell_count": len(cell_summaries),
    "completed_pair_count": len(cell_summaries) * QUALIFICATION_PAIRS_PER_CELL,
    "completed_candidate_count": len(candidates),
    "qualified_candidate_count": sum(
      candidate["qualified"] is True for candidate in candidates
    ),
    "candidates": candidates,
    "selected_candidate": (
      None if selected is None else qualification_selection(selected)
    ),
    "evidence_eligible": evidence_eligible,
    "promotion_eligible": False,
    "training_eligible": False,
    "checkpoint": None,
    "next_step": next_step,
  }
  payload["detector_hash"] = canonical_hash(payload, hash_field="detector_hash")
  if classification == "INNOVATION_DETECTOR_QUALIFIED":
    parse_innovation_detector_qualification(
      payload,
      predictor_hash=PREDICTOR_HASH,
      floor_payload=floor_payload,
    )
  output = args.output_dir / "c2_innovation_detector_qualification.json"
  output.write_text(
    json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  print(f"CLASSIFICATION={classification}")
  print(f"DETECTOR_HASH={payload['detector_hash']}")
  print(f"RESULT={output.resolve()}")


if __name__ == "__main__":
  main()
