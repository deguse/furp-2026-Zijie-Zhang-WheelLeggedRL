#!/usr/bin/env python3
"""Measure the final-classical HopperTrex direct-roll stair boundary (R0).

The formal path binds the five frozen final-C1 artifacts, applies an exactly
zero six-dimensional residual, and reports a sampled bracket. ``--smoke``
constructs real sub-centimetre terrain on CPU but is never evidence eligible.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

PROJECT_PATH = Path(__file__).resolve().parents[1]
SRC_PATH = Path(__file__).resolve().parents[2]
REPOSITORY_PATH = Path(__file__).resolve().parents[3]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

try:
  from hoppertrex_mjlab import tasks
  from hoppertrex_mjlab.assets.HopperTrex_CFG import (
    RMD_L_9025_35T_PEAK_TORQUE,
    WHEEL_VELOCITY_DAMPING,
  )
  from hoppertrex_mjlab.hybrid.identification import (
    NOMINAL_WHEEL_RADIUS_M,
  )
  from hoppertrex_mjlab.hybrid.roll_assist import (
    ROLL_FIRST_ARTIFACT_SPECS,
    ROLL_FIRST_CELL_PASS_SUCCESSES,
    ROLL_FIRST_CONTROL_DECIMATION,
    ROLL_FIRST_CONTROL_FREQUENCY_HZ,
    ROLL_FIRST_DRIVE_STEPS,
    ROLL_FIRST_ENVS_PER_HEIGHT,
    ROLL_FIRST_FORMAL_CAP_M,
    ROLL_FIRST_PHYSICS_TIMESTEP_S,
    ROLL_FIRST_POSTURE_CARDS,
    ROLL_FIRST_REPEATS,
    ROLL_FIRST_RESET_JOINT_STATE,
    ROLL_FIRST_RESET_ORIENTATION,
    ROLL_FIRST_SETTLE_STEPS,
    ROLL_FIRST_STABLE_STEPS,
    ROLL_FIRST_SUBSTEP_SUPPORT_SCOPE,
    ROLL_FIRST_TASK,
    ROLL_FIRST_TERRAIN_PROTOCOL,
    ROLL_FIRST_WHEEL_CONTACT_SOLIMP,
    ROLL_FIRST_WHEEL_CONTACT_SOLREF,
    roll_first_artifact_paths,
  )
  from hoppertrex_mjlab.hybrid.roll_pose_schedule import (
    CONTROL_DT_S as ROLL_POSE_CONTROL_DT_S,
  )
  from hoppertrex_mjlab.hybrid.roll_pose_schedule import (
    RollPoseSchedule,
    make_roll_pose_schedule_state,
    roll_pose_schedule_step,
  )
  from hoppertrex_mjlab.tasks.hoppertrex_balance_task import (
    NON_WHEEL_GROUND_SENSOR_NAME,
    non_wheel_ground_contact,
  )
  from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
    hybrid_provenance_lines,
    make_hoppertrex_hybrid_env_cfg,
  )
except ImportError:
  import tasks  # type: ignore[no-redef]  # noqa: F401
  from assets.HopperTrex_CFG import (  # type: ignore[no-redef]
    RMD_L_9025_35T_PEAK_TORQUE,
    WHEEL_VELOCITY_DAMPING,
  )
  from hybrid.identification import (
    NOMINAL_WHEEL_RADIUS_M,  # type: ignore[no-redef]
  )
  from hybrid.roll_assist import (  # type: ignore[no-redef]
    ROLL_FIRST_ARTIFACT_SPECS,
    ROLL_FIRST_CELL_PASS_SUCCESSES,
    ROLL_FIRST_CONTROL_DECIMATION,
    ROLL_FIRST_CONTROL_FREQUENCY_HZ,
    ROLL_FIRST_DRIVE_STEPS,
    ROLL_FIRST_ENVS_PER_HEIGHT,
    ROLL_FIRST_FORMAL_CAP_M,
    ROLL_FIRST_PHYSICS_TIMESTEP_S,
    ROLL_FIRST_POSTURE_CARDS,
    ROLL_FIRST_REPEATS,
    ROLL_FIRST_RESET_JOINT_STATE,
    ROLL_FIRST_RESET_ORIENTATION,
    ROLL_FIRST_SETTLE_STEPS,
    ROLL_FIRST_STABLE_STEPS,
    ROLL_FIRST_SUBSTEP_SUPPORT_SCOPE,
    ROLL_FIRST_TASK,
    ROLL_FIRST_TERRAIN_PROTOCOL,
    ROLL_FIRST_WHEEL_CONTACT_SOLIMP,
    ROLL_FIRST_WHEEL_CONTACT_SOLREF,
    roll_first_artifact_paths,
  )
  from hybrid.roll_pose_schedule import (  # type: ignore[no-redef]
    CONTROL_DT_S as ROLL_POSE_CONTROL_DT_S,
  )
  from hybrid.roll_pose_schedule import (
    RollPoseSchedule,
    make_roll_pose_schedule_state,
    roll_pose_schedule_step,
  )
  from tasks.hoppertrex_balance_task import (  # type: ignore[no-redef]
    NON_WHEEL_GROUND_SENSOR_NAME,
    non_wheel_ground_contact,
  )
  from tasks.hoppertrex_hybrid_task import (  # type: ignore[no-redef]
    hybrid_provenance_lines,
    make_hoppertrex_hybrid_env_cfg,
  )

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.config import flat, pyramid_stairs

TASK = ROLL_FIRST_TASK
PROBE_NAME = "hoppertrex_roll_boundary_r0"
SEED = 1
HEIGHT_STEP_UM = 2_500
FORMAL_CAP_UM = round(ROLL_FIRST_FORMAL_CAP_M * 1_000_000)
SMOKE_HEIGHTS_M = (0.0025, 0.005, 0.0075)
POSTURE_CARDS = ROLL_FIRST_POSTURE_CARDS
TERRAIN_SIZE_M = (8.0, 8.0)
TERRAIN_BORDER_WIDTH_M = 1.0
STEP_WIDTH_M = 0.30
PLATFORM_WIDTH_M = 3.0
START_OFFSET_M = 0.25
CROSS_DEPTH_M = 0.15
RESET_X_JITTER_M = 0.02
RESET_Y_JITTER_M = 0.03
RESET_VX_JITTER_MPS = 0.01
RESET_PITCH_RATE_JITTER_RADPS = 0.02
COMMAND_VX_MPS = 0.07
CONTROL_FREQUENCY_HZ = ROLL_FIRST_CONTROL_FREQUENCY_HZ
OFFICIAL_ENVS_PER_HEIGHT = ROLL_FIRST_ENVS_PER_HEIGHT
OFFICIAL_REPEATS = ROLL_FIRST_REPEATS
OFFICIAL_SETTLE_STEPS = ROLL_FIRST_SETTLE_STEPS
OFFICIAL_DRIVE_STEPS = ROLL_FIRST_DRIVE_STEPS
OFFICIAL_STABLE_STEPS = ROLL_FIRST_STABLE_STEPS
CELL_PASS_SUCCESSES = ROLL_FIRST_CELL_PASS_SUCCESSES
PITCH_LIMIT_RAD = 0.10
ROLL_LIMIT_RAD = 0.10
PITCH_RATE_LIMIT_RADPS = 0.5
LEFT_SENSOR = "roll_boundary_left_wheel_contact"
RIGHT_SENSOR = "roll_boundary_right_wheel_contact"
SENSOR_FIELDS = ("found", "force", "dist", "pos", "normal")
SENSOR_SLOTS = 8
ZERO_ACTION_MASK = (False, False, False, False, False, False)
ROLL_BOUNDARY_WHEEL_SOLREF = ROLL_FIRST_WHEEL_CONTACT_SOLREF
# Evidence-backed R0 contact model. The original wheel collision impedance
# solref=(0.005, 1) / solimp=(0.95, 0.99, 0.001) produced real
# 0.08--0.16 mm bilateral gaps on a finite box under MJWarp even after a
# posture-consistent reset. MuJoCo clamps positive solref time constants below
# 2*dt to 2*dt when refsafe is enabled; explicitly using the 20 ms terrain
# time constant plus the terrain impedance avoids relying on that hidden
# clamp. A native-MuJoCo cross-check did not reproduce the finite-box failure,
# and MJWarp currently lacks cylinder-box multicontact support (mujoco_warp#1555). This softer,
# still high-impedance setting removed the chatter without changing dt,
# controller cadence, actuator limits, or the strict no-airborne contract.
ROLL_BOUNDARY_WHEEL_SOLIMP = ROLL_FIRST_WHEEL_CONTACT_SOLIMP

EXPECTED_SCHEDULE_HASH = "8fe8548bca85978c164bbd7de39d2d6463cdfd8d7ab91796cf57696b0f64e203"
ARTIFACT_SPECS = ROLL_FIRST_ARTIFACT_SPECS
CLASSIFICATIONS = (
  "NO_POSITIVE_CLASSICAL_CROLL", "CLASSICAL_CROLL_BRACKETED",
  "NEXT_HEIGHT_UNSAFE_STOP", "CLASSICAL_CROLL_AT_LEAST_CAP",
  "EXTEND_ROLL_BOUNDARY_SWEEP", "NON_MONOTONIC_STOP",
  "INVALID_FLAT_CONTROL_STOP",
)


def frozen_artifact_paths(repository_path: Path = REPOSITORY_PATH) -> dict[str, Path]:
  return roll_first_artifact_paths(repository_path)


def height_to_micrometres(height_m: float) -> int:
  value = float(height_m)
  if not math.isfinite(value) or value < 0.0:
    raise ValueError("RollBoundary heights must be finite and non-negative.")
  integer = round(value * 1_000_000.0)
  if not math.isclose(value, integer / 1_000_000.0, rel_tol=0.0, abs_tol=1.0e-12):
    raise ValueError("RollBoundary heights must be exact to one micrometre.")
  return integer


def validate_heights(heights: Iterable[float]) -> tuple[float, ...]:
  values = tuple(float(value) for value in heights)
  if not values:
    raise ValueError("RollBoundary requires at least one height.")
  integers = tuple(height_to_micrometres(value) for value in values)
  if len(integers) != len(set(integers)):
    raise ValueError("RollBoundary heights collide at micrometre resolution.")
  if any(right <= left for left, right in pairwise(integers)):
    raise ValueError("RollBoundary heights must be strictly increasing.")
  return tuple(value / 1_000_000.0 for value in integers)


def terrain_key(height_m: float) -> str:
  return f"stair_{height_to_micrometres(height_m):06d}um"


def formal_heights(max_height_um: int) -> tuple[float, ...]:
  if max_height_um not in (10_000, 20_000, 30_000):
    raise ValueError("Formal maximum must be 10, 20, or 30 mm.")
  return tuple(value / 1_000_000.0 for value in range(0, max_height_um + 1, HEIGHT_STEP_UM))


def roll_boundary_sub_terrains(heights: Iterable[float]) -> dict[str, Any]:
  canonical = validate_heights(heights)
  result = {}
  for height in canonical:
    key = terrain_key(height)
    if height == 0.0:
      # A zero-height pyramid is not flat: MjLab emits many 2-micrometre-thick
      # adjacent boxes. Their seams amplified cylinder-box contact chatter and
      # invalidated the flat qualification cell. Use one finite, 1 m-thick box
      # with the same patch size and z=0 top instead.
      result[key] = flat(proportion=1.0, size=TERRAIN_SIZE_M)
    else:
      result[key] = pyramid_stairs(
        proportion=1.0, step_height_range=(height, height),
        step_width=STEP_WIDTH_M, platform_width=PLATFORM_WIDTH_M,
        border_width=TERRAIN_BORDER_WIDTH_M,
      )
  if len(result) != len(canonical):
    raise RuntimeError("RollBoundary terrain-key construction lost a height.")
  return result


def wheel_sensor_cfg(*, name: str, wheel_geom: str) -> ContactSensorCfg:
  expected = {LEFT_SENSOR: "wheel_left_collision", RIGHT_SENSOR: "wheel_right_collision"}
  if expected.get(name) != wheel_geom:
    raise ValueError("RollBoundary wheel sensor identity is invalid.")
  return ContactSensorCfg(
    name=name, primary=ContactMatch(mode="geom", pattern=wheel_geom, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"), fields=SENSOR_FIELDS,
    reduce="none", num_slots=SENSOR_SLOTS,
  )


def make_roll_boundary_env_cfg(
  heights: Iterable[float],
  envs_per_height: int,
  *,
  repository_path: Path = REPOSITORY_PATH,
  residual_mask: tuple[bool, ...] = ZERO_ACTION_MASK,
  action_scales: tuple[float, ...] | None = None,
):
  canonical = validate_heights(heights)
  if isinstance(envs_per_height, bool) or envs_per_height < 1:
    raise ValueError("RollBoundary envs_per_height must be positive.")
  cfg = make_hoppertrex_hybrid_env_cfg(
    stage=5, play=True, **frozen_artifact_paths(repository_path)
  )
  robot_cfg = cfg.scene.entities["robot"]
  if len(robot_cfg.collisions) != 1:
    raise ValueError("RollBoundary expects exactly one robot collision config.")
  wheel_collision = robot_cfg.collisions[0]
  wheel_collision.solref["wheel_.*_collision"] = ROLL_BOUNDARY_WHEEL_SOLREF
  wheel_collision.solimp["wheel_.*_collision"] = ROLL_BOUNDARY_WHEEL_SOLIMP
  cfg.seed = SEED
  cfg.auto_reset = False
  num_envs = len(canonical) * int(envs_per_height)
  cfg.scene.num_envs = num_envs
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      seed=SEED, curriculum=True, size=TERRAIN_SIZE_M, num_rows=1,
      num_cols=len(canonical), difficulty_range=(0.0, 0.0),
      sub_terrains=roll_boundary_sub_terrains(canonical),
    ),
    max_init_terrain_level=0, num_envs=num_envs,
  )
  cfg.scene.sensors = tuple(cfg.scene.sensors) + (
    wheel_sensor_cfg(name=LEFT_SENSOR, wheel_geom="wheel_left_collision"),
    wheel_sensor_cfg(name=RIGHT_SENSOR, wheel_geom="wheel_right_collision"),
  )
  cfg.episode_length_s = 1.0e9
  action = cfg.actions["hybrid_wheel_leg"]
  if len(residual_mask) != 6:
    raise ValueError("RollBoundary residual mask must have six entries.")
  action.action_mask = tuple(bool(value) for value in residual_mask)
  if action_scales is not None:
    if len(action_scales) != 6:
      raise ValueError("RollBoundary action scales must have six entries.")
    action.action_scales = tuple(float(value) for value in action_scales)
  action.dynamic_stair_maneuver = None
  action.dynamic_stair_request_command_name = None
  action.dynamic_stair_left_sensor_name = None
  action.dynamic_stair_right_sensor_name = None
  action.stair_trigger_sensor_name = None
  action.stair_mode_freezes_leg_reference = False
  action.stair_mode_forced = False
  action.__post_init__()
  validate_roll_boundary_env_cfg(
    cfg, canonical, expected_action_mask=tuple(bool(value) for value in residual_mask)
  )
  return cfg


def validate_roll_boundary_env_cfg(
  cfg: Any,
  heights: Iterable[float],
  *,
  expected_action_mask: tuple[bool, ...] = ZERO_ACTION_MASK,
) -> None:
  canonical = validate_heights(heights)
  terrain = cfg.scene.terrain
  generator = None if terrain is None else terrain.terrain_generator
  if (terrain is None or terrain.terrain_type != "generator" or generator is None
      or generator.num_rows != 1 or generator.num_cols != len(canonical)):
    raise ValueError("RollBoundary terrain grid does not match the request.")
  expected_keys = tuple(terrain_key(height) for height in canonical)
  if tuple(generator.sub_terrains) != expected_keys or len(generator.sub_terrains) != len(canonical):
    raise ValueError("RollBoundary terrain keys/order drifted.")
  observed_ranges = tuple(
    (0.0, 0.0) if height == 0.0 else
    tuple(float(value) for value in sub.step_height_range)
    for height, sub in zip(canonical, generator.sub_terrains.values(), strict=True)
  )
  if observed_ranges != tuple((height, height) for height in canonical):
    raise ValueError("RollBoundary terrain heights do not match their indices.")
  if canonical[0] == 0.0:
    flat_cfg = generator.sub_terrains[terrain_key(0.0)]
    if type(flat_cfg).__name__ != "BoxFlatTerrainCfg" or tuple(flat_cfg.size) != TERRAIN_SIZE_M:
      raise ValueError("RollBoundary zero-height cell must be one finite flat box.")
  robot_cfg = cfg.scene.entities["robot"]
  if len(robot_cfg.collisions) != 1:
    raise ValueError("RollBoundary robot collision config count drifted.")
  wheel_collision = robot_cfg.collisions[0]
  if (tuple(wheel_collision.solref["wheel_.*_collision"])
      != ROLL_BOUNDARY_WHEEL_SOLREF
      or tuple(wheel_collision.solimp["wheel_.*_collision"])
      != ROLL_BOUNDARY_WHEEL_SOLIMP):
    raise ValueError("RollBoundary wheel-contact model drifted.")
  action = cfg.actions.get("hybrid_wheel_leg")
  if action is None or tuple(action.action_mask) != expected_action_mask:
    raise ValueError("RollBoundary residual mask differs from the requested contract.")
  if any(expected_action_mask[:2]):
    raise ValueError("RollBoundary evaluation can never grant wheel residual authority.")
  if action.controller_gain_hash != EXPECTED_SCHEDULE_HASH:
    raise ValueError("RollBoundary is not bound to the final C1 schedule.")
  if not all((action.controller_qualified, action.calibration_hash is not None,
              action.yaw_calibration_qualified, action.posture_map_qualified,
              action.posture_artifact_hash is not None, action.station_calibration_qualified)):
    raise ValueError("RollBoundary final classical artifact stack is incomplete.")
  if (action.dynamic_stair_maneuver is not None or action.stair_trigger_sensor_name is not None
      or action.stair_mode_freezes_leg_reference or action.stair_mode_forced):
    raise ValueError("RollBoundary accidentally enabled a dynamic stair path.")
  sensors = {sensor.name: sensor for sensor in cfg.scene.sensors}
  for name, geom in {LEFT_SENSOR: "wheel_left_collision", RIGHT_SENSOR: "wheel_right_collision"}.items():
    sensor = sensors.get(name)
    if (sensor is None or sensor.primary.mode != "geom" or sensor.primary.pattern != geom
        or sensor.primary.entity != "robot" or sensor.secondary.mode != "body"
        or sensor.secondary.pattern != "terrain" or tuple(sensor.fields) != SENSOR_FIELDS
        or sensor.reduce != "none" or sensor.num_slots != SENSOR_SLOTS):
      raise ValueError(f"RollBoundary sensor {name!r} identity drifted.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--max-height-mm", type=int, choices=(10, 20, 30), default=10)
  parser.add_argument("--smoke", action="store_true")
  args = parser.parse_args(argv)
  if args.output.exists():
    parser.error(f"Refusing to overwrite RollBoundary output: {args.output}")
  if not args.smoke and args.device != "cuda:0":
    parser.error("The formal RollBoundary protocol is pinned to cuda:0.")
  if args.smoke and args.max_height_mm != 10:
    parser.error("Smoke uses the fixed 2.5/5/7.5 mm construction set.")
  return args


def protocol_for_mode(smoke: bool, max_height_mm: int = 10) -> dict[str, Any]:
  if smoke:
    return {
      "heights_m": SMOKE_HEIGHTS_M, "envs_per_height": 1, "repeats": 1,
      "settle_steps": 2, "drive_steps": 5, "stable_steps": 2,
      "evidence_eligible": False, "formal_cap_m": FORMAL_CAP_UM / 1_000_000.0,
    }
  return {
    "heights_m": formal_heights(int(max_height_mm) * 1_000),
    "envs_per_height": OFFICIAL_ENVS_PER_HEIGHT, "repeats": OFFICIAL_REPEATS,
    "settle_steps": OFFICIAL_SETTLE_STEPS, "drive_steps": OFFICIAL_DRIVE_STEPS,
    "stable_steps": OFFICIAL_STABLE_STEPS, "evidence_eligible": True,
    "formal_cap_m": FORMAL_CAP_UM / 1_000_000.0,
  }


def approach_geometry(origin_x: float) -> dict[str, float]:
  inner_half_width = 0.5 * (TERRAIN_SIZE_M[0] - 2.0 * TERRAIN_BORDER_WIDTH_M)
  face = float(origin_x) - inner_half_width
  return {"outer_face_x": face, "start_x": face - START_OFFSET_M, "cross_x": face + CROSS_DEPTH_M}


def _force_commands(
  env: ManagerBasedRlEnv,
  *,
  active: torch.Tensor,
  vx: float,
  height: float | torch.Tensor,
  pitch: float | torch.Tensor,
) -> None:
  twist = env.command_manager.get_term("twist")
  command_vx = active.float() * float(vx)
  for attribute in ("vel_command_b", "vel_command_w"):
    command = getattr(twist, attribute)
    command[:, :] = 0.0
    command[:, 0] = command_vx
  for attribute in ("is_standing_env", "is_heading_env", "is_world_env", "is_forward_env"):
    value = getattr(twist, attribute, None)
    if value is not None:
      value[:] = False
  posture = env.command_manager.get_term("posture")
  command = getattr(posture, "_command", None)
  if command is None:
    raise AttributeError("Posture command term does not expose _command.")
  target = getattr(posture, "_target", None)
  if target is not None:
    target[:, 0], target[:, 1] = height, pitch
  command[:, 0], command[:, 1] = height, pitch


def reset_perturbations(*, slots: int, card_name: str, repeat: int) -> torch.Tensor:
  if slots < 1 or repeat < 1:
    raise ValueError("Reset perturbations require positive slots and repeat.")
  names = [str(card["name"]) for card in POSTURE_CARDS]
  if card_name not in names:
    raise ValueError(f"Unknown RollBoundary posture card: {card_name}")
  generator = torch.Generator(device="cpu")
  generator.manual_seed(SEED * 10_000 + names.index(card_name) * 100 + repeat)
  unit = 2.0 * torch.rand((slots, 4), generator=generator) - 1.0
  return unit * torch.tensor([
    RESET_X_JITTER_M, RESET_Y_JITTER_M, RESET_VX_JITTER_MPS,
    RESET_PITCH_RATE_JITTER_RADPS,
  ])


def posture_target_from_coefficients(
  coefficients: torch.Tensor, *, height: float, pitch: float,
) -> torch.Tensor:
  if coefficients.shape != (3, 4):
    raise ValueError("RollBoundary posture coefficients must have shape [3, 4].")
  features = torch.tensor(
    [1.0, float(height), float(pitch)],
    device=coefficients.device, dtype=coefficients.dtype,
  )
  return features @ coefficients


def _posture_joint_targets(env: ManagerBasedRlEnv, *, height: float,
                           pitch: float) -> tuple[torch.Tensor, torch.Tensor]:
  """Return registered absolute leg targets and their joint ids."""
  term = env.action_manager.get_term("hybrid_wheel_leg")
  target = posture_target_from_coefficients(
    term._posture_coefficients, height=height, pitch=pitch,
  )
  return target, term._leg_ids


def _root_quaternion_for_pitch(pitch: float, *, device: str) -> torch.Tensor:
  """World quaternion for a pure body-y pitch."""
  half = 0.5 * float(pitch)
  return torch.tensor(
    [math.cos(half), 0.0, math.sin(half), 0.0],
    device=device, dtype=torch.float,
  )


def _reset_to_approach(env: ManagerBasedRlEnv, *, root_height: float, card_name: str,
                       repeat: int, height_count: int):
  env.reset()
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_types is None:
    raise RuntimeError("RollBoundary requires generated terrain types.")
  terrain_types = terrain.terrain_types.clone()
  counts = torch.bincount(terrain_types, minlength=height_count)
  if int(terrain_types.min()) != 0 or len(counts) != height_count or torch.any(counts == 0):
    raise RuntimeError("RollBoundary did not instantiate every requested height.")
  if not torch.all(counts == counts[0]):
    raise RuntimeError("RollBoundary heights have unequal environment counts.")
  slot_ids = torch.empty_like(terrain_types)
  for terrain_type in range(height_count):
    env_ids = torch.nonzero(terrain_types == terrain_type, as_tuple=False).squeeze(-1)
    slot_ids[env_ids] = torch.arange(len(env_ids), device=env.device)
  reset_values = reset_perturbations(
    slots=int(counts[0]), card_name=card_name, repeat=repeat,
  ).to(env.device)[slot_ids]
  origins = env.scene.env_origins
  geometry = approach_geometry(0.0)
  robot = env.scene["robot"]
  card = next(card for card in POSTURE_CARDS if str(card["name"]) == card_name)
  target, leg_ids = _posture_joint_targets(
    env, height=float(root_height), pitch=float(card["pitch_rad"]),
  )
  joint_pos = robot.data.default_joint_pos.clone()
  joint_vel = torch.zeros_like(joint_pos)
  joint_pos[:, leg_ids] = target
  robot.write_joint_state_to_sim(joint_pos, joint_vel)
  roots = robot.data.default_root_state.clone()
  roots[:, 0] = origins[:, 0] + geometry["start_x"] + reset_values[:, 0]
  roots[:, 1] = origins[:, 1] + reset_values[:, 1]
  roots[:, 2] = float(root_height)
  roots[:, 3:7] = _root_quaternion_for_pitch(
    float(card["pitch_rad"]), device=env.device,
  )
  roots[:, 7:13] = 0.0
  roots[:, 7], roots[:, 11] = reset_values[:, 2], reset_values[:, 3]
  robot.write_root_state_to_sim(roots)
  env.sim.forward()
  env.sim.sense()
  face_x = origins[:, 0] + geometry["outer_face_x"]
  metadata = {
    "x_relative_to_face_m": roots[:, 0] - face_x,
    "y_relative_to_center_m": roots[:, 1] - origins[:, 1],
    "root_height_m": roots[:, 2], "root_linear_velocity_mps": roots[:, 7:10],
    "root_angular_velocity_radps": roots[:, 10:13],
    "root_quaternion_wxyz": roots[:, 3:7],
    "leg_joint_position_rad": joint_pos[:, leg_ids],
    "leg_joint_velocity_radps": joint_vel[:, leg_ids],
  }
  return terrain_types, face_x, origins[:, 0] + geometry["cross_x"], metadata


def wheel_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  force = env.scene[sensor_name].data.force
  if force is None:
    raise RuntimeError(f"RollBoundary sensor {sensor_name} exposes no force field.")
  magnitude = torch.linalg.vector_norm(force.reshape(force.shape[0], -1, 3), dim=-1)
  return torch.any(magnitude > 0.0, dim=-1)


def bilateral_airborne(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
  if left.shape != right.shape:
    raise ValueError("Left/right contact tensors must have identical shape.")
  return ~left.bool() & ~right.bool()


def latch_before_reset(history: torch.Tensor, event: torch.Tensor,
                       was_active: torch.Tensor) -> torch.Tensor:
  return history.bool() | (event.bool() & was_active.bool())



def wheel_clearance_above_flat_m(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Exact cylinder support clearance above the z=0 flat top, shape [B, 2]."""
  robot = env.scene["robot"]
  local_ids, names = robot.find_geoms(
    ("wheel_left_collision", "wheel_right_collision"), preserve_order=True,
  )
  if tuple(names) != ("wheel_left_collision", "wheel_right_collision"):
    raise RuntimeError("RollBoundary wheel geom identity drifted.")
  center = robot.data.geom_pos_w[:, local_ids]
  quaternion = robot.data.geom_quat_w[:, local_ids]
  w, x, y, z = quaternion.unbind(dim=-1)
  del w, z
  # Cylinder symmetry axis is local z. R_zz is its world-z component.
  axis_z = 1.0 - 2.0 * (x.square() + y.square())
  radius = float(NOMINAL_WHEEL_RADIUS_M)
  half_length = 0.018
  vertical_extent = (
    radius * torch.sqrt(torch.clamp(1.0 - axis_z.square(), min=0.0))
    + half_length * axis_z.abs()
  )
  return center[:, :, 2] - vertical_extent


def install_strict_substep_support_recorder(
  env: ManagerBasedRlEnv,
) -> tuple[dict[str, Any], Any]:
  """Latch bilateral zero-force support at the unchanged 5 ms physics cadence."""
  if int(env.cfg.decimation) != ROLL_FIRST_CONTROL_DECIMATION or not math.isclose(
    float(env.physics_dt), ROLL_FIRST_PHYSICS_TIMESTEP_S,
    rel_tol=0.0, abs_tol=1.0e-12,
  ):
    raise ValueError("Strict RollBoundary recorder requires 4 x 5 ms physics substeps.")
  state = {
    "enabled": False,
    "active_mask": torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device,
    ),
    "bilateral_unsupported_ever": torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device,
    ),
    "bilateral_unsupported_substeps": torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device,
    ),
    "bilateral_positive_clearance_ever": torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device,
    ),
    "max_flat_clearance_m": torch.full(
      (env.num_envs, 2), -torch.inf, device=env.device,
    ),
    "max_actual_wheel_force_nm": torch.zeros(
      env.num_envs, device=env.device,
    ),
  }
  robot = env.scene["robot"]
  term = env.action_manager.get_term("hybrid_wheel_leg")
  original_update = env.scene.update

  def update(dt: float) -> None:
    original_update(dt)
    if not state["enabled"]:
      return
    left_force = torch.linalg.vector_norm(env.scene[LEFT_SENSOR].data.force, dim=-1)
    right_force = torch.linalg.vector_norm(env.scene[RIGHT_SENSOR].data.force, dim=-1)
    left = torch.any(left_force > 0.0, dim=-1)
    right = torch.any(right_force > 0.0, dim=-1)
    unsupported = state["active_mask"] & bilateral_airborne(left, right)
    state["bilateral_unsupported_ever"].logical_or_(unsupported)
    state["bilateral_unsupported_substeps"].add_(unsupported.long())
    clearance = wheel_clearance_above_flat_m(env)
    state["max_flat_clearance_m"].copy_(torch.where(
      state["active_mask"].unsqueeze(1),
      torch.maximum(state["max_flat_clearance_m"], clearance),
      state["max_flat_clearance_m"],
    ))
    state["bilateral_positive_clearance_ever"].logical_or_(
      unsupported & torch.all(clearance > 0.0, dim=-1)
    )
    # `_wheel_ids` are entity-local joint indices; this robot keeps actuator
    # ordering aligned with joint ordering (`sort_actuators=True`).
    actual = robot.data.actuator_force[:, term._wheel_ids]
    state["max_actual_wheel_force_nm"].copy_(torch.where(
      state["active_mask"],
      torch.maximum(
        state["max_actual_wheel_force_nm"], actual.abs().amax(dim=-1),
      ),
      state["max_actual_wheel_force_nm"],
    ))

  env.scene.update = update
  return state, original_update


def model_wheel_torque(target: torch.Tensor, actual: torch.Tensor):
  raw = WHEEL_VELOCITY_DAMPING * (target - actual)
  return (
    torch.clamp(raw, -RMD_L_9025_35T_PEAK_TORQUE, RMD_L_9025_35T_PEAK_TORQUE),
    raw.abs() >= RMD_L_9025_35T_PEAK_TORQUE,
  )


def _pitch_roll(robot: Any) -> tuple[torch.Tensor, torch.Tensor]:
  gravity = robot.data.projected_gravity_b
  denominator = torch.clamp(-gravity[:, 2], min=1.0e-6)
  return torch.atan2(gravity[:, 0], denominator), torch.atan2(-gravity[:, 1], denominator)


def _masked(values: torch.Tensor, valid: torch.Tensor, env_id: int) -> torch.Tensor:
  return values[:, env_id][valid[:, env_id]]


def _stat(values: torch.Tensor, kind: str) -> float | None:
  if values.numel() == 0:
    return None
  if kind == "mean":
    result = values.float().mean()
  elif kind == "max":
    result = values.float().max()
  elif kind == "p95":
    result = values.float().quantile(0.95)
  elif kind == "last":
    result = values.float().reshape(-1)[-1]
  else:
    raise ValueError(kind)
  return float(result.item())


def run_card_repeat(
  env: ManagerBasedRlEnv,
  *,
  heights: tuple[float, ...],
  card: Mapping[str, float | str],
  repeat: int,
  settle_steps: int,
  drive_steps: int,
  stable_steps: int,
  policy: Any | None = None,
  wheel_residual_exact_zero: bool = True,
  episode_wide_safety: bool = False,
  diagnostic_continue_after_support_loss: bool = False,
  roll_pose_schedule: RollPoseSchedule | None = None,
  require_pure_classical_authority: bool = False,
):
  heights = validate_heights(heights)
  if episode_wide_safety and not wheel_residual_exact_zero:
    raise ValueError("Episode-wide RollAssist safety requires exact-zero wheel residuals.")
  if require_pure_classical_authority and policy is not None:
    raise ValueError("Pure-classical RollBoundary diagnostics do not accept a policy.")
  terrain_types, face_x, cross_x, reset = _reset_to_approach(
    env, root_height=float(card["height_m"]), card_name=str(card["name"]),
    repeat=repeat, height_count=len(heights),
  )
  if int(terrain_types.max()) >= len(heights):
    raise RuntimeError("Terrain type index exceeds the RollBoundary height table.")
  robot = env.scene["robot"]
  term = env.action_manager.get_term("hybrid_wheel_leg")
  wheel_ids = term._wheel_ids
  leg_ids = (
    term._leg_ids
    if roll_pose_schedule is not None or require_pure_classical_authority
    else None
  )
  actions = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
  observation = env.get_observations()
  active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  terminated_ever = torch.zeros_like(active)
  non_wheel_ever = torch.zeros_like(active)
  left_ever, right_ever, airborne_ever = (torch.zeros_like(active) for _ in range(3))
  success = torch.zeros_like(active)
  support_failed = torch.zeros_like(active)
  success_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
  stable = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  left_steps, right_steps, air_steps = (torch.zeros_like(stable) for _ in range(3))
  left_run, right_run, left_max_run, right_max_run = (torch.zeros_like(stable) for _ in range(4))
  max_progress = reset["x_relative_to_face_m"].clone()
  peak_pitch = torch.zeros(env.num_envs, device=env.device)
  peak_roll, peak_pitch_rate = torch.zeros_like(peak_pitch), torch.zeros_like(peak_pitch)
  wheel_residual_max = torch.zeros_like(peak_pitch)
  applied_residual_max = torch.zeros_like(peak_pitch)
  schedule_state = (
    None if roll_pose_schedule is None else make_roll_pose_schedule_state(
      roll_pose_schedule, robot.data.root_link_pos_w[:, 0],
    )
  )
  schedule_alpha_max = torch.zeros_like(peak_pitch)
  schedule_height_lag_max = torch.zeros_like(peak_pitch)
  schedule_pitch_lag_max = torch.zeros_like(peak_pitch)
  schedule_completion_step = torch.full_like(success_step, -1)
  schedule_completed_before_face = torch.zeros_like(active)
  first_support_loss_progress = torch.full_like(peak_pitch, math.nan)
  schedule_desired_height = torch.full_like(peak_pitch, float(card["height_m"]))
  schedule_desired_pitch = torch.full_like(peak_pitch, float(card["pitch_rad"]))
  schedule_applied_height = schedule_desired_height.clone()
  schedule_applied_pitch = schedule_desired_pitch.clone()
  wheel_classical_path_delta_max = torch.zeros_like(peak_pitch)
  dynamic_leg_feedforward_max = torch.zeros_like(peak_pitch)
  dynamic_drive_feedforward_max = torch.zeros_like(peak_pitch)
  samples: dict[str, list[torch.Tensor]] = defaultdict(list)
  valid_samples: list[torch.Tensor] = []
  substep_support, original_scene_update = install_strict_substep_support_recorder(env)

  def step(vx: float, drive_index: int | None) -> None:
    nonlocal active, actions, observation
    was_active = active.clone()
    if roll_pose_schedule is not None and schedule_state is not None:
      schedule_output = roll_pose_schedule_step(
        roll_pose_schedule,
        schedule_state,
        root_x_m=robot.data.root_link_pos_w[:, 0],
        face_x_m=face_x,
        active_mask=was_active,
        drive_active=drive_index is not None,
        dt=ROLL_POSE_CONTROL_DT_S,
      )
      schedule_desired_height.copy_(schedule_output.desired_height_m)
      schedule_desired_pitch.copy_(schedule_output.desired_pitch_rad)
      schedule_applied_height.copy_(schedule_output.applied_height_m)
      schedule_applied_pitch.copy_(schedule_output.applied_pitch_rad)
      if drive_index is not None:
        schedule_alpha_max.copy_(torch.maximum(
          schedule_alpha_max,
          torch.where(was_active, schedule_output.alpha, 0.0),
        ))
        height_lag = (
          schedule_output.desired_height_m - schedule_output.applied_height_m
        ).abs()
        pitch_lag = (
          schedule_output.desired_pitch_rad - schedule_output.applied_pitch_rad
        ).abs()
        schedule_height_lag_max.copy_(torch.maximum(
          schedule_height_lag_max, torch.where(was_active, height_lag, 0.0),
        ))
        schedule_pitch_lag_max.copy_(torch.maximum(
          schedule_pitch_lag_max, torch.where(was_active, pitch_lag, 0.0),
        ))
        completed = (
          was_active
          & (schedule_output.alpha >= 1.0)
          & torch.isclose(
            schedule_output.applied_height_m,
            torch.full_like(schedule_output.applied_height_m, roll_pose_schedule.climb_height_m),
            rtol=0.0, atol=1.0e-7,
          )
          & torch.isclose(
            schedule_output.applied_pitch_rad,
            torch.full_like(schedule_output.applied_pitch_rad, roll_pose_schedule.climb_pitch_rad),
            rtol=0.0, atol=1.0e-7,
          )
        )
        newly_completed = completed & (schedule_completion_step < 0)
        schedule_completion_step[newly_completed] = drive_index + 1
        schedule_completed_before_face.logical_or_(
          newly_completed & (robot.data.root_link_pos_w[:, 0] <= face_x)
        )
      command_height: float | torch.Tensor = schedule_output.applied_height_m
      command_pitch: float | torch.Tensor = schedule_output.applied_pitch_rad
    else:
      command_height = float(card["height_m"])
      command_pitch = float(card["pitch_rad"])
    _force_commands(env, active=was_active, vx=vx,
                    height=command_height, pitch=command_pitch)
    previous_wheel_targets = (
      term._previous_wheel_targets.detach().clone()
      if require_pure_classical_authority else None
    )
    if policy is None:
      actions.zero_()
    else:
      observation = env.get_observations()
      with torch.inference_mode():
        candidate = policy(observation)
      if candidate.shape != actions.shape:
        raise RuntimeError("RollBoundary policy action shape drifted.")
      actions = candidate.detach()
    monitor_support = episode_wide_safety or drive_index is not None
    substep_support["enabled"] = monitor_support
    substep_support["active_mask"].copy_(
      was_active if monitor_support else torch.zeros_like(was_active)
    )
    observation, _reward, terminated, timeouts, _extras = env.step(actions)
    substep_support["enabled"] = False
    applied_residual = term.applied_residual.abs().amax(dim=1)
    applied_residual_max.copy_(torch.maximum(
      applied_residual_max, applied_residual,
    ))
    wheel_residual = term.applied_residual[:, :2].abs().amax(dim=1)
    wheel_residual_max.copy_(torch.maximum(wheel_residual_max, wheel_residual))
    if require_pure_classical_authority and previous_wheel_targets is not None:
      classical_delta = torch.clamp(
        term.controller_baseline - previous_wheel_targets,
        -term.cfg.wheel_slew_limit,
        term.cfg.wheel_slew_limit,
      )
      classical_target = torch.clamp(
        previous_wheel_targets + classical_delta,
        -term.cfg.wheel_velocity_limit,
        term.cfg.wheel_velocity_limit,
      )
      wheel_classical_path_delta_max.copy_(torch.maximum(
        wheel_classical_path_delta_max,
        (term.wheel_targets - classical_target).abs().amax(dim=1),
      ))
      dynamic_leg_feedforward_max.copy_(torch.maximum(
        dynamic_leg_feedforward_max,
        term.dynamic_leg_feedforward.abs().amax(dim=1),
      ))
      dynamic_drive_feedforward_max.copy_(torch.maximum(
        dynamic_drive_feedforward_max,
        term.dynamic_drive_feedforward.abs(),
      ))
      if float((term.wheel_targets - classical_target).abs().max().item()) != 0.0:
        raise RuntimeError("Roll-pose schedule changed the pure-classical wheel path.")
      if float(term.dynamic_leg_feedforward.abs().max().item()) != 0.0:
        raise RuntimeError("Roll-pose schedule enabled dynamic leg feedforward.")
      if float(term.dynamic_drive_feedforward.abs().max().item()) != 0.0:
        raise RuntimeError("Roll-pose schedule enabled dynamic drive feedforward.")
      if float(term.applied_residual.abs().max().item()) != 0.0:
        raise RuntimeError("Roll-pose schedule observed a nonzero applied residual.")
    wheel_max = float(wheel_residual.max().item())
    if wheel_residual_exact_zero and wheel_max != 0.0:
      raise RuntimeError("RollBoundary observed a nonzero wheel residual.")
    left, right = wheel_contact(env, LEFT_SENSOR), wheel_contact(env, RIGHT_SENSOR)
    airborne = bilateral_airborne(left, right)
    substep_airborne = (
      substep_support["bilateral_unsupported_ever"] & was_active
      if monitor_support else torch.zeros_like(was_active)
    )
    non_wheel = (
      non_wheel_ground_contact(env, NON_WHEEL_GROUND_SENSOR_NAME).bool()
      | env.termination_manager.get_term("non_wheel_ground_contact").bool()
    )
    done = terminated.bool() | timeouts.bool()
    pitch, roll = _pitch_roll(robot)
    pitch_rate = robot.data.root_link_ang_vel_b[:, 1]
    progress = robot.data.root_link_pos_w[:, 0] - face_x
    safety_active = (
      was_active
      if episode_wide_safety
      else was_active & (drive_index is not None)
    )
    terminated_ever.copy_(latch_before_reset(terminated_ever, done, safety_active))
    non_wheel_ever.copy_(latch_before_reset(non_wheel_ever, non_wheel, safety_active))
    left_ever.copy_(latch_before_reset(left_ever, left, was_active))
    right_ever.copy_(latch_before_reset(right_ever, right, was_active))
    airborne_ever.copy_(latch_before_reset(
      airborne_ever, airborne | substep_airborne, safety_active,
    ))
    if drive_index is not None:
      max_progress.copy_(torch.where(
        was_active, torch.maximum(max_progress, progress), max_progress
      ))
    peak_pitch.copy_(torch.maximum(peak_pitch, torch.where(was_active, pitch.abs(), 0.0)))
    peak_roll.copy_(torch.maximum(peak_roll, torch.where(was_active, roll.abs(), 0.0)))
    peak_pitch_rate.copy_(torch.maximum(
      peak_pitch_rate, torch.where(was_active, pitch_rate.abs(), 0.0)
    ))
    unsafe = (
      done | non_wheel | airborne | substep_airborne
      if episode_wide_safety or drive_index is not None
      else done | non_wheel
    )
    support_loss_now = was_active & (airborne | substep_airborne)
    first_support_loss = support_loss_now & ~support_failed
    first_support_loss_progress.copy_(torch.where(
      first_support_loss, progress, first_support_loss_progress,
    ))
    support_failed.logical_or_(support_loss_now)
    valid_now = was_active & ~unsafe & ~support_failed
    if drive_index is not None:
      left_unloaded, right_unloaded = was_active & ~left, was_active & ~right
      left_steps.add_(left_unloaded.long())
      right_steps.add_(right_unloaded.long())
      air_steps.add_((was_active & airborne).long())
      left_run.copy_(torch.where(left_unloaded, left_run + 1, 0))
      right_run.copy_(torch.where(right_unloaded, right_run + 1, 0))
      left_max_run.copy_(torch.maximum(left_max_run, left_run))
      right_max_run.copy_(torch.maximum(right_max_run, right_run))
      posture_ok = ((pitch.abs() <= PITCH_LIMIT_RAD) & (roll.abs() <= ROLL_LIMIT_RAD)
                    & (pitch_rate.abs() <= PITCH_RATE_LIMIT_RADPS))
      stable.copy_(torch.where(
        valid_now & (robot.data.root_link_pos_w[:, 0] >= cross_x) & posture_ok,
        stable + 1, torch.zeros_like(stable),
      ))
      newly = valid_now & (stable >= stable_steps) & ~success
      success_step[newly] = drive_index + 1
      success.logical_or_(newly)
      target = term.wheel_targets.detach()
      speed = robot.data.joint_vel[:, wheel_ids].detach()
      torque, saturated = model_wheel_torque(target, speed)
      body_vx = robot.data.root_link_lin_vel_b[:, 0].detach()
      wheel_linear = torch.stack(
        (-speed[:, 0] * NOMINAL_WHEEL_RADIUS_M, speed[:, 1] * NOMINAL_WHEEL_RADIUS_M), dim=1
      )
      samples["target_abs"].append(target.abs())
      samples["speed_abs"].append(speed.abs())
      samples["target_forward"].append(0.5 * (target[:, 1] - target[:, 0]))
      samples["speed_forward"].append(0.5 * (speed[:, 1] - speed[:, 0]))
      samples["torque_abs"].append(torque.abs())
      samples["saturated"].append(saturated)
      samples["slip"].append((wheel_linear - body_vx.unsqueeze(1)).abs())
      if roll_pose_schedule is not None:
        samples["schedule_alpha"].append(schedule_alpha_max.clone())
        samples["desired_height"].append(schedule_desired_height.clone())
        samples["desired_pitch"].append(schedule_desired_pitch.clone())
        samples["applied_height"].append(schedule_applied_height.clone())
        samples["applied_pitch"].append(schedule_applied_pitch.clone())
      if leg_ids is not None:
        samples["leg_target"].append(term.leg_targets.detach().clone())
        leg_position = robot.data.joint_pos[:, leg_ids].detach()
        samples["leg_position"].append(leg_position.clone())
        samples["leg_tracking_error"].append(
          (term.leg_targets.detach() - leg_position).abs()
        )
        samples["leg_force_abs"].append(
          robot.data.actuator_force[:, leg_ids].detach().abs().clone()
        )
      valid_samples.append(was_active.detach().clone())
    if diagnostic_continue_after_support_loss:
      # Diagnostic traces may continue after the first force-defined support
      # loss so the post-event recovery can be inspected. The failure remains
      # permanently latched and can never become a success. Formal callers
      # leave this disabled and retain the original fail-closed behavior.
      active &= ~(done | non_wheel) & ~success
    else:
      active &= ~unsafe & ~success
    # Manual-reset mode requires every done environment to be reset before the
    # next vector step, including trials that became inactive earlier.
    reset_event = (
      (was_active & (done | non_wheel)) | done
      if diagnostic_continue_after_support_loss
      else (was_active & unsafe) | done
    )
    reset_ids = torch.nonzero(reset_event, as_tuple=False).squeeze(-1)
    if reset_ids.numel() > 0:
      env.reset(env_ids=reset_ids)
      observation = env.get_observations()

  try:
    for _ in range(settle_steps):
      step(0.0, None)
    for drive_index in range(drive_steps):
      step(COMMAND_VX_MPS, drive_index)
  finally:
    substep_support["enabled"] = False
    substep_support["active_mask"].zero_()
    env.scene.update = original_scene_update
  airborne_ever.logical_or_(substep_support["bilateral_unsupported_ever"])
  air_steps.copy_(substep_support["bilateral_unsupported_substeps"])
  success.logical_and_(~substep_support["bilateral_unsupported_ever"])
  if not valid_samples:
    raise RuntimeError("RollBoundary drive produced no metric samples.")
  stacked = {name: torch.stack(values) for name, values in samples.items()}
  validity = torch.stack(valid_samples)
  rows = []
  for env_id, terrain_type in enumerate(terrain_types.cpu().tolist()):
    data = {name: _masked(value, validity, env_id) for name, value in stacked.items()}
    success_index = int(success_step[env_id]) if bool(success[env_id]) else -1
    row = {
      "posture_card": str(card["name"]), "target_height_m": float(card["height_m"]),
      "target_pitch_rad": float(card["pitch_rad"]), "stair_height_m": heights[terrain_type],
      "terrain_key": terrain_key(heights[terrain_type]), "terrain_index": int(terrain_type),
      "repeat": int(repeat), "env_id": env_id, "success": bool(success[env_id]),
      "time_to_success_s": None if success_index < 0 else success_index / CONTROL_FREQUENCY_HZ,
      "termination": bool(terminated_ever[env_id]),
      "non_wheel_contact": bool(non_wheel_ever[env_id]),
      "left_wheel_contact_ever": bool(left_ever[env_id]),
      "right_wheel_contact_ever": bool(right_ever[env_id]),
      "left_unload_steps": int(left_steps[env_id]), "right_unload_steps": int(right_steps[env_id]),
      "left_unload_max_consecutive_steps": int(left_max_run[env_id]),
      "right_unload_max_consecutive_steps": int(right_max_run[env_id]),
      "bilateral_airborne_steps": int(air_steps[env_id]),
      "bilateral_airborne_ever": bool(airborne_ever[env_id]),
      "bilateral_unsupported_physics_substeps": int(
        substep_support["bilateral_unsupported_substeps"][env_id]
      ),
      "bilateral_positive_clearance_ever": bool(
        substep_support["bilateral_positive_clearance_ever"][env_id]
      ),
      "max_flat_wheel_clearance_m": (
        [float(value) for value in
         substep_support["max_flat_clearance_m"][env_id].tolist()]
        if heights[terrain_type] == 0.0 else None
      ),
      "actual_wheel_actuator_force_abs_max_nm": float(
        substep_support["max_actual_wheel_force_nm"][env_id]
      ),
      "wheel_residual_abs_max": float(wheel_residual_max[env_id].item()),
      "peak_pitch_abs_rad": float(peak_pitch[env_id]), "peak_roll_abs_rad": float(peak_roll[env_id]),
      "peak_pitch_rate_abs_radps": float(peak_pitch_rate[env_id]),
      "wheel_target_radps_mean": _stat(data["target_abs"], "mean"),
      "wheel_target_radps_max": _stat(data["target_abs"], "max"),
      "wheel_speed_radps_mean": _stat(data["speed_abs"], "mean"),
      "wheel_speed_radps_max": _stat(data["speed_abs"], "max"),
      "wheel_target_forward_radps_mean": _stat(data["target_forward"], "mean"),
      "wheel_speed_forward_radps_mean": _stat(data["speed_forward"], "mean"),
      "model_torque_abs_nm_mean": _stat(data["torque_abs"], "mean"),
      "model_torque_abs_nm_p95": _stat(data["torque_abs"], "p95"),
      "torque_saturation_fraction": _stat(data["saturated"].float(), "mean"),
      "wheel_slip_mps_mean": _stat(data["slip"], "mean"),
      "wheel_slip_mps_p95": _stat(data["slip"], "p95"),
      "max_progress_past_face_m": float(max_progress[env_id]),
      "root_reset": {
        "x_relative_to_face_m": float(reset["x_relative_to_face_m"][env_id]),
        "y_relative_to_center_m": float(reset["y_relative_to_center_m"][env_id]),
        "root_height_m": float(reset["root_height_m"][env_id]),
        "root_linear_velocity_mps": [float(v) for v in reset["root_linear_velocity_mps"][env_id].tolist()],
        "root_angular_velocity_radps": [float(v) for v in reset["root_angular_velocity_radps"][env_id].tolist()],
        "root_quaternion_wxyz": [float(v) for v in reset["root_quaternion_wxyz"][env_id].tolist()],
        "leg_joint_position_rad": [float(v) for v in reset["leg_joint_position_rad"][env_id].tolist()],
        "leg_joint_velocity_radps": [float(v) for v in reset["leg_joint_velocity_radps"][env_id].tolist()],
      },
    }
    if require_pure_classical_authority:
      row.update({
        "applied_residual_abs_max": float(applied_residual_max[env_id]),
        "wheel_target_classical_path_abs_max_radps": float(
          wheel_classical_path_delta_max[env_id]
        ),
        "dynamic_leg_feedforward_abs_max_rad": float(
          dynamic_leg_feedforward_max[env_id]
        ),
        "dynamic_drive_feedforward_abs_max_radps": float(
          dynamic_drive_feedforward_max[env_id]
        ),
      })
    if leg_ids is not None:
      row.update({
        "leg_target_abs_max_rad": _stat(data["leg_target"].abs(), "max"),
        "leg_position_abs_max_rad": _stat(data["leg_position"].abs(), "max"),
        "leg_tracking_error_abs_max_rad": _stat(
          data["leg_tracking_error"], "max"
        ),
        "leg_tracking_error_abs_p95_rad": _stat(
          data["leg_tracking_error"], "p95"
        ),
        "leg_actuator_force_abs_max_nm": _stat(data["leg_force_abs"], "max"),
      })
    if roll_pose_schedule is not None and schedule_state is not None:
      support_loss_progress = float(first_support_loss_progress[env_id])
      row.update({
        "roll_pose_schedule": roll_pose_schedule.to_dict(),
        "drive_start_x_m": float(schedule_state.drive_start_x_m[env_id]),
        "end_distance_to_riser_m": roll_pose_schedule.end_distance_to_riser_m,
        "schedule_alpha_max": float(schedule_alpha_max[env_id]),
        "desired_height_m_final": _stat(data["desired_height"], "last"),
        "desired_pitch_rad_final": _stat(data["desired_pitch"], "last"),
        "applied_height_m_final": _stat(data["applied_height"], "last"),
        "applied_pitch_rad_final": _stat(data["applied_pitch"], "last"),
        "maximum_height_tracking_lag_m": float(schedule_height_lag_max[env_id]),
        "maximum_pitch_tracking_lag_rad": float(schedule_pitch_lag_max[env_id]),
        "transition_completion_step": (
          None if int(schedule_completion_step[env_id]) < 0
          else int(schedule_completion_step[env_id])
        ),
        "transition_completed_before_face": bool(
          schedule_completed_before_face[env_id]
        ),
        "first_support_loss_progress_m": (
          support_loss_progress if math.isfinite(support_loss_progress) else None
        ),
      })
    rows.append(row)
  return rows


def _cell_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
  if not rows:
    raise ValueError("A RollBoundary cell cannot be empty.")
  successes = sum(bool(row["success"]) for row in rows)
  terminations = sum(bool(row["termination"]) for row in rows)
  contacts = sum(bool(row["non_wheel_contact"]) for row in rows)
  airborne = sum(bool(row["bilateral_airborne_ever"]) for row in rows)
  return {
    "trials": len(rows), "successes": successes, "success_rate": successes / len(rows),
    "terminated_trials": terminations, "termination_rate": terminations / len(rows),
    "non_wheel_contact_trials": contacts, "non_wheel_contact_rate": contacts / len(rows),
    "bilateral_airborne_trials": airborne, "bilateral_airborne_rate": airborne / len(rows),
    "passed": (successes >= CELL_PASS_SUCCESSES and terminations == 0
               and contacts == 0 and airborne == 0),
  }


def aggregate_trials(trials: list[dict[str, Any]], *, heights: tuple[float, ...],
                     expected_repeats: int, expected_envs_per_height: int,
                     cards: tuple[Mapping[str, float | str], ...] = POSTURE_CARDS):
  canonical = validate_heights(heights)
  expected_total = len(cards) * expected_repeats * expected_envs_per_height * len(canonical)
  if len(trials) != expected_total:
    raise ValueError(
      f"RollBoundary received {len(trials)} total trials; expected {expected_total}."
    )
  groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
  repeats: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
  valid_cards = {str(card["name"]) for card in cards}
  valid_heights = set(canonical)
  for row in trials:
    key = str(row["posture_card"]), float(row["stair_height_m"])
    if key[0] not in valid_cards or key[1] not in valid_heights:
      raise ValueError(f"RollBoundary trial contains an unexpected cell: {key}.")
    groups[key].append(row)
    repeats[(key[0], key[1], int(row["repeat"]))].append(row)
  expected_ids = set(range(len(canonical) * expected_envs_per_height))
  for card in cards:
    name = str(card["name"])
    for repeat in range(1, expected_repeats + 1):
      ids = [
        int(row["env_id"])
        for height in canonical
        for row in repeats.get((name, height, repeat), [])
      ]
      if len(ids) != len(expected_ids) or set(ids) != expected_ids:
        raise ValueError("RollBoundary repeat env ids do not cover the vector batch.")
  cells, repeat_cells = [], []
  for card in cards:
    name = str(card["name"])
    for height in canonical:
      rows = groups.get((name, height), [])
      expected = expected_repeats * expected_envs_per_height
      if len(rows) != expected:
        raise ValueError(f"RollBoundary cell {(name, height)} has {len(rows)} trials; expected {expected}.")
      cells.append({"posture_card": name, "stair_height_m": height, **_cell_summary(rows)})
      for repeat in range(1, expected_repeats + 1):
        repeat_rows = repeats.get((name, height, repeat), [])
        if len(repeat_rows) != expected_envs_per_height:
          raise ValueError(f"RollBoundary repeat {(name, height, repeat)} has wrong trial count.")
        env_ids = [int(row["env_id"]) for row in repeat_rows]
        if len(env_ids) != len(set(env_ids)):
          raise ValueError("RollBoundary repeat contains duplicate env ids.")
        repeat_cells.append({
          "posture_card": name, "stair_height_m": height, "repeat": repeat,
          **_cell_summary(repeat_rows),
        })
  return cells, repeat_cells


def classify_results(cells: list[dict[str, Any]], *, heights: tuple[float, ...],
                     formal_cap_m: float = FORMAL_CAP_UM / 1_000_000.0,
                     cards: tuple[Mapping[str, float | str], ...] = POSTURE_CARDS):
  canonical = validate_heights(heights)
  if not math.isclose(canonical[0], 0.0, abs_tol=1.0e-12):
    raise ValueError("Formal RollBoundary classification requires flat first.")
  by_key = {(str(cell["posture_card"]), float(cell["stair_height_m"])): cell for cell in cells}
  expected = {(str(card["name"]), height) for card in cards for height in canonical}
  if set(by_key) != expected:
    raise ValueError("RollBoundary classification received incomplete cells.")
  common = [all(bool(by_key[(str(card["name"]), h)]["passed"]) for card in cards)
            for h in canonical]
  flat_valid = common[0]
  non_monotonic = False
  for flags in [common] + [[bool(by_key[(str(card["name"]), h)]["passed"]) for h in canonical]
                           for card in cards]:
    seen_failure = False
    for passed in flags:
      if not passed:
        seen_failure = True
      elif seen_failure:
        non_monotonic = True
  pass_index = 0
  for index, passed in enumerate(common):
    if not passed:
      break
    pass_index = index
  hpass = canonical[pass_index] if flat_valid else None
  hfail = None if all(common) else canonical[pass_index + 1]
  unsafe = False
  if hfail is not None:
    unsafe = any(
      int(by_key[(str(card["name"]), hfail)][field]) > 0
      for card in cards
      for field in ("terminated_trials", "non_wheel_contact_trials", "bilateral_airborne_trials")
    )
  if not flat_valid:
    classification = "INVALID_FLAT_CONTROL_STOP"
  elif non_monotonic:
    classification = "NON_MONOTONIC_STOP"
  elif hfail is not None and math.isclose(hpass or 0.0, 0.0, abs_tol=1.0e-12):
    classification = "NO_POSITIVE_CLASSICAL_CROLL"
  elif hfail is not None and unsafe:
    classification = "NEXT_HEIGHT_UNSAFE_STOP"
  elif hfail is not None:
    classification = "CLASSICAL_CROLL_BRACKETED"
  elif canonical[-1] >= formal_cap_m - 1.0e-12:
    classification = "CLASSICAL_CROLL_AT_LEAST_CAP"
  else:
    classification = "EXTEND_ROLL_BOUNDARY_SWEEP"
  return {
    "classification": classification, "flat_control_valid": flat_valid,
    "non_monotonic": non_monotonic, "max_common_passing_height_m": hpass,
    "first_non_common_height_m": hfail,
    "croll_bracket_m": None if hpass is None else [hpass, hfail],
    "next_height_unsafe": unsafe,
    "training_eligible": classification == "CLASSICAL_CROLL_BRACKETED",
    "common_height_results": [
      {"stair_height_m": h, "both_cards_passed": passed}
      for h, passed in zip(canonical, common, strict=True)
    ],
  }


def _git_sha(path: Path) -> str | None:
  result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=False,
                          capture_output=True, text=True)
  return result.stdout.strip() if result.returncode == 0 else None


def _runtime_metadata(device: str) -> dict[str, object]:
  gpu_name = driver = None
  if device.startswith("cuda") and torch.cuda.is_available():
    index_text = device.partition(":")[2]
    index = int(index_text) if index_text else torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(index)
    query = subprocess.run(
      ["nvidia-smi", f"--id={index}", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
      check=False, capture_output=True, text=True, timeout=10,
    )
    if query.returncode == 0 and query.stdout.strip():
      driver = query.stdout.splitlines()[0].strip()
  return {"device": device, "cuda_available": bool(torch.cuda.is_available()),
          "gpu_name": gpu_name, "driver_version": driver,
          "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda}


def build_payload(*, trials, cells, repeat_cells, verdict, action_cfg, protocol,
                  device: str, runtime_metadata: dict[str, object]) -> dict[str, Any]:
  smoke = not bool(protocol["evidence_eligible"])
  mjlab_root = Path(mjlab.__file__).resolve().parents[2]
  return {
    "schema_version": 1, "probe": PROBE_NAME, "evidence_eligible": not smoke,
    "promotion_eligible": False,
    "classification": "SMOKE_ONLY" if smoke else verdict["classification"],
    "training_eligible": False if smoke else verdict["training_eligible"],
    "max_common_passing_height_m": None if smoke else verdict["max_common_passing_height_m"],
    "first_non_common_height_m": None if smoke else verdict["first_non_common_height_m"],
    "croll_bracket_m": None if smoke else verdict["croll_bracket_m"],
    "task": TASK, "seed": SEED, "git_sha": _git_sha(REPOSITORY_PATH),
    "mjlab_git_sha": _git_sha(mjlab_root), "device": device, "runtime": runtime_metadata,
    "checkpoint": None, "checkpoint_file_sha256": None,
    "controller_schedule_hash": action_cfg.controller_gain_hash,
    "calibration_hash": action_cfg.calibration_hash,
    "yaw_calibration_hash": action_cfg.yaw_calibration_hash,
    "posture_map_hash": action_cfg.posture_map_hash,
    "posture_artifact_hash": action_cfg.posture_artifact_hash,
    "station_calibration_hash": action_cfg.station_calibration_hash,
    "artifact_files": {
      name: {"path": relative, "file_sha256": expected}
      for name, (relative, expected) in ARTIFACT_SPECS.items()
    },
    "action_mask": list(action_cfg.action_mask), "action_scales": list(action_cfg.action_scales),
    "protocol": {
      "terrain": ROLL_FIRST_TERRAIN_PROTOCOL, "terrain_key_unit": "integer_micrometre",
      "terrain_keys": [terrain_key(h) for h in protocol["heights_m"]],
      "terrain_size_m": list(TERRAIN_SIZE_M),
      "terrain_border_width_m": TERRAIN_BORDER_WIDTH_M, "step_width_m": STEP_WIDTH_M,
      "platform_width_m": PLATFORM_WIDTH_M, "heights_m": list(protocol["heights_m"]),
      "height_step_m": HEIGHT_STEP_UM / 1_000_000.0,
      "physics_timestep_s": ROLL_FIRST_PHYSICS_TIMESTEP_S,
      "control_frequency_hz": CONTROL_FREQUENCY_HZ,
      "control_decimation": ROLL_FIRST_CONTROL_DECIMATION,
      "formal_cap_m": float(protocol["formal_cap_m"]), "environment_seed": SEED,
      "envs_per_height": int(protocol["envs_per_height"]), "repeats": int(protocol["repeats"]),
      "settle_steps": int(protocol["settle_steps"]), "drive_steps": int(protocol["drive_steps"]),
      "stable_steps": int(protocol["stable_steps"]),
      "settle_duration_s": int(protocol["settle_steps"]) / CONTROL_FREQUENCY_HZ,
      "maximum_drive_duration_s": int(protocol["drive_steps"]) / CONTROL_FREQUENCY_HZ,
      "required_stable_duration_s": int(protocol["stable_steps"]) / CONTROL_FREQUENCY_HZ,
      "command_vx_mps": COMMAND_VX_MPS, "cell_pass_successes": CELL_PASS_SUCCESSES,
      "cell_trials": OFFICIAL_ENVS_PER_HEIGHT * OFFICIAL_REPEATS,
      "posture_cards": [dict(card) for card in POSTURE_CARDS],
      "policy_action": [0.0] * 6, "commanded_yaw_rate": 0.0,
      "dynamic_stair_fsm": False, "contact_trigger_control": False,
      "leg_feedforward": False, "drive_feedforward": False, "auto_reset": False,
      "wheel_contact_solref": list(ROLL_BOUNDARY_WHEEL_SOLREF),
      "wheel_contact_solimp": list(ROLL_BOUNDARY_WHEEL_SOLIMP),
      "strict_physics_substep_support_required": True,
      "strict_physics_substep_support_scope": ROLL_FIRST_SUBSTEP_SUPPORT_SCOPE,
      "stability_limits": {
        "pitch_abs_rad": PITCH_LIMIT_RAD, "roll_abs_rad": ROLL_LIMIT_RAD,
        "pitch_rate_abs_radps": PITCH_RATE_LIMIT_RADPS,
      },
      "safety": {
        "termination_trials_required": 0, "non_wheel_contact_trials_required": 0,
        "bilateral_airborne_trials_required": 0,
        "terminal_state_latched_before_reset": True,
      },
      "root_reset": {
        "reference": "first_riser_outer_face", "start_offset_outside_m": START_OFFSET_M,
        "success_line_inside_m": CROSS_DEPTH_M, "x_jitter_abs_m": RESET_X_JITTER_M,
        "y_jitter_abs_m": RESET_Y_JITTER_M, "vx_jitter_abs_mps": RESET_VX_JITTER_MPS,
        "pitch_rate_jitter_abs_radps": RESET_PITCH_RATE_JITTER_RADPS,
        "joint_state": ROLL_FIRST_RESET_JOINT_STATE,
        "orientation": ROLL_FIRST_RESET_ORIENTATION,
      },
      "wheel_model": {
        "radius_m": NOMINAL_WHEEL_RADIUS_M, "peak_torque_nm": RMD_L_9025_35T_PEAK_TORQUE,
        "velocity_damping": WHEEL_VELOCITY_DAMPING,
        "torque_is_model_value_not_sensor": True,
        "contact_backend_caveat": "MJWarp cylinder-box multicontact unavailable; strict substep gate retained",
      },
    },
    "cells": cells, "repeat_cells": repeat_cells, "trials": trials, "verdict": verdict,
  }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
  if path.exists():
    raise FileExistsError(f"Refusing to overwrite RollBoundary output: {path}")
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.incomplete")
  if temporary.exists():
    raise FileExistsError(f"Stale RollBoundary temporary output: {temporary}")
  try:
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)
  finally:
    if temporary.exists():
      temporary.unlink()


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  protocol = protocol_for_mode(args.smoke, args.max_height_mm)
  heights = validate_heights(protocol["heights_m"])
  cfg = make_roll_boundary_env_cfg(heights, int(protocol["envs_per_height"]))
  for line in hybrid_provenance_lines(cfg):
    print(line)
  action_cfg = cfg.actions["hybrid_wheel_leg"]
  trials = []
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    for card in POSTURE_CARDS:
      for repeat in range(1, int(protocol["repeats"]) + 1):
        print(f"[roll-boundary] card={card['name']} repeat={repeat}")
        trials.extend(run_card_repeat(
          env, heights=heights, card=card, repeat=repeat,
          settle_steps=int(protocol["settle_steps"]), drive_steps=int(protocol["drive_steps"]),
          stable_steps=int(protocol["stable_steps"]),
          episode_wide_safety=True,
        ))
  finally:
    env.close()
  cells, repeat_cells = aggregate_trials(
    trials, heights=heights, expected_repeats=int(protocol["repeats"]),
    expected_envs_per_height=int(protocol["envs_per_height"]),
  )
  verdict = (classify_results(cells, heights=heights,
                              formal_cap_m=float(protocol["formal_cap_m"]))
             if protocol["evidence_eligible"] else None)
  payload = build_payload(
    trials=trials, cells=cells, repeat_cells=repeat_cells, verdict=verdict,
    action_cfg=action_cfg, protocol=protocol, device=args.device,
    runtime_metadata=_runtime_metadata(args.device),
  )
  _atomic_write_json(args.output, payload)
  print(f"[roll-boundary] output={args.output}")
  print(f"[roll-boundary] classification={payload['classification']}")


if __name__ == "__main__":
  main()
