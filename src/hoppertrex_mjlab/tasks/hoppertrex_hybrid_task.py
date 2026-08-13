"""Hybrid v2 controller-residual tasks for the two-leg HopperTrex robot."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from assets.HopperTrex_CFG import INIT_JOINT_POS
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import (
  ActionTerm,
  ActionTermCfg,
  CommandTerm,
  CommandTermCfg,
  CurriculumTermCfg,
  EventTermCfg,
  MetricsTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  SceneEntityCfg,
  TerminationTermCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.config import flat, pyramid_stairs
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from hoppertrex_mjlab.hybrid.calibration import (
  VelocityCalibration,
  parse_calibration_artifact,
)
from hoppertrex_mjlab.hybrid.config import (
  HYBRID_ACTION_NAMES,
  HYBRID_STAGES,
  STAIR_CAMP_ACTION_MASK,
  STAIR_CAMP_FAILURE_LADDER_VARIANT,
  STAIR_CAMP_FAILURE_LQR_GAIN_SCALE,
  STAIR_CAMP_LEG_RESIDUAL_SCALE,
  STAIR_CAMP_LQR_ALPHA05_TASK_ID,
  STAIR_CAMP_TASK_ID,
  action_scales_with_leg_authority,
)
from hoppertrex_mjlab.hybrid.controller_schedule import (
  SCHEDULE_ARTIFACT_TYPE,
  ControllerSchedule,
  parse_controller_schedule,
)
from hoppertrex_mjlab.hybrid.identification import (
  CONTROLLER_STATE_NAMES,
  NOMINAL_WHEEL_RADIUS_M,
  STATE_DEFINITION_VERSION,
)
from hoppertrex_mjlab.hybrid.mismatch import (
  STAGE1_MISMATCH_FRACTION,
  STAGE1_MISMATCH_PROFILE_VERSION,
  apply_stage1_mild_mismatch,
)
from hoppertrex_mjlab.hybrid.posture import (
  LEG_JOINT_NAMES,
  POSTURE_ENVELOPE_VERIFICATION_METHODS,
  POSTURE_FEATURE_NAMES,
  posture_artifact_hash,
)
from hoppertrex_mjlab.hybrid.roll_assist import (
  ROLL_ASSIST_ACTION_MASK,
  ROLL_ASSIST_ACTION_SCALES,
  ROLL_ASSIST_ACTOR_TERMS,
  ROLL_ASSIST_ACTOR_WIDTH,
  ROLL_ASSIST_COMMAND_VX_MPS,
  ROLL_ASSIST_CONTROLLER_SCHEDULE_HASH,
  ROLL_ASSIST_CRITIC_TAIL,
  ROLL_ASSIST_CRITIC_WIDTH,
  ROLL_ASSIST_FLAT_ENVS,
  ROLL_ASSIST_HEIGHT_STEP_M,
  ROLL_ASSIST_NUM_ENVS,
  ROLL_ASSIST_SETTLE_STEPS,
  ROLL_ASSIST_STAIR_POSTURE_HEIGHT_M,
  ROLL_ASSIST_STAIR_POSTURE_PITCH_RAD,
  ROLL_ASSIST_STEPS_PER_UPDATE,
  ROLL_ASSIST_SWITCH_UPDATE,
  ROLL_ASSIST_TASK_ID,
  ROLL_ASSIST_TERM_WIDTHS,
  ROLL_FIRST_CONTROL_DECIMATION,
  ROLL_FIRST_PHYSICS_TIMESTEP_S,
  ROLL_FIRST_WHEEL_CONTACT_SOLIMP,
  ROLL_FIRST_WHEEL_CONTACT_SOLREF,
  RollAssistCurriculumState,
  file_sha256,
  load_roll_boundary_verdict,
  roll_first_artifact_paths,
  validate_reward_calibration,
)
from hoppertrex_mjlab.hybrid.stair_camp_contract import (
  STAIR_CAMP_ACTOR_WIDTH,
  STAIR_CAMP_CONTRACT_SCHEMA_VERSION,
  STAIR_CAMP_CRITIC_WIDTH,
  STAIR_CAMP_EXPECTED_ACTOR_TERMS,
  STAIR_CAMP_EXPECTED_CRITIC_TAIL,
  STAIR_CAMP_EXPECTED_TERM_WIDTHS,
  STAIR_CAMP_WITHDRAWN_CRITIC_TERMS,
)
from hoppertrex_mjlab.hybrid.stair_classical import PHASE_COUNT, StairPhase
from hoppertrex_mjlab.hybrid.stair_dynamic import (
  DYNAMIC_STAIR_CEM_LOWER,
  DYNAMIC_STAIR_CEM_UPPER,
  DYNAMIC_STAIR_FEEDFORWARD_LIMIT_RAD,
  DYNAMIC_STAIR_TASK_ID,
  DYNAMIC_STAIR_TIME_EPS_S,
  DynamicLiftMode,
  DynamicStairManeuver,
  LeadSide,
  load_dynamic_maneuver,
)
from hoppertrex_mjlab.hybrid.stair_dynamic_contract import (
  DYNAMIC_STAIR_ACTION_MASK,
  DYNAMIC_STAIR_ACTION_SCALES,
  DYNAMIC_STAIR_ACTOR_TERMS,
  DYNAMIC_STAIR_CRITIC_TAIL_TERMS,
  DYNAMIC_STAIR_FLAT_ENVS,
  DYNAMIC_STAIR_NUM_ENVS,
  validate_dynamic_stair_observation_layout,
)
from hoppertrex_mjlab.hybrid.stair_residual import (
  StairCurriculumState,
  update_stair_curriculum,
)
from hoppertrex_mjlab.hybrid.stair_trigger import (
  STAIR_TRIGGER_FORCE_N,
  STAIR_TRIGGER_SENSOR_FIELDS,
  STAIR_TRIGGER_SENSOR_NAME,
  STAIR_TRIGGER_SLOTS_PER_WHEEL,
  STAIR_TRIGGER_WINDOW,
  stair_trigger_metric,
  update_stair_trigger,
)
from hoppertrex_mjlab.hybrid.station_calibration import (
  parse_station_calibration_artifact,
  validate_station_breakpoints,
)
from hoppertrex_mjlab.hybrid.yaw_calibration import (
  parse_yaw_calibration_artifact,
  validate_yaw_breakpoints,
)
from hoppertrex_mjlab.tasks.hoppertrex_balance_task import (
  ROOT_HEIGHT_TARGET,
  WHEEL_GROUND_GEOMS,
  joint_pos_rel_without_wheel_position,
  make_hoppertrex_balance_env_cfg,
)

REPOSITORY_PATH = Path(__file__).resolve().parents[3]
WHEEL_JOINT_NAMES = ("wheel_left", "wheel_right")
ROLL_ASSIST_VERDICT_PATH_ENV = "HOPPERTREX_ROLL_ASSIST_R0_PATH"
ROLL_ASSIST_REWARD_CALIBRATION_PATH_ENV = "HOPPERTREX_ROLL_ASSIST_REWARD_CALIBRATION_PATH"
ROLL_ASSIST_LEFT_SENSOR_NAME = "roll_assist_left_wheel_contact"
ROLL_ASSIST_RIGHT_SENSOR_NAME = "roll_assist_right_wheel_contact"
ROLL_ASSIST_SENSOR_NAMES = (ROLL_ASSIST_LEFT_SENSOR_NAME, ROLL_ASSIST_RIGHT_SENSOR_NAME)
ROLL_ASSIST_SENSOR_FIELDS = ("found", "force", "normal")
ROLL_ASSIST_SENSOR_SLOTS = 8
ROLL_ASSIST_TERRAIN_SIZE_M = (8.0, 8.0)
ROLL_ASSIST_TERRAIN_BORDER_WIDTH_M = 1.0
ROLL_ASSIST_STEP_WIDTH_M = 0.30
ROLL_ASSIST_PLATFORM_WIDTH_M = 3.0
ROLL_ASSIST_TERRAIN_ROWS = 2
ROLL_ASSIST_EPISODE_LENGTH_S = 12.0
ROLL_ASSIST_START_OFFSET_M = 0.25
ROLL_ASSIST_CROSS_DEPTH_M = 0.15
ROLL_ASSIST_RISER_OFFSET_M = 0.5 * (
  ROLL_ASSIST_TERRAIN_SIZE_M[0] - 2.0 * ROLL_ASSIST_TERRAIN_BORDER_WIDTH_M
)
ROLL_ASSIST_PITCH_LIMIT_RAD = 0.10
ROLL_ASSIST_ROLL_LIMIT_RAD = 0.10
ROLL_ASSIST_PITCH_RATE_LIMIT_RADPS = 0.5
ROLL_ASSIST_STABLE_STEPS = 25
DYNAMIC_STAIR_MANEUVER_PATH_ENV = "HOPPERTREX_DYNAMIC_STAIR_MANEUVER_PATH"
DYNAMIC_STAIR_LEFT_SENSOR_NAME = "stair_dynamic_left_contact"
DYNAMIC_STAIR_RIGHT_SENSOR_NAME = "stair_dynamic_right_contact"
DYNAMIC_STAIR_SENSOR_NAMES = (
  DYNAMIC_STAIR_LEFT_SENSOR_NAME,
  DYNAMIC_STAIR_RIGHT_SENSOR_NAME,
)
DYNAMIC_STAIR_SENSOR_FIELDS = ("found", "force", "normal")
DYNAMIC_STAIR_SENSOR_SLOTS = 8
DYNAMIC_STAIR_TERRAIN_SIZE_M = (8.0, 8.0)
DYNAMIC_STAIR_TERRAIN_BORDER_WIDTH_M = 1.0
DYNAMIC_STAIR_STEP_WIDTH_M = 0.30
DYNAMIC_STAIR_PLATFORM_WIDTH_M = 3.0
DYNAMIC_STAIR_HEIGHT_STEP_M = 0.01
DYNAMIC_STAIR_MAX_HEIGHT_M = 0.03
DYNAMIC_STAIR_TERRAIN_ROWS = 4
DYNAMIC_STAIR_EPISODE_LENGTH_S = 20.0
DYNAMIC_STAIR_EVALUATION_INTERVAL_ITERS = 50
DYNAMIC_STAIR_PROGRESS_WEIGHT = 320.0
DYNAMIC_STAIR_RISER_EVENT_BONUS = 24.0
DYNAMIC_STAIR_COMMAND_VX_MPS = 0.07
DYNAMIC_STAIR_RISER_OFFSET_M = 0.5 * (
  DYNAMIC_STAIR_TERRAIN_SIZE_M[0]
  - 2.0 * DYNAMIC_STAIR_TERRAIN_BORDER_WIDTH_M
)
HYBRID_TASK_IDS = tuple(
  f"HopperTrex-Hybrid-v2-Stage{stage}" for stage in range(6)
)
HOPPERTREX_HYBRID_TASK_IDS = HYBRID_TASK_IDS

DEFAULT_PD_GAIN = (8.0, 1.0, 3.0, 0.2)
DEFAULT_WHEEL_RADIUS = NOMINAL_WHEEL_RADIUS_M
DEFAULT_WHEEL_VELOCITY_LIMIT = 12.0
DEFAULT_WHEEL_SLEW_LIMIT = 6.0
STAGE1_ACTIVE_LIN_VEL_X_ABS_RANGE = (0.03, 0.07)
STAGE1_EXTENSION_LIN_VEL_X_ABS_RANGE = (0.07, 0.10)
STAGE1_STANDING_ENVS = 0.20
STAGE1_EXTENSION_ENVS = 0.20
STAGE1_COMMAND_RESAMPLING_TIME_RANGE = (2.0, 4.0)
STAGE1_TRACK_LIN_VEL_STD = 0.02
STAGE1_GLOBAL_RESIDUAL_L2_WEIGHT = -0.02
STAGE1_HEALTHY_RESIDUAL_L2_WEIGHT = -0.25
STAGE1_HEALTHY_VELOCITY_ERROR = 0.015
STAGE1_HEALTHY_PITCH_ABS = math.radians(2.0)
STAGE1_HEALTHY_PITCH_RATE_ABS = 0.15
HYBRID_PLANAR_COMMAND_RESAMPLING_TIME_RANGE = (3.0, 6.0)
HYBRID_PLANAR_STANDING_ENVS = 0.10
HYBRID_PLANAR_LINEAR_ONLY_ENVS = 0.25
HYBRID_PLANAR_YAW_ONLY_ENVS = 0.25
HYBRID_PLANAR_LIN_VEL_X_ABS_RANGE = (0.035, 0.07)
HYBRID_PLANAR_YAW_RATE_ABS_RANGE = (0.05, 0.10)
STAGE2_LINEAR_RETENTION_ENVS = 0.10
STAGE2_LINEAR_RETENTION_ABS_RANGE = STAGE1_EXTENSION_LIN_VEL_X_ABS_RANGE
HYBRID_RESIDUAL_L2_WEIGHT = -0.10
# Probe measurement (2026-07-15, qualified LQR): the legacy 0.20 std against
# ±0.10 yaw commands put the reward equilibrium at ~55% of the commanded rate
# once the residual tax was paid. With the feedforward owning the nominal
# differential, 0.08 keeps the tracking gradient alive across the full band.
HYBRID_TRACK_ANG_VEL_STD = 0.08
CONTROLLER_PATH_ENV = "HOPPERTREX_HYBRID_CONTROLLER_PATH"
POSTURE_MAP_PATH_ENV = "HOPPERTREX_HYBRID_POSTURE_MAP_PATH"
CALIBRATION_PATH_ENV = "HOPPERTREX_HYBRID_CALIBRATION_PATH"
YAW_CALIBRATION_PATH_ENV = "HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH"
# With no artifact the feedforward is identically zero, matching pre-Stage-2.0
# behavior; the map must pin (0, 0), so Stage0/1 zero-yaw commands are
# byte-identical with or without an artifact.
YAW_FEEDFORWARD_FALLBACK_BREAKPOINTS = (
  (-1.0, 0.0),
  (0.0, 0.0),
  (1.0, 0.0),
)
STATION_CALIBRATION_PATH_ENV = "HOPPERTREX_HYBRID_STATION_CALIBRATION_PATH"
LEG_RESIDUAL_SCALE_ENV = "HOPPERTREX_HYBRID_LEG_RESIDUAL_SCALE"
# With no artifact the station compensation is identically zero. The artifact
# is only ever loaded for stages that command postures (stage_cfg
# .posture_commands with a qualified posture map), so stages 0-2 are
# byte-identical with or without the environment variable set.
STATION_DRIFT_FALLBACK_BREAKPOINTS = (
  (-1.0, 0.0),
  (1.0, 0.0),
)
# Stage 3.0 reference shaping (probe 2247602, 2026-07-16): STEP posture
# commands excited |vx| surges up to 0.996 m/s (10x the +-0.10 command
# domain), pitch-rate spikes of 6.09 rad/s (26x the steady floor), and
# pitch overshoot of 79% of the envelope width. The published posture
# command is therefore rate-limited toward the raw target. The machine-room
# matrix (b1edcb6 runs, full-span traverse {0.5, 1.0, 2.0} s) selected the
# 2.0 s tier by the pre-registered rule: worst |vx| 0.055 <= 0.15,
# pitch-rate 0.413 <= 1.0, pitch overshoot 0.0189 <= 0.02, settling
# 1.742 <= 2.5 s, zero terminations (the 1.0 s tier missed on overshoot
# 0.0209).
POSTURE_HEIGHT_SLEW_RATE = 0.01215
POSTURE_PITCH_SLEW_RATE = 0.07755


@dataclass(frozen=True)
class _ControllerArtifact:
  gain: tuple[float, float, float, float]
  controller_type: str
  qualified: bool
  source: str
  gain_hash: str | None
  schedule: ControllerSchedule | None = None


@dataclass(frozen=True)
class _PostureArtifact:
  coefficients: tuple[tuple[float, float, float, float], ...]
  height_range: tuple[float, float]
  pitch_range: tuple[float, float]
  qualified: bool
  source: str
  map_hash: str | None
  artifact_hash: str | None
  controller_gain_hash: str | None
  calibration_hash: str | None


def _artifact_path(
  explicit_path: Path | None,
  environment_variable: str,
) -> Path | None:
  if explicit_path is not None:
    return Path(explicit_path).expanduser().resolve()
  value = os.environ.get(environment_variable)
  if not value:
    return None
  return Path(value).expanduser().resolve()


def _read_json_object(path: Path, artifact_name: str) -> dict[str, object]:
  if not path.is_file():
    raise FileNotFoundError(f"{artifact_name} artifact does not exist: {path}")
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise ValueError(f"{artifact_name} artifact must contain a JSON object.")
  return payload


def _stable_hash(payload: dict[str, object]) -> str:
  encoded = json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
  ).encode("ascii")
  return hashlib.sha256(encoded).hexdigest()


_STATE_CONSTRUCTION_WARNED: set[str] = set()
_YAW_CALIBRATION_WARNED: set[int] = set()
_STATION_CALIBRATION_WARNED: set[int] = set()


def _warn_missing_state_construction(path: Path) -> None:
  key = str(path)
  if key in _STATE_CONSTRUCTION_WARNED:
    return
  _STATE_CONSTRUCTION_WARNED.add(key)
  print(
    "[hybrid][WARN] Controller artifact has no state_construction block: "
    f"{path}. Assuming wheel_radius={DEFAULT_WHEEL_RADIUS} and state "
    f"definition {STATE_DEFINITION_VERSION!r}. Regenerate the artifact with "
    "identify_hybrid_controller.py to record state provenance; this becomes "
    "a hard requirement when Stage0 artifacts are next regenerated."
  )


def _validate_state_construction(payload: dict[str, object], path: Path) -> None:
  """Reject a gain identified against a different state construction."""

  state_construction = payload.get("state_construction")
  if state_construction is None:
    _warn_missing_state_construction(path)
    return
  if not isinstance(state_construction, dict):
    raise ValueError("Controller state_construction must be a JSON object.")
  version = state_construction.get("state_definition_version")
  if version != STATE_DEFINITION_VERSION:
    raise ValueError(
      f"Controller artifact state definition {version!r} does not match the "
      f"runtime state definition {STATE_DEFINITION_VERSION!r}."
    )
  radius = state_construction.get("wheel_radius")
  if (
    not isinstance(radius, (int, float))
    or isinstance(radius, bool)
    or not math.isfinite(float(radius))
    or abs(float(radius) - DEFAULT_WHEEL_RADIUS) > 1.0e-9
  ):
    raise ValueError(
      f"Controller artifact wheel_radius {radius!r} does not match the "
      f"runtime wheel radius {DEFAULT_WHEEL_RADIUS}."
    )


def _load_controller(path: Path | None) -> _ControllerArtifact:
  if path is None:
    return _ControllerArtifact(
      gain=DEFAULT_PD_GAIN,
      controller_type="pd",
      qualified=False,
      source="local-unqualified-pd-fallback",
      gain_hash=None,
    )

  payload = _read_json_object(path, "Controller")
  if payload.get("artifact_type") == SCHEDULE_ARTIFACT_TYPE:
    schedule = parse_controller_schedule(payload, source=str(path))
    center_gain = schedule.gains[1, 1]
    return _ControllerArtifact(
      gain=tuple(float(value) for value in center_gain),
      controller_type="lqr",
      qualified=True,
      source=str(path),
      gain_hash=schedule.schedule_hash,
      schedule=schedule,
    )
  if payload.get("schema_version") != 1:
    raise ValueError("Controller schema_version must be 1.")
  if tuple(payload.get("state_names", ())) != CONTROLLER_STATE_NAMES:
    raise ValueError(
      "Controller state_names must match the Hybrid v2 controller state order."
    )
  gain_array = np.asarray(payload.get("gain"), dtype=np.float64)
  if gain_array.shape == (1, 4):
    gain_array = gain_array[0]
  if gain_array.shape != (4,) or not np.all(np.isfinite(gain_array)):
    raise ValueError("Controller gain must contain four finite values.")

  controller_type = str(payload.get("controller_type", "")).lower()
  nrmse = payload.get("heldout_one_step_nrmse")
  maximum_nrmse = (
    float(nrmse.get("maximum", math.inf))
    if isinstance(nrmse, dict)
    else math.inf
  )
  fallback_reasons = payload.get("fallback_reasons", ())
  qualified = (
    controller_type == "lqr"
    and int(payload.get("controllability_rank", -1)) == 4
    and maximum_nrmse <= 0.15
    and isinstance(fallback_reasons, list)
    and not fallback_reasons
  )
  if controller_type not in ("lqr", "pd"):
    raise ValueError("Controller artifact must label its type as 'lqr' or 'pd'.")
  gain_hash = payload.get("gain_hash")
  expected_gain_hash = _stable_hash(
    {
      "controller_type": controller_type,
      "state_names": CONTROLLER_STATE_NAMES,
      "gain": [gain_array.tolist()],
    }
  )
  if gain_hash != expected_gain_hash:
    raise ValueError("Controller gain_hash does not match its controller data.")
  _validate_state_construction(payload, path)
  if controller_type == "lqr" and not qualified:
    raise ValueError(
      "LQR controller artifact does not meet controllability and NRMSE qualification."
    )
  return _ControllerArtifact(
    gain=tuple(float(value) for value in gain_array),
    controller_type=controller_type,
    qualified=qualified,
    source=str(path),
    gain_hash=str(gain_hash),
  )


def _default_posture_artifact() -> _PostureArtifact:
  initial = tuple(float(INIT_JOINT_POS[name]) for name in LEG_JOINT_NAMES)
  return _PostureArtifact(
    coefficients=(
      initial,
      (0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0),
    ),
    height_range=(ROOT_HEIGHT_TARGET, ROOT_HEIGHT_TARGET),
    pitch_range=(0.0, 0.0),
    qualified=False,
    source="local-unqualified-initial-posture",
    map_hash=None,
    artifact_hash=None,
    controller_gain_hash=None,
    calibration_hash=None,
  )


def _load_posture_map(path: Path | None) -> _PostureArtifact:
  if path is None:
    return _default_posture_artifact()

  payload = _read_json_object(path, "Posture map")
  if payload.get("schema_version") != 1:
    raise ValueError("Posture map schema_version must be 1.")
  if tuple(payload.get("joint_names", ())) != LEG_JOINT_NAMES:
    raise ValueError("Posture map joint_names must match the four leg joints.")
  if tuple(payload.get("feature_names", ())) != POSTURE_FEATURE_NAMES:
    raise ValueError("Posture map feature_names must be bias, height, pitch.")
  coefficients = np.asarray(payload.get("coefficients"), dtype=np.float64)
  if coefficients.shape != (3, 4) or not np.all(np.isfinite(coefficients)):
    raise ValueError("Posture map coefficients must have finite shape (3, 4).")
  envelope = payload.get("training_envelope")
  if not isinstance(envelope, dict):
    raise ValueError("Posture map must contain a training_envelope object.")
  height_range = tuple(float(value) for value in envelope.get("height", ()))
  pitch_range = tuple(float(value) for value in envelope.get("pitch", ()))
  if len(height_range) != 2 or height_range[0] > height_range[1]:
    raise ValueError("Posture height range must contain two ordered values.")
  if (
    len(pitch_range) != 2
    or pitch_range[0] > pitch_range[1]
    or max(abs(value) for value in pitch_range) > 0.08 + 1.0e-12
  ):
    raise ValueError("Posture pitch range must be ordered and stay within 0.08 rad.")
  verification = payload.get("envelope_verification")
  if not isinstance(verification, dict):
    raise ValueError("Posture map must document a verified grid rectangle.")
  grid_shape = verification.get("grid_shape")
  if (
    verification.get("method") not in POSTURE_ENVELOPE_VERIFICATION_METHODS
    or not isinstance(grid_shape, list)
    or len(grid_shape) != 2
    or any(not isinstance(value, int) or value < 2 for value in grid_shape)
  ):
    raise ValueError("Posture map must document a verified grid rectangle.")
  map_hash = payload.get("map_hash")
  expected_map_hash = _stable_hash(
    {
      "feature_names": POSTURE_FEATURE_NAMES,
      "joint_names": LEG_JOINT_NAMES,
      "coefficients": coefficients.tolist(),
    }
  )
  if map_hash != expected_map_hash:
    raise ValueError("Posture map_hash does not match its posture data.")
  artifact_hash = payload.get("posture_artifact_hash")
  if artifact_hash is not None:
    if artifact_hash != posture_artifact_hash(payload):
      raise ValueError(
        "Posture artifact hash does not match its envelope and fit data."
      )
  source_sweep = payload.get('source_sweep')
  controller_gain_hash = (
    source_sweep.get('controller_gain_hash')
    if isinstance(source_sweep, dict)
    else None
  )
  if not isinstance(controller_gain_hash, str) or not controller_gain_hash:
    raise ValueError(
      'Posture map must record its source controller gain hash.'
    )
  calibration_hash = (
    source_sweep.get('calibration_hash')
    if isinstance(source_sweep, dict)
    else None
  )
  if not isinstance(calibration_hash, str) or not calibration_hash:
    raise ValueError('Posture map must record its source calibration hash.')
  return _PostureArtifact(
    coefficients=tuple(
      tuple(float(value) for value in row) for row in coefficients
    ),
    height_range=(height_range[0], height_range[1]),
    pitch_range=(pitch_range[0], pitch_range[1]),
    qualified=True,
    source=str(path),
    map_hash=str(map_hash),
    artifact_hash=(str(artifact_hash) if artifact_hash is not None else None),
    controller_gain_hash=controller_gain_hash,
    calibration_hash=calibration_hash,
  )


@dataclass(frozen=True)
class _YawCalibrationArtifact:
  breakpoints: tuple[tuple[float, float], ...]
  qualified: bool
  source: str
  yaw_calibration_hash: str | None


def _load_yaw_calibration(
  path: Path | None,
  controller_gain_hash: str,
) -> _YawCalibrationArtifact:
  if path is None:
    return _YawCalibrationArtifact(
      breakpoints=YAW_FEEDFORWARD_FALLBACK_BREAKPOINTS,
      qualified=False,
      source="local-zero-feedforward-fallback",
      yaw_calibration_hash=None,
    )
  parsed = parse_yaw_calibration_artifact(
    _read_json_object(path, "Yaw calibration"),
    controller_gain_hash=controller_gain_hash,
  )
  return _YawCalibrationArtifact(
    breakpoints=parsed.breakpoints,
    qualified=True,
    source=str(path),
    yaw_calibration_hash=parsed.yaw_calibration_hash,
  )


@dataclass(frozen=True)
class _StationCalibrationArtifact:
  breakpoints: tuple[tuple[float, float], ...]
  qualified: bool
  source: str
  station_calibration_hash: str | None


_STATION_FALLBACK_ARTIFACT = _StationCalibrationArtifact(
  breakpoints=STATION_DRIFT_FALLBACK_BREAKPOINTS,
  qualified=False,
  source="local-zero-drift-fallback",
  station_calibration_hash=None,
)


def _load_station_calibration(
  path: Path | None,
  controller_gain_hash: str,
  posture_map_hash: str | None,
  posture_artifact_hash: str | None,
) -> _StationCalibrationArtifact:
  if path is None:
    return _STATION_FALLBACK_ARTIFACT
  if not posture_map_hash:
    raise ValueError(
      "Station calibration binds to a qualified posture map; set "
      "HOPPERTREX_HYBRID_POSTURE_MAP_PATH or unset "
      f"{STATION_CALIBRATION_PATH_ENV}."
    )
  parsed = parse_station_calibration_artifact(
    _read_json_object(path, "Station calibration"),
    controller_gain_hash=controller_gain_hash,
    posture_map_hash=posture_map_hash,
    posture_artifact_hash=posture_artifact_hash,
  )
  return _StationCalibrationArtifact(
    breakpoints=parsed.breakpoints,
    qualified=True,
    source=str(path),
    station_calibration_hash=parsed.station_calibration_hash,
  )


@dataclass(kw_only=True)
class PostureCommandCfg(CommandTermCfg):
  """Uniform ``[target_height, target_pitch]`` command with rate limiting."""

  height_range: tuple[float, float]
  pitch_range: tuple[float, float]
  height_slew_rate: float | None = POSTURE_HEIGHT_SLEW_RATE
  pitch_slew_rate: float | None = POSTURE_PITCH_SLEW_RATE
  qualified: bool = False
  source: str = "local-unqualified-initial-posture"
  map_hash: str | None = None

  def __post_init__(self) -> None:
    parent_post_init = getattr(super(), "__post_init__", None)
    if parent_post_init is not None:
      parent_post_init()
    for name in ("height_slew_rate", "pitch_slew_rate"):
      value = getattr(self, name)
      if value is not None and (not math.isfinite(value) or value <= 0.0):
        raise ValueError(f"Posture {name} must be positive or None.")

  @property
  def command_dim(self) -> int:
    return 2

  def build(self, env: ManagerBasedRlEnv) -> "PostureCommand":
    return PostureCommand(self, env)


class PostureCommand(CommandTerm):
  """Sample raw posture targets and publish a rate-limited command.

  Stage 3.0 reference shaping: the resampled ``_target`` is the raw goal;
  the published ``command`` slews toward it at the configured axis rates,
  so every consumer (posture map, station feedforward, rewards, gates)
  sees one consistent shaped reference. Episodes snap to their initial
  target on reset - shaping only acts on WITHIN-episode target changes.
  A ``None`` rate publishes the target instantly (the legacy step).
  """

  cfg: PostureCommandCfg

  def __init__(self, cfg: PostureCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._command = torch.zeros(self.num_envs, 2, device=self.device)
    self._target = torch.zeros_like(self._command)
    self._step_dt = 0.0

  @property
  def command(self) -> torch.Tensor:
    return self._command

  @property
  def target(self) -> torch.Tensor:
    return self._target

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    extras = super().reset(env_ids)
    self._command[env_ids] = self._target[env_ids]
    return extras

  def compute(self, dt: float) -> None:
    self._step_dt = float(dt)
    super().compute(dt)

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # NOTE: the previous implementation called ``uniform_`` on
    # ``self._command[env_ids, 0]`` - advanced indexing returns a COPY, so
    # the in-place sample never reached the command tensor. The latent
    # no-op was harmless only because Stage3 never trained and every probe
    # forces the command; setitem assignment actually writes.
    count = len(env_ids)
    self._target[env_ids, 0] = torch.empty(
      count, device=self.device
    ).uniform_(*self.cfg.height_range)
    self._target[env_ids, 1] = torch.empty(
      count, device=self.device
    ).uniform_(*self.cfg.pitch_range)

  def _update_command(self) -> None:
    delta = self._target - self._command
    if self.cfg.height_slew_rate is not None:
      step = self.cfg.height_slew_rate * self._step_dt
      delta[:, 0] = torch.clamp(delta[:, 0], min=-step, max=step)
    if self.cfg.pitch_slew_rate is not None:
      step = self.cfg.pitch_slew_rate * self._step_dt
      delta[:, 1] = torch.clamp(delta[:, 1], min=-step, max=step)
    self._command += delta


@dataclass(kw_only=True)
class Stage1VelocityCommandCfg(UniformVelocityCommandCfg):
  """Stratified standing, nominal, and speed-extension commands."""

  nominal_abs_range: tuple[float, float] = STAGE1_ACTIVE_LIN_VEL_X_ABS_RANGE
  extension_abs_range: tuple[float, float] = (
    STAGE1_EXTENSION_LIN_VEL_X_ABS_RANGE
  )
  rel_extension_envs: float = STAGE1_EXTENSION_ENVS

  def __post_init__(self) -> None:
    super().__post_init__()
    if not 0.0 <= self.rel_extension_envs <= 1.0:
      raise ValueError("Stage1 extension fraction must be in [0, 1].")
    if self.rel_standing_envs + self.rel_extension_envs > 1.0:
      raise ValueError("Stage1 standing and extension fractions exceed 1.")
    nominal_low, nominal_high = self.nominal_abs_range
    extension_low, extension_high = self.extension_abs_range
    if nominal_low <= 0.0 or nominal_high < nominal_low:
      raise ValueError("Stage1 nominal command range must be positive and ordered.")
    if extension_low < nominal_high or extension_high < extension_low:
      raise ValueError(
        "Stage1 extension range must begin at or above the nominal maximum."
      )

  def build(self, env: ManagerBasedRlEnv) -> "Stage1VelocityCommand":
    return Stage1VelocityCommand(self, env)


class Stage1VelocityCommand(UniformVelocityCommand):
  """Sample Stage1-B categories without relying on measure-zero commands."""

  cfg: Stage1VelocityCommandCfg

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    count = len(env_ids)
    if count == 0:
      return
    self.vel_command_b[env_ids, :] = 0.0
    self.vel_command_w[env_ids, :] = 0.0
    category = torch.rand(count, device=self.device)
    standing_cutoff = self.cfg.rel_standing_envs
    extension_cutoff = standing_cutoff + self.cfg.rel_extension_envs
    standing = category < standing_cutoff
    extension = (category >= standing_cutoff) & (category < extension_cutoff)
    nominal = category >= extension_cutoff

    def signed_band(value_range: tuple[float, float]) -> torch.Tensor:
      magnitude = torch.empty(count, device=self.device).uniform_(*value_range)
      sign = torch.where(
        torch.rand(count, device=self.device) < 0.5,
        -torch.ones(count, device=self.device),
        torch.ones(count, device=self.device),
      )
      return sign * magnitude

    nominal_values = signed_band(self.cfg.nominal_abs_range)
    extension_values = signed_band(self.cfg.extension_abs_range)
    self.vel_command_b[env_ids[nominal], 0] = nominal_values[nominal]
    self.vel_command_b[env_ids[extension], 0] = extension_values[extension]
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
    self.is_standing_env[env_ids] = standing
    self.is_heading_env[env_ids] = False
    self.is_world_env[env_ids] = False
    self.is_forward_env[env_ids] = False


@dataclass(kw_only=True)
class HybridPlanarVelocityCommandCfg(UniformVelocityCommandCfg):
  """Stratified stop, linear-only, yaw-only, and combined commands."""

  rel_linear_only_envs: float = HYBRID_PLANAR_LINEAR_ONLY_ENVS
  rel_yaw_only_envs: float = HYBRID_PLANAR_YAW_ONLY_ENVS
  rel_linear_retention_envs: float = 0.0
  lin_vel_x_abs_range: tuple[float, float] = (
    HYBRID_PLANAR_LIN_VEL_X_ABS_RANGE
  )
  yaw_rate_abs_range: tuple[float, float] = HYBRID_PLANAR_YAW_RATE_ABS_RANGE
  linear_retention_abs_range: tuple[float, float] = (
    STAGE2_LINEAR_RETENTION_ABS_RANGE
  )

  def __post_init__(self) -> None:
    super().__post_init__()
    fractions = (
      self.rel_standing_envs,
      self.rel_linear_only_envs,
      self.rel_yaw_only_envs,
      self.rel_linear_retention_envs,
    )
    if any(value < 0.0 or value > 1.0 for value in fractions):
      raise ValueError("Hybrid planar command fractions must be in [0, 1].")
    if sum(fractions) > 1.0:
      raise ValueError("Hybrid planar command fractions must sum to at most 1.")
    for name, value_range in (
      ("lin_vel_x_abs_range", self.lin_vel_x_abs_range),
      ("yaw_rate_abs_range", self.yaw_rate_abs_range),
      ("linear_retention_abs_range", self.linear_retention_abs_range),
    ):
      low, high = value_range
      if low <= 0.0 or high < low:
        raise ValueError(f"{name} must be a positive ordered range.")

  def build(self, env: ManagerBasedRlEnv) -> "HybridPlanarVelocityCommand":
    return HybridPlanarVelocityCommand(self, env)


class HybridPlanarVelocityCommand(UniformVelocityCommand):
  """Sample explicit axis cases instead of relying on measure-zero zeros."""

  cfg: HybridPlanarVelocityCommandCfg

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    count = len(env_ids)
    if count == 0:
      return
    random = torch.empty(count, device=self.device)
    category = random.uniform_(0.0, 1.0)
    standing_cutoff = self.cfg.rel_standing_envs
    linear_cutoff = standing_cutoff + self.cfg.rel_linear_only_envs
    yaw_cutoff = linear_cutoff + self.cfg.rel_yaw_only_envs
    retention_cutoff = yaw_cutoff + self.cfg.rel_linear_retention_envs
    standing = category < standing_cutoff
    linear_only = (category >= standing_cutoff) & (category < linear_cutoff)
    yaw_only = (category >= linear_cutoff) & (category < yaw_cutoff)
    linear_retention = (category >= yaw_cutoff) & (category < retention_cutoff)
    active = ~standing

    def signed_band(value_range: tuple[float, float]) -> torch.Tensor:
      magnitude = torch.empty(count, device=self.device).uniform_(*value_range)
      sign = torch.where(
        torch.empty(count, device=self.device).uniform_(0.0, 1.0) < 0.5,
        -torch.ones(count, device=self.device),
        torch.ones(count, device=self.device),
      )
      return sign * magnitude

    linear = signed_band(self.cfg.lin_vel_x_abs_range)
    yaw = signed_band(self.cfg.yaw_rate_abs_range)
    retention_linear = signed_band(self.cfg.linear_retention_abs_range)
    self.vel_command_b[env_ids[active], 0] = linear[active]
    self.vel_command_b[env_ids[active], 2] = yaw[active]
    self.vel_command_b[env_ids[linear_retention], 0] = retention_linear[
      linear_retention
    ]
    self.vel_command_b[env_ids[standing], :] = 0.0
    self.vel_command_b[env_ids[linear_only], 2] = 0.0
    self.vel_command_b[env_ids[yaw_only], 0] = 0.0
    self.vel_command_b[env_ids[linear_retention], 2] = 0.0
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
    self.is_standing_env[env_ids] = standing


@dataclass(kw_only=True)
class StairRequestCommandCfg(CommandTermCfg):
  """Deterministic terrain-mode bit: flat retention first, stairs second."""

  flat_env_count: int = DYNAMIC_STAIR_FLAT_ENVS

  def __post_init__(self) -> None:
    parent_post_init = getattr(super(), "__post_init__", None)
    if parent_post_init is not None:
      parent_post_init()
    if self.flat_env_count < 0:
      raise ValueError("Stair request flat_env_count cannot be negative.")

  @property
  def command_dim(self) -> int:
    return 1

  def build(self, env: ManagerBasedRlEnv) -> StairRequestCommand:
    return StairRequestCommand(self, env)


class StairRequestCommand(CommandTerm):
  """Publish 0 for retention slots and 1 for regular-stair slots."""

  cfg: StairRequestCommandCfg

  def __init__(self, cfg: StairRequestCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    if cfg.flat_env_count > self.num_envs:
      raise ValueError("Stair request flat_env_count exceeds num_envs.")
    self._command = torch.zeros(self.num_envs, 1, device=self.device)
    if cfg.flat_env_count < self.num_envs:
      self._command[cfg.flat_env_count :, 0] = 1.0

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._command[env_ids, 0] = (
      env_ids >= self.cfg.flat_env_count
    ).to(self._command.dtype)

  def _update_command(self) -> None:
    pass


@dataclass(kw_only=True)
class StairDynamicVelocityCommandCfg(HybridPlanarVelocityCommandCfg):
  """Stage5 retention commands plus fixed forward stair commands."""

  flat_env_count: int = DYNAMIC_STAIR_FLAT_ENVS
  stair_vx: float = DYNAMIC_STAIR_COMMAND_VX_MPS

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.flat_env_count < 0:
      raise ValueError("StairDynamic flat_env_count cannot be negative.")
    if not math.isfinite(self.stair_vx) or self.stair_vx <= 0.0:
      raise ValueError("StairDynamic stair_vx must be finite and positive.")

  def build(self, env: ManagerBasedRlEnv) -> StairDynamicVelocityCommand:
    return StairDynamicVelocityCommand(self, env)


class StairDynamicVelocityCommand(HybridPlanarVelocityCommand):
  cfg: StairDynamicVelocityCommandCfg

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    stair = env_ids >= self.cfg.flat_env_count
    stair_ids = env_ids[stair]
    if len(stair_ids) == 0:
      return
    self.vel_command_b[stair_ids, :] = 0.0
    self.vel_command_b[stair_ids, 0] = self.cfg.stair_vx
    self.vel_command_w[stair_ids] = self.vel_command_b[stair_ids]
    self.is_standing_env[stair_ids] = False
    self.is_heading_env[stair_ids] = False
    self.is_world_env[stair_ids] = False
    self.is_forward_env[stair_ids] = False


@dataclass(kw_only=True)
class StairDynamicPostureCommandCfg(PostureCommandCfg):
  """Stage5 retention posture plus one fixed stair posture."""

  flat_env_count: int = DYNAMIC_STAIR_FLAT_ENVS
  stair_height: float = ROOT_HEIGHT_TARGET
  stair_pitch: float = 0.0

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.flat_env_count < 0:
      raise ValueError("StairDynamic flat_env_count cannot be negative.")
    if not all(math.isfinite(value) for value in (self.stair_height, self.stair_pitch)):
      raise ValueError("StairDynamic fixed posture must be finite.")

  def build(self, env: ManagerBasedRlEnv) -> StairDynamicPostureCommand:
    return StairDynamicPostureCommand(self, env)


class StairDynamicPostureCommand(PostureCommand):
  cfg: StairDynamicPostureCommandCfg

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    stair_ids = env_ids[env_ids >= self.cfg.flat_env_count]
    if len(stair_ids) == 0:
      return
    self._target[stair_ids, 0] = self.cfg.stair_height
    self._target[stair_ids, 1] = self.cfg.stair_pitch


@dataclass(kw_only=True)
class HybridWheelLegActionCfg(ActionTermCfg):
  """One invariant six-dimensional controller-residual action term."""

  action_mask: tuple[bool, ...]
  action_scales: tuple[float, ...]
  controller_gain: tuple[float, float, float, float]
  posture_coefficients: tuple[tuple[float, float, float, float], ...]
  wheel_joint_names: tuple[str, str] = WHEEL_JOINT_NAMES
  leg_joint_names: tuple[str, str, str, str] = LEG_JOINT_NAMES
  action_names: tuple[str, ...] = HYBRID_ACTION_NAMES
  controller_type: str = "pd"
  controller_qualified: bool = False
  controller_source: str = "local-unqualified-pd-fallback"
  controller_gain_hash: str | None = None
  controller_schedule: ControllerSchedule | None = None
  velocity_command_scale: float = 1.0
  velocity_command_bias: float = 0.0
  calibration_hash: str | None = None
  posture_map_qualified: bool = False
  posture_map_source: str = "local-unqualified-initial-posture"
  posture_map_hash: str | None = None
  posture_artifact_hash: str | None = None
  yaw_feedforward_breakpoints: tuple[tuple[float, float], ...] = (
    YAW_FEEDFORWARD_FALLBACK_BREAKPOINTS
  )
  yaw_calibration_qualified: bool = False
  yaw_calibration_source: str = "local-zero-feedforward-fallback"
  yaw_calibration_hash: str | None = None
  station_drift_breakpoints: tuple[tuple[float, float], ...] = (
    STATION_DRIFT_FALLBACK_BREAKPOINTS
  )
  station_calibration_qualified: bool = False
  station_calibration_source: str = "local-zero-drift-fallback"
  station_calibration_hash: str | None = None
  velocity_command_name: str = "twist"
  posture_command_name: str = "posture"
  wheel_radius: float = DEFAULT_WHEEL_RADIUS
  wheel_velocity_limit: float = DEFAULT_WHEEL_VELOCITY_LIMIT
  wheel_slew_limit: float = DEFAULT_WHEEL_SLEW_LIMIT
  # Probe-only sim-to-real tolerance injection. Defaults keep the term
  # byte-identical to the frozen Stage0-5 behavior: zero delay applies the
  # freshly composed targets directly and zero stds skip the noise draw
  # entirely. The latency/noise probe is the only intended writer.
  action_delay_steps: int = 0
  sensor_noise_pitch_std: float = 0.0
  sensor_noise_pitch_rate_std: float = 0.0
  sensor_noise_vx_std: float = 0.0
  sensor_noise_wheel_vel_std: float = 0.0
  sensor_noise_seed: int = 0
  # Residual stair camp (mainline doc S5B). Every default here is inert, so
  # the frozen Stage0-5 configs keep composing leg targets exactly as before:
  # with no trigger sensor and no forced mode, `stair_mode` stays False for
  # the whole episode and the leg reference tracks the posture map every step.
  stair_trigger_sensor_name: str | None = None
  stair_trigger_force_n: float = STAIR_TRIGGER_FORCE_N
  stair_trigger_window: int = STAIR_TRIGGER_WINDOW
  # Deviation minute 3: on the rising edge of `stair_mode` the classical leg
  # reference is latched and held for the rest of the episode, so the posture
  # map can no longer pull the legs back and fight the residual. The earlier
  # "classical leg output zeroed" wording was withdrawn as incoherent against
  # this code path - `_nominal_leg_targets` is an absolute joint target, not a
  # delta, so literal zeroing would snap the legs to 0 rad at the trigger.
  stair_mode_freezes_leg_reference: bool = False
  # Trigger-off ablation (S5B ablation set): `stair_mode` True from t=0.
  stair_mode_forced: bool = False
  # Registered failure-ladder branch; round 1 remains exactly 1.0.
  stair_mode_lqr_gain_scale: float = 1.0
  # Hybrid-v3 dynamic stair path. Defaults are inert and therefore leave the
  # frozen Stage0-5 and v2 StairCamp numerical paths untouched.
  dynamic_stair_maneuver: DynamicStairManeuver | None = None
  dynamic_stair_request_command_name: str | None = None
  dynamic_stair_left_sensor_name: str | None = None
  dynamic_stair_right_sensor_name: str | None = None
  dynamic_stair_control_dt: float = 0.02

  def __post_init__(self) -> None:
    if len(self.action_mask) != 6 or len(self.action_scales) != 6:
      raise ValueError("Hybrid action mask and scales must each contain six values.")
    if len(self.controller_gain) != 4:
      raise ValueError("Hybrid controller gain must contain four values.")
    if not math.isfinite(self.velocity_command_scale) or self.velocity_command_scale <= 0.0:
      raise ValueError("Velocity command scale must be finite and positive.")
    if not math.isfinite(self.velocity_command_bias):
      raise ValueError("Velocity command bias must be finite.")
    if np.asarray(self.posture_coefficients).shape != (3, 4):
      raise ValueError("Posture coefficients must have shape (3, 4).")
    self.yaw_feedforward_breakpoints = validate_yaw_breakpoints(
      self.yaw_feedforward_breakpoints
    )
    self.station_drift_breakpoints = validate_station_breakpoints(
      self.station_drift_breakpoints
    )
    if self.action_delay_steps < 0 or self.action_delay_steps > 8:
      raise ValueError("Action delay must be within [0, 8] control steps.")
    for name in (
      "sensor_noise_pitch_std",
      "sensor_noise_pitch_rate_std",
      "sensor_noise_vx_std",
      "sensor_noise_wheel_vel_std",
    ):
      value = getattr(self, name)
      if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    if not math.isfinite(self.stair_trigger_force_n) or (
      self.stair_trigger_force_n <= 0.0
    ):
      raise ValueError("Stair trigger force threshold must be positive.")
    if self.stair_trigger_window < 1:
      raise ValueError("Stair trigger window must be at least one sample.")
    if self.stair_mode_forced and self.stair_trigger_sensor_name is not None:
      raise ValueError(
        "The trigger-off ablation forces stair_mode on, so it must not also "
        "configure a contact trigger sensor."
      )
    if self.stair_mode_lqr_gain_scale not in (0.5, 1.0):
      raise ValueError("Stair-mode LQR gain scale must be exactly 1.0 or 0.5.")
    dynamic_fields = (
      self.dynamic_stair_maneuver,
      self.dynamic_stair_request_command_name,
      self.dynamic_stair_left_sensor_name,
      self.dynamic_stair_right_sensor_name,
    )
    if any(value is not None for value in dynamic_fields) and not all(
      value is not None for value in dynamic_fields
    ):
      raise ValueError("StairDynamic maneuver, command, and both sensors are atomic.")
    if self.dynamic_stair_maneuver is not None:
      if self.stair_trigger_sensor_name is not None or self.stair_mode_forced:
        raise ValueError("StairDynamic cannot share the v2 StairCamp trigger path.")
      if tuple(self.action_mask) != DYNAMIC_STAIR_ACTION_MASK:
        raise ValueError("StairDynamic requires all six PPO feedback heads.")
      if tuple(float(value) for value in self.action_scales) != DYNAMIC_STAIR_ACTION_SCALES:
        raise ValueError("StairDynamic must preserve the Stage5 action scales.")
    if not math.isfinite(self.dynamic_stair_control_dt) or self.dynamic_stair_control_dt <= 0.0:
      raise ValueError("StairDynamic control dt must be finite and positive.")

  @property
  def action_dim(self) -> int:
    return 6

  def build(self, env: ManagerBasedRlEnv) -> "HybridWheelLegAction":
    return HybridWheelLegAction(self, env)


def _torch_linear_interpolate(
  x: torch.Tensor,
  xp: torch.Tensor,
  fp: torch.Tensor,
) -> torch.Tensor:
  """Match numpy.interp on a strictly increasing grid, clamped at the ends.

  The numpy contract is the reference implementation used to fit and verify
  yaw calibration artifacts; an equivalence test pins the two together.
  """

  clamped = torch.clamp(x, min=xp[0], max=xp[-1])
  upper = torch.clamp(
    torch.bucketize(clamped, xp, right=True),
    min=1,
    max=xp.numel() - 1,
  )
  lower = upper - 1
  x0 = xp[lower]
  x1 = xp[upper]
  weight = (clamped - x0) / (x1 - x0)
  return fp[lower] + weight * (fp[upper] - fp[lower])


def _torch_bilinear_interpolate(
  x: torch.Tensor,
  y: torch.Tensor,
  xp: torch.Tensor,
  yp: torch.Tensor,
  values: torch.Tensor,
) -> torch.Tensor:
  """Clamp and bilinearly interpolate a rectangular schedule grid."""

  x_clamped = torch.clamp(x, min=xp[0], max=xp[-1])
  y_clamped = torch.clamp(y, min=yp[0], max=yp[-1])
  x_upper = torch.clamp(torch.bucketize(x_clamped, xp, right=True), 1, xp.numel() - 1)
  y_upper = torch.clamp(torch.bucketize(y_clamped, yp, right=True), 1, yp.numel() - 1)
  x_lower = x_upper - 1
  y_lower = y_upper - 1
  x_weight = (x_clamped - xp[x_lower]) / (xp[x_upper] - xp[x_lower])
  y_weight = (y_clamped - yp[y_lower]) / (yp[y_upper] - yp[y_lower])
  v00 = values[x_lower, y_lower]
  v01 = values[x_lower, y_upper]
  v10 = values[x_upper, y_lower]
  v11 = values[x_upper, y_upper]
  while x_weight.ndim < v00.ndim:
    x_weight = x_weight.unsqueeze(-1)
    y_weight = y_weight.unsqueeze(-1)
  low = (1.0 - y_weight) * v00 + y_weight * v01
  high = (1.0 - y_weight) * v10 + y_weight * v11
  return (1.0 - x_weight) * low + x_weight * high


def stair_dynamic_safe_control_inputs(
  velocity_command: torch.Tensor,
  applied_residual: torch.Tensor,
  abort_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Mask the command and all PPO authority for v3 ABORT slots."""

  if (
    velocity_command.ndim != 2
    or velocity_command.shape[1] < 3
    or applied_residual.ndim != 2
    or applied_residual.shape[0] != velocity_command.shape[0]
    or abort_mask.shape != (velocity_command.shape[0],)
  ):
    raise ValueError("StairDynamic safe-control tensors have incompatible shapes.")
  if not bool(abort_mask.any()):
    # Preserve the exact Stage0-5/request-false numerical path and allocations.
    return velocity_command, applied_residual
  safe_command = velocity_command.clone()
  safe_residual = applied_residual.clone()
  safe_command[abort_mask, 0] = 0.0
  safe_command[abort_mask, 2] = 0.0
  safe_residual[abort_mask] = 0.0
  return safe_command, safe_residual


def stair_dynamic_target_saturation_mask(
  desired_legs: torch.Tensor,
  soft_limits: torch.Tensor,
  dynamic_active: torch.Tensor,
) -> torch.Tensor:
  """Return dynamic slots whose composed target exceeds a soft joint limit."""

  if (
    desired_legs.ndim != 2
    or soft_limits.shape != (*desired_legs.shape, 2)
    or dynamic_active.shape != (desired_legs.shape[0],)
  ):
    raise ValueError("StairDynamic target-limit tensors have incompatible shapes.")
  outside = (desired_legs < soft_limits[..., 0]) | (
    desired_legs > soft_limits[..., 1]
  )
  return dynamic_active & outside.any(dim=1)


class HybridWheelLegAction(ActionTerm):
  """Apply controller wheel targets and four two-leg posture targets."""

  cfg: HybridWheelLegActionCfg

  def __init__(self, cfg: HybridWheelLegActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    wheel_ids, wheel_names = self._entity.find_joints(
      cfg.wheel_joint_names,
      preserve_order=True,
    )
    leg_ids, leg_names = self._entity.find_joints(
      cfg.leg_joint_names,
      preserve_order=True,
    )
    if tuple(wheel_names) != cfg.wheel_joint_names:
      raise ValueError(f"Expected wheel joints {cfg.wheel_joint_names}, got {wheel_names}.")
    if tuple(leg_names) != cfg.leg_joint_names:
      raise ValueError(f"Expected leg joints {cfg.leg_joint_names}, got {leg_names}.")
    self._wheel_ids = torch.tensor(wheel_ids, device=self.device, dtype=torch.long)
    self._leg_ids = torch.tensor(leg_ids, device=self.device, dtype=torch.long)
    self._gain = torch.tensor(
      cfg.controller_gain,
      device=self.device,
      dtype=torch.float,
    )
    self._schedule_enabled = cfg.controller_schedule is not None
    if cfg.controller_schedule is not None:
      schedule = cfg.controller_schedule
      self._schedule_heights = torch.tensor(
        schedule.height_nodes, device=self.device, dtype=torch.float
      )
      self._schedule_pitches = torch.tensor(
        schedule.pitch_nodes, device=self.device, dtype=torch.float
      )
      self._schedule_gains = torch.tensor(
        schedule.gains, device=self.device, dtype=torch.float
      )
      self._schedule_equilibrium_pitch = torch.tensor(
        schedule.equilibrium_pitch, device=self.device, dtype=torch.float
      )
      self._schedule_equilibrium_state = torch.tensor(
        schedule.equilibrium_state, device=self.device, dtype=torch.float
      )
      self._schedule_equilibrium_input = torch.tensor(
        schedule.equilibrium_input, device=self.device, dtype=torch.float
      )
    self._mask = torch.tensor(
      cfg.action_mask,
      device=self.device,
      dtype=torch.float,
    )
    self._scales = torch.tensor(
      cfg.action_scales,
      device=self.device,
      dtype=torch.float,
    )
    self._posture_coefficients = torch.tensor(
      cfg.posture_coefficients,
      device=self.device,
      dtype=torch.float,
    )
    yaw_breakpoints = validate_yaw_breakpoints(cfg.yaw_feedforward_breakpoints)
    self._yaw_feedforward_wz = torch.tensor(
      [point[0] for point in yaw_breakpoints],
      device=self.device,
      dtype=torch.float,
    )
    self._yaw_feedforward_diff = torch.tensor(
      [point[1] for point in yaw_breakpoints],
      device=self.device,
      dtype=torch.float,
    )
    station_breakpoints = validate_station_breakpoints(
      cfg.station_drift_breakpoints
    )
    self._station_pitch = torch.tensor(
      [point[0] for point in station_breakpoints],
      device=self.device,
      dtype=torch.float,
    )
    self._station_drift = torch.tensor(
      [point[1] for point in station_breakpoints],
      device=self.device,
      dtype=torch.float,
    )
    self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
    self._applied_residual = torch.zeros_like(self._raw_actions)
    self._previous_applied_residual = torch.zeros_like(self._raw_actions)
    self._previous_previous_applied_residual = torch.zeros_like(
      self._raw_actions
    )
    self._controller_baseline = torch.zeros(self.num_envs, 2, device=self.device)
    self._classical_errors = torch.zeros(self.num_envs, 4, device=self.device)
    self._previous_wheel_targets = torch.zeros_like(self._controller_baseline)
    self._wheel_targets = torch.zeros_like(self._controller_baseline)
    self._nominal_leg_targets = torch.zeros(self.num_envs, 4, device=self.device)
    self._leg_targets = torch.zeros_like(self._nominal_leg_targets)
    initial = torch.tensor(
      [INIT_JOINT_POS[name] for name in cfg.leg_joint_names],
      device=self.device,
      dtype=torch.float,
    )
    self._nominal_leg_targets[:] = initial
    self._leg_targets[:] = initial
    # Residual stair camp state (S5B). `_leg_reference` is the pose the leg
    # residual is added to: it tracks `_nominal_leg_targets` step by step
    # until `stair_mode` latches, then holds (deviation minute 3). With the
    # camp disabled it equals `_nominal_leg_targets` at every step, so the
    # frozen stages compose exactly the same leg targets as before.
    self._leg_reference = self._nominal_leg_targets.clone()
    self._stair_mode = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    self._stair_trigger_streak = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.long
    )
    self._stair_mode_forced = bool(cfg.stair_mode_forced)
    self._stair_freeze_enabled = bool(cfg.stair_mode_freezes_leg_reference)
    self._stair_mode_lqr_gain_scale = float(cfg.stair_mode_lqr_gain_scale)
    self._stair_trigger_sensor_name = cfg.stair_trigger_sensor_name
    if self._stair_trigger_sensor_name is not None:
      # Fail at construction rather than on the first control step: a missing
      # sensor would otherwise leave `stair_mode` silently False forever and
      # the camp would train as an un-triggered always-classical baseline.
      if self._stair_trigger_sensor_name not in env.scene.sensors:
        raise ValueError(
          "Stair trigger sensor "
          f"'{self._stair_trigger_sensor_name}' is not in the scene."
        )
    if self._stair_mode_forced:
      self._stair_mode[:] = True
    # Hybrid-v3 dynamic stair state. All tensors exist even when disabled so
    # observation helpers have stable properties; the disabled hot path only
    # clears feedforward and is numerically identical to Stage0-5/v2.
    self._dynamic_maneuver = cfg.dynamic_stair_maneuver
    self._dynamic_enabled = self._dynamic_maneuver is not None
    self._dynamic_request_command_name = cfg.dynamic_stair_request_command_name
    self._dynamic_left_sensor_name = cfg.dynamic_stair_left_sensor_name
    self._dynamic_right_sensor_name = cfg.dynamic_stair_right_sensor_name
    self._dynamic_dt = float(cfg.dynamic_stair_control_dt)
    self._dynamic_stair_request = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    self._dynamic_phase = torch.full(
      (self.num_envs,), int(StairPhase.IDLE), device=self.device, dtype=torch.long
    )
    self._dynamic_phase_elapsed = torch.zeros(self.num_envs, device=self.device)
    self._dynamic_step_progress = torch.zeros(self.num_envs, device=self.device)
    self._dynamic_step_index = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.long
    )
    env_index = torch.arange(self.num_envs, device=self.device)
    self._dynamic_preferred_side = torch.where(
      env_index.remainder(2) == 0,
      torch.full_like(env_index, int(LeadSide.LEFT)),
      torch.full_like(env_index, int(LeadSide.RIGHT)),
    )
    self._dynamic_lead_side = torch.zeros_like(self._dynamic_preferred_side)
    self._dynamic_left_streak = torch.zeros_like(self._dynamic_preferred_side)
    self._dynamic_right_streak = torch.zeros_like(self._dynamic_preferred_side)
    self._dynamic_left_loaded = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    self._dynamic_right_loaded = torch.zeros_like(self._dynamic_left_loaded)
    self._dynamic_left_force = torch.zeros(self.num_envs, device=self.device)
    self._dynamic_right_force = torch.zeros_like(self._dynamic_left_force)
    self._dynamic_trail_contact_elapsed = torch.full(
      (self.num_envs,), -1.0, device=self.device
    )
    self._dynamic_recover_stable = torch.zeros_like(self._dynamic_preferred_side)
    self._dynamic_traversal_mode = torch.zeros_like(self._dynamic_preferred_side)
    self._dynamic_abort_code = torch.zeros_like(self._dynamic_preferred_side)
    self._dynamic_episode_unsafe = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    self._dynamic_target_saturation = torch.zeros_like(
      self._dynamic_episode_unsafe
    )
    self._dynamic_leg_feedforward = torch.zeros(
      self.num_envs, 4, device=self.device
    )
    self._dynamic_drive_feedforward = torch.zeros(self.num_envs, device=self.device)
    self._dynamic_riser_cross_event = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    self._dynamic_previous_root_x = self._entity.data.root_link_pos_w[:, 0].clone()
    self._dynamic_candidate_parameters = torch.zeros(
      self.num_envs, 4, device=self.device
    )
    if self._dynamic_enabled:
      assert self._dynamic_maneuver is not None
      try:
        request_command = env.command_manager.get_command(
          str(self._dynamic_request_command_name)
        )
      except (AttributeError, KeyError) as exc:
        raise ValueError(
          f"StairDynamic request command {self._dynamic_request_command_name!r} is missing."
        ) from exc
      if request_command.shape != (self.num_envs, 1):
        raise ValueError("StairDynamic request command must have shape [B, 1].")
      for sensor_name in (
        self._dynamic_left_sensor_name,
        self._dynamic_right_sensor_name,
      ):
        if sensor_name not in env.scene.sensors:
          raise ValueError(f"StairDynamic sensor {sensor_name!r} is missing.")
      self._dynamic_split_left = torch.tensor(
        self._dynamic_maneuver.split_basis_left,
        device=self.device,
        dtype=torch.float,
      )
      self._dynamic_split_right = torch.tensor(
        self._dynamic_maneuver.split_basis_right,
        device=self.device,
        dtype=torch.float,
      )
      self._dynamic_lift_left = torch.tensor(
        self._dynamic_maneuver.lift_basis_left,
        device=self.device,
        dtype=torch.float,
      )
      self._dynamic_lift_right = torch.tensor(
        self._dynamic_maneuver.lift_basis_right,
        device=self.device,
        dtype=torch.float,
      )
      self._dynamic_candidate_parameters[:] = torch.tensor(
        (
          self._dynamic_maneuver.split_amplitude_rad,
          self._dynamic_maneuver.lift_amplitude_rad,
          self._dynamic_maneuver.trailing_delay_s,
          self._dynamic_maneuver.drive_feedforward_radps,
        ),
        device=self.device,
        dtype=torch.float,
      )
    # Probe-only sim-to-real tolerance injection state. With the default
    # cfg (zero delay, zero stds) none of it is touched on the hot path.
    self._noise_enabled = (
      cfg.sensor_noise_pitch_std > 0.0
      or cfg.sensor_noise_pitch_rate_std > 0.0
      or cfg.sensor_noise_vx_std > 0.0
      or cfg.sensor_noise_wheel_vel_std > 0.0
    )
    if self._noise_enabled:
      self._noise_generator = torch.Generator(device=self.device)
      self._noise_generator.manual_seed(int(cfg.sensor_noise_seed))
    self._delay_steps = int(cfg.action_delay_steps)
    if self._delay_steps > 0:
      self._delayed_wheel_ring = torch.zeros(
        self._delay_steps, self.num_envs, 2, device=self.device
      )
      self._delayed_leg_ring = initial.expand(
        self._delay_steps, self.num_envs, 4
      ).clone()
      self._delay_pointer = 0
      self._delay_dirty = False
      self._delayed_wheel_output = torch.zeros(
        self.num_envs, 2, device=self.device
      )
      self._delayed_leg_output = initial.expand(self.num_envs, 4).clone()

  @property
  def action_dim(self) -> int:
    return 6

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def applied_residual(self) -> torch.Tensor:
    return self._applied_residual

  @property
  def previous_applied_residual(self) -> torch.Tensor:
    return self._previous_applied_residual

  @property
  def previous_previous_applied_residual(self) -> torch.Tensor:
    return self._previous_previous_applied_residual

  @property
  def controller_baseline(self) -> torch.Tensor:
    return self._controller_baseline

  @property
  def classical_errors(self) -> torch.Tensor:
    """`[B, 4]` the state the classical layer regulates to zero."""

    return self._classical_errors

  @property
  def wheel_targets(self) -> torch.Tensor:
    return self._wheel_targets

  @property
  def nominal_leg_targets(self) -> torch.Tensor:
    return self._nominal_leg_targets

  @property
  def leg_targets(self) -> torch.Tensor:
    return self._leg_targets

  @property
  def leg_reference(self) -> torch.Tensor:
    """The pose the leg residual is added to (frozen once `stair_mode`)."""

    return self._leg_reference

  @property
  def stair_mode(self) -> torch.Tensor:
    """`[B]` bool: latched CTBC-style contact trigger (S5B Protocol 2)."""

    return self._stair_mode

  @property
  def stair_trigger_streak(self) -> torch.Tensor:
    return self._stair_trigger_streak

  @property
  def dynamic_stair_request(self) -> torch.Tensor:
    return self._dynamic_stair_request

  @property
  def dynamic_phase(self) -> torch.Tensor:
    return self._dynamic_phase

  @property
  def dynamic_loaded_contact(self) -> torch.Tensor:
    return torch.stack(
      (self._dynamic_left_loaded, self._dynamic_right_loaded), dim=1
    )

  @property
  def dynamic_contact_force(self) -> torch.Tensor:
    return torch.stack(
      (self._dynamic_left_force, self._dynamic_right_force), dim=1
    )

  @property
  def dynamic_lead_side(self) -> torch.Tensor:
    return self._dynamic_lead_side

  @property
  def dynamic_leg_feedforward(self) -> torch.Tensor:
    return self._dynamic_leg_feedforward

  @property
  def dynamic_drive_feedforward(self) -> torch.Tensor:
    return self._dynamic_drive_feedforward

  @property
  def dynamic_step_index(self) -> torch.Tensor:
    return self._dynamic_step_index

  @property
  def dynamic_traversal_mode(self) -> torch.Tensor:
    return self._dynamic_traversal_mode

  @property
  def dynamic_abort_code(self) -> torch.Tensor:
    return self._dynamic_abort_code

  @property
  def dynamic_episode_unsafe(self) -> torch.Tensor:
    return self._dynamic_episode_unsafe

  @property
  def dynamic_target_saturation(self) -> torch.Tensor:
    return self._dynamic_target_saturation

  @property
  def dynamic_riser_cross_event(self) -> torch.Tensor:
    return self._dynamic_riser_cross_event

  @property
  def dynamic_candidate_parameters(self) -> torch.Tensor:
    return self._dynamic_candidate_parameters

  def set_dynamic_candidate_parameters(self, values: torch.Tensor) -> None:
    """Assign one bounded CEM candidate per environment without rebuilding."""

    if not self._dynamic_enabled:
      raise ValueError("Dynamic candidate parameters require StairDynamic mode.")
    converted = torch.as_tensor(values, device=self.device, dtype=torch.float)
    if converted.shape != (self.num_envs, 4) or not bool(
      torch.isfinite(converted).all()
    ):
      raise ValueError("Dynamic candidates must be finite with shape [B, 4].")
    lower = torch.tensor(DYNAMIC_STAIR_CEM_LOWER, device=self.device)
    upper = torch.tensor(DYNAMIC_STAIR_CEM_UPPER, device=self.device)
    if bool(((converted < lower) | (converted > upper)).any()):
      raise ValueError("Dynamic candidate parameters exceed frozen CEM bounds.")
    self._dynamic_candidate_parameters.copy_(converted)

  def _update_stair_mode(self) -> None:
    """Advance the latched contact trigger by one control step."""

    if self._stair_mode_forced:
      # Trigger-off ablation: the mode is on from t=0 and never re-evaluated.
      return
    if self._stair_trigger_sensor_name is None:
      return
    data = self._env.scene.sensors[self._stair_trigger_sensor_name].data
    metric = stair_trigger_metric(
      found=data.found,
      force_contact_frame=data.force,
      normal_global=data.normal,
    )
    latched, streak = update_stair_trigger(
      latched=self._stair_mode,
      streak=self._stair_trigger_streak,
      metric=metric,
      threshold=self.cfg.stair_trigger_force_n,
      window=self.cfg.stair_trigger_window,
    )
    # Write in place so the buffers keep a stable identity: observation and
    # reward terms read them through the properties every control step, and
    # `reset()` writes into whichever tensor is current.
    self._stair_mode.copy_(latched)
    self._stair_trigger_streak.copy_(streak)

  def _dynamic_contact_metric(self, sensor_name: str | None) -> torch.Tensor:
    if sensor_name is None:
      return torch.zeros(self.num_envs, device=self.device)
    data = self._env.scene.sensors[sensor_name].data
    return stair_trigger_metric(
      found=data.found,
      force_contact_frame=data.force,
      normal_global=data.normal,
    )

  def _update_dynamic_stair(
    self,
    *,
    pitch: torch.Tensor,
    pitch_rate: torch.Tensor,
    projected_gravity: torch.Tensor,
  ) -> None:
    """Vectorized mirror of ``stair_dynamic.dynamic_stair_step``."""

    self._dynamic_riser_cross_event.zero_()
    self._dynamic_leg_feedforward.zero_()
    self._dynamic_drive_feedforward.zero_()
    if not self._dynamic_enabled:
      self._dynamic_stair_request.zero_()
      return
    maneuver = self._dynamic_maneuver
    assert maneuver is not None
    assert self._dynamic_request_command_name is not None
    request_command = self._env.command_manager.get_command(
      self._dynamic_request_command_name
    )
    if request_command.shape != (self.num_envs, 1):
      raise ValueError("StairDynamic request command must have shape [B, 1].")
    request = request_command[:, 0] > 0.5
    self._dynamic_stair_request.copy_(request)

    left_force = self._dynamic_contact_metric(self._dynamic_left_sensor_name)
    right_force = self._dynamic_contact_metric(self._dynamic_right_sensor_name)
    self._dynamic_left_force.copy_(left_force)
    self._dynamic_right_force.copy_(right_force)

    root_x = self._entity.data.root_link_pos_w[:, 0]
    delta = root_x - self._dynamic_previous_root_x
    self._dynamic_previous_root_x.copy_(root_x)
    phase = self._dynamic_phase
    starting = request & (
      (phase == int(StairPhase.IDLE)) | (phase == int(StairPhase.DONE))
    )
    inactive = ~request
    if bool(inactive.any()):
      phase[inactive] = int(StairPhase.IDLE)
      self._dynamic_phase_elapsed[inactive] = 0.0
      self._dynamic_step_progress[inactive] = 0.0
      self._dynamic_step_index[inactive] = 0
      self._dynamic_lead_side[inactive] = int(LeadSide.NONE)
      self._dynamic_left_streak[inactive] = 0
      self._dynamic_right_streak[inactive] = 0
      self._dynamic_left_loaded[inactive] = False
      self._dynamic_right_loaded[inactive] = False
      self._dynamic_trail_contact_elapsed[inactive] = -1.0
      self._dynamic_recover_stable[inactive] = 0
      self._dynamic_traversal_mode[inactive] = 0
      self._dynamic_abort_code[inactive] = 0
      self._dynamic_episode_unsafe[inactive] = False
      self._dynamic_target_saturation[inactive] = False
    if bool(starting.any()):
      phase[starting] = int(StairPhase.APPROACH)
      self._dynamic_phase_elapsed[starting] = 0.0
      self._dynamic_step_progress[starting] = 0.0
      self._dynamic_lead_side[starting] = int(LeadSide.NONE)
      self._dynamic_left_streak[starting] = 0
      self._dynamic_right_streak[starting] = 0
      self._dynamic_left_loaded[starting] = False
      self._dynamic_right_loaded[starting] = False
      self._dynamic_trail_contact_elapsed[starting] = -1.0
      self._dynamic_recover_stable[starting] = 0
      self._dynamic_traversal_mode[starting] = 0
      self._dynamic_abort_code[starting] = 0
      self._dynamic_episode_unsafe[starting] = False
      self._dynamic_target_saturation[starting] = False

    running = request & ~starting & (phase != int(StairPhase.ABORT))
    self._dynamic_phase_elapsed[running] += self._dynamic_dt
    self._dynamic_step_progress[running] += delta[running]

    trigger_active = request & (
      (phase == int(StairPhase.APPROACH))
      | (phase == int(StairPhase.PRELOAD))
      | (phase == int(StairPhase.CONTACT_WAIT))
      | (phase == int(StairPhase.LEAD_LIFT))
    )
    left_hit = trigger_active & (left_force >= maneuver.trigger_force_n)
    right_hit = trigger_active & (right_force >= maneuver.trigger_force_n)
    self._dynamic_left_streak.copy_(
      torch.where(
        trigger_active,
        torch.where(
          left_hit,
          torch.clamp(self._dynamic_left_streak + 1, max=maneuver.trigger_window),
          torch.zeros_like(self._dynamic_left_streak),
        ),
        self._dynamic_left_streak,
      )
    )
    self._dynamic_right_streak.copy_(
      torch.where(
        trigger_active,
        torch.where(
          right_hit,
          torch.clamp(self._dynamic_right_streak + 1, max=maneuver.trigger_window),
          torch.zeros_like(self._dynamic_right_streak),
        ),
        self._dynamic_right_streak,
      )
    )
    self._dynamic_left_loaded.logical_or_(
      trigger_active & (self._dynamic_left_streak >= maneuver.trigger_window)
    )
    self._dynamic_right_loaded.logical_or_(
      trigger_active & (self._dynamic_right_streak >= maneuver.trigger_window)
    )

    try:
      non_wheel = self._env.termination_manager.get_term(
        "non_wheel_ground_contact"
      ).bool()
    except (AttributeError, KeyError):
      non_wheel = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    soft_limits = self._entity.data.soft_joint_pos_limits[:, self._leg_ids]
    joint_pos = self._entity.data.joint_pos[:, self._leg_ids]
    actuator_limit = (
      (joint_pos < soft_limits[..., 0]) | (joint_pos > soft_limits[..., 1])
    ).any(dim=1)
    roll = torch.atan2(
      -projected_gravity[:, 1],
      torch.clamp(-projected_gravity[:, 2], min=1.0e-6),
    )
    orientation_limit = (pitch.abs() > 0.35) | (roll.abs() > 0.35)
    backward = self._dynamic_step_progress < -0.10
    abort = request & (non_wheel | actuator_limit | orientation_limit | backward)
    abort_code = torch.where(
      non_wheel,
      torch.ones_like(self._dynamic_abort_code),
      torch.where(
        actuator_limit,
        torch.full_like(self._dynamic_abort_code, 2),
        torch.where(
          orientation_limit,
          torch.full_like(self._dynamic_abort_code, 3),
          torch.full_like(self._dynamic_abort_code, 4),
        ),
      ),
    )
    if bool(abort.any()):
      phase[abort] = int(StairPhase.ABORT)
      self._dynamic_phase_elapsed[abort] = 0.0
      self._dynamic_traversal_mode[abort] = 3
      self._dynamic_abort_code[abort] = abort_code[abort]

    cross_distance = torch.where(
      self._dynamic_step_index == 0,
      torch.full_like(self._dynamic_step_progress, maneuver.first_cross_m),
      torch.full_like(self._dynamic_step_progress, maneuver.next_cross_m),
    )
    crossed = self._dynamic_step_progress >= cross_distance
    pre_contact = request & ~abort & (
      (phase == int(StairPhase.APPROACH))
      | (phase == int(StairPhase.PRELOAD))
      | (phase == int(StairPhase.CONTACT_WAIT))
    )
    left_only = self._dynamic_left_loaded & ~self._dynamic_right_loaded
    right_only = self._dynamic_right_loaded & ~self._dynamic_left_loaded
    both = self._dynamic_left_loaded & self._dynamic_right_loaded
    selected = torch.zeros_like(self._dynamic_lead_side)
    selected[left_only] = int(LeadSide.LEFT)
    selected[right_only] = int(LeadSide.RIGHT)
    left_stronger = both & (left_force > right_force)
    right_stronger = both & (right_force > left_force)
    tie = both & ~(left_stronger | right_stronger)
    selected[left_stronger] = int(LeadSide.LEFT)
    selected[right_stronger] = int(LeadSide.RIGHT)
    selected[tie] = self._dynamic_preferred_side[tie]

    roll_cross = pre_contact & crossed & (selected == int(LeadSide.NONE))
    if bool(roll_cross.any()):
      phase[roll_cross] = int(StairPhase.RECOVER)
      self._dynamic_phase_elapsed[roll_cross] = 0.0
      self._dynamic_traversal_mode[roll_cross] = 1
      self._dynamic_riser_cross_event[roll_cross] = True
    dynamic_start = pre_contact & (selected != int(LeadSide.NONE))
    if bool(dynamic_start.any()):
      phase[dynamic_start] = int(StairPhase.LEAD_LIFT)
      self._dynamic_phase_elapsed[dynamic_start] = 0.0
      self._dynamic_lead_side[dynamic_start] = selected[dynamic_start]
      selected_trail_loaded = torch.where(
        selected == int(LeadSide.LEFT),
        self._dynamic_right_loaded,
        self._dynamic_left_loaded,
      )
      self._dynamic_trail_contact_elapsed[dynamic_start] = torch.where(
        selected_trail_loaded[dynamic_start],
        torch.zeros_like(self._dynamic_trail_contact_elapsed[dynamic_start]),
        torch.full_like(self._dynamic_trail_contact_elapsed[dynamic_start], -1.0),
      )
      self._dynamic_traversal_mode[dynamic_start] = 2

    approach_done = (
      pre_contact
      & ~roll_cross
      & ~dynamic_start
      & (phase == int(StairPhase.APPROACH))
      & (
        self._dynamic_phase_elapsed + DYNAMIC_STAIR_TIME_EPS_S
        >= maneuver.approach_duration_s
      )
    )
    if bool(approach_done.any()):
      phase[approach_done] = int(StairPhase.PRELOAD)
      self._dynamic_phase_elapsed[approach_done] = 0.0
    preload_done = (
      pre_contact
      & ~roll_cross
      & ~dynamic_start
      & (phase == int(StairPhase.PRELOAD))
      & (
        self._dynamic_phase_elapsed + DYNAMIC_STAIR_TIME_EPS_S
        >= maneuver.preload_duration_s
      )
    )
    if bool(preload_done.any()):
      phase[preload_done] = int(StairPhase.CONTACT_WAIT)
      self._dynamic_phase_elapsed[preload_done] = 0.0
    contact_timeout = (
      pre_contact
      & ~roll_cross
      & ~dynamic_start
      & (phase == int(StairPhase.CONTACT_WAIT))
      & (
        self._dynamic_phase_elapsed + DYNAMIC_STAIR_TIME_EPS_S
        >= maneuver.contact_timeout_s
      )
    )
    if bool(contact_timeout.any()):
      phase[contact_timeout] = int(StairPhase.ABORT)
      self._dynamic_phase_elapsed[contact_timeout] = 0.0
      self._dynamic_traversal_mode[contact_timeout] = 3
      self._dynamic_abort_code[contact_timeout] = 5

    lead_phase = request & ~abort & (phase == int(StairPhase.LEAD_LIFT))
    if maneuver.lift_mode == DynamicLiftMode.SYNCHRONIZED:
      lead_done = lead_phase & (
        self._dynamic_phase_elapsed + DYNAMIC_STAIR_TIME_EPS_S
        >= maneuver.lift_duration_s
      )
      lead_timeout = torch.zeros_like(lead_done)
    else:
      trail_loaded = torch.where(
        self._dynamic_lead_side == int(LeadSide.LEFT),
        self._dynamic_right_loaded,
        self._dynamic_left_loaded,
      )
      new_trail_edge = (
        lead_phase
        & trail_loaded
        & (self._dynamic_trail_contact_elapsed < 0.0)
      )
      self._dynamic_trail_contact_elapsed[new_trail_edge] = (
        self._dynamic_phase_elapsed[new_trail_edge]
      )
      has_trail_edge = self._dynamic_trail_contact_elapsed >= 0.0
      ready_time = torch.maximum(
        torch.full_like(
          self._dynamic_phase_elapsed, maneuver.lift_duration_s
        ),
        self._dynamic_trail_contact_elapsed
        + self._dynamic_candidate_parameters[:, 2],
      )
      lead_done = (
        lead_phase
        & has_trail_edge
        & (self._dynamic_phase_elapsed + DYNAMIC_STAIR_TIME_EPS_S >= ready_time)
      )
      lead_timeout = (
        lead_phase
        & ~has_trail_edge
        & (
          self._dynamic_phase_elapsed + DYNAMIC_STAIR_TIME_EPS_S
          >= maneuver.lift_duration_s + maneuver.trail_contact_timeout_s
        )
      )
    if bool(lead_done.any()):
      phase[lead_done] = int(StairPhase.TRAIL_LIFT)
      self._dynamic_phase_elapsed[lead_done] = 0.0
    if bool(lead_timeout.any()):
      phase[lead_timeout] = int(StairPhase.ABORT)
      self._dynamic_phase_elapsed[lead_timeout] = 0.0
      self._dynamic_traversal_mode[lead_timeout] = 3
      self._dynamic_abort_code[lead_timeout] = 6

    trail_phase = request & ~abort & (phase == int(StairPhase.TRAIL_LIFT))
    required_lift = (
      maneuver.lift_duration_s
      if maneuver.lift_mode == DynamicLiftMode.ALTERNATING
      else 0.0
    )
    dynamic_cross = (
      trail_phase
      & crossed
      & (self._dynamic_phase_elapsed + DYNAMIC_STAIR_TIME_EPS_S >= required_lift)
    )
    if bool(dynamic_cross.any()):
      phase[dynamic_cross] = int(StairPhase.RECOVER)
      self._dynamic_phase_elapsed[dynamic_cross] = 0.0
      self._dynamic_riser_cross_event[dynamic_cross] = True
    cross_timeout = (
      trail_phase
      & ~dynamic_cross
      & (
        self._dynamic_phase_elapsed + DYNAMIC_STAIR_TIME_EPS_S
        >= required_lift + maneuver.cross_timeout_s
      )
    )
    if bool(cross_timeout.any()):
      phase[cross_timeout] = int(StairPhase.ABORT)
      self._dynamic_phase_elapsed[cross_timeout] = 0.0
      self._dynamic_traversal_mode[cross_timeout] = 3
      self._dynamic_abort_code[cross_timeout] = 7

    recover = request & (phase == int(StairPhase.RECOVER))
    stable = (
      recover
      & ~non_wheel
      & (pitch.abs() <= 0.10)
      & (roll.abs() <= 0.10)
      & (pitch_rate.abs() <= 0.5)
    )
    self._dynamic_recover_stable.copy_(
      torch.where(
        recover,
        torch.where(
          stable,
          self._dynamic_recover_stable + 1,
          torch.zeros_like(self._dynamic_recover_stable),
        ),
        self._dynamic_recover_stable,
      )
    )
    recovered = (
      recover
      & (
        self._dynamic_phase_elapsed + DYNAMIC_STAIR_TIME_EPS_S
        >= maneuver.recover_duration_s
      )
      & (self._dynamic_recover_stable >= maneuver.recover_stable_steps)
    )
    if bool(recovered.any()):
      self._dynamic_step_index[recovered] += 1
      self._dynamic_preferred_side[recovered] = torch.where(
        self._dynamic_preferred_side[recovered] == int(LeadSide.LEFT),
        torch.full_like(self._dynamic_preferred_side[recovered], int(LeadSide.RIGHT)),
        torch.full_like(self._dynamic_preferred_side[recovered], int(LeadSide.LEFT)),
      )
      phase[recovered] = int(StairPhase.APPROACH)
      self._dynamic_phase_elapsed[recovered] = 0.0
      self._dynamic_step_progress[recovered] = 0.0
      self._dynamic_lead_side[recovered] = int(LeadSide.NONE)
      self._dynamic_left_streak[recovered] = 0
      self._dynamic_right_streak[recovered] = 0
      self._dynamic_left_loaded[recovered] = False
      self._dynamic_right_loaded[recovered] = False
      self._dynamic_trail_contact_elapsed[recovered] = -1.0
      self._dynamic_recover_stable[recovered] = 0
      self._dynamic_traversal_mode[recovered] = 0
      self._dynamic_abort_code[recovered] = 0

    # ABORT is an episode-level unsafe fact even if a later request/reset would
    # otherwise clear the current traversal label before curriculum scoring.
    self._dynamic_episode_unsafe.logical_or_(
      request & (phase == int(StairPhase.ABORT))
    )

    # Compose the same bounded split/lift reference as the scalar contract.
    phase = self._dynamic_phase
    split_fraction = torch.zeros(self.num_envs, device=self.device)
    preload = phase == int(StairPhase.PRELOAD)
    split_fraction[preload] = torch.clamp(
      self._dynamic_phase_elapsed[preload] / maneuver.preload_duration_s,
      0.0,
      1.0,
    )
    held_split = (
      (phase == int(StairPhase.CONTACT_WAIT))
      | (phase == int(StairPhase.LEAD_LIFT))
      | (phase == int(StairPhase.TRAIL_LIFT))
    )
    split_fraction[held_split] = 1.0
    recover = phase == int(StairPhase.RECOVER)
    split_fraction[recover] = torch.clamp(
      1.0 - self._dynamic_phase_elapsed[recover] / maneuver.recover_duration_s,
      0.0,
      1.0,
    )
    chosen_left = torch.where(
      self._dynamic_lead_side != int(LeadSide.NONE),
      self._dynamic_lead_side,
      self._dynamic_preferred_side,
    ) == int(LeadSide.LEFT)
    left_sign = torch.where(
      chosen_left,
      torch.ones(self.num_envs, device=self.device),
      -torch.ones(self.num_envs, device=self.device),
    )
    split = self._dynamic_candidate_parameters[:, 0] * split_fraction
    self._dynamic_leg_feedforward[:, 0] += left_sign * split * self._dynamic_split_left[0]
    self._dynamic_leg_feedforward[:, 2] += left_sign * split * self._dynamic_split_left[1]
    self._dynamic_leg_feedforward[:, 1] -= left_sign * split * self._dynamic_split_right[0]
    self._dynamic_leg_feedforward[:, 3] -= left_sign * split * self._dynamic_split_right[1]

    def bump(elapsed: torch.Tensor) -> torch.Tensor:
      inside = (elapsed > 0.0) & (elapsed < maneuver.lift_duration_s)
      value = 0.5 * (
        1.0
        - torch.cos(
          2.0 * math.pi * elapsed / maneuver.lift_duration_s
        )
      )
      return torch.where(inside, value, torch.zeros_like(value))

    lead_phase = phase == int(StairPhase.LEAD_LIFT)
    lead_lift = (
      self._dynamic_candidate_parameters[:, 1]
      * bump(self._dynamic_phase_elapsed)
    )
    if maneuver.lift_mode == DynamicLiftMode.SYNCHRONIZED:
      left_lift_mask = lead_phase
      right_lift_mask = lead_phase
    else:
      left_lift_mask = lead_phase & chosen_left
      right_lift_mask = lead_phase & ~chosen_left
    self._dynamic_leg_feedforward[left_lift_mask, 0] += (
      lead_lift[left_lift_mask] * self._dynamic_lift_left[0]
    )
    self._dynamic_leg_feedforward[left_lift_mask, 2] += (
      lead_lift[left_lift_mask] * self._dynamic_lift_left[1]
    )
    self._dynamic_leg_feedforward[right_lift_mask, 1] += (
      lead_lift[right_lift_mask] * self._dynamic_lift_right[0]
    )
    self._dynamic_leg_feedforward[right_lift_mask, 3] += (
      lead_lift[right_lift_mask] * self._dynamic_lift_right[1]
    )
    if maneuver.lift_mode == DynamicLiftMode.ALTERNATING:
      trail_phase = phase == int(StairPhase.TRAIL_LIFT)
      trail_lift = (
        self._dynamic_candidate_parameters[:, 1]
        * bump(self._dynamic_phase_elapsed)
      )
      trail_left = trail_phase & ~chosen_left
      trail_right = trail_phase & chosen_left
      self._dynamic_leg_feedforward[trail_left, 0] += (
        trail_lift[trail_left] * self._dynamic_lift_left[0]
      )
      self._dynamic_leg_feedforward[trail_left, 2] += (
        trail_lift[trail_left] * self._dynamic_lift_left[1]
      )
      self._dynamic_leg_feedforward[trail_right, 1] += (
        trail_lift[trail_right] * self._dynamic_lift_right[0]
      )
      self._dynamic_leg_feedforward[trail_right, 3] += (
        trail_lift[trail_right] * self._dynamic_lift_right[1]
      )
    self._dynamic_leg_feedforward.clamp_(
      -DYNAMIC_STAIR_FEEDFORWARD_LIMIT_RAD,
      DYNAMIC_STAIR_FEEDFORWARD_LIMIT_RAD,
    )
    drive = (
      (phase == int(StairPhase.LEAD_LIFT))
      | (phase == int(StairPhase.TRAIL_LIFT))
    )
    self._dynamic_drive_feedforward[drive] = self._dynamic_candidate_parameters[
      drive, 3
    ]
    self._dynamic_drive_feedforward[recover] = (
      self._dynamic_candidate_parameters[recover, 3]
      * torch.clamp(
        1.0
        - self._dynamic_phase_elapsed[recover]
        / maneuver.recover_duration_s,
        0.0,
        1.0,
      )
    )

  def process_actions(self, actions: torch.Tensor) -> None:
    if actions.shape != self._raw_actions.shape:
      raise ValueError(
        f"Hybrid action expects shape {tuple(self._raw_actions.shape)}, "
        f"got {tuple(actions.shape)}."
      )
    self._previous_previous_applied_residual[:] = (
      self._previous_applied_residual
    )
    self._previous_applied_residual[:] = self._applied_residual
    self._raw_actions[:] = actions
    self._applied_residual[:] = (
      torch.clamp(actions, -1.0, 1.0) * self._mask * self._scales
    )
    # Measure, then decide, then act - once per control step. The contact
    # sample read here is the one produced by the previous step's physics,
    # which is the only sample a causal loop can act on; the trigger
    # therefore responds one control step (20 ms at 50 Hz) after the third
    # consecutive above-threshold sample. The frozen C2-j3 replay measured a
    # weakest-pair above-threshold run of 4 ticks against the 3-tick window,
    # so this latency does not consume the detection margin.
    was_latched = self._stair_mode.clone()
    self._update_stair_mode()

    velocity_command = self._env.command_manager.get_command(
      self.cfg.velocity_command_name
    )
    posture_command = self._env.command_manager.get_command(
      self.cfg.posture_command_name
    )
    projected_gravity = self._entity.data.projected_gravity_b
    pitch = torch.atan2(
      projected_gravity[:, 0],
      torch.clamp(-projected_gravity[:, 2], min=1.0e-6),
    )
    pitch_rate = self._entity.data.root_link_ang_vel_b[:, 1]
    measured_vx = self._entity.data.root_link_lin_vel_b[:, 0]
    wheel_speed = self._entity.data.joint_vel[:, self._wheel_ids]
    self._update_dynamic_stair(
      pitch=pitch,
      pitch_rate=pitch_rate,
      projected_gravity=projected_gravity,
    )
    if self._dynamic_enabled:
      dynamic_abort = self._dynamic_stair_request & (
        self._dynamic_phase == int(StairPhase.ABORT)
      )
      control_velocity_command, control_residual = (
        stair_dynamic_safe_control_inputs(
          velocity_command, self._applied_residual, dynamic_abort
        )
      )
    else:
      # Preserve the frozen Stage0-5 hot path without even a device sync.
      control_velocity_command = velocity_command
      control_residual = self._applied_residual
    if self._noise_enabled:
      # Probe-only: corrupt the classical layer's direct sensor reads the
      # way a real IMU/encoder would. The policy observation pipeline has
      # its own noise fields; this covers the LQR path they cannot reach.
      pitch = pitch + self._sensor_noise(
        pitch.shape, self.cfg.sensor_noise_pitch_std
      )
      pitch_rate = pitch_rate + self._sensor_noise(
        pitch_rate.shape, self.cfg.sensor_noise_pitch_rate_std
      )
      measured_vx = measured_vx + self._sensor_noise(
        measured_vx.shape, self.cfg.sensor_noise_vx_std
      )
      wheel_speed = wheel_speed + self._sensor_noise(
        wheel_speed.shape, self.cfg.sensor_noise_wheel_vel_std
      )
    # Stage 3.0: the classical layer owns station keeping across the posture
    # envelope. The probe measured a steady drift, affine in the commanded
    # pitch, from holding any pitch other than the identified LQR reference;
    # subtracting the interpolated drift from the velocity reference cancels
    # it. The fallback breakpoints are identically zero and the artifact is
    # only loaded for posture-commanding stages, so stages 0-2 are unchanged.
    station_drift = _torch_linear_interpolate(
      posture_command[:, 1],
      self._station_pitch,
      self._station_drift,
    )
    calibrated_vx = (
      self.cfg.velocity_command_scale * control_velocity_command[:, 0]
      + self.cfg.velocity_command_bias
      - station_drift
    )
    vx_error = measured_vx - calibrated_vx
    signed_wheel_speed = 0.5 * (wheel_speed[:, 1] - wheel_speed[:, 0])
    desired_wheel_speed = calibrated_vx / self.cfg.wheel_radius
    wheel_speed_error = signed_wheel_speed - desired_wheel_speed
    state = torch.stack((pitch, pitch_rate, vx_error, wheel_speed_error), dim=1)
    equilibrium_input = torch.zeros_like(pitch)
    if self._schedule_enabled:
      scheduled_gain = _torch_bilinear_interpolate(
        posture_command[:, 0],
        posture_command[:, 1],
        self._schedule_heights,
        self._schedule_pitches,
        self._schedule_gains,
      )
      equilibrium_state = _torch_bilinear_interpolate(
        posture_command[:, 0],
        posture_command[:, 1],
        self._schedule_heights,
        self._schedule_pitches,
        self._schedule_equilibrium_state,
      )
      equilibrium_input = _torch_bilinear_interpolate(
        posture_command[:, 0],
        posture_command[:, 1],
        self._schedule_heights,
        self._schedule_pitches,
        self._schedule_equilibrium_input,
      )
      state = state - equilibrium_state
    # The residual stair camp observes what the classical layer is actually
    # regulating to zero, which under gain scheduling is the equilibrium-
    # relative state. Without a schedule the subtraction above is skipped and
    # this is the raw LQR state, so the six frozen stages are unaffected -
    # nothing reads it unless the camp wires the observation term.
    self._classical_errors[:] = state
    feedback = (
      torch.sum(state * scheduled_gain, dim=1)
      if self._schedule_enabled
      else state @ self._gain
    )
    if self._stair_mode_lqr_gain_scale != 1.0:
      gain_scale = torch.where(
        self._stair_mode,
        torch.full_like(feedback, self._stair_mode_lqr_gain_scale),
        torch.ones_like(feedback),
      )
      feedback = feedback * gain_scale
    control = equilibrium_input - feedback
    # The classical layer owns nominal yaw: the probe-fitted feedforward maps
    # the commanded yaw rate to a same-sign wheel differential and is part of
    # the baseline, so observations, gate collectors, and the compose contract
    # all see it as controller output rather than residual authority.
    yaw_feedforward = _torch_linear_interpolate(
      control_velocity_command[:, 2],
      self._yaw_feedforward_wz,
      self._yaw_feedforward_diff,
    )
    self._controller_baseline[:, 0] = (
      -control + yaw_feedforward - self._dynamic_drive_feedforward
    )
    self._controller_baseline[:, 1] = (
      control + yaw_feedforward + self._dynamic_drive_feedforward
    )

    balance_residual = control_residual[:, 0]
    yaw_residual = control_residual[:, 1]
    desired_wheels = self._controller_baseline.clone()
    desired_wheels[:, 0] += -balance_residual + yaw_residual
    desired_wheels[:, 1] += balance_residual + yaw_residual
    features = torch.stack(
      (
        torch.ones(self.num_envs, device=self.device),
        posture_command[:, 0],
        posture_command[:, 1],
      ),
      dim=1,
    )
    self._nominal_leg_targets[:] = features @ self._posture_coefficients
    if self._stair_freeze_enabled:
      # Deviation minute 3 (freeze-at-trigger). `was_latched` is the state
      # BEFORE this step's trigger update, so on the rising edge the
      # reference first adopts this step's posture target and only then
      # holds: the switch is continuous, with no jump at the trigger. From
      # the next step on the posture map can no longer pull the legs back
      # and fight the residual, which is the interference this switch exists
      # to remove, and the residual then perturbs a held pose - exactly the
      # regime the [U]-confirmed physics table measured.
      self._leg_reference[:] = torch.where(
        was_latched.unsqueeze(1),
        self._leg_reference,
        self._nominal_leg_targets,
      )
    else:
      self._leg_reference[:] = self._nominal_leg_targets
    desired_legs = (
      self._leg_reference
      + self._dynamic_leg_feedforward
      + control_residual[:, 2:]
    )
    soft_limits = self._entity.data.soft_joint_pos_limits[:, self._leg_ids]
    if self._dynamic_enabled:
      dynamic_active = self._dynamic_stair_request & (
        self._dynamic_phase != int(StairPhase.IDLE)
      )
      target_saturation = stair_dynamic_target_saturation_mask(
        desired_legs, soft_limits, dynamic_active
      )
      if bool(target_saturation.any()):
        # A composed target outside the registered actuator envelope is itself
        # unsafe. Abort on this same control step, remove FF/PPO authority, and
        # slew the wheels toward zero instead of silently clipping and driving on.
        self._dynamic_phase[target_saturation] = int(StairPhase.ABORT)
        self._dynamic_phase_elapsed[target_saturation] = 0.0
        self._dynamic_traversal_mode[target_saturation] = 3
        self._dynamic_abort_code[target_saturation] = 8
        self._dynamic_episode_unsafe[target_saturation] = True
        self._dynamic_target_saturation[target_saturation] = True
        self._dynamic_riser_cross_event[target_saturation] = False
        self._dynamic_leg_feedforward[target_saturation] = 0.0
        self._dynamic_drive_feedforward[target_saturation] = 0.0
        self._controller_baseline[target_saturation] = 0.0
        desired_legs[target_saturation] = self._leg_reference[target_saturation]
        desired_wheels[target_saturation] = 0.0

    wheel_delta = torch.clamp(
      desired_wheels - self._previous_wheel_targets,
      -self.cfg.wheel_slew_limit,
      self.cfg.wheel_slew_limit,
    )
    self._wheel_targets[:] = torch.clamp(
      self._previous_wheel_targets + wheel_delta,
      -self.cfg.wheel_velocity_limit,
      self.cfg.wheel_velocity_limit,
    )
    self._previous_wheel_targets[:] = self._wheel_targets
    self._leg_targets[:] = torch.clamp(
      desired_legs,
      min=soft_limits[..., 0],
      max=soft_limits[..., 1],
    )
    if self._delay_steps > 0:
      self._delay_dirty = True

  def _sensor_noise(
    self,
    shape: torch.Size,
    std: float,
  ) -> torch.Tensor:
    if std <= 0.0:
      return torch.zeros(shape, device=self.device)
    return std * torch.randn(
      shape,
      generator=self._noise_generator,
      device=self.device,
    )

  def apply_actions(self) -> None:
    wheel_targets = self._wheel_targets
    leg_targets = self._leg_targets
    if self._delay_steps > 0:
      # Probe-only control-loop latency: apply the targets composed
      # delay_steps control steps ago. The ring holds exactly the last
      # delay_steps compositions; the pointer wraps once per env step
      # (apply_actions runs once per physics substep, so only rotate when
      # a fresh composition arrived).
      if self._delay_dirty:
        delayed_wheels = self._delayed_wheel_ring[self._delay_pointer].clone()
        delayed_legs = self._delayed_leg_ring[self._delay_pointer].clone()
        self._delayed_wheel_ring[self._delay_pointer] = self._wheel_targets
        self._delayed_leg_ring[self._delay_pointer] = self._leg_targets
        self._delay_pointer = (self._delay_pointer + 1) % self._delay_steps
        self._delayed_wheel_output = delayed_wheels
        self._delayed_leg_output = delayed_legs
        self._delay_dirty = False
      wheel_targets = self._delayed_wheel_output
      leg_targets = self._delayed_leg_output
    self._entity.set_joint_velocity_target(
      wheel_targets,
      joint_ids=self._wheel_ids,
    )
    encoder_bias = self._entity.data.encoder_bias[:, self._leg_ids]
    self._entity.set_joint_position_target(
      leg_targets - encoder_bias,
      joint_ids=self._leg_ids,
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._applied_residual[env_ids] = 0.0
    self._previous_applied_residual[env_ids] = 0.0
    self._previous_previous_applied_residual[env_ids] = 0.0
    self._controller_baseline[env_ids] = 0.0
    self._classical_errors[env_ids] = 0.0
    self._previous_wheel_targets[env_ids] = 0.0
    self._wheel_targets[env_ids] = 0.0
    initial = torch.tensor(
      [INIT_JOINT_POS[name] for name in self.cfg.leg_joint_names],
      device=self.device,
      dtype=torch.float,
    )
    self._nominal_leg_targets[env_ids] = initial
    self._leg_targets[env_ids] = initial
    # S5B Protocol 2: `stair_mode` latches ON once triggered and resets only
    # at episode reset. There is no mid-episode exit path.
    self._leg_reference[env_ids] = initial
    self._stair_trigger_streak[env_ids] = 0
    self._stair_mode[env_ids] = self._stair_mode_forced
    ids = (
      torch.arange(self.num_envs, device=self.device)[env_ids]
      if isinstance(env_ids, slice)
      else env_ids
    )
    self._dynamic_stair_request[env_ids] = False
    self._dynamic_phase[env_ids] = int(StairPhase.IDLE)
    self._dynamic_phase_elapsed[env_ids] = 0.0
    self._dynamic_step_progress[env_ids] = 0.0
    self._dynamic_step_index[env_ids] = 0
    self._dynamic_preferred_side[env_ids] = torch.where(
      ids.remainder(2) == 0,
      torch.full_like(ids, int(LeadSide.LEFT)),
      torch.full_like(ids, int(LeadSide.RIGHT)),
    )
    self._dynamic_lead_side[env_ids] = int(LeadSide.NONE)
    self._dynamic_left_streak[env_ids] = 0
    self._dynamic_right_streak[env_ids] = 0
    self._dynamic_left_loaded[env_ids] = False
    self._dynamic_right_loaded[env_ids] = False
    self._dynamic_trail_contact_elapsed[env_ids] = -1.0
    self._dynamic_left_force[env_ids] = 0.0
    self._dynamic_right_force[env_ids] = 0.0
    self._dynamic_recover_stable[env_ids] = 0
    self._dynamic_traversal_mode[env_ids] = 0
    self._dynamic_abort_code[env_ids] = 0
    self._dynamic_episode_unsafe[env_ids] = False
    self._dynamic_target_saturation[env_ids] = False
    self._dynamic_leg_feedforward[env_ids] = 0.0
    self._dynamic_drive_feedforward[env_ids] = 0.0
    self._dynamic_riser_cross_event[env_ids] = False
    self._dynamic_previous_root_x[env_ids] = self._entity.data.root_link_pos_w[
      env_ids, 0
    ]
    if self._delay_steps > 0:
      # Reset envs restart from the freshly composed neutral targets; a
      # stale ring would leak pre-reset actions across the episode boundary.
      self._delayed_wheel_ring[:, env_ids] = 0.0
      self._delayed_leg_ring[:, env_ids] = initial
      self._delayed_wheel_output[env_ids] = 0.0
      self._delayed_leg_output[env_ids] = initial


def _hybrid_action_term(
  env: ManagerBasedRlEnv,
  attribute: str,
) -> torch.Tensor:
  term = env.action_manager.get_term("hybrid_wheel_leg")
  value = getattr(term, attribute)
  if not isinstance(value, torch.Tensor):
    raise TypeError(f"Hybrid action attribute '{attribute}' must be a tensor.")
  return value


def controller_baseline_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _hybrid_action_term(env, "controller_baseline")


def applied_residual_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _hybrid_action_term(env, "applied_residual")


def classical_errors_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _hybrid_action_term(env, "classical_errors")


def leg_reference_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The leg pose the residual acts about.

  S5B registers this actor channel as `nominal_leg_targets`. Under
  freeze-at-trigger (deviation minute 3) the classical layer's leg output
  IS the latched reference once `stair_mode` holds, so this publishes
  `leg_reference` rather than the posture-map value: the policy must see
  the pose it is actually perturbing, or its residual is credited against
  a reference it does not act on. Before the trigger the two are equal by
  construction, so this is a strict superset of the registered semantics.
  """

  return _hybrid_action_term(env, "leg_reference")


def stair_mode_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
  term = env.action_manager.get_term("hybrid_wheel_leg")
  return term.stair_mode.to(torch.float).unsqueeze(-1)


def stair_phase_one_hot_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Constant IDLE one-hot, kept for interface stability (S5B Protocol 1).

  The camp pins `stair_maneuver = None`, so the classical stair FSM never
  runs and no phase other than IDLE is reachable. The channel stays in the
  contract so a later maneuver-enabled variant keeps the same observation
  layout; a contract test asserts it is informationless here.
  """

  one_hot = torch.zeros(env.num_envs, PHASE_COUNT, device=env.device)
  one_hot[:, int(StairPhase.IDLE)] = 1.0
  return one_hot


def stair_dynamic_request_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _hybrid_action_term(env, "dynamic_stair_request").to(torch.float).unsqueeze(-1)


def stair_dynamic_phase_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
  phase = _hybrid_action_term(env, "dynamic_phase").long()
  one_hot = torch.zeros(env.num_envs, PHASE_COUNT, device=env.device)
  one_hot.scatter_(1, phase.unsqueeze(1), 1.0)
  return one_hot


def stair_dynamic_loaded_contact_observation(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  return _hybrid_action_term(env, "dynamic_loaded_contact").to(torch.float)


def stair_dynamic_lead_side_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
  side = _hybrid_action_term(env, "dynamic_lead_side").long()
  return torch.stack(
    (
      (side == int(LeadSide.LEFT)).to(torch.float),
      (side == int(LeadSide.RIGHT)).to(torch.float),
    ),
    dim=1,
  )


def stair_dynamic_leg_feedforward_observation(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  return _hybrid_action_term(env, "dynamic_leg_feedforward")


def _roll_assist_action(env: ManagerBasedRlEnv) -> HybridWheelLegAction:
  term = env.action_manager.get_term("hybrid_wheel_leg")
  if not isinstance(term, HybridWheelLegAction):
    raise TypeError("RollAssist hybrid action term is missing.")
  return term


def roll_assist_step_height_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
  terrain = env.scene.terrain
  if terrain is None:
    raise ValueError("RollAssist requires generated terrain.")
  state = getattr(env, "roll_assist_curriculum_state", None)
  if state is None:
    hpass = float(env.cfg.roll_assist_hpass_m)
    hnext = float(env.cfg.roll_assist_hnext_m)
    height = torch.where(
      terrain.terrain_levels == 0,
      torch.full_like(terrain.terrain_levels, hpass, dtype=torch.float),
      torch.full_like(terrain.terrain_levels, hnext, dtype=torch.float),
    )
  else:
    curriculum = state.state
    height = torch.where(
      terrain.terrain_levels == 0,
      torch.full_like(terrain.terrain_levels, curriculum.hpass_m, dtype=torch.float),
      torch.full_like(terrain.terrain_levels, curriculum.hnext_m, dtype=torch.float),
    )
  height[: int(getattr(env.cfg, "roll_assist_flat_env_count", 0))] = 0.0
  return height.unsqueeze(-1)


def roll_assist_distance_to_riser_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
  root_x = env.scene["robot"].data.root_link_pos_w[:, 0]
  riser_x = env.scene.env_origins[:, 0] - ROLL_ASSIST_RISER_OFFSET_M
  distance = riser_x - root_x
  flat = torch.arange(env.num_envs, device=env.device) < int(
    getattr(env.cfg, "roll_assist_flat_env_count", 0)
  )
  return torch.where(flat, torch.zeros_like(distance), distance).unsqueeze(-1)


def _roll_assist_contact_force(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  force = sensor.data.force
  if force is None:
    raise RuntimeError("RollAssist exact wheel sensor exposes no force field.")
  return torch.linalg.vector_norm(force.reshape(force.shape[0], -1, 3), dim=-1).sum(dim=-1)


def roll_assist_left_contact_force_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _roll_assist_contact_force(env, ROLL_ASSIST_LEFT_SENSOR_NAME).unsqueeze(-1)


def roll_assist_right_contact_force_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _roll_assist_contact_force(env, ROLL_ASSIST_RIGHT_SENSOR_NAME).unsqueeze(-1)


def roll_assist_wheel_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  force = env.scene[sensor_name].data.force
  if force is None:
    raise RuntimeError("RollAssist exact wheel sensor exposes no force field.")
  magnitude = torch.linalg.vector_norm(force.reshape(force.shape[0], -1, 3), dim=-1)
  return torch.any(magnitude > 0.0, dim=-1)


def roll_assist_bilateral_airborne(env: ManagerBasedRlEnv) -> torch.Tensor:
  left = roll_assist_wheel_contact(env, ROLL_ASSIST_LEFT_SENSOR_NAME)
  right = roll_assist_wheel_contact(env, ROLL_ASSIST_RIGHT_SENSOR_NAME)
  flat_count = int(getattr(env.cfg, "roll_assist_flat_env_count", 0))
  stair = torch.arange(env.num_envs, device=env.device) >= flat_count
  return stair & ~left & ~right


def roll_assist_progress_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Positive stair-only speed clipped to the immutable 0.07 m/s command."""

  velocity = env.scene["robot"].data.root_link_lin_vel_w[:, 0].clamp(
    min=0.0, max=ROLL_ASSIST_COMMAND_VX_MPS
  )
  stair = torch.arange(env.num_envs, device=env.device) >= int(
    getattr(env.cfg, "roll_assist_flat_env_count", 0)
  )
  after_settle = env.episode_length_buf > ROLL_ASSIST_SETTLE_STEPS
  return velocity * (stair & after_settle).to(velocity.dtype)


class RollAssistEpisodeEvidence:
  """Latch continuous-contact safety and 25-step stable crossing before reset."""

  SETTLE_STEPS = ROLL_ASSIST_SETTLE_STEPS

  def __init__(self, cfg: object, env: ManagerBasedRlEnv):
    del cfg
    self._env = env
    env.roll_assist_episode_evidence = self  # type: ignore[attr-defined]
    self.success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.stable_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.bilateral_airborne_ever = torch.zeros_like(self.success)
    self.left_unload_steps = torch.zeros_like(self.stable_steps)
    self.right_unload_steps = torch.zeros_like(self.stable_steps)
    self.max_progress_m = torch.full(
      (env.num_envs,), -math.inf, dtype=torch.float, device=env.device
    )
    self.wheel_residual_abs_max = torch.zeros(env.num_envs, device=env.device)

  def _support_state(
    self, env: ManagerBasedRlEnv,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_count = int(getattr(env.cfg, "roll_assist_flat_env_count", 0))
    stair = torch.arange(env.num_envs, device=env.device) >= flat_count
    left = roll_assist_wheel_contact(env, ROLL_ASSIST_LEFT_SENSOR_NAME)
    right = roll_assist_wheel_contact(env, ROLL_ASSIST_RIGHT_SENSOR_NAME)
    return stair, left, right

  def record_physics_substep(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Latch any unsupported 5 ms physics state before control-step reset."""

    stair, left, right = self._support_state(env)
    airborne = ~left & ~right
    self.bilateral_airborne_ever |= airborne
    self.left_unload_steps += (stair & ~left).long()
    self.right_unload_steps += (stair & ~right).long()
    return airborne

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    stair, left, right = self._support_state(env)
    after_settle = env.episode_length_buf > self.SETTLE_STEPS
    # The per-substep metric has already latched all four 5 ms states.  OR the
    # current control sample as a fail-closed guard; never clear an earlier hit.
    airborne = ~left & ~right
    self.bilateral_airborne_ever |= airborne
    robot = env.scene["robot"].data
    riser_x = env.scene.env_origins[:, 0] - ROLL_ASSIST_RISER_OFFSET_M
    progress = robot.root_link_pos_w[:, 0] - riser_x
    self.max_progress_m.copy_(torch.maximum(self.max_progress_m, progress))
    gravity = robot.projected_gravity_b
    pitch = torch.atan2(
      gravity[:, 0], torch.clamp(-gravity[:, 2], min=1.0e-6)
    )
    roll = torch.atan2(
      -gravity[:, 1], torch.clamp(-gravity[:, 2], min=1.0e-6)
    )
    stable = (
      stair
      & after_settle
      & ~self.bilateral_airborne_ever
      & (progress >= ROLL_ASSIST_CROSS_DEPTH_M)
      & (pitch.abs() <= ROLL_ASSIST_PITCH_LIMIT_RAD)
      & (roll.abs() <= ROLL_ASSIST_ROLL_LIMIT_RAD)
      & (robot.root_link_ang_vel_b[:, 1].abs() <= ROLL_ASSIST_PITCH_RATE_LIMIT_RADPS)
    )
    self.stable_steps.copy_(torch.where(stable, self.stable_steps + 1, 0))
    self.success |= self.stable_steps >= ROLL_ASSIST_STABLE_STEPS
    residual = _roll_assist_action(env).applied_residual
    self.wheel_residual_abs_max.copy_(torch.maximum(
      self.wheel_residual_abs_max, residual[:, :2].abs().amax(dim=1)
    ))
    self.success &= ~self.bilateral_airborne_ever
    return self.bilateral_airborne_ever

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.success[ids] = False
    self.stable_steps[ids] = 0
    self.bilateral_airborne_ever[ids] = False
    self.left_unload_steps[ids] = 0
    self.right_unload_steps[ids] = 0
    self.max_progress_m[ids] = -math.inf
    self.wheel_residual_abs_max[ids] = 0.0


def _roll_assist_episode_evidence(env: ManagerBasedRlEnv) -> RollAssistEpisodeEvidence:
  state = getattr(env, "roll_assist_episode_evidence", None)
  if state is None:
    raise RuntimeError("RollAssist episode evidence metric is not installed.")
  return state


def roll_assist_stable_success(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _roll_assist_episode_evidence(env).success.to(torch.float)


def roll_assist_substep_support_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Record strict support every physics substep; output is diagnostic only."""

  return _roll_assist_episode_evidence(env).record_physics_substep(env).to(torch.float)


def roll_assist_episode_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  state = _roll_assist_episode_evidence(env)
  curriculum = getattr(env, "roll_assist_curriculum_state", None)
  if curriculum is not None:
    curriculum.record_step(env)
  return state.success.to(torch.float)


def roll_assist_actor_terms(stage5_terms: Mapping[str, ObservationTermCfg]) -> dict[str, ObservationTermCfg]:
  if tuple(stage5_terms) != ROLL_ASSIST_ACTOR_TERMS:
    raise ValueError("RollAssist no longer preserves the exact Stage5 34-D actor prefix.")
  return dict(stage5_terms)


def roll_assist_critic_terms(actor_terms: Mapping[str, ObservationTermCfg]) -> dict[str, ObservationTermCfg]:
  terms = {
    name: ObservationTermCfg(func=term.func, params=dict(term.params))
    for name, term in actor_terms.items()
  }
  terms.update({
    "step_height": ObservationTermCfg(func=roll_assist_step_height_observation),
    "distance_to_riser": ObservationTermCfg(func=roll_assist_distance_to_riser_observation),
    "left_contact_force": ObservationTermCfg(func=roll_assist_left_contact_force_observation),
    "right_contact_force": ObservationTermCfg(func=roll_assist_right_contact_force_observation),
  })
  return terms


def stair_dynamic_step_height_observation(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  terrain = env.scene.terrain
  if terrain is None:
    raise ValueError("StairDynamic step height requires generated terrain.")
  return (
    terrain.terrain_levels.to(torch.float) * DYNAMIC_STAIR_HEIGHT_STEP_M
  ).unsqueeze(-1)


def stair_dynamic_distance_to_next_riser_observation(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  terrain = env.scene.terrain
  if terrain is None:
    raise ValueError("StairDynamic riser distance requires generated terrain.")
  action = env.action_manager.get_term("hybrid_wheel_leg")
  root_x = env.scene["robot"].data.root_link_pos_w[:, 0]
  first_riser = env.scene.env_origins[:, 0] - DYNAMIC_STAIR_RISER_OFFSET_M
  next_riser = first_riser + action.dynamic_step_index.to(torch.float) * (
    DYNAMIC_STAIR_STEP_WIDTH_M
  )
  return (next_riser - root_x).unsqueeze(-1)


def stair_dynamic_left_contact_force_observation(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  return _hybrid_action_term(env, "dynamic_contact_force")[:, :1]


def stair_dynamic_right_contact_force_observation(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  return _hybrid_action_term(env, "dynamic_contact_force")[:, 1:2]


def stair_dynamic_progress_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Contact-gated progress capped at the registered +0.07 m/s command."""

  action = env.action_manager.get_term("hybrid_wheel_leg")
  loaded = action.dynamic_loaded_contact.any(dim=1)
  requested = action.dynamic_stair_request
  forward = env.scene["robot"].data.root_link_lin_vel_w[:, 0].clamp(
    min=0.0,
    max=DYNAMIC_STAIR_COMMAND_VX_MPS,
  )
  return forward * (loaded & requested).to(forward.dtype)


def stair_dynamic_riser_event_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  """One control-step pulse whose dt-scaled integral is exactly one."""

  action = env.action_manager.get_term("hybrid_wheel_leg")
  return action.dynamic_riser_cross_event.to(torch.float) / action.cfg.dynamic_stair_control_dt


def stair_dynamic_three_step_success_mask(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  """Require three recovered risers with no episode unsafe/termination event."""

  action = env.action_manager.get_term("hybrid_wheel_leg")
  unsafe = action.dynamic_episode_unsafe
  termination_manager = getattr(env, "termination_manager", None)
  terminated = getattr(termination_manager, "terminated", None)
  if terminated is None:
    terminated = torch.zeros_like(unsafe)
  return (action.dynamic_step_index >= 3) & ~unsafe & ~terminated.bool()


def stair_dynamic_three_step_success(env: ManagerBasedRlEnv) -> torch.Tensor:
  return stair_dynamic_three_step_success_mask(env).to(torch.float)


def stair_dynamic_actor_terms(
  base_terms: Mapping[str, ObservationTermCfg],
) -> dict[str, ObservationTermCfg]:
  """Append the exact 18-dimensional v3 tail to the Stage5 34-D prefix."""

  expected_prefix = DYNAMIC_STAIR_ACTOR_TERMS[:9]
  if tuple(base_terms) != expected_prefix:
    raise ValueError("StairDynamic requires the unchanged Stage5 actor prefix.")
  result = dict(base_terms)
  result.update(
    {
      "stair_request": ObservationTermCfg(
        func=stair_dynamic_request_observation
      ),
      "phase_one_hot": ObservationTermCfg(
        func=stair_dynamic_phase_observation
      ),
      "loaded_contact": ObservationTermCfg(
        func=stair_dynamic_loaded_contact_observation
      ),
      "lead_side": ObservationTermCfg(
        func=stair_dynamic_lead_side_observation
      ),
      "leg_feedforward": ObservationTermCfg(
        func=stair_dynamic_leg_feedforward_observation
      ),
    }
  )
  if tuple(result) != DYNAMIC_STAIR_ACTOR_TERMS:
    raise ValueError("StairDynamic actor term order drifted.")
  return result


def stair_dynamic_critic_terms(
  actor_terms: Mapping[str, ObservationTermCfg],
) -> dict[str, ObservationTermCfg]:
  """Critic = actor plus four registered terrain/contact fields."""

  if tuple(actor_terms) != DYNAMIC_STAIR_ACTOR_TERMS:
    raise ValueError("StairDynamic critic requires the exact actor prefix.")
  result = {
    name: ObservationTermCfg(func=term.func, params=dict(term.params))
    for name, term in actor_terms.items()
  }
  result.update(
    {
      "step_height": ObservationTermCfg(
        func=stair_dynamic_step_height_observation
      ),
      "distance_to_next_riser": ObservationTermCfg(
        func=stair_dynamic_distance_to_next_riser_observation
      ),
      "left_contact_force": ObservationTermCfg(
        func=stair_dynamic_left_contact_force_observation
      ),
      "right_contact_force": ObservationTermCfg(
        func=stair_dynamic_right_contact_force_observation
      ),
    }
  )
  validate_dynamic_stair_observation_layout(dict(actor_terms), result)
  return result


class StairDynamicCurriculum:
  """Exact 50-update, three-window curriculum for the 192 stair slots."""

  STATE_SCHEMA_VERSION = 1

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    evaluation_interval_steps: int,
    flat_env_count: int = DYNAMIC_STAIR_FLAT_ENVS,
    initial_upper_height_m: float = DYNAMIC_STAIR_HEIGHT_STEP_M,
  ):
    if evaluation_interval_steps < 1:
      raise ValueError("StairDynamic evaluation interval must be positive.")
    if flat_env_count < 0 or flat_env_count > env.num_envs:
      raise ValueError("StairDynamic flat env count is invalid.")
    initial_level = round(initial_upper_height_m / DYNAMIC_STAIR_HEIGHT_STEP_M)
    if (
      initial_level < 1
      or initial_level >= DYNAMIC_STAIR_TERRAIN_ROWS
      or not math.isclose(
        initial_upper_height_m,
        initial_level * DYNAMIC_STAIR_HEIGHT_STEP_M,
        rel_tol=0.0,
        abs_tol=1.0e-12,
      )
    ):
      raise ValueError("StairDynamic initial height must be 0.01, 0.02, or 0.03 m.")
    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
      raise ValueError("StairDynamic curriculum requires generated terrain.")
    self.state = StairCurriculumState(
      lower_height_m=DYNAMIC_STAIR_HEIGHT_STEP_M,
      upper_height_m=float(initial_upper_height_m),
    )
    self.evaluation_interval_steps = int(evaluation_interval_steps)
    self.flat_env_count = int(flat_env_count)
    self.next_evaluation_step = self.evaluation_interval_steps
    self.episodes_at_upper = 0
    self.successes_at_upper = 0
    self.completed_stair_episodes = 0
    self.successful_stair_episodes = 0
    self.evaluations = 0
    self.last_processed_step = -1
    self.started = False

  @property
  def upper_level(self) -> int:
    return round(self.state.upper_height_m / DYNAMIC_STAIR_HEIGHT_STEP_M)

  def _score_finished_episodes(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
  ) -> None:
    stair_ids = env_ids[env_ids >= self.flat_env_count]
    if len(stair_ids) == 0:
      return
    terrain = env.scene.terrain
    assert terrain is not None
    success = stair_dynamic_three_step_success_mask(env)[stair_ids]
    at_upper = terrain.terrain_levels[stair_ids] == self.upper_level
    self.completed_stair_episodes += len(stair_ids)
    self.successful_stair_episodes += int(success.sum().item())
    if bool(at_upper.any()):
      self.episodes_at_upper += int(at_upper.sum().item())
      self.successes_at_upper += int((success & at_upper).sum().item())

  def _evaluate_once(self) -> None:
    rate = (
      self.successes_at_upper / self.episodes_at_upper
      if self.episodes_at_upper
      else 0.0
    )
    self.state = update_stair_curriculum(
      self.state,
      success_rate=rate,
      maximum_height_m=DYNAMIC_STAIR_MAX_HEIGHT_M,
    )
    self.episodes_at_upper = 0
    self.successes_at_upper = 0
    self.evaluations += 1
    self.next_evaluation_step += self.evaluation_interval_steps

  def record_step(self, env: ManagerBasedRlEnv) -> None:
    step = int(env.common_step_counter)
    if step <= self.last_processed_step:
      return
    reset_buf = getattr(env, "reset_buf", None)
    if reset_buf is not None:
      finished = reset_buf.nonzero(as_tuple=False).squeeze(-1)
      if len(finished):
        self._score_finished_episodes(env, finished)
    while step >= self.next_evaluation_step:
      self._evaluate_once()
    self.last_processed_step = step

  def progress_snapshot(self) -> dict[str, float | int]:
    return {
      "upper_height_m": float(self.state.upper_height_m),
      "consecutive_ready_evaluations": int(
        self.state.consecutive_ready_evaluations
      ),
      "evaluations": self.evaluations,
      "completed_stair_episodes": self.completed_stair_episodes,
      "successful_stair_episodes": self.successful_stair_episodes,
      "stair_success_rate": (
        self.successful_stair_episodes
        / max(self.completed_stair_episodes, 1)
      ),
    }

  def state_dict(self) -> dict[str, object]:
    return {
      "schema_version": self.STATE_SCHEMA_VERSION,
      "lower_height_m": float(self.state.lower_height_m),
      "upper_height_m": float(self.state.upper_height_m),
      "consecutive_ready_evaluations": int(
        self.state.consecutive_ready_evaluations
      ),
      "evaluation_interval_steps": self.evaluation_interval_steps,
      "flat_env_count": self.flat_env_count,
      "next_evaluation_step": self.next_evaluation_step,
      "episodes_at_upper": self.episodes_at_upper,
      "successes_at_upper": self.successes_at_upper,
      "completed_stair_episodes": self.completed_stair_episodes,
      "successful_stair_episodes": self.successful_stair_episodes,
      "evaluations": self.evaluations,
      "last_processed_step": self.last_processed_step,
      "started": self.started,
    }

  def load_state_dict(self, payload: Mapping[str, object]) -> None:
    if not isinstance(payload, Mapping) or set(payload) != set(self.state_dict()):
      raise ValueError("StairDynamic curriculum state schema does not match.")

    def exact_int(name: str, minimum: int | None = None) -> int:
      value = payload[name]
      if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"StairDynamic curriculum {name} must be an integer.")
      result = int(value)
      if minimum is not None and result < minimum:
        raise ValueError(f"StairDynamic curriculum {name} is out of range.")
      return result

    def finite(name: str) -> float:
      value = payload[name]
      if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"StairDynamic curriculum {name} must be numeric.")
      result = float(value)
      if not math.isfinite(result):
        raise ValueError(f"StairDynamic curriculum {name} must be finite.")
      return result

    if exact_int("schema_version", 0) != self.STATE_SCHEMA_VERSION:
      raise ValueError("Unsupported StairDynamic curriculum schema.")
    interval = exact_int("evaluation_interval_steps", 1)
    flat_count = exact_int("flat_env_count", 0)
    if interval != self.evaluation_interval_steps or flat_count != self.flat_env_count:
      raise ValueError("StairDynamic curriculum configuration drifted.")
    lower = finite("lower_height_m")
    upper = finite("upper_height_m")
    upper_level = upper / DYNAMIC_STAIR_HEIGHT_STEP_M
    if (
      not math.isclose(
        lower,
        DYNAMIC_STAIR_HEIGHT_STEP_M,
        rel_tol=0.0,
        abs_tol=1.0e-12,
      )
      or upper < lower
      or upper > DYNAMIC_STAIR_MAX_HEIGHT_M
      or abs(upper_level - round(upper_level)) > 1.0e-9
    ):
      raise ValueError("StairDynamic curriculum height state is invalid.")
    ready = exact_int("consecutive_ready_evaluations", 0)
    episodes = exact_int("episodes_at_upper", 0)
    successes = exact_int("successes_at_upper", 0)
    completed = exact_int("completed_stair_episodes", 0)
    successful = exact_int("successful_stair_episodes", 0)
    evaluations = exact_int("evaluations", 0)
    next_step = exact_int("next_evaluation_step", interval)
    last_step = exact_int("last_processed_step")
    started = payload["started"]
    if not isinstance(started, bool):
      raise TypeError("StairDynamic curriculum started must be boolean.")
    if (
      ready > 2
      or successes > episodes
      or successful > completed
      or next_step != (evaluations + 1) * interval
      or last_step < -1
      or last_step >= next_step
    ):
      raise ValueError("StairDynamic curriculum counters are inconsistent.")
    self.state = StairCurriculumState(
      lower_height_m=lower,
      upper_height_m=upper,
      consecutive_ready_evaluations=ready,
    )
    self.next_evaluation_step = next_step
    self.episodes_at_upper = episodes
    self.successes_at_upper = successes
    self.completed_stair_episodes = completed
    self.successful_stair_episodes = successful
    self.evaluations = evaluations
    self.last_processed_step = last_step
    self.started = started

  def _assign_levels(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
  ) -> None:
    terrain = env.scene.terrain
    assert terrain is not None and terrain.terrain_origins is not None
    flat = env_ids < self.flat_env_count
    flat_ids = env_ids[flat]
    stair_ids = env_ids[~flat]
    if len(flat_ids):
      terrain.terrain_levels[flat_ids] = 0
    if len(stair_ids):
      terrain.terrain_levels[stair_ids] = torch.randint(
        1,
        self.upper_level + 1,
        (len(stair_ids),),
        device=terrain.terrain_levels.device,
        dtype=terrain.terrain_levels.dtype,
      )
    if len(env_ids):
      assert terrain.env_origins is not None
      terrain.env_origins[env_ids] = terrain.terrain_origins[
        terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
      ]

  def compute(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
  ) -> dict[str, float]:
    if len(env_ids):
      self._assign_levels(env, env_ids)
      self.started = True
    terrain = env.scene.terrain
    assert terrain is not None
    stair_levels = terrain.terrain_levels[self.flat_env_count :].float()
    return {
      "upper_height_m": float(self.state.upper_height_m),
      "consecutive_ready": float(self.state.consecutive_ready_evaluations),
      "evaluations": float(self.evaluations),
      "mean_stair_level": (
        float(stair_levels.mean().item()) if len(stair_levels) else 0.0
      ),
    }


def _stair_dynamic_curriculum_state(
  env: ManagerBasedRlEnv,
  evaluation_interval_steps: int,
  flat_env_count: int = DYNAMIC_STAIR_FLAT_ENVS,
  initial_upper_height_m: float = DYNAMIC_STAIR_HEIGHT_STEP_M,
) -> StairDynamicCurriculum:
  state = getattr(env, "stair_dynamic_curriculum_state", None)
  if state is None:
    state = StairDynamicCurriculum(
      env,
      evaluation_interval_steps,
      flat_env_count,
      initial_upper_height_m,
    )
    env.stair_dynamic_curriculum_state = state  # type: ignore[attr-defined]
  elif not isinstance(state, StairDynamicCurriculum):
    raise ValueError("StairDynamic curriculum state has an invalid type.")
  elif (
    state.evaluation_interval_steps != int(evaluation_interval_steps)
    or state.flat_env_count != int(flat_env_count)
  ):
    raise ValueError("StairDynamic curriculum configuration changed.")
  return state


def stair_dynamic_step_metric(
  env: ManagerBasedRlEnv,
  evaluation_interval_steps: int,
  flat_env_count: int = DYNAMIC_STAIR_FLAT_ENVS,
  initial_upper_height_m: float = DYNAMIC_STAIR_HEIGHT_STEP_M,
) -> torch.Tensor:
  state = _stair_dynamic_curriculum_state(
    env,
    evaluation_interval_steps,
    flat_env_count,
    initial_upper_height_m,
  )
  state.record_step(env)
  return stair_dynamic_three_step_success(env)


def stair_dynamic_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  evaluation_interval_steps: int,
  flat_env_count: int = DYNAMIC_STAIR_FLAT_ENVS,
  initial_upper_height_m: float = DYNAMIC_STAIR_HEIGHT_STEP_M,
) -> dict[str, float]:
  state = _stair_dynamic_curriculum_state(
    env,
    evaluation_interval_steps,
    flat_env_count,
    initial_upper_height_m,
  )
  all_ids = torch.arange(env.num_envs, device=env.device)
  if env_ids is None:
    ids = all_ids
  elif isinstance(env_ids, slice):
    ids = all_ids[env_ids]
  else:
    ids = env_ids
  return state.compute(env, ids)


def applied_residual_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  current = _hybrid_action_term(env, "applied_residual")
  previous = _hybrid_action_term(env, "previous_applied_residual")
  return torch.sum(torch.square(current - previous), dim=1)


def applied_residual_acc_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  current = _hybrid_action_term(env, "applied_residual")
  previous = _hybrid_action_term(env, "previous_applied_residual")
  previous_previous = _hybrid_action_term(
    env,
    "previous_previous_applied_residual",
  )
  acceleration = current - 2 * previous + previous_previous
  return torch.sum(torch.square(acceleration), dim=1)


def applied_residual_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize controller authority used by the learned residual."""

  residual = _hybrid_action_term(env, "applied_residual")
  return torch.sum(torch.square(residual), dim=1)


def healthy_applied_residual_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  max_velocity_error: float,
  max_pitch_abs: float,
  max_pitch_rate_abs: float,
  action_indices: tuple[int, ...] | None = None,
) -> torch.Tensor:
  """Penalize Stage1 residual authority only while the LQR is healthy.

  The residual remains free to assist after a reset or push, but is driven to
  zero once nominal tracking and pitch stability have recovered.
  """

  command = env.command_manager.get_command(command_name)
  robot = env.scene["robot"]
  robot_data = robot.data
  velocity_error = command[:, 0] - robot_data.root_link_lin_vel_b[:, 0]
  projected_gravity = robot_data.projected_gravity_b
  pitch = torch.atan2(
    projected_gravity[:, 0],
    torch.clamp(-projected_gravity[:, 2], min=1.0e-6),
  )
  healthy = (
    (velocity_error.abs() <= max_velocity_error)
    & (pitch.abs() <= max_pitch_abs)
    & (robot_data.root_link_ang_vel_b[:, 1].abs() <= max_pitch_rate_abs)
  )
  residual = _hybrid_action_term(env, "applied_residual")
  if action_indices is not None:
    if not action_indices:
      raise ValueError("action_indices cannot be empty.")
    residual = residual[:, list(action_indices)]
  penalty = torch.sum(torch.square(residual), dim=1)
  return penalty * healthy.to(dtype=velocity_error.dtype)


def stair_mode_forward_progress(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Contact-gated forward progress for the residual stair camp (S5B).

  S5B lists the gating as a REQUIRED element, not a tuning choice: the R2
  literature verdict is that guidance is the decisive variable at small env
  counts, and the CTBC v3 trigger ablation (86% -> 2% at 20 cm) shows the
  trigger is load-bearing in this architecture class. Gating on the latched
  contact trigger is also what bounds the false-trigger damage: `stair_mode`
  never latches on flat ground, so this term is identically zero throughout
  all four no-regression gate runs and cannot pay the policy for speeding
  up where the frozen classical stack already owns the behavior.

  The reward manager scales by dt, so this returns a rate (m/s), clipped at
  zero: rolling backwards earns nothing here, and penalising it is left to
  the existing tracking and residual terms rather than doubled up.
  """

  term = env.action_manager.get_term("hybrid_wheel_leg")
  forward = env.scene["robot"].data.root_link_lin_vel_w[:, 0].clamp(min=0.0)
  return forward * term.stair_mode.to(forward.dtype)


def posture_height_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  robot = env.scene["robot"]
  return torch.square(robot.data.root_link_pos_w[:, 2] - command[:, 0])


def posture_pitch_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  robot = env.scene["robot"]
  projected_gravity = robot.data.projected_gravity_b
  pitch = torch.atan2(
    projected_gravity[:, 0],
    torch.clamp(-projected_gravity[:, 2], min=1.0e-6),
  )
  return torch.square(pitch - command[:, 1])


STAIR_CAMP_PROPRIOCEPTION_TERMS = (
  "base_lin_vel",
  "base_ang_vel",
  "projected_gravity",
  "velocity_command",
  "posture_command",
  "joint_pos",
  "joint_vel",
)
# Registered actor layout (S5B): proprioception, phase one-hot (9),
# classical_wheel_baseline (2), nominal_leg_targets (4), classical_errors (4),
# previous_residual (6), stair_mode (1). Term dicts preserve insertion order
# and the group concatenates in that order, so this tuple IS the layout.
STAIR_CAMP_ACTOR_TAIL_TERMS = (
  "phase_one_hot",
  "classical_wheel_baseline",
  "nominal_leg_targets",
  "classical_errors",
  "previous_residual",
  "stair_mode",
)
STAIR_CAMP_ACTOR_TERM_ORDER = (
  STAIR_CAMP_PROPRIOCEPTION_TERMS + STAIR_CAMP_ACTOR_TAIL_TERMS
)


def stair_camp_actor_terms(
  base_terms: dict[str, ObservationTermCfg],
) -> dict[str, ObservationTermCfg]:
  """Assemble the camp actor group in the registered field order (S5B).

  `stair_residual.build_stair_residual_observations` is the single-env
  NumPy statement of this contract; the runtime path is the manager-based
  group built here, so the two must agree on field order and width. The
  proprioception block is taken verbatim from the frozen Stage5 actor group
  rather than re-declared, so a change there cannot silently desynchronise
  the camp from the stack it layers on.

  Note on `previous_residual`: observations are computed after
  `process_actions`, so `applied_residual` at observation time is the
  residual just applied - which is exactly the "previous action" from the
  policy's next decision. `previous_applied_residual` would be one step
  staler than the registered field.
  """

  missing = [
    name for name in STAIR_CAMP_PROPRIOCEPTION_TERMS if name not in base_terms
  ]
  if missing:
    raise ValueError(
      f"Stage5 actor group is missing proprioception terms: {missing}."
    )
  terms: dict[str, ObservationTermCfg] = {
    name: base_terms[name] for name in STAIR_CAMP_PROPRIOCEPTION_TERMS
  }
  terms["phase_one_hot"] = ObservationTermCfg(
    func=stair_phase_one_hot_observation
  )
  terms["classical_wheel_baseline"] = ObservationTermCfg(
    func=controller_baseline_observation
  )
  terms["nominal_leg_targets"] = ObservationTermCfg(
    func=leg_reference_observation
  )
  terms["classical_errors"] = ObservationTermCfg(
    func=classical_errors_observation
  )
  terms["previous_residual"] = ObservationTermCfg(
    func=applied_residual_observation
  )
  terms["stair_mode"] = ObservationTermCfg(func=stair_mode_observation)
  return terms


# Residual stair camp terrain (S5B). The geometry constants reproduce the
# C0/C2 probe terrain (`probe_hybrid_stair_height`) so the camp trains on the
# same staircase the classical boundary C* was measured on; a different
# staircase would make the boundary-extension comparison incommensurable.
STAIR_CAMP_TERRAIN_SIZE_M = (8.0, 8.0)
STAIR_CAMP_TERRAIN_BORDER_WIDTH_M = 1.0
STAIR_CAMP_STEP_WIDTH_M = 0.30
STAIR_CAMP_PLATFORM_WIDTH_M = 3.0
STAIR_CAMP_START_OFFSET_M = 0.25
STAIR_CAMP_CROSS_DEPTH_M = 0.15
# The registered curriculum ladder is 0.01 m steps capped at 0.15 m. With
# `curriculum=True` mjlab gives row i difficulty i/(rows-1) and
# `BoxPyramidStairsTerrainCfg` interpolates `step_height_range` by difficulty,
# so 16 rows over (0.0, 0.15) reproduce the ladder EXACTLY: row i = i * 0.01.
STAIR_CAMP_HEIGHT_STEP_M = 0.01
STAIR_CAMP_MAX_HEIGHT_M = 0.15
STAIR_CAMP_TERRAIN_ROWS = 16
# The staircase's outer riser face sits this far in -x from the terrain
# origin: half the inner (border-excluded) terrain span. Same derivation as
# `probe_hybrid_stair_height.approach_geometry`, so forward travel is +x.
STAIR_CAMP_RISER_OFFSET_M = 0.5 * (
  STAIR_CAMP_TERRAIN_SIZE_M[0] - 2.0 * STAIR_CAMP_TERRAIN_BORDER_WIDTH_M
)
# Travel budget: the success criterion is crossing riser + 0.15 m from a start
# 0.25 m short of it, i.e. 0.40 m. At the 0.07 m/s speed of the C0 stall
# regime that is 5.7 s, so 20 s leaves better than 3x margin and room to
# recover from a stall rather than truncating mid-attempt.
STAIR_CAMP_EPISODE_LENGTH_S = 20.0
# Reward weights, re-registered per deviation minute 8 ([U] approved option A
# 2026-08-10) after the first frozen weighting (progress 2.0 / success 5.0)
# measurably failed: at iteration 999 of the seed-1 fresh-1000 run the two
# stair terms contributed 0.0088/s against an inherited positive income of
# 11.07/s while parked at the riser (0.079% of the positive budget), and the
# policy rationally converged to standing latched at the riser for the full
# 1000-step episode (latch occupancy 0.80, climb success 0.12 -> 0.00 as
# exploration annealed). Derivation rule, from those measurements: each camp
# term at its IDEAL value in the latched regime must dominate the parked
# income with margin >= 2.
#   progress: ideal value = command speed 0.07 m/s
#     -> weight >= 2 * 11.07 / 0.07 = 316.3 -> 320 (pays 320 per metre of
#        latched forward progress; 22.4/s while advancing at command speed)
#   climb success: state-valued 1.0 after crossing
#     -> weight >= 2 * 11.07 = 22.1 -> 24 (post-success income 2.2x parking)
# Both terms stay gated on the stair_mode latch, which measured 0/96000 false
# positives on flat, so the four no-regression gates see identical incentives
# and the frozen Stage0-5 ladder is untouched. Evidence: transferred seed-1
# archive (SHA256 e9410c41.../bcbb6f48...), experiment log 3.76/3.77.
STAIR_CAMP_PROGRESS_WEIGHT = 320.0
STAIR_CAMP_CLIMB_SUCCESS_WEIGHT = 24.0
# F1 RESOLUTION ([U] 2026-08-04): the privileged critic set is narrowed to the
# three fields that actually vary across envs in this env. `friction` and
# `randomization_parameters` are WITHDRAWN for round 1 - the hybrid env has no
# friction/mass/actuator randomization, so they would be structurally
# degenerate and the builder would silently zero-fill them.
STAIR_CAMP_PRIVILEGED_TERMS = (
  "step_height",
  "distance_to_riser",
  "contact_force",
)
STAIR_CAMP_WITHDRAWN_PRIVILEGED_TERMS = (
  "friction",
  "randomization_parameters",
)


def _stair_camp_riser_x(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene.env_origins[:, 0] - STAIR_CAMP_RISER_OFFSET_M


def stair_camp_step_height_observation(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  """Privileged: the step height of each env's terrain row."""

  terrain = env.scene.terrain
  if terrain is None:
    raise ValueError("The stair camp requires a generated terrain.")
  height = terrain.terrain_levels.to(torch.float) * (
    STAIR_CAMP_MAX_HEIGHT_M / max(STAIR_CAMP_TERRAIN_ROWS - 1, 1)
  )
  return height.unsqueeze(-1)


def stair_camp_distance_to_riser_observation(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  """Privileged: signed +x distance from the root to the first riser face."""

  root_x = env.scene["robot"].data.root_link_pos_w[:, 0]
  return (_stair_camp_riser_x(env) - root_x).unsqueeze(-1)


def stair_camp_contact_force_observation(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  """Privileged: the raw trigger metric the threshold is applied to.

  The actor only sees the thresholded, latched `stair_mode` bit (CTBC uses
  force as a trigger, not an observation - S5B Protocol 2). The critic sees
  the underlying continuous quantity, which is exactly the asymmetry an
  asymmetric actor-critic is for.
  """

  data = env.scene.sensors[STAIR_TRIGGER_SENSOR_NAME].data
  metric = stair_trigger_metric(
    found=data.found,
    force_contact_frame=data.force,
    normal_global=data.normal,
  )
  return metric.unsqueeze(-1)


def stair_camp_climb_success(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Contact-gated climb-success shaping (S5B).

  Pays while the root is past the crest reference (`riser + 0.15 m`, the
  same crossing depth the C0/C2 probes score success at) AND the contact
  trigger has latched. Gating on `stair_mode` for the same reason the
  progress term is gated: on flat ground the trigger never fires, so this
  term is identically zero across all four no-regression gate runs and
  cannot pay the policy for behavior the frozen classical stack owns.
  """

  term = env.action_manager.get_term("hybrid_wheel_leg")
  root_x = env.scene["robot"].data.root_link_pos_w[:, 0]
  crossed = root_x >= _stair_camp_riser_x(env) + STAIR_CAMP_CROSS_DEPTH_M
  return (crossed & term.stair_mode).to(torch.float)


# Curriculum (S5B Protocol 4). The frozen `update_stair_curriculum` owns the
# promotion rule (success_rate >= 0.80 for 3 consecutive evaluations -> upper
# +0.01 m, cap 0.15 m); everything here is the wiring that feeds it.
#
# The registered tier grid is {0.01, 0.02, ..., upper}: the lower bound is
# 0.01 m and NEVER moves, so level 0 (flat) is NOT part of the training
# distribution. Sampling in [0, upper] instead - which is what mjlab's
# `max_init_terrain_level` does on its own - would put roughly half the envs
# on flat ground at the registered initial state (upper = 0.01).
STAIR_CAMP_CURRICULUM_LOWER_LEVEL = 1
STAIR_CAMP_CURRICULUM_INITIAL_UPPER_M = 0.01
STAIR_CAMP_EVALUATION_INTERVAL_ITERS = 50
# `num_steps_per_env` of the registered runner; a contract test pins it.
STAIR_CAMP_STEPS_PER_ITERATION = 24


def _stair_camp_level(height_m: float) -> int:
  return int(round(float(height_m) / STAIR_CAMP_HEIGHT_STEP_M))


class StairCampCurriculum:
  """Exact-step S5B curriculum shared by metrics and reset terms."""

  STATE_SCHEMA_VERSION = 1

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    evaluation_interval_steps: int,
    initial_upper_height_m: float = STAIR_CAMP_CURRICULUM_INITIAL_UPPER_M,
  ):
    if evaluation_interval_steps < 1:
      raise ValueError("StairCamp evaluation interval must be positive.")
    self.state = StairCurriculumState(
      lower_height_m=STAIR_CAMP_HEIGHT_STEP_M,
      upper_height_m=initial_upper_height_m,
    )
    self.evaluation_interval_steps = int(evaluation_interval_steps)
    self.next_evaluation_step = self.evaluation_interval_steps
    self.episodes_at_upper = 0
    self.successes_at_upper = 0
    self.evaluations = 0
    self.last_processed_step = -1
    self._started = False
    self.triggered_episodes = 0
    self.completed_episodes = 0
    self.residual_abs_sum = 0.0
    self.residual_sq_sum = 0.0
    self.residual_sample_count = 0
    self.residual_abs_max = 0.0
    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
      raise ValueError("The stair camp curriculum requires generated terrain.")

  @property
  def upper_level(self) -> int:
    return _stair_camp_level(self.state.upper_height_m)

  def _score_finished_episodes(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor
  ) -> None:
    terrain = env.scene.terrain
    assert terrain is not None
    if len(env_ids) == 0:
      return
    at_upper = terrain.terrain_levels[env_ids] == self.upper_level
    success = stair_camp_climb_success(env)[env_ids] > 0.5
    self.completed_episodes += int(len(env_ids))
    action_manager = getattr(env, "action_manager", None)
    if action_manager is not None:
      term = action_manager.get_term("hybrid_wheel_leg")
      self.triggered_episodes += int(term.stair_mode[env_ids].sum())
    if not bool(at_upper.any()):
      return
    self.episodes_at_upper += int(at_upper.sum())
    self.successes_at_upper += int((success & at_upper).sum())

  def _evaluate_once(self) -> None:
    rate = (
      self.successes_at_upper / self.episodes_at_upper
      if self.episodes_at_upper > 0
      else 0.0
    )
    self.state = update_stair_curriculum(
      self.state,
      success_rate=rate,
      maximum_height_m=STAIR_CAMP_MAX_HEIGHT_M,
    )
    self.episodes_at_upper = 0
    self.successes_at_upper = 0
    self.evaluations += 1
    self.next_evaluation_step += self.evaluation_interval_steps

  def _maybe_evaluate(self, env: ManagerBasedRlEnv) -> None:
    """Evaluate every crossed registered boundary, never only on reset."""

    step = int(env.common_step_counter)
    while step >= self.next_evaluation_step:
      self._evaluate_once()

  def record_step(self, env: ManagerBasedRlEnv) -> None:
    """Consume one post-reward env step exactly once."""

    step = int(env.common_step_counter)
    if step <= self.last_processed_step:
      return
    reset_buf = getattr(env, "reset_buf", None)
    if reset_buf is not None:
      env_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
      if len(env_ids):
        self._score_finished_episodes(env, env_ids)
    self._maybe_evaluate(env)
    self.last_processed_step = step

  def record_residual(self, env: ManagerBasedRlEnv) -> None:
    term = env.action_manager.get_term("hybrid_wheel_leg")
    values = term.applied_residual[:, 2:].detach()
    absolute = values.abs()
    self.residual_abs_sum += float(absolute.sum().item())
    self.residual_sq_sum += float(torch.square(values).sum().item())
    self.residual_sample_count += int(values.numel())
    if values.numel():
      self.residual_abs_max = max(self.residual_abs_max, float(absolute.max().item()))

  def progress_snapshot(self) -> dict[str, float | int]:
    count = max(self.residual_sample_count, 1)
    return {
      "upper_height_m": float(self.state.upper_height_m),
      "trigger_rate": self.triggered_episodes / max(self.completed_episodes, 1),
      "residual_abs_mean": self.residual_abs_sum / count,
      "residual_rms": math.sqrt(self.residual_sq_sum / count),
      "residual_abs_max": self.residual_abs_max,
      "evaluations": self.evaluations,
    }

  def state_dict(self) -> dict[str, object]:
    return {
      "schema_version": self.STATE_SCHEMA_VERSION,
      "lower_height_m": float(self.state.lower_height_m),
      "upper_height_m": float(self.state.upper_height_m),
      "consecutive_ready_evaluations": int(
        self.state.consecutive_ready_evaluations
      ),
      "evaluation_interval_steps": self.evaluation_interval_steps,
      "next_evaluation_step": self.next_evaluation_step,
      "episodes_at_upper": self.episodes_at_upper,
      "successes_at_upper": self.successes_at_upper,
      "evaluations": self.evaluations,
      "last_processed_step": self.last_processed_step,
      "started": self._started,
      "triggered_episodes": self.triggered_episodes,
      "completed_episodes": self.completed_episodes,
      "residual_abs_sum": self.residual_abs_sum,
      "residual_sq_sum": self.residual_sq_sum,
      "residual_sample_count": self.residual_sample_count,
      "residual_abs_max": self.residual_abs_max,
    }

  def load_state_dict(self, payload: Mapping[str, object]) -> None:
    """Restore a complete, validated curriculum snapshot atomically."""

    if not isinstance(payload, Mapping):
      raise ValueError("StairCamp curriculum state must be a mapping.")
    expected = set(self.state_dict())
    if set(payload) != expected:
      raise ValueError("StairCamp curriculum state schema does not match.")

    def exact_int(name: str, *, minimum: int | None = None) -> int:
      value = payload[name]
      if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"StairCamp curriculum {name} must be an integer.")
      result = int(value)
      if minimum is not None and result < minimum:
        raise ValueError(f"StairCamp curriculum {name} is negative.")
      return result

    def finite_number(name: str) -> float:
      value = payload[name]
      if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"StairCamp curriculum {name} must be numeric.")
      result = float(value)
      if not math.isfinite(result):
        raise ValueError(f"StairCamp curriculum {name} must be finite.")
      return result

    schema_version = exact_int("schema_version", minimum=0)
    if schema_version != self.STATE_SCHEMA_VERSION:
      raise ValueError("Unsupported StairCamp curriculum state schema.")
    interval = exact_int("evaluation_interval_steps", minimum=1)
    if interval != self.evaluation_interval_steps:
      raise ValueError("StairCamp curriculum evaluation interval drifted.")

    lower = finite_number("lower_height_m")
    upper = finite_number("upper_height_m")
    lower_level = lower / STAIR_CAMP_HEIGHT_STEP_M
    upper_level = upper / STAIR_CAMP_HEIGHT_STEP_M
    if (
      not math.isclose(
        lower,
        STAIR_CAMP_HEIGHT_STEP_M,
        rel_tol=0.0,
        abs_tol=1.0e-12,
      )
      or upper < lower
      or upper > STAIR_CAMP_MAX_HEIGHT_M
      or abs(lower_level - round(lower_level)) > 1.0e-9
      or abs(upper_level - round(upper_level)) > 1.0e-9
    ):
      raise ValueError("Invalid StairCamp curriculum height state.")

    ready = exact_int("consecutive_ready_evaluations", minimum=0)
    episodes = exact_int("episodes_at_upper", minimum=0)
    successes = exact_int("successes_at_upper", minimum=0)
    evaluations = exact_int("evaluations", minimum=0)
    next_step = exact_int("next_evaluation_step", minimum=interval)
    last_step = exact_int("last_processed_step")
    triggered = exact_int("triggered_episodes", minimum=0)
    completed = exact_int("completed_episodes", minimum=0)
    residual_count = exact_int("residual_sample_count", minimum=0)
    started = payload["started"]
    if not isinstance(started, bool):
      raise ValueError("StairCamp curriculum started must be a boolean.")

    if ready > 2 or successes > episodes:
      raise ValueError("Invalid StairCamp curriculum counters.")
    if triggered > completed:
      raise ValueError("StairCamp triggered episodes exceed completed episodes.")
    if next_step % interval != 0 or next_step != (evaluations + 1) * interval:
      raise ValueError("Invalid StairCamp next evaluation step.")
    if last_step < -1 or last_step >= next_step:
      raise ValueError("Invalid StairCamp last processed step.")

    residual_abs_sum = finite_number("residual_abs_sum")
    residual_sq_sum = finite_number("residual_sq_sum")
    residual_abs_max = finite_number("residual_abs_max")
    if min(residual_abs_sum, residual_sq_sum, residual_abs_max) < 0.0:
      raise ValueError("Invalid StairCamp residual progress state.")
    if residual_count == 0:
      if any(value != 0.0 for value in (residual_abs_sum, residual_sq_sum, residual_abs_max)):
        raise ValueError("Empty StairCamp residual state must have zero totals.")
    else:
      # These inequalities catch truncated/forged aggregates without relying on
      # a particular rollout batch size. The sums are over absolute residual
      # samples, so both are bounded by the recorded maximum.
      absolute_bound = residual_count * residual_abs_max
      square_bound = residual_count * residual_abs_max**2
      if residual_abs_sum > absolute_bound + 1.0e-5 * max(1.0, absolute_bound):
        raise ValueError("StairCamp residual absolute sum is inconsistent.")
      if residual_sq_sum > square_bound + 1.0e-5 * max(1.0, square_bound):
        raise ValueError("StairCamp residual square sum is inconsistent.")

    # Assign only after every field has passed validation, so a rejected
    # checkpoint cannot leave a partially restored state behind.
    self.state = StairCurriculumState(
      lower_height_m=lower,
      upper_height_m=upper,
      consecutive_ready_evaluations=ready,
    )
    self.evaluation_interval_steps = interval
    self.next_evaluation_step = next_step
    self.episodes_at_upper = episodes
    self.successes_at_upper = successes
    self.evaluations = evaluations
    self.last_processed_step = last_step
    self._started = started
    self.triggered_episodes = triggered
    self.completed_episodes = completed
    self.residual_abs_sum = residual_abs_sum
    self.residual_sq_sum = residual_sq_sum
    self.residual_sample_count = residual_count
    self.residual_abs_max = residual_abs_max

  def _assign_levels(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor
  ) -> None:
    terrain = env.scene.terrain
    assert terrain is not None and terrain.terrain_origins is not None
    levels = torch.randint(
      STAIR_CAMP_CURRICULUM_LOWER_LEVEL,
      self.upper_level + 1,
      (len(env_ids),),
      device=terrain.terrain_levels.device,
      dtype=terrain.terrain_levels.dtype,
    )
    terrain.terrain_levels[env_ids] = levels
    assert terrain.env_origins is not None
    terrain.env_origins[env_ids] = terrain.terrain_origins[
      terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    ]

  def compute(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor
  ) -> dict[str, float]:
    if len(env_ids):
      self._assign_levels(env, env_ids)
      self._started = True
    terrain = env.scene.terrain
    assert terrain is not None
    return {
      "upper_height_m": float(self.state.upper_height_m),
      "consecutive_ready": float(self.state.consecutive_ready_evaluations),
      "evaluations": self.evaluations,
      "mean_level": float(terrain.terrain_levels.float().mean()),
    }

def _stair_camp_curriculum_state(
  env: ManagerBasedRlEnv,
  evaluation_interval_steps: int,
  initial_upper_height_m: float,
) -> StairCampCurriculum:
  state = getattr(env, "stair_camp_curriculum_state", None)
  if state is None:
    state = StairCampCurriculum(
      env, evaluation_interval_steps, initial_upper_height_m
    )
    env.stair_camp_curriculum_state = state  # type: ignore[attr-defined]
  elif not isinstance(state, StairCampCurriculum):
    raise ValueError("StairCamp curriculum state has an invalid type.")
  elif state.evaluation_interval_steps != int(evaluation_interval_steps):
    raise ValueError("StairCamp curriculum interval changed after construction.")
  return state


def stair_camp_step_metric(
  env: ManagerBasedRlEnv,
  evaluation_interval_steps: int,
  initial_upper_height_m: float = STAIR_CAMP_CURRICULUM_INITIAL_UPPER_M,
) -> torch.Tensor:
  """Per-step hook: score terminal episodes and evaluate exact boundaries."""

  state = _stair_camp_curriculum_state(
    env, evaluation_interval_steps, initial_upper_height_m
  )
  previous_step = state.last_processed_step
  state.record_step(env)
  if state.last_processed_step != previous_step:
    state.record_residual(env)
  term = env.action_manager.get_term("hybrid_wheel_leg")
  return term.stair_mode.to(dtype=torch.float)


def stair_camp_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  evaluation_interval_steps: int,
  initial_upper_height_m: float = STAIR_CAMP_CURRICULUM_INITIAL_UPPER_M,
) -> dict[str, float]:
  """Reset hook: assign the current registered height band only."""

  state = _stair_camp_curriculum_state(
    env, evaluation_interval_steps, initial_upper_height_m
  )
  if env_ids is None:
    ids = torch.arange(env.num_envs, device=env.device)
  elif isinstance(env_ids, slice):
    ids = torch.arange(env.num_envs, device=env.device)[env_ids]
  else:
    ids = env_ids
  return state.compute(env, ids)

def stair_camp_critic_terms(

  actor_terms: dict[str, ObservationTermCfg],
) -> dict[str, ObservationTermCfg]:
  """Critic = actor superset ⊕ the retained privileged fields (S5B)."""

  terms = dict(actor_terms)
  terms["step_height"] = ObservationTermCfg(
    func=stair_camp_step_height_observation
  )
  terms["distance_to_riser"] = ObservationTermCfg(
    func=stair_camp_distance_to_riser_observation
  )
  terms["contact_force"] = ObservationTermCfg(
    func=stair_camp_contact_force_observation
  )
  return terms


def validate_stair_camp_observation_contract(
  cfg: ManagerBasedRlEnvCfg,
) -> None:
  """Fail closed if the asymmetric S5B observation surface degenerates."""

  observations = cfg.observations
  actor = observations.get("actor")
  critic = observations.get("critic")
  if actor is None or critic is None:
    raise ValueError("StairCamp requires actor and critic observation groups.")
  actor_names = tuple(actor.terms)
  critic_names = tuple(critic.terms)
  if actor_names != STAIR_CAMP_EXPECTED_ACTOR_TERMS:
    raise ValueError("StairCamp actor terms or insertion order drifted.")
  if critic_names != actor_names + STAIR_CAMP_EXPECTED_CRITIC_TAIL:
    raise ValueError("StairCamp critic must append exactly three privileged terms.")
  actor_width = sum(STAIR_CAMP_EXPECTED_TERM_WIDTHS[name] for name in actor_names)
  critic_width = sum(STAIR_CAMP_EXPECTED_TERM_WIDTHS[name] for name in critic_names)
  if actor_width != STAIR_CAMP_ACTOR_WIDTH or critic_width != STAIR_CAMP_CRITIC_WIDTH:
    raise ValueError("StairCamp observation widths must be exactly 52/55.")
  if any(name in critic_names for name in STAIR_CAMP_WITHDRAWN_CRITIC_TERMS):
    raise ValueError("Withdrawn StairCamp privileged terms must be absent.")
  action = cfg.actions.get("hybrid_wheel_leg")
  if action is None:
    raise ValueError("StairCamp requires the hybrid_wheel_leg action term.")
  if tuple(action.action_mask) != STAIR_CAMP_ACTION_MASK:
    raise ValueError("StairCamp runtime action mask drifted.")
  expected_scales = action_scales_with_leg_authority(STAIR_CAMP_LEG_RESIDUAL_SCALE)
  if tuple(action.action_scales) != expected_scales:
    raise ValueError("StairCamp action scales drifted from 0.070 rad authority.")
  if not action.stair_mode_freezes_leg_reference:
    raise ValueError("StairCamp must freeze the leg reference at the trigger.")
  if action.stair_trigger_sensor_name != STAIR_TRIGGER_SENSOR_NAME:
    raise ValueError("StairCamp trigger sensor binding is missing.")
  sensor_names = {getattr(sensor, "name", None) for sensor in cfg.scene.sensors}
  if STAIR_TRIGGER_SENSOR_NAME not in sensor_names:
    raise ValueError("StairCamp contact-force sensor is missing.")
  terrain = cfg.scene.terrain
  generator = None if terrain is None else terrain.terrain_generator
  if (
    generator is None
    or not generator.curriculum
    or generator.num_rows != STAIR_CAMP_TERRAIN_ROWS
    or generator.num_cols != 1
  ):
    raise ValueError("StairCamp terrain cannot vary the privileged step height.")
  if critic.terms["step_height"].func is not stair_camp_step_height_observation:
    raise ValueError("StairCamp step-height critic field is not live.")
  if critic.terms["distance_to_riser"].func is not stair_camp_distance_to_riser_observation:
    raise ValueError("StairCamp distance critic field is not live.")
  if critic.terms["contact_force"].func is not stair_camp_contact_force_observation:
    raise ValueError("StairCamp contact-force critic field is not live.")

def roll_assist_wheel_sensor_cfg(
  *,
  name: str,
  wheel_geom: str,
) -> ContactSensorCfg:
  """Build one exact per-wheel terrain-contact sensor for RollAssist."""

  expected = {
    ROLL_ASSIST_LEFT_SENSOR_NAME: "wheel_left_collision",
    ROLL_ASSIST_RIGHT_SENSOR_NAME: "wheel_right_collision",
  }
  if expected.get(name) != wheel_geom:
    raise ValueError("Unknown RollAssist wheel sensor identity.")
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(mode="geom", pattern=wheel_geom, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=ROLL_ASSIST_SENSOR_FIELDS,
    reduce="none",
    num_slots=ROLL_ASSIST_SENSOR_SLOTS,
  )


def dynamic_stair_wheel_sensor_cfg(
  *,
  name: str,
  wheel_geom: str,
) -> ContactSensorCfg:
  """Build one unreduced contact-frame sensor for one exact wheel geom."""

  if name not in DYNAMIC_STAIR_SENSOR_NAMES:
    raise ValueError("Unknown StairDynamic wheel sensor name.")
  if wheel_geom not in ("wheel_left_collision", "wheel_right_collision"):
    raise ValueError("StairDynamic sensor must bind one exact wheel geom.")
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(mode="geom", pattern=wheel_geom, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=DYNAMIC_STAIR_SENSOR_FIELDS,
    reduce="none",
    num_slots=DYNAMIC_STAIR_SENSOR_SLOTS,
  )


def validate_stair_dynamic_observation_contract(
  cfg: ManagerBasedRlEnvCfg,
) -> None:
  """Fail closed on the v3 52/56 interface and live FSM prerequisites."""

  if getattr(cfg, "stair_dynamic_task_id", None) != DYNAMIC_STAIR_TASK_ID:
    raise ValueError("StairDynamic task marker is missing.")
  actor = cfg.observations.get("actor")
  critic = cfg.observations.get("critic")
  if actor is None or critic is None:
    raise ValueError("StairDynamic actor and critic groups are required.")
  validate_dynamic_stair_observation_layout(actor.terms, critic.terms)
  action = cfg.actions.get("hybrid_wheel_leg")
  if not isinstance(action, HybridWheelLegActionCfg):
    raise TypeError("StairDynamic hybrid action term is missing.")
  if tuple(action.action_mask) != DYNAMIC_STAIR_ACTION_MASK:
    raise ValueError("StairDynamic runtime action mask drifted.")
  if tuple(float(value) for value in action.action_scales) != DYNAMIC_STAIR_ACTION_SCALES:
    raise ValueError("StairDynamic action scales drifted.")
  if action.dynamic_stair_maneuver is None:
    raise ValueError("StairDynamic maneuver is missing.")
  if action.stair_trigger_sensor_name is not None or action.stair_mode_forced:
    raise ValueError("StairDynamic must not invoke the archived global trigger.")
  if action.stair_mode_freezes_leg_reference:
    raise ValueError("StairDynamic uses live posture plus explicit feedforward.")
  expected_sensor_bindings = {
    DYNAMIC_STAIR_LEFT_SENSOR_NAME: "wheel_left_collision",
    DYNAMIC_STAIR_RIGHT_SENSOR_NAME: "wheel_right_collision",
  }
  sensors = {sensor.name: sensor for sensor in cfg.scene.sensors}
  for sensor_name, geom in expected_sensor_bindings.items():
    sensor = sensors.get(sensor_name)
    if sensor is None:
      raise ValueError(f"StairDynamic sensor {sensor_name!r} is missing.")
    if (
      sensor.primary.mode != "geom"
      or sensor.primary.pattern != geom
      or sensor.primary.entity != "robot"
      or tuple(sensor.fields) != DYNAMIC_STAIR_SENSOR_FIELDS
      or sensor.reduce != "none"
    ):
      raise ValueError(f"StairDynamic sensor {sensor_name!r} identity drifted.")
  terrain = cfg.scene.terrain
  generator = None if terrain is None else terrain.terrain_generator
  if (
    terrain is None
    or terrain.terrain_type != "generator"
    or generator is None
    or generator.num_rows != DYNAMIC_STAIR_TERRAIN_ROWS
    or generator.num_cols != 1
  ):
    raise ValueError("StairDynamic regular-stair terrain contract drifted.")
  commands = cfg.commands
  if not isinstance(commands.get("stair_request"), StairRequestCommandCfg):
    raise TypeError("StairDynamic stair_request command is missing.")
  if not isinstance(commands.get("twist"), StairDynamicVelocityCommandCfg):
    raise TypeError("StairDynamic velocity command is not split flat/stair.")
  if not isinstance(commands.get("posture"), StairDynamicPostureCommandCfg):
    raise TypeError("StairDynamic posture command is not split flat/stair.")
  if tuple(actor.terms)[:9] != DYNAMIC_STAIR_ACTOR_TERMS[:9]:
    raise ValueError("StairDynamic no longer preserves the Stage5 prefix.")
  privileged = set(DYNAMIC_STAIR_CRITIC_TAIL_TERMS)
  if privileged.intersection(actor.terms):
    raise ValueError("StairDynamic privileged fields leaked into the actor.")


def stair_trigger_sensor_cfg() -> ContactSensorCfg:
  """Contact sensor feeding the camp's CTBC-style trigger (S5B Protocol 2).

  Deliberately a dedicated sensor rather than a reuse of the non-wheel
  termination sensor: the trigger metric needs the per-slot contact-frame
  force and the global contact normal, which that sensor does not carry.
  The primary/secondary match, the slot count and the un-reduced layout
  reproduce the C2-j3 capture that produced the frozen 288/288 detection
  and zero-false-positive evidence, so the runtime metric is computed over
  the same contact set the evidence was measured on.

  `reduce="none"` with `global_frame` left at its default keeps `force` in
  the contact frame (component 0 is the normal force) and `normal` in the
  global frame, which is exactly what `stair_trigger_metric` expects.
  """

  return ContactSensorCfg(
    name=STAIR_TRIGGER_SENSOR_NAME,
    primary=ContactMatch(
      mode="geom", pattern=WHEEL_GROUND_GEOMS, entity="robot"
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=STAIR_TRIGGER_SENSOR_FIELDS,
    reduce="none",
    num_slots=STAIR_TRIGGER_SLOTS_PER_WHEEL,
  )


def _base_env_cfg(stage: int, play: bool) -> ManagerBasedRlEnvCfg:
  if stage == 0:
    return make_hoppertrex_balance_env_cfg(play=play)
  if stage == 1:
    return make_hoppertrex_balance_env_cfg(
      play=play,
      robust=not play,
      robust_level=1,
      slow_speed=True,
      speed_level=0,
      slow_speed_lin_sign=True,
      slow_speed_obs_scale=True,
      zero_wheel_joint_pos_obs=True,
    )
  return make_hoppertrex_balance_env_cfg(
    play=play,
    robust=(stage == 5 or (stage == 2 and not play)),
    robust_level=(1 if stage == 2 else 2),
    slow_speed_turn=True,
    slow_speed_turn_sign=True,
    slow_speed_turn_obs_scale=True,
    slow_speed_turn_safe_v2=True,
    slow_speed_turn_safe_v2_yaw_scale2p5=True,
    slow_speed_turn_safe_v2_yaw_smooth=True,
    slow_speed_turn_bidirectional=True,
    slow_speed_turn_bidirectional_lin_sign=True,
    slow_speed_turn_target_slew=True,
    zero_wheel_joint_pos_obs=True,
  )


def stage1_mismatch_event_cfg(
  *,
  gate_group_count: int | None = None,
  gate_group_index: int | None = None,
) -> EventTermCfg:
  """Build training or gate-targeted Stage1 mild mismatch configuration."""

  gate_indices = (
    () if gate_group_index is None else (int(gate_group_index),)
  )
  return EventTermCfg(
    func=apply_stage1_mild_mismatch,
    mode="startup",
    params={
      "chassis_cfg": SceneEntityCfg(
        "robot",
        body_names=("chassis_base",),
      ),
      "wheel_geom_cfg": SceneEntityCfg(
        "robot",
        geom_names=("wheel_left_collision", "wheel_right_collision"),
      ),
      "wheel_actuator_cfg": SceneEntityCfg(
        "robot",
        actuator_names=WHEEL_JOINT_NAMES,
      ),
      "mismatch_fraction": STAGE1_MISMATCH_FRACTION,
      "group_count": gate_group_count,
      "mismatch_group_indices": gate_indices,
    },
  )


def make_hoppertrex_hybrid_env_cfg(
  stage: int,
  play: bool = False,
  controller_path: Path | None = None,
  posture_map_path: Path | None = None,
  calibration_path: Path | None = None,
  yaw_calibration_path: Path | None = None,
  station_calibration_path: Path | None = None,
  leg_residual_scale: float | None = None,
) -> ManagerBasedRlEnvCfg:
  """Build one Hybrid v2 stage without changing the legacy task factory."""

  if stage not in HYBRID_STAGES:
    raise ValueError(f"Unsupported Hybrid stage {stage}; expected an integer from 0 to 5.")
  stage_cfg = HYBRID_STAGES[stage]
  environment_leg_scale = os.environ.get(LEG_RESIDUAL_SCALE_ENV)
  if leg_residual_scale is not None and environment_leg_scale is not None:
    raise ValueError(
      "Set the leg residual scale either explicitly or through "
      f"{LEG_RESIDUAL_SCALE_ENV}, not both."
    )
  resolved_leg_scale = leg_residual_scale
  if environment_leg_scale is not None:
    try:
      resolved_leg_scale = float(environment_leg_scale)
    except ValueError as exc:
      raise ValueError(
        f"{LEG_RESIDUAL_SCALE_ENV} must contain a finite number."
      ) from exc
  if leg_residual_scale is not None and stage != 5:
    raise ValueError(
      "The experimental leg-residual authority override is restricted to "
      "Stage5 so earlier curriculum and frozen evidence cannot change."
    )
  if stage != 5:
    # Task registration constructs all six stages in one process. A Stage5
    # experiment environment variable must leave the frozen Stage0-4 configs
    # byte-identical instead of making their registration fail.
    resolved_leg_scale = None
  action_scales = action_scales_with_leg_authority(resolved_leg_scale)
  controller = _load_controller(
    _artifact_path(controller_path, CONTROLLER_PATH_ENV)
  )
  # A gain-scheduled artifact's own gain_hash is the schedule_hash, but its
  # companion artifacts (velocity/yaw/station calibration, posture map) were
  # created against the identification incumbent controller. Binding checks
  # must therefore compare against the schedule's registered
  # identification_controller_gain_hash, not the schedule_hash.
  schedule_bindings = (
    controller.schedule.bindings if controller.schedule is not None else {}
  )
  binding_gain_hash = (
    schedule_bindings.get("identification_controller_gain_hash")
    if controller.schedule is not None
    else controller.gain_hash
  )
  resolved_calibration = _artifact_path(
    calibration_path, CALIBRATION_PATH_ENV
  )
  calibration = VelocityCalibration(
    scale=1.0,
    bias=0.0,
    calibration_hash="uncalibrated",
    controller_gain_hash=binding_gain_hash or "",
  )
  if resolved_calibration is not None:
    calibration = parse_calibration_artifact(
      _read_json_object(resolved_calibration, "Calibration"),
      controller_gain_hash=binding_gain_hash or "",
    )
    if (
      controller.schedule is not None
      and calibration.calibration_hash
      != schedule_bindings.get("identification_calibration_hash")
    ):
      raise ValueError(
        "Controller schedule was identified with a different velocity "
        "calibration artifact."
      )
  posture = _load_posture_map(
    _artifact_path(posture_map_path, POSTURE_MAP_PATH_ENV)
  )
  yaw_calibration = _load_yaw_calibration(
    _artifact_path(yaw_calibration_path, YAW_CALIBRATION_PATH_ENV),
    binding_gain_hash or "",
  )
  if (
    not yaw_calibration.qualified
    and stage_cfg.yaw_rate_range != (0.0, 0.0)
    and stage not in _YAW_CALIBRATION_WARNED
  ):
    _YAW_CALIBRATION_WARNED.add(stage)
    print(
      f"[hybrid][WARN] Stage {stage} commands nonzero yaw rates but no yaw "
      "calibration artifact is set: the wheel-differential feedforward is "
      "zero and nominal yaw tracking is unowned. Set "
      f"{YAW_CALIBRATION_PATH_ENV} to the probe-fitted artifact."
    )
  expected_posture_controller_hash = binding_gain_hash
  expected_posture_calibration_hash = (
    schedule_bindings.get("identification_calibration_hash")
    if controller.schedule is not None
    else calibration.calibration_hash
  )
  if (
    stage_cfg.posture_commands
    and posture.qualified
    and posture.controller_gain_hash != expected_posture_controller_hash
  ):
    raise ValueError(
      'Posture map was collected with a different controller artifact.'
    )
  if (
    stage_cfg.posture_commands
    and posture.qualified
    and posture.calibration_hash != expected_posture_calibration_hash
  ):
    raise ValueError(
      'Posture map was collected with a different calibration artifact.'
    )
  if (
    stage_cfg.posture_commands
    and posture.qualified
    and controller.schedule is not None
    and posture.artifact_hash != schedule_bindings.get("posture_artifact_hash")
  ):
    raise ValueError(
      "Controller schedule was identified with a different posture artifact."
    )
  # Stage 3.0 invariance hardening: non-posture stages pin the ACTION term
  # to the default posture artifact. Their frozen evidence ran without a
  # map (leg targets = nominal via the zero height/pitch rows), and the
  # resample fix would otherwise let an exported map env var change stage
  # 0-2 leg targets through the now-live (0.325, 0) posture command.
  # Loading/validation above stays unconditional so binding errors still
  # surface in any session.
  action_posture = (
    posture if stage_cfg.posture_commands else _default_posture_artifact()
  )
  # Stage 3.0: only posture-commanding stages load the station artifact, so
  # stages 0-2 keep bit-identical behavior even when the environment
  # variable is set (their compensation stays the zero fallback).
  station_calibration = _STATION_FALLBACK_ARTIFACT
  if stage_cfg.posture_commands and posture.qualified:
    station_calibration = _load_station_calibration(
      _artifact_path(station_calibration_path, STATION_CALIBRATION_PATH_ENV),
      binding_gain_hash or "",
      posture.map_hash,
      posture.artifact_hash,
    )
    if (
      not station_calibration.qualified
      and stage not in _STATION_CALIBRATION_WARNED
    ):
      _STATION_CALIBRATION_WARNED.add(stage)
      print(
        f"[hybrid][WARN] Stage {stage} commands postures but no station "
        "calibration artifact is set: commanded pitches settle into the "
        "measured steady drift instead of station keeping. Set "
        f"{STATION_CALIBRATION_PATH_ENV} to the probe-fitted artifact."
      )
  cfg = _base_env_cfg(stage, play)
  cfg.rewards["action_rate_l2"].func = applied_residual_rate_l2
  if "action_acc_l2" in cfg.rewards:
    cfg.rewards["action_acc_l2"].func = applied_residual_acc_l2
  if stage >= 1:
    cfg.rewards["applied_residual_l2"] = RewardTermCfg(
      func=applied_residual_l2,
      weight=(
        STAGE1_GLOBAL_RESIDUAL_L2_WEIGHT
        if stage == 1
        else HYBRID_RESIDUAL_L2_WEIGHT
      ),
    )
    # The identified LQR owns nominal forward/reverse direction in every
    # Hybrid stage. The legacy pure-PPO sign reward saturates at the target
    # speed and can drive the balance residual to relearn or override that
    # behavior while a later head (yaw/posture) is being introduced.
    cfg.rewards.pop("lin_vel_x_sign_alignment", None)

  if stage == 1:
    cfg.rewards["track_linear_velocity"].params["std"] = (
      STAGE1_TRACK_LIN_VEL_STD
    )
  if stage >= 2:
    # From Stage 2.0 the probe-fitted feedforward owns nominal yaw sign and
    # magnitude. The legacy sign reward would pay the residual for relearning
    # what the classical layer already provides, and the legacy 0.20 tracking
    # std leaves no gradient near the ±0.10 command band.
    cfg.rewards.pop("yaw_sign_alignment", None)
    if "track_angular_velocity" in cfg.rewards:
      cfg.rewards["track_angular_velocity"].params["std"] = (
        HYBRID_TRACK_ANG_VEL_STD
      )
  if stage in (1, 2):
    cfg.rewards["healthy_applied_residual_l2"] = RewardTermCfg(
      func=healthy_applied_residual_l2,
      weight=STAGE1_HEALTHY_RESIDUAL_L2_WEIGHT,
      params={
        "command_name": "twist",
        "max_velocity_error": STAGE1_HEALTHY_VELOCITY_ERROR,
        "max_pitch_abs": STAGE1_HEALTHY_PITCH_ABS,
        "max_pitch_rate_abs": STAGE1_HEALTHY_PITCH_RATE_ABS,
        # From Stage2 on the yaw head is a residual around the calibrated
        # feedforward, so it is anchored to zero in the healthy state exactly
        # like the balance head around the LQR.
        "action_indices": (0,) if stage == 1 else (0, 1),
      },
    )

  cfg.actions = {
    "hybrid_wheel_leg": HybridWheelLegActionCfg(
      entity_name="robot",
      action_mask=stage_cfg.action_mask,
      action_scales=action_scales,
      controller_gain=controller.gain,
      controller_type=controller.controller_type,
      controller_qualified=controller.qualified,
      controller_source=controller.source,
      controller_gain_hash=controller.gain_hash,
      controller_schedule=controller.schedule,
      velocity_command_scale=calibration.scale,
      velocity_command_bias=calibration.bias,
      calibration_hash=(
        None if calibration.calibration_hash == "uncalibrated"
        else calibration.calibration_hash
      ),
      posture_coefficients=action_posture.coefficients,
      posture_map_qualified=action_posture.qualified,
      posture_map_source=action_posture.source,
      posture_map_hash=action_posture.map_hash,
      posture_artifact_hash=action_posture.artifact_hash,
      yaw_feedforward_breakpoints=yaw_calibration.breakpoints,
      yaw_calibration_qualified=yaw_calibration.qualified,
      yaw_calibration_source=yaw_calibration.source,
      yaw_calibration_hash=yaw_calibration.yaw_calibration_hash,
      station_drift_breakpoints=station_calibration.breakpoints,
      station_calibration_qualified=station_calibration.qualified,
      station_calibration_source=station_calibration.source,
      station_calibration_hash=station_calibration.station_calibration_hash,
    )
  }
  velocity_command_cfg_cls = (
    Stage1VelocityCommandCfg
    if stage == 1
    else (
      HybridPlanarVelocityCommandCfg
      if stage in (2, 4, 5)
      else UniformVelocityCommandCfg
    )
  )
  velocity_command_kwargs = {
    "entity_name": "robot",
    "resampling_time_range": (
      STAGE1_COMMAND_RESAMPLING_TIME_RANGE
      if stage == 1 and not play
      else (
        HYBRID_PLANAR_COMMAND_RESAMPLING_TIME_RANGE
        if stage in (2, 4, 5) and not play
        else (5.0, 10.0)
      )
    ),
    "rel_standing_envs": (
      STAGE1_STANDING_ENVS
      if stage == 1
      else (
        HYBRID_PLANAR_STANDING_ENVS if stage in (2, 4, 5) else 0.0
      )
    ),
    "rel_heading_envs": 0.0,
    "rel_forward_envs": 0.0,
    "heading_command": False,
    "debug_vis": play,
    "ranges": UniformVelocityCommandCfg.Ranges(
      lin_vel_x=stage_cfg.lin_vel_x_range,
      lin_vel_y=(0.0, 0.0),
      ang_vel_z=stage_cfg.yaw_rate_range,
    ),
  }
  if stage == 1:
    velocity_command_kwargs["nominal_abs_range"] = (
      STAGE1_ACTIVE_LIN_VEL_X_ABS_RANGE
    )
    velocity_command_kwargs["extension_abs_range"] = (
      STAGE1_EXTENSION_LIN_VEL_X_ABS_RANGE
    )
    velocity_command_kwargs["rel_extension_envs"] = STAGE1_EXTENSION_ENVS
  if stage == 2:
    velocity_command_kwargs["rel_linear_retention_envs"] = (
      STAGE2_LINEAR_RETENTION_ENVS
    )
    velocity_command_kwargs["linear_retention_abs_range"] = (
      STAGE2_LINEAR_RETENTION_ABS_RANGE
    )

  cfg.commands = {
    "twist": velocity_command_cfg_cls(
      **velocity_command_kwargs,
    ),
    "posture": PostureCommandCfg(
      resampling_time_range=(5.0, 10.0),
      debug_vis=False,
      height_range=(
        posture.height_range
        if stage_cfg.posture_commands and posture.qualified
        else (ROOT_HEIGHT_TARGET, ROOT_HEIGHT_TARGET)
      ),
      pitch_range=(
        posture.pitch_range
        if stage_cfg.posture_commands and posture.qualified
        else (0.0, 0.0)
      ),
      qualified=posture.qualified and stage_cfg.posture_commands,
      source=posture.source,
      map_hash=posture.map_hash,
    ),
  }

  actor_terms = {
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "velocity_command": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "posture_command": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "posture"},
    ),
    "joint_pos": ObservationTermCfg(
      func=joint_pos_rel_without_wheel_position,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "wheel_joint_names": WHEEL_JOINT_NAMES,
      },
      noise=Unoise(n_min=-0.002, n_max=0.002),
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": SceneEntityCfg("robot")},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "controller_baseline": ObservationTermCfg(
      func=controller_baseline_observation
    ),
    "applied_residual": ObservationTermCfg(func=applied_residual_observation),
  }
  critic_terms = {
    name: ObservationTermCfg(func=term.func, params=dict(term.params))
    for name, term in actor_terms.items()
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  if stage_cfg.posture_commands:
    cfg.rewards.pop("flat_orientation_l2", None)
    cfg.rewards.pop("root_height_l2", None)
    cfg.rewards["posture_height_l2"] = RewardTermCfg(
      func=posture_height_l2,
      weight=-10.0,
      params={"command_name": "posture"},
    )
    cfg.rewards["posture_pitch_l2"] = RewardTermCfg(
      func=posture_pitch_l2,
      weight=-6.0,
      params={"command_name": "posture"},
    )

  if stage in (1, 2, 5) and not play:
    cfg.events["push_robot"] = EventTermCfg(
      func=envs_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=stage_cfg.push_interval_s,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "velocity_range": {
          "x": (-stage_cfg.push_lin_vel_x, stage_cfg.push_lin_vel_x),
          "pitch": (-stage_cfg.push_pitch_rate, stage_cfg.push_pitch_rate),
        },
      },
    )
  else:
    cfg.events.pop("push_robot", None)
    cfg.events.pop("slow_speed_turn_push_robot", None)

  if stage in (1, 2) and not play:
    cfg.events["stage1_mild_mismatch"] = stage1_mismatch_event_cfg()
  if stage == 1 and not play:
    setattr(cfg, "stage1_profile_version", STAGE1_MISMATCH_PROFILE_VERSION)

  return cfg


def _uniform_se3_samples(
  ranges: dict[str, tuple[float, float]] | None,
  count: int,
  device: torch.device | str,
) -> torch.Tensor:
  """Sample (x, y, z, roll, pitch, yaw) uniformly, mjlab's key convention.

  Written out rather than importing mjlab's private `_sample_se3_range` so
  the camp does not depend on a private symbol.
  """

  keys = ("x", "y", "z", "roll", "pitch", "yaw")
  source = ranges or {}
  low = torch.tensor(
    [float(source.get(key, (0.0, 0.0))[0]) for key in keys], device=device
  )
  high = torch.tensor(
    [float(source.get(key, (0.0, 0.0))[1]) for key in keys], device=device
  )
  return low + (high - low) * torch.rand((count, 6), device=device)


def reset_root_to_stair_approach(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]] | None = None,
  root_height: float = ROOT_HEIGHT_TARGET,
  x_offset_from_origin_m: float | None = None,
  flat_env_count: int | None = None,
) -> None:
  """Reset the robot onto the approach run, 0.25 m short of the first riser.

  Required, not cosmetic. `env_origins` for a generated pyramid staircase
  sits on the TOP platform, so its z is the terrain-dependent platform
  elevation rather than the approach-ground elevation. mjlab's stock
  `reset_root_state_uniform`
  places the robot at `default_z + origin_z`, which spawns it that far above
  the flat outer border and drops it. The camp therefore replaces that event
  rather than running after it, and pins z in ABSOLUTE world coordinates,
  following `probe_hybrid_stair_height._reset_to_approach` so the camp starts
  from the same pose the classical boundary C* was measured from.

  Composition mirrors `reset_root_state_uniform` exactly - same
  `default_root_state` base, same uniform pose/velocity offsets, same
  `quat_mul(default, delta)` order - and the ranges are read from the Stage5
  event this supersedes, so the camp inherits the identical disturbance
  envelope with no re-typed constants.

  It must NOT be written as an event that runs after the stock reset and
  patches the position: inside a reset event `robot.data.root_link_*` is a
  stale view of the PRE-reset state (measured: it reads all-zero position on
  the first reset), so reading the pose back and rewriting it re-applies a
  crashed attitude every episode. That formulation measured 96
  `bad_orientation` terminations in 60 steps against a Stage5 control of 8.

  `x_offset_from_origin_m` exists for FLAT evaluation sessions only. On a
  flat tile there is no riser, so the stair-approach spawn (origin - 3.25 m)
  is meaningless - and measured harmful: it parks the robot 0.75 m from the
  west tile seam, and 58/58 stair-mode false latches in the flat FP
  diagnostic occurred at x in [-4.075, -4.016] under the backward command,
  where mesh-edge contact normals reach 140 N and overlap the 20.96 N frozen
  stair-impact floor (no threshold separates them). Passing 0.0 spawns at
  the tile center; the seam margin itself comes from the enlarged 16 m flat
  evaluation tile (deviation minute 7 - evaluation drives the full 3000-step
  block as ONE episode, 4.2 m of travel, which the 8 m tile cannot hold from
  any spawn point). The default None preserves the registered
  stair-approach behavior byte-for-byte; the TRAINING event never sets this
  key, so the canonical contract hash is unchanged.
  """

  ids = (
    torch.arange(env.num_envs, device=env.device)
    if env_ids is None
    else env_ids
  )
  if len(ids) == 0:
    return
  robot = env.scene["robot"]
  root_states = robot.data.default_root_state[ids].clone()
  pose_samples = _uniform_se3_samples(pose_range, len(ids), env.device)
  velocity_samples = _uniform_se3_samples(velocity_range, len(ids), env.device)
  origins = env.scene.env_origins[ids]

  # Start from mjlab's stock reset exactly.  Hybrid-v3 keeps this branch for
  # the first ``flat_env_count`` retention slots and overrides only stair slots.
  positions = root_states[:, 0:3] + pose_samples[:, 0:3] + origins
  if flat_env_count is None:
    stair_slots = torch.ones(len(ids), device=env.device, dtype=torch.bool)
  else:
    if flat_env_count < 0 or flat_env_count > env.num_envs:
      raise ValueError("flat_env_count must stay within [0, num_envs].")
    stair_slots = ids >= int(flat_env_count)
  if bool(stair_slots.any()):
    stair_origins = origins[stair_slots]
    if x_offset_from_origin_m is None:
      spawn_x = stair_origins[:, 0] - (
        STAIR_CAMP_RISER_OFFSET_M + STAIR_CAMP_START_OFFSET_M
      )
    else:
      spawn_x = stair_origins[:, 0] + float(x_offset_from_origin_m)
    positions[stair_slots, 0] = spawn_x + pose_samples[stair_slots, 0]
    positions[stair_slots, 1] = (
      stair_origins[:, 1] + pose_samples[stair_slots, 1]
    )
    positions[stair_slots, 2] = (
      float(root_height) + pose_samples[stair_slots, 2]
    )
  orientations = quat_mul(
    root_states[:, 3:7],
    quat_from_euler_xyz(
      pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    ),
  )
  robot.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=ids
  )
  robot.write_root_link_velocity_to_sim(
    root_states[:, 7:13] + velocity_samples, env_ids=ids
  )


def reset_roll_assist_posture_consistent(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]] | None = None,
  root_height: float = ROOT_HEIGHT_TARGET,
  x_offset_from_origin_m: float | None = None,
  flat_env_count: int = ROLL_ASSIST_FLAT_ENVS,
  stair_posture_height: float = ROLL_ASSIST_STAIR_POSTURE_HEIGHT_M,
  stair_posture_pitch: float = ROLL_ASSIST_STAIR_POSTURE_PITCH_RAD,
) -> None:
  """Reset stair slots to the same root/leg posture-map state as R0."""

  ids = (
    torch.arange(env.num_envs, device=env.device)
    if env_ids is None
    else env_ids
  )
  if len(ids) == 0:
    return
  robot = env.scene["robot"]
  root_states = robot.data.default_root_state[ids].clone()
  # Preserve the Stage5 disturbance sampler only for flat-retention slots.
  # Stair slots use exactly the same axes/bounds as the formal R0 protocol;
  # random roll/root pitch would violate the posture-card reset contract.
  pose_samples = _uniform_se3_samples(pose_range, len(ids), env.device)
  velocity_samples = _uniform_se3_samples(velocity_range, len(ids), env.device)
  origins = env.scene.env_origins[ids]
  positions = root_states[:, 0:3] + pose_samples[:, 0:3] + origins
  orientations = quat_mul(
    root_states[:, 3:7],
    quat_from_euler_xyz(
      pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    ),
  )
  stair_slots = ids >= int(flat_env_count)
  if bool(stair_slots.any()):
    pose_samples[stair_slots] = 0.0
    velocity_samples[stair_slots] = 0.0
    stair_origins = origins[stair_slots]
    if x_offset_from_origin_m is None:
      spawn_x = stair_origins[:, 0] - (
        ROLL_ASSIST_RISER_OFFSET_M + ROLL_ASSIST_START_OFFSET_M
      )
    else:
      spawn_x = stair_origins[:, 0] + float(x_offset_from_origin_m)
    positions[stair_slots, 0] = spawn_x + pose_samples[stair_slots, 0]
    positions[stair_slots, 1] = (
      stair_origins[:, 1] + pose_samples[stair_slots, 1]
    )
    positions[stair_slots, 2] = (
      float(root_height) + pose_samples[stair_slots, 2]
    )
    # R0 binds the root orientation to the posture card.  RollAssist has one
    # fixed stair posture card, so its stair root pitch is exactly that card;
    # Stage5 random attitude remains unchanged for flat retention slots.
    orientations[stair_slots] = quat_from_euler_xyz(
      torch.zeros_like(positions[stair_slots, 0]),
      torch.full_like(
        positions[stair_slots, 0], float(stair_posture_pitch)
      ),
      torch.zeros_like(positions[stair_slots, 0]),
    )
  robot.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=ids
  )
  robot.write_root_link_velocity_to_sim(
    root_states[:, 7:13] + velocity_samples, env_ids=ids
  )
  stair_ids = ids[stair_slots]
  if len(stair_ids) == 0:
    return
  action = _roll_assist_action(env)
  leg_ids, leg_names = robot.find_joints(LEG_JOINT_NAMES, preserve_order=True)
  if tuple(leg_names) != LEG_JOINT_NAMES:
    raise RuntimeError("RollAssist leg joint order drifted.")
  dtype = robot.data.joint_pos.dtype
  height = torch.full(
    (len(stair_ids),), float(stair_posture_height),
    device=env.device, dtype=dtype,
  )
  pitch = torch.full_like(height, float(stair_posture_pitch))
  coefficients = action._posture_coefficients.to(device=env.device, dtype=dtype)
  features = torch.stack((torch.ones_like(height), height, pitch), dim=1)
  targets = features @ coefficients
  robot.write_joint_state_to_sim(
    targets, torch.zeros_like(targets), joint_ids=leg_ids, env_ids=stair_ids,
  )


def push_flat_retention_envs(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  *,
  flat_env_count: int,
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg,
) -> None:
  """Apply the unchanged Stage5 interval push to flat retention slots only."""

  if flat_env_count < 0 or flat_env_count > env.num_envs:
    raise ValueError("flat_env_count must stay within [0, num_envs].")
  all_ids = torch.arange(env.num_envs, device=env.device)
  if env_ids is None:
    ids = all_ids
  elif isinstance(env_ids, slice):
    ids = all_ids[env_ids]
  else:
    ids = env_ids
  ids = ids[ids < int(flat_env_count)]
  if len(ids):
    envs_mdp.push_by_setting_velocity(
      env,
      ids,
      velocity_range=velocity_range,
      asset_cfg=asset_cfg,
    )


def make_stair_camp_env_cfg(
  play: bool = False,
  *,
  initial_upper_height_m: float = STAIR_CAMP_CURRICULUM_INITIAL_UPPER_M,
  steps_per_iteration: int = STAIR_CAMP_STEPS_PER_ITERATION,
  leg_residual_scale: float = STAIR_CAMP_LEG_RESIDUAL_SCALE,
  **artifact_paths: Path | None,
) -> ManagerBasedRlEnvCfg:
  """Build the residual stair camp (mainline doc S5B).

  Starts from the frozen Stage5 classical stack - same controller schedule,
  yaw feedforward, posture map and station calibration - and changes only
  what S5B registers: legs-only residual authority at 0.070 rad, staircase
  terrain, the CTBC-style contact trigger with freeze-at-trigger leg
  release, the camp observation groups, the two gated stair rewards and a
  finite episode.

  Deliberately NOT built on `probe_hybrid_stair_height.make_stair_env_cfg`:
  that is an evaluation probe builder (play mode, `episode_length_s = 1e9`,
  a fixed tuple of heights, no reward/curriculum wiring) and reusing it
  would silently produce a non-episodic training env.

  Args:
    play: build the play variant (no pushes, debug vis).
    initial_upper_height_m: `upper_height_m` of the initial
      `stair_residual.StairCurriculumState`. S5B pins it to 0.01 m; the
      curriculum term advances it from there with the frozen
      `update_stair_curriculum`. The terrain grid always spans the full
      0.15 m cap - the band is enforced by the curriculum's per-reset row
      assignment, not by the terrain, because a terrain cannot be
      regenerated mid-run.
    steps_per_iteration: `num_steps_per_env` of the runner this env is
      registered with. Only used to convert the registered 50-ITERATION
      evaluation cadence into the env-step counter the curriculum sees; a
      contract test pins it against the actual runner cfg.
    leg_residual_scale: leg residual authority, [U]-confirmed at 0.070 rad.
  """

  if not math.isfinite(initial_upper_height_m):
    raise ValueError("Initial upper height must be finite.")
  if initial_upper_height_m > STAIR_CAMP_MAX_HEIGHT_M + 1.0e-12:
    raise ValueError(
      "Initial upper height exceeds the registered "
      f"{STAIR_CAMP_MAX_HEIGHT_M} m cap."
    )
  level = _stair_camp_level(initial_upper_height_m)
  if abs(initial_upper_height_m - level * STAIR_CAMP_HEIGHT_STEP_M) > 1.0e-9:
    raise ValueError(
      "Initial upper height must land on the registered "
      f"{STAIR_CAMP_HEIGHT_STEP_M} m ladder."
    )
  if level < STAIR_CAMP_CURRICULUM_LOWER_LEVEL:
    raise ValueError(
      "The registered tier grid starts at "
      f"{STAIR_CAMP_HEIGHT_STEP_M} m and its lower bound never moves, so "
      "the initial upper height cannot sit below it."
    )
  if steps_per_iteration < 1:
    raise ValueError("steps_per_iteration must be positive.")

  cfg = make_hoppertrex_hybrid_env_cfg(
    stage=5,
    play=play,
    leg_residual_scale=leg_residual_scale,
    **artifact_paths,
  )
  if not play:
    cfg.scene.num_envs = 256
  setattr(cfg, "stair_camp_task_id", STAIR_CAMP_TASK_ID)
  setattr(cfg, "stair_camp_zero_initialize_actor_output", True)
  setattr(cfg, "stair_camp_training_contract", not play)
  setattr(
    cfg,
    "stair_camp_contract_schema_version",
    STAIR_CAMP_CONTRACT_SCHEMA_VERSION,
  )
  # The full hash includes the qualified artifact bindings and PPO config, so
  # train preflight binds it only after both surfaces are available. Ordinary
  # Stage0--5 configs intentionally do not carry this StairCamp-only field.
  setattr(cfg, "stair_camp_contract_sha256", None)

  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      curriculum=True,
      size=STAIR_CAMP_TERRAIN_SIZE_M,
      num_rows=STAIR_CAMP_TERRAIN_ROWS,
      num_cols=1,
      difficulty_range=(0.0, 1.0),
      sub_terrains={
        "stair": pyramid_stairs(
          proportion=1.0,
          step_height_range=(0.0, STAIR_CAMP_MAX_HEIGHT_M),
          step_width=STAIR_CAMP_STEP_WIDTH_M,
          platform_width=STAIR_CAMP_PLATFORM_WIDTH_M,
          border_width=STAIR_CAMP_TERRAIN_BORDER_WIDTH_M,
        )
      },
    ),
    # The initial draw only has to be legal; the curriculum term rewrites
    # every resetting env's row to the registered tier grid before its first
    # episode, including on the very first reset. No `randomize_terrain`
    # event is added - that helper re-draws over the FULL grid and would
    # silently break the band.
    max_init_terrain_level=level,
    num_envs=cfg.scene.num_envs,
  )
  cfg.scene.sensors = tuple(cfg.scene.sensors) + (stair_trigger_sensor_cfg(),)
  cfg.episode_length_s = STAIR_CAMP_EPISODE_LENGTH_S

  action_cfg = cfg.actions["hybrid_wheel_leg"]
  action_cfg.action_mask = STAIR_CAMP_ACTION_MASK
  action_cfg.stair_trigger_sensor_name = STAIR_TRIGGER_SENSOR_NAME
  action_cfg.stair_mode_freezes_leg_reference = True

  # Spawn on the approach, not on the staircase. The camp REPLACES the Stage5
  # root reset rather than running after it (see the event docstring: a
  # patch-afterwards formulation reads a stale pre-reset pose and re-applies
  # crashed attitudes). The disturbance ranges are read out of the event being
  # replaced, so the camp inherits Stage5's exact envelope.
  inherited_reset = cfg.events.pop("reset_root_state_with_small_disturbance", None)
  if inherited_reset is None:
    raise ValueError(
      "Stage5 must provide the root-state reset the camp replaces."
    )
  cfg.events["reset_root_to_stair_approach"] = EventTermCfg(
    func=reset_root_to_stair_approach,
    mode="reset",
    params={
      "root_height": ROOT_HEIGHT_TARGET,
      "pose_range": inherited_reset.params["pose_range"],
      "velocity_range": inherited_reset.params["velocity_range"],
    },
  )

  actor_terms = stair_camp_actor_terms(cfg.observations["actor"].terms)
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms=stair_camp_critic_terms(
        {
          name: ObservationTermCfg(func=term.func, params=dict(term.params))
          for name, term in actor_terms.items()
        }
      ),
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  cfg.rewards["stair_progress"] = RewardTermCfg(
    func=stair_mode_forward_progress,
    weight=STAIR_CAMP_PROGRESS_WEIGHT,
    params={},
  )
  cfg.rewards["stair_climb_success"] = RewardTermCfg(
    func=stair_camp_climb_success,
    weight=STAIR_CAMP_CLIMB_SUCCESS_WEIGHT,
    params={},
  )

  # S5B Protocol 4. Without this the band never moves: `max_init_terrain_level`
  # is an INITIAL draw, so the registered "0.80 x 3 evaluations -> +0.01 m"
  # promotion would never happen and a 1000-iteration run would spend its whole
  # budget on the initial tier. The term also enforces the registered tier grid
  # {0.01, ..., upper}; mjlab's own initial draw spans [0, upper] and would put
  # roughly half the envs on flat ground at the registered initial state.
  cadence_params = {
    "evaluation_interval_steps": (
      STAIR_CAMP_EVALUATION_INTERVAL_ITERS * steps_per_iteration
    ),
    "initial_upper_height_m": initial_upper_height_m,
  }
  cfg.curriculum = {
    "stair_height_band": CurriculumTermCfg(
      func=stair_camp_curriculum,
      params=dict(cadence_params),
    )
  }
  if not play:
    cfg.metrics = {
      **dict(getattr(cfg, "metrics", {})),
      "stair_camp_step": MetricsTermCfg(
        func=stair_camp_step_metric,
        params=dict(cadence_params),
        reduce="last",
      ),
    }
  validate_stair_camp_observation_contract(cfg)
  return cfg


@dataclass(kw_only=True)
class RollAssistVelocityCommandCfg(HybridPlanarVelocityCommandCfg):
  flat_env_count: int = ROLL_ASSIST_FLAT_ENVS
  stair_vx: float = ROLL_ASSIST_COMMAND_VX_MPS

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.flat_env_count < 0 or not math.isfinite(self.stair_vx) or self.stair_vx <= 0.0:
      raise ValueError("RollAssist fixed stair command is invalid.")

  def build(self, env: ManagerBasedRlEnv) -> RollAssistVelocityCommand:
    return RollAssistVelocityCommand(self, env)


class RollAssistVelocityCommand(HybridPlanarVelocityCommand):
  """Expose and control the same zero command during each 2 s stair settle."""

  cfg: RollAssistVelocityCommandCfg

  def _apply_stair_command(self, env_ids: torch.Tensor) -> None:
    stair_ids = env_ids[env_ids >= self.cfg.flat_env_count]
    if not len(stair_ids):
      return
    settled = self._env.episode_length_buf[stair_ids] >= ROLL_ASSIST_SETTLE_STEPS
    self.vel_command_b[stair_ids, :] = 0.0
    self.vel_command_b[stair_ids, 0] = torch.where(
      settled,
      torch.full_like(self.vel_command_b[stair_ids, 0], self.cfg.stair_vx),
      torch.zeros_like(self.vel_command_b[stair_ids, 0]),
    )
    self.vel_command_w[stair_ids] = self.vel_command_b[stair_ids]
    self.is_standing_env[stair_ids] = ~settled
    self.is_heading_env[stair_ids] = False
    self.is_world_env[stair_ids] = False
    self.is_forward_env[stair_ids] = False

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self._apply_stair_command(env_ids)

  def _update_command(self) -> None:
    super()._update_command()
    all_ids = torch.arange(self._env.num_envs, device=self.device)
    self._apply_stair_command(all_ids)


@dataclass(kw_only=True)
class RollAssistPostureCommandCfg(PostureCommandCfg):
  flat_env_count: int = ROLL_ASSIST_FLAT_ENVS
  stair_height: float = ROOT_HEIGHT_TARGET
  stair_pitch: float = 0.0

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.flat_env_count < 0 or not all(
      math.isfinite(value) for value in (self.stair_height, self.stair_pitch)
    ):
      raise ValueError("RollAssist fixed posture is invalid.")

  def build(self, env: ManagerBasedRlEnv) -> RollAssistPostureCommand:
    return RollAssistPostureCommand(self, env)


class RollAssistPostureCommand(PostureCommand):
  cfg: RollAssistPostureCommandCfg

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    stair_ids = env_ids[env_ids >= self.cfg.flat_env_count]
    if len(stair_ids):
      self._target[stair_ids, 0] = self.cfg.stair_height
      self._target[stair_ids, 1] = self.cfg.stair_pitch
      self._command[stair_ids] = self._target[stair_ids]

def _dataclass_init_kwargs(value: object) -> dict[str, object]:
  return {
    field.name: getattr(value, field.name)
    for field in fields(value)
    if field.init
  }


def _default_unqualified_dynamic_maneuver() -> DynamicStairManeuver:
  """Registration-only seed maneuver; formal training requires a CEM artifact."""

  return DynamicStairManeuver(
    lift_mode=DynamicLiftMode.ALTERNATING,
    split_amplitude_rad=0.035,
    lift_amplitude_rad=0.045,
    trailing_delay_s=0.20,
    drive_feedforward_radps=1.0,
    source="registration-unqualified",
  )


def _validate_dynamic_maneuver_classical_bindings(
  maneuver: DynamicStairManeuver,
  action: HybridWheelLegActionCfg,
) -> None:
  bindings = maneuver.bindings
  if bindings is None:
    raise ValueError("Qualified StairDynamic maneuver has no bindings.")
  expected = {
    "controller_gain_hash": action.controller_gain_hash,
    "calibration_hash": action.calibration_hash,
    "yaw_calibration_hash": action.yaw_calibration_hash,
    "posture_map_hash": action.posture_map_hash,
    "posture_artifact_hash": action.posture_artifact_hash,
    "station_calibration_hash": action.station_calibration_hash,
  }
  for name, value in expected.items():
    if not isinstance(value, str) or not value:
      raise ValueError(
        f"StairDynamic classical artifact {name} is missing."
      )
    if bindings.get(name) != value:
      raise ValueError(
        f"StairDynamic maneuver was built with a different {name}."
      )


class RollAssistCurriculum:
  """Assign flat retention plus exactly Hpass -> optional Hnext."""

  def __init__(self, env: ManagerBasedRlEnv, flat_env_count: int, hpass_m: float, hnext_m: float):
    if flat_env_count < 0 or flat_env_count > env.num_envs:
      raise ValueError("RollAssist flat env count is invalid.")
    self.state = RollAssistCurriculumState(hpass_m=hpass_m, hnext_m=hnext_m)
    self.flat_env_count = int(flat_env_count)
    self.last_processed_step = -1

  @property
  def active_level(self) -> int:
    return 1 if self.state.switched_to_hnext else 0

  def _assign(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
    terrain = env.scene.terrain
    assert terrain is not None and terrain.terrain_origins is not None
    flat = env_ids < self.flat_env_count
    flat_ids, stair_ids = env_ids[flat], env_ids[~flat]
    if len(flat_ids):
      terrain.terrain_types[flat_ids] = 0
      terrain.terrain_levels[flat_ids] = 0
    if len(stair_ids):
      terrain.terrain_types[stair_ids] = 1
      terrain.terrain_levels[stair_ids] = self.active_level
    if len(env_ids):
      terrain.env_origins[env_ids] = terrain.terrain_origins[
        terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
      ]

  def _record_pre_reset_episodes(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor
  ) -> None:
    if self.state.decision_made or not len(env_ids):
      return
    stair_ids = env_ids[env_ids >= self.flat_env_count]
    if not len(stair_ids):
      return
    evidence = _roll_assist_episode_evidence(env)
    terminated = env.reset_terminated[stair_ids].bool()
    bilateral = evidence.bilateral_airborne_ever[stair_ids]
    try:
      non_wheel = env.termination_manager.get_term("non_wheel_ground_contact")[
        stair_ids
      ].bool()
    except (AttributeError, KeyError):
      non_wheel = torch.zeros_like(terminated)
    self.state.record_completed_episodes(
      completed=len(stair_ids),
      successes=int(evidence.success[stair_ids].sum().item()),
      terminations=int(terminated.sum().item()),
      non_wheel_contacts=int(non_wheel.sum().item()),
      bilateral_airborne=int(bilateral.sum().item()),
    )

  def compute(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> dict[str, float]:
    self._assign(env, env_ids)
    return {
      "active_height_m": self.state.active_height_m,
      "switched_to_hnext": float(self.state.switched_to_hnext),
      "completed_stair_episodes": float(self.state.completed_stair_episodes),
      "online_success_rate": self.state.online_success_rate,
    }

  def record_step(self, env: ManagerBasedRlEnv) -> None:
    step = int(env.common_step_counter)
    if step <= self.last_processed_step:
      return
    finished = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    self._record_pre_reset_episodes(env, finished)
    # A complete update contains exactly 24 environment steps.  Therefore
    # common step 599 is still inside update 25 and step 600 is its final
    # collected sample; floor division prevents an early decision at step 577.
    completed_updates = step // ROLL_ASSIST_STEPS_PER_UPDATE
    if not self.state.decision_made and completed_updates >= ROLL_ASSIST_SWITCH_UPDATE:
      if completed_updates != ROLL_ASSIST_SWITCH_UPDATE:
        raise ValueError("RollAssist missed its immutable update-25 decision boundary.")
      self.state.evaluate_update25(completed_updates=ROLL_ASSIST_SWITCH_UPDATE)
      # Do not rewrite origins for live Hpass episodes. The active level is
      # applied by CurriculumManager to each env at its next reset.
    self.last_processed_step = step

  def state_dict(self) -> dict[str, Any]:
    return {
      **self.state.state_dict(),
      "flat_env_count": self.flat_env_count,
      "last_processed_step": self.last_processed_step,
    }

  def load_state_dict(self, payload: Mapping[str, Any]) -> None:
    expected = set(self.state.state_dict()) | {"flat_env_count", "last_processed_step"}
    if not isinstance(payload, Mapping) or set(payload) != expected:
      raise ValueError("RollAssist runtime curriculum state schema drifted.")
    flat_count = payload["flat_env_count"]
    last_step = payload["last_processed_step"]
    if isinstance(flat_count, bool) or not isinstance(flat_count, int):
      raise TypeError("RollAssist flat_env_count must be an integer.")
    if flat_count != self.flat_env_count:
      raise ValueError("RollAssist restored flat/stair split drifted.")
    if isinstance(last_step, bool) or not isinstance(last_step, int) or last_step < -1:
      raise ValueError("RollAssist last_processed_step is invalid.")
    state_payload = {name: payload[name] for name in self.state.state_dict()}
    restored = RollAssistCurriculumState.from_state_dict(state_payload)
    if (
      not math.isclose(restored.hpass_m, self.state.hpass_m, abs_tol=1.0e-12)
      or not math.isclose(restored.hnext_m, self.state.hnext_m, abs_tol=1.0e-12)
    ):
      raise ValueError("RollAssist restored height pair drifted.")
    self.state = restored
    self.last_processed_step = last_step

  def progress_snapshot(self) -> dict[str, Any]:
    return {
      **self.state.state_dict(),
      "active_height_m": self.state.active_height_m,
      "online_success_rate": self.state.online_success_rate,
    }


def _roll_assist_curriculum_state(
  env: ManagerBasedRlEnv, flat_env_count: int, hpass_m: float, hnext_m: float
) -> RollAssistCurriculum:
  state = getattr(env, "roll_assist_curriculum_state", None)
  if state is None:
    state = RollAssistCurriculum(env, flat_env_count, hpass_m, hnext_m)
    env.roll_assist_curriculum_state = state  # type: ignore[attr-defined]
  elif not isinstance(state, RollAssistCurriculum):
    raise ValueError("RollAssist curriculum state has invalid type.")
  elif state.flat_env_count != flat_env_count or not math.isclose(state.state.hpass_m, hpass_m) or not math.isclose(state.state.hnext_m, hnext_m):
    raise ValueError("RollAssist curriculum configuration drifted.")
  return state


def roll_assist_curriculum(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice | None,
  flat_env_count: int, hpass_m: float, hnext_m: float,
) -> dict[str, float]:
  state = _roll_assist_curriculum_state(env, flat_env_count, hpass_m, hnext_m)
  all_ids = torch.arange(env.num_envs, device=env.device)
  ids = all_ids if env_ids is None else all_ids[env_ids] if isinstance(env_ids, slice) else env_ids
  return state.compute(env, ids)


def roll_assist_repository_head() -> str:
  return subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_PATH, check=True,
    capture_output=True, text=True,
  ).stdout.strip()


def _roll_assist_default_paths() -> tuple[Path | None, Path | None]:
  verdict_text = os.environ.get(ROLL_ASSIST_VERDICT_PATH_ENV)
  reward_text = os.environ.get(ROLL_ASSIST_REWARD_CALIBRATION_PATH_ENV)
  return (
    None if verdict_text is None else Path(verdict_text),
    None if reward_text is None else Path(reward_text),
  )


def reward_content_sha256(path: Path) -> str:
  """Validate a reward artifact and return its canonical self-hash."""

  payload = json.loads(path.read_text(encoding="utf-8-sig"))
  if not isinstance(payload, Mapping):
    raise TypeError("RollAssist reward calibration must be a JSON object.")
  return str(validate_reward_calibration(payload)["calibration_sha256"])


def validate_roll_assist_observation_contract(cfg: ManagerBasedRlEnvCfg) -> None:
  if getattr(cfg, "roll_assist_task_id", None) != ROLL_ASSIST_TASK_ID:
    raise ValueError("RollAssist task marker is missing.")
  actor, critic = cfg.observations.get("actor"), cfg.observations.get("critic")
  if actor is None or critic is None:
    raise ValueError("RollAssist actor/critic groups are required.")
  if tuple(actor.terms) != ROLL_ASSIST_ACTOR_TERMS:
    raise ValueError("RollAssist actor is not the original Stage5 proprioceptive interface.")
  if tuple(critic.terms) != ROLL_ASSIST_ACTOR_TERMS + ROLL_ASSIST_CRITIC_TAIL:
    raise ValueError("RollAssist critic privileged tail drifted.")
  if sum(ROLL_ASSIST_TERM_WIDTHS[name] for name in actor.terms) != ROLL_ASSIST_ACTOR_WIDTH:
    raise ValueError("RollAssist actor width is not 34.")
  if sum(ROLL_ASSIST_TERM_WIDTHS[name] for name in critic.terms) != ROLL_ASSIST_CRITIC_WIDTH:
    raise ValueError("RollAssist critic width is not 38.")
  action = cfg.actions.get("hybrid_wheel_leg")
  if tuple(action.action_mask) != ROLL_ASSIST_ACTION_MASK:
    raise ValueError("RollAssist runtime action mask drifted.")
  if tuple(float(value) for value in action.action_scales) != ROLL_ASSIST_ACTION_SCALES:
    raise ValueError("RollAssist action scales drifted.")
  if (
    action.controller_gain_hash != ROLL_ASSIST_CONTROLLER_SCHEDULE_HASH
    or not action.controller_qualified
    or not action.yaw_calibration_qualified
    or not action.posture_map_qualified
    or not action.station_calibration_qualified
  ):
    raise ValueError("RollAssist is not bound to the frozen final-C1 artifact stack.")
  if (
    action.dynamic_stair_maneuver is not None
    or action.dynamic_stair_request_command_name is not None
    or action.dynamic_stair_left_sensor_name is not None
    or action.dynamic_stair_right_sensor_name is not None
    or action.stair_trigger_sensor_name is not None
    or action.stair_mode_freezes_leg_reference
    or action.stair_mode_forced
  ):
    raise ValueError("RollAssist must not enable trigger, freeze, feedforward, or dynamic FSM.")
  if cfg.seed != 1:
    raise ValueError("RollAssist environment seed must remain 1.")
  if (
    cfg.decimation != ROLL_FIRST_CONTROL_DECIMATION
    or not math.isclose(
      float(cfg.sim.mujoco.timestep), ROLL_FIRST_PHYSICS_TIMESTEP_S,
      rel_tol=0.0, abs_tol=1.0e-12,
    )
  ):
    raise ValueError("RollAssist must preserve 50 Hz control and 5 ms physics.")
  robot_cfg = cfg.scene.entities.get("robot")
  if robot_cfg is None or len(robot_cfg.collisions) != 1:
    raise ValueError("RollAssist robot collision config count drifted.")
  wheel_collision = robot_cfg.collisions[0]
  if (
    tuple(wheel_collision.solref.get("wheel_.*_collision", ()))
    != ROLL_FIRST_WHEEL_CONTACT_SOLREF
    or tuple(wheel_collision.solimp.get("wheel_.*_collision", ()))
    != ROLL_FIRST_WHEEL_CONTACT_SOLIMP
  ):
    raise ValueError("RollAssist wheel-contact model differs from R0.")
  reset = cfg.events.get("reset_root_to_roll_assist")
  if reset is None or reset.func is not reset_roll_assist_posture_consistent:
    raise ValueError("RollAssist reset is not posture-map consistent.")
  if cfg.scene.num_envs == ROLL_ASSIST_NUM_ENVS:
    if getattr(cfg, "roll_assist_flat_env_count", None) != ROLL_ASSIST_FLAT_ENVS:
      raise ValueError("RollAssist flat/stair split marker drifted.")
    terrain = cfg.scene.terrain
    generator = None if terrain is None else terrain.terrain_generator
    if (
      generator is None
      or generator.seed != 1
      or generator.num_rows != ROLL_ASSIST_TERRAIN_ROWS
      or generator.num_cols != 2
      or tuple(generator.sub_terrains) != ("flat_retention", "stair_roll_assist")
    ):
      raise ValueError("RollAssist two-column/two-level terrain contract drifted.")
    stair_cfg = generator.sub_terrains["stair_roll_assist"]
    expected_heights = (
      float(cfg.roll_assist_hpass_m),
      float(cfg.roll_assist_hnext_m),
    )
    if tuple(float(value) for value in stair_cfg.step_height_range) != expected_heights:
      raise ValueError("RollAssist Hpass/Hnext terrain heights drifted.")
  expected_reset_x = -(ROLL_ASSIST_RISER_OFFSET_M + ROLL_ASSIST_START_OFFSET_M)
  if (
    reset is None
    or not math.isclose(
      float(reset.params.get("x_offset_from_origin_m", math.nan)), expected_reset_x
    )
    or not math.isclose(
      float(reset.params.get("stair_posture_height", math.nan)),
      ROLL_ASSIST_STAIR_POSTURE_HEIGHT_M
    )
    or not math.isclose(
      float(reset.params.get("stair_posture_pitch", math.nan)),
      ROLL_ASSIST_STAIR_POSTURE_PITCH_RAD
    )
    or reset.params.get("pose_range") is not None
    or reset.params.get("velocity_range") is not None
  ):
    raise ValueError("RollAssist reset is not aligned with the R0 riser approach.")
  twist = cfg.commands.get("twist")
  if (
    not isinstance(twist, RollAssistVelocityCommandCfg)
    or twist.flat_env_count != getattr(cfg, "roll_assist_flat_env_count", None)
    or not math.isclose(twist.stair_vx, ROLL_ASSIST_COMMAND_VX_MPS)
    or ROLL_ASSIST_SETTLE_STEPS * cfg.sim.mujoco.timestep * cfg.decimation != 2.0
  ):
    raise ValueError("RollAssist command/settle contract drifted.")
  progress = cfg.rewards.get("roll_assist_progress")
  success = cfg.rewards.get("roll_assist_success")
  if getattr(cfg, "roll_assist_qualified", False) and (
    progress is None
    or success is None
    or not math.isclose(progress.weight, float(cfg.roll_assist_progress_weight))
    or not math.isclose(success.weight, float(cfg.roll_assist_success_weight))
    or not isinstance(cfg.roll_assist_r0_sha256, str)
    or not isinstance(cfg.roll_assist_reward_calibration_sha256, str)
    or not isinstance(cfg.roll_assist_reward_calibration_content_sha256, str)
    or file_sha256(Path(cfg.roll_assist_reward_calibration_path))
    != cfg.roll_assist_reward_calibration_sha256
    or reward_content_sha256(Path(cfg.roll_assist_reward_calibration_path))
    != cfg.roll_assist_reward_calibration_content_sha256
  ):
    raise ValueError("RollAssist reward/artifact bindings drifted.")
  sensors = {sensor.name: sensor for sensor in cfg.scene.sensors}
  for name, geom in {
    ROLL_ASSIST_LEFT_SENSOR_NAME: "wheel_left_collision",
    ROLL_ASSIST_RIGHT_SENSOR_NAME: "wheel_right_collision",
  }.items():
    sensor = sensors.get(name)
    if (
      sensor is None
      or sensor.primary.mode != "geom"
      or sensor.primary.pattern != geom
      or sensor.primary.entity != "robot"
      or sensor.secondary is None
      or sensor.secondary.mode != "body"
      or sensor.secondary.pattern != "terrain"
      or tuple(sensor.fields) != ROLL_ASSIST_SENSOR_FIELDS
      or sensor.reduce != "none"
      or sensor.num_slots != ROLL_ASSIST_SENSOR_SLOTS
    ):
      raise ValueError(f"RollAssist sensor {name!r} drifted.")
  non_wheel = cfg.terminations.get("non_wheel_ground_contact")
  if non_wheel is None:
    raise ValueError("RollAssist non-wheel-contact termination is missing.")
  bilateral = cfg.terminations.get("bilateral_airborne")
  if bilateral is None or bilateral.func is not RollAssistEpisodeEvidence:
    raise ValueError("RollAssist bilateral-airborne termination is missing.")
  substep = cfg.metrics.get("roll_assist_substep_support")
  if (
    substep is None
    or substep.func is not roll_assist_substep_support_metric
    or substep.per_substep is not True
  ):
    raise ValueError("RollAssist strict 5 ms support recorder is missing.")


def make_stair_roll_assist_env_cfg(
  play: bool = False,
  *,
  verdict_path: Path | None = None,
  reward_calibration_path: Path | None = None,
  steps_per_iteration: int = ROLL_ASSIST_STEPS_PER_UPDATE,
  **artifact_paths: Path | None,
) -> ManagerBasedRlEnvCfg:
  """Build continuous-contact leg-only RollAssist from a formal R0 bracket."""

  if steps_per_iteration != ROLL_ASSIST_STEPS_PER_UPDATE:
    raise ValueError("RollAssist is pinned to 24 steps per update.")
  env_verdict, env_reward = _roll_assist_default_paths()
  if verdict_path is not None and env_verdict is not None:
    raise ValueError("Provide RollAssist R0 verdict by argument or environment, not both.")
  if reward_calibration_path is not None and env_reward is not None:
    raise ValueError("Provide reward calibration by argument or environment, not both.")
  resolved_verdict = (verdict_path or env_verdict)
  resolved_reward = (reward_calibration_path or env_reward)
  qualified = resolved_verdict is not None and resolved_reward is not None
  if qualified:
    verdict = load_roll_boundary_verdict(
      resolved_verdict, expected_git_sha=roll_assist_repository_head()
    )
    reward_payload = json.loads(resolved_reward.read_text(encoding="utf-8-sig"))
    reward = validate_reward_calibration(
      reward_payload, expected_roll_boundary_sha256=verdict["file_sha256"]
    )
    reward_file_sha256 = file_sha256(resolved_reward.resolve())
    hpass, hnext = float(verdict["hpass_m"]), float(verdict["hnext_m"])
  else:
    # Registration/smoke placeholder only. train.py rejects this marker.
    hpass, hnext = ROLL_ASSIST_HEIGHT_STEP_M, 2.0 * ROLL_ASSIST_HEIGHT_STEP_M
    verdict = {
      "path": None, "file_sha256": None, "git_sha": None,
      "controller_schedule_hash": None,
    }
    reward = {"progress_weight": 0.0, "success_weight": 0.0,
              "calibration_sha256": None}
    reward_file_sha256 = None
  frozen_artifacts = roll_first_artifact_paths(REPOSITORY_PATH)
  supplied = {name: path for name, path in artifact_paths.items() if path is not None}
  if supplied:
    for name, path in supplied.items():
      if name not in frozen_artifacts:
        raise ValueError(f"Unknown RollAssist artifact path: {name}.")
      if path.resolve() != frozen_artifacts[name]:
        raise ValueError(f"RollAssist {name} differs from the frozen R0 stack.")
  cfg = make_hoppertrex_hybrid_env_cfg(
    stage=5, play=play, leg_residual_scale=ROLL_ASSIST_ACTION_SCALES[2],
    **frozen_artifacts,
  )
  robot_cfg = cfg.scene.entities["robot"]
  if len(robot_cfg.collisions) != 1:
    raise ValueError("RollAssist expects exactly one robot collision config.")
  wheel_collision = robot_cfg.collisions[0]
  wheel_collision.solref["wheel_.*_collision"] = ROLL_FIRST_WHEEL_CONTACT_SOLREF
  wheel_collision.solimp["wheel_.*_collision"] = ROLL_FIRST_WHEEL_CONTACT_SOLIMP
  cfg.scene.num_envs = 1 if play else ROLL_ASSIST_NUM_ENVS
  flat_count = 0 if play else ROLL_ASSIST_FLAT_ENVS
  cfg.roll_assist_task_id = ROLL_ASSIST_TASK_ID
  cfg.roll_assist_training_contract = not play
  cfg.roll_assist_qualified = qualified
  cfg.roll_assist_hpass_m = hpass
  cfg.roll_assist_hnext_m = hnext
  cfg.roll_assist_flat_env_count = flat_count
  cfg.roll_assist_r0_path = verdict["path"]
  cfg.roll_assist_r0_sha256 = verdict["file_sha256"]
  cfg.roll_assist_r0_git_sha = verdict["git_sha"]
  cfg.roll_assist_r0_schedule_hash = verdict["controller_schedule_hash"]
  cfg.roll_assist_reward_calibration_path = (
    None if resolved_reward is None else str(resolved_reward.resolve())
  )
  # This checkpointed SHA binds exact file bytes. The canonical JSON self-hash
  # remains separate so whitespace-only or encoding drift also fails closed.
  cfg.roll_assist_reward_calibration_sha256 = reward_file_sha256
  cfg.roll_assist_reward_calibration_content_sha256 = reward["calibration_sha256"]
  cfg.roll_assist_progress_weight = float(reward["progress_weight"])
  cfg.roll_assist_success_weight = float(reward["success_weight"])
  cfg.roll_assist_settle_steps = ROLL_ASSIST_SETTLE_STEPS
  cfg.roll_assist_zero_initialize_actor_output = True
  cfg.seed = 1
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      seed=1,       curriculum=True, size=ROLL_ASSIST_TERRAIN_SIZE_M,
      num_rows=ROLL_ASSIST_TERRAIN_ROWS, num_cols=2, difficulty_range=(0.0, 1.0),
      sub_terrains={
        "flat_retention": flat(proportion=0.25),
        "stair_roll_assist": pyramid_stairs(
          proportion=0.75, step_height_range=(hpass, hnext),
          step_width=ROLL_ASSIST_STEP_WIDTH_M,
          platform_width=ROLL_ASSIST_PLATFORM_WIDTH_M,
          border_width=ROLL_ASSIST_TERRAIN_BORDER_WIDTH_M,
        )
      },
    ),
    max_init_terrain_level=0, num_envs=cfg.scene.num_envs,
  )
  cfg.scene.sensors = tuple(cfg.scene.sensors) + (
    roll_assist_wheel_sensor_cfg(
      name=ROLL_ASSIST_LEFT_SENSOR_NAME, wheel_geom="wheel_left_collision"
    ),
    roll_assist_wheel_sensor_cfg(
      name=ROLL_ASSIST_RIGHT_SENSOR_NAME, wheel_geom="wheel_right_collision"
    ),
  )
  cfg.episode_length_s = ROLL_ASSIST_EPISODE_LENGTH_S
  action = cfg.actions["hybrid_wheel_leg"]
  action.action_mask = ROLL_ASSIST_ACTION_MASK
  action.action_scales = ROLL_ASSIST_ACTION_SCALES
  action.dynamic_stair_maneuver = None
  action.dynamic_stair_request_command_name = None
  action.dynamic_stair_left_sensor_name = None
  action.dynamic_stair_right_sensor_name = None
  action.stair_trigger_sensor_name = None
  action.stair_mode_freezes_leg_reference = False
  action.stair_mode_forced = False
  action.__post_init__()
  inherited_reset = cfg.events.pop("reset_root_state_with_small_disturbance", None)
  if inherited_reset is None:
    raise ValueError("Stage5 root reset is required by RollAssist.")
  cfg.events["reset_root_to_roll_assist"] = EventTermCfg(
    func=reset_roll_assist_posture_consistent, mode="reset",
    params={
      "root_height": ROLL_ASSIST_STAIR_POSTURE_HEIGHT_M,
      "pose_range": None,
      "velocity_range": None,
      "x_offset_from_origin_m": -(
        ROLL_ASSIST_RISER_OFFSET_M + ROLL_ASSIST_START_OFFSET_M
      ),
      "flat_env_count": flat_count,
      "stair_posture_height": ROLL_ASSIST_STAIR_POSTURE_HEIGHT_M,
      "stair_posture_pitch": ROLL_ASSIST_STAIR_POSTURE_PITCH_RAD,
    },
  )
  inherited_push = cfg.events.get("push_robot")
  if inherited_push is not None:
    cfg.events["push_robot"] = EventTermCfg(
      func=push_flat_retention_envs, mode=inherited_push.mode,
      interval_range_s=inherited_push.interval_range_s,
      params={**dict(inherited_push.params), "flat_env_count": flat_count},
    )
  twist_kwargs = _dataclass_init_kwargs(cfg.commands["twist"])
  twist_kwargs.update(flat_env_count=flat_count, stair_vx=ROLL_ASSIST_COMMAND_VX_MPS)
  posture_kwargs = _dataclass_init_kwargs(cfg.commands["posture"])
  posture_kwargs.update(
    flat_env_count=flat_count,
    stair_height=ROLL_ASSIST_STAIR_POSTURE_HEIGHT_M,
    stair_pitch=ROLL_ASSIST_STAIR_POSTURE_PITCH_RAD,
  )
  cfg.commands = {
    "twist": RollAssistVelocityCommandCfg(**twist_kwargs),
    "posture": RollAssistPostureCommandCfg(**posture_kwargs),
  }
  actor_terms = roll_assist_actor_terms(cfg.observations["actor"].terms)
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True, enable_corruption=not play
    ),
    "critic": ObservationGroupCfg(
      terms=roll_assist_critic_terms(actor_terms), concatenate_terms=True,
      enable_corruption=False,
    ),
  }
  cfg.curriculum = {
    "roll_assist_height": CurriculumTermCfg(
      func=roll_assist_curriculum,
      params={"flat_env_count": flat_count, "hpass_m": hpass, "hnext_m": hnext},
    )
  }
  cfg.rewards["roll_assist_progress"] = RewardTermCfg(
    func=roll_assist_progress_reward, weight=float(reward["progress_weight"])
  )
  cfg.rewards["roll_assist_success"] = RewardTermCfg(
    func=roll_assist_stable_success, weight=float(reward["success_weight"])
  )
  # Terminations run before reward/metrics and before auto-reset. This stateful
  # term therefore latches bilateral flight and stable success on the live
  # terminal sample; reward and curriculum read the same evidence afterward.
  cfg.terminations["bilateral_airborne"] = TerminationTermCfg(
    func=RollAssistEpisodeEvidence
  )
  # `RollAssistEpisodeEvidence` is constructed while the termination manager is
  # loaded, before this per-substep metric is resolved by the metrics manager.
  cfg.metrics = {
    **dict(getattr(cfg, "metrics", {})),
    "roll_assist_substep_support": MetricsTermCfg(
      func=roll_assist_substep_support_metric, per_substep=True,
    ),
    "roll_assist_episode": MetricsTermCfg(
      func=roll_assist_episode_metric, reduce="last"
    ),
  }
  validate_roll_assist_observation_contract(cfg)
  return cfg

def make_stair_dynamic_env_cfg(
  play: bool = False,
  *,
  initial_upper_height_m: float = DYNAMIC_STAIR_HEIGHT_STEP_M,
  steps_per_iteration: int = 24,
  maneuver_path: Path | None = None,
  dynamic_maneuver: DynamicStairManeuver | None = None,
  **artifact_paths: Path | None,
) -> ManagerBasedRlEnvCfg:
  """Build Hybrid-v3 for regular, frontal, equal-height 0.30 m stairs."""

  if steps_per_iteration < 1:
    raise ValueError("StairDynamic steps_per_iteration must be positive.")
  initial_level = round(initial_upper_height_m / DYNAMIC_STAIR_HEIGHT_STEP_M)
  if (
    initial_level < 1
    or initial_level >= DYNAMIC_STAIR_TERRAIN_ROWS
    or not math.isclose(
      initial_upper_height_m,
      initial_level * DYNAMIC_STAIR_HEIGHT_STEP_M,
      rel_tol=0.0,
      abs_tol=1.0e-12,
    )
  ):
    raise ValueError("StairDynamic initial height must be 0.01, 0.02, or 0.03 m.")
  resolved_maneuver_path = _artifact_path(
    maneuver_path,
    DYNAMIC_STAIR_MANEUVER_PATH_ENV,
  )
  if dynamic_maneuver is not None and resolved_maneuver_path is not None:
    raise ValueError("Provide StairDynamic maneuver by object or path, not both.")
  maneuver = dynamic_maneuver
  if maneuver is None:
    maneuver = (
      load_dynamic_maneuver(resolved_maneuver_path)
      if resolved_maneuver_path is not None
      else _default_unqualified_dynamic_maneuver()
    )
  qualified_maneuver = bool(maneuver.maneuver_hash and maneuver.bindings)

  cfg = make_hoppertrex_hybrid_env_cfg(
    stage=5,
    play=play,
    leg_residual_scale=DYNAMIC_STAIR_ACTION_SCALES[2],
    **artifact_paths,
  )
  cfg.scene.num_envs = 1 if play else DYNAMIC_STAIR_NUM_ENVS
  flat_env_count = DYNAMIC_STAIR_FLAT_ENVS if not play else 0
  cfg.stair_dynamic_task_id = DYNAMIC_STAIR_TASK_ID
  cfg.stair_dynamic_training_contract = not play
  cfg.stair_dynamic_contract_schema_version = 1
  cfg.stair_dynamic_contract_sha256 = None
  cfg.stair_dynamic_maneuver_qualified = qualified_maneuver
  cfg.stair_dynamic_maneuver_bindings = dict(maneuver.bindings or {})

  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      curriculum=True,
      size=DYNAMIC_STAIR_TERRAIN_SIZE_M,
      num_rows=DYNAMIC_STAIR_TERRAIN_ROWS,
      num_cols=1,
      difficulty_range=(0.0, 1.0),
      sub_terrains={
        "stair": pyramid_stairs(
          proportion=1.0,
          step_height_range=(0.0, DYNAMIC_STAIR_MAX_HEIGHT_M),
          step_width=DYNAMIC_STAIR_STEP_WIDTH_M,
          platform_width=DYNAMIC_STAIR_PLATFORM_WIDTH_M,
          border_width=DYNAMIC_STAIR_TERRAIN_BORDER_WIDTH_M,
        )
      },
    ),
    max_init_terrain_level=initial_level,
    num_envs=cfg.scene.num_envs,
  )
  cfg.scene.sensors = tuple(cfg.scene.sensors) + (
    dynamic_stair_wheel_sensor_cfg(
      name=DYNAMIC_STAIR_LEFT_SENSOR_NAME,
      wheel_geom="wheel_left_collision",
    ),
    dynamic_stair_wheel_sensor_cfg(
      name=DYNAMIC_STAIR_RIGHT_SENSOR_NAME,
      wheel_geom="wheel_right_collision",
    ),
  )
  cfg.episode_length_s = DYNAMIC_STAIR_EPISODE_LENGTH_S

  action = cfg.actions["hybrid_wheel_leg"]
  action.action_mask = DYNAMIC_STAIR_ACTION_MASK
  action.action_scales = DYNAMIC_STAIR_ACTION_SCALES
  action.dynamic_stair_maneuver = maneuver
  action.dynamic_stair_request_command_name = "stair_request"
  action.dynamic_stair_left_sensor_name = DYNAMIC_STAIR_LEFT_SENSOR_NAME
  action.dynamic_stair_right_sensor_name = DYNAMIC_STAIR_RIGHT_SENSOR_NAME
  action.dynamic_stair_control_dt = float(cfg.sim.mujoco.timestep * cfg.decimation)
  action.stair_trigger_sensor_name = None
  action.stair_mode_freezes_leg_reference = False
  action.stair_mode_forced = False
  action.__post_init__()
  if qualified_maneuver:
    _validate_dynamic_maneuver_classical_bindings(maneuver, action)

  inherited_reset = cfg.events.pop(
    "reset_root_state_with_small_disturbance",
    None,
  )
  if inherited_reset is None:
    raise ValueError("Stage5 root reset is required by StairDynamic.")
  cfg.events["reset_root_to_stair_dynamic"] = EventTermCfg(
    func=reset_root_to_stair_approach,
    mode="reset",
    params={
      "root_height": ROOT_HEIGHT_TARGET,
      "pose_range": inherited_reset.params["pose_range"],
      "velocity_range": inherited_reset.params["velocity_range"],
      "flat_env_count": flat_env_count,
    },
  )
  inherited_push = cfg.events.get("push_robot")
  if inherited_push is not None:
    cfg.events["push_robot"] = EventTermCfg(
      func=push_flat_retention_envs,
      mode=inherited_push.mode,
      interval_range_s=inherited_push.interval_range_s,
      params={
        **dict(inherited_push.params),
        "flat_env_count": flat_env_count,
      },
    )

  twist_cfg = cfg.commands["twist"]
  twist_kwargs = _dataclass_init_kwargs(twist_cfg)
  twist_kwargs.update(
    flat_env_count=flat_env_count,
    stair_vx=DYNAMIC_STAIR_COMMAND_VX_MPS,
  )
  posture_cfg = cfg.commands["posture"]
  posture_kwargs = _dataclass_init_kwargs(posture_cfg)
  posture_kwargs.update(
    flat_env_count=flat_env_count,
    stair_height=ROOT_HEIGHT_TARGET,
    stair_pitch=0.0,
  )
  cfg.commands = {
    "twist": StairDynamicVelocityCommandCfg(**twist_kwargs),
    "posture": StairDynamicPostureCommandCfg(**posture_kwargs),
    "stair_request": StairRequestCommandCfg(
      resampling_time_range=(DYNAMIC_STAIR_EPISODE_LENGTH_S,) * 2,
      debug_vis=False,
      flat_env_count=flat_env_count,
    ),
  }

  actor_terms = stair_dynamic_actor_terms(cfg.observations["actor"].terms)
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms=stair_dynamic_critic_terms(actor_terms),
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }
  cfg.rewards["stair_dynamic_progress"] = RewardTermCfg(
    func=stair_dynamic_progress_reward,
    weight=DYNAMIC_STAIR_PROGRESS_WEIGHT,
  )
  cfg.rewards["stair_dynamic_edge_bonus"] = RewardTermCfg(
    func=stair_dynamic_riser_event_reward,
    weight=DYNAMIC_STAIR_RISER_EVENT_BONUS,
  )

  cadence_params = {
    "evaluation_interval_steps": (
      DYNAMIC_STAIR_EVALUATION_INTERVAL_ITERS * steps_per_iteration
    ),
    "flat_env_count": flat_env_count,
    "initial_upper_height_m": initial_upper_height_m,
  }
  cfg.curriculum = {
    "stair_dynamic_height": CurriculumTermCfg(
      func=stair_dynamic_curriculum,
      params=dict(cadence_params),
    )
  }
  if not play:
    cfg.metrics = {
      **dict(getattr(cfg, "metrics", {})),
      "stair_dynamic_step": MetricsTermCfg(
        func=stair_dynamic_step_metric,
        params=dict(cadence_params),
        reduce="last",
      ),
    }
  validate_stair_dynamic_observation_contract(cfg)
  return cfg


def make_stair_camp_lqr_alpha05_env_cfg(
  play: bool = False,
  **artifact_paths: Path | None,
) -> ManagerBasedRlEnvCfg:
  """Build the one preregistered non-primary LQR alpha=0.5 failure rung.

  This is a fresh seed-1 diagnostic retrain at the campaign's final main-run
  budget. It is a separate registered task so its checkpoint can never be
  mistaken for primary promotion evidence or a 1000->3000 continuation.
  """

  cfg = make_stair_camp_env_cfg(play=play, **artifact_paths)
  cfg.stair_camp_task_id = STAIR_CAMP_LQR_ALPHA05_TASK_ID
  cfg.stair_camp_failure_ladder_variant = STAIR_CAMP_FAILURE_LADDER_VARIANT
  cfg.actions["hybrid_wheel_leg"].stair_mode_lqr_gain_scale = (
    STAIR_CAMP_FAILURE_LQR_GAIN_SCALE
  )
  validate_stair_camp_observation_contract(cfg)
  return cfg


def hybrid_provenance_lines(env_cfg: object) -> list[str]:
  """Human-visible controller/calibration provenance for play sessions.

  make_hoppertrex_hybrid_env_cfg silently falls back to the unqualified local
  PD gain and an uncalibrated velocity command when the artifact environment
  variables are missing. train.py rejects that for training; play/Viser
  sessions print these lines instead so a fallback session cannot be mistaken
  for the qualified Stage0 controller.
  """

  actions = getattr(env_cfg, "actions", None)
  action = actions.get("hybrid_wheel_leg") if isinstance(actions, dict) else None
  if not isinstance(action, HybridWheelLegActionCfg):
    return []
  lines = [
    (
      f"[hybrid] controller_type={action.controller_type} "
      f"qualified={action.controller_qualified} "
      f"gain_hash={action.controller_gain_hash or 'none'}"
    ),
    f"[hybrid] controller_source={action.controller_source}",
    (
      f"[hybrid] calibration_hash={action.calibration_hash or 'uncalibrated'} "
      f"scale={action.velocity_command_scale} "
      f"bias={action.velocity_command_bias}"
    ),
    (
      f"[hybrid] posture_map_qualified={action.posture_map_qualified} "
      f"artifact_hash={action.posture_artifact_hash or 'legacy'} "
      f"source={action.posture_map_source}"
    ),
    (
      f"[hybrid] yaw_calibration_qualified={action.yaw_calibration_qualified} "
      f"hash={action.yaw_calibration_hash or 'none'} "
      f"source={action.yaw_calibration_source}"
    ),
    (
      f"[hybrid] station_calibration_qualified="
      f"{action.station_calibration_qualified} "
      f"hash={action.station_calibration_hash or 'none'} "
      f"source={action.station_calibration_source}"
    ),
  ]
  if not action.controller_qualified:
    lines.append(
      "[hybrid] WARNING: unqualified controller fallback is active. Viewer "
      "verdicts from this session do not describe the qualified Stage0 LQR. "
      "Set HOPPERTREX_HYBRID_CONTROLLER_PATH and restart."
    )
  if not action.calibration_hash:
    lines.append(
      "[hybrid] WARNING: velocity command is uncalibrated (scale=1, bias=0). "
      "Set HOPPERTREX_HYBRID_CALIBRATION_PATH and restart."
    )
  if action.action_mask[1] and not action.yaw_calibration_qualified:
    lines.append(
      "[hybrid] WARNING: the yaw residual head is active but the yaw "
      "feedforward is the zero fallback, so nominal yaw tracking is unowned. "
      "Set HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH and restart."
    )
  return lines


__all__ = [
  "DYNAMIC_STAIR_MANEUVER_PATH_ENV",
  "DYNAMIC_STAIR_SENSOR_NAMES",
  "HOPPERTREX_HYBRID_TASK_IDS",
  "HYBRID_TASK_IDS",
  "ROLL_ASSIST_FLAT_ENVS",
  "ROLL_ASSIST_SETTLE_STEPS",
  "WHEEL_JOINT_NAMES",
  "HybridPlanarVelocityCommand",
  "HybridPlanarVelocityCommandCfg",
  "HybridWheelLegAction",
  "HybridWheelLegActionCfg",
  "PostureCommand",
  "PostureCommandCfg",
  "RollAssistEpisodeEvidence",
  "Stage1VelocityCommand",
  "Stage1VelocityCommandCfg",
  "StairDynamicCurriculum",
  "StairDynamicPostureCommand",
  "StairDynamicPostureCommandCfg",
  "StairDynamicVelocityCommand",
  "StairDynamicVelocityCommandCfg",
  "StairRequestCommand",
  "StairRequestCommandCfg",
  "hybrid_provenance_lines",
  "make_hoppertrex_hybrid_env_cfg",
  "make_stair_camp_env_cfg",
  "make_stair_camp_lqr_alpha05_env_cfg",
  "make_stair_dynamic_env_cfg",
  "make_stair_roll_assist_env_cfg",
  "stage1_mismatch_event_cfg",
  "stair_dynamic_actor_terms",
  "stair_dynamic_critic_terms",
  "validate_stair_dynamic_observation_contract",
]
