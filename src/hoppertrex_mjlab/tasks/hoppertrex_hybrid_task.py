"""Hybrid v2 controller-residual tasks for the two-leg HopperTrex robot."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Mapping
from pathlib import Path

import numpy as np
import torch
from assets.HopperTrex_CFG import INIT_JOINT_POS
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.config import pyramid_stairs
from mjlab.managers import (
  ActionTerm,
  ActionTermCfg,
  CommandTerm,
  CommandTermCfg,
  CurriculumTermCfg,
  MetricsTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  SceneEntityCfg,
)
from mjlab.tasks.velocity.mdp import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

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
from hoppertrex_mjlab.hybrid.stair_residual import (
  StairCurriculumState,
  update_stair_curriculum,
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
from hoppertrex_mjlab.hybrid.station_calibration import (
  parse_station_calibration_artifact,
  validate_station_breakpoints,
)
from hoppertrex_mjlab.hybrid.stair_classical import PHASE_COUNT, StairPhase
from hoppertrex_mjlab.hybrid.stair_trigger import (
  STAIR_TRIGGER_FORCE_N,
  STAIR_TRIGGER_SENSOR_FIELDS,
  STAIR_TRIGGER_SENSOR_NAME,
  STAIR_TRIGGER_SLOTS_PER_WHEEL,
  STAIR_TRIGGER_WINDOW,
  stair_trigger_metric,
  update_stair_trigger,
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

WHEEL_JOINT_NAMES = ("wheel_left", "wheel_right")
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
      self.cfg.velocity_command_scale * velocity_command[:, 0]
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
      velocity_command[:, 2],
      self._yaw_feedforward_wz,
      self._yaw_feedforward_diff,
    )
    self._controller_baseline[:, 0] = -control + yaw_feedforward
    self._controller_baseline[:, 1] = control + yaw_feedforward

    balance_residual = self._applied_residual[:, 0]
    yaw_residual = self._applied_residual[:, 1]
    desired_wheels = self._controller_baseline.clone()
    desired_wheels[:, 0] += -balance_residual + yaw_residual
    desired_wheels[:, 1] += balance_residual + yaw_residual
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
    desired_legs = self._leg_reference + self._applied_residual[:, 2:]
    soft_limits = self._entity.data.soft_joint_pos_limits
    self._leg_targets[:] = torch.clamp(
      desired_legs,
      min=soft_limits[:, self._leg_ids, 0],
      max=soft_limits[:, self._leg_ids, 1],
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
# Reward weights, frozen with the implementation commit (S5B freeze clause).
STAIR_CAMP_PROGRESS_WEIGHT = 2.0
STAIR_CAMP_CLIMB_SUCCESS_WEIGHT = 5.0
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

  positions = root_states[:, 0:3] + pose_samples[:, 0:3]
  positions[:, 0] = (
    origins[:, 0]
    - (STAIR_CAMP_RISER_OFFSET_M + STAIR_CAMP_START_OFFSET_M)
    + pose_samples[:, 0]
  )
  positions[:, 1] = origins[:, 1] + pose_samples[:, 1]
  positions[:, 2] = float(root_height) + pose_samples[:, 2]
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
  "HOPPERTREX_HYBRID_TASK_IDS",
  "HYBRID_TASK_IDS",
  "HybridPlanarVelocityCommand",
  "HybridPlanarVelocityCommandCfg",
  "HybridWheelLegAction",
  "HybridWheelLegActionCfg",
  "PostureCommand",
  "PostureCommandCfg",
  "Stage1VelocityCommand",
  "Stage1VelocityCommandCfg",
  "WHEEL_JOINT_NAMES",
  "hybrid_provenance_lines",
  "make_hoppertrex_hybrid_env_cfg",
  "make_stair_camp_env_cfg",
  "make_stair_camp_lqr_alpha05_env_cfg",
  "stage1_mismatch_event_cfg",
]
