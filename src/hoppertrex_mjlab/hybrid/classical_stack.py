"""Portable classical stack: the full baseline pipeline in pure numpy.

This is the R2 deployment payload of the sim-to-real ladder. It replays the
exact classical math of ``HybridWheelLegAction.process_actions`` — LQR state
construction, velocity calibration, station-keeping compensation, yaw
feedforward, residual mixing with slew/velocity limits, and posture-map IK —
with no torch or mjlab imports, so it runs on any onboard computer.

Design contracts:

- ``classical_step`` consumes explicit sensor fields. ``vx`` is an input,
  not magic: simulation reads the privileged body velocity, hardware must
  supply an odometry estimate (wheel-speed based; an R2 decision point in
  the runbook).
- Artifact loading mirrors the task-module validation (schema, hash
  re-computation, cross-artifact binding) so a JSON that the training env
  would reject is rejected here too.
- The residual path reuses ``compose_hybrid_targets`` — the same audited
  contract the simulation runtime is pinned against bit-for-bit.
- All arithmetic is float32 to match the torch runtime; the equivalence
  test in tests/test_hybrid_classical_stack.py pins the two pipelines
  against each other inside a real CPU environment.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .calibration import parse_calibration_artifact
from .control import compose_hybrid_targets
from .controller_schedule import (
  SCHEDULE_ARTIFACT_TYPE,
  ControllerSchedule,
  parse_controller_schedule,
)
from .identification import (
  CONTROLLER_STATE_NAMES,
  NOMINAL_WHEEL_RADIUS_M,
  STATE_DEFINITION_VERSION,
)
from .posture import (
  LEG_JOINT_NAMES,
  POSTURE_ENVELOPE_VERIFICATION_METHODS,
  POSTURE_FEATURE_NAMES,
  posture_artifact_hash,
)
from .stair_classical import (
  StairControllerState,
  StairManeuver,
  StairPhase,
  StairSensors,
  contact_detector_wheel_reference_radps,
  load_stair_maneuver,
  stair_controller_step,
)
from .station_calibration import parse_station_calibration_artifact
from .yaw_calibration import parse_yaw_calibration_artifact

# Mirrors of the task-module runtime constants. The equivalence test pins
# them against the task values so a drift there fails CI here.
DEFAULT_WHEEL_VELOCITY_LIMIT = 12.0
DEFAULT_WHEEL_SLEW_LIMIT = 6.0
POSTURE_HEIGHT_SLEW_RATE = 0.01215
POSTURE_PITCH_SLEW_RATE = 0.07755
CONTROL_DT_S = 0.02
ZERO_YAW_BREAKPOINTS = ((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0))
ZERO_STATION_BREAKPOINTS = ((-1.0, 0.0), (1.0, 0.0))


def _stable_hash(payload: dict[str, object]) -> str:
  encoded = json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
  ).encode("ascii")
  return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: Path, artifact_name: str) -> dict[str, object]:
  if not path.is_file():
    raise FileNotFoundError(f"{artifact_name} artifact does not exist: {path}")
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise ValueError(f"{artifact_name} artifact must contain a JSON object.")
  return payload


@dataclass(frozen=True)
class ClassicalStackArtifacts:
  """Validated classical-layer parameters plus their provenance hashes."""

  controller_gain: tuple[float, float, float, float]
  controller_type: str
  controller_qualified: bool
  controller_gain_hash: str | None
  velocity_command_scale: float
  velocity_command_bias: float
  calibration_hash: str | None
  yaw_feedforward_breakpoints: tuple[tuple[float, float], ...]
  yaw_calibration_hash: str | None
  station_drift_breakpoints: tuple[tuple[float, float], ...]
  station_calibration_hash: str | None
  posture_coefficients: tuple[tuple[float, float, float, float], ...]
  posture_map_hash: str | None
  posture_height_range: tuple[float, float]
  posture_pitch_range: tuple[float, float]
  posture_artifact_hash: str | None = None
  wheel_radius: float = NOMINAL_WHEEL_RADIUS_M
  wheel_velocity_limit: float = DEFAULT_WHEEL_VELOCITY_LIMIT
  wheel_slew_limit: float = DEFAULT_WHEEL_SLEW_LIMIT
  controller_schedule: ControllerSchedule | None = None
  stair_maneuver: StairManeuver | None = None


def _load_controller_payload(path: Path) -> dict[str, object]:
  payload = _read_json_object(path, "Controller")
  if payload.get("artifact_type") == SCHEDULE_ARTIFACT_TYPE:
    schedule = parse_controller_schedule(payload, source=str(path.resolve()))
    payload["gain_hash"] = schedule.schedule_hash
    payload["controller_type"] = "lqr"
    payload["_gain"] = tuple(float(value) for value in schedule.gains[1, 1])
    payload["_qualified"] = True
    payload["_schedule"] = schedule
    return payload
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
  if controller_type not in ("lqr", "pd"):
    raise ValueError(
      "Controller artifact must label its type as 'lqr' or 'pd'."
    )
  expected_gain_hash = _stable_hash(
    {
      "controller_type": controller_type,
      "state_names": CONTROLLER_STATE_NAMES,
      "gain": [gain_array.tolist()],
    }
  )
  if payload.get("gain_hash") != expected_gain_hash:
    raise ValueError("Controller gain_hash does not match its controller data.")
  state_construction = payload.get("state_construction")
  if state_construction is not None:
    if not isinstance(state_construction, dict):
      raise ValueError("Controller state_construction must be a JSON object.")
    if (
      state_construction.get("state_definition_version")
      != STATE_DEFINITION_VERSION
    ):
      raise ValueError(
        "Controller artifact state definition does not match the runtime."
      )
    radius = state_construction.get("wheel_radius")
    if (
      not isinstance(radius, (int, float))
      or isinstance(radius, bool)
      or not math.isfinite(float(radius))
      or abs(float(radius) - NOMINAL_WHEEL_RADIUS_M) > 1.0e-9
    ):
      raise ValueError(
        "Controller artifact wheel_radius does not match the runtime."
      )
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
  if controller_type == "lqr" and not qualified:
    raise ValueError(
      "LQR controller artifact does not meet controllability and NRMSE "
      "qualification."
    )
  payload["_gain"] = tuple(float(value) for value in gain_array)
  payload["_qualified"] = qualified
  return payload


def _load_posture_payload(path: Path) -> dict[str, object]:
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
  expected_map_hash = _stable_hash(
    {
      "feature_names": POSTURE_FEATURE_NAMES,
      "joint_names": LEG_JOINT_NAMES,
      "coefficients": coefficients.tolist(),
    }
  )
  if payload.get("map_hash") != expected_map_hash:
    raise ValueError("Posture map_hash does not match its posture data.")
  artifact_hash = payload.get("posture_artifact_hash")
  if artifact_hash is not None and artifact_hash != posture_artifact_hash(
    payload
  ):
    raise ValueError(
      "Posture artifact hash does not match its envelope and fit data."
    )
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
    raise ValueError(
      "Posture pitch range must be ordered and stay within 0.08 rad."
    )
  verification = payload.get("envelope_verification")
  grid_shape = (
    verification.get("grid_shape") if isinstance(verification, dict) else None
  )
  if (
    not isinstance(verification, dict)
    or verification.get("method") not in POSTURE_ENVELOPE_VERIFICATION_METHODS
    or not isinstance(grid_shape, list)
    or len(grid_shape) != 2
    or any(not isinstance(value, int) or value < 2 for value in grid_shape)
  ):
    raise ValueError("Posture map must document a verified grid rectangle.")
  source_sweep = payload.get("source_sweep")
  controller_gain_hash = (
    source_sweep.get("controller_gain_hash")
    if isinstance(source_sweep, dict)
    else None
  )
  if not isinstance(controller_gain_hash, str) or not controller_gain_hash:
    raise ValueError("Posture map must record its source controller gain hash.")
  calibration_hash = (
    source_sweep.get("calibration_hash")
    if isinstance(source_sweep, dict)
    else None
  )
  if not isinstance(calibration_hash, str) or not calibration_hash:
    raise ValueError("Posture map must record its source calibration hash.")
  payload["_coefficients"] = tuple(
    tuple(float(value) for value in row) for row in coefficients
  )
  payload["_height_range"] = (height_range[0], height_range[1])
  payload["_pitch_range"] = (pitch_range[0], pitch_range[1])
  payload["_controller_gain_hash"] = controller_gain_hash
  payload["_calibration_hash"] = calibration_hash
  payload["_posture_artifact_hash"] = artifact_hash
  return payload


def load_classical_stack_artifacts(
  *,
  controller_path: Path,
  calibration_path: Path,
  posture_map_path: Path,
  yaw_calibration_path: Path | None = None,
  station_calibration_path: Path | None = None,
  stair_maneuver_path: Path | None = None,
) -> ClassicalStackArtifacts:
  """Load and cross-bind the five classical artifacts for deployment.

  Controller, velocity calibration, and posture map are mandatory: a
  deployment without them has no qualified balance/velocity/legs story.
  Yaw and station calibrations fall back to identically-zero maps with a
  ``None`` hash, matching the training-env fallback semantics.
  """

  controller = _load_controller_payload(Path(controller_path))
  gain_hash = str(controller["gain_hash"])
  calibration = parse_calibration_artifact(
    _read_json_object(Path(calibration_path), "Calibration"),
    controller_gain_hash=gain_hash,
  )
  posture = _load_posture_payload(Path(posture_map_path))
  schedule = controller.get("_schedule")
  schedule_bindings = schedule.bindings if isinstance(schedule, ControllerSchedule) else {}
  expected_posture_controller_hash = (
    schedule_bindings.get("identification_controller_gain_hash")
    if schedule is not None
    else gain_hash
  )
  expected_posture_calibration_hash = (
    schedule_bindings.get("identification_calibration_hash")
    if schedule is not None
    else calibration.calibration_hash
  )
  if posture["_controller_gain_hash"] != expected_posture_controller_hash:
    raise ValueError(
      "Posture map was produced for a different controller gain."
    )
  if posture["_calibration_hash"] != expected_posture_calibration_hash:
    raise ValueError(
      "Posture map was produced for a different identification calibration."
    )
  if (
    schedule is not None
    and posture["_posture_artifact_hash"]
    != schedule_bindings.get("posture_artifact_hash")
  ):
    raise ValueError(
      "Controller schedule was identified with a different posture artifact."
    )
  yaw_breakpoints = ZERO_YAW_BREAKPOINTS
  yaw_hash: str | None = None
  if yaw_calibration_path is not None:
    parsed_yaw = parse_yaw_calibration_artifact(
      _read_json_object(Path(yaw_calibration_path), "Yaw calibration"),
      controller_gain_hash=gain_hash,
    )
    yaw_breakpoints = parsed_yaw.breakpoints
    yaw_hash = parsed_yaw.yaw_calibration_hash
  station_breakpoints = ZERO_STATION_BREAKPOINTS
  station_hash: str | None = None
  if station_calibration_path is not None:
    parsed_station = parse_station_calibration_artifact(
      _read_json_object(
        Path(station_calibration_path), "Station calibration"
      ),
      controller_gain_hash=gain_hash,
      posture_map_hash=str(posture["map_hash"]),
      posture_artifact_hash=(
        str(posture["_posture_artifact_hash"])
        if posture["_posture_artifact_hash"] is not None
        else None
      ),
    )
    station_breakpoints = parsed_station.breakpoints
    station_hash = parsed_station.station_calibration_hash
  stair_maneuver = (
    None
    if stair_maneuver_path is None
    else load_stair_maneuver(Path(stair_maneuver_path))
  )
  if stair_maneuver is not None:
    expected = stair_maneuver.bindings.get("controller_schedule_hash")
    if expected != gain_hash:
      raise ValueError("Stair maneuver is bound to a different controller schedule.")
  return ClassicalStackArtifacts(
    controller_gain=controller["_gain"],  # type: ignore[arg-type]
    controller_type=str(controller["controller_type"]),
    controller_qualified=bool(controller["_qualified"]),
    controller_gain_hash=gain_hash,
    velocity_command_scale=calibration.scale,
    velocity_command_bias=calibration.bias,
    calibration_hash=calibration.calibration_hash,
    yaw_feedforward_breakpoints=yaw_breakpoints,
    yaw_calibration_hash=yaw_hash,
    station_drift_breakpoints=station_breakpoints,
    station_calibration_hash=station_hash,
    posture_coefficients=posture["_coefficients"],  # type: ignore[arg-type]
    posture_map_hash=str(posture["map_hash"]),
    posture_artifact_hash=(
      str(posture["_posture_artifact_hash"])
      if posture["_posture_artifact_hash"] is not None
      else None
    ),
    posture_height_range=posture["_height_range"],  # type: ignore[arg-type]
    posture_pitch_range=posture["_pitch_range"],  # type: ignore[arg-type]
    controller_schedule=controller.get("_schedule"),  # type: ignore[arg-type]
    stair_maneuver=stair_maneuver,
  )


@dataclass(frozen=True)
class ClassicalStackConfig:
  """Runtime parameters of the classical pipeline (one robot).

  ``from_artifacts`` is the deployment path; tests may also build it
  directly from a training env's action-term config to pin equivalence.
  """

  controller_gain: tuple[float, float, float, float]
  velocity_command_scale: float
  velocity_command_bias: float
  yaw_feedforward_breakpoints: tuple[tuple[float, float], ...]
  station_drift_breakpoints: tuple[tuple[float, float], ...]
  posture_coefficients: tuple[tuple[float, float, float, float], ...]
  action_mask: tuple[bool, ...]
  action_scales: tuple[float, ...]
  leg_position_lower: tuple[float, float, float, float]
  leg_position_upper: tuple[float, float, float, float]
  wheel_radius: float = NOMINAL_WHEEL_RADIUS_M
  wheel_velocity_limit: float = DEFAULT_WHEEL_VELOCITY_LIMIT
  wheel_slew_limit: float = DEFAULT_WHEEL_SLEW_LIMIT
  controller_schedule: ControllerSchedule | None = None
  stair_maneuver: StairManeuver | None = None

  @classmethod
  def from_artifacts(
    cls,
    artifacts: ClassicalStackArtifacts,
    *,
    action_mask: tuple[bool, ...],
    action_scales: tuple[float, ...],
    leg_position_lower: tuple[float, float, float, float],
    leg_position_upper: tuple[float, float, float, float],
  ) -> "ClassicalStackConfig":
    return cls(
      controller_gain=artifacts.controller_gain,
      velocity_command_scale=artifacts.velocity_command_scale,
      velocity_command_bias=artifacts.velocity_command_bias,
      yaw_feedforward_breakpoints=artifacts.yaw_feedforward_breakpoints,
      station_drift_breakpoints=artifacts.station_drift_breakpoints,
      posture_coefficients=artifacts.posture_coefficients,
      action_mask=action_mask,
      action_scales=action_scales,
      leg_position_lower=leg_position_lower,
      leg_position_upper=leg_position_upper,
      wheel_radius=artifacts.wheel_radius,
      wheel_velocity_limit=artifacts.wheel_velocity_limit,
      wheel_slew_limit=artifacts.wheel_slew_limit,
      controller_schedule=artifacts.controller_schedule,
      stair_maneuver=artifacts.stair_maneuver,
    )


@dataclass(frozen=True)
class ClassicalStackState:
  """Per-tick pipeline state; treat as immutable and thread the returns."""

  previous_wheel_targets: tuple[float, float] = (0.0, 0.0)
  posture_command: tuple[float, float] = (0.0, 0.0)
  posture_target: tuple[float, float] = (0.0, 0.0)
  stair_state: StairControllerState = field(default_factory=StairControllerState)


def reset_state(height: float, pitch: float) -> ClassicalStackState:
  """Fresh state: shaped command snaps to the target, wheels neutral."""

  return ClassicalStackState(
    previous_wheel_targets=(0.0, 0.0),
    posture_command=(float(height), float(pitch)),
    posture_target=(float(height), float(pitch)),
  )


def set_posture_target(
  state: ClassicalStackState,
  height: float,
  pitch: float,
) -> ClassicalStackState:
  return replace(state, posture_target=(float(height), float(pitch)))


def shape_posture_command(
  state: ClassicalStackState,
  *,
  dt: float = CONTROL_DT_S,
  height_slew_rate: float = POSTURE_HEIGHT_SLEW_RATE,
  pitch_slew_rate: float = POSTURE_PITCH_SLEW_RATE,
) -> ClassicalStackState:
  """One reference-shaping tick: slew the command toward the raw target.

  Mirrors PostureCommand._update_command (the T=2.0 tier pinned at
  57aff94): per-axis delta clamped to rate*dt.
  """

  command = list(state.posture_command)
  for axis, rate in ((0, height_slew_rate), (1, pitch_slew_rate)):
    delta = state.posture_target[axis] - command[axis]
    step = rate * dt
    command[axis] += min(max(delta, -step), step)
  return replace(state, posture_command=(command[0], command[1]))


@dataclass(frozen=True)
class ClassicalSensors:
  """Explicit hardware inputs for one control tick (SI units, body frame).

  ``vx`` must come from an estimator on hardware (wheel odometry fusion);
  simulation supplies the privileged body velocity. Leg positions are not
  consumed: the classical layer commands legs purely feedforward.
  """

  pitch: float
  pitch_rate: float
  vx: float
  body_deceleration: float
  wheel_vel_left: float
  wheel_vel_right: float
  non_wheel_contact: bool = False
  actuator_limit: bool = False


@dataclass(frozen=True)
class ClassicalCommands:
  vx: float = 0.0
  wz: float = 0.0
  height: float = 0.0
  pitch: float = 0.0
  stair_mode: bool = False


def _interp_f32(x: float, breakpoints: tuple[tuple[float, float], ...]) -> float:
  xp = np.asarray([point[0] for point in breakpoints], dtype=np.float32)
  fp = np.asarray([point[1] for point in breakpoints], dtype=np.float32)
  return float(
    np.interp(np.float32(x), xp, fp).astype(np.float32)
  )


def classical_step(
  config: ClassicalStackConfig,
  state: ClassicalStackState,
  sensors: ClassicalSensors,
  commands: ClassicalCommands,
  residual_actions: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32], ClassicalStackState]:
  """One 50 Hz classical tick: sensors + commands -> wheel/leg targets.

  Replays HybridWheelLegAction.process_actions in float32 numpy:
  station-compensated velocity calibration -> LQR -> yaw feedforward ->
  residual mixing with slew and velocity limits -> posture-map IK with
  joint clamping. ``residual_actions`` is the raw six-dim policy output
  (zeros for pure classical R2 operation).
  """

  signed_wheel_speed = np.float32(0.5) * (
    np.float32(sensors.wheel_vel_right) - np.float32(sensors.wheel_vel_left)
  )
  effective_commands = commands
  drive_feedforward = 0.0
  stair_state = state.stair_state
  maneuver_active = (
    config.stair_maneuver is not None
    and (
      commands.stair_mode
      or state.stair_state.phase not in (StairPhase.IDLE, StairPhase.DONE)
    )
  )
  if maneuver_active and config.stair_maneuver is not None:
    detector_reference_vx = state.stair_state.detector_reference_vx
    detector_reference_pitch = state.stair_state.detector_reference_pitch
    if state.stair_state.phase == StairPhase.IDLE:
      detector_reference_vx = config.stair_maneuver.approach_vx
      detector_reference_pitch = commands.pitch
    detector_station_drift = _interp_f32(
      detector_reference_pitch, config.station_drift_breakpoints
    )
    detector_wheel_reference = contact_detector_wheel_reference_radps(
      command_vx=detector_reference_vx,
      velocity_command_scale=config.velocity_command_scale,
      velocity_command_bias=config.velocity_command_bias,
      station_drift_mps=detector_station_drift,
      wheel_radius=config.wheel_radius,
    )
    preliminary_error = float(
      signed_wheel_speed - detector_wheel_reference
    )
    stair_targets, stair_state = stair_controller_step(
      config.stair_maneuver,
      state.stair_state,
      StairSensors(
        pitch=sensors.pitch,
        pitch_rate=sensors.pitch_rate,
        body_deceleration=sensors.body_deceleration,
        signed_wheel_speed=float(signed_wheel_speed),
        wheel_speed_error=preliminary_error,
        non_wheel_contact=sensors.non_wheel_contact,
        actuator_limit=sensors.actuator_limit,
      ),
      stair_mode=commands.stair_mode,
      nominal_height=commands.height,
      nominal_pitch=commands.pitch,
    )
    effective_commands = ClassicalCommands(
      vx=stair_targets.vx,
      wz=commands.wz,
      height=stair_targets.height,
      pitch=stair_targets.pitch,
      stair_mode=commands.stair_mode,
    )
    drive_feedforward = stair_targets.drive_feedforward_radps

  shaped_posture = state.posture_command
  if config.stair_maneuver is not None:
    height_delta = min(
      max(
        effective_commands.height - state.posture_command[0],
        -POSTURE_HEIGHT_SLEW_RATE * CONTROL_DT_S,
      ),
      POSTURE_HEIGHT_SLEW_RATE * CONTROL_DT_S,
    )
    pitch_delta = min(
      max(
        effective_commands.pitch - state.posture_command[1],
        -POSTURE_PITCH_SLEW_RATE * CONTROL_DT_S,
      ),
      POSTURE_PITCH_SLEW_RATE * CONTROL_DT_S,
    )
    shaped_posture = (
      state.posture_command[0] + height_delta,
      state.posture_command[1] + pitch_delta,
    )
    effective_commands = ClassicalCommands(
      vx=effective_commands.vx,
      wz=effective_commands.wz,
      height=shaped_posture[0],
      pitch=shaped_posture[1],
      stair_mode=effective_commands.stair_mode,
    )

  height_cmd = np.float32(effective_commands.height)
  pitch_cmd = np.float32(effective_commands.pitch)
  station_drift = np.float32(
    _interp_f32(float(pitch_cmd), config.station_drift_breakpoints)
  )
  calibrated_vx = (
    np.float32(config.velocity_command_scale) * np.float32(effective_commands.vx)
    + np.float32(config.velocity_command_bias)
    - station_drift
  )
  vx_error = np.float32(sensors.vx) - calibrated_vx
  desired_wheel_speed = calibrated_vx / np.float32(config.wheel_radius)
  wheel_speed_error = signed_wheel_speed - desired_wheel_speed
  gain = np.asarray(config.controller_gain, dtype=np.float32)
  equilibrium_input = np.float32(0.0)
  state_vector = np.asarray(
    [
      np.float32(sensors.pitch),
      np.float32(sensors.pitch_rate),
      vx_error,
      wheel_speed_error,
    ],
    dtype=np.float32,
  )
  if config.controller_schedule is not None:
    scheduled, equilibrium, equilibrium_input_value, _ = (
      config.controller_schedule.interpolate_affine(
      float(height_cmd), float(pitch_cmd)
      )
    )
    gain = scheduled.astype(np.float32)
    state_vector -= equilibrium.astype(np.float32)
    equilibrium_input = np.float32(equilibrium_input_value)
  control = (
    equilibrium_input
    - np.float32(state_vector @ gain)
    + np.float32(drive_feedforward)
  )
  yaw_feedforward = np.float32(
    _interp_f32(float(effective_commands.wz), config.yaw_feedforward_breakpoints)
  )
  baseline = np.asarray(
    [[-control + yaw_feedforward, control + yaw_feedforward]],
    dtype=np.float32,
  )

  features = np.asarray(
    [[np.float32(1.0), height_cmd, pitch_cmd]], dtype=np.float32
  )
  coefficients = np.asarray(config.posture_coefficients, dtype=np.float32)
  nominal_legs = features @ coefficients

  actions = (
    np.zeros((1, 6), dtype=np.float32)
    if residual_actions is None
    else np.asarray(residual_actions, dtype=np.float32).reshape(1, 6)
  )
  output = compose_hybrid_targets(
    policy_actions=actions,
    action_mask=config.action_mask,
    action_scales=config.action_scales,
    baseline_wheel_targets=baseline,
    nominal_leg_targets=nominal_legs,
    previous_wheel_targets=np.asarray(
      [list(state.previous_wheel_targets)], dtype=np.float32
    ),
    wheel_slew_limit=config.wheel_slew_limit,
    wheel_velocity_limit=config.wheel_velocity_limit,
    leg_position_lower=np.asarray(
      config.leg_position_lower, dtype=np.float32
    ),
    leg_position_upper=np.asarray(
      config.leg_position_upper, dtype=np.float32
    ),
  )
  wheel_targets = output.wheel_targets.astype(np.float32)[0]
  leg_targets = output.leg_targets.astype(np.float32)[0]
  new_state = replace(
    state,
    previous_wheel_targets=(
      float(wheel_targets[0]),
      float(wheel_targets[1]),
    ),
    stair_state=stair_state,
    posture_command=shaped_posture,
    posture_target=(float(commands.height), float(commands.pitch)),
  )
  return wheel_targets, leg_targets, new_state
