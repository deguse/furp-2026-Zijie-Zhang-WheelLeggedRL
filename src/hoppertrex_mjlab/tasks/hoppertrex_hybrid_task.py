"""Hybrid v2 controller-residual tasks for the two-leg HopperTrex robot."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import (
  ActionTerm,
  ActionTermCfg,
  CommandTerm,
  CommandTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  SceneEntityCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from assets.HopperTrex_CFG import INIT_JOINT_POS
from hoppertrex_mjlab.hybrid.config import (
  HYBRID_ACTION_NAMES,
  HYBRID_STAGES,
)
from hoppertrex_mjlab.hybrid.identification import CONTROLLER_STATE_NAMES
from hoppertrex_mjlab.hybrid.posture import (
  LEG_JOINT_NAMES,
  POSTURE_FEATURE_NAMES,
)
from hoppertrex_mjlab.tasks.hoppertrex_balance_task import (
  ROOT_HEIGHT_TARGET,
  joint_pos_rel_without_wheel_position,
  make_hoppertrex_balance_env_cfg,
)


WHEEL_JOINT_NAMES = ("wheel_left", "wheel_right")
HYBRID_TASK_IDS = tuple(
  f"HopperTrex-Hybrid-v2-Stage{stage}" for stage in range(6)
)
HOPPERTREX_HYBRID_TASK_IDS = HYBRID_TASK_IDS

DEFAULT_PD_GAIN = (8.0, 1.0, 3.0, 0.2)
DEFAULT_WHEEL_RADIUS = 0.100
DEFAULT_WHEEL_VELOCITY_LIMIT = 12.0
DEFAULT_WHEEL_SLEW_LIMIT = 6.0
CONTROLLER_PATH_ENV = "HOPPERTREX_HYBRID_CONTROLLER_PATH"
POSTURE_MAP_PATH_ENV = "HOPPERTREX_HYBRID_POSTURE_MAP_PATH"


@dataclass(frozen=True)
class _ControllerArtifact:
  gain: tuple[float, float, float, float]
  controller_type: str
  qualified: bool
  source: str
  gain_hash: str | None


@dataclass(frozen=True)
class _PostureArtifact:
  coefficients: tuple[tuple[float, float, float, float], ...]
  height_range: tuple[float, float]
  pitch_range: tuple[float, float]
  qualified: bool
  source: str
  map_hash: str | None


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
  return _PostureArtifact(
    coefficients=tuple(
      tuple(float(value) for value in row) for row in coefficients
    ),
    height_range=(height_range[0], height_range[1]),
    pitch_range=(pitch_range[0], pitch_range[1]),
    qualified=True,
    source=str(path),
    map_hash=str(map_hash),
  )


@dataclass(kw_only=True)
class PostureCommandCfg(CommandTermCfg):
  """Uniform ``[target_height, target_pitch]`` command."""

  height_range: tuple[float, float]
  pitch_range: tuple[float, float]
  qualified: bool = False
  source: str = "local-unqualified-initial-posture"
  map_hash: str | None = None

  @property
  def command_dim(self) -> int:
    return 2

  def build(self, env: ManagerBasedRlEnv) -> "PostureCommand":
    return PostureCommand(self, env)


class PostureCommand(CommandTerm):
  """Sample target height and pitch independently for each environment."""

  cfg: PostureCommandCfg

  def __init__(self, cfg: PostureCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._command = torch.zeros(self.num_envs, 2, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._command[env_ids, 0].uniform_(*self.cfg.height_range)
    self._command[env_ids, 1].uniform_(*self.cfg.pitch_range)

  def _update_command(self) -> None:
    pass


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
  posture_map_qualified: bool = False
  posture_map_source: str = "local-unqualified-initial-posture"
  posture_map_hash: str | None = None
  velocity_command_name: str = "twist"
  posture_command_name: str = "posture"
  wheel_radius: float = DEFAULT_WHEEL_RADIUS
  wheel_velocity_limit: float = DEFAULT_WHEEL_VELOCITY_LIMIT
  wheel_slew_limit: float = DEFAULT_WHEEL_SLEW_LIMIT

  def __post_init__(self) -> None:
    if len(self.action_mask) != 6 or len(self.action_scales) != 6:
      raise ValueError("Hybrid action mask and scales must each contain six values.")
    if len(self.controller_gain) != 4:
      raise ValueError("Hybrid controller gain must contain four values.")
    if np.asarray(self.posture_coefficients).shape != (3, 4):
      raise ValueError("Posture coefficients must have shape (3, 4).")

  @property
  def action_dim(self) -> int:
    return 6

  def build(self, env: ManagerBasedRlEnv) -> "HybridWheelLegAction":
    return HybridWheelLegAction(self, env)


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
    self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
    self._applied_residual = torch.zeros_like(self._raw_actions)
    self._controller_baseline = torch.zeros(self.num_envs, 2, device=self.device)
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
  def controller_baseline(self) -> torch.Tensor:
    return self._controller_baseline

  @property
  def wheel_targets(self) -> torch.Tensor:
    return self._wheel_targets

  @property
  def nominal_leg_targets(self) -> torch.Tensor:
    return self._nominal_leg_targets

  @property
  def leg_targets(self) -> torch.Tensor:
    return self._leg_targets

  def process_actions(self, actions: torch.Tensor) -> None:
    if actions.shape != self._raw_actions.shape:
      raise ValueError(
        f"Hybrid action expects shape {tuple(self._raw_actions.shape)}, "
        f"got {tuple(actions.shape)}."
      )
    self._raw_actions[:] = actions
    self._applied_residual[:] = (
      torch.clamp(actions, -1.0, 1.0) * self._mask * self._scales
    )

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
    vx_error = (
      self._entity.data.root_link_lin_vel_b[:, 0] - velocity_command[:, 0]
    )
    wheel_speed = self._entity.data.joint_vel[:, self._wheel_ids]
    signed_wheel_speed = 0.5 * (wheel_speed[:, 1] - wheel_speed[:, 0])
    desired_wheel_speed = velocity_command[:, 0] / self.cfg.wheel_radius
    wheel_speed_error = signed_wheel_speed - desired_wheel_speed
    state = torch.stack(
      (pitch, pitch_rate, vx_error, wheel_speed_error),
      dim=1,
    )
    control = -(state @ self._gain)
    self._controller_baseline[:, 0] = -control
    self._controller_baseline[:, 1] = control

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
    desired_legs = self._nominal_leg_targets + self._applied_residual[:, 2:]
    soft_limits = self._entity.data.soft_joint_pos_limits
    self._leg_targets[:] = torch.clamp(
      desired_legs,
      min=soft_limits[:, self._leg_ids, 0],
      max=soft_limits[:, self._leg_ids, 1],
    )

  def apply_actions(self) -> None:
    self._entity.set_joint_velocity_target(
      self._wheel_targets,
      joint_ids=self._wheel_ids,
    )
    encoder_bias = self._entity.data.encoder_bias[:, self._leg_ids]
    self._entity.set_joint_position_target(
      self._leg_targets - encoder_bias,
      joint_ids=self._leg_ids,
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._applied_residual[env_ids] = 0.0
    self._controller_baseline[env_ids] = 0.0
    self._previous_wheel_targets[env_ids] = 0.0
    self._wheel_targets[env_ids] = 0.0
    initial = torch.tensor(
      [INIT_JOINT_POS[name] for name in self.cfg.leg_joint_names],
      device=self.device,
      dtype=torch.float,
    )
    self._nominal_leg_targets[env_ids] = initial
    self._leg_targets[env_ids] = initial


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


def _base_env_cfg(stage: int, play: bool) -> ManagerBasedRlEnvCfg:
  if stage == 0:
    return make_hoppertrex_balance_env_cfg(play=play)
  if stage == 1:
    return make_hoppertrex_balance_env_cfg(
      play=play,
      slow_speed=True,
      speed_level=0,
      slow_speed_lin_sign=True,
      slow_speed_obs_scale=True,
      zero_wheel_joint_pos_obs=True,
    )
  return make_hoppertrex_balance_env_cfg(
    play=play,
    robust=stage == 5,
    robust_level=2,
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


def make_hoppertrex_hybrid_env_cfg(
  stage: int,
  play: bool = False,
  controller_path: Path | None = None,
  posture_map_path: Path | None = None,
) -> ManagerBasedRlEnvCfg:
  """Build one Hybrid v2 stage without changing the legacy task factory."""

  if stage not in HYBRID_STAGES:
    raise ValueError(f"Unsupported Hybrid stage {stage}; expected an integer from 0 to 5.")
  stage_cfg = HYBRID_STAGES[stage]
  controller = _load_controller(
    _artifact_path(controller_path, CONTROLLER_PATH_ENV)
  )
  posture = _load_posture_map(
    _artifact_path(posture_map_path, POSTURE_MAP_PATH_ENV)
  )
  cfg = _base_env_cfg(stage, play)

  cfg.actions = {
    "hybrid_wheel_leg": HybridWheelLegActionCfg(
      entity_name="robot",
      action_mask=stage_cfg.action_mask,
      action_scales=stage_cfg.action_scales,
      controller_gain=controller.gain,
      controller_type=controller.controller_type,
      controller_qualified=controller.qualified,
      controller_source=controller.source,
      controller_gain_hash=controller.gain_hash,
      posture_coefficients=posture.coefficients,
      posture_map_qualified=posture.qualified,
      posture_map_source=posture.source,
      posture_map_hash=posture.map_hash,
    )
  }
  cfg.commands = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(5.0, 10.0),
      rel_standing_envs=0.0,
      rel_heading_envs=0.0,
      rel_forward_envs=0.0,
      heading_command=False,
      debug_vis=play,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=stage_cfg.lin_vel_x_range,
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=stage_cfg.yaw_rate_range,
      ),
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

  if stage == 5 and not play:
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

  return cfg


__all__ = [
  "HOPPERTREX_HYBRID_TASK_IDS",
  "HYBRID_TASK_IDS",
  "HybridWheelLegAction",
  "HybridWheelLegActionCfg",
  "PostureCommand",
  "PostureCommandCfg",
  "WHEEL_JOINT_NAMES",
  "make_hoppertrex_hybrid_env_cfg",
]
