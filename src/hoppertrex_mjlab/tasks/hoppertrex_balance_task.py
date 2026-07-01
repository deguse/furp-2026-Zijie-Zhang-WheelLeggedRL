"""HopperTrex two-wheel balance task for MjLab."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import (
  JointPositionAction,
  JointPositionActionCfg,
  JointVelocityAction,
  JointVelocityActionCfg,
)
from mjlab.managers import (
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  SceneEntityCfg,
  TerminationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from assets.HopperTrex_CFG import (
  INIT_JOINT_POS,
  LEG_JOINT_NAMES,
  WHEEL_JOINT_NAMES,
  WHEEL_VELOCITY_ACTION_SCALE,
  get_hoppertrex_robot_cfg,
)

LEG_INIT_JOINT_POS = {
  name: INIT_JOINT_POS[name]
  for name in LEG_JOINT_NAMES
}
NON_WHEEL_GROUND_SENSOR_NAME = "non_wheel_ground_touch"
NON_WHEEL_GROUND_GEOMS = (
  "thigh_left_collision",
  "thigh_right_collision",
  "calf_left_collision",
  "calf_right_collision",
  "chassis_base_collision",
)
WHEEL_GROUND_SENSOR_NAME = "wheel_ground_touch"
WHEEL_GROUND_GEOMS = (
  "wheel_left_collision",
  "wheel_right_collision",
)

WHEEL_ACTION_CLIP = 1.0
ROOT_HEIGHT_TARGET = 0.325
ROOT_HEIGHT_SOFT_MIN = 0.30
ROOT_HEIGHT_HARD_MIN = 0.26
BAD_ORIENTATION_LIMIT_ANGLE = 0.55
NON_WHEEL_CONTACT_GRACE_STEPS = 5
CLEAN_SUPPORT_MIN_HEIGHT = 0.29
CLEAN_SUPPORT_MAX_TILT_XY = 0.20
ROBUST_INIT_ANGLE_RANGE = math.radians(2.0)
ROBUST_INIT_LIN_VEL_X_RANGE = 0.05
ROBUST_INIT_ANG_VEL_XY_RANGE = 0.10
ROBUST_L2_INIT_ANGLE_RANGE = math.radians(5.0)
ROBUST_L2_INIT_LIN_VEL_X_RANGE = 0.10
ROBUST_L2_INIT_ANG_VEL_XY_RANGE = 0.20
PUSH_L3_INTERVAL_RANGE_S = (2.0, 4.0)
PUSH_L3_LIN_VEL_X_RANGE = 0.15
PUSH_L3_ANG_VEL_PITCH_RANGE = 0.25
SLOW_SPEED_LIN_VEL_X_RANGE = 0.10
SLOW_SPEED_STANDING_ENVS = 0.20
SLOW_SPEED_TRACK_LIN_VEL_WEIGHT = 2.0
SLOW_SPEED_TRACK_LIN_VEL_STD = 0.10
SLOW_SPEED_LIN_VEL_XY_PENALTY_WEIGHT = -0.002
SLOW_SPEED_EASY_LIN_VEL_X_RANGE = 0.05
SLOW_SPEED_EASY_STANDING_ENVS = 0.10
SLOW_SPEED_EASY_TRACK_LIN_VEL_WEIGHT = 3.0
SLOW_SPEED_EASY_TRACK_LIN_VEL_STD = 0.08
SLOW_SPEED_EASY_LIN_VEL_XY_PENALTY_WEIGHT = -0.001
SLOW_SPEED_TURN_LIN_VEL_X_RANGE = (0.03, 0.08)
SLOW_SPEED_TURN_ANG_VEL_Z_RANGE = 0.10
SLOW_SPEED_TURN_STANDING_ENVS = 0.0
SLOW_SPEED_TURN_TRACK_LIN_VEL_WEIGHT = 2.0
SLOW_SPEED_TURN_TRACK_LIN_VEL_STD = 0.08
SLOW_SPEED_TURN_TRACK_ANG_VEL_WEIGHT = 2.0
SLOW_SPEED_TURN_TRACK_ANG_VEL_STD = 0.20
SLOW_SPEED_TURN_LIN_VEL_XY_PENALTY_WEIGHT = -0.001
SLOW_SPEED_TURN_YAW_SCALE = 2.0
SLOW_SPEED_TURN_SIGN_YAW_WEIGHT = 4.0
SLOW_SPEED_TURN_OBS_COMMAND_SCALE = (10.0, 1.0, 10.0)
SLOW_SPEED_TURN_SAFE_CLEAN_WHEEL_SUPPORT_WEIGHT = 6.0
SLOW_SPEED_TURN_SAFE_WHEEL_GROUND_CONTACT_WEIGHT = 2.0
SLOW_SPEED_TURN_SAFE_NON_WHEEL_GROUND_CONTACT_WEIGHT = -8.0
SLOW_SPEED_TURN_SAFE_TRACK_ANG_VEL_WEIGHT = 1.0
SLOW_SPEED_TURN_SAFE_YAW_SIGN_WEIGHT = 1.5
SLOW_SPEED_TURN_SAFE_V2_CLEAN_WHEEL_SUPPORT_WEIGHT = 5.0
SLOW_SPEED_TURN_SAFE_V2_WHEEL_GROUND_CONTACT_WEIGHT = 1.5
SLOW_SPEED_TURN_SAFE_V2_NON_WHEEL_GROUND_CONTACT_WEIGHT = -7.0
SLOW_SPEED_TURN_SAFE_V2_TRACK_ANG_VEL_WEIGHT = 1.5
SLOW_SPEED_TURN_SAFE_V2_YAW_SIGN_WEIGHT = 2.5
SLOW_SPEED_TURN_SAFE_V2_YAW_SCALE_3 = 3.0
TURN_L4_ANG_VEL_Z_RANGE = 0.30
TURN_L4_STANDING_ENVS = 0.20
TURN_L4_ANG_VEL_WEIGHT = 2.0
TURN_L4_ANG_VEL_STD = 0.25
TURN_L4_TRACK_STANDING_ENVS = 0.05
TURN_L4_TRACK_ANG_VEL_WEIGHT = 5.0
TURN_L4_TRACK_ANG_VEL_STD = 0.18
TURN_L4_TRACK_LIN_VEL_XY_PENALTY_WEIGHT = -0.005
TURN_L4_TRACK_WHEEL_VEL_PENALTY_WEIGHT = -2.0e-4
TURN_L4_TRACK_ACTION_RATE_PENALTY_WEIGHT = -0.003
TURN_L4_TRACK_V2_STANDING_ENVS = 0.05
TURN_L4_TRACK_V2_ANG_VEL_WEIGHT = 4.0
TURN_L4_TRACK_V2_ANG_VEL_STD = 0.22
TURN_L4_TRACK_V2_LIN_VEL_XY_PENALTY_WEIGHT = -0.005
TURN_L4_TRACK_V2_WHEEL_VEL_PENALTY_WEIGHT = -3.0e-4
TURN_L4_TRACK_V2_ACTION_RATE_PENALTY_WEIGHT = -0.006
TURN_L4_EASY_ANG_VEL_Z_RANGE = 0.10
TURN_L4_EASY_STANDING_ENVS = 0.10
TURN_L4_EASY_ANG_VEL_WEIGHT = 3.0
TURN_L4_EASY_ANG_VEL_STD = 0.20
TURN_L4_EASY_LIN_VEL_XY_PENALTY_WEIGHT = -0.005
TURN_L4_EASY_WHEEL_VEL_PENALTY_WEIGHT = -3.0e-4
TURN_L4_EASY_ACTION_RATE_PENALTY_WEIGHT = -0.006
TURN_L4_EASY_LOW_YAW_SCALE = 2.0
TURN_L4_SIGN_YAW_ABS = 0.10
TURN_L4_SIGN_YAW_WEIGHT = 2.0
TURN_L4_SIGN_YAW_DEADBAND = 0.02


@dataclass(kw_only=True)
class FixedJointPositionActionCfg(JointPositionActionCfg):
  """Hold joints at fixed position targets without exposing policy actions."""

  def build(self, env: ManagerBasedRlEnv) -> "FixedJointPositionAction":
    return FixedJointPositionAction(self, env)


class FixedJointPositionAction(JointPositionAction):
  """Apply fixed joint position targets with zero action dimension."""

  def __init__(self, cfg: FixedJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._action_dim = 0
    self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)
    if isinstance(self._offset, torch.Tensor):
      self._processed_actions = self._offset.clone()
    else:
      self._processed_actions = torch.full(
        (self.num_envs, self._num_targets),
        float(self._offset),
        device=self.device,
      )

  def process_actions(self, actions: torch.Tensor):
    if actions.shape[-1] != 0:
      raise ValueError(
        "FixedJointPositionAction expects action dimension 0, "
        f"got action shape {tuple(actions.shape)}."
      )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids


@dataclass(kw_only=True)
class CoupledWheelVelocityActionCfg(JointVelocityActionCfg):
  """One-dimensional symmetric wheel velocity action for pitch balance."""

  def build(self, env: ManagerBasedRlEnv) -> "CoupledWheelVelocityAction":
    return CoupledWheelVelocityAction(self, env)


class CoupledWheelVelocityAction(JointVelocityAction):
  """Map one policy scalar to opposite wheel velocity targets.

  The wheel joint axes are mirrored, so a forward/backward chassis correction
  uses opposite signed joint targets: left=-u, right=+u.
  """

  def __init__(self, cfg: CoupledWheelVelocityActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    if self._num_targets != 2:
      raise ValueError(
        "CoupledWheelVelocityAction expects exactly two wheel joints, "
        f"got {self._num_targets}: {self._target_names}"
      )
    if not isinstance(cfg.scale, (float, int)):
      raise ValueError("CoupledWheelVelocityAction expects cfg.scale to be a float.")

    self._left_idx = self._target_names.index("wheel_left")
    self._right_idx = self._target_names.index("wheel_right")
    self._action_dim = 1
    self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
    self._processed_actions = torch.zeros(
      self.num_envs,
      self._num_targets,
      device=self.device,
    )
    self._coupled_scale = float(cfg.scale)

  def process_actions(self, actions: torch.Tensor):
    if actions.shape[-1] != 1:
      raise ValueError(
        "CoupledWheelVelocityAction expects action dimension 1, "
        f"got action shape {tuple(actions.shape)}."
      )

    raw = torch.clamp(actions[:, 0], -WHEEL_ACTION_CLIP, WHEEL_ACTION_CLIP)
    self._raw_actions[:, 0] = raw
    u = raw * self._coupled_scale
    self._processed_actions[:, self._left_idx] = -u
    self._processed_actions[:, self._right_idx] = u

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._processed_actions[env_ids] = 0.0


@dataclass(kw_only=True)
class DifferentialWheelVelocityActionCfg(JointVelocityActionCfg):
  """Two-dimensional wheel velocity action for pitch balance plus yaw."""

  yaw_scale: float | None = None
  """Optional scale for the yaw action channel. Defaults to ``scale``."""

  def build(self, env: ManagerBasedRlEnv) -> "DifferentialWheelVelocityAction":
    return DifferentialWheelVelocityAction(self, env)


class DifferentialWheelVelocityAction(JointVelocityAction):
  """Map policy scalars to wheel targets.

  ``actions[:, 0]`` is the existing pitch/forward balance channel.
  ``actions[:, 1]`` is the new yaw channel. Wheel joint axes are mirrored, so
  forward balance uses opposite signed joint targets, while yaw adds the same
  signed target to both joints.
  """

  def __init__(self, cfg: DifferentialWheelVelocityActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    if self._num_targets != 2:
      raise ValueError(
        "DifferentialWheelVelocityAction expects exactly two wheel joints, "
        f"got {self._num_targets}: {self._target_names}"
      )
    if not isinstance(cfg.scale, (float, int)):
      raise ValueError(
        "DifferentialWheelVelocityAction expects cfg.scale to be a float."
      )

    self._left_idx = self._target_names.index("wheel_left")
    self._right_idx = self._target_names.index("wheel_right")
    self._action_dim = 2
    self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
    self._processed_actions = torch.zeros(
      self.num_envs,
      self._num_targets,
      device=self.device,
    )
    self._balance_scale = float(cfg.scale)
    self._yaw_scale = float(cfg.scale if cfg.yaw_scale is None else cfg.yaw_scale)

  def process_actions(self, actions: torch.Tensor):
    if actions.shape[-1] != 2:
      raise ValueError(
        "DifferentialWheelVelocityAction expects action dimension 2, "
        f"got action shape {tuple(actions.shape)}."
      )

    raw = torch.clamp(actions[:, :2], -WHEEL_ACTION_CLIP, WHEEL_ACTION_CLIP)
    self._raw_actions[:, :] = raw
    balance = raw[:, 0] * self._balance_scale
    yaw = raw[:, 1] * self._yaw_scale
    left = torch.clamp(-balance + yaw, -self._balance_scale, self._balance_scale)
    right = torch.clamp(balance + yaw, -self._balance_scale, self._balance_scale)
    self._processed_actions[:, self._left_idx] = left
    self._processed_actions[:, self._right_idx] = right

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._processed_actions[env_ids] = 0.0


@dataclass(kw_only=True)
class BinaryYawVelocityCommandCfg(UniformVelocityCommandCfg):
  """Velocity command that samples only positive or negative yaw targets."""

  yaw_abs: float = TURN_L4_SIGN_YAW_ABS

  def build(self, env: ManagerBasedRlEnv) -> "BinaryYawVelocityCommand":
    return BinaryYawVelocityCommand(self, env)


class BinaryYawVelocityCommand(UniformVelocityCommand):
  cfg: BinaryYawVelocityCommandCfg

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.vel_command_b[env_ids, :] = 0.0
    self.vel_command_w[env_ids, :] = 0.0
    signs = torch.where(
      torch.rand(len(env_ids), device=self.device) < 0.5,
      -1.0,
      1.0,
    )
    self.vel_command_b[env_ids, 2] = signs * self.cfg.yaw_abs
    self.is_heading_env[env_ids] = False
    self.is_standing_env[env_ids] = False
    self.is_world_env[env_ids] = False
    self.is_forward_env[env_ids] = False


@dataclass(kw_only=True)
class BinarySlowSpeedTurnCommandCfg(UniformVelocityCommandCfg):
  """Slow forward command with binary left/right yaw targets."""

  yaw_abs: float = SLOW_SPEED_TURN_ANG_VEL_Z_RANGE

  def build(self, env: ManagerBasedRlEnv) -> "BinarySlowSpeedTurnCommand":
    return BinarySlowSpeedTurnCommand(self, env)


class BinarySlowSpeedTurnCommand(UniformVelocityCommand):
  cfg: BinarySlowSpeedTurnCommandCfg

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, :] = 0.0
    self.vel_command_w[env_ids, :] = 0.0
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    signs = torch.where(
      torch.rand(len(env_ids), device=self.device) < 0.5,
      -1.0,
      1.0,
    )
    self.vel_command_b[env_ids, 2] = signs * self.cfg.yaw_abs
    self.is_heading_env[env_ids] = False
    self.is_standing_env[env_ids] = False
    self.is_world_env[env_ids] = False
    self.is_forward_env[env_ids] = False


def lin_vel_z_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = env.scene["robot"]
  return torch.square(robot.data.root_link_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = env.scene["robot"]
  return torch.sum(torch.square(robot.data.root_link_ang_vel_b[:, :2]), dim=1)


def lin_vel_xy_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = env.scene["robot"]
  return torch.sum(torch.square(robot.data.root_link_lin_vel_b[:, :2]), dim=1)


def root_height_l2(env: ManagerBasedRlEnv, target_height: float) -> torch.Tensor:
  robot = env.scene["robot"]
  return torch.square(robot.data.root_link_pos_w[:, 2] - target_height)


def root_height_below_minimum_l2(
  env: ManagerBasedRlEnv,
  minimum_height: float,
) -> torch.Tensor:
  robot = env.scene["robot"]
  height_error = torch.clamp(minimum_height - robot.data.root_link_pos_w[:, 2], min=0.0)
  return torch.square(height_error)


def _contact_any(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  if found is not None:
    return torch.any(found.reshape(found.shape[0], -1) > 0, dim=-1)

  assert sensor.data.force is not None
  force = torch.norm(sensor.data.force.reshape(sensor.data.force.shape[0], -1, 3), dim=-1)
  return torch.any(force > 0.0, dim=-1)


def _contact_all(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  if found is not None:
    flat_found = found.reshape(found.shape[0], -1) > 0
    return torch.all(flat_found, dim=-1)

  assert sensor.data.force is not None
  force = torch.norm(sensor.data.force.reshape(sensor.data.force.shape[0], -1, 3), dim=-1)
  return torch.all(force > 0.0, dim=-1)


def wheel_ground_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  return _contact_all(env, sensor_name).float()


def non_wheel_ground_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  return _contact_any(env, sensor_name).float()


def clean_wheel_support(
  env: ManagerBasedRlEnv,
  wheel_sensor_name: str,
  non_wheel_sensor_name: str,
  minimum_height: float,
  max_tilt_xy: float,
) -> torch.Tensor:
  robot = env.scene["robot"]
  wheel_contact = _contact_all(env, wheel_sensor_name)
  non_wheel_contact = _contact_any(env, non_wheel_sensor_name)
  root_ok = robot.data.root_link_pos_w[:, 2] > minimum_height
  tilt_xy = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=-1)
  tilt_ok = tilt_xy < max_tilt_xy
  return (wheel_contact & ~non_wheel_contact & root_ok & tilt_ok).float()


def yaw_sign_alignment(
  env: ManagerBasedRlEnv,
  command_name: str,
  deadband: float,
) -> torch.Tensor:
  robot = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  cmd_yaw = command[:, 2]
  actual_yaw = robot.data.root_link_ang_vel_b[:, 2]
  active = torch.abs(cmd_yaw) > deadband
  normalized = torch.clamp(
    (cmd_yaw * actual_yaw) / torch.clamp(torch.square(cmd_yaw), min=deadband**2),
    min=-1.0,
    max=1.0,
  )
  return torch.where(active, normalized, torch.zeros_like(normalized))


def scaled_velocity_commands(
  env: ManagerBasedRlEnv,
  command_name: str,
  scale: tuple[float, float, float],
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  scale_tensor = torch.tensor(scale, device=command.device, dtype=command.dtype)
  return command * scale_tensor


def non_wheel_ground_contact_after_grace(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  grace_steps: int,
) -> torch.Tensor:
  return _contact_any(env, sensor_name) & (env.episode_length_buf > grace_steps)


def make_hoppertrex_balance_env_cfg(
  play: bool = False,
  robust: bool = False,
  robust_level: int = 1,
  push_l3: bool = False,
  slow_speed: bool = False,
  speed_level: int = 1,
  slow_speed_turn: bool = False,
  slow_speed_turn_sign: bool = False,
  slow_speed_turn_obs_scale: bool = False,
  slow_speed_turn_safe: bool = False,
  slow_speed_turn_safe_v2: bool = False,
  slow_speed_turn_safe_v2_yaw_scale3: bool = False,
  turn_l4: bool = False,
  turn_level: int = 1,
) -> ManagerBasedRlEnvCfg:
  robot_cfg = get_hoppertrex_robot_cfg()
  num_envs = 16 if play else 4096
  command_lin_vel_x_range = (0.0, 0.0)
  command_ang_vel_z_range = (0.0, 0.0)
  rel_standing_envs = 1.0
  lin_vel_xy_penalty_weight = -0.02
  wheel_vel_penalty_weight = -5.0e-4
  action_rate_penalty_weight = -0.01
  wheel_yaw_scale: float | None = None
  binary_yaw_command = False
  binary_slow_speed_turn_command = False
  yaw_sign_reward = False
  track_lin_vel_weight = SLOW_SPEED_TRACK_LIN_VEL_WEIGHT
  track_lin_vel_std = SLOW_SPEED_TRACK_LIN_VEL_STD
  track_ang_vel_weight = TURN_L4_ANG_VEL_WEIGHT
  track_ang_vel_std = TURN_L4_ANG_VEL_STD
  clean_wheel_support_weight = 4.0
  wheel_ground_contact_weight = 1.0
  non_wheel_ground_contact_weight = -6.0
  yaw_sign_weight = SLOW_SPEED_TURN_SIGN_YAW_WEIGHT
  command_obs_func = envs_mdp.generated_commands
  command_obs_params: dict[str, object] = {"command_name": "twist"}
  use_differential_wheel_action = turn_l4 or slow_speed_turn
  if slow_speed:
    if speed_level == 0:
      command_lin_vel_x_range = (
        -SLOW_SPEED_EASY_LIN_VEL_X_RANGE,
        SLOW_SPEED_EASY_LIN_VEL_X_RANGE,
      )
      rel_standing_envs = SLOW_SPEED_EASY_STANDING_ENVS
      lin_vel_xy_penalty_weight = SLOW_SPEED_EASY_LIN_VEL_XY_PENALTY_WEIGHT
      track_lin_vel_weight = SLOW_SPEED_EASY_TRACK_LIN_VEL_WEIGHT
      track_lin_vel_std = SLOW_SPEED_EASY_TRACK_LIN_VEL_STD
    elif speed_level == 1:
      command_lin_vel_x_range = (
        -SLOW_SPEED_LIN_VEL_X_RANGE,
        SLOW_SPEED_LIN_VEL_X_RANGE,
      )
      rel_standing_envs = SLOW_SPEED_STANDING_ENVS
      lin_vel_xy_penalty_weight = SLOW_SPEED_LIN_VEL_XY_PENALTY_WEIGHT
    else:
      raise ValueError(f"Unsupported speed_level={speed_level}. Expected 0 or 1.")
  if slow_speed_turn:
    command_lin_vel_x_range = SLOW_SPEED_TURN_LIN_VEL_X_RANGE
    command_ang_vel_z_range = (
      -SLOW_SPEED_TURN_ANG_VEL_Z_RANGE,
      SLOW_SPEED_TURN_ANG_VEL_Z_RANGE,
    )
    rel_standing_envs = SLOW_SPEED_TURN_STANDING_ENVS
    track_lin_vel_weight = SLOW_SPEED_TURN_TRACK_LIN_VEL_WEIGHT
    track_lin_vel_std = SLOW_SPEED_TURN_TRACK_LIN_VEL_STD
    track_ang_vel_weight = SLOW_SPEED_TURN_TRACK_ANG_VEL_WEIGHT
    track_ang_vel_std = SLOW_SPEED_TURN_TRACK_ANG_VEL_STD
    lin_vel_xy_penalty_weight = SLOW_SPEED_TURN_LIN_VEL_XY_PENALTY_WEIGHT
    wheel_yaw_scale = SLOW_SPEED_TURN_YAW_SCALE
    if slow_speed_turn_sign:
      command_ang_vel_z_range = (
        -SLOW_SPEED_TURN_ANG_VEL_Z_RANGE,
        SLOW_SPEED_TURN_ANG_VEL_Z_RANGE,
      )
      binary_slow_speed_turn_command = True
      yaw_sign_reward = True
    if slow_speed_turn_obs_scale:
      command_obs_func = scaled_velocity_commands
      command_obs_params = {
        "command_name": "twist",
        "scale": SLOW_SPEED_TURN_OBS_COMMAND_SCALE,
      }
    if slow_speed_turn_safe:
      # Keep the learned yaw sign, but make PPO prefer clean two-wheel support.
      clean_wheel_support_weight = SLOW_SPEED_TURN_SAFE_CLEAN_WHEEL_SUPPORT_WEIGHT
      wheel_ground_contact_weight = SLOW_SPEED_TURN_SAFE_WHEEL_GROUND_CONTACT_WEIGHT
      non_wheel_ground_contact_weight = (
        SLOW_SPEED_TURN_SAFE_NON_WHEEL_GROUND_CONTACT_WEIGHT
      )
      track_ang_vel_weight = SLOW_SPEED_TURN_SAFE_TRACK_ANG_VEL_WEIGHT
      yaw_sign_weight = SLOW_SPEED_TURN_SAFE_YAW_SIGN_WEIGHT
    if slow_speed_turn_safe_v2:
      # Middle ground after Safe-v1 over-regularized yaw into a weak-turn policy.
      clean_wheel_support_weight = SLOW_SPEED_TURN_SAFE_V2_CLEAN_WHEEL_SUPPORT_WEIGHT
      wheel_ground_contact_weight = SLOW_SPEED_TURN_SAFE_V2_WHEEL_GROUND_CONTACT_WEIGHT
      non_wheel_ground_contact_weight = (
        SLOW_SPEED_TURN_SAFE_V2_NON_WHEEL_GROUND_CONTACT_WEIGHT
      )
      track_ang_vel_weight = SLOW_SPEED_TURN_SAFE_V2_TRACK_ANG_VEL_WEIGHT
      yaw_sign_weight = SLOW_SPEED_TURN_SAFE_V2_YAW_SIGN_WEIGHT
    if slow_speed_turn_safe_v2_yaw_scale3:
      wheel_yaw_scale = SLOW_SPEED_TURN_SAFE_V2_YAW_SCALE_3
  if turn_l4:
    if turn_level == 1:
      command_ang_vel_z_range = (
        -TURN_L4_ANG_VEL_Z_RANGE,
        TURN_L4_ANG_VEL_Z_RANGE,
      )
      rel_standing_envs = TURN_L4_STANDING_ENVS
      track_ang_vel_weight = TURN_L4_ANG_VEL_WEIGHT
      track_ang_vel_std = TURN_L4_ANG_VEL_STD
    elif turn_level == 2:
      command_ang_vel_z_range = (
        -TURN_L4_ANG_VEL_Z_RANGE,
        TURN_L4_ANG_VEL_Z_RANGE,
      )
      rel_standing_envs = TURN_L4_TRACK_STANDING_ENVS
      track_ang_vel_weight = TURN_L4_TRACK_ANG_VEL_WEIGHT
      track_ang_vel_std = TURN_L4_TRACK_ANG_VEL_STD
      lin_vel_xy_penalty_weight = TURN_L4_TRACK_LIN_VEL_XY_PENALTY_WEIGHT
      wheel_vel_penalty_weight = TURN_L4_TRACK_WHEEL_VEL_PENALTY_WEIGHT
      action_rate_penalty_weight = TURN_L4_TRACK_ACTION_RATE_PENALTY_WEIGHT
    elif turn_level == 3:
      command_ang_vel_z_range = (
        -TURN_L4_ANG_VEL_Z_RANGE,
        TURN_L4_ANG_VEL_Z_RANGE,
      )
      rel_standing_envs = TURN_L4_TRACK_V2_STANDING_ENVS
      track_ang_vel_weight = TURN_L4_TRACK_V2_ANG_VEL_WEIGHT
      track_ang_vel_std = TURN_L4_TRACK_V2_ANG_VEL_STD
      lin_vel_xy_penalty_weight = TURN_L4_TRACK_V2_LIN_VEL_XY_PENALTY_WEIGHT
      wheel_vel_penalty_weight = TURN_L4_TRACK_V2_WHEEL_VEL_PENALTY_WEIGHT
      action_rate_penalty_weight = TURN_L4_TRACK_V2_ACTION_RATE_PENALTY_WEIGHT
    elif turn_level == 4:
      command_ang_vel_z_range = (
        -TURN_L4_EASY_ANG_VEL_Z_RANGE,
        TURN_L4_EASY_ANG_VEL_Z_RANGE,
      )
      rel_standing_envs = TURN_L4_EASY_STANDING_ENVS
      track_ang_vel_weight = TURN_L4_EASY_ANG_VEL_WEIGHT
      track_ang_vel_std = TURN_L4_EASY_ANG_VEL_STD
      lin_vel_xy_penalty_weight = TURN_L4_EASY_LIN_VEL_XY_PENALTY_WEIGHT
      wheel_vel_penalty_weight = TURN_L4_EASY_WHEEL_VEL_PENALTY_WEIGHT
      action_rate_penalty_weight = TURN_L4_EASY_ACTION_RATE_PENALTY_WEIGHT
    elif turn_level == 5:
      command_ang_vel_z_range = (
        -TURN_L4_EASY_ANG_VEL_Z_RANGE,
        TURN_L4_EASY_ANG_VEL_Z_RANGE,
      )
      rel_standing_envs = TURN_L4_EASY_STANDING_ENVS
      track_ang_vel_weight = TURN_L4_EASY_ANG_VEL_WEIGHT
      track_ang_vel_std = TURN_L4_EASY_ANG_VEL_STD
      lin_vel_xy_penalty_weight = TURN_L4_EASY_LIN_VEL_XY_PENALTY_WEIGHT
      wheel_vel_penalty_weight = TURN_L4_EASY_WHEEL_VEL_PENALTY_WEIGHT
      action_rate_penalty_weight = TURN_L4_EASY_ACTION_RATE_PENALTY_WEIGHT
      wheel_yaw_scale = TURN_L4_EASY_LOW_YAW_SCALE
    elif turn_level == 6:
      command_ang_vel_z_range = (
        -TURN_L4_SIGN_YAW_ABS,
        TURN_L4_SIGN_YAW_ABS,
      )
      rel_standing_envs = 0.0
      track_ang_vel_weight = TURN_L4_EASY_ANG_VEL_WEIGHT
      track_ang_vel_std = TURN_L4_EASY_ANG_VEL_STD
      lin_vel_xy_penalty_weight = TURN_L4_EASY_LIN_VEL_XY_PENALTY_WEIGHT
      wheel_vel_penalty_weight = TURN_L4_EASY_WHEEL_VEL_PENALTY_WEIGHT
      action_rate_penalty_weight = TURN_L4_EASY_ACTION_RATE_PENALTY_WEIGHT
      wheel_yaw_scale = TURN_L4_EASY_LOW_YAW_SCALE
      binary_yaw_command = True
      yaw_sign_reward = True
    else:
      raise ValueError(
        f"Unsupported turn_level={turn_level}. Expected 1, 2, 3, 4, 5, or 6."
      )
  non_wheel_ground_cfg = ContactSensorCfg(
    name=NON_WHEEL_GROUND_SENSOR_NAME,
    primary=ContactMatch(mode="geom", pattern=NON_WHEEL_GROUND_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  wheel_ground_cfg = ContactSensorCfg(
    name=WHEEL_GROUND_SENSOR_NAME,
    primary=ContactMatch(mode="geom", pattern=WHEEL_GROUND_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=2,
  )

  observations = {
    "actor": ObservationGroupCfg(
      terms={
        "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
        "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
        "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
        "velocity_commands": ObservationTermCfg(
          func=command_obs_func,
          params=command_obs_params,
        ),
        "joint_pos": ObservationTermCfg(
          func=envs_mdp.joint_pos_rel,
          params={"asset_cfg": SceneEntityCfg("robot")},
          noise=Unoise(n_min=-0.002, n_max=0.002),
        ),
        "joint_vel": ObservationTermCfg(
          func=envs_mdp.joint_vel_rel,
          params={"asset_cfg": SceneEntityCfg("robot")},
          noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "actions": ObservationTermCfg(func=envs_mdp.last_action),
      },
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms={
        "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
        "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
        "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
        "velocity_commands": ObservationTermCfg(
          func=command_obs_func,
          params=command_obs_params,
        ),
        "joint_pos": ObservationTermCfg(
          func=envs_mdp.joint_pos_rel,
          params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "joint_vel": ObservationTermCfg(
          func=envs_mdp.joint_vel_rel,
          params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "actions": ObservationTermCfg(func=envs_mdp.last_action),
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  actions = {
    "fixed_leg_pos": FixedJointPositionActionCfg(
      entity_name="robot",
      actuator_names=LEG_JOINT_NAMES,
      scale=0.0,
      offset=LEG_INIT_JOINT_POS,
      use_default_offset=False,
      preserve_order=True,
    ),
  }
  wheel_action_cfg_cls = (
    DifferentialWheelVelocityActionCfg
    if use_differential_wheel_action
    else CoupledWheelVelocityActionCfg
  )
  wheel_action_kwargs = {
    "entity_name": "robot",
    "actuator_names": WHEEL_JOINT_NAMES,
    "scale": WHEEL_VELOCITY_ACTION_SCALE,
    "offset": 0.0,
    "use_default_offset": False,
    "preserve_order": True,
  }
  if use_differential_wheel_action:
    wheel_action_kwargs["yaw_scale"] = wheel_yaw_scale
  actions["wheel_balance"] = wheel_action_cfg_cls(**wheel_action_kwargs)

  if binary_slow_speed_turn_command:
    command_cfg_cls = BinarySlowSpeedTurnCommandCfg
  elif binary_yaw_command:
    command_cfg_cls = BinaryYawVelocityCommandCfg
  else:
    command_cfg_cls = UniformVelocityCommandCfg
  command_kwargs = {
    "entity_name": "robot",
    "resampling_time_range": (5.0, 10.0),
    "rel_standing_envs": rel_standing_envs,
    "rel_heading_envs": 0.0,
    "rel_forward_envs": 0.0,
    "heading_command": False,
    "debug_vis": play,
    "ranges": UniformVelocityCommandCfg.Ranges(
      lin_vel_x=command_lin_vel_x_range,
      lin_vel_y=(0.0, 0.0),
      ang_vel_z=command_ang_vel_z_range,
    ),
  }
  if binary_yaw_command:
    command_kwargs["yaw_abs"] = TURN_L4_SIGN_YAW_ABS
  if binary_slow_speed_turn_command:
    command_kwargs["yaw_abs"] = SLOW_SPEED_TURN_ANG_VEL_Z_RANGE
  commands = {
    "twist": command_cfg_cls(
      **command_kwargs,
    )
  }

  rewards = {
    "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=0.5),
    "clean_wheel_support": RewardTermCfg(
      func=clean_wheel_support,
      weight=clean_wheel_support_weight,
      params={
        "wheel_sensor_name": WHEEL_GROUND_SENSOR_NAME,
        "non_wheel_sensor_name": NON_WHEEL_GROUND_SENSOR_NAME,
        "minimum_height": CLEAN_SUPPORT_MIN_HEIGHT,
        "max_tilt_xy": CLEAN_SUPPORT_MAX_TILT_XY,
      },
    ),
    "wheel_ground_contact": RewardTermCfg(
      func=wheel_ground_contact,
      weight=wheel_ground_contact_weight,
      params={"sensor_name": WHEEL_GROUND_SENSOR_NAME},
    ),
    "non_wheel_ground_contact": RewardTermCfg(
      func=non_wheel_ground_contact,
      weight=non_wheel_ground_contact_weight,
      params={"sensor_name": NON_WHEEL_GROUND_SENSOR_NAME},
    ),
    "upright": RewardTermCfg(
      func=vel_mdp.upright,
      weight=4.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=("chassis_base",)),
      },
    ),
    "flat_orientation_l2": RewardTermCfg(func=envs_mdp.flat_orientation_l2, weight=-6.0),
    "root_height_l2": RewardTermCfg(
      func=root_height_l2,
      weight=-10.0,
      params={"target_height": ROOT_HEIGHT_TARGET},
    ),
    "root_height_below_minimum_l2": RewardTermCfg(
      func=root_height_below_minimum_l2,
      weight=-20.0,
      params={"minimum_height": ROOT_HEIGHT_SOFT_MIN},
    ),
    "ang_vel_xy_l2": RewardTermCfg(func=ang_vel_xy_l2, weight=-0.15),
    "lin_vel_xy_l2": RewardTermCfg(
      func=lin_vel_xy_l2, weight=lin_vel_xy_penalty_weight
    ),
    "lin_vel_z_l2": RewardTermCfg(func=lin_vel_z_l2, weight=-0.15),
    "wheel_vel_l2": RewardTermCfg(
      func=envs_mdp.joint_vel_l2,
      weight=wheel_vel_penalty_weight,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    ),
    "action_rate_l2": RewardTermCfg(
      func=envs_mdp.action_rate_l2,
      weight=action_rate_penalty_weight,
    ),
  }
  if slow_speed or slow_speed_turn:
    rewards["track_linear_velocity"] = RewardTermCfg(
      func=vel_mdp.track_linear_velocity,
      weight=track_lin_vel_weight,
      params={
        "command_name": "twist",
        "std": track_lin_vel_std,
      },
    )
  if turn_l4 or slow_speed_turn:
    rewards["track_angular_velocity"] = RewardTermCfg(
      func=vel_mdp.track_angular_velocity,
      weight=track_ang_vel_weight,
      params={
        "command_name": "twist",
        "std": track_ang_vel_std,
      },
    )
  if yaw_sign_reward:
    rewards["yaw_sign_alignment"] = RewardTermCfg(
      func=yaw_sign_alignment,
      weight=yaw_sign_weight if slow_speed_turn_sign else TURN_L4_SIGN_YAW_WEIGHT,
      params={
        "command_name": "twist",
        "deadband": TURN_L4_SIGN_YAW_DEADBAND,
      },
    )

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "bad_orientation": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"limit_angle": BAD_ORIENTATION_LIMIT_ANGLE},
    ),
    "root_too_low": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": ROOT_HEIGHT_HARD_MIN},
    ),
    "non_wheel_ground_contact": TerminationTermCfg(
      func=non_wheel_ground_contact_after_grace,
      params={
        "sensor_name": NON_WHEEL_GROUND_SENSOR_NAME,
        "grace_steps": NON_WHEEL_CONTACT_GRACE_STEPS,
      },
    ),
    "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=num_envs,
      env_spacing=2.5,
      terrain=TerrainEntityCfg(terrain_type="plane", env_spacing=2.5),
      entities={"robot": robot_cfg},
      sensors=(non_wheel_ground_cfg, wheel_ground_cfg),
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    rewards=rewards,
    terminations=terminations,
    sim=SimulationCfg(
      nconmax=50,
      njmax=1500,
      contact_sensor_maxmatch=64,
      mujoco=MujocoCfg(
        timestep=0.005,
        integrator="implicitfast",
        cone="elliptic",
        iterations=50,
        ls_iterations=20,
        impratio=10.0,
      ),
    ),
    decimation=4,
    episode_length_s=10.0 if not play else 1.0e9,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="chassis_base",
      distance=2.0,
      elevation=-12.0,
      azimuth=90.0,
    ),
  )

  if push_l3 and not robust:
    raise ValueError("push_l3=True requires robust=True.")
  if slow_speed and not robust:
    raise ValueError("slow_speed=True requires robust=True.")
  if slow_speed_turn and not robust:
    raise ValueError("slow_speed_turn=True requires robust=True.")
  if slow_speed_turn_sign and not slow_speed_turn:
    raise ValueError("slow_speed_turn_sign=True requires slow_speed_turn=True.")
  if slow_speed_turn_obs_scale and not slow_speed_turn:
    raise ValueError("slow_speed_turn_obs_scale=True requires slow_speed_turn=True.")
  if slow_speed_turn_safe and not (
    slow_speed_turn and slow_speed_turn_sign and slow_speed_turn_obs_scale
  ):
    raise ValueError(
      "slow_speed_turn_safe=True requires slow_speed_turn=True, "
      "slow_speed_turn_sign=True, and slow_speed_turn_obs_scale=True."
    )
  if slow_speed_turn_safe_v2 and not (
    slow_speed_turn and slow_speed_turn_sign and slow_speed_turn_obs_scale
  ):
    raise ValueError(
      "slow_speed_turn_safe_v2=True requires slow_speed_turn=True, "
      "slow_speed_turn_sign=True, and slow_speed_turn_obs_scale=True."
    )
  if slow_speed_turn_safe_v2_yaw_scale3 and not slow_speed_turn_safe_v2:
    raise ValueError(
      "slow_speed_turn_safe_v2_yaw_scale3=True requires "
      "slow_speed_turn_safe_v2=True."
    )
  if slow_speed_turn_safe and slow_speed_turn_safe_v2:
    raise ValueError(
      "slow_speed_turn_safe and slow_speed_turn_safe_v2 are mutually exclusive."
    )
  if turn_l4 and not robust:
    raise ValueError("turn_l4=True requires robust=True.")
  if slow_speed and push_l3:
    raise ValueError("slow_speed=True should not be combined with push_l3 in v1.")
  if turn_l4 and push_l3:
    raise ValueError("turn_l4=True should not be combined with push_l3 in v1.")
  if turn_l4 and slow_speed:
    raise ValueError("turn_l4=True should not be combined with slow_speed in v1.")
  if slow_speed_turn and (slow_speed or turn_l4 or push_l3):
    raise ValueError(
      "slow_speed_turn=True should not be combined with slow_speed, turn_l4, "
      "or push_l3 in v1."
    )

  if robust:
    if robust_level == 1:
      robust_angle_range = ROBUST_INIT_ANGLE_RANGE
      robust_lin_vel_x_range = ROBUST_INIT_LIN_VEL_X_RANGE
      robust_ang_vel_xy_range = ROBUST_INIT_ANG_VEL_XY_RANGE
    elif robust_level == 2:
      robust_angle_range = ROBUST_L2_INIT_ANGLE_RANGE
      robust_lin_vel_x_range = ROBUST_L2_INIT_LIN_VEL_X_RANGE
      robust_ang_vel_xy_range = ROBUST_L2_INIT_ANG_VEL_XY_RANGE
    else:
      raise ValueError(f"Unsupported robust_level={robust_level}. Expected 1 or 2.")

    cfg.events = {
      "reset_scene_to_default": EventTermCfg(
        func=envs_mdp.reset_scene_to_default,
        mode="reset",
      ),
      "reset_root_state_with_small_disturbance": EventTermCfg(
        func=envs_mdp.reset_root_state_uniform,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot"),
          "pose_range": {
            "roll": (-robust_angle_range, robust_angle_range),
            "pitch": (-robust_angle_range, robust_angle_range),
          },
          "velocity_range": {
            "x": (-robust_lin_vel_x_range, robust_lin_vel_x_range),
            "roll": (-robust_ang_vel_xy_range, robust_ang_vel_xy_range),
            "pitch": (-robust_ang_vel_xy_range, robust_ang_vel_xy_range),
          },
        },
      ),
    }

  if push_l3:
    cfg.events["push_robot"] = EventTermCfg(
      func=envs_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=PUSH_L3_INTERVAL_RANGE_S,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "velocity_range": {
          "x": (-PUSH_L3_LIN_VEL_X_RANGE, PUSH_L3_LIN_VEL_X_RANGE),
          "pitch": (-PUSH_L3_ANG_VEL_PITCH_RANGE, PUSH_L3_ANG_VEL_PITCH_RANGE),
        },
      },
    )

  return cfg
