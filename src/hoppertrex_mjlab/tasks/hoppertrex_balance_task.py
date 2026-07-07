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
SLOW_SPEED_FEEDFORWARD_GAIN = 2.0
SLOW_SPEED_FEEDFORWARD_CLIP = 0.25
SLOW_SPEED_RESIDUAL_ACTION_SCALE = 0.5
SLOW_SPEED_LOW_RESIDUAL_ACTION_SCALE = 0.15
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
SLOW_SPEED_TURN_PUSH_INTERVAL_RANGE_S = (3.0, 5.0)
SLOW_SPEED_TURN_PUSH_LIN_VEL_X_RANGE = 0.08
SLOW_SPEED_TURN_PUSH_ANG_VEL_PITCH_RANGE = 0.12
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
SLOW_SPEED_EASY_FORWARD_ONLY_LIN_VEL_X_RANGE = (0.02, 0.05)
SLOW_SPEED_EASY_BACKWARD_ONLY_LIN_VEL_X_RANGE = (-0.05, -0.02)
SLOW_SPEED_BACKWARD_STRICT_LIN_VEL_X_RANGE = (-0.08, -0.04)
SLOW_SPEED_BACKWARD_STRICT_TRACK_LIN_VEL_WEIGHT = 4.0
SLOW_SPEED_BACKWARD_STRICT_TRACK_LIN_VEL_STD = 0.035
SLOW_SPEED_BACKWARD_STRICT_LIN_SIGN_WEIGHT = 8.0
SLOW_SPEED_BACKWARD_STRICT_FORWARD_RATIO_WEIGHT = -6.0
SLOW_SPEED_BACKWARD_PITCH_TARGET_GAIN = 1.0
SLOW_SPEED_BACKWARD_PITCH_TARGET_CLIP = 0.08
SLOW_SPEED_BACKWARD_PITCH_TARGET_WEIGHT = -4.0
SLOW_SPEED_OBS_COMMAND_SCALE = (20.0, 1.0, 1.0)
SLOW_SPEED_LIN_SIGN_WEIGHT = 2.0
SLOW_SPEED_LIN_SIGN_DEADBAND = 0.01
LIMITED_LEG_ASSIST_ACTION_SCALE = 0.035
LIMITED_LEG_ASSIST_JOINT_POS_WEIGHT = -2.0
LIMITED_LEG_ASSIST_JOINT_VEL_WEIGHT = -0.01
LIMITED_LEG_ASSIST_SAFE_ACTION_SCALE = 0.015
LIMITED_LEG_ASSIST_SAFE_JOINT_POS_WEIGHT = -10.0
LIMITED_LEG_ASSIST_SAFE_JOINT_VEL_WEIGHT = -0.05
SLOW_SPEED_TURN_LIN_VEL_X_RANGE = (0.03, 0.08)
SLOW_SPEED_TURN_LOW_FORWARD_LIN_VEL_X_RANGE = (0.015, 0.05)
SLOW_SPEED_TURN_MID_FORWARD_LIN_VEL_X_RANGE = (0.02, 0.065)
SLOW_SPEED_TURN_BIDIRECTIONAL_LIN_VEL_X_RANGE = (-0.05, 0.065)
SLOW_SPEED_TURN_ANG_VEL_Z_RANGE = 0.10
SLOW_SPEED_TURN_BIDIRECTIONAL_LOW_YAW_RANGE = 0.05
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
SLOW_SPEED_TURN_SAFE_V2_YAW_SCALE_2P5 = 2.5
SLOW_SPEED_TURN_SAFE_V2_YAW_SMOOTHING_ALPHA = 0.65
SLOW_SPEED_TURN_SAFE_V2_EFFECTIVE_YAW_RATE_WEIGHT = -0.03
SLOW_SPEED_TURN_SAFE_V2_WHEEL_TARGET_RATE_WEIGHT = -5.0e-4
SLOW_SPEED_TURN_SAFE_V2_STABLE_WHEEL_TARGET_RATE_WEIGHT = -7.5e-4
SLOW_SPEED_TURN_SAFE_V2_TARGET_SLEW_LIMIT = 6.0
SLOW_SPEED_TURN_VARIABLE_YAW_ABS_RANGE = (0.04, 0.10)
SLOW_SPEED_TURN_NO_BACKWARD_WEIGHT = -0.6
SLOW_SPEED_TURN_BIDIRECTIONAL_LIN_SIGN_WEIGHT = 1.0
SLOW_SPEED_TURN_BIDIRECTIONAL_LIN_SIGN_STRONG_WEIGHT = 2.0
SLOW_SPEED_TURN_BIDIRECTIONAL_LIN_SIGN_DEADBAND = 0.01
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
TURN_L4_SIGN_MEDIUM_ANG_VEL_WEIGHT = 4.0
TURN_L4_SIGN_MEDIUM_ANG_VEL_STD = 0.14
TURN_L4_SIGN_MEDIUM_YAW_WEIGHT = 3.0
TURN_L4_SIGN_MEDIUM_YAW_SCALE = 2.5
TURN_L4_SIGN_TRACK_ANG_VEL_WEIGHT = 5.0
TURN_L4_SIGN_TRACK_ANG_VEL_STD = 0.12
TURN_L4_SIGN_TRACK_YAW_ABS = 0.07
TURN_L4_SIGN_TRACK_YAW_WEIGHT = 0.5
TURN_L4_SIGN_TRACK_YAW_SCALE = 2.0
TURN_L4_SIGN_TRACK_YAW_ERROR_WEIGHT = -20.0
TURN_L4_SIGN_TRACK_ACTION_RATE_WEIGHT = -0.04
TURN_L4_SIGN_TRACK_WHEEL_TARGET_RATE_WEIGHT = -1.0e-3
TURN_L4_SIGN_TRACK_TARGET_SLEW_LIMIT = 12.0
TURN_L4_SIGN_STRONG_ANG_VEL_WEIGHT = 5.0
TURN_L4_SIGN_STRONG_ANG_VEL_STD = 0.12
TURN_L4_SIGN_STRONG_YAW_WEIGHT = 4.0
TURN_L4_SIGN_STRONG_YAW_SCALE = 3.0
SCRATCH_STAGE0_STABLE_LIN_VEL_XY_WEIGHT = -0.08
SCRATCH_STAGE0_STABLE_WHEEL_VEL_WEIGHT = -1.0e-3
SCRATCH_STAGE0_STABLE_ACTION_RATE_WEIGHT = -0.03
SCRATCH_STAGE1_CLEAR_FORWARD_LIN_VEL_X_RANGE = (0.06, 0.10)
SCRATCH_STAGE1_CLEAR_FORWARD_STANDING_ENVS = 0.25
SCRATCH_STAGE1_CLEAR_FORWARD_TRACK_LIN_VEL_WEIGHT = 4.0
SCRATCH_STAGE1_CLEAR_FORWARD_TRACK_LIN_VEL_STD = 0.04
SCRATCH_STAGE1_CLEAR_FORWARD_LIN_SIGN_WEIGHT = 5.0
SCRATCH_STAGE1_CLEAR_FORWARD_LIN_VEL_XY_WEIGHT = -0.002
SCRATCH_STAGE1_FORWARD_ONLY_CLEAR_LIN_VEL_X_RANGE = (0.08, 0.12)
SCRATCH_STAGE1_FORWARD_ONLY_CLEAR_STANDING_ENVS = 0.0
SCRATCH_STAGE1_FORWARD_ONLY_CLEAR_TRACK_LIN_VEL_WEIGHT = 5.0
SCRATCH_STAGE1_FORWARD_ONLY_CLEAR_TRACK_LIN_VEL_STD = 0.04
SCRATCH_STAGE1_FORWARD_ONLY_CLEAR_LIN_SIGN_WEIGHT = 6.0
SCRATCH_STAGE1_FORWARD_NOSPIKE_LIN_VEL_X_RANGE = (0.055, 0.085)
SCRATCH_STAGE1_FORWARD_NOSPIKE_STANDING_ENVS = 0.25
SCRATCH_STAGE1_FORWARD_NOSPIKE_TRACK_LIN_VEL_WEIGHT = 3.5
SCRATCH_STAGE1_FORWARD_NOSPIKE_TRACK_LIN_VEL_STD = 0.06
SCRATCH_STAGE1_FORWARD_NOSPIKE_LIN_SIGN_WEIGHT = 4.0
SCRATCH_STAGE1_FORWARD_NOSPIKE_ACTION_RATE_WEIGHT = -0.04
SCRATCH_STAGE1_FORWARD_NOSPIKE_WHEEL_TARGET_RATE_WEIGHT = -1.0e-3
SCRATCH_STAGE1_FORWARD_NOSPIKE_PITCH_TAIL_LIMIT = 0.16
SCRATCH_STAGE1_FORWARD_NOSPIKE_PITCH_TAIL_WEIGHT = -30.0
SCRATCH_STAGE1_FORWARD_NOSPIKE_PITCH_RATE_TAIL_LIMIT = 0.90
SCRATCH_STAGE1_FORWARD_NOSPIKE_PITCH_RATE_TAIL_WEIGHT = -0.40
SCRATCH_STAGE1_FORWARD_NOSPIKE_STRONG_LIN_VEL_X_RANGE = (0.055, 0.085)
SCRATCH_STAGE1_FORWARD_NOSPIKE_STRONG_STANDING_ENVS = 0.25
SCRATCH_STAGE1_FORWARD_NOSPIKE_STRONG_TRACK_LIN_VEL_WEIGHT = 3.2
SCRATCH_STAGE1_FORWARD_NOSPIKE_STRONG_TRACK_LIN_VEL_STD = 0.065
SCRATCH_STAGE1_FORWARD_NOSPIKE_STRONG_LIN_SIGN_WEIGHT = 4.0
SCRATCH_STAGE1_FORWARD_NOSPIKE_STRONG_ACTION_RATE_WEIGHT = -0.05
SCRATCH_STAGE1_FORWARD_NOSPIKE_STRONG_WHEEL_TARGET_RATE_WEIGHT = -3.0e-3
SCRATCH_STAGE1_FORWARD_NOSPIKE_STRONG_PITCH_TAIL_LIMIT = 0.10
SCRATCH_STAGE1_FORWARD_NOSPIKE_STRONG_PITCH_TAIL_WEIGHT = -120.0
SCRATCH_STAGE1_FORWARD_NOSPIKE_STRONG_PITCH_RATE_TAIL_LIMIT = 0.75
SCRATCH_STAGE1_FORWARD_NOSPIKE_STRONG_PITCH_RATE_TAIL_WEIGHT = -1.0
SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_LIN_VEL_X_RANGE = (0.055, 0.085)
SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_STANDING_ENVS = 0.25
SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_TRACK_LIN_VEL_WEIGHT = 3.6
SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_TRACK_LIN_VEL_STD = 0.065
SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_LIN_SIGN_WEIGHT = 4.5
SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_ACTION_RATE_WEIGHT = -0.035
SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_WHEEL_TARGET_RATE_WEIGHT = -5.0e-4
SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_TARGET_SLEW_LIMIT = 12.0
SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_PITCH_TAIL_LIMIT = 0.12
SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_PITCH_TAIL_WEIGHT = -60.0
SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_PITCH_RATE_TAIL_LIMIT = 0.85
SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_PITCH_RATE_TAIL_WEIGHT = -0.7
SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_NOREV_STANDING_ENVS = 0.10
SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_NOREV_TRACK_LIN_VEL_WEIGHT = 4.0
SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_NOREV_TRACK_LIN_VEL_STD = 0.055
SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_NOREV_LIN_SIGN_WEIGHT = 6.0
SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_NOREV_BACKWARD_WEIGHT = -12.0
SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_LIN_VEL_X_RANGE = (-0.085, 0.085)
SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_ACTIVE_LIN_VEL_X_ABS_RANGE = (0.05, 0.085)
SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_STANDING_ENVS = 0.20
SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_TRACK_LIN_VEL_WEIGHT = 4.0
SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_TRACK_LIN_VEL_STD = 0.055
SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_LIN_SIGN_WEIGHT = 6.0
SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_ACTION_RATE_WEIGHT = -0.035
SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_WHEEL_TARGET_RATE_WEIGHT = -5.0e-4
SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_TARGET_SLEW_LIMIT = 12.0
SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_PITCH_TAIL_LIMIT = 0.12
SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_PITCH_TAIL_WEIGHT = -60.0
SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_PITCH_RATE_TAIL_LIMIT = 0.85
SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_PITCH_RATE_TAIL_WEIGHT = -0.7
SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_LONG_EPISODE_S = 60.0
SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_SUSTAINED_RESAMPLE_TIME_RANGE = (30.0, 60.0)
SCRATCH_STAGE1_FORWARD_GUARDED_LIN_VEL_X_RANGE = (0.055, 0.085)
SCRATCH_STAGE1_FORWARD_GUARDED_STANDING_ENVS = 0.0
SCRATCH_STAGE1_FORWARD_GUARDED_TRACK_LIN_VEL_WEIGHT = 4.0
SCRATCH_STAGE1_FORWARD_GUARDED_TRACK_LIN_VEL_STD = 0.055
SCRATCH_STAGE1_FORWARD_GUARDED_LIN_SIGN_WEIGHT = 5.0
SCRATCH_STAGE1_FORWARD_GUARDED_UNSAFE_FORWARD_WEIGHT = -8.0
SCRATCH_STAGE1_FORWARD_GUARDED_SAFE_PITCH_ABS = 0.08
SCRATCH_STAGE1_FORWARD_GUARDED_SAFE_PITCH_RATE_ABS = 0.8
SCRATCH_STAGE1_FORWARD_SUPPORT_GUARDED_LIN_VEL_X_RANGE = (0.055, 0.085)
SCRATCH_STAGE1_FORWARD_SUPPORT_GUARDED_STANDING_ENVS = 0.0
SCRATCH_STAGE1_FORWARD_SUPPORT_GUARDED_TRACK_LIN_VEL_WEIGHT = 4.0
SCRATCH_STAGE1_FORWARD_SUPPORT_GUARDED_TRACK_LIN_VEL_STD = 0.055
SCRATCH_STAGE1_FORWARD_SUPPORT_GUARDED_LIN_SIGN_WEIGHT = 5.0
SCRATCH_STAGE1_FORWARD_SUPPORT_GUARDED_UNSAFE_FORWARD_WEIGHT = -8.0
SCRATCH_STAGE1_GENTLE_FORWARD_LIN_VEL_X_RANGE = (0.045, 0.075)
SCRATCH_STAGE1_GENTLE_FORWARD_STANDING_ENVS = 0.40
SCRATCH_STAGE1_GENTLE_FORWARD_TRACK_LIN_VEL_WEIGHT = 3.5
SCRATCH_STAGE1_GENTLE_FORWARD_TRACK_LIN_VEL_STD = 0.06
SCRATCH_STAGE1_GENTLE_FORWARD_LIN_SIGN_WEIGHT = 4.0
SCRATCH_STAGE1_GENTLE_FORWARD_STABLE_WHEEL_TARGET_RATE_WEIGHT = -7.5e-4
SCRATCH_STAGE1_GENTLE_FORWARD_SLEW6_TARGET_SLEW_LIMIT = 6.0
SCRATCH_STAGE1_GENTLE_FORWARD_ZEROHOLD_ACTION_RATE_WEIGHT = -0.01
SCRATCH_STAGE1_GENTLE_FORWARD_ZEROHOLD_ZERO_CMD_RATE_WEIGHT = -1.0e-3
SCRATCH_STAGE1_GENTLE_FORWARD_ZEROHOLD_BACKWARD_WEIGHT = -6.0


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

  target_slew_limit: float | None = None
  """Optional per-step clamp for final wheel target changes in rad/s."""

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
    self._prev_processed_actions = torch.zeros_like(self._processed_actions)
    self._coupled_scale = float(cfg.scale)
    self._target_slew_limit = cfg.target_slew_limit
    if self._target_slew_limit is not None and self._target_slew_limit <= 0.0:
      raise ValueError(
        "CoupledWheelVelocityAction target_slew_limit must be positive, "
        f"got {self._target_slew_limit}."
      )

  def process_actions(self, actions: torch.Tensor):
    if actions.shape[-1] != 1:
      raise ValueError(
        "CoupledWheelVelocityAction expects action dimension 1, "
        f"got action shape {tuple(actions.shape)}."
      )

    raw = torch.clamp(actions[:, 0], -WHEEL_ACTION_CLIP, WHEEL_ACTION_CLIP)
    self._raw_actions[:, 0] = raw
    u = raw * self._coupled_scale
    target = torch.zeros_like(self._processed_actions)
    target[:, self._left_idx] = -u
    target[:, self._right_idx] = u
    self._prev_processed_actions[:] = self._processed_actions
    if self._target_slew_limit is not None:
      delta = torch.clamp(
        target - self._processed_actions,
        -self._target_slew_limit,
        self._target_slew_limit,
      )
      target = self._processed_actions + delta
    self._processed_actions[:] = target

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._processed_actions[env_ids] = 0.0
    self._prev_processed_actions[env_ids] = 0.0


@dataclass(kw_only=True)
class CommandFeedforwardCoupledWheelVelocityActionCfg(CoupledWheelVelocityActionCfg):
  """One-dimensional wheel action with a command-derived velocity bias."""

  command_name: str = "twist"
  command_gain: float = SLOW_SPEED_FEEDFORWARD_GAIN
  feedforward_clip: float = SLOW_SPEED_FEEDFORWARD_CLIP
  residual_scale: float = SLOW_SPEED_RESIDUAL_ACTION_SCALE

  def build(self, env: ManagerBasedRlEnv) -> "CommandFeedforwardCoupledWheelVelocityAction":
    return CommandFeedforwardCoupledWheelVelocityAction(self, env)


class CommandFeedforwardCoupledWheelVelocityAction(CoupledWheelVelocityAction):
  """Bias wheel balance action by commanded x velocity.

  The learned policy output remains a residual around the command bias. Negative
  x velocity commands need positive raw wheel action for this robot's mirrored
  wheel axes, hence ``feedforward = -cmd_lin_x * command_gain``.
  """

  def __init__(
    self,
    cfg: CommandFeedforwardCoupledWheelVelocityActionCfg,
    env: ManagerBasedRlEnv,
  ):
    super().__init__(cfg=cfg, env=env)
    self._command_env = env
    self._command_name = cfg.command_name
    self._command_gain = float(cfg.command_gain)
    self._feedforward_clip = float(cfg.feedforward_clip)
    self._residual_scale = float(cfg.residual_scale)
    if self._feedforward_clip < 0.0:
      raise ValueError("feedforward_clip must be non-negative.")
    if self._residual_scale <= 0.0:
      raise ValueError("residual_scale must be positive.")
    self._residual_actions = torch.zeros_like(self._raw_actions)
    self._feedforward_actions = torch.zeros_like(self._raw_actions)

  def process_actions(self, actions: torch.Tensor):
    if actions.shape[-1] != 1:
      raise ValueError(
        "CommandFeedforwardCoupledWheelVelocityAction expects action dimension 1, "
        f"got action shape {tuple(actions.shape)}."
      )

    command = self._command_env.command_manager.get_command(self._command_name)
    assert command is not None, f"Command '{self._command_name}' not found."
    residual = (
      torch.clamp(actions[:, 0], -WHEEL_ACTION_CLIP, WHEEL_ACTION_CLIP)
      * self._residual_scale
    )
    feedforward = torch.clamp(
      -command[:, 0] * self._command_gain,
      -self._feedforward_clip,
      self._feedforward_clip,
    )
    raw = torch.clamp(residual + feedforward, -WHEEL_ACTION_CLIP, WHEEL_ACTION_CLIP)

    self._residual_actions[:, 0] = residual
    self._feedforward_actions[:, 0] = feedforward
    self._raw_actions[:, 0] = raw
    u = raw * self._coupled_scale
    target = torch.zeros_like(self._processed_actions)
    target[:, self._left_idx] = -u
    target[:, self._right_idx] = u
    self._prev_processed_actions[:] = self._processed_actions
    if self._target_slew_limit is not None:
      delta = torch.clamp(
        target - self._processed_actions,
        -self._target_slew_limit,
        self._target_slew_limit,
      )
      target = self._processed_actions + delta
    self._processed_actions[:] = target

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    self._residual_actions[env_ids] = 0.0
    self._feedforward_actions[env_ids] = 0.0


@dataclass(kw_only=True)
class DifferentialWheelVelocityActionCfg(JointVelocityActionCfg):
  """Two-dimensional wheel velocity action for pitch balance plus yaw."""

  yaw_scale: float | None = None
  """Optional scale for the yaw action channel. Defaults to ``scale``."""

  yaw_smoothing_alpha: float | None = None
  """EMA previous-action weight for yaw. ``None`` disables smoothing."""

  target_slew_limit: float | None = None
  """Optional per-step clamp for final wheel target changes in rad/s."""

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
    self._prev_processed_actions = torch.zeros_like(self._processed_actions)
    self._balance_scale = float(cfg.scale)
    self._yaw_scale = float(cfg.scale if cfg.yaw_scale is None else cfg.yaw_scale)
    self._yaw_smoothing_alpha = cfg.yaw_smoothing_alpha
    if self._yaw_smoothing_alpha is not None and not (
      0.0 <= self._yaw_smoothing_alpha < 1.0
    ):
      raise ValueError(
        "DifferentialWheelVelocityAction yaw_smoothing_alpha must be in [0, 1), "
        f"got {self._yaw_smoothing_alpha}."
      )
    self._target_slew_limit = cfg.target_slew_limit
    if self._target_slew_limit is not None and self._target_slew_limit <= 0.0:
      raise ValueError(
        "DifferentialWheelVelocityAction target_slew_limit must be positive, "
        f"got {self._target_slew_limit}."
      )
    self._smoothed_yaw_action = torch.zeros(self.num_envs, device=self.device)
    self._prev_smoothed_yaw_action = torch.zeros(self.num_envs, device=self.device)

  def process_actions(self, actions: torch.Tensor):
    if actions.shape[-1] != 2:
      raise ValueError(
        "DifferentialWheelVelocityAction expects action dimension 2, "
        f"got action shape {tuple(actions.shape)}."
      )

    raw = torch.clamp(actions[:, :2], -WHEEL_ACTION_CLIP, WHEEL_ACTION_CLIP)
    self._raw_actions[:, :] = raw
    balance = raw[:, 0] * self._balance_scale
    yaw_action = raw[:, 1]
    if self._yaw_smoothing_alpha is not None:
      alpha = self._yaw_smoothing_alpha
      self._prev_smoothed_yaw_action[:] = self._smoothed_yaw_action
      self._smoothed_yaw_action[:] = (
        alpha * self._smoothed_yaw_action + (1.0 - alpha) * yaw_action
      )
      yaw_action = self._smoothed_yaw_action
    else:
      self._prev_smoothed_yaw_action[:] = yaw_action
    yaw = yaw_action * self._yaw_scale
    left = torch.clamp(-balance + yaw, -self._balance_scale, self._balance_scale)
    right = torch.clamp(balance + yaw, -self._balance_scale, self._balance_scale)
    target = torch.zeros_like(self._processed_actions)
    target[:, self._left_idx] = left
    target[:, self._right_idx] = right
    self._prev_processed_actions[:] = self._processed_actions
    if self._target_slew_limit is not None:
      delta = torch.clamp(
        target - self._processed_actions,
        -self._target_slew_limit,
        self._target_slew_limit,
      )
      target = self._processed_actions + delta
    self._processed_actions[:] = target

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._processed_actions[env_ids] = 0.0
    self._prev_processed_actions[env_ids] = 0.0
    self._smoothed_yaw_action[env_ids] = 0.0
    self._prev_smoothed_yaw_action[env_ids] = 0.0


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


@dataclass(kw_only=True)
class VariableYawSlowSpeedTurnCommandCfg(BinarySlowSpeedTurnCommandCfg):
  """Slow forward command with random yaw magnitude and explicit sign."""

  yaw_abs_range: tuple[float, float] = SLOW_SPEED_TURN_VARIABLE_YAW_ABS_RANGE

  def build(self, env: ManagerBasedRlEnv) -> "VariableYawSlowSpeedTurnCommand":
    return VariableYawSlowSpeedTurnCommand(self, env)


class VariableYawSlowSpeedTurnCommand(BinarySlowSpeedTurnCommand):
  cfg: VariableYawSlowSpeedTurnCommandCfg

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    yaw_mag = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, :] = 0.0
    self.vel_command_w[env_ids, :] = 0.0
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    signs = torch.where(
      torch.rand(len(env_ids), device=self.device) < 0.5,
      -1.0,
      1.0,
    )
    self.vel_command_b[env_ids, 2] = signs * yaw_mag.uniform_(
      *self.cfg.yaw_abs_range
    )
    self.is_heading_env[env_ids] = False
    self.is_standing_env[env_ids] = False
    self.is_world_env[env_ids] = False
    self.is_forward_env[env_ids] = False


@dataclass(kw_only=True)
class BidirBandVelocityCommandCfg(UniformVelocityCommandCfg):
  """Bidirectional linear command that avoids ambiguous near-zero speeds."""

  lin_vel_x_abs_range: tuple[float, float] = (
    SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_ACTIVE_LIN_VEL_X_ABS_RANGE
  )

  def build(self, env: ManagerBasedRlEnv) -> "BidirBandVelocityCommand":
    return BidirBandVelocityCommand(self, env)


class BidirBandVelocityCommand(UniformVelocityCommand):
  cfg: BidirBandVelocityCommandCfg

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    low, high = self.cfg.lin_vel_x_abs_range
    if not (0.0 < low <= high):
      raise ValueError(
        "BidirBandVelocityCommand lin_vel_x_abs_range must satisfy "
        f"0 < low <= high, got {self.cfg.lin_vel_x_abs_range}."
      )

    self.vel_command_b[env_ids, :] = 0.0
    self.vel_command_w[env_ids, :] = 0.0
    r = torch.empty(len(env_ids), device=self.device)
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
    self.is_heading_env[env_ids] = False
    self.is_world_env[env_ids] = False
    self.is_forward_env[env_ids] = False

    active_ids = env_ids[~self.is_standing_env[env_ids]]
    if len(active_ids) > 0:
      signs = torch.where(
        torch.rand(len(active_ids), device=self.device) < 0.5,
        -1.0,
        1.0,
      )
      magnitudes = torch.empty(len(active_ids), device=self.device).uniform_(low, high)
      self.vel_command_b[active_ids, 0] = signs * magnitudes
      self.vel_command_w[active_ids, 0] = self.vel_command_b[active_ids, 0]


def lin_vel_z_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = env.scene["robot"]
  return torch.square(robot.data.root_link_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = env.scene["robot"]
  return torch.sum(torch.square(robot.data.root_link_ang_vel_b[:, :2]), dim=1)


def lin_vel_xy_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = env.scene["robot"]
  return torch.sum(torch.square(robot.data.root_link_lin_vel_b[:, :2]), dim=1)


def _pitch_proxy_and_rate(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
  robot = env.scene["robot"]
  projected_gravity = robot.data.projected_gravity_b
  pitch_proxy = torch.atan2(
    projected_gravity[:, 0],
    torch.clamp(-projected_gravity[:, 2], min=1.0e-6),
  )
  pitch_rate = robot.data.root_link_ang_vel_b[:, 1]
  return pitch_proxy, pitch_rate


def _safe_pitch_posture(
  env: ManagerBasedRlEnv,
  pitch_abs_limit: float,
  pitch_rate_abs_limit: float,
) -> torch.Tensor:
  pitch_proxy, pitch_rate = _pitch_proxy_and_rate(env)
  return (
    (torch.abs(pitch_proxy) < float(pitch_abs_limit))
    & (torch.abs(pitch_rate) < float(pitch_rate_abs_limit))
  ).float()


def pitch_abs_above_limit_l2(
  env: ManagerBasedRlEnv,
  limit: float,
) -> torch.Tensor:
  pitch_proxy, _ = _pitch_proxy_and_rate(env)
  excess = torch.clamp(torch.abs(pitch_proxy) - float(limit), min=0.0)
  return torch.square(excess)


def pitch_rate_abs_above_limit_l2(
  env: ManagerBasedRlEnv,
  limit: float,
) -> torch.Tensor:
  _, pitch_rate = _pitch_proxy_and_rate(env)
  excess = torch.clamp(torch.abs(pitch_rate) - float(limit), min=0.0)
  return torch.square(excess)


def safe_posture_track_linear_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  pitch_abs_limit: float,
  pitch_rate_abs_limit: float,
) -> torch.Tensor:
  robot = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = robot.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  tracking = torch.exp(-(xy_error + z_error) / float(std) ** 2)
  safe = _safe_pitch_posture(
    env,
    pitch_abs_limit=pitch_abs_limit,
    pitch_rate_abs_limit=pitch_rate_abs_limit,
  )
  return safe * tracking


def unsafe_forward_velocity_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  deadband: float,
  pitch_abs_limit: float,
  pitch_rate_abs_limit: float,
) -> torch.Tensor:
  robot = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  active_forward_cmd = command[:, 0] > float(deadband)
  unsafe = 1.0 - _safe_pitch_posture(
    env,
    pitch_abs_limit=pitch_abs_limit,
    pitch_rate_abs_limit=pitch_rate_abs_limit,
  )
  forward_velocity = torch.clamp(robot.data.root_link_lin_vel_b[:, 0], min=0.0)
  penalty = unsafe * torch.square(forward_velocity)
  return torch.where(active_forward_cmd, penalty, torch.zeros_like(penalty))


def safe_support_track_linear_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  wheel_sensor_name: str,
  non_wheel_sensor_name: str,
  minimum_height: float,
  max_tilt_xy: float,
  pitch_rate_abs_limit: float,
) -> torch.Tensor:
  robot = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = robot.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  tracking = torch.exp(-(xy_error + z_error) / float(std) ** 2)
  _ = non_wheel_sensor_name
  safe = wheel_support_posture(
    env=env,
    wheel_sensor_name=wheel_sensor_name,
    minimum_height=minimum_height,
    max_tilt_xy=max_tilt_xy,
  )
  pitch_rate_ok = torch.abs(robot.data.root_link_ang_vel_b[:, 1]) < float(
    pitch_rate_abs_limit
  )
  return safe * pitch_rate_ok.float() * tracking


def unsafe_support_forward_velocity_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  deadband: float,
  wheel_sensor_name: str,
  non_wheel_sensor_name: str,
  minimum_height: float,
  max_tilt_xy: float,
  pitch_rate_abs_limit: float,
) -> torch.Tensor:
  robot = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  active_forward_cmd = command[:, 0] > float(deadband)
  _ = non_wheel_sensor_name
  safe = wheel_support_posture(
    env=env,
    wheel_sensor_name=wheel_sensor_name,
    minimum_height=minimum_height,
    max_tilt_xy=max_tilt_xy,
  )
  pitch_rate_ok = torch.abs(robot.data.root_link_ang_vel_b[:, 1]) < float(
    pitch_rate_abs_limit
  )
  safe = safe * pitch_rate_ok.float()
  forward_velocity = torch.clamp(robot.data.root_link_lin_vel_b[:, 0], min=0.0)
  penalty = (1.0 - safe) * torch.square(forward_velocity)
  return torch.where(active_forward_cmd, penalty, torch.zeros_like(penalty))


def backward_lin_vel_x_l2(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  robot = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  active = command[:, 0] > 0.01
  backward = torch.clamp(-robot.data.root_link_lin_vel_b[:, 0], min=0.0)
  return torch.where(active, torch.square(backward), torch.zeros_like(backward))


def forward_lin_vel_x_ratio_on_backward_command(
  env: ManagerBasedRlEnv,
  command_name: str,
  deadband: float,
) -> torch.Tensor:
  robot = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  cmd_lin_x = command[:, 0]
  actual_lin_x = robot.data.root_link_lin_vel_b[:, 0]
  active = cmd_lin_x < -deadband
  forward_ratio = torch.clamp(
    torch.clamp(actual_lin_x, min=0.0)
    / torch.clamp(torch.abs(cmd_lin_x), min=deadband),
    min=0.0,
    max=1.0,
  )
  return torch.where(active, forward_ratio, torch.zeros_like(forward_ratio))


def pitch_target_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  sign: float,
  gain: float,
  target_clip: float,
) -> torch.Tensor:
  robot = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  projected_gravity = robot.data.projected_gravity_b
  pitch_proxy = torch.atan2(
    projected_gravity[:, 0],
    torch.clamp(-projected_gravity[:, 2], min=1.0e-6),
  )
  target = torch.clamp(
    float(sign) * float(gain) * command[:, 0],
    min=-float(target_clip),
    max=float(target_clip),
  )
  return torch.square(pitch_proxy - target)


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


def wheel_support_posture(
  env: ManagerBasedRlEnv,
  wheel_sensor_name: str,
  minimum_height: float,
  max_tilt_xy: float,
) -> torch.Tensor:
  robot = env.scene["robot"]
  wheel_contact = _contact_all(env, wheel_sensor_name)
  root_ok = robot.data.root_link_pos_w[:, 2] > minimum_height
  tilt_xy = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=-1)
  tilt_ok = tilt_xy < max_tilt_xy
  return (wheel_contact & root_ok & tilt_ok).float()


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


def yaw_velocity_error_l2(
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
  error = torch.square(actual_yaw - cmd_yaw)
  return torch.where(active, error, torch.zeros_like(error))


def lin_vel_x_sign_alignment(
  env: ManagerBasedRlEnv,
  command_name: str,
  deadband: float,
) -> torch.Tensor:
  robot = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  cmd_lin_x = command[:, 0]
  actual_lin_x = robot.data.root_link_lin_vel_b[:, 0]
  active = torch.abs(cmd_lin_x) > deadband
  normalized = torch.clamp(
    (cmd_lin_x * actual_lin_x) / torch.clamp(torch.square(cmd_lin_x), min=deadband**2),
    min=-1.0,
    max=1.0,
  )
  return torch.where(active, normalized, torch.zeros_like(normalized))


def joint_pos_deviation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  robot = env.scene[asset_cfg.name]
  default_joint_pos = robot.data.default_joint_pos
  assert default_joint_pos is not None
  return torch.sum(
    torch.square(
      robot.data.joint_pos[:, asset_cfg.joint_ids]
      - default_joint_pos[:, asset_cfg.joint_ids]
    ),
    dim=1,
  )


def effective_yaw_rate_l2(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  action_term = env.action_manager.get_term(action_name)
  current = getattr(action_term, "_smoothed_yaw_action", None)
  previous = getattr(action_term, "_prev_smoothed_yaw_action", None)
  if current is None or previous is None:
    raise AttributeError(
      f"Action term '{action_name}' does not expose smoothed yaw action buffers."
    )
  return torch.square(current - previous)


def wheel_target_rate_l2(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  action_term = env.action_manager.get_term(action_name)
  current = getattr(action_term, "_processed_actions", None)
  previous = getattr(action_term, "_prev_processed_actions", None)
  if current is None or previous is None:
    raise AttributeError(
      f"Action term '{action_name}' does not expose processed action buffers."
    )
  return torch.sum(torch.square(current - previous), dim=1)


def stable_wheel_target_rate_l2(
  env: ManagerBasedRlEnv,
  action_name: str,
  wheel_sensor_name: str,
  non_wheel_sensor_name: str,
  minimum_height: float,
  max_tilt_xy: float,
) -> torch.Tensor:
  """Penalize target jumps only when the robot is already in a safe support state."""

  stable = clean_wheel_support(
    env=env,
    wheel_sensor_name=wheel_sensor_name,
    non_wheel_sensor_name=non_wheel_sensor_name,
    minimum_height=minimum_height,
    max_tilt_xy=max_tilt_xy,
  )
  return stable * wheel_target_rate_l2(env, action_name=action_name)


def zero_command_stable_wheel_target_rate_l2(
  env: ManagerBasedRlEnv,
  action_name: str,
  command_name: str,
  deadband: float,
  wheel_sensor_name: str,
  non_wheel_sensor_name: str,
  minimum_height: float,
  max_tilt_xy: float,
) -> torch.Tensor:
  """Penalize wheel target jumps only for near-zero commands."""

  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  zero_cmd = torch.all(torch.abs(command) <= deadband, dim=1).float()
  stable = clean_wheel_support(
    env=env,
    wheel_sensor_name=wheel_sensor_name,
    non_wheel_sensor_name=non_wheel_sensor_name,
    minimum_height=minimum_height,
    max_tilt_xy=max_tilt_xy,
  )
  return zero_cmd * stable * wheel_target_rate_l2(env, action_name=action_name)


def scaled_velocity_commands(
  env: ManagerBasedRlEnv,
  command_name: str,
  scale: tuple[float, float, float],
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  scale_tensor = torch.tensor(scale, device=command.device, dtype=command.dtype)
  return command * scale_tensor


def joint_pos_rel_without_wheel_position(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  wheel_joint_names: tuple[str, ...],
) -> torch.Tensor:
  joint_pos = envs_mdp.joint_pos_rel(env, asset_cfg=asset_cfg)
  robot = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, slice):
    selected_joint_ids = list(range(len(robot.joint_names)))[joint_ids]
  elif isinstance(joint_ids, torch.Tensor):
    selected_joint_ids = joint_ids.detach().cpu().tolist()
  else:
    selected_joint_ids = list(joint_ids)

  joint_pos = joint_pos.clone()
  for obs_index, joint_id in enumerate(selected_joint_ids):
    if robot.joint_names[joint_id] in wheel_joint_names:
      joint_pos[:, obs_index] = 0.0
  return joint_pos


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
  slow_speed_lin_sign: bool = False,
  slow_speed_obs_scale: bool = False,
  slow_speed_forward_only: bool = False,
  slow_speed_backward_only: bool = False,
  slow_speed_backward_strict: bool = False,
  slow_speed_command_feedforward: bool = False,
  slow_speed_command_feedforward_low_residual: bool = False,
  slow_speed_pitch_target_pos: bool = False,
  slow_speed_pitch_target_neg: bool = False,
  limited_leg_assist: bool = False,
  limited_leg_assist_safe: bool = False,
  slow_speed_turn: bool = False,
  slow_speed_turn_sign: bool = False,
  slow_speed_turn_obs_scale: bool = False,
  slow_speed_turn_safe: bool = False,
  slow_speed_turn_safe_v2: bool = False,
  slow_speed_turn_safe_v2_yaw_scale3: bool = False,
  slow_speed_turn_safe_v2_yaw_scale2p5: bool = False,
  slow_speed_turn_safe_v2_yaw_smooth: bool = False,
  slow_speed_turn_safe_v2_yaw_smooth_v2: bool = False,
  slow_speed_turn_safe_v2_wheel_rate: bool = False,
  slow_speed_turn_low_forward: bool = False,
  slow_speed_turn_mid_forward: bool = False,
  slow_speed_turn_bidirectional: bool = False,
  slow_speed_turn_bidirectional_low_yaw: bool = False,
  slow_speed_turn_bidirectional_lin_sign: bool = False,
  slow_speed_turn_bidirectional_lin_sign_strong: bool = False,
  slow_speed_turn_stable_rate: bool = False,
  slow_speed_turn_target_slew: bool = False,
  slow_speed_turn_variable_yaw: bool = False,
  slow_speed_turn_no_backward: bool = False,
  slow_speed_turn_push: bool = False,
  turn_l4: bool = False,
  turn_level: int = 1,
  scratch_stage0_stable: bool = False,
  scratch_stable_regularization: bool = False,
  scratch_stage1_clear_forward: bool = False,
  scratch_stage1_forward_only_clear: bool = False,
  scratch_stage1_forward_nospike: bool = False,
  scratch_stage1_forward_nospike_strong: bool = False,
  scratch_stage1_forward_smooth_slew12: bool = False,
  scratch_stage1_forward_smooth_slew12_norev: bool = False,
  scratch_stage2_bidir_smooth_slew12: bool = False,
  scratch_stage1_forward_guarded: bool = False,
  scratch_stage1_forward_support_guarded: bool = False,
  scratch_stage1_gentle_forward: bool = False,
  scratch_stage1_gentle_forward_slew6: bool = False,
  scratch_stage1_gentle_forward_zero_hold: bool = False,
  training_episode_length_s: float | None = None,
  command_resampling_time_range: tuple[float, float] = (5.0, 10.0),
  zero_wheel_joint_pos_obs: bool = False,
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
  wheel_yaw_smoothing_alpha: float | None = None
  wheel_target_slew_limit: float | None = None
  binary_yaw_command = False
  binary_yaw_abs = TURN_L4_SIGN_YAW_ABS
  binary_slow_speed_turn_command = False
  binary_slow_speed_turn_yaw_abs = SLOW_SPEED_TURN_ANG_VEL_Z_RANGE
  variable_yaw_slow_speed_turn_command = False
  bidir_band_velocity_command = False
  yaw_sign_reward = False
  track_lin_vel_weight = SLOW_SPEED_TRACK_LIN_VEL_WEIGHT
  track_lin_vel_std = SLOW_SPEED_TRACK_LIN_VEL_STD
  track_ang_vel_weight = TURN_L4_ANG_VEL_WEIGHT
  track_ang_vel_std = TURN_L4_ANG_VEL_STD
  yaw_velocity_error_weight: float | None = None
  clean_wheel_support_weight = 4.0
  wheel_ground_contact_weight = 1.0
  non_wheel_ground_contact_weight = -6.0
  yaw_sign_weight = TURN_L4_SIGN_YAW_WEIGHT
  lin_sign_weight = SLOW_SPEED_TURN_BIDIRECTIONAL_LIN_SIGN_WEIGHT
  slow_speed_lin_sign_weight = SLOW_SPEED_LIN_SIGN_WEIGHT
  slow_speed_pitch_target_sign: float | None = None
  slow_speed_residual_action_scale = SLOW_SPEED_RESIDUAL_ACTION_SCALE
  effective_yaw_rate_weight: float | None = None
  wheel_target_rate_weight: float | None = None
  stable_wheel_target_rate_weight: float | None = None
  zero_command_stable_wheel_target_rate_weight: float | None = None
  forward_backward_lin_vel_weight: float | None = None
  safe_posture_track_lin_vel_weight: float | None = None
  safe_posture_track_lin_vel_std: float | None = None
  unsafe_forward_velocity_weight: float | None = None
  safe_support_track_lin_vel_weight: float | None = None
  safe_support_track_lin_vel_std: float | None = None
  unsafe_support_forward_velocity_weight: float | None = None
  pitch_abs_tail_weight: float | None = None
  pitch_abs_tail_limit: float | None = None
  pitch_rate_abs_tail_weight: float | None = None
  pitch_rate_abs_tail_limit: float | None = None
  leg_joint_pos_weight: float | None = None
  leg_joint_vel_weight: float | None = None
  leg_action_scale: float | None = None
  command_obs_func = envs_mdp.generated_commands
  command_obs_params: dict[str, object] = {"command_name": "twist"}
  use_differential_wheel_action = turn_l4 or slow_speed_turn
  if scratch_stage0_stable:
    lin_vel_xy_penalty_weight = SCRATCH_STAGE0_STABLE_LIN_VEL_XY_WEIGHT
    wheel_vel_penalty_weight = SCRATCH_STAGE0_STABLE_WHEEL_VEL_WEIGHT
    action_rate_penalty_weight = SCRATCH_STAGE0_STABLE_ACTION_RATE_WEIGHT
  if limited_leg_assist:
    leg_action_scale = LIMITED_LEG_ASSIST_ACTION_SCALE
    leg_joint_pos_weight = LIMITED_LEG_ASSIST_JOINT_POS_WEIGHT
    leg_joint_vel_weight = LIMITED_LEG_ASSIST_JOINT_VEL_WEIGHT
  if limited_leg_assist_safe:
    leg_action_scale = LIMITED_LEG_ASSIST_SAFE_ACTION_SCALE
    leg_joint_pos_weight = LIMITED_LEG_ASSIST_SAFE_JOINT_POS_WEIGHT
    leg_joint_vel_weight = LIMITED_LEG_ASSIST_SAFE_JOINT_VEL_WEIGHT
  if slow_speed:
    if slow_speed_obs_scale:
      command_obs_func = scaled_velocity_commands
      command_obs_params = {
        "command_name": "twist",
        "scale": SLOW_SPEED_OBS_COMMAND_SCALE,
      }
    if speed_level == 0:
      command_lin_vel_x_range = (
        -SLOW_SPEED_EASY_LIN_VEL_X_RANGE,
        SLOW_SPEED_EASY_LIN_VEL_X_RANGE,
      )
      rel_standing_envs = SLOW_SPEED_EASY_STANDING_ENVS
      lin_vel_xy_penalty_weight = SLOW_SPEED_EASY_LIN_VEL_XY_PENALTY_WEIGHT
      track_lin_vel_weight = SLOW_SPEED_EASY_TRACK_LIN_VEL_WEIGHT
      track_lin_vel_std = SLOW_SPEED_EASY_TRACK_LIN_VEL_STD
      if slow_speed_forward_only:
        command_lin_vel_x_range = SLOW_SPEED_EASY_FORWARD_ONLY_LIN_VEL_X_RANGE
        rel_standing_envs = 0.0
        if scratch_stage1_clear_forward:
          command_lin_vel_x_range = SCRATCH_STAGE1_CLEAR_FORWARD_LIN_VEL_X_RANGE
          rel_standing_envs = SCRATCH_STAGE1_CLEAR_FORWARD_STANDING_ENVS
          track_lin_vel_weight = SCRATCH_STAGE1_CLEAR_FORWARD_TRACK_LIN_VEL_WEIGHT
          track_lin_vel_std = SCRATCH_STAGE1_CLEAR_FORWARD_TRACK_LIN_VEL_STD
          slow_speed_lin_sign_weight = SCRATCH_STAGE1_CLEAR_FORWARD_LIN_SIGN_WEIGHT
          lin_vel_xy_penalty_weight = SCRATCH_STAGE1_CLEAR_FORWARD_LIN_VEL_XY_WEIGHT
          wheel_vel_penalty_weight = SCRATCH_STAGE0_STABLE_WHEEL_VEL_WEIGHT
          action_rate_penalty_weight = SCRATCH_STAGE0_STABLE_ACTION_RATE_WEIGHT
        if scratch_stage1_forward_only_clear:
          command_lin_vel_x_range = SCRATCH_STAGE1_FORWARD_ONLY_CLEAR_LIN_VEL_X_RANGE
          rel_standing_envs = SCRATCH_STAGE1_FORWARD_ONLY_CLEAR_STANDING_ENVS
          track_lin_vel_weight = SCRATCH_STAGE1_FORWARD_ONLY_CLEAR_TRACK_LIN_VEL_WEIGHT
          track_lin_vel_std = SCRATCH_STAGE1_FORWARD_ONLY_CLEAR_TRACK_LIN_VEL_STD
          slow_speed_lin_sign_weight = SCRATCH_STAGE1_FORWARD_ONLY_CLEAR_LIN_SIGN_WEIGHT
          lin_vel_xy_penalty_weight = SCRATCH_STAGE1_CLEAR_FORWARD_LIN_VEL_XY_WEIGHT
          wheel_vel_penalty_weight = SCRATCH_STAGE0_STABLE_WHEEL_VEL_WEIGHT
          action_rate_penalty_weight = SCRATCH_STAGE0_STABLE_ACTION_RATE_WEIGHT
        if scratch_stage1_forward_nospike:
          command_lin_vel_x_range = SCRATCH_STAGE1_FORWARD_NOSPIKE_LIN_VEL_X_RANGE
          rel_standing_envs = SCRATCH_STAGE1_FORWARD_NOSPIKE_STANDING_ENVS
          track_lin_vel_weight = SCRATCH_STAGE1_FORWARD_NOSPIKE_TRACK_LIN_VEL_WEIGHT
          track_lin_vel_std = SCRATCH_STAGE1_FORWARD_NOSPIKE_TRACK_LIN_VEL_STD
          slow_speed_lin_sign_weight = SCRATCH_STAGE1_FORWARD_NOSPIKE_LIN_SIGN_WEIGHT
          lin_vel_xy_penalty_weight = SCRATCH_STAGE1_CLEAR_FORWARD_LIN_VEL_XY_WEIGHT
          wheel_vel_penalty_weight = SCRATCH_STAGE0_STABLE_WHEEL_VEL_WEIGHT
          action_rate_penalty_weight = SCRATCH_STAGE1_FORWARD_NOSPIKE_ACTION_RATE_WEIGHT
          wheel_target_rate_weight = (
            SCRATCH_STAGE1_FORWARD_NOSPIKE_WHEEL_TARGET_RATE_WEIGHT
          )
          pitch_abs_tail_weight = SCRATCH_STAGE1_FORWARD_NOSPIKE_PITCH_TAIL_WEIGHT
          pitch_abs_tail_limit = SCRATCH_STAGE1_FORWARD_NOSPIKE_PITCH_TAIL_LIMIT
          pitch_rate_abs_tail_weight = (
            SCRATCH_STAGE1_FORWARD_NOSPIKE_PITCH_RATE_TAIL_WEIGHT
          )
          pitch_rate_abs_tail_limit = (
            SCRATCH_STAGE1_FORWARD_NOSPIKE_PITCH_RATE_TAIL_LIMIT
          )
        if scratch_stage1_forward_nospike_strong:
          command_lin_vel_x_range = (
            SCRATCH_STAGE1_FORWARD_NOSPIKE_STRONG_LIN_VEL_X_RANGE
          )
          rel_standing_envs = SCRATCH_STAGE1_FORWARD_NOSPIKE_STRONG_STANDING_ENVS
          track_lin_vel_weight = (
            SCRATCH_STAGE1_FORWARD_NOSPIKE_STRONG_TRACK_LIN_VEL_WEIGHT
          )
          track_lin_vel_std = (
            SCRATCH_STAGE1_FORWARD_NOSPIKE_STRONG_TRACK_LIN_VEL_STD
          )
          slow_speed_lin_sign_weight = (
            SCRATCH_STAGE1_FORWARD_NOSPIKE_STRONG_LIN_SIGN_WEIGHT
          )
          lin_vel_xy_penalty_weight = SCRATCH_STAGE1_CLEAR_FORWARD_LIN_VEL_XY_WEIGHT
          wheel_vel_penalty_weight = SCRATCH_STAGE0_STABLE_WHEEL_VEL_WEIGHT
          action_rate_penalty_weight = (
            SCRATCH_STAGE1_FORWARD_NOSPIKE_STRONG_ACTION_RATE_WEIGHT
          )
          wheel_target_rate_weight = (
            SCRATCH_STAGE1_FORWARD_NOSPIKE_STRONG_WHEEL_TARGET_RATE_WEIGHT
          )
          pitch_abs_tail_weight = (
            SCRATCH_STAGE1_FORWARD_NOSPIKE_STRONG_PITCH_TAIL_WEIGHT
          )
          pitch_abs_tail_limit = (
            SCRATCH_STAGE1_FORWARD_NOSPIKE_STRONG_PITCH_TAIL_LIMIT
          )
          pitch_rate_abs_tail_weight = (
            SCRATCH_STAGE1_FORWARD_NOSPIKE_STRONG_PITCH_RATE_TAIL_WEIGHT
          )
          pitch_rate_abs_tail_limit = (
            SCRATCH_STAGE1_FORWARD_NOSPIKE_STRONG_PITCH_RATE_TAIL_LIMIT
          )
        if scratch_stage1_forward_smooth_slew12:
          command_lin_vel_x_range = (
            SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_LIN_VEL_X_RANGE
          )
          rel_standing_envs = SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_STANDING_ENVS
          track_lin_vel_weight = (
            SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_TRACK_LIN_VEL_WEIGHT
          )
          track_lin_vel_std = SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_TRACK_LIN_VEL_STD
          slow_speed_lin_sign_weight = (
            SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_LIN_SIGN_WEIGHT
          )
          lin_vel_xy_penalty_weight = SCRATCH_STAGE1_CLEAR_FORWARD_LIN_VEL_XY_WEIGHT
          wheel_vel_penalty_weight = SCRATCH_STAGE0_STABLE_WHEEL_VEL_WEIGHT
          action_rate_penalty_weight = (
            SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_ACTION_RATE_WEIGHT
          )
          wheel_target_rate_weight = (
            SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_WHEEL_TARGET_RATE_WEIGHT
          )
          wheel_target_slew_limit = (
            SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_TARGET_SLEW_LIMIT
          )
          pitch_abs_tail_weight = (
            SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_PITCH_TAIL_WEIGHT
          )
          pitch_abs_tail_limit = SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_PITCH_TAIL_LIMIT
          pitch_rate_abs_tail_weight = (
            SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_PITCH_RATE_TAIL_WEIGHT
          )
          pitch_rate_abs_tail_limit = (
            SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_PITCH_RATE_TAIL_LIMIT
          )
        if scratch_stage1_forward_smooth_slew12_norev:
          command_lin_vel_x_range = (
            SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_LIN_VEL_X_RANGE
          )
          rel_standing_envs = SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_NOREV_STANDING_ENVS
          track_lin_vel_weight = (
            SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_NOREV_TRACK_LIN_VEL_WEIGHT
          )
          track_lin_vel_std = (
            SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_NOREV_TRACK_LIN_VEL_STD
          )
          slow_speed_lin_sign_weight = (
            SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_NOREV_LIN_SIGN_WEIGHT
          )
          lin_vel_xy_penalty_weight = SCRATCH_STAGE1_CLEAR_FORWARD_LIN_VEL_XY_WEIGHT
          wheel_vel_penalty_weight = SCRATCH_STAGE0_STABLE_WHEEL_VEL_WEIGHT
          action_rate_penalty_weight = (
            SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_ACTION_RATE_WEIGHT
          )
          wheel_target_rate_weight = (
            SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_WHEEL_TARGET_RATE_WEIGHT
          )
          wheel_target_slew_limit = (
            SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_TARGET_SLEW_LIMIT
          )
          pitch_abs_tail_weight = (
            SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_PITCH_TAIL_WEIGHT
          )
          pitch_abs_tail_limit = SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_PITCH_TAIL_LIMIT
          pitch_rate_abs_tail_weight = (
            SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_PITCH_RATE_TAIL_WEIGHT
          )
          pitch_rate_abs_tail_limit = (
            SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_PITCH_RATE_TAIL_LIMIT
          )
          forward_backward_lin_vel_weight = (
            SCRATCH_STAGE1_FORWARD_SMOOTH_SLEW12_NOREV_BACKWARD_WEIGHT
          )
        if scratch_stage1_forward_guarded:
          command_lin_vel_x_range = SCRATCH_STAGE1_FORWARD_GUARDED_LIN_VEL_X_RANGE
          rel_standing_envs = SCRATCH_STAGE1_FORWARD_GUARDED_STANDING_ENVS
          track_lin_vel_weight = 0.0
          safe_posture_track_lin_vel_weight = (
            SCRATCH_STAGE1_FORWARD_GUARDED_TRACK_LIN_VEL_WEIGHT
          )
          safe_posture_track_lin_vel_std = (
            SCRATCH_STAGE1_FORWARD_GUARDED_TRACK_LIN_VEL_STD
          )
          slow_speed_lin_sign_weight = SCRATCH_STAGE1_FORWARD_GUARDED_LIN_SIGN_WEIGHT
          unsafe_forward_velocity_weight = (
            SCRATCH_STAGE1_FORWARD_GUARDED_UNSAFE_FORWARD_WEIGHT
          )
          lin_vel_xy_penalty_weight = SCRATCH_STAGE1_CLEAR_FORWARD_LIN_VEL_XY_WEIGHT
          wheel_vel_penalty_weight = SCRATCH_STAGE0_STABLE_WHEEL_VEL_WEIGHT
          action_rate_penalty_weight = SCRATCH_STAGE0_STABLE_ACTION_RATE_WEIGHT
        if scratch_stage1_forward_support_guarded:
          command_lin_vel_x_range = (
            SCRATCH_STAGE1_FORWARD_SUPPORT_GUARDED_LIN_VEL_X_RANGE
          )
          rel_standing_envs = SCRATCH_STAGE1_FORWARD_SUPPORT_GUARDED_STANDING_ENVS
          track_lin_vel_weight = 0.0
          safe_support_track_lin_vel_weight = (
            SCRATCH_STAGE1_FORWARD_SUPPORT_GUARDED_TRACK_LIN_VEL_WEIGHT
          )
          safe_support_track_lin_vel_std = (
            SCRATCH_STAGE1_FORWARD_SUPPORT_GUARDED_TRACK_LIN_VEL_STD
          )
          slow_speed_lin_sign_weight = (
            SCRATCH_STAGE1_FORWARD_SUPPORT_GUARDED_LIN_SIGN_WEIGHT
          )
          unsafe_support_forward_velocity_weight = (
            SCRATCH_STAGE1_FORWARD_SUPPORT_GUARDED_UNSAFE_FORWARD_WEIGHT
          )
          lin_vel_xy_penalty_weight = SCRATCH_STAGE1_CLEAR_FORWARD_LIN_VEL_XY_WEIGHT
          wheel_vel_penalty_weight = SCRATCH_STAGE0_STABLE_WHEEL_VEL_WEIGHT
          action_rate_penalty_weight = SCRATCH_STAGE0_STABLE_ACTION_RATE_WEIGHT
        if (
          scratch_stage1_gentle_forward
          or scratch_stage1_gentle_forward_slew6
          or scratch_stage1_gentle_forward_zero_hold
        ):
          command_lin_vel_x_range = SCRATCH_STAGE1_GENTLE_FORWARD_LIN_VEL_X_RANGE
          rel_standing_envs = SCRATCH_STAGE1_GENTLE_FORWARD_STANDING_ENVS
          track_lin_vel_weight = SCRATCH_STAGE1_GENTLE_FORWARD_TRACK_LIN_VEL_WEIGHT
          track_lin_vel_std = SCRATCH_STAGE1_GENTLE_FORWARD_TRACK_LIN_VEL_STD
          slow_speed_lin_sign_weight = SCRATCH_STAGE1_GENTLE_FORWARD_LIN_SIGN_WEIGHT
          lin_vel_xy_penalty_weight = SCRATCH_STAGE1_CLEAR_FORWARD_LIN_VEL_XY_WEIGHT
          wheel_vel_penalty_weight = SCRATCH_STAGE0_STABLE_WHEEL_VEL_WEIGHT
          action_rate_penalty_weight = SCRATCH_STAGE0_STABLE_ACTION_RATE_WEIGHT
          stable_wheel_target_rate_weight = (
            SCRATCH_STAGE1_GENTLE_FORWARD_STABLE_WHEEL_TARGET_RATE_WEIGHT
          )
          if scratch_stage1_gentle_forward_slew6:
            wheel_target_slew_limit = (
              SCRATCH_STAGE1_GENTLE_FORWARD_SLEW6_TARGET_SLEW_LIMIT
            )
          if scratch_stage1_gentle_forward_zero_hold:
            action_rate_penalty_weight = (
              SCRATCH_STAGE1_GENTLE_FORWARD_ZEROHOLD_ACTION_RATE_WEIGHT
            )
            stable_wheel_target_rate_weight = None
            zero_command_stable_wheel_target_rate_weight = (
              SCRATCH_STAGE1_GENTLE_FORWARD_ZEROHOLD_ZERO_CMD_RATE_WEIGHT
            )
            forward_backward_lin_vel_weight = (
              SCRATCH_STAGE1_GENTLE_FORWARD_ZEROHOLD_BACKWARD_WEIGHT
            )
      if slow_speed_backward_only:
        command_lin_vel_x_range = SLOW_SPEED_EASY_BACKWARD_ONLY_LIN_VEL_X_RANGE
        rel_standing_envs = 0.0
      if slow_speed_backward_strict:
        command_lin_vel_x_range = SLOW_SPEED_BACKWARD_STRICT_LIN_VEL_X_RANGE
        rel_standing_envs = 0.0
        track_lin_vel_weight = SLOW_SPEED_BACKWARD_STRICT_TRACK_LIN_VEL_WEIGHT
        track_lin_vel_std = SLOW_SPEED_BACKWARD_STRICT_TRACK_LIN_VEL_STD
        slow_speed_lin_sign_weight = SLOW_SPEED_BACKWARD_STRICT_LIN_SIGN_WEIGHT
      if scratch_stage2_bidir_smooth_slew12:
        command_lin_vel_x_range = (
          SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_LIN_VEL_X_RANGE
        )
        bidir_band_velocity_command = True
        rel_standing_envs = SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_STANDING_ENVS
        track_lin_vel_weight = (
          SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_TRACK_LIN_VEL_WEIGHT
        )
        track_lin_vel_std = SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_TRACK_LIN_VEL_STD
        slow_speed_lin_sign_weight = (
          SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_LIN_SIGN_WEIGHT
        )
        lin_vel_xy_penalty_weight = SCRATCH_STAGE1_CLEAR_FORWARD_LIN_VEL_XY_WEIGHT
        wheel_vel_penalty_weight = SCRATCH_STAGE0_STABLE_WHEEL_VEL_WEIGHT
        action_rate_penalty_weight = (
          SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_ACTION_RATE_WEIGHT
        )
        wheel_target_rate_weight = (
          SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_WHEEL_TARGET_RATE_WEIGHT
        )
        wheel_target_slew_limit = (
          SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_TARGET_SLEW_LIMIT
        )
        pitch_abs_tail_weight = (
          SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_PITCH_TAIL_WEIGHT
        )
        pitch_abs_tail_limit = (
          SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_PITCH_TAIL_LIMIT
        )
        pitch_rate_abs_tail_weight = (
          SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_PITCH_RATE_TAIL_WEIGHT
        )
        pitch_rate_abs_tail_limit = (
          SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_PITCH_RATE_TAIL_LIMIT
        )
      if slow_speed_command_feedforward_low_residual:
        slow_speed_residual_action_scale = SLOW_SPEED_LOW_RESIDUAL_ACTION_SCALE
      if slow_speed_pitch_target_pos:
        slow_speed_pitch_target_sign = 1.0
      if slow_speed_pitch_target_neg:
        slow_speed_pitch_target_sign = -1.0
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
      yaw_sign_weight = SLOW_SPEED_TURN_SIGN_YAW_WEIGHT
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
    if slow_speed_turn_safe_v2_yaw_scale2p5:
      wheel_yaw_scale = SLOW_SPEED_TURN_SAFE_V2_YAW_SCALE_2P5
    if slow_speed_turn_safe_v2_yaw_smooth:
      wheel_yaw_smoothing_alpha = SLOW_SPEED_TURN_SAFE_V2_YAW_SMOOTHING_ALPHA
    if slow_speed_turn_safe_v2_yaw_smooth_v2:
      effective_yaw_rate_weight = SLOW_SPEED_TURN_SAFE_V2_EFFECTIVE_YAW_RATE_WEIGHT
    if slow_speed_turn_safe_v2_wheel_rate:
      wheel_target_rate_weight = SLOW_SPEED_TURN_SAFE_V2_WHEEL_TARGET_RATE_WEIGHT
    if slow_speed_turn_low_forward:
      command_lin_vel_x_range = SLOW_SPEED_TURN_LOW_FORWARD_LIN_VEL_X_RANGE
    if slow_speed_turn_mid_forward:
      command_lin_vel_x_range = SLOW_SPEED_TURN_MID_FORWARD_LIN_VEL_X_RANGE
    if slow_speed_turn_bidirectional:
      command_lin_vel_x_range = SLOW_SPEED_TURN_BIDIRECTIONAL_LIN_VEL_X_RANGE
    if slow_speed_turn_bidirectional_low_yaw:
      command_ang_vel_z_range = (
        -SLOW_SPEED_TURN_BIDIRECTIONAL_LOW_YAW_RANGE,
        SLOW_SPEED_TURN_BIDIRECTIONAL_LOW_YAW_RANGE,
      )
      binary_slow_speed_turn_yaw_abs = SLOW_SPEED_TURN_BIDIRECTIONAL_LOW_YAW_RANGE
    if slow_speed_turn_bidirectional_lin_sign_strong:
      lin_sign_weight = SLOW_SPEED_TURN_BIDIRECTIONAL_LIN_SIGN_STRONG_WEIGHT
    if slow_speed_turn_stable_rate:
      stable_wheel_target_rate_weight = (
        SLOW_SPEED_TURN_SAFE_V2_STABLE_WHEEL_TARGET_RATE_WEIGHT
      )
    if slow_speed_turn_target_slew:
      wheel_target_slew_limit = SLOW_SPEED_TURN_SAFE_V2_TARGET_SLEW_LIMIT
    if slow_speed_turn_variable_yaw:
      variable_yaw_slow_speed_turn_command = True
      binary_slow_speed_turn_command = False
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
    elif turn_level == 7:
      command_ang_vel_z_range = (
        -TURN_L4_SIGN_YAW_ABS,
        TURN_L4_SIGN_YAW_ABS,
      )
      rel_standing_envs = 0.0
      track_ang_vel_weight = TURN_L4_SIGN_STRONG_ANG_VEL_WEIGHT
      track_ang_vel_std = TURN_L4_SIGN_STRONG_ANG_VEL_STD
      lin_vel_xy_penalty_weight = TURN_L4_EASY_LIN_VEL_XY_PENALTY_WEIGHT
      wheel_vel_penalty_weight = TURN_L4_EASY_WHEEL_VEL_PENALTY_WEIGHT
      action_rate_penalty_weight = TURN_L4_EASY_ACTION_RATE_PENALTY_WEIGHT
      wheel_yaw_scale = TURN_L4_SIGN_STRONG_YAW_SCALE
      binary_yaw_command = True
      yaw_sign_reward = True
      yaw_sign_weight = TURN_L4_SIGN_STRONG_YAW_WEIGHT
    elif turn_level == 8:
      command_ang_vel_z_range = (
        -TURN_L4_SIGN_YAW_ABS,
        TURN_L4_SIGN_YAW_ABS,
      )
      rel_standing_envs = 0.0
      track_ang_vel_weight = TURN_L4_SIGN_MEDIUM_ANG_VEL_WEIGHT
      track_ang_vel_std = TURN_L4_SIGN_MEDIUM_ANG_VEL_STD
      lin_vel_xy_penalty_weight = TURN_L4_EASY_LIN_VEL_XY_PENALTY_WEIGHT
      wheel_vel_penalty_weight = TURN_L4_EASY_WHEEL_VEL_PENALTY_WEIGHT
      action_rate_penalty_weight = TURN_L4_EASY_ACTION_RATE_PENALTY_WEIGHT
      wheel_yaw_scale = TURN_L4_SIGN_MEDIUM_YAW_SCALE
      binary_yaw_command = True
      yaw_sign_reward = True
      yaw_sign_weight = TURN_L4_SIGN_MEDIUM_YAW_WEIGHT
    elif turn_level == 9:
      command_ang_vel_z_range = (
        -TURN_L4_SIGN_TRACK_YAW_ABS,
        TURN_L4_SIGN_TRACK_YAW_ABS,
      )
      rel_standing_envs = 0.0
      binary_yaw_abs = TURN_L4_SIGN_TRACK_YAW_ABS
      track_ang_vel_weight = TURN_L4_SIGN_TRACK_ANG_VEL_WEIGHT
      track_ang_vel_std = TURN_L4_SIGN_TRACK_ANG_VEL_STD
      yaw_velocity_error_weight = TURN_L4_SIGN_TRACK_YAW_ERROR_WEIGHT
      lin_vel_xy_penalty_weight = TURN_L4_EASY_LIN_VEL_XY_PENALTY_WEIGHT
      wheel_vel_penalty_weight = TURN_L4_EASY_WHEEL_VEL_PENALTY_WEIGHT
      action_rate_penalty_weight = TURN_L4_SIGN_TRACK_ACTION_RATE_WEIGHT
      wheel_target_rate_weight = TURN_L4_SIGN_TRACK_WHEEL_TARGET_RATE_WEIGHT
      wheel_target_slew_limit = TURN_L4_SIGN_TRACK_TARGET_SLEW_LIMIT
      wheel_yaw_scale = TURN_L4_SIGN_TRACK_YAW_SCALE
      binary_yaw_command = True
      yaw_sign_reward = True
      yaw_sign_weight = TURN_L4_SIGN_TRACK_YAW_WEIGHT
    else:
      raise ValueError(
        f"Unsupported turn_level={turn_level}. Expected 1, 2, 3, 4, 5, 6, 7, 8, or 9."
      )
  if scratch_stable_regularization:
    lin_vel_xy_penalty_weight = SCRATCH_STAGE0_STABLE_LIN_VEL_XY_WEIGHT
    wheel_vel_penalty_weight = SCRATCH_STAGE0_STABLE_WHEEL_VEL_WEIGHT
    action_rate_penalty_weight = SCRATCH_STAGE0_STABLE_ACTION_RATE_WEIGHT
  joint_pos_obs_func = envs_mdp.joint_pos_rel
  joint_pos_obs_params: dict[str, object] = {"asset_cfg": SceneEntityCfg("robot")}
  if zero_wheel_joint_pos_obs:
    joint_pos_obs_func = joint_pos_rel_without_wheel_position
    joint_pos_obs_params = {
      "asset_cfg": SceneEntityCfg("robot"),
      "wheel_joint_names": WHEEL_JOINT_NAMES,
    }
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
          func=joint_pos_obs_func,
          params=joint_pos_obs_params,
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
          func=joint_pos_obs_func,
          params=joint_pos_obs_params,
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

  actions = {}
  if use_differential_wheel_action:
    wheel_action_cfg_cls = DifferentialWheelVelocityActionCfg
  elif slow_speed_command_feedforward:
    wheel_action_cfg_cls = CommandFeedforwardCoupledWheelVelocityActionCfg
  else:
    wheel_action_cfg_cls = CoupledWheelVelocityActionCfg
  wheel_action_kwargs = {
    "entity_name": "robot",
    "actuator_names": WHEEL_JOINT_NAMES,
    "scale": WHEEL_VELOCITY_ACTION_SCALE,
    "offset": 0.0,
    "use_default_offset": False,
    "preserve_order": True,
  }
  if wheel_target_slew_limit is not None:
    wheel_action_kwargs["target_slew_limit"] = wheel_target_slew_limit
  if use_differential_wheel_action:
    wheel_action_kwargs["yaw_scale"] = wheel_yaw_scale
    wheel_action_kwargs["yaw_smoothing_alpha"] = wheel_yaw_smoothing_alpha
  elif slow_speed_command_feedforward:
    wheel_action_kwargs["command_name"] = "twist"
    wheel_action_kwargs["command_gain"] = SLOW_SPEED_FEEDFORWARD_GAIN
    wheel_action_kwargs["feedforward_clip"] = SLOW_SPEED_FEEDFORWARD_CLIP
    wheel_action_kwargs["residual_scale"] = slow_speed_residual_action_scale
  actions["wheel_balance"] = wheel_action_cfg_cls(**wheel_action_kwargs)

  if limited_leg_assist:
    assert leg_action_scale is not None
    actions["leg_assist_pos"] = JointPositionActionCfg(
      entity_name="robot",
      actuator_names=LEG_JOINT_NAMES,
      scale=leg_action_scale,
      offset=LEG_INIT_JOINT_POS,
      use_default_offset=False,
      preserve_order=True,
    )
  else:
    actions["fixed_leg_pos"] = FixedJointPositionActionCfg(
      entity_name="robot",
      actuator_names=LEG_JOINT_NAMES,
      scale=0.0,
      offset=LEG_INIT_JOINT_POS,
      use_default_offset=False,
      preserve_order=True,
    )

  if variable_yaw_slow_speed_turn_command:
    command_cfg_cls = VariableYawSlowSpeedTurnCommandCfg
  elif bidir_band_velocity_command:
    command_cfg_cls = BidirBandVelocityCommandCfg
  elif binary_slow_speed_turn_command:
    command_cfg_cls = BinarySlowSpeedTurnCommandCfg
  elif binary_yaw_command:
    command_cfg_cls = BinaryYawVelocityCommandCfg
  else:
    command_cfg_cls = UniformVelocityCommandCfg
  command_kwargs = {
    "entity_name": "robot",
    "resampling_time_range": command_resampling_time_range,
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
    command_kwargs["yaw_abs"] = binary_yaw_abs
  if binary_slow_speed_turn_command:
    command_kwargs["yaw_abs"] = binary_slow_speed_turn_yaw_abs
  if variable_yaw_slow_speed_turn_command:
    command_kwargs["yaw_abs_range"] = SLOW_SPEED_TURN_VARIABLE_YAW_ABS_RANGE
  if bidir_band_velocity_command:
    command_kwargs["lin_vel_x_abs_range"] = (
      SCRATCH_STAGE2_BIDIR_SMOOTH_SLEW12_ACTIVE_LIN_VEL_X_ABS_RANGE
    )
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
  if safe_posture_track_lin_vel_weight is not None:
    assert safe_posture_track_lin_vel_std is not None
    rewards["safe_posture_track_linear_velocity"] = RewardTermCfg(
      func=safe_posture_track_linear_velocity,
      weight=safe_posture_track_lin_vel_weight,
      params={
        "command_name": "twist",
        "std": safe_posture_track_lin_vel_std,
        "pitch_abs_limit": SCRATCH_STAGE1_FORWARD_GUARDED_SAFE_PITCH_ABS,
        "pitch_rate_abs_limit": SCRATCH_STAGE1_FORWARD_GUARDED_SAFE_PITCH_RATE_ABS,
      },
    )
  if safe_support_track_lin_vel_weight is not None:
    assert safe_support_track_lin_vel_std is not None
    rewards["safe_support_track_linear_velocity"] = RewardTermCfg(
      func=safe_support_track_linear_velocity,
      weight=safe_support_track_lin_vel_weight,
      params={
        "command_name": "twist",
        "std": safe_support_track_lin_vel_std,
        "wheel_sensor_name": WHEEL_GROUND_SENSOR_NAME,
        "non_wheel_sensor_name": NON_WHEEL_GROUND_SENSOR_NAME,
        "minimum_height": CLEAN_SUPPORT_MIN_HEIGHT,
        "max_tilt_xy": CLEAN_SUPPORT_MAX_TILT_XY,
        "pitch_rate_abs_limit": SCRATCH_STAGE1_FORWARD_GUARDED_SAFE_PITCH_RATE_ABS,
      },
    )
  if slow_speed_lin_sign:
    rewards["lin_vel_x_sign_alignment"] = RewardTermCfg(
      func=lin_vel_x_sign_alignment,
      weight=slow_speed_lin_sign_weight,
      params={
        "command_name": "twist",
        "deadband": SLOW_SPEED_LIN_SIGN_DEADBAND,
      },
    )
  if slow_speed_backward_strict:
    rewards["forward_lin_vel_x_ratio_on_backward_command"] = RewardTermCfg(
      func=forward_lin_vel_x_ratio_on_backward_command,
      weight=SLOW_SPEED_BACKWARD_STRICT_FORWARD_RATIO_WEIGHT,
      params={
        "command_name": "twist",
        "deadband": SLOW_SPEED_LIN_SIGN_DEADBAND,
      },
    )
  if forward_backward_lin_vel_weight is not None:
    rewards["backward_lin_vel_x_l2_on_forward_command"] = RewardTermCfg(
      func=backward_lin_vel_x_l2,
      weight=forward_backward_lin_vel_weight,
      params={"command_name": "twist"},
    )
  if unsafe_forward_velocity_weight is not None:
    rewards["unsafe_forward_velocity_l2"] = RewardTermCfg(
      func=unsafe_forward_velocity_l2,
      weight=unsafe_forward_velocity_weight,
      params={
        "command_name": "twist",
        "deadband": SLOW_SPEED_LIN_SIGN_DEADBAND,
        "pitch_abs_limit": SCRATCH_STAGE1_FORWARD_GUARDED_SAFE_PITCH_ABS,
        "pitch_rate_abs_limit": SCRATCH_STAGE1_FORWARD_GUARDED_SAFE_PITCH_RATE_ABS,
      },
    )
  if unsafe_support_forward_velocity_weight is not None:
    rewards["unsafe_support_forward_velocity_l2"] = RewardTermCfg(
      func=unsafe_support_forward_velocity_l2,
      weight=unsafe_support_forward_velocity_weight,
      params={
        "command_name": "twist",
        "deadband": SLOW_SPEED_LIN_SIGN_DEADBAND,
        "wheel_sensor_name": WHEEL_GROUND_SENSOR_NAME,
        "non_wheel_sensor_name": NON_WHEEL_GROUND_SENSOR_NAME,
        "minimum_height": CLEAN_SUPPORT_MIN_HEIGHT,
        "max_tilt_xy": CLEAN_SUPPORT_MAX_TILT_XY,
        "pitch_rate_abs_limit": SCRATCH_STAGE1_FORWARD_GUARDED_SAFE_PITCH_RATE_ABS,
      },
    )
  if slow_speed_pitch_target_sign is not None:
    rewards["pitch_target_l2"] = RewardTermCfg(
      func=pitch_target_l2,
      weight=SLOW_SPEED_BACKWARD_PITCH_TARGET_WEIGHT,
      params={
        "command_name": "twist",
        "sign": slow_speed_pitch_target_sign,
        "gain": SLOW_SPEED_BACKWARD_PITCH_TARGET_GAIN,
        "target_clip": SLOW_SPEED_BACKWARD_PITCH_TARGET_CLIP,
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
      weight=yaw_sign_weight,
      params={
        "command_name": "twist",
        "deadband": TURN_L4_SIGN_YAW_DEADBAND,
      },
    )
  if yaw_velocity_error_weight is not None:
    rewards["yaw_velocity_error_l2"] = RewardTermCfg(
      func=yaw_velocity_error_l2,
      weight=yaw_velocity_error_weight,
      params={
        "command_name": "twist",
        "deadband": TURN_L4_SIGN_YAW_DEADBAND,
      },
    )
  if slow_speed_turn_no_backward:
    rewards["backward_lin_vel_x_l2"] = RewardTermCfg(
      func=backward_lin_vel_x_l2,
      weight=SLOW_SPEED_TURN_NO_BACKWARD_WEIGHT,
      params={"command_name": "twist"},
    )
  if slow_speed_turn_bidirectional_lin_sign:
    rewards["lin_vel_x_sign_alignment"] = RewardTermCfg(
      func=lin_vel_x_sign_alignment,
      weight=lin_sign_weight,
      params={
        "command_name": "twist",
        "deadband": SLOW_SPEED_TURN_BIDIRECTIONAL_LIN_SIGN_DEADBAND,
      },
    )
  if effective_yaw_rate_weight is not None:
    rewards["effective_yaw_rate_l2"] = RewardTermCfg(
      func=effective_yaw_rate_l2,
      weight=effective_yaw_rate_weight,
      params={"action_name": "wheel_balance"},
    )
  if wheel_target_rate_weight is not None:
    rewards["wheel_target_rate_l2"] = RewardTermCfg(
      func=wheel_target_rate_l2,
      weight=wheel_target_rate_weight,
      params={"action_name": "wheel_balance"},
    )
  if pitch_abs_tail_weight is not None:
    assert pitch_abs_tail_limit is not None
    rewards["pitch_abs_above_limit_l2"] = RewardTermCfg(
      func=pitch_abs_above_limit_l2,
      weight=pitch_abs_tail_weight,
      params={"limit": pitch_abs_tail_limit},
    )
  if pitch_rate_abs_tail_weight is not None:
    assert pitch_rate_abs_tail_limit is not None
    rewards["pitch_rate_abs_above_limit_l2"] = RewardTermCfg(
      func=pitch_rate_abs_above_limit_l2,
      weight=pitch_rate_abs_tail_weight,
      params={"limit": pitch_rate_abs_tail_limit},
    )
  if leg_joint_pos_weight is not None:
    rewards["leg_joint_pos_deviation_l2"] = RewardTermCfg(
      func=joint_pos_deviation_l2,
      weight=leg_joint_pos_weight,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
  if leg_joint_vel_weight is not None:
    rewards["leg_joint_vel_l2"] = RewardTermCfg(
      func=envs_mdp.joint_vel_l2,
      weight=leg_joint_vel_weight,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
  if stable_wheel_target_rate_weight is not None:
    rewards["stable_wheel_target_rate_l2"] = RewardTermCfg(
      func=stable_wheel_target_rate_l2,
      weight=stable_wheel_target_rate_weight,
      params={
        "action_name": "wheel_balance",
        "wheel_sensor_name": WHEEL_GROUND_SENSOR_NAME,
        "non_wheel_sensor_name": NON_WHEEL_GROUND_SENSOR_NAME,
        "minimum_height": CLEAN_SUPPORT_MIN_HEIGHT,
        "max_tilt_xy": CLEAN_SUPPORT_MAX_TILT_XY,
      },
    )
  if zero_command_stable_wheel_target_rate_weight is not None:
    rewards["zero_command_stable_wheel_target_rate_l2"] = RewardTermCfg(
      func=zero_command_stable_wheel_target_rate_l2,
      weight=zero_command_stable_wheel_target_rate_weight,
      params={
        "action_name": "wheel_balance",
        "command_name": "twist",
        "deadband": SLOW_SPEED_LIN_SIGN_DEADBAND,
        "wheel_sensor_name": WHEEL_GROUND_SENSOR_NAME,
        "non_wheel_sensor_name": NON_WHEEL_GROUND_SENSOR_NAME,
        "minimum_height": CLEAN_SUPPORT_MIN_HEIGHT,
        "max_tilt_xy": CLEAN_SUPPORT_MAX_TILT_XY,
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
    episode_length_s=(
      1.0e9 if play else training_episode_length_s or 10.0
    ),
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
  if scratch_stage0_stable and (
    robust
    or push_l3
    or slow_speed
    or slow_speed_turn
    or turn_l4
    or limited_leg_assist
  ):
    raise ValueError(
      "scratch_stage0_stable is only valid for clean balance-only fixed-leg tasks."
    )
  if scratch_stage1_clear_forward and not (slow_speed and slow_speed_forward_only):
    raise ValueError(
      "scratch_stage1_clear_forward requires a slow_speed_forward_only task."
    )
  if scratch_stage1_forward_only_clear and not (
    slow_speed and slow_speed_forward_only
  ):
    raise ValueError(
      "scratch_stage1_forward_only_clear requires a slow_speed_forward_only task."
    )
  if scratch_stage1_forward_nospike and not (
    slow_speed and slow_speed_forward_only
  ):
    raise ValueError(
      "scratch_stage1_forward_nospike requires a slow_speed_forward_only task."
    )
  if scratch_stage1_forward_nospike_strong and not (
    slow_speed and slow_speed_forward_only
  ):
    raise ValueError(
      "scratch_stage1_forward_nospike_strong requires a slow_speed_forward_only task."
    )
  if scratch_stage1_forward_smooth_slew12 and not (
    slow_speed and slow_speed_forward_only
  ):
    raise ValueError(
      "scratch_stage1_forward_smooth_slew12 requires a slow_speed_forward_only task."
    )
  if scratch_stage1_forward_smooth_slew12_norev and not (
    slow_speed and slow_speed_forward_only
  ):
    raise ValueError(
      "scratch_stage1_forward_smooth_slew12_norev requires a slow_speed_forward_only task."
    )
  if scratch_stage2_bidir_smooth_slew12 and not (
    slow_speed
    and not slow_speed_forward_only
    and not slow_speed_backward_only
    and not slow_speed_backward_strict
  ):
    raise ValueError(
      "scratch_stage2_bidir_smooth_slew12 requires a bidirectional slow_speed task."
    )
  if scratch_stage1_forward_guarded and not (slow_speed and slow_speed_forward_only):
    raise ValueError(
      "scratch_stage1_forward_guarded requires a slow_speed_forward_only task."
    )
  if scratch_stage1_forward_support_guarded and not (
    slow_speed and slow_speed_forward_only
  ):
    raise ValueError(
      "scratch_stage1_forward_support_guarded requires a slow_speed_forward_only task."
    )
  if (
    (
      scratch_stage1_gentle_forward
      or scratch_stage1_gentle_forward_slew6
      or scratch_stage1_gentle_forward_zero_hold
    )
    and not (slow_speed and slow_speed_forward_only)
  ):
    raise ValueError(
      "scratch stage1 gentle forward tasks require a slow_speed_forward_only task."
    )
  scratch_stage1_variant_count = sum(
    (
      scratch_stage1_clear_forward,
      scratch_stage1_forward_only_clear,
      scratch_stage1_forward_nospike,
      scratch_stage1_forward_nospike_strong,
      scratch_stage1_forward_smooth_slew12,
      scratch_stage1_forward_smooth_slew12_norev,
      scratch_stage1_forward_guarded,
      scratch_stage1_forward_support_guarded,
      scratch_stage1_gentle_forward,
      scratch_stage1_gentle_forward_slew6,
      scratch_stage1_gentle_forward_zero_hold,
    )
  )
  if scratch_stage1_variant_count > 1:
    raise ValueError(
      "scratch stage1 forward variants are mutually exclusive."
    )
  if slow_speed_lin_sign and not slow_speed:
    raise ValueError("slow_speed_lin_sign=True requires slow_speed=True.")
  if slow_speed_obs_scale and not slow_speed:
    raise ValueError("slow_speed_obs_scale=True requires slow_speed=True.")
  if slow_speed_forward_only and not slow_speed:
    raise ValueError("slow_speed_forward_only=True requires slow_speed=True.")
  if slow_speed_backward_only and not slow_speed:
    raise ValueError("slow_speed_backward_only=True requires slow_speed=True.")
  if slow_speed_backward_strict and not slow_speed_backward_only:
    raise ValueError(
      "slow_speed_backward_strict=True requires slow_speed_backward_only=True."
    )
  if slow_speed_command_feedforward and not slow_speed:
    raise ValueError("slow_speed_command_feedforward=True requires slow_speed=True.")
  if slow_speed_command_feedforward_low_residual and not slow_speed_command_feedforward:
    raise ValueError(
      "slow_speed_command_feedforward_low_residual=True requires "
      "slow_speed_command_feedforward=True."
    )
  if slow_speed_command_feedforward and slow_speed_turn:
    raise ValueError(
      "slow_speed_command_feedforward is only enabled for 1D fixed-leg slow_speed."
    )
  if slow_speed_pitch_target_pos and slow_speed_pitch_target_neg:
    raise ValueError(
      "slow_speed_pitch_target_pos and slow_speed_pitch_target_neg are mutually exclusive."
    )
  pitch_target_allowed = (
    (slow_speed_backward_only and slow_speed_backward_strict)
    or (scratch_stage1_gentle_forward and slow_speed_forward_only)
  )
  if (
    (slow_speed_pitch_target_pos or slow_speed_pitch_target_neg)
    and not pitch_target_allowed
  ):
    raise ValueError(
      "slow_speed pitch-target variants require backward-only strict or "
      "scratch stage1 gentle-forward slow-speed."
    )
  if slow_speed_forward_only and slow_speed_backward_only:
    raise ValueError(
      "slow_speed_forward_only and slow_speed_backward_only are mutually exclusive."
    )
  if limited_leg_assist and not (slow_speed or slow_speed_turn):
    raise ValueError(
      "limited_leg_assist=True is currently only enabled for slow_speed "
      "or slow_speed_turn tasks."
    )
  if limited_leg_assist_safe and not limited_leg_assist:
    raise ValueError(
      "limited_leg_assist_safe=True requires limited_leg_assist=True."
    )
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
  if slow_speed_turn_safe_v2_yaw_scale2p5 and not slow_speed_turn_safe_v2:
    raise ValueError(
      "slow_speed_turn_safe_v2_yaw_scale2p5=True requires "
      "slow_speed_turn_safe_v2=True."
    )
  if slow_speed_turn_safe_v2_yaw_smooth and not (
    slow_speed_turn_safe_v2_yaw_scale3 or slow_speed_turn_safe_v2_yaw_scale2p5
  ):
    raise ValueError(
      "slow_speed_turn_safe_v2_yaw_smooth=True requires "
      "slow_speed_turn_safe_v2_yaw_scale3=True or "
      "slow_speed_turn_safe_v2_yaw_scale2p5=True."
    )
  if slow_speed_turn_safe_v2_yaw_smooth_v2 and not slow_speed_turn_safe_v2_yaw_smooth:
    raise ValueError(
      "slow_speed_turn_safe_v2_yaw_smooth_v2=True requires "
      "slow_speed_turn_safe_v2_yaw_smooth=True."
    )
  if slow_speed_turn_safe_v2_wheel_rate and not slow_speed_turn_safe_v2_yaw_smooth:
    raise ValueError(
      "slow_speed_turn_safe_v2_wheel_rate=True requires "
      "slow_speed_turn_safe_v2_yaw_smooth=True."
    )
  if slow_speed_turn_low_forward and not slow_speed_turn:
    raise ValueError(
      "slow_speed_turn_low_forward=True requires slow_speed_turn=True."
    )
  if slow_speed_turn_mid_forward and not slow_speed_turn:
    raise ValueError(
      "slow_speed_turn_mid_forward=True requires slow_speed_turn=True."
    )
  if slow_speed_turn_bidirectional and not slow_speed_turn:
    raise ValueError(
      "slow_speed_turn_bidirectional=True requires slow_speed_turn=True."
    )
  if slow_speed_turn_bidirectional_low_yaw and not slow_speed_turn_bidirectional:
    raise ValueError(
      "slow_speed_turn_bidirectional_low_yaw=True requires "
      "slow_speed_turn_bidirectional=True."
    )
  if slow_speed_turn_bidirectional_lin_sign and not slow_speed_turn_bidirectional:
    raise ValueError(
      "slow_speed_turn_bidirectional_lin_sign=True requires "
      "slow_speed_turn_bidirectional=True."
    )
  if slow_speed_turn_bidirectional_lin_sign_strong and not (
    slow_speed_turn_bidirectional and slow_speed_turn_bidirectional_lin_sign
  ):
    raise ValueError(
      "slow_speed_turn_bidirectional_lin_sign_strong=True requires "
      "slow_speed_turn_bidirectional=True and "
      "slow_speed_turn_bidirectional_lin_sign=True."
    )
  if sum(
    (
      bool(slow_speed_turn_low_forward),
      bool(slow_speed_turn_mid_forward),
      bool(slow_speed_turn_bidirectional),
    )
  ) > 1:
    raise ValueError(
      "slow_speed_turn_low_forward, slow_speed_turn_mid_forward, and "
      "slow_speed_turn_bidirectional are mutually exclusive."
    )
  if slow_speed_turn_safe and slow_speed_turn_safe_v2:
    raise ValueError(
      "slow_speed_turn_safe and slow_speed_turn_safe_v2 are mutually exclusive."
    )
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
  if slow_speed_turn_push and not slow_speed_turn:
    raise ValueError(
      "slow_speed_turn_push=True requires slow_speed_turn=True."
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
  if slow_speed_turn_push:
    cfg.events["slow_speed_turn_push_robot"] = EventTermCfg(
      func=envs_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=SLOW_SPEED_TURN_PUSH_INTERVAL_RANGE_S,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "velocity_range": {
          "x": (
            -SLOW_SPEED_TURN_PUSH_LIN_VEL_X_RANGE,
            SLOW_SPEED_TURN_PUSH_LIN_VEL_X_RANGE,
          ),
          "pitch": (
            -SLOW_SPEED_TURN_PUSH_ANG_VEL_PITCH_RANGE,
            SLOW_SPEED_TURN_PUSH_ANG_VEL_PITCH_RANGE,
          ),
        },
      },
    )

  return cfg
