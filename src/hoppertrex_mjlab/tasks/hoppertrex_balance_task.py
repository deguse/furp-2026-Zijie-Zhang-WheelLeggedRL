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
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
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
) -> ManagerBasedRlEnvCfg:
  robot_cfg = get_hoppertrex_robot_cfg()
  num_envs = 16 if play else 4096
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
          func=envs_mdp.generated_commands,
          params={"command_name": "twist"},
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
          func=envs_mdp.generated_commands,
          params={"command_name": "twist"},
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
    "wheel_balance": CoupledWheelVelocityActionCfg(
      entity_name="robot",
      actuator_names=WHEEL_JOINT_NAMES,
      scale=WHEEL_VELOCITY_ACTION_SCALE,
      offset=0.0,
      use_default_offset=False,
      preserve_order=True,
    ),
  }

  commands = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(5.0, 10.0),
      rel_standing_envs=1.0,
      rel_heading_envs=0.0,
      rel_forward_envs=0.0,
      heading_command=False,
      debug_vis=play,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(0.0, 0.0),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(0.0, 0.0),
      ),
    )
  }

  rewards = {
    "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=0.5),
    "clean_wheel_support": RewardTermCfg(
      func=clean_wheel_support,
      weight=4.0,
      params={
        "wheel_sensor_name": WHEEL_GROUND_SENSOR_NAME,
        "non_wheel_sensor_name": NON_WHEEL_GROUND_SENSOR_NAME,
        "minimum_height": CLEAN_SUPPORT_MIN_HEIGHT,
        "max_tilt_xy": CLEAN_SUPPORT_MAX_TILT_XY,
      },
    ),
    "wheel_ground_contact": RewardTermCfg(
      func=wheel_ground_contact,
      weight=1.0,
      params={"sensor_name": WHEEL_GROUND_SENSOR_NAME},
    ),
    "non_wheel_ground_contact": RewardTermCfg(
      func=non_wheel_ground_contact,
      weight=-6.0,
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
    "lin_vel_xy_l2": RewardTermCfg(func=lin_vel_xy_l2, weight=-0.02),
    "lin_vel_z_l2": RewardTermCfg(func=lin_vel_z_l2, weight=-0.15),
    "wheel_vel_l2": RewardTermCfg(
      func=envs_mdp.joint_vel_l2,
      weight=-5.0e-4,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    ),
    "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01),
  }

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

  return cfg
