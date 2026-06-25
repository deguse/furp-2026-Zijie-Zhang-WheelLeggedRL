"""HopperTrex two-wheel balance task for MjLab."""

from __future__ import annotations

import math

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers import (
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  SceneEntityCfg,
  TerminationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from assets.HopperTrex_CFG import (
  HIP_INITIAL_ANGLE,
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


def lin_vel_z_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = env.scene["robot"]
  return torch.square(robot.data.root_link_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = env.scene["robot"]
  return torch.sum(torch.square(robot.data.root_link_ang_vel_b[:, :2]), dim=1)


def lin_vel_xy_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = env.scene["robot"]
  return torch.sum(torch.square(robot.data.root_link_lin_vel_b[:, :2]), dim=1)


def make_hoppertrex_balance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
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
    "leg_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=LEG_JOINT_NAMES,
      scale=0.0,
      offset=LEG_INIT_JOINT_POS,
      use_default_offset=False,
      preserve_order=True,
    ),
    "wheel_vel": JointVelocityActionCfg(
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
    "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=1.0),
    "upright": RewardTermCfg(
      func=vel_mdp.upright,
      weight=2.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=("chassis_base",)),
      },
    ),
    "flat_orientation_l2": RewardTermCfg(func=envs_mdp.flat_orientation_l2, weight=-8.0),
    "ang_vel_xy_l2": RewardTermCfg(func=ang_vel_xy_l2, weight=-0.2),
    "lin_vel_xy_l2": RewardTermCfg(func=lin_vel_xy_l2, weight=-0.1),
    "lin_vel_z_l2": RewardTermCfg(func=lin_vel_z_l2, weight=-0.25),
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
      params={"limit_angle": 0.6},
    ),
    "root_too_low": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.18},
    ),
    "illegal_contact": TerminationTermCfg(
      func=vel_mdp.illegal_contact,
      params={"sensor_name": NON_WHEEL_GROUND_SENSOR_NAME, "force_threshold": 5.0},
    ),
    "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=num_envs,
      env_spacing=2.5,
      terrain=TerrainEntityCfg(terrain_type="plane", env_spacing=2.5),
      entities={"robot": robot_cfg},
      sensors=(non_wheel_ground_cfg,),
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

  return cfg
