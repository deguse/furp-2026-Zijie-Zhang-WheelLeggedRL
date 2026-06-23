"""HopperTrex robot configuration for MjLab."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco

from mjlab.actuator import (
  BuiltinPositionActuatorCfg,
  BuiltinVelocityActuatorCfg,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

PROJECT_PATH = Path(__file__).resolve().parents[1]
HOPPERTREX_XML = PROJECT_PATH / "assets" / "hoppertrex" / "robot.xml"

HIP_INITIAL_ANGLE = math.radians(39.0)
WHEEL_VELOCITY_ACTION_SCALE = 12.0


def _rpm_to_rad_per_sec(rpm: float) -> float:
  return rpm * 2.0 * math.pi / 60.0


# Damiao DM-J6248P leg joint motor.
DM_J6248P_RATED_TORQUE = 30.0
DM_J6248P_PEAK_TORQUE = 97.0
DM_J6248P_RATED_SPEED = _rpm_to_rad_per_sec(40.0)
DM_J6248P_NO_LOAD_SPEED = _rpm_to_rad_per_sec(60.0)

# Lingkong/MyActuator RMD-L-9025-35T wheel motor.
RMD_L_9025_35T_RATED_TORQUE = 2.79
RMD_L_9025_35T_PEAK_TORQUE = 5.8
RMD_L_9025_35T_RATED_SPEED = _rpm_to_rad_per_sec(130.0)
RMD_L_9025_35T_NO_LOAD_SPEED = _rpm_to_rad_per_sec(280.0)

LEG_POSITION_STIFFNESS = 8000.0
LEG_POSITION_DAMPING = 80.0
WHEEL_VELOCITY_DAMPING = 200.0

LEG_JOINT_NAMES = (
  "thigh_left_01",
  "thigh_right_01",
  "knee_left",
  "knee_right",
)
WHEEL_JOINT_NAMES = ("wheel_left", "wheel_right")
ALL_JOINT_NAMES = (
  "thigh_left_01",
  "knee_left",
  "wheel_left",
  "thigh_right_01",
  "knee_right",
  "wheel_right",
)

INIT_JOINT_POS = {
  "thigh_left_01": HIP_INITIAL_ANGLE,
  "thigh_right_01": -HIP_INITIAL_ANGLE,
  "knee_left": 0.0,
  "knee_right": 0.0,
  "wheel_left": 0.0,
  "wheel_right": 0.0,
}


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(HOPPERTREX_XML))


HOPPERTREX_INIT_STATE = EntityCfg.InitialStateCfg(
  # The XML is authored around its CAD assembly origin. This height places the
  # simplified wheel cylinders just above the ground plane before settling.
  pos=(0.0, 0.0, 0.42),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos=INIT_JOINT_POS,
  joint_vel={".*": 0.0},
)

HOPPERTREX_LEG_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=LEG_JOINT_NAMES,
  stiffness=LEG_POSITION_STIFFNESS,
  damping=LEG_POSITION_DAMPING,
  effort_limit=DM_J6248P_PEAK_TORQUE,
  armature=0.02,
  frictionloss=0.0,
  viscous_damping=0.0,
)

HOPPERTREX_WHEEL_ACTUATOR_CFG = BuiltinVelocityActuatorCfg(
  target_names_expr=WHEEL_JOINT_NAMES,
  damping=WHEEL_VELOCITY_DAMPING,
  effort_limit=RMD_L_9025_35T_PEAK_TORQUE,
  armature=0.005,
  frictionloss=0.0,
  viscous_damping=0.0,
)

HOPPERTREX_FULL_COLLISION = CollisionCfg(
  geom_names_expr=(
    ".*_collision",
  ),
  condim={
    "wheel_.*_collision": 6,
    ".*_collision": 3,
  },
  priority={
    "wheel_.*_collision": 1,
  },
  friction={
    "wheel_.*_collision": (1.6, 2.0e-2, 2.0e-3),
    ".*_collision": (0.8, 3.0e-3, 1.0e-4),
  },
  solref={
    "wheel_.*_collision": (0.005, 1.0),
    ".*_collision": (0.01, 1.0),
  },
  solimp={
    "wheel_.*_collision": (0.95, 0.99, 0.001),
    ".*_collision": (0.9, 0.95, 0.001),
  },
)


def get_hoppertrex_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOPPERTREX_INIT_STATE,
    spec_fn=get_spec,
    articulation=EntityArticulationInfoCfg(
      actuators=(HOPPERTREX_LEG_ACTUATOR_CFG, HOPPERTREX_WHEEL_ACTUATOR_CFG),
      soft_joint_pos_limit_factor=0.95,
    ),
    collisions=(HOPPERTREX_FULL_COLLISION,),
    sort_actuators=True,
  )

