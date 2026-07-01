"""Task registration for the standalone HopperTrex MjLab package."""

from mjlab.tasks.registry import register_mjlab_task

from .agents import hoppertrex_balance_ppo_runner_cfg
from .hoppertrex_balance_task import make_hoppertrex_balance_env_cfg

HOPPERTREX_BALANCE_TASK_ID = "Mjlab-HopperTrex-Balance-v0"
HOPPERTREX_BALANCE_ROBUST_TASK_ID = "Mjlab-HopperTrex-Balance-Robust-v0"
HOPPERTREX_BALANCE_ROBUST_L2_TASK_ID = "Mjlab-HopperTrex-Balance-Robust-L2-v0"
HOPPERTREX_BALANCE_PUSH_L3_TASK_ID = "Mjlab-HopperTrex-Balance-Push-L3-v0"
HOPPERTREX_BALANCE_SLOW_SPEED_TASK_ID = "Mjlab-HopperTrex-Balance-SlowSpeed-v0"
HOPPERTREX_BALANCE_SLOW_SPEED_EASY_TASK_ID = (
  "Mjlab-HopperTrex-Balance-SlowSpeed-Easy-v0"
)
HOPPERTREX_BALANCE_SLOW_SPEED_TURN_TASK_ID = (
  "Mjlab-HopperTrex-Balance-SlowSpeedTurn-v0"
)
HOPPERTREX_BALANCE_SLOW_SPEED_TURN_SIGN_TASK_ID = (
  "Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-v0"
)
HOPPERTREX_BALANCE_SLOW_SPEED_TURN_SIGN_OBS_SCALE_TASK_ID = (
  "Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-v0"
)
HOPPERTREX_BALANCE_SLOW_SPEED_TURN_SIGN_OBS_SCALE_SAFE_TASK_ID = (
  "Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-Safe-v0"
)
HOPPERTREX_BALANCE_SLOW_SPEED_TURN_SIGN_OBS_SCALE_SAFE_V2_TASK_ID = (
  "Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-v0"
)
HOPPERTREX_BALANCE_SLOW_SPEED_TURN_SIGN_OBS_SCALE_SAFE_V2_YAW_SCALE3_TASK_ID = (
  "Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale3-v0"
)
HOPPERTREX_BALANCE_TURN_L4_TASK_ID = "Mjlab-HopperTrex-Balance-Turn-L4-v0"
HOPPERTREX_BALANCE_TURN_L4_TRACK_TASK_ID = (
  "Mjlab-HopperTrex-Balance-Turn-L4-Track-v0"
)
HOPPERTREX_BALANCE_TURN_L4_TRACK_V2_TASK_ID = (
  "Mjlab-HopperTrex-Balance-Turn-L4-Track-v2"
)
HOPPERTREX_BALANCE_TURN_L4_EASY_TASK_ID = (
  "Mjlab-HopperTrex-Balance-Turn-L4-Easy-v0"
)
HOPPERTREX_BALANCE_TURN_L4_EASY_LOW_YAW_SCALE_TASK_ID = (
  "Mjlab-HopperTrex-Balance-Turn-L4-Easy-LowYawScale-v0"
)
HOPPERTREX_BALANCE_TURN_L4_SIGN_YAW_TASK_ID = (
  "Mjlab-HopperTrex-Balance-Turn-L4-SignYaw-v0"
)


def _register(
  task_id: str,
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
) -> None:
  register_mjlab_task(
    task_id=task_id,
    env_cfg=make_hoppertrex_balance_env_cfg(
      play=False,
      robust=robust,
      robust_level=robust_level,
      push_l3=push_l3,
      slow_speed=slow_speed,
      speed_level=speed_level,
      slow_speed_turn=slow_speed_turn,
      slow_speed_turn_sign=slow_speed_turn_sign,
      slow_speed_turn_obs_scale=slow_speed_turn_obs_scale,
      slow_speed_turn_safe=slow_speed_turn_safe,
      slow_speed_turn_safe_v2=slow_speed_turn_safe_v2,
      slow_speed_turn_safe_v2_yaw_scale3=slow_speed_turn_safe_v2_yaw_scale3,
      turn_l4=turn_l4,
      turn_level=turn_level,
    ),
    play_env_cfg=make_hoppertrex_balance_env_cfg(
      play=True,
      robust=robust,
      robust_level=robust_level,
      push_l3=push_l3,
      slow_speed=slow_speed,
      speed_level=speed_level,
      slow_speed_turn=slow_speed_turn,
      slow_speed_turn_sign=slow_speed_turn_sign,
      slow_speed_turn_obs_scale=slow_speed_turn_obs_scale,
      slow_speed_turn_safe=slow_speed_turn_safe,
      slow_speed_turn_safe_v2=slow_speed_turn_safe_v2,
      slow_speed_turn_safe_v2_yaw_scale3=slow_speed_turn_safe_v2_yaw_scale3,
      turn_l4=turn_l4,
      turn_level=turn_level,
    ),
    rl_cfg=hoppertrex_balance_ppo_runner_cfg(),
    runner_cls=None,
  )


_register(HOPPERTREX_BALANCE_TASK_ID)
_register("hoppertrex-balance-v0")
_register(HOPPERTREX_BALANCE_ROBUST_TASK_ID, robust=True)
_register("hoppertrex-balance-robust-v0", robust=True)
_register(HOPPERTREX_BALANCE_ROBUST_L2_TASK_ID, robust=True, robust_level=2)
_register("hoppertrex-balance-robust-l2-v0", robust=True, robust_level=2)
_register(
  HOPPERTREX_BALANCE_PUSH_L3_TASK_ID,
  robust=True,
  robust_level=2,
  push_l3=True,
)
_register(
  "hoppertrex-balance-push-l3-v0",
  robust=True,
  robust_level=2,
  push_l3=True,
)
_register(
  HOPPERTREX_BALANCE_SLOW_SPEED_TASK_ID,
  robust=True,
  robust_level=2,
  slow_speed=True,
)
_register(
  "hoppertrex-balance-slow-speed-v0",
  robust=True,
  robust_level=2,
  slow_speed=True,
)
_register(
  HOPPERTREX_BALANCE_SLOW_SPEED_EASY_TASK_ID,
  robust=True,
  robust_level=2,
  slow_speed=True,
  speed_level=0,
)
_register(
  "hoppertrex-balance-slow-speed-easy-v0",
  robust=True,
  robust_level=2,
  slow_speed=True,
  speed_level=0,
)
_register(
  HOPPERTREX_BALANCE_SLOW_SPEED_TURN_TASK_ID,
  robust=True,
  robust_level=2,
  slow_speed_turn=True,
)
_register(
  "hoppertrex-balance-slow-speed-turn-v0",
  robust=True,
  robust_level=2,
  slow_speed_turn=True,
)
_register(
  HOPPERTREX_BALANCE_SLOW_SPEED_TURN_SIGN_TASK_ID,
  robust=True,
  robust_level=2,
  slow_speed_turn=True,
  slow_speed_turn_sign=True,
)
_register(
  "hoppertrex-balance-slow-speed-turn-sign-v0",
  robust=True,
  robust_level=2,
  slow_speed_turn=True,
  slow_speed_turn_sign=True,
)
_register(
  HOPPERTREX_BALANCE_SLOW_SPEED_TURN_SIGN_OBS_SCALE_TASK_ID,
  robust=True,
  robust_level=2,
  slow_speed_turn=True,
  slow_speed_turn_sign=True,
  slow_speed_turn_obs_scale=True,
)
_register(
  "hoppertrex-balance-slow-speed-turn-sign-obs-scale-v0",
  robust=True,
  robust_level=2,
  slow_speed_turn=True,
  slow_speed_turn_sign=True,
  slow_speed_turn_obs_scale=True,
)
_register(
  HOPPERTREX_BALANCE_SLOW_SPEED_TURN_SIGN_OBS_SCALE_SAFE_TASK_ID,
  robust=True,
  robust_level=2,
  slow_speed_turn=True,
  slow_speed_turn_sign=True,
  slow_speed_turn_obs_scale=True,
  slow_speed_turn_safe=True,
)
_register(
  "hoppertrex-balance-slow-speed-turn-sign-obs-scale-safe-v0",
  robust=True,
  robust_level=2,
  slow_speed_turn=True,
  slow_speed_turn_sign=True,
  slow_speed_turn_obs_scale=True,
  slow_speed_turn_safe=True,
)
_register(
  HOPPERTREX_BALANCE_SLOW_SPEED_TURN_SIGN_OBS_SCALE_SAFE_V2_TASK_ID,
  robust=True,
  robust_level=2,
  slow_speed_turn=True,
  slow_speed_turn_sign=True,
  slow_speed_turn_obs_scale=True,
  slow_speed_turn_safe_v2=True,
)
_register(
  "hoppertrex-balance-slow-speed-turn-sign-obs-scale-safe-v2-v0",
  robust=True,
  robust_level=2,
  slow_speed_turn=True,
  slow_speed_turn_sign=True,
  slow_speed_turn_obs_scale=True,
  slow_speed_turn_safe_v2=True,
)
_register(
  HOPPERTREX_BALANCE_SLOW_SPEED_TURN_SIGN_OBS_SCALE_SAFE_V2_YAW_SCALE3_TASK_ID,
  robust=True,
  robust_level=2,
  slow_speed_turn=True,
  slow_speed_turn_sign=True,
  slow_speed_turn_obs_scale=True,
  slow_speed_turn_safe_v2=True,
  slow_speed_turn_safe_v2_yaw_scale3=True,
)
_register(
  "hoppertrex-balance-slow-speed-turn-sign-obs-scale-safe-v2-yaw-scale3-v0",
  robust=True,
  robust_level=2,
  slow_speed_turn=True,
  slow_speed_turn_sign=True,
  slow_speed_turn_obs_scale=True,
  slow_speed_turn_safe_v2=True,
  slow_speed_turn_safe_v2_yaw_scale3=True,
)
_register(
  HOPPERTREX_BALANCE_TURN_L4_TASK_ID,
  robust=True,
  robust_level=2,
  turn_l4=True,
)
_register(
  "hoppertrex-balance-turn-l4-v0",
  robust=True,
  robust_level=2,
  turn_l4=True,
)
_register(
  HOPPERTREX_BALANCE_TURN_L4_TRACK_TASK_ID,
  robust=True,
  robust_level=2,
  turn_l4=True,
  turn_level=2,
)
_register(
  "hoppertrex-balance-turn-l4-track-v0",
  robust=True,
  robust_level=2,
  turn_l4=True,
  turn_level=2,
)
_register(
  HOPPERTREX_BALANCE_TURN_L4_TRACK_V2_TASK_ID,
  robust=True,
  robust_level=2,
  turn_l4=True,
  turn_level=3,
)
_register(
  "hoppertrex-balance-turn-l4-track-v2",
  robust=True,
  robust_level=2,
  turn_l4=True,
  turn_level=3,
)
_register(
  HOPPERTREX_BALANCE_TURN_L4_EASY_TASK_ID,
  robust=True,
  robust_level=2,
  turn_l4=True,
  turn_level=4,
)
_register(
  "hoppertrex-balance-turn-l4-easy-v0",
  robust=True,
  robust_level=2,
  turn_l4=True,
  turn_level=4,
)
_register(
  HOPPERTREX_BALANCE_TURN_L4_EASY_LOW_YAW_SCALE_TASK_ID,
  robust=True,
  robust_level=2,
  turn_l4=True,
  turn_level=5,
)
_register(
  "hoppertrex-balance-turn-l4-easy-low-yaw-scale-v0",
  robust=True,
  robust_level=2,
  turn_l4=True,
  turn_level=5,
)
_register(
  HOPPERTREX_BALANCE_TURN_L4_SIGN_YAW_TASK_ID,
  robust=True,
  robust_level=2,
  turn_l4=True,
  turn_level=6,
)
_register(
  "hoppertrex-balance-turn-l4-sign-yaw-v0",
  robust=True,
  robust_level=2,
  turn_l4=True,
  turn_level=6,
)
