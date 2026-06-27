"""Task registration for the standalone HopperTrex MjLab package."""

from mjlab.tasks.registry import register_mjlab_task

from .agents import hoppertrex_balance_ppo_runner_cfg
from .hoppertrex_balance_task import make_hoppertrex_balance_env_cfg

HOPPERTREX_BALANCE_TASK_ID = "Mjlab-HopperTrex-Balance-v0"
HOPPERTREX_BALANCE_ROBUST_TASK_ID = "Mjlab-HopperTrex-Balance-Robust-v0"
HOPPERTREX_BALANCE_ROBUST_L2_TASK_ID = "Mjlab-HopperTrex-Balance-Robust-L2-v0"


def _register(task_id: str, robust: bool = False, robust_level: int = 1) -> None:
  register_mjlab_task(
    task_id=task_id,
    env_cfg=make_hoppertrex_balance_env_cfg(
      play=False,
      robust=robust,
      robust_level=robust_level,
    ),
    play_env_cfg=make_hoppertrex_balance_env_cfg(
      play=True,
      robust=robust,
      robust_level=robust_level,
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
