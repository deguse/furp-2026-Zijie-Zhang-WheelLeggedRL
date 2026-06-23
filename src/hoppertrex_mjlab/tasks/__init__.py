"""Task registration for the standalone HopperTrex MjLab package."""

from mjlab.tasks.registry import register_mjlab_task

from .agents import hoppertrex_balance_ppo_runner_cfg
from .hoppertrex_balance_task import make_hoppertrex_balance_env_cfg

HOPPERTREX_BALANCE_TASK_ID = "Mjlab-HopperTrex-Balance-v0"


def _register(task_id: str) -> None:
  register_mjlab_task(
    task_id=task_id,
    env_cfg=make_hoppertrex_balance_env_cfg(play=False),
    play_env_cfg=make_hoppertrex_balance_env_cfg(play=True),
    rl_cfg=hoppertrex_balance_ppo_runner_cfg(),
    runner_cls=None,
  )


_register(HOPPERTREX_BALANCE_TASK_ID)
_register("hoppertrex-balance-v0")
