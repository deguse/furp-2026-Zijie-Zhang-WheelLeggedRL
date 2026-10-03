"""Temporary tested launcher: force play env/runner seed to checkpoint seed 1."""
from mjlab.tasks.registry import load_env_cfg as _registry_load_env_cfg
from mjlab.tasks.registry import load_rl_cfg as _registry_load_rl_cfg
import mjlab.scripts.play as _mjlab_play


def _seeded_env_cfg(task_id: str, play: bool = False):
    cfg = _registry_load_env_cfg(task_id, play=play)
    cfg.seed = 1
    return cfg


def _seeded_rl_cfg(task_id: str):
    cfg = _registry_load_rl_cfg(task_id)
    cfg.seed = 1
    return cfg


_mjlab_play.load_env_cfg = _seeded_env_cfg
_mjlab_play.load_rl_cfg = _seeded_rl_cfg

from hoppertrex_mjlab.scripts.rsl_rl.play import main

main()
