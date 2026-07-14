from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

import hoppertrex_mjlab.tasks  # noqa: F401
from hoppertrex_mjlab.hybrid.runner import (
  HybridOnPolicyRunner,
  merge_hybrid_checkpoint_infos,
)
from hoppertrex_mjlab.hybrid.config import HYBRID_STAGES
from hoppertrex_mjlab.tasks.agents.hoppertrex_balance_rsl_rl_ppo import (
  HYBRID_DISTRIBUTION_CLASS,
)


def test_merge_preserves_loaded_provenance_and_updates_save_infos():
  merged = merge_hybrid_checkpoint_infos(
    {
      "hybrid_stage1_bootstrap": {"stage": 1},
      "env_state": {"common_step_counter": 10},
    },
    {"env_state": {"common_step_counter": 20}},
  )

  assert merged == {
    "hybrid_stage1_bootstrap": {"stage": 1},
    "env_state": {"common_step_counter": 20},
  }


def test_all_hybrid_tasks_use_provenance_preserving_runner():
  for stage in range(6):
    assert (
      load_runner_cls(f"HopperTrex-Hybrid-v2-Stage{stage}")
      is HybridOnPolicyRunner
    )


def test_all_hybrid_tasks_mask_inactive_ppo_distribution_heads():
  for stage, stage_cfg in HYBRID_STAGES.items():
    cfg = load_rl_cfg(f"HopperTrex-Hybrid-v2-Stage{stage}")
    assert cfg.actor.distribution_cfg is not None
    assert cfg.actor.distribution_cfg["class_name"] == HYBRID_DISTRIBUTION_CLASS
    assert tuple(cfg.actor.distribution_cfg["active_mask"]) == stage_cfg.action_mask
