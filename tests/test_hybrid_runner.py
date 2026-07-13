from mjlab.tasks.registry import load_runner_cls

import hoppertrex_mjlab.tasks  # noqa: F401
from hoppertrex_mjlab.hybrid.runner import (
  HybridOnPolicyRunner,
  merge_hybrid_checkpoint_infos,
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
