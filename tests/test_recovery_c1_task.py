"""Task/runner contract tests; actual CPU physics smoke is separately recorded."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from hoppertrex_mjlab.hybrid.recovery_pilot import ACTION_SCALES, ACTION_STD, TASK_ID, digest, verify_artifacts
from hoppertrex_mjlab.hybrid.runner import zero_initialize_actor_output
from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import make_hoppertrex_hybrid_env_cfg
from hoppertrex_mjlab.tasks.recovery_c1_pilot import (
  REPO, RecoveryPilotRunner, configuration, make_env_cfg, normalized,
)
from hoppertrex_mjlab.scripts.rsl_rl.train import resolve_and_validate_hybrid_resume

GOLDEN = {
  "0_False": "82a746881b2037681bbbfbcaff997b00b0e881e2eb6bd42f4d064c8b91f42b05",
  "0_True": "d12701751fcdbe4509333ea2cd02a7a97e4d4fb92991e392096a52f36bcdc202",
  "1_False": "c381a37579414709f1dc3d337cbfd2272029e9c7d61e1b0fb8e551a6af7dd6b1",
  "1_True": "2142d99bc075a5d2f291c8a35940dd9ad12b9dc2085a654ba1843f02ed725b09",
  "2_False": "0a8c059b02a47a452bf7c19a9190b08792e5b4731b416ea0244f4168de2ec8a0",
  "2_True": "dcab9ffbc278e8cf9563c045242c1cdd85e2c760bf1f4dbe65b88a36e36c31fc",
  "3_False": "c83cefb2e73a78833e35049d6e01462018d509a618b19bc1b774eb07064e48f7",
  "3_True": "ef3f0467779af97ecbfc72b8723aa7e0262583ada98a81766d89d813dbdbf350",
  "4_False": "4d4820def6b065be0bb3c7336693d31dc1d9d1ff5f65dba82a0d2850e2627b4e",
  "4_True": "ae9fd8abc0516da9093c1ed264dd2add6eff1ddcbf30a94f3472b50dd5218a51",
  "5_False": "c6806c704d96fc9900532507862ac7ab2ad22adbc3345ffbb087322bfecc85b4",
  "5_True": "1c4bd6feade4441915b7c4193361562bc42a78de15e48023957cf39abdba430d"
}


class PilotTaskTest(unittest.TestCase):
  def test_twelve_legacy_configs_unchanged(self):
    for stage in range(6):
      for play in (False, True):
        with self.subTest(stage=stage, play=play):
          self.assertEqual(digest(normalized(make_hoppertrex_hybrid_env_cfg(stage=stage, play=play))), GOLDEN[f"{stage}_{play}"])

  def test_explicit_five_artifacts_and_registration(self):
    self.assertEqual(len(verify_artifacts(REPO)), 5)
    env, agent = load_env_cfg(TASK_ID), load_rl_cfg(TASK_ID)
    self.assertEqual(env.scene.num_envs, 256)
    self.assertEqual(env.actions["hybrid_wheel_leg"].action_scales, ACTION_SCALES)
    self.assertEqual(agent.max_iterations, 1000)
    self.assertEqual(agent.num_steps_per_env, 24)
    self.assertFalse(agent.resume)

  def test_resolved_configuration_is_repeatable(self):
    self.assertEqual(digest(configuration()), digest(configuration()))
    c = configuration()
    self.assertEqual(c["initial_std_override"], list(ACTION_STD))
    self.assertEqual(c["randomization_audit"]["training_events"],
                     ["reset_scene_to_default", "reset_root_state_with_small_disturbance", "push_robot"])
    self.assertNotIn("push_robot", c["randomization_audit"]["play_events"])

  def test_legacy_leg_override_not_inherited_or_destroyed(self):
    key = "HOPPERTREX_HYBRID_LEG_RESIDUAL_SCALE"
    with patch.dict(os.environ, {key: "0.070"}):
      cfg = make_env_cfg(play=False)
      self.assertEqual(cfg.actions["hybrid_wheel_leg"].action_scales, ACTION_SCALES)
      self.assertEqual(os.environ[key], "0.070")
      old = make_hoppertrex_hybrid_env_cfg(stage=5)
      self.assertEqual(old.actions["hybrid_wheel_leg"].action_scales[2], .070)

  def test_generic_training_cannot_bypass_approval(self):
    with self.assertRaisesRegex(ValueError, "wrapper|run_recovery"):
      resolve_and_validate_hybrid_resume(TASK_ID, None)

  def test_zero_output_not_zero_hidden_layers(self):
    actor = torch.nn.Sequential(torch.nn.Linear(34, 128), torch.nn.ELU(), torch.nn.Linear(128, 6))
    before = actor[0].weight.detach().clone()
    zero_initialize_actor_output(actor, label="test")
    self.assertTrue(torch.equal(actor(torch.randn(5, 34)), torch.zeros(5, 6)))
    self.assertTrue(torch.equal(before, actor[0].weight))


class SaveGridTest(unittest.TestCase):
  def test_completed_update_grid(self):
    runner = RecoveryPilotRunner.__new__(RecoveryPilotRunner)
    runner._mode = "training"
    runner._identity = {"git_sha": "a" * 40}
    runner._seed = 11
    runner._config_hash = "cfg"
    with tempfile.TemporaryDirectory() as t, patch.object(MjlabOnPolicyRunner, "save") as base:
      for i in (0, 98, 99, 100, 199, 999):
        runner.current_learning_iteration = i
        runner.save(str(Path(t) / f"model_{i}.pt"))
      self.assertEqual([x.args[1]["recovery_c1_pilot"]["completed_updates"] for x in base.call_args_list], [100, 200, 1000])

  def test_no_resume_or_learning_in_evaluation(self):
    runner = RecoveryPilotRunner.__new__(RecoveryPilotRunner)
    runner.current_learning_iteration = 0
    runner._mode = "evaluation"
    with self.assertRaises(ValueError):
      runner.learn(1000)
    runner._mode = "training"
    with self.assertRaises(ValueError):
      runner.learn(100)
    runner.current_learning_iteration = 10
    with self.assertRaises(ValueError):
      runner.learn(1000)


if __name__ == "__main__":
  unittest.main()
