import unittest

import torch

from mjlab.envs import ManagerBasedRlEnv

from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
  make_hoppertrex_hybrid_env_cfg,
)


def _one_env_cfg(stage: int):
  cfg = make_hoppertrex_hybrid_env_cfg(stage=stage)
  cfg.scene.num_envs = 1
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.num_envs = 1
  return cfg


class HybridTaskRuntimeTest(unittest.TestCase):
  def test_stage0_zero_residual_matches_controller_with_limits(self):
    env = ManagerBasedRlEnv(cfg=_one_env_cfg(stage=0), device="cpu")
    try:
      observations, _ = env.reset()
      self.assertEqual(env.action_space.shape[-1], 6)
      self.assertEqual(observations["actor"].shape[-1], 34)

      env.step(torch.zeros(env.action_space.shape, device=env.device))
      action = env.action_manager.get_term("hybrid_wheel_leg")
      expected = torch.clamp(
        action.controller_baseline,
        -action.cfg.wheel_slew_limit,
        action.cfg.wheel_slew_limit,
      )
      expected = torch.clamp(
        expected,
        -action.cfg.wheel_velocity_limit,
        action.cfg.wheel_velocity_limit,
      )

      torch.testing.assert_close(
        action.applied_residual,
        torch.zeros_like(action.applied_residual),
      )
      torch.testing.assert_close(action.wheel_targets, expected)
    finally:
      env.close()

  def test_posture_and_robust_stages_reset_and_step_on_cpu(self):
    for stage in (3, 5):
      with self.subTest(stage=stage):
        env = ManagerBasedRlEnv(cfg=_one_env_cfg(stage=stage), device="cpu")
        try:
          observations, _ = env.reset()
          next_observations, rewards, terminated, truncated, _ = env.step(
            torch.zeros(env.action_space.shape, device=env.device)
          )

          self.assertEqual(env.action_space.shape[-1], 6)
          self.assertEqual(observations["actor"].shape[-1], 34)
          self.assertEqual(next_observations["actor"].shape[-1], 34)
          self.assertTrue(torch.isfinite(next_observations["actor"]).all())
          self.assertTrue(torch.isfinite(rewards).all())
          self.assertFalse(terminated.any())
          self.assertFalse(truncated.any())
        finally:
          env.close()


if __name__ == "__main__":
  unittest.main()
