import unittest

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv

from hoppertrex_mjlab.hybrid.control import compose_hybrid_targets
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

  def test_residual_composition_matches_numpy_contract(self):
    """Bind the runtime torch composition to hybrid.control step by step.

    Stage 3 activates all six heads. The wheel limits are shrunk so the slew
    clamp and the velocity saturation both engage inside the schedule, and
    the schedule includes |action| > 1 rows so residual clipping engages.
    Legs stay inside their soft limits; that clamp is covered by the numpy
    unit tests.
    """

    cfg = _one_env_cfg(stage=3)
    action_cfg = cfg.actions["hybrid_wheel_leg"]
    action_cfg.wheel_slew_limit = 0.4
    action_cfg.wheel_velocity_limit = 0.6
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
      env.reset()
      term = env.action_manager.get_term("hybrid_wheel_leg")
      robot = env.scene["robot"]
      leg_ids, _leg_names = robot.find_joints(
        term.cfg.leg_joint_names,
        preserve_order=True,
      )
      soft_limits = (
        robot.data.soft_joint_pos_limits[0, leg_ids, :].detach().cpu().numpy()
      )
      schedule = (
        (1.5, -1.4, 1.5, -1.5, 1.2, -1.2),
        (-1.5, 1.4, -0.6, 0.6, -1.0, 1.0),
        (0.10, 0.05, 0.2, -0.2, 0.3, -0.3),
        (-1.2, -1.3, -1.5, 1.5, -0.4, 0.4),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1.5, 1.5, 1.0, -1.0, 0.5, -0.5),
      )
      previous = np.zeros((1, 2))
      slew_engaged = False
      saturation_engaged = False
      for row in schedule:
        actions = torch.tensor([row], dtype=torch.float32)
        _obs, _rewards, terminated, truncated, _extras = env.step(actions)
        self.assertFalse(terminated.any())
        self.assertFalse(truncated.any())
        expected = compose_hybrid_targets(
          policy_actions=actions.numpy(),
          action_mask=term.cfg.action_mask,
          action_scales=term.cfg.action_scales,
          baseline_wheel_targets=(
            term.controller_baseline.detach().cpu().numpy()
          ),
          nominal_leg_targets=(
            term.nominal_leg_targets.detach().cpu().numpy()
          ),
          previous_wheel_targets=previous,
          wheel_slew_limit=term.cfg.wheel_slew_limit,
          wheel_velocity_limit=term.cfg.wheel_velocity_limit,
          leg_position_lower=soft_limits[:, 0],
          leg_position_upper=soft_limits[:, 1],
        )
        residual = expected.applied_residual
        balance = residual[:, 0]
        yaw = residual[:, 1]
        desired = expected.wheel_targets * 0.0
        desired[:, 0] = -balance + yaw
        desired[:, 1] = balance + yaw
        desired += term.controller_baseline.detach().cpu().numpy()
        delta = desired - previous
        slew_engaged |= bool(
          np.any(np.abs(delta) > term.cfg.wheel_slew_limit + 1.0e-9)
        )
        clamped_delta = np.clip(
          delta,
          -term.cfg.wheel_slew_limit,
          term.cfg.wheel_slew_limit,
        )
        saturation_engaged |= bool(
          np.any(
            np.abs(previous + clamped_delta)
            > term.cfg.wheel_velocity_limit + 1.0e-9
          )
        )
        np.testing.assert_allclose(
          term.applied_residual.detach().cpu().numpy(),
          expected.applied_residual,
          rtol=1.0e-5,
          atol=1.0e-5,
        )
        np.testing.assert_allclose(
          term.wheel_targets.detach().cpu().numpy(),
          expected.wheel_targets,
          rtol=1.0e-5,
          atol=1.0e-5,
        )
        np.testing.assert_allclose(
          term.leg_targets.detach().cpu().numpy(),
          expected.leg_targets,
          rtol=1.0e-5,
          atol=1.0e-5,
        )
        previous = term.wheel_targets.detach().cpu().numpy().copy()
      self.assertTrue(slew_engaged)
      self.assertTrue(saturation_engaged)
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
