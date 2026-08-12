import unittest
from unittest.mock import patch

import torch
from mjlab.envs import ManagerBasedRlEnv

from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
  ROLL_ASSIST_FLAT_ENVS,
  ROLL_ASSIST_SETTLE_STEPS,
  RollAssistEpisodeEvidence,
  make_stair_roll_assist_env_cfg,
  roll_assist_progress_reward,
)


class RollAssistRuntimeTest(unittest.TestCase):
  def test_real_reset_enforces_64_192_split_and_two_second_command_settle(self):
    cfg = make_stair_roll_assist_env_cfg(play=False)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
      env.reset()
      terrain = env.scene.terrain
      self.assertIsNotNone(terrain)
      self.assertTrue(torch.all(terrain.terrain_types[:ROLL_ASSIST_FLAT_ENVS] == 0))
      self.assertTrue(torch.all(terrain.terrain_types[ROLL_ASSIST_FLAT_ENVS:] == 1))
      self.assertTrue(torch.all(terrain.terrain_levels[ROLL_ASSIST_FLAT_ENVS:] == 0))
      twist = env.command_manager.get_term("twist")
      self.assertTrue(torch.all(twist.command[ROLL_ASSIST_FLAT_ENVS:, 0] == 0.0))
      env.episode_length_buf[ROLL_ASSIST_FLAT_ENVS:] = ROLL_ASSIST_SETTLE_STEPS
      env.command_manager.compute(dt=env.step_dt)
      self.assertTrue(torch.allclose(
        twist.command[ROLL_ASSIST_FLAT_ENVS:, 0],
        torch.full_like(twist.command[ROLL_ASSIST_FLAT_ENVS:, 0], 0.07),
      ))
    finally:
      env.close()

  def test_progress_reward_is_zero_until_settle_completes(self):
    env = type("Env", (), {})()
    env.num_envs = 2
    env.device = "cpu"
    env.cfg = type("Cfg", (), {"roll_assist_flat_env_count": 1})()
    data = type("Data", (), {
      "root_link_lin_vel_w": torch.tensor([[0.07, 0.0, 0.0], [0.07, 0.0, 0.0]])
    })()
    env.scene = {"robot": type("Robot", (), {"data": data})()}
    env.episode_length_buf = torch.tensor([ROLL_ASSIST_SETTLE_STEPS, ROLL_ASSIST_SETTLE_STEPS - 1])
    self.assertEqual(roll_assist_progress_reward(env).tolist(), [0.0, 0.0])
    env.episode_length_buf[1] = ROLL_ASSIST_SETTLE_STEPS
    self.assertEqual(float(roll_assist_progress_reward(env)[1]), 0.0)
    env.episode_length_buf[1] = ROLL_ASSIST_SETTLE_STEPS + 1
    self.assertAlmostEqual(float(roll_assist_progress_reward(env)[1]), 0.07, places=6)


  def test_bilateral_airborne_latches_before_episode_reset(self):
    class Sensor:
      def __init__(self, found):
        self.data = type("Data", (), {"found": found})()

    found = torch.ones((2, 1), dtype=torch.bool)
    env = type("Env", (), {})()
    env.num_envs = 2
    env.device = "cpu"
    env.cfg = type("Cfg", (), {"roll_assist_flat_env_count": 0})()
    env.episode_length_buf = torch.tensor([ROLL_ASSIST_SETTLE_STEPS, ROLL_ASSIST_SETTLE_STEPS])
    env.scene = {
      "roll_assist_left_wheel_contact": Sensor(found.clone()),
      "roll_assist_right_wheel_contact": Sensor(found.clone()),
    }
    env.scene["roll_assist_left_wheel_contact"].data.found[1] = False
    env.scene["roll_assist_right_wheel_contact"].data.found[1] = False
    # Keep this unit-level latch check focused on contact state; the remaining
    # evidence inputs are minimal valid tensors.
    robot_data = type("RobotData", (), {
      "root_link_pos_w": torch.zeros((2, 3)),
      "projected_gravity_b": torch.tensor([[0.0, 0.0, -1.0]] * 2),
      "root_link_ang_vel_b": torch.zeros((2, 3)),
    })()
    env.scene["robot"] = type("Robot", (), {"data": robot_data})()
    env.scene = type("Scene", (dict,), {
      "env_origins": torch.zeros((2, 3)),
    })(env.scene)
    action = type("Action", (), {"applied_residual": torch.zeros((2, 6))})()
    env.action_manager = type("Actions", (), {
      "get_term": lambda self, name: action,
    })()
    evidence = RollAssistEpisodeEvidence(None, env)
    with patch(
      "hoppertrex_mjlab.tasks.hoppertrex_hybrid_task._roll_assist_action",
      return_value=action,
    ):
      result = evidence(env)
    self.assertEqual(result.tolist(), [False, True])
    self.assertEqual(evidence.bilateral_airborne_ever.tolist(), [False, True])

  def test_stable_success_starts_only_after_command_was_observed(self):
    class Sensor:
      def __init__(self):
        self.data = type("Data", (), {"found": torch.ones((1, 1), dtype=torch.bool)})()

    env = type("Env", (), {})()
    env.num_envs = 1
    env.device = "cpu"
    env.cfg = type("Cfg", (), {"roll_assist_flat_env_count": 0})()
    env.episode_length_buf = torch.tensor([ROLL_ASSIST_SETTLE_STEPS])
    robot_data = type("RobotData", (), {
      "root_link_pos_w": torch.tensor([[-2.0, 0.0, 0.0]]),
      "projected_gravity_b": torch.tensor([[0.0, 0.0, -1.0]]),
      "root_link_ang_vel_b": torch.zeros((1, 3)),
    })()
    env.scene = type("Scene", (dict,), {
      "env_origins": torch.zeros((1, 3)),
    })({
      "robot": type("Robot", (), {"data": robot_data})(),
      "roll_assist_left_wheel_contact": Sensor(),
      "roll_assist_right_wheel_contact": Sensor(),
    })
    action = type("Action", (), {"applied_residual": torch.zeros((1, 6))})()
    evidence = RollAssistEpisodeEvidence(None, env)
    with patch(
      "hoppertrex_mjlab.tasks.hoppertrex_hybrid_task._roll_assist_action",
      return_value=action,
    ):
      evidence(env)
      self.assertEqual(int(evidence.stable_steps[0]), 0)
      env.episode_length_buf[0] = ROLL_ASSIST_SETTLE_STEPS + 1
      evidence(env)
    self.assertEqual(int(evidence.stable_steps[0]), 1)


if __name__ == "__main__":
  unittest.main()
