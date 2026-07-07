import unittest

from mjlab.tasks.registry import load_env_cfg

import hoppertrex_mjlab.tasks as hoppertrex_tasks
from hoppertrex_mjlab.tasks.hoppertrex_balance_task import (
  BidirBandVelocityCommandCfg,
  WHEEL_JOINT_NAMES,
  joint_pos_rel_without_wheel_position,
  make_hoppertrex_balance_env_cfg,
)


class Stage2CommandConfigTest(unittest.TestCase):
  def test_stage2_smooth_slew12_uses_bidirectional_speed_band_command(self):
    cfg = make_hoppertrex_balance_env_cfg(
      slow_speed=True,
      speed_level=0,
      slow_speed_lin_sign=True,
      slow_speed_obs_scale=True,
      scratch_stage2_bidir_smooth_slew12=True,
    )

    twist = cfg.commands["twist"]

    self.assertIsInstance(twist, BidirBandVelocityCommandCfg)
    self.assertEqual(twist.lin_vel_x_abs_range, (0.05, 0.085))
    self.assertEqual(twist.rel_standing_envs, 0.20)

  def test_stage2_no_wheel_pos_obs_zeroes_continuous_wheel_positions(self):
    cfg = make_hoppertrex_balance_env_cfg(
      slow_speed=True,
      speed_level=0,
      slow_speed_lin_sign=True,
      slow_speed_obs_scale=True,
      scratch_stage2_bidir_smooth_slew12=True,
      zero_wheel_joint_pos_obs=True,
    )

    actor_joint_pos = cfg.observations["actor"].terms["joint_pos"]
    critic_joint_pos = cfg.observations["critic"].terms["joint_pos"]

    self.assertIs(actor_joint_pos.func, joint_pos_rel_without_wheel_position)
    self.assertIs(critic_joint_pos.func, joint_pos_rel_without_wheel_position)
    self.assertEqual(actor_joint_pos.params["wheel_joint_names"], WHEEL_JOINT_NAMES)
    self.assertEqual(critic_joint_pos.params["wheel_joint_names"], WHEEL_JOINT_NAMES)

  def test_later_scratch_stages_drop_continuous_wheel_position_obs(self):
    task_ids = (
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_MEDIUM_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_STRONG_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE4_SMALL_LIN_SMALL_YAW_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE5_FULL_LIN_FULL_YAW_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE6_PUSH_NOISE_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE8_LEG_ASSIST_SAFE_TASK_ID,
    )

    for task_id in task_ids:
      with self.subTest(task_id=task_id):
        cfg = load_env_cfg(task_id)
        actor_joint_pos = cfg.observations["actor"].terms["joint_pos"]
        critic_joint_pos = cfg.observations["critic"].terms["joint_pos"]

        self.assertIs(actor_joint_pos.func, joint_pos_rel_without_wheel_position)
        self.assertIs(critic_joint_pos.func, joint_pos_rel_without_wheel_position)
        self.assertEqual(actor_joint_pos.params["wheel_joint_names"], WHEEL_JOINT_NAMES)
        self.assertEqual(critic_joint_pos.params["wheel_joint_names"], WHEEL_JOINT_NAMES)

  def test_stage3_yaw_only_strong_uses_stronger_yaw_authority_and_reward(self):
    cfg = load_env_cfg(hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_STRONG_TASK_ID)

    wheel_balance = cfg.actions["wheel_balance"]
    track_yaw = cfg.rewards["track_angular_velocity"]
    yaw_sign = cfg.rewards["yaw_sign_alignment"]

    self.assertEqual(wheel_balance.yaw_scale, 3.0)
    self.assertEqual(track_yaw.weight, 5.0)
    self.assertEqual(track_yaw.params["std"], 0.12)
    self.assertEqual(yaw_sign.weight, 4.0)

  def test_stage3_yaw_only_medium_sits_between_base_and_strong(self):
    cfg = load_env_cfg(hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_MEDIUM_TASK_ID)

    wheel_balance = cfg.actions["wheel_balance"]
    track_yaw = cfg.rewards["track_angular_velocity"]
    yaw_sign = cfg.rewards["yaw_sign_alignment"]

    self.assertEqual(wheel_balance.yaw_scale, 2.5)
    self.assertEqual(track_yaw.weight, 4.0)
    self.assertEqual(track_yaw.params["std"], 0.14)
    self.assertEqual(yaw_sign.weight, 3.0)


if __name__ == "__main__":
  unittest.main()
