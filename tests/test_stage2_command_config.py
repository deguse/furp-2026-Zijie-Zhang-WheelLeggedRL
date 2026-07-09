import unittest
from types import SimpleNamespace

import torch
from mjlab.tasks.registry import load_env_cfg

import hoppertrex_mjlab.tasks as hoppertrex_tasks
from hoppertrex_mjlab.tasks.hoppertrex_balance_task import (
  BidirBandVelocityCommandCfg,
  CommandFeedforwardCoupledWheelVelocityActionCfg,
  WHEEL_JOINT_NAMES,
  apply_action_ema,
  joint_pos_rel_without_wheel_position,
  lin_velocity_band_l2,
  lin_velocity_delta_l2,
  low_speed_lin_overspeed_l2,
  safe_posture_lin_velocity_band_l2,
  safe_posture_lin_velocity_delta_l2,
  safe_posture_low_speed_lin_overspeed_l2,
  make_hoppertrex_balance_env_cfg,
  yaw_velocity_band_l2,
  yaw_velocity_error_l2,
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

  def test_stage2_slew6_band_repair_targets_sustained_overspeed_and_jitter(self):
    cfg = load_env_cfg(
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE2_BIDIR_LIN_SMOOTH_SLEW6_BAND_TASK_ID
    )

    wheel_balance = cfg.actions["wheel_balance"]
    twist = cfg.commands["twist"]
    lin_band = cfg.rewards["lin_velocity_band_l2"]
    wheel_rate = cfg.rewards["wheel_target_rate_l2"]
    actor_joint_pos = cfg.observations["actor"].terms["joint_pos"]
    critic_joint_pos = cfg.observations["critic"].terms["joint_pos"]

    self.assertIsInstance(twist, BidirBandVelocityCommandCfg)
    self.assertEqual(twist.lin_vel_x_abs_range, (0.05, 0.085))
    self.assertEqual(twist.rel_standing_envs, 0.20)
    self.assertEqual(wheel_balance.target_slew_limit, 6.0)
    self.assertEqual(lin_band.weight, -30.0)
    self.assertEqual(lin_band.params["lower_fraction"], 0.5)
    self.assertEqual(lin_band.params["upper_fraction"], 1.5)
    self.assertEqual(lin_band.params["under_scale"], 1.0)
    self.assertEqual(lin_band.params["over_scale"], 4.0)
    self.assertEqual(wheel_rate.weight, -1.0e-3)
    self.assertIs(actor_joint_pos.func, joint_pos_rel_without_wheel_position)
    self.assertIs(critic_joint_pos.func, joint_pos_rel_without_wheel_position)

  def test_stage2_slew6_band_repair_requires_bidirectional_slow_speed(self):
    with self.assertRaisesRegex(ValueError, "scratch stage2 bidirectional variants"):
      make_hoppertrex_balance_env_cfg(
        scratch_stage2_bidir_smooth_slew6_band=True,
      )

  def test_stage2_repair_matrix_variants_target_distinct_failure_modes(self):
    variants = (
      (
        hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE2_BIDIR_LIN_SMOOTH_SLEW6_BAND_DELTA_TASK_ID,
        True,
        4.0,
      ),
      (
        hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE2_BIDIR_LIN_SMOOTH_SLEW6_BAND_OVER8_TASK_ID,
        False,
        8.0,
      ),
      (
        hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE2_BIDIR_LIN_SMOOTH_SLEW6_BAND_DELTA_OVER8_TASK_ID,
        True,
        8.0,
      ),
    )

    for task_id, expect_delta, over_scale in variants:
      with self.subTest(task_id=task_id):
        cfg = load_env_cfg(task_id)
        wheel_balance = cfg.actions["wheel_balance"]
        twist = cfg.commands["twist"]
        lin_band = cfg.rewards["lin_velocity_band_l2"]
        wheel_rate = cfg.rewards["wheel_target_rate_l2"]
        actor_joint_pos = cfg.observations["actor"].terms["joint_pos"]
        critic_joint_pos = cfg.observations["critic"].terms["joint_pos"]

        self.assertIsInstance(twist, BidirBandVelocityCommandCfg)
        self.assertEqual(twist.lin_vel_x_abs_range, (0.05, 0.085))
        self.assertEqual(wheel_balance.target_slew_limit, 6.0)
        self.assertEqual(lin_band.params["over_scale"], over_scale)
        self.assertEqual(wheel_rate.weight, -1.0e-3)
        if expect_delta:
          lin_delta = cfg.rewards["lin_velocity_delta_l2"]
          self.assertEqual(lin_delta.weight, -8.0)
          self.assertEqual(lin_delta.params["deadband"], 0.01)
        else:
          self.assertNotIn("lin_velocity_delta_l2", cfg.rewards)
        self.assertIs(actor_joint_pos.func, joint_pos_rel_without_wheel_position)
        self.assertIs(critic_joint_pos.func, joint_pos_rel_without_wheel_position)

  def test_stage2_slew6_norm_repair_uses_normalized_velocity_constraints(self):
    cfg = load_env_cfg(
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE2_BIDIR_LIN_SMOOTH_SLEW6_NORM_TASK_ID
    )

    wheel_balance = cfg.actions["wheel_balance"]
    twist = cfg.commands["twist"]
    lin_band = cfg.rewards["lin_velocity_band_l2"]
    lin_delta = cfg.rewards["lin_velocity_delta_l2"]
    wheel_rate = cfg.rewards["wheel_target_rate_l2"]
    actor_joint_pos = cfg.observations["actor"].terms["joint_pos"]
    critic_joint_pos = cfg.observations["critic"].terms["joint_pos"]

    self.assertIsInstance(twist, BidirBandVelocityCommandCfg)
    self.assertEqual(twist.lin_vel_x_abs_range, (0.05, 0.085))
    self.assertEqual(twist.rel_standing_envs, 0.20)
    self.assertEqual(cfg.episode_length_s, 60.0)
    self.assertEqual(wheel_balance.target_slew_limit, 6.0)
    self.assertEqual(lin_band.weight, -4.0)
    self.assertTrue(lin_band.params["normalize_by_command"])
    self.assertEqual(lin_band.params["lower_fraction"], 0.5)
    self.assertEqual(lin_band.params["upper_fraction"], 1.5)
    self.assertEqual(lin_band.params["under_scale"], 1.0)
    self.assertEqual(lin_band.params["over_scale"], 4.0)
    self.assertEqual(lin_delta.weight, -0.75)
    self.assertTrue(lin_delta.params["normalize_by_command"])
    self.assertEqual(wheel_rate.weight, -1.0e-3)
    self.assertIs(actor_joint_pos.func, joint_pos_rel_without_wheel_position)
    self.assertIs(critic_joint_pos.func, joint_pos_rel_without_wheel_position)

  def test_stage2_slew6_norm_acc_repair_targets_policy_action_chatter(self):
    cfg = load_env_cfg(
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE2_BIDIR_LIN_SMOOTH_SLEW6_NORM_ACC_TASK_ID
    )

    wheel_balance = cfg.actions["wheel_balance"]
    twist = cfg.commands["twist"]
    lin_band = cfg.rewards["lin_velocity_band_l2"]
    lin_delta = cfg.rewards["lin_velocity_delta_l2"]
    action_acc = cfg.rewards["action_acc_l2"]
    wheel_rate = cfg.rewards["wheel_target_rate_l2"]
    actor_joint_pos = cfg.observations["actor"].terms["joint_pos"]
    critic_joint_pos = cfg.observations["critic"].terms["joint_pos"]

    self.assertIsInstance(twist, BidirBandVelocityCommandCfg)
    self.assertEqual(twist.lin_vel_x_abs_range, (0.05, 0.085))
    self.assertEqual(twist.rel_standing_envs, 0.20)
    self.assertEqual(cfg.episode_length_s, 60.0)
    self.assertEqual(wheel_balance.target_slew_limit, 6.0)
    self.assertEqual(lin_band.weight, -4.0)
    self.assertTrue(lin_band.params["normalize_by_command"])
    self.assertEqual(lin_delta.weight, -0.75)
    self.assertTrue(lin_delta.params["normalize_by_command"])
    self.assertEqual(action_acc.weight, -0.08)
    self.assertEqual(wheel_rate.weight, -1.0e-3)
    self.assertIs(actor_joint_pos.func, joint_pos_rel_without_wheel_position)
    self.assertIs(critic_joint_pos.func, joint_pos_rel_without_wheel_position)

  def test_stage2_slew6_reward_balance_prioritizes_tracking_over_sign(self):
    cfg = load_env_cfg(
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE2_BIDIR_LIN_SMOOTH_SLEW6_REWARD_BALANCE_TASK_ID
    )

    wheel_balance = cfg.actions["wheel_balance"]
    twist = cfg.commands["twist"]
    track_lin = cfg.rewards["track_linear_velocity"]
    lin_sign = cfg.rewards["lin_vel_x_sign_alignment"]
    lin_band = cfg.rewards["lin_velocity_band_l2"]
    lin_delta = cfg.rewards["lin_velocity_delta_l2"]
    overspeed = cfg.rewards["low_speed_lin_overspeed_l2"]
    action_acc = cfg.rewards["action_acc_l2"]
    wheel_rate = cfg.rewards["wheel_target_rate_l2"]
    actor_joint_pos = cfg.observations["actor"].terms["joint_pos"]
    critic_joint_pos = cfg.observations["critic"].terms["joint_pos"]

    self.assertIsInstance(twist, BidirBandVelocityCommandCfg)
    self.assertEqual(twist.lin_vel_x_abs_range, (0.05, 0.085))
    self.assertEqual(twist.rel_standing_envs, 0.20)
    self.assertEqual(cfg.episode_length_s, 60.0)
    self.assertEqual(wheel_balance.target_slew_limit, 6.0)
    self.assertEqual(wheel_balance.balance_smoothing_alpha, 0.65)
    self.assertEqual(track_lin.weight, 4.0)
    self.assertEqual(lin_sign.weight, 2.5)
    self.assertEqual(lin_band.weight, -6.0)
    self.assertTrue(lin_band.params["normalize_by_command"])
    self.assertEqual(lin_delta.weight, -2.25)
    self.assertTrue(lin_delta.params["normalize_by_command"])
    self.assertEqual(overspeed.weight, -2.0)
    self.assertTrue(overspeed.params["normalize_by_command"])
    self.assertEqual(overspeed.params["margin"], 0.005)
    self.assertEqual(overspeed.params["max_command_abs"], 0.12)
    self.assertEqual(action_acc.weight, -0.08)
    self.assertEqual(wheel_rate.weight, -1.0e-3)
    self.assertIs(actor_joint_pos.func, joint_pos_rel_without_wheel_position)
    self.assertIs(critic_joint_pos.func, joint_pos_rel_without_wheel_position)

  def test_stage2_slew6_reward_balance_precision_targets_sustained_low_speed_pulsing(self):
    cfg = load_env_cfg(
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE2_BIDIR_LIN_SMOOTH_SLEW6_REWARD_BALANCE_PRECISION_TASK_ID
    )

    wheel_balance = cfg.actions["wheel_balance"]
    twist = cfg.commands["twist"]
    track_lin = cfg.rewards["track_linear_velocity"]
    lin_sign = cfg.rewards["lin_vel_x_sign_alignment"]
    lin_band = cfg.rewards["lin_velocity_band_l2"]
    lin_delta = cfg.rewards["lin_velocity_delta_l2"]
    overspeed = cfg.rewards["low_speed_lin_overspeed_l2"]
    action_acc = cfg.rewards["action_acc_l2"]
    wheel_rate = cfg.rewards["wheel_target_rate_l2"]
    actor_joint_pos = cfg.observations["actor"].terms["joint_pos"]
    critic_joint_pos = cfg.observations["critic"].terms["joint_pos"]

    self.assertIsInstance(twist, BidirBandVelocityCommandCfg)
    self.assertEqual(twist.lin_vel_x_abs_range, (0.05, 0.085))
    self.assertEqual(twist.rel_standing_envs, 0.20)
    self.assertEqual(cfg.episode_length_s, 60.0)
    self.assertEqual(twist.resampling_time_range, (30.0, 60.0))
    self.assertEqual(wheel_balance.target_slew_limit, 6.0)
    self.assertEqual(wheel_balance.balance_smoothing_alpha, 0.65)
    self.assertEqual(track_lin.weight, 4.0)
    self.assertEqual(lin_sign.weight, 1.0)
    self.assertEqual(lin_band.weight, -8.0)
    self.assertTrue(lin_band.params["normalize_by_command"])
    self.assertEqual(lin_delta.weight, -5.0)
    self.assertTrue(lin_delta.params["normalize_by_command"])
    self.assertEqual(overspeed.weight, -4.0)
    self.assertTrue(overspeed.params["normalize_by_command"])
    self.assertEqual(overspeed.params["margin"], 0.0)
    self.assertEqual(overspeed.params["max_command_abs"], 0.12)
    self.assertEqual(action_acc.weight, -0.08)
    self.assertEqual(wheel_rate.weight, -1.0e-3)
    self.assertIs(actor_joint_pos.func, joint_pos_rel_without_wheel_position)
    self.assertIs(critic_joint_pos.func, joint_pos_rel_without_wheel_position)

  def test_stage2_slew6_reward_balance_precision_tight_band_aligns_training_with_promotion_speed(self):
    cfg = load_env_cfg(
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE2_BIDIR_LIN_SMOOTH_SLEW6_REWARD_BALANCE_PRECISION_TIGHT_BAND_TASK_ID
    )

    wheel_balance = cfg.actions["wheel_balance"]
    lin_band = cfg.rewards["lin_velocity_band_l2"]
    lin_delta = cfg.rewards["lin_velocity_delta_l2"]
    overspeed = cfg.rewards["low_speed_lin_overspeed_l2"]
    lin_sign = cfg.rewards["lin_vel_x_sign_alignment"]
    actor_joint_pos = cfg.observations["actor"].terms["joint_pos"]
    critic_joint_pos = cfg.observations["critic"].terms["joint_pos"]

    self.assertEqual(wheel_balance.target_slew_limit, 6.0)
    self.assertEqual(wheel_balance.balance_smoothing_alpha, 0.65)
    self.assertEqual(lin_sign.weight, 1.0)
    self.assertEqual(lin_band.weight, -8.0)
    self.assertEqual(lin_band.params["lower_fraction"], 0.75)
    self.assertEqual(lin_band.params["upper_fraction"], 1.25)
    self.assertTrue(lin_band.params["normalize_by_command"])
    self.assertEqual(lin_delta.weight, -5.0)
    self.assertTrue(lin_delta.params["normalize_by_command"])
    self.assertEqual(overspeed.weight, -4.0)
    self.assertEqual(overspeed.params["margin"], 0.0)
    self.assertTrue(overspeed.params["normalize_by_command"])
    self.assertIs(actor_joint_pos.func, joint_pos_rel_without_wheel_position)
    self.assertIs(critic_joint_pos.func, joint_pos_rel_without_wheel_position)

  def test_stage2_slew6_reward_balance_tight_targets_low_speed_pulsing(self):
    cfg = load_env_cfg(
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE2_BIDIR_LIN_SMOOTH_SLEW6_REWARD_BALANCE_TIGHT_TASK_ID
    )

    wheel_balance = cfg.actions["wheel_balance"]
    twist = cfg.commands["twist"]
    track_lin = cfg.rewards["track_linear_velocity"]
    lin_sign = cfg.rewards["lin_vel_x_sign_alignment"]
    lin_band = cfg.rewards["lin_velocity_band_l2"]
    lin_delta = cfg.rewards["lin_velocity_delta_l2"]
    overspeed = cfg.rewards["low_speed_lin_overspeed_l2"]
    action_acc = cfg.rewards["action_acc_l2"]
    wheel_rate = cfg.rewards["wheel_target_rate_l2"]
    actor_joint_pos = cfg.observations["actor"].terms["joint_pos"]
    critic_joint_pos = cfg.observations["critic"].terms["joint_pos"]

    self.assertIsInstance(twist, BidirBandVelocityCommandCfg)
    self.assertEqual(twist.lin_vel_x_abs_range, (0.05, 0.085))
    self.assertEqual(twist.rel_standing_envs, 0.20)
    self.assertEqual(cfg.episode_length_s, 60.0)
    self.assertEqual(twist.resampling_time_range, (30.0, 60.0))
    self.assertEqual(wheel_balance.target_slew_limit, 5.0)
    self.assertEqual(wheel_balance.balance_smoothing_alpha, 0.75)
    self.assertEqual(track_lin.weight, 4.0)
    self.assertEqual(lin_sign.weight, 1.5)
    self.assertEqual(lin_band.weight, -10.0)
    self.assertTrue(lin_band.params["normalize_by_command"])
    self.assertEqual(lin_delta.weight, -4.0)
    self.assertTrue(lin_delta.params["normalize_by_command"])
    self.assertEqual(overspeed.weight, -4.0)
    self.assertTrue(overspeed.params["normalize_by_command"])
    self.assertEqual(overspeed.params["margin"], 0.0)
    self.assertEqual(overspeed.params["max_command_abs"], 0.12)
    self.assertEqual(action_acc.weight, -0.12)
    self.assertEqual(wheel_rate.weight, -1.5e-3)
    self.assertIs(actor_joint_pos.func, joint_pos_rel_without_wheel_position)
    self.assertIs(critic_joint_pos.func, joint_pos_rel_without_wheel_position)

  def test_stage2_slew6_reward_balance_moderate_keeps_stability_authority(self):
    cfg = load_env_cfg(
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE2_BIDIR_LIN_SMOOTH_SLEW6_REWARD_BALANCE_MODERATE_TASK_ID
    )

    wheel_balance = cfg.actions["wheel_balance"]
    twist = cfg.commands["twist"]
    track_lin = cfg.rewards["track_linear_velocity"]
    lin_sign = cfg.rewards["lin_vel_x_sign_alignment"]
    lin_band = cfg.rewards["lin_velocity_band_l2"]
    lin_delta = cfg.rewards["lin_velocity_delta_l2"]
    overspeed = cfg.rewards["low_speed_lin_overspeed_l2"]
    action_acc = cfg.rewards["action_acc_l2"]
    wheel_rate = cfg.rewards["wheel_target_rate_l2"]
    actor_joint_pos = cfg.observations["actor"].terms["joint_pos"]
    critic_joint_pos = cfg.observations["critic"].terms["joint_pos"]

    self.assertIsInstance(twist, BidirBandVelocityCommandCfg)
    self.assertEqual(twist.lin_vel_x_abs_range, (0.05, 0.085))
    self.assertEqual(twist.rel_standing_envs, 0.20)
    self.assertEqual(cfg.episode_length_s, 60.0)
    self.assertEqual(twist.resampling_time_range, (30.0, 60.0))
    self.assertEqual(wheel_balance.target_slew_limit, 6.0)
    self.assertEqual(wheel_balance.balance_smoothing_alpha, 0.65)
    self.assertEqual(track_lin.weight, 4.0)
    self.assertEqual(lin_sign.weight, 2.0)
    self.assertEqual(lin_band.weight, -7.0)
    self.assertTrue(lin_band.params["normalize_by_command"])
    self.assertEqual(lin_delta.weight, -3.0)
    self.assertTrue(lin_delta.params["normalize_by_command"])
    self.assertEqual(overspeed.weight, -3.0)
    self.assertTrue(overspeed.params["normalize_by_command"])
    self.assertEqual(overspeed.params["margin"], 0.002)
    self.assertEqual(overspeed.params["max_command_abs"], 0.12)
    self.assertEqual(action_acc.weight, -0.10)
    self.assertEqual(wheel_rate.weight, -1.2e-3)
    self.assertIs(actor_joint_pos.func, joint_pos_rel_without_wheel_position)
    self.assertIs(critic_joint_pos.func, joint_pos_rel_without_wheel_position)

  def test_stage2_slew6_feedforward_residual_splits_speed_bias_from_balance(self):
    cfg = load_env_cfg(
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE2_BIDIR_LIN_SMOOTH_SLEW6_REWARD_BALANCE_FF_TASK_ID
    )

    wheel_balance = cfg.actions["wheel_balance"]
    twist = cfg.commands["twist"]
    track_lin = cfg.rewards["track_linear_velocity"]
    lin_sign = cfg.rewards["lin_vel_x_sign_alignment"]
    lin_band = cfg.rewards["lin_velocity_band_l2"]
    lin_delta = cfg.rewards["lin_velocity_delta_l2"]
    overspeed = cfg.rewards["low_speed_lin_overspeed_l2"]
    action_acc = cfg.rewards["action_acc_l2"]
    wheel_rate = cfg.rewards["wheel_target_rate_l2"]
    actor_joint_pos = cfg.observations["actor"].terms["joint_pos"]
    critic_joint_pos = cfg.observations["critic"].terms["joint_pos"]

    self.assertIsInstance(twist, BidirBandVelocityCommandCfg)
    self.assertIsInstance(wheel_balance, CommandFeedforwardCoupledWheelVelocityActionCfg)
    self.assertEqual(twist.lin_vel_x_abs_range, (0.05, 0.085))
    self.assertEqual(twist.rel_standing_envs, 0.20)
    self.assertEqual(cfg.episode_length_s, 60.0)
    self.assertEqual(twist.resampling_time_range, (30.0, 60.0))
    self.assertEqual(wheel_balance.target_slew_limit, 6.0)
    self.assertEqual(wheel_balance.balance_smoothing_alpha, 0.65)
    self.assertEqual(wheel_balance.residual_scale, 0.15)
    self.assertEqual(wheel_balance.command_gain, 2.0)
    self.assertEqual(wheel_balance.feedforward_clip, 0.25)
    self.assertEqual(track_lin.weight, 4.0)
    self.assertEqual(lin_sign.weight, 2.5)
    self.assertEqual(lin_band.weight, -6.0)
    self.assertTrue(lin_band.params["normalize_by_command"])
    self.assertEqual(lin_delta.weight, -2.25)
    self.assertTrue(lin_delta.params["normalize_by_command"])
    self.assertEqual(overspeed.weight, -2.0)
    self.assertTrue(overspeed.params["normalize_by_command"])
    self.assertEqual(overspeed.params["margin"], 0.005)
    self.assertEqual(overspeed.params["max_command_abs"], 0.12)
    self.assertEqual(action_acc.weight, -0.08)
    self.assertEqual(wheel_rate.weight, -1.0e-3)
    self.assertIs(actor_joint_pos.func, joint_pos_rel_without_wheel_position)
    self.assertIs(critic_joint_pos.func, joint_pos_rel_without_wheel_position)

  def test_stage2_slew6_feedforward_tiny_residual_limits_policy_override(self):
    cfg = load_env_cfg(
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE2_BIDIR_LIN_SMOOTH_SLEW6_REWARD_BALANCE_FF_TINY_TASK_ID
    )

    wheel_balance = cfg.actions["wheel_balance"]
    twist = cfg.commands["twist"]
    track_lin = cfg.rewards["track_linear_velocity"]
    lin_sign = cfg.rewards["lin_vel_x_sign_alignment"]
    lin_band = cfg.rewards["lin_velocity_band_l2"]
    lin_delta = cfg.rewards["lin_velocity_delta_l2"]
    overspeed = cfg.rewards["low_speed_lin_overspeed_l2"]
    action_acc = cfg.rewards["action_acc_l2"]
    wheel_rate = cfg.rewards["wheel_target_rate_l2"]

    self.assertIsInstance(twist, BidirBandVelocityCommandCfg)
    self.assertIsInstance(wheel_balance, CommandFeedforwardCoupledWheelVelocityActionCfg)
    self.assertEqual(twist.lin_vel_x_abs_range, (0.05, 0.085))
    self.assertEqual(twist.rel_standing_envs, 0.20)
    self.assertEqual(cfg.episode_length_s, 60.0)
    self.assertEqual(twist.resampling_time_range, (30.0, 60.0))
    self.assertEqual(wheel_balance.target_slew_limit, 6.0)
    self.assertEqual(wheel_balance.balance_smoothing_alpha, 0.65)
    self.assertEqual(wheel_balance.residual_scale, 0.05)
    self.assertEqual(wheel_balance.command_gain, 2.0)
    self.assertEqual(wheel_balance.feedforward_clip, 0.25)
    self.assertEqual(track_lin.weight, 4.0)
    self.assertEqual(lin_sign.weight, 2.5)
    self.assertEqual(lin_band.weight, -6.0)
    self.assertTrue(lin_band.params["normalize_by_command"])
    self.assertEqual(lin_delta.weight, -2.25)
    self.assertTrue(lin_delta.params["normalize_by_command"])
    self.assertEqual(overspeed.weight, -2.0)
    self.assertTrue(overspeed.params["normalize_by_command"])
    self.assertEqual(overspeed.params["margin"], 0.005)
    self.assertEqual(action_acc.weight, -0.08)
    self.assertEqual(wheel_rate.weight, -1.0e-3)

  def test_stage2_slew6_reward_balance_guarded_shapes_velocity_only_when_safe(self):
    cfg = load_env_cfg(
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE2_BIDIR_LIN_SMOOTH_SLEW6_REWARD_BALANCE_GUARDED_TASK_ID
    )

    wheel_balance = cfg.actions["wheel_balance"]
    twist = cfg.commands["twist"]
    lin_sign = cfg.rewards["lin_vel_x_sign_alignment"]
    lin_band = cfg.rewards["safe_posture_lin_velocity_band_l2"]
    lin_delta = cfg.rewards["safe_posture_lin_velocity_delta_l2"]
    overspeed = cfg.rewards["safe_posture_low_speed_lin_overspeed_l2"]
    action_acc = cfg.rewards["action_acc_l2"]
    wheel_rate = cfg.rewards["wheel_target_rate_l2"]

    self.assertIsInstance(twist, BidirBandVelocityCommandCfg)
    self.assertEqual(twist.lin_vel_x_abs_range, (0.05, 0.085))
    self.assertEqual(twist.rel_standing_envs, 0.20)
    self.assertEqual(cfg.episode_length_s, 60.0)
    self.assertEqual(twist.resampling_time_range, (30.0, 60.0))
    self.assertEqual(wheel_balance.target_slew_limit, 6.0)
    self.assertEqual(wheel_balance.balance_smoothing_alpha, 0.65)
    self.assertEqual(lin_sign.weight, 2.0)
    self.assertEqual(lin_band.weight, -7.0)
    self.assertTrue(lin_band.params["normalize_by_command"])
    self.assertEqual(lin_band.params["pitch_abs_limit"], 0.08)
    self.assertEqual(lin_band.params["pitch_rate_abs_limit"], 0.8)
    self.assertEqual(lin_delta.weight, -3.0)
    self.assertTrue(lin_delta.params["normalize_by_command"])
    self.assertEqual(overspeed.weight, -3.0)
    self.assertTrue(overspeed.params["normalize_by_command"])
    self.assertEqual(overspeed.params["margin"], 0.002)
    self.assertEqual(action_acc.weight, -0.08)
    self.assertEqual(wheel_rate.weight, -1.0e-3)

  def test_later_linear_yaw_stages_smooth_balance_channel(self):
    task_ids = (
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE4_SMALL_LIN_SMALL_YAW_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE5_FULL_LIN_FULL_YAW_TASK_ID,
    )

    for task_id in task_ids:
      with self.subTest(task_id=task_id):
        cfg = load_env_cfg(task_id)
        wheel_balance = cfg.actions["wheel_balance"]
        twist = cfg.commands["twist"]

        self.assertEqual(wheel_balance.balance_smoothing_alpha, 0.65)
        self.assertEqual(wheel_balance.yaw_smoothing_alpha, 0.50)
        self.assertEqual(cfg.episode_length_s, 60.0)
        self.assertEqual(twist.resampling_time_range, (30.0, 60.0))

  def test_stage3_yaw_scratch_stages_use_sustained_training_window(self):
    task_ids = (
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_MEDIUM_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_MEDIUM_ALIGNED_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_MEDIUM_ALIGNED_LITE_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_MEDIUM_ALIGNED_SMOOTH_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_MEDIUM_ALIGNED_NARROW_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_MEDIUM_ALIGNED_ANTIPULSE_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_MEDIUM_ALIGNED_ANTIPULSE_V2_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_TRACK_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_TRACK_V2_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_STRONG_TASK_ID,
    )

    for task_id in task_ids:
      with self.subTest(task_id=task_id):
        cfg = load_env_cfg(task_id)
        twist = cfg.commands["twist"]

        self.assertEqual(cfg.episode_length_s, 60.0)
        self.assertEqual(twist.resampling_time_range, (30.0, 60.0))

  def test_stage2_slew6_norm_repair_requires_bidirectional_slow_speed(self):
    with self.assertRaisesRegex(ValueError, "scratch stage2 bidirectional variants"):
      make_hoppertrex_balance_env_cfg(
        scratch_stage2_bidir_smooth_slew6_norm=True,
      )

  def test_stage2_slew6_reward_balance_requires_bidirectional_slow_speed(self):
    with self.assertRaisesRegex(ValueError, "scratch stage2 bidirectional variants"):
      make_hoppertrex_balance_env_cfg(
        scratch_stage2_bidir_smooth_slew6_reward_balance=True,
      )

  def test_later_scratch_stages_drop_continuous_wheel_position_obs(self):
    task_ids = (
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_MEDIUM_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_MEDIUM_ALIGNED_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_MEDIUM_ALIGNED_LITE_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_MEDIUM_ALIGNED_SMOOTH_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_MEDIUM_ALIGNED_NARROW_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_MEDIUM_ALIGNED_ANTIPULSE_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_MEDIUM_ALIGNED_ANTIPULSE_V2_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_TRACK_TASK_ID,
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_TRACK_V2_TASK_ID,
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

  def test_stage3_yaw_only_medium_aligned_keeps_drive_without_trackv2_damping(self):
    cfg = load_env_cfg(
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_MEDIUM_ALIGNED_TASK_ID
    )

    wheel_balance = cfg.actions["wheel_balance"]
    track_yaw = cfg.rewards["track_angular_velocity"]
    yaw_sign = cfg.rewards["yaw_sign_alignment"]
    action_rate = cfg.rewards["action_rate_l2"]
    twist = cfg.commands["twist"]

    self.assertEqual(twist.yaw_abs, 0.07)
    self.assertEqual(twist.ranges.ang_vel_z, (-0.07, 0.07))
    self.assertEqual(wheel_balance.yaw_scale, 2.5)
    self.assertIsNone(wheel_balance.target_slew_limit)
    self.assertEqual(track_yaw.weight, 4.0)
    self.assertEqual(track_yaw.params["std"], 0.14)
    self.assertEqual(yaw_sign.weight, 3.0)
    self.assertEqual(action_rate.weight, -0.006)
    self.assertNotIn("yaw_velocity_error_l2", cfg.rewards)
    self.assertNotIn("wheel_target_rate_l2", cfg.rewards)

  def test_stage3_yaw_only_medium_aligned_lite_reduces_pulsed_turning(self):
    cfg = load_env_cfg(
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_MEDIUM_ALIGNED_LITE_TASK_ID
    )

    wheel_balance = cfg.actions["wheel_balance"]
    track_yaw = cfg.rewards["track_angular_velocity"]
    yaw_sign = cfg.rewards["yaw_sign_alignment"]
    yaw_error = cfg.rewards["yaw_velocity_error_l2"]
    action_rate = cfg.rewards["action_rate_l2"]
    twist = cfg.commands["twist"]

    self.assertEqual(twist.yaw_abs, 0.07)
    self.assertEqual(twist.ranges.ang_vel_z, (-0.07, 0.07))
    self.assertEqual(wheel_balance.yaw_scale, 2.1)
    self.assertEqual(wheel_balance.target_slew_limit, 12.0)
    self.assertEqual(track_yaw.weight, 4.0)
    self.assertEqual(track_yaw.params["std"], 0.12)
    self.assertEqual(yaw_sign.weight, 2.0)
    self.assertEqual(yaw_error.weight, -8.0)
    self.assertEqual(action_rate.weight, -0.006)
    self.assertNotIn("wheel_target_rate_l2", cfg.rewards)

  def test_stage3_yaw_only_medium_aligned_smooth_targets_yaw_channel_jitter(self):
    cfg = load_env_cfg(
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_MEDIUM_ALIGNED_SMOOTH_TASK_ID
    )

    wheel_balance = cfg.actions["wheel_balance"]
    track_yaw = cfg.rewards["track_angular_velocity"]
    yaw_sign = cfg.rewards["yaw_sign_alignment"]
    yaw_error = cfg.rewards["yaw_velocity_error_l2"]
    effective_yaw_rate = cfg.rewards["effective_yaw_rate_l2"]
    action_rate = cfg.rewards["action_rate_l2"]
    twist = cfg.commands["twist"]

    self.assertEqual(twist.yaw_abs, 0.07)
    self.assertEqual(twist.ranges.ang_vel_z, (-0.07, 0.07))
    self.assertEqual(wheel_balance.yaw_scale, 2.1)
    self.assertEqual(wheel_balance.yaw_smoothing_alpha, 0.50)
    self.assertEqual(wheel_balance.target_slew_limit, 12.0)
    self.assertEqual(track_yaw.weight, 4.0)
    self.assertEqual(track_yaw.params["std"], 0.12)
    self.assertEqual(yaw_sign.weight, 2.0)
    self.assertEqual(yaw_error.weight, -8.0)
    self.assertEqual(effective_yaw_rate.weight, -0.01)
    self.assertEqual(action_rate.weight, -0.006)
    self.assertNotIn("wheel_target_rate_l2", cfg.rewards)

  def test_stage3_yaw_only_medium_aligned_narrow_tightens_yaw_tracking(self):
    cfg = load_env_cfg(
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_MEDIUM_ALIGNED_NARROW_TASK_ID
    )

    wheel_balance = cfg.actions["wheel_balance"]
    track_yaw = cfg.rewards["track_angular_velocity"]
    yaw_sign = cfg.rewards["yaw_sign_alignment"]
    yaw_error = cfg.rewards["yaw_velocity_error_l2"]
    action_rate = cfg.rewards["action_rate_l2"]
    twist = cfg.commands["twist"]

    self.assertEqual(twist.yaw_abs, 0.07)
    self.assertEqual(twist.ranges.ang_vel_z, (-0.07, 0.07))
    self.assertEqual(wheel_balance.yaw_scale, 2.1)
    self.assertIsNone(wheel_balance.yaw_smoothing_alpha)
    self.assertEqual(wheel_balance.target_slew_limit, 12.0)
    self.assertEqual(track_yaw.weight, 4.0)
    self.assertEqual(track_yaw.params["std"], 0.07)
    self.assertEqual(yaw_sign.weight, 1.0)
    self.assertEqual(yaw_error.weight, -12.0)
    self.assertEqual(action_rate.weight, -0.006)
    self.assertNotIn("effective_yaw_rate_l2", cfg.rewards)
    self.assertNotIn("wheel_target_rate_l2", cfg.rewards)

  def test_stage3_yaw_only_medium_aligned_antipulse_aligns_reward_with_gate_band(self):
    cfg = load_env_cfg(
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_MEDIUM_ALIGNED_ANTIPULSE_TASK_ID
    )

    wheel_balance = cfg.actions["wheel_balance"]
    track_yaw = cfg.rewards["track_angular_velocity"]
    yaw_sign = cfg.rewards["yaw_sign_alignment"]
    yaw_error = cfg.rewards["yaw_velocity_error_l2"]
    yaw_band = cfg.rewards["yaw_velocity_band_l2"]
    action_rate = cfg.rewards["action_rate_l2"]
    twist = cfg.commands["twist"]

    self.assertEqual(twist.yaw_abs, 0.07)
    self.assertEqual(twist.ranges.ang_vel_z, (-0.07, 0.07))
    self.assertEqual(wheel_balance.yaw_scale, 2.1)
    self.assertIsNone(wheel_balance.yaw_smoothing_alpha)
    self.assertEqual(wheel_balance.target_slew_limit, 12.0)
    self.assertEqual(track_yaw.weight, 4.0)
    self.assertEqual(track_yaw.params["std"], 0.07)
    self.assertEqual(yaw_sign.weight, 1.0)
    self.assertEqual(yaw_error.weight, -12.0)
    self.assertEqual(yaw_band.weight, -24.0)
    self.assertEqual(yaw_band.params["lower_fraction"], 0.5)
    self.assertEqual(yaw_band.params["upper_fraction"], 1.5)
    self.assertEqual(yaw_band.params["over_scale"], 1.5)
    self.assertEqual(action_rate.weight, -0.006)
    self.assertNotIn("effective_yaw_rate_l2", cfg.rewards)
    self.assertNotIn("wheel_target_rate_l2", cfg.rewards)

  def test_stage3_yaw_only_medium_aligned_antipulse_v2_prioritizes_slow_yaw_recovery(self):
    cfg = load_env_cfg(
      hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_MEDIUM_ALIGNED_ANTIPULSE_V2_TASK_ID
    )

    wheel_balance = cfg.actions["wheel_balance"]
    track_yaw = cfg.rewards["track_angular_velocity"]
    yaw_sign = cfg.rewards["yaw_sign_alignment"]
    yaw_error = cfg.rewards["yaw_velocity_error_l2"]
    yaw_band = cfg.rewards["yaw_velocity_band_l2"]
    action_rate = cfg.rewards["action_rate_l2"]
    twist = cfg.commands["twist"]

    self.assertEqual(twist.yaw_abs, 0.07)
    self.assertEqual(twist.ranges.ang_vel_z, (-0.07, 0.07))
    self.assertEqual(wheel_balance.yaw_scale, 2.1)
    self.assertIsNone(wheel_balance.yaw_smoothing_alpha)
    self.assertEqual(wheel_balance.target_slew_limit, 12.0)
    self.assertEqual(track_yaw.weight, 4.0)
    self.assertEqual(track_yaw.params["std"], 0.07)
    self.assertEqual(yaw_sign.weight, 1.0)
    self.assertEqual(yaw_error.weight, -12.0)
    self.assertEqual(yaw_band.weight, -24.0)
    self.assertEqual(yaw_band.params["lower_fraction"], 0.5)
    self.assertEqual(yaw_band.params["upper_fraction"], 1.5)
    self.assertEqual(yaw_band.params["under_scale"], 4.0)
    self.assertEqual(yaw_band.params["over_scale"], 1.0)
    self.assertEqual(action_rate.weight, -0.006)
    self.assertNotIn("effective_yaw_rate_l2", cfg.rewards)
    self.assertNotIn("wheel_target_rate_l2", cfg.rewards)

  def test_stage3_yaw_only_track_prioritizes_tracking_over_sign_reward(self):
    cfg = load_env_cfg(hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_TRACK_TASK_ID)

    wheel_balance = cfg.actions["wheel_balance"]
    track_yaw = cfg.rewards["track_angular_velocity"]
    yaw_sign = cfg.rewards["yaw_sign_alignment"]
    yaw_error = cfg.rewards["yaw_velocity_error_l2"]
    twist = cfg.commands["twist"]

    self.assertEqual(wheel_balance.yaw_scale, 2.0)
    self.assertEqual(wheel_balance.target_slew_limit, 12.0)
    self.assertEqual(twist.yaw_abs, 0.07)
    self.assertEqual(twist.ranges.ang_vel_z, (-0.07, 0.07))
    self.assertEqual(track_yaw.weight, 5.0)
    self.assertEqual(track_yaw.params["std"], 0.12)
    self.assertEqual(yaw_sign.weight, 0.5)
    self.assertEqual(yaw_error.weight, -20.0)

  def test_stage3_yaw_only_track_v2_restores_yaw_drive_with_aligned_target(self):
    cfg = load_env_cfg(hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_TRACK_V2_TASK_ID)

    wheel_balance = cfg.actions["wheel_balance"]
    track_yaw = cfg.rewards["track_angular_velocity"]
    yaw_sign = cfg.rewards["yaw_sign_alignment"]
    yaw_error = cfg.rewards["yaw_velocity_error_l2"]
    wheel_rate = cfg.rewards["wheel_target_rate_l2"]
    action_rate = cfg.rewards["action_rate_l2"]
    twist = cfg.commands["twist"]

    self.assertEqual(twist.yaw_abs, 0.07)
    self.assertEqual(twist.ranges.ang_vel_z, (-0.07, 0.07))
    self.assertEqual(wheel_balance.yaw_scale, 2.3)
    self.assertEqual(wheel_balance.target_slew_limit, 12.0)
    self.assertEqual(track_yaw.weight, 5.0)
    self.assertEqual(track_yaw.params["std"], 0.08)
    self.assertEqual(yaw_sign.weight, 1.8)
    self.assertEqual(yaw_error.weight, -40.0)
    self.assertEqual(wheel_rate.weight, -5.0e-4)
    self.assertEqual(action_rate.weight, -0.02)

  def test_yaw_velocity_error_l2_penalizes_active_yaw_error_only(self):
    command = torch.tensor(
      [
        [0.0, 0.0, 0.07],
        [0.0, 0.0, -0.07],
        [0.0, 0.0, 0.0],
      ]
    )
    actual_yaw = torch.tensor([0.10, -0.02, 0.10])
    env = SimpleNamespace(
      command_manager=SimpleNamespace(get_command=lambda _name: command),
      scene={
        "robot": SimpleNamespace(
          data=SimpleNamespace(
            root_link_ang_vel_b=torch.stack(
              [torch.zeros_like(actual_yaw), torch.zeros_like(actual_yaw), actual_yaw],
              dim=1,
            )
          )
        )
      },
    )

    penalty = yaw_velocity_error_l2(env, command_name="twist", deadband=0.01)

    torch.testing.assert_close(
      penalty,
      torch.tensor([0.0009, 0.0025, 0.0]),
    )

  def test_yaw_velocity_band_l2_penalizes_slow_wrong_and_fast_yaw_only(self):
    command = torch.tensor(
      [
        [0.0, 0.0, 0.08],
        [0.0, 0.0, 0.08],
        [0.0, 0.0, 0.08],
        [0.0, 0.0, -0.08],
        [0.0, 0.0, 0.0],
      ]
    )
    actual_yaw = torch.tensor([0.08, 0.02, 0.14, 0.02, 0.20])
    env = SimpleNamespace(
      command_manager=SimpleNamespace(get_command=lambda _name: command),
      scene={
        "robot": SimpleNamespace(
          data=SimpleNamespace(
            root_link_ang_vel_b=torch.stack(
              [torch.zeros_like(actual_yaw), torch.zeros_like(actual_yaw), actual_yaw],
              dim=1,
            )
          )
        )
      },
    )

    penalty = yaw_velocity_band_l2(
      env,
      command_name="twist",
      deadband=0.01,
      lower_fraction=0.5,
      upper_fraction=1.5,
      under_scale=1.0,
      over_scale=1.5,
    )

    torch.testing.assert_close(
      penalty,
      torch.tensor([0.0, 0.0004, 0.0006, 0.0036, 0.0]),
    )

  def test_yaw_velocity_band_l2_supports_stronger_under_speed_penalty(self):
    command = torch.tensor(
      [
        [0.0, 0.0, 0.08],
        [0.0, 0.0, 0.08],
      ]
    )
    actual_yaw = torch.tensor([0.02, 0.14])
    env = SimpleNamespace(
      command_manager=SimpleNamespace(get_command=lambda _name: command),
      scene={
        "robot": SimpleNamespace(
          data=SimpleNamespace(
            root_link_ang_vel_b=torch.stack(
              [torch.zeros_like(actual_yaw), torch.zeros_like(actual_yaw), actual_yaw],
              dim=1,
            )
          )
        )
      },
    )

    penalty = yaw_velocity_band_l2(
      env,
      command_name="twist",
      deadband=0.01,
      lower_fraction=0.5,
      upper_fraction=1.5,
      under_scale=4.0,
      over_scale=1.0,
    )

    torch.testing.assert_close(
      penalty,
      torch.tensor([0.0016, 0.0004]),
    )

  def test_lin_velocity_band_l2_penalizes_slow_wrong_and_fast_lin_x_only(self):
    command = torch.tensor(
      [
        [0.08, 0.0, 0.0],
        [0.08, 0.0, 0.0],
        [0.08, 0.0, 0.0],
        [-0.08, 0.0, 0.0],
        [0.0, 0.0, 0.0],
      ]
    )
    actual_lin_x = torch.tensor([0.08, 0.02, 0.14, 0.02, 0.20])
    env = SimpleNamespace(
      command_manager=SimpleNamespace(get_command=lambda _name: command),
      scene={
        "robot": SimpleNamespace(
          data=SimpleNamespace(
            root_link_lin_vel_b=torch.stack(
              [
                actual_lin_x,
                torch.zeros_like(actual_lin_x),
                torch.zeros_like(actual_lin_x),
              ],
              dim=1,
            )
          )
        )
      },
    )

    penalty = lin_velocity_band_l2(
      env,
      command_name="twist",
      deadband=0.01,
      lower_fraction=0.5,
      upper_fraction=1.5,
      under_scale=1.0,
      over_scale=4.0,
    )

    torch.testing.assert_close(
      penalty,
      torch.tensor([0.0, 0.0004, 0.0016, 0.0036, 0.0]),
    )

  def test_lin_velocity_band_l2_can_normalize_by_command_magnitude(self):
    command = torch.tensor(
      [
        [0.08, 0.0, 0.0],
        [0.08, 0.0, 0.0],
        [-0.08, 0.0, 0.0],
        [0.0, 0.0, 0.0],
      ]
    )
    actual_lin_x = torch.tensor([0.14, 0.02, 0.02, 0.20])
    env = SimpleNamespace(
      command_manager=SimpleNamespace(get_command=lambda _name: command),
      scene={
        "robot": SimpleNamespace(
          data=SimpleNamespace(
            root_link_lin_vel_b=torch.stack(
              [
                actual_lin_x,
                torch.zeros_like(actual_lin_x),
                torch.zeros_like(actual_lin_x),
              ],
              dim=1,
            )
          )
        )
      },
    )

    penalty = lin_velocity_band_l2(
      env,
      command_name="twist",
      deadband=0.01,
      lower_fraction=0.5,
      upper_fraction=1.5,
      under_scale=1.0,
      over_scale=4.0,
      normalize_by_command=True,
    )

    torch.testing.assert_close(
      penalty,
      torch.tensor([0.25, 0.0625, 0.5625, 0.0]),
    )

  def test_lin_velocity_delta_l2_penalizes_active_lin_x_jitter_after_first_sample(self):
    command = torch.tensor(
      [
        [0.07, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-0.07, 0.0, 0.0],
      ]
    )
    actual_lin_x = torch.tensor([0.10, 0.20, -0.10])
    data = SimpleNamespace(
      root_link_lin_vel_b=torch.stack(
        [actual_lin_x, torch.zeros_like(actual_lin_x), torch.zeros_like(actual_lin_x)],
        dim=1,
      )
    )
    env = SimpleNamespace(
      command_manager=SimpleNamespace(get_command=lambda _name: command),
      scene={"robot": SimpleNamespace(data=data)},
      episode_length_buf=torch.tensor([5, 5, 5]),
    )

    first = lin_velocity_delta_l2(env, command_name="twist", deadband=0.01)
    data.root_link_lin_vel_b[:, 0] = torch.tensor([0.12, 0.25, -0.14])
    second = lin_velocity_delta_l2(env, command_name="twist", deadband=0.01)
    data.root_link_lin_vel_b[:, 0] = torch.tensor([0.11, 0.30, -0.20])
    env.episode_length_buf = torch.tensor([6, 6, 1])
    third = lin_velocity_delta_l2(env, command_name="twist", deadband=0.01)

    torch.testing.assert_close(first, torch.zeros(3))
    torch.testing.assert_close(second, torch.tensor([0.0004, 0.0, 0.0016]))
    torch.testing.assert_close(third, torch.tensor([0.0001, 0.0, 0.0]))

  def test_lin_velocity_delta_l2_can_normalize_and_still_ignores_zero_or_reset(self):
    command = torch.tensor(
      [
        [0.08, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-0.04, 0.0, 0.0],
      ]
    )
    actual_lin_x = torch.tensor([0.10, 0.20, -0.10])
    data = SimpleNamespace(
      root_link_lin_vel_b=torch.stack(
        [actual_lin_x, torch.zeros_like(actual_lin_x), torch.zeros_like(actual_lin_x)],
        dim=1,
      )
    )
    env = SimpleNamespace(
      command_manager=SimpleNamespace(get_command=lambda _name: command),
      scene={"robot": SimpleNamespace(data=data)},
      episode_length_buf=torch.tensor([5, 5, 5]),
    )

    first = lin_velocity_delta_l2(
      env,
      command_name="twist",
      deadband=0.01,
      normalize_by_command=True,
    )
    data.root_link_lin_vel_b[:, 0] = torch.tensor([0.12, 0.25, -0.14])
    second = lin_velocity_delta_l2(
      env,
      command_name="twist",
      deadband=0.01,
      normalize_by_command=True,
    )
    data.root_link_lin_vel_b[:, 0] = torch.tensor([0.11, 0.30, -0.20])
    env.episode_length_buf = torch.tensor([6, 6, 1])
    third = lin_velocity_delta_l2(
      env,
      command_name="twist",
      deadband=0.01,
      normalize_by_command=True,
    )

    torch.testing.assert_close(first, torch.zeros(3))
    torch.testing.assert_close(second, torch.tensor([0.0625, 0.0, 1.0]))
    torch.testing.assert_close(third, torch.tensor([0.015625, 0.0, 0.0]))

  def test_lin_velocity_delta_l2_ignores_first_step_after_command_change(self):
    command = torch.tensor(
      [
        [0.08, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-0.08, 0.0, 0.0],
      ]
    )
    actual_lin_x = torch.tensor([0.04, 0.00, 0.04])
    data = SimpleNamespace(
      root_link_lin_vel_b=torch.stack(
        [actual_lin_x, torch.zeros_like(actual_lin_x), torch.zeros_like(actual_lin_x)],
        dim=1,
      )
    )
    env = SimpleNamespace(
      command_manager=SimpleNamespace(get_command=lambda _name: command),
      scene={"robot": SimpleNamespace(data=data)},
      episode_length_buf=torch.tensor([5, 5, 5]),
    )

    first = lin_velocity_delta_l2(
      env,
      command_name="twist",
      deadband=0.01,
      normalize_by_command=True,
    )
    data.root_link_lin_vel_b[:, 0] = torch.tensor([0.06, 0.02, 0.02])
    second = lin_velocity_delta_l2(
      env,
      command_name="twist",
      deadband=0.01,
      normalize_by_command=True,
    )
    command[:, 0] = torch.tensor([-0.08, 0.08, 0.08])
    data.root_link_lin_vel_b[:, 0] = torch.tensor([0.05, 0.03, 0.00])
    after_change = lin_velocity_delta_l2(
      env,
      command_name="twist",
      deadband=0.01,
      normalize_by_command=True,
    )

    torch.testing.assert_close(first, torch.zeros(3))
    torch.testing.assert_close(second, torch.tensor([0.0625, 0.0, 0.0625]))
    torch.testing.assert_close(after_change, torch.zeros(3))

  def test_apply_action_ema_filters_and_resets_selected_envs(self):
    previous = torch.tensor([0.0, 0.5, -0.5])
    current = torch.tensor([1.0, -1.0, 0.0])

    filtered = apply_action_ema(current, previous, alpha=0.65)

    torch.testing.assert_close(filtered, 0.65 * previous + 0.35 * current)

    with self.assertRaisesRegex(ValueError, "alpha"):
      apply_action_ema(current, previous, alpha=1.0)

  def test_low_speed_lin_overspeed_l2_penalizes_only_same_direction_overspeed(self):
    command = torch.tensor(
      [
        [0.07, 0.0, 0.0],
        [0.07, 0.0, 0.0],
        [-0.07, 0.0, 0.0],
        [-0.07, 0.0, 0.0],
        [0.00, 0.0, 0.0],
        [0.14, 0.0, 0.0],
      ]
    )
    actual_lin_x = torch.tensor([0.09, 0.06, -0.09, 0.02, 0.20, 0.20])
    env = SimpleNamespace(
      command_manager=SimpleNamespace(get_command=lambda _name: command),
      scene={
        "robot": SimpleNamespace(
          data=SimpleNamespace(
            root_link_lin_vel_b=torch.stack(
              [
                actual_lin_x,
                torch.zeros_like(actual_lin_x),
                torch.zeros_like(actual_lin_x),
              ],
              dim=1,
            )
          )
        )
      },
    )

    penalty = low_speed_lin_overspeed_l2(
      env,
      command_name="twist",
      deadband=0.01,
      margin=0.005,
      max_command_abs=0.12,
      normalize_by_command=True,
    )

    expected = torch.tensor(
      [
        ((0.09 - 0.07 - 0.005) / 0.07) ** 2,
        0.0,
        ((0.09 - 0.07 - 0.005) / 0.07) ** 2,
        0.0,
        0.0,
        0.0,
      ]
    )
    torch.testing.assert_close(penalty, expected)

  def test_safe_posture_velocity_shaping_masks_unsafe_pitch_or_pitch_rate(self):
    command = torch.tensor(
      [
        [0.08, 0.0, 0.0],
        [0.08, 0.0, 0.0],
        [0.08, 0.0, 0.0],
      ]
    )
    actual_lin_x = torch.tensor([0.14, 0.14, 0.14])
    data = SimpleNamespace(
      root_link_lin_vel_b=torch.stack(
        [actual_lin_x, torch.zeros_like(actual_lin_x), torch.zeros_like(actual_lin_x)],
        dim=1,
      ),
      root_link_ang_vel_b=torch.tensor(
        [
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
        ]
      ),
      projected_gravity_b=torch.tensor(
        [
          [0.0, 0.0, -1.0],
          [0.2, 0.0, -1.0],
          [0.0, 0.0, -1.0],
        ]
      ),
    )
    env = SimpleNamespace(
      command_manager=SimpleNamespace(get_command=lambda _name: command),
      scene={"robot": SimpleNamespace(data=data)},
      episode_length_buf=torch.tensor([5, 5, 5]),
    )

    band = safe_posture_lin_velocity_band_l2(
      env,
      command_name="twist",
      deadband=0.01,
      lower_fraction=0.5,
      upper_fraction=1.5,
      over_scale=4.0,
      normalize_by_command=True,
      pitch_abs_limit=0.08,
      pitch_rate_abs_limit=0.8,
    )
    overspeed = safe_posture_low_speed_lin_overspeed_l2(
      env,
      command_name="twist",
      deadband=0.01,
      margin=0.002,
      max_command_abs=0.12,
      normalize_by_command=True,
      pitch_abs_limit=0.08,
      pitch_rate_abs_limit=0.8,
    )
    first_delta = safe_posture_lin_velocity_delta_l2(
      env,
      command_name="twist",
      deadband=0.01,
      normalize_by_command=True,
      pitch_abs_limit=0.08,
      pitch_rate_abs_limit=0.8,
    )
    data.root_link_lin_vel_b[:, 0] = torch.tensor([0.16, 0.16, 0.16])
    second_delta = safe_posture_lin_velocity_delta_l2(
      env,
      command_name="twist",
      deadband=0.01,
      normalize_by_command=True,
      pitch_abs_limit=0.08,
      pitch_rate_abs_limit=0.8,
    )

    torch.testing.assert_close(band, torch.tensor([0.25, 0.0, 0.0]))
    torch.testing.assert_close(overspeed, torch.tensor([0.525625, 0.0, 0.0]))
    torch.testing.assert_close(first_delta, torch.zeros(3))
    torch.testing.assert_close(second_delta, torch.tensor([0.0625, 0.0, 0.0]))


if __name__ == "__main__":
  unittest.main()
