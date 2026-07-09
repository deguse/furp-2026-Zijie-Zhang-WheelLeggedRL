import unittest

import torch

from hoppertrex_mjlab.scripts.rsl_rl.evaluate_fixed_command import (
  _bucketed_dynamic_stats,
  _command_tracking_health,
  _late_command_health,
  _summarize_reward_contributions,
  parse_args,
)


class LateCommandHealthTest(unittest.TestCase):
  def test_backward_command_with_negative_velocity_is_not_slow_or_wrong_direction(self):
    late_lin_x = torch.tensor(
      [
        [-0.09, -0.08],
        [-0.10, -0.07],
      ]
    )

    health = _late_command_health(
      late_lin_x=late_lin_x,
      target_lin_x=-0.08,
      stuck_speed=0.01,
    )

    self.assertFalse(torch.any(health["slow_env"]))
    self.assertFalse(torch.any(health["wrong_direction_env"]))

  def test_forward_command_with_negative_velocity_is_wrong_direction(self):
    late_lin_x = torch.tensor(
      [
        [-0.01, 0.08],
        [-0.02, 0.07],
      ]
    )

    health = _late_command_health(
      late_lin_x=late_lin_x,
      target_lin_x=0.08,
      stuck_speed=0.01,
    )

    self.assertTrue(health["wrong_direction_env"][0])
    self.assertFalse(health["wrong_direction_env"][1])

  def test_command_tracking_reports_sample_and_late_direction_rates(self):
    lin_x_by_step = torch.tensor(
      [
        [0.08, 0.08],
        [-0.02, 0.07],
        [0.09, 0.06],
      ]
    )

    metrics = _command_tracking_health(
      lin_x_by_step=lin_x_by_step,
      target_lin_x=0.06,
      stuck_speed=0.01,
      window_steps=2,
    )

    self.assertAlmostEqual(metrics["command_match_frac"], 5.0 / 6.0)
    self.assertAlmostEqual(metrics["wrong_direction_frac"], 1.0 / 6.0)
    self.assertAlmostEqual(metrics["late_wrong_direction_sample_frac"], 0.25)
    self.assertAlmostEqual(metrics["late_wrong_direction_env_frac"], 0.5)

  def test_command_tracking_reports_band_and_fast_fractions(self):
    lin_x_by_step = torch.tensor(
      [
        [0.00, 0.04, 0.08, 0.13],
        [0.00, 0.04, 0.08, 0.13],
      ],
      dtype=torch.float32,
    )

    metrics = _command_tracking_health(
      lin_x_by_step=lin_x_by_step,
      target_lin_x=0.08,
      stuck_speed=0.01,
      window_steps=2,
    )

    self.assertEqual(metrics["in_band_frac"], 0.5)
    self.assertEqual(metrics["fast_frac"], 0.25)
    self.assertEqual(metrics["late_in_band_frac"], 0.5)
    self.assertEqual(metrics["late_fast_sample_frac"], 0.25)

  def test_command_tracking_reports_velocity_delta_metrics(self):
    lin_x_by_step = torch.tensor(
      [
        [0.02, 0.02],
        [0.05, 0.01],
        [0.11, 0.03],
      ],
      dtype=torch.float32,
    )

    metrics = _command_tracking_health(
      lin_x_by_step=lin_x_by_step,
      target_lin_x=0.06,
      stuck_speed=0.01,
      window_steps=3,
    )

    delta = torch.tensor([0.03, -0.01, 0.06, 0.02])
    self.assertAlmostEqual(metrics["lin_x_delta_rms"], torch.sqrt(torch.mean(delta.square())).item())
    self.assertAlmostEqual(
      metrics["lin_x_delta_abs_p95"],
      torch.quantile(delta.abs(), 0.95).item(),
    )
    self.assertAlmostEqual(metrics["late_lin_x_delta_rms"], metrics["lin_x_delta_rms"])
    self.assertAlmostEqual(metrics["late_lin_x_delta_abs_p95"], metrics["lin_x_delta_abs_p95"])

  def test_reward_contribution_summary_reports_raw_weighted_and_relative_scale(self):
    weighted_rewards = {
      "track_linear_velocity": [torch.tensor([2.0, 4.0])],
      "lin_vel_x_sign_alignment": [torch.tensor([0.4, 0.8])],
      "lin_velocity_delta_l2": [torch.tensor([-0.12, -0.18])],
    }
    reward_weights = {
      "track_linear_velocity": 4.0,
      "lin_vel_x_sign_alignment": 2.0,
      "lin_velocity_delta_l2": -0.75,
    }

    summary = _summarize_reward_contributions(
      weighted_rewards=weighted_rewards,
      reward_weights=reward_weights,
      term_names=(
        "track_linear_velocity",
        "lin_vel_x_sign_alignment",
        "lin_velocity_delta_l2",
        "wheel_target_rate_l2",
      ),
      main_term="track_linear_velocity",
    )

    self.assertAlmostEqual(summary["track_linear_velocity_raw_mean"], 0.75)
    self.assertAlmostEqual(summary["track_linear_velocity_weighted_mean"], 3.0)
    self.assertAlmostEqual(summary["track_linear_velocity_relative_to_main"], 1.0)
    self.assertAlmostEqual(summary["lin_velocity_delta_l2_raw_mean"], 0.2)
    self.assertAlmostEqual(summary["lin_velocity_delta_l2_weighted_mean"], -0.15)
    self.assertAlmostEqual(summary["lin_velocity_delta_l2_relative_to_main"], -0.05)
    self.assertTrue(torch.isnan(torch.tensor(summary["wheel_target_rate_l2_raw_mean"])))

  def test_bucketed_dynamic_stats_reports_action_wheel_pitch_and_lin_delta(self):
    slow_mask = torch.tensor([True, False, False, True])
    in_band_mask = torch.tensor([False, True, False, False])
    fast_mask = torch.tensor([False, False, True, False])
    lin_delta_slow_mask = torch.tensor([False, True, False])
    lin_delta_in_band_mask = torch.tensor([True, False, False])
    lin_delta_fast_mask = torch.tensor([False, False, True])

    stats = _bucketed_dynamic_stats(
      slow_mask=slow_mask,
      in_band_mask=in_band_mask,
      fast_mask=fast_mask,
      lin_x_delta=torch.tensor([0.02, 0.04, 0.08]),
      lin_x_delta_slow_mask=lin_delta_slow_mask,
      lin_x_delta_in_band_mask=lin_delta_in_band_mask,
      lin_x_delta_fast_mask=lin_delta_fast_mask,
      wheel_target_rate=torch.tensor([1.0, 2.0, 3.0, 4.0]),
      action_delta=torch.tensor([0.10, 0.20, 0.30, 0.40]),
      pitch_rate_abs=torch.tensor([0.50, 0.60, 0.70, 0.80]),
    )

    self.assertAlmostEqual(
      stats["slow_action_delta_rms"],
      torch.sqrt(torch.mean(torch.tensor([0.10, 0.40]).square())).item(),
    )
    self.assertAlmostEqual(stats["in_band_wheel_target_rate_rms"], 2.0)
    self.assertAlmostEqual(stats["fast_pitch_rate_abs_p99"], 0.70)
    self.assertAlmostEqual(stats["slow_lin_x_delta_rms"], 0.04)
    self.assertAlmostEqual(stats["in_band_lin_x_delta_rms"], 0.02)
    self.assertAlmostEqual(stats["fast_lin_x_delta_rms"], 0.08)

  def test_argparse_accepts_multiple_lin_x_values(self):
    args = parse_args(
      [
        "--task",
        "task",
        "--checkpoint-file",
        "model.pt",
        "--lin-x",
        "-0.06",
        "0.06",
        "0.08",
      ]
    )

    self.assertEqual(args.lin_x, [-0.06, 0.06, 0.08])

  def test_argparse_accepts_episode_length_override(self):
    args = parse_args(
      [
        "--task",
        "task",
        "--checkpoint-file",
        "model.pt",
        "--episode-length-s",
        "1000000000",
      ]
    )

    self.assertEqual(args.episode_length_s, 1000000000.0)


if __name__ == "__main__":
  unittest.main()
