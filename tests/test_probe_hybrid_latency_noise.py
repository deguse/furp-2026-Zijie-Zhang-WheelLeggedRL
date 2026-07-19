import unittest

import torch

from mjlab.envs import ManagerBasedRlEnv

from hoppertrex_mjlab.scripts.probe_hybrid_latency_noise import (
  DEFAULT_DELAYS,
  NOISE_TIERS,
  _cell_env_cfg,
  _run_policy_kick_cell,
  _run_tracking_cell,
  _zero_policy,
  parse_args,
  probe_payload,
)


class _ActionCfg:
  controller_gain_hash = "c" * 64
  controller_qualified = True
  calibration_hash = "v" * 64
  yaw_calibration_hash = "y" * 64
  posture_map_hash = "p" * 64
  station_calibration_hash = "s" * 64


def _cell(policy, delay, tier, recovery, terminated) -> dict[str, object]:
  return {
    "policy": policy,
    "delay_steps": delay,
    "delay_ms": 20.0 * delay,
    "noise_tier": tier,
    "standing_abs_vx_mean": 0.01,
    "tracking_abs_error_mean": 0.01,
    "tracking_abs_error_p95": 0.03,
    "pitch_abs_p95": 0.02,
    "recovery_time_s": recovery,
    "post_kick_lin_x_abs_max": 0.10,
    "kick_event_count": 64.0,
    "terminated_events": terminated,
    "non_wheel_contact_rate": 0.0,
  }


class LatencyNoiseProbeProtocolTest(unittest.TestCase):
  def test_default_arguments_cover_the_scan_grid(self):
    args = parse_args([])
    self.assertEqual(tuple(args.delays), DEFAULT_DELAYS)
    self.assertEqual(tuple(args.noise_tiers), tuple(NOISE_TIERS))
    self.assertIsNone(args.checkpoint_file)
    self.assertIsNone(args.fit_output)
    self.assertEqual(args.num_envs, 16)

  def test_noise_tiers_are_ordered_scan_inputs(self):
    self.assertEqual(
      tuple(NOISE_TIERS), ("none", "encoder", "mems_imu", "mems_imu_2x")
    )
    clean = NOISE_TIERS["none"]
    self.assertTrue(all(value == 0.0 for value in clean.values()))
    doubled = NOISE_TIERS["mems_imu_2x"]
    base = NOISE_TIERS["mems_imu"]
    for key, value in base.items():
      self.assertAlmostEqual(doubled[key], 2.0 * value)

  def test_payload_reports_delay_knee_per_tier(self):
    cells = [
      _cell("zero_residual", 0, "none", 0.30, 0.0),
      _cell("zero_residual", 1, "none", 0.35, 0.0),
      _cell("zero_residual", 2, "none", 0.70, 0.0),
      _cell("zero_residual", 0, "mems_imu", 0.35, 0.0),
      _cell("zero_residual", 1, "mems_imu", 0.90, 0.0),
      _cell("zero_residual", 2, "mems_imu", 1.20, 4.0),
    ]

    payload = probe_payload(
      cells=cells,
      action_cfg=_ActionCfg(),
      source_probe={"git_sha": "deadbeef"},
      checkpoint=None,
    )

    knees = payload["zero_residual_delay_knee_by_tier"]
    # none tier: recovery 0.70 exceeds 2x the 0.30 reference at delay 2.
    self.assertEqual(knees["none"], 2)
    # mems_imu tier: 0.90 > 2x 0.30 already at delay 1.
    self.assertEqual(knees["mems_imu"], 1)
    self.assertEqual(payload["probe"], "hybrid_latency_noise_tolerance")
    self.assertIsNone(payload["checkpoint"])
    self.assertIn("scan inputs", payload["noise_tier_provenance"])
    self.assertEqual(payload["control_step_ms"], 20.0)

  def test_payload_with_no_degradation_reports_null_knee(self):
    cells = [
      _cell("zero_residual", delay, "none", 0.30 + 0.01 * delay, 0.0)
      for delay in (0, 1, 2, 3, 4)
    ]

    payload = probe_payload(
      cells=cells,
      action_cfg=_ActionCfg(),
      source_probe={},
      checkpoint=None,
    )

    self.assertIsNone(
      payload["zero_residual_delay_knee_by_tier"]["none"]
    )


class LatencyNoiseProbeSmokeTest(unittest.TestCase):
  def test_single_cell_runs_on_cpu_with_delay_and_noise(self):
    """Mechanical correctness only: one delayed+noisy cell end to end."""

    torch.manual_seed(1)
    cfg = _cell_env_cfg(
      "HopperTrex-Hybrid-v2-Stage5",
      num_envs=1,
      delay_steps=2,
      tier="mems_imu",
      seed=1,
      policy_obs_noise=False,
    )
    posture = cfg.commands["posture"]
    center = (
      0.5 * (posture.height_range[0] + posture.height_range[1]),
      0.5 * (posture.pitch_range[0] + posture.pitch_range[1]),
    )
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
      policy = _zero_policy(int(env.action_manager.total_action_dim))
      tracking = _run_tracking_cell(
        env,
        policy,
        settle_steps=3,
        tracking_steps=5,
        center=center,
      )
      kick = _run_policy_kick_cell(
        env,
        policy,
        center=center,
        kicks=1,
        kick_interval=5,
        settle_steps=3,
      )
    finally:
      env.close()

    for key in (
      "standing_abs_vx_mean",
      "tracking_abs_error_p95",
      "pitch_abs_p95",
      "terminated_events",
    ):
      self.assertIn(key, tracking)
    for key in (
      "recovery_time_s",
      "post_kick_lin_x_abs_max",
      "kick_event_count",
      "non_wheel_contact_rate",
    ):
      self.assertIn(key, kick)
    self.assertEqual(kick["kick_event_count"], 1.0)


if __name__ == "__main__":
  unittest.main()
