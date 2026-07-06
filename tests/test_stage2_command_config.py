import unittest

from hoppertrex_mjlab.tasks.hoppertrex_balance_task import (
  BidirBandVelocityCommandCfg,
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


if __name__ == "__main__":
  unittest.main()
