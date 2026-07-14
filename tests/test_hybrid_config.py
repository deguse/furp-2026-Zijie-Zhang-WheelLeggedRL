import unittest

from hoppertrex_mjlab.hybrid.config import (
  HYBRID_ACTION_NAMES,
  HYBRID_STAGES,
  HybridStageCfg,
)


class HybridStageConfigTest(unittest.TestCase):
  def test_action_order_is_fixed_for_every_stage(self):
    self.assertEqual(
      HYBRID_ACTION_NAMES,
      (
        "wheel_balance_residual",
        "wheel_yaw_residual",
        "left_thigh_residual",
        "right_thigh_residual",
        "left_knee_residual",
        "right_knee_residual",
      ),
    )
    for stage in HYBRID_STAGES.values():
      with self.subTest(stage=stage.index):
        self.assertEqual(stage.action_dim, 6)
        self.assertEqual(len(stage.action_mask), 6)
        self.assertEqual(len(stage.action_scales), 6)

  def test_stage_masks_enable_capabilities_monotonically(self):
    self.assertEqual(HYBRID_STAGES[0].action_mask, (False,) * 6)
    self.assertEqual(
      HYBRID_STAGES[1].action_mask,
      (True, False, False, False, False, False),
    )
    self.assertEqual(
      HYBRID_STAGES[2].action_mask,
      (True, True, False, False, False, False),
    )
    for index in (3, 4, 5):
      self.assertEqual(HYBRID_STAGES[index].action_mask, (True,) * 6)

  def test_curriculum_ranges_and_gate_suites_match_capabilities(self):
    self.assertEqual(HYBRID_STAGES[1].action_scales[0], 0.5)
    self.assertEqual(HYBRID_STAGES[1].lin_vel_x_range, (-0.10, 0.10))
    self.assertEqual(HYBRID_STAGES[1].yaw_rate_range, (0.0, 0.0))
    self.assertEqual(HYBRID_STAGES[2].yaw_rate_range, (-0.10, 0.10))
    self.assertEqual(HYBRID_STAGES[3].lin_vel_x_range, (0.0, 0.0))
    self.assertEqual(HYBRID_STAGES[3].yaw_rate_range, (0.0, 0.0))
    self.assertEqual(
      [HYBRID_STAGES[index].gate_suite for index in range(6)],
      ["controller", "residual", "planar", "posture", "integrated", "robust"],
    )
    self.assertEqual(HYBRID_STAGES[1].randomization_level, 1)
    self.assertEqual(HYBRID_STAGES[1].push_interval_s, (5.0, 8.0))
    self.assertEqual(HYBRID_STAGES[1].push_lin_vel_x, 0.04)
    self.assertEqual(HYBRID_STAGES[1].push_pitch_rate, 0.06)
    self.assertEqual(HYBRID_STAGES[5].randomization_level, 2)
    self.assertEqual(HYBRID_STAGES[5].push_interval_s, (3.0, 5.0))
    self.assertEqual(HYBRID_STAGES[5].push_lin_vel_x, 0.08)
    self.assertEqual(HYBRID_STAGES[5].push_pitch_rate, 0.12)

  def test_invalid_stage_shape_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "six"):
      HybridStageCfg(
        index=9,
        capability="invalid",
        action_mask=(True,),
        action_scales=(1.0,),
        lin_vel_x_range=(0.0, 0.0),
        yaw_rate_range=(0.0, 0.0),
        posture_commands=False,
        randomization_level=0,
        gate_suite="invalid",
      )


if __name__ == "__main__":
  unittest.main()
