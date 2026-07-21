import unittest

from hoppertrex_mjlab.hybrid.config import (
  HYBRID_ACTION_NAMES,
  HYBRID_STAGES,
  HybridStageCfg,
  action_scales_with_leg_authority,
)


class HybridStageConfigTest(unittest.TestCase):
  def test_leg_authority_override_changes_only_four_leg_heads(self):
    self.assertEqual(
      action_scales_with_leg_authority(0.07),
      (0.5, 0.3, 0.07, 0.07, 0.07, 0.07),
    )
    self.assertEqual(
      action_scales_with_leg_authority(), HYBRID_STAGES[5].action_scales
    )
    for invalid in (-0.01, float("nan"), float("inf")):
      with self.subTest(invalid=invalid):
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
          action_scales_with_leg_authority(invalid)

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
    # The yaw residual scale is deliberately a fraction of the ~0.55 nominal
    # feedforward differential at wz = 0.10: nominal yaw tracking is owned by
    # the Stage 2.0 calibrated feedforward, not the policy.
    self.assertEqual(HYBRID_STAGES[2].action_scales[1], 0.3)
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
    self.assertEqual(HYBRID_STAGES[2].randomization_level, 1)
    self.assertEqual(HYBRID_STAGES[2].push_interval_s, (5.0, 8.0))
    self.assertEqual(HYBRID_STAGES[2].push_lin_vel_x, 0.04)
    self.assertEqual(HYBRID_STAGES[2].push_pitch_rate, 0.06)
    self.assertEqual(HYBRID_STAGES[5].randomization_level, 2)
    self.assertEqual(HYBRID_STAGES[5].push_interval_s, (3.0, 5.0))
    # Stage5 campaign: training pushes span the pre-registered eval band
    # (8x the Stage1 kick impulse).
    self.assertEqual(HYBRID_STAGES[5].push_lin_vel_x, 0.32)
    self.assertEqual(HYBRID_STAGES[5].push_pitch_rate, 0.48)

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
