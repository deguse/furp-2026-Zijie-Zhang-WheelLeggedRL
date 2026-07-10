import unittest

import numpy as np

from hoppertrex_mjlab.hybrid.control import compose_hybrid_targets


class HybridActionCompositionTest(unittest.TestCase):
  def test_masked_policy_noise_does_not_reach_applied_residual(self):
    output = compose_hybrid_targets(
      policy_actions=np.ones((1, 6)),
      action_mask=(True, False, False, False, False, False),
      action_scales=(0.5, 0.4, 0.03, 0.03, 0.03, 0.03),
      baseline_wheel_targets=np.array([[-2.0, 2.0]]),
      nominal_leg_targets=np.array([[0.1, 0.2, 0.3, 0.4]]),
      previous_wheel_targets=np.array([[-2.0, 2.0]]),
      wheel_slew_limit=6.0,
      wheel_velocity_limit=20.0,
      leg_position_lower=np.full(4, -1.0),
      leg_position_upper=np.full(4, 1.0),
    )

    np.testing.assert_allclose(
      output.applied_residual,
      [[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]],
    )
    np.testing.assert_allclose(output.wheel_targets, [[-2.5, 2.5]])
    np.testing.assert_allclose(output.leg_targets, [[0.1, 0.2, 0.3, 0.4]])

  def test_balance_and_yaw_residual_use_mirrored_wheel_signs(self):
    output = compose_hybrid_targets(
      policy_actions=np.array([[0.5, -0.25, 0.0, 0.0, 0.0, 0.0]]),
      action_mask=(True, True, False, False, False, False),
      action_scales=(2.0, 4.0, 0.0, 0.0, 0.0, 0.0),
      baseline_wheel_targets=np.array([[-3.0, 3.0]]),
      nominal_leg_targets=np.zeros((1, 4)),
      previous_wheel_targets=np.array([[-3.0, 3.0]]),
      wheel_slew_limit=10.0,
      wheel_velocity_limit=20.0,
      leg_position_lower=np.full(4, -1.0),
      leg_position_upper=np.full(4, 1.0),
    )

    # balance=+1 gives [-1,+1], yaw=-1 gives [-1,-1].
    np.testing.assert_allclose(output.wheel_targets, [[-5.0, 3.0]])

  def test_targets_apply_policy_clip_slew_and_actuator_limits(self):
    output = compose_hybrid_targets(
      policy_actions=np.array([[4.0, 0.0, 2.0, -2.0, 2.0, -2.0]]),
      action_mask=(True,) * 6,
      action_scales=(10.0, 1.0, 0.5, 0.5, 0.5, 0.5),
      baseline_wheel_targets=np.array([[0.0, 0.0]]),
      nominal_leg_targets=np.zeros((1, 4)),
      previous_wheel_targets=np.array([[1.0, -1.0]]),
      wheel_slew_limit=2.0,
      wheel_velocity_limit=2.5,
      leg_position_lower=np.full(4, -0.25),
      leg_position_upper=np.full(4, 0.25),
    )

    np.testing.assert_allclose(output.wheel_targets, [[-1.0, 1.0]])
    np.testing.assert_allclose(output.leg_targets, [[0.25, -0.25, 0.25, -0.25]])
    np.testing.assert_allclose(output.applied_residual[0, 2:], [0.5, -0.5, 0.5, -0.5])

  def test_shape_mismatch_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "policy_actions"):
      compose_hybrid_targets(
        policy_actions=np.zeros((2, 5)),
        action_mask=(True,) * 6,
        action_scales=(1.0,) * 6,
        baseline_wheel_targets=np.zeros((2, 2)),
        nominal_leg_targets=np.zeros((2, 4)),
        previous_wheel_targets=np.zeros((2, 2)),
        wheel_slew_limit=1.0,
        wheel_velocity_limit=2.0,
        leg_position_lower=np.full(4, -1.0),
        leg_position_upper=np.full(4, 1.0),
      )


if __name__ == "__main__":
  unittest.main()
