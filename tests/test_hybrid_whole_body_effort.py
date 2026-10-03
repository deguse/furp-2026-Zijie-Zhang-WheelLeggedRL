import unittest

import numpy as np

from hoppertrex_mjlab.hybrid.whole_body_effort import (
  WholeBodyEffortConfig,
  compute_whole_body_effort,
)


class WholeBodyEffortTest(unittest.TestCase):
  def setUp(self) -> None:
    self.nv = 12
    self.mass = np.eye(self.nv, dtype=np.float64)
    self.bias = np.zeros(self.nv, dtype=np.float64)
    self.left_jacobian = np.zeros(self.nv, dtype=np.float64)
    self.right_jacobian = np.zeros(self.nv, dtype=np.float64)
    self.left_jacobian[[6, 8]] = (0.10, 0.20)
    self.right_jacobian[[7, 9]] = (0.10, 0.20)
    self.kwargs = {
      "mass_matrix": self.mass,
      "bias_effort": self.bias,
      "left_vertical_jacobian": self.left_jacobian,
      "right_vertical_jacobian": self.right_jacobian,
      "leg_dof_ids": (6, 7, 8, 9),
      "wheel_dof_ids": (10, 11),
      "roll_dof_id": 3,
      "track_width_m": 0.4768,
      "roll_rad": 0.0,
      "roll_rate_rad_s": 0.0,
      "leg_positions_rad": (0.0, 0.0, 0.0, 0.0),
      "leg_velocities_rad_s": (0.0, 0.0, 0.0, 0.0),
      "leg_position_targets_rad": (0.0, 0.0, 0.0, 0.0),
      "wheel_velocities_rad_s": (0.0, 0.0),
      "wheel_velocity_targets_rad_s": (0.0, 0.0),
    }

  def _compute(self, **overrides: object):
    values = dict(self.kwargs)
    values.update(overrides)
    return compute_whole_body_effort(WholeBodyEffortConfig(), **values)

  def test_symmetric_state_allocates_symmetric_load_and_leg_effort(self) -> None:
    result = self._compute()
    expected = 0.5 * WholeBodyEffortConfig().total_support_force_n
    self.assertAlmostEqual(result.allocation.left_load_n, expected, places=9)
    self.assertAlmostEqual(result.allocation.right_load_n, expected, places=9)
    self.assertAlmostEqual(result.inverse_effort_nm[0], result.inverse_effort_nm[1])
    self.assertAlmostEqual(result.inverse_effort_nm[2], result.inverse_effort_nm[3])
    np.testing.assert_allclose(result.leg_kp, 400.0)
    np.testing.assert_allclose(result.leg_kd, 5.0)
    self.assertTrue(result.allocation.bilateral_minimum_satisfied)

  def test_positive_roll_requests_restoring_load_difference(self) -> None:
    result = self._compute(roll_rad=0.01)
    self.assertLess(result.desired_roll_moment_nm, 0.0)
    self.assertLess(result.allocation.left_load_n, result.allocation.right_load_n)

  def test_roll_moment_clamps_before_bilateral_minimum_is_violated(self) -> None:
    result = self._compute(roll_rad=100.0)
    self.assertAlmostEqual(
      abs(result.desired_roll_moment_nm),
      result.roll_moment_limit_nm,
    )
    self.assertGreaterEqual(
      result.allocation.left_load_n,
      WholeBodyEffortConfig().minimum_side_load_n - 1.0e-9,
    )
    self.assertGreaterEqual(
      result.allocation.right_load_n,
      WholeBodyEffortConfig().minimum_side_load_n - 1.0e-9,
    )

  def test_leg_and_wheel_efforts_are_hard_bounded(self) -> None:
    biased = self.bias.copy()
    biased[10:] = (1.0, -1.0)
    result = self._compute(
      bias_effort=biased,
      leg_position_targets_rad=(1.0, 1.0, 1.0, 1.0),
      wheel_velocity_targets_rad_s=(1.0, -1.0),
    )
    np.testing.assert_allclose(np.abs(result.leg_effort_nm), 30.0)
    np.testing.assert_allclose(np.abs(result.wheel_effort_nm), 5.8)
    self.assertTrue(bool(result.leg_saturated.all()))
    self.assertTrue(bool(result.wheel_saturated.all()))
    self.assertTrue(bool(result.wheel_above_rated.all()))

  def test_held_outer_target_still_refreshes_inner_effort_from_state(self) -> None:
    first = self._compute(
      leg_position_targets_rad=(0.05, 0.05, 0.05, 0.05),
    )
    second = self._compute(
      leg_positions_rad=(0.04, 0.04, 0.04, 0.04),
      leg_position_targets_rad=(0.05, 0.05, 0.05, 0.05),
    )
    self.assertFalse(np.array_equal(first.leg_effort_nm, second.leg_effort_nm))

  def test_same_numeric_state_is_backend_independent(self) -> None:
    native = self._compute()
    warp_mirror = self._compute(
      mass_matrix=self.mass.astype(np.float32),
      bias_effort=self.bias.astype(np.float32),
      left_vertical_jacobian=self.left_jacobian.astype(np.float32),
      right_vertical_jacobian=self.right_jacobian.astype(np.float32),
    )
    np.testing.assert_allclose(native.leg_effort_nm, warp_mirror.leg_effort_nm)
    np.testing.assert_allclose(native.wheel_effort_nm, warp_mirror.wheel_effort_nm)

  def test_rejects_overlapping_actuator_dofs(self) -> None:
    with self.assertRaisesRegex(ValueError, "disjoint"):
      self._compute(wheel_dof_ids=(9, 10))


if __name__ == "__main__":
  unittest.main()

