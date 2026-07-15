import unittest

from hoppertrex_mjlab.hybrid.yaw_calibration import (
  parse_yaw_calibration_artifact,
  yaw_calibration_artifact,
  yaw_feedforward,
)
from hoppertrex_mjlab.scripts.probe_hybrid_yaw_transfer import (
  DEFAULT_YAW_ACTIONS,
  FIT_DEFAULT_YAW_ACTIONS,
  fit_yaw_breakpoints,
  parse_args,
)


CONTROLLER_HASH = "c" * 64


class FitYawBreakpointsTest(unittest.TestCase):
  def test_monotone_samples_pass_through_with_pinned_zero(self):
    samples = [
      (-0.176, -0.75),
      (-0.049, -0.35),
      (0.049, 0.35),
      (0.109, 0.55),
      (0.271, 1.0),
    ]

    breakpoints = fit_yaw_breakpoints(samples)

    self.assertIn((0.0, 0.0), breakpoints)
    self.assertEqual(
      breakpoints,
      (
        (-0.176, -0.75),
        (-0.049, -0.35),
        (0.0, 0.0),
        (0.049, 0.35),
        (0.109, 0.55),
        (0.271, 1.0),
      ),
    )

  def test_noise_inversions_are_monotonized_outward_from_zero(self):
    samples = [
      (0.049, 0.35),
      (0.060, 0.33),
      (0.109, 0.55),
      (-0.049, -0.35),
      (-0.060, -0.33),
    ]

    breakpoints = fit_yaw_breakpoints(samples)

    diffs = [differential for _, differential in breakpoints]
    self.assertEqual(diffs, sorted(diffs))
    self.assertIn((0.06, 0.35), breakpoints)
    self.assertIn((-0.06, -0.35), breakpoints)

  def test_sign_violating_rows_are_dropped_not_projected(self):
    samples = [
      (0.049, 0.35),
      (0.002, -0.05),
      (-0.049, -0.35),
    ]

    breakpoints = fit_yaw_breakpoints(samples)

    self.assertEqual(
      breakpoints,
      ((-0.049, -0.35), (0.0, 0.0), (0.049, 0.35)),
    )

  def test_fitted_breakpoints_build_a_parseable_artifact(self):
    breakpoints = fit_yaw_breakpoints(
      [(0.049, 0.35), (0.109, 0.55), (-0.049, -0.35), (-0.109, -0.55)]
    )
    payload = yaw_calibration_artifact(
      controller_gain_hash=CONTROLLER_HASH,
      breakpoints=breakpoints,
      source_probe={"git_sha": "test", "device": "cpu"},
    )

    parsed = parse_yaw_calibration_artifact(
      payload,
      controller_gain_hash=CONTROLLER_HASH,
    )
    self.assertEqual(parsed.breakpoints, breakpoints)
    self.assertAlmostEqual(
      float(yaw_feedforward(0.10, parsed.breakpoints)),
      0.35 + (0.10 - 0.049) / (0.109 - 0.049) * (0.55 - 0.35),
    )

  def test_all_noise_rows_leave_too_few_points(self):
    with self.assertRaisesRegex(ValueError, "at least two"):
      fit_yaw_breakpoints([(0.002, -0.05), (-0.002, 0.05)])


class ProbeArgumentsTest(unittest.TestCase):
  def test_fit_defaults_densify_both_signs(self):
    self.assertGreater(
      len(FIT_DEFAULT_YAW_ACTIONS), len(DEFAULT_YAW_ACTIONS)
    )
    positives = [value for value in FIT_DEFAULT_YAW_ACTIONS if value > 0]
    negatives = [value for value in FIT_DEFAULT_YAW_ACTIONS if value < 0]
    self.assertEqual(
      sorted(positives),
      sorted(-value for value in negatives),
    )

  def test_yaw_actions_default_is_deferred_to_fit_mode_choice(self):
    args = parse_args([])
    self.assertIsNone(args.yaw_actions)
    self.assertIsNone(args.fit_output)
    self.assertEqual(args.probe_yaw_scale, 1.0)


if __name__ == "__main__":
  unittest.main()
