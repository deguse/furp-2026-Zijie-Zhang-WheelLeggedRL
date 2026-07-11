import math
import unittest

from hoppertrex_mjlab.hybrid.calibration import (
  CalibrationCandidate,
  apply_velocity_calibration,
  calibration_artifact,
  calibration_hash,
  candidate_from_envelope,
  fine_grid,
  parse_calibration_artifact,
  score_candidate,
)


class VelocityCalibrationTest(unittest.TestCase):
  def test_scale_and_bias_are_applied_to_requested_command(self):
    self.assertAlmostEqual(
      apply_velocity_calibration(0.07, scale=0.856, bias=-0.0116),
      0.04832,
    )

  def test_artifact_hash_binds_lqr_and_parameters(self):
    payload = calibration_artifact(
      controller_gain_hash='a' * 64,
      scale=0.86,
      bias=-0.012,
      seed=1,
      candidates=[{'scale': 0.86, 'bias': -0.012, 'score': 0.01}],
    )
    self.assertEqual(payload['schema_version'], 1)
    self.assertEqual(payload['calibration_hash'], calibration_hash(payload))
    changed = dict(payload)
    changed['velocity_command_bias'] = -0.010
    self.assertNotEqual(payload['calibration_hash'], calibration_hash(changed))
    parsed = parse_calibration_artifact(payload, controller_gain_hash='a' * 64)
    self.assertEqual((parsed.scale, parsed.bias), (0.86, -0.012))
    with self.assertRaisesRegex(ValueError, 'different controller'):
      parse_calibration_artifact(payload, controller_gain_hash='b' * 64)

  def test_candidate_rejects_instability_and_scores_tracking(self):
    stable = CalibrationCandidate(
      scale=0.86,
      bias=-0.012,
      scenarios=(
        {'requested_vx': -0.07, 'actual_vx': -0.068, 'terminated_event_rate': 0.0, 'p95_pitch': 0.03, 'p99_pitch_rate': 0.2},
        {'requested_vx': 0.0, 'actual_vx': 0.002, 'terminated_event_rate': 0.0, 'p95_pitch': 0.02, 'p99_pitch_rate': 0.2},
        {'requested_vx': 0.07, 'actual_vx': 0.071, 'terminated_event_rate': 0.0, 'p95_pitch': 0.02, 'p99_pitch_rate': 0.2},
      ),
    )
    score = score_candidate(stable)
    self.assertTrue(score.accepted)
    self.assertAlmostEqual(score.score, math.sqrt(9.0e-6 / 3.0) + 0.004)

    unstable = CalibrationCandidate(
      scale=0.86,
      bias=-0.012,
      scenarios=({
        'requested_vx': 0.0, 'actual_vx': 0.0,
        'terminated_event_rate': 0.02, 'p95_pitch': 0.02,
        'p99_pitch_rate': 0.2,
      },),
    )
    self.assertFalse(score_candidate(unstable).accepted)

  def test_fine_grid_is_centered_and_deterministic(self):
    self.assertEqual(
      fine_grid(0.86, -0.012),
      tuple(
        (scale, bias)
        for scale in (0.82, 0.84, 0.86, 0.88, 0.90)
        for bias in (-0.016, -0.014, -0.012, -0.010, -0.008)
      ),
    )

  def test_candidate_is_extracted_from_gate_envelope(self):
    envelope = {'scenarios': [
      {'lin_x': command, 'metrics': {
        'mean_actual_lin_x': actual, 'terminated_event_rate': 0.0,
        'p95_pitch': 0.02, 'p99_pitch_rate': 0.2,
      }}
      for command, actual in ((-0.07, -0.068), (0.0, 0.002), (0.07, 0.071))
    ]}
    candidate = candidate_from_envelope(envelope, scale=0.86, bias=-0.012)
    self.assertEqual(len(candidate.scenarios), 3)
    self.assertEqual(candidate.scenarios[1]['requested_vx'], 0.0)


if __name__ == '__main__':
  unittest.main()
