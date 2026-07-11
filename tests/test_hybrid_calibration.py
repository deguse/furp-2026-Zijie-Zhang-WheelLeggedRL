import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
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
from hoppertrex_mjlab.scripts.calibrate_hybrid_velocity import (
  _candidate_manifest,
  _is_reusable_candidate,
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

  def test_candidate_is_extracted_from_serialized_metrics_envelope(self):
    envelope = {'metrics': {
      'controller_vx_-0.070': {'lin_x': -0.07, 'mean_actual_lin_x': -0.068,
        'terminated_event_rate': 0.0, 'p95_pitch': 0.02, 'p99_pitch_rate': 0.2},
      'controller_stand': {'lin_x': 0.0, 'mean_actual_lin_x': 0.002,
        'terminated_event_rate': 0.0, 'p95_pitch': 0.02, 'p99_pitch_rate': 0.2},
      'controller_vx_+0.070': {'lin_x': 0.07, 'mean_actual_lin_x': 0.071,
        'terminated_event_rate': 0.0, 'p95_pitch': 0.02, 'p99_pitch_rate': 0.2},
    }}
    candidate = candidate_from_envelope(envelope, scale=0.86, bias=-0.012)
    self.assertEqual([row['requested_vx'] for row in candidate.scenarios], [-0.07, 0.0, 0.07])

  def test_resume_requires_matching_run_manifest_and_gate_contract(self):
    args = SimpleNamespace(seed=1, device='cuda:0', num_envs=16, steps=600,
      warmup_steps=150, window_steps=300)
    manifest = _candidate_manifest(args, gain_hash='a' * 64, scale=0.86, bias=-0.012)
    envelope = {'schema_version': 1, 'suite': 'controller',
      'task': 'HopperTrex-Hybrid-v2-Stage0', 'seed': 1,
      'controller_gain_hash': 'a' * 64,
      'calibration_hash': manifest['calibration_hash'], 'metrics': {
        name: {'lin_x': command, 'mean_actual_lin_x': command,
          'terminated_event_rate': 0.0, 'p95_pitch': 0.02,
          'p99_pitch_rate': 0.2, 'duration_s': 12.0}
        for name, command in (('controller_vx_-0.070', -0.07),
          ('controller_stand', 0.0), ('controller_vx_+0.070', 0.07))}}
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      (root / 'manifest.json').write_text(__import__('json').dumps(manifest), encoding='utf-8')
      (root / 'gate.json').write_text(__import__('json').dumps(envelope), encoding='utf-8')
      self.assertTrue(_is_reusable_candidate(root / 'manifest.json', root / 'gate.json', manifest))
      changed = dict(manifest, steps=601)
      self.assertFalse(_is_reusable_candidate(root / 'manifest.json', root / 'gate.json', changed))


if __name__ == '__main__':
  unittest.main()
