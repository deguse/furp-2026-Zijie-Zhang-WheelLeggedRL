from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = (
  ROOT
  / 'docs'
  / 'experiments'
  / 'artifacts'
  / 'c1_posture_requalification_seed1'
)


class C1PostureArtifactTest(unittest.TestCase):
  def test_manifest_hashes_and_bindings_are_frozen(self):
    manifest = json.loads((ARTIFACTS / 'manifest.json').read_text())
    entries = (
      manifest['posture_map'],
      manifest['uncompensated_qualification'],
      manifest['station_calibration'],
      manifest['compensated_qualification'],
    )
    for entry in entries:
      raw = (ARTIFACTS / entry['path']).read_bytes()
      self.assertEqual(hashlib.sha256(raw).hexdigest(), entry['file_sha256'])

    posture = json.loads(
      (ARTIFACTS / manifest['posture_map']['path']).read_text()
    )
    station = json.loads(
      (ARTIFACTS / manifest['station_calibration']['path']).read_text()
    )
    compensated = json.loads(
      (ARTIFACTS / manifest['compensated_qualification']['path']).read_text()
    )
    bindings = manifest['bindings']
    self.assertEqual(posture['posture_artifact_hash'], bindings['posture_artifact_hash'])
    self.assertEqual(station['controller_gain_hash'], bindings['controller_gain_hash'])
    self.assertEqual(station['posture_artifact_hash'], bindings['posture_artifact_hash'])
    self.assertEqual(
      station['station_calibration_hash'], bindings['station_calibration_hash']
    )
    self.assertEqual(compensated['calibration_hash'], bindings['velocity_calibration_hash'])
    self.assertEqual(
      compensated['station_calibration_hash'], bindings['station_calibration_hash']
    )

  def test_compensated_qualification_passes_registered_limits(self):
    payload = json.loads(
      (ARTIFACTS / 'balance_compensated_seed1.json').read_text()
    )
    self.assertTrue(payload['controller_qualified'])
    self.assertTrue(payload['posture_map_qualified'])
    self.assertTrue(payload['station_calibration_qualified'])
    self.assertEqual(payload['source_probe']['git_sha'], 'c2d1ffe224549462a7858a082772a400e74b6f86')
    self.assertEqual(len(payload['grid_cells']), 9)
    self.assertEqual(len(payload['vx_checks']), 6)
    cells = payload['grid_cells'] + payload['vx_checks']
    self.assertTrue(all(cell['terminated_events'] == 0.0 for cell in cells))
    self.assertTrue(all(cell['non_wheel_contact_rate'] == 0.0 for cell in cells))
    summary = payload['summary']
    self.assertLessEqual(summary['worst_abs_station_drift'], 0.015)
    self.assertLessEqual(summary['worst_height_rmse'], 0.002)
    self.assertLessEqual(summary['worst_pitch_rmse'], 0.015)


if __name__ == '__main__':
  unittest.main()
