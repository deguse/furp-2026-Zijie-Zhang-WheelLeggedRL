from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hoppertrex_mjlab.scripts.rsl_rl.preflight_stair_camp import (
  FROZEN_C2_MJLAB_GIT_SHA,
  FROZEN_C2_SOURCE_GIT_SHA,
  PASS_CLASSIFICATION,
  STOP_CLASSIFICATION,
  StairCampPreflightError,
  _replay_trigger_arrays,
  finalize_preflight,
  replay_frozen_c2_trigger,
  validate_live_false_positive_result,
  write_new_atomic,
)


def _fp(domain: str, false_positives: int = 0) -> dict[str, object]:
  return {
    'schema_version': 1,
    'kind': 'stair_camp_trigger_false_positive_check',
    'domain': domain,
    'threshold_n': 18.0,
    'window_steps': 3,
    'events': 128,
    'stair_mode_false_positives': false_positives,
    'completed': True,
  }


def _c2_pass() -> dict[str, object]:
  return {
    'kind': 'stair_camp_c2_trigger_replay',
    'classification': PASS_CLASSIFICATION,
    'passed': True,
    'files_unchanged': True,
    'completed_pairs': 288,
    'detections': 288,
    'pre_impact_triggers': 0,
    'sha256s_file_sha256': 'a' * 64,
  }


class TriggerArrayReplayTest(unittest.TestCase):
  def _arrays(self):
    found = np.zeros((8, 2, 1), dtype=np.float32)
    force = np.zeros((8, 2, 1, 3), dtype=np.float32)
    normal = np.zeros_like(force)
    normal[..., 0] = 1.0
    impacts = np.array([2, 3], dtype=np.int64)
    for pair, impact in enumerate(impacts):
      found[impact : impact + 3, pair, 0] = 1.0
      force[impact : impact + 3, pair, 0, 0] = 20.0
    return found, force, normal, impacts

  def test_exact_three_frame_runtime_replay_detects_both_pairs(self) -> None:
    found, force, normal, impacts = self._arrays()
    result = _replay_trigger_arrays(
      found=found,
      force=force,
      normal=normal,
      impact_steps=impacts,
      post_impact_steps=4,
    )
    self.assertEqual(result['pairs'], 2)
    self.assertEqual(result['detections'], 2)
    self.assertEqual(result['pre_impact_triggers'], 0)
    self.assertEqual(result['missing_or_late'], 0)
    self.assertEqual(result['minimum_delay_steps'], 2)
    self.assertEqual(result['maximum_delay_steps'], 2)

  def test_preimpact_trigger_is_not_misreported_as_detection(self) -> None:
    found, force, normal, impacts = self._arrays()
    found[:3, 0, 0] = 1.0
    force[:3, 0, 0, 0] = 20.0
    impacts[0] = 4
    result = _replay_trigger_arrays(
      found=found,
      force=force,
      normal=normal,
      impact_steps=impacts,
      post_impact_steps=3,
    )
    self.assertEqual(result['pre_impact_triggers'], 1)
    self.assertEqual(result['detections'], 1)

  def test_malformed_nonfinite_and_fractional_arrays_fail_closed(self) -> None:
    found, force, normal, impacts = self._arrays()
    with self.assertRaisesRegex(StairCampPreflightError, 'shapes'):
      _replay_trigger_arrays(
        found=found,
        force=force[..., :2],
        normal=normal,
        impact_steps=impacts,
        post_impact_steps=3,
      )
    force[0, 0, 0, 0] = np.nan
    with self.assertRaisesRegex(StairCampPreflightError, 'NaN'):
      _replay_trigger_arrays(
        found=found,
        force=force,
        normal=normal,
        impact_steps=impacts,
        post_impact_steps=3,
      )
    force[0, 0, 0, 0] = 0.0
    found[0, 0, 0] = 0.5
    with self.assertRaisesRegex(StairCampPreflightError, 'match counts'):
      _replay_trigger_arrays(
        found=found,
        force=force,
        normal=normal,
        impact_steps=impacts,
        post_impact_steps=3,
      )


class FrozenDirectoryReplayTest(unittest.TestCase):
  def _make_fixture(self, root: Path) -> str:
    found = np.zeros((8, 2, 1), dtype=np.float32)
    force = np.zeros((8, 2, 1, 3), dtype=np.float32)
    normal = np.zeros_like(force)
    normal[..., 0] = 1.0
    impacts = np.array([2, 3], dtype=np.int64)
    truth = np.zeros((8, 2), dtype=np.bool_)
    for pair, impact in enumerate(impacts):
      truth[impact:, pair] = True
      found[impact : impact + 3, pair, 0] = 1.0
      force[impact : impact + 3, pair, 0, 0] = 20.0
    np.savez(
      root / 'cell_00.npz',
      stair_contact_found=found,
      stair_contact_force_contact_frame=force,
      stair_contact_normal_global=normal,
      impact_steps=impacts,
      stair_riser_contact=truth,
    )
    result = {
      'classification': 'C2_INNOVATION_DETECTOR_UNQUALIFIED_STOP',
      'evidence_eligible': True,
      'git_sha': FROZEN_C2_SOURCE_GIT_SHA,
      'mjlab_git_sha': FROZEN_C2_MJLAB_GIT_SHA,
      'completed_cell_count': 1,
      'completed_pair_count': 2,
      'protocol': {'post_impact_steps': 4},
      'cells': [
        {
          'raw_file': 'cell_00.npz',
          'raw_sha256': '',
          'impact_steps': [2, 3],
        }
      ],
    }
    raw_hash = hashlib.sha256((root / 'cell_00.npz').read_bytes()).hexdigest()
    result['cells'][0]['raw_sha256'] = raw_hash
    result_path = root / 'c2_innovation_detector_qualification.json'
    result_path.write_text(json.dumps(result), encoding='utf-8')
    result_hash = hashlib.sha256(result_path.read_bytes()).hexdigest()
    sums = f'{result_hash}  {result_path.name}\n{raw_hash}  cell_00.npz\n'
    sums_path = root / 'SHA256SUMS.txt'
    sums_path.write_text(sums, encoding='ascii', newline='\n')
    return hashlib.sha256(sums_path.read_bytes()).hexdigest()

  def test_directory_replay_binds_every_byte_and_stays_read_only(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      manifest_hash = self._make_fixture(root)
      before = {path.name: path.read_bytes() for path in root.iterdir()}
      result = replay_frozen_c2_trigger(
        root,
        expected_manifest_sha256=manifest_hash,
        expected_cells=1,
        expected_pairs_per_cell=2,
        expected_ticks=8,
        expected_slots=1,
      )
      self.assertEqual(result['classification'], PASS_CLASSIFICATION)
      self.assertEqual(result['detections'], 2)
      self.assertTrue(result['files_unchanged'])
      self.assertEqual(before, {path.name: path.read_bytes() for path in root.iterdir()})

  def test_manifest_or_file_mutation_is_a_protocol_error(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      manifest_hash = self._make_fixture(root)
      with (root / 'cell_00.npz').open('ab') as stream:
        stream.write(b'mutation')
      with self.assertRaisesRegex(StairCampPreflightError, 'hash drifted'):
        replay_frozen_c2_trigger(
          root,
          expected_manifest_sha256=manifest_hash,
          expected_cells=1,
          expected_pairs_per_cell=2,
          expected_ticks=8,
          expected_slots=1,
        )


class LiveFalsePositiveContractTest(unittest.TestCase):
  def test_exact_zero_false_positive_evidence_passes(self) -> None:
    result = validate_live_false_positive_result(
      _fp('camp_flat_rolling'), expected_domain='camp_flat_rolling'
    )
    self.assertTrue(result['passed'])

  def test_nonzero_false_positive_is_a_valid_scientific_stop(self) -> None:
    result = finalize_preflight(
      _c2_pass(),
      _fp('camp_flat_rolling', false_positives=1),
      _fp('stage5_kick'),
    )
    self.assertEqual(result['classification'], STOP_CLASSIFICATION)
    self.assertFalse(result['training_authorized'])

  def test_schema_threshold_window_and_domain_mutations_fail_closed(self) -> None:
    for field, value, pattern in (
      ('threshold_n', 17.9, 'threshold'),
      ('window_steps', 2, 'window'),
      ('domain', 'stairs', 'domain'),
      ('completed', False, 'did not complete'),
    ):
      payload = _fp('camp_flat_rolling')
      payload[field] = value
      with self.subTest(field=field), self.assertRaisesRegex(
        StairCampPreflightError, pattern
      ):
        validate_live_false_positive_result(
          payload, expected_domain='camp_flat_rolling'
        )

  def test_full_preflight_requires_288_of_288_and_both_fp_domains(self) -> None:
    result = finalize_preflight(
      _c2_pass(), _fp('camp_flat_rolling'), _fp('stage5_kick')
    )
    self.assertEqual(result['classification'], PASS_CLASSIFICATION)
    self.assertTrue(result['training_authorized'])
    invalid = _c2_pass()
    invalid['detections'] = 287
    with self.assertRaisesRegex(StairCampPreflightError, 'registered pass'):
      finalize_preflight(
        invalid, _fp('camp_flat_rolling'), _fp('stage5_kick')
      )

  def test_atomic_output_refuses_overwrite(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      output = Path(temp_dir) / 'preflight.json'
      payload = {'classification': PASS_CLASSIFICATION}
      write_new_atomic(output, payload)
      self.assertEqual(json.loads(output.read_text(encoding='utf-8')), payload)
      with self.assertRaises(FileExistsError):
        write_new_atomic(output, payload)
      self.assertEqual(list(output.parent.glob('*.tmp')), [])


if __name__ == '__main__':
  unittest.main()
