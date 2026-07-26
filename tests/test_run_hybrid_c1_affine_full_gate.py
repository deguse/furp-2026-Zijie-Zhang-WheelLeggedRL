from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / 'scripts' / 'run_hybrid_c1_affine_full_gate.ps1'
SELF_HASH = WRAPPER.with_suffix(WRAPPER.suffix + '.sha256')


class AffineFullGateWrapperTests(unittest.TestCase):
  def test_wrapper_is_fixed_single_candidate_protocol(self) -> None:
    text = WRAPPER.read_text(encoding='utf-8')
    required = (
      '9fe48c31a5cc1c3cbea8b163d3fafe860e3aba53',
      '43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6',
      '18cea95353b227b47370af25265f16c2450ba25e224069e08c52f92d6d472f07',
      '86521c7e5762b669a2c179c590f5c08fbd6454d165087ee8a02b86ae293f14dd',
      'evaluate_hybrid_c1_affine_full_gate',
      '--num-envs',
      "'16'",
      '--settle-steps',
      "'100'",
      '--measure-steps',
      "'200'",
      '[int]$result.candidate.index -ne 24',
      '[int]$result.completed_candidate_count -ne 1',
      '[int]$result.completed_cell_count -ne 15',
      'c1_affine_full_gate_selection.json',
      'affine_full_gate_selected',
      '@($selection.screened_candidates).Count -ne 27',
      'HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH',
      '.incomplete.',
      'Compress-Archive',
      'nvidia-smi',
      '--help',
    )
    for token in required:
      self.assertIn(token, text)
    forbidden = (
      'train.py',
      'migrate_hybrid_stage',
      'checkpoint-path',
      'evaluate_hybrid_c1_affine_center_smoke',
      'fit_all_candidates',
      'CEM',
      'PPO',
    )
    for token in forbidden:
      self.assertNotIn(token, text)

  def test_self_hash_matches_wrapper(self) -> None:
    expected = SELF_HASH.read_text(encoding='ascii').strip()
    actual = hashlib.sha256(WRAPPER.read_bytes()).hexdigest()
    self.assertEqual(actual, expected)


if __name__ == '__main__':
  unittest.main()
