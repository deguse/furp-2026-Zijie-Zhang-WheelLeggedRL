from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from hoppertrex_mjlab.hybrid.controller_schedule import (
  qr_candidate_grid,
  validate_flat_gate_selection,
)
from hoppertrex_mjlab.scripts import build_hybrid_controller_schedule as builder
from hoppertrex_mjlab.scripts import evaluate_hybrid_c1_affine_full_gate as gate


def _node_facts() -> dict[str, dict[str, object]]:
  return {
    stem: {
      'controllability_rank': 4,
      'max_nrmse': 0.1,
      'fallback_reasons': [],
      'command_gain_ratio': 0.74,
      'gain': [1.0, 2.0, 3.0, 4.0],
    }
    for stem in gate.NODE_STEMS
  }


def _retry_payload() -> dict[str, object]:
  candidates = []
  for index, registered in enumerate(qr_candidate_grid()):
    score = float(index + 1)
    candidate = {
      'index': index,
      'q_diag': registered['q_diag'],
      'r_diag': registered['r_diag'],
      'anchor_alpha': 0.25,
      'flat_gate_passed': True,
      'worst_velocity_error': score,
      'p95_pitch': score,
      'p99_pitch_rate': score,
      'wheel_target_rate': score,
      'node_facts': _node_facts(),
    }
    candidates.append(candidate)
  selected = candidates[24]
  selected.update(
    {
      'q_diag': gate.SELECTED_Q_DIAG,
      'r_diag': gate.SELECTED_R_DIAG,
      'worst_velocity_error': 0.005,
      'p95_pitch': 0.011,
      'p99_pitch_rate': 0.176,
      'wheel_target_rate': 0.365,
    }
  )
  return {
    'classification': 'AFFINE_CENTER_SMOKE_HAS_CANDIDATES',
    'git_sha': gate.RETRY_EVALUATOR_GIT_SHA,
    'collection_git_sha': gate.SOURCE_COLLECTION_GIT_SHA,
    'mjlab_git_sha': gate.EXPECTED_MJLAB_GIT_SHA,
    'passed_candidate_count': 27,
    'completed_candidate_count': 27,
    'completed_node_fit_count': 243,
    'evidence_eligible': True,
    'promotion_eligible': False,
    'training_eligible': False,
    'checkpoint': None,
    'next_step': 'DOWNLOAD_FOR_REVIEW',
    'bindings': gate.EXPECTED_BINDINGS,
    'candidates': candidates,
  }


def _retry_protocol() -> dict[str, object]:
  return {
    'kind': 'c1_affine_center_smoke_retry',
    'git_sha': gate.RETRY_EVALUATOR_GIT_SHA,
    'source_collection_git_sha': gate.SOURCE_COLLECTION_GIT_SHA,
    'source_zip_sha256': gate.SOURCE_COLLECTION_ZIP_SHA256,
    'mjlab_git_sha': gate.EXPECTED_MJLAB_GIT_SHA,
    'passed_candidate_count': 27,
    'classification': 'AFFINE_CENTER_SMOKE_HAS_CANDIDATES',
  }


class AffineFullGateTests(unittest.TestCase):
  def test_registered_protocol_has_exactly_fifteen_cells(self) -> None:
    cells = gate.evaluation_cells(gate.FORMAL_VX_CHECK)
    self.assertEqual(len(cells), 15)
    self.assertEqual(cells[:9], gate.evaluation_cells(0.0)[:9])
    self.assertEqual(cells[9], (0.3092089487, 0.0, 0.05))

  def test_load_selected_evidence_recovers_only_candidate_24(self) -> None:
    payload = _retry_payload()
    with (
      mock.patch.object(gate, '_require_sha'),
      mock.patch.object(
        gate, '_read_json', side_effect=[payload, _retry_protocol()]
      ),
    ):
      actual, selected, gains = gate.load_selected_evidence(
        source_zip=Path('source.zip'),
        retry_result=Path('result.json'),
        retry_protocol=Path('protocol.json'),
        retry_zip=Path('retry.zip'),
      )
    self.assertIs(actual, payload)
    self.assertEqual(selected['index'], 24)
    self.assertEqual(gains.shape, (3, 3, 4))
    self.assertTrue(np.all(gains == np.asarray([1.0, 2.0, 3.0, 4.0])))

  def test_selection_drift_is_rejected(self) -> None:
    payload = _retry_payload()
    payload['candidates'][0]['worst_velocity_error'] = 0.0  # type: ignore[index]
    with (
      mock.patch.object(gate, '_require_sha'),
      mock.patch.object(
        gate, '_read_json', side_effect=[payload, _retry_protocol()]
      ),
      self.assertRaisesRegex(ValueError, 'candidate 24'),
    ):
      gate.load_selected_evidence(
        source_zip=Path('source.zip'),
        retry_result=Path('result.json'),
        retry_protocol=Path('protocol.json'),
        retry_zip=Path('retry.zip'),
      )

  def test_payload_classifies_pass_and_failure_without_training(self) -> None:
    selected = _retry_payload()['candidates'][24]  # type: ignore[index]
    cell = {
      'target_height': 0.3092089487,
      'target_pitch': 0.0,
      'vx_command': 0.0,
      'velocity_error_abs': 0.001,
      'pitch_error_abs_p95': 0.001,
      'pitch_rate_abs_p99': 0.01,
      'wheel_target_rate_rms': 0.1,
      'terminated_events': 0.0,
      'non_wheel_contact_rate': 0.0,
    }
    retry = _retry_payload()
    passed = gate.make_payload(
      git_sha='a' * 40,
      retry_payload=retry,
      selected=selected,
      caps={
        'worst_velocity_error': 0.01,
        'p95_pitch': 0.02,
        'p99_pitch_rate': 0.3,
      },
      floors={},
      cells=[cell] * 15,
    )
    self.assertEqual(passed['classification'], 'C1_AFFINE_FULL_GATE_SELECTED')
    self.assertEqual(passed['completed_candidate_count'], 1)
    self.assertFalse(passed['training_eligible'])
    failed_cell = dict(cell, terminated_events=1.0)
    failed = gate.make_payload(
      git_sha='a' * 40,
      retry_payload=retry,
      selected=selected,
      caps=passed['caps'],
      floors={},
      cells=[failed_cell] * 15,
    )
    self.assertEqual(failed['classification'], 'C1_AFFINE_FULL_GATE_FAILED_STOP')
    self.assertEqual(failed['next_step'], 'STOP')

  def test_selection_evidence_is_schedule_builder_compatible(self) -> None:
    retry = _retry_payload()
    selected = retry['candidates'][24]  # type: ignore[index]
    cell = {
      'target_height': 0.3092089487,
      'target_pitch': 0.0,
      'vx_command': 0.0,
      'velocity_error_abs': 0.001,
      'pitch_error_abs_p95': 0.001,
      'pitch_rate_abs_p99': 0.01,
      'wheel_target_rate_rms': 0.1,
      'terminated_events': 0.0,
      'non_wheel_contact_rate': 0.0,
    }
    detail = gate.make_payload(
      git_sha='a' * 40,
      retry_payload=retry,
      selected=selected,
      caps={
        'worst_velocity_error': 0.01,
        'p95_pitch': 0.02,
        'p99_pitch_rate': 0.3,
      },
      floors={},
      cells=[cell] * 15,
    )
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      detail_path = root / 'c1_affine_full_gate.json'
      detail_path.write_text(json.dumps(detail), encoding='utf-8')
      selection = gate.make_selection_payload(
        detail=detail,
        retry_payload=retry,
        detail_path=detail_path,
      )
      selection_path = root / 'c1_affine_full_gate_selection.json'
      selection_path.write_text(json.dumps(selection), encoding='utf-8')
      loaded = builder._validated_selection(
        {
          'evaluation_artifact_path': selection_path.name,
          'evaluation_artifact_sha256': hashlib.sha256(
            selection_path.read_bytes()
          ).hexdigest(),
        },
        root,
      )
    validate_flat_gate_selection(
      loaded,
      selected_q_diag=tuple(gate.SELECTED_Q_DIAG),
      selected_r_diag=tuple(gate.SELECTED_R_DIAG),
      selected_anchor_alpha=gate.SELECTED_ANCHOR_ALPHA,
    )
    loaded['final_gate_candidate']['flat_gate_passed'] = False
    with self.assertRaisesRegex(ValueError, 'did not pass'):
      validate_flat_gate_selection(
        loaded,
        selected_q_diag=tuple(gate.SELECTED_Q_DIAG),
        selected_r_diag=tuple(gate.SELECTED_R_DIAG),
        selected_anchor_alpha=gate.SELECTED_ANCHOR_ALPHA,
      )

  def test_formal_arguments_are_frozen(self) -> None:
    args = argparse.Namespace(
      task=gate.FORMAL_TASK,
      device=gate.FORMAL_DEVICE,
      num_envs=gate.FORMAL_NUM_ENVS,
      settle_steps=gate.FORMAL_SETTLE_STEPS,
      measure_steps=gate.FORMAL_MEASURE_STEPS,
      vx_check=gate.FORMAL_VX_CHECK,
      mjlab_git_sha=gate.EXPECTED_MJLAB_GIT_SHA,
      git_sha='a' * 40,
    )
    gate._validate_formal_args(args)
    args.num_envs = 32
    with self.assertRaisesRegex(ValueError, 'frozen'):
      gate._validate_formal_args(args)


if __name__ == '__main__':
  unittest.main()
