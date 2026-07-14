import unittest

import torch

from hoppertrex_mjlab.scripts.rsl_rl.prepare_hybrid_stage1_extension import (
  prepare_stage1_extension_checkpoint,
  validate_stage1_source,
)


ACTION_NAMES = [
  'wheel_balance_residual',
  'wheel_yaw_residual',
  'left_thigh_residual',
  'right_thigh_residual',
  'left_knee_residual',
  'right_knee_residual',
]


def _checkpoint(std: float = 0.10) -> dict:
  return {
    'actor_state_dict': {
      'distribution.std_param': torch.full((6,), std),
      'actor.0.weight': torch.zeros((6, 3)),
    },
    'optimizer_state_dict': {
      'state': {1: {'momentum_buffer': torch.ones(1)}},
      'param_groups': [],
    },
    'iter': 123,
    'infos': {
      'hybrid_stage1_bootstrap': {
        'task': 'HopperTrex-Hybrid-v2-Stage1',
        'stage': 1,
        'action_order': ACTION_NAMES,
      },
      'hybrid_training': {'git_sha': 'training-revision'},
    },
  }


def _gate(sha: str = 'source-sha') -> dict:
  return {
    'suite': 'residual',
    'gate_pass': True,
    'checkpoint_file_sha256': sha,
  }


class PrepareHybridStage1ExtensionTest(unittest.TestCase):
  def test_prepares_same_stage_handoff_with_fresh_optimizer(self):
    prepared, provenance = prepare_stage1_extension_checkpoint(
      _checkpoint(),
      _gate(),
      source_checkpoint='C:/runs/stage1a/model.pt',
      source_checkpoint_sha256='source-sha',
      source_gate='C:/runs/stage1a/screen.json',
      source_gate_sha256='gate-sha',
      created_at='2026-07-14T12:00:00+08:00',
    )

    self.assertEqual(prepared['iter'], 0)
    self.assertEqual(prepared['optimizer_state_dict']['state'], {})
    self.assertNotIn('hybrid_training', prepared['infos'])
    self.assertEqual(
      provenance['target_profile_version'], 'stage1b_speed010_mild_v1',
    )
    self.assertEqual(provenance['source_iteration'], 123)
    self.assertEqual(
      prepared['infos']['hybrid_stage1_extension']['source_checkpoint_sha256'],
      'source-sha',
    )

  def test_rejects_gate_for_a_different_checkpoint(self):
    with self.assertRaisesRegex(ValueError, 'SHA256 does not match'):
      validate_stage1_source(
        _checkpoint(), _gate('other-sha'), source_sha256='source-sha',
      )

  def test_collapsed_balance_std_requires_explicit_reset(self):
    with self.assertRaisesRegex(ValueError, 'balance action std is collapsed'):
      prepare_stage1_extension_checkpoint(
        _checkpoint(std=0.01),
        _gate(),
        source_checkpoint='source.pt',
        source_checkpoint_sha256='source-sha',
        source_gate='screen.json',
        source_gate_sha256='gate-sha',
        created_at='2026-07-14T12:00:00+08:00',
      )

    prepared, provenance = prepare_stage1_extension_checkpoint(
      _checkpoint(std=0.01),
      _gate(),
      source_checkpoint='source.pt',
      source_checkpoint_sha256='source-sha',
      source_gate='screen.json',
      source_gate_sha256='gate-sha',
      reset_collapsed_active_std=True,
      created_at='2026-07-14T12:00:00+08:00',
    )

    self.assertAlmostEqual(
      float(prepared['actor_state_dict']['distribution.std_param'][0]), 0.15,
    )
    self.assertEqual(provenance['collapsed_active_actions'], [
      'wheel_balance_residual',
    ])


if __name__ == '__main__':
  unittest.main()
