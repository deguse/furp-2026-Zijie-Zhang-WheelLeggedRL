from types import SimpleNamespace
import unittest

from hoppertrex_mjlab.scripts.rsl_rl.train import (
  resolve_and_validate_hybrid_resume,
  validate_hybrid_repository_status,
  validate_hybrid_training_artifacts,
  validate_hybrid_training_checkpoint,
)


def _env_cfg(*, controller: bool, posture: bool, calibration: bool = True):
  return SimpleNamespace(
    actions={
      'hybrid_wheel_leg': SimpleNamespace(
        controller_qualified=controller,
        controller_gain_hash='controller123',
        posture_map_qualified=posture,
        calibration_hash=('calibration123' if calibration else None),
      )
    }
  )


def _checkpoint(*, target_stage: int | None = None):
  infos = {
    'hybrid_stage1_bootstrap': {
      'task': 'HopperTrex-Hybrid-v2-Stage1',
      'stage': 1,
      'controller_gain_hash': 'controller123',
      'calibration_hash': 'calibration123',
      'action_order': [
        'wheel_balance_residual',
        'wheel_yaw_residual',
        'left_thigh_residual',
        'right_thigh_residual',
        'left_knee_residual',
        'right_knee_residual',
      ],
    }
  }
  if target_stage is not None:
    infos['hybrid_stage_migration'] = {
      'source_stage': target_stage - 1,
      'target_stage': target_stage,
      'source_action_std': [0.1] * 6,
      'collapsed_active_actions': [],
      'reset_collapsed_active_std': False,
    }
  return {'infos': infos}


class HybridTrainPreflightTest(unittest.TestCase):
  def test_hybrid_training_requires_clean_repository(self):
    validate_hybrid_repository_status(
      'HopperTrex-Hybrid-v2-Stage1',
      '',
    )
    validate_hybrid_repository_status('legacy-task', ' M local.py')

    with self.assertRaisesRegex(ValueError, 'clean git worktree'):
      validate_hybrid_repository_status(
        'HopperTrex-Hybrid-v2-Stage1',
        ' M local.py',
      )

  def test_stage1_rejects_unqualified_controller(self):
    with self.assertRaisesRegex(ValueError, 'qualified controller'):
      validate_hybrid_training_artifacts(
        'HopperTrex-Hybrid-v2-Stage1',
        _env_cfg(controller=False, posture=False),
      )

  def test_stage3_rejects_missing_posture_map(self):
    with self.assertRaisesRegex(ValueError, 'qualified posture map'):
      validate_hybrid_training_artifacts(
        'HopperTrex-Hybrid-v2-Stage3',
        _env_cfg(controller=True, posture=False),
      )

  def test_stage1_rejects_missing_velocity_calibration(self):
    with self.assertRaisesRegex(ValueError, 'velocity calibration'):
      validate_hybrid_training_artifacts(
        'HopperTrex-Hybrid-v2-Stage1',
        _env_cfg(controller=True, posture=False, calibration=False),
      )

  def test_qualified_hybrid_and_legacy_tasks_pass(self):
    validate_hybrid_training_artifacts(
      'HopperTrex-Hybrid-v2-Stage4',
      _env_cfg(controller=True, posture=True),
    )
    validate_hybrid_training_artifacts('legacy-task', SimpleNamespace())

  def test_stage1_checkpoint_requires_matching_bootstrap_provenance(self):
    validate_hybrid_training_checkpoint(
      'HopperTrex-Hybrid-v2-Stage1',
      _env_cfg(controller=True, posture=False),
      _checkpoint(),
    )

    mismatched = _checkpoint()
    mismatched['infos']['hybrid_stage1_bootstrap']['controller_gain_hash'] = 'wrong'
    with self.assertRaisesRegex(ValueError, 'controller hash'):
      validate_hybrid_training_checkpoint(
        'HopperTrex-Hybrid-v2-Stage1',
        _env_cfg(controller=True, posture=False),
        mismatched,
      )

  def test_later_stage_requires_targeted_migration_and_std_audit(self):
    validate_hybrid_training_checkpoint(
      'HopperTrex-Hybrid-v2-Stage3',
      _env_cfg(controller=True, posture=True),
      _checkpoint(target_stage=3),
    )

    wrong_target = _checkpoint(target_stage=2)
    with self.assertRaisesRegex(ValueError, 'target'):
      validate_hybrid_training_checkpoint(
        'HopperTrex-Hybrid-v2-Stage3',
        _env_cfg(controller=True, posture=True),
        wrong_target,
      )

    skipped = _checkpoint(target_stage=3)
    skipped['infos']['hybrid_stage_migration']['source_stage'] = 1
    with self.assertRaisesRegex(ValueError, 'immediately preceding'):
      validate_hybrid_training_checkpoint(
        'HopperTrex-Hybrid-v2-Stage3',
        _env_cfg(controller=True, posture=True),
        skipped,
      )

  def test_later_stage_rejects_unreset_collapsed_active_action(self):
    checkpoint = _checkpoint(target_stage=2)
    migration = checkpoint['infos']['hybrid_stage_migration']
    migration['collapsed_active_actions'] = ['wheel_balance_residual']

    with self.assertRaisesRegex(ValueError, 'were not reset'):
      validate_hybrid_training_checkpoint(
        'HopperTrex-Hybrid-v2-Stage2',
        _env_cfg(controller=True, posture=False),
        checkpoint,
      )

  def test_hybrid_training_rejects_random_initialization(self):
    cfg = SimpleNamespace(agent=SimpleNamespace(resume=False))

    with self.assertRaisesRegex(ValueError, 'must resume'):
      resolve_and_validate_hybrid_resume(
        'HopperTrex-Hybrid-v2-Stage1',
        cfg,
      )


if __name__ == '__main__':
  unittest.main()
