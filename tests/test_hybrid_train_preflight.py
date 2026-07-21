from types import SimpleNamespace
import unittest

from hoppertrex_mjlab.scripts.rsl_rl.train import (
  resolve_and_validate_hybrid_resume,
  validate_hybrid_repository_status,
  validate_hybrid_training_artifacts,
  validate_hybrid_training_checkpoint,
)


def _env_cfg(
  *, controller: bool, posture: bool, calibration: bool = True,
  yaw: bool = True,
  station: bool = True,
  stage1_profile_version: str | None = None,
  action_scales: tuple[float, ...] = (0.5, 0.3, 0.035, 0.035, 0.035, 0.035),
):
  return SimpleNamespace(
    actions={
      'hybrid_wheel_leg': SimpleNamespace(
        controller_qualified=controller,
        controller_gain_hash='controller123',
        posture_map_qualified=posture,
        posture_map_hash=('map123' if posture else None),
        calibration_hash=('calibration123' if calibration else None),
        yaw_calibration_qualified=yaw,
        yaw_calibration_hash=('yaw123' if yaw else None),
        station_calibration_qualified=station,
        station_calibration_hash=('station123' if station else None),
        action_scales=action_scales,
      )
    },
    stage1_profile_version=stage1_profile_version,
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
    if target_stage >= 2:
      infos['hybrid_stage_migration']['yaw_calibration_hash'] = 'yaw123'
    if target_stage >= 3:
      infos['hybrid_stage_migration'].update({
        'posture_map_hash': 'map123',
        'station_calibration_hash': 'station123',
      })
    if target_stage == 2:
      infos['hybrid_stage_migration'].update({
        'source_checkpoint_sha256': 'source-checkpoint-sha',
        'source_gate': 'stage1-formal.json',
        'source_gate_sha256': 'source-gate-sha',
        'source_gate_profile': 'formal',
        'source_gate_suite': 'residual',
        'source_gate_stage1_profile_version': 'stage1b_speed010_mild_v1',
      })
  return {'infos': infos}


def _stage1b_checkpoint(*, collapsed: list[str] | None = None, reset: bool = False):
  checkpoint = _checkpoint()
  checkpoint['infos']['hybrid_stage1_extension'] = {
    'target_profile_version': 'stage1b_speed010_mild_v1',
    'source_action_std': [0.1] * 6,
    'collapsed_active_actions': [] if collapsed is None else collapsed,
    'reset_collapsed_active_std': reset,
  }
  return checkpoint


class HybridTrainPreflightTest(unittest.TestCase):
  def test_experimental_leg_authority_requires_matching_migration(self):
    checkpoint = _checkpoint(target_stage=5)
    checkpoint["infos"]["hybrid_stage_migration"]["target_action_scales"] = (
      [0.5, 0.3, 0.07, 0.07, 0.07, 0.07]
    )
    env_cfg = _env_cfg(
      controller=True,
      posture=True,
      action_scales=(0.5, 0.3, 0.07, 0.07, 0.07, 0.07),
    )
    validate_hybrid_training_checkpoint(
      "HopperTrex-Hybrid-v2-Stage5", env_cfg, checkpoint
    )
    checkpoint["infos"]["hybrid_stage_migration"]["target_action_scales"] = (
      [0.5, 0.3, 0.10, 0.10, 0.10, 0.10]
    )
    with self.assertRaisesRegex(ValueError, "action scales"):
      validate_hybrid_training_checkpoint(
        "HopperTrex-Hybrid-v2-Stage5", env_cfg, checkpoint
      )

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

  def test_stage3_rejects_missing_station_calibration(self):
    # Stage 3.0 (2026-07-15): the probe measured a steady drift affine in the
    # commanded pitch; training a residual on top of that unowned defect
    # would repeat the yaw misallocation.
    with self.assertRaisesRegex(ValueError, 'station-keeping'):
      validate_hybrid_training_artifacts(
        'HopperTrex-Hybrid-v2-Stage3',
        _env_cfg(controller=True, posture=True, station=False),
      )
    # Stages 1-2 predate posture commands and stay launchable without it.
    validate_hybrid_training_artifacts(
      'HopperTrex-Hybrid-v2-Stage2',
      _env_cfg(controller=True, posture=False, station=False),
    )

  def test_stage1_rejects_missing_velocity_calibration(self):
    with self.assertRaisesRegex(ValueError, 'velocity calibration'):
      validate_hybrid_training_artifacts(
        'HopperTrex-Hybrid-v2-Stage1',
        _env_cfg(controller=True, posture=False, calibration=False),
      )

  def test_stage2_rejects_missing_yaw_calibration(self):
    with self.assertRaisesRegex(ValueError, 'yaw calibration'):
      validate_hybrid_training_artifacts(
        'HopperTrex-Hybrid-v2-Stage2',
        _env_cfg(controller=True, posture=False, yaw=False),
      )
    # Stage1 predates yaw calibration and must stay launchable without it.
    validate_hybrid_training_artifacts(
      'HopperTrex-Hybrid-v2-Stage1',
      _env_cfg(controller=True, posture=False, yaw=False),
    )

  def test_stage2_checkpoint_rejects_yaw_calibration_hash_mismatch(self):
    checkpoint = _checkpoint(target_stage=2)
    checkpoint['infos']['hybrid_stage_migration']['yaw_calibration_hash'] = (
      'other-yaw'
    )

    with self.assertRaisesRegex(ValueError, 'yaw calibration hash'):
      validate_hybrid_training_checkpoint(
        'HopperTrex-Hybrid-v2-Stage2',
        _env_cfg(controller=True, posture=False),
        checkpoint,
      )

    missing = _checkpoint(target_stage=2)
    missing['infos']['hybrid_stage_migration'].pop('yaw_calibration_hash')
    with self.assertRaisesRegex(ValueError, 'yaw calibration hash'):
      validate_hybrid_training_checkpoint(
        'HopperTrex-Hybrid-v2-Stage2',
        _env_cfg(controller=True, posture=False),
        missing,
      )

  def test_stage3_checkpoint_rejects_station_or_posture_hash_mismatch(self):
    checkpoint = _checkpoint(target_stage=3)
    checkpoint['infos']['hybrid_stage_migration'][
      'station_calibration_hash'
    ] = 'other-station'
    with self.assertRaisesRegex(ValueError, 'station calibration hash'):
      validate_hybrid_training_checkpoint(
        'HopperTrex-Hybrid-v2-Stage3',
        _env_cfg(controller=True, posture=True),
        checkpoint,
      )

    wrong_map = _checkpoint(target_stage=3)
    wrong_map['infos']['hybrid_stage_migration']['posture_map_hash'] = 'other'
    with self.assertRaisesRegex(ValueError, 'posture map hash'):
      validate_hybrid_training_checkpoint(
        'HopperTrex-Hybrid-v2-Stage3',
        _env_cfg(controller=True, posture=True),
        wrong_map,
      )

    validate_hybrid_training_checkpoint(
      'HopperTrex-Hybrid-v2-Stage3',
      _env_cfg(controller=True, posture=True),
      _checkpoint(target_stage=3),
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

  def test_stage1b_requires_matching_same_stage_extension(self):
    env_cfg = _env_cfg(
      controller=True,
      posture=False,
      stage1_profile_version='stage1b_speed010_mild_v1',
    )
    with self.assertRaisesRegex(ValueError, 'prepare_hybrid_stage1_extension'):
      validate_hybrid_training_checkpoint(
        'HopperTrex-Hybrid-v2-Stage1', env_cfg, _checkpoint(),
      )

    validate_hybrid_training_checkpoint(
      'HopperTrex-Hybrid-v2-Stage1', env_cfg, _stage1b_checkpoint(),
    )

  def test_stage1b_rejects_unreset_collapsed_exploration(self):
    env_cfg = _env_cfg(
      controller=True,
      posture=False,
      stage1_profile_version='stage1b_speed010_mild_v1',
    )
    with self.assertRaisesRegex(ValueError, 'not reset'):
      validate_hybrid_training_checkpoint(
        'HopperTrex-Hybrid-v2-Stage1',
        env_cfg,
        _stage1b_checkpoint(
          collapsed=['wheel_balance_residual'],
          reset=False,
        ),
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

  def test_stage2_requires_stage1_formal_gate_audit(self):
    checkpoint = _checkpoint(target_stage=2)
    checkpoint['infos']['hybrid_stage_migration'].pop('source_gate_sha256')

    with self.assertRaisesRegex(ValueError, 'formal gate audit'):
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
