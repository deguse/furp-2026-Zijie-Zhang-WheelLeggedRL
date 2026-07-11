from types import SimpleNamespace
import unittest

from hoppertrex_mjlab.scripts.rsl_rl.train import (
  validate_hybrid_training_artifacts,
)


def _env_cfg(*, controller: bool, posture: bool, calibration: bool = True):
  return SimpleNamespace(
    actions={
      'hybrid_wheel_leg': SimpleNamespace(
        controller_qualified=controller,
        posture_map_qualified=posture,
        calibration_hash=('c' * 64 if calibration else None),
      )
    }
  )


class HybridTrainPreflightTest(unittest.TestCase):
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


if __name__ == '__main__':
  unittest.main()
