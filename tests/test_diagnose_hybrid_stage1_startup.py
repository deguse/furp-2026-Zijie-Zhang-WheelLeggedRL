import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from hoppertrex_mjlab.scripts.diagnose_hybrid_stage1_startup import (
  classify_matrix,
  classify_startup,
  collect_matrix_case,
  first_bad_substep,
  matrix_cases,
  scenario_actions,
  summarize,
)


def _scenario(*, raw=(0.35, 0.34, 0.33), derived=None, contact=True):
  derived = raw if derived is None else derived
  return {
    'raw_root_z': list(raw),
    'derived_root_z': list(derived),
    'both_wheels_contact': [contact] * len(raw),
  }


class HybridStage1StartupDiagnosticTest(unittest.TestCase):
  def test_matrix_cases_cover_two_stages_and_five_scales(self):
    self.assertEqual(
      matrix_cases(),
      tuple((stage, envs) for stage in (0, 1) for envs in (1, 2, 4, 8, 16)),
    )

  def test_first_bad_substep_uses_velocity_drop_and_non_finite_rules(self):
    healthy = [
      {'substep': -1, 'raw_root_z': 0.35, 'raw_root_qvel_z': 0.0},
      {'substep': 0, 'raw_root_z': 0.3499, 'raw_root_qvel_z': -0.049},
    ]
    self.assertIsNone(first_bad_substep(healthy, timestep=0.005))
    violent = healthy + [
      {'substep': 1, 'raw_root_z': 0.31, 'raw_root_qvel_z': -2.0},
    ]
    self.assertEqual(first_bad_substep(violent, timestep=0.005), 1)
    non_finite = healthy + [
      {'substep': 1, 'raw_root_z': math.nan, 'raw_root_qvel_z': -0.1},
    ]
    self.assertEqual(first_bad_substep(non_finite, timestep=0.005), 1)

  def test_matrix_classification_prioritizes_reset_scale_stage_and_contact(self):
    def result(stage, envs, *, bad=None, reset_qz=0.35, reset_qvz=0.0, contact=True):
      return {
        'stage': stage, 'num_envs': envs, 'first_bad_substep': bad,
        'reset_qz_min': reset_qz, 'reset_qvz_abs_max': abs(reset_qvz),
        'wheel_contact_any': contact, 'finite': True,
      }

    reset_bad = [result(s, n, reset_qvz=-4.0) for s, n in matrix_cases()]
    self.assertEqual(classify_matrix(reset_bad), 'invalid_reset_state')

    scale_bad = [
      result(s, n, bad=(0 if n >= 4 else None)) for s, n in matrix_cases()
    ]
    self.assertEqual(classify_matrix(scale_bad), 'cuda_scale_dependent_dynamics')

    stage_bad = [
      result(s, n, bad=(0 if s == 1 else None)) for s, n in matrix_cases()
    ]
    self.assertEqual(classify_matrix(stage_bad), 'stage_specific_startup')

    no_contact = [result(s, n, contact=False) for s, n in matrix_cases()]
    self.assertEqual(classify_matrix(no_contact), 'contact_initialization_failure')

  def test_cpu_matrix_reset_and_first_step_are_stable_for_stage0_and_stage1(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      args = SimpleNamespace(
        controller_path=None,
        calibration_path=None,
        device='cpu',
        seed=1,
        environment_log_path=Path(temp_dir) / 'environment.log',
      )
      for stage in (0, 1):
        for num_envs in (1, 16):
          with self.subTest(stage=stage, num_envs=num_envs):
            _rows, result = collect_matrix_case(
              args,
              stage=stage,
              num_envs=num_envs,
            )
            self.assertAlmostEqual(result['reset_qz_mean'], 0.35, places=5)
            self.assertAlmostEqual(result['reset_qvz_abs_max'], 0.0, places=6)
            self.assertIsNone(result['first_bad_substep'])
  def test_module_imports_with_only_repository_src_on_pythonpath(self):
    repository = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env['PYTHONPATH'] = str(repository / 'src')
    completed = subprocess.run(
      [
        sys.executable,
        '-c',
        'import hoppertrex_mjlab.scripts.diagnose_hybrid_stage1_startup',
      ],
      cwd=repository,
      env=env,
      capture_output=True,
      text=True,
    )
    self.assertEqual(completed.returncode, 0, completed.stderr)

  def test_scenario_actions_are_deterministic_and_only_enable_balance(self):
    first = scenario_actions('std', num_envs=4, seed=1)
    second = scenario_actions('std', num_envs=4, seed=1)
    np.testing.assert_array_equal(first, second)
    self.assertEqual(first.shape, (4, 6))
    np.testing.assert_array_equal(first[:, 1:], 0.0)
    np.testing.assert_array_equal(
      scenario_actions('zero', num_envs=4, seed=1),
      np.zeros((4, 6)),
    )

  def test_classifies_stale_derived_state_before_physical_failures(self):
    scenarios = {
      'zero': _scenario(derived=(0.0, 0.0, 0.0)),
      'std': _scenario(),
      'controller_off': _scenario(),
    }
    self.assertEqual(classify_startup(scenarios), 'derived_state_stale')

  def test_classifies_reset_controller_exploration_and_contact_failures(self):
    invalid = {name: _scenario(raw=(0.20, 0.20, 0.20)) for name in (
      'zero', 'std', 'controller_off'
    )}
    self.assertEqual(classify_startup(invalid), 'invalid_reset_height')

    controller = {
      'zero': _scenario(raw=(0.35, 0.28, 0.20)),
      'std': _scenario(raw=(0.35, 0.28, 0.20)),
      'controller_off': _scenario(raw=(0.35, 0.28, 0.20)),
    }
    self.assertEqual(classify_startup(controller), 'controller_startup_failure')

    exploration = {
      'zero': _scenario(),
      'std': _scenario(raw=(0.35, 0.28, 0.20)),
      'controller_off': _scenario(),
    }
    self.assertEqual(classify_startup(exploration), 'exploration_startup_failure')

    no_contact = {name: _scenario(contact=False) for name in (
      'zero', 'std', 'controller_off'
    )}
    self.assertEqual(
      classify_startup(no_contact),
      'contact_initialization_failure',
    )

  def test_rejects_non_finite_measurements(self):
    scenarios = {
      'zero': _scenario(raw=(0.35, math.nan, 0.33)),
      'std': _scenario(),
      'controller_off': _scenario(),
    }
    with self.assertRaisesRegex(ValueError, 'finite'):
      classify_startup(scenarios)

  def test_summary_is_compact_and_aggregates_each_step(self):
    rows = []
    for scenario in ('zero', 'std', 'controller_off'):
      for step in range(2):
        for env_id, root_z in enumerate((0.30 - step * 0.05, 0.32 - step * 0.05)):
          rows.append({
            'scenario': scenario,
            'step': step,
            'env_id': env_id,
            'raw_root_z': root_z,
            'derived_root_z': root_z,
            'both_wheels_contact': step == 1,
            'would_root_too_low': root_z < 0.26,
          })

    summary = summarize(rows)

    self.assertNotIn('raw_root_z', summary['scenarios']['zero'])
    self.assertEqual(summary['scenarios']['zero']['first_both_wheels_contact_step'], 1)
    self.assertEqual(summary['scenarios']['zero']['first_root_too_low_step'], 1)
    self.assertEqual(
      summary['scenarios']['zero']['steps'][0],
      {
        'step': 0,
        'raw_root_z': {'min': 0.30, 'mean': 0.31, 'max': 0.32},
        'derived_root_z': {'min': 0.30, 'mean': 0.31, 'max': 0.32},
        'both_wheels_contact_rate': 0.0,
        'root_too_low_rate': 0.0,
      },
    )


if __name__ == '__main__':
  unittest.main()
