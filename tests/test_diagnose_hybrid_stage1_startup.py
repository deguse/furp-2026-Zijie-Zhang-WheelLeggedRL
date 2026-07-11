import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from hoppertrex_mjlab.scripts.diagnose_hybrid_stage1_startup import (
  classify_startup,
  scenario_actions,
)


def _scenario(*, raw=(0.35, 0.34, 0.33), derived=None, contact=True):
  derived = raw if derived is None else derived
  return {
    'raw_root_z': list(raw),
    'derived_root_z': list(derived),
    'both_wheels_contact': [contact] * len(raw),
  }


class HybridStage1StartupDiagnosticTest(unittest.TestCase):
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


if __name__ == '__main__':
  unittest.main()
