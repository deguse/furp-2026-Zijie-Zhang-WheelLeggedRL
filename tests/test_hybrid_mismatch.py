import unittest

import torch

from hoppertrex_mjlab.hybrid.mismatch import (
  STAGE1_MISMATCH_PROFILE_VERSION,
  Stage1MismatchAudit,
  _selected_mismatch_ids,
  stage1_mismatch_spec,
)


class Stage1MismatchTest(unittest.TestCase):
  def test_gate_group_selection_is_exact_and_reproducible(self):
    env_ids = torch.arange(16)

    selected = _selected_mismatch_ids(
      env_ids,
      mismatch_fraction=0.50,
      group_count=8,
      mismatch_group_indices=(7,),
    )

    self.assertEqual(selected.tolist(), [7, 15])

  def test_gate_group_selection_rejects_invalid_group(self):
    with self.assertRaisesRegex(ValueError, 'outside group_count'):
      _selected_mismatch_ids(
        torch.arange(4),
        mismatch_fraction=0.50,
        group_count=4,
        mismatch_group_indices=(4,),
      )

  def test_audit_selection_and_profile_document_symmetry(self):
    audit = Stage1MismatchAudit(
      is_mismatch=torch.tensor([True, False]),
      mass_inertia_scale=torch.tensor([1.02, 1.0]),
      com_x_offset_m=torch.tensor([0.001, 0.0]),
      com_z_offset_m=torch.tensor([-0.002, 0.0]),
      wheel_friction_scale=torch.tensor([0.95, 1.0]),
      wheel_radius_scale=torch.tensor([1.01, 1.0]),
      wheel_actuator_gain_scale=torch.tensor([1.03, 1.0]),
    )

    values = audit.select(torch.tensor([0]))
    spec = stage1_mismatch_spec()

    self.assertEqual(values['is_mismatch'], [True])
    self.assertAlmostEqual(values['wheel_radius_scale'][0], 1.01)
    self.assertEqual(spec['profile_version'], STAGE1_MISMATCH_PROFILE_VERSION)
    self.assertTrue(spec['left_right_shared'])


if __name__ == '__main__':
  unittest.main()
