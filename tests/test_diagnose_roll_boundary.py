import unittest
from unittest.mock import patch

import torch

from hoppertrex_mjlab.scripts import diagnose_roll_boundary as diag


class DiagnoseRollBoundaryTest(unittest.TestCase):
  def test_schedule_grid_is_frozen_to_twelve_dynamic_and_two_sentinels(self):
    candidates = diag.schedule_grid_candidates()
    self.assertEqual(len(candidates), 14)
    self.assertEqual(
      sum(candidate['kind'] == 'position_indexed_schedule' for candidate in candidates),
      12,
    )
    self.assertEqual(
      [candidate['name'] for candidate in candidates[-2:]],
      [
        'static_low_h290732um_p-032000urad',
        'static_high_h327686um_p+032000urad',
      ],
    )

  def test_schedule_grid_reuses_identical_matched_reset_perturbations(self):
    perturbations = []
    for candidate in diag.schedule_grid_candidates():
      card = candidate['posture_card']
      with patch.object(diag.rb, 'POSTURE_CARDS', (card,)):
        perturbations.append(diag.rb.reset_perturbations(
          slots=8, card_name=card['name'], repeat=1,
        ))
    for observed in perturbations[1:]:
      self.assertTrue(torch.equal(observed, perturbations[0]))

  def test_schedule_grid_cli_mode_is_registered(self):
    args = diag.parse_args([
      '--mode', 'schedule-grid', '--output', 'outside.json',
      '--envs-per-height', '1', '--repeats', '1',
      '--settle-steps', '1', '--drive-steps', '1', '--stable-steps', '1',
    ])
    self.assertEqual(args.mode, 'schedule-grid')

  def test_posture_grid_uses_registered_nodes_and_endpoints(self):
    payload = {
      'fit_criteria': {'fixed_height_nodes': [0.29, 0.31, 0.33]},
      'training_envelope': {'pitch': [-0.032, 0.032]},
    }
    cards = diag.posture_grid(payload, pitch_count=5)
    self.assertEqual(len(cards), 15)
    self.assertEqual(cards[0]['height_m'], 0.29)
    self.assertAlmostEqual(cards[0]['pitch_rad'], -0.032)
    self.assertAlmostEqual(cards[4]['pitch_rad'], 0.032)
    self.assertEqual(cards[-1]['height_m'], 0.33)

  def test_event_window_keeps_pre_trigger_and_post_trigger_samples(self):
    collector = diag.EventWindowCollector(1, pre_substeps=2, post_substeps=2)
    for step in range(1, 6):
      collector.observe(
        0, {'substep': step}, active=True, unsupported=(step == 3),
      )
    self.assertEqual(
      [item['substep'] for item in collector.events[0]['samples']],
      [1, 2, 3, 4, 5],
    )

  def test_summarize_trials_counts_force_events(self):
    rows = []
    for height in diag.DIAGNOSTIC_HEIGHTS_M:
      rows.append({
        'stair_height_m': height, 'success': height == 0.0,
        'bilateral_airborne_ever': height != 0.0,
        'bilateral_unsupported_physics_substeps': int(height != 0.0),
        'non_wheel_contact': False, 'termination': False,
        'max_progress_past_face_m': 0.1,
      })
    summaries = diag.summarize_trials(rows)
    self.assertEqual(summaries[0]['successes'], 1)
    self.assertEqual(summaries[1]['bilateral_airborne_trials'], 1)
    self.assertEqual(summaries[1]['bilateral_unsupported_physics_substeps'], 1)
    self.assertNotIn('applied_residual_abs_max', summaries[0])
    self.assertNotIn('dynamic_leg_feedforward_abs_max_rad', summaries[0])

  def test_summarize_trials_reports_only_fully_measured_authority_metrics(self):
    rows = []
    for height in diag.DIAGNOSTIC_HEIGHTS_M:
      for value in (0.0, 0.25):
        rows.append({
          'stair_height_m': height, 'success': False,
          'bilateral_airborne_ever': False,
          'bilateral_unsupported_physics_substeps': 0,
          'non_wheel_contact': False, 'termination': False,
          'max_progress_past_face_m': 0.1,
          'applied_residual_abs_max': value,
          'wheel_target_classical_path_abs_max_radps': 0.0,
          'dynamic_leg_feedforward_abs_max_rad': 0.0,
          'dynamic_drive_feedforward_abs_max_radps': 0.0,
        })
    summaries = diag.summarize_trials(rows)
    self.assertEqual(summaries[0]['applied_residual_abs_max'], 0.25)
    del rows[0]['applied_residual_abs_max']
    with self.assertRaisesRegex(ValueError, 'partially missing'):
      diag.summarize_trials(rows)


if __name__ == '__main__':
  unittest.main()
