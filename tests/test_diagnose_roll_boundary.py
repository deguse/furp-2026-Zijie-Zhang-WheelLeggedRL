import unittest

from hoppertrex_mjlab.scripts import diagnose_roll_boundary as diag


class DiagnoseRollBoundaryTest(unittest.TestCase):
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


if __name__ == '__main__':
  unittest.main()
