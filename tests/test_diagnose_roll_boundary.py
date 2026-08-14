import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from hoppertrex_mjlab.scripts import diagnose_roll_boundary as diag


def _summary_row(height: float, **updates):
  row = {
    'stair_height_m': height,
    'success': height == 0.0,
    'bilateral_airborne_ever': height != 0.0,
    'bilateral_unsupported_physics_substeps': int(height != 0.0),
    'non_wheel_contact': False,
    'termination': False,
    'max_progress_past_face_m': 0.1,
    'peak_pitch_rate_abs_radps': 0.2,
    'torque_saturation_fraction': 0.25,
    'wheel_residual_abs_max': 0.0,
  }
  row.update(updates)
  return row


def _schedule_trial(
  candidate, *, terrain_index: int, repeat: int, env_id: int,
):
  card = candidate['posture_card']
  schedule = candidate['schedule']
  nominal_alpha = None
  applied_alpha = None
  height_alpha = None
  pitch_alpha = None
  applied_height = None
  applied_pitch = None
  channel_gap = None
  if schedule is not None:
    nominal_alpha = 0.1
    desired_height = schedule.start_height_m + nominal_alpha * (
      schedule.climb_height_m - schedule.start_height_m
    )
    desired_pitch = schedule.start_pitch_rad + nominal_alpha * (
      schedule.climb_pitch_rad - schedule.start_pitch_rad
    )
    if candidate.get('slew_mode') == diag.SYNCHRONIZED_SLEW_MODE:
      alpha_step = min(
        diag.POSTURE_HEIGHT_SLEW_RATE_MPS
        / abs(schedule.climb_height_m - schedule.start_height_m),
        diag.POSTURE_PITCH_SLEW_RATE_RADPS
        / abs(schedule.climb_pitch_rad - schedule.start_pitch_rad),
      ) * diag.CONTROL_DT_S
      applied_alpha = min(nominal_alpha, alpha_step)
      height_alpha = applied_alpha
      pitch_alpha = applied_alpha
      applied_height = schedule.start_height_m + height_alpha * (
        schedule.climb_height_m - schedule.start_height_m
      )
      applied_pitch = schedule.start_pitch_rad + pitch_alpha * (
        schedule.climb_pitch_rad - schedule.start_pitch_rad
      )
    else:
      height_step = diag.POSTURE_HEIGHT_SLEW_RATE_MPS * diag.CONTROL_DT_S
      pitch_step = diag.POSTURE_PITCH_SLEW_RATE_RADPS * diag.CONTROL_DT_S
      applied_height = schedule.start_height_m + min(
        max(desired_height - schedule.start_height_m, -height_step), height_step,
      )
      applied_pitch = schedule.start_pitch_rad + min(
        max(desired_pitch - schedule.start_pitch_rad, -pitch_step), pitch_step,
      )
      height_alpha = (
        (applied_height - schedule.start_height_m)
        / (schedule.climb_height_m - schedule.start_height_m)
      )
      pitch_alpha = (
        (applied_pitch - schedule.start_pitch_rad)
        / (schedule.climb_pitch_rad - schedule.start_pitch_rad)
      )
      applied_alpha = min(height_alpha, pitch_alpha)
    channel_gap = abs(height_alpha - pitch_alpha)
  row = {
    'posture_card': candidate['name'],
    'target_height_m': float(card['height_m']),
    'target_pitch_rad': float(card['pitch_rad']),
    'stair_height_m': diag.DIAGNOSTIC_HEIGHTS_M[terrain_index],
    'terrain_key': diag.rb.terrain_key(diag.DIAGNOSTIC_HEIGHTS_M[terrain_index]),
    'terrain_index': terrain_index,
    'repeat': repeat,
    'env_id': env_id,
    'success': False,
    'time_to_success_s': None,
    'termination': False,
    'non_wheel_contact': False,
    'bilateral_airborne_ever': False,
    'bilateral_unsupported_physics_substeps': 0,
    'bilateral_positive_clearance_ever': False,
    'actual_wheel_actuator_force_abs_max_nm': 1.0,
    'wheel_residual_abs_max': 0.0,
    'peak_pitch_abs_rad': 0.1,
    'peak_roll_abs_rad': 0.1,
    'peak_pitch_rate_abs_radps': 0.2,
    'torque_saturation_fraction': 0.25,
    'max_progress_past_face_m': -0.01,
    'root_reset': {},
    'applied_residual_abs_max': 0.0,
    'wheel_target_classical_path_abs_max_radps': 0.0,
    'dynamic_leg_feedforward_abs_max_rad': 0.0,
    'dynamic_drive_feedforward_abs_max_radps': 0.0,
    'first_support_loss_progress_m': None,
    'left_vertical_normal_load_n_mean': 100.0,
    'right_vertical_normal_load_n_mean': 100.0,
    'total_vertical_normal_load_n_mean': 200.0,
    'total_vertical_normal_load_n_min_control_step': 150.0,
    'control_trace': [{
      'control_step': 1,
      'progress_m': -0.1,
      'root_z_m': 0.3,
      'root_vz_mps': 0.0,
      'pitch_rad': float(card['pitch_rad']),
      'pitch_rate_radps': 0.0,
      'left_vertical_normal_load_n': 100.0,
      'right_vertical_normal_load_n': 100.0,
      'total_vertical_normal_load_n': 200.0,
      'schedule_nominal_alpha': nominal_alpha,
      'schedule_applied_alpha': applied_alpha,
      'schedule_applied_height_alpha': height_alpha,
      'schedule_applied_pitch_alpha': pitch_alpha,
      'applied_height_m': applied_height,
      'applied_pitch_rad': applied_pitch,
    }],
  }
  if schedule is None:
    row.update({
      'roll_pose_schedule': None,
      'drive_start_x_m': None,
      'end_distance_to_riser_m': None,
      'schedule_slew_mode': None,
      'schedule_alpha_max': None,
      'schedule_nominal_alpha_final': None,
      'schedule_applied_alpha_final': None,
      'schedule_applied_height_alpha_final': None,
      'schedule_applied_pitch_alpha_final': None,
      'maximum_applied_channel_alpha_gap': None,
      'desired_height_m_final': float(card['height_m']),
      'desired_pitch_rad_final': float(card['pitch_rad']),
      'applied_height_m_final': float(card['height_m']),
      'applied_pitch_rad_final': float(card['pitch_rad']),
      'maximum_height_tracking_lag_m': 0.0,
      'maximum_pitch_tracking_lag_rad': 0.0,
      'height_transition_completion_step': None,
      'pitch_transition_completion_step': None,
      'transition_completion_step': None,
      'transition_completed_before_face': None,
    })
  else:
    row.update({
      'roll_pose_schedule': schedule.to_dict(),
      'drive_start_x_m': -0.2,
      'end_distance_to_riser_m': schedule.end_distance_to_riser_m,
      'schedule_slew_mode': candidate.get(
        'slew_mode', diag.INDEPENDENT_SLEW_MODE,
      ),
      'schedule_alpha_max': nominal_alpha,
      'schedule_nominal_alpha_final': nominal_alpha,
      'schedule_applied_alpha_final': applied_alpha,
      'schedule_applied_height_alpha_final': height_alpha,
      'schedule_applied_pitch_alpha_final': pitch_alpha,
      'maximum_applied_channel_alpha_gap': channel_gap,
      'desired_height_m_final': desired_height,
      'desired_pitch_rad_final': desired_pitch,
      'applied_height_m_final': applied_height,
      'applied_pitch_rad_final': applied_pitch,
      'maximum_height_tracking_lag_m': 0.0,
      'maximum_pitch_tracking_lag_rad': 0.0,
      'height_transition_completion_step': None,
      'pitch_transition_completion_step': None,
      'transition_completion_step': None,
      'transition_completed_before_face': False,
    })
  return row


def _schedule_trials(candidate, *, repeats: int = 1, envs_per_height: int = 1):
  rows = []
  for repeat in range(1, repeats + 1):
    for terrain_index in range(len(diag.DIAGNOSTIC_HEIGHTS_M)):
      for slot in range(envs_per_height):
        rows.append(_schedule_trial(
          candidate,
          terrain_index=terrain_index,
          repeat=repeat,
          env_id=terrain_index * envs_per_height + slot,
        ))
  return rows


class DiagnoseRollBoundaryTest(unittest.TestCase):
  def test_schedule_grid_is_frozen_to_twelve_dynamic_and_two_sentinels(self):
    candidates = diag.schedule_grid_candidates()
    self.assertEqual(diag.SCHEDULE_DIAGNOSTIC_SCHEMA_VERSION, 3)
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
    diag._validate_schedule_candidate_set(candidates)
    with self.assertRaisesRegex(ValueError, 'exactly 14'):
      diag._validate_schedule_candidate_set(candidates[:-1])

  def test_r0c_sync_screen_is_frozen_to_legacy_and_shared_alpha_modes(self):
    candidates = diag.r0c_sync_candidates()
    self.assertEqual(diag.R0C_SYNC_SCHEMA_VERSION, 1)
    self.assertEqual(
      [candidate['slew_mode'] for candidate in candidates],
      [diag.INDEPENDENT_SLEW_MODE, diag.SYNCHRONIZED_SLEW_MODE],
    )
    self.assertEqual(
      {candidate['schedule'].name for candidate in candidates},
      {diag.R0C_SYNC_BASE_SCHEDULE_NAME},
    )
    diag._validate_r0c_sync_candidate_set(candidates)
    with self.assertRaisesRegex(ValueError, 'exactly two'):
      diag._validate_r0c_sync_candidate_set(candidates[:1])

  def test_r0c_sync_cli_freezes_the_8_by_1_rejection_protocol(self):
    args = diag.parse_args([
      '--mode', 'r0c-sync', '--output', 'outside.json', '--device', 'cuda:0',
    ])
    self.assertEqual(args.envs_per_height, 8)
    self.assertEqual(args.repeats, 1)
    self.assertEqual(args.settle_steps, 100)
    self.assertEqual(args.drive_steps, 500)
    self.assertEqual(args.stable_steps, 25)
    with self.assertRaises(SystemExit):
      diag.parse_args([
        '--mode', 'r0c-sync', '--output', 'outside.json', '--device', 'cuda:0',
        '--envs-per-height', '16',
      ])

  def test_r0c_sync_reset_claim_is_derived_from_complete_trial_metadata(self):
    candidates = diag.r0c_sync_candidates()
    results = [
      {'trials': _schedule_trials(candidate)} for candidate in candidates
    ]
    self.assertTrue(diag._validate_matched_r0c_resets(results))
    results[1]['trials'][0]['root_reset'] = {'x_relative_to_face_m': 1.0}
    with self.assertRaisesRegex(ValueError, 'reset mismatch'):
      diag._validate_matched_r0c_resets(results)

  def test_r0c_sync_trial_validator_enforces_shared_alpha_contract(self):
    candidate = diag.r0c_sync_candidates()[1]
    rows = _schedule_trials(candidate)
    diag._validate_schedule_candidate_trials(
      candidate,
      rows,
      expected_repeats=1,
      expected_envs_per_height=1,
      drive_steps=1,
      require_control_trace=True,
    )
    rows[0]['maximum_applied_channel_alpha_gap'] = 1.0e-4
    with self.assertRaisesRegex(ValueError, 'share applied alpha'):
      diag._validate_schedule_candidate_trials(
        candidate,
        rows,
        expected_repeats=1,
        expected_envs_per_height=1,
        drive_steps=1,
      )
    jump_rows = _schedule_trials(candidate)
    sample = jump_rows[0]['control_trace'][0]
    sample.update({
      'schedule_nominal_alpha': 1.0,
      'schedule_applied_alpha': 1.0,
      'schedule_applied_height_alpha': 1.0,
      'schedule_applied_pitch_alpha': 1.0,
      'applied_height_m': candidate['schedule'].climb_height_m,
      'applied_pitch_rad': candidate['schedule'].climb_pitch_rad,
    })
    with self.assertRaisesRegex(ValueError, 'declared slew controller'):
      diag._validate_schedule_candidate_trials(
        candidate,
        jump_rows,
        expected_repeats=1,
        expected_envs_per_height=1,
        drive_steps=1,
        require_control_trace=True,
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
    events = collector.finalize()
    self.assertEqual(
      [item['substep'] for item in events[0]['samples']],
      [1, 2, 3, 4, 5],
    )

  def test_event_window_finalize_rejects_an_incomplete_post_window(self):
    collector = diag.EventWindowCollector(1, pre_substeps=2, post_substeps=2)
    for step in range(1, 5):
      collector.observe(
        0, {'substep': step}, active=True, unsupported=(step == 3),
      )
    with self.assertRaisesRegex(ValueError, 'before post samples completed'):
      collector.finalize()

  def test_schedule_events_include_candidate_and_repeat_identity(self):
    events = diag._identified_schedule_events(
      [{'env_id': 3, 'samples': []}], candidate='candidate-a', repeat=2,
    )
    self.assertEqual(events, [{
      'env_id': 3,
      'samples': [],
      'candidate': 'candidate-a',
      'repeat': 2,
    }])

  def test_schedule_event_validator_binds_each_raw_loss_to_a_complete_window(self):
    candidate = diag.schedule_grid_candidates()[0]
    rows = _schedule_trials(candidate)
    for row in rows:
      row.pop('control_trace')
    rows[0]['bilateral_unsupported_physics_substeps'] = 1
    rows[0]['bilateral_airborne_ever'] = True
    rows[0]['first_support_loss_progress_m'] = -0.1
    samples = []
    trigger_substep = 409
    for substep in range(401, 422):
      episode_step = (substep - 1) // 4 + 1
      trigger = substep == trigger_substep
      samples.append({
        'substep': substep,
        'episode_control_step': episode_step,
        'drive_control_step': episode_step - 100,
        'phase': 'drive',
        'terrain_index': 0.0,
        'left_force_n': 0.0 if trigger else 100.0,
        'right_force_n': 0.0 if trigger else 100.0,
        'left_vertical_normal_load_n': 0.0 if trigger else 100.0,
        'right_vertical_normal_load_n': 0.0 if trigger else 100.0,
        'total_vertical_normal_load_n': 0.0 if trigger else 200.0,
        'schedule_nominal_alpha': 0.1,
        'schedule_applied_alpha': 0.05,
        'schedule_applied_height_alpha': 0.05,
        'schedule_applied_pitch_alpha': 0.05,
      })
    event = {
      'candidate': candidate['name'],
      'repeat': 1,
      'env_id': 0,
      'trigger_substep': trigger_substep,
      'pre_substeps_requested': 8,
      'post_substeps_requested': 12,
      'samples': samples,
    }
    diag._validate_schedule_events(
      candidate, rows, [event],
      pre_substeps=8, post_substeps=12, settle_steps=100,
    )
    event['samples'] = samples[:-1]
    with self.assertRaisesRegex(ValueError, 'sample count is incomplete'):
      diag._validate_schedule_events(
        candidate, rows, [event],
        pre_substeps=8, post_substeps=12, settle_steps=100,
      )

  def test_summarize_trials_counts_force_events(self):
    rows = [_summary_row(height) for height in diag.DIAGNOSTIC_HEIGHTS_M]
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
        rows.append(_summary_row(
          height,
          applied_residual_abs_max=value,
          wheel_target_classical_path_abs_max_radps=0.0,
          dynamic_leg_feedforward_abs_max_rad=0.0,
          dynamic_drive_feedforward_abs_max_radps=0.0,
        ))
    summaries = diag.summarize_trials(rows)
    self.assertEqual(summaries[0]['applied_residual_abs_max'], 0.25)
    del rows[0]['applied_residual_abs_max']
    with self.assertRaisesRegex(ValueError, 'partially missing'):
      diag.summarize_trials(rows)

  def test_schedule_trial_schema_is_uniform_for_dynamic_and_static_candidates(self):
    dynamic = diag.schedule_grid_candidates()[0]
    static = diag.schedule_grid_candidates()[-1]
    dynamic_rows = _schedule_trials(dynamic)
    dynamic_keys = diag._validate_schedule_candidate_trials(
      dynamic,
      dynamic_rows,
      expected_repeats=1,
      expected_envs_per_height=1,
      drive_steps=2,
    )
    static_rows = _schedule_trials(static)
    static_keys = diag._validate_schedule_candidate_trials(
      static,
      static_rows,
      expected_repeats=1,
      expected_envs_per_height=1,
      drive_steps=2,
      expected_keys=dynamic_keys,
    )
    self.assertEqual(dynamic_keys, static_keys)

  def test_schedule_trial_validator_rejects_missing_authority_and_wheel_metrics(self):
    candidate = diag.schedule_grid_candidates()[0]
    cases = []
    all_authority_missing = _schedule_trials(candidate)
    for row in all_authority_missing:
      for field in diag.SCHEDULE_AUTHORITY_METRICS:
        del row[field]
    cases.append(('required trial fields', all_authority_missing))
    missing_wheel_residual = _schedule_trials(candidate)
    del missing_wheel_residual[0]['wheel_residual_abs_max']
    cases.append(('inconsistent trial keys', missing_wheel_residual))
    nonzero_authority = _schedule_trials(candidate)
    nonzero_authority[0]['applied_residual_abs_max'] = 1e-12
    cases.append(('must be exactly zero', nonzero_authority))
    for message, rows in cases:
      with (
        self.subTest(message=message),
        self.assertRaisesRegex(ValueError, message),
      ):
        diag._validate_schedule_candidate_trials(
          candidate,
          rows,
          expected_repeats=1,
          expected_envs_per_height=1,
          drive_steps=2,
        )

  def test_schedule_trial_validator_cross_checks_the_raw_5ms_support_gate(self):
    candidate = diag.schedule_grid_candidates()[0]
    rows = _schedule_trials(candidate)
    rows[0]['bilateral_unsupported_physics_substeps'] = 1
    with self.assertRaisesRegex(ValueError, 'boolean/count disagree'):
      diag._validate_schedule_candidate_trials(
        candidate,
        rows,
        expected_repeats=1,
        expected_envs_per_height=1,
        drive_steps=2,
      )

  def test_schedule_trial_validator_rejects_invalid_ranking_and_counts(self):
    candidate = diag.schedule_grid_candidates()[0]
    for value in (None, math.nan):
      rows = _schedule_trials(candidate)
      rows[0]['torque_saturation_fraction'] = value
      with (
        self.subTest(torque=value),
        self.assertRaisesRegex(ValueError, 'torque_saturation_fraction'),
      ):
        diag._validate_schedule_candidate_trials(
          candidate,
          rows,
          expected_repeats=1,
          expected_envs_per_height=1,
          drive_steps=2,
        )
    rows = _schedule_trials(candidate)
    with self.assertRaisesRegex(ValueError, 'produced 1 trials; expected 2'):
      diag._validate_schedule_candidate_trials(
        candidate,
        rows[:-1],
        expected_repeats=1,
        expected_envs_per_height=1,
        drive_steps=2,
      )

  def test_r0c_sync_verdict_requires_flat_retention_and_baseline_reproduction(self):
    def summary(height, successes, unsafe, stalls):
      return {
        'stair_height_m': height,
        'trials': 8,
        'successes': successes,
        'unsafe_trials': unsafe,
        'safe_stalls': stalls,
        'bilateral_airborne_trials': unsafe,
        'bilateral_unsupported_physics_substeps': unsafe,
        'non_wheel_contact_trials': 0,
        'terminated_trials': 0,
      }

    results = [
      {
        'candidate_definition': {'kind': 'legacy_independent_slew_baseline'},
        'summaries': [summary(0.0, 8, 0, 0), summary(0.0025, 3, 5, 0)],
      },
      {
        'candidate_definition': {'kind': 'shared_alpha_synchronized_slew'},
        'summaries': [summary(0.0, 8, 0, 0), summary(0.0025, 7, 0, 1)],
      },
    ]
    verdict = diag.classify_r0c_sync_screen(results)
    self.assertEqual(
      verdict['decision'], 'SYNC_SCREEN_PASS_FORMAL_REPLICATION_REQUIRED',
    )
    self.assertFalse(verdict['promotion_eligible'])
    results[1]['summaries'][1]['bilateral_unsupported_physics_substeps'] = 1
    verdict = diag.classify_r0c_sync_screen(results)
    self.assertEqual(verdict['decision'], 'SYNC_REJECTED_ADVANCE_TO_R0C_LRG')
    results[1]['summaries'][1] = summary(0.0025, 6, 1, 1)
    verdict = diag.classify_r0c_sync_screen(results)
    self.assertEqual(verdict['decision'], 'SYNC_REJECTED_ADVANCE_TO_R0C_LRG')
    results[0]['summaries'][1] = summary(0.0025, 5, 3, 0)
    verdict = diag.classify_r0c_sync_screen(results)
    self.assertEqual(verdict['decision'], 'INVALID_BASELINE_NOT_REPRODUCED')

  def test_output_must_be_outside_project_and_mjlab_checkouts(self):
    repositories = (
      diag.rb.REPOSITORY_PATH,
      Path(diag.mjlab.__file__).resolve().parents[2],
    )
    for repository in repositories:
      with (
        self.subTest(repository=repository),
        self.assertRaisesRegex(ValueError, 'outside the Git checkout'),
      ):
        diag._outside_repository(repository / 'diagnostic.json')

  def test_existing_output_fails_before_schedule_execution(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'schedule.json'
      output.write_text('{}', encoding='utf-8')
      with (
        patch.object(diag, 'run_schedule_grid') as run,
        self.assertRaises(FileExistsError),
      ):
        diag.main(['--mode', 'schedule-grid', '--output', str(output)])
      run.assert_not_called()

  def test_dirty_flag_is_rejected_for_cuda_schedule(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'schedule.json'
      argv = [
        '--mode', 'schedule-grid', '--output', str(output),
        '--device', 'cuda:0', '--allow-dirty',
      ]
      with (
        patch.object(diag, '_git_dirty', return_value=False),
        patch.object(diag, '_diagnostic_provenance') as capture,
        patch.object(diag, 'run_schedule_grid') as run,
        self.assertRaisesRegex(RuntimeError, 'cannot use --allow-dirty'),
      ):
        diag.main(argv)
      capture.assert_not_called()
      run.assert_not_called()
      self.assertFalse(output.exists())
      self.assertFalse(output.with_name(f'.{output.name}.reserved').exists())

  def test_provenance_drift_aborts_before_atomic_write(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'schedule.json'
      provenance = {'git_sha': 'a' * 40, 'device': 'cpu'}
      payload = {'kind': 'mock', **provenance}
      with (
        patch.object(diag, '_git_dirty', return_value=False),
        patch.object(diag, '_diagnostic_provenance', return_value=provenance),
        patch.object(diag, 'run_schedule_grid', return_value=payload) as run,
        patch.object(
          diag,
          '_verify_provenance_unchanged',
          side_effect=RuntimeError('Diagnostic provenance changed during execution.'),
        ),
        patch.object(diag.rb, '_atomic_write_json') as write,
        self.assertRaisesRegex(RuntimeError, 'provenance changed'),
      ):
        diag.main(['--mode', 'schedule-grid', '--output', str(output)])
      self.assertIs(run.call_args.kwargs['provenance'], provenance)
      write.assert_not_called()
      self.assertFalse(output.exists())
      self.assertFalse(output.with_name(f'.{output.name}.reserved').exists())

  def test_provenance_verifier_rejects_source_drift(self):
    provenance = {
      'git_sha': 'a',
      'mjlab_git_sha': 'b',
      'project_dirty': True,
      'mjlab_dirty': False,
      'project_worktree_fingerprint': 'project-before',
      'mjlab_worktree_fingerprint': 'mjlab-before',
      'source_file_sha256': {'source': 'before'},
    }
    with (
      patch.object(diag.rb, '_git_sha', side_effect=['a', 'b']),
      patch.object(diag, '_git_dirty', side_effect=[True, False]),
      patch.object(
        diag,
        '_git_worktree_fingerprint',
        side_effect=['project-before', 'mjlab-before'],
      ),
      patch.object(diag, '_diagnostic_source_hashes', return_value={'source': 'after'}),
      self.assertRaisesRegex(RuntimeError, 'source_file_sha256'),
    ):
      diag._verify_provenance_unchanged(provenance)


if __name__ == '__main__':
  unittest.main()
