import math
import unittest

import torch

from hoppertrex_mjlab.hybrid import roll_pose_schedule as schedule


class RollPoseScheduleTest(unittest.TestCase):
  def setUp(self) -> None:
    self.schedule = schedule.roll_pose_schedule_candidates()[0]

  def test_frozen_grid_has_twelve_exact_ordered_candidates(self):
    observed = tuple(
      (
        candidate.name,
        candidate.start_height_m,
        candidate.start_pitch_rad,
        candidate.climb_height_m,
        candidate.climb_pitch_rad,
        candidate.end_distance_to_riser_m,
      )
      for candidate in schedule.roll_pose_schedule_candidates()
    )
    expected = (
      ("roll_pose_sa_cc_d030mm", 0.2907321708, -0.032, 0.3092089487, 0.032, 0.030),
      ("roll_pose_sa_cc_d015mm", 0.2907321708, -0.032, 0.3092089487, 0.032, 0.015),
      ("roll_pose_sa_cc_d000mm", 0.2907321708, -0.032, 0.3092089487, 0.032, 0.000),
      ("roll_pose_sa_cd_d030mm", 0.2907321708, -0.032, 0.3276857266, 0.032, 0.030),
      ("roll_pose_sa_cd_d015mm", 0.2907321708, -0.032, 0.3276857266, 0.032, 0.015),
      ("roll_pose_sa_cd_d000mm", 0.2907321708, -0.032, 0.3276857266, 0.032, 0.000),
      ("roll_pose_sb_cc_d030mm", 0.2907321708, -0.016, 0.3092089487, 0.032, 0.030),
      ("roll_pose_sb_cc_d015mm", 0.2907321708, -0.016, 0.3092089487, 0.032, 0.015),
      ("roll_pose_sb_cc_d000mm", 0.2907321708, -0.016, 0.3092089487, 0.032, 0.000),
      ("roll_pose_sb_cd_d030mm", 0.2907321708, -0.016, 0.3276857266, 0.032, 0.030),
      ("roll_pose_sb_cd_d015mm", 0.2907321708, -0.016, 0.3276857266, 0.032, 0.015),
      ("roll_pose_sb_cd_d000mm", 0.2907321708, -0.016, 0.3276857266, 0.032, 0.000),
    )
    self.assertEqual(observed, expected)

  def test_settle_and_staggered_drive_start_latch_each_environment(self):
    root_x = torch.tensor([-0.25, -0.23])
    face_x = torch.zeros(2)
    state = schedule.make_roll_pose_schedule_state(self.schedule, root_x)
    settled = schedule.roll_pose_schedule_step(
      self.schedule, state, root_x_m=root_x, face_x_m=face_x,
      active_mask=torch.ones(2, dtype=torch.bool), drive_active=False,
    )
    self.assertEqual(settled.alpha.tolist(), [0.0, 0.0])
    self.assertTrue(torch.allclose(
      settled.applied_height_m,
      torch.full((2,), self.schedule.start_height_m),
    ))

    first = schedule.roll_pose_schedule_step(
      self.schedule, state, root_x_m=root_x, face_x_m=face_x,
      active_mask=torch.tensor([True, False]), drive_active=True,
    )
    self.assertEqual(state.drive_started.tolist(), [True, False])
    self.assertAlmostEqual(float(state.drive_start_x_m[0]), -0.25)
    self.assertAlmostEqual(float(state.drive_start_x_m[1]), -0.23)
    self.assertAlmostEqual(float(state.required_transition_progress_m[0]), 0.22)
    self.assertEqual(float(state.required_transition_progress_m[1]), 0.0)
    inactive_height = float(first.applied_height_m[1])

    second_root = torch.tensor([-0.20, -0.10])
    schedule.roll_pose_schedule_step(
      self.schedule, state, root_x_m=second_root, face_x_m=face_x,
      active_mask=torch.tensor([False, True]), drive_active=True,
    )
    self.assertEqual(state.drive_started.tolist(), [True, True])
    self.assertAlmostEqual(float(state.drive_start_x_m[0]), -0.25)
    self.assertAlmostEqual(float(state.drive_start_x_m[1]), -0.10)
    self.assertAlmostEqual(float(state.required_transition_progress_m[1]), 0.07)
    self.assertEqual(float(state.max_forward_progress_m[0]), 0.0)
    self.assertAlmostEqual(float(state.applied_height_m[1]), inactive_height)

  def test_progress_and_transition_distance_remain_latched_without_rewind(self):
    root_x = torch.tensor([-0.25])
    face_x = torch.zeros(1)
    active = torch.ones(1, dtype=torch.bool)
    state = schedule.make_roll_pose_schedule_state(self.schedule, root_x)
    schedule.roll_pose_schedule_step(
      self.schedule, state, root_x_m=root_x, face_x_m=face_x,
      active_mask=active, drive_active=True,
    )
    forward = schedule.roll_pose_schedule_step(
      self.schedule, state, root_x_m=torch.tensor([-0.15]), face_x_m=face_x,
      active_mask=active, drive_active=True,
    )
    expected_alpha = 0.10 / 0.22
    self.assertAlmostEqual(float(forward.alpha[0]), expected_alpha, places=6)

    backward_with_changed_face = schedule.roll_pose_schedule_step(
      self.schedule, state, root_x_m=torch.tensor([-0.18]),
      face_x_m=torch.tensor([0.10]), active_mask=active, drive_active=True,
    )
    self.assertAlmostEqual(float(state.max_forward_progress_m[0]), 0.10, places=6)
    self.assertAlmostEqual(
      float(backward_with_changed_face.alpha[0]), expected_alpha, places=6,
    )
    self.assertAlmostEqual(float(state.required_transition_progress_m[0]), 0.22)

  def test_alpha_formula_and_slew_move_exactly_toward_target_without_overshoot(self):
    root_x = torch.tensor([-0.25])
    face_x = torch.zeros(1)
    active = torch.ones(1, dtype=torch.bool)
    state = schedule.make_roll_pose_schedule_state(self.schedule, root_x)
    schedule.roll_pose_schedule_step(
      self.schedule, state, root_x_m=root_x, face_x_m=face_x,
      active_mask=active, drive_active=True,
    )
    before_height = state.applied_height_m.clone()
    before_pitch = state.applied_pitch_rad.clone()
    midpoint = schedule.roll_pose_schedule_step(
      self.schedule, state, root_x_m=torch.tensor([-0.14]), face_x_m=face_x,
      active_mask=active, drive_active=True,
    )
    self.assertAlmostEqual(float(midpoint.alpha[0]), 0.5, places=6)
    self.assertAlmostEqual(
      float(midpoint.desired_height_m[0]),
      0.5 * (self.schedule.start_height_m + self.schedule.climb_height_m),
      places=6,
    )
    self.assertAlmostEqual(
      float(midpoint.desired_pitch_rad[0]),
      0.5 * (self.schedule.start_pitch_rad + self.schedule.climb_pitch_rad),
      places=6,
    )
    self.assertAlmostEqual(
      float(midpoint.applied_height_m[0] - before_height[0]),
      schedule.POSTURE_HEIGHT_SLEW_RATE_MPS * schedule.CONTROL_DT_S,
      places=7,
    )
    self.assertAlmostEqual(
      float(midpoint.applied_pitch_rad[0] - before_pitch[0]),
      schedule.POSTURE_PITCH_SLEW_RATE_RADPS * schedule.CONTROL_DT_S,
      places=7,
    )

    terminal_root = torch.tensor([0.10])
    for _ in range(200):
      terminal = schedule.roll_pose_schedule_step(
        self.schedule, state, root_x_m=terminal_root, face_x_m=face_x,
        active_mask=active, drive_active=True,
      )
    self.assertEqual(float(terminal.alpha[0]), 1.0)
    self.assertAlmostEqual(
      float(terminal.applied_height_m[0]), self.schedule.climb_height_m, places=7,
    )
    self.assertAlmostEqual(
      float(terminal.applied_pitch_rad[0]), self.schedule.climb_pitch_rad, places=7,
    )

  def test_all_candidates_converge_exactly_to_dtype_endpoints(self):
    for candidate in schedule.roll_pose_schedule_candidates():
      for dtype in (torch.float32, torch.float64):
        with self.subTest(candidate=candidate.name, dtype=dtype):
          root_x = torch.tensor([-0.25], dtype=dtype)
          face_x = torch.zeros(1, dtype=dtype)
          active = torch.ones(1, dtype=torch.bool)
          state = schedule.make_roll_pose_schedule_state(candidate, root_x)
          schedule.roll_pose_schedule_step(
            candidate, state, root_x_m=root_x, face_x_m=face_x,
            active_mask=active, drive_active=True,
          )
          terminal_root = torch.tensor([0.10], dtype=dtype)
          for _ in range(220):
            output = schedule.roll_pose_schedule_step(
              candidate, state, root_x_m=terminal_root, face_x_m=face_x,
              active_mask=active, drive_active=True,
            )
          expected_height = torch.full_like(root_x, candidate.climb_height_m)
          expected_pitch = torch.full_like(root_x, candidate.climb_pitch_rad)
          self.assertTrue(torch.equal(output.desired_height_m, expected_height))
          self.assertTrue(torch.equal(output.desired_pitch_rad, expected_pitch))
          self.assertTrue(torch.equal(output.applied_height_m, expected_height))
          self.assertTrue(torch.equal(output.applied_pitch_rad, expected_pitch))
          repeated = schedule.roll_pose_schedule_step(
            candidate, state, root_x_m=terminal_root, face_x_m=face_x,
            active_mask=active, drive_active=True,
          )
          self.assertTrue(torch.equal(repeated.applied_height_m, expected_height))
          self.assertTrue(torch.equal(repeated.applied_pitch_rad, expected_pitch))

  def test_synchronized_mode_uses_one_rate_limited_applied_alpha(self):
    candidate = next(
      item for item in schedule.roll_pose_schedule_candidates()
      if item.name == "roll_pose_sa_cd_d030mm"
    )
    root_x = torch.tensor([-0.25])
    face_x = torch.zeros(1)
    active = torch.ones(1, dtype=torch.bool)
    independent = schedule.make_roll_pose_schedule_state(candidate, root_x)
    synchronized = schedule.make_roll_pose_schedule_state(
      candidate, root_x, slew_mode=schedule.SYNCHRONIZED_SLEW_MODE,
    )
    for state in (independent, synchronized):
      schedule.roll_pose_schedule_step(
        candidate, state, root_x_m=root_x, face_x_m=face_x,
        active_mask=active, drive_active=True,
      )
    terminal_root = torch.tensor([0.10])
    independent_output = schedule.roll_pose_schedule_step(
      candidate, independent, root_x_m=terminal_root, face_x_m=face_x,
      active_mask=active, drive_active=True,
    )
    synchronized_output = schedule.roll_pose_schedule_step(
      candidate, synchronized, root_x_m=terminal_root, face_x_m=face_x,
      active_mask=active, drive_active=True,
    )
    expected_step = min(
      schedule.POSTURE_HEIGHT_SLEW_RATE_MPS
      / (candidate.climb_height_m - candidate.start_height_m),
      schedule.POSTURE_PITCH_SLEW_RATE_RADPS
      / (candidate.climb_pitch_rad - candidate.start_pitch_rad),
    ) * schedule.CONTROL_DT_S
    self.assertAlmostEqual(
      float(synchronized_output.applied_alpha[0]), expected_step, places=7,
    )
    self.assertAlmostEqual(
      float(synchronized_output.applied_height_alpha[0]), expected_step, places=6,
    )
    self.assertAlmostEqual(
      float(synchronized_output.applied_pitch_alpha[0]), expected_step, places=6,
    )
    self.assertGreater(
      float(independent_output.applied_pitch_alpha[0]),
      float(independent_output.applied_height_alpha[0]),
    )

  def test_synchronized_channels_complete_together_without_overshoot(self):
    candidate = next(
      item for item in schedule.roll_pose_schedule_candidates()
      if item.name == "roll_pose_sa_cd_d030mm"
    )
    root_x = torch.tensor([-0.25], dtype=torch.float64)
    face_x = torch.zeros(1, dtype=torch.float64)
    active = torch.ones(1, dtype=torch.bool)
    state = schedule.make_roll_pose_schedule_state(
      candidate, root_x, slew_mode=schedule.SYNCHRONIZED_SLEW_MODE,
    )
    schedule.roll_pose_schedule_step(
      candidate, state, root_x_m=root_x, face_x_m=face_x,
      active_mask=active, drive_active=True,
    )
    terminal_root = torch.tensor([0.10], dtype=torch.float64)
    completion = None
    for step in range(1, 220):
      output = schedule.roll_pose_schedule_step(
        candidate, state, root_x_m=terminal_root, face_x_m=face_x,
        active_mask=active, drive_active=True,
      )
      height_done = bool(output.applied_height_alpha[0] >= 1.0)
      pitch_done = bool(output.applied_pitch_alpha[0] >= 1.0)
      self.assertEqual(height_done, pitch_done)
      self.assertAlmostEqual(
        float(output.applied_height_alpha[0]),
        float(output.applied_pitch_alpha[0]),
        places=12,
      )
      if height_done:
        completion = step
        break
    self.assertIsNotNone(completion)
    self.assertEqual(float(output.applied_alpha[0]), 1.0)
    self.assertEqual(float(output.applied_height_m[0]), candidate.climb_height_m)
    self.assertEqual(float(output.applied_pitch_rad[0]), candidate.climb_pitch_rad)

  def test_historical_dataclass_constructor_order_remains_compatible(self):
    numeric = torch.zeros(1)
    state = schedule.RollPoseScheduleState(
      torch.zeros(1, dtype=torch.bool),
      numeric.clone(), numeric.clone(), numeric.clone(),
      torch.full((1,), self.schedule.start_height_m),
      torch.full((1,), self.schedule.start_pitch_rad),
    )
    self.assertEqual(state.slew_mode, schedule.INDEPENDENT_SLEW_MODE)
    self.assertTrue(torch.equal(state.applied_alpha, numeric))
    output = schedule.RollPoseScheduleOutput(
      numeric, numeric, numeric, numeric, numeric, numeric,
    )
    self.assertIsNone(output.applied_alpha)

  def test_invalid_slew_mode_is_rejected_before_state_creation(self):
    with self.assertRaisesRegex(ValueError, "slew_mode"):
      schedule.make_roll_pose_schedule_state(
        self.schedule, torch.tensor([-0.25]), slew_mode="unknown",
      )

  def test_invalid_numeric_contract_fails_before_mutating_state(self):
    with self.assertRaisesRegex(TypeError, "float32 or float64"):
      schedule.make_roll_pose_schedule_state(
        self.schedule, torch.tensor([-1], dtype=torch.int64),
      )
    with self.assertRaisesRegex(ValueError, "nonempty"):
      schedule.make_roll_pose_schedule_state(self.schedule, torch.empty(0))
    with self.assertRaisesRegex(ValueError, "finite"):
      schedule.make_roll_pose_schedule_state(self.schedule, torch.tensor([math.nan]))

    root_x = torch.tensor([-0.25])
    state = schedule.make_roll_pose_schedule_state(self.schedule, root_x)
    initial = (
      state.drive_started.clone(),
      state.drive_start_x_m.clone(),
      state.required_transition_progress_m.clone(),
    )
    with self.assertRaisesRegex(TypeError, "share dtype"):
      schedule.roll_pose_schedule_step(
        self.schedule, state, root_x_m=root_x,
        face_x_m=torch.zeros(1, dtype=torch.float64),
        active_mask=torch.ones(1, dtype=torch.bool), drive_active=True,
      )
    self.assertTrue(torch.equal(state.drive_started, initial[0]))
    self.assertTrue(torch.equal(state.drive_start_x_m, initial[1]))
    self.assertTrue(torch.equal(state.required_transition_progress_m, initial[2]))

    with self.assertRaisesRegex(ValueError, "finite"):
      schedule.roll_pose_schedule_step(
        self.schedule, state, root_x_m=root_x, face_x_m=torch.tensor([math.inf]),
        active_mask=torch.ones(1, dtype=torch.bool), drive_active=True,
      )
    with self.assertRaisesRegex(TypeError, "active_mask must be a tensor"):
      schedule.roll_pose_schedule_step(
        self.schedule, state, root_x_m=root_x, face_x_m=torch.zeros(1),
        active_mask=[True], drive_active=True,
      )
    with self.assertRaisesRegex(TypeError, "drive_active"):
      schedule.roll_pose_schedule_step(
        self.schedule, state, root_x_m=root_x, face_x_m=torch.zeros(1),
        active_mask=torch.ones(1, dtype=torch.bool), drive_active=1,
      )
    with self.assertRaisesRegex(ValueError, "finite and positive"):
      schedule.roll_pose_schedule_step(
        self.schedule, state, root_x_m=root_x, face_x_m=torch.zeros(1),
        active_mask=torch.ones(1, dtype=torch.bool), drive_active=True, dt=math.nan,
      )

    state.drive_start_x_m = state.drive_start_x_m.double()
    with self.assertRaisesRegex(TypeError, "dtype drifted"):
      schedule.roll_pose_schedule_step(
        self.schedule, state, root_x_m=root_x, face_x_m=torch.zeros(1),
        active_mask=torch.ones(1, dtype=torch.bool), drive_active=True,
      )

  @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
  def test_mixed_device_is_rejected_before_state_mutation(self):
    root_x = torch.tensor([-0.25], device="cuda")
    state = schedule.make_roll_pose_schedule_state(self.schedule, root_x)
    with self.assertRaisesRegex(ValueError, "share device"):
      schedule.roll_pose_schedule_step(
        self.schedule, state, root_x_m=root_x, face_x_m=torch.zeros(1),
        active_mask=torch.ones(1, dtype=torch.bool, device="cuda"),
        drive_active=True,
      )
    self.assertFalse(bool(state.drive_started[0]))

  def test_schedule_rejects_values_outside_registered_envelope(self):
    with self.assertRaisesRegex(ValueError, "registered envelope"):
      schedule.RollPoseSchedule(
        name="bad", start_height_m=0.28, start_pitch_rad=0.0,
        climb_height_m=0.31, climb_pitch_rad=0.0,
        end_distance_to_riser_m=0.015,
      )


if __name__ == "__main__":
  unittest.main()
