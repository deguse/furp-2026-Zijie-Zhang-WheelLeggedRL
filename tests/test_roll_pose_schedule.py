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

  def test_settle_holds_start_pose_and_drive_latches_each_environment(self):
    root_x = torch.tensor([-0.25, -0.23])
    face_x = torch.zeros(2)
    active = torch.ones(2, dtype=torch.bool)
    state = schedule.make_roll_pose_schedule_state(self.schedule, root_x)
    output = schedule.roll_pose_schedule_step(
      self.schedule, state, root_x_m=root_x, face_x_m=face_x,
      active_mask=active, drive_active=False,
    )
    self.assertEqual(output.alpha.tolist(), [0.0, 0.0])
    self.assertTrue(torch.allclose(
      output.applied_height_m,
      torch.full((2,), self.schedule.start_height_m),
    ))

    schedule.roll_pose_schedule_step(
      self.schedule, state, root_x_m=root_x, face_x_m=face_x,
      active_mask=active, drive_active=True,
    )
    self.assertEqual(state.drive_start_x_m.tolist(), root_x.tolist())
    self.assertEqual(state.drive_started.tolist(), [True, True])

  def test_progress_is_monotone_when_robot_rolls_back(self):
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
    backward = schedule.roll_pose_schedule_step(
      self.schedule, state, root_x_m=torch.tensor([-0.18]), face_x_m=face_x,
      active_mask=active, drive_active=True,
    )
    self.assertAlmostEqual(float(state.max_forward_progress_m[0]), 0.10, places=6)
    self.assertAlmostEqual(float(backward.alpha[0]), float(forward.alpha[0]), places=6)

  def test_alpha_clamps_and_slew_limits_are_enforced(self):
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
    output = schedule.roll_pose_schedule_step(
      self.schedule, state, root_x_m=torch.tensor([0.10]), face_x_m=face_x,
      active_mask=active, drive_active=True,
    )
    self.assertEqual(float(output.alpha[0]), 1.0)
    self.assertLessEqual(
      float((output.applied_height_m - before_height).abs().max()),
      schedule.POSTURE_HEIGHT_SLEW_RATE_MPS * schedule.CONTROL_DT_S + 1e-8,
    )
    self.assertLessEqual(
      float((output.applied_pitch_rad - before_pitch).abs().max()),
      schedule.POSTURE_PITCH_SLEW_RATE_RADPS * schedule.CONTROL_DT_S + 1e-8,
    )

  def test_schedule_rejects_values_outside_registered_envelope(self):
    with self.assertRaisesRegex(ValueError, "registered envelope"):
      schedule.RollPoseSchedule(
        name="bad", start_height_m=0.28, start_pitch_rad=0.0,
        climb_height_m=0.31, climb_pitch_rad=0.0,
        end_distance_to_riser_m=0.015,
      )


if __name__ == "__main__":
  unittest.main()
