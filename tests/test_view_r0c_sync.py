import json
import tempfile
import unittest
from pathlib import Path

from hoppertrex_mjlab.scripts import diagnose_roll_boundary as diag
from hoppertrex_mjlab.scripts import view_r0c_sync as view


def _reset():
  return {
    "x_relative_to_face_m": -0.25,
    "y_relative_to_center_m": 0.0,
    "root_height_m": 0.29,
    "root_linear_velocity_mps": [0.0, 0.0, 0.0],
    "root_angular_velocity_radps": [0.0, 0.0, 0.0],
    "root_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    "leg_joint_position_rad": [0.1, -0.1, 0.2, -0.2],
    "leg_joint_velocity_radps": [0.0, 0.0, 0.0, 0.0],
  }


def _trial(env_id: int, *, unsupported: int, progress: float):
  return {
    "repeat": 1,
    "terrain_index": 1,
    "env_id": env_id,
    "stair_height_m": 0.0025,
    "root_reset": _reset(),
    "bilateral_unsupported_physics_substeps": unsupported,
    "max_progress_past_face_m": progress,
    "termination": False,
    "non_wheel_contact": False,
  }


def _payload():
  candidates = []
  for index, candidate in enumerate(diag.r0c_sync_candidates()):
    unsupported = 0 if index == 0 else 3
    candidates.append({
      "candidate_definition": diag._schedule_candidate_definition(candidate),
      "trials": [_trial(14, unsupported=unsupported, progress=-0.014)],
      "first_support_loss_events": (
        [] if unsupported == 0 else [{"repeat": 1, "env_id": 14}]
      ),
    })
  return {
    "kind": "r0c_synchronized_reference_rejection_screen",
    "git_sha": view.EXPECTED_RESULT_GIT_SHA,
    "project_dirty": False,
    "mjlab_dirty": False,
    "matched_reset_perturbations_across_candidates": True,
    "candidates": candidates,
  }


class R0cSyncViewerTests(unittest.TestCase):
  def test_load_view_case_selects_the_reviewed_candidate_and_reset(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "result.json"
      path.write_text(json.dumps(_payload()), encoding="utf-8")
      case = view.load_view_case(
        path, candidate_key="c1", env_id=14, enforce_file_hash=False,
      )
    self.assertEqual(case.candidate_key, "c1")
    self.assertEqual(case.env_id, 14)
    self.assertEqual(case.trial["root_reset"], _reset())
    self.assertIsNotNone(case.event)

  def test_load_view_case_rejects_incomplete_event_coverage(self):
    payload = _payload()
    payload["candidates"][1]["first_support_loss_events"] = []
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "result.json"
      path.write_text(json.dumps(payload), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "event coverage"):
        view.load_view_case(
          path, candidate_key="c1", env_id=14, enforce_file_hash=False,
        )

  def test_counterfactual_counts_are_explicitly_total_substep_allowances(self):
    losses = [0, 0, 1, 2, 2, 3, 4, 4]
    rows = [
      _trial(8 + index, unsupported=loss, progress=(0.2 if index != 5 else 0.1))
      for index, loss in enumerate(losses)
    ]
    self.assertEqual(
      view.counterfactual_crossing_counts({"trials": rows}),
      {0: 2, 1: 3, 2: 5, 4: 7},
    )

  def test_parse_args_pins_candidate_and_step_environment_choices(self):
    args = view.parse_args([
      "--result", "result.json", "--candidate", "c0", "--env-id", "15",
    ])
    self.assertEqual(args.candidate, "c0")
    self.assertEqual(args.env_id, 15)
    self.assertEqual(args.device, "cuda:0")
    with self.assertRaises(SystemExit):
      view.parse_args(["--result", "result.json", "--env-id", "7"])


if __name__ == "__main__":
  unittest.main()