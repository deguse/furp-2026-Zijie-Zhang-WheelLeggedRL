"""Tests for the paired P2 stall causal capture."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from hoppertrex_mjlab.scripts import probe_hybrid_stair_height as stair
from hoppertrex_mjlab.scripts import probe_hybrid_stall_causal_v2 as causal


def _cell(name: str, *, valid_pairs: int = 16, flat_success: float = 1.0):
  return {
    "cell": name,
    "flat_success_rate": flat_success,
    "flat_terminated_trials": 0,
    "flat_timeout_trials": 0,
    "flat_non_wheel_contact_trials": 0,
    "valid_capture_pairs": valid_pairs,
  }


def _capture(name: str, slot: int, *, valid: bool = True):
  return {
    "cell": name,
    "terrain_slot": slot,
    "valid": valid,
  }


class CausalProtocolTest(unittest.TestCase):
  def test_official_protocol_is_frozen(self):
    protocol = causal.protocol_for_mode(False, "cuda:0")

    self.assertEqual(protocol["heights_m"], (0.0, 0.01))
    self.assertEqual(protocol["envs_per_height"], 16)
    self.assertEqual(protocol["settle_steps"], 200)
    self.assertEqual(protocol["drive_steps"], 500)
    self.assertEqual(protocol["pre_impact_steps"], 25)
    self.assertEqual(protocol["post_impact_steps"], 75)
    self.assertEqual(
      [cell["name"] for cell in protocol["command_cells"]],
      ["pitch_zero", "fast_lean_0p032"],
    )
    self.assertTrue(protocol["evidence_eligible"])

  def test_probe_sensor_does_not_modify_default_stair_config(self):
    default_cfg = stair.make_stair_env_cfg((0.0, 0.01), 1)
    causal_cfg = causal.make_causal_env_cfg((0.0, 0.01), 1)
    default_names = [sensor.name for sensor in default_cfg.scene.sensors]
    causal_names = [sensor.name for sensor in causal_cfg.scene.sensors]

    self.assertNotIn(causal.DIAGNOSTIC_SENSOR_NAME, default_names)
    self.assertEqual(causal_names[:-1], default_names)
    self.assertEqual(causal_names[-1], causal.DIAGNOSTIC_SENSOR_NAME)
    sensor = causal_cfg.scene.sensors[-1]
    self.assertEqual(sensor.reduce, "none")
    self.assertEqual(sensor.num_slots, 8)
    self.assertEqual(sensor.fields, causal.DIAGNOSTIC_SENSOR_FIELDS)


class PairAndImpactSelectorTest(unittest.TestCase):
  def test_pairs_by_within_terrain_slot(self):
    terrain_types = torch.tensor([0, 0, 1, 1])

    self.assertEqual(
      causal.paired_environment_ids(terrain_types),
      [
        {"slot": 0, "flat_env_id": 0, "stair_env_id": 2},
        {"slot": 1, "flat_env_id": 1, "stair_env_id": 3},
      ],
    )

  def test_pairs_reject_unbalanced_terrain_counts(self):
    with self.assertRaisesRegex(ValueError, "equal nonzero counts"):
      causal.paired_environment_ids(torch.tensor([0, 1, 1]))

  def test_riser_selector_uses_geometry_normal_and_force(self):
    found = torch.ones((2, 2), dtype=torch.bool)
    force = torch.zeros((2, 2, 3))
    force[..., 0] = torch.tensor([[20.0, 20.0], [0.5, 20.0]])
    pos = torch.zeros((2, 2, 3))
    pos[..., 0] = torch.tensor([[-3.0, -2.95], [-3.0, -3.0]])
    normal = torch.zeros((2, 2, 3))
    normal[..., 0] = torch.tensor([[0.44, 0.44], [0.44, 0.10]])

    mask = causal.riser_contact_mask(
      found=found,
      force_contact_frame=force,
      pos_global=pos,
      normal_global=normal,
      outer_face_x=torch.tensor([-3.0, -3.0]),
    )

    torch.testing.assert_close(
      mask,
      torch.tensor([[True, False], [False, False]]),
    )


class AlignedSeriesTest(unittest.TestCase):
  def test_same_time_stair_minus_flat_alignment(self):
    base = torch.tensor([
      [0.0, 10.0],
      [1.0, 12.0],
      [2.0, 14.0],
      [3.0, 16.0],
      [4.0, 18.0],
    ])
    samples = {field: base.clone() for field in causal.SERIES_FIELDS}

    series = causal.build_aligned_series(
      samples,
      flat_env_id=0,
      stair_env_id=1,
      impact_step=2,
      pre_steps=1,
      post_steps=2,
    )

    self.assertEqual(series["relative_steps"], [-1, 0, 1, 2])
    self.assertEqual(
      series["stair_minus_flat"]["body_vx_mps"],
      [11.0, 12.0, 13.0, 14.0],
    )
    summary = causal.summarize_aligned_series(series)
    self.assertEqual(
      summary["body_vx_mps"]["stair_minus_flat"]["impact"],
      12.0,
    )
    self.assertEqual(
      summary["body_vx_mps"]["stair_minus_flat"]
      ["impact_and_post_mean"],
      13.0,
    )

  def test_alignment_rejects_missing_history(self):
    samples = {
      field: torch.zeros((4, 2)) for field in causal.SERIES_FIELDS
    }
    with self.assertRaisesRegex(ValueError, "pre-impact"):
      causal.build_aligned_series(
        samples,
        flat_env_id=0,
        stair_env_id=1,
        impact_step=0,
        pre_steps=1,
        post_steps=1,
      )


class CaptureClassificationTest(unittest.TestCase):
  def test_aggregate_rejects_missing_cell_capture(self):
    trials = [
      {
        "cell": name,
        "stair_height_m": height,
        "success": height == 0.0,
        "terminated": False,
        "timeout": False,
        "non_wheel_contact": False,
      }
      for name in ("pitch_zero", "fast_lean_0p032")
      for height in (0.0, 0.01)
    ]
    captures = [_capture("pitch_zero", 0, valid=False)]

    with self.assertRaisesRegex(ValueError, "capture count"):
      causal.aggregate_cells(
        trials,
        captures,
        command_cells=causal.COMMAND_CELLS,
        expected_pairs=1,
      )

  def test_complete_capture_is_analysis_ready_without_cause_label(self):
    cells = [_cell(name) for name in ("pitch_zero", "fast_lean_0p032")]
    captures = [
      _capture(name, slot)
      for name in ("pitch_zero", "fast_lean_0p032")
      for slot in range(16)
    ]

    verdict = causal.classify_capture(
      cells,
      captures,
      expected_cells=2,
      expected_pairs_per_cell=16,
    )

    self.assertEqual(verdict["classification"], "ANALYSIS_READY")
    self.assertIsNone(verdict["single_cause_label"])
    self.assertEqual(verdict["invalid_reasons"], [])

  def test_invalid_pair_or_flat_control_invalidates_capture(self):
    cells = [
      _cell("pitch_zero", flat_success=0.5),
      _cell("fast_lean_0p032"),
    ]
    captures = [
      _capture(name, slot, valid=not (name == "pitch_zero" and slot == 3))
      for name in ("pitch_zero", "fast_lean_0p032")
      for slot in range(16)
    ]

    verdict = causal.classify_capture(
      cells,
      captures,
      expected_cells=2,
      expected_pairs_per_cell=16,
    )

    self.assertEqual(verdict["classification"], "INVALID_CAPTURE")
    self.assertIn("invalid_aligned_pairs", verdict["invalid_reasons"])
    self.assertIn("invalid_flat_control", verdict["invalid_reasons"])
    self.assertEqual(verdict["invalid_pairs"], ["pitch_zero:slot3"])


class CausalCaptureSmokeTest(unittest.TestCase):
  def test_cpu_smoke_exercises_sensor_and_schema(self):
    with tempfile.TemporaryDirectory() as tmp:
      output = Path(tmp) / "causal_v2_smoke.json"
      causal.main([
        "--device", "cpu", "--smoke", "--output", str(output)
      ])
      payload = json.loads(output.read_text(encoding="utf-8"))

    self.assertEqual(payload["probe"], "hybrid_p2_stall_causal_capture_v2")
    self.assertFalse(payload["evidence_eligible"])
    self.assertFalse(payload["promotion_eligible"])
    self.assertFalse(payload["training_eligible"])
    self.assertIsNone(payload["classification"])
    self.assertIsNone(payload["single_cause_label"])
    self.assertIsNone(payload["checkpoint"])
    self.assertIsNone(payload["checkpoint_file_sha256"])
    self.assertEqual(len(payload["trials"]), 2)
    self.assertEqual(len(payload["paired_captures"]), 1)
    self.assertEqual(
      payload["protocol"]["contact_sensor"]["name"],
      causal.DIAGNOSTIC_SENSOR_NAME,
    )
    self.assertEqual(payload["protocol"]["policy_action"], [0.0] * 6)
    self.assertEqual(payload["protocol"]["commanded_yaw_rate"], 0.0)


if __name__ == "__main__":
  unittest.main()
