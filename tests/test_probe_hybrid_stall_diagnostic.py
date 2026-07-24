"""Tests for the P2 stall-mechanism diagnostic probe."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from hoppertrex_mjlab.assets.HopperTrex_CFG import (
  RMD_L_9025_35T_PEAK_TORQUE,
  WHEEL_VELOCITY_DAMPING,
)
from hoppertrex_mjlab.hybrid.identification import NOMINAL_WHEEL_RADIUS_M
from hoppertrex_mjlab.scripts import probe_hybrid_stall_diagnostic as stall


def _cell(
  name: str,
  *,
  stair_height_m: float,
  success_rate: float = 0.0,
  terminated_trials: int = 0,
  non_wheel_contact_trials: int = 0,
  saturation: float = 0.0,
  slip: float = 0.0,
  target: float = 0.7,
) -> dict:
  return {
    "cell": name,
    "pitch_rad": 0.0,
    "vx_mps": 0.07,
    "stair_height_m": stair_height_m,
    "success_rate": success_rate,
    "terminated_trials": terminated_trials,
    "non_wheel_contact_trials": non_wheel_contact_trials,
    "stall_window": {
      "torque_saturation_frac": saturation,
      "wheel_slip_mps_mean": slip,
      "wheel_target_radps_mean": target,
    },
  }


class StallTorqueModelTest(unittest.TestCase):
  def test_torque_model_matches_actuator_law_and_clips(self):
    target = torch.tensor([[0.7, 0.7], [0.7, 0.0]])
    actual = torch.tensor([[0.69, 0.0], [0.7, 0.0]])
    torque, saturated = stall.model_wheel_torque(target, actual)
    self.assertAlmostEqual(
      float(torque[0, 0]), WHEEL_VELOCITY_DAMPING * 0.01, places=4
    )
    self.assertAlmostEqual(
      float(torque[0, 1]), RMD_L_9025_35T_PEAK_TORQUE, places=6
    )
    self.assertTrue(bool(saturated[0, 1]))
    self.assertFalse(bool(saturated[0, 0]))
    self.assertAlmostEqual(float(torque[1, 0]), 0.0, places=9)
    negative, negative_saturated = stall.model_wheel_torque(
      torch.tensor([[0.0]]), torch.tensor([[2.0]])
    )
    self.assertAlmostEqual(
      float(negative[0, 0]), -RMD_L_9025_35T_PEAK_TORQUE, places=6
    )
    self.assertTrue(bool(negative_saturated[0, 0]))

  def test_signed_balance_channel_preserves_opposite_sign_forward_drive(self):
    wheel_values = torch.tensor([[-0.7, 0.7], [0.2, 0.6]])

    signed = stall.signed_balance_channel(wheel_values)

    torch.testing.assert_close(signed, torch.tensor([0.7, 0.2]))
    with self.assertRaisesRegex(ValueError, "shape"):
      stall.signed_balance_channel(torch.zeros((2, 3)))

  def test_command_cells_stay_inside_calibration_domain(self):
    for cell in stall.COMMAND_CELLS:
      self.assertLessEqual(
        abs(float(cell["vx_mps"])), stall.CALIBRATION_VX_LIMIT_MPS
      )
    flags = stall.cell_flags(
      {"name": "x", "pitch_rad": -0.032, "vx_mps": 0.10}
    )
    self.assertFalse(flags["pitch_in_qualified_envelope"])
    self.assertFalse(flags["vx_in_stage5_range"])
    self.assertTrue(flags["vx_in_calibration_domain"])
    baseline_flags = stall.cell_flags(
      {"name": "b", "pitch_rad": 0.016, "vx_mps": 0.07}
    )
    self.assertTrue(baseline_flags["pitch_in_qualified_envelope"])
    self.assertTrue(baseline_flags["vx_in_stage5_range"])


class StallClassificationTest(unittest.TestCase):
  def test_candidate_branch_fires_on_any_stair_success(self):
    cells = [
      _cell("pitch_zero", stair_height_m=0.01, saturation=1.0),
      _cell("lean_in_0p032", stair_height_m=0.01, success_rate=0.8),
      _cell("pitch_zero", stair_height_m=0.0, success_rate=1.0),
      _cell("lean_in_0p032", stair_height_m=0.0, success_rate=1.0),
    ]
    verdict = stall.classify_cells(cells)
    self.assertEqual(
      verdict["classification"], "CLASSICAL_CARD_CANDIDATE_FOUND"
    )
    self.assertEqual(verdict["best_cell"], "lean_in_0p032")

  def test_invalid_baseline_flat_control_stops_classification(self):
    cells = [
      _cell("pitch_zero", stair_height_m=0.01, saturation=1.0),
      _cell("pitch_zero", stair_height_m=0.0, success_rate=0.5),
    ]

    verdict = stall.classify_cells(cells)

    self.assertEqual(
      verdict["classification"], "INVALID_FLAT_CONTROL_STOP"
    )
    self.assertEqual(verdict["invalid_flat_cells"], ["pitch_zero"])

  def test_candidate_requires_its_paired_flat_control(self):
    cells = [
      _cell("pitch_zero", stair_height_m=0.01, saturation=1.0),
      _cell("lean_in_0p032", stair_height_m=0.01, success_rate=0.8),
      _cell("pitch_zero", stair_height_m=0.0, success_rate=1.0),
      _cell("lean_in_0p032", stair_height_m=0.0, success_rate=0.5),
    ]

    verdict = stall.classify_cells(cells)

    self.assertEqual(verdict["classification"], "TORQUE_SATURATED_STALL")
    self.assertEqual(verdict["invalid_flat_cells"], ["lean_in_0p032"])

  def test_saturated_stationary_stall_is_torque_saturated(self):
    cells = [
      _cell(
        "pitch_zero", stair_height_m=0.01, saturation=0.97, slip=0.005
      ),
      _cell("pitch_zero", stair_height_m=0.0, success_rate=1.0),
    ]
    self.assertEqual(
      stall.classify_cells(cells)["classification"],
      "TORQUE_SATURATED_STALL",
    )

  def test_saturated_spinning_stall_is_friction_limited(self):
    cells = [
      _cell(
        "pitch_zero", stair_height_m=0.01, saturation=0.95, slip=0.08
      ),
      _cell("pitch_zero", stair_height_m=0.0, success_rate=1.0),
    ]
    self.assertEqual(
      stall.classify_cells(cells)["classification"],
      "WHEEL_SPIN_FRICTION_LIMITED",
    )

  def test_collapsed_drive_target_is_balance_conflict(self):
    tracking = 0.07 / NOMINAL_WHEEL_RADIUS_M
    cells = [
      _cell(
        "pitch_zero",
        stair_height_m=0.01,
        saturation=0.2,
        target=0.3 * tracking,
      ),
      _cell("pitch_zero", stair_height_m=0.0, success_rate=1.0),
    ]
    self.assertEqual(
      stall.classify_cells(cells)["classification"],
      "DRIVE_TARGET_COLLAPSED",
    )

  def test_everything_nominal_is_mixed(self):
    cells = [
      _cell("pitch_zero", stair_height_m=0.01, saturation=0.2),
      _cell("pitch_zero", stair_height_m=0.0, success_rate=1.0),
    ]
    self.assertEqual(
      stall.classify_cells(cells)["classification"],
      "MIXED_STALL_MECHANISM",
    )


class StallDiagnosticSmokeTest(unittest.TestCase):
  def test_cpu_smoke_end_to_end(self):
    # No mocks: real terrain env, real command forcing, real stall-window
    # capture through the exact runtime path the machine room executes.
    with tempfile.TemporaryDirectory() as tmp:
      output = Path(tmp) / "stall_diagnostic_smoke.json"
      stall.main([
        "--device", "cpu",
        "--smoke",
        "--output", str(output),
      ])
      payload = json.loads(output.read_text(encoding="utf-8"))

    self.assertEqual(
      payload["probe"], "hybrid_p2_stall_mechanism_diagnostic"
    )
    self.assertFalse(payload["evidence_eligible"])
    self.assertFalse(payload["promotion_eligible"])
    self.assertFalse(payload["training_eligible"])
    self.assertIsNone(payload["checkpoint"])
    self.assertIsNone(payload["checkpoint_file_sha256"])
    self.assertIsNone(payload["classification"])
    expected_cells = len(stall.SMOKE_CELL_NAMES) * len(
      stall.DIAGNOSTIC_HEIGHTS_M
    )
    self.assertEqual(len(payload["cells"]), expected_cells)
    self.assertEqual(len(payload["trials"]), expected_cells)
    lean_cells = [
      cell
      for cell in payload["cells"]
      if not cell["pitch_in_qualified_envelope"]
    ]
    self.assertTrue(lean_cells)
    self.assertTrue(
      all(cell["pitch_rad"] < 0.0 for cell in lean_cells)
    )
    for cell in payload["cells"]:
      window = cell["stall_window"]
      for key in (
        "wheel_target_radps_mean",
        "model_torque_abs_nm_mean",
        "torque_saturation_frac",
        "wheel_slip_mps_mean",
        "pitch_error_rad_mean",
        "progress_rate_mps",
      ):
        self.assertIn(key, window)
        self.assertTrue(
          window[key] == window[key],  # not NaN
          f"{cell['cell']} {key} is NaN",
        )
    protocol = payload["protocol"]
    self.assertTrue(protocol["paired_resets_across_cells"])
    self.assertEqual(
      protocol["wheel_model"]["peak_torque_nm"],
      RMD_L_9025_35T_PEAK_TORQUE,
    )
    self.assertEqual(
      protocol["wheel_model"]["forward_channel"],
      "0.5 * (right - left)",
    )


if __name__ == "__main__":
  unittest.main()
