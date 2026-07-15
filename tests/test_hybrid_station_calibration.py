import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from hoppertrex_mjlab.hybrid.station_calibration import (
  parse_station_calibration_artifact,
  station_calibration_artifact,
  station_drift,
  validate_station_breakpoints,
)
from hoppertrex_mjlab.scripts.fit_hybrid_station_calibration import (
  _validated_qualification,
  fit_station_breakpoints,
)
from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
  _torch_linear_interpolate,
)


# Measured on the 2026-07-15 uncompensated probe: drift is affine in the
# commanded pitch (~ -0.0136 - 0.751 * pitch) and independent of height.
PROBE_BREAKPOINTS = [
  [-0.08, 0.0465],
  [0.0, -0.0136],
  [0.0116, -0.0223],
  [0.0458, -0.0480],
  [0.08, -0.0737],
]
CONTROLLER_HASH = "a" * 64
POSTURE_MAP_HASH = "b" * 64


def _artifact() -> dict[str, object]:
  return station_calibration_artifact(
    controller_gain_hash=CONTROLLER_HASH,
    posture_map_hash=POSTURE_MAP_HASH,
    breakpoints=PROBE_BREAKPOINTS,
    source_probe={"git_sha": "test", "device": "cpu"},
  )


class StationBreakpointValidationTest(unittest.TestCase):
  def test_accepts_monotone_map_without_zero_pin(self):
    # Unlike the yaw feedforward there is no (0, 0) pin: the drift at zero
    # commanded pitch is nonzero because the LQR reference pitch is nonzero.
    points = validate_station_breakpoints([[-0.08, 0.05], [0.08, -0.07]])
    self.assertEqual(points, ((-0.08, 0.05), (0.08, -0.07)))

  def test_rejects_short_nonincreasing_or_nonfinite_maps(self):
    with self.assertRaisesRegex(ValueError, "at least two"):
      validate_station_breakpoints([[0.0, 0.0]])
    with self.assertRaisesRegex(ValueError, "strictly increase"):
      validate_station_breakpoints([[0.08, 0.0], [-0.08, 0.0]])
    with self.assertRaisesRegex(ValueError, "non-increasing"):
      validate_station_breakpoints([[-0.08, -0.01], [0.08, 0.05]])
    with self.assertRaisesRegex(ValueError, "finite"):
      validate_station_breakpoints([[0.0, float("nan")], [0.1, 0.0]])

  def test_interpolation_matches_numpy_and_clamps_out_of_domain(self):
    breakpoints = validate_station_breakpoints(PROBE_BREAKPOINTS)
    pitch = np.array([-0.2, -0.08, -0.04, 0.0058, 0.0458, 0.08, 0.2])
    drift = station_drift(pitch, breakpoints)
    reference = np.interp(
      pitch,
      [point[0] for point in breakpoints],
      [point[1] for point in breakpoints],
    )
    np.testing.assert_allclose(drift, reference)
    self.assertEqual(drift[0], drift[1])
    self.assertEqual(drift[-1], drift[-2])

  def test_torch_interpolation_matches_numpy_contract(self):
    # The action term interpolates with _torch_linear_interpolate while the
    # artifact is fitted and verified against numpy.interp; pin the two
    # implementations together over and beyond the calibrated domain.
    breakpoints = validate_station_breakpoints(PROBE_BREAKPOINTS)
    pitch = np.linspace(-0.15, 0.15, 301)
    expected = station_drift(pitch, breakpoints)
    actual = _torch_linear_interpolate(
      torch.tensor(pitch, dtype=torch.double),
      torch.tensor([point[0] for point in breakpoints], dtype=torch.double),
      torch.tensor([point[1] for point in breakpoints], dtype=torch.double),
    )
    np.testing.assert_allclose(actual.numpy(), expected, atol=1.0e-12)


class StationArtifactTest(unittest.TestCase):
  def test_round_trip_parses_with_both_bindings(self):
    payload = _artifact()
    parsed = parse_station_calibration_artifact(
      payload,
      controller_gain_hash=CONTROLLER_HASH,
      posture_map_hash=POSTURE_MAP_HASH,
    )
    self.assertEqual(
      parsed.breakpoints,
      tuple((pitch, drift) for pitch, drift in PROBE_BREAKPOINTS),
    )
    self.assertEqual(
      parsed.station_calibration_hash,
      payload["station_calibration_hash"],
    )

  def test_rejects_tampered_or_misbound_artifacts(self):
    payload = _artifact()
    tampered = dict(payload)
    tampered["breakpoints"] = [[-0.08, 0.05], [0.08, -0.09]]
    with self.assertRaisesRegex(ValueError, "does not match its artifact"):
      parse_station_calibration_artifact(
        tampered,
        controller_gain_hash=CONTROLLER_HASH,
        posture_map_hash=POSTURE_MAP_HASH,
      )
    with self.assertRaisesRegex(ValueError, "different controller"):
      parse_station_calibration_artifact(
        payload,
        controller_gain_hash="c" * 64,
        posture_map_hash=POSTURE_MAP_HASH,
      )
    with self.assertRaisesRegex(ValueError, "different posture map"):
      parse_station_calibration_artifact(
        payload,
        controller_gain_hash=CONTROLLER_HASH,
        posture_map_hash="d" * 64,
      )
    with self.assertRaisesRegex(ValueError, "schema_version"):
      parse_station_calibration_artifact(
        {**payload, "schema_version": 2},
        controller_gain_hash=CONTROLLER_HASH,
        posture_map_hash=POSTURE_MAP_HASH,
      )

  def test_artifact_requires_full_length_hashes(self):
    with self.assertRaisesRegex(ValueError, "Controller gain hash"):
      station_calibration_artifact(
        controller_gain_hash="short",
        posture_map_hash=POSTURE_MAP_HASH,
        breakpoints=PROBE_BREAKPOINTS,
        source_probe={},
      )
    with self.assertRaisesRegex(ValueError, "Posture map hash"):
      station_calibration_artifact(
        controller_gain_hash=CONTROLLER_HASH,
        posture_map_hash="short",
        breakpoints=PROBE_BREAKPOINTS,
        source_probe={},
      )


def _qualification_payload(
  *,
  pitches: list[float],
  heights: list[float],
  drift_of_pitch,
  height_offsets: list[float] | None = None,
  station_active: bool = False,
) -> dict[str, object]:
  offsets = height_offsets or [0.0] * len(heights)
  cells = []
  for pitch in pitches:
    for height, offset in zip(heights, offsets):
      cells.append(
        {
          "target_height": height,
          "target_pitch": pitch,
          "vx_command": 0.0,
          "mean_actual_lin_x": drift_of_pitch(pitch) + offset,
          "terminated_events": 0.0,
        }
      )
  return {
    "schema_version": 1,
    "kind": "posture_balance_qualification",
    "controller_qualified": True,
    "posture_map_qualified": True,
    "station_calibration_qualified": station_active,
    "controller_gain_hash": CONTROLLER_HASH,
    "posture_map_hash": POSTURE_MAP_HASH,
    "grid_cells": cells,
  }


class StationFitterTest(unittest.TestCase):
  PITCHES = [0.0116, 0.0287, 0.0458, 0.0629, 0.08]
  HEIGHTS = [0.299, 0.302, 0.304, 0.307, 0.309]

  @staticmethod
  def _drift(pitch: float) -> float:
    return -0.0136 - 0.751 * pitch

  def test_fit_averages_heights_per_pitch_level(self):
    payload = _qualification_payload(
      pitches=self.PITCHES,
      heights=self.HEIGHTS,
      drift_of_pitch=self._drift,
      height_offsets=[-0.0004, -0.0002, 0.0, 0.0002, 0.0004],
    )
    breakpoints = fit_station_breakpoints(
      payload["grid_cells"], max_height_spread=0.01
    )
    self.assertEqual([point[0] for point in breakpoints], self.PITCHES)
    for pitch, drift in breakpoints:
      self.assertAlmostEqual(drift, self._drift(pitch), places=12)

  def test_fit_ignores_vx_cells_and_rejects_bad_inputs(self):
    payload = _qualification_payload(
      pitches=self.PITCHES,
      heights=self.HEIGHTS,
      drift_of_pitch=self._drift,
    )
    cells = list(payload["grid_cells"])
    # A vx spot-check cell with an off-trend drift must not enter the fit.
    cells.append(
      {
        "target_height": 0.304,
        "target_pitch": 0.0458,
        "vx_command": 0.05,
        "mean_actual_lin_x": 0.0018,
        "terminated_events": 0.0,
      }
    )
    breakpoints = fit_station_breakpoints(cells, max_height_spread=0.01)
    self.assertEqual(len(breakpoints), len(self.PITCHES))

    with self.assertRaisesRegex(ValueError, "three distinct pitch"):
      fit_station_breakpoints(
        _qualification_payload(
          pitches=self.PITCHES[:2],
          heights=self.HEIGHTS,
          drift_of_pitch=self._drift,
        )["grid_cells"],
        max_height_spread=0.01,
      )
    with self.assertRaisesRegex(ValueError, "height-independence"):
      fit_station_breakpoints(
        _qualification_payload(
          pitches=self.PITCHES,
          heights=self.HEIGHTS,
          drift_of_pitch=self._drift,
          height_offsets=[-0.02, 0.0, 0.0, 0.0, 0.02],
        )["grid_cells"],
        max_height_spread=0.01,
      )
    terminated = _qualification_payload(
      pitches=self.PITCHES,
      heights=self.HEIGHTS,
      drift_of_pitch=self._drift,
    )
    terminated["grid_cells"][0]["terminated_events"] = 1.0
    with self.assertRaisesRegex(ValueError, "termination-free"):
      fit_station_breakpoints(
        terminated["grid_cells"], max_height_spread=0.01
      )

  def test_validation_rejects_unqualified_or_compensated_inputs(self):
    payload = _qualification_payload(
      pitches=self.PITCHES,
      heights=self.HEIGHTS,
      drift_of_pitch=self._drift,
    )
    self.assertIs(_validated_qualification(payload), payload)
    with self.assertRaisesRegex(ValueError, "qualified LQR"):
      _validated_qualification({**payload, "controller_qualified": False})
    with self.assertRaisesRegex(ValueError, "qualified posture map"):
      _validated_qualification({**payload, "posture_map_qualified": False})
    with self.assertRaisesRegex(ValueError, "already ran with station"):
      _validated_qualification(
        {**payload, "station_calibration_qualified": True}
      )
    with self.assertRaisesRegex(ValueError, "64-character"):
      _validated_qualification({**payload, "posture_map_hash": "short"})
    with self.assertRaisesRegex(ValueError, "not a posture_balance"):
      _validated_qualification({**payload, "kind": "other"})

  def test_cli_writes_parseable_artifact(self):
    payload = _qualification_payload(
      pitches=self.PITCHES,
      heights=self.HEIGHTS,
      drift_of_pitch=self._drift,
    )
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)
      input_path = temp_path / "qualification.json"
      output_path = temp_path / "station_calibration.json"
      input_path.write_text(json.dumps(payload), encoding="utf-8")
      env = os.environ.copy()
      env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

      completed = subprocess.run(
        [
          sys.executable,
          "-m",
          "hoppertrex_mjlab.scripts.fit_hybrid_station_calibration",
          "--input",
          str(input_path),
          "--output",
          str(output_path),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
      )

      self.assertEqual(completed.returncode, 0, completed.stderr)
      artifact = json.loads(output_path.read_text(encoding="utf-8"))
      parsed = parse_station_calibration_artifact(
        artifact,
        controller_gain_hash=CONTROLLER_HASH,
        posture_map_hash=POSTURE_MAP_HASH,
      )
      self.assertEqual(len(parsed.breakpoints), len(self.PITCHES))
      self.assertEqual(
        artifact["source_probe"]["kind"], "posture_balance_qualification"
      )
      self.assertEqual(len(artifact["source_probe"]["input_sha256"]), 64)


if __name__ == "__main__":
  unittest.main()
