import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from hoppertrex_mjlab.hybrid.posture import (
  LEG_JOINT_NAMES,
  PostureEnvelope,
  fit_posture_map,
  posture_map_to_dict,
  predict_leg_targets,
  select_feasible_samples,
  training_envelope,
)


class HybridPostureTest(unittest.TestCase):
  def test_feasibility_filters_contact_joint_margin_and_actuator_load(self):
    joint_lower = np.full(4, -1.0)
    joint_upper = np.full(4, 1.0)
    joint_positions = np.array(
      [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [-0.81, 0.0, 0.0, 0.0],
        [-0.80, 0.80, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
      ]
    )
    non_wheel_contact = np.array([False, True, False, False, False])
    actuator_load = np.array(
      [
        [0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.8],
      ]
    )

    feasible = select_feasible_samples(
      non_wheel_contact=non_wheel_contact,
      joint_positions=joint_positions,
      joint_lower=joint_lower,
      joint_upper=joint_upper,
      actuator_load_fraction=actuator_load,
    )

    np.testing.assert_array_equal(feasible, [True, False, False, True, False])

  def test_training_envelope_shrinks_ranges_and_caps_pitch(self):
    envelope = training_envelope(
      heights=np.array([0.30, 0.35, 0.40, 0.45, 0.50]),
      pitches=np.array([-0.20, -0.10, 0.0, 0.10, 0.20]),
      feasible=np.ones(5, dtype=bool),
      inward_fraction=0.10,
      pitch_limit=0.08,
    )

    self.assertEqual(
      envelope,
      PostureEnvelope(
        height_range=(0.32, 0.48),
        pitch_range=(-0.08, 0.08),
      ),
    )

  def test_posture_map_fits_height_and_pitch_to_four_leg_joint_targets(self):
    heights = np.array([0.30, 0.34, 0.38, 0.42, 0.46, 0.50])
    pitches = np.array([-0.06, 0.02, 0.05, -0.03, 0.07, -0.01])
    coefficients = np.array(
      [
        [-0.2, 0.2, -0.4, 0.4],
        [-1.0, 1.0, -0.8, 0.8],
        [0.5, 0.5, -0.3, -0.3],
      ]
    )
    features = np.column_stack((np.ones(heights.size), heights, pitches))
    joint_positions = features @ coefficients

    posture_map = fit_posture_map(heights, pitches, joint_positions)
    predicted = predict_leg_targets(
      posture_map,
      heights=np.array([0.36, 0.44]),
      pitches=np.array([0.04, -0.02]),
    )

    np.testing.assert_allclose(posture_map.coefficients, coefficients)
    np.testing.assert_allclose(
      predicted,
      np.column_stack(
        (
          np.ones(2),
          np.array([0.36, 0.44]),
          np.array([0.04, -0.02]),
        )
      )
      @ coefficients,
    )

  def test_posture_map_json_names_all_four_joints_and_has_stable_hash(self):
    posture_map = fit_posture_map(
      heights=np.array([0.30, 0.35, 0.40, 0.45]),
      pitches=np.array([-0.04, 0.04, 0.02, -0.02]),
      joint_positions=np.array(
        [
          [-0.5, 0.5, -0.4, 0.4],
          [-0.6, 0.6, -0.5, 0.5],
          [-0.7, 0.7, -0.6, 0.6],
          [-0.8, 0.8, -0.7, 0.7],
        ]
      ),
    )

    payload = posture_map_to_dict(
      posture_map,
      PostureEnvelope((0.32, 0.43), (-0.04, 0.04)),
      feasible_sample_count=4,
      total_sample_count=4,
    )

    self.assertEqual(payload["joint_names"], list(LEG_JOINT_NAMES))
    self.assertEqual(payload["feature_names"], ["bias", "height", "pitch"])
    self.assertEqual(len(payload["map_hash"]), 64)
    self.assertEqual(payload["map_hash"], posture_map.map_hash)
    self.assertEqual(payload["feasible_sample_count"], 4)

  def test_cli_filters_sweep_and_writes_posture_map_json(self):
    heights = np.array([0.30, 0.35, 0.40, 0.45, 0.50])
    pitches = np.array([-0.10, 0.02, 0.0, -0.03, 0.10])
    features = np.column_stack((np.ones(5), heights, pitches))
    coefficients = np.array(
      [
        [-0.2, 0.2, -0.4, 0.4],
        [-1.0, 1.0, -0.8, 0.8],
        [0.5, 0.5, -0.3, -0.3],
      ]
    )
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)
      input_path = temp_path / "posture_sweep.npz"
      output_path = temp_path / "posture_map.json"
      np.savez(
        input_path,
        heights=heights,
        pitches=pitches,
        joint_positions=features @ coefficients,
        non_wheel_contact=np.array([False, False, False, False, False]),
        joint_lower=np.full(4, -2.0),
        joint_upper=np.full(4, 2.0),
        actuator_load_fraction=np.full((5, 4), 0.5),
      )
      env = os.environ.copy()
      env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

      completed = subprocess.run(
        [
          sys.executable,
          "-m",
          "hoppertrex_mjlab.scripts.fit_hybrid_posture_map",
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
      payload = json.loads(output_path.read_text(encoding="utf-8"))
      self.assertEqual(payload["source_npz"], str(input_path.resolve()))
      self.assertEqual(payload["training_envelope"]["pitch"], [-0.08, 0.08])
      self.assertEqual(payload["feasible_sample_count"], 5)


if __name__ == "__main__":
  unittest.main()
