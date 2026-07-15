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
from hoppertrex_mjlab.scripts.fit_hybrid_posture_map import (
  validated_sweep_metadata,
)


def _write_qualified_sweep_sidecar(path: Path, sample_count: int) -> None:
  path.with_suffix('.json').write_text(
    json.dumps(
      {
        'schema_version': 1,
        'git_sha': 'abc123',
        'seed': 1,
        'point_count': sample_count,
        'joint_names': list(LEG_JOINT_NAMES),
        'controller': {
          'type': 'lqr',
          'qualified': True,
          'gain_hash': 'gain123',
        },
        'calibration': {'hash': 'calibration123'},
      }
    ),
    encoding='utf-8',
  )


class HybridPostureTest(unittest.TestCase):
  def test_sweep_metadata_rejects_unqualified_controller(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      input_path = Path(temp_dir) / 'posture_sweep.npz'
      input_path.touch()
      metadata_path = input_path.with_suffix('.json')
      metadata_path.write_text(
        json.dumps(
          {
            'schema_version': 1,
            'git_sha': 'abc123',
            'point_count': 4,
            'joint_names': list(LEG_JOINT_NAMES),
            'controller': {
              'type': 'pd',
              'qualified': False,
              'gain_hash': None,
            },
          }
        ),
        encoding='utf-8',
      )

      with self.assertRaisesRegex(ValueError, 'qualified LQR'):
        validated_sweep_metadata(input_path)

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

  def test_absolute_joint_margin_overrides_fraction_of_range(self):
    # 2026-07-15 diagnosis: the 0.10 fraction margin scales with the joint
    # range (0.279 rad on the knees) and rejected the nominal standing
    # posture at 0.245 rad from its limit with actuator loads <= 0.33. The
    # absolute margin grounds the headroom in the dynamic excursion instead.
    joint_lower = np.full(4, -1.0)
    joint_upper = np.full(4, 1.0)
    joint_positions = np.array(
      [
        [-0.85, 0.0, 0.0, 0.0],
        [-0.90, 0.0, 0.0, 0.0],
      ]
    )
    kwargs = dict(
      non_wheel_contact=np.zeros(2, dtype=bool),
      joint_positions=joint_positions,
      joint_lower=joint_lower,
      joint_upper=joint_upper,
      actuator_load_fraction=np.full((2, 4), 0.3),
    )

    fraction = select_feasible_samples(**kwargs)
    np.testing.assert_array_equal(fraction, [False, False])

    absolute = select_feasible_samples(**kwargs, joint_margin_rad=0.12)
    np.testing.assert_array_equal(absolute, [True, False])

  def test_absolute_joint_margin_rejects_invalid_values(self):
    kwargs = dict(
      non_wheel_contact=np.zeros(1, dtype=bool),
      joint_positions=np.zeros((1, 4)),
      joint_lower=np.full(4, -1.0),
      joint_upper=np.full(4, 1.0),
      actuator_load_fraction=np.full((1, 4), 0.3),
    )
    with self.assertRaisesRegex(ValueError, "finite and non-negative"):
      select_feasible_samples(**kwargs, joint_margin_rad=-0.1)
    with self.assertRaisesRegex(ValueError, "usable room"):
      select_feasible_samples(**kwargs, joint_margin_rad=1.0)

  def test_training_envelope_shrinks_ranges_and_caps_pitch(self):
    heights, pitches = np.meshgrid(
      np.linspace(0.30, 0.50, 5),
      np.linspace(-0.20, 0.20, 5),
      indexing="ij",
    )
    envelope = training_envelope(
      heights=heights.ravel(),
      pitches=pitches.ravel(),
      feasible=np.ones(heights.size, dtype=bool),
      inward_fraction=0.10,
      pitch_limit=0.08,
    )

    self.assertEqual(
      envelope,
      PostureEnvelope(
        height_range=(0.32, 0.48),
        pitch_range=(-0.08, 0.08),
        verified_grid_shape=(5, 5),
      ),
    )

  def test_training_envelope_selects_only_an_all_feasible_grid_rectangle(self):
    heights, pitches = np.meshgrid(
      np.array([0.30, 0.40, 0.50]),
      np.array([-0.08, 0.00, 0.08]),
      indexing="ij",
    )
    feasible = np.array(
      [
        [True, True, True],
        [True, True, False],
        [True, True, False],
      ]
    )

    envelope = training_envelope(
      heights=heights.ravel(),
      pitches=pitches.ravel(),
      feasible=feasible.ravel(),
      inward_fraction=0.10,
      pitch_limit=0.08,
    )

    self.assertEqual(
      envelope,
      PostureEnvelope(
        height_range=(0.32, 0.48),
        pitch_range=(-0.072, -0.008),
        verified_grid_shape=(3, 2),
      ),
    )

  def test_training_envelope_rejects_feasible_set_without_a_2d_rectangle(self):
    heights, pitches = np.meshgrid(
      np.array([0.30, 0.40, 0.50]),
      np.array([-0.08, 0.00, 0.08]),
      indexing="ij",
    )
    feasible = np.array(
      [
        [False, True, False],
        [True, True, True],
        [False, True, False],
      ]
    )

    with self.assertRaisesRegex(ValueError, "verified 2D rectangle"):
      training_envelope(
        heights=heights.ravel(),
        pitches=pitches.ravel(),
        feasible=feasible.ravel(),
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
      PostureEnvelope(
        (0.32, 0.43),
        (-0.04, 0.04),
        verified_grid_shape=(2, 2),
      ),
      feasible_sample_count=4,
      total_sample_count=4,
    )

    self.assertEqual(payload["joint_names"], list(LEG_JOINT_NAMES))
    self.assertEqual(payload["feature_names"], ["bias", "height", "pitch"])
    self.assertEqual(len(payload["map_hash"]), 64)
    self.assertEqual(payload["map_hash"], posture_map.map_hash)
    self.assertEqual(payload["feasible_sample_count"], 4)
    self.assertEqual(
      payload["envelope_verification"]["method"],
      "all_feasible_grid_rectangle",
    )

  def test_cli_filters_sweep_and_writes_posture_map_json(self):
    height_grid, pitch_grid = np.meshgrid(
      np.linspace(0.30, 0.50, 3),
      np.linspace(-0.10, 0.10, 3),
      indexing="ij",
    )
    heights = height_grid.ravel()
    pitches = pitch_grid.ravel()
    features = np.column_stack((np.ones(heights.size), heights, pitches))
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
        non_wheel_contact=np.zeros(heights.size, dtype=bool),
        joint_lower=np.full(4, -2.0),
        joint_upper=np.full(4, 2.0),
        actuator_load_fraction=np.full((heights.size, 4), 0.5),
      )
      _write_qualified_sweep_sidecar(input_path, heights.size)
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
      self.assertEqual(
        payload["envelope_verification"],
        {
          "method": "all_feasible_grid_rectangle",
          "grid_shape": [3, 3],
        },
      )
      self.assertEqual(payload["feasible_sample_count"], 9)

  def test_cli_accepts_measured_outputs_from_joint_coordinate_sweep(self):
    hip_grid, knee_grid = np.meshgrid(
      np.linspace(-1.0, 1.0, 3),
      np.linspace(-1.0, 1.0, 3),
      indexing="ij",
    )
    hip_offsets = hip_grid.ravel()
    knee_offsets = knee_grid.ravel()
    heights = 0.40 + 0.04 * hip_offsets + 0.02 * knee_offsets
    pitches = -0.03 * hip_offsets + 0.05 * knee_offsets
    features = np.column_stack((np.ones(heights.size), heights, pitches))
    coefficients = np.array(
      [
        [-0.2, 0.2, -0.4, 0.4],
        [-0.6, 0.6, -0.5, 0.5],
        [0.3, 0.3, -0.2, -0.2],
      ]
    )
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)
      input_path = temp_path / "measured_posture_sweep.npz"
      output_path = temp_path / "posture_map.json"
      np.savez(
        input_path,
        heights=heights,
        pitches=pitches,
        joint_positions=features @ coefficients,
        non_wheel_contact=np.zeros(heights.size, dtype=bool),
        joint_lower=np.full(4, -2.0),
        joint_upper=np.full(4, 2.0),
        actuator_load_fraction=np.full((heights.size, 4), 0.5),
        hip_offsets=hip_offsets,
        knee_offsets=knee_offsets,
      )
      _write_qualified_sweep_sidecar(input_path, heights.size)
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
      self.assertEqual(
        payload["envelope_verification"],
        {
          "method": "all_feasible_sweep_grid_hull_rectangle",
          "grid_shape": [3, 3],
        },
      )
      height_range = payload["training_envelope"]["height"]
      pitch_range = payload["training_envelope"]["pitch"]
      np.testing.assert_allclose(height_range, [0.37504, 0.42496])
      np.testing.assert_allclose(pitch_range, [-0.03328, 0.03328])


if __name__ == "__main__":
  unittest.main()
