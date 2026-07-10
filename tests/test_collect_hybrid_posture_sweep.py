import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from hoppertrex_mjlab.scripts.collect_hybrid_posture_sweep import (
  POSTURE_REQUIRED_ARRAY_NAMES,
  build_symmetric_leg_targets,
  posture_sweep_grid,
  summarize_posture_samples,
  write_posture_sweep_dataset,
)


class CollectHybridPostureSweepTest(unittest.TestCase):
  def test_runtime_dependencies_load_without_duplicate_task_registration(self):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    completed = subprocess.run(
      [
        sys.executable,
        "-c",
        (
          "from hoppertrex_mjlab.scripts.collect_hybrid_posture_sweep "
          "import load_runtime_dependencies;"
          "deps = load_runtime_dependencies();"
          "assert deps['peak_torque'] == 97.0"
        ),
      ],
      check=False,
      capture_output=True,
      env=env,
      text=True,
    )

    self.assertEqual(completed.returncode, 0, completed.stderr)

  def test_grid_is_deterministic_and_covers_every_coordinate_pair(self):
    hip_offsets, knee_offsets = posture_sweep_grid(
      hip_range=(-0.10, 0.10),
      knee_range=(-0.20, 0.20),
      hip_points=3,
      knee_points=2,
    )

    np.testing.assert_allclose(
      hip_offsets,
      [-0.10, -0.10, 0.0, 0.0, 0.10, 0.10],
    )
    np.testing.assert_allclose(
      knee_offsets,
      [-0.20, 0.20, -0.20, 0.20, -0.20, 0.20],
    )

  def test_symmetric_targets_preserve_two_leg_mirror_signs(self):
    initial = np.array([-0.50, 0.50, -0.40, 0.40])

    targets = build_symmetric_leg_targets(
      initial,
      hip_offsets=np.array([0.10, -0.20]),
      knee_offsets=np.array([-0.05, 0.15]),
    )

    np.testing.assert_allclose(
      targets,
      [
        [-0.60, 0.60, -0.35, 0.35],
        [-0.30, 0.30, -0.55, 0.55],
      ],
    )

  def test_summary_uses_means_but_conservative_contact_and_peak_load(self):
    heights = np.array([[0.30, 0.40], [0.32, 0.42], [0.34, 0.44]])
    pitches = np.array([[-0.02, 0.01], [0.00, 0.03], [0.02, 0.05]])
    joint_positions = np.array(
      [
        [[-0.5, 0.5, -0.4, 0.4], [-0.6, 0.6, -0.5, 0.5]],
        [[-0.4, 0.4, -0.3, 0.3], [-0.5, 0.5, -0.4, 0.4]],
        [[-0.3, 0.3, -0.2, 0.2], [-0.4, 0.4, -0.3, 0.3]],
      ]
    )
    contact = np.array(
      [[False, False], [True, False], [False, False]]
    )
    loads = np.array(
      [
        [[0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4]],
        [[0.3, 0.2, 0.5, 0.4], [0.2, 0.3, 0.4, 0.5]],
        [[0.1, 0.4, 0.3, 0.2], [0.3, 0.1, 0.2, 0.4]],
      ]
    )

    summary = summarize_posture_samples(
      heights=heights,
      pitches=pitches,
      joint_positions=joint_positions,
      non_wheel_contact=contact,
      actuator_load_fraction=loads,
      invalid=np.array([False, True]),
    )

    np.testing.assert_allclose(summary["heights"], [0.32, 0.42])
    np.testing.assert_allclose(summary["pitches"], [0.0, 0.03])
    np.testing.assert_allclose(
      summary["joint_positions"],
      joint_positions.mean(axis=0),
    )
    np.testing.assert_array_equal(
      summary["non_wheel_contact"],
      [True, True],
    )
    np.testing.assert_allclose(
      summary["actuator_load_fraction"],
      loads.max(axis=0),
    )

  def test_writer_preserves_fitter_arrays_and_diagnostic_targets(self):
    arrays = {
      "heights": np.array([0.31, 0.35, 0.39, 0.43]),
      "pitches": np.array([-0.04, 0.04, -0.02, 0.02]),
      "joint_positions": np.zeros((4, 4)),
      "non_wheel_contact": np.zeros(4, dtype=bool),
      "joint_lower": np.full(4, -1.0),
      "joint_upper": np.full(4, 1.0),
      "actuator_load_fraction": np.full((4, 4), 0.2),
      "target_joint_positions": np.ones((4, 4)),
      "hip_offsets": np.array([-0.1, -0.1, 0.1, 0.1]),
      "knee_offsets": np.array([-0.1, 0.1, -0.1, 0.1]),
      "invalid": np.zeros(4, dtype=bool),
    }
    metadata = {"device": "cpu", "controller_gain_hash": "abc"}

    with tempfile.TemporaryDirectory() as temp_dir:
      output = Path(temp_dir) / "posture_sweep.npz"
      metadata_path = write_posture_sweep_dataset(output, arrays, metadata)

      with np.load(output, allow_pickle=False) as saved:
        self.assertTrue(set(POSTURE_REQUIRED_ARRAY_NAMES) <= set(saved.files))
        np.testing.assert_array_equal(
          saved["target_joint_positions"],
          arrays["target_joint_positions"],
        )
      self.assertEqual(metadata_path, output.with_suffix(".json"))
      self.assertEqual(
        json.loads(metadata_path.read_text(encoding="utf-8")),
        metadata,
      )


if __name__ == "__main__":
  unittest.main()
