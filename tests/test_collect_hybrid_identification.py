import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from hoppertrex_mjlab.scripts.collect_hybrid_identification import (
  AFFINE_EQUILIBRIUM_WINDOW_STEPS,
  IDENTIFICATION_ARRAY_NAMES,
  build_controller_state,
  calibrated_velocity_reference,
  deterministic_transition_split,
  filter_valid_transitions,
  signed_balance_input,
  write_identification_dataset,
)


class CollectHybridIdentificationTest(unittest.TestCase):
  def test_affine_equilibrium_window_is_frozen(self):
    self.assertEqual(AFFINE_EQUILIBRIUM_WINDOW_STEPS, 100)
  def test_calibrated_reference_matches_runtime_scale_bias_and_station_drift(self):
    reference = calibrated_velocity_reference(
      torch.tensor([0.10, 0.00, -0.10]),
      torch.tensor([-0.02, 0.00, 0.02]),
      scale=0.86,
      bias=-0.012,
      station_breakpoints=((-0.02, 0.01), (0.0, 0.0), (0.02, -0.01)),
    )
    torch.testing.assert_close(
      reference,
      torch.tensor([0.064, -0.012, -0.088]),
    )

  def test_module_prepares_legacy_task_import_path(self):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    completed = subprocess.run(
      [
        sys.executable,
        "-c",
        (
          "import hoppertrex_mjlab.scripts.collect_hybrid_identification;"
          "import hoppertrex_mjlab.tasks.hoppertrex_hybrid_task"
        ),
      ],
      check=False,
      capture_output=True,
      env=env,
      text=True,
    )

    self.assertEqual(completed.returncode, 0, completed.stderr)

  def test_build_controller_state_uses_documented_order_and_signs(self):
    projected_gravity = torch.tensor(
      [
        [0.10, 0.0, -0.99],
        [-0.20, 0.0, -0.98],
      ]
    )
    pitch_rate = torch.tensor([0.3, -0.4])
    root_vx = torch.tensor([0.08, -0.02])
    commanded_vx = torch.tensor([0.03, -0.01])
    wheel_velocity = torch.tensor(
      [
        [-2.0, 4.0],
        [3.0, -1.0],
      ]
    )

    state = build_controller_state(
      projected_gravity=projected_gravity,
      pitch_rate=pitch_rate,
      root_vx=root_vx,
      commanded_vx=commanded_vx,
      wheel_velocity=wheel_velocity,
      wheel_radius=0.10,
    )

    expected = torch.tensor(
      [
        [torch.atan2(torch.tensor(0.10), torch.tensor(0.99)), 0.3, 0.05, 2.7],
        [torch.atan2(torch.tensor(-0.20), torch.tensor(0.98)), -0.4, -0.01, -1.9],
      ]
    )
    torch.testing.assert_close(state, expected)

  def test_signed_balance_input_uses_actual_left_and_right_targets(self):
    wheel_targets = torch.tensor(
      [
        [-3.0, 5.0],
        [2.0, -4.0],
      ]
    )

    inputs = signed_balance_input(wheel_targets)

    torch.testing.assert_close(inputs, torch.tensor([[4.0], [-3.0]]))

  def test_filter_valid_transitions_drops_terminated_and_truncated_rows(self):
    states = np.arange(16, dtype=np.float64).reshape(4, 4)
    inputs = np.arange(4, dtype=np.float64).reshape(4, 1)
    next_states = states + 100.0

    filtered = filter_valid_transitions(
      states,
      inputs,
      next_states,
      terminated=np.array([False, True, False, False]),
      truncated=np.array([False, False, True, False]),
    )

    np.testing.assert_array_equal(filtered[0], states[[0, 3]])
    np.testing.assert_array_equal(filtered[1], inputs[[0, 3]])
    np.testing.assert_array_equal(filtered[2], next_states[[0, 3]])

  def test_deterministic_split_is_reproducible_and_disjoint(self):
    states = np.arange(80, dtype=np.float64).reshape(20, 4)
    inputs = np.arange(20, dtype=np.float64).reshape(20, 1)
    next_states = states + 0.5

    first = deterministic_transition_split(
      states,
      inputs,
      next_states,
      heldout_fraction=0.25,
      seed=17,
    )
    second = deterministic_transition_split(
      states,
      inputs,
      next_states,
      heldout_fraction=0.25,
      seed=17,
    )

    self.assertEqual(tuple(first), IDENTIFICATION_ARRAY_NAMES)
    for name in IDENTIFICATION_ARRAY_NAMES:
      np.testing.assert_array_equal(first[name], second[name])
    self.assertEqual(first["states"].shape, (15, 4))
    self.assertEqual(first["heldout_states"].shape, (5, 4))
    train_rows = {tuple(row) for row in first["states"]}
    heldout_rows = {tuple(row) for row in first["heldout_states"]}
    self.assertFalse(train_rows & heldout_rows)
    self.assertEqual(train_rows | heldout_rows, {tuple(row) for row in states})

  def test_writer_emits_fitter_schema_and_json_metadata(self):
    states = np.arange(48, dtype=np.float64).reshape(12, 4)
    inputs = np.arange(12, dtype=np.float64).reshape(12, 1)
    arrays = deterministic_transition_split(
      states,
      inputs,
      states + 1.0,
      heldout_fraction=0.25,
      seed=5,
    )
    metadata = {
      "seed": 5,
      "device": "cpu",
      "state_names": [
        "pitch",
        "pitch_rate",
        "vx_error",
        "signed_wheel_speed_error",
      ],
    }

    with tempfile.TemporaryDirectory() as temp_dir:
      output = Path(temp_dir) / "identification.npz"
      metadata_path = write_identification_dataset(output, arrays, metadata)

      with np.load(output, allow_pickle=False) as saved:
        self.assertEqual(tuple(saved.files), IDENTIFICATION_ARRAY_NAMES)
        for name in IDENTIFICATION_ARRAY_NAMES:
          np.testing.assert_array_equal(saved[name], arrays[name])
      self.assertEqual(metadata_path, output.with_suffix(".json"))
      self.assertEqual(
        json.loads(metadata_path.read_text(encoding="utf-8")),
        metadata,
      )


if __name__ == "__main__":
  unittest.main()
