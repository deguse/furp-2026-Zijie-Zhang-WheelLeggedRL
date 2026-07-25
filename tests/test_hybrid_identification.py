import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from hoppertrex_mjlab.hybrid.identification import (
  AffineEquilibrium,
  CONTROLLER_STATE_NAMES,
  NOMINAL_WHEEL_RADIUS_M,
  anchor_gains_to_incumbent,
  center_affine_transitions,
  closed_loop_spectral_radius,
  controllability_rank,
  controller_design_to_dict,
  estimate_affine_equilibrium,
  fit_discrete_model,
  identify_controller,
  one_step_nrmse,
  solve_lqr_gain,
  state_construction_spec,
)


def _controllable_system() -> tuple[np.ndarray, np.ndarray]:
  a = np.array(
    [
      [1.0, 0.02, 0.0, 0.0],
      [0.0, 0.96, 0.02, 0.0],
      [0.0, 0.0, 0.93, 0.02],
      [0.0, 0.0, 0.0, 0.90],
    ]
  )
  b = np.array([[0.0], [0.01], [0.04], [0.20]])
  return a, b


def _samples(
  a: np.ndarray,
  b: np.ndarray,
  *,
  count: int,
  seed: int,
  noise_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  rng = np.random.default_rng(seed)
  states = rng.normal(size=(count, 4))
  inputs = rng.normal(size=(count, 1))
  next_states = states @ a.T + inputs @ b.T
  next_states += rng.normal(scale=noise_std, size=next_states.shape)
  return states, inputs, next_states


class HybridIdentificationTest(unittest.TestCase):
  def test_affine_equilibrium_centers_state_input_and_next_state(self):
    states = np.asarray([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]])
    inputs = np.asarray([[0.5], [1.5]])
    equilibrium = estimate_affine_equilibrium(states, inputs)
    np.testing.assert_allclose(equilibrium.state, [2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(equilibrium.input, [1.0])
    centered = center_affine_transitions(
      states,
      inputs,
      states + 0.25,
      equilibrium,
    )
    np.testing.assert_allclose(np.mean(centered[0], axis=0), 0.0)
    np.testing.assert_allclose(np.mean(centered[1], axis=0), 0.0)
    np.testing.assert_allclose(centered[2], centered[0] + 0.25)

  def test_affine_equilibrium_rejects_wrong_dimensions(self):
    with self.assertRaisesRegex(ValueError, "four states and one input"):
      estimate_affine_equilibrium(np.ones((4, 3)), np.ones((4, 1)))
    with self.assertRaisesRegex(ValueError, "dimensions"):
      center_affine_transitions(
        np.ones((4, 4)),
        np.ones((4, 1)),
        np.ones((4, 4)),
        AffineEquilibrium(np.ones(3), np.ones(1)),
      )

  def test_closed_loop_spectral_radius_uses_u_equals_minus_kx(self):
    a = np.diag([1.1, 0.8])
    b = np.asarray([[1.0], [0.0]])
    gain = np.asarray([[0.3, 0.0]])
    self.assertAlmostEqual(closed_loop_spectral_radius(a, b, gain), 0.8)

  def test_gain_anchor_selects_largest_nonregressing_global_alpha(self):
    models = (
      type("Model", (), {"a": np.asarray([[1.1]]), "b": np.asarray([[1.0]])})(),
      type("Model", (), {"a": np.asarray([[1.2]]), "b": np.asarray([[1.0]])})(),
    )
    anchored = anchor_gains_to_incumbent(
      models,
      np.asarray([[[0.0]], [[0.0]]]),
      np.asarray([[0.4]]),
      spectral_margin=0.05,
      alpha_grid=(1.0, 0.5, 0.25, 0.0),
    )
    self.assertEqual(anchored.alpha, 0.0)
    np.testing.assert_allclose(anchored.gains, [[[0.4]], [[0.4]]])

  def test_gain_anchor_rejects_invalid_alpha_grid(self):
    model = type(
      "Model", (), {"a": np.asarray([[0.9]]), "b": np.asarray([[1.0]])}
    )()
    with self.assertRaisesRegex(ValueError, "alpha_grid"):
      anchor_gains_to_incumbent(
        (model,), np.asarray([[[0.1]]]), np.asarray([[0.2]]), alpha_grid=(0.5,)
      )
  def test_controller_state_order_is_explicit(self):
    self.assertEqual(
      CONTROLLER_STATE_NAMES,
      ("pitch", "pitch_rate", "vx_error", "signed_wheel_speed_error"),
    )

  def test_fit_recovers_discrete_model_and_full_controllability(self):
    a, b = _controllable_system()
    states, inputs, next_states = _samples(a, b, count=500, seed=3)

    model = fit_discrete_model(states, inputs, next_states)

    np.testing.assert_allclose(model.a, a, atol=1.0e-10)
    np.testing.assert_allclose(model.b, b, atol=1.0e-10)
    self.assertEqual(controllability_rank(model.a, model.b), 4)

  def test_nrmse_uses_each_state_range_and_reports_worst_dimension(self):
    actual = np.array(
      [
        [-1.0, -10.0],
        [0.0, 0.0],
        [1.0, 10.0],
      ]
    )
    predicted = actual + np.array(
      [
        [0.2, 1.0],
        [0.2, 1.0],
        [0.2, 1.0],
      ]
    )

    result = one_step_nrmse(actual, predicted)

    np.testing.assert_allclose(result.by_state, [0.10, 0.05])
    self.assertAlmostEqual(result.maximum, 0.10)

  def test_qualified_model_uses_dare_lqr_and_stable_gain_hash(self):
    a, b = _controllable_system()
    train = _samples(a, b, count=500, seed=4, noise_std=1.0e-4)
    heldout = _samples(a, b, count=200, seed=5, noise_std=1.0e-4)

    design = identify_controller(
      *train,
      heldout_states=heldout[0],
      heldout_inputs=heldout[1],
      heldout_next_states=heldout[2],
      q_diag=(20.0, 2.0, 4.0, 0.5),
      r_diag=(1.0,),
      pd_gain=(8.0, 1.0, 3.0, 0.2),
    )
    expected_gain = solve_lqr_gain(
      design.model.a,
      design.model.b,
      np.diag([20.0, 2.0, 4.0, 0.5]),
      np.diag([1.0]),
    )

    self.assertEqual(design.controller_type, "lqr")
    self.assertEqual(design.controllability_rank, 4)
    self.assertLessEqual(design.heldout_nrmse.maximum, 0.15)
    np.testing.assert_allclose(design.gain, expected_gain)
    self.assertEqual(len(design.gain_hash), 64)
    self.assertEqual(design.gain_hash, design.gain_hash)

  def test_uncontrollable_model_uses_explicit_pd_fallback(self):
    a, _ = _controllable_system()
    b = np.zeros((4, 1))
    train = _samples(a, b, count=300, seed=6)
    heldout = _samples(a, b, count=100, seed=7)

    design = identify_controller(
      *train,
      heldout_states=heldout[0],
      heldout_inputs=heldout[1],
      heldout_next_states=heldout[2],
      q_diag=(1.0, 1.0, 1.0, 1.0),
      r_diag=(1.0,),
      pd_gain=(8.0, 1.0, 3.0, 0.2),
    )

    self.assertEqual(design.controller_type, "pd")
    self.assertEqual(design.controllability_rank, 0)
    np.testing.assert_allclose(design.gain, [[8.0, 1.0, 3.0, 0.2]])
    self.assertIn("controllability", design.fallback_reasons[0])

  def test_bad_heldout_prediction_uses_pd_fallback(self):
    a, b = _controllable_system()
    train = _samples(a, b, count=300, seed=8)
    heldout = list(_samples(a, b, count=100, seed=9))
    heldout[2] = heldout[2] + np.linspace(-2.0, 2.0, 100)[:, None]

    design = identify_controller(
      *train,
      heldout_states=heldout[0],
      heldout_inputs=heldout[1],
      heldout_next_states=heldout[2],
      q_diag=(1.0, 1.0, 1.0, 1.0),
      r_diag=(1.0,),
      pd_gain=(8.0, 1.0, 3.0, 0.2),
    )

    self.assertEqual(design.controller_type, "pd")
    self.assertGreater(design.heldout_nrmse.maximum, 0.15)
    self.assertTrue(any("NRMSE" in reason for reason in design.fallback_reasons))

  def test_controller_design_serializes_complete_audit_metadata(self):
    a, b = _controllable_system()
    train = _samples(a, b, count=300, seed=10)
    heldout = _samples(a, b, count=100, seed=11)
    design = identify_controller(
      *train,
      heldout_states=heldout[0],
      heldout_inputs=heldout[1],
      heldout_next_states=heldout[2],
      q_diag=(20.0, 2.0, 4.0, 0.5),
      r_diag=(1.0,),
      pd_gain=(8.0, 1.0, 3.0, 0.2),
    )

    payload = controller_design_to_dict(design)

    self.assertEqual(payload["schema_version"], 1)
    self.assertEqual(payload["controller_type"], "lqr")
    self.assertEqual(payload["state_names"], list(CONTROLLER_STATE_NAMES))
    np.testing.assert_allclose(payload["model"]["a"], design.model.a)
    np.testing.assert_allclose(payload["model"]["b"], design.model.b)
    np.testing.assert_allclose(payload["gain"], design.gain)
    self.assertEqual(payload["controllability_rank"], 4)
    self.assertEqual(
      set(payload["heldout_one_step_nrmse"]["by_state"]),
      set(CONTROLLER_STATE_NAMES),
    )
    self.assertEqual(
      payload["heldout_one_step_nrmse"]["maximum"],
      design.heldout_nrmse.maximum,
    )
    self.assertEqual(payload["fallback_reasons"], [])
    self.assertEqual(payload["gain_hash"], design.gain_hash)

  def test_cli_reads_npz_and_writes_auditable_json(self):
    a, b = _controllable_system()
    train = _samples(a, b, count=300, seed=12)
    heldout = _samples(a, b, count=100, seed=13)
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)
      input_path = temp_path / "sweep.npz"
      output_path = temp_path / "controller.json"
      np.savez(
        input_path,
        states=train[0],
        inputs=train[1],
        next_states=train[2],
        heldout_states=heldout[0],
        heldout_inputs=heldout[1],
        heldout_next_states=heldout[2],
      )
      env = os.environ.copy()
      env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

      completed = subprocess.run(
        [
          sys.executable,
          "-m",
          "hoppertrex_mjlab.scripts.identify_hybrid_controller",
          "--input",
          str(input_path),
          "--output",
          str(output_path),
          "--q-diag",
          "20",
          "2",
          "4",
          "0.5",
          "--r",
          "1",
          "--pd-gain",
          "8",
          "1",
          "3",
          "0.2",
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
      )

      self.assertEqual(completed.returncode, 0, completed.stderr)
      payload = json.loads(output_path.read_text(encoding="utf-8"))
      self.assertEqual(payload["controller_type"], "lqr")
      self.assertEqual(payload["source_npz"], str(input_path.resolve()))
      self.assertEqual(
        payload["source_npz_sha256"],
        hashlib.sha256(input_path.read_bytes()).hexdigest(),
      )
      self.assertIsNone(payload["source_metadata_sha256"])
      self.assertEqual(payload["q_diag"], [20.0, 2.0, 4.0, 0.5])
      self.assertEqual(payload["r_diag"], [1.0])
      self.assertEqual(len(payload["gain_hash"]), 64)
      self.assertEqual(
        payload["state_construction"],
        state_construction_spec(NOMINAL_WHEEL_RADIUS_M),
      )

  def test_cli_rejects_wheel_radius_contradicting_collection_sidecar(self):
    a, b = _controllable_system()
    train = _samples(a, b, count=300, seed=14)
    heldout = _samples(a, b, count=100, seed=15)
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)
      input_path = temp_path / "sweep.npz"
      output_path = temp_path / "controller.json"
      np.savez(
        input_path,
        states=train[0],
        inputs=train[1],
        next_states=train[2],
        heldout_states=heldout[0],
        heldout_inputs=heldout[1],
        heldout_next_states=heldout[2],
      )
      input_path.with_suffix(".json").write_text(
        json.dumps({"wheel_radius": 2.0 * NOMINAL_WHEEL_RADIUS_M}),
        encoding="utf-8",
      )
      env = os.environ.copy()
      env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

      completed = subprocess.run(
        [
          sys.executable,
          "-m",
          "hoppertrex_mjlab.scripts.identify_hybrid_controller",
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

      self.assertNotEqual(completed.returncode, 0)
      self.assertIn("wheel_radius", completed.stderr)
      self.assertFalse(output_path.exists())

  def test_cli_carries_affine_equilibrium_and_spectral_provenance(self):
    a, b = _controllable_system()
    train = _samples(a, b, count=300, seed=16)
    heldout = _samples(a, b, count=100, seed=17)
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      input_path = root / "affine.npz"
      output_path = root / "controller.json"
      np.savez(
        input_path,
        states=train[0],
        inputs=train[1],
        next_states=train[2],
        heldout_states=heldout[0],
        heldout_inputs=heldout[1],
        heldout_next_states=heldout[2],
      )
      input_path.with_suffix(".json").write_text(
        json.dumps(
          {
            "wheel_radius": NOMINAL_WHEEL_RADIUS_M,
            "state_definition_version": "hybrid_lqr_affine_equilibrium_v3",
            "equilibrium_state": [0.01, 0.02, 0.03, 0.04],
            "equilibrium_input": [0.1],
            "controller": {"gain": [0.2, 0.1, 0.05, 0.01]},
          }
        ),
        encoding="utf-8",
      )
      env = os.environ.copy()
      env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
      completed = subprocess.run(
        [
          sys.executable,
          "-m",
          "hoppertrex_mjlab.scripts.identify_hybrid_controller",
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
      self.assertEqual(payload["equilibrium_input"], [0.1])
      self.assertEqual(len(payload["equilibrium_state"]), 4)
      self.assertIn("closed_loop_spectral_radius", payload)
      self.assertIn("incumbent_closed_loop_spectral_radius", payload)


if __name__ == "__main__":
  unittest.main()
