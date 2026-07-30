import copy
import unittest

import numpy as np

from hoppertrex_mjlab.hybrid.controller_schedule import canonical_hash
from hoppertrex_mjlab.hybrid.innovation_detector import (
  EXPECTED_BINDINGS,
  OFFICIAL_IDENTIFICATION_PROTOCOL,
  GRID_SHA256,
  PREDICTOR_ARTIFACT_TYPE,
  fit_predictor_node,
  parse_innovation_predictor,
  signed_balance_channel,
  velocity_prbs,
)


class HybridInnovationDetectorTest(unittest.TestCase):
  def test_signed_balance_channel_uses_right_minus_left(self):
    np.testing.assert_array_equal(
      signed_balance_channel([[-2.0, 4.0], [3.0, -1.0]]),
      np.array([3.0, -2.0]),
    )

  def test_prbs_is_registered_pcg64_five_tick_binary_stream(self):
    stream = velocity_prbs(2, 7)
    self.assertEqual(stream.shape, (2750,))
    self.assertTrue(np.all(np.isin(stream, (0.0, 0.10))))
    np.testing.assert_array_equal(
      stream.reshape(-1, 5), np.repeat(stream[::5, None], 5, axis=1)
    )
    np.testing.assert_array_equal(stream, velocity_prbs(2, 7))
    self.assertFalse(np.array_equal(stream, velocity_prbs(2, 8)))

  def test_affine_fit_recovers_model_and_range_nrmse(self):
    rng = np.random.default_rng(4)
    z = rng.normal(size=(500, 2))
    u = rng.uniform(-2.0, 3.0, size=(500, 1))
    a = np.array([[0.8, 0.1], [-0.2, 0.6]])
    b = np.array([[0.4], [-0.3]])
    c = np.array([0.05, -0.07])
    y = z @ a.T + u @ b.T + c
    result = fit_predictor_node(z[:400], u[:400], y[:400], z[400:], u[400:], y[400:])
    self.assertEqual(result["regression_rank"], 4)
    np.testing.assert_allclose(result["a"], a, atol=1e-12)
    np.testing.assert_allclose(result["b"], b, atol=1e-12)
    np.testing.assert_allclose(result["c"], c, atol=1e-12)
    np.testing.assert_allclose(result["heldout_nrmse"], 0.0, atol=1e-14)

  def _payload(self):
    nodes = []
    for index in range(9):
      nodes.append({
        "node_index": index, "a": [[1.0, 0.0], [0.0, 1.0]],
        "b": [[1.0], [2.0]], "c": [0.0, 0.0],
        "regression_rank": 4, "heldout_nrmse": [0.01, 0.02],
        "fit_u_min_radps": -3.0 + 0.1 * index,
        "fit_u_max_radps": 3.0 + 0.1 * index,
      })
    payload = {
      "schema_version": 1, "artifact_type": PREDICTOR_ARTIFACT_TYPE,
      "probe": "hybrid_c2_predictor_identification_v1",
      "classification": "PREDICTOR_IDENTIFICATION_QUALIFIED",
      "state_names": ["imu_pitch_rate_radps", "signed_wheel_speed_radps"],
      "grid_sha256": GRID_SHA256,
      "height_nodes": [0.2907321708, 0.3092089487, 0.3276857266],
      "pitch_nodes": [-0.032, 0.0, 0.032], "nodes": nodes,
      "protocol": copy.deepcopy(OFFICIAL_IDENTIFICATION_PROTOCOL),
      "bindings": EXPECTED_BINDINGS,
      "evidence_eligible": False, "detector_fit_eligible": False,
      "promotion_eligible": False, "training_eligible": False,
      "checkpoint": None,
    }
    payload["predictor_hash"] = canonical_hash(payload, hash_field="predictor_hash")
    return payload

  def test_parser_interpolates_model_and_enforces_input_domain(self):
    predictor = parse_innovation_predictor(self._payload())
    np.testing.assert_allclose(
      predictor.predict([1.0, 2.0], 1.0, 0.3092089487, 0.0), [2.0, 4.0]
    )
    with self.assertRaisesRegex(ValueError, "outside the fitted domain"):
      predictor.predict([1.0, 2.0], 20.0, 0.3092089487, 0.0)

  def test_parser_rejects_hash_or_qualification_drift(self):
    for mutate in (
      lambda value: value.update(predictor_hash="0" * 64),
      lambda value: value["nodes"][0].update(regression_rank=3),
      lambda value: value["nodes"][0].update(heldout_nrmse=[0.16, 0.01]),
      lambda value: value["nodes"][0].update(heldout_nrmse=[float("nan"), 0.01]),
      lambda value: value.update(bindings={}),
      lambda value: value["bindings"].update(controller_schedule_hash="wrong"),
      lambda value: value["nodes"][0].update(a=[[1.0, 0.0]]),
      lambda value: value["protocol"].update(control_dt_s=9.0),
      lambda value: value["protocol"]["prbs"].update(seed_formula="wrong"),
      lambda value: value["protocol"].update(residual_action=[1.0] * 6),
      lambda value: value["protocol"].update(yaw_command=1.0),
    ):
      payload = copy.deepcopy(self._payload())
      mutate(payload)
      if payload["predictor_hash"] != "0" * 64:
        payload["predictor_hash"] = canonical_hash(payload, hash_field="predictor_hash")
      with self.assertRaises(ValueError):
        parse_innovation_predictor(payload)


if __name__ == "__main__":
  unittest.main()
