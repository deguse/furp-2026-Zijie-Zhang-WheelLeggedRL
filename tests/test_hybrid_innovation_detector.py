import copy
import unittest

import numpy as np

from hoppertrex_mjlab.hybrid.controller_schedule import canonical_hash
from hoppertrex_mjlab.hybrid.innovation_detector import (
  EXPECTED_BINDINGS,
  FEATURE_NAMES,
  GRID_SHA256,
  OFFICIAL_IDENTIFICATION_PROTOCOL,
  OFFICIAL_QUALIFICATION_PROTOCOL,
  OFFICIAL_TRANSITION_FLOOR_PROTOCOL,
  PREDICTOR_ARTIFACT_TYPE,
  PREDICTOR_POSTURE_DOMAIN_ATOL,
  QUALIFICATION_ARTIFACT_TYPE,
  QUALIFICATION_DRIVE_STEPS,
  QUALIFICATION_MAX_DELAY_TICKS,
  QUALIFICATION_PAIRS_PER_CELL,
  REGISTERED_HEIGHT_NODES,
  REGISTERED_PITCH_NODES,
  _first_trigger_tick,
  evaluate_qualification_candidate,
  fit_predictor_node,
  parse_innovation_detector_qualification,
  parse_innovation_predictor,
  parse_transition_floor,
  qualification_cells,
  qualification_selection,
  select_qualification_candidate,
  signed_balance_channel,
  threshold_table,
  threshold_table_hash,
  transition_floor_cells,
  velocity_prbs,
)


class HybridInnovationDetectorTest(unittest.TestCase):
  @staticmethod
  def _threshold_row(index=0, value=1.0):
    return {"index": index, **{name: value for name in FEATURE_NAMES}}

  @staticmethod
  def _qualification_evidence(timely_counts=None, *, delay=3):
    if timely_counts is None:
      timely_counts = [QUALIFICATION_PAIRS_PER_CELL] * len(qualification_cells())
    cells = []
    impact = 25
    for registered, timely_count in zip(
      qualification_cells(), timely_counts, strict=True
    ):
      flat = np.zeros(
        (QUALIFICATION_DRIVE_STEPS, QUALIFICATION_PAIRS_PER_CELL, 3),
        dtype=np.float64,
      )
      stair = np.zeros_like(flat)
      for slot in range(timely_count):
        trigger = impact + delay
        stair[trigger - 1 : trigger + 1, slot, :2] = 2.0
      cells.append({
        "cell": registered,
        "flat_features": flat,
        "stair_features": stair,
        "flat_active": np.ones(flat.shape[:2], dtype=np.bool_),
        "stair_active": np.ones(stair.shape[:2], dtype=np.bool_),
        "impact_steps": np.full(
          QUALIFICATION_PAIRS_PER_CELL, impact, dtype=np.int64
        ),
      })
    return cells

  @staticmethod
  def _candidate_for_row(row):
    cells = []
    for index in range(len(qualification_cells())):
      cells.append({
        "cell_index": index,
        "flat_trigger_count": 0,
        "stair_pre_impact_trigger_count": 0,
        "timely_detection_count": QUALIFICATION_PAIRS_PER_CELL,
        "late_detection_count": 0,
        "missing_detection_count": 0,
        "timely_detection_rate": 1.0,
        "timely_delays_ticks": [1] * QUALIFICATION_PAIRS_PER_CELL,
      })
    return {
      "threshold_table_index": row["index"],
      "thresholds": {name: row[name] for name in FEATURE_NAMES},
      "qualified": True,
      "flat_trigger_count": 0,
      "stair_pre_impact_trigger_count": 0,
      "timely_detection_count": 288,
      "timely_detection_rate": 1.0,
      "late_detection_count": 0,
      "missing_detection_count": 0,
      "mean_timely_delay_ticks": 1.0,
      "timely_delays_ticks": [1] * 288,
      "cells": cells,
    }

  def test_threshold_table_is_pitch_then_wheel_then_deceleration_major(self):
    rows = threshold_table([2.0, 4.0, 8.0])
    self.assertEqual(len(rows), 125)
    self.assertEqual(rows[0]["index"], 0)
    self.assertEqual(rows[0]["pitch_rate_innovation_radps"], 2.1)
    self.assertEqual(rows[1]["forward_deceleration_mps2"], 10.0)
    self.assertEqual(rows[5]["wheel_speed_innovation_radps"], 5.0)
    self.assertEqual(rows[25]["pitch_rate_innovation_radps"], 2.5)
    self.assertEqual(len(threshold_table_hash(rows)), 64)
    self.assertEqual(len(transition_floor_cells()), 10)

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

  def test_qualification_requires_native_boolean_activation_masks(self):
    evidence = self._qualification_evidence()
    evidence[0]["flat_active"] = evidence[0]["flat_active"].astype(np.uint8)
    with self.assertRaisesRegex(ValueError, "Boolean dtype"):
      evaluate_qualification_candidate(self._threshold_row(), evidence)

    evidence = self._qualification_evidence()
    evidence[0]["stair_active"] = evidence[0]["stair_active"].tolist()
    with self.assertRaisesRegex(ValueError, "Boolean dtype"):
      evaluate_qualification_candidate(self._threshold_row(), evidence)

  def test_qualification_requires_finite_nonnegative_features(self):
    for value in (float("nan"), -1.0):
      evidence = self._qualification_evidence()
      evidence[0]["stair_features"][100, 0, 0] = value
      with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
        evaluate_qualification_candidate(self._threshold_row(), evidence)

  def test_delay_three_is_timely_and_delay_four_is_late(self):
    impact = 25
    features = np.zeros((40, 3), dtype=np.float64)
    active = np.ones(40, dtype=np.bool_)
    thresholds = np.ones(3, dtype=np.float64)
    features[impact + 2 : impact + 4, :2] = 2.0
    trigger = _first_trigger_tick(features, active, thresholds)
    self.assertEqual(trigger - impact, 3)
    self.assertLessEqual(trigger - impact, QUALIFICATION_MAX_DELAY_TICKS)

    features[:] = 0.0
    features[impact + 3 : impact + 5, :2] = 2.0
    trigger = _first_trigger_tick(features, active, thresholds)
    self.assertEqual(trigger - impact, 4)
    self.assertGreater(trigger - impact, QUALIFICATION_MAX_DELAY_TICKS)

  def test_overall_and_per_cell_timely_boundaries_are_exact(self):
    pass_counts = [15] * 14 + [16] * 4
    passed = evaluate_qualification_candidate(
      self._threshold_row(), self._qualification_evidence(pass_counts)
    )
    self.assertEqual(passed["timely_detection_count"], 274)
    self.assertTrue(passed["qualified"])
    self.assertEqual(
      passed["stair_pre_impact_trigger_count"]
      + passed["timely_detection_count"]
      + passed["late_detection_count"]
      + passed["missing_detection_count"],
      288,
    )
    for cell in passed["cells"]:
      self.assertEqual(
        cell["stair_pre_impact_trigger_count"]
        + cell["timely_detection_count"]
        + cell["late_detection_count"]
        + cell["missing_detection_count"],
        QUALIFICATION_PAIRS_PER_CELL,
      )

    overall_fail_counts = [15] * 15 + [16] * 3
    overall_failed = evaluate_qualification_candidate(
      self._threshold_row(), self._qualification_evidence(overall_fail_counts)
    )
    self.assertEqual(overall_failed["timely_detection_count"], 273)
    self.assertFalse(overall_failed["qualified"])

    cell_fail_counts = [14] + [16] * 17
    cell_failed = evaluate_qualification_candidate(
      self._threshold_row(), self._qualification_evidence(cell_fail_counts)
    )
    self.assertEqual(cell_failed["timely_detection_count"], 286)
    self.assertEqual(cell_failed["cells"][0]["timely_detection_count"], 14)
    self.assertFalse(cell_failed["qualified"])

  def test_first_preimpact_trigger_cannot_be_redeemed_by_later_trigger(self):
    impact = 25
    features = np.zeros((40, 3), dtype=np.float64)
    features[impact - 2 : impact, :2] = 2.0
    features[impact : impact + 2, :2] = 2.0
    trigger = _first_trigger_tick(
      features, np.ones(40, dtype=np.bool_), np.ones(3, dtype=np.float64)
    )
    self.assertEqual(trigger, impact - 1)

  def test_pitch_feature_is_voted_as_direct_innovation_without_redifferencing(self):
    impact = 25
    features = np.zeros((40, 3), dtype=np.float64)
    # A constant two-tick innovation plateau must trigger. Re-differencing the
    # pitch innovation would erase the second vote and miss this attempt.
    features[impact - 1 : impact + 1, 0] = 2.0
    features[impact - 1 : impact + 1, 1] = 2.0
    trigger = _first_trigger_tick(
      features, np.ones(40, dtype=np.bool_), np.ones(3, dtype=np.float64)
    )
    self.assertEqual(trigger, impact)

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

  def test_predictor_clamps_only_float32_posture_boundary_roundoff(self):
    predictor = parse_innovation_predictor(self._payload())
    for height in REGISTERED_HEIGHT_NODES:
      for pitch in REGISTERED_PITCH_NODES:
        expected = predictor.predict([1.0, 2.0], 1.0, height, pitch)
        observed = predictor.predict(
          [1.0, 2.0],
          1.0,
          float(np.float32(height)),
          float(np.float32(pitch)),
        )
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)
    with self.assertRaisesRegex(ValueError, "height.*outside"):
      predictor.predict(
        [1.0, 2.0],
        1.0,
        REGISTERED_HEIGHT_NODES[0] - 1.01 * PREDICTOR_POSTURE_DOMAIN_ATOL,
        0.0,
      )
    with self.assertRaisesRegex(ValueError, "pitch.*outside"):
      predictor.predict(
        [1.0, 2.0],
        1.0,
        REGISTERED_HEIGHT_NODES[1],
        REGISTERED_PITCH_NODES[-1] + 1.01 * PREDICTOR_POSTURE_DOMAIN_ATOL,
      )

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

  def _floor_payload(self):
    maxima = {
      "forward_deceleration_mps2": 3.0,
      "pitch_rate_innovation_radps": 1.0,
      "wheel_speed_innovation_radps": 2.0,
    }
    cells = []
    for index, registered in enumerate(transition_floor_cells()):
      cells.append({
        "cell_index": index,
        "name": registered["name"],
        "kind": registered["kind"],
        "target": registered["target"],
        "raw_file": f"cell_{index:02d}.npz",
        "raw_sha256": f"{index:064x}",
        "raw_shape": [500, 16],
        "active_voting_ticks": 160,
        "active_voting_ticks_per_env": [10] * 16,
        "feature_maxima": dict(maxima),
        "domain_violation_count": 0,
        "termination_count": 0,
        "timeout_count": 0,
        "non_wheel_contact_count": 0,
      })
    ordered_maxima = [maxima[name] for name in (
      "pitch_rate_innovation_radps",
      "wheel_speed_innovation_radps",
      "forward_deceleration_mps2",
    )]
    table = threshold_table(ordered_maxima)
    payload = {
      "schema_version": 1,
      "artifact_type": "c2_innovation_transition_floor",
      "probe": "hybrid_c2_transition_floor_v1",
      "classification": "INNOVATION_FLOOR_QUALIFIED",
      "predictor_hash": "p" * 64,
      "bindings": copy.deepcopy(EXPECTED_BINDINGS),
      "protocol": copy.deepcopy(OFFICIAL_TRANSITION_FLOOR_PROTOCOL),
      "cells": cells,
      "pooled_feature_maxima": maxima,
      "threshold_table": table,
      "threshold_table_hash": threshold_table_hash(table),
      "evidence_eligible": False,
      "detector_fit_eligible": False,
      "promotion_eligible": False,
      "training_eligible": False,
      "checkpoint": None,
    }
    payload["floor_hash"] = canonical_hash(payload, hash_field="floor_hash")
    return payload

  def _qualification_payload(self):
    floor = self._floor_payload()
    candidates = [
      self._candidate_for_row(row) for row in floor["threshold_table"]
    ]
    selected = select_qualification_candidate(candidates)
    result_cells = []
    for index, registered in enumerate(qualification_cells()):
      impact = 100
      result_cells.append({
        "cell": registered,
        "raw_file": f"cell_{index:02d}.npz",
        "raw_sha256": f"{index:064x}",
        "raw_shape": [QUALIFICATION_DRIVE_STEPS, QUALIFICATION_PAIRS_PER_CELL],
        "impact_steps": [impact] * QUALIFICATION_PAIRS_PER_CELL,
        "diagnostic_windows": [
          {
            "slot": slot,
            "start_tick": impact - 25,
            "impact_tick": impact,
            "end_tick": impact + 75,
          }
          for slot in range(QUALIFICATION_PAIRS_PER_CELL)
        ],
        "paired_reset_max_abs_error": 0.0,
        "written_reset_max_abs_error": 0.0,
        "written_paired_reset_max_abs_error": 0.0,
        "root_pitch_max_abs_error_rad": 0.0,
        "root_roll_yaw_max_abs_rad": 0.0,
        "other_root_velocity_max_abs": 0.0,
        "portable_max_abs_target_error_radps": 0.0,
        "health": {
          "flat_termination_count": 0,
          "stair_termination_count": 0,
          "flat_timeout_count": 0,
          "stair_timeout_count": 0,
          "flat_non_wheel_contact_count": 0,
          "stair_non_wheel_contact_count": 0,
          "settle_riser_contact_count": 0,
          "drive_start_past_face_count": 0,
          "missing_impact_count": 0,
          "invalid_window_count": 0,
          "predictor_domain_violation_count": 0,
          "posture_violation_count": 0,
          "predictor_evaluation_error_count": 0,
          "nonfinite_sample_count": 0,
          "negative_feature_sample_count": 0,
          "portable_target_violation_count": 0,
          "outer_face_binding_violation_count": 0,
        },
      })
    payload = {
      "schema_version": 1,
      "artifact_type": QUALIFICATION_ARTIFACT_TYPE,
      "probe": OFFICIAL_QUALIFICATION_PROTOCOL["probe"],
      "classification": "INNOVATION_DETECTOR_QUALIFIED",
      "git_sha": "a" * 40,
      "mjlab_git_sha": "b" * 40,
      "predictor_hash": "p" * 64,
      "floor_hash": floor["floor_hash"],
      "threshold_table_hash": floor["threshold_table_hash"],
      "bindings": copy.deepcopy(EXPECTED_BINDINGS),
      "protocol": copy.deepcopy(OFFICIAL_QUALIFICATION_PROTOCOL),
      "cells": result_cells,
      "candidates": candidates,
      "selected_candidate": qualification_selection(selected),
      "completed_cell_count": 18,
      "completed_pair_count": 288,
      "completed_candidate_count": len(candidates),
      "qualified_candidate_count": len(candidates),
      "evidence_eligible": True,
      "promotion_eligible": False,
      "training_eligible": False,
      "checkpoint": None,
      "next_step": "FREEZE_AND_INDEPENDENT_AUDIT_BEFORE_C3",
    }
    payload["detector_hash"] = canonical_hash(payload, hash_field="detector_hash")
    return payload, floor

  def test_qualification_parser_rejects_candidate_count_or_selection_drift(self):
    valid, floor = self._qualification_payload()
    parse_innovation_detector_qualification(
      valid, predictor_hash="p" * 64, floor_payload=floor
    )
    mutations = (
      lambda payload: payload["candidates"][0].update(
        timely_detection_count=287
      ),
      lambda payload: payload["candidates"][0].update(
        late_detection_count=1
      ),
      lambda payload: payload["candidates"][0]["cells"][0].update(
        timely_detection_count=15
      ),
      lambda payload: payload["cells"][0]["health"].update(
        flat_termination_count=1
      ),
      lambda payload: payload["cells"][0]["diagnostic_windows"][0].update(
        end_tick=999
      ),
      lambda payload: payload.update(selected_candidate=None),
      lambda payload: payload.update(
        classification="C2_INNOVATION_DETECTOR_UNQUALIFIED_STOP"
      ),
    )
    for mutate in mutations:
      payload, floor = self._qualification_payload()
      mutate(payload)
      payload["detector_hash"] = canonical_hash(payload, hash_field="detector_hash")
      with self.assertRaises(ValueError):
        parse_innovation_detector_qualification(
          payload, predictor_hash="p" * 64, floor_payload=floor
        )

  def test_floor_parser_accepts_sorted_maxima_keys(self):
    payload = self._floor_payload()
    payload = dict(sorted(payload.items()))
    payload["pooled_feature_maxima"] = dict(
      sorted(payload["pooled_feature_maxima"].items())
    )
    parse_transition_floor(payload, predictor_hash="p" * 64)

  def test_floor_parser_rejects_missing_attempt_votes(self):
    payload = self._floor_payload()
    payload["cells"][0]["active_voting_ticks_per_env"][3] = 0
    payload["floor_hash"] = canonical_hash(payload, hash_field="floor_hash")
    with self.assertRaisesRegex(ValueError, "cell"):
      parse_transition_floor(payload, predictor_hash="p" * 64)

  def test_floor_parser_rejects_deployment_binding_drift(self):
    payload = self._floor_payload()
    payload["bindings"]["controller_schedule_hash"] = "0" * 64
    payload["floor_hash"] = canonical_hash(payload, hash_field="floor_hash")
    with self.assertRaisesRegex(ValueError, "bindings"):
      parse_transition_floor(payload, predictor_hash="p" * 64)


if __name__ == "__main__":
  unittest.main()
