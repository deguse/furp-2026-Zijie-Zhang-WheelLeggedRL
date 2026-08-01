import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from mjlab.utils.lab_api.math import euler_xyz_from_quat

from hoppertrex_mjlab.hybrid.controller_schedule import canonical_hash
from hoppertrex_mjlab.hybrid.innovation_detector import (
  EXPECTED_BINDINGS,
  FEATURE_NAMES,
  OFFICIAL_QUALIFICATION_PROTOCOL,
  QUALIFICATION_ARTIFACT_TYPE,
  RESET_PERTURBATION_BOUNDS,
  evaluate_qualification_candidate,
  qualification_cells,
  qualification_selection,
  select_qualification_candidate,
)
from hoppertrex_mjlab.scripts import (
  validate_hybrid_c2_innovation_qualification as validator,
)


class _ZeroPredictor:
  def predict(self, _z, _u, _height, _pitch):
    return np.zeros(2, dtype=np.float64)


def _reset_arrays(cell):
  generator = torch.Generator(device="cpu")
  generator.manual_seed(30_000 + int(cell["cell_index"]))
  unit = 2.0 * torch.rand((16, 4), generator=generator) - 1.0
  bounds = torch.tensor(RESET_PERTURBATION_BOUNDS, dtype=torch.float32)
  perturbations = (unit * bounds).numpy()
  canonical = np.zeros((16, 13), dtype=np.float32)
  canonical[:, 0] = -0.25 + perturbations[:, 0]
  canonical[:, 1] = perturbations[:, 1]
  canonical[:, 2] = float(cell["height_m"])
  half = 0.5 * float(cell["pitch_rad"])
  canonical[:, 3:7] = np.asarray(
    [math.cos(half), 0.0, math.sin(half), 0.0], dtype=np.float32
  )
  canonical[:, 7] = perturbations[:, 2]
  canonical[:, 11] = perturbations[:, 3]
  quaternion = torch.from_numpy(canonical[:, 3:7])
  roll, pitch, yaw = euler_xyz_from_quat(quaternion)
  producer_metrics = {
    "paired_reset_max_abs_error": 0.0,
    "written_reset_max_abs_error": 0.0,
    "written_paired_reset_max_abs_error": 0.0,
    "root_pitch_max_abs_error_rad": float(
      torch.max(torch.abs(pitch - float(cell["pitch_rad"]))).item()
    ),
    "root_roll_yaw_max_abs_rad": float(
      torch.max(torch.abs(torch.stack((roll, yaw), dim=1))).item()
    ),
    "other_root_velocity_max_abs": 0.0,
  }
  return perturbations, canonical, producer_metrics


def _raw_cell(cell):
  steps, slots, impact = 500, 16, 25
  z = np.zeros((steps, slots, 2), dtype=np.float64)
  next_z = np.zeros_like(z)
  stair_next_z = next_z.copy()
  stair_next_z[impact - 1 : impact + 1] = 2.0
  flat_features = np.zeros((steps, slots, 3), dtype=np.float64)
  stair_features = flat_features.copy()
  stair_features[:, :, :2] = np.abs(stair_next_z)
  posture = np.broadcast_to(
    np.asarray([cell["height_m"], cell["pitch_rad"]], dtype=np.float64),
    (steps, slots, 2),
  ).copy()
  targets = np.zeros((steps, slots, 2), dtype=np.float64)
  scalar = np.zeros((steps, slots, 1), dtype=np.float64)
  active = np.ones((steps, slots), dtype=np.bool_)
  riser = np.zeros((steps, slots), dtype=np.bool_)
  riser[impact] = True
  impacts = np.full(slots, impact, dtype=np.int64)
  contact_found = np.zeros((steps, slots, 1), dtype=np.float32)
  contact_force = np.zeros((steps, slots, 1, 3), dtype=np.float32)
  contact_position = np.zeros((steps, slots, 1, 3), dtype=np.float32)
  contact_normal = np.zeros((steps, slots, 1, 3), dtype=np.float32)
  outer_face_x = np.linspace(1.0, 2.5, slots, dtype=np.float32)
  contact_found[impact, :, 0] = 1.0
  contact_found[0, :, 0] = 2.0
  contact_force[impact, :, 0, 0] = 1.0
  contact_position[impact, :, 0, 0] = outer_face_x
  contact_normal[impact, :, 0, 0] = 0.25
  perturbations, reset, producer_metrics = _reset_arrays(cell)
  false = np.zeros(slots, dtype=np.bool_)
  raw = {
    "flat_z": z.copy(), "stair_z": z.copy(),
    "flat_u": scalar.copy(), "stair_u": scalar.copy(),
    "flat_next_z": next_z.copy(), "stair_next_z": stair_next_z,
    "flat_shaped_posture": posture.copy(), "stair_shaped_posture": posture.copy(),
    "flat_features": flat_features, "stair_features": stair_features,
    "flat_active": active.copy(), "stair_active": active.copy(),
    "stair_riser_contact": riser, "impact_steps": impacts,
    "stair_contact_found": contact_found,
    "stair_contact_force_contact_frame": contact_force,
    "stair_contact_pos_global": contact_position,
    "stair_contact_normal_global": contact_normal,
    "stair_outer_face_x": outer_face_x,
    "stair_terrain_origin_x": outer_face_x + np.float32(3.0),
    "reset_perturbations": perturbations,
    "flat_reset_relative": reset.copy(), "stair_reset_relative": reset.copy(),
    "flat_written_reset_relative": reset.copy(),
    "stair_written_reset_relative": reset.copy(),
    "flat_wheel_targets": targets.copy(), "stair_wheel_targets": targets.copy(),
    "flat_portable_targets": targets.copy(), "stair_portable_targets": targets.copy(),
    "flat_specific_force_x": scalar.copy(), "stair_specific_force_x": scalar.copy(),
    "flat_projected_gravity_x": scalar.copy(),
    "stair_projected_gravity_x": scalar.copy(),
    "flat_terminated": false.copy(), "stair_terminated": false.copy(),
    "flat_timeout": false.copy(), "stair_timeout": false.copy(),
    "flat_non_wheel_contact": false.copy(),
    "stair_non_wheel_contact": false.copy(),
    "flat_settle_riser_contact": false.copy(),
    "stair_settle_riser_contact": false.copy(),
    "flat_drive_start_past_face": false.copy(),
    "stair_drive_start_past_face": false.copy(),
  }
  return raw, producer_metrics


class ValidateHybridC2InnovationReplayTest(unittest.TestCase):
  GIT_SHA = "a" * 40
  MJLAB_SHA = "b" * 40

  @classmethod
  def setUpClass(cls):
    cls.temporary = tempfile.TemporaryDirectory()
    cls.root = Path(cls.temporary.name)
    cls.predictor = _ZeroPredictor()
    cls.row = {"index": 0, **{name: 1.0 for name in FEATURE_NAMES}}
    cls.floor = {"threshold_table": [cls.row]}
    summaries = []
    evaluation = []
    for cell in qualification_cells():
      index = int(cell["cell_index"])
      raw, reset_metrics = _raw_cell(cell)
      path = cls.root / f"cell_{index:02d}.npz"
      np.savez(path, **raw)
      impacts = raw["impact_steps"]
      summaries.append({
        "cell": cell,
        "raw_file": path.name,
        "raw_sha256": validator._sha256(path),
        "raw_shape": [500, 16],
        "impact_steps": impacts.tolist(),
        "diagnostic_windows": [
          {
            "slot": slot,
            "start_tick": int(impact - 25),
            "impact_tick": int(impact),
            "end_tick": int(impact + 75),
          }
          for slot, impact in enumerate(impacts)
        ],
        **reset_metrics,
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
      evaluation.append({
        "cell": cell,
        "flat_features": raw["flat_features"],
        "stair_features": raw["stair_features"],
        "flat_active": raw["flat_active"],
        "stair_active": raw["stair_active"],
        "impact_steps": impacts,
      })
    candidates = [evaluate_qualification_candidate(cls.row, evaluation)]
    selected = select_qualification_candidate(candidates)
    cls.base_payload = {
      "schema_version": 1,
      "artifact_type": QUALIFICATION_ARTIFACT_TYPE,
      "probe": OFFICIAL_QUALIFICATION_PROTOCOL["probe"],
      "classification": "INNOVATION_DETECTOR_QUALIFIED",
      "git_sha": cls.GIT_SHA,
      "mjlab_git_sha": cls.MJLAB_SHA,
      "predictor_hash": validator.PREDICTOR_HASH,
      "floor_hash": validator.FLOOR_HASH,
      "threshold_table_hash": validator.THRESHOLD_TABLE_HASH,
      "bindings": copy.deepcopy(EXPECTED_BINDINGS),
      "protocol": copy.deepcopy(OFFICIAL_QUALIFICATION_PROTOCOL),
      "cells": summaries,
      "completed_cell_count": 18,
      "completed_pair_count": 288,
      "completed_candidate_count": 1,
      "qualified_candidate_count": 1,
      "candidates": candidates,
      "selected_candidate": qualification_selection(selected),
      "evidence_eligible": True,
      "promotion_eligible": False,
      "training_eligible": False,
      "checkpoint": None,
      "next_step": "FREEZE_AND_INDEPENDENT_AUDIT_BEFORE_C3",
    }
    cls.base_payload["detector_hash"] = canonical_hash(
      cls.base_payload, hash_field="detector_hash"
    )
    cls.base_cell0 = (cls.root / "cell_00.npz").read_bytes()

  @classmethod
  def tearDownClass(cls):
    cls.temporary.cleanup()

  def setUp(self):
    self._restore()

  def _restore(self):
    (self.root / "cell_00.npz").write_bytes(self.base_cell0)
    self._write_payload(copy.deepcopy(self.base_payload))

  def _write_payload(self, payload):
    payload["detector_hash"] = canonical_hash(payload, hash_field="detector_hash")
    (self.root / "c2_innovation_detector_qualification.json").write_text(
      json.dumps(payload, allow_nan=False), encoding="utf-8"
    )

  def _payload(self):
    return json.loads(
      (self.root / "c2_innovation_detector_qualification.json").read_text(
        encoding="utf-8"
      )
    )

  def _raw0(self):
    with np.load(self.root / "cell_00.npz", allow_pickle=False) as loaded:
      return {name: loaded[name] for name in loaded.files}

  def _write_raw0(self, raw, payload):
    path = self.root / "cell_00.npz"
    np.savez(path, **raw)
    payload["cells"][0]["raw_sha256"] = validator._sha256(path)
    self._write_payload(payload)

  def _validate(self):
    with mock.patch.object(
      validator, "parse_innovation_detector_qualification"
    ) as parser:
      result = validator.validate_output(
        self.root,
        predictor=self.predictor,
        floor_payload={},
        floor=self.floor,
        expected_git_sha=self.GIT_SHA,
        expected_mjlab_git_sha=self.MJLAB_SHA,
      )
    parser.assert_called_once()
    return result

  @staticmethod
  def _mark_invalid(payload):
    payload.update(
      classification="INVALID_INNOVATION_CAPTURE",
      completed_candidate_count=0,
      qualified_candidate_count=0,
      candidates=[],
      selected_candidate=None,
      evidence_eligible=False,
      next_step="INDEPENDENT_IMPLEMENTATION_DIAGNOSIS_ONLY",
    )

  def test_full_raw_replay_accepts_all_three_registered_pitch_resets(self):
    payload = self._validate()
    self.assertEqual(payload["classification"], "INNOVATION_DETECTOR_QUALIFIED")
    self.assertEqual(
      {cell["cell"]["pitch_rad"] for cell in payload["cells"]},
      {-0.032, 0.0, 0.032},
    )

  def test_rejects_raw_hash_field_shape_mask_and_feature_tamper(self):
    payload = self._payload()
    payload["cells"][0]["raw_sha256"] = "0" * 64
    self._write_payload(payload)
    with self.assertRaisesRegex(ValueError, "raw hash"):
      self._validate()

    mutations = (
      ("field", lambda raw: raw.pop("flat_z"), "field set"),
      ("shape", lambda raw: raw.update(flat_z=raw["flat_z"][:-1]), "shape drifted"),
      (
        "mask",
        lambda raw: raw.update(flat_active=raw["flat_active"].astype(np.uint8)),
        "mask is not full true",
      ),
      (
        "feature",
        lambda raw: raw["flat_features"].__setitem__((0, 0, 0), 0.5),
        "features do not reproduce",
      ),
      (
        "nonbinary_contact_found",
        lambda raw: raw["stair_contact_found"].__setitem__(
          (0, 0, 0), np.float32(0.5)
        ),
        "raw contact values",
      ),
    )
    for name, mutate, message in mutations:
      with self.subTest(name=name):
        self._restore()
        payload = self._payload()
        raw = self._raw0()
        mutate(raw)
        self._write_raw0(raw, payload)
        with self.assertRaisesRegex(ValueError, message):
          self._validate()

  def test_rejects_impact_reset_and_summary_tamper_after_rehash(self):
    mutations = (
      (
        "impact",
        lambda raw, payload: (
          raw["impact_steps"].__setitem__(0, 26),
          payload["cells"][0]["impact_steps"].__setitem__(0, 26),
        ),
        "First-riser impact",
      ),
      (
        "reset",
        lambda raw, _payload: raw["reset_perturbations"].__setitem__(
          (0, 0), raw["reset_perturbations"][0, 0] + np.float32(0.001)
        ),
        "frozen CPU generator",
      ),
    )
    for name, mutate, message in mutations:
      with self.subTest(name=name):
        self._restore()
        payload = self._payload()
        raw = self._raw0()
        mutate(raw, payload)
        self._write_raw0(raw, payload)
        with self.assertRaisesRegex(ValueError, message):
          self._validate()

    for field in (
      "written_reset_max_abs_error",
      "root_pitch_max_abs_error_rad",
      "portable_max_abs_target_error_radps",
    ):
      with self.subTest(summary_field=field):
        self._restore()
        payload = self._payload()
        payload["cells"][0][field] = 1.0e-6
        self._write_payload(payload)
        with self.assertRaisesRegex(ValueError, "summary drifted|maximum summary"):
          self._validate()

  def test_rejects_raw_contact_tamper_with_derived_truth_left_unchanged(self):
    payload = self._payload()
    raw = self._raw0()
    raw["stair_contact_force_contact_frame"][25, 0, 0, 0] = 0.99
    self._write_raw0(raw, payload)
    with self.assertRaisesRegex(ValueError, "riser mask"):
      self._validate()

  def test_rejects_outer_face_translation_with_contact_truth_preserved(self):
    payload = self._payload()
    raw = self._raw0()
    shift = np.float32(0.1)
    raw["stair_outer_face_x"][0] += shift
    raw["stair_contact_pos_global"][:, 0, :, 0] += shift
    self._write_raw0(raw, payload)
    with self.assertRaisesRegex(ValueError, "health summary drifted"):
      self._validate()

  def test_nonfinite_contact_is_archived_as_invalid_capture(self):
    payload = self._payload()
    raw = self._raw0()
    raw["stair_contact_force_contact_frame"][0, 0, 0, 0] = np.nan
    payload["cells"][0]["health"]["nonfinite_sample_count"] = 1
    self._mark_invalid(payload)
    self._write_raw0(raw, payload)
    with mock.patch.object(
      validator, "parse_innovation_detector_qualification"
    ) as parser:
      result = validator.validate_output(
        self.root,
        predictor=self.predictor,
        floor_payload={},
        floor=self.floor,
        expected_git_sha=self.GIT_SHA,
        expected_mjlab_git_sha=self.MJLAB_SHA,
      )
    parser.assert_not_called()
    self.assertEqual(result["classification"], "INVALID_INNOVATION_CAPTURE")

  def test_valid_raw_cannot_be_forged_invalid_with_summary_only(self):
    payload = self._payload()
    payload["cells"][0]["health"]["predictor_domain_violation_count"] = 1
    self._mark_invalid(payload)
    self._write_payload(payload)
    with self.assertRaisesRegex(ValueError, "health summary drifted"):
      validator.validate_output(
        self.root,
        predictor=self.predictor,
        floor_payload={},
        floor=self.floor,
        expected_git_sha=self.GIT_SHA,
        expected_mjlab_git_sha=self.MJLAB_SHA,
      )

  def test_rejects_candidate_classification_and_selection_raw_drift(self):
    mutations = (
      lambda payload: payload["candidates"][0].update(timely_detection_count=287),
      lambda payload: payload.update(
        classification="C2_INNOVATION_DETECTOR_UNQUALIFIED_STOP",
        next_step="STOP_FOR_USER_ROUTE_DECISION",
      ),
      lambda payload: payload.update(selected_candidate=None),
    )
    for mutate in mutations:
      self._restore()
      payload = self._payload()
      mutate(payload)
      self._write_payload(payload)
      with self.assertRaises(ValueError):
        self._validate()


if __name__ == "__main__":
  unittest.main()
