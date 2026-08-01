import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hoppertrex_mjlab.scripts import (
  probe_hybrid_c2_innovation_qualification as probe,
)
from hoppertrex_mjlab.scripts import (
  validate_hybrid_c2_innovation_qualification as validator,
)

ROOT = Path(__file__).resolve().parents[1]


class ValidateHybridC2InnovationQualificationTest(unittest.TestCase):
  def test_validator_replay_does_not_import_shared_candidate_or_selection_code(self):
    source = Path(validator.__file__).read_text(encoding="utf-8-sig")
    self.assertNotIn("evaluate_qualification_candidate", source)
    self.assertNotIn("select_qualification_candidate", source)
    self.assertNotIn("qualification_selection", source)
    self.assertIn("class _ReplayPredictor", source)
    self.assertIn("def _evaluate_candidate", source)

  def test_independent_predictor_accepts_only_registered_float32_roundoff(self):
    payload = json.loads((
      ROOT
      / "docs"
      / "experiments"
      / "artifacts"
      / "c2_innovation_predictor_2cccb36_seed1"
      / "c2_innovation_predictor.json"
    ).read_text(encoding="utf-8-sig"))
    predictor = validator._ReplayPredictor(payload)
    for height in validator.REPLAY_HEIGHT_NODES:
      for pitch in validator.REPLAY_PITCH_NODES:
        predicted = predictor.predict(
          np.zeros(2),
          0.0,
          float(np.float32(height)),
          float(np.float32(pitch)),
        )
        self.assertTrue(np.all(np.isfinite(predicted)))
    with self.assertRaises(validator._ReplayPostureDomainError):
      predictor.predict(
        np.zeros(2),
        0.0,
        validator.REPLAY_HEIGHT_NODES[1],
        validator.REPLAY_PITCH_NODES[-1] + 1.01 * validator.REPLAY_POSTURE_ATOL,
      )

  def test_validator_contract_covers_raw_signals_and_all_formal_outcomes(self):
    for required in (
      "flat_z",
      "stair_z",
      "flat_u",
      "stair_u",
      "flat_next_z",
      "stair_next_z",
      "flat_features",
      "stair_features",
      "flat_active",
      "stair_active",
      "stair_riser_contact",
      "impact_steps",
      "stair_contact_found",
      "stair_contact_force_contact_frame",
      "stair_contact_pos_global",
      "stair_contact_normal_global",
      "stair_outer_face_x",
      "stair_terrain_origin_x",
      "reset_perturbations",
      "flat_reset_relative",
      "stair_reset_relative",
      "flat_written_reset_relative",
      "stair_written_reset_relative",
      "flat_wheel_targets",
      "stair_wheel_targets",
      "flat_portable_targets",
      "stair_portable_targets",
      "flat_specific_force_x",
      "stair_specific_force_x",
      "flat_projected_gravity_x",
      "stair_projected_gravity_x",
    ):
      self.assertIn(required, validator.RAW_KEYS)
    self.assertEqual(
      validator.FORMAL_CLASSIFICATIONS,
      {
        "INNOVATION_DETECTOR_QUALIFIED",
        "C2_INNOVATION_DETECTOR_UNQUALIFIED_STOP",
        "INVALID_INNOVATION_CAPTURE",
      },
    )

  def test_validator_reproduces_the_same_seed3_reset_table_as_producer(self):
    for cell_index in range(18):
      np.testing.assert_array_equal(
        validator._expected_perturbations(cell_index),
        probe.reset_perturbations(cell_index, 16).numpy(),
      )

  def test_validator_recomputes_direct_innovation_without_redifferencing(self):
    shape = (500, 16)
    z = np.zeros((*shape, 2), dtype=np.float64)
    next_z = np.zeros_like(z)
    next_z[100:102, :, 0] = 2.0
    targets = np.zeros((*shape, 2), dtype=np.float64)
    features = np.zeros((*shape, 3), dtype=np.float64)
    features[:, :, :2] = np.abs(next_z)
    raw = {
      "flat_z": z,
      "flat_u": np.zeros((*shape, 1), dtype=np.float64),
      "flat_next_z": next_z,
      "flat_shaped_posture": np.broadcast_to(
        np.asarray([0.3092089487, 0.0]), (*shape, 2)
      ).copy(),
      "flat_features": features,
      "flat_active": np.ones(shape, dtype=np.bool_),
      "flat_wheel_targets": targets,
      "flat_portable_targets": targets.copy(),
      "flat_specific_force_x": np.zeros((*shape, 1), dtype=np.float64),
      "flat_projected_gravity_x": np.zeros((*shape, 1), dtype=np.float64),
    }
    predictor = SimpleNamespace(predict=lambda state, _u, _h, _p: state)
    health, replayed = validator._validate_side(
      raw,
      prefix="flat",
      predictor=predictor,
      cell={"height_m": 0.3092089487, "pitch_rad": 0.0},
    )
    self.assertTrue(all(value == 0 for key, value in health.items() if key.endswith("_count")))
    self.assertEqual(health["portable_max_abs_target_error_radps"], 0.0)
    np.testing.assert_array_equal(replayed[100:102, :, 0], 2.0)
    raw["flat_active"] = raw["flat_active"].astype(np.uint8)
    with self.assertRaisesRegex(ValueError, "mask is not full true"):
      validator._validate_side(
        raw,
        prefix="flat",
        predictor=predictor,
        cell={"height_m": 0.3092089487, "pitch_rad": 0.0},
      )

  def test_validator_replays_archived_deceleration_with_deployment_float32_math(self):
    shape = (500, 16)
    z = np.zeros((*shape, 2), dtype=np.float64)
    targets = np.zeros((*shape, 2), dtype=np.float64)
    specific_force_f32 = np.full(shape, np.float32(0.12345679), dtype=np.float32)
    gravity_f32 = np.full(shape, np.float32(-0.2345679), dtype=np.float32)
    expected_deceleration = np.maximum(
      np.float32(0.0),
      -(
        specific_force_f32
        + np.float32(validator.GRAVITY_MPS2) * gravity_f32
      ),
    ).astype(np.float64)
    float64_replay = np.maximum(
      0.0,
      -(
        specific_force_f32.astype(np.float64)
        + validator.GRAVITY_MPS2 * gravity_f32.astype(np.float64)
      ),
    )
    self.assertGreater(
      float(np.max(np.abs(expected_deceleration - float64_replay))),
      1.0e-12,
    )
    features = np.zeros((*shape, 3), dtype=np.float64)
    features[:, :, 2] = expected_deceleration
    raw = {
      "flat_z": z,
      "flat_u": np.zeros((*shape, 1), dtype=np.float64),
      "flat_next_z": z.copy(),
      "flat_shaped_posture": np.broadcast_to(
        np.asarray([0.3092089487, 0.0]), (*shape, 2)
      ).copy(),
      "flat_features": features,
      "flat_active": np.ones(shape, dtype=np.bool_),
      "flat_wheel_targets": targets,
      "flat_portable_targets": targets.copy(),
      "flat_specific_force_x": specific_force_f32.astype(np.float64)[..., None],
      "flat_projected_gravity_x": gravity_f32.astype(np.float64)[..., None],
    }
    predictor = SimpleNamespace(predict=lambda state, _u, _h, _p: state)
    health, replayed = validator._validate_side(
      raw,
      prefix="flat",
      predictor=predictor,
      cell={"height_m": 0.3092089487, "pitch_rad": 0.0},
    )
    self.assertTrue(
      all(value == 0 for key, value in health.items() if key.endswith("_count"))
    )
    np.testing.assert_array_equal(replayed[:, :, 2], expected_deceleration)

  def test_inputs_only_is_the_only_mode_without_output_and_sha_pins(self):
    parsed = validator.parse_args([
      "--predictor", "predictor.json",
      "--transition-floor", "floor.json",
      "--inputs-only",
    ])
    self.assertTrue(parsed.inputs_only)
    with self.assertRaises(SystemExit):
      validator.parse_args([
        "--predictor", "predictor.json",
        "--transition-floor", "floor.json",
      ])


if __name__ == "__main__":
  unittest.main()
