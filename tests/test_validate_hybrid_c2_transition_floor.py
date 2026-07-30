import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from hoppertrex_mjlab.scripts import validate_hybrid_c2_transition_floor as validator


class ValidateHybridC2TransitionFloorTest(unittest.TestCase):
  def test_validator_rejects_protocol_drift_before_consuming_raw_npz(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      floor = {
        "schema_version": 1,
        "artifact_type": "c2_innovation_transition_floor",
        "probe": "hybrid_c2_transition_floor_v1",
        "protocol": {},
      }
      (root / "c2_innovation_floor.json").write_text(json.dumps(floor))
      predictor = root / "predictor.json"
      predictor.write_text("{}")
      parsed = mock.Mock(predictor_hash="p" * 64)
      with mock.patch.object(validator, "parse_innovation_predictor", return_value=parsed):
        with self.assertRaisesRegex(ValueError, "protocol"):
          validator.validate_capture(root, predictor)

  def test_validator_raw_contract_covers_all_registered_arrays(self):
    self.assertEqual(
      validator.RAW_KEYS,
      (
        "z",
        "u",
        "next_z",
        "shaped_posture",
        "innovation",
        "accelerometer_specific_force_x",
        "projected_gravity_x",
        "forward_deceleration",
        "active",
        "raw_command",
      ),
    )

  def test_empty_voting_and_nonfinite_samples_are_archivable_invalid_data(self):
    features = np.ones((4, 2, 3), dtype=np.float64)
    maxima = validator._voting_maxima(features, np.zeros((4, 2), dtype=bool))
    self.assertTrue(np.isnan(maxima).all())
    actual = np.asarray([1.0, np.nan, 3.0])
    expected = np.asarray([1.0, 2.0, 3.0])
    self.assertTrue(validator._matches_or_nonfinite(actual, expected, 0.0))
    self.assertFalse(
      validator._matches_or_nonfinite(np.asarray([1.0, 9.0, 3.0]), expected, 0.0)
    )


if __name__ == "__main__":
  unittest.main()
