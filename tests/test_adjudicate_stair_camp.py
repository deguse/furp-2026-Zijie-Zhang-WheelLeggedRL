from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hoppertrex_mjlab.hybrid.stair_residual import (
  residual_promotion_decision as frozen_residual_promotion_decision,
)
from hoppertrex_mjlab.scripts.rsl_rl import adjudicate_stair_camp as adjudicator


def _row(height: float, *, passed: bool) -> dict[str, float | int]:
  return {
    "height_m": height,
    "success_rate": 44.0 / 48.0 if passed else 43.0 / 48.0,
    "terminations": 0,
    "non_wheel_contacts": 0,
    "trials": 48,
  }


def _rows(*, passing_prefix: int) -> list[dict[str, float | int]]:
  return [
    _row(height, passed=index < passing_prefix)
    for index, height in enumerate(adjudicator.REGISTERED_HEIGHT_GRID_M)
  ]


def _envelope(seed: int) -> dict[str, object]:
  return {
    "training_seed": seed,
    "evaluation_seed": 1,
    "budget_iterations": 1000,
    "git_sha": "0123456789abcdef0123456789abcdef01234567",
    "contract_hash": adjudicator.STAIR_CAMP_CANONICAL_CONTRACT_SHA256,
    "artifact_bindings": {
      "controller_gain_hash": "b" * 64,
      "calibration_hash": "c" * 64,
      "yaw_calibration_hash": "d" * 64,
      "posture_map_hash": "e" * 64,
      "posture_artifact_hash": "f" * 64,
      "station_calibration_hash": "0" * 64,
    },
    # Classical fails the first stair tier; the registered synthetic flat row
    # supplies its contiguous height-zero boundary.
    "classical_rows": _rows(passing_prefix=0),
    "residual_rows": _rows(passing_prefix=1),
    "flat_gate_passed": True,
    "standing_gate_passed": True,
    "velocity_gate_passed": True,
    "stage5_gate_passed": True,
    "gate_stair_mode_false_positives": {
      gate: 0 for gate in adjudicator.GATE_NAMES
    },
    "completed_ablations": list(adjudicator.REQUIRED_ABLATIONS),
    "ablations_complete": True,
    "evidence_eligible": True,
    "checkpoint": f"seed-{seed}/model_999.pt",
    "checkpoint_file_sha256": str(seed) * 64,
  }


def _envelopes() -> list[dict[str, object]]:
  return [_envelope(seed) for seed in (1, 2, 3)]


class StairCampAdjudicatorSuccessTest(unittest.TestCase):
  def test_promotes_only_after_one_frozen_decision_call_per_seed(self) -> None:
    envelopes = list(reversed(_envelopes()))
    original = copy.deepcopy(envelopes)

    with patch.object(
      adjudicator,
      "residual_promotion_decision",
      wraps=frozen_residual_promotion_decision,
    ) as decision:
      result = adjudicator.adjudicate_stair_camp(envelopes)

    self.assertEqual(decision.call_count, 3)
    self.assertEqual(envelopes, original, "adjudication must not mutate evidence")
    self.assertEqual(
      result["classification"], adjudicator.PROMOTION_CLASSIFICATION
    )
    self.assertTrue(result["promotion_eligible"])
    self.assertEqual(result["training_seeds"], [1, 2, 3])
    self.assertEqual(result["minimum_boundary_extension_m"], 0.01)
    self.assertTrue(result["minimum_boundary_extension_passed"])

    for call in decision.call_args_list:
      classical = call.kwargs["classical_rows"]
      residual = call.kwargs["residual_rows"]
      self.assertEqual(classical[0], adjudicator.SYNTHETIC_FLAT_ROW)
      self.assertEqual(residual[0], adjudicator.SYNTHETIC_FLAT_ROW)
      self.assertIsNot(classical[0], residual[0])
      self.assertEqual(len(classical), 8)
      self.assertEqual(len(residual), 8)
      self.assertEqual(
        [row["height_m"] for row in classical[1:]],
        list(adjudicator.REGISTERED_HEIGHT_GRID_M),
      )

    self.assertEqual(
      [row["training_seed"] for row in result["seed_results"]],
      [1, 2, 3],
    )
    for seed_result in result["seed_results"]:
      self.assertEqual(
        seed_result["decision"]["classification"],
        adjudicator.PROMOTION_CLASSIFICATION,
      )
      self.assertEqual(
        seed_result["classical_rows"][0], adjudicator.SYNTHETIC_FLAT_ROW
      )
      self.assertEqual(
        seed_result["residual_rows"][0], adjudicator.SYNTHETIC_FLAT_ROW
      )

  def test_accepts_registered_3000_iteration_extension_budget(self) -> None:
    envelopes = _envelopes()
    for envelope in envelopes:
      envelope["budget_iterations"] = 3000

    result = adjudicator.adjudicate_stair_camp(envelopes)

    self.assertEqual(result["budget_iterations"], 3000)
    self.assertTrue(result["promotion_eligible"])

  def test_uses_float_tolerance_for_a_true_one_centimetre_extension(self) -> None:
    envelopes = _envelopes()
    for envelope in envelopes:
      envelope["classical_rows"] = _rows(passing_prefix=2)
      envelope["residual_rows"] = _rows(passing_prefix=3)

    result = adjudicator.adjudicate_stair_camp(envelopes)

    self.assertAlmostEqual(result["minimum_boundary_extension_m"], 0.01)
    self.assertTrue(result["minimum_boundary_extension_passed"])
    self.assertEqual(result["classification"], adjudicator.PROMOTION_CLASSIFICATION)

  def test_deterministic_json_is_strict_and_newline_terminated(self) -> None:
    result = adjudicator.adjudicate_stair_camp(_envelopes())

    encoded_a = adjudicator.to_deterministic_json(result)
    encoded_b = adjudicator.to_deterministic_json(result)

    self.assertEqual(encoded_a, encoded_b)
    self.assertTrue(encoded_a.endswith("\n"))
    self.assertNotIn("NaN", encoded_a)
    self.assertEqual(
      json.loads(encoded_a)["classification"],
      adjudicator.PROMOTION_CLASSIFICATION,
    )


class StairCampScientificStopTest(unittest.TestCase):
  def _assert_valid_stop(self, envelopes: list[dict[str, object]]) -> None:
    result = adjudicator.adjudicate_stair_camp(envelopes)
    self.assertEqual(result["classification"], adjudicator.STOP_CLASSIFICATION)
    self.assertFalse(result["promotion_eligible"])

  def test_boundary_failure_is_a_valid_stop_not_a_protocol_error(self) -> None:
    envelopes = _envelopes()
    envelopes[1]["residual_rows"] = _rows(passing_prefix=0)

    self._assert_valid_stop(envelopes)

  def test_each_failed_gate_is_a_valid_stop(self) -> None:
    for gate in adjudicator.GATE_NAMES:
      with self.subTest(gate=gate):
        envelopes = _envelopes()
        envelopes[2][gate] = False
        result = adjudicator.adjudicate_stair_camp(envelopes)
        self.assertEqual(result["classification"], adjudicator.STOP_CLASSIFICATION)
        self.assertFalse(result["all_regression_gates_passed"])

  def test_incomplete_global_ablations_are_a_valid_stop(self) -> None:
    envelopes = _envelopes()
    envelopes[0]["completed_ablations"] = list(
      adjudicator.REQUIRED_ABLATIONS[:-1]
    )
    envelopes[0]["ablations_complete"] = False

    result = adjudicator.adjudicate_stair_camp(envelopes)

    self.assertEqual(result["classification"], adjudicator.STOP_CLASSIFICATION)
    self.assertFalse(result["ablations_complete"])
    self.assertFalse(result["promotion_eligible"])

  def test_safety_count_failure_is_a_valid_stop(self) -> None:
    envelopes = _envelopes()
    residual_rows = copy.deepcopy(envelopes[0]["residual_rows"])
    residual_rows[0]["terminations"] = 1
    envelopes[0]["residual_rows"] = residual_rows

    self._assert_valid_stop(envelopes)


class StairCampEnvelopeProtocolTest(unittest.TestCase):
  def assert_protocol_error(
    self,
    envelopes: object,
    pattern: str,
  ) -> None:
    with self.assertRaisesRegex(adjudicator.StairCampAdjudicationError, pattern):
      adjudicator.adjudicate_stair_camp(envelopes)  # type: ignore[arg-type]

  def test_requires_exactly_three_envelopes(self) -> None:
    self.assert_protocol_error(_envelopes()[:2], "exactly three")
    self.assert_protocol_error(
      [*_envelopes(), _envelope(4)],
      "exactly three",
    )

  def test_rejects_duplicate_missing_and_extra_training_seeds(self) -> None:
    duplicate = _envelopes()
    duplicate[2]["training_seed"] = 2
    self.assert_protocol_error(duplicate, "duplicate training seeds")

    wrong_set = _envelopes()
    wrong_set[2]["training_seed"] = 4
    self.assert_protocol_error(wrong_set, r"exactly \{1, 2, 3\}")

    missing = _envelopes()
    del missing[0]["training_seed"]
    self.assert_protocol_error(missing, "missing required fields.*training_seed")

  def test_rejects_non_integer_or_boolean_seeds(self) -> None:
    for value in (1.0, "1", True):
      with self.subTest(value=value):
        envelopes = _envelopes()
        envelopes[0]["training_seed"] = value
        self.assert_protocol_error(envelopes, "training_seed: must be an integer")

  def test_evaluation_seed_is_pinned_to_one(self) -> None:
    for value in (0, 2, True):
      with self.subTest(value=value):
        envelopes = _envelopes()
        envelopes[1]["evaluation_seed"] = value
        expected = "must be an integer" if value is True else "registered seed 1"
        self.assert_protocol_error(envelopes, expected)

  def test_rejects_mismatched_or_unregistered_budget(self) -> None:
    mismatch = _envelopes()
    mismatch[2]["budget_iterations"] = 3000
    self.assert_protocol_error(mismatch, "disagree on budget_iterations")

    invalid = _envelopes()
    for envelope in invalid:
      envelope["budget_iterations"] = 2000
    self.assert_protocol_error(invalid, "must be one of.*1000.*3000")

  def test_rejects_consistent_noncanonical_contract_hash(self) -> None:
    envelopes = _envelopes()
    for envelope in envelopes:
      envelope["contract_hash"] = "a" * 64
    with self.assertRaisesRegex(
      adjudicator.StairCampAdjudicationError, "canonical StairCamp contract"
    ):
      adjudicator.adjudicate_stair_camp(envelopes)

  def test_rejects_each_mismatched_campaign_binding(self) -> None:
    mutations = {
      "git_sha": "f" * 40,
      "contract_hash": "0" * 64,
    }
    for field, value in mutations.items():
      with self.subTest(field=field):
        envelopes = _envelopes()
        envelopes[1][field] = value
        pattern = (
          "canonical StairCamp contract"
          if field == "contract_hash"
          else f"disagree on {field}"
        )
        self.assert_protocol_error(envelopes, pattern)
    envelopes = _envelopes()
    envelopes[1]["artifact_bindings"] = dict(envelopes[1]["artifact_bindings"])
    envelopes[1]["artifact_bindings"]["controller_gain_hash"] = "1" * 64
    self.assert_protocol_error(envelopes, "disagree on artifact_bindings")

  def test_artifact_bindings_require_exact_digest_schema(self) -> None:
    for mutation, pattern in (
      ({"value": 1}, "six frozen bindings"),
      ({name: "x" for name in adjudicator.ARTIFACT_BINDING_NAMES}, "SHA256"),
    ):
      envelopes = _envelopes()
      envelopes[0]["artifact_bindings"] = mutation
      self.assert_protocol_error(envelopes, pattern)

  def test_rejects_empty_binding_and_hash_fields(self) -> None:
    mutations = (
      ("git_sha", "", "git_sha"),
      ("contract_hash", " ", "contract_hash"),
      ("artifact_bindings", {}, "artifact_bindings"),
    )
    for field, value, pattern in mutations:
      with self.subTest(field=field):
        envelopes = _envelopes()
        envelopes[0][field] = value
        self.assert_protocol_error(envelopes, pattern)

  def test_all_protocol_validation_precedes_any_frozen_decision_call(self) -> None:
    envelopes = _envelopes()
    envelopes[2]["evaluation_seed"] = 9

    with patch.object(adjudicator, "residual_promotion_decision") as decision:
      self.assert_protocol_error(envelopes, "registered seed 1")

    decision.assert_not_called()

  def test_gate_and_ablation_flags_must_be_actual_booleans(self) -> None:
    for field in (*adjudicator.GATE_NAMES, "ablations_complete"):
      for value in (0, 1, "true", None):
        with self.subTest(field=field, value=value):
          envelopes = _envelopes()
          envelopes[0][field] = value
          self.assert_protocol_error(envelopes, f"{field}: must be a boolean")

  def test_false_positive_evidence_and_ablation_set_are_cross_validated(self) -> None:
    envelopes = _envelopes()
    envelopes[0]["gate_stair_mode_false_positives"]["stage5_gate_passed"] = 1
    envelopes[0]["stage5_gate_passed"] = False
    result = adjudicator.adjudicate_stair_camp(envelopes)
    self.assertEqual(result["classification"], adjudicator.STOP_CLASSIFICATION)
    self.assertFalse(result["all_trigger_false_positive_checks_passed"])

    inconsistent = _envelopes()
    inconsistent[0]["gate_stair_mode_false_positives"]["stage5_gate_passed"] = 1
    self.assert_protocol_error(inconsistent, "cannot pass")
    incomplete = _envelopes()
    incomplete[0]["completed_ablations"] = []
    self.assert_protocol_error(incomplete, "does not match")

  def test_formal_evidence_checkpoint_and_digest_fields_are_mandatory(self) -> None:
    for field, value, pattern in (
      ("evidence_eligible", False, "formal evidence"),
      ("checkpoint", "", "checkpoint"),
      ("checkpoint_file_sha256", "x", "SHA256"),
      ("git_sha", "short", "Git SHA"),
      ("contract_hash", "short", "SHA256"),
    ):
      envelopes = _envelopes()
      envelopes[0][field] = value
      self.assert_protocol_error(envelopes, pattern)

  def test_rejects_nan_and_infinity_anywhere_in_envelope(self) -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
      with self.subTest(value=value):
        envelopes = _envelopes()
        envelopes[2]["ignored_metadata"] = {"bad": value}
        self.assert_protocol_error(envelopes, "NaN and infinity are forbidden")


class StairCampRowProtocolTest(unittest.TestCase):
  def assert_row_error(
    self,
    mutate: object,
    pattern: str,
    *,
    row_set: str = "residual_rows",
  ) -> None:
    envelopes = _envelopes()
    rows = copy.deepcopy(envelopes[1][row_set])
    mutate(rows)  # type: ignore[operator]
    envelopes[1][row_set] = rows
    with self.assertRaisesRegex(adjudicator.StairCampAdjudicationError, pattern):
      adjudicator.adjudicate_stair_camp(envelopes)

  def test_requires_exact_registered_grid_for_both_row_sets(self) -> None:
    for row_set in ("classical_rows", "residual_rows"):
      with self.subTest(row_set=row_set, mutation="missing"):
        self.assert_row_error(
          lambda rows: rows.pop(), "exactly one row", row_set=row_set
        )
      with self.subTest(row_set=row_set, mutation="extra"):
        self.assert_row_error(
          lambda rows: rows.append(copy.deepcopy(rows[-1])),
          "exactly one row",
          row_set=row_set,
        )
      with self.subTest(row_set=row_set, mutation="duplicate"):
        self.assert_row_error(
          lambda rows: rows[1].update(height_m=rows[0]["height_m"]),
          "duplicates registered height",
          row_set=row_set,
        )
      with self.subTest(row_set=row_set, mutation="near-but-not-exact"):
        self.assert_row_error(
          lambda rows: rows[0].update(height_m=0.0100000000001),
          "must be exactly one of",
          row_set=row_set,
        )

  def test_requires_exactly_48_trials_per_height(self) -> None:
    for value in (47, 49, 48.0, True, "48"):
      with self.subTest(value=value):
        pattern = "must equal 48" if value in (47, 49) else "must be an integer"
        self.assert_row_error(
          lambda rows, value=value: rows[3].update(trials=value),
          pattern,
        )

  def test_rejects_missing_and_malformed_required_fields(self) -> None:
    mutations = (
      (lambda rows: rows[0].pop("height_m"), "missing required fields"),
      (lambda rows: rows[0].pop("success_rate"), "missing required fields"),
      (lambda rows: rows[0].update(success_rate="0.95"), "finite number"),
      (lambda rows: rows[0].update(success_rate=True), "finite number"),
      (lambda rows: rows[0].update(success_rate=1.01), r"must be in \[0, 1\]"),
      (lambda rows: rows[0].update(success_rate=-0.01), r"must be in \[0, 1\]"),
      (lambda rows: rows[0].update(terminations=-1), "between 0 and 48"),
      (lambda rows: rows[0].update(terminations=49), "between 0 and 48"),
      (lambda rows: rows[0].update(terminations=True), "must be an integer"),
      (
        lambda rows: rows[0].update(non_wheel_contacts=49),
        "between 0 and 48",
      ),
    )
    for mutate, pattern in mutations:
      with self.subTest(pattern=pattern):
        self.assert_row_error(mutate, pattern)

  def test_rejects_non_mapping_rows_and_non_array_row_sets(self) -> None:
    self.assert_row_error(lambda rows: rows.__setitem__(0, "bad"), "must be an object")

    envelopes = _envelopes()
    envelopes[0]["classical_rows"] = {"not": "an array"}
    with self.assertRaisesRegex(adjudicator.StairCampAdjudicationError, "must be an array"):
      adjudicator.adjudicate_stair_camp(envelopes)


class StairCampAtomicCliTest(unittest.TestCase):
  def test_atomic_writer_refuses_existing_destination_without_temp_leak(self) -> None:
    result = adjudicator.adjudicate_stair_camp(_envelopes())
    with tempfile.TemporaryDirectory() as temporary:
      output = Path(temporary) / "adjudication.json"
      output.write_text("old evidence\n", encoding="utf-8")

      with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
        adjudicator.write_atomic_json(output, result)

      self.assertEqual(output.read_text(encoding="utf-8"), "old evidence\n")
      self.assertEqual(
        list(Path(temporary).glob(".adjudication.json.incomplete.*")), []
      )

  def test_atomic_writer_does_not_publish_or_leak_if_link_fails(self) -> None:
    result = adjudicator.adjudicate_stair_camp(_envelopes())
    with tempfile.TemporaryDirectory() as temporary:
      output = Path(temporary) / "adjudication.json"

      with (
        patch.object(adjudicator.os, "link", side_effect=OSError("blocked")),
        self.assertRaisesRegex(OSError, "blocked"),
      ):
        adjudicator.write_atomic_json(output, result)

      self.assertFalse(output.exists())
      self.assertEqual(
        list(Path(temporary).glob(".adjudication.json.incomplete.*")), []
      )

  def test_cli_writes_atomic_json_and_prints_only_classification(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      source = Path(temporary) / "envelopes.json"
      output = Path(temporary) / "result.json"
      source.write_text(json.dumps(_envelopes()), encoding="utf-8")
      stdout = io.StringIO()

      with redirect_stdout(stdout):
        exit_code = adjudicator.main([
          "--input",
          str(source),
          "--output",
          str(output),
        ])

      self.assertEqual(exit_code, 0)
      self.assertEqual(stdout.getvalue(), f"{adjudicator.PROMOTION_CLASSIFICATION}\n")
      self.assertEqual(
        json.loads(output.read_text(encoding="utf-8"))["classification"],
        adjudicator.PROMOTION_CLASSIFICATION,
      )

  def test_cli_accepts_exact_envelopes_wrapper(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      source = Path(temporary) / "envelopes.json"
      output = Path(temporary) / "result.json"
      source.write_text(
        json.dumps({"envelopes": _envelopes()}), encoding="utf-8"
      )

      with redirect_stdout(io.StringIO()):
        adjudicator.main(["--input", str(source), "--output", str(output)])

      self.assertTrue(output.is_file())

  def test_protocol_error_does_not_publish_output(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      source = Path(temporary) / "envelopes.json"
      output = Path(temporary) / "result.json"
      bad = _envelopes()
      bad[2]["evaluation_seed"] = 2
      source.write_text(json.dumps(bad), encoding="utf-8")

      with self.assertRaises(adjudicator.StairCampAdjudicationError):
        adjudicator.main(["--input", str(source), "--output", str(output)])

      self.assertFalse(output.exists())

  def test_json_loader_rejects_nonfinite_constants_and_duplicate_keys(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      source = Path(temporary) / "bad.json"
      source.write_text("[NaN]", encoding="utf-8")
      with self.assertRaisesRegex(
        adjudicator.StairCampAdjudicationError, "non-finite JSON constant"
      ):
        adjudicator.load_envelopes(source)

      source.write_text(
        '[{"training_seed": 1, "training_seed": 2}]', encoding="utf-8"
      )
      with self.assertRaisesRegex(
        adjudicator.StairCampAdjudicationError, "duplicate JSON object key"
      ):
        adjudicator.load_envelopes(source)

  def test_json_wrapper_rejects_unregistered_extra_metadata(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      source = Path(temporary) / "bad.json"
      source.write_text(
        json.dumps({"envelopes": _envelopes(), "extra": True}),
        encoding="utf-8",
      )

      with self.assertRaisesRegex(
        adjudicator.StairCampAdjudicationError, "only the key"
      ):
        adjudicator.load_envelopes(source)


if __name__ == "__main__":
  unittest.main()
