"""Pure contract tests for the StairCamp-aware evaluator sidecar."""

from __future__ import annotations

import copy
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hoppertrex_mjlab.hybrid.config import (
  STAIR_CAMP_ACTION_MASK,
  STAIR_CAMP_STAGE,
  STAIR_CAMP_TASK_ID,
)
from hoppertrex_mjlab.scripts.rsl_rl import evaluate_stair_camp as evaluator

_GIT_SHA = "a" * 40
_CONTRACT_SHA256 = evaluator.STAIR_CAMP_CANONICAL_CONTRACT_SHA256
_ARTIFACT_BINDINGS = {
  "calibration_hash": "1" * 64,
  "controller_gain_hash": "2" * 64,
  "posture_artifact_hash": "3" * 64,
  "posture_map_hash": "4" * 64,
  "station_calibration_hash": "5" * 64,
  "yaw_calibration_hash": "6" * 64,
}


def _checkpoint_envelope(
  *,
  updates: int = 1000,
  seed: int = 1,
  git_sha: str = _GIT_SHA,
  contract_sha256: str = _CONTRACT_SHA256,
  checkpoint_sha: str | None = None,
) -> dict[str, object]:
  return {
    "schema_version": evaluator.EVALUATOR_SCHEMA_VERSION,
    "kind": evaluator.CHECKPOINT_ENVELOPE_KIND,
    "checkpoint_file": f"D:/evidence/model_{updates}.pt",
    "checkpoint_file_sha256": checkpoint_sha or f"{updates:064x}",
    "training": {
      "schema_version": evaluator.STAIR_CAMP_CONTRACT_SCHEMA_VERSION,
      "task": STAIR_CAMP_TASK_ID,
      "training_seed": seed,
      "git_sha": git_sha,
      "contract_sha256": contract_sha256,
      "artifact_bindings": dict(_ARTIFACT_BINDINGS),
      "action_scales": list(STAIR_CAMP_STAGE.action_scales),
      "zero_initialized_deterministic_mean": True,
      "init_std": 0.6,
      "completed_updates": updates,
    },
  }


@contextmanager
def _verified_checkpoint_envelope(*, updates: int = 1000, seed: int = 1):
  with tempfile.TemporaryDirectory() as temporary:
    path = Path(temporary) / f"model_{updates}.pt"
    path.write_bytes(f"checkpoint-{updates}-{seed}".encode("ascii"))
    training = _checkpoint_envelope(updates=updates, seed=seed)["training"]
    envelope = evaluator.checkpoint_envelope_from_loaded_checkpoint(
      path,
      {
        "iter": updates - 1,
        "infos": {evaluator.STAIR_CAMP_TRAINING_INFO_KEY: training},
      },
    )
    yield envelope


def _scan_rows(
  protocol: evaluator.DomainProtocol,
  *,
  fail_index: int | None = None,
) -> list[dict[str, object]]:
  assert protocol.cell_key is not None
  assert protocol.events_per_cell is not None
  passing_successes = (
    evaluator.OFFICIAL_MIN_SUCCESSES
    if protocol.events_per_cell == evaluator.OFFICIAL_EVENTS_PER_CELL
    else evaluator.math.ceil(evaluator.SUCCESS_RATE_LIMIT * protocol.events_per_cell)
  )
  rows = []
  for index, cell in enumerate(protocol.cells):
    successes = passing_successes - 1 if index == fail_index else passing_successes
    rows.append(
      {
        protocol.cell_key: cell,
        "trials": protocol.events_per_cell,
        "successes": successes,
        "terminations": 0,
        "non_wheel_contacts": 0,
        "stair_mode_false_positives": 0,
      }
    )
  return rows


def _classical_scan_rows() -> list[dict[str, object]]:
  """Classical rows on the frozen C0 grid (no 0.15 m cell exists)."""

  key = evaluator.STAIRS_PROTOCOL.cell_key
  assert key is not None
  return [
    row
    for row in _scan_rows(evaluator.STAIRS_PROTOCOL)
    if row[key] in evaluator.CLASSICAL_HEIGHTS_M
  ]


def _rehash_config(config: dict[str, object]) -> dict[str, object]:
  mutated = copy.deepcopy(config)
  mutated.pop("config_sha256", None)
  mutated["config_sha256"] = evaluator._canonical_sha256(mutated)
  return mutated


def _collection_for(config: dict[str, object], **overrides: object) -> dict[str, object]:
  payload: dict[str, object] = {
    "config_sha256": config["config_sha256"],
    "evaluation_source": "fake_cpu_adapter",
    "adapter_metadata": {"smoke": config["profile"] == "smoke"},
  }
  domain = config["domain"]
  protocol = evaluator.protocol_for(str(domain), str(config["profile"]))
  if domain == "flat":
    gates = []
    for binding in evaluator.gate_bindings_for_profile(str(config["profile"])).values():
      gates.append(
        {
          "name": binding.name,
          "upstream_gate_passed": True,
          "num_envs": binding.num_envs,
          "steps": binding.steps,
          "scenario_count": binding.scenario_count,
          "kick_events": binding.minimum_kick_events,
          "terminations": 0,
          "non_wheel_contacts": 0,
          "stair_mode_false_positives": 0,
        }
      )
    payload["gates"] = gates
  else:
    payload["rows"] = _scan_rows(protocol)
  payload.update(overrides)
  return payload


def _k3_candidate(updates: int, *, passed: bool = True, seed: int = 1):
  gates = {name: True for name in evaluator.GATE_NAMES}
  if not passed:
    gates["flat_gate_passed"] = False
  false_positives = {name: 0 for name in evaluator.GATE_NAMES}
  return evaluator.make_k3_screen_candidate(
    checkpoint_envelope=_checkpoint_envelope(updates=updates, seed=seed),
    budget_updates=1000,
    gate_passes=gates,
    gate_stair_mode_false_positives=false_positives,
    height_row={
      "height_m": 0.01,
      "trials": 16,
      "successes": 15,
      "terminations": 0,
      "non_wheel_contacts": 0,
      "stair_mode_false_positives": 0,
    },
  )


class RegisteredProtocolTest(unittest.TestCase):
  def test_stairs_protocol_is_exactly_the_registered_scan(self) -> None:
    self.assertEqual(
      evaluator.STAIR_HEIGHTS_M,
      (0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15),
    )
    self.assertEqual(evaluator.STAIRS_PROTOCOL.events_per_cell, 48)
    self.assertEqual(evaluator.OFFICIAL_MIN_SUCCESSES, 44)
    self.assertEqual(evaluator.STAIRS_PROTOCOL.settle_steps, 100)
    self.assertEqual(evaluator.STAIRS_PROTOCOL.drive_steps, 500)
    self.assertEqual(evaluator.STAIRS_PROTOCOL.stable_steps, 25)
    self.assertAlmostEqual(evaluator.SUCCESS_TRAVEL_DISTANCE_M, 0.40)

  def test_slope_is_secondary_three_cell_protocol(self) -> None:
    self.assertEqual(evaluator.SLOPE_DEGREES, (5.0, 10.0, 15.0))
    self.assertEqual(evaluator.SLOPE_PROTOCOL.events_per_cell, 48)
    config = evaluator.make_adapter_config(
      domain="slope", checkpoint_envelope=_checkpoint_envelope()
    )
    result = evaluator.finalize_adapter_output(config, _collection_for(config))
    self.assertTrue(result["secondary_metric_only"])
    self.assertIsNone(result["registered_pass_threshold"])
    self.assertIsNone(result["result_passed"])
    self.assertTrue(all(row["passed"] is None for row in result["rows"]))
    self.assertFalse(result["promotion_evidence_eligible"])

  def test_smoke_protocol_is_tiny_and_never_evidence(self) -> None:
    for domain in ("stairs", "slope"):
      protocol = evaluator.protocol_for(domain, "smoke")
      self.assertEqual(len(protocol.cells), 1)
      self.assertEqual(protocol.events_per_cell, 1)
      self.assertEqual(protocol.drive_steps, 5)
      self.assertFalse(protocol.evidence_eligible)
    self.assertFalse(evaluator.protocol_for("flat", "smoke").evidence_eligible)
    with self.assertRaisesRegex(ValueError, "Unknown"):
      evaluator.protocol_for("unknown")
    with self.assertRaisesRegex(ValueError, "formal.*smoke"):
      evaluator.protocol_for("stairs", "screen")

  def test_four_gate_bindings_are_exact_and_camp_aware(self) -> None:
    self.assertEqual(
      tuple(evaluator.GATE_BINDINGS),
      (
        "flat_gate_passed",
        "standing_gate_passed",
        "velocity_gate_passed",
        "stage5_gate_passed",
      ),
    )
    flat = evaluator.GATE_BINDINGS["flat_gate_passed"]
    self.assertEqual((flat.num_envs, flat.scenario_count, flat.steps), (16, 15, 300))
    standing = evaluator.GATE_BINDINGS["standing_gate_passed"]
    self.assertEqual((standing.num_envs, standing.steps, standing.commands), (16, 3000, ((0.0, 0.0),)))
    velocity = evaluator.GATE_BINDINGS["velocity_gate_passed"]
    self.assertEqual(velocity.commands, ((-0.07, 0.0), (0.07, 0.0)))
    robust = evaluator.GATE_BINDINGS["stage5_gate_passed"]
    self.assertEqual((robust.num_envs, robust.minimum_kick_events), (32, 128))
    self.assertEqual(robust.kick_scale, 8.0)
    self.assertNotIn("evaluate_hybrid_gate", robust.source_suite)

  def test_smoke_gate_bindings_preserve_names_but_not_evidence(self) -> None:
    bindings = evaluator.gate_bindings_for_profile("smoke")
    self.assertEqual(tuple(bindings), evaluator.GATE_NAMES)
    self.assertTrue(all(binding.num_envs == 1 for binding in bindings.values()))
    self.assertTrue(all(not binding.evidence_eligible for binding in bindings.values()))
    self.assertEqual(bindings["stage5_gate_passed"].minimum_kick_events, 1)


class AblationDescriptorTest(unittest.TestCase):
  def test_all_registered_ablations_are_explicitly_non_promotable(self) -> None:
    self.assertEqual(evaluator.LEG_OFF_ABLATION.zero_action_indices, (2, 3, 4, 5))
    self.assertEqual(
      tuple(item.deployment_leg_scale_rad for item in evaluator.ZERO_SHOT_SCALE_ABLATIONS),
      (0.035, 0.070, 0.100),
    )
    self.assertTrue(evaluator.BASELINE_ABLATION.promotion_evidence_eligible)
    for name, descriptor in evaluator.ABLATION_DESCRIPTORS.items():
      if name != "baseline":
        self.assertFalse(descriptor.promotion_evidence_eligible)

  def test_mode_always_on_is_labelled_as_three_factor_composite(self) -> None:
    descriptor = evaluator.MODE_ALWAYS_ON_ABLATION
    self.assertTrue(descriptor.force_stair_mode_from_reset)
    self.assertEqual(len(descriptor.coupled_factors), 3)
    self.assertIn("never attribute", descriptor.interpretation.lower())

  def test_ablations_are_rejected_outside_stairs(self) -> None:
    for domain in ("flat", "slope"):
      with self.assertRaisesRegex(ValueError, "only for the stairs"):
        evaluator.make_adapter_config(
          domain=domain,
          checkpoint_envelope=_checkpoint_envelope(),
          ablation="leg-off",
        )
    with self.assertRaisesRegex(ValueError, "Unknown StairCamp ablation"):
      evaluator.resolve_ablation("parameter-shopping")


class CheckpointEnvelopeTest(unittest.TestCase):
  def test_valid_checkpoint_is_normalized_and_bound(self) -> None:
    expected = evaluator.CheckpointExpectation(
      git_sha=_GIT_SHA,
      contract_sha256=_CONTRACT_SHA256,
      artifact_bindings=_ARTIFACT_BINDINGS,
      training_seed=1,
      completed_updates=1000,
    )
    checkpoint = evaluator.validate_stair_camp_checkpoint_envelope(
      _checkpoint_envelope(), expectation=expected
    )
    training = checkpoint["training"]
    self.assertEqual(training["action_mask"], list(STAIR_CAMP_ACTION_MASK))
    self.assertEqual(training["completed_updates"], 1000)
    self.assertFalse(checkpoint["checkpoint_file_verified"])
    self.assertTrue(training["zero_initialized_deterministic_mean"])

  def test_checkpoint_file_can_be_bound_without_importing_torch(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      checkpoint_file = Path(temporary) / "model.pt"
      checkpoint_file.write_bytes(b"opaque runner checkpoint")
      loaded = {
        "infos": {
          evaluator.STAIR_CAMP_TRAINING_INFO_KEY: _checkpoint_envelope()[
            "training"
          ]
        }
      }
      envelope = evaluator.checkpoint_envelope_from_loaded_checkpoint(
        checkpoint_file, loaded
      )
      self.assertEqual(envelope["checkpoint_file"], str(checkpoint_file.resolve()))
      self.assertTrue(envelope["checkpoint_file_verified"])
      evaluator.validate_stair_camp_checkpoint_envelope(
        envelope, verify_file=True
      )
      checkpoint_file.write_bytes(b"tampered")
      with self.assertRaisesRegex(ValueError, "SHA256"):
        evaluator.validate_stair_camp_checkpoint_envelope(
          envelope, verify_file=True
        )

  def test_lazy_checkpoint_file_loader_creates_verified_envelope(self) -> None:
    import torch

    with tempfile.TemporaryDirectory() as temporary:
      path = Path(temporary) / "model_999.pt"
      torch.save(
        {
          "infos": {
            evaluator.STAIR_CAMP_TRAINING_INFO_KEY: _checkpoint_envelope()[
              "training"
            ]
          }
        },
        path,
      )
      envelope = evaluator.checkpoint_envelope_from_file(path)
      self.assertTrue(envelope["checkpoint_file_verified"])
      self.assertEqual(envelope["training"]["completed_updates"], 1000)
      self.assertEqual(
        envelope["checkpoint_file_sha256"],
        evaluator.hashlib.sha256(path.read_bytes()).hexdigest(),
      )

  def test_missing_loaded_checkpoint_info_fails_closed(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      path = Path(temporary) / "model.pt"
      path.write_bytes(b"x")
      with self.assertRaisesRegex(ValueError, "checkpoint infos"):
        evaluator.checkpoint_envelope_from_loaded_checkpoint(path, {})
      with self.assertRaisesRegex(ValueError, "stair_camp_training"):
        evaluator.checkpoint_envelope_from_loaded_checkpoint(
          path, {"infos": {}}
        )

  def test_checkpoint_schema_task_and_seed_mutations_are_rejected(self) -> None:
    mutations = (
      (("schema_version",), 2),
      (("kind",), "hybrid_stage5_checkpoint"),
      (("training", "schema_version"), 2),
      (("training", "task"), "HopperTrex-Hybrid-v2-Stage5"),
      (("training", "training_seed"), 4),
    )
    for path, value in mutations:
      with self.subTest(path=path):
        envelope = _checkpoint_envelope()
        target = envelope
        for key in path[:-1]:
          target = target[key]
        target[path[-1]] = value
        with self.assertRaises(ValueError):
          evaluator.validate_stair_camp_checkpoint_envelope(envelope)

  def test_digest_artifact_and_scale_mutations_are_rejected(self) -> None:
    mutations = []
    bad_checkpoint_sha = _checkpoint_envelope()
    bad_checkpoint_sha["checkpoint_file_sha256"] = "short"
    mutations.append(bad_checkpoint_sha)
    bad_git = _checkpoint_envelope()
    bad_git["training"]["git_sha"] = "a" * 39
    mutations.append(bad_git)
    bad_contract = _checkpoint_envelope()
    bad_contract["training"]["contract_sha256"] = "z" * 64
    mutations.append(bad_contract)
    missing_artifacts = _checkpoint_envelope()
    missing_artifacts["training"]["artifact_bindings"] = {}
    mutations.append(missing_artifacts)
    bad_artifact = _checkpoint_envelope()
    bad_artifact["training"]["artifact_bindings"]["calibration_hash"] = "x"
    mutations.append(bad_artifact)
    bad_scale = _checkpoint_envelope()
    bad_scale["training"]["action_scales"][2] = 0.071
    mutations.append(bad_scale)
    for envelope in mutations:
      with self.subTest(envelope=envelope), self.assertRaises(ValueError):
        evaluator.validate_stair_camp_checkpoint_envelope(envelope)

  def test_initialization_and_update_attestations_are_mandatory(self) -> None:
    for value in (False, None, "true"):
      envelope = _checkpoint_envelope()
      envelope["training"]["zero_initialized_deterministic_mean"] = value
      with self.assertRaisesRegex(ValueError, "zero initialization"):
        evaluator.validate_stair_camp_checkpoint_envelope(envelope)
    for value in (0.5, 0.6001, float("nan"), None):
      envelope = _checkpoint_envelope()
      envelope["training"]["init_std"] = value
      with self.assertRaises(ValueError):
        evaluator.validate_stair_camp_checkpoint_envelope(envelope)
    for value in (0, -1, 1.5, True):
      envelope = _checkpoint_envelope()
      envelope["training"]["completed_updates"] = value
      with self.assertRaises(ValueError):
        evaluator.validate_stair_camp_checkpoint_envelope(envelope)

  def test_expectation_mismatches_are_rejected_independently(self) -> None:
    expectations = (
      evaluator.CheckpointExpectation(git_sha="c" * 40),
      evaluator.CheckpointExpectation(contract_sha256="d" * 64),
      evaluator.CheckpointExpectation(
        artifact_bindings={**_ARTIFACT_BINDINGS, "calibration_hash": "e" * 64}
      ),
      evaluator.CheckpointExpectation(training_seed=2),
      evaluator.CheckpointExpectation(completed_updates=900),
    )
    for expectation in expectations:
      with self.subTest(expectation=expectation), self.assertRaises(ValueError):
        evaluator.validate_stair_camp_checkpoint_envelope(
          _checkpoint_envelope(), expectation=expectation
        )


class AdapterContractTest(unittest.TestCase):
  def test_formal_baseline_config_is_digest_bound_and_promotable_evidence(self) -> None:
    with _verified_checkpoint_envelope() as checkpoint:
      first = evaluator.make_adapter_config(
        domain="stairs",
        checkpoint_envelope=checkpoint,
        device="cuda:0",
        verify_checkpoint_file=True,
      )
      second = evaluator.make_adapter_config(
        domain="stairs",
        checkpoint_envelope=checkpoint,
        device="cuda:0",
        verify_checkpoint_file=True,
      )
    self.assertEqual(first, second)
    self.assertEqual(len(first["config_sha256"]), 64)
    self.assertTrue(first["evidence_eligible"])
    self.assertTrue(first["promotion_evidence_eligible"])
    self.assertEqual(first["policy_interface"]["actor_observation_width"], 52)
    self.assertTrue(first["policy_interface"]["stage5_actor_adapter_forbidden"])

  def test_unverified_formal_checkpoint_can_run_but_cannot_be_evidence(self) -> None:
    config = evaluator.make_adapter_config(
      domain="stairs", checkpoint_envelope=_checkpoint_envelope()
    )
    self.assertFalse(config["checkpoint"]["checkpoint_file_verified"])
    self.assertFalse(config["evidence_eligible"])
    self.assertFalse(config["promotion_evidence_eligible"])

  def test_ablation_config_is_never_promotion_evidence(self) -> None:
    with _verified_checkpoint_envelope() as checkpoint:
      for ablation in evaluator.ABLATION_DESCRIPTORS:
        config = evaluator.make_adapter_config(
          domain="stairs",
          checkpoint_envelope=checkpoint,
          ablation=ablation,
          verify_checkpoint_file=True,
        )
        self.assertEqual(config["ablation"]["name"], ablation)
        self.assertEqual(
          config["promotion_evidence_eligible"], ablation == "baseline"
        )

  def test_smoke_adapter_config_and_callable_are_cpu_friendly(self) -> None:
    config = evaluator.make_adapter_config(
      domain="stairs",
      profile="smoke",
      checkpoint_envelope=_checkpoint_envelope(),
      device="cpu",
    )
    seen = []

    def collect(request):
      seen.append(request)
      return _collection_for(dict(request))

    result = evaluator.run_live_adapter(config, collect)
    self.assertEqual(seen, [config])
    self.assertFalse(result["evidence_eligible"])
    self.assertFalse(result["promotion_evidence_eligible"])
    self.assertEqual(result["rows"][0]["trials"], 1)

  def test_object_adapter_collect_method_is_supported(self) -> None:
    config = evaluator.make_adapter_config(
      domain="slope",
      profile="smoke",
      checkpoint_envelope=_checkpoint_envelope(),
    )

    class Adapter:
      def collect(self, request):
        return _collection_for(dict(request))

    result = evaluator.run_live_adapter(config, Adapter())
    self.assertEqual(result["domain"], "slope")

  def test_dynamic_adapter_loading_is_lazy_and_strict(self) -> None:
    module = types.ModuleType("fake_stair_adapter")
    module.collect = lambda config: _collection_for(dict(config))
    with patch.dict(sys.modules, {"fake_stair_adapter": module}):
      self.assertIs(
        evaluator.load_live_adapter("fake_stair_adapter:collect"), module.collect
      )
      with self.assertRaisesRegex(ValueError, "does not exist"):
        evaluator.load_live_adapter("fake_stair_adapter:missing")
    for invalid in ("no_colon", ":missing", "too:many:parts"):
      with self.assertRaisesRegex(ValueError, "module:callable"):
        evaluator.load_live_adapter(invalid)

  def test_semantic_config_forgery_fails_even_with_a_recomputed_digest(self) -> None:
    config = evaluator.make_adapter_config(
      domain="stairs", checkpoint_envelope=_checkpoint_envelope()
    )
    collection = _collection_for(config)

    wrong_width = copy.deepcopy(config)
    wrong_width["policy_interface"]["actor_observation_width"] = 34
    with self.assertRaisesRegex(ValueError, "52-D"):
      evaluator.finalize_adapter_output(_rehash_config(wrong_width), collection)

    forged_evidence = copy.deepcopy(config)
    forged_evidence["evidence_eligible"] = True
    forged_evidence["promotion_evidence_eligible"] = True
    with self.assertRaisesRegex(ValueError, "evidence eligibility"):
      evaluator.finalize_adapter_output(_rehash_config(forged_evidence), collection)

    forged_verification = copy.deepcopy(config)
    forged_verification["checkpoint"]["checkpoint_file_verified"] = True
    with self.assertRaisesRegex(ValueError, "does not exist"):
      evaluator.finalize_adapter_output(_rehash_config(forged_verification), collection)

    wrong_seed = copy.deepcopy(config)
    wrong_seed["evaluation_seed"] = 2
    with self.assertRaisesRegex(ValueError, "seed must be 1"):
      evaluator.finalize_adapter_output(_rehash_config(wrong_seed), collection)

  def test_adapter_config_and_output_digest_mutations_fail_closed(self) -> None:
    config = evaluator.make_adapter_config(
      domain="stairs", checkpoint_envelope=_checkpoint_envelope()
    )
    collection = _collection_for(config)
    tampered_config = copy.deepcopy(config)
    tampered_config["device"] = "different"
    with self.assertRaisesRegex(ValueError, "digest"):
      evaluator.finalize_adapter_output(tampered_config, collection)
    collection["config_sha256"] = "f" * 64
    with self.assertRaisesRegex(ValueError, "not bound"):
      evaluator.finalize_adapter_output(config, collection)

  def test_stair_rows_compute_boundary_and_keep_later_rows_observational(self) -> None:
    config = evaluator.make_adapter_config(
      domain="stairs", checkpoint_envelope=_checkpoint_envelope()
    )
    rows = _scan_rows(evaluator.STAIRS_PROTOCOL, fail_index=2)
    result = evaluator.finalize_adapter_output(
      config, _collection_for(config, rows=rows)
    )
    self.assertEqual(result["highest_contiguous_passing_height_m"], 0.02)
    self.assertFalse(result["rows"][2]["passed"])
    self.assertTrue(result["rows"][3]["passed"])
    self.assertFalse(result["all_cells_passed"])

  def test_44_of_48_is_the_first_passing_integer_count(self) -> None:
    config = evaluator.make_adapter_config(
      domain="stairs", checkpoint_envelope=_checkpoint_envelope()
    )
    rows = _scan_rows(evaluator.STAIRS_PROTOCOL)
    result = evaluator.finalize_adapter_output(
      config, _collection_for(config, rows=rows)
    )
    self.assertTrue(all(row["passed"] for row in result["rows"]))
    rows[0]["successes"] = 43
    result = evaluator.finalize_adapter_output(
      config, _collection_for(config, rows=rows)
    )
    self.assertFalse(result["rows"][0]["passed"])

  def test_scan_requires_exact_cells_and_exact_trial_counts(self) -> None:
    config = evaluator.make_adapter_config(
      domain="stairs", checkpoint_envelope=_checkpoint_envelope()
    )
    rows = _scan_rows(evaluator.STAIRS_PROTOCOL)
    mutations = []
    missing = copy.deepcopy(rows)
    missing.pop()
    mutations.append(missing)
    duplicate = copy.deepcopy(rows)
    duplicate[-1]["height_m"] = duplicate[0]["height_m"]
    mutations.append(duplicate)
    wrong_cell = copy.deepcopy(rows)
    wrong_cell[-1]["height_m"] = 0.14
    mutations.append(wrong_cell)
    wrong_trials = copy.deepcopy(rows)
    wrong_trials[0]["trials"] = 47
    mutations.append(wrong_trials)
    for mutation in mutations:
      with self.subTest(mutation=mutation), self.assertRaises(ValueError):
        evaluator.finalize_adapter_output(
          config, _collection_for(config, rows=mutation)
        )

  def test_scan_rejects_impossible_counts_rates_and_nonfinite_values(self) -> None:
    config = evaluator.make_adapter_config(
      domain="stairs", checkpoint_envelope=_checkpoint_envelope()
    )
    rows = _scan_rows(evaluator.STAIRS_PROTOCOL)
    impossible = copy.deepcopy(rows)
    impossible[0]["successes"] = 49
    inconsistent = copy.deepcopy(rows)
    inconsistent[0]["success_rate"] = 0.1
    nonfinite = copy.deepcopy(rows)
    nonfinite[0]["height_m"] = float("nan")
    bool_count = copy.deepcopy(rows)
    bool_count[0]["terminations"] = False
    for mutation in (impossible, inconsistent, nonfinite, bool_count):
      with self.subTest(mutation=mutation), self.assertRaises(ValueError):
        evaluator.finalize_adapter_output(
          config, _collection_for(config, rows=mutation)
        )

  def test_termination_and_non_wheel_contact_each_fail_a_height(self) -> None:
    config = evaluator.make_adapter_config(
      domain="stairs", checkpoint_envelope=_checkpoint_envelope()
    )
    for field in ("terminations", "non_wheel_contacts"):
      rows = _scan_rows(evaluator.STAIRS_PROTOCOL)
      rows[0][field] = 1
      result = evaluator.finalize_adapter_output(
        config, _collection_for(config, rows=rows)
      )
      self.assertFalse(result["rows"][0]["passed"])
      self.assertIsNone(result["highest_contiguous_passing_height_m"])

  def test_flat_collection_maps_exactly_four_named_booleans(self) -> None:
    config = evaluator.make_adapter_config(
      domain="flat", checkpoint_envelope=_checkpoint_envelope()
    )
    result = evaluator.finalize_adapter_output(config, _collection_for(config))
    self.assertEqual(tuple(result["gate_booleans"]), evaluator.GATE_NAMES)
    self.assertTrue(result["all_gates_passed"])
    self.assertEqual(result["gates"][-1]["kick_events"], 128)

  def test_gate_safety_failure_is_valid_negative_evidence(self) -> None:
    config = evaluator.make_adapter_config(
      domain="flat", checkpoint_envelope=_checkpoint_envelope()
    )
    collection = _collection_for(config)
    collection["gates"][1]["stair_mode_false_positives"] = 1
    result = evaluator.finalize_adapter_output(config, collection)
    self.assertFalse(result["gate_booleans"]["standing_gate_passed"])
    self.assertFalse(result["all_gates_passed"])

  def test_gate_protocol_mismatch_duplicate_and_false_boolean_are_rejected(self) -> None:
    config = evaluator.make_adapter_config(
      domain="flat", checkpoint_envelope=_checkpoint_envelope()
    )
    base = _collection_for(config)
    wrong_count = copy.deepcopy(base)
    wrong_count["gates"][-1]["kick_events"] = 127
    duplicate = copy.deepcopy(base)
    duplicate["gates"][-1]["name"] = "flat_gate_passed"
    inconsistent = copy.deepcopy(base)
    inconsistent["gates"][0]["passed"] = False
    for mutation in (wrong_count, duplicate, inconsistent):
      with self.subTest(mutation=mutation), self.assertRaises(ValueError):
        evaluator.finalize_adapter_output(config, mutation)

  def test_adapter_metadata_must_be_strict_json(self) -> None:
    config = evaluator.make_adapter_config(
      domain="stairs", checkpoint_envelope=_checkpoint_envelope()
    )
    with self.assertRaisesRegex(ValueError, "non-finite"):
      evaluator.finalize_adapter_output(
        config,
        _collection_for(config, adapter_metadata={"loss": float("inf")}),
      )


class K3SelectionTest(unittest.TestCase):
  def test_newest_passing_of_latest_three_is_selected(self) -> None:
    candidates = [
      _k3_candidate(801, passed=True),
      _k3_candidate(1000, passed=False),
      _k3_candidate(901, passed=True),
    ]
    result = evaluator.select_newest_passing_checkpoint(candidates)
    self.assertEqual(result["status"], "selected")
    self.assertEqual(result["classification"], "STAIR_CAMP_CHECKPOINT_SELECTED")
    self.assertEqual(
      result["selected_checkpoint"]["training"]["completed_updates"], 901
    )
    self.assertEqual(
      [row["completed_updates"] for row in result["ordered_candidates"]],
      [1000, 901, 801],
    )

  def test_no_passer_is_a_machine_readable_stop_not_an_exception(self) -> None:
    result = evaluator.select_newest_passing_checkpoint(
      [_k3_candidate(update, passed=False) for update in (801, 901, 1000)]
    )
    self.assertEqual(result["status"], "no_passing_checkpoint")
    self.assertEqual(result["classification"], "STOP_NO_PROMOTION")
    self.assertIsNone(result["selected_checkpoint"])

  def test_screen_pass_requires_gates_height_and_all_false_positive_counts(self) -> None:
    passing = _k3_candidate(1000)
    self.assertTrue(passing["screen_passed"])
    gates = {name: True for name in evaluator.GATE_NAMES}
    counts = {name: 0 for name in evaluator.GATE_NAMES}
    counts["stage5_gate_passed"] = 1
    failed = evaluator.make_k3_screen_candidate(
      checkpoint_envelope=_checkpoint_envelope(updates=1000),
      budget_updates=1000,
      gate_passes=gates,
      gate_stair_mode_false_positives=counts,
      height_row={
        "height_m": 0.01,
        "trials": 16,
        "successes": 15,
        "terminations": 0,
        "non_wheel_contacts": 0,
        "stair_mode_false_positives": 0,
      },
    )
    self.assertFalse(failed["screen_passed"])

  def test_selection_requires_exactly_latest_three_unique_checkpoints(self) -> None:
    with self.assertRaisesRegex(ValueError, "exactly three"):
      evaluator.select_newest_passing_checkpoint([_k3_candidate(1000)])
    wrong_updates = [_k3_candidate(update) for update in (700, 901, 1000)]
    with self.assertRaisesRegex(ValueError, "latest three"):
      evaluator.select_newest_passing_checkpoint(wrong_updates)
    duplicates = [_k3_candidate(update) for update in (801, 901, 1000)]
    duplicates[1]["checkpoint"]["checkpoint_file_sha256"] = duplicates[0][
      "checkpoint"
    ]["checkpoint_file_sha256"]
    with self.assertRaisesRegex(ValueError, "distinct"):
      evaluator.select_newest_passing_checkpoint(duplicates)

  def test_selection_rejects_mixed_seed_contract_and_budget_pools(self) -> None:
    base = [_k3_candidate(update) for update in (801, 901, 1000)]
    mixed_seed = copy.deepcopy(base)
    mixed_seed[0]["checkpoint"]["training"]["training_seed"] = 2
    with self.assertRaisesRegex(ValueError, "training_seed"):
      evaluator.select_newest_passing_checkpoint(mixed_seed)
    mixed_contract = copy.deepcopy(base)
    mixed_contract[0]["checkpoint"]["training"]["contract_sha256"] = "c" * 64
    with self.assertRaisesRegex(ValueError, "contract"):
      evaluator.select_newest_passing_checkpoint(mixed_contract)
    mixed_budget = copy.deepcopy(base)
    mixed_budget[0]["budget_updates"] = 3000
    with self.assertRaisesRegex(ValueError, "one budget pool"):
      evaluator.select_newest_passing_checkpoint(mixed_budget)

  def test_unregistered_gate_keys_and_forged_screen_boolean_are_rejected(self) -> None:
    gates = {name: True for name in evaluator.GATE_NAMES}
    gates.pop("flat_gate_passed")
    with self.assertRaisesRegex(ValueError, "exactly the four"):
      evaluator.make_k3_screen_candidate(
        checkpoint_envelope=_checkpoint_envelope(),
        budget_updates=1000,
        gate_passes=gates,
        gate_stair_mode_false_positives={name: 0 for name in evaluator.GATE_NAMES},
        height_row={
          "height_m": 0.01,
          "trials": 16,
          "successes": 15,
          "terminations": 0,
          "non_wheel_contacts": 0,
          "stair_mode_false_positives": 0,
        },
      )
    candidate = _k3_candidate(1000, passed=False)
    candidate["screen_passed"] = True
    with self.assertRaisesRegex(ValueError, "inconsistent"):
      evaluator.validate_k3_screen_candidate(candidate)


class AdjudicationCompositionTest(unittest.TestCase):
  def _formal_result(
    self,
    checkpoint: dict[str, object],
    *,
    domain: str,
    ablation: str = "baseline",
  ) -> dict[str, object]:
    config = evaluator.make_adapter_config(
      domain=domain,
      checkpoint_envelope=checkpoint,
      profile="formal",
      ablation=ablation,
      device="cpu",
      verify_checkpoint_file=True,
    )
    return evaluator.finalize_adapter_output(config, _collection_for(config))

  def _selection(
    self, selected_checkpoint: dict[str, object]
  ) -> dict[str, object]:
    candidates = []
    for update in (801, 901):
      candidates.append(_k3_candidate(update, passed=False))
    candidates.append(
      evaluator.make_k3_screen_candidate(
        checkpoint_envelope=selected_checkpoint,
        budget_updates=1000,
        gate_passes={name: True for name in evaluator.GATE_NAMES},
        gate_stair_mode_false_positives={
          name: 0 for name in evaluator.GATE_NAMES
        },
        height_row={
          "height_m": 0.01,
          "trials": 16,
          "successes": 15,
          "terminations": 0,
          "non_wheel_contacts": 0,
          "stair_mode_false_positives": 0,
        },
      )
    )
    return evaluator.select_newest_passing_checkpoint(candidates)

  def test_compose_seed_projects_evaluator_outputs_to_adjudicator_schema(self) -> None:
    with _verified_checkpoint_envelope() as checkpoint:
      stairs = self._formal_result(checkpoint, domain="stairs")
      flat = self._formal_result(checkpoint, domain="flat")
      ablations = [
        self._formal_result(checkpoint, domain="stairs", ablation=name)
        for name in evaluator.ADJUDICATION_ABLATION_NAMES
      ]
      envelope = evaluator.compose_adjudication_seed_envelope(
        stairs_result=stairs,
        flat_result=flat,
        classical_rows=_classical_scan_rows(),
        ablation_results=ablations,
        k3_selection=self._selection(checkpoint),
        budget_iterations=1000,
      )
    self.assertEqual(envelope["training_seed"], 1)
    self.assertEqual(envelope["evaluation_seed"], 1)
    self.assertEqual(envelope["budget_iterations"], 1000)
    self.assertEqual(
      envelope["contract_hash"], evaluator.STAIR_CAMP_CANONICAL_CONTRACT_SHA256
    )
    self.assertEqual(
      envelope["completed_ablations"],
      list(evaluator.ADJUDICATION_ABLATION_NAMES),
    )
    self.assertTrue(envelope["ablations_complete"])
    self.assertTrue(envelope["evidence_eligible"])
    self.assertEqual(
      envelope["gate_stair_mode_false_positives"],
      {name: 0 for name in evaluator.GATE_NAMES},
    )
    self.assertEqual(
      set(envelope["residual_rows"][0]),
      {
        "height_m",
        "success_rate",
        "terminations",
        "non_wheel_contacts",
        "trials",
      },
    )

  def test_compose_rejects_wrong_ablation_order_and_unverified_selection(self) -> None:
    with _verified_checkpoint_envelope() as checkpoint:
      stairs = self._formal_result(checkpoint, domain="stairs")
      flat = self._formal_result(checkpoint, domain="flat")
      ablations = [
        self._formal_result(checkpoint, domain="stairs", ablation=name)
        for name in evaluator.ADJUDICATION_ABLATION_NAMES
      ]
      selection = self._selection(checkpoint)
      with self.assertRaisesRegex(ValueError, "ablation descriptor"):
        evaluator.compose_adjudication_seed_envelope(
          stairs_result=stairs,
          flat_result=flat,
          classical_rows=_classical_scan_rows(),
          ablation_results=list(reversed(ablations)),
          k3_selection=selection,
          budget_iterations=1000,
        )
      selection["selected_checkpoint"]["checkpoint_file_sha256"] = "0" * 64
      with self.assertRaises(ValueError):
        evaluator.compose_adjudication_seed_envelope(
          stairs_result=stairs,
          flat_result=flat,
          classical_rows=_classical_scan_rows(),
          ablation_results=ablations,
          k3_selection=selection,
          budget_iterations=1000,
        )


class MachineOutputAndCliTest(unittest.TestCase):
  def test_manifest_is_deterministic_and_complete(self) -> None:
    manifest = evaluator.manifest_payload()
    self.assertEqual(manifest["task"], STAIR_CAMP_TASK_ID)
    self.assertEqual(tuple(manifest["protocols"]), ("stairs", "flat", "slope"))
    self.assertEqual(tuple(manifest["gate_bindings"]), evaluator.GATE_NAMES)
    self.assertEqual(manifest["k3"]["pool_size"], 3)
    self.assertEqual(manifest["checkpoint_contract"]["action_mask"], list(STAIR_CAMP_ACTION_MASK))
    self.assertEqual(manifest["checkpoint_contract"]["actor_observation_width"], 52)
    self.assertEqual(manifest["checkpoint_contract"]["critic_observation_width"], 55)
    first = evaluator.deterministic_json(manifest)
    second = evaluator.deterministic_json(manifest)
    self.assertEqual(first, second)
    self.assertTrue(first.endswith("\n"))
    self.assertFalse(first.endswith("\n\n"))
    json.loads(first)

  def test_deterministic_json_rejects_nan_and_non_string_keys(self) -> None:
    with self.assertRaisesRegex(ValueError, "non-finite"):
      evaluator.deterministic_json({"bad": float("nan")})
    with self.assertRaisesRegex(ValueError, "non-string"):
      evaluator.deterministic_json({1: "bad"})

  def test_atomic_writer_refuses_overwrite_and_leaves_no_temporary_file(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      output = Path(temporary) / "nested" / "result.json"
      payload = {"status": "ok"}
      evaluator.write_machine_output(payload, output)
      self.assertEqual(json.loads(output.read_text(encoding="utf-8")), payload)
      self.assertEqual(list(output.parent.glob("*.incomplete.*")), [])
      with self.assertRaises(FileExistsError):
        evaluator.write_machine_output(payload, output)

  def test_manifest_cli_prints_machine_json(self) -> None:
    stream = io.StringIO()
    with redirect_stdout(stream):
      return_code = evaluator.main(["manifest", "--profile", "smoke"])
    self.assertEqual(return_code, 0)
    payload = json.loads(stream.getvalue())
    self.assertEqual(payload["kind"], "stair_camp_evaluator_manifest")
    self.assertFalse(payload["protocols"]["stairs"]["evidence_eligible"])

  def test_finalize_cli_round_trip(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      checkpoint_file = root / "checkpoint.json"
      collection_file = root / "collection.json"
      output_file = root / "result.json"
      checkpoint = _checkpoint_envelope()
      checkpoint_file.write_text(
        evaluator.deterministic_json(checkpoint), encoding="utf-8"
      )
      config = evaluator.make_adapter_config(
        domain="stairs", profile="smoke", checkpoint_envelope=checkpoint
      )
      collection_file.write_text(
        evaluator.deterministic_json(_collection_for(config)), encoding="utf-8"
      )
      return_code = evaluator.main(
        [
          "finalize",
          "--domain",
          "stairs",
          "--profile",
          "smoke",
          "--checkpoint-envelope",
          str(checkpoint_file),
          "--collection",
          str(collection_file),
          "--output",
          str(output_file),
        ]
      )
      self.assertEqual(return_code, 0)
      result = json.loads(output_file.read_text(encoding="utf-8"))
      self.assertEqual(result["kind"], evaluator.EVALUATION_ENVELOPE_KIND)
      self.assertEqual(result["rows"][0]["trials"], 1)

  def test_validate_checkpoint_cli_applies_same_commit_binding(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      path = Path(temporary) / "checkpoint.json"
      path.write_text(
        evaluator.deterministic_json(_checkpoint_envelope()), encoding="utf-8"
      )
      stream = io.StringIO()
      with redirect_stdout(stream):
        evaluator.main(
          [
            "validate-checkpoint",
            "--envelope",
            str(path),
            "--expected-git-sha",
            _GIT_SHA,
            "--expected-contract-hash",
            _CONTRACT_SHA256,
            "--expected-training-seed",
            "1",
          ]
        )
      payload = json.loads(stream.getvalue())
      self.assertTrue(payload["valid"])

  def test_select_k3_cli_outputs_valid_stop_with_exit_zero(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      paths = []
      for update in (801, 901, 1000):
        path = root / f"candidate_{update}.json"
        path.write_text(
          evaluator.deterministic_json(_k3_candidate(update, passed=False)),
          encoding="utf-8",
        )
        paths.append(path)
      stream = io.StringIO()
      with redirect_stdout(stream):
        return_code = evaluator.main(
          ["select-k3", "--candidate", *(str(path) for path in paths)]
        )
      self.assertEqual(return_code, 0)
      self.assertEqual(json.loads(stream.getvalue())["classification"], "STOP_NO_PROMOTION")


if __name__ == "__main__":
  unittest.main()

