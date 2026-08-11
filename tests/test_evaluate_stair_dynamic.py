"""Pure tests for the HopperTrex v3 StairDynamic evaluator contract."""

from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

from hoppertrex_mjlab.hybrid.stair_dynamic import DYNAMIC_STAIR_TASK_ID
from hoppertrex_mjlab.hybrid.stair_dynamic_contract import (
  DYNAMIC_STAIR_ACTION_SCALES,
  DYNAMIC_STAIR_MIGRATION_INFO_KEY,
  DYNAMIC_STAIR_TRAINING_INFO_KEY,
)
from hoppertrex_mjlab.scripts.rsl_rl import evaluate_stair_camp
from hoppertrex_mjlab.scripts.rsl_rl import evaluate_stair_dynamic as evaluator

_GIT_SHA = "a" * 40
_CONTRACT_SHA = "b" * 64
_MANEUVER_SHA = "c" * 64
_STAGE5_CHECKPOINT_SHA = "d" * 64
_STAGE5_GATE_SHA = "e" * 64
_ARTIFACTS = {
  "controller_gain_hash": "1" * 64,
  "calibration_hash": "2" * 64,
  "yaw_calibration_hash": "3" * 64,
  "posture_map_hash": "4" * 64,
  "posture_artifact_hash": "5" * 64,
  "station_calibration_hash": "6" * 64,
  "dynamic_maneuver_hash": _MANEUVER_SHA,
}


def _training(
  updates: int = 100,
  *,
  contract_sha: str = _CONTRACT_SHA,
) -> dict[str, object]:
  return {
    "schema_version": 1,
    "task": DYNAMIC_STAIR_TASK_ID,
    "training_seed": 1,
    "git_sha": _GIT_SHA,
    "contract_sha256": contract_sha,
    "artifact_bindings": dict(_ARTIFACTS),
    "action_scales": list(DYNAMIC_STAIR_ACTION_SCALES),
    "maneuver_sha256": _MANEUVER_SHA,
    "source_stage5_checkpoint_sha256": _STAGE5_CHECKPOINT_SHA,
    "source_stage5_gate_sha256": _STAGE5_GATE_SHA,
    "stage5_prefix_preserved_and_new_columns_zero": True,
    "completed_updates": updates,
  }


def _expectation(updates: int | None = None) -> evaluator.CheckpointExpectation:
  return evaluator.CheckpointExpectation(
    git_sha=_GIT_SHA,
    contract_sha256=_CONTRACT_SHA,
    artifact_bindings=dict(_ARTIFACTS),
    maneuver_sha256=_MANEUVER_SHA,
    source_stage5_checkpoint_sha256=_STAGE5_CHECKPOINT_SHA,
    source_stage5_gate_sha256=_STAGE5_GATE_SHA,
    completed_updates=updates,
  )


def _verified_envelope(path: Path, updates: int = 100) -> dict[str, object]:
  path.write_bytes(f"checkpoint-{updates}-{path.name}".encode("ascii"))
  return evaluator.checkpoint_envelope_from_loaded_checkpoint(
    path,
    {
      "iter": updates - 1,
      "infos": {DYNAMIC_STAIR_TRAINING_INFO_KEY: _training(updates)},
    },
  )


def _synthetic_envelope(
  updates: int,
  *,
  checkpoint_sha: str | None = None,
  contract_sha: str = _CONTRACT_SHA,
) -> dict[str, object]:
  return {
    "schema_version": evaluator.EVALUATOR_SCHEMA_VERSION,
    "kind": evaluator.CHECKPOINT_ENVELOPE_KIND,
    "checkpoint_file": f"C:/evidence/model_{updates}.pt",
    "checkpoint_file_sha256": checkpoint_sha or f"{updates:064x}",
    "checkpoint_iteration": updates - 1,
    "training": _training(updates, contract_sha=contract_sha),
  }



def _zero_update_checkpoint() -> dict[str, object]:
  std = [0.07, 0.09, 0.05, 0.05, 0.05, 0.05]
  migration = {
    "source_checkpoint_sha256": _STAGE5_CHECKPOINT_SHA,
    "source_gate_sha256": _STAGE5_GATE_SHA,
    "source_task": "HopperTrex-Hybrid-v2-Stage5",
    "source_seed": 1,
    "source_completed_updates": 100,
    "target_task": DYNAMIC_STAIR_TASK_ID,
    "source_actor_width": 34,
    "target_actor_width": 52,
    "source_critic_width": 34,
    "target_critic_width": 56,
    "actor_first_layer": "mlp.0.weight",
    "critic_first_layer": "mlp.0.weight",
    "std_key": "distribution.std_param",
    "source_action_std": std,
    "target_action_std": std,
    "collapsed_std_threshold": 0.02,
    "collapsed_active_indices": [],
    "collapsed_active_actions": [],
    "reset_collapsed_active_std": False,
    "created_at": "2026-08-11T12:00:00+00:00",
  }
  return {
    "iter": 0,
    "actor_state_dict": {"mlp.0.weight": torch.zeros(4, 52)},
    "critic_state_dict": {"mlp.0.weight": torch.zeros(3, 56)},
    "optimizer_state_dict": {"state": {}, "param_groups": []},
    "infos": {
      DYNAMIC_STAIR_MIGRATION_INFO_KEY: migration,
      "env_state": {"common_step_counter": 0},
    },
  }


def _zero_update_envelope(path: Path) -> dict[str, object]:
  checkpoint = _zero_update_checkpoint()
  torch.save(checkpoint, path)
  return evaluator.migration_checkpoint_envelope_from_loaded_checkpoint(
    path, checkpoint, expectation=_expectation(0)
  )


def _phase_durations() -> dict[str, float]:
  return {phase: 0.1 for phase in evaluator.PHASE_NAMES}


def _trial(
  protocol: evaluator.StairEvaluationProtocol,
  *,
  height: float,
  env_index: int,
  repeat_index: int,
  ablation: str = "full",
  success: bool = True,
  lift_mode: str = "alternating",
) -> dict[str, object]:
  descriptor = evaluator.resolve_ablation(ablation)
  mode = "DYNAMIC"
  lead = "LEFT"
  left_trigger: float | None = 0.4
  right_trigger: float | None = None if lift_mode == "synchronized" else 0.6
  feedforward = 0.05
  wheel_rms, wheel_max = 0.08, 0.12
  leg_rms, leg_max = 0.01, 0.02
  if descriptor.force_stair_request_false:
    mode, lead = "ROLL", "NONE"
    left_trigger = right_trigger = None
  if descriptor.disable_feedforward:
    feedforward = 0.0
  if {0, 1}.issubset(descriptor.zero_action_indices):
    wheel_rms = wheel_max = 0.0
  if {2, 3, 4, 5}.issubset(descriptor.zero_action_indices):
    leg_rms = leg_max = 0.0
  steps = protocol.risers_per_trial
  abort_reason = None
  stable_steps = protocol.stable_steps
  if not success:
    mode, lead = "ABORT", "NONE"
    left_trigger = right_trigger = None
    steps = 0
    stable_steps = 0
    abort_reason = "contact_timeout"
  return {
    "height_m": height,
    "env_index": env_index,
    "repeat_index": repeat_index,
    "success": success,
    "traversal_mode": mode,
    "lift_mode": lift_mode,
    "lead_side": lead,
    "left_trigger_time_s": left_trigger,
    "right_trigger_time_s": right_trigger,
    "phase_durations_s": _phase_durations(),
    "wheel_ppo_rms": wheel_rms,
    "wheel_ppo_max_abs": wheel_max,
    "leg_ppo_rms": leg_rms,
    "leg_ppo_max_abs": leg_max,
    "feedforward_max_abs_rad": feedforward,
    "peak_abs_pitch_rad": 0.12,
    "peak_abs_roll_rad": 0.03,
    "steps_completed": steps,
    "step_recovery_times_s": [0.5] * steps,
    "stable_steps": stable_steps,
    "terminated": False,
    "non_wheel_contact": False,
    "abort_reason": abort_reason,
  }


def _trials(
  protocol: evaluator.StairEvaluationProtocol,
  *,
  ablation: str = "full",
  failures_per_height: int = 0,
  lift_mode: str = "alternating",
) -> list[dict[str, object]]:
  trials = []
  for height in protocol.heights_m:
    count = 0
    for repeat in range(protocol.repeats):
      for env_index in range(protocol.num_envs_per_height):
        failed = count < failures_per_height
        trials.append(
          _trial(
            protocol,
            height=height,
            env_index=env_index,
            repeat_index=repeat,
            ablation=ablation,
            success=not failed,
            lift_mode=lift_mode,
          )
        )
        count += 1
  return trials


def _gate_rows(*, failed_gate: str | None = None) -> list[dict[str, object]]:
  rows = []
  for name, binding in evaluator.GATE_BINDINGS.items():
    rows.append(
      {
        "name": name,
        "upstream_gate_passed": name != failed_gate,
        "num_envs": binding.num_envs,
        "steps": binding.steps,
        "scenario_count": binding.scenario_count,
        "kick_events": binding.minimum_kick_events,
        "terminations": 0,
        "non_wheel_contacts": 0,
        "stair_mode_false_positives": 0,
      }
    )
  return rows


def _collection(request: dict[str, object], trials: list[dict[str, object]]) -> dict[str, object]:
  return {
    "request_sha256": request["request_sha256"],
    "evaluation_source": "fake_live_hook",
    "adapter_metadata": {"simulation_started": False},
    "trials": trials,
  }


def _k3_candidate(updates: int, *, passed: bool = True) -> dict[str, object]:
  gate_passes = {name: True for name in evaluator.GATE_NAMES}
  false_positives = {name: 0 for name in evaluator.GATE_NAMES}
  return evaluator.make_k3_screen_candidate(
    checkpoint_envelope=_synthetic_envelope(updates),
    budget_updates=100,
    gate_passes=gate_passes,
    gate_stair_mode_false_positives=false_positives,
    height_row={
      "height_m": 0.01,
      "trials": 16,
      "successes": 15 if passed else 14,
      "terminations": 0,
      "non_wheel_contacts": 0,
      "stair_mode_false_positives": 0,
    },
  )


class RegisteredContractTest(unittest.TestCase):
  def test_manifest_pins_task_widths_protocols_and_provisional_status(self) -> None:
    manifest = evaluator.manifest_payload()
    self.assertEqual(manifest["task"], DYNAMIC_STAIR_TASK_ID)
    self.assertEqual(manifest["checkpoint_contract"]["actor_observation_width"], 52)
    self.assertEqual(manifest["checkpoint_contract"]["critic_observation_width"], 56)
    self.assertEqual(manifest["protocols"]["single-riser"]["heights_m"], [0.01, 0.02, 0.03])
    self.assertEqual(manifest["protocols"]["single-riser"]["trials_per_height"], 48)
    self.assertEqual(manifest["protocols"]["continuous-stairs"]["risers_per_trial"], 3)
    self.assertEqual(manifest["single_seed_status"], "provisional")
    self.assertFalse(manifest["promotion_claim_eligible"])
    self.assertFalse(manifest["live_hook"]["implemented_here"])
    self.assertTrue(manifest["live_hook"]["implemented"])
    self.assertEqual(
      manifest["live_hook"]["module"],
      "hoppertrex_mjlab.scripts.rsl_rl.stair_dynamic_live_adapter",
    )

  def test_gate_bindings_are_reused_not_reauthored(self) -> None:
    self.assertIs(evaluator.GATE_BINDINGS, evaluate_stair_camp.GATE_BINDINGS)
    self.assertIs(evaluator.gate_bindings_for_profile, evaluate_stair_camp.gate_bindings_for_profile)
    self.assertEqual(tuple(evaluator.GATE_BINDINGS), evaluator.GATE_NAMES)

  def test_six_ablation_contracts_are_exact(self) -> None:
    self.assertEqual(
      evaluator.ABLATION_ORDER,
      (
        "roll-only",
        "feedforward-only",
        "policy-only",
        "full",
        "leg-PPO-off",
        "wheel-PPO-off",
      ),
    )
    self.assertEqual(
      evaluator.ABLATION_DESCRIPTORS["feedforward-only"].zero_action_indices,
      tuple(range(6)),
    )
    self.assertEqual(
      evaluator.ABLATION_DESCRIPTORS["leg-PPO-off"].zero_action_indices,
      (2, 3, 4, 5),
    )
    self.assertTrue(evaluator.ABLATION_DESCRIPTORS["full"].primary_evidence_eligible)


class CheckpointAndRequestTest(unittest.TestCase):
  def setUp(self) -> None:
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.root = Path(self.temporary.name)
    self.checkpoint = self.root / "model_99.pt"
    self.envelope = _verified_envelope(self.checkpoint)

  def test_checkpoint_and_request_bind_every_frozen_value(self) -> None:
    request = evaluator.make_evaluation_request(
      suite="continuous-stairs",
      checkpoint_envelope=self.envelope,
      expectation=_expectation(100),
    )
    validated = evaluator.validate_evaluation_request(
      request, expectation=_expectation(100)
    )
    self.assertEqual(validated["checkpoint"]["training"]["training_seed"], 1)
    self.assertEqual(validated["evaluation_protocol_sha256"], evaluator.EVALUATION_PROTOCOL_SHA256)
    self.assertEqual(validated["policy_interface"]["actor_observation_width"], 52)
    self.assertEqual(validated["policy_interface"]["critic_observation_width"], 56)

  def test_checkpoint_rejects_task_seed_artifact_maneuver_and_iteration_mutations(self) -> None:
    mutations = []
    wrong_task = copy.deepcopy(self.envelope)
    wrong_task["training"]["task"] = "HopperTrex-Hybrid-v2-StairCamp"
    mutations.append(wrong_task)
    wrong_seed = copy.deepcopy(self.envelope)
    wrong_seed["training"]["training_seed"] = 2
    mutations.append(wrong_seed)
    wrong_git = copy.deepcopy(self.envelope)
    wrong_git["training"]["git_sha"] = "f" * 40
    mutations.append(wrong_git)
    wrong_contract = copy.deepcopy(self.envelope)
    wrong_contract["training"]["contract_sha256"] = "f" * 64
    mutations.append(wrong_contract)
    wrong_source = copy.deepcopy(self.envelope)
    wrong_source["training"]["source_stage5_gate_sha256"] = "f" * 64
    mutations.append(wrong_source)
    artifact = copy.deepcopy(self.envelope)
    artifact["training"]["artifact_bindings"]["calibration_hash"] = "f" * 64
    mutations.append(artifact)
    maneuver = copy.deepcopy(self.envelope)
    maneuver["training"]["maneuver_sha256"] = "f" * 64
    mutations.append(maneuver)
    iteration = copy.deepcopy(self.envelope)
    iteration["checkpoint_iteration"] = 100
    mutations.append(iteration)
    for candidate in mutations:
      with self.subTest(candidate=candidate), self.assertRaises(ValueError):
        evaluator.validate_checkpoint_envelope(candidate, expectation=_expectation())


  def test_zero_update_migration_has_distinct_honest_envelope_and_request(self) -> None:
    path = self.root / "migrated_zero.pt"
    envelope = _zero_update_envelope(path)
    self.assertEqual(
      envelope["kind"], evaluator.MIGRATION_CHECKPOINT_ENVELOPE_KIND
    )
    self.assertEqual(envelope["runtime_binding"]["completed_updates"], 0)
    self.assertNotIn("training", envelope)
    request = evaluator.make_evaluation_request(
      suite="single-riser",
      checkpoint_envelope=envelope,
      expectation=_expectation(0),
    )
    validated = evaluator.validate_evaluation_request(
      request, expectation=_expectation(0)
    )
    self.assertEqual(
      validated["checkpoint"]["kind"],
      evaluator.MIGRATION_CHECKPOINT_ENVELOPE_KIND,
    )

  def test_zero_update_rejects_nonzero_new_columns_and_fake_training(self) -> None:
    checkpoint = _zero_update_checkpoint()
    checkpoint["actor_state_dict"]["mlp.0.weight"][:, 34] = 1.0
    path = self.root / "bad_zero.pt"
    torch.save(checkpoint, path)
    with self.assertRaisesRegex(ValueError, "added observation columns"):
      evaluator.migration_checkpoint_envelope_from_loaded_checkpoint(
        path, checkpoint, expectation=_expectation(0)
      )
    checkpoint = _zero_update_checkpoint()
    checkpoint["infos"][DYNAMIC_STAIR_TRAINING_INFO_KEY] = _training(1)
    torch.save(checkpoint, path)
    with self.assertRaisesRegex(ValueError, "must not contain training"):
      evaluator.migration_checkpoint_envelope_from_loaded_checkpoint(
        path, checkpoint, expectation=_expectation(0)
      )

  def test_request_rejects_rehashed_protocol_and_gate_mutations(self) -> None:
    request = evaluator.make_evaluation_request(
      suite="single-riser",
      checkpoint_envelope=self.envelope,
      expectation=_expectation(),
    )
    protocol = copy.deepcopy(request)
    protocol["protocol"]["stable_steps"] = 24
    protocol["request_sha256"] = evaluator._canonical_sha256(
      {key: value for key, value in protocol.items() if key != "request_sha256"}
    )
    with self.assertRaisesRegex(ValueError, "protocol drifted"):
      evaluator.validate_evaluation_request(protocol)
    gate = copy.deepcopy(request)
    gate["gate_bindings"]["standing_gate_passed"]["steps"] = 2999
    gate["request_sha256"] = evaluator._canonical_sha256(
      {key: value for key, value in gate.items() if key != "request_sha256"}
    )
    with self.assertRaisesRegex(ValueError, "gate bindings"):
      evaluator.validate_evaluation_request(gate)

  def test_formal_request_requires_complete_expectation_and_verified_file(self) -> None:
    with self.assertRaisesRegex(ValueError, "missing exact"):
      evaluator.make_evaluation_request(
        suite="single-riser",
        checkpoint_envelope=self.envelope,
        expectation=evaluator.CheckpointExpectation(git_sha=_GIT_SHA),
      )
    missing = copy.deepcopy(self.envelope)
    missing["checkpoint_file"] = str(self.root / "missing.pt")
    with self.assertRaisesRegex(ValueError, "does not exist"):
      evaluator.make_evaluation_request(
        suite="single-riser",
        checkpoint_envelope=missing,
        expectation=_expectation(),
      )



class TrialAggregationTest(unittest.TestCase):
  def setUp(self) -> None:
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.root = Path(self.temporary.name)
    self.envelope = _verified_envelope(self.root / "model_99.pt")

  def _request(self, suite: str = "single-riser", ablation: str = "full") -> dict[str, object]:
    return evaluator.make_evaluation_request(
      suite=suite,
      checkpoint_envelope=self.envelope,
      expectation=_expectation(),
      ablation=ablation,
    )

  def test_single_riser_44_of_48_passes_and_keeps_every_trial_metric(self) -> None:
    request = self._request()
    protocol = evaluator.SINGLE_RISER_PROTOCOL
    result = evaluator.finalize_collection(
      request,
      _collection(request, _trials(protocol, failures_per_height=4)),
    )
    self.assertTrue(result["primary_gate_passed"])
    self.assertEqual([row["successes"] for row in result["rows"]], [44, 44, 44])
    self.assertEqual(result["rows"][0]["mode_counts"], {"ROLL": 0, "DYNAMIC": 44, "ABORT": 4})
    trial = result["trials"][4]
    self.assertEqual(trial["lead_side"], "LEFT")
    self.assertEqual(trial["left_trigger_time_s"], 0.4)
    self.assertEqual(tuple(trial["phase_durations_s"]), evaluator.PHASE_NAMES)
    self.assertEqual(trial["step_recovery_times_s"], [0.5])
    self.assertEqual(result["single_seed_status"], "provisional")
    self.assertFalse(result["promotion_claim_eligible"])
    self.assertEqual(evaluator.validate_evaluation_result(result), result)

  def test_dynamic_trigger_contract_depends_on_honest_lift_mode(self) -> None:
    request = self._request()
    protocol = evaluator.SINGLE_RISER_PROTOCOL

    synchronized = _trials(protocol, lift_mode="synchronized")
    synchronized[0]["right_trigger_time_s"] = 0.6
    synchronized[1]["lead_side"] = "RIGHT"
    synchronized[1]["left_trigger_time_s"] = None
    synchronized[1]["right_trigger_time_s"] = 0.4
    result = evaluator.finalize_collection(
      request, _collection(request, synchronized)
    )
    self.assertEqual(result["trials"][0]["lift_mode"], "synchronized")
    self.assertEqual(result["trials"][0]["right_trigger_time_s"], 0.6)
    self.assertIsNone(result["trials"][2]["right_trigger_time_s"])

    alternating = _trials(protocol)
    alternating[0]["right_trigger_time_s"] = 0.4
    result = evaluator.finalize_collection(
      request, _collection(request, alternating)
    )
    self.assertEqual(result["trials"][0]["lift_mode"], "alternating")
    self.assertEqual(result["trials"][0]["left_trigger_time_s"], 0.4)
    self.assertEqual(result["trials"][0]["right_trigger_time_s"], 0.4)

  def test_dynamic_trigger_contract_rejects_mode_specific_mutations(self) -> None:
    request = self._request()
    protocol = evaluator.SINGLE_RISER_PROTOCOL

    cases: list[tuple[str, list[dict[str, object]], str]] = []
    missing_alternating_trail = _trials(protocol)
    missing_alternating_trail[0]["right_trigger_time_s"] = None
    cases.append(
      ("alternating missing trail", missing_alternating_trail, "both observed")
    )

    reversed_alternating = _trials(protocol)
    reversed_alternating[0]["left_trigger_time_s"] = 0.7
    cases.append(
      ("alternating reversed", reversed_alternating, "cannot precede")
    )

    missing_synchronized_lead = _trials(protocol, lift_mode="synchronized")
    missing_synchronized_lead[0]["left_trigger_time_s"] = None
    missing_synchronized_lead[0]["right_trigger_time_s"] = 0.6
    cases.append(
      ("synchronized missing lead", missing_synchronized_lead, "lead trigger")
    )

    early_synchronized_other = _trials(protocol, lift_mode="synchronized")
    early_synchronized_other[0]["right_trigger_time_s"] = 0.3
    cases.append(
      ("synchronized early other", early_synchronized_other, "cannot precede")
    )

    invalid_lift_mode = _trials(protocol)
    invalid_lift_mode[0]["lift_mode"] = "fabricated"
    cases.append(("invalid lift mode", invalid_lift_mode, "lift mode"))

    for name, trials, error in cases:
      with self.subTest(name=name), self.assertRaisesRegex(ValueError, error):
        evaluator.finalize_collection(request, _collection(request, trials))

  def test_continuous_three_riser_contract_and_primary_only_blocking(self) -> None:
    request = self._request("continuous-stairs")
    protocol = evaluator.CONTINUOUS_STAIRS_PROTOCOL
    trials = _trials(protocol)
    # 2/3 cm are capability rows, so make 2 cm fail without blocking 1 cm.
    for trial in trials:
      if trial["height_m"] == 0.02 and trial["env_index"] < 5 and trial["repeat_index"] == 0:
        replacement = _trial(
          protocol,
          height=0.02,
          env_index=int(trial["env_index"]),
          repeat_index=0,
          success=False,
        )
        trial.clear()
        trial.update(replacement)
    result = evaluator.finalize_collection(request, _collection(request, trials))
    self.assertTrue(result["result_passed"])
    self.assertTrue(result["rows"][0]["passed"])
    self.assertFalse(result["rows"][1]["passed"])
    self.assertEqual(result["trials"][0]["steps_completed"], 3)
    self.assertEqual(result["trials"][0]["step_recovery_times_s"], [0.5, 0.5, 0.5])

  def test_trial_matrix_and_schema_mutations_fail_closed(self) -> None:
    request = self._request()
    protocol = evaluator.SINGLE_RISER_PROTOCOL
    trials = _trials(protocol)
    with self.assertRaisesRegex(ValueError, "exactly 144"):
      evaluator.finalize_collection(request, _collection(request, trials[:-1]))
    duplicate = copy.deepcopy(trials)
    duplicate[-1]["height_m"] = 0.01
    duplicate[-1]["repeat_index"] = 0
    duplicate[-1]["env_index"] = 0
    with self.assertRaisesRegex(ValueError, "Duplicate"):
      evaluator.finalize_collection(request, _collection(request, duplicate))
    missing_phase = copy.deepcopy(trials)
    missing_phase[0]["phase_durations_s"].pop("RECOVER")
    with self.assertRaisesRegex(ValueError, "phase duration schema"):
      evaluator.finalize_collection(request, _collection(request, missing_phase))

  def test_trigger_safety_limits_and_nan_mutations_fail_closed(self) -> None:
    request = self._request()
    protocol = evaluator.SINGLE_RISER_PROTOCOL
    cases = []
    wrong_lead = _trials(protocol)
    wrong_lead[0]["lead_side"] = "RIGHT"
    cases.append(wrong_lead)
    dangerous_success = _trials(protocol)
    dangerous_success[0]["terminated"] = True
    cases.append(dangerous_success)
    feedforward = _trials(protocol)
    feedforward[0]["feedforward_max_abs_rad"] = 0.071
    cases.append(feedforward)
    leg = _trials(protocol)
    leg[0]["leg_ppo_max_abs"] = 0.036
    cases.append(leg)
    nan_case = _trials(protocol)
    nan_case[0]["peak_abs_pitch_rad"] = float("nan")
    cases.append(nan_case)
    for trials in cases:
      with self.subTest(), self.assertRaises(ValueError):
        evaluator.finalize_collection(request, _collection(request, trials))

  def test_each_ablation_enforces_its_zeroed_channels(self) -> None:
    protocol = evaluator.SINGLE_RISER_PROTOCOL
    for name in evaluator.ABLATION_ORDER:
      request = self._request(ablation=name)
      result = evaluator.finalize_collection(
        request,
        _collection(request, _trials(protocol, ablation=name)),
      )
      self.assertTrue(result["primary_gate_passed"], name)
    request = self._request(ablation="policy-only")
    invalid = _trials(protocol, ablation="policy-only")
    invalid[0]["feedforward_max_abs_rad"] = 0.01
    with self.assertRaisesRegex(ValueError, "zero stair feedforward"):
      evaluator.finalize_collection(request, _collection(request, invalid))
    request = self._request(ablation="leg-PPO-off")
    invalid = _trials(protocol, ablation="leg-PPO-off")
    invalid[0]["leg_ppo_max_abs"] = 0.01
    with self.assertRaisesRegex(ValueError, "zero leg PPO"):
      evaluator.finalize_collection(request, _collection(request, invalid))


class GateReuseTest(unittest.TestCase):
  def setUp(self) -> None:
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    root = Path(self.temporary.name)
    envelope = _verified_envelope(root / "model_99.pt")
    self.request = evaluator.make_evaluation_request(
      suite=evaluator.RETENTION_SUITE,
      checkpoint_envelope=envelope,
      expectation=_expectation(),
    )

  def test_four_formal_gates_use_existing_normalizer(self) -> None:
    result = evaluator.finalize_collection(
      self.request,
      {
        "request_sha256": self.request["request_sha256"],
        "evaluation_source": "fake_gate_hook",
        "adapter_metadata": {},
        "gates": _gate_rows(),
      },
    )
    self.assertTrue(result["all_gates_passed"])
    self.assertEqual(tuple(result["gate_booleans"]), evaluator.GATE_NAMES)
    for row in result["gates"]:
      self.assertEqual(row["binding"], evaluator.GATE_BINDINGS[row["name"]].to_dict())

  def test_gate_failure_and_false_positive_are_not_reinterpreted(self) -> None:
    rows = _gate_rows(failed_gate="standing_gate_passed")
    rows[-1]["stair_mode_false_positives"] = 1
    result = evaluator.finalize_collection(
      self.request,
      {
        "request_sha256": self.request["request_sha256"],
        "evaluation_source": "fake_gate_hook",
        "adapter_metadata": {},
        "gates": rows,
      },
    )
    self.assertFalse(result["result_passed"])
    self.assertFalse(result["gate_booleans"]["standing_gate_passed"])
    self.assertFalse(result["gate_booleans"]["stage5_gate_passed"])

  def test_retention_rejects_non_full_ablation(self) -> None:
    with self.assertRaisesRegex(ValueError, "only full"):
      evaluator.make_evaluation_request(
        suite=evaluator.RETENTION_SUITE,
        checkpoint_envelope=self.request["checkpoint"],
        expectation=_expectation(),
        ablation="wheel-PPO-off",
      )


class K3SelectionTest(unittest.TestCase):
  def test_newest_passing_checkpoint_is_selected(self) -> None:
    candidates = [
      _k3_candidate(51, passed=True),
      _k3_candidate(76, passed=True),
      _k3_candidate(100, passed=False),
    ]
    result = evaluator.select_newest_passing_checkpoint(candidates)
    self.assertEqual(result["classification"], "STAIR_DYNAMIC_CHECKPOINT_SELECTED")
    self.assertEqual(result["ordered_candidates"][0]["completed_updates"], 100)
    self.assertEqual(
      result["selected_checkpoint"]["training"]["completed_updates"], 76
    )

  def test_all_failed_is_valid_stop(self) -> None:
    result = evaluator.select_newest_passing_checkpoint(
      [_k3_candidate(update, passed=False) for update in (51, 76, 100)]
    )
    self.assertEqual(result["classification"], "STOP_DYNAMIC_STAIR_UNQUALIFIED")
    self.assertIsNone(result["selected_checkpoint"])

  def test_wrong_save_set_duplicate_hash_and_binding_drift_are_rejected(self) -> None:
    with self.assertRaisesRegex(ValueError, "latest three"):
      evaluator.select_newest_passing_checkpoint(
        [_k3_candidate(update) for update in (50, 75, 100)]
      )
    duplicate = [_k3_candidate(update) for update in (51, 76, 100)]
    duplicate[1]["checkpoint"]["checkpoint_file_sha256"] = duplicate[0]["checkpoint"][
      "checkpoint_file_sha256"
    ]
    with self.assertRaises(ValueError):
      evaluator.select_newest_passing_checkpoint(duplicate)
    drift = [
      _k3_candidate(51),
      _k3_candidate(76),
      evaluator.make_k3_screen_candidate(
        checkpoint_envelope=_synthetic_envelope(100, contract_sha="f" * 64),
        budget_updates=100,
        gate_passes={name: True for name in evaluator.GATE_NAMES},
        gate_stair_mode_false_positives={name: 0 for name in evaluator.GATE_NAMES},
        height_row={
          "height_m": 0.01,
          "trials": 16,
          "successes": 15,
          "terminations": 0,
          "non_wheel_contacts": 0,
          "stair_mode_false_positives": 0,
        },
      ),
    ]
    with self.assertRaisesRegex(ValueError, "contract_sha256"):
      evaluator.select_newest_passing_checkpoint(drift)

  def test_screen_requires_four_gates_zero_fp_and_15_of_16(self) -> None:
    candidate = _k3_candidate(100)
    self.assertTrue(candidate["screen_passed"])
    gates = {name: True for name in evaluator.GATE_NAMES}
    fps = {name: 0 for name in evaluator.GATE_NAMES}
    fps["velocity_gate_passed"] = 1
    candidate = evaluator.make_k3_screen_candidate(
      checkpoint_envelope=_synthetic_envelope(100),
      budget_updates=100,
      gate_passes=gates,
      gate_stair_mode_false_positives=fps,
      height_row={
        "height_m": 0.01,
        "trials": 16,
        "successes": 16,
        "terminations": 0,
        "non_wheel_contacts": 0,
        "stair_mode_false_positives": 0,
      },
    )
    self.assertFalse(candidate["screen_passed"])




class ExtensionAuthorizationTest(unittest.TestCase):
  def setUp(self) -> None:
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    root = Path(self.temporary.name)
    candidates = []
    for updates in (51, 76, 100):
      envelope = _verified_envelope(root / f"model_{updates - 1}.pt", updates)
      candidates.append(
        evaluator.make_k3_screen_candidate(
          checkpoint_envelope=envelope,
          budget_updates=100,
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
    self.selection = evaluator.select_newest_passing_checkpoint(candidates)
    selected = self.selection["selected_checkpoint"]
    self.retention_request = evaluator.make_evaluation_request(
      suite=evaluator.RETENTION_SUITE,
      checkpoint_envelope=selected,
      expectation=_expectation(100),
    )
    self.stairs_request = evaluator.make_evaluation_request(
      suite="single-riser",
      checkpoint_envelope=selected,
      expectation=_expectation(100),
    )

  def _retention(self, *, failed_gate: str | None = None):
    return evaluator.finalize_collection(
      self.retention_request,
      {
        "request_sha256": self.retention_request["request_sha256"],
        "evaluation_source": "fake_live_hook",
        "adapter_metadata": {},
        "gates": _gate_rows(failed_gate=failed_gate),
      },
    )

  def _stairs(self, *, failures: int = 0):
    return evaluator.finalize_collection(
      self.stairs_request,
      _collection(
        self.stairs_request,
        _trials(evaluator.SINGLE_RISER_PROTOCOL, failures_per_height=failures),
      ),
    )

  def test_authorization_binds_selected_checkpoint_and_formal_predicates(self) -> None:
    authorization = evaluator.make_extension_authorization(
      k3_selection=self.selection,
      retention_result=self._retention(),
      single_riser_result=self._stairs(failures=4),
    )
    validated = evaluator.validate_extension_authorization(authorization)
    self.assertEqual(validated["selected_completed_updates"], 100)
    self.assertEqual(validated["target_total_updates"], 500)
    self.assertEqual(len(validated["authorization_sha256"]), 64)

  def test_authorization_rejects_gate_failure_under_44_and_cross_checkpoint(self) -> None:
    with self.assertRaisesRegex(ValueError, "four formal"):
      evaluator.make_extension_authorization(
        k3_selection=self.selection,
        retention_result=self._retention(failed_gate="standing_gate_passed"),
        single_riser_result=self._stairs(failures=4),
      )
    with self.assertRaisesRegex(ValueError, "44/48"):
      evaluator.make_extension_authorization(
        k3_selection=self.selection,
        retention_result=self._retention(),
        single_riser_result=self._stairs(failures=5),
      )
    other = _verified_envelope(
      Path(self.temporary.name) / "other_model.pt", 100
    )
    request = evaluator.make_evaluation_request(
      suite="single-riser", checkpoint_envelope=other, expectation=_expectation(100)
    )
    cross = evaluator.finalize_collection(
      request, _collection(request, _trials(evaluator.SINGLE_RISER_PROTOCOL))
    )
    with self.assertRaisesRegex(ValueError, "selected checkpoint"):
      evaluator.make_extension_authorization(
        k3_selection=self.selection,
        retention_result=self._retention(),
        single_riser_result=cross,
      )


class AblationBundleAndOutputTest(unittest.TestCase):
  def setUp(self) -> None:
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.root = Path(self.temporary.name)
    self.envelope = _verified_envelope(self.root / "model_99.pt")

  def _ablation_results(self) -> list[dict[str, object]]:
    results = []
    protocol = evaluator.CONTINUOUS_STAIRS_PROTOCOL
    for name in evaluator.ABLATION_ORDER:
      request = evaluator.make_evaluation_request(
        suite="continuous-stairs",
        checkpoint_envelope=self.envelope,
        expectation=_expectation(),
        ablation=name,
      )
      results.append(
        evaluator.finalize_collection(
          request,
          _collection(request, _trials(protocol, ablation=name)),
        )
      )
    return results

  def test_ablation_bundle_requires_exact_six_for_same_checkpoint(self) -> None:
    results = self._ablation_results()
    bundle = evaluator.make_ablation_bundle(list(reversed(results)))
    self.assertEqual(bundle["completed_ablations"], list(evaluator.ABLATION_ORDER))
    self.assertEqual(bundle["single_seed_status"], "provisional")
    self.assertFalse(bundle["promotion_claim_eligible"])
    with self.assertRaisesRegex(ValueError, "exactly six"):
      evaluator.make_ablation_bundle(results[:-1])
    duplicate = list(results)
    duplicate[-1] = results[0]
    with self.assertRaisesRegex(ValueError, "Duplicate"):
      evaluator.make_ablation_bundle(duplicate)

  def test_ablation_bundle_rejects_cross_checkpoint_mix(self) -> None:
    results = self._ablation_results()
    other_envelope = _verified_envelope(self.root / "other_model_99.pt")
    request = evaluator.make_evaluation_request(
      suite="continuous-stairs",
      checkpoint_envelope=other_envelope,
      expectation=_expectation(),
      ablation="full",
    )
    replacement = evaluator.finalize_collection(
      request,
      _collection(
        request,
        _trials(evaluator.CONTINUOUS_STAIRS_PROTOCOL, ablation="full"),
      ),
    )
    full_index = evaluator.ABLATION_ORDER.index("full")
    results[full_index] = replacement
    with self.assertRaisesRegex(ValueError, "disagree"):
      evaluator.make_ablation_bundle(results)

  def test_atomic_json_never_overwrites_and_cli_manifest_is_strict_json(self) -> None:
    output = self.root / "nested" / "result.json"
    evaluator.write_machine_output(evaluator.manifest_payload(), output)
    self.assertEqual(
      json.loads(output.read_text(encoding="utf-8"))["task"],
      DYNAMIC_STAIR_TASK_ID,
    )
    with self.assertRaises(FileExistsError):
      evaluator.write_machine_output(evaluator.manifest_payload(), output)
    stream = io.StringIO()
    with redirect_stdout(stream):
      return_code = evaluator.main(["manifest"])
    self.assertEqual(return_code, 0)
    payload = json.loads(stream.getvalue())
    self.assertEqual(payload["kind"], "stair_dynamic_evaluator_manifest")


if __name__ == "__main__":
  unittest.main()
