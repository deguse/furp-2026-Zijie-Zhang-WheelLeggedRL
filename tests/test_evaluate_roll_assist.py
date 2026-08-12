import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from mjlab.tasks.registry import load_env_cfg

import hoppertrex_mjlab.tasks  # noqa: F401
from hoppertrex_mjlab.hybrid.roll_assist import (
  ROLL_ASSIST_ACTION_SCALES,
  ROLL_ASSIST_TASK_ID,
  canonical_json_sha256,
)
from hoppertrex_mjlab.scripts.adjudicate_roll_assist import (
  adjudicate_continuation,
  select_k3,
  validate_k3_selection,
)
from hoppertrex_mjlab.scripts.evaluate_roll_assist import (
  CHECKPOINT_KIND,
  _roll_assist_stage5_env_cfg,
  _trials_by_pair,
  checkpoint_envelope,
  screen_checkpoint_envelope,
  validate_checkpoint_envelope,
)


class RollAssistEvaluatorTest(unittest.TestCase):
  def _checkpoint(self, path: Path) -> None:
    curriculum = {
      "schema_version": 2,
      "hpass_m": 0.005,
      "hnext_m": 0.0075,
      "switched_to_hnext": True,
      "decision_made": True,
      "update25_success_rate": 0.9,
      "update25_safe": True,
      "completed_stair_episodes": 20,
      "successful_stair_episodes": 18,
      "termination_episodes": 0,
      "non_wheel_contact_episodes": 0,
      "bilateral_airborne_episodes": 0,
      "flat_env_count": 64,
      "last_processed_step": 2400,
    }
    progress = {
      key: value
      for key, value in curriculum.items()
      if key not in ("flat_env_count", "last_processed_step")
    }
    progress["active_height_m"] = 0.0075
    progress["online_success_rate"] = 0.9
    torch.save({
      "iter": 99,
      "infos": {
        "roll_assist_training": {
          "schema_version": 1,
          "task": ROLL_ASSIST_TASK_ID,
          "training_seed": 1,
          "git_sha": "a" * 40,
          "r0_sha256": "b" * 64,
          "reward_calibration_sha256": "c" * 64,
          "action_scales": list(ROLL_ASSIST_ACTION_SCALES),
          "wheel_residual_exact_zero": True,
          "zero_initialized_deterministic_mean": True,
          "update25_curriculum_decided": True,
          "active_height_m": 0.0075,
          "completed_updates": 100,
        },
        "roll_assist_curriculum": curriculum,
        "roll_assist_progress": progress,
        "env_state": {"common_step_counter": 2400},
      },
    }, path)

  def test_checkpoint_envelope_is_byte_and_hash_bound(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "model_99.pt"
      self._checkpoint(path)
      from unittest.mock import patch
      with patch("hoppertrex_mjlab.scripts.evaluate_roll_assist._git_sha", return_value="a" * 40):
        envelope = checkpoint_envelope(path)
        self.assertEqual(envelope["kind"], CHECKPOINT_KIND)
        self.assertEqual(envelope["completed_updates"], 100)
        self.assertEqual(validate_checkpoint_envelope(envelope), envelope)
        runtime_drift = copy.deepcopy(envelope)
        runtime_drift["progress"]["active_height_m"] = 0.005
        unsigned = dict(runtime_drift)
        unsigned.pop("envelope_sha256")
        runtime_drift["envelope_sha256"] = canonical_json_sha256(unsigned)
        with self.assertRaises(ValueError):
          validate_checkpoint_envelope(runtime_drift, verify_file=False)
        drifted = copy.deepcopy(envelope)
        drifted["completed_updates"] = 75
        with self.assertRaises(ValueError):
          validate_checkpoint_envelope(drifted, verify_file=False)

  def test_full_formal_evaluator_envelope_drives_continuation_and_stops_after_final_pass(self):
    baseline = [index / 10_000 for index in range(96)]
    candidate = [value + 0.01 for value in baseline]
    continuation = {
      "authorized": True,
      "classification": "ROLL_ASSIST_EXTEND_BLOCK",
      "checks": {
        "flat_retention": True, "hpass_retained": True, "hnext_safe": True,
        "wheel_residual_exact_zero": True, "positive_success_evidence": True,
        "paired_progress_positive": True,
      },
      "paired_progress_bootstrap": {
        "pairs": 96, "samples": 10000, "seed": 1,
        "mean_delta_m": 0.01, "lower_95_m": 0.01, "upper_95_m": 0.01,
      },
    }
    evidence = {
      "schema_version": 1, "kind": "roll_assist_evaluation",
      "profile": "formal", "evidence_eligible": True,
      "task": ROLL_ASSIST_TASK_ID, "git_sha": "a" * 40,
      "checkpoint": {}, "roll_boundary_file": "r0.json",
      "roll_boundary_sha256": "b" * 64, "hpass_m": 0.005,
      "hnext_m": 0.0075, "stage5_retention": {},
      "flat_c1_diagnostic": {}, "hpass_candidate": {},
      "hnext_candidate": {}, "hnext_zero_residual": {},
      "paired_candidate_progress": candidate,
      "paired_baseline_progress": baseline,
      "wheel_residual_abs_max": 0.0, "hnext_unsafe": {},
      "screen": None, "continuation": continuation,
      "final": {"passed": False, "classification": "ROLL_ASSIST_NO_EXPANSION"},
      "recovery_claim": {
        "eligible": False, "classification": "RECOVERY_CLAIM_NOT_EVALUATED",
        "reason": "paired_recovery_time_bootstrap_not_implemented",
      },
    }
    evidence["evaluation_sha256"] = canonical_json_sha256(evidence)
    self.assertEqual(adjudicate_continuation(evidence), continuation)
    stopped = copy.deepcopy(evidence)
    stopped["final"] = {
      "passed": True, "classification": "ROLL_ASSIST_BOUNDARY_EXPANDED"
    }
    stopped.pop("evaluation_sha256")
    stopped["evaluation_sha256"] = canonical_json_sha256(stopped)
    with self.assertRaisesRegex(ValueError, "stop immediately"):
      adjudicate_continuation(stopped)


  def test_stair_evaluator_enforces_episode_wide_safety(self):
    source = Path(__import__(
      "hoppertrex_mjlab.scripts.evaluate_roll_assist", fromlist=["x"]
    ).__file__).read_text(encoding="utf-8")
    start = source.index("def _evaluate_stair_arm")
    call = source.index("run_card_repeat(", start)
    self.assertIn(
      "episode_wide_safety=True", source[call:source.index("))", call) + 2]
    )

  def test_flat_evaluator_refreshes_observation_after_forcing_command(self):
    source = Path(__import__(
      "hoppertrex_mjlab.scripts.evaluate_roll_assist", fromlist=["x"]
    ).__file__).read_text(encoding="utf-8")
    start = source.index("def _run_flat_cell")
    force = source.index("_force_commands(env, vx=vx", start)
    refresh = source.index("observation = env.get_observations()", force)
    policy = source.index("actions = policy(observation)", refresh)
    step = source.index("env.step(actions)", policy)
    self.assertLess(force, refresh)
    self.assertLess(refresh, policy)
    self.assertLess(policy, step)
  def test_stage5_retention_mutator_grants_legs_only(self):
    cfg = load_env_cfg("HopperTrex-Hybrid-v2-Stage5", play=True)
    _roll_assist_stage5_env_cfg(cfg)
    action = cfg.actions["hybrid_wheel_leg"]
    self.assertEqual(tuple(action.action_mask), (False, False, True, True, True, True))
    self.assertEqual(tuple(action.action_scales), ROLL_ASSIST_ACTION_SCALES)
    self.assertIsNone(action.dynamic_stair_maneuver)
    self.assertIsNone(action.stair_trigger_sensor_name)

  def test_paired_trials_reject_duplicate_keys(self):
    row = {
      "posture_card": "envelope_center", "repeat": 1, "env_id": 0,
      "max_progress_past_face_m": 0.01,
    }
    with self.assertRaises(ValueError):
      _trials_by_pair({"trials": [row, dict(row)]}, {"trials": [row, dict(row)]})
    rows = [
      {**row, "repeat": repeat, "env_id": env_id, "posture_card": card}
      for card in ("envelope_center", "envelope_edge")
      for repeat in range(1, 4)
      for env_id in range(16)
    ]
    candidate, baseline = _trials_by_pair({"trials": rows}, {"trials": rows})
    self.assertEqual((len(candidate), len(baseline)), (96, 96))
  def test_k3_no_passer_selection_preserves_order_and_screen_byte_bindings(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      paths = []
      for updates in (100, 51, 76):
        checkpoint = root / f"model_{updates}.pt"
        checkpoint.write_bytes(f"checkpoint-{updates}".encode())
        digest = __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest()
        path = root / f"u{updates}.json"
        path.write_text(json.dumps({
          "schema_version": 1,
          "kind": "roll_assist_k3_screen",
          "checkpoint_file": str(checkpoint),
          "checkpoint_file_sha256": digest,
          "completed_updates": updates,
          "passed": False,
          "checks": {
            "flat_retention_passed": True,
            "hpass_retained": True,
            "hnext_safe": True,
            "wheel_residual_exact_zero": False,
          },
        }), encoding="utf-8")
        paths.append(path)
      result = select_k3(paths)
      self.assertEqual(result["kind"], "roll_assist_k3_selection")
      self.assertEqual(result["classification"], "ROLL_ASSIST_K3_NO_PASSER")
      self.assertIsNone(result["selected"])
      self.assertEqual(
        [candidate["completed_updates"] for candidate in result["candidates"]],
        [51, 76, 100],
      )
      for candidate in result["candidates"]:
        source = Path(candidate["screen_envelope_file"])
        self.assertEqual(
          candidate["screen_envelope_sha256"],
          __import__("hashlib").sha256(source.read_bytes()).hexdigest(),
        )

  def test_k3_validator_rejects_order_and_screen_byte_drift(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      paths = []
      for updates in (51, 76, 100):
        checkpoint = root / f"model_{updates}.pt"
        checkpoint.write_bytes(f"checkpoint-{updates}".encode())
        digest = __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest()
        path = root / f"u{updates}.json"
        path.write_text(json.dumps({
          "schema_version": 1,
          "kind": "roll_assist_k3_screen",
          "checkpoint_file": str(checkpoint),
          "checkpoint_file_sha256": digest,
          "completed_updates": updates,
          "passed": False,
          "checks": {
            "flat_retention_passed": True,
            "hpass_retained": True,
            "hnext_safe": True,
            "wheel_residual_exact_zero": False,
          },
        }), encoding="utf-8")
        paths.append(path)
      selection = select_k3(paths)
      self.assertEqual(validate_k3_selection(selection), selection)
      reordered = copy.deepcopy(selection)
      reordered["candidates"] = list(reversed(reordered["candidates"]))
      with self.assertRaisesRegex(ValueError, "update-ordered"):
        validate_k3_selection(reordered)
      first_checkpoint = Path(selection["candidates"][0]["checkpoint_file"])
      original_checkpoint = first_checkpoint.read_bytes()
      first_checkpoint.write_bytes(original_checkpoint + b"drift")
      with self.assertRaisesRegex(ValueError, "checkpoint bytes drifted"):
        validate_k3_selection(selection)
      first_checkpoint.write_bytes(original_checkpoint)
      paths[0].write_text(paths[0].read_text(encoding="utf-8") + "\n", encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "screen-envelope bytes drifted"):
        validate_k3_selection(selection)

  def test_screen_envelope_is_rejection_only(self):
    envelope = {
      "schema_version": 1,
      "kind": CHECKPOINT_KIND,
      "checkpoint_file": str(Path("model_99.pt").resolve()),
      "checkpoint_file_sha256": "d" * 64,
      "completed_updates": 100,
      "iteration": 99,
      "training": {
        "schema_version": 1, "task": ROLL_ASSIST_TASK_ID,
        "training_seed": 1, "git_sha": "a" * 40,
        "r0_sha256": "b" * 64, "reward_calibration_sha256": "c" * 64,
        "action_scales": list(ROLL_ASSIST_ACTION_SCALES),
        "wheel_residual_exact_zero": True,
        "zero_initialized_deterministic_mean": True,
        "update25_curriculum_decided": True,
        "active_height_m": 0.0075,
        "completed_updates": 100,
      },
      "curriculum": {"decision_made": True},
      "progress": {"active_height_m": 0.0075},
      "common_step_counter": 2400,
    }
    envelope["envelope_sha256"] = canonical_json_sha256(envelope)
    checks = {
      "flat_retention_passed": True,
      "hpass_retained": True,
      "hnext_safe": True,
      "wheel_residual_exact_zero": True,
    }
    full_screen = {
      "kind": "roll_assist_evaluation", "profile": "screen",
      "evidence_eligible": False, "checkpoint": envelope, "screen": checks,
    }
    from unittest.mock import patch
    with patch("hoppertrex_mjlab.scripts.evaluate_roll_assist._git_sha", return_value="a" * 40):
      result = screen_checkpoint_envelope(envelope, checks)
      self.assertEqual(screen_checkpoint_envelope(envelope, full_screen), result)
    self.assertTrue(result["passed"])
    self.assertNotIn("score", result)


if __name__ == "__main__":
  unittest.main()
