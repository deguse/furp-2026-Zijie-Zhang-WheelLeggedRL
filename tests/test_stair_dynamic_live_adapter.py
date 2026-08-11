"""Pure mock tests for the StairDynamic live adapter seam."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hoppertrex_mjlab.hybrid.stair_dynamic import DYNAMIC_STAIR_TASK_ID
from hoppertrex_mjlab.hybrid.stair_dynamic_contract import (
  DYNAMIC_STAIR_ACTION_SCALES,
)
from hoppertrex_mjlab.scripts.rsl_rl import evaluate_stair_dynamic as evaluator
from hoppertrex_mjlab.scripts.rsl_rl import stair_dynamic_live_adapter as adapter

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


def _training(updates: int = 100) -> dict[str, object]:
  return {
    "schema_version": 1,
    "task": DYNAMIC_STAIR_TASK_ID,
    "training_seed": 1,
    "git_sha": _GIT_SHA,
    "contract_sha256": _CONTRACT_SHA,
    "artifact_bindings": dict(_ARTIFACTS),
    "action_scales": list(DYNAMIC_STAIR_ACTION_SCALES),
    "maneuver_sha256": _MANEUVER_SHA,
    "source_stage5_checkpoint_sha256": _STAGE5_CHECKPOINT_SHA,
    "source_stage5_gate_sha256": _STAGE5_GATE_SHA,
    "stage5_prefix_preserved_and_new_columns_zero": True,
    "completed_updates": updates,
  }


def _runtime_binding() -> dict[str, object]:
  training = _training(100)
  training.pop("schema_version")
  training["completed_updates"] = 0
  return training


def _migration() -> dict[str, object]:
  std = [0.07, 0.09, 0.05, 0.05, 0.05, 0.05]
  return {
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


def _expectation(updates: int) -> evaluator.CheckpointExpectation:
  return evaluator.CheckpointExpectation(
    git_sha=_GIT_SHA,
    contract_sha256=_CONTRACT_SHA,
    artifact_bindings=dict(_ARTIFACTS),
    maneuver_sha256=_MANEUVER_SHA,
    source_stage5_checkpoint_sha256=_STAGE5_CHECKPOINT_SHA,
    source_stage5_gate_sha256=_STAGE5_GATE_SHA,
    completed_updates=updates,
  )


def _envelope(root: Path, *, migration: bool = False) -> dict[str, object]:
  path = root / ("migration.pt" if migration else "model_99.pt")
  path.write_bytes(b"migration-checkpoint" if migration else b"trained-checkpoint")
  common = {
    "schema_version": evaluator.EVALUATOR_SCHEMA_VERSION,
    "checkpoint_file": str(path.resolve()),
    "checkpoint_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
  }
  if migration:
    value = {
      **common,
      "kind": evaluator.MIGRATION_CHECKPOINT_ENVELOPE_KIND,
      "checkpoint_iteration": 0,
      "migration": _migration(),
      "runtime_binding": _runtime_binding(),
    }
    return evaluator.validate_migration_checkpoint_envelope(
      value, expectation=_expectation(0), verify_file=True
    )
  value = {
    **common,
    "kind": evaluator.CHECKPOINT_ENVELOPE_KIND,
    "checkpoint_iteration": 99,
    "training": _training(),
  }
  return evaluator.validate_checkpoint_envelope(
    value, expectation=_expectation(100), verify_file=True
  )


def _request(
  root: Path,
  *,
  suite: str = "single-riser",
  ablation: str = "full",
  migration: bool = False,
) -> dict[str, object]:
  return evaluator.make_evaluation_request(
    suite=suite,
    checkpoint_envelope=_envelope(root, migration=migration),
    expectation=_expectation(0 if migration else 100),
    ablation=ablation,
    device="cpu",
  )


def _trial(
  protocol: evaluator.StairEvaluationProtocol,
  descriptor: evaluator.AblationDescriptor,
  height: float,
  repeat: int,
  env_index: int,
  *,
  lift_mode: str = "alternating",
) -> dict[str, object]:
  roll_only = descriptor.force_stair_request_false
  zero = set(descriptor.zero_action_indices)
  wheel_zero = {0, 1}.issubset(zero)
  leg_zero = {2, 3, 4, 5}.issubset(zero)
  return {
    "height_m": height,
    "env_index": env_index,
    "repeat_index": repeat,
    "success": True,
    "traversal_mode": "ROLL" if roll_only else "DYNAMIC",
    "lift_mode": lift_mode,
    "lead_side": "NONE" if roll_only else "LEFT",
    "left_trigger_time_s": None if roll_only else 0.4,
    "right_trigger_time_s": (
      None if roll_only or lift_mode == "synchronized" else 0.6
    ),
    "phase_durations_s": {
      phase: 0.1 for phase in evaluator.PHASE_NAMES
    },
    "wheel_ppo_rms": 0.0 if wheel_zero else 0.08,
    "wheel_ppo_max_abs": 0.0 if wheel_zero else 0.12,
    "leg_ppo_rms": 0.0 if leg_zero else 0.01,
    "leg_ppo_max_abs": 0.0 if leg_zero else 0.02,
    "feedforward_max_abs_rad": 0.0 if descriptor.disable_feedforward else 0.05,
    "peak_abs_pitch_rad": 0.12,
    "peak_abs_roll_rad": 0.03,
    "steps_completed": protocol.risers_per_trial,
    "step_recovery_times_s": [0.5] * protocol.risers_per_trial,
    "stable_steps": protocol.stable_steps,
    "terminated": False,
    "non_wheel_contact": False,
    "abort_reason": None,
  }


class FakeBackend:
  def __init__(self, *, lift_mode: str = "alternating") -> None:
    self.lift_mode = lift_mode
    self.stair_calls: list[tuple[str, str]] = []
    self.stair_protocols = []
    self.gate_calls = []
    self.config: dict[str, object] = {}

  def metadata(self) -> dict[str, object]:
    return {
      "actor_observation_width": 52,
      "critic_observation_width": 56,
      "action_width": 6,
      "stage5_actor_adapter_used": False,
      "mocked": True,
      "completed_calls": len(self.stair_calls) + len(self.gate_calls),
    }

  def run_stair_suite(self, protocol, descriptor):
    self.stair_calls.append((protocol.suite, descriptor.name))
    self.stair_protocols.append(protocol)
    return [
      _trial(
        protocol,
        descriptor,
        height,
        repeat,
        env_index,
        lift_mode=self.lift_mode,
      )
      for height in protocol.heights_m
      for repeat in range(protocol.repeats)
      for env_index in range(protocol.num_envs_per_height)
    ]

  def run_gate(self, request):
    self.gate_calls.append(request)
    return {
      "name": request.name,
      "num_envs": request.num_envs,
      "steps": request.steps,
      "scenario_count": request.scenario_count,
      "kick_events": request.minimum_kick_events,
      "upstream_gate_passed": True,
      "terminations": 0,
      "non_wheel_contacts": 0,
      "stair_mode_false_positives": 0,
    }


class FakeTensor:
  def __init__(self, values: list[list[float]]) -> None:
    self.values = [list(row) for row in values]
    self.shape = (len(values), len(values[0]))

  def clone(self) -> FakeTensor:
    return FakeTensor(self.values)

  def __setitem__(self, key: tuple[object, int], value: float) -> None:
    rows, column = key
    if rows is not Ellipsis:
      raise AssertionError("Unexpected fake tensor index.")
    for row in self.values:
      row[column] = value


class FakeBuffer:
  def __init__(self) -> None:
    self.value = 1.0

  def zero_(self) -> None:
    self.value = 0.0


class LiveAdapterPureTest(unittest.TestCase):
  def setUp(self) -> None:
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.root = Path(self.temporary.name)

  def test_import_does_not_import_torch_mjlab_or_task_registry(self) -> None:
    source = Path(__file__).resolve().parents[1] / "src"
    code = (
      "import json,sys; "
      "import hoppertrex_mjlab.scripts.rsl_rl.stair_dynamic_live_adapter; "
      "print(json.dumps(sorted(name for name in sys.modules if "
      "name == 'torch' or name.startswith('torch.') or name == 'mjlab' "
      "or name.startswith('mjlab.') or name == 'hoppertrex_mjlab.tasks' "
      "or name.startswith('hoppertrex_mjlab.tasks.'))))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(source)
    completed = subprocess.run(
      [sys.executable, "-c", code],
      check=True,
      capture_output=True,
      text=True,
      env=env,
    )
    self.assertEqual(json.loads(completed.stdout), [])

  def test_all_six_stair_ablations_finalize_through_mock_backend(self) -> None:
    for name in evaluator.ABLATION_ORDER:
      with self.subTest(ablation=name):
        backend = FakeBackend()
        request = _request(self.root, ablation=name)
        collection = adapter.collect_with_backend(request, backend)
        self.assertEqual(set(collection), {
          "request_sha256",
          "evaluation_source",
          "adapter_metadata",
          "trials",
        })
        self.assertEqual(backend.stair_calls, [("single-riser", name)])
        self.assertEqual(collection["adapter_metadata"]["completed_calls"], 1)
        result = evaluator.finalize_collection(request, collection)
        self.assertTrue(result["primary_gate_passed"])

  def test_synchronized_collection_keeps_null_unobserved_trail_trigger(self) -> None:
    backend = FakeBackend(lift_mode="synchronized")
    request = _request(self.root)
    collection = adapter.collect_with_backend(request, backend)

    trial = collection["trials"][0]
    self.assertEqual(trial["lift_mode"], "synchronized")
    self.assertEqual(trial["left_trigger_time_s"], 0.4)
    self.assertIsNone(trial["right_trigger_time_s"])
    result = evaluator.finalize_collection(request, collection)
    self.assertTrue(result["primary_gate_passed"])
    self.assertIsNone(result["trials"][0]["right_trigger_time_s"])

  def test_retention_reuses_exact_four_gate_requests_and_normalizer(self) -> None:
    backend = FakeBackend()
    request = _request(self.root, suite=evaluator.RETENTION_SUITE)
    collection = adapter.collect_with_backend(request, backend)
    self.assertEqual(
      tuple(item.name for item in backend.gate_calls), evaluator.GATE_NAMES
    )
    self.assertEqual(collection["adapter_metadata"]["completed_calls"], 4)
    result = evaluator.finalize_collection(request, collection)
    self.assertTrue(result["all_gates_passed"])

  def test_zero_update_migration_request_stays_distinct_and_collects(self) -> None:
    backend = FakeBackend()
    request = _request(self.root, migration=True)
    checkpoint = request["checkpoint"]
    self.assertEqual(
      checkpoint["kind"], evaluator.MIGRATION_CHECKPOINT_ENVELOPE_KIND
    )
    self.assertNotIn("training", checkpoint)
    self.assertEqual(adapter._checkpoint_binding(checkpoint)["completed_updates"], 0)
    collection = adapter.collect_with_backend(request, backend)
    self.assertEqual(len(collection["trials"]), 144)

  def test_loaded_info_validation_dispatches_without_fake_training(self) -> None:
    trained = _envelope(self.root)
    adapter._validate_loaded_checkpoint_infos(
      trained, {"stair_dynamic_training": _training()}
    )
    bad_training = _training()
    bad_training["completed_updates"] = 101
    with self.assertRaisesRegex(RuntimeError, "differs"):
      adapter._validate_loaded_checkpoint_infos(
        trained, {"stair_dynamic_training": bad_training}
      )

    migration = _envelope(self.root, migration=True)
    adapter._validate_loaded_checkpoint_infos(
      migration, {"stair_dynamic_migration": _migration()}
    )
    with self.assertRaisesRegex(RuntimeError, "fabricated"):
      adapter._validate_loaded_checkpoint_infos(
        migration,
        {
          "stair_dynamic_migration": _migration(),
          "stair_dynamic_training": _training(),
        },
      )
    changed = _migration()
    changed["created_at"] = "changed"
    with self.assertRaisesRegex(RuntimeError, "differs"):
      adapter._validate_loaded_checkpoint_infos(
        migration, {"stair_dynamic_migration": changed}
      )

  def test_collect_loads_heavy_dependencies_only_after_validation(self) -> None:
    request = _request(self.root)
    backend = FakeBackend()
    with (
      patch.object(adapter, "_load_live_dependencies", return_value="deps") as load,
      patch.object(adapter, "_DynamicMjLabBackend", return_value=backend) as cls,
    ):
      collection = adapter.collect(request)
    self.assertEqual(collection["adapter_metadata"]["completed_calls"], 1)
    load.assert_called_once_with()
    cls.assert_called_once()

    invalid = copy.deepcopy(request)
    invalid["task"] = "wrong-task"
    with (
      patch.object(adapter, "_load_live_dependencies") as forbidden,
      self.assertRaises(ValueError),
    ):
      adapter.collect(invalid)
    forbidden.assert_not_called()


  def test_k3_is_smoke_gates_plus_exact_rejection_only_height_screen(self) -> None:
    backend = FakeBackend()
    checkpoint = _envelope(self.root)
    candidate = adapter.collect_k3_with_backend(
      checkpoint, 100, "cpu", backend
    )
    self.assertEqual(candidate["kind"], evaluator.K3_SCREEN_KIND)
    self.assertEqual(candidate["profile"], "screen")
    self.assertFalse(candidate["evidence_eligible"])
    self.assertTrue(candidate["screen_passed"])
    self.assertEqual(
      tuple(request.name for request in backend.gate_calls), evaluator.GATE_NAMES
    )
    self.assertTrue(
      all(request.profile == "smoke" for request in backend.gate_calls)
    )
    protocol = backend.stair_protocols[0]
    self.assertEqual(protocol.profile, "screen")
    self.assertFalse(protocol.evidence_eligible)
    self.assertEqual(protocol.heights_m, (0.01,))
    self.assertEqual(protocol.num_envs_per_height, 16)
    self.assertEqual(protocol.repeats, 1)
    self.assertEqual(candidate["height_screen"]["trials"], 16)

  def test_k3_is_rejection_only_and_rejects_migration_or_bad_budget(self) -> None:
    class FailedGateBackend(FakeBackend):
      def run_gate(self, request):
        outcome = super().run_gate(request)
        if request.name == "velocity_gate_passed":
          outcome["upstream_gate_passed"] = False
        return outcome

    checkpoint = _envelope(self.root)
    candidate = adapter.collect_k3_with_backend(
      checkpoint, 100, "cpu", FailedGateBackend()
    )
    self.assertFalse(candidate["screen_passed"])
    with self.assertRaises(ValueError):
      adapter.collect_k3_with_backend(
        _envelope(self.root, migration=True), 100, "cpu", FakeBackend()
      )
    with (
      patch.object(adapter, "_load_live_dependencies") as forbidden,
      self.assertRaisesRegex(ValueError, "100 or 500"),
    ):
      adapter.collect_k3(checkpoint, 300, "cpu")
    forbidden.assert_not_called()

  def test_collect_k3_loads_one_backend_after_pure_validation(self) -> None:
    checkpoint = _envelope(self.root)
    backend = FakeBackend()
    with (
      patch.object(adapter, "_load_live_dependencies", return_value="deps") as load,
      patch.object(adapter, "_DynamicMjLabBackend", return_value=backend) as cls,
    ):
      candidate = adapter.collect_k3(checkpoint, 100, "cpu")
    self.assertTrue(candidate["screen_passed"])
    self.assertEqual(backend.config["domain"], "stairs")
    load.assert_called_once_with()
    cls.assert_called_once()


  def test_cli_collect_and_k3_flags_are_direct_and_mockable(self) -> None:
    request_path = self.root / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    output = self.root / "collection.json"
    with (
      patch.object(adapter, "collect", return_value={"ok": True}) as collect,
      patch.object(adapter.stair_camp, "_write_output") as write,
    ):
      self.assertEqual(
        adapter.main(
          ["collect", "--request", str(request_path), "--output", str(output)]
        ),
        0,
      )
    collect.assert_called_once_with({})
    write.assert_called_once_with({"ok": True}, output)
    parsed = adapter.parse_args(
      [
        "collect-k3",
        "--checkpoint-envelope",
        str(request_path),
        "--budget-updates",
        "500",
        "--device",
        "cuda:2",
        "--output",
        str(output),
      ]
    )
    self.assertEqual(parsed.command, "collect-k3")
    self.assertEqual(parsed.budget_updates, 500)
    self.assertEqual(parsed.device, "cuda:2")

  def test_policy_head_ablations_zero_exact_registered_indices(self) -> None:
    original = [float(index + 1) for index in range(6)]

    def policy(_observations):
      return FakeTensor([original])

    for name in evaluator.ABLATION_ORDER:
      with self.subTest(ablation=name):
        descriptor = evaluator.resolve_ablation(name)
        wrapped = adapter.apply_policy_ablation(policy, descriptor.to_dict())
        actual = wrapped(None).values[0]
        expected = list(original)
        for index in descriptor.zero_action_indices:
          expected[index] = 0.0
        self.assertEqual(actual, expected)
    mutated = evaluator.resolve_ablation("full").to_dict()
    mutated["interpretation"] = "drift"
    with self.assertRaisesRegex(ValueError, "drifted"):
      adapter.apply_policy_ablation(policy, mutated)

  def test_feedforward_ablation_keeps_fsm_update_but_zeros_outputs(self) -> None:
    class Action:
      def __init__(self) -> None:
        self._dynamic_leg_feedforward = FakeBuffer()
        self._dynamic_drive_feedforward = FakeBuffer()
        self.calls = 0

      def _update_dynamic_stair(self, value: int) -> int:
        self.calls += 1
        self._dynamic_leg_feedforward.value = float(value)
        self._dynamic_drive_feedforward.value = float(value)
        return value + 1

    action = Action()
    adapter._disable_feedforward(action)
    self.assertEqual(action._update_dynamic_stair(2), 3)
    self.assertEqual(action.calls, 1)
    self.assertEqual(action._dynamic_leg_feedforward.value, 0.0)
    self.assertEqual(action._dynamic_drive_feedforward.value, 0.0)
    with self.assertRaisesRegex(RuntimeError, "twice"):
      adapter._disable_feedforward(action)

  def test_interface_metadata_and_riser_geometry_fail_closed(self) -> None:
    class Shape:
      def __init__(self, width: int) -> None:
        self.shape = (2, width)

    adapter.assert_policy_interface(
      {"actor": Shape(52), "critic": Shape(56)}, action_width=6
    )
    with self.assertRaisesRegex(RuntimeError, "critic width"):
      adapter.assert_policy_interface(
        {"actor": Shape(52), "critic": Shape(55)}, action_width=6
      )
    for risers, expected_platform in ((1, 5.4), (3, 4.2)):
      platform = adapter._platform_width_for_risers(
        risers,
        terrain_length_m=8.0,
        border_width_m=1.0,
        step_width_m=0.3,
      )
      self.assertAlmostEqual(platform, expected_platform)
      # Mirror MjLab BoxPyramidStairsTerrainCfg.function exactly.
      generated = int((8.0 - 2.0 * 1.0 - platform) / (2.0 * 0.3))
      self.assertEqual(generated, risers)
    backend = FakeBackend()
    backend.metadata = lambda: {
      "actor_observation_width": 52,
      "critic_observation_width": 55,
      "action_width": 6,
      "stage5_actor_adapter_used": False,
    }
    with self.assertRaisesRegex(ValueError, "critic_observation_width"):
      adapter.collect_with_backend(_request(self.root), backend)


if __name__ == "__main__":
  unittest.main()
