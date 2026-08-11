import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from hoppertrex_mjlab.hybrid.config import HYBRID_ACTION_NAMES, HYBRID_ACTION_STD
from hoppertrex_mjlab.hybrid.stair_dynamic import DYNAMIC_STAIR_TASK_ID
from hoppertrex_mjlab.hybrid.stair_dynamic_contract import (
  DYNAMIC_STAIR_MIGRATION_INFO_KEY,
)
from hoppertrex_mjlab.hybrid.runner import stair_dynamic_effective_load_cfg
from hoppertrex_mjlab.scripts.rsl_rl.migrate_hybrid_stage import (
  COLLAPSED_ACTION_STD_THRESHOLD,
)
from hoppertrex_mjlab.scripts.rsl_rl.migrate_stage5_to_stair_dynamic import (
  STAGE5_TASK_ID,
  atomic_torch_save_no_clobber,
  main,
  migrate_checkpoint,
  validate_stage5_source,
)

SOURCE_SHA = "a" * 64
GATE_SHA = "b" * 64
GIT_SHA = "c" * 40
CREATED_AT = "2026-08-11T12:00:00+00:00"


def _actor_state(*, collapsed: tuple[int, ...] = ()) -> dict[str, torch.Tensor]:
  std = torch.tensor([0.07, 0.09, 0.05, 0.05, 0.05, 0.05])
  for index in collapsed:
    std[index] = COLLAPSED_ACTION_STD_THRESHOLD / 2.0
  return {
    "distribution.std_param": std,
    "mlp.0.weight": torch.arange(4 * 34, dtype=torch.float32).reshape(4, 34),
    "mlp.0.bias": torch.arange(4, dtype=torch.float32),
    "mlp.2.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
    "mlp.2.bias": torch.arange(4, dtype=torch.float32) + 10.0,
    "mlp.4.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4),
    "mlp.4.bias": torch.arange(6, dtype=torch.float32) + 20.0,
  }


def _critic_state() -> dict[str, torch.Tensor]:
  return {
    "mlp.0.weight": torch.arange(3 * 34, dtype=torch.float64).reshape(3, 34),
    "mlp.0.bias": torch.arange(3, dtype=torch.float64),
    "mlp.2.weight": torch.arange(9, dtype=torch.float64).reshape(3, 3),
    "mlp.2.bias": torch.arange(3, dtype=torch.float64) + 10.0,
    "mlp.4.weight": torch.arange(3, dtype=torch.float64).reshape(1, 3),
    "mlp.4.bias": torch.tensor([20.0], dtype=torch.float64),
  }


def _checkpoint(*, collapsed: tuple[int, ...] = ()) -> dict[str, object]:
  return {
    "actor_state_dict": _actor_state(collapsed=collapsed),
    "critic_state_dict": _critic_state(),
    "optimizer_state_dict": {
      "state": {1: {"step": torch.tensor(12), "moment": torch.tensor([1.0])}},
      "param_groups": [{"params": [1, 2], "lr": 3.0e-4}],
    },
    "iter": 99,
    "infos": {
      "hybrid_stage1_bootstrap": {
        "task": "HopperTrex-Hybrid-v2-Stage1",
        "stage": 1,
        "seed": 1,
        "action_order": list(HYBRID_ACTION_NAMES),
      },
      "hybrid_stage_migration": {
        "source_stage": 4,
        "target_stage": 5,
        "source_action_std": [0.07, 0.09, 0.05, 0.05, 0.05, 0.05],
        "collapsed_active_actions": [],
        "reset_collapsed_active_std": False,
      },
      "hybrid_training": {"git_sha": GIT_SHA},
      "env_state": {"common_step_counter": 9600},
    },
  }


def _gate() -> dict[str, object]:
  return {
    "schema_version": 2,
    "suite": "robust",
    "task": STAGE5_TASK_ID,
    "evaluation_profile": "formal",
    "evaluation_source": "live",
    "gate_pass": True,
    "seed": 1,
    "git_sha": GIT_SHA,
    "checkpoint": r"C:\machine\run\model_99.pt",
    "checkpoint_file_sha256": SOURCE_SHA,
    "rollout": {
      "steps": 3000,
      "num_envs": 32,
      "leg_residuals_ablated": False,
    },
    "checks": [
      {
        "name": "candidate_terminated_event_rate",
        "scenario": "robust_pushes",
        "pass": True,
        "value": 0.0,
      },
      {
        "name": "recovery_kick_event_count",
        "scenario": "stage5_ablation",
        "pass": True,
        "value": 128.0,
      },
    ],
  }


def _migrate(
  checkpoint: dict[str, object] | None = None,
  gate: dict[str, object] | None = None,
  *,
  reset: bool = False,
):
  return migrate_checkpoint(
    checkpoint or _checkpoint(),
    gate or _gate(),
    source_checkpoint_sha256=SOURCE_SHA,
    source_gate_sha256=GATE_SHA,
    reset_collapsed_active_std=reset,
    created_at=CREATED_AT,
  )


class MigrateStage5ToStairDynamicTest(unittest.TestCase):
  def test_positive_migration_expands_only_first_layers_and_records_provenance(
    self,
  ) -> None:
    source = _checkpoint()
    original = copy.deepcopy(source)

    migrated, report = _migrate(source)

    actor = migrated["actor_state_dict"]
    critic = migrated["critic_state_dict"]
    assert isinstance(actor, dict)
    assert isinstance(critic, dict)
    self.assertEqual(tuple(actor["mlp.0.weight"].shape), (4, 52))
    self.assertEqual(tuple(critic["mlp.0.weight"].shape), (3, 56))
    original_actor = original["actor_state_dict"]
    original_critic = original["critic_state_dict"]
    assert isinstance(original_actor, dict)
    assert isinstance(original_critic, dict)
    torch.testing.assert_close(
      actor["mlp.0.weight"][:, :34],
      original_actor["mlp.0.weight"],
      rtol=0.0,
      atol=0.0,
    )
    torch.testing.assert_close(
      critic["mlp.0.weight"][:, :34],
      original_critic["mlp.0.weight"],
      rtol=0.0,
      atol=0.0,
    )
    self.assertEqual(torch.count_nonzero(actor["mlp.0.weight"][:, 34:]).item(), 0)
    self.assertEqual(
      torch.count_nonzero(critic["mlp.0.weight"][:, 34:]).item(), 0
    )
    for key, tensor in original_actor.items():
      if key != "mlp.0.weight":
        torch.testing.assert_close(actor[key], tensor, rtol=0.0, atol=0.0)
    for key, tensor in original_critic.items():
      if key != "mlp.0.weight":
        torch.testing.assert_close(critic[key], tensor, rtol=0.0, atol=0.0)

    optimizer = migrated["optimizer_state_dict"]
    original_optimizer = original["optimizer_state_dict"]
    assert isinstance(optimizer, dict)
    assert isinstance(original_optimizer, dict)
    self.assertEqual(optimizer["state"], {})
    self.assertEqual(optimizer["param_groups"], original_optimizer["param_groups"])
    # Source param_groups remain auditable bytes only: a migration-only v3
    # load explicitly keeps the newly constructed optimizer and defaults.
    effective = stair_dynamic_effective_load_cfg(migrated, None)
    assert isinstance(effective, dict)
    self.assertFalse(effective["optimizer"])
    self.assertTrue(effective["iteration"])
    self.assertEqual(optimizer["param_groups"][0]["lr"], 3.0e-4)
    self.assertEqual(migrated["iter"], 0)
    self.assertEqual(source["iter"], 99)
    source_actor = source["actor_state_dict"]
    source_critic = source["critic_state_dict"]
    assert isinstance(source_actor, dict)
    assert isinstance(source_critic, dict)
    for key, tensor in original_actor.items():
      torch.testing.assert_close(source_actor[key], tensor)
    for key, tensor in original_critic.items():
      torch.testing.assert_close(source_critic[key], tensor)

    infos = migrated["infos"]
    assert isinstance(infos, dict)
    self.assertIs(infos[DYNAMIC_STAIR_MIGRATION_INFO_KEY], report)
    self.assertEqual(infos["env_state"], {"common_step_counter": 0})
    self.assertEqual(report["source_checkpoint_sha256"], SOURCE_SHA)
    self.assertEqual(report["source_gate_sha256"], GATE_SHA)
    self.assertEqual(report["source_task"], STAGE5_TASK_ID)
    self.assertEqual(report["source_seed"], 1)
    self.assertEqual(report["source_completed_updates"], 100)
    self.assertEqual(report["target_task"], DYNAMIC_STAIR_TASK_ID)
    self.assertEqual(
      (
        report["source_actor_width"],
        report["target_actor_width"],
        report["source_critic_width"],
        report["target_critic_width"],
      ),
      (34, 52, 34, 56),
    )
    self.assertEqual(report["created_at"], CREATED_AT)
    self.assertEqual(report["collapsed_active_actions"], [])
    self.assertEqual(report["source_action_std"], report["target_action_std"])

  def test_trained_checkpoint_keeps_normal_optimizer_resume_semantics(self) -> None:
    migrated, _report = _migrate()
    infos = migrated["infos"]
    assert isinstance(infos, dict)
    infos["stair_dynamic_training"] = {"completed_updates": 100}
    self.assertIsNone(stair_dynamic_effective_load_cfg(migrated, None))

  def test_collapsed_active_std_requires_explicit_reset(self) -> None:
    checkpoint = _checkpoint(collapsed=(2, 5))
    with self.assertRaisesRegex(ValueError, "collapsed active action std"):
      _migrate(checkpoint)

    migrated, report = _migrate(checkpoint, reset=True)
    actor = migrated["actor_state_dict"]
    source_actor = checkpoint["actor_state_dict"]
    assert isinstance(actor, dict)
    assert isinstance(source_actor, dict)
    target_std = actor["distribution.std_param"]
    source_std = source_actor["distribution.std_param"]
    self.assertEqual(report["collapsed_active_indices"], [2, 5])
    self.assertEqual(
      report["collapsed_active_actions"],
      [HYBRID_ACTION_NAMES[2], HYBRID_ACTION_NAMES[5]],
    )
    self.assertAlmostEqual(float(target_std[2]), HYBRID_ACTION_STD[2])
    self.assertAlmostEqual(float(target_std[5]), HYBRID_ACTION_STD[5])
    for index in (0, 1, 3, 4):
      self.assertEqual(float(target_std[index]), float(source_std[index]))
    self.assertAlmostEqual(
      float(source_std[2]), COLLAPSED_ACTION_STD_THRESHOLD / 2.0
    )

  def test_log_std_preserves_noncollapsed_indices(self) -> None:
    checkpoint = _checkpoint()
    actor = checkpoint["actor_state_dict"]
    assert isinstance(actor, dict)
    std = actor.pop("distribution.std_param")
    std[3] = COLLAPSED_ACTION_STD_THRESHOLD / 2.0
    actor["distribution.log_std_param"] = torch.log(std)

    migrated, report = _migrate(checkpoint, reset=True)

    target_actor = migrated["actor_state_dict"]
    assert isinstance(target_actor, dict)
    target = target_actor["distribution.log_std_param"]
    source = actor["distribution.log_std_param"]
    self.assertEqual(report["std_key"], "distribution.log_std_param")
    self.assertEqual(report["collapsed_active_indices"], [3])
    for index in (0, 1, 2, 4, 5):
      self.assertEqual(float(target[index]), float(source[index]))
    self.assertAlmostEqual(float(target[3]), math.log(HYBRID_ACTION_STD[3]), places=6)

  def test_rejects_source_provenance_failures(self) -> None:
    cases = []
    staircamp = _checkpoint()
    staircamp_infos = staircamp["infos"]
    assert isinstance(staircamp_infos, dict)
    staircamp_infos["stair_camp_training"] = {"task": "StairCamp"}
    cases.append(("StairCamp", staircamp, _gate(), "StairCamp"))

    extended = _checkpoint()
    extended["iter"] = 499
    cases.append(("500-update", extended, _gate(), "100-update"))

    wrong_task = _gate()
    wrong_task["task"] = "HopperTrex-Hybrid-v2-StairCamp"
    cases.append(("wrong task", _checkpoint(), wrong_task, "task"))

    wrong_seed = _checkpoint()
    wrong_seed_infos = wrong_seed["infos"]
    assert isinstance(wrong_seed_infos, dict)
    bootstrap = wrong_seed_infos["hybrid_stage1_bootstrap"]
    assert isinstance(bootstrap, dict)
    bootstrap["seed"] = 2
    cases.append(("wrong seed", wrong_seed, _gate(), "seed"))

    missing = _checkpoint()
    missing_infos = missing["infos"]
    assert isinstance(missing_infos, dict)
    del missing_infos["hybrid_stage_migration"]
    cases.append(("missing provenance", missing, _gate(), "provenance"))

    for name, checkpoint, gate, message in cases:
      with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
        _migrate(checkpoint, gate)

  def test_rejects_invalid_gate_envelopes(self) -> None:
    def ablate(gate):
      rollout = gate["rollout"]
      assert isinstance(rollout, dict)
      rollout["leg_residuals_ablated"] = True

    mutations = {
      "nonformal": (lambda gate: gate.update(evaluation_profile="screen"), "formal"),
      "not live": (lambda gate: gate.update(evaluation_source="scenario_file"), "live"),
      "failed": (lambda gate: gate.update(gate_pass=False), "gate_pass"),
      "ablated": (ablate, "ablated"),
      "wrong sha": (
        lambda gate: gate.update(checkpoint_file_sha256="f" * 64),
        "checkpoint_file_sha256",
      ),
      "wrong seed": (lambda gate: gate.update(seed=2), "seed"),
    }
    for name, (mutate, message) in mutations.items():
      with self.subTest(name=name):
        gate = _gate()
        mutate(gate)
        with self.assertRaisesRegex(ValueError, message):
          _migrate(gate=gate)

  def test_rejects_malformed_gate_checks_and_network_widths(self) -> None:
    gate = _gate()
    checks = gate["checks"]
    assert isinstance(checks, list)
    checks[0]["pass"] = False
    with self.assertRaisesRegex(ValueError, "failed or malformed"):
      _migrate(gate=gate)

    checkpoint = _checkpoint()
    actor = checkpoint["actor_state_dict"]
    assert isinstance(actor, dict)
    actor["mlp.0.weight"] = torch.zeros(4, 33)
    with self.assertRaisesRegex(ValueError, "34-wide first layer"):
      _migrate(checkpoint)

  def test_cli_atomically_writes_and_never_overwrites(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source_path = root / "model_99.pt"
      gate_path = root / "stage5_formal.json"
      output_path = root / "stair_dynamic.pt"
      torch.save(_checkpoint(collapsed=(2, 3, 4, 5)), source_path)
      source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
      gate = _gate()
      gate["checkpoint_file_sha256"] = source_sha
      gate_path.write_text(json.dumps(gate), encoding="utf-8")
      argv = [
        "migrate_stage5_to_stair_dynamic.py",
        "--source-checkpoint",
        str(source_path),
        "--source-gate-json",
        str(gate_path),
        "--output-checkpoint",
        str(output_path),
        "--reset-collapsed-active-std",
      ]

      with patch.object(sys, "argv", argv):
        main()

      saved = torch.load(output_path, map_location="cpu", weights_only=False)
      report = saved["infos"][DYNAMIC_STAIR_MIGRATION_INFO_KEY]
      self.assertEqual(report["source_checkpoint_sha256"], source_sha)
      self.assertEqual(
        report["source_gate_sha256"],
        hashlib.sha256(gate_path.read_bytes()).hexdigest(),
      )
      first_bytes = output_path.read_bytes()
      with (
        patch.object(sys, "argv", argv),
        self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"),
      ):
        main()
      self.assertEqual(output_path.read_bytes(), first_bytes)
      self.assertEqual(list(root.glob("*.incomplete.*")), [])

  def test_atomic_helper_preserves_preexisting_file(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / "exists.pt"
      output.write_bytes(b"sentinel")
      with self.assertRaises(FileExistsError):
        atomic_torch_save_no_clobber({"new": True}, output)
      self.assertEqual(output.read_bytes(), b"sentinel")

  def test_validator_rejects_gate_seed(self) -> None:
    gate = _gate()
    gate["seed"] = 2
    with self.assertRaisesRegex(ValueError, "seed"):
      validate_stage5_source(
        _checkpoint(), gate, source_checkpoint_sha256=SOURCE_SHA
      )


if __name__ == "__main__":
  unittest.main()
