from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hoppertrex_mjlab.hybrid.runner import HybridOnPolicyRunner
from hoppertrex_mjlab.hybrid.stair_dynamic import DYNAMIC_MANEUVER_REQUIRED_BINDINGS
from hoppertrex_mjlab.scripts.rsl_rl.train import (
  DYNAMIC_STAIR_EXTENSION_AUTHORIZATION_PATH_ENV,
  validate_stair_dynamic_extension_authorization,
  validate_stair_dynamic_extension_checkpoint,
  validate_stair_dynamic_migration_checkpoint,
)
from tests.test_migrate_stage5_to_stair_dynamic import (
  GATE_SHA,
  SOURCE_SHA,
  _migrate,
)


class StairDynamicTrainPreflightTest(unittest.TestCase):
  def _cfg(self, git_sha):
    bindings = {
      name: (git_sha if name == "git_sha" else "a" * 64)
      for name in DYNAMIC_MANEUVER_REQUIRED_BINDINGS
    }
    bindings["stage5_checkpoint_sha256"] = SOURCE_SHA
    bindings["stage5_formal_gate_sha256"] = GATE_SHA
    action = SimpleNamespace(
      controller_gain_hash=None,
      calibration_hash=None,
      yaw_calibration_hash=None,
      posture_map_hash=None,
      posture_artifact_hash=None,
      station_calibration_hash=None,
    )
    env = SimpleNamespace(
      stair_dynamic_maneuver_bindings=bindings,
      actions={"hybrid_wheel_leg": action},
    )
    return SimpleNamespace(
      agent=SimpleNamespace(max_iterations=100, seed=1),
      env=env,
    )

  def test_valid_stage5_migration_starts_fresh_v3_counter(self):
    migrated, _report = _migrate()
    git_sha = "c" * 40
    cfg = self._cfg(git_sha)
    with patch(
      "hoppertrex_mjlab.scripts.rsl_rl.train._repository_head",
      return_value=git_sha,
    ):
      validate_stair_dynamic_migration_checkpoint(cfg, migrated)

  def test_stage5_500_checkpoint_and_round1_are_rejected(self):
    migrated, _report = _migrate()
    cfg = self._cfg("c" * 40)
    migrated["infos"]["stair_camp_training"] = {}
    with patch(
      "hoppertrex_mjlab.scripts.rsl_rl.train._repository_head",
      return_value="c" * 40,
    ), self.assertRaises(ValueError):
      validate_stair_dynamic_migration_checkpoint(cfg, migrated)
    migrated["infos"].pop("stair_camp_training")
    migrated["infos"]["stair_dynamic_migration"]["source_completed_updates"] = 500
    with patch(
      "hoppertrex_mjlab.scripts.rsl_rl.train._repository_head",
      return_value="c" * 40,
    ), self.assertRaises(ValueError):
      validate_stair_dynamic_migration_checkpoint(cfg, migrated)


  def test_extension_accepts_k3_selected_76_update_checkpoint(self):
    action = SimpleNamespace(
      action_scales=(0.5, 0.3, 0.035, 0.035, 0.035, 0.035),
      dynamic_stair_maneuver=SimpleNamespace(maneuver_hash="d" * 64),
    )
    bindings = {
      "stage5_checkpoint_sha256": SOURCE_SHA,
      "stage5_formal_gate_sha256": GATE_SHA,
    }
    env = SimpleNamespace(
      actions={"hybrid_wheel_leg": action},
      stair_dynamic_maneuver_bindings=bindings,
      stair_dynamic_contract_sha256="e" * 64,
    )
    cfg = SimpleNamespace(
      agent=SimpleNamespace(max_iterations=500, seed=1), env=env
    )
    record = {
      "schema_version": 1,
      "task": "HopperTrex-Hybrid-v3-StairDynamic",
      "training_seed": 1,
      "git_sha": "c" * 40,
      "contract_sha256": "e" * 64,
      "artifact_bindings": {"artifact": "f" * 64},
      "action_scales": list(action.action_scales),
      "maneuver_sha256": "d" * 64,
      "source_stage5_checkpoint_sha256": SOURCE_SHA,
      "source_stage5_gate_sha256": GATE_SHA,
      "stage5_prefix_preserved_and_new_columns_zero": True,
      "completed_updates": 76,
    }
    checkpoint = {
      "iter": 75,
      "actor_state_dict": {},
      "critic_state_dict": {},
      "infos": {
        "stair_dynamic_training": record,
        "stair_dynamic_curriculum": {},
        "stair_dynamic_progress": {},
        "stair_dynamic_migration": {},
        "env_state": {"common_step_counter": 1},
      },
    }
    with (
      patch(
        "hoppertrex_mjlab.scripts.rsl_rl.train._repository_head",
        return_value="c" * 40,
      ),
      patch(
        "hoppertrex_mjlab.scripts.rsl_rl.train.dynamic_stair_contract_hash",
        return_value="e" * 64,
      ),
      patch(
        "hoppertrex_mjlab.scripts.rsl_rl.train.dynamic_stair_artifact_bindings",
        return_value={"artifact": "f" * 64},
      ),
      patch(
        "hoppertrex_mjlab.scripts.rsl_rl.train.validate_dynamic_stair_progress_payload"
      ),
    ):
      validate_stair_dynamic_extension_checkpoint(cfg, checkpoint)
      checkpoint["iter"] = 74
      record["completed_updates"] = 75
      with self.assertRaisesRegex(ValueError, "51/76/100"):
        validate_stair_dynamic_extension_checkpoint(cfg, checkpoint)

  def test_extension_authorization_binds_exact_checkpoint_bytes(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint_path = root / "model_75.pt"
      checkpoint_path.write_bytes(b"selected-checkpoint")
      authorization_path = root / "authorization.json"
      authorization_path.write_text("{}", encoding="utf-8")
      authorization = {
        "selected_checkpoint_file": str(checkpoint_path),
        "selected_checkpoint_sha256": hashlib.sha256(
          checkpoint_path.read_bytes()
        ).hexdigest(),
        "selected_completed_updates": 76,
        "target_total_updates": 500,
      }
      checkpoint = {
        "infos": {"stair_dynamic_training": {"completed_updates": 76}}
      }
      with (
        patch.dict(
          os.environ,
          {
            DYNAMIC_STAIR_EXTENSION_AUTHORIZATION_PATH_ENV: str(
              authorization_path
            )
          },
        ),
        patch(
          "hoppertrex_mjlab.scripts.rsl_rl.evaluate_stair_dynamic.validate_extension_authorization",
          return_value=authorization,
        ),
      ):
        validate_stair_dynamic_extension_authorization(
          checkpoint_path, checkpoint
        )
        authorization["selected_checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "authorized K=3"):
          validate_stair_dynamic_extension_authorization(
            checkpoint_path, checkpoint
          )

  def test_runner_migration_load_keeps_fresh_optimizer_defaults(self):
    migrated, _report = _migrate()
    runner = object.__new__(HybridOnPolicyRunner)
    runner._stair_camp = False
    runner._stair_dynamic = True
    runner._stair_dynamic_maneuver_bindings = {
      "stage5_checkpoint_sha256": SOURCE_SHA,
      "stage5_formal_gate_sha256": GATE_SHA,
    }
    runner._stair_dynamic_loaded_completed_updates = -1
    runner.current_learning_iteration = 0
    runner.env = SimpleNamespace(
      unwrapped=SimpleNamespace(common_step_counter=0),
      reset=Mock(),
    )
    expected_load_cfg = {
      "actor": True,
      "critic": True,
      "optimizer": False,
      "iteration": True,
      "rnd": True,
    }
    base_runner = HybridOnPolicyRunner.__mro__[1]
    with (
      patch(
        "hoppertrex_mjlab.hybrid.runner.torch.load", return_value=migrated
      ),
      patch.object(
        base_runner, "load", return_value=migrated["infos"]
      ) as base_load,
    ):
      runner.load("migration.pt")
    base_load.assert_called_once_with("migration.pt", expected_load_cfg, True, None)
    self.assertEqual(runner._stair_dynamic_loaded_completed_updates, 0)
    runner.env.reset.assert_called_once_with()

  def test_runner_interprets_100_to_500_as_total_budget(self):
    runner = object.__new__(HybridOnPolicyRunner)
    runner._stair_camp = False
    runner._stair_dynamic = True
    runner._stair_dynamic_loaded_completed_updates = 100
    runner.current_learning_iteration = 99
    with patch.object(HybridOnPolicyRunner.__mro__[1], "learn") as base_learn:
      runner.learn(500, True)
    base_learn.assert_called_once_with(400, True)
    self.assertEqual(runner.current_learning_iteration, 100)


if __name__ == "__main__":
  unittest.main()
