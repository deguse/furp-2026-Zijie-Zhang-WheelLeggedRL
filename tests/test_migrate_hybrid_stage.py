import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from hoppertrex_mjlab.scripts.rsl_rl.migrate_hybrid_stage import (
  main,
  migrate_hybrid_actor_state,
  validate_stage1_formal_gate_for_migration,
)
from hoppertrex_mjlab.hybrid.mismatch import stage1_mismatch_spec


def _actor_state() -> dict[str, torch.Tensor]:
  return {
    "mlp.0.weight": torch.arange(24, dtype=torch.float32).reshape(4, 6),
    "mlp.0.bias": torch.arange(4, dtype=torch.float32),
    "mlp.4.weight": torch.arange(18, dtype=torch.float32).reshape(6, 3),
    "mlp.4.bias": torch.arange(6, dtype=torch.float32),
    "distribution.std_param": torch.tensor(
      [0.012, 0.022, 0.032, 0.042, 0.052, 0.062]
    ),
  }


class MigrateHybridStageTest(unittest.TestCase):
  def test_stage1_to_stage2_gate_must_match_formal_source(self):
    checkpoint = {'infos': {'hybrid_training': {'git_sha': 'training-sha'}}}
    gate = {
      'suite': 'residual',
      'task': 'HopperTrex-Hybrid-v2-Stage1',
      'evaluation_profile': 'formal',
      'evaluation_source': 'live',
      'gate_pass': True,
      'stage1_profile_version': 'stage1b_speed010_mild_v1',
      'mismatch_profile': stage1_mismatch_spec(),
      'checkpoint_file_sha256': 'checkpoint-sha',
      'git_sha': 'training-sha',
    }

    validate_stage1_formal_gate_for_migration(
      checkpoint,
      gate,
      source_checkpoint_sha256='checkpoint-sha',
    )

    gate['checkpoint_file_sha256'] = 'other-checkpoint'
    with self.assertRaisesRegex(ValueError, 'checkpoint SHA256'):
      validate_stage1_formal_gate_for_migration(
        checkpoint,
        gate,
        source_checkpoint_sha256='checkpoint-sha',
      )

  def test_stage1_to_stage2_cli_saves_formal_gate_audit(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      source = root / "stage1.pt"
      gate_path = root / "stage1_formal.json"
      output = root / "stage2_bootstrap.pt"
      checkpoint = {
        "actor_state_dict": _actor_state(),
        "optimizer_state_dict": {"state": {1: {"step": 10}}},
        "iter": 99,
        "infos": {"hybrid_training": {"git_sha": "training-sha"}},
      }
      torch.save(checkpoint, source)
      source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
      gate = {
        "suite": "residual",
        "task": "HopperTrex-Hybrid-v2-Stage1",
        "evaluation_profile": "formal",
        "evaluation_source": "live",
        "gate_pass": True,
        "stage1_profile_version": "stage1b_speed010_mild_v1",
        "mismatch_profile": stage1_mismatch_spec(),
        "checkpoint_file_sha256": source_sha,
        "git_sha": "training-sha",
      }
      gate_path.write_text(json.dumps(gate), encoding="utf-8")

      argv = [
        "migrate_hybrid_stage.py",
        "--source-checkpoint",
        str(source),
        "--source-gate-json",
        str(gate_path),
        "--output-checkpoint",
        str(output),
        "--source-stage",
        "1",
        "--target-stage",
        "2",
        "--reset-collapsed-active-std",
      ]
      with patch.object(sys, "argv", argv):
        main()

      saved = torch.load(output, map_location="cpu", weights_only=False)
      migration = saved["infos"]["hybrid_stage_migration"]
      self.assertEqual(migration["source_checkpoint_sha256"], source_sha)
      self.assertEqual(migration["source_gate_profile"], "formal")
      self.assertEqual(migration["source_gate_suite"], "residual")
      self.assertEqual(
        migration["source_gate_sha256"],
        hashlib.sha256(gate_path.read_bytes()).hexdigest(),
      )
      self.assertEqual(saved["optimizer_state_dict"]["state"], {})
      self.assertEqual(saved["iter"], 0)

  def test_stage1_to_stage2_only_resets_new_yaw_head_and_std(self):
    source = _actor_state()

    migrated, report = migrate_hybrid_actor_state(
      source,
      source_stage=1,
      target_stage=2,
    )

    torch.testing.assert_close(migrated["mlp.4.weight"][0], source["mlp.4.weight"][0])
    torch.testing.assert_close(migrated["mlp.4.bias"][0], source["mlp.4.bias"][0])
    torch.testing.assert_close(
      migrated["mlp.4.weight"][1],
      torch.zeros_like(migrated["mlp.4.weight"][1]),
    )
    self.assertEqual(float(migrated["mlp.4.bias"][1]), 0.0)
    torch.testing.assert_close(
      migrated["distribution.std_param"],
      torch.tensor([0.012, 0.10, 0.032, 0.042, 0.052, 0.062]),
    )
    self.assertEqual(report["activated_indices"], [1])

  def test_stage2_to_stage3_resets_four_leg_joint_heads_to_point_zero_five_std(self):
    source = _actor_state()

    migrated, report = migrate_hybrid_actor_state(
      source,
      source_stage=2,
      target_stage=3,
    )

    torch.testing.assert_close(
      migrated["mlp.4.weight"][:2],
      source["mlp.4.weight"][:2],
    )
    torch.testing.assert_close(
      migrated["mlp.4.weight"][2:],
      torch.zeros_like(migrated["mlp.4.weight"][2:]),
    )
    torch.testing.assert_close(
      migrated["mlp.4.bias"][2:],
      torch.zeros_like(migrated["mlp.4.bias"][2:]),
    )
    torch.testing.assert_close(
      migrated["distribution.std_param"],
      torch.tensor([0.012, 0.022, 0.05, 0.05, 0.05, 0.05]),
    )
    self.assertEqual(report["activated_indices"], [2, 3, 4, 5])

  def test_source_tensors_are_not_mutated(self):
    source = _actor_state()
    original = {key: value.clone() for key, value in source.items()}

    migrate_hybrid_actor_state(source, source_stage=0, target_stage=1)

    for key in source:
      torch.testing.assert_close(source[key], original[key])

  def test_reports_collapsed_active_std_without_silently_resetting_it(self):
    source = _actor_state()

    migrated, report = migrate_hybrid_actor_state(
      source,
      source_stage=1,
      target_stage=2,
    )

    self.assertEqual(report["collapsed_active_indices"], [0])
    self.assertEqual(
      report["collapsed_active_actions"],
      ["wheel_balance_residual"],
    )
    self.assertAlmostEqual(
      float(migrated["distribution.std_param"][0]),
      0.012,
    )

  def test_can_explicitly_reset_collapsed_active_std(self):
    migrated, report = migrate_hybrid_actor_state(
      _actor_state(),
      source_stage=1,
      target_stage=2,
      reset_collapsed_active_std=True,
    )

    self.assertTrue(report["reset_collapsed_active_std"])
    self.assertAlmostEqual(
      float(migrated["distribution.std_param"][0]),
      0.15,
    )

  def test_non_forward_or_non_six_dimensional_migration_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "exactly one"):
      migrate_hybrid_actor_state(_actor_state(), source_stage=3, target_stage=2)

    with self.assertRaisesRegex(ValueError, "exactly one"):
      migrate_hybrid_actor_state(_actor_state(), source_stage=1, target_stage=5)

    malformed = _actor_state()
    malformed["mlp.4.weight"] = torch.zeros(5, 3)
    with self.assertRaisesRegex(ValueError, "six"):
      migrate_hybrid_actor_state(malformed, source_stage=1, target_stage=2)


if __name__ == "__main__":
  unittest.main()
