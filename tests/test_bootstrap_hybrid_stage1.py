import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from hoppertrex_mjlab.scripts.rsl_rl.bootstrap_hybrid_stage1 import (
  STAGE1_ACTION_STD,
  bootstrap_stage1_checkpoint,
  qualified_controller_provenance,
)


def _stable_hash(payload):
  encoded = json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
  ).encode("ascii")
  return hashlib.sha256(encoded).hexdigest()


def _controller_payload(*, controller_type="lqr"):
  gain = [[1.0, 2.0, 3.0, 4.0]]
  return {
    "schema_version": 1,
    "controller_type": controller_type,
    "state_names": [
      "pitch",
      "pitch_rate",
      "vx_error",
      "signed_wheel_speed_error",
    ],
    "gain": gain,
    "controllability_rank": 4 if controller_type == "lqr" else 0,
    "heldout_one_step_nrmse": {
      "maximum": 0.10 if controller_type == "lqr" else 0.40
    },
    "fallback_reasons": (
      [] if controller_type == "lqr" else ["held-out NRMSE exceeds limit"]
    ),
    "gain_hash": _stable_hash(
      {
        "controller_type": controller_type,
        "state_names": (
          "pitch",
          "pitch_rate",
          "vx_error",
          "signed_wheel_speed_error",
        ),
        "gain": gain,
      }
    ),
  }


def _checkpoint():
  return {
    "actor_state_dict": {
      "mlp.0.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
      "mlp.4.weight": torch.arange(18, dtype=torch.float32).reshape(6, 3),
      "mlp.4.bias": torch.arange(6, dtype=torch.float32),
      "distribution.std_param": torch.full((6,), 0.6),
    },
    "critic_state_dict": {
      "mlp.0.weight": torch.ones(3, 4),
    },
    "optimizer_state_dict": {
      "state": {0: {"step": torch.tensor(7.0)}},
      "param_groups": [{"lr": 1.0e-3, "params": [0]}],
    },
    "iter": 99,
    "infos": {"old": "value"},
  }


class BootstrapHybridStage1Test(unittest.TestCase):
  def test_bootstrap_zeroes_six_dimensional_head_and_sets_per_action_std(self):
    source = _checkpoint()
    original_actor = {
      key: value.clone() for key, value in source["actor_state_dict"].items()
    }

    result = bootstrap_stage1_checkpoint(
      source,
      task="HopperTrex-Hybrid-v2-Stage1",
      seed=3,
      controller_provenance={
        "path": "C:/artifacts/controller.json",
        "gain_hash": "gain123",
        "file_sha256": "file456",
        "controller_type": "lqr",
      },
      git_sha="abc123",
      created_at="2026-07-11T12:00:00+08:00",
    )

    torch.testing.assert_close(
      result["actor_state_dict"]["mlp.4.weight"],
      torch.zeros(6, 3),
    )
    torch.testing.assert_close(
      result["actor_state_dict"]["mlp.4.bias"],
      torch.zeros(6),
    )
    torch.testing.assert_close(
      result["actor_state_dict"]["distribution.std_param"],
      torch.tensor(STAGE1_ACTION_STD),
    )
    torch.testing.assert_close(
      result["actor_state_dict"]["mlp.0.weight"],
      source["actor_state_dict"]["mlp.0.weight"],
    )
    for key, value in original_actor.items():
      torch.testing.assert_close(source["actor_state_dict"][key], value)

  def test_bootstrap_clears_optimizer_resets_iteration_and_records_provenance(self):
    result = bootstrap_stage1_checkpoint(
      _checkpoint(),
      task="HopperTrex-Hybrid-v2-Stage1",
      seed=3,
      controller_provenance={
        "path": "C:/artifacts/controller.json",
        "gain_hash": "gain123",
        "file_sha256": "file456",
        "controller_type": "lqr",
      },
      git_sha="abc123",
      created_at="2026-07-11T12:00:00+08:00",
    )

    self.assertEqual(result["optimizer_state_dict"]["state"], {})
    self.assertEqual(result["iter"], 0)
    self.assertEqual(
      result["infos"]["hybrid_stage1_bootstrap"],
      {
        "created_at": "2026-07-11T12:00:00+08:00",
        "task": "HopperTrex-Hybrid-v2-Stage1",
        "stage": 1,
        "seed": 3,
        "git_sha": "abc123",
        "controller_path": "C:/artifacts/controller.json",
        "controller_gain_hash": "gain123",
        "controller_file_sha256": "file456",
        "controller_type": "lqr",
        "action_std": list(STAGE1_ACTION_STD),
      },
    )

  def test_controller_provenance_requires_qualified_artifact(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      path = Path(temp_dir) / "controller.json"
      path.write_text(
        json.dumps(_controller_payload()),
        encoding="utf-8",
      )

      provenance = qualified_controller_provenance(path)

      self.assertEqual(provenance["path"], str(path.resolve()))
      self.assertEqual(
        provenance["gain_hash"],
        _controller_payload()["gain_hash"],
      )
      self.assertEqual(provenance["controller_type"], "lqr")
      self.assertEqual(len(provenance["file_sha256"]), 64)

      path.write_text(
        json.dumps(_controller_payload(controller_type="pd")),
        encoding="utf-8",
      )
      with self.assertRaisesRegex(ValueError, "qualified"):
        qualified_controller_provenance(path)


if __name__ == "__main__":
  unittest.main()
