from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hoppertrex_mjlab.hybrid.stair_dynamic import DYNAMIC_MANEUVER_REQUIRED_BINDINGS
from hoppertrex_mjlab.scripts.rsl_rl.search_stair_dynamic import run_search
from hoppertrex_mjlab.scripts.rsl_rl.stair_dynamic_search_live_adapter import (
  STAGE5_CHECKPOINT_PATH_ENV,
  _dynamic_danger,
  _stage5_checkpoint_path,
)


def _bindings():
  return {
    name: ("b" * 40 if name == "git_sha" else "a" * 64)
    for name in DYNAMIC_MANEUVER_REQUIRED_BINDINGS
  }


def _score(progress, *, successes=1, unsafe=0, error=0.1):
  return {
    "safe_successes": successes,
    "median_progress": progress,
    "peak_pitch": error,
    "energy": error,
    "target_smoothness": error,
    "unsafe_trials": unsafe,
  }


class _Adapter:
  def __init__(self, *, qualify=True):
    self.qualify = qualify
    self.batch_sizes = []
    self.devices = []
    self.stage5_hashes = []

  def __call__(self, request):
    self.devices.append(request["device"])
    self.stage5_hashes.append(request["expected_stage5_checkpoint_sha256"])
    if request["kind"] == "family_screen":
      unsafe = 0 if self.qualify else 1
      return {
        "scores": {
          "roll_only": _score(0.20, successes=0),
          "synchronized": _score(0.30, unsafe=unsafe),
          "alternating": _score(0.40, unsafe=unsafe, error=0.05),
        },
        "trigger_qualification": {
          "metric": "abs(F0*nx)",
          "threshold_n": 18.0,
          "window": 3,
          "left_sensor_identity": True,
          "right_sensor_identity": True,
          "left_live_detected": True,
          "right_live_detected": True,
          "flat_false_positives": 0,
          "kick_false_positives": 0,
          "evidence_sha256": "a" * 64,
        },
      }
    candidates = request["candidates"]
    self.batch_sizes.append(len(candidates))
    scores = []
    for candidate in candidates:
      target = (0.04, 0.05, 0.2, 1.0)
      error = sum((float(a) - b) ** 2 for a, b in zip(candidate, target))
      scores.append(_score(0.50 - error, error=error))
    return {"scores": scores}


class SearchStairDynamicTest(unittest.TestCase):
  def test_safe_family_runs_one_parallel_batch_per_cem_iteration(self):
    adapter = _Adapter()
    report, artifact = run_search(adapter, bindings=_bindings())
    self.assertEqual(report["classification"], "DYNAMIC_STAIR_MANEUVER_QUALIFIED")
    self.assertEqual(report["screen"]["selected_family"], "alternating")
    self.assertIsNotNone(artifact)
    self.assertEqual(adapter.batch_sizes, [1, 32, 32, 32, 32, 32])
    self.assertEqual(adapter.stage5_hashes, ["a" * 64] * 7)
    self.assertEqual(artifact["cem"]["population"], 32)
    self.assertEqual(artifact["cem"]["iterations"], 5)

  def test_device_is_forwarded_without_a_launcher_layer(self):
    adapter = _Adapter(qualify=False)
    run_search(adapter, bindings=_bindings(), device="cuda:3")
    self.assertEqual(adapter.devices, ["cuda:3"])
    with self.assertRaisesRegex(ValueError, "device"):
      run_search(adapter, bindings=_bindings(), device="")

  def test_unsafe_families_stop_before_cem(self):
    adapter = _Adapter(qualify=False)
    report, artifact = run_search(adapter, bindings=_bindings())
    self.assertEqual(report["classification"], "STOP_DYNAMIC_STAIR_UNQUALIFIED")
    self.assertIsNone(artifact)
    self.assertEqual(adapter.batch_sizes, [])

  def test_stage5_policy_path_is_bound_to_the_signed_checkpoint_bytes(self):
    with tempfile.TemporaryDirectory() as directory:
      checkpoint = Path(directory) / "stage5.pt"
      checkpoint.write_bytes(b"selected-stage5-policy")
      digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
      with patch.dict(
        "os.environ", {STAGE5_CHECKPOINT_PATH_ENV: str(checkpoint)}, clear=False
      ):
        self.assertEqual(_stage5_checkpoint_path(digest), checkpoint.resolve())
        with self.assertRaisesRegex(ValueError, "SHA256"):
          _stage5_checkpoint_path("0" * 64)

  def test_fsm_abort_code_episode_latch_and_target_saturation_are_danger(self):
    action = SimpleNamespace(
      dynamic_traversal_mode=torch.tensor([0, 3, 0, 0, 0, 0]),
      dynamic_abort_code=torch.tensor([0, 0, 7, 0, 0, 0]),
      dynamic_episode_unsafe=torch.tensor([False, False, False, True, False, False]),
      dynamic_target_saturation=torch.tensor([False, False, False, False, True, False]),
    )
    danger = _dynamic_danger(
      action,
      torch.tensor([False, False, False, False, False, True]),
      torch.zeros(6, dtype=torch.bool),
    )
    self.assertEqual(danger.tolist(), [False, True, True, True, True, True])

  def test_trigger_false_positive_blocks_search(self):
    adapter = _Adapter()
    original = adapter.__call__

    def bad(request):
      result = original(request)
      if request["kind"] == "family_screen":
        result["trigger_qualification"]["flat_false_positives"] = 1
      return result

    with self.assertRaises(ValueError):
      run_search(bad, bindings=_bindings())


if __name__ == "__main__":
  unittest.main()
