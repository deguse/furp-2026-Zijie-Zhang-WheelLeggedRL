from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hoppertrex_mjlab.scripts.rsl_rl import preflight_stair_dynamic as preflight
from hoppertrex_mjlab.scripts.rsl_rl.qualify_stair_dynamic_trigger import (
  collect_with_backend,
)
from tests.test_qualify_stair_dynamic_trigger import FakeBackend

_GIT = "a" * 40
_MANEUVER = "b" * 64
_ARTIFACTS = {
  "controller_gain_hash": "1" * 64,
  "calibration_hash": "2" * 64,
  "yaw_calibration_hash": "3" * 64,
  "posture_map_hash": "4" * 64,
  "posture_artifact_hash": "5" * 64,
  "station_calibration_hash": "6" * 64,
  "dynamic_maneuver_hash": _MANEUVER,
}
_BINDINGS = {
  "git_sha": _GIT,
  "stage5_checkpoint_sha256": "c" * 64,
  "stage5_formal_gate_sha256": "d" * 64,
}


class StairDynamicPreflightTest(unittest.TestCase):
  def test_real_registered_cfg_exposes_exact_per_wheel_sensor_cfgs(self) -> None:
    # Registry/config-only check: this must not instantiate ManagerBasedRlEnv.
    from mjlab.tasks.registry import load_env_cfg

    import hoppertrex_mjlab.tasks  # noqa: F401
    from hoppertrex_mjlab.hybrid.stair_dynamic import DYNAMIC_STAIR_TASK_ID
    from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
      DYNAMIC_STAIR_LEFT_SENSOR_NAME,
      DYNAMIC_STAIR_RIGHT_SENSOR_NAME,
    )

    cfg = load_env_cfg(DYNAMIC_STAIR_TASK_ID, play=False)
    self.assertIsInstance(cfg.scene.sensors, tuple)
    self.assertEqual(
      preflight._registered_per_wheel_sensor_names(cfg),
      (DYNAMIC_STAIR_LEFT_SENSOR_NAME, DYNAMIC_STAIR_RIGHT_SENSOR_NAME),
    )

  def test_expectation_is_exact_and_zero_update_is_honest(self) -> None:
    result = preflight._expectation_payload(
      git_sha=_GIT,
      contract_sha256="e" * 64,
      artifact_bindings=_ARTIFACTS,
      maneuver_sha256=_MANEUVER,
      maneuver_bindings=_BINDINGS,
      completed_updates=0,
    )
    self.assertEqual(result["completed_updates"], 0)
    self.assertEqual(result["source_stage5_checkpoint_sha256"], "c" * 64)
    self.assertEqual(result["artifact_bindings"], _ARTIFACTS)

  def test_maneuver_hash_and_git_drift_are_rejected(self) -> None:
    with self.assertRaisesRegex(ValueError, "maneuver hashes"):
      preflight._expectation_payload(
        git_sha=_GIT,
        contract_sha256="e" * 64,
        artifact_bindings={**_ARTIFACTS, "dynamic_maneuver_hash": "f" * 64},
        maneuver_sha256=_MANEUVER,
        maneuver_bindings=_BINDINGS,
        completed_updates=None,
      )
    with self.assertRaisesRegex(ValueError, "Git SHA"):
      preflight._expectation_payload(
        git_sha="f" * 40,
        contract_sha256="e" * 64,
        artifact_bindings=_ARTIFACTS,
        maneuver_sha256=_MANEUVER,
        maneuver_bindings=_BINDINGS,
        completed_updates=None,
      )


  def test_search_bindings_use_file_hashes_and_registered_classical_values(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = root / "stage5.pt"
      gate = root / "gate.json"
      qualification = root / "qualification.json"
      checkpoint.write_bytes(b"stage5")
      gate.write_bytes(b"gate")
      qualification.write_text(
        json.dumps(collect_with_backend(FakeBackend())), encoding="utf-8"
      )
      classical = {
        "controller_gain_hash": "1" * 64,
        "calibration_hash": "2" * 64,
        "yaw_calibration_hash": "3" * 64,
        "posture_map_hash": "4" * 64,
        "posture_artifact_hash": "5" * 64,
        "station_calibration_hash": "6" * 64,
      }
      with (
        patch.object(
          preflight,
          "_registered_classical_bindings",
          return_value=(_GIT, classical),
        ),
        patch.object(preflight, "_validate_stage5_search_source") as validate_source,
      ):
        result = preflight.collect_search_bindings(
          stage5_checkpoint=checkpoint,
          stage5_gate=gate,
          trigger_qualification=qualification,
        )
      self.assertEqual(result["git_sha"], _GIT)
      self.assertEqual(
        result["stage5_checkpoint_sha256"],
        preflight._file_sha256(checkpoint),
      )
      self.assertEqual(
        result["per_wheel_trigger_qualification_sha256"],
        preflight._file_sha256(qualification),
      )
      self.assertEqual(result["controller_gain_hash"], "1" * 64)
      validate_source.assert_called_once_with(
        checkpoint,
        gate,
        checkpoint_sha256=preflight._file_sha256(checkpoint),
      )

  def test_cli_writes_only_direct_expectation_and_never_overwrites(self) -> None:
    expectation = preflight._expectation_payload(
      git_sha=_GIT,
      contract_sha256="e" * 64,
      artifact_bindings=_ARTIFACTS,
      maneuver_sha256=_MANEUVER,
      maneuver_bindings=_BINDINGS,
      completed_updates=0,
    )
    report = {"expectation": expectation}
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / "expectation.json"
      with patch.object(preflight, "collect_runtime_preflight", return_value=report):
        self.assertEqual(
          preflight.main(
            [
              "runtime-expectation",
              "--completed-updates",
              "0",
              "--output",
              str(output),
            ]
          ),
          0,
        )
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), expectation)
        with self.assertRaises(FileExistsError):
          preflight.main(
            [
              "runtime-expectation",
              "--completed-updates",
              "0",
              "--output",
              str(output),
            ]
          )


if __name__ == "__main__":
  unittest.main()
