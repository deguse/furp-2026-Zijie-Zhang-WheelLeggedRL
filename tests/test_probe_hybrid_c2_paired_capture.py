from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from hoppertrex_mjlab.scripts import probe_hybrid_c2_paired_capture_v1 as probe
from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
  make_hoppertrex_hybrid_env_cfg,
)


class CaptureProvenanceContractTest(unittest.TestCase):
  """Locks the payload keys the machine-room wrapper hard-fails without.

  Regression for the 2026-07-26 finding: the probe once spread
  ``hybrid_provenance_lines(env)`` (a ``list[str]``) with ``**`` into the
  payload, which would TypeError after the full GPU capture and leave the
  wrapper with none of the provenance keys it validates.
  """

  WRAPPER_CONSUMED_KEYS = (
    "git_sha",
    "mjlab_git_sha",
    "calibration_hash",
    "posture_artifact_hash",
    "station_calibration_hash",
  )

  def _fake_cfg(self) -> SimpleNamespace:
    action_cfg = SimpleNamespace(
      controller_gain_hash="a" * 64,
      calibration_hash="b" * 64,
      posture_artifact_hash="c" * 64,
      station_calibration_hash="d" * 64,
    )
    return SimpleNamespace(actions={"hybrid_wheel_leg": action_cfg})

  def test_provenance_is_mapping_with_wrapper_keys(self) -> None:
    with (
      mock.patch.object(probe.stair, "_git_sha", side_effect=["0" * 40, "1" * 40]),
      mock.patch.object(
        probe.stair, "_runtime_metadata", return_value={"python": "x"}
      ),
    ):
      provenance = probe._capture_provenance(self._fake_cfg(), "cpu")
    self.assertIsInstance(provenance, dict)
    for key in self.WRAPPER_CONSUMED_KEYS:
      self.assertIn(key, provenance)
    merged = {"schema_version": 1, **provenance}
    self.assertEqual(merged["git_sha"], "0" * 40)
    self.assertEqual(merged["mjlab_git_sha"], "1" * 40)
    self.assertEqual(merged["calibration_hash"], "b" * 64)
    self.assertEqual(merged["posture_artifact_hash"], "c" * 64)
    self.assertEqual(merged["station_calibration_hash"], "d" * 64)

  def test_provenance_lines_is_not_a_mapping(self) -> None:
    lines = probe.hybrid_provenance_lines(self._fake_cfg())
    self.assertIsInstance(lines, list)

  def test_classifications_match_wrapper_allowed_set(self) -> None:
    self.assertEqual(
      probe.CLASSIFICATIONS, ("ANALYSIS_READY", "INVALID_CAPTURE")
    )

  def test_task_identity_matches_wrapper_expectation(self) -> None:
    # The wrapper hard-fails unless payload task equals this registry id;
    # the payload value comes from the env cfg built via stair.TASK. The
    # 364e053 machine-room artifact proved the emitted value is Stage5.
    self.assertEqual(probe.stair.TASK, "HopperTrex-Hybrid-v2-Stage5")


class C2ScheduleStackBindingTest(unittest.TestCase):
  """Builds a stage cfg on the real frozen C2 artifact stack.

  Regression for the 2026-07-27 machine-room preflight crash: with a
  gain-scheduled controller artifact, companion-artifact binding checks
  compared against the schedule_hash instead of the schedule's registered
  identification_controller_gain_hash, so the registered C2 stack could
  never load.
  """

  ARTIFACTS = Path(__file__).resolve().parents[1] / (
    "docs/experiments/artifacts"
  )

  def test_stage5_cfg_builds_with_registered_c2_stack(self) -> None:
    cfg = make_hoppertrex_hybrid_env_cfg(
      stage=5,
      play=True,
      controller_path=self.ARTIFACTS
      / "c1_schedule_candidate24_1f54968_seed1/c1_schedule.json",
      calibration_path=self.ARTIFACTS
      / "hybrid_runtime_seed1/velocity_calibration_seed1.json",
      posture_map_path=self.ARTIFACTS
      / "c1_posture_requalification_seed1/posture_map_seed1_registered_p032.json",
      station_calibration_path=self.ARTIFACTS
      / "c1_posture_requalification_seed1/station_calibration_seed1.json",
    )
    action_cfg = cfg.actions["hybrid_wheel_leg"]
    self.assertIsNotNone(action_cfg.controller_schedule)
    self.assertEqual(
      action_cfg.controller_schedule.schedule_hash,
      "8fe8548bca85978c164bbd7de39d2d6463cdfd8d7ab91796cf57696b0f64e203",
    )
    self.assertEqual(
      action_cfg.calibration_hash,
      "f62648b57bd17a3503bcbdbf58f349f91fcd8de8ef0cf04551c200401233ed01",
    )
    self.assertEqual(
      action_cfg.posture_artifact_hash,
      "3b96fd3dae66ad781b5b875c74184db101c42da02c53dfcc40a5137a6b5de11a",
    )
    self.assertEqual(
      action_cfg.station_calibration_hash,
      "c00e859b3093b4812d54799253accdaeb99171a2cf4028b08bc39e68eaaa7d8a",
    )


if __name__ == "__main__":
  unittest.main()
