from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from hoppertrex_mjlab.scripts import probe_hybrid_c2_paired_capture_v1 as probe


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


if __name__ == "__main__":
  unittest.main()
