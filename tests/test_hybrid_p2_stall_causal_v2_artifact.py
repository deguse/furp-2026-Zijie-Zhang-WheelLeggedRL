from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = (
  REPOSITORY
  / "docs"
  / "experiments"
  / "artifacts"
  / "hybrid_p2_stall_causal_v2_seed1"
)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class HybridP2StallCausalV2ArtifactTest(unittest.TestCase):
  def test_frozen_bytes_and_provenance(self) -> None:
    manifest = json.loads(
      (ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    for name, expected in manifest["files"].items():
      self.assertEqual(_sha256(ARTIFACT_DIR / name), expected)

    result = json.loads(
      (ARTIFACT_DIR / "stall_causal_v2.json").read_text(
        encoding="utf-8-sig"
      )
    )
    self.assertEqual(result["classification"], "ANALYSIS_READY")
    self.assertTrue(result["evidence_eligible"])
    self.assertFalse(result["promotion_eligible"])
    self.assertFalse(result["training_eligible"])
    self.assertIsNone(result["single_cause_label"])
    self.assertIsNone(result["checkpoint"])
    self.assertIsNone(result["yaw_calibration_hash"])
    self.assertEqual(len(result["trials"]), 64)
    self.assertEqual(len(result["paired_captures"]), 32)
    self.assertEqual(result["git_sha"], manifest["git_sha"])
    self.assertEqual(result["mjlab_git_sha"], manifest["mjlab_git_sha"])
    self.assertEqual(
      result["controller_gain_hash"],
      manifest["provenance"]["controller_gain_hash"],
    )
    self.assertEqual(
      result["calibration_hash"],
      manifest["provenance"]["velocity_calibration_hash"],
    )
    self.assertEqual(
      result["posture_artifact_hash"],
      manifest["provenance"]["posture_artifact_hash"],
    )
    self.assertEqual(
      result["station_calibration_hash"],
      manifest["provenance"]["station_calibration_hash"],
    )


if __name__ == "__main__":
  unittest.main()
