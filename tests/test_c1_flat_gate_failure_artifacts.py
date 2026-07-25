from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
  ROOT / "docs" / "experiments" / "artifacts" / "c1_flat_gate_failure_seed1"
)


class C1FlatGateFailureArtifactTest(unittest.TestCase):
  def test_manifest_hashes_every_frozen_file(self) -> None:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
      self.assertEqual(
        hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest(), expected
      )

  def test_adjudication_is_valid_all_failed_stop_evidence(self) -> None:
    adjudication = json.loads(
      (ARTIFACT / "flat_gate_adjudication.json").read_text(encoding="utf-8-sig")
    )
    self.assertEqual(
      adjudication["classification"], "NO_QR_CANDIDATE_PASSED_FLAT_GATE"
    )
    self.assertEqual(adjudication["completed_candidate_count"], 27)
    self.assertEqual(adjudication["completed_node_fit_count"], 243)
    self.assertEqual(adjudication["passed_candidate_count"], 0)
    self.assertEqual(adjudication["next_step"], "STOP")
    self.assertTrue(adjudication["evidence_eligible"])
    self.assertFalse(adjudication["promotion_eligible"])
    self.assertFalse(adjudication["training_eligible"])
    self.assertIsNone(adjudication["selected_candidate_index"])

  def test_detail_reproduces_registered_failure_diagnosis(self) -> None:
    detail = json.loads(
      (ARTIFACT / "flat_gate_evaluation_detail.json").read_text(
        encoding="utf-8-sig"
      )
    )
    candidates = detail["candidates"]
    self.assertEqual(len(candidates), 27)
    self.assertEqual(sum(item["safety_clean"] for item in candidates), 0)
    self.assertEqual(
      max(
        cell["non_wheel_contact_rate"]
        for item in candidates
        for cell in item["cells"]
      ),
      0.0,
    )
    self.assertEqual(
      min(
        sum(cell["terminated_events"] for cell in item["cells"])
        for item in candidates
      ),
      164,
    )
    self.assertAlmostEqual(
      min(item["worst_velocity_error"] for item in candidates),
      0.4038919687271118,
    )


if __name__ == "__main__":
  unittest.main()
