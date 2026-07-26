import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "docs"
    / "experiments"
    / "artifacts"
    / "c1_affine_center_smoke_failure_seed1"
    / "manifest.json"
)


class C1AffineCenterSmokeFailureArtifactTest(unittest.TestCase):
    def test_manifest_records_valid_velocity_only_failure(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["classification"], "AFFINE_CENTER_SMOKE_NO_CANDIDATE_STOP"
        )
        self.assertEqual(payload["source_zip_sha256"], "10e0f8f498107406e969e9f7d8390f8ac8c22f5838b60d5254e65196453eb4f9")
        self.assertEqual(payload["candidate_count"], 27)
        self.assertEqual(payload["safety_clean_candidate_count"], 27)
        self.assertEqual(payload["pitch_cap_pass_candidate_count"], 27)
        self.assertEqual(payload["pitch_rate_cap_pass_candidate_count"], 27)
        self.assertEqual(payload["velocity_cap_pass_candidate_count"], 0)
        self.assertTrue(payload["legacy_incumbent"]["flat_gate_passed"])
        self.assertFalse(payload["best_candidate"]["flat_gate_passed"])
        self.assertGreater(
            payload["best_candidate"]["worst_velocity_error"],
            payload["caps"]["worst_velocity_error"],
        )
        self.assertTrue(
            payload["codex_correction"]["affine_incumbent_zero_blend_was_not_tested"]
        )
        self.assertEqual(
            payload["codex_correction"]["retry_expected_anchor_alpha"], 0.25
        )


if __name__ == "__main__":
    unittest.main()
