from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_hybrid_c1_affine_center_smoke_retry.ps1"
SELF_HASH = ROOT / "scripts" / "run_hybrid_c1_affine_center_smoke_retry.ps1.sha256"


class HybridC1AffineCenterSmokeRetryWrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_pins_frozen_source_and_runtime(self):
        for fragment in (
            "'codex/p2-classical-upper-bound'",
            "'0c7bd78893998f0a1c6d58615fb3ea7fd97f0bdd'",
            "'43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6'",
            "'10e0f8f498107406e969e9f7d8390f8ac8c22f5838b60d5254e65196453eb4f9'",
            "'6609c3086a88b07a9a903c15897a4aab838c80ab9bc23ddf862618a16793d341'",
            "'e5d692831b2c676ecbe37d3124527e72abf146b2708919fdce8cde9a68fec1ee'",
            "'a2d65437f094a604d4f47145d63c7342e81d59cef096d7153fca27ba64fcd1b8'",
            "c1_affine_identification_nodes_0c7bd78_seed1",
            "$sourceHashLines.Count -ne 30",
            "Affine center-smoke retry wrapper self-hash mismatch",
        ):
            self.assertIn(fragment, self.source)

    def test_requires_affine_incumbent_and_drive_retention(self):
        for fragment in (
            "$result.affine_incumbent.flat_gate_passed -ne $true",
            "$result.affine_incumbent.anchor_alpha",
            "$candidate.anchor_alpha - 0.25",
            "$MinimumCommandGainRatio = 0.70",
            "$fact.command_gain_ratio",
            "'Retry classification, pass count, and next step disagree.'",
            "'--settle-steps', '100'",
            "'--measure-steps', '200'",
            "'--vx-check', '0.05'",
        ):
            self.assertIn(fragment, self.source)

    def test_is_atomic_and_does_not_recollect_or_train(self):
        for fragment in (
            "'.incomplete.' + $runToken",
            "Refusing to overwrite existing C1 retry output",
            "Move-Item -LiteralPath $WorkingDirectory -Destination $OutputDirectory",
            "Move-Item -LiteralPath $WorkingZip -Destination $OutputZip",
            "'SHA256SUMS.txt'",
            "'c1_affine_center_smoke_retry'",
        ):
            self.assertIn(fragment, self.source)
        for forbidden in (
            "collect_hybrid_identification",
            "rsl_rl.train",
            "migrate_hybrid_stage",
            "checkpoint-file",
            "probe_hybrid_stair",
            "hoppertrex_mjlab.scripts.run_hybrid_cem",
            "hoppertrex_mjlab.scripts.rsl_rl.train_stair_ppo",
        ):
            self.assertNotIn(forbidden.lower(), self.source.lower())

    def test_self_hash_matches_wrapper_bytes(self):
        import hashlib

        self.assertEqual(
            hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
            SELF_HASH.read_text(encoding="ascii").strip(),
        )


if __name__ == "__main__":
    unittest.main()
