from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_hybrid_c1_affine_identification_collection.ps1"
SELF_HASH = (
    ROOT / "scripts" / "run_hybrid_c1_affine_identification_collection.ps1.sha256"
)


class HybridC1IdentificationCollectionWrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_pins_branch_runtime_and_all_frozen_artifacts(self):
        for fragment in (
            "'codex/p2-classical-upper-bound'",
            "'fc940b9f0116608bfdfc2e08f996ecd5e9e76e5e'",
            "'43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6'",
            "'663ab77f77521581cde77ea2bd8c72c7f395f33b05b62348ef6d82a752aad7fc'",
            "'ef002d0d622725509b47c8ff40d8af658fd42f705bdeac67ac35bae4458f889d'",
            "'b8e627f85b53d21dd8d9c26edbe2943151d9bcf9e5864ff998ede5f909118e23'",
            "'4ae258eaf73121fd1cffc1186c5611b20a3c95b1ef684060fafa39383b55ca06'",
            "'f22a9b66f734004ff14b6586a22a991d527f360806bbbdefe096e9f0474db72a'",
            "'c003192963b257c8d497ffd347be2cd60695c5ce8653932403709d8193c88e55'",
            "'c00e859b3093b4812d54799253accdaeb99171a2cf4028b08bc39e68eaaa7d8a'",
            "'run_hybrid_c1_schedule_preflight.ps1'",
            "$env:PYTHONPATH =",
            "Affine collection wrapper self-hash mismatch",
        ):
            self.assertIn(fragment, self.source)

    def test_pins_registered_nine_node_collection_protocol(self):
        for fragment in (
            "$HeightNodes = @(0.2907321708, 0.3092089487, 0.3276857266)",
            "$PitchNodes = @(-0.032, 0.0, 0.032)",
            "'--device', 'cuda:0'",
            "'--num-envs', '32'",
            "'--steps', '2500'",
            "'--warmup-steps', '250'",
            "'--hold-steps', '5'",
            "'--balance-amplitude', '0.35'",
            "'--heldout-fraction', '0.20'",
            "'--seed', '1'",
            "'hybrid_lqr_affine_equilibrium_v3'",
            "'delta_actual_signed_balance_wheel_velocity_target'",
            "$EquilibriumWindowSteps = 100",
            "'c1_affine_identification_collection'",
            "'hoppertrex_mjlab.scripts.evaluate_hybrid_c1_affine_center_smoke'",
            "'--settle-steps', '100'",
            "'--measure-steps', '200'",
            "'--git-sha', $fullSha",
            "'--mjlab-git-sha', $mjlabSha",
            "'AFFINE_CENTER_SMOKE_HAS_CANDIDATES'",
            "'AFFINE_CENTER_SMOKE_NO_CANDIDATE_STOP'",
            "'DOWNLOAD_FOR_REVIEW'",
            "$centerSmoke.fit_qualification.minimum_controllability_rank",
            "$centerSmoke.fit_qualification.maximum_heldout_nrmse -gt 0.15",
            "$centerSmoke.fit_qualification.fallback_count",
            "'Affine center smoke classification, pass count, and next step disagree.'",
        ):
            self.assertIn(fragment, self.source)

    def test_uses_atomic_non_overwriting_outputs_and_sha_manifest(self):
        for fragment in (
            "'.incomplete.' + $runToken",
            "Refusing to overwrite existing C1 output",
            "Move-Item -LiteralPath $WorkingDirectory -Destination $OutputDirectory",
            "Move-Item -LiteralPath $WorkingZip -Destination $OutputZip",
            "Remove-Item -LiteralPath $OutputZip -Force",
            "'SHA256SUMS.txt'",
            "Compress-Archive -Path (Join-Path $WorkingDirectory '*')",
            'Write-Host "NEXT=$nextStep"',
        ):
            self.assertIn(fragment, self.source)

    def test_rejects_training_migration_checkpoint_and_later_stages(self):
        for forbidden in (
            "hoppertrex_mjlab.scripts.rsl_rl.train",
            "migrate_hybrid_stage",
            "--checkpoint-file",
            "run_hybrid_leg_authority_seed1.ps1",
            "probe_hybrid_stair",
            "cem",
            "ppo",
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
