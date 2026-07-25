from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_hybrid_c1_flat_gate.ps1"
SELF_HASH = ROOT / "scripts" / "run_hybrid_c1_flat_gate.ps1.sha256"


class HybridC1FlatGateWrapperTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    cls.source = SCRIPT.read_text(encoding="utf-8")

  def test_pins_branch_remote_implementation_and_self(self) -> None:
    for fragment in (
      "'codex/p2-classical-upper-bound'",
      "'ffbb01850787ceead53ba407a0a7bf9c6f6a9b11'",
      "'merge-base', '--is-ancestor', $RequiredImplementation, 'HEAD'",
      "'fetch', '--quiet', 'origin', $RequiredBranch",
      'git rev-parse "origin/$RequiredBranch"',
      "Repository must be clean",
      "Wrapper self-hash mismatch",
    ):
      self.assertIn(fragment, self.source)
    expected = SELF_HASH.read_text(encoding="ascii").strip()
    actual = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    self.assertEqual(actual, expected)
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    self.assertIn("scripts/run_hybrid_c1_flat_gate.ps1 -text", attributes)
    self.assertIn("scripts/run_hybrid_c1_flat_gate.ps1.sha256 -text", attributes)

  def test_pins_mjlab_nodes_and_all_runtime_artifacts(self) -> None:
    for fragment in (
      "'43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6'",
      "'c1_identification_nodes_e54bd1a_seed1'",
      "'364590b8d9f2f5c66fdaac2b3fa124ee914236e33f6fc47e31e75f64d53c72e2'",
      "'663ab77f77521581cde77ea2bd8c72c7f395f33b05b62348ef6d82a752aad7fc'",
      "'ef002d0d622725509b47c8ff40d8af658fd42f705bdeac67ac35bae4458f889d'",
      "'b8e627f85b53d21dd8d9c26edbe2943151d9bcf9e5864ff998ede5f909118e23'",
      "'f22a9b66f734004ff14b6586a22a991d527f360806bbbdefe096e9f0474db72a'",
      "'c003192963b257c8d497ffd347be2cd60695c5ce8653932403709d8193c88e55'",
      "'8fee25a0339dd1e99127cbed912941dc3ad8ef2030ce49a0d310d1563cb87d98'",
      "'f62648b57bd17a3503bcbdbf58f349f91fcd8de8ef0cf04551c200401233ed01'",
      "'3b96fd3dae66ad781b5b875c74184db101c42da02c53dfcc40a5137a6b5de11a'",
      "'c00e859b3093b4812d54799253accdaeb99171a2cf4028b08bc39e68eaaa7d8a'",
      'path = "../mjlab-main"',
      "pyproject.toml no longer pins the expected editable MjLab source",
      "pathlib.Path(mjlab.__file__).resolve().parents[2]",
      "Python imports MjLab from",
      "function Find-NodesDirectory",
      "Join-Path $Repository 'experiments'",
      "$NodesDirectory = Find-NodesDirectory -Repository $RepoRoot",
    ):
      self.assertIn(fragment, self.source)

  def test_pins_powershell_cuda_and_evaluator_protocol(self) -> None:
    for fragment in (
      "$PSVersionTable.PSEdition -ne 'Desktop'",
      "$PSVersionTable.PSVersion.Major -ne 5",
      "$PSVersionTable.PSVersion.Minor -ne 1",
      "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader",
      "'uv' -Arguments @(",
      "'sync', '--frozen', '--python', '3.11'",
      "'hoppertrex_mjlab.scripts.evaluate_hybrid_c1_flat_gate', '--help'",
      "'--task', 'HopperTrex-Hybrid-v2-Stage3'",
      "'--device', 'cuda:0'",
      "'--num-envs', '16'",
      "'--settle-steps', '100'",
      "'--measure-steps', '200'",
      "'--vx-check', '0.05'",
      "Remove-Item Env:HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH",
    ):
      self.assertIn(fragment, self.source)

  def test_atomic_nonoverwriting_evidence_and_all_failed_semantics(self) -> None:
    for fragment in (
      "'experiments/c1_flat_gate_' + $shortSha + '_seed1'",
      "Refusing to overwrite existing C1 flat-gate output",
      "'.incomplete.' + $runToken",
      "'flat_gate_evaluation_detail.json'",
      "'flat_gate_adjudication.json'",
      "'flat_gate_selection.json'",
      "'console.log'",
      "'protocol_note.json'",
      "'SHA256SUMS.txt'",
      "Compress-Archive",
      "Move-Item -LiteralPath $WorkingDirectory -Destination $OutputDirectory",
      "Move-Item -LiteralPath $WorkingZip -Destination $OutputZip",
      "Remove-Item -LiteralPath $OutputZip -Force",
      "'NO_QR_CANDIDATE_PASSED_FLAT_GATE'",
      "'STOP'",
      "'DOWNLOAD_FOR_OFFLINE_SCHEDULE_BUILD'",
    ):
      self.assertIn(fragment, self.source)
    zip_publish = self.source.index(
      "Move-Item -LiteralPath $WorkingZip -Destination $OutputZip"
    )
    directory_publish = self.source.index(
      "Move-Item -LiteralPath $WorkingDirectory -Destination $OutputDirectory"
    )
    self.assertLess(zip_publish, directory_publish)

  def test_contains_no_later_stage_or_model_mutation_entrypoint(self) -> None:
    lowered = self.source.lower()
    for forbidden in (
      "hoppertrex_mjlab.scripts.rsl_rl.train",
      "migrate_hybrid_stage",
      "--checkpoint-file",
      "build_hybrid_controller_schedule",
      "fit_hybrid_stair_contact_detector",
      "probe_hybrid_stair",
      "run_hybrid_leg_authority_seed1",
      "cem.py",
      "ppo.py",
    ):
      self.assertNotIn(forbidden, lowered)


if __name__ == "__main__":
  unittest.main()
