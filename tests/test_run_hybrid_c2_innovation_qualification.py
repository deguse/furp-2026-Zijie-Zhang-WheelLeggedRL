import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_hybrid_c2_innovation_qualification.ps1"
SIDECAR = SCRIPT.with_suffix(SCRIPT.suffix + ".sha256")
ARTIFACTS = ROOT / "docs" / "experiments" / "artifacts"


def canonical_script_hash(path: Path) -> str:
  text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace(
    "\r", "\n"
  )
  return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RunHybridC2InnovationQualificationTest(unittest.TestCase):
  def test_wrapper_pins_the_one_shot_seed3_protocol(self):
    text = SCRIPT.read_text(encoding="utf-8-sig")
    for required in (
      "43c379b919d36465ef4e666254e708e26b1a2c6e",
      "43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6",
      "hybrid_c2_innovation_qualification_v1",
      "INNOVATION_DETECTOR_QUALIFIED",
      "C2_INNOVATION_DETECTOR_UNQUALIFIED_STOP",
      "INVALID_INNOVATION_CAPTURE",
      "c2_innovation_qualification_{0}_seed3",
      "$shortSha = $fullSha.Substring(0, 7)",
      "[int]$result.protocol.seed -ne 3",
      "[int]$result.completed_cell_count -ne 18",
      "[int]$result.completed_pair_count -ne 288",
      "[int]$result.completed_candidate_count -ne 125",
      ".attempt_mask",
      "'full_true'",
      "overall_timely_min",
      "per_cell_timely_min",
      "--device', 'cuda:0'",
    ):
      self.assertIn(required, text)
    self.assertNotIn("'--seed'", text)
    self.assertNotIn("seed4", text.lower())
    self.assertNotIn("seed5", text.lower())

  def test_wrapper_pins_frozen_artifact_bytes_and_real_fields(self):
    expected = {
      ARTIFACTS
      / "c1_schedule_candidate24_1f54968_seed1"
      / "c1_schedule.json": "9b21125e7cc48be3ea61e12a67171a855892ad3ced1f54b3176ed979e76224ec",
      ARTIFACTS
      / "hybrid_runtime_seed1"
      / "velocity_calibration_seed1.json": "ef002d0d622725509b47c8ff40d8af658fd42f705bdeac67ac35bae4458f889d",
      ARTIFACTS
      / "c1_posture_requalification_seed1"
      / "posture_map_seed1_registered_p032.json": "b8e627f85b53d21dd8d9c26edbe2943151d9bcf9e5864ff998ede5f909118e23",
      ARTIFACTS
      / "c1_posture_requalification_seed1"
      / "station_calibration_seed1.json": "f22a9b66f734004ff14b6586a22a991d527f360806bbbdefe096e9f0474db72a",
      ARTIFACTS
      / "c2_innovation_predictor_2cccb36_seed1"
      / "c2_innovation_predictor.json": "fe43855f6c34b440b007c0628e0bf4aacf39d3e8c0a4b209501398c99e4ee877",
      ARTIFACTS
      / "c2_innovation_floor_b527766_seed2"
      / "c2_innovation_floor.json": "cffc0e0877025af2cfc2e7292cf7e7b0ef79e6f3c743f3ec23589f38a296b4fd",
    }
    text = SCRIPT.read_text(encoding="utf-8-sig")
    for path, expected_hash in expected.items():
      self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected_hash)
      self.assertIn(expected_hash, text)

    schedule = json.loads(next(iter(expected)).read_text(encoding="utf-8-sig"))
    calibration = json.loads(
      (
        ARTIFACTS
        / "hybrid_runtime_seed1"
        / "velocity_calibration_seed1.json"
      ).read_text(encoding="utf-8-sig")
    )
    station = json.loads(
      (
        ARTIFACTS
        / "c1_posture_requalification_seed1"
        / "station_calibration_seed1.json"
      ).read_text(encoding="utf-8-sig")
    )
    predictor = json.loads(
      (
        ARTIFACTS
        / "c2_innovation_predictor_2cccb36_seed1"
        / "c2_innovation_predictor.json"
      ).read_text(encoding="utf-8-sig")
    )
    floor = json.loads(
      (
        ARTIFACTS
        / "c2_innovation_floor_b527766_seed2"
        / "c2_innovation_floor.json"
      ).read_text(encoding="utf-8-sig")
    )
    self.assertEqual(schedule["artifact_type"], "gain_scheduled_lqr")
    self.assertIn("identification_controller_gain_hash", schedule["bindings"])
    self.assertIn("identification_calibration_hash", schedule["bindings"])
    self.assertIn("calibration_hash", calibration)
    self.assertIn("station_calibration_hash", station)
    self.assertIn("velocity_calibration_hash", predictor["bindings"])
    self.assertIsNone(predictor["bindings"]["yaw_calibration_hash"])
    self.assertEqual(len(floor["threshold_table"]), 125)
    self.assertEqual(floor["predictor_hash"], predictor["predictor_hash"])

  def test_wrapper_pins_core_producer_and_validator_sources(self):
    expected = {
      ROOT
      / "src"
      / "hoppertrex_mjlab"
      / "hybrid"
      / "innovation_detector.py": "8ff70de0ae6bb47827509860f85337e85095acebddcaa0ecc1b4b996332751fe",
      ROOT
      / "src"
      / "hoppertrex_mjlab"
      / "scripts"
      / "probe_hybrid_c2_innovation_qualification.py": "045c21a1a779cfba38672f7c589c049ab83086953ffe3ac132f852965b415cf6",
      ROOT
      / "src"
      / "hoppertrex_mjlab"
      / "scripts"
      / "validate_hybrid_c2_innovation_qualification.py": "76aaf3d9e5781e0d3c7ba35aeaa99c640f8aaee1f51e4449cf06cd6f0836bd4d",
    }
    text = SCRIPT.read_text(encoding="utf-8-sig")
    for path, expected_hash in expected.items():
      self.assertEqual(canonical_script_hash(path), expected_hash)
      self.assertIn(expected_hash, text)
    self.assertIn("$PinnedSources", text)
    self.assertIn("source_canonical_sha256", text)

  def test_wrapper_uses_fail_closed_runtime_and_atomic_outputs(self):
    text = SCRIPT.read_text(encoding="utf-8-sig")
    for required in (
      "$ErrorActionPreference = 'Stop'",
      "Set-StrictMode -Version Latest",
      "Invoke-NativeChecked",
      "Invoke-NativeLogged",
      "Tee-Object -FilePath $LogPath -Append",
      "$ErrorActionPreference = 'Continue'",
      "$FrozenSourcePathspecs",
      "'diff', '--quiet', $RequiredBase, 'HEAD', '--'",
      "transitive source dependencies drifted",
      "$MjLabSourceDeclaration",
      "pyproject.toml no longer pins the expected editable MjLab source",
      "mjlab_root=str(pathlib.Path(mjlab.__file__).resolve().parents[2])",
      "Python imports MjLab from",
      "mjlab_import_root = $importedMjLabRoot",
      "$env:PYTHONPATH",
      "Remove-Item Env:HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH",
      "--inputs-only",
      "torch.cuda.is_available()",
      "python_version = [string]$runtime.python",
      "torch_version = [string]$runtime.torch",
      "torch_cuda_version = [string]$runtime.torch_cuda",
      "cuda_device_name = [string]$runtime.cuda_device",
      "mujoco_version = [string]$runtime.mujoco",
      "warp_version = [string]$runtime.warp",
      "posture_boundary_snap_atol -ne 1.0e-7",
      "impact_truth.archived_raw_replay -ne $true",
      "outer_face_offset_from_terrain_origin_m -ne -3.0",
      "outer_face_binding_atol_m -ne 2.0e-5",
      "$ConsoleTemp = $WorkingDirectory + '.console.log'",
      "$WorkingDirectory = $OutputDirectory + '.incomplete.'",
      "$TemporaryZip = $OutputZip + '.incomplete.'",
      "SHA256SUMS.txt",
      "Compress-Archive",
      "Move-Item -LiteralPath $WorkingDirectory -Destination $OutputDirectory",
      "Move-Item -LiteralPath $OutputDirectory -Destination $WorkingDirectory",
      "ARCHIVED_THEN_NONZERO",
      "throw ('INVALID_INNOVATION_CAPTURE archived at {0}'",
    ):
      self.assertIn(required, text)
    self.assertLess(text.index("$env:PYTHONPATH"), text.index("'--help'"))
    self.assertNotIn("Invoke-Expression", text)
    self.assertNotIn("Tee-Object -LiteralPath $LogPath -Append", text)
    self.assertNotIn("uv sync", text.lower())
    self.assertNotIn("train.py", text.lower())
    self.assertNotIn("migrate_hybrid_stage", text.lower())
    self.assertNotIn("build_hybrid_stair_maneuver", text.lower())
    self.assertNotIn("--checkpoint", text.lower())
    self.assertNotIn(".pt'", text.lower())

  def test_wrapper_keeps_all_failed_as_a_zero_exit_scientific_result(self):
    text = SCRIPT.read_text(encoding="utf-8-sig")
    unqualified = text.index(
      "$result.classification -eq 'C2_INNOVATION_DETECTOR_UNQUALIFIED_STOP'"
    )
    invalid_throw = text.rindex(
      "$result.classification -eq 'INVALID_INNOVATION_CAPTURE'"
    )
    self.assertLess(unqualified, invalid_throw)
    self.assertIn("[COMPLETE] C2-j3 ran all 125 candidates; none qualified.", text)
    self.assertNotIn("C2_INNOVATION_DETECTOR_UNQUALIFIED_STOP archived at", text)

  def test_wrapper_canonical_hash_matches_sidecar(self):
    self.assertTrue(SIDECAR.is_file())
    self.assertRegex(SIDECAR.read_text(encoding="ascii").strip(), r"^[0-9a-f]{64}$")
    self.assertEqual(
      SIDECAR.read_text(encoding="ascii").strip(), canonical_script_hash(SCRIPT)
    )

  def test_gitattributes_protects_wrapper_and_sidecar(self):
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8-sig")
    self.assertIn("scripts/run_hybrid_c2_innovation_qualification.ps1 -text", attributes)
    self.assertIn(
      "scripts/run_hybrid_c2_innovation_qualification.ps1.sha256 -text",
      attributes,
    )


if __name__ == "__main__":
  unittest.main()
