import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_hybrid_c2_predictor_identification.ps1"
SIDECAR = SCRIPT.with_suffix(SCRIPT.suffix + ".sha256")


def canonical_script_hash(path: Path) -> str:
  text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
  return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RunHybridC2PredictorIdentificationTest(unittest.TestCase):
  def test_wrapper_pins_protocol_and_validates_every_node(self):
    text = SCRIPT.read_text(encoding="utf-8-sig")
    for required in (
      "4b0210420d3dd35f5c8b74561b49bcb4e8b49034",
      "hybrid_c2_predictor_identification_v1",
      "PREDICTOR_IDENTIFICATION_QUALIFIED",
      "regression_rank",
      "heldout_nrmse",
      "termination_count",
      "non_wheel_contact_count",
      "raw_sha256",
      "parse_innovation_predictor",
      "shaped_posture",
      "FREEZE_AND_INDEPENDENT_AUDIT_BEFORE_C2_J2",
    ):
      self.assertIn(required, text)
    self.assertNotIn("Invoke-Expression", text)

  def test_wrapper_canonical_hash_matches_sidecar(self):
    self.assertTrue(SIDECAR.is_file())
    self.assertEqual(SIDECAR.read_text(encoding="ascii").strip(), canonical_script_hash(SCRIPT))


if __name__ == "__main__":
  unittest.main()
