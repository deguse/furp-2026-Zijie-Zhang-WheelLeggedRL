import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_hybrid_c2_transition_floor.ps1"
SIDECAR = SCRIPT.with_suffix(SCRIPT.suffix + ".sha256")


def canonical_script_hash(path: Path) -> str:
  text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
  return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RunHybridC2TransitionFloorTest(unittest.TestCase):
  def test_wrapper_pins_protocol_and_validates_all_raw_cells(self):
    text = SCRIPT.read_text(encoding="utf-8-sig")
    for required in (
      "16da4416c80a3a6bbbe53c39c21a15ad45bdc69f",
      "hybrid_c2_transition_floor_v1",
      "validate_hybrid_c2_transition_floor",
      "INNOVATION_FLOOR_QUALIFIED",
      "PREDICTOR_DOMAIN_UNCOVERED_STOP",
      "INVALID_INNOVATION_FLOOR",
      "threshold_table_hash",
      "floor_hash",
      "SHA256SUMS.txt",
      "TemporaryZip",
      "Move-Item -LiteralPath $OutputDirectory -Destination $WorkingDirectory",
      "FREEZE_AND_INDEPENDENT_AUDIT_BEFORE_C2_J3",
      "ARCHIVE_EVIDENCE_AND_STOP_AT_USER_ROUTE_DECISION",
    ):
      self.assertIn(required, text)
    self.assertNotIn("Invoke-Expression", text)

  def test_wrapper_canonical_hash_matches_sidecar(self):
    self.assertTrue(SIDECAR.is_file())
    self.assertEqual(
      SIDECAR.read_text(encoding="ascii").strip(), canonical_script_hash(SCRIPT)
    )


if __name__ == "__main__":
  unittest.main()
