from __future__ import annotations

import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts" / "run_hybrid_c1_schedule_preflight.ps1"


class HybridC1SchedulePreflightTest(unittest.TestCase):
  def test_protocol_is_pinned_and_training_free(self) -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for token in (
      "codex/p2-classical-upper-bound",
      "59ff3cd4d86c569d7d0ea8e207640a6d11c178ab",
      "9787fe5b4fe00b7c77665e7da73fa359f0ee196c",
      "43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6",
      "0.2907321708",
      "0.3092089487",
      "0.3276857266",
      "0.032, 0.024, 0.016",
      "uv sync --frozen --python 3.11",
      "..\\\\mjlab-main",
      "..\\\\..\\\\..\\\\mjlab-main",
      "C1_GPU_NODE_COLLECTION_READY_NO_TRAINING",
    ):
      self.assertIn(token, text)
    for forbidden in (" train ", "checkpoint", "migrate_hybrid_stage", "P3"):
      self.assertNotIn(forbidden, text)


if __name__ == "__main__":
  unittest.main()
