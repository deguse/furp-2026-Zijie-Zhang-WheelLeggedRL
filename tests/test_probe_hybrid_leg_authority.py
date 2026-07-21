import json
from pathlib import Path
import tempfile
import unittest

from hoppertrex_mjlab.scripts.probe_hybrid_leg_authority import (
  _validated_baseline,
  parse_args,
)


class HybridLegAuthorityProbeTest(unittest.TestCase):
  def test_defaults_pin_the_pre_registered_event_count(self):
    args = parse_args(["--output", "result.json"])
    self.assertEqual(args.seed, 1)
    self.assertEqual(args.num_envs, 32)
    self.assertEqual(args.warmup_steps, 300)

  def test_baseline_artifact_requires_exactly_two_repeats(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      path = Path(temp_dir) / "baseline.json"
      path.write_text(
        json.dumps(
          {
            "probe": "hybrid_leg_authority",
            "baseline_repeats": [
              {"recovery_time_s": 1.0},
              {"recovery_time_s": 1.01},
            ],
          }
        ),
        encoding="utf-8",
      )
      self.assertEqual(len(_validated_baseline(path)), 2)
      payload = json.loads(path.read_text(encoding="utf-8"))
      payload["baseline_repeats"].pop()
      path.write_text(json.dumps(payload), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "exactly two"):
        _validated_baseline(path)


if __name__ == "__main__":
  unittest.main()
