import unittest

from hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate import (
  STAGE1_KICK_LIN_X,
  STAGE1_KICK_PITCH_RATE,
  STAGE5_RECOVERY_KICK_SCALE,
)
from hoppertrex_mjlab.scripts.rsl_rl.play_with_manual_push import (
  RECOVERY_KICK_LIN_X,
  RECOVERY_KICK_PITCH_RATE,
  _push_payload,
)


class RecoveryKickDemoTest(unittest.TestCase):
  def test_demo_kick_matches_the_preregistered_gate_magnitude(self):
    self.assertAlmostEqual(
      RECOVERY_KICK_LIN_X, STAGE5_RECOVERY_KICK_SCALE * STAGE1_KICK_LIN_X
    )
    self.assertAlmostEqual(
      RECOVERY_KICK_PITCH_RATE,
      STAGE5_RECOVERY_KICK_SCALE * STAGE1_KICK_PITCH_RATE,
    )

  def test_recovery_x_buttons_apply_the_full_combined_kick(self):
    payload = _push_payload("+X", recovery=True)
    self.assertAlmostEqual(payload["x"], RECOVERY_KICK_LIN_X)
    self.assertAlmostEqual(payload["pitch"], RECOVERY_KICK_PITCH_RATE)
    negative = _push_payload("-X", recovery=True)
    self.assertAlmostEqual(negative["x"], -RECOVERY_KICK_LIN_X)
    self.assertAlmostEqual(negative["pitch"], -RECOVERY_KICK_PITCH_RATE)

  def test_light_and_strong_buttons_are_unchanged(self):
    self.assertEqual(
      _push_payload("+X", light=True),
      {"type": "manual_push", "x": 0.08, "pitch": 0.0},
    )
    self.assertEqual(
      _push_payload("+X", light=False),
      {"type": "manual_push", "x": 0.15, "pitch": 0.0},
    )


if __name__ == "__main__":
  unittest.main()
