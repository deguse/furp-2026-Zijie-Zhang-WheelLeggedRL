import unittest
from types import SimpleNamespace

import torch

from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
  POSTURE_HEIGHT_SLEW_RATE,
  POSTURE_PITCH_SLEW_RATE,
  PostureCommandCfg,
)


def _cfg(**overrides) -> PostureCommandCfg:
  kwargs = dict(
    resampling_time_range=(1.0e9, 1.0e9),
    height_range=(0.3024, 0.3267),
    pitch_range=(-0.0791, 0.0760),
  )
  kwargs.update(overrides)
  return PostureCommandCfg(**kwargs)


def _term(cfg: PostureCommandCfg):
  env = SimpleNamespace(num_envs=3, device="cpu")
  term = cfg.build(env)
  term.reset(torch.arange(3))
  return term


class PostureCommandShapingTest(unittest.TestCase):
  def test_defaults_carry_preliminary_probe_rates(self):
    cfg = _cfg()
    self.assertEqual(cfg.height_slew_rate, POSTURE_HEIGHT_SLEW_RATE)
    self.assertEqual(cfg.pitch_slew_rate, POSTURE_PITCH_SLEW_RATE)

  def test_cfg_rejects_non_positive_rates(self):
    with self.assertRaisesRegex(ValueError, "positive or None"):
      _cfg(height_slew_rate=0.0)
    with self.assertRaisesRegex(ValueError, "positive or None"):
      _cfg(pitch_slew_rate=-0.1)
    _cfg(height_slew_rate=None, pitch_slew_rate=None)

  def test_reset_snaps_command_to_fresh_target(self):
    term = _term(_cfg())
    torch.testing.assert_close(term.command, term.target)
    # float32 storage of the float64 range bounds needs a hair of slack.
    self.assertTrue(
      bool((term.target[:, 0] >= 0.3024 - 1.0e-6).all())
      and bool((term.target[:, 0] <= 0.3267 + 1.0e-6).all())
    )

  def test_command_slews_toward_target_at_axis_rates(self):
    term = _term(_cfg(height_slew_rate=0.025, pitch_slew_rate=0.155))
    term.target[:, 0] = term.command[:, 0] + 0.02
    term.target[:, 1] = term.command[:, 1] - 0.10
    start = term.command.clone()
    dt = 0.02

    term.compute(dt)
    torch.testing.assert_close(
      term.command[:, 0], start[:, 0] + 0.025 * dt
    )
    torch.testing.assert_close(
      term.command[:, 1], start[:, 1] - 0.155 * dt
    )

    # Ramp must reach the target exactly and then hold with no overshoot.
    for _ in range(200):
      term.compute(dt)
    torch.testing.assert_close(term.command, term.target)
    settled = term.command.clone()
    term.compute(dt)
    torch.testing.assert_close(term.command, settled)

  def test_none_rate_publishes_target_instantly(self):
    term = _term(_cfg(height_slew_rate=None, pitch_slew_rate=None))
    term.target[:, 0] = 0.99
    term.target[:, 1] = -0.99
    term.compute(0.02)
    torch.testing.assert_close(term.command, term.target)

  def test_degenerate_range_keeps_command_bitwise_constant(self):
    # Stages 0-2 use a collapsed posture range; shaping must be a no-op so
    # frozen-stage behavior stays byte-identical with the rates configured.
    term = _term(
      _cfg(height_range=(0.325, 0.325), pitch_range=(0.0, 0.0))
    )
    before = term.command.clone()
    for _ in range(10):
      term.compute(0.02)
    self.assertTrue(torch.equal(term.command, before))


if __name__ == "__main__":
  unittest.main()
