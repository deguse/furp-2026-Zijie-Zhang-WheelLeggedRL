import math
import unittest

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from hoppertrex_mjlab.scripts.probe_hybrid_kick_sweep import (
  parse_args,
  sweep_payload,
  sweep_postures,
)
from hoppertrex_mjlab.scripts.probe_hybrid_posture_transition import (
  run_kick_cell,
)


def _cell(height, pitch, scale, recovery, terminated) -> dict[str, float]:
  return {
    "target_height": height,
    "target_pitch": pitch,
    "kick_scale": scale,
    "kick_lin_x": scale * 0.04,
    "kick_pitch_rate": scale * 0.06,
    "recovery_time_s": recovery,
    "kick_event_count": 64.0,
    "post_kick_lin_x_abs_max": 0.1 * scale,
    "non_wheel_contact_rate": 0.0,
    "terminated_events": terminated,
  }


class SweepProtocolTest(unittest.TestCase):
  def test_postures_are_center_weak_corner_and_opposite(self):
    postures = sweep_postures((0.3024, 0.3267), (-0.0791, 0.0760))
    self.assertEqual(len(postures), 3)
    self.assertAlmostEqual(postures[0][0], 0.31455, places=6)
    self.assertEqual(postures[1], (0.3267, 0.0760))
    self.assertEqual(postures[2], (0.3024, -0.0791))

  def test_default_arguments(self):
    args = parse_args([])
    self.assertEqual(args.kick_scales, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0))
    self.assertEqual(args.kick_interval, 300)
    self.assertIsNone(args.fit_output)

  def test_payload_reports_knee_and_max_survived_scale(self):
    cells = [
      _cell(0.315, 0.0, 1.0, 0.25, 0.0),
      _cell(0.315, 0.0, 2.0, 0.60, 0.0),
      _cell(0.315, 0.0, 4.0, 1.90, 0.0),
      _cell(0.315, 0.0, 6.0, 2.50, 5.0),
      _cell(0.327, 0.076, 1.0, 0.40, 0.0),
      _cell(0.327, 0.076, 2.0, 1.20, 3.0),
    ]
    payload = sweep_payload(
      cells=cells,
      controller_gain_hash="gain" * 16,
      controller_qualified=True,
      posture_map_hash="map" * 16,
      posture_map_qualified=True,
      station_calibration_hash="station-hash",
      station_calibration_qualified=True,
      source_probe={"git_sha": "test"},
    )

    self.assertEqual(payload["kind"], "kick_magnitude_sweep_qualification")
    per_posture = payload["summary"]["per_posture"]
    center = per_posture["(0.3150,+0.0000)"]
    self.assertEqual(center["baseline_recovery_time_s"], 0.25)
    self.assertEqual(center["termination_knee_scale"], 6.0)
    self.assertEqual(center["max_survived_scale"], 4.0)
    self.assertEqual(center["max_recovery_time_s"], 2.50)
    weak = per_posture["(0.3270,+0.0760)"]
    self.assertEqual(weak["termination_knee_scale"], 2.0)
    self.assertEqual(weak["max_survived_scale"], 1.0)
    self.assertEqual(payload["summary"]["terminated_events_total"], 8.0)

    with self.assertRaisesRegex(ValueError, "at least one"):
      sweep_payload(
        cells=[],
        controller_gain_hash=None,
        controller_qualified=False,
        posture_map_hash=None,
        posture_map_qualified=False,
        station_calibration_hash=None,
        station_calibration_qualified=False,
        source_probe={},
      )


class SweepSmokeTest(unittest.TestCase):
  def test_scaled_kick_cell_runs_on_cpu_with_default_artifacts(self):
    cfg = load_env_cfg("HopperTrex-Hybrid-v2-Stage3", play=True)
    cfg.scene.num_envs = 1
    if cfg.scene.terrain is not None:
      cfg.scene.terrain.num_envs = 1
    postures = sweep_postures(
      tuple(cfg.commands["posture"].height_range),
      tuple(cfg.commands["posture"].pitch_range),
    )

    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
      mild = run_kick_cell(
        env,
        height=postures[0][0],
        pitch=postures[0][1],
        kicks=1,
        kick_interval=25,
        settle_steps=5,
        kick_scale=1.0,
      )
      strong = run_kick_cell(
        env,
        height=postures[0][0],
        pitch=postures[0][1],
        kicks=1,
        kick_interval=25,
        settle_steps=5,
        kick_scale=3.0,
      )
    finally:
      env.close()

    self.assertTrue(math.isfinite(mild["recovery_time_s"]))
    self.assertEqual(mild["kick_scale"], 1.0)
    self.assertAlmostEqual(strong["kick_lin_x"], 0.12, places=9)
    # A 3x impulse must displace the plant visibly harder than 1x.
    self.assertGreater(
      strong["post_kick_lin_x_abs_max"],
      mild["post_kick_lin_x_abs_max"],
    )


if __name__ == "__main__":
  unittest.main()
