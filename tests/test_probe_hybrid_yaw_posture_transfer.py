import math
import unittest

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from hoppertrex_mjlab.scripts.probe_hybrid_yaw_posture_transfer import (
  DEVIATION_LIMIT,
  parse_args,
  run_transfer_cell,
  transfer_deviation_summary,
)


def _cell(height, pitch, action, transfer) -> dict[str, float]:
  return {
    "target_height": height,
    "target_pitch": pitch,
    "yaw_action": action,
    "mean_mapped_yaw": action,
    "mean_body_yaw": action * transfer,
    "transfer": transfer,
    "lin_x_abs_mean": 0.01,
    "pitch_error_abs_p95": 0.008,
    "terminated_events": 0.0,
  }


class TransferSummaryTest(unittest.TestCase):
  CENTER = (0.315, 0.0)

  def test_deviation_measured_against_center_per_action(self):
    cells = [
      _cell(0.315, 0.0, 0.55, 0.200),
      _cell(0.315, 0.0, -0.55, 0.210),
      _cell(0.327, 0.076, 0.55, 0.170),
      _cell(0.327, 0.076, -0.55, 0.230),
      _cell(0.302, -0.079, 0.55, 0.198),
    ]
    summary = transfer_deviation_summary(cells, center=self.CENTER)

    worst = summary["worst_deviation"]
    self.assertEqual(
      (worst["target_height"], worst["target_pitch"]), (0.327, 0.076)
    )
    self.assertAlmostEqual(worst["deviation"], 0.03 / 0.20, places=9)
    working = summary["worst_working_point_deviation"]
    self.assertEqual(working["deviation"], worst["deviation"])
    self.assertEqual(summary["deviation_limit"], DEVIATION_LIMIT)
    # 15% at the working point stays under the 20% Q3-style trigger.
    self.assertLess(working["deviation"], DEVIATION_LIMIT)

  def test_missing_center_reference_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "center posture"):
      transfer_deviation_summary(
        [_cell(0.327, 0.076, 0.55, 0.2)], center=self.CENTER
      )
    with self.assertRaisesRegex(ValueError, "No center reference"):
      transfer_deviation_summary(
        [
          _cell(0.315, 0.0, 0.55, 0.2),
          _cell(0.327, 0.076, 0.35, 0.2),
        ],
        center=self.CENTER,
      )

  def test_default_arguments_target_stage4(self):
    args = parse_args([])
    self.assertEqual(args.task, "HopperTrex-Hybrid-v2-Stage4")
    self.assertIn(0.55, args.yaw_actions)
    self.assertIn(-0.55, args.yaw_actions)
    self.assertEqual(args.probe_yaw_scale, 1.0)
    self.assertIsNone(args.fit_output)


class TransferSmokeTest(unittest.TestCase):
  def test_single_cell_runs_on_cpu_with_default_artifacts(self):
    cfg = load_env_cfg("HopperTrex-Hybrid-v2-Stage4", play=True)
    cfg.scene.num_envs = 1
    if cfg.scene.terrain is not None:
      cfg.scene.terrain.num_envs = 1
    action_cfg = cfg.actions["hybrid_wheel_leg"]
    scales = list(action_cfg.action_scales)
    scales[1] = 1.0
    action_cfg.action_scales = tuple(scales)
    action_cfg.yaw_feedforward_breakpoints = (
      (-1.0, 0.0), (0.0, 0.0), (1.0, 0.0),
    )
    action_cfg.yaw_calibration_qualified = False
    posture = cfg.commands["posture"]

    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
      cell = run_transfer_cell(
        env,
        height=float(posture.height_range[0]),
        pitch=float(posture.pitch_range[0]),
        value=0.55,
        settle_steps=5,
        measure_steps=25,
      )
    finally:
      env.close()

    self.assertTrue(math.isfinite(cell["transfer"]))
    # The driven differential must actually reach the wheels (half-sum
    # isolates it from balance/station common mode).
    self.assertGreater(cell["mean_mapped_yaw"], 0.05)
    self.assertEqual(cell["yaw_action"], 0.55)


if __name__ == "__main__":
  unittest.main()
