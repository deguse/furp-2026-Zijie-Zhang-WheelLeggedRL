import math
import unittest

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from hoppertrex_mjlab.scripts.probe_hybrid_posture_balance import (
  build_grid,
  parse_args,
  qualification_payload,
  run_cell,
  vx_check_postures,
)


class BuildGridTest(unittest.TestCase):
  def test_grid_covers_center_and_corners(self):
    grid = build_grid((0.32, 0.48), (-0.08, 0.08), 5, 5)

    self.assertEqual(len(grid), 25)
    self.assertIn((0.32, -0.08), grid)
    self.assertIn((0.48, 0.08), grid)
    self.assertIn((0.32, 0.08), grid)
    self.assertIn((0.48, -0.08), grid)
    center = ((0.32 + 0.48) / 2, 0.0)
    self.assertTrue(any(
      math.isclose(height, center[0]) and math.isclose(pitch, center[1])
      for height, pitch in grid
    ))

  def test_degenerate_envelope_collapses_to_one_cell(self):
    grid = build_grid((0.42, 0.42), (0.0, 0.0), 5, 5)
    self.assertEqual(grid, [(0.42, 0.0)])

  def test_vx_checks_pick_center_and_extreme_corners(self):
    grid = build_grid((0.32, 0.48), (-0.08, 0.08), 5, 5)
    postures = vx_check_postures(grid)
    self.assertIn((0.32, -0.08), postures)
    self.assertIn((0.48, 0.08), postures)
    self.assertLessEqual(len(postures), 3)


class QualificationPayloadTest(unittest.TestCase):
  def test_payload_carries_double_binding_and_summary(self):
    cell = {
      "target_height": 0.42,
      "target_pitch": 0.0,
      "vx_command": 0.0,
      "height_rmse": 0.004,
      "pitch_rmse": 0.010,
      "pitch_error_abs_p95": 0.02,
      "pitch_rate_abs_p99": 0.20,
      "mean_actual_lin_x": 0.001,
      "lin_x_abs_mean": 0.002,
      "non_wheel_contact_rate": 0.0,
      "terminated_events": 0.0,
    }
    payload = qualification_payload(
      grid_cells=[cell],
      vx_cells=[],
      controller_gain_hash="gain" * 16,
      controller_qualified=True,
      posture_map_hash="map" * 16,
      posture_map_qualified=True,
      calibration_hash="calibration",
      source_probe={"git_sha": "test", "device": "cpu"},
    )

    self.assertEqual(payload["schema_version"], 1)
    self.assertEqual(payload["kind"], "posture_balance_qualification")
    self.assertEqual(payload["controller_gain_hash"], "gain" * 16)
    self.assertEqual(payload["posture_map_hash"], "map" * 16)
    self.assertEqual(payload["summary"]["cells"], 1)
    self.assertEqual(payload["summary"]["worst_height_rmse"], 0.004)

    with self.assertRaisesRegex(ValueError, "at least one"):
      qualification_payload(
        grid_cells=[],
        vx_cells=[],
        controller_gain_hash=None,
        controller_qualified=False,
        posture_map_hash=None,
        posture_map_qualified=False,
        calibration_hash=None,
        source_probe={},
      )

  def test_default_arguments_target_stage3(self):
    args = parse_args([])
    self.assertEqual(args.task, "HopperTrex-Hybrid-v2-Stage3")
    self.assertEqual((args.height_points, args.pitch_points), (5, 5))
    self.assertIsNone(args.fit_output)


class ProbeSmokeTest(unittest.TestCase):
  def test_single_cell_runs_on_cpu_with_default_artifacts(self):
    """Mechanical correctness only: the default (unqualified) posture
    artifact collapses the envelope, so official data must come from the
    machine-room posture map."""

    cfg = load_env_cfg("HopperTrex-Hybrid-v2-Stage3", play=True)
    cfg.scene.num_envs = 1
    if cfg.scene.terrain is not None:
      cfg.scene.terrain.num_envs = 1
    grid = build_grid(
      tuple(cfg.commands["posture"].height_range),
      tuple(cfg.commands["posture"].pitch_range),
      5,
      5,
    )
    self.assertEqual(len(grid), 1)

    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
      cell = run_cell(
        env,
        height=grid[0][0],
        pitch=grid[0][1],
        vx=0.0,
        settle_steps=5,
        measure_steps=10,
      )
    finally:
      env.close()

    for key in (
      "height_rmse",
      "pitch_rmse",
      "pitch_error_abs_p95",
      "pitch_rate_abs_p99",
      "mean_actual_lin_x",
      "non_wheel_contact_rate",
      "terminated_events",
    ):
      self.assertTrue(math.isfinite(cell[key]), key)
    self.assertEqual(cell["target_height"], grid[0][0])


if __name__ == "__main__":
  unittest.main()
