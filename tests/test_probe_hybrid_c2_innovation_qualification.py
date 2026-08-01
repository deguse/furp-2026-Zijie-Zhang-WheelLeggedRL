import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from hoppertrex_mjlab.hybrid.innovation_detector import (
  OFFICIAL_QUALIFICATION_PROTOCOL,
  QUALIFICATION_PAIRS_PER_CELL,
  RESET_PERTURBATION_BOUNDS,
  qualification_cells,
)
from hoppertrex_mjlab.scripts import (
  probe_hybrid_c2_innovation_qualification as probe,
)


class ProbeHybridC2InnovationQualificationTest(unittest.TestCase):
  def test_direct_cli_preconfigures_artifacts_before_task_imports(self):
    flags = {
      "--controller-path": "controller.json",
      "--calibration-path": "velocity.json",
      "--posture-map-path": "posture.json",
      "--station-calibration-path": "station.json",
    }
    argv = [item for pair in flags.items() for item in pair]
    with mock.patch.dict(probe.os.environ, {}, clear=True):
      probe._preconfigure_artifact_environment(argv)
      for flag, value in flags.items():
        variable = probe._EARLY_ARTIFACT_FLAGS[flag]
        self.assertEqual(probe.os.environ[variable], str(probe.Path(value).resolve()))
      self.assertNotIn(
        "HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH", probe.os.environ
      )
      with self.assertRaisesRegex(RuntimeError, "all four artifact paths"):
        probe._preconfigure_artifact_environment(argv[:-2])

  def test_official_protocol_is_one_seed3_formal_capture(self):
    protocol = probe.protocol(False, "cuda:0")
    self.assertEqual(protocol, OFFICIAL_QUALIFICATION_PROTOCOL)
    self.assertEqual(protocol["seed"], 3)
    self.assertEqual(len(protocol["cells"]), 18)
    self.assertEqual(protocol["pairs_per_cell"], 16)
    self.assertEqual(protocol["settle_steps"], 200)
    self.assertEqual(protocol["drive_steps"], 500)
    self.assertEqual(protocol["voting"]["max_delay_ticks"], 3)
    self.assertEqual(protocol["qualification"]["overall_timely_min"], 274)
    self.assertEqual(protocol["qualification"]["per_cell_timely_min"], 15)
    self.assertTrue(protocol["evidence_eligible"])
    with self.assertRaises(SystemExit):
      probe.parse_args([
        "--output-dir", "out", "--predictor", "predictor.json",
        "--transition-floor", "floor.json", "--controller-path", "controller.json",
        "--calibration-path", "velocity.json", "--posture-map-path", "posture.json",
        "--station-calibration-path", "station.json", "--device", "cpu",
      ])

  def test_reset_perturbations_reproduce_frozen_cpu_generator_exactly(self):
    for cell_index in (0, 7, 17):
      generator = torch.Generator(device="cpu")
      generator.manual_seed(30_000 + cell_index)
      unit = 2.0 * torch.rand(
        (QUALIFICATION_PAIRS_PER_CELL, 4), generator=generator
      ) - 1.0
      bounds = torch.tensor(RESET_PERTURBATION_BOUNDS, dtype=torch.float32)
      expected = unit * bounds
      actual = probe.reset_perturbations(cell_index, QUALIFICATION_PAIRS_PER_CELL)
      self.assertEqual(actual.device.type, "cpu")
      self.assertEqual(actual.dtype, torch.float32)
      self.assertEqual(tuple(actual.shape), (16, 4))
      self.assertTrue(torch.equal(actual, expected))
    self.assertFalse(torch.equal(
      probe.reset_perturbations(0, 16), probe.reset_perturbations(1, 16)
    ))
    for cell_index, slots in ((-1, 16), (18, 16), (0, 15)):
      with self.assertRaises(ValueError):
        probe.reset_perturbations(cell_index, slots)

  def test_paired_reset_encodes_registered_root_state_and_pair_identity(self):
    num_envs = 32
    origins = torch.zeros((num_envs, 3), dtype=torch.float32)
    origins[:, 0] = torch.arange(num_envs, dtype=torch.float32) * 10.0
    origins[:, 1] = torch.arange(num_envs, dtype=torch.float32) * -3.0

    class Robot:
      def __init__(self):
        self.data = SimpleNamespace(
          default_root_state=torch.full((num_envs, 13), 99.0, dtype=torch.float32)
        )
        self.written = None

      def write_root_state_to_sim(self, value):
        self.written = value.clone()

    robot = Robot()

    class Scene:
      terrain = SimpleNamespace(
        terrain_types=torch.arange(num_envs, dtype=torch.long)
      )
      env_origins = origins

      def __getitem__(self, name):
        self.test.assertEqual(name, "robot")
        return robot

    Scene.test = self

    class Sim:
      def __init__(self):
        self.forward_count = 0
        self.sense_count = 0

      def forward(self):
        self.forward_count += 1

      def sense(self):
        self.sense_count += 1

    env = SimpleNamespace(
      scene=Scene(), device="cpu", num_envs=num_envs, sim=Sim(), reset=mock.Mock()
    )
    pairs = [
      {"slot": slot, "flat_env_id": slot, "stair_env_id": 16 + slot}
      for slot in range(16)
    ]
    cell = qualification_cells()[2]
    geometry = {"outer_face_x": 3.0, "cross_x": 3.1}
    with (
      mock.patch.object(probe.capture, "paired_environment_ids", return_value=pairs),
      mock.patch.object(probe.stair, "approach_geometry", return_value=geometry),
    ):
      reset = probe._paired_reset(env, cell=cell)

    env.reset.assert_called_once_with()
    self.assertEqual(env.sim.forward_count, 1)
    self.assertEqual(env.sim.sense_count, 1)
    self.assertIsNotNone(robot.written)
    expected_perturbations = probe.reset_perturbations(cell["cell_index"], 16)
    self.assertTrue(torch.equal(reset["perturbations"], expected_perturbations))

    root = reset["root_states"]
    relative = reset["relative_root_states"]
    half_pitch = 0.5 * float(cell["pitch_rad"])
    expected_quaternion = torch.tensor(
      [np.cos(half_pitch), 0.0, np.sin(half_pitch), 0.0],
      dtype=root.dtype,
    )
    for pair in pairs:
      slot = pair["slot"]
      flat_id = pair["flat_env_id"]
      stair_id = pair["stair_env_id"]
      perturbation = expected_perturbations[slot]
      self.assertTrue(torch.equal(relative[flat_id], relative[stair_id]))
      self.assertAlmostEqual(
        float(relative[flat_id, 0]), -0.25 + float(perturbation[0]), places=6
      )
      self.assertAlmostEqual(
        float(relative[flat_id, 1]), float(perturbation[1]), places=6
      )
      self.assertAlmostEqual(
        float(root[flat_id, 2]), float(cell["height_m"]), places=7
      )
      torch.testing.assert_close(root[flat_id, 3:7], expected_quaternion)
      self.assertEqual(float(root[flat_id, 7]), float(perturbation[2]))
      self.assertEqual(float(root[flat_id, 11]), float(perturbation[3]))
      self.assertTrue(torch.equal(
        root[flat_id, [8, 9, 10, 12]], torch.zeros(4, dtype=root.dtype)
      ))
    self.assertEqual(reset["paired_reset_max_abs_error"], 0.0)
    self.assertLessEqual(
      reset["written_reset_max_abs_error"], probe.QUALIFICATION_RESET_WRITE_ATOL
    )
    self.assertLessEqual(
      reset["written_paired_reset_max_abs_error"],
      probe.QUALIFICATION_RESET_WRITE_ATOL,
    )
    self.assertLessEqual(reset["root_pitch_max_abs_error_rad"], 1.0e-7)
    self.assertLessEqual(reset["root_roll_yaw_max_abs_rad"], 1.0e-7)
    self.assertEqual(reset["other_root_velocity_max_abs"], 0.0)

  @staticmethod
  def _synthetic_contact_raw(slots=2):
    shape = (500, 16, slots)
    outer_face_x = np.linspace(1.0, 2.5, 16, dtype=np.float32)
    return {
      "stair_contact_found": np.zeros(shape, dtype=np.float32),
      "stair_contact_force_contact_frame": np.zeros(
        (*shape, 3), dtype=np.float32
      ),
      "stair_contact_pos_global": np.zeros((*shape, 3), dtype=np.float32),
      "stair_contact_normal_global": np.zeros((*shape, 3), dtype=np.float32),
      "stair_outer_face_x": outer_face_x,
      "stair_terrain_origin_x": outer_face_x + np.float32(3.0),
    }

  def test_stair_contact_raw_arrays_preserve_shape_dtype_and_trial_geometry(self):
    steps, envs, slots = 500, 32, 3
    found = torch.zeros((steps, envs, slots), dtype=torch.float32)
    for env_id in range(envs):
      found[:, env_id, :] = float(env_id)
    sensor_stacked = {
      "found": found,
      "force": torch.zeros((steps, envs, slots, 3), dtype=torch.float32),
      "pos": torch.zeros((steps, envs, slots, 3), dtype=torch.float32),
      "normal": torch.zeros((steps, envs, slots, 3), dtype=torch.float32),
    }
    stair_ids = torch.arange(1, envs, 2, dtype=torch.long)
    outer_face_x = torch.arange(envs, dtype=torch.float32) + 10.0
    terrain_origin_x = outer_face_x + 3.0

    raw = probe._stair_contact_raw_arrays(
      sensor_stacked, stair_ids, outer_face_x, terrain_origin_x
    )

    self.assertEqual(set(raw), probe._STAIR_CONTACT_RAW_KEYS)
    expected_shapes = {
      "stair_contact_found": (500, 16, 3),
      "stair_contact_force_contact_frame": (500, 16, 3, 3),
      "stair_contact_pos_global": (500, 16, 3, 3),
      "stair_contact_normal_global": (500, 16, 3, 3),
      "stair_outer_face_x": (16,),
      "stair_terrain_origin_x": (16,),
    }
    for name, expected_shape in expected_shapes.items():
      self.assertEqual(raw[name].shape, expected_shape, msg=name)
      self.assertEqual(raw[name].dtype, np.float32, msg=name)
    np.testing.assert_array_equal(
      raw["stair_contact_found"][0, :, 0],
      stair_ids.numpy().astype(np.float32),
    )
    np.testing.assert_array_equal(
      raw["stair_outer_face_x"], outer_face_x[stair_ids].numpy()
    )
    np.testing.assert_array_equal(
      raw["stair_terrain_origin_x"], terrain_origin_x[stair_ids].numpy()
    )

  def test_recompute_first_riser_truth_uses_archived_raw_and_detects_tamper(self):
    raw = self._synthetic_contact_raw()
    impact_tick = 125
    raw["stair_contact_found"][impact_tick, :, 0] = 1.0
    raw["stair_contact_force_contact_frame"][impact_tick, :, 0, 0] = 1.0
    raw["stair_contact_pos_global"][impact_tick, :, 0, 0] = (
      raw["stair_outer_face_x"]
    )
    raw["stair_contact_normal_global"][impact_tick, :, 0, 0] = 0.25

    impacts, mask = probe._recompute_first_riser_truth(raw)

    self.assertEqual(impacts.shape, (16,))
    self.assertEqual(impacts.dtype, np.int64)
    self.assertEqual(mask.shape, (500, 16))
    self.assertEqual(mask.dtype, np.bool_)
    np.testing.assert_array_equal(impacts, np.full(16, impact_tick, np.int64))
    self.assertTrue(np.all(mask[impact_tick]))
    self.assertEqual(int(mask.sum()), 16)

    tampered = {name: value.copy() for name, value in raw.items()}
    tampered["stair_contact_force_contact_frame"][impact_tick, 7, 0, 0] = 0.99
    tampered_impacts, tampered_mask = probe._recompute_first_riser_truth(tampered)
    self.assertEqual(tampered_impacts[7], -1)
    self.assertFalse(tampered_mask[impact_tick, 7])
    np.testing.assert_array_equal(
      np.delete(tampered_impacts, 7), np.full(15, impact_tick, np.int64)
    )

  def test_stair_contact_raw_validation_rejects_shape_and_dtype_drift(self):
    base = self._synthetic_contact_raw()
    mutations = {
      "short_tick_axis": lambda raw: raw.__setitem__(
        "stair_contact_found", raw["stair_contact_found"][:-1]
      ),
      "short_env_axis": lambda raw: raw.__setitem__(
        "stair_contact_found", raw["stair_contact_found"][:, :-1]
      ),
      "wrong_vector_axis": lambda raw: raw.__setitem__(
        "stair_contact_pos_global", raw["stair_contact_pos_global"][..., :2]
      ),
      "wrong_outer_face": lambda raw: raw.__setitem__(
        "stair_outer_face_x", raw["stair_outer_face_x"][:-1]
      ),
      "wrong_dtype": lambda raw: raw.__setitem__(
        "stair_contact_found", raw["stair_contact_found"].astype(np.float64)
      ),
      "fractional_found_count": lambda raw: raw["stair_contact_found"].__setitem__(
        (0, 0, 0), np.float32(0.5)
      ),
    }
    for name, mutate in mutations.items():
      with self.subTest(name=name):
        raw = {key: value.copy() for key, value in base.items()}
        mutate(raw)
        with self.assertRaises(ValueError):
          probe._recompute_first_riser_truth(raw)

    def sensor_history():
      return {
        "found": torch.zeros((500, 32, 2), dtype=torch.float32),
        "force": torch.zeros((500, 32, 2, 3), dtype=torch.float32),
        "pos": torch.zeros((500, 32, 2, 3), dtype=torch.float32),
        "normal": torch.zeros((500, 32, 2, 3), dtype=torch.float32),
      }

    extraction_cases = {}
    history = sensor_history()
    history["found"] = history["found"][:-1]
    extraction_cases["short_tick_axis"] = (
      history,
      torch.arange(16, 32, dtype=torch.long),
      torch.zeros(32, dtype=torch.float32),
      torch.zeros(32, dtype=torch.float32),
    )
    history = sensor_history()
    history["found"] = history["found"][:, :31]
    extraction_cases["short_env_axis"] = (
      history,
      torch.arange(16, 32, dtype=torch.long),
      torch.zeros(32, dtype=torch.float32),
      torch.zeros(32, dtype=torch.float32),
    )
    history = sensor_history()
    history["pos"] = history["pos"][..., :2]
    extraction_cases["wrong_vector_axis"] = (
      history,
      torch.arange(16, 32, dtype=torch.long),
      torch.zeros(32, dtype=torch.float32),
      torch.zeros(32, dtype=torch.float32),
    )
    history = sensor_history()
    extraction_cases["wrong_outer_face"] = (
      history,
      torch.arange(16, 32, dtype=torch.long),
      torch.zeros(31, dtype=torch.float32),
      torch.zeros(32, dtype=torch.float32),
    )
    history = sensor_history()
    history["found"] = history["found"].to(torch.float64)
    extraction_cases["wrong_dtype"] = (
      history,
      torch.arange(16, 32, dtype=torch.long),
      torch.zeros(32, dtype=torch.float32),
      torch.zeros(32, dtype=torch.float32),
    )
    history = sensor_history()
    extraction_cases["wrong_terrain_origin"] = (
      history,
      torch.arange(16, 32, dtype=torch.long),
      torch.zeros(32, dtype=torch.float32),
      torch.zeros(31, dtype=torch.float32),
    )
    for name, (
      history, stair_ids, outer_face_x, terrain_origin_x
    ) in extraction_cases.items():
      with self.subTest(extraction=name):
        with self.assertRaises(ValueError):
          probe._stair_contact_raw_arrays(
            history, stair_ids, outer_face_x, terrain_origin_x
          )

  def test_contact_raw_health_counts_nonfinite_and_geometry_drift(self):
    raw = self._synthetic_contact_raw()
    raw["stair_contact_found"][0, 0, 0] = np.float32(2.0)
    self.assertEqual(
      probe._contact_raw_health(raw),
      {
        "nonfinite_sample_count": 0,
        "outer_face_binding_violation_count": 0,
      },
    )
    raw["stair_contact_force_contact_frame"][0, 0, 0, 0] = np.nan
    raw["stair_outer_face_x"][1] += np.float32(0.1)
    self.assertEqual(
      probe._contact_raw_health(raw),
      {
        "nonfinite_sample_count": 1,
        "outer_face_binding_violation_count": 1,
      },
    )

  def test_run_cell_archives_the_raw_arrays_used_for_riser_truth(self):
    source = inspect.getsource(probe.run_cell)
    self.assertIn(
      "first_impact, riser_mask = _recompute_first_riser_truth(contact_raw)",
      source,
    )
    self.assertIn("**contact_raw", source)

  def test_cell_invalid_rejects_any_capture_health_or_reset_drift(self):
    summary = {
      "raw_shape": [500, 16],
      "paired_reset_max_abs_error": 0.0,
      "written_reset_max_abs_error": 0.0,
      "written_paired_reset_max_abs_error": 0.0,
      "root_pitch_max_abs_error_rad": 0.0,
      "root_roll_yaw_max_abs_rad": 0.0,
      "other_root_velocity_max_abs": 0.0,
      "impact_steps": [100] * 16,
      "diagnostic_windows": [{}] * 16,
      "health": {
        "flat_termination_count": 0,
        "stair_termination_count": 0,
        "flat_timeout_count": 0,
        "stair_timeout_count": 0,
        "flat_non_wheel_contact_count": 0,
        "stair_non_wheel_contact_count": 0,
        "settle_riser_contact_count": 0,
        "drive_start_past_face_count": 0,
        "missing_impact_count": 0,
        "invalid_window_count": 0,
        "predictor_domain_violation_count": 0,
        "posture_violation_count": 0,
        "predictor_evaluation_error_count": 0,
        "nonfinite_sample_count": 0,
        "negative_feature_sample_count": 0,
        "portable_target_violation_count": 0,
        "outer_face_binding_violation_count": 0,
      },
    }
    self.assertFalse(probe._cell_invalid(summary))
    for key in summary["health"]:
      changed = {**summary, "health": dict(summary["health"])}
      changed["health"][key] = 1
      self.assertTrue(probe._cell_invalid(changed), msg=key)


if __name__ == "__main__":
  unittest.main()
