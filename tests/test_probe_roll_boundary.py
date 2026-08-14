import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hoppertrex_mjlab.scripts import probe_roll_boundary as roll


def _cells(flags, unsafe_height=None):
  result = []
  for card in roll.POSTURE_CARDS:
    for height in roll.formal_heights(10_000):
      passed = bool(flags[card["name"]][height])
      unsafe = unsafe_height is not None and math.isclose(height, unsafe_height)
      result.append({
        "posture_card": card["name"],
        "stair_height_m": height,
        "trials": 48,
        "successes": 48 if passed else 0,
        "success_rate": 1.0 if passed else 0.0,
        "terminated_trials": 1 if unsafe else 0,
        "termination_rate": 1 / 48 if unsafe else 0.0,
        "non_wheel_contact_trials": 0,
        "non_wheel_contact_rate": 0.0,
        "bilateral_airborne_trials": 0,
        "bilateral_airborne_rate": 0.0,
        "passed": passed and not unsafe,
      })
  return result


def _flags(first_failure):
  heights = roll.formal_heights(10_000)
  return {
    card["name"]: {height: first_failure is None or height < first_failure for height in heights}
    for card in roll.POSTURE_CARDS
  }


class TerrainContractTest(unittest.TestCase):
  def test_five_subcentimetre_heights_are_unique_ordered_and_exact(self):
    heights = (0.0, 0.0025, 0.005, 0.0075, 0.01)
    sub = roll.roll_boundary_sub_terrains(heights)
    self.assertEqual(tuple(sub), (
      "stair_000000um", "stair_002500um", "stair_005000um",
      "stair_007500um", "stair_010000um",
    ))
    self.assertEqual(type(sub["stair_000000um"]).__name__, "BoxFlatTerrainCfg")
    self.assertEqual(tuple(sub["stair_000000um"].size), roll.TERRAIN_SIZE_M)
    self.assertEqual(
      tuple(cfg.step_height_range for cfg in tuple(sub.values())[1:]),
      tuple((height, height) for height in heights[1:]),
    )

  def test_invalid_height_requests_fail_closed(self):
    for heights in (
      (0.0, 0.005, 0.0025),
      (0.0, 0.0025, 0.0025),
      (0.0, 0.0025004),
      (0.0, float("nan")),
      (0.0, -0.0025),
    ):
      with self.subTest(heights=heights), self.assertRaises(ValueError):
        roll.roll_boundary_sub_terrains(heights)

  def test_environment_binds_final_stack_and_zeroes_all_residual_heads(self):
    cfg = roll.make_roll_boundary_env_cfg((0.0, 0.0025, 0.005), 1)
    action = cfg.actions["hybrid_wheel_leg"]
    self.assertEqual(action.action_mask, roll.ZERO_ACTION_MASK)
    self.assertEqual(action.controller_gain_hash, roll.EXPECTED_SCHEDULE_HASH)
    self.assertTrue(action.yaw_calibration_qualified)
    self.assertTrue(action.posture_map_qualified)
    self.assertTrue(action.station_calibration_qualified)
    self.assertIsNone(action.dynamic_stair_maneuver)
    self.assertIsNone(action.stair_trigger_sensor_name)
    self.assertFalse(cfg.auto_reset)
    collision = cfg.scene.entities["robot"].collisions[0]
    self.assertEqual(
      collision.solref["wheel_.*_collision"], roll.ROLL_BOUNDARY_WHEEL_SOLREF
    )
    self.assertEqual(
      collision.solimp["wheel_.*_collision"], roll.ROLL_BOUNDARY_WHEEL_SOLIMP
    )
    self.assertIs(
      roll.ROLL_BOUNDARY_WHEEL_SOLREF, roll.ROLL_FIRST_WHEEL_CONTACT_SOLREF
    )
    self.assertIs(
      roll.ROLL_BOUNDARY_WHEEL_SOLIMP, roll.ROLL_FIRST_WHEEL_CONTACT_SOLIMP
    )
    sensors = {sensor.name: sensor for sensor in cfg.scene.sensors}
    for name in (roll.LEFT_SENSOR, roll.RIGHT_SENSOR):
      self.assertFalse(sensors[name].global_frame)
    generator = cfg.scene.terrain.terrain_generator
    self.assertEqual(generator.num_cols, 3)
    self.assertEqual(tuple(generator.sub_terrains), (
      "stair_000000um", "stair_002500um", "stair_005000um",
    ))

  def test_artifact_sha_drift_is_rejected(self):
    with (
      patch(
        "hoppertrex_mjlab.hybrid.roll_assist.file_sha256",
        return_value="0" * 64,
      ),
      self.assertRaises(ValueError),
    ):
      roll.frozen_artifact_paths()


class VerticalNormalLoadTest(unittest.TestCase):
  def test_projects_found_contact_normal_force_onto_world_z(self):
    found = torch.tensor([[True, True, False], [True, False, False]])
    force = torch.zeros(2, 3, 3)
    force[..., 0] = torch.tensor([[10.0, -20.0, 999.0], [8.0, 7.0, 6.0]])
    normal = torch.zeros_like(force)
    normal[..., 2] = torch.tensor([[1.0, 0.5, 1.0], [-0.25, 1.0, 1.0]])
    observed = roll.vertical_normal_load_n(
      found=found, force_contact_frame=force, normal_global=normal,
    )
    self.assertTrue(torch.equal(observed, torch.tensor([20.0, 2.0])))

  def test_diagnostic_control_trace_binds_loads_and_schedule_commands(self):
    data = {
      'control_step': torch.tensor([1.0, 2.0]),
      'progress': torch.tensor([-0.10, -0.09]),
      'root_z': torch.tensor([0.30, 0.31]),
      'root_vz': torch.tensor([0.0, 0.01]),
      'pitch': torch.tensor([0.02, 0.03]),
      'pitch_rate': torch.tensor([0.1, 0.2]),
      'left_vertical_normal_load': torch.tensor([100.0, 90.0]),
      'right_vertical_normal_load': torch.tensor([110.0, 80.0]),
      'total_vertical_normal_load': torch.tensor([210.0, 170.0]),
      'schedule_alpha': torch.tensor([0.2, 0.3]),
      'schedule_applied_alpha': torch.tensor([0.1, 0.2]),
      'schedule_applied_height_alpha': torch.tensor([0.1, 0.2]),
      'schedule_applied_pitch_alpha': torch.tensor([0.1, 0.2]),
      'applied_height': torch.tensor([0.30, 0.31]),
      'applied_pitch': torch.tensor([0.01, 0.02]),
    }
    trace = roll._diagnostic_control_trace(data, schedule_enabled=True)
    self.assertEqual([sample['control_step'] for sample in trace], [1, 2])
    self.assertEqual(trace[0]['total_vertical_normal_load_n'], 210.0)
    self.assertAlmostEqual(trace[1]['schedule_applied_alpha'], 0.2, places=6)
    static = roll._diagnostic_control_trace(data, schedule_enabled=False)
    self.assertIsNone(static[0]['schedule_nominal_alpha'])
    self.assertIsNone(static[0]['applied_height_m'])

  def test_vertical_load_contract_rejects_invalid_masks_and_nonfinite_data(self):
    force = torch.zeros(1, 2, 3)
    normal = torch.zeros_like(force)
    found = torch.ones(1, 2, dtype=torch.bool)
    with self.assertRaisesRegex(ValueError, "shape"):
      roll.vertical_normal_load_n(
        found=found[:, :1], force_contact_frame=force, normal_global=normal,
      )
    counted = roll.vertical_normal_load_n(
      found=torch.tensor([[2, 0]]),
      force_contact_frame=force,
      normal_global=normal,
    )
    self.assertTrue(torch.equal(counted, torch.zeros(1)))
    invalid_count = roll.vertical_normal_load_n(
      found=torch.tensor([[-1, 0]]),
      force_contact_frame=force,
      normal_global=normal,
    )
    self.assertTrue(torch.isnan(invalid_count).all())
    force[0, 0, 0] = math.nan
    invalid_force = roll.vertical_normal_load_n(
      found=found, force_contact_frame=force, normal_global=normal,
    )
    self.assertTrue(torch.isnan(invalid_force).all())


class ResetContractTest(unittest.TestCase):
  def test_posture_card_quaternion_encodes_requested_pitch(self):
    for card in roll.POSTURE_CARDS:
      q = roll._root_quaternion_for_pitch(float(card["pitch_rad"]), device="cpu")
      self.assertAlmostEqual(float(q.square().sum()), 1.0, places=6)
      observed = 2.0 * math.atan2(float(q[2]), float(q[0]))
      self.assertAlmostEqual(observed, float(card["pitch_rad"]), places=7)

  def test_posture_target_helper_matches_registered_affine_map(self):
    coefficients = torch.tensor([
      [1.0, 2.0, 3.0, 4.0],
      [10.0, 20.0, 30.0, 40.0],
      [-1.0, -2.0, -3.0, -4.0],
    ])
    observed = roll.posture_target_from_coefficients(
      coefficients, height=0.3, pitch=0.1,
    )
    expected = torch.tensor([3.9, 7.8, 11.7, 15.6])
    self.assertTrue(torch.allclose(observed, expected))

  def test_runtime_reset_sets_joint_and_root_state_before_forward(self):
    source = Path(roll.__file__).read_text(encoding="utf-8")
    start = source.index("def _reset_to_approach")
    joint_write = source.index("robot.write_joint_state_to_sim", start)
    root_write = source.index("robot.write_root_state_to_sim", joint_write)
    forward = source.index("env.sim.forward()", root_write)
    self.assertLess(joint_write, root_write)
    self.assertLess(root_write, forward)

  def test_strict_substep_latch_is_used_without_changing_decimation(self):
    source = Path(roll.__file__).read_text(encoding="utf-8")
    start = source.index("def run_card_repeat")
    install = source.index("install_strict_substep_support_recorder", start)
    step = source.index("env.step(actions)", install)
    latch = source.index('substep_support["bilateral_unsupported_ever"]', step)
    invalidate = source.index(
      'success.logical_and_(~substep_support["bilateral_unsupported_ever"])',
      latch,
    )
    self.assertLess(install, step)
    self.assertLess(step, latch)
    self.assertLess(latch, invalidate)
    self.assertNotIn("cfg.decimation = 1", source)

  def test_formal_probe_monitors_substeps_from_settle_through_success(self):
    source = Path(roll.__file__).read_text(encoding="utf-8")
    run_start = source.index("def run_card_repeat")
    monitor = source.index(
      "monitor_support = episode_wide_safety or drive_index is not None", run_start,
    )
    enable = source.index('substep_support["enabled"] = monitor_support', monitor)
    substep_failure = source.index(
      'substep_support["bilateral_unsupported_ever"] & was_active', enable,
    )
    unsafe = source.index(
      "done | non_wheel | airborne | substep_airborne", substep_failure,
    )
    main_start = source.index("def main")
    formal_call = source.index("episode_wide_safety=True", main_start)
    self.assertLess(monitor, enable)
    self.assertLess(enable, substep_failure)
    self.assertLess(substep_failure, unsafe)
    self.assertGreater(formal_call, main_start)
    self.assertEqual(
      roll.ROLL_FIRST_SUBSTEP_SUPPORT_SCOPE,
      "post_reset_settle_through_success",
    )


class SafetyTest(unittest.TestCase):
  def test_pure_classical_authority_rejects_a_policy_before_runtime(self):
    with self.assertRaisesRegex(ValueError, "do not accept a policy"):
      roll.run_card_repeat(
        object(), heights=(0.0,), card={}, repeat=1,
        settle_steps=1, drive_steps=1, stable_steps=1,
        policy=lambda observation: observation,
        require_pure_classical_authority=True,
      )

  def test_bilateral_airborne_requires_both_wheels_unloaded(self):
    left = torch.tensor([True, False, False, True])
    right = torch.tensor([True, True, False, False])
    self.assertEqual(roll.bilateral_airborne(left, right).tolist(), [False, False, True, False])

  def test_event_is_latched_before_reset(self):
    history = torch.tensor([False, False])
    event = torch.tensor([True, True])
    was_active = torch.tensor([True, False])
    self.assertEqual(roll.latch_before_reset(history, event, was_active).tolist(), [True, False])

  def test_model_torque_is_clipped_and_reports_saturation(self):
    torque, saturated = roll.model_wheel_torque(
      torch.tensor([[1.0, 0.001]]), torch.zeros(1, 2)
    )
    self.assertAlmostEqual(float(torque[0, 0]), roll.RMD_L_9025_35T_PEAK_TORQUE, places=6)
    self.assertEqual(saturated.tolist(), [[True, False]])

  def test_progress_metric_starts_only_when_drive_starts(self):
    source = Path(roll.__file__).read_text(encoding="utf-8")
    start = source.index("def run_card_repeat")
    drive_guard = source.index("if drive_index is not None:", start)
    progress_update = source.index("max_progress.copy_", drive_guard)
    self.assertLess(drive_guard, progress_update)

  def test_policy_observes_the_command_forced_on_the_same_step(self):
    source = Path(roll.__file__).read_text(encoding="utf-8")
    force = source.index("_force_commands(env, active=was_active")
    refresh = source.index("observation = env.get_observations()", force)
    policy = source.index("candidate = policy(observation)", refresh)
    step = source.index("env.step(actions)", policy)
    self.assertLess(force, refresh)
    self.assertLess(refresh, policy)
    self.assertLess(policy, step)

  def test_settle_substep_failure_is_final_and_inactive_done_is_reset(self):
    class FakeScene:
      def __init__(self, robot):
        self.robot = robot
        self.update = object()

      def __getitem__(self, name):
        if name != "robot":
          raise KeyError(name)
        return self.robot

    class FakeEnv:
      def __init__(self, *, fail_on_step=False):
        self.num_envs = 2
        self.device = "cpu"
        self.action_space = SimpleNamespace(shape=(2, 6))
        data = SimpleNamespace(
          root_link_ang_vel_b=torch.zeros(2, 3),
          root_link_pos_w=torch.tensor([[1.0, 0.0, 0.3], [1.0, 0.0, 0.3]]),
          joint_vel=torch.zeros(2, 2),
          root_link_lin_vel_b=torch.zeros(2, 3),
        )
        self.scene = FakeScene(SimpleNamespace(data=data))
        self.term = SimpleNamespace(
          _wheel_ids=torch.tensor([0, 1]),
          applied_residual=torch.zeros(2, 6),
          wheel_targets=torch.zeros(2, 2),
        )
        self.action_manager = SimpleNamespace(get_term=lambda _name: self.term)
        self.termination_manager = SimpleNamespace(
          get_term=lambda _name: torch.zeros(2, dtype=torch.bool)
        )
        self.pending = torch.zeros(2, dtype=torch.bool)
        self.reset_calls = []
        self.step_count = 0
        self.substep_state = None
        self.fail_on_step = fail_on_step

      def get_observations(self):
        return torch.zeros(2, 1)

      def step(self, _actions):
        if bool(torch.any(self.pending)):
          raise RuntimeError("manual reset pending")
        self.step_count += 1
        if self.fail_on_step:
          raise RuntimeError("injected step failure")
        terminated = torch.zeros(2, dtype=torch.bool)
        if self.step_count == 1:
          self.substep_state["bilateral_unsupported_ever"][0] = True
          self.substep_state["bilateral_unsupported_substeps"][0] = 4
        elif self.step_count == 2:
          # The already-failed env terminates again while the peer succeeds.
          terminated[0] = True
          self.pending[0] = True
        return (
          self.get_observations(), torch.zeros(2), terminated,
          torch.zeros(2, dtype=torch.bool), {},
        )

      def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs) if env_ids is None else env_ids.cpu()
        self.pending[ids] = False
        self.reset_calls.append(ids.tolist())

    env = FakeEnv()
    original_update = env.scene.update
    recorder = {
      "enabled": False,
      "active_mask": torch.zeros(2, dtype=torch.bool),
      "bilateral_unsupported_ever": torch.zeros(2, dtype=torch.bool),
      "bilateral_unsupported_substeps": torch.zeros(2, dtype=torch.long),
      "bilateral_positive_clearance_ever": torch.zeros(2, dtype=torch.bool),
      "max_flat_clearance_m": torch.zeros(2, 2),
      "max_actual_wheel_force_nm": torch.zeros(2),
      "schedule_nominal_alpha": torch.zeros(2),
      "schedule_applied_alpha": torch.zeros(2),
      "schedule_applied_height_alpha": torch.zeros(2),
      "schedule_applied_pitch_alpha": torch.zeros(2),
    }
    env.substep_state = recorder
    reset = {
      "x_relative_to_face_m": torch.zeros(2),
      "y_relative_to_center_m": torch.zeros(2),
      "root_height_m": torch.full((2,), 0.3),
      "root_linear_velocity_mps": torch.zeros(2, 3),
      "root_angular_velocity_radps": torch.zeros(2, 3),
      "root_quaternion_wxyz": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2),
      "leg_joint_position_rad": torch.zeros(2, 4),
      "leg_joint_velocity_radps": torch.zeros(2, 4),
    }
    with (
      patch.object(
        roll, "_reset_to_approach",
        return_value=(torch.zeros(2, dtype=torch.long), torch.zeros(2), torch.zeros(2), reset),
      ),
      patch.object(roll, "_force_commands"),
      patch.object(
        roll, "install_strict_substep_support_recorder",
        return_value=(recorder, original_update),
      ),
      patch.object(roll, "wheel_contact", return_value=torch.ones(2, dtype=torch.bool)),
      patch.object(
        roll, "non_wheel_ground_contact", return_value=torch.zeros(2, dtype=torch.bool)
      ),
      patch.object(
        roll, "_pitch_roll", return_value=(torch.zeros(2), torch.zeros(2))
      ),
      patch.object(
        roll, "model_wheel_torque",
        return_value=(torch.zeros(2, 2), torch.zeros(2, 2, dtype=torch.bool)),
      ),
    ):
      rows = roll.run_card_repeat(
        env,
        heights=(0.0,),
        card=roll.POSTURE_CARDS[0],
        repeat=1,
        settle_steps=1,
        drive_steps=2,
        stable_steps=1,
        episode_wide_safety=True,
      )

    self.assertFalse(rows[0]["success"])
    self.assertIsNone(rows[0]["time_to_success_s"])
    self.assertTrue(rows[0]["bilateral_airborne_ever"])
    self.assertEqual(rows[0]["bilateral_unsupported_physics_substeps"], 4)
    self.assertTrue(rows[1]["success"])
    self.assertIsNotNone(rows[1]["time_to_success_s"])
    self.assertEqual(env.reset_calls, [[0], [0]])
    self.assertIs(env.scene.update, original_update)
    self.assertFalse(recorder["enabled"])
    self.assertFalse(bool(torch.any(recorder["active_mask"])))

    failing_env = FakeEnv(fail_on_step=True)
    failing_original_update = failing_env.scene.update
    failing_recorder = {
      "enabled": False,
      "active_mask": torch.zeros(2, dtype=torch.bool),
      "bilateral_unsupported_ever": torch.zeros(2, dtype=torch.bool),
      "bilateral_unsupported_substeps": torch.zeros(2, dtype=torch.long),
      "bilateral_positive_clearance_ever": torch.zeros(2, dtype=torch.bool),
      "max_flat_clearance_m": torch.zeros(2, 2),
      "max_actual_wheel_force_nm": torch.zeros(2),
      "schedule_nominal_alpha": torch.zeros(2),
      "schedule_applied_alpha": torch.zeros(2),
      "schedule_applied_height_alpha": torch.zeros(2),
      "schedule_applied_pitch_alpha": torch.zeros(2),
    }
    with (
      patch.object(
        roll, "_reset_to_approach",
        return_value=(torch.zeros(2, dtype=torch.long), torch.zeros(2), torch.zeros(2), reset),
      ),
      patch.object(roll, "_force_commands"),
      patch.object(
        roll, "install_strict_substep_support_recorder",
        return_value=(failing_recorder, failing_original_update),
      ),
      self.assertRaisesRegex(RuntimeError, "injected step failure"),
    ):
      roll.run_card_repeat(
        failing_env,
        heights=(0.0,),
        card=roll.POSTURE_CARDS[0],
        repeat=1,
        settle_steps=1,
        drive_steps=1,
        stable_steps=1,
        episode_wide_safety=True,
      )
    self.assertIs(failing_env.scene.update, failing_original_update)
    self.assertFalse(failing_recorder["enabled"])
    self.assertFalse(bool(torch.any(failing_recorder["active_mask"])))



class AggregationTest(unittest.TestCase):
  def test_unexpected_trial_cell_fails_closed(self):
    row = {
      "posture_card": "unexpected", "stair_height_m": 0.0,
      "repeat": 1, "env_id": 0, "success": False,
      "termination": False, "non_wheel_contact": False,
      "bilateral_airborne_ever": False,
    }
    with self.assertRaises(ValueError):
      roll.aggregate_trials(
        [row], heights=(0.0,), expected_repeats=1,
        expected_envs_per_height=1, cards=({"name": "expected"},),
      )

class VerdictTest(unittest.TestCase):
  def test_positive_safe_failure_produces_training_eligible_bracket(self):
    verdict = roll.classify_results(_cells(_flags(0.0075)), heights=roll.formal_heights(10_000))
    self.assertEqual(verdict["classification"], "CLASSICAL_CROLL_BRACKETED")
    self.assertEqual(verdict["croll_bracket_m"], [0.005, 0.0075])
    self.assertTrue(verdict["training_eligible"])

  def test_no_positive_height_stops_ppo(self):
    verdict = roll.classify_results(_cells(_flags(0.0025)), heights=roll.formal_heights(10_000))
    self.assertEqual(verdict["classification"], "NO_POSITIVE_CLASSICAL_CROLL")
    self.assertFalse(verdict["training_eligible"])

  def test_unsafe_next_height_stops_training(self):
    verdict = roll.classify_results(
      _cells(_flags(0.0075), unsafe_height=0.0075), heights=roll.formal_heights(10_000)
    )
    self.assertEqual(verdict["classification"], "NEXT_HEIGHT_UNSAFE_STOP")
    self.assertFalse(verdict["training_eligible"])

  def test_all_pass_at_10mm_requests_extension(self):
    verdict = roll.classify_results(_cells(_flags(None)), heights=roll.formal_heights(10_000))
    self.assertEqual(verdict["classification"], "EXTEND_ROLL_BOUNDARY_SWEEP")
    self.assertEqual(verdict["croll_bracket_m"], [0.01, None])

  def test_nonmonotonic_result_fails_closed(self):
    flags = _flags(0.005)
    for card in roll.POSTURE_CARDS:
      flags[card["name"]][0.0075] = True
    verdict = roll.classify_results(_cells(flags), heights=roll.formal_heights(10_000))
    self.assertEqual(verdict["classification"], "NON_MONOTONIC_STOP")


class CliTest(unittest.TestCase):
  def test_smoke_is_not_evidence(self):
    args = roll.parse_args(["--output", "x.json", "--device", "cpu", "--smoke"])
    self.assertTrue(args.smoke)
    protocol = roll.protocol_for_mode(True)
    self.assertFalse(protocol["evidence_eligible"])
    self.assertEqual(protocol["heights_m"], (0.0025, 0.005, 0.0075))

  def test_existing_output_is_rejected_before_runtime(self):
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / "existing.json"
      output.write_text("{}", encoding="utf-8")
      with self.assertRaises(SystemExit):
        roll.parse_args(["--output", str(output), "--device", "cpu", "--smoke"])

  def test_formal_cpu_is_rejected(self):
    with self.assertRaises(SystemExit):
      roll.parse_args(["--output", "x.json", "--device", "cpu"])


if __name__ == "__main__":
  unittest.main()
