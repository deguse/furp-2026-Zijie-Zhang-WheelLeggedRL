import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from torch import nn

import hoppertrex_mjlab.tasks  # noqa: F401
from hoppertrex_mjlab.hybrid.roll_assist import (
  ROLL_ASSIST_ACTION_MASK,
  ROLL_ASSIST_ACTION_SCALES,
  ROLL_ASSIST_ACTOR_TERMS,
  ROLL_ASSIST_CONTROLLER_SCHEDULE_HASH,
  ROLL_ASSIST_CRITIC_TAIL,
  ROLL_ASSIST_SETTLE_STEPS,
  ROLL_ASSIST_STAIR_POSTURE_HEIGHT_M,
  ROLL_ASSIST_STAIR_POSTURE_PITCH_RAD,
  ROLL_ASSIST_TASK_ID,
  ROLL_FIRST_MJLAB_GIT_SHA,
  ROLL_FIRST_WHEEL_CONTACT_SOLIMP,
  ROLL_FIRST_WHEEL_CONTACT_SOLREF,
  RollAssistCurriculumState,
  build_extension_authorization,
  build_reward_calibration,
  continuation_gate,
  file_sha256,
  final_expansion_gate,
  load_roll_boundary_verdict,
  newest_passer,
  reward_weights,
  validate_extension_authorization,
  validate_reward_calibration,
  validate_roll_assist_training_record,
)
from hoppertrex_mjlab.hybrid.runner import (
  HybridOnPolicyRunner,
  is_roll_assist_env,
  zero_initialize_actor_output,
)
from hoppertrex_mjlab.scripts import probe_roll_boundary as roll_boundary
from hoppertrex_mjlab.scripts.calibrate_roll_assist_reward import (
  positive_reward_rate_from_stall,
)
from hoppertrex_mjlab.scripts.rsl_rl.train import validate_roll_assist_training_request
from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
  ROLL_ASSIST_LEFT_SENSOR_NAME,
  ROLL_ASSIST_RIGHT_SENSOR_NAME,
  RollAssistCurriculum,
  make_stair_roll_assist_env_cfg,
  validate_roll_assist_observation_contract,
)


class ContractTest(unittest.TestCase):
  def _bind_reward_file(self, env_cfg, root: Path) -> None:
    artifact = build_reward_calibration(
      baseline_positive_reward_rate=3.5,
      source_stall_sha256="1" * 64,
      roll_boundary_sha256="a" * 64,
    )
    path = root / "reward.json"
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    env_cfg.roll_assist_reward_calibration_path = str(path)
    env_cfg.roll_assist_reward_calibration_sha256 = file_sha256(path)
    env_cfg.roll_assist_reward_calibration_content_sha256 = artifact[
      "calibration_sha256"
    ]

  def test_registered_task_has_frozen_shapes_masks_and_budget(self):
    cfg = load_env_cfg(ROLL_ASSIST_TASK_ID)
    rl = load_rl_cfg(ROLL_ASSIST_TASK_ID)
    self.assertEqual(tuple(cfg.observations["actor"].terms), ROLL_ASSIST_ACTOR_TERMS)
    self.assertEqual(tuple(cfg.observations["critic"].terms), ROLL_ASSIST_ACTOR_TERMS + ROLL_ASSIST_CRITIC_TAIL)
    self.assertEqual(cfg.actions["hybrid_wheel_leg"].action_mask, ROLL_ASSIST_ACTION_MASK)
    self.assertEqual(cfg.actions["hybrid_wheel_leg"].action_scales, ROLL_ASSIST_ACTION_SCALES)
    self.assertEqual(tuple(rl.actor.distribution_cfg["active_mask"]), ROLL_ASSIST_ACTION_MASK)
    self.assertEqual((rl.seed, rl.num_steps_per_env, rl.max_iterations, rl.save_interval), (1, 24, 100, 25))
    self.assertIs(load_runner_cls(ROLL_ASSIST_TASK_ID), HybridOnPolicyRunner)
    self.assertEqual(cfg.scene.num_envs, 256)
    self.assertEqual(cfg.seed, 1)
    self.assertEqual(cfg.scene.terrain.terrain_generator.seed, 1)
    self.assertEqual(cfg.roll_assist_flat_env_count, 64)

  def test_roll_assist_always_binds_same_frozen_classical_stack_as_r0(self):
    cfg = make_stair_roll_assist_env_cfg(play=True)
    action = cfg.actions["hybrid_wheel_leg"]
    self.assertEqual(
      action.controller_gain_hash, ROLL_ASSIST_CONTROLLER_SCHEDULE_HASH
    )
    self.assertTrue(action.controller_qualified)
    self.assertTrue(action.yaw_calibration_qualified)
    self.assertTrue(action.posture_map_qualified)
    self.assertTrue(action.station_calibration_qualified)

  def test_actor_has_no_privileged_fields_and_dynamic_paths_are_off(self):
    cfg = make_stair_roll_assist_env_cfg(play=True)
    action = cfg.actions["hybrid_wheel_leg"]
    self.assertFalse(set(ROLL_ASSIST_CRITIC_TAIL).intersection(cfg.observations["actor"].terms))
    self.assertIsNone(action.dynamic_stair_maneuver)
    self.assertIsNone(action.stair_trigger_sensor_name)
    self.assertFalse(action.stair_mode_freezes_leg_reference)
    self.assertFalse(action.stair_mode_forced)
    reset = cfg.events["reset_root_to_roll_assist"]
    self.assertAlmostEqual(reset.params["x_offset_from_origin_m"], -3.25)
    self.assertAlmostEqual(
      reset.params["root_height"], ROLL_ASSIST_STAIR_POSTURE_HEIGHT_M
    )
    self.assertAlmostEqual(
      reset.params["stair_posture_pitch"], ROLL_ASSIST_STAIR_POSTURE_PITCH_RAD
    )
    self.assertIsNone(reset.params["pose_range"])
    self.assertIsNone(reset.params["velocity_range"])
    self.assertIn("non_wheel_ground_contact", cfg.terminations)
    self.assertIn("bilateral_airborne", cfg.terminations)
    self.assertTrue(cfg.metrics["roll_assist_substep_support"].per_substep)
    collision = cfg.scene.entities["robot"].collisions[0]
    self.assertEqual(
      collision.solref["wheel_.*_collision"], ROLL_FIRST_WHEEL_CONTACT_SOLREF
    )
    self.assertEqual(
      collision.solimp["wheel_.*_collision"], ROLL_FIRST_WHEEL_CONTACT_SOLIMP
    )
    validate_roll_assist_observation_contract(cfg)
    names = {sensor.name for sensor in cfg.scene.sensors}
    self.assertTrue({ROLL_ASSIST_LEFT_SENSOR_NAME, ROLL_ASSIST_RIGHT_SENSOR_NAME}.issubset(names))

  def test_stair_command_has_exact_two_second_settle_contract(self):
    self.assertEqual(ROLL_ASSIST_SETTLE_STEPS, 100)
    cfg = make_stair_roll_assist_env_cfg(play=True)
    self.assertAlmostEqual(
      cfg.sim.mujoco.timestep * cfg.decimation * ROLL_ASSIST_SETTLE_STEPS,
      2.0,
    )

  def test_zero_output_head_matches_classical_deterministic_action(self):
    actor = nn.Sequential(nn.Linear(34, 128), nn.ELU(), nn.Linear(128, 6))
    head = zero_initialize_actor_output(actor, label="RollAssist")
    torch.testing.assert_close(head.weight, torch.zeros_like(head.weight))
    torch.testing.assert_close(actor(torch.randn(8, 34)), torch.zeros(8, 6))

  def test_training_request_defers_resume_total_to_byte_bound_authorization(self):
    with tempfile.TemporaryDirectory() as directory:
      env_cfg = load_env_cfg(ROLL_ASSIST_TASK_ID)
      env_cfg.roll_assist_qualified = True
      env_cfg.roll_assist_r0_sha256 = "a" * 64
      env_cfg.roll_assist_r0_git_sha = "a" * 40
      env_cfg.roll_assist_r0_schedule_hash = ROLL_ASSIST_CONTROLLER_SCHEDULE_HASH
      self._bind_reward_file(env_cfg, Path(directory))
      agent = load_rl_cfg(ROLL_ASSIST_TASK_ID)
      from unittest.mock import patch
      with patch(
        "hoppertrex_mjlab.scripts.rsl_rl.train._repository_head",
        return_value="a" * 40,
      ):
        validate_roll_assist_training_request(env_cfg, agent, resume=False)
      agent.resume = True
      for target in (151, 176, 200, 251, 276, 300, 400, 500):
        agent.max_iterations = target
        with patch(
          "hoppertrex_mjlab.scripts.rsl_rl.train._repository_head",
          return_value="a" * 40,
        ):
          validate_roll_assist_training_request(env_cfg, agent, resume=True)
      for target in (101, 150, 501, 600):
        agent.max_iterations = target
        with patch(
          "hoppertrex_mjlab.scripts.rsl_rl.train._repository_head",
          return_value="a" * 40,
        ), self.assertRaises(ValueError):
          validate_roll_assist_training_request(env_cfg, agent, resume=True)

  def test_marker_detection_is_task_specific(self):
    env = SimpleNamespace(cfg=SimpleNamespace(
      roll_assist_task_id=ROLL_ASSIST_TASK_ID,
      roll_assist_zero_initialize_actor_output=True,
    ))
    self.assertTrue(is_roll_assist_env(env))
    self.assertFalse(is_roll_assist_env(SimpleNamespace(cfg=SimpleNamespace())))


class ArtifactTest(unittest.TestCase):
  def _r0(self, max_height_um=10_000):
    heights = roll_boundary.formal_heights(max_height_um)
    trials = []
    for card in roll_boundary.POSTURE_CARDS:
      for repeat in range(1, roll_boundary.OFFICIAL_REPEATS + 1):
        for height_index, height in enumerate(heights):
          passed = height <= 0.005
          for slot in range(roll_boundary.OFFICIAL_ENVS_PER_HEIGHT):
            trials.append({
              "posture_card": card["name"],
              "target_height_m": card["height_m"],
              "target_pitch_rad": card["pitch_rad"],
              "stair_height_m": height,
              "terrain_key": roll_boundary.terrain_key(height),
              "terrain_index": height_index,
              "repeat": repeat,
              "env_id": height_index * roll_boundary.OFFICIAL_ENVS_PER_HEIGHT + slot,
              "success": passed,
              "time_to_success_s": 1.0 if passed else None,
              "termination": False,
              "non_wheel_contact": False,
              "bilateral_airborne_ever": False,
              "bilateral_unsupported_physics_substeps": 0,
              "wheel_residual_abs_max": 0.0,
            })
    cells, repeat_cells = roll_boundary.aggregate_trials(
      trials,
      heights=heights,
      expected_repeats=roll_boundary.OFFICIAL_REPEATS,
      expected_envs_per_height=roll_boundary.OFFICIAL_ENVS_PER_HEIGHT,
    )
    verdict = roll_boundary.classify_results(cells, heights=heights)
    return {
      "schema_version": 1, "probe": "hoppertrex_roll_boundary_r0",
      "task": "HopperTrex-Hybrid-v2-Stage5", "promotion_eligible": False,
      "evidence_eligible": True, "training_eligible": True,
      "classification": verdict["classification"],
      "max_common_passing_height_m": verdict["max_common_passing_height_m"],
      "first_non_common_height_m": verdict["first_non_common_height_m"],
      "croll_bracket_m": verdict["croll_bracket_m"],
      "verdict": verdict,
      "action_mask": [False] * 6, "checkpoint": None,
      "seed": 1, "device": "cuda:0",
      "git_sha": "a" * 40,
      "mjlab_git_sha": ROLL_FIRST_MJLAB_GIT_SHA,
      "runtime": {
        "device": "cuda:0", "cuda_available": True, "gpu_name": "test gpu",
      },
      "controller_schedule_hash": ROLL_ASSIST_CONTROLLER_SCHEDULE_HASH,
      "protocol": {
        "heights_m": list(heights),
        "terrain_keys": [roll_boundary.terrain_key(height) for height in heights],
        "height_step_m": 0.0025,
        "physics_timestep_s": 0.005,
        "control_frequency_hz": 50.0,
        "control_decimation": 4,
        "formal_cap_m": 0.030,
        "envs_per_height": roll_boundary.OFFICIAL_ENVS_PER_HEIGHT,
        "repeats": roll_boundary.OFFICIAL_REPEATS,
        "cell_pass_successes": roll_boundary.CELL_PASS_SUCCESSES,
        "cell_trials": (
          roll_boundary.OFFICIAL_ENVS_PER_HEIGHT * roll_boundary.OFFICIAL_REPEATS
        ),
        "settle_steps": roll_boundary.OFFICIAL_SETTLE_STEPS,
        "drive_steps": roll_boundary.OFFICIAL_DRIVE_STEPS,
        "stable_steps": roll_boundary.OFFICIAL_STABLE_STEPS,
        "posture_cards": [dict(card) for card in roll_boundary.POSTURE_CARDS],
        "terrain": "flat_box_at_zero_else_pyramid_stairs",
        "strict_physics_substep_support_required": True,
        "strict_physics_substep_support_scope": "post_reset_settle_through_success",
        "safety": {
          "termination_trials_required": 0,
          "non_wheel_contact_trials_required": 0,
          "bilateral_airborne_trials_required": 0,
          "terminal_state_latched_before_reset": True,
        },
        "wheel_contact_solref": list(ROLL_FIRST_WHEEL_CONTACT_SOLREF),
        "wheel_contact_solimp": list(ROLL_FIRST_WHEEL_CONTACT_SOLIMP),
        "root_reset": {
          "joint_state": "registered_posture_map_absolute_targets",
          "orientation": "posture_card_pitch_quaternion",
        },
      },
      "cells": cells,
      "repeat_cells": repeat_cells,
      "trials": trials,
    }

  def test_training_request_rejects_r0_git_or_schedule_drift(self):
    env_cfg = load_env_cfg(ROLL_ASSIST_TASK_ID)
    env_cfg.roll_assist_qualified = True
    env_cfg.roll_assist_r0_sha256 = "a" * 64
    env_cfg.roll_assist_r0_git_sha = "b" * 40
    env_cfg.roll_assist_r0_schedule_hash = ROLL_ASSIST_CONTROLLER_SCHEDULE_HASH
    agent = load_rl_cfg(ROLL_ASSIST_TASK_ID)
    from unittest.mock import patch
    with patch(
      "hoppertrex_mjlab.scripts.rsl_rl.train._repository_head",
      return_value="a" * 40,
    ), self.assertRaisesRegex(ValueError, "Git SHA"):
      validate_roll_assist_training_request(env_cfg, agent, resume=False)
    env_cfg.roll_assist_r0_git_sha = "a" * 40
    env_cfg.roll_assist_r0_schedule_hash = "b" * 64
    with patch(
      "hoppertrex_mjlab.scripts.rsl_rl.train._repository_head",
      return_value="a" * 40,
    ), self.assertRaisesRegex(ValueError, "frozen C1 schedule"):
      validate_roll_assist_training_request(env_cfg, agent, resume=False)

  def test_r0_verdict_accepts_only_positive_safe_adjacent_bracket(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "r0.json"
      path.write_text(json.dumps(self._r0()), encoding="utf-8")
      result = load_roll_boundary_verdict(path, expected_git_sha="a" * 40)
      self.assertEqual((result["hpass_m"], result["hnext_m"]), (0.005, 0.0075))
      with self.assertRaisesRegex(ValueError, "current checkout"):
        load_roll_boundary_verdict(path, expected_git_sha="b" * 40)
      bad = self._r0()
      bad["verdict"]["next_height_unsafe"] = True
      path.write_text(json.dumps(bad), encoding="utf-8")
      with self.assertRaises(ValueError):
        load_roll_boundary_verdict(path)

      bad = self._r0()
      bad["device"] = "cpu"
      path.write_text(json.dumps(bad), encoding="utf-8")
      with self.assertRaises(ValueError):
        load_roll_boundary_verdict(path)

      bad = self._r0()
      bad["controller_schedule_hash"] = "b" * 64
      path.write_text(json.dumps(bad), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "frozen C1 schedule"):
        load_roll_boundary_verdict(path)

      for mutate in (
        lambda payload: payload.update(mjlab_git_sha="b" * 40),
        lambda payload: payload["runtime"].update(cuda_available=False),
        lambda payload: payload["runtime"].update(gpu_name=""),
      ):
        bad = self._r0()
        mutate(bad)
        path.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "MjLab SHA|CUDA runtime"):
          load_roll_boundary_verdict(path)

  def test_r0_verdict_accepts_all_registered_formal_sweep_maxima(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "r0.json"
      for max_height_um in (10_000, 20_000, 30_000):
        payload = self._r0(max_height_um)
        path.write_text(json.dumps(payload), encoding="utf-8")
        result = load_roll_boundary_verdict(path, expected_git_sha="a" * 40)
        with self.subTest(max_height_um=max_height_um):
          self.assertEqual((result["hpass_m"], result["hnext_m"]), (0.005, 0.0075))

  def test_r0_verdict_rejects_unregistered_formal_sweep_maximum(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "r0.json"
      payload = self._r0()
      payload["protocol"]["heights_m"] = payload["protocol"]["heights_m"][:-1]
      path.write_text(json.dumps(payload), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "unregistered formal sweep maximum"):
        load_roll_boundary_verdict(path)

  def test_r0_verdict_rejects_old_terrain_contact_and_reset_contracts(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "r0.json"
      for mutate in (
        lambda payload: payload["protocol"].update(
          terrain="pyramid_stairs"
        ),
        lambda payload: payload["protocol"].update(
          strict_physics_substep_support_required=False
        ),
        lambda payload: payload["protocol"].update(
          strict_physics_substep_support_scope="drive_only"
        ),
        lambda payload: payload["protocol"].update(physics_timestep_s=0.01),
        lambda payload: payload["protocol"].update(control_frequency_hz=100.0),
        lambda payload: payload["protocol"].update(control_decimation=2),
        lambda payload: payload["protocol"].update(
          wheel_contact_solref=[0.005, 1.0]
        ),
        lambda payload: payload["protocol"]["root_reset"].update(
          joint_state="default_joint_state"
        ),
      ):
        payload = self._r0()
        mutate(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ValueError):
          load_roll_boundary_verdict(path)

  def test_r0_verdict_recomputes_raw_trials_cells_and_success_times(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "r0.json"

      missing = self._r0()
      del missing["trials"]
      path.write_text(json.dumps(missing), encoding="utf-8")
      with self.assertRaisesRegex((TypeError, ValueError), "raw trial"):
        load_roll_boundary_verdict(path)

      inconsistent = self._r0()
      inconsistent["trials"][0]["bilateral_unsupported_physics_substeps"] = 1
      path.write_text(json.dumps(inconsistent), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "substep failure"):
        load_roll_boundary_verdict(path)

      stale_time = self._r0()
      failed = next(row for row in stale_time["trials"] if not row["success"])
      failed["time_to_success_s"] = 2.0
      path.write_text(json.dumps(stale_time), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "retained a success time"):
        load_roll_boundary_verdict(path)

      stale_cell = self._r0()
      stale_cell["cells"][0]["successes"] -= 1
      path.write_text(json.dumps(stale_cell), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "summaries disagree"):
        load_roll_boundary_verdict(path)

      stale_verdict = self._r0()
      stale_verdict["verdict"]["max_common_passing_height_m"] = 0.0025
      path.write_text(json.dumps(stale_verdict), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "verdict disagrees"):
        load_roll_boundary_verdict(path)

      stale_top_level = self._r0()
      stale_top_level.update({
        "max_common_passing_height_m": 0.0025,
        "first_non_common_height_m": 0.005,
        "croll_bracket_m": [0.0025, 0.005],
      })
      path.write_text(json.dumps(stale_top_level), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "top-level"):
        load_roll_boundary_verdict(path)

      reused_env_ids = self._r0()
      for row in reused_env_ids["trials"]:
        row["env_id"] %= roll_boundary.OFFICIAL_ENVS_PER_HEIGHT
      path.write_text(json.dumps(reused_env_ids), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "env ids"):
        load_roll_boundary_verdict(path)

      for value in (0.0, 10.02, 0.51):
        bad_time = self._r0()
        successful = next(row for row in bad_time["trials"] if row["success"])
        successful["time_to_success_s"] = value
        path.write_text(json.dumps(bad_time), encoding="utf-8")
        with self.subTest(time_to_success_s=value), self.assertRaisesRegex(
          ValueError, "control-step grid"
        ):
          load_roll_boundary_verdict(path)

      for field, value in (
        ("terrain_index", 99),
        ("terrain_key", "stair_wrong"),
        ("target_height_m", 0.0),
        ("target_pitch_rad", 1.0),
      ):
        bad_identity = self._r0()
        bad_identity["trials"][0][field] = value
        path.write_text(json.dumps(bad_identity), encoding="utf-8")
        with self.subTest(field=field), self.assertRaisesRegex(
          ValueError, "terrain|posture"
        ):
          load_roll_boundary_verdict(path)

  def test_r0_verdict_requires_complete_zero_safety_contract(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "r0.json"
      for field in (
        "termination_trials_required", "non_wheel_contact_trials_required",
      ):
        payload = self._r0()
        del payload["protocol"]["safety"][field]
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.subTest(field=field), self.assertRaisesRegex(ValueError, "safety contract"):
          load_roll_boundary_verdict(path)

  def test_roll_assist_contact_or_substep_contract_drift_fails_closed(self):
    cfg = make_stair_roll_assist_env_cfg(play=True)
    cfg.scene.entities["robot"].collisions[0].solref[
      "wheel_.*_collision"
    ] = (0.005, 1.0)
    with self.assertRaisesRegex(ValueError, "differs from R0"):
      validate_roll_assist_observation_contract(cfg)
    cfg = make_stair_roll_assist_env_cfg(play=True)
    cfg.metrics["roll_assist_substep_support"].per_substep = False
    with self.assertRaisesRegex(ValueError, "strict 5 ms"):
      validate_roll_assist_observation_contract(cfg)

  def test_checkpoint_record_and_one_block_authorization_are_hash_bound(self):
    record = {
      "schema_version": 1,
      "task": ROLL_ASSIST_TASK_ID,
      "training_seed": 1,
      "git_sha": "a" * 40,
      "r0_sha256": "b" * 64,
      "reward_calibration_sha256": "c" * 64,
      "action_scales": list(ROLL_ASSIST_ACTION_SCALES),
      "wheel_residual_exact_zero": True,
      "zero_initialized_deterministic_mean": True,
      "update25_curriculum_decided": True,
      "active_height_m": 0.0075,
      "completed_updates": 100,
    }
    self.assertEqual(validate_roll_assist_training_record(
      record, git_sha="a" * 40, r0_sha256="b" * 64,
      reward_calibration_sha256="c" * 64,
    ), 100)
    authorization = build_extension_authorization(
      selected_checkpoint_file=Path("model_99.pt"),
      selected_checkpoint_sha256="d" * 64,
      selected_completed_updates=100,
      target_total_updates=200,
      continuation_evidence_sha256="e" * 64,
    )
    self.assertEqual(validate_extension_authorization(authorization), authorization)
    intermediate = build_extension_authorization(
      selected_checkpoint_file=Path("model_75.pt"),
      selected_checkpoint_sha256="f" * 64,
      selected_completed_updates=76,
      target_total_updates=176,
      continuation_evidence_sha256="e" * 64,
    )
    self.assertEqual(validate_extension_authorization(intermediate), intermediate)
    drifted = dict(authorization)
    drifted["target_total_updates"] = 300
    with self.assertRaises(ValueError):
      validate_extension_authorization(drifted)

  def test_reward_stall_requires_full_rollout_safety(self):
    stall = {
      "kind": "roll_assist_zero_residual_stall",
      "evidence_eligible": True,
      "protocol": {
        "policy_action": [0.0] * 6, "wheel_residual_exact_zero": True,
        "measurement_window_s": 3.0, "height_role": "Hnext",
        "strict_physics_substep_support_required": True,
      },
      "safety": {
        "scope": "final_3s_measurement_window", "terminations": 0,
        "non_wheel_contacts": 0, "bilateral_airborne": 0,
      },
      "full_rollout_safety": {
        "scope": "post_reset_settle_and_drive", "terminations": 0,
        "non_wheel_contacts": 0, "bilateral_airborne": 0,
      },
      "measurement": {"inherited_positive_reward_rate": 3.5, "samples": 150},
    }
    self.assertEqual(positive_reward_rate_from_stall(stall), 3.5)
    stall["full_rollout_safety"]["bilateral_airborne"] = 1
    with self.assertRaises(ValueError):
      positive_reward_rate_from_stall(stall)
    stall["full_rollout_safety"]["bilateral_airborne"] = 0
    stall["protocol"]["strict_physics_substep_support_required"] = False
    with self.assertRaisesRegex(ValueError, "every 5 ms"):
      positive_reward_rate_from_stall(stall)

  def test_reward_calibration_is_formula_and_hash_bound(self):
    progress, success = reward_weights(3.5)
    self.assertAlmostEqual(progress, 100.0)
    self.assertEqual(success, 7.0)
    artifact = build_reward_calibration(
      baseline_positive_reward_rate=3.5,
      source_stall_sha256="1" * 64,
      roll_boundary_sha256="2" * 64,
    )
    self.assertEqual(validate_reward_calibration(
      artifact, expected_roll_boundary_sha256="2" * 64
    ), artifact)
    drifted = dict(artifact)
    drifted["progress_weight"] += 1.0
    with self.assertRaises(ValueError):
      validate_reward_calibration(drifted)

  def test_qualified_reward_marker_binds_exact_file_bytes(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "reward.json"
      artifact = build_reward_calibration(
        baseline_positive_reward_rate=3.5,
        source_stall_sha256="1" * 64,
        roll_boundary_sha256="2" * 64,
      )
      path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
      self.assertNotEqual(file_sha256(path), artifact["calibration_sha256"])
      original = file_sha256(path)
      path.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
      self.assertNotEqual(file_sha256(path), original)
      self.assertEqual(validate_reward_calibration(json.loads(path.read_text())), artifact)



class CurriculumAndGateTest(unittest.TestCase):
  def test_curriculum_can_only_switch_once_at_update25(self):
    state = RollAssistCurriculumState(0.005, 0.0075)
    self.assertEqual(state.active_height_m, 0.005)
    with self.assertRaises(ValueError):
      state.evaluate_update25(completed_updates=24, success_rate=1.0,
                              terminations=0, non_wheel_contacts=0, bilateral_airborne=0)
    self.assertTrue(state.evaluate_update25(
      completed_updates=25, success_rate=0.8,
      terminations=0, non_wheel_contacts=0, bilateral_airborne=0,
    ))
    self.assertEqual(state.active_height_m, 0.0075)
    with self.assertRaises(ValueError):
      state.evaluate_update25(completed_updates=25, success_rate=1.0,
                              terminations=0, non_wheel_contacts=0, bilateral_airborne=0)

  def test_runtime_update_boundary_waits_for_common_step_600(self):
    curriculum = RollAssistCurriculum.__new__(RollAssistCurriculum)
    curriculum.state = RollAssistCurriculumState(0.005, 0.0075)
    curriculum.state.record_completed_episodes(
      completed=10, successes=8, terminations=0,
      non_wheel_contacts=0, bilateral_airborne=0,
    )
    curriculum.flat_env_count = 64
    curriculum.last_processed_step = 598
    env = SimpleNamespace(
      common_step_counter=599,
      reset_buf=torch.zeros(256, dtype=torch.bool),
    )
    curriculum.record_step(env)
    self.assertFalse(curriculum.state.decision_made)
    env.common_step_counter = 600
    curriculum.record_step(env)
    self.assertTrue(curriculum.state.decision_made)
    self.assertTrue(curriculum.state.switched_to_hnext)

  def test_cumulative_episode_window_round_trips_and_drives_update25(self):
    state = RollAssistCurriculumState(0.005, 0.0075)
    state.record_completed_episodes(
      completed=10, successes=8, terminations=0,
      non_wheel_contacts=0, bilateral_airborne=0,
    )
    state.record_completed_episodes(
      completed=5, successes=4, terminations=0,
      non_wheel_contacts=0, bilateral_airborne=0,
    )
    restored = RollAssistCurriculumState.from_state_dict(state.state_dict())
    self.assertEqual(restored.completed_stair_episodes, 15)
    self.assertEqual(restored.successful_stair_episodes, 12)
    self.assertTrue(restored.evaluate_update25(completed_updates=25))

  def test_cumulative_unsafe_episode_prevents_promotion(self):
    state = RollAssistCurriculumState(0.005, 0.0075)
    state.record_completed_episodes(
      completed=10, successes=10, terminations=0,
      non_wheel_contacts=0, bilateral_airborne=1,
    )
    self.assertFalse(state.evaluate_update25(completed_updates=25))
    self.assertFalse(state.update25_safe)

  def test_unsafe_or_low_success_update25_never_promotes(self):
    for rate, airborne in ((0.79, 0), (1.0, 1)):
      state = RollAssistCurriculumState(0.005, 0.0075)
      self.assertFalse(state.evaluate_update25(
        completed_updates=25, success_rate=rate, terminations=0,
        non_wheel_contacts=0, bilateral_airborne=airborne,
      ))
      self.assertEqual(state.active_height_m, 0.005)

  def test_continuation_requires_all_six_gates_and_bootstrap(self):
    baseline = np.linspace(0.0, 0.02, 96)
    candidate = baseline + 0.01
    result = continuation_gate(
      flat_retention_passed=True, hpass_card_successes=(44, 48),
      hnext_terminations=0, hnext_non_wheel_contacts=0,
      hnext_bilateral_airborne=0, wheel_residual_abs_max=0.0,
      hnext_candidate_successes=2, hnext_baseline_successes=0,
      paired_candidate_progress=candidate, paired_baseline_progress=baseline,
    )
    self.assertTrue(result["authorized"])
    self.assertGreater(result["paired_progress_bootstrap"]["lower_95_m"], 0.0)
    rejected = continuation_gate(
      flat_retention_passed=True, hpass_card_successes=(44, 48),
      hnext_terminations=0, hnext_non_wheel_contacts=0,
      hnext_bilateral_airborne=0, wheel_residual_abs_max=1e-12,
      hnext_candidate_successes=2, hnext_baseline_successes=0,
      paired_candidate_progress=candidate, paired_baseline_progress=baseline,
    )
    self.assertFalse(rejected["authorized"])

  def test_training_record_accepts_actual_intermediate_save_grid(self):
    base = {
      "schema_version": 1, "task": ROLL_ASSIST_TASK_ID,
      "training_seed": 1, "git_sha": "a" * 40,
      "r0_sha256": "b" * 64, "reward_calibration_sha256": "c" * 64,
      "action_scales": list(ROLL_ASSIST_ACTION_SCALES),
      "wheel_residual_exact_zero": True,
      "zero_initialized_deterministic_mean": True,
      "update25_curriculum_decided": True,
      "active_height_m": 0.0075,
      "completed_updates": 51,
    }
    for updates in (51, 76, 100, 101, 126, 151, 176, 200, 500):
      record = {**base, "completed_updates": updates}
      self.assertEqual(validate_roll_assist_training_record(
        record, git_sha="a" * 40, r0_sha256="b" * 64,
        reward_calibration_sha256="c" * 64,
      ), updates)
    for updates in (50, 75, 99, 102, 501):
      with self.assertRaises(ValueError):
        validate_roll_assist_training_record(
          {**base, "completed_updates": updates}, git_sha="a" * 40,
          r0_sha256="b" * 64, reward_calibration_sha256="c" * 64,
        )

  def test_final_gate_and_k3_use_thresholds_not_score_ranking(self):
    self.assertTrue(final_expansion_gate(
      hnext_card_successes=(44, 48), safety_gate_passed=True,
      wheel_residual_abs_max=0.0,
    )["passed"])
    checkpoints = [
      {"completed_updates": 51, "passed": True, "score": 100},
      {"completed_updates": 76, "passed": True, "score": 1},
      {"completed_updates": 100, "passed": False, "score": 1000},
    ]
    self.assertEqual(newest_passer(checkpoints)["completed_updates"], 76)
    with self.assertRaises(ValueError):
      newest_passer([
        {"completed_updates": 50, "passed": True},
        {"completed_updates": 75, "passed": True},
        {"completed_updates": 100, "passed": True},
      ])
    with self.assertRaises(TypeError):
      newest_passer([
        {"completed_updates": 51, "passed": True},
        {"completed_updates": 76, "passed": None},
        {"completed_updates": 100, "passed": False},
      ])


if __name__ == "__main__":
  unittest.main()
