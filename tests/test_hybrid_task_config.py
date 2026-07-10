import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from mjlab.tasks.registry import list_tasks, load_env_cfg

import hoppertrex_mjlab.tasks as hoppertrex_tasks
from hoppertrex_mjlab.hybrid.config import HYBRID_ACTION_NAMES, HYBRID_STAGES
from hoppertrex_mjlab.hybrid.posture import LEG_JOINT_NAMES
from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
  HYBRID_TASK_IDS,
  WHEEL_JOINT_NAMES,
  HybridWheelLegActionCfg,
  PostureCommandCfg,
  _load_controller,
  _load_posture_map,
  make_hoppertrex_hybrid_env_cfg,
)


EXPECTED_OBSERVATION_TERMS = (
  "base_lin_vel",
  "base_ang_vel",
  "projected_gravity",
  "velocity_command",
  "posture_command",
  "joint_pos",
  "joint_vel",
  "controller_baseline",
  "applied_residual",
)


def _stable_hash(payload):
  encoded = json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
  ).encode("ascii")
  return hashlib.sha256(encoded).hexdigest()


def _controller_payload():
  gain = [[1.0, 2.0, 3.0, 4.0]]
  return {
    "schema_version": 1,
    "controller_type": "lqr",
    "state_names": [
      "pitch",
      "pitch_rate",
      "vx_error",
      "signed_wheel_speed_error",
    ],
    "gain": gain,
    "controllability_rank": 4,
    "heldout_one_step_nrmse": {"maximum": 0.15},
    "fallback_reasons": [],
    "gain_hash": _stable_hash(
      {
        "controller_type": "lqr",
        "state_names": (
          "pitch",
          "pitch_rate",
          "vx_error",
          "signed_wheel_speed_error",
        ),
        "gain": gain,
      }
    ),
  }


def _posture_payload():
  coefficients = [
    [-0.2, 0.2, -0.4, 0.4],
    [-1.0, 1.0, -0.8, 0.8],
    [0.5, 0.5, -0.3, -0.3],
  ]
  return {
    "schema_version": 1,
    "feature_names": ["bias", "height", "pitch"],
    "joint_names": [
      "thigh_left_01",
      "thigh_right_01",
      "knee_left",
      "knee_right",
    ],
    "coefficients": coefficients,
    "training_envelope": {
      "height": [0.32, 0.48],
      "pitch": [-0.08, 0.08],
    },
    "map_hash": _stable_hash(
      {
        "feature_names": ("bias", "height", "pitch"),
        "joint_names": (
          "thigh_left_01",
          "thigh_right_01",
          "knee_left",
          "knee_right",
        ),
        "coefficients": coefficients,
      }
    ),
  }


def _write_json(directory: str, name: str, payload) -> Path:
  path = Path(directory) / name
  path.write_text(json.dumps(payload), encoding="utf-8")
  return path


class HybridTaskConfigTest(unittest.TestCase):
  def test_action_cfg_controls_two_wheels_and_four_joints_of_two_legs(self):
    cfg = make_hoppertrex_hybrid_env_cfg(stage=0)
    self.assertEqual(tuple(cfg.actions), ("hybrid_wheel_leg",))

    action = cfg.actions["hybrid_wheel_leg"]
    self.assertIsInstance(action, HybridWheelLegActionCfg)
    self.assertEqual(action.action_names, HYBRID_ACTION_NAMES)
    self.assertEqual(action.action_dim, 6)
    self.assertEqual(action.wheel_joint_names, ("wheel_left", "wheel_right"))
    self.assertEqual(
      action.leg_joint_names,
      (
        "thigh_left_01",
        "thigh_right_01",
        "knee_left",
        "knee_right",
      ),
    )
    self.assertEqual(action.wheel_joint_names, WHEEL_JOINT_NAMES)
    self.assertEqual(action.leg_joint_names, LEG_JOINT_NAMES)

  def test_stage_masks_change_without_changing_action_or_observation_shape(self):
    observation_terms = None
    for stage_index, stage in HYBRID_STAGES.items():
      with self.subTest(stage=stage_index):
        cfg = make_hoppertrex_hybrid_env_cfg(stage=stage_index)
        action = cfg.actions["hybrid_wheel_leg"]
        self.assertEqual(action.action_mask, stage.action_mask)
        self.assertEqual(action.action_scales, stage.action_scales)
        self.assertEqual(action.action_dim, 6)

        actor_terms = tuple(cfg.observations["actor"].terms)
        critic_terms = tuple(cfg.observations["critic"].terms)
        self.assertEqual(actor_terms, EXPECTED_OBSERVATION_TERMS)
        self.assertEqual(critic_terms, EXPECTED_OBSERVATION_TERMS)
        self.assertNotIn("actions", actor_terms)
        self.assertNotIn("last_action", actor_terms)
        if observation_terms is None:
          observation_terms = actor_terms
        self.assertEqual(actor_terms, observation_terms)

  def test_stage0_is_controller_only_and_uses_unqualified_local_pd_fallback(self):
    cfg = make_hoppertrex_hybrid_env_cfg(stage=0)
    action = cfg.actions["hybrid_wheel_leg"]

    self.assertEqual(action.action_mask, (False,) * 6)
    self.assertEqual(action.controller_type, "pd")
    self.assertFalse(action.controller_qualified)
    self.assertEqual(action.controller_source, "local-unqualified-pd-fallback")

  def test_posture_command_is_always_two_dimensional(self):
    for stage_index in HYBRID_STAGES:
      with self.subTest(stage=stage_index):
        cfg = make_hoppertrex_hybrid_env_cfg(stage=stage_index)
        posture = cfg.commands["posture"]
        self.assertIsInstance(posture, PostureCommandCfg)
        self.assertEqual(posture.command_dim, 2)

    stage3 = make_hoppertrex_hybrid_env_cfg(stage=3)
    posture = stage3.commands["posture"]
    self.assertFalse(posture.qualified)
    self.assertEqual(posture.height_range[0], posture.height_range[1])
    self.assertEqual(posture.pitch_range, (0.0, 0.0))

  def test_valid_controller_and_posture_artifacts_load_with_verified_hashes(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      controller_path = _write_json(
        temp_dir,
        "controller.json",
        _controller_payload(),
      )
      posture_path = _write_json(
        temp_dir,
        "posture.json",
        _posture_payload(),
      )

      controller = _load_controller(controller_path)
      posture = _load_posture_map(posture_path)

    self.assertEqual(controller.controller_type, "lqr")
    self.assertTrue(controller.qualified)
    self.assertEqual(controller.gain, (1.0, 2.0, 3.0, 4.0))
    self.assertTrue(posture.qualified)
    self.assertEqual(posture.height_range, (0.32, 0.48))
    self.assertEqual(posture.pitch_range, (-0.08, 0.08))

  def test_tampered_artifact_hashes_are_rejected(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      controller_payload = _controller_payload()
      controller_payload["gain"][0][0] = 99.0
      controller_path = _write_json(
        temp_dir,
        "controller.json",
        controller_payload,
      )
      posture_payload = _posture_payload()
      posture_payload["coefficients"][0][0] = 99.0
      posture_path = _write_json(
        temp_dir,
        "posture.json",
        posture_payload,
      )

      with self.assertRaisesRegex(ValueError, "gain_hash"):
        _load_controller(controller_path)
      with self.assertRaisesRegex(ValueError, "map_hash"):
        _load_posture_map(posture_path)

  def test_lqr_artifact_that_fails_qualification_is_rejected(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      payload = _controller_payload()
      payload["heldout_one_step_nrmse"]["maximum"] = 0.150001
      path = _write_json(temp_dir, "controller.json", payload)

      with self.assertRaisesRegex(ValueError, "does not meet"):
        _load_controller(path)

  def test_posture_command_qualification_only_marks_active_posture_stage(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      posture_path = _write_json(
        temp_dir,
        "posture.json",
        _posture_payload(),
      )

      stage0 = make_hoppertrex_hybrid_env_cfg(
        stage=0,
        posture_map_path=posture_path,
      )
      stage3 = make_hoppertrex_hybrid_env_cfg(
        stage=3,
        posture_map_path=posture_path,
      )

    self.assertTrue(
      stage0.actions["hybrid_wheel_leg"].posture_map_qualified
    )
    self.assertFalse(stage0.commands["posture"].qualified)
    self.assertEqual(
      stage0.commands["posture"].height_range,
      (stage0.commands["posture"].height_range[0],) * 2,
    )
    self.assertTrue(stage3.commands["posture"].qualified)
    self.assertEqual(stage3.commands["posture"].height_range, (0.32, 0.48))

  def test_stage5_uses_robust_level2_reset_and_exact_push(self):
    cfg = make_hoppertrex_hybrid_env_cfg(stage=5)
    reset = cfg.events["reset_root_state_with_small_disturbance"]
    push = cfg.events["push_robot"]

    self.assertEqual(
      reset.params["pose_range"]["pitch"],
      (-math.radians(5.0), math.radians(5.0)),
    )
    self.assertEqual(reset.params["velocity_range"]["x"], (-0.10, 0.10))
    self.assertEqual(reset.params["velocity_range"]["pitch"], (-0.20, 0.20))
    self.assertEqual(push.interval_range_s, (3.0, 5.0))
    self.assertEqual(push.params["velocity_range"]["x"], (-0.08, 0.08))
    self.assertEqual(push.params["velocity_range"]["pitch"], (-0.12, 0.12))

    play_cfg = make_hoppertrex_hybrid_env_cfg(stage=5, play=True)
    self.assertNotIn("push_robot", play_cfg.events)

  def test_all_six_train_and_play_tasks_are_registered(self):
    registered = set(list_tasks())
    self.assertEqual(
      HYBRID_TASK_IDS,
      tuple(f"HopperTrex-Hybrid-v2-Stage{index}" for index in range(6)),
    )
    for task_id in HYBRID_TASK_IDS:
      with self.subTest(task=task_id):
        self.assertIn(task_id, registered)
        train_cfg = load_env_cfg(task_id)
        play_cfg = load_env_cfg(task_id, play=True)
        self.assertIsInstance(
          train_cfg.actions["hybrid_wheel_leg"],
          HybridWheelLegActionCfg,
        )
        self.assertIsInstance(
          play_cfg.actions["hybrid_wheel_leg"],
          HybridWheelLegActionCfg,
        )
        self.assertGreaterEqual(play_cfg.episode_length_s, 1.0e9)

  def test_tasks_package_exports_hybrid_task_ids(self):
    self.assertEqual(hoppertrex_tasks.HOPPERTREX_HYBRID_TASK_IDS, HYBRID_TASK_IDS)


if __name__ == "__main__":
  unittest.main()
