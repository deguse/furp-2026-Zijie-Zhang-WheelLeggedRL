import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand
import torch

import hoppertrex_mjlab.tasks as hoppertrex_tasks
import hoppertrex_mjlab.tasks.hoppertrex_hybrid_task as hybrid_task
from hoppertrex_mjlab.hybrid.config import HYBRID_ACTION_NAMES, HYBRID_STAGES
from hoppertrex_mjlab.hybrid.calibration import calibration_artifact
from hoppertrex_mjlab.hybrid.identification import (
  STATE_DEFINITION_VERSION,
  state_construction_spec,
)
from hoppertrex_mjlab.hybrid.posture import LEG_JOINT_NAMES
from hoppertrex_mjlab.hybrid.station_calibration import (
  station_calibration_artifact,
)
from hoppertrex_mjlab.hybrid.yaw_calibration import yaw_calibration_artifact
from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
  HYBRID_RESIDUAL_L2_WEIGHT,
  HYBRID_TASK_IDS,
  HYBRID_PLANAR_COMMAND_RESAMPLING_TIME_RANGE,
  HYBRID_PLANAR_LINEAR_ONLY_ENVS,
  HYBRID_PLANAR_LIN_VEL_X_ABS_RANGE,
  HYBRID_PLANAR_STANDING_ENVS,
  HYBRID_PLANAR_YAW_ONLY_ENVS,
  HYBRID_PLANAR_YAW_RATE_ABS_RANGE,
  STAGE1_ACTIVE_LIN_VEL_X_ABS_RANGE,
  STAGE1_COMMAND_RESAMPLING_TIME_RANGE,
  STAGE1_EXTENSION_ENVS,
  STAGE1_EXTENSION_LIN_VEL_X_ABS_RANGE,
  STAGE1_GLOBAL_RESIDUAL_L2_WEIGHT,
  STAGE1_HEALTHY_PITCH_ABS,
  STAGE1_HEALTHY_PITCH_RATE_ABS,
  STAGE1_HEALTHY_RESIDUAL_L2_WEIGHT,
  STAGE1_HEALTHY_VELOCITY_ERROR,
  STAGE1_STANDING_ENVS,
  STAGE1_TRACK_LIN_VEL_STD,
  WHEEL_JOINT_NAMES,
  HybridPlanarVelocityCommand,
  HybridPlanarVelocityCommandCfg,
  HybridWheelLegActionCfg,
  PostureCommandCfg,
  Stage1VelocityCommand,
  Stage1VelocityCommandCfg,
  _load_controller,
  _load_posture_map,
  hybrid_provenance_lines,
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
    "envelope_verification": {
      "method": "all_feasible_grid_rectangle",
      "grid_shape": [3, 3],
    },
    'source_sweep': {
      'git_sha': 'abc123',
      'seed': 1,
      'controller_gain_hash': _controller_payload()['gain_hash'],
      'calibration_hash': 'calibration-hash',
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


def _calibration_payload():
  return calibration_artifact(
    controller_gain_hash=_controller_payload()['gain_hash'],
    scale=0.86,
    bias=-0.012,
    seed=1,
    candidates=[],
  )


def _write_json(directory: str, name: str, payload) -> Path:
  path = Path(directory) / name
  path.write_text(json.dumps(payload), encoding="utf-8")
  return path


class HybridTaskConfigTest(unittest.TestCase):
  def test_leg_authority_environment_override_is_stage5_only(self):
    with patch.dict(
      hybrid_task.os.environ,
      {hybrid_task.LEG_RESIDUAL_SCALE_ENV: "0.070"},
      clear=False,
    ):
      cfg = make_hoppertrex_hybrid_env_cfg(stage=5)
      self.assertEqual(
        cfg.actions["hybrid_wheel_leg"].action_scales,
        (0.5, 0.3, 0.07, 0.07, 0.07, 0.07),
      )
      with self.assertRaisesRegex(ValueError, "restricted to Stage5"):
        make_hoppertrex_hybrid_env_cfg(stage=4)

  def test_stage1_healthy_residual_penalty_turns_off_during_recovery(self):
    command = torch.tensor(
      [
        [0.04, 0.00, 0.00],
        [-0.04, 0.00, 0.00],
      ]
    )
    root_link_lin_vel_b = torch.tensor(
      [
        [0.04, 0.00, 0.00],
        [0.00, 0.00, 0.00],
      ]
    )
    action = SimpleNamespace(
      applied_residual=torch.tensor(
        [[0.20, 0.00, 0.00, 0.00, 0.00, 0.00]] * 2
      )
    )
    env = SimpleNamespace(
      command_manager=SimpleNamespace(
        get_command=lambda name: command,
      ),
      action_manager=SimpleNamespace(get_term=lambda name: action),
      scene={
        "robot": SimpleNamespace(
          data=SimpleNamespace(
            root_link_lin_vel_b=root_link_lin_vel_b,
            root_link_ang_vel_b=torch.zeros((2, 3)),
            projected_gravity_b=torch.tensor(
              [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]
            ),
          )
        )
      },
    )

    torch.testing.assert_close(
      hybrid_task.healthy_applied_residual_l2(
        env,
        command_name="twist",
        max_velocity_error=STAGE1_HEALTHY_VELOCITY_ERROR,
        max_pitch_abs=STAGE1_HEALTHY_PITCH_ABS,
        max_pitch_rate_abs=STAGE1_HEALTHY_PITCH_RATE_ABS,
      ),
      torch.tensor([0.20**2, 0.0]),
    )

  def test_applied_residual_magnitude_reward_uses_scaled_masked_residual(self):
    action = SimpleNamespace(
      applied_residual=torch.tensor(
        [[0.50, -0.25, 0.00, 0.00, 0.00, 0.00]]
      )
    )
    env = SimpleNamespace(
      action_manager=SimpleNamespace(get_term=lambda name: action)
    )

    torch.testing.assert_close(
      hybrid_task.applied_residual_l2(env),
      torch.tensor([0.50**2 + 0.25**2]),
    )

  def test_action_smoothness_rewards_use_only_applied_residual_history(self):
    action = SimpleNamespace(
      applied_residual=torch.tensor(
        [[0.50, 0.00, 0.00, 0.00, 0.00, 0.00]]
      ),
      previous_applied_residual=torch.tensor(
        [[0.25, 0.00, 0.00, 0.00, 0.00, 0.00]]
      ),
      previous_previous_applied_residual=torch.tensor(
        [[0.10, 0.00, 0.00, 0.00, 0.00, 0.00]]
      ),
    )
    env = SimpleNamespace(
      action_manager=SimpleNamespace(
        get_term=lambda name: action,
        action=torch.full((1, 6), 100.0),
        prev_action=torch.full((1, 6), -100.0),
        prev_prev_action=torch.full((1, 6), 50.0),
      )
    )

    torch.testing.assert_close(
      hybrid_task.applied_residual_rate_l2(env),
      torch.tensor([0.25**2]),
    )
    torch.testing.assert_close(
      hybrid_task.applied_residual_acc_l2(env),
      torch.tensor([(0.50 - 2 * 0.25 + 0.10) ** 2]),
    )

  def test_hybrid_configs_replace_raw_action_smoothness_rewards(self):
    for stage in HYBRID_STAGES:
      with self.subTest(stage=stage):
        cfg = make_hoppertrex_hybrid_env_cfg(stage=stage)
        self.assertIs(
          cfg.rewards["action_rate_l2"].func,
          hybrid_task.applied_residual_rate_l2,
        )
        if "action_acc_l2" in cfg.rewards:
          self.assertIs(
            cfg.rewards["action_acc_l2"].func,
            hybrid_task.applied_residual_acc_l2,
          )

        if stage == 0:
          self.assertNotIn("applied_residual_l2", cfg.rewards)
        else:
          self.assertIs(
            cfg.rewards["applied_residual_l2"].func,
            hybrid_task.applied_residual_l2,
          )
          expected_weight = (
            STAGE1_GLOBAL_RESIDUAL_L2_WEIGHT
            if stage == 1
            else HYBRID_RESIDUAL_L2_WEIGHT
          )
          self.assertEqual(
            cfg.rewards["applied_residual_l2"].weight,
            expected_weight,
          )

  def test_stage1_sampling_and_reward_match_linear_gate(self):
    cfg = make_hoppertrex_hybrid_env_cfg(stage=1)
    command = cfg.commands["twist"]

    self.assertIsInstance(command, Stage1VelocityCommandCfg)
    self.assertEqual(
      command.nominal_abs_range,
      STAGE1_ACTIVE_LIN_VEL_X_ABS_RANGE,
    )
    self.assertEqual(
      command.extension_abs_range,
      STAGE1_EXTENSION_LIN_VEL_X_ABS_RANGE,
    )
    self.assertEqual(command.rel_standing_envs, STAGE1_STANDING_ENVS)
    self.assertEqual(command.rel_extension_envs, STAGE1_EXTENSION_ENVS)
    self.assertEqual(
      command.resampling_time_range,
      STAGE1_COMMAND_RESAMPLING_TIME_RANGE,
    )
    self.assertEqual(
      cfg.rewards["track_linear_velocity"].params["std"],
      STAGE1_TRACK_LIN_VEL_STD,
    )
    self.assertNotIn("lin_vel_x_sign_alignment", cfg.rewards)
    self.assertNotIn("track_linear_velocity_x_fine", cfg.rewards)
    healthy_reward = cfg.rewards["healthy_applied_residual_l2"]
    self.assertIs(
      healthy_reward.func,
      hybrid_task.healthy_applied_residual_l2,
    )
    self.assertEqual(
      healthy_reward.weight,
      STAGE1_HEALTHY_RESIDUAL_L2_WEIGHT,
    )
    self.assertEqual(
      healthy_reward.params,
      {
        "command_name": "twist",
        "max_velocity_error": STAGE1_HEALTHY_VELOCITY_ERROR,
        "max_pitch_abs": STAGE1_HEALTHY_PITCH_ABS,
        "max_pitch_rate_abs": STAGE1_HEALTHY_PITCH_RATE_ABS,
        "action_indices": (0,),
      },
    )

  def test_stage2_rehearses_stage1_robustness_without_legacy_sign_reward(self):
    cfg = make_hoppertrex_hybrid_env_cfg(stage=2)

    self.assertNotIn("lin_vel_x_sign_alignment", cfg.rewards)
    self.assertNotIn("yaw_sign_alignment", cfg.rewards)
    self.assertEqual(
      cfg.rewards["track_angular_velocity"].params["std"],
      hybrid_task.HYBRID_TRACK_ANG_VEL_STD,
    )
    healthy = cfg.rewards["healthy_applied_residual_l2"]
    self.assertEqual(healthy.params["action_indices"], (0, 1))
    self.assertIn("push_robot", cfg.events)
    self.assertEqual(cfg.events["push_robot"].interval_range_s, (5.0, 8.0))
    self.assertIn("stage1_mild_mismatch", cfg.events)
    self.assertIn("reset_root_state_with_small_disturbance", cfg.events)
    self.assertEqual(
      cfg.commands["twist"].rel_linear_retention_envs,
      hybrid_task.STAGE2_LINEAR_RETENTION_ENVS,
    )
    self.assertEqual(
      cfg.commands["twist"].linear_retention_abs_range,
      hybrid_task.STAGE2_LINEAR_RETENTION_ABS_RANGE,
    )

    play_cfg = make_hoppertrex_hybrid_env_cfg(stage=2, play=True)
    self.assertNotIn("push_robot", play_cfg.events)
    self.assertNotIn("stage1_mild_mismatch", play_cfg.events)
    self.assertNotIn("reset_root_state_with_small_disturbance", play_cfg.events)

  def test_stage1_uses_level1_reset_and_weaker_push_than_stage5(self):
    cfg = make_hoppertrex_hybrid_env_cfg(stage=1)
    reset = cfg.events["reset_root_state_with_small_disturbance"]
    push = cfg.events["push_robot"]

    self.assertEqual(
      reset.params["pose_range"]["pitch"],
      (-math.radians(2.0), math.radians(2.0)),
    )
    self.assertEqual(reset.params["velocity_range"]["x"], (-0.05, 0.05))
    self.assertEqual(reset.params["velocity_range"]["pitch"], (-0.10, 0.10))
    self.assertEqual(push.interval_range_s, (5.0, 8.0))
    self.assertEqual(push.params["velocity_range"]["x"], (-0.04, 0.04))
    self.assertEqual(push.params["velocity_range"]["pitch"], (-0.06, 0.06))
    self.assertIn("stage1_mild_mismatch", cfg.events)

    play_cfg = make_hoppertrex_hybrid_env_cfg(stage=1, play=True)
    self.assertNotIn("reset_root_state_with_small_disturbance", play_cfg.events)
    self.assertNotIn("push_robot", play_cfg.events)
    self.assertNotIn("stage1_mild_mismatch", play_cfg.events)

  def test_stage1_sampler_stratifies_standing_nominal_and_extension(self):
    count = 4000
    command = object.__new__(Stage1VelocityCommand)
    command.cfg = SimpleNamespace(
      rel_standing_envs=STAGE1_STANDING_ENVS,
      rel_extension_envs=STAGE1_EXTENSION_ENVS,
      nominal_abs_range=STAGE1_ACTIVE_LIN_VEL_X_ABS_RANGE,
      extension_abs_range=STAGE1_EXTENSION_LIN_VEL_X_ABS_RANGE,
    )
    command._env = SimpleNamespace(device="cpu")
    command.vel_command_b = torch.zeros((count, 3))
    command.vel_command_w = torch.zeros_like(command.vel_command_b)
    command.is_standing_env = torch.zeros(count, dtype=torch.bool)
    command.is_heading_env = torch.zeros(count, dtype=torch.bool)
    command.is_world_env = torch.zeros(count, dtype=torch.bool)
    command.is_forward_env = torch.zeros(count, dtype=torch.bool)

    torch.manual_seed(7)
    command._resample_command(torch.arange(count))

    speed = command.vel_command_b[:, 0]
    standing = speed == 0.0
    extension = speed.abs() > STAGE1_ACTIVE_LIN_VEL_X_ABS_RANGE[1]
    nominal = ~standing & ~extension
    self.assertGreater(int(standing.sum()), 650)
    self.assertGreater(int(nominal.sum()), 2200)
    self.assertGreater(int(extension.sum()), 650)
    self.assertTrue(torch.all(
      speed[nominal].abs() >= STAGE1_ACTIVE_LIN_VEL_X_ABS_RANGE[0]
    ))
    self.assertTrue(torch.all(
      speed[extension].abs() >= STAGE1_EXTENSION_LIN_VEL_X_ABS_RANGE[0]
    ))
    self.assertGreater(int((speed > 0.0).sum()), 1400)
    self.assertGreater(int((speed < 0.0).sum()), 1400)

  def test_planar_stages_sample_axis_and_combined_commands(self):
    for stage in (2, 4, 5):
      with self.subTest(stage=stage):
        cfg = make_hoppertrex_hybrid_env_cfg(stage=stage)
        command = cfg.commands["twist"]
        self.assertIsInstance(command, HybridPlanarVelocityCommandCfg)
        self.assertEqual(
          command.resampling_time_range,
          HYBRID_PLANAR_COMMAND_RESAMPLING_TIME_RANGE,
        )
        self.assertEqual(
          command.rel_standing_envs,
          HYBRID_PLANAR_STANDING_ENVS,
        )
        self.assertEqual(
          command.rel_linear_only_envs,
          HYBRID_PLANAR_LINEAR_ONLY_ENVS,
        )
        self.assertEqual(
          command.rel_yaw_only_envs,
          HYBRID_PLANAR_YAW_ONLY_ENVS,
        )
        self.assertEqual(
          command.lin_vel_x_abs_range,
          HYBRID_PLANAR_LIN_VEL_X_ABS_RANGE,
        )
        self.assertEqual(
          command.yaw_rate_abs_range,
          HYBRID_PLANAR_YAW_RATE_ABS_RANGE,
        )

  def test_planar_sampler_produces_nonzero_axis_and_combo_groups(self):
    count = 2000
    command = object.__new__(HybridPlanarVelocityCommand)
    command.cfg = SimpleNamespace(
      rel_standing_envs=HYBRID_PLANAR_STANDING_ENVS,
      rel_linear_only_envs=HYBRID_PLANAR_LINEAR_ONLY_ENVS,
      rel_yaw_only_envs=HYBRID_PLANAR_YAW_ONLY_ENVS,
      rel_linear_retention_envs=0.0,
      lin_vel_x_abs_range=HYBRID_PLANAR_LIN_VEL_X_ABS_RANGE,
      yaw_rate_abs_range=HYBRID_PLANAR_YAW_RATE_ABS_RANGE,
      linear_retention_abs_range=STAGE1_EXTENSION_LIN_VEL_X_ABS_RANGE,
    )
    command._env = SimpleNamespace(device="cpu")
    command.vel_command_b = torch.zeros((count, 3))
    command.vel_command_w = torch.zeros_like(command.vel_command_b)
    command.is_standing_env = torch.zeros(count, dtype=torch.bool)
    env_ids = torch.arange(count)

    torch.manual_seed(7)
    with patch.object(UniformVelocityCommand, "_resample_command"):
      command._resample_command(env_ids)

    lin_active = command.vel_command_b[:, 0].abs() > 0.0
    yaw_active = command.vel_command_b[:, 2].abs() > 0.0
    standing = ~lin_active & ~yaw_active
    linear_only = lin_active & ~yaw_active
    yaw_only = ~lin_active & yaw_active
    combined = lin_active & yaw_active
    self.assertGreater(int(standing.sum()), 100)
    self.assertGreater(int(linear_only.sum()), 350)
    self.assertGreater(int(yaw_only.sum()), 350)
    self.assertGreater(int(combined.sum()), 600)
    self.assertTrue(torch.all(
      command.vel_command_b[lin_active, 0].abs()
      >= HYBRID_PLANAR_LIN_VEL_X_ABS_RANGE[0]
    ))
    self.assertTrue(torch.all(
      command.vel_command_b[yaw_active, 2].abs()
      >= HYBRID_PLANAR_YAW_RATE_ABS_RANGE[0]
    ))

  def test_stage2_sampler_rehearses_stage1_extension_as_linear_only(self):
    count = 4000
    command = object.__new__(HybridPlanarVelocityCommand)
    command.cfg = make_hoppertrex_hybrid_env_cfg(stage=2).commands["twist"]
    command._env = SimpleNamespace(device="cpu")
    command.vel_command_b = torch.zeros((count, 3))
    command.vel_command_w = torch.zeros_like(command.vel_command_b)
    command.is_standing_env = torch.zeros(count, dtype=torch.bool)
    env_ids = torch.arange(count)

    torch.manual_seed(11)
    with patch.object(UniformVelocityCommand, "_resample_command"):
      command._resample_command(env_ids)

    linear_speed = command.vel_command_b[:, 0].abs()
    yaw_speed = command.vel_command_b[:, 2].abs()
    retention = linear_speed > STAGE1_EXTENSION_LIN_VEL_X_ABS_RANGE[0]
    self.assertGreater(int(retention.sum()), 300)
    self.assertLess(int(retention.sum()), 500)
    self.assertTrue(torch.all(yaw_speed[retention] == 0.0))
    self.assertTrue(torch.all(
      linear_speed[retention] <= STAGE1_EXTENSION_LIN_VEL_X_ABS_RANGE[1]
    ))

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

  def test_posture_artifact_accepts_sweep_grid_hull_verification(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      payload = _posture_payload()
      payload["envelope_verification"]["method"] = (
        "all_feasible_sweep_grid_hull_rectangle"
      )
      posture_path = _write_json(temp_dir, "posture.json", payload)

      posture = _load_posture_map(posture_path)

    self.assertTrue(posture.qualified)

  def test_posture_map_must_match_loaded_controller(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      controller_path = _write_json(
        temp_dir, 'controller.json', _controller_payload()
      )
      posture_payload = _posture_payload()
      posture_payload['source_sweep']['controller_gain_hash'] = 'other-gain'
      posture_path = _write_json(temp_dir, 'posture.json', posture_payload)

      with self.assertRaisesRegex(ValueError, 'different controller'):
        make_hoppertrex_hybrid_env_cfg(
          stage=3,
          play=False,
          controller_path=controller_path,
          posture_map_path=posture_path,
        )

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

  def test_posture_artifact_without_verified_rectangle_is_rejected(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      payload = _posture_payload()
      payload.pop("envelope_verification")
      path = _write_json(temp_dir, "posture.json", payload)

      with self.assertRaisesRegex(ValueError, "verified grid rectangle"):
        _load_posture_map(path)

  def test_lqr_artifact_that_fails_qualification_is_rejected(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      payload = _controller_payload()
      payload["heldout_one_step_nrmse"]["maximum"] = 0.150001
      path = _write_json(temp_dir, "controller.json", payload)

      with self.assertRaisesRegex(ValueError, "does not meet"):
        _load_controller(path)

  def test_controller_state_construction_block_is_cross_checked(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      payload = _controller_payload()
      payload["state_construction"] = state_construction_spec(
        hybrid_task.DEFAULT_WHEEL_RADIUS
      )
      path = _write_json(temp_dir, "controller.json", payload)
      controller = _load_controller(path)
    self.assertTrue(controller.qualified)

  def test_controller_with_wrong_wheel_radius_is_rejected(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      payload = _controller_payload()
      payload["state_construction"] = state_construction_spec(
        0.5 * hybrid_task.DEFAULT_WHEEL_RADIUS
      )
      path = _write_json(temp_dir, "controller.json", payload)
      with self.assertRaisesRegex(ValueError, "wheel_radius"):
        _load_controller(path)

  def test_controller_with_wrong_state_definition_is_rejected(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      payload = _controller_payload()
      payload["state_construction"] = {
        "state_definition_version": "other_state_v9",
        "wheel_radius": hybrid_task.DEFAULT_WHEEL_RADIUS,
      }
      path = _write_json(temp_dir, "controller.json", payload)
      with self.assertRaisesRegex(ValueError, "state definition"):
        _load_controller(path)

  def test_controller_without_state_construction_loads_with_warning(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      path = _write_json(temp_dir, "controller.json", _controller_payload())
      stdout = io.StringIO()
      with contextlib.redirect_stdout(stdout):
        controller = _load_controller(path)
    self.assertTrue(controller.qualified)
    self.assertIn("state_construction", stdout.getvalue())
    self.assertEqual(
      state_construction_spec(hybrid_task.DEFAULT_WHEEL_RADIUS)[
        "state_definition_version"
      ],
      STATE_DEFINITION_VERSION,
    )

  def test_hybrid_provenance_lines_warn_on_fallback_and_uncalibrated(self):
    lines = hybrid_provenance_lines(make_hoppertrex_hybrid_env_cfg(stage=1))
    joined = "\n".join(lines)
    self.assertIn("local-unqualified-pd-fallback", joined)
    self.assertIn("WARNING: unqualified controller fallback", joined)
    self.assertIn("WARNING: velocity command is uncalibrated", joined)
    self.assertIn("yaw_calibration_qualified=False", joined)
    # Stage1 commands zero yaw, so the missing yaw calibration must not add
    # a warning there; Stage2 activates the yaw residual head and must warn.
    self.assertNotIn("yaw feedforward is the zero fallback", joined)
    stage2_lines = hybrid_provenance_lines(
      make_hoppertrex_hybrid_env_cfg(stage=2)
    )
    stage2_joined = "\n".join(stage2_lines)
    self.assertIn("yaw feedforward is the zero fallback", stage2_joined)
    self.assertEqual(hybrid_provenance_lines(object()), [])

  def test_hybrid_provenance_lines_are_clean_for_qualified_artifacts(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      controller_payload = _controller_payload()
      controller_payload["state_construction"] = state_construction_spec(
        hybrid_task.DEFAULT_WHEEL_RADIUS
      )
      controller_path = _write_json(
        temp_dir,
        "controller.json",
        controller_payload,
      )
      calibration_path = _write_json(
        temp_dir,
        "calibration.json",
        calibration_artifact(
          controller_gain_hash=controller_payload["gain_hash"],
          scale=1.05,
          bias=0.001,
          seed=1,
          candidates=[],
        ),
      )
      lines = hybrid_provenance_lines(
        make_hoppertrex_hybrid_env_cfg(
          stage=1,
          controller_path=controller_path,
          calibration_path=calibration_path,
        )
      )
    joined = "\n".join(lines)
    self.assertIn("qualified=True", joined)
    self.assertNotIn("WARNING", joined)

  def test_posture_command_qualification_only_marks_active_posture_stage(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      controller_path = _write_json(
        temp_dir, 'controller.json', _controller_payload()
      )
      calibration_payload = _calibration_payload()
      calibration_path = _write_json(
        temp_dir, 'calibration.json', calibration_payload
      )
      posture_payload = _posture_payload()
      posture_payload['source_sweep']['calibration_hash'] = (
        calibration_payload['calibration_hash']
      )
      posture_path = _write_json(
        temp_dir, "posture.json", posture_payload
      )

      stage0 = make_hoppertrex_hybrid_env_cfg(
        stage=0,
        controller_path=controller_path,
        posture_map_path=posture_path,
        calibration_path=calibration_path,
      )
      stage3 = make_hoppertrex_hybrid_env_cfg(
        stage=3,
        controller_path=controller_path,
        posture_map_path=posture_path,
        calibration_path=calibration_path,
      )

    # Invariance hardening (Stage 3.0 resample fix): non-posture stages pin
    # the action term to the DEFAULT posture artifact so an exported map
    # env var can never change stage 0-2 leg targets; loading/validation
    # still ran (a bad binding would have raised above).
    self.assertFalse(
      stage0.actions["hybrid_wheel_leg"].posture_map_qualified
    )
    self.assertFalse(stage0.commands["posture"].qualified)
    self.assertEqual(
      stage0.commands["posture"].height_range,
      (stage0.commands["posture"].height_range[0],) * 2,
    )
    self.assertTrue(stage3.commands["posture"].qualified)
    self.assertEqual(stage3.commands["posture"].height_range, (0.32, 0.48))

  def test_station_calibration_loads_only_for_posture_stages(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      controller_payload = _controller_payload()
      controller_path = _write_json(
        temp_dir, "controller.json", controller_payload
      )
      calibration_payload = _calibration_payload()
      calibration_path = _write_json(
        temp_dir, "calibration.json", calibration_payload
      )
      posture_payload = _posture_payload()
      posture_payload["source_sweep"]["calibration_hash"] = (
        calibration_payload["calibration_hash"]
      )
      posture_path = _write_json(temp_dir, "posture.json", posture_payload)
      station_payload = station_calibration_artifact(
        controller_gain_hash=controller_payload["gain_hash"],
        posture_map_hash=posture_payload["map_hash"],
        breakpoints=[[-0.08, 0.0465], [0.0, -0.0136], [0.08, -0.0737]],
        source_probe={"git_sha": "test", "device": "cpu"},
      )
      station_path = _write_json(temp_dir, "station.json", station_payload)

      with patch.object(hybrid_task, "_STATION_CALIBRATION_WARNED", set()):
        stage3 = make_hoppertrex_hybrid_env_cfg(
          stage=3,
          controller_path=controller_path,
          posture_map_path=posture_path,
          calibration_path=calibration_path,
          station_calibration_path=station_path,
        )
      action = stage3.actions["hybrid_wheel_leg"]
      self.assertTrue(action.station_calibration_qualified)
      self.assertEqual(
        action.station_calibration_hash,
        station_payload["station_calibration_hash"],
      )
      self.assertEqual(
        action.station_drift_breakpoints,
        ((-0.08, 0.0465), (0.0, -0.0136), (0.08, -0.0737)),
      )
      self.assertIn(
        "station_calibration_qualified=True",
        "\n".join(hybrid_provenance_lines(stage3)),
      )

      # Stages without posture commands never load the artifact: a stage2
      # build with a nonexistent station path must not even try to read it,
      # keeping stages 0-2 byte-identical with or without the variable set.
      stage2 = make_hoppertrex_hybrid_env_cfg(
        stage=2,
        controller_path=controller_path,
        posture_map_path=posture_path,
        calibration_path=calibration_path,
        station_calibration_path=Path(temp_dir) / "missing.json",
      )
      stage2_action = stage2.actions["hybrid_wheel_leg"]
      self.assertFalse(stage2_action.station_calibration_qualified)
      self.assertEqual(
        stage2_action.station_drift_breakpoints,
        hybrid_task.STATION_DRIFT_FALLBACK_BREAKPOINTS,
      )

      # A station artifact bound to a different posture map must be rejected
      # instead of silently compensating with the wrong drift law.
      other_station = station_calibration_artifact(
        controller_gain_hash=controller_payload["gain_hash"],
        posture_map_hash="e" * 64,
        breakpoints=[[-0.08, 0.0465], [0.08, -0.0737]],
        source_probe={},
      )
      other_path = _write_json(temp_dir, "station_other.json", other_station)
      with patch.object(hybrid_task, "_STATION_CALIBRATION_WARNED", set()):
        with self.assertRaisesRegex(ValueError, "different posture map"):
          make_hoppertrex_hybrid_env_cfg(
            stage=3,
            controller_path=controller_path,
            posture_map_path=posture_path,
            calibration_path=calibration_path,
            station_calibration_path=other_path,
          )

  def test_stage3_without_station_artifact_warns_and_falls_back(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      controller_path = _write_json(
        temp_dir, "controller.json", _controller_payload()
      )
      calibration_payload = _calibration_payload()
      calibration_path = _write_json(
        temp_dir, "calibration.json", calibration_payload
      )
      posture_payload = _posture_payload()
      posture_payload["source_sweep"]["calibration_hash"] = (
        calibration_payload["calibration_hash"]
      )
      posture_path = _write_json(temp_dir, "posture.json", posture_payload)

      stdout = io.StringIO()
      with patch.object(hybrid_task, "_STATION_CALIBRATION_WARNED", set()):
        with contextlib.redirect_stdout(stdout):
          cfg = make_hoppertrex_hybrid_env_cfg(
            stage=3,
            controller_path=controller_path,
            posture_map_path=posture_path,
            calibration_path=calibration_path,
          )
      self.assertIn("no station calibration artifact", stdout.getvalue())
      action = cfg.actions["hybrid_wheel_leg"]
      self.assertFalse(action.station_calibration_qualified)
      self.assertIn(
        "station_calibration_qualified=False",
        "\n".join(hybrid_provenance_lines(cfg)),
      )

  def test_yaw_calibration_artifact_loads_into_stage2_action_cfg(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      controller_payload = _controller_payload()
      controller_path = _write_json(
        temp_dir, "controller.json", controller_payload
      )
      yaw_payload = yaw_calibration_artifact(
        controller_gain_hash=controller_payload["gain_hash"],
        breakpoints=[[-0.10, -0.55], [0.0, 0.0], [0.10, 0.55]],
        source_probe={"git_sha": "test", "device": "cpu"},
      )
      yaw_path = _write_json(temp_dir, "yaw.json", yaw_payload)

      cfg = make_hoppertrex_hybrid_env_cfg(
        stage=2,
        controller_path=controller_path,
        yaw_calibration_path=yaw_path,
      )
      action = cfg.actions["hybrid_wheel_leg"]
      self.assertTrue(action.yaw_calibration_qualified)
      self.assertEqual(
        action.yaw_calibration_hash,
        yaw_payload["yaw_calibration_hash"],
      )
      self.assertEqual(
        action.yaw_feedforward_breakpoints,
        ((-0.10, -0.55), (0.0, 0.0), (0.10, 0.55)),
      )

      # Without the matching controller artifact the binding must reject the
      # yaw calibration instead of silently pairing it with the PD fallback.
      with self.assertRaisesRegex(ValueError, "different controller"):
        make_hoppertrex_hybrid_env_cfg(
          stage=2,
          yaw_calibration_path=yaw_path,
        )

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
    # Stage5 campaign pushes span the pre-registered 8x eval kick band.
    self.assertEqual(push.params["velocity_range"]["x"], (-0.32, 0.32))
    self.assertEqual(push.params["velocity_range"]["pitch"], (-0.48, 0.48))

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
