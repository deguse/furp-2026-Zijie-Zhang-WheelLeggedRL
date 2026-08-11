from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch

from hoppertrex_mjlab.hybrid.stair_classical import StairPhase
from hoppertrex_mjlab.hybrid.stair_dynamic import (
  DynamicLiftMode,
  DynamicStairManeuver,
  DynamicStairSensors,
  DynamicStairState,
  LeadSide,
  StairTraversalMode,
  dynamic_stair_step,
)
from hoppertrex_mjlab.hybrid.stair_dynamic_contract import (
  DYNAMIC_STAIR_ACTION_MASK,
  DYNAMIC_STAIR_ACTOR_TERMS,
  DYNAMIC_STAIR_CRITIC_TAIL_TERMS,
)
from hoppertrex_mjlab.tasks.agents import hoppertrex_stair_dynamic_ppo_runner_cfg
from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
  DYNAMIC_STAIR_LEFT_SENSOR_NAME,
  DYNAMIC_STAIR_RIGHT_SENSOR_NAME,
  HybridWheelLegAction,
  StairDynamicCurriculum,
  make_stair_dynamic_env_cfg,
  stair_dynamic_safe_control_inputs,
  stair_dynamic_target_saturation_mask,
  stair_dynamic_three_step_success_mask,
)


def _maneuver(**overrides) -> DynamicStairManeuver:
  base = DynamicStairManeuver(
    lift_mode=DynamicLiftMode.ALTERNATING,
    split_amplitude_rad=0.035,
    lift_amplitude_rad=0.050,
    trailing_delay_s=0.20,
    drive_feedforward_radps=1.0,
  )
  return replace(base, **overrides)


class _ActionManager:
  def __init__(self, action):
    self.action = action

  def get_term(self, name):
    if name != "hybrid_wheel_leg":
      raise KeyError(name)
    return self.action


class _CurriculumEnv:
  def __init__(self):
    self.num_envs = 8
    self.device = "cpu"
    self.common_step_counter = 0
    self.reset_buf = torch.zeros(8, dtype=torch.bool)
    self.action = SimpleNamespace(
      dynamic_step_index=torch.zeros(8, dtype=torch.long),
      dynamic_episode_unsafe=torch.zeros(8, dtype=torch.bool),
    )
    self.action_manager = _ActionManager(self.action)
    self.termination_manager = SimpleNamespace(
      terminated=torch.zeros(8, dtype=torch.bool)
    )
    terrain = SimpleNamespace(
      terrain_levels=torch.zeros(8, dtype=torch.long),
      terrain_types=torch.zeros(8, dtype=torch.long),
      terrain_origins=torch.zeros(4, 1, 3),
      env_origins=torch.zeros(8, 3),
    )
    self.scene = SimpleNamespace(terrain=terrain)


class StairDynamicConfigTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.cfg = make_stair_dynamic_env_cfg(play=False)
    cls.play_cfg = make_stair_dynamic_env_cfg(play=True)

  def test_registered_static_contract(self):
    cfg = self.cfg
    self.assertEqual(cfg.scene.num_envs, 256)
    self.assertEqual(tuple(cfg.observations["actor"].terms), DYNAMIC_STAIR_ACTOR_TERMS)
    self.assertEqual(
      tuple(cfg.observations["critic"].terms),
      DYNAMIC_STAIR_ACTOR_TERMS + DYNAMIC_STAIR_CRITIC_TAIL_TERMS,
    )
    action = cfg.actions["hybrid_wheel_leg"]
    self.assertEqual(tuple(action.action_mask), DYNAMIC_STAIR_ACTION_MASK)
    self.assertEqual(action.action_scales, (0.5, 0.3, 0.035, 0.035, 0.035, 0.035))
    self.assertEqual(cfg.commands["stair_request"].flat_env_count, 64)
    self.assertEqual(cfg.events["push_robot"].func.__name__, "push_flat_retention_envs")
    sensors = {sensor.name: sensor for sensor in cfg.scene.sensors}
    self.assertEqual(
      sensors[DYNAMIC_STAIR_LEFT_SENSOR_NAME].primary.pattern,
      "wheel_left_collision",
    )
    self.assertEqual(
      sensors[DYNAMIC_STAIR_RIGHT_SENSOR_NAME].primary.pattern,
      "wheel_right_collision",
    )

  def test_viser_cfg_is_one_stair_env(self):
    self.assertEqual(self.play_cfg.scene.num_envs, 1)
    self.assertEqual(self.play_cfg.commands["stair_request"].flat_env_count, 0)

  def test_runner_defaults_are_seed1_probe_budget(self):
    cfg = hoppertrex_stair_dynamic_ppo_runner_cfg()
    self.assertEqual(cfg.seed, 1)
    self.assertTrue(cfg.resume)
    self.assertEqual(cfg.num_steps_per_env, 24)
    self.assertEqual(cfg.max_iterations, 100)
    self.assertEqual(cfg.save_interval, 25)
    self.assertEqual(
      tuple(cfg.actor.distribution_cfg["active_mask"]),
      DYNAMIC_STAIR_ACTION_MASK,
    )


class StairDynamicFailClosedControlTest(unittest.TestCase):
  def test_abort_masks_command_and_all_six_ppo_heads(self):
    command = torch.tensor([[0.07, 0.0, 0.2], [0.07, 0.0, -0.2]])
    residual = torch.arange(12, dtype=torch.float).reshape(2, 6)
    abort = torch.tensor([False, True])
    safe_command, safe_residual = stair_dynamic_safe_control_inputs(
      command, residual, abort
    )
    torch.testing.assert_close(safe_command[0], command[0])
    torch.testing.assert_close(safe_residual[0], residual[0])
    self.assertEqual(float(safe_command[1, 0]), 0.0)
    self.assertEqual(float(safe_command[1, 2]), 0.0)
    self.assertEqual(int(torch.count_nonzero(safe_residual[1])), 0)

    unchanged_command, unchanged_residual = stair_dynamic_safe_control_inputs(
      command, residual, torch.zeros(2, dtype=torch.bool)
    )
    self.assertIs(unchanged_command, command)
    self.assertIs(unchanged_residual, residual)

  def test_composed_target_saturation_is_detected_only_for_dynamic_slots(self):
    desired = torch.tensor([[0.2, 0.0], [0.0, -0.2], [0.2, 0.0]])
    limits = torch.tensor([[[-0.1, 0.1], [-0.1, 0.1]]] * 3)
    mask = stair_dynamic_target_saturation_mask(
      desired, limits, torch.tensor([True, True, False])
    )
    self.assertEqual(mask.tolist(), [True, True, False])


class StairDynamicCurriculumTest(unittest.TestCase):
  def test_three_exact_windows_promote_and_flat_slots_stay_flat(self):
    env = _CurriculumEnv()
    curriculum = StairDynamicCurriculum(
      env,
      evaluation_interval_steps=2,
      flat_env_count=2,
    )
    ids = torch.arange(env.num_envs)
    curriculum.compute(env, ids)
    self.assertTrue(torch.equal(env.scene.terrain.terrain_levels[:2], torch.zeros(2, dtype=torch.long)))
    for boundary in (2, 4, 6):
      env.scene.terrain.terrain_levels[2:] = curriculum.upper_level
      env.action.dynamic_step_index[:] = 0
      env.action.dynamic_step_index[2:] = 3
      env.reset_buf[:] = False
      env.reset_buf[2:] = True
      env.common_step_counter = boundary
      curriculum.record_step(env)
    self.assertAlmostEqual(curriculum.state.upper_height_m, 0.02)
    self.assertEqual(curriculum.state.consecutive_ready_evaluations, 0)
    self.assertEqual(curriculum.evaluations, 3)

  def test_third_step_followed_by_unsafe_or_termination_is_not_success(self):
    env = _CurriculumEnv()
    curriculum = StairDynamicCurriculum(
      env, evaluation_interval_steps=2, flat_env_count=2
    )
    stair_ids = torch.arange(2, 8)
    env.scene.terrain.terrain_levels[stair_ids] = curriculum.upper_level
    env.action.dynamic_step_index[stair_ids] = 3
    env.action.dynamic_episode_unsafe[2] = True
    env.termination_manager.terminated[3] = True
    expected = stair_dynamic_three_step_success_mask(env)
    self.assertEqual(expected[2:].tolist(), [False, False, True, True, True, True])
    env.reset_buf[stair_ids] = True
    env.common_step_counter = 2
    curriculum.record_step(env)
    self.assertEqual(curriculum.completed_stair_episodes, 6)
    self.assertEqual(curriculum.successful_stair_episodes, 4)
    self.assertEqual(curriculum.successes_at_upper, 0)  # window consumed at step 2
    self.assertAlmostEqual(curriculum.state.upper_height_m, 0.01)
    self.assertEqual(curriculum.state.consecutive_ready_evaluations, 0)

  def test_state_round_trip_and_rejects_drift(self):
    env = _CurriculumEnv()
    source = StairDynamicCurriculum(env, 1200, flat_env_count=2)
    source.compute(env, torch.arange(8))
    source.completed_stair_episodes = 9
    source.successful_stair_episodes = 7
    payload = source.state_dict()
    target = StairDynamicCurriculum(env, 1200, flat_env_count=2)
    target.load_state_dict(payload)
    self.assertEqual(target.state_dict(), payload)
    malformed = dict(payload)
    malformed["evaluation_interval_steps"] = 1199
    with self.assertRaises(ValueError):
      target.load_state_dict(malformed)


class StairDynamicScalarTorchParityTest(unittest.TestCase):
  def _batched_action(self, maneuver: DynamicStairManeuver):
    env = SimpleNamespace(num_envs=1, device="cpu")
    env.command_manager = SimpleNamespace(
      get_command=lambda name: torch.ones(1, 1)
    )
    env.termination_manager = SimpleNamespace(
      get_term=lambda name: torch.zeros(1, dtype=torch.bool)
    )
    action = object.__new__(HybridWheelLegAction)
    action._env = env
    action._dynamic_enabled = True
    action._dynamic_maneuver = maneuver
    action._dynamic_request_command_name = "stair_request"
    action._dynamic_left_sensor_name = DYNAMIC_STAIR_LEFT_SENSOR_NAME
    action._dynamic_right_sensor_name = DYNAMIC_STAIR_RIGHT_SENSOR_NAME
    action._dynamic_dt = 0.02
    action._dynamic_stair_request = torch.zeros(1, dtype=torch.bool)
    action._dynamic_phase = torch.full((1,), int(StairPhase.IDLE), dtype=torch.long)
    action._dynamic_phase_elapsed = torch.zeros(1)
    action._dynamic_step_progress = torch.zeros(1)
    action._dynamic_step_index = torch.zeros(1, dtype=torch.long)
    action._dynamic_preferred_side = torch.tensor([int(LeadSide.LEFT)])
    action._dynamic_lead_side = torch.tensor([int(LeadSide.NONE)])
    action._dynamic_left_streak = torch.zeros(1, dtype=torch.long)
    action._dynamic_right_streak = torch.zeros(1, dtype=torch.long)
    action._dynamic_left_loaded = torch.zeros(1, dtype=torch.bool)
    action._dynamic_right_loaded = torch.zeros(1, dtype=torch.bool)
    action._dynamic_left_force = torch.zeros(1)
    action._dynamic_right_force = torch.zeros(1)
    action._dynamic_trail_contact_elapsed = torch.full((1,), -1.0)
    action._dynamic_recover_stable = torch.zeros(1, dtype=torch.long)
    action._dynamic_traversal_mode = torch.zeros(1, dtype=torch.long)
    action._dynamic_abort_code = torch.zeros(1, dtype=torch.long)
    action._dynamic_episode_unsafe = torch.zeros(1, dtype=torch.bool)
    action._dynamic_target_saturation = torch.zeros(1, dtype=torch.bool)
    action._dynamic_leg_feedforward = torch.zeros(1, 4)
    action._dynamic_drive_feedforward = torch.zeros(1)
    action._dynamic_riser_cross_event = torch.zeros(1, dtype=torch.bool)
    action._dynamic_previous_root_x = torch.zeros(1)
    action._dynamic_split_left = torch.tensor(maneuver.split_basis_left)
    action._dynamic_split_right = torch.tensor(maneuver.split_basis_right)
    action._dynamic_lift_left = torch.tensor(maneuver.lift_basis_left)
    action._dynamic_lift_right = torch.tensor(maneuver.lift_basis_right)
    action._dynamic_candidate_parameters = torch.tensor([[
      maneuver.split_amplitude_rad,
      maneuver.lift_amplitude_rad,
      maneuver.trailing_delay_s,
      maneuver.drive_feedforward_radps,
    ]])
    data = SimpleNamespace(
      root_link_pos_w=torch.zeros(1, 3),
      soft_joint_pos_limits=torch.tensor([[[-2.0, 2.0]] * 4]),
      joint_pos=torch.zeros(1, 4),
    )
    action._entity = SimpleNamespace(data=data)
    action._leg_ids = torch.arange(4)
    forces = {"left": torch.zeros(1), "right": torch.zeros(1)}
    action._dynamic_contact_metric = lambda name: (
      forces["left"] if name == DYNAMIC_STAIR_LEFT_SENSOR_NAME else forces["right"]
    )
    return action, forces

  def test_scalar_and_batched_fsm_match_step_by_step(self):
    maneuver = _maneuver()
    action, forces = self._batched_action(maneuver)
    state = DynamicStairState(preferred_side=LeadSide.LEFT)
    mode_code = {
      StairTraversalMode.NONE: 0,
      StairTraversalMode.ROLL: 1,
      StairTraversalMode.DYNAMIC: 2,
      StairTraversalMode.ABORT: 3,
    }
    root_x = 0.0
    for tick in range(140):
      delta = 0.01 if tick < 110 else 0.0
      root_x += delta
      action._entity.data.root_link_pos_w[0, 0] = root_x
      left = 20.0 if 35 <= tick <= 37 else 0.0
      right = 20.0 if 45 <= tick <= 47 else 0.0
      forces["left"][0] = left
      forces["right"][0] = right
      sensors = DynamicStairSensors(
        progress_delta_m=delta,
        left_force_n=left,
        right_force_n=right,
        stable=True,
      )
      target, state = dynamic_stair_step(
        maneuver,
        state,
        sensors,
        stair_request=True,
      )
      action._update_dynamic_stair(
        pitch=torch.zeros(1),
        pitch_rate=torch.zeros(1),
        projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
      )
      self.assertEqual(int(action.dynamic_phase[0]), int(state.phase), tick)
      self.assertEqual(int(action.dynamic_step_index[0]), state.step_index, tick)
      self.assertEqual(int(action.dynamic_lead_side[0]), int(state.lead_side), tick)
      self.assertEqual(
        int(action.dynamic_traversal_mode[0]), mode_code[state.traversal_mode], tick
      )
      torch.testing.assert_close(
        action.dynamic_leg_feedforward[0],
        torch.tensor(target.leg_feedforward),
        rtol=0.0,
        atol=2.0e-6,
        msg=lambda msg, tick=tick: f"tick={tick}: {msg}",
      )
      self.assertAlmostEqual(
        float(action.dynamic_drive_feedforward[0]),
        target.drive_feedforward_radps,
        places=6,
        msg=f"tick={tick}",
      )
    self.assertGreaterEqual(state.step_index, 1)


if __name__ == "__main__":
  unittest.main()
