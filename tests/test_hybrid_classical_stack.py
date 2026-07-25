import json
import time
import unittest

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv

from hoppertrex_mjlab.hybrid.classical_stack import (
  CONTROL_DT_S,
  DEFAULT_WHEEL_SLEW_LIMIT,
  DEFAULT_WHEEL_VELOCITY_LIMIT,
  POSTURE_HEIGHT_SLEW_RATE,
  POSTURE_PITCH_SLEW_RATE,
  ClassicalCommands,
  ClassicalSensors,
  ClassicalStackConfig,
  ClassicalStackState,
  classical_step,
  load_classical_stack_artifacts,
  reset_state,
  set_posture_target,
  shape_posture_command,
)
from hoppertrex_mjlab.hybrid.stair_classical import (
  ContactDetectorCfg,
  StairManeuver,
)
from hoppertrex_mjlab.tasks import hoppertrex_hybrid_task as hybrid_task
from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
  make_hoppertrex_hybrid_env_cfg,
)


def _one_env_cfg(stage: int):
  cfg = make_hoppertrex_hybrid_env_cfg(stage=stage)
  cfg.scene.num_envs = 1
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.num_envs = 1
  return cfg


def _config_from_env(env: ManagerBasedRlEnv) -> ClassicalStackConfig:
  term = env.action_manager.get_term("hybrid_wheel_leg")
  robot = env.scene["robot"]
  leg_ids, _names = robot.find_joints(
    term.cfg.leg_joint_names, preserve_order=True
  )
  soft = robot.data.soft_joint_pos_limits[0, leg_ids, :].detach().cpu().numpy()
  return ClassicalStackConfig(
    controller_gain=term.cfg.controller_gain,
    velocity_command_scale=term.cfg.velocity_command_scale,
    velocity_command_bias=term.cfg.velocity_command_bias,
    yaw_feedforward_breakpoints=term.cfg.yaw_feedforward_breakpoints,
    station_drift_breakpoints=term.cfg.station_drift_breakpoints,
    posture_coefficients=term.cfg.posture_coefficients,
    action_mask=term.cfg.action_mask,
    action_scales=term.cfg.action_scales,
    leg_position_lower=tuple(float(v) for v in soft[:, 0]),
    leg_position_upper=tuple(float(v) for v in soft[:, 1]),
    wheel_radius=term.cfg.wheel_radius,
    wheel_velocity_limit=term.cfg.wheel_velocity_limit,
    wheel_slew_limit=term.cfg.wheel_slew_limit,
  )


def _sensors_from_env(env: ManagerBasedRlEnv) -> ClassicalSensors:
  robot = env.scene["robot"]
  term = env.action_manager.get_term("hybrid_wheel_leg")
  data = robot.data
  gravity = data.projected_gravity_b[0]
  pitch = float(
    torch.atan2(
      gravity[0], torch.clamp(-gravity[2], min=1.0e-6)
    ).item()
  )
  wheel_vel = data.joint_vel[0, term._wheel_ids]
  return ClassicalSensors(
    pitch=pitch,
    pitch_rate=float(data.root_link_ang_vel_b[0, 1].item()),
    vx=float(data.root_link_lin_vel_b[0, 0].item()),
    wheel_vel_left=float(wheel_vel[0].item()),
    wheel_vel_right=float(wheel_vel[1].item()),
  )


def _commands_from_env(env: ManagerBasedRlEnv) -> ClassicalCommands:
  twist = env.command_manager.get_command("twist")[0]
  posture = env.command_manager.get_command("posture")[0]
  return ClassicalCommands(
    vx=float(twist[0].item()),
    wz=float(twist[2].item()),
    height=float(posture[0].item()),
    pitch=float(posture[1].item()),
  )


class ClassicalStackEquivalenceTest(unittest.TestCase):
  def test_stage5_pipeline_matches_runtime_to_float32_ulp(self):
    """Core acceptance: the portable stack replays the torch runtime.

    The env is stepped with a mixed residual schedule; before each step the
    exact sensor/command inputs the runtime will consume are captured, and
    the portable classical_step must reproduce the applied wheel and leg
    targets. Tolerance is one float32 ULP (atol 1e-6, rtol 0), not zero:
    compose_hybrid_targets — the audited reference contract — computes in
    float64 by design while the torch runtime is float32, so single-ULP
    rounding differences are irreducible without forking the contract.
    (The repo's runtime contract test uses 1e-5; this is tighter.)
    """

    env = ManagerBasedRlEnv(cfg=_one_env_cfg(stage=5), device="cpu")
    try:
      env.reset(seed=2026)
      term = env.action_manager.get_term("hybrid_wheel_leg")
      config = _config_from_env(env)
      state = ClassicalStackState()
      schedule = (
        (0.6, 0.2, 0.1, -0.1, 0.3, -0.3),
        (-1.5, 1.4, -0.6, 0.6, -1.0, 1.0),
        (0.2, 0.4, -0.2, 0.0, 0.1, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1.5, 1.5, 1.0, -1.0, 0.5, -0.5),
      )
      for row in schedule:
        sensors = _sensors_from_env(env)
        commands = _commands_from_env(env)
        actions = torch.tensor([row], dtype=torch.float32)
        env.step(actions)
        wheel_targets, leg_targets, state = classical_step(
          config,
          state,
          sensors,
          commands,
          residual_actions=np.asarray(row, dtype=np.float32),
        )
        np.testing.assert_allclose(
          term.wheel_targets.detach().cpu().numpy()[0],
          wheel_targets,
          rtol=0.0,
          atol=1.0e-6,
        )
        np.testing.assert_allclose(
          term.leg_targets.detach().cpu().numpy()[0],
          leg_targets,
          rtol=0.0,
          atol=1.0e-6,
        )
        # Thread the runtime's applied wheel targets into the portable
        # state so slew references stay aligned across the schedule.
        applied = term.wheel_targets.detach().cpu().numpy()[0]
        state = ClassicalStackState(
          previous_wheel_targets=(
            float(applied[0]),
            float(applied[1]),
          ),
          posture_command=state.posture_command,
          posture_target=state.posture_target,
        )
    finally:
      env.close()

  def test_posture_shaping_matches_runtime_command(self):
    """The portable slew shaper replays PostureCommand tick by tick."""

    env = ManagerBasedRlEnv(cfg=_one_env_cfg(stage=5), device="cpu")
    try:
      env.reset(seed=7)
      posture_term = env.command_manager.get_term("posture")
      start = posture_term.command[0].detach().cpu().numpy().copy()
      state = reset_state(float(start[0]), float(start[1]))
      target = (float(start[0]) + 0.02, float(start[1]) - 0.05)
      posture_term._target[0, 0] = target[0]
      posture_term._target[0, 1] = target[1]
      state = set_posture_target(state, *target)
      for _ in range(12):
        env.step(torch.zeros((1, 6), dtype=torch.float32))
        state = shape_posture_command(state, dt=CONTROL_DT_S)
        runtime = posture_term.command[0].detach().cpu().numpy()
        np.testing.assert_allclose(
          np.asarray(state.posture_command, dtype=np.float64),
          runtime.astype(np.float64),
          rtol=0.0,
          atol=1.0e-6,
        )
    finally:
      env.close()

  def test_runtime_constants_match_task_module(self):
    self.assertEqual(
      DEFAULT_WHEEL_VELOCITY_LIMIT,
      hybrid_task.DEFAULT_WHEEL_VELOCITY_LIMIT,
    )
    self.assertEqual(
      DEFAULT_WHEEL_SLEW_LIMIT, hybrid_task.DEFAULT_WHEEL_SLEW_LIMIT
    )
    self.assertEqual(
      POSTURE_HEIGHT_SLEW_RATE, hybrid_task.POSTURE_HEIGHT_SLEW_RATE
    )
    self.assertEqual(
      POSTURE_PITCH_SLEW_RATE, hybrid_task.POSTURE_PITCH_SLEW_RATE
    )


class ClassicalStackArtifactTest(unittest.TestCase):
  def test_loads_and_binds_the_test_fixture_artifacts(self):
    import tempfile
    from pathlib import Path

    from tests.test_hybrid_task_config import (
      _calibration_payload,
      _controller_payload,
      _posture_payload,
    )

    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      controller = root / "controller.json"
      controller.write_text(json.dumps(_controller_payload()))
      calibration = root / "calibration.json"
      calibration.write_text(json.dumps(_calibration_payload()))
      posture = root / "posture.json"
      posture.write_text(json.dumps(_posture_payload()))

      artifacts = load_classical_stack_artifacts(
        controller_path=controller,
        calibration_path=calibration,
        posture_map_path=posture,
      )

      self.assertTrue(artifacts.controller_qualified)
      self.assertEqual(
        artifacts.controller_gain_hash,
        _controller_payload()["gain_hash"],
      )
      self.assertEqual(artifacts.velocity_command_scale, 0.86)
      self.assertIsNone(artifacts.yaw_calibration_hash)
      self.assertIsNone(artifacts.station_calibration_hash)
      # Fallback maps are identically zero.
      self.assertEqual(
        artifacts.yaw_feedforward_breakpoints,
        ((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)),
      )

  def test_rejects_posture_map_from_a_different_controller(self):
    import tempfile
    from pathlib import Path

    from tests.test_hybrid_task_config import (
      _calibration_payload,
      _controller_payload,
      _posture_payload,
    )

    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      controller = root / "controller.json"
      controller.write_text(json.dumps(_controller_payload()))
      calibration = root / "calibration.json"
      calibration.write_text(json.dumps(_calibration_payload()))
      payload = _posture_payload()
      payload["source_sweep"]["controller_gain_hash"] = "f" * 64
      posture = root / "posture.json"
      posture.write_text(json.dumps(payload))

      with self.assertRaisesRegex(ValueError, "different controller"):
        load_classical_stack_artifacts(
          controller_path=controller,
          calibration_path=calibration,
          posture_map_path=posture,
        )

  def test_loads_registered_symmetric_posture_envelope(self):
    import tempfile
    from pathlib import Path

    from tests.test_hybrid_task_config import (
      _calibration_payload,
      _controller_payload,
      _posture_payload,
    )

    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      controller = root / 'controller.json'
      controller.write_text(json.dumps(_controller_payload()))
      calibration = root / 'calibration.json'
      calibration.write_text(json.dumps(_calibration_payload()))
      payload = _posture_payload()
      payload['envelope_verification']['method'] = (
        'registered_fixed_symmetric_hull_rectangle'
      )
      posture = root / 'posture.json'
      posture.write_text(json.dumps(payload))

      artifacts = load_classical_stack_artifacts(
        controller_path=controller,
        calibration_path=calibration,
        posture_map_path=posture,
      )

    self.assertEqual(
      artifacts.posture_map_hash,
      _posture_payload()['map_hash'],
    )
    self.assertEqual(artifacts.posture_height_range, (0.32, 0.48))
    self.assertEqual(artifacts.posture_pitch_range, (-0.08, 0.08))


class ClassicalStackBudgetTest(unittest.TestCase):
  def test_stair_maneuver_posture_is_slew_limited(self):
    maneuver = StairManeuver(
      approach_vx=0.08,
      preload_trigger_m=0.01,
      preload_duration_s=0.02,
      preload_height_m=0.29,
      preload_pitch_rad=-0.032,
      contact_vx=0.06,
      climb_vx=0.08,
      drive_feedforward_radps=1.0,
      climb_height_m=0.327,
      climb_pitch_rad=0.032,
      climb_timeout_s=1.0,
      crest_progress_m=0.40,
      recover_duration_s=0.5,
      detector=ContactDetectorCfg(0.1, 0.2, 1.0),
      maneuver_hash="a" * 64,
      bindings={"controller_schedule_hash": "b" * 64},
    )
    config = ClassicalStackConfig(
      controller_gain=(8.0, 1.0, 3.0, 0.2),
      velocity_command_scale=1.0,
      velocity_command_bias=0.0,
      yaw_feedforward_breakpoints=((-1.0, 0.0), (1.0, 0.0)),
      station_drift_breakpoints=((-1.0, 0.0), (1.0, 0.0)),
      posture_coefficients=((0.0,) * 4, (1.0,) * 4, (1.0,) * 4),
      action_mask=(False,) * 6,
      action_scales=(0.5, 0.3, 0.035, 0.035, 0.035, 0.035),
      leg_position_lower=(-2.0,) * 4,
      leg_position_upper=(2.0,) * 4,
      stair_maneuver=maneuver,
    )
    state = reset_state(0.31, 0.0)
    sensors = ClassicalSensors(0.0, 0.0, 0.0, 0.0, 0.0)
    _wheels, _legs, state = classical_step(
      config,
      state,
      sensors,
      ClassicalCommands(height=0.31, pitch=0.0, stair_mode=True),
    )
    for _ in range(8):
      previous = state.posture_command
      _wheels, _legs, state = classical_step(
        config,
        state,
        sensors,
        ClassicalCommands(height=0.31, pitch=0.0, stair_mode=True),
      )
      self.assertLessEqual(
        abs(state.posture_command[0] - previous[0]),
        POSTURE_HEIGHT_SLEW_RATE * CONTROL_DT_S + 1.0e-12,
      )
      self.assertLessEqual(
        abs(state.posture_command[1] - previous[1]),
        POSTURE_PITCH_SLEW_RATE * CONTROL_DT_S + 1.0e-12,
      )

  def test_single_step_fits_the_50hz_budget_on_cpu(self):
    """Deployment budget evidence: one tick must cost well under 20 ms."""

    config = ClassicalStackConfig(
      controller_gain=(8.0, 1.0, 3.0, 0.2),
      velocity_command_scale=0.86,
      velocity_command_bias=-0.012,
      yaw_feedforward_breakpoints=((-1.0, -0.6), (0.0, 0.0), (1.0, 0.6)),
      station_drift_breakpoints=((-0.08, 0.05), (0.08, -0.05)),
      posture_coefficients=(
        (-0.2, 0.2, -0.4, 0.4),
        (-1.0, 1.0, -0.8, 0.8),
        (0.5, 0.5, -0.3, -0.3),
      ),
      action_mask=(True,) * 6,
      action_scales=(0.5, 0.3, 0.035, 0.035, 0.035, 0.035),
      leg_position_lower=(-1.0, -1.0, -1.5, -1.5),
      leg_position_upper=(1.0, 1.0, 1.5, 1.5),
    )
    state = ClassicalStackState()
    sensors = ClassicalSensors(
      pitch=0.01,
      pitch_rate=-0.05,
      vx=0.03,
      wheel_vel_left=0.4,
      wheel_vel_right=0.6,
    )
    commands = ClassicalCommands(vx=0.05, wz=0.1, height=0.31, pitch=0.0)
    residual = np.zeros(6, dtype=np.float32)
    # Warm up, then measure.
    for _ in range(50):
      _w, _l, state = classical_step(
        config, state, sensors, commands, residual
      )
    started = time.perf_counter()
    ticks = 500
    for _ in range(ticks):
      _w, _l, state = classical_step(
        config, state, sensors, commands, residual
      )
    per_tick_ms = 1000.0 * (time.perf_counter() - started) / ticks
    print(f"[budget] classical_step: {per_tick_ms:.3f} ms/tick")
    self.assertLess(per_tick_ms, 2.0)


if __name__ == "__main__":
  unittest.main()
