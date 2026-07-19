import unittest

import torch

from mjlab.envs import ManagerBasedRlEnv

from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
  HybridWheelLegActionCfg,
  make_hoppertrex_hybrid_env_cfg,
)


def _one_env_cfg(stage: int):
  cfg = make_hoppertrex_hybrid_env_cfg(stage=stage)
  cfg.scene.num_envs = 1
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.num_envs = 1
  return cfg


ACTION_SCHEDULE = (
  (0.6, 0.2, 0.1, -0.1, 0.3, -0.3),
  (-0.4, -0.2, 0.0, 0.2, -0.1, 0.1),
  (0.2, 0.4, -0.2, 0.0, 0.1, 0.0),
  (0.0, -0.3, 0.3, -0.2, 0.0, 0.2),
)


def _run_trajectory(stage: int, mutate=None):
  torch.manual_seed(2026)
  cfg = _one_env_cfg(stage)
  if mutate is not None:
    mutate(cfg.actions["hybrid_wheel_leg"])
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset(seed=2026)
    robot = env.scene["robot"]
    steps = []
    for row in ACTION_SCHEDULE:
      env.step(torch.tensor([row], dtype=torch.float32))
      term = env.action_manager.get_term("hybrid_wheel_leg")
      steps.append(
        (
          term.wheel_targets.detach().clone(),
          term.leg_targets.detach().clone(),
          robot.data.joint_vel_target.detach().clone(),
          robot.data.joint_pos_target.detach().clone(),
        )
      )
    return steps
  finally:
    env.close()


class LatencyNoiseInjectionTest(unittest.TestCase):
  def test_defaults_leave_stage5_trajectories_bit_identical(self):
    """Frozen-evidence guard: default cfg keeps behavior byte-identical.

    The probe fields ship default-off; a Stage5 env built from the default
    cfg must match one where the fields are explicitly written with their
    defaults, element for element, across states and applied targets.
    """

    def pin_defaults(action: HybridWheelLegActionCfg) -> None:
      action.action_delay_steps = 0
      action.sensor_noise_pitch_std = 0.0
      action.sensor_noise_pitch_rate_std = 0.0
      action.sensor_noise_vx_std = 0.0
      action.sensor_noise_wheel_vel_std = 0.0
      action.sensor_noise_seed = 0

    base = _run_trajectory(5)
    pinned = _run_trajectory(5, pin_defaults)
    for (bw, bl, bvt, bpt), (pw, pl, pvt, ppt) in zip(base, pinned):
      torch.testing.assert_close(pw, bw, rtol=0.0, atol=0.0)
      torch.testing.assert_close(pl, bl, rtol=0.0, atol=0.0)
      torch.testing.assert_close(pvt, bvt, rtol=0.0, atol=0.0)
      torch.testing.assert_close(ppt, bpt, rtol=0.0, atol=0.0)

  def test_action_delay_shifts_applied_targets_by_k_steps(self):
    """delay=k applies the composition from k control steps earlier."""

    def delay_two(action: HybridWheelLegActionCfg) -> None:
      action.action_delay_steps = 2

    base = _run_trajectory(5)
    delayed = _run_trajectory(5, delay_two)

    # The composed (pre-delay) pipeline is identical: same commands, same
    # policy actions. Divergence only enters through the plant reacting to
    # delayed targets, so the first delayed step must apply the neutral
    # ring content rather than the fresh composition.
    first_wheels = delayed[0][2][:, [2, 5]]
    self.assertTrue(torch.all(first_wheels == 0.0))
    fresh_first_wheels = base[0][2][:, [2, 5]]
    self.assertFalse(torch.all(fresh_first_wheels == 0.0))
    # By step k+1 the ring starts replaying real compositions: the applied
    # wheel velocity target at step 2 equals the step-0 composition.
    torch.testing.assert_close(
      delayed[2][2][:, [2, 5]],
      delayed[0][0],
      rtol=0.0,
      atol=1.0e-6,
    )

  def test_delay_rejects_out_of_range_and_negative_noise(self):
    cfg = _one_env_cfg(5)
    action = cfg.actions["hybrid_wheel_leg"]
    with self.assertRaises(ValueError):
      type(action)(**{**action.__dict__, "action_delay_steps": 9})
    with self.assertRaises(ValueError):
      type(action)(**{**action.__dict__, "sensor_noise_pitch_std": -0.1})

  def test_sensor_noise_perturbs_baseline_deterministically(self):
    """Same seed reproduces the same corrupted baseline; noise != clean."""

    def noisy(action: HybridWheelLegActionCfg) -> None:
      action.sensor_noise_pitch_std = 0.01
      action.sensor_noise_pitch_rate_std = 0.05
      action.sensor_noise_vx_std = 0.01
      action.sensor_noise_wheel_vel_std = 0.05
      action.sensor_noise_seed = 7

    clean = _run_trajectory(5)
    noisy_a = _run_trajectory(5, noisy)
    noisy_b = _run_trajectory(5, noisy)

    self.assertFalse(
      torch.allclose(noisy_a[0][0], clean[0][0], rtol=0.0, atol=0.0)
    )
    for (aw, al, _avt, _apt), (bw, bl, _bvt, _bpt) in zip(noisy_a, noisy_b):
      torch.testing.assert_close(bw, aw, rtol=0.0, atol=0.0)
      torch.testing.assert_close(bl, al, rtol=0.0, atol=0.0)


if __name__ == "__main__":
  unittest.main()
