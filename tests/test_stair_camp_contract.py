"""Contract tests for the residual stair camp (mainline doc S5B).

Every assertion here pins a value or a mechanism that S5B preregisters, so a
silent drift between the registration and the implementation fails the suite
rather than reaching a training run.
"""

from dataclasses import asdict
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import mock
import os
import unittest

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from assets.HopperTrex_CFG import INIT_JOINT_POS
from hoppertrex_mjlab.hybrid.config import (
  DEFAULT_ACTION_SCALES,
  HYBRID_STAGES,
  LEG_RESIDUAL_INDICES,
  STAIR_CAMP_ACTION_MASK,
  STAIR_CAMP_LEG_RESIDUAL_SCALE,
  STAIR_CAMP_STAGE,
  STAIR_CAMP_TASK_ID,
  action_scales_with_leg_authority,
)
from hoppertrex_mjlab.hybrid.posture import LEG_JOINT_NAMES
from hoppertrex_mjlab.hybrid.runner import HybridOnPolicyRunner
from hoppertrex_mjlab.hybrid.stair_classical import PHASE_COUNT, StairPhase
from hoppertrex_mjlab.hybrid.stair_residual import ACTOR_FIELD_NAMES
from hoppertrex_mjlab.hybrid.stair_trigger import (
  STAIR_TRIGGER_FORCE_N,
  STAIR_TRIGGER_SENSOR_FIELDS,
  STAIR_TRIGGER_SENSOR_NAME,
  STAIR_TRIGGER_SLOTS_PER_WHEEL,
  STAIR_TRIGGER_WINDOW,
  stair_trigger_metric,
  update_stair_trigger,
)
from hoppertrex_mjlab.scripts.rsl_rl.train import (
  _hybrid_stage,
  validate_hybrid_repository_status,
  validate_hybrid_training_artifacts,
)
from hoppertrex_mjlab.tasks.agents import hoppertrex_stair_camp_ppo_runner_cfg
import hoppertrex_mjlab.tasks.hoppertrex_hybrid_task as hybrid_task
from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
  ROOT_HEIGHT_TARGET,
  STAIR_CAMP_EVALUATION_INTERVAL_ITERS,
  STAIR_CAMP_STEPS_PER_ITERATION,
  StairCampCurriculum,
  STAIR_CAMP_ACTOR_TAIL_TERMS,
  STAIR_CAMP_ACTOR_TERM_ORDER,
  STAIR_CAMP_PROPRIOCEPTION_TERMS,
  STAIR_CAMP_RISER_OFFSET_M,
  STAIR_CAMP_START_OFFSET_M,
  make_hoppertrex_hybrid_env_cfg,
  make_stair_camp_env_cfg,
  stair_camp_actor_terms,
  stair_camp_distance_to_riser_observation,
  stair_camp_step_height_observation,
  stair_mode_forward_progress,
  stair_phase_one_hot_observation,
  stair_trigger_sensor_cfg,
)

# Settle at zero command before driving, as the C0/C1/C2 probes do: settling
# at the command velocity would strike the riser inside the settle window.
_SETTLE_STEPS = 200
_APPROACH_VX_MPS = 0.07


def _env_cfg(
  *,
  controller: bool = True,
  calibration: bool = True,
  yaw: bool = True,
  posture: bool = True,
  station: bool = True,
):
  return SimpleNamespace(
    actions={
      'hybrid_wheel_leg': SimpleNamespace(
        controller_qualified=controller,
        controller_gain_hash='controller123',
        calibration_hash=('calibration123' if calibration else None),
        yaw_calibration_qualified=yaw,
        posture_map_qualified=posture,
        station_calibration_qualified=station,
      )
    },
  )


# A non-degenerate posture map: the shipped fallback artifact has zero height
# and pitch rows, so the classical leg reference would be constant and every
# freeze assertion below would pass vacuously. These coefficients make the
# posture map respond to the command without pushing any target into the soft
# joint limits, which would clamp away the effect being measured.
_LIVE_POSTURE_COEFFICIENTS = (
  tuple(float(INIT_JOINT_POS[name]) for name in LEG_JOINT_NAMES),
  (0.10, 0.10, 0.10, 0.10),
  (0.0, 0.0, 0.0, 0.0),
)
_POSTURE_HEIGHT_STEP = 0.10
_ACTION_ROW = (0.6, 0.2, 0.4, -0.4, 0.5, -0.5)


def _stage5_cfg(*, attach_trigger_sensor: bool = False, **camp_fields):
  cfg = make_hoppertrex_hybrid_env_cfg(stage=5)
  cfg.scene.num_envs = 1
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.num_envs = 1
  if attach_trigger_sensor:
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (stair_trigger_sensor_cfg(),)
  action_cfg = cfg.actions['hybrid_wheel_leg']
  action_cfg.posture_coefficients = _LIVE_POSTURE_COEFFICIENTS
  for name, value in camp_fields.items():
    setattr(action_cfg, name, value)
  return cfg


def _run_camp_trajectory(
  steps: int = 3, *, attach_trigger_sensor: bool = False, **camp_fields
):
  """Step a one-env Stage5 stack and record the leg composition per step."""

  torch.manual_seed(2026)
  env = ManagerBasedRlEnv(
    cfg=_stage5_cfg(attach_trigger_sensor=attach_trigger_sensor, **camp_fields),
    device='cpu',
  )
  try:
    env.reset(seed=2026)
    term = env.action_manager.get_term('hybrid_wheel_leg')
    # Drive the classical posture reference away from its reset value so a
    # frozen reference is distinguishable from a live one.
    posture = env.command_manager.get_term('posture')
    posture.target[:, 0] += _POSTURE_HEIGHT_STEP
    rows = []
    for _ in range(steps):
      env.step(torch.tensor([_ACTION_ROW], dtype=torch.float32))
      row = {
        'stair_mode': term.stair_mode.clone(),
        'leg_reference': term.leg_reference.detach().clone(),
        'nominal_leg_targets': term.nominal_leg_targets.detach().clone(),
        'leg_targets': term.leg_targets.detach().clone(),
        'applied_residual': term.applied_residual.detach().clone(),
      }
      if attach_trigger_sensor:
        data = env.scene.sensors[STAIR_TRIGGER_SENSOR_NAME].data
        row['contacts_found'] = int((data.found > 0).sum())
        row['peak_normal_force'] = float(
          torch.where(
            data.found > 0,
            data.force[..., 0].abs(),
            torch.zeros_like(data.force[..., 0]),
          ).max()
        )
        row['trigger_metric'] = stair_trigger_metric(
          found=data.found,
          force_contact_frame=data.force,
          normal_global=data.normal,
        ).clone()
      rows.append(row)
    return rows
  finally:
    env.close()


class StairCampFrozenValuesTest(unittest.TestCase):
  """S5B Protocol 1: masks and action scales."""

  def test_task_id_matches_registration(self) -> None:
    self.assertEqual(STAIR_CAMP_TASK_ID, 'HopperTrex-Hybrid-v2-StairCamp')

  def test_action_mask_is_legs_only(self) -> None:
    self.assertEqual(
      STAIR_CAMP_ACTION_MASK, (False, False, True, True, True, True)
    )

  def test_stage_action_mask_matches_the_frozen_tuple(self) -> None:
    # Surface 1 (runtime gate) and surface 2 (PPO statistics) must be the
    # same tuple; the task registration feeds this one value to both.
    self.assertEqual(STAIR_CAMP_STAGE.action_mask, STAIR_CAMP_ACTION_MASK)

  def test_leg_residual_scale_is_the_confirmed_value(self) -> None:
    self.assertEqual(STAIR_CAMP_LEG_RESIDUAL_SCALE, 0.070)

  def test_leg_entries_carry_the_confirmed_authority(self) -> None:
    scales = STAIR_CAMP_STAGE.action_scales
    for index in LEG_RESIDUAL_INDICES:
      self.assertEqual(scales[index], 0.070)

  def test_wheel_entries_stay_at_the_frozen_defaults(self) -> None:
    # Irrelevant at runtime under the mask, but pinned for hash stability.
    scales = STAIR_CAMP_STAGE.action_scales
    self.assertEqual(scales[0], DEFAULT_ACTION_SCALES[0])
    self.assertEqual(scales[1], DEFAULT_ACTION_SCALES[1])

  def test_scales_come_from_the_registered_helper(self) -> None:
    self.assertEqual(
      STAIR_CAMP_STAGE.action_scales,
      action_scales_with_leg_authority(0.070),
    )

  def test_default_scales_are_untouched_by_the_camp(self) -> None:
    # The camp must not mutate the frozen six-stage ladder's scales.
    self.assertEqual(
      DEFAULT_ACTION_SCALES, (0.5, 0.3, 0.035, 0.035, 0.035, 0.035)
    )

  def test_camp_inherits_the_stage5_envelope(self) -> None:
    stage5 = HYBRID_STAGES[5]
    self.assertEqual(STAIR_CAMP_STAGE.randomization_level, 2)
    self.assertEqual(
      STAIR_CAMP_STAGE.lin_vel_x_range, stage5.lin_vel_x_range
    )
    self.assertEqual(STAIR_CAMP_STAGE.yaw_rate_range, stage5.yaw_rate_range)
    self.assertEqual(
      STAIR_CAMP_STAGE.posture_commands, stage5.posture_commands
    )
    self.assertEqual(STAIR_CAMP_STAGE.gate_suite, stage5.gate_suite)

  def test_camp_is_not_part_of_the_frozen_stage_ladder(self) -> None:
    # HYBRID_STAGES drives the six frozen task ids and their hashes.
    self.assertEqual(sorted(HYBRID_STAGES), [0, 1, 2, 3, 4, 5])
    for stage in HYBRID_STAGES.values():
      self.assertEqual(stage.action_scales, DEFAULT_ACTION_SCALES)


class StairCampTrainGuardTest(unittest.TestCase):
  """S5B Protocol 1 guard-coverage obligation (final audit A1-iv/B-v).

  The camp's task id does not match HYBRID_TASK_PREFIX, so before this
  obligation was implemented every artifact guard and the clean-worktree
  refusal were silently skipped for it.
  """

  def test_camp_task_resolves_to_the_strictest_stage(self) -> None:
    self.assertEqual(_hybrid_stage(STAIR_CAMP_TASK_ID), 5)

  def test_prefix_tasks_still_resolve(self) -> None:
    self.assertEqual(_hybrid_stage('HopperTrex-Hybrid-v2-Stage3'), 3)

  def test_unrelated_tasks_stay_unguarded(self) -> None:
    self.assertIsNone(_hybrid_stage('Mjlab-HopperTrex-Balance-v0'))

  def test_camp_accepts_a_fully_qualified_environment(self) -> None:
    validate_hybrid_training_artifacts(STAIR_CAMP_TASK_ID, _env_cfg())

  def test_camp_rejects_unqualified_controller(self) -> None:
    with self.assertRaises(ValueError):
      validate_hybrid_training_artifacts(
        STAIR_CAMP_TASK_ID, _env_cfg(controller=False)
      )

  def test_camp_rejects_missing_velocity_calibration(self) -> None:
    with self.assertRaises(ValueError):
      validate_hybrid_training_artifacts(
        STAIR_CAMP_TASK_ID, _env_cfg(calibration=False)
      )

  def test_camp_rejects_unqualified_yaw_calibration(self) -> None:
    # The legs-only residual mask does not release the classical yaw
    # feedforward, which still consumes the frozen artifact.
    with self.assertRaises(ValueError):
      validate_hybrid_training_artifacts(
        STAIR_CAMP_TASK_ID, _env_cfg(yaw=False)
      )

  def test_camp_rejects_unqualified_posture_map(self) -> None:
    with self.assertRaises(ValueError):
      validate_hybrid_training_artifacts(
        STAIR_CAMP_TASK_ID, _env_cfg(posture=False)
      )

  def test_camp_rejects_unqualified_station_calibration(self) -> None:
    with self.assertRaises(ValueError):
      validate_hybrid_training_artifacts(
        STAIR_CAMP_TASK_ID, _env_cfg(station=False)
      )

  def test_camp_rejects_a_dirty_worktree(self) -> None:
    with self.assertRaises(ValueError):
      validate_hybrid_repository_status(
        STAIR_CAMP_TASK_ID, ' M src/hoppertrex_mjlab/hybrid/config.py\n'
      )

  def test_camp_accepts_a_clean_worktree(self) -> None:
    validate_hybrid_repository_status(STAIR_CAMP_TASK_ID, '')


class StairTriggerRuleTest(unittest.TestCase):
  """S5B Protocol 2: the CTBC-style 18 N x 3-frame contact trigger."""

  def test_frozen_threshold_and_window(self) -> None:
    self.assertEqual(STAIR_TRIGGER_FORCE_N, 18.0)
    self.assertEqual(STAIR_TRIGGER_WINDOW, 3)

  def test_metric_is_the_normal_force_projected_on_global_x(self) -> None:
    # The likeliest implementation slip is using |F0| (or the force norm)
    # instead of |F0 * nx|. A 20 N normal force on a nearly horizontal
    # contact normal is a floor contact, not a riser strike, and must stay
    # far below the 18 N threshold.
    found = torch.ones(1, 1)
    force = torch.zeros(1, 1, 3)
    normal = torch.zeros(1, 1, 3)
    force[0, 0, 0] = 20.0
    normal[0, 0, 0] = 0.5
    metric = stair_trigger_metric(
      found=found, force_contact_frame=force, normal_global=normal
    )
    self.assertAlmostEqual(float(metric[0]), 10.0, places=5)
    self.assertLess(float(metric[0]), STAIR_TRIGGER_FORCE_N)

  def test_metric_ignores_slots_without_contact(self) -> None:
    found = torch.tensor([[0.0, 1.0]])
    force = torch.zeros(1, 2, 3)
    normal = torch.zeros(1, 2, 3)
    force[0, 0, 0] = 99.0  # reported by a slot that found no contact
    normal[0, 0, 0] = 1.0
    force[0, 1, 0] = 25.0
    normal[0, 1, 0] = 1.0
    metric = stair_trigger_metric(
      found=found, force_contact_frame=force, normal_global=normal
    )
    self.assertAlmostEqual(float(metric[0]), 25.0, places=5)

  def test_metric_takes_the_maximum_over_slots(self) -> None:
    found = torch.ones(1, 3)
    force = torch.zeros(1, 3, 3)
    normal = torch.zeros(1, 3, 3)
    force[0, :, 0] = torch.tensor([5.0, -30.0, 12.0])
    normal[0, :, 0] = 1.0
    metric = stair_trigger_metric(
      found=found, force_contact_frame=force, normal_global=normal
    )
    self.assertAlmostEqual(float(metric[0]), 30.0, places=5)

  def test_metric_rejects_mismatched_shapes(self) -> None:
    found = torch.ones(1, 2)
    with self.assertRaises(ValueError):
      stair_trigger_metric(
        found=found,
        force_contact_frame=torch.zeros(1, 3, 3),
        normal_global=torch.zeros(1, 2, 3),
      )

  def test_three_consecutive_samples_fire(self) -> None:
    latched = torch.zeros(1, dtype=torch.bool)
    streak = torch.zeros(1, dtype=torch.long)
    fired_at = None
    for step in range(3):
      latched, streak = update_stair_trigger(
        latched=latched, streak=streak, metric=torch.tensor([18.0])
      )
      if bool(latched[0]) and fired_at is None:
        fired_at = step
    self.assertEqual(fired_at, 2)

  def test_two_consecutive_samples_do_not_fire(self) -> None:
    latched = torch.zeros(1, dtype=torch.bool)
    streak = torch.zeros(1, dtype=torch.long)
    for metric in (25.0, 25.0, 17.9, 25.0, 25.0):
      latched, streak = update_stair_trigger(
        latched=latched, streak=streak, metric=torch.tensor([metric])
      )
      self.assertFalse(bool(latched[0]))

  def test_threshold_is_inclusive_and_below_threshold_resets(self) -> None:
    latched = torch.zeros(2, dtype=torch.bool)
    streak = torch.zeros(2, dtype=torch.long)
    # Env 0 sits exactly at the threshold; env 1 sits just below it.
    for _ in range(3):
      latched, streak = update_stair_trigger(
        latched=latched, streak=streak, metric=torch.tensor([18.0, 17.999])
      )
    self.assertTrue(bool(latched[0]))
    self.assertFalse(bool(latched[1]))
    self.assertEqual(int(streak[1]), 0)

  def test_latch_survives_the_force_dropping_away(self) -> None:
    latched = torch.zeros(1, dtype=torch.bool)
    streak = torch.zeros(1, dtype=torch.long)
    for _ in range(3):
      latched, streak = update_stair_trigger(
        latched=latched, streak=streak, metric=torch.tensor([50.0])
      )
    for _ in range(20):
      latched, streak = update_stair_trigger(
        latched=latched, streak=streak, metric=torch.tensor([0.0])
      )
    # No mid-episode exit path exists: the latch resets only at env reset.
    self.assertTrue(bool(latched[0]))
    self.assertEqual(int(streak[0]), 0)

  def test_streak_saturates_at_the_window(self) -> None:
    latched = torch.zeros(1, dtype=torch.bool)
    streak = torch.zeros(1, dtype=torch.long)
    for _ in range(50):
      latched, streak = update_stair_trigger(
        latched=latched, streak=streak, metric=torch.tensor([50.0])
      )
    self.assertEqual(int(streak[0]), STAIR_TRIGGER_WINDOW)

  def test_window_and_threshold_are_configurable(self) -> None:
    latched = torch.zeros(1, dtype=torch.bool)
    streak = torch.zeros(1, dtype=torch.long)
    latched, streak = update_stair_trigger(
      latched=latched,
      streak=streak,
      metric=torch.tensor([5.0]),
      threshold=4.0,
      window=1,
    )
    self.assertTrue(bool(latched[0]))

  def test_zero_window_is_rejected(self) -> None:
    with self.assertRaises(ValueError):
      update_stair_trigger(
        latched=torch.zeros(1, dtype=torch.bool),
        streak=torch.zeros(1, dtype=torch.long),
        metric=torch.tensor([0.0]),
        window=0,
      )

  def test_envs_latch_independently(self) -> None:
    latched = torch.zeros(3, dtype=torch.bool)
    streak = torch.zeros(3, dtype=torch.long)
    # Env 0 climbs, env 1 is interrupted on the second sample, env 2 is idle.
    schedule = (
      (30.0, 30.0, 0.0),
      (30.0, 0.0, 0.0),
      (30.0, 30.0, 0.0),
    )
    for row in schedule:
      latched, streak = update_stair_trigger(
        latched=latched, streak=streak, metric=torch.tensor(row)
      )
    self.assertEqual(latched.tolist(), [True, False, False])


class StairCampActionCfgTest(unittest.TestCase):
  """Camp fields must be inert by default and self-consistent when set."""

  def _cfg_kwargs(self, **overrides):
    kwargs = {
      'entity_name': 'robot',
      'action_mask': STAIR_CAMP_ACTION_MASK,
      'action_scales': STAIR_CAMP_STAGE.action_scales,
      'controller_gain': (8.0, 1.0, 3.0, 0.2),
      'posture_coefficients': _LIVE_POSTURE_COEFFICIENTS,
    }
    kwargs.update(overrides)
    return kwargs

  def test_camp_fields_default_to_inert(self) -> None:
    from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
      HybridWheelLegActionCfg,
    )

    cfg = HybridWheelLegActionCfg(**self._cfg_kwargs())
    self.assertIsNone(cfg.stair_trigger_sensor_name)
    self.assertFalse(cfg.stair_mode_forced)
    self.assertFalse(cfg.stair_mode_freezes_leg_reference)
    self.assertEqual(cfg.stair_trigger_force_n, STAIR_TRIGGER_FORCE_N)
    self.assertEqual(cfg.stair_trigger_window, STAIR_TRIGGER_WINDOW)

  def test_forced_mode_and_a_trigger_sensor_are_mutually_exclusive(
    self,
  ) -> None:
    from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
      HybridWheelLegActionCfg,
    )

    with self.assertRaises(ValueError):
      HybridWheelLegActionCfg(
        **self._cfg_kwargs(
          stair_mode_forced=True,
          stair_trigger_sensor_name='some_sensor',
        )
      )

  def test_non_positive_threshold_and_empty_window_are_rejected(self) -> None:
    from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
      HybridWheelLegActionCfg,
    )

    with self.assertRaises(ValueError):
      HybridWheelLegActionCfg(**self._cfg_kwargs(stair_trigger_force_n=0.0))
    with self.assertRaises(ValueError):
      HybridWheelLegActionCfg(**self._cfg_kwargs(stair_trigger_window=0))


class StairCampLegReleaseTest(unittest.TestCase):
  """S5B deviation minute 3: freeze-at-trigger leg-release semantics.

  This is the load-bearing mode effect that decision-tree branch 2 hinges
  on, so it ships with its own contract test against the real action term
  running inside a one-env Stage5 stack on CPU.
  """

  def test_rising_edge_freezes_the_latest_same_step_nominal_target(self) -> None:
    torch.manual_seed(2026)
    cfg = _stage5_cfg(stair_mode_freezes_leg_reference=True)
    env = ManagerBasedRlEnv(cfg=cfg, device='cpu')
    try:
      env.reset(seed=2026)
      term = env.action_manager.get_term('hybrid_wheel_leg')
      posture = env.command_manager.get_term('posture')
      actions = torch.zeros(1, 6)

      posture.command[:, 0] = ROOT_HEIGHT_TARGET + 0.01
      posture.command[:, 1] = 0.02

      def latch_now() -> None:
        term._stair_mode[:] = True

      with mock.patch.object(term, '_update_stair_mode', side_effect=latch_now):
        env.action_manager.process_action(actions)
      latest_nominal = term.nominal_leg_targets.clone()
      torch.testing.assert_close(term.leg_reference, latest_nominal)

      posture.command[:, 0] = ROOT_HEIGHT_TARGET - 0.01
      posture.command[:, 1] = -0.02
      env.action_manager.process_action(actions)
      self.assertFalse(torch.allclose(term.nominal_leg_targets, latest_nominal))
      torch.testing.assert_close(term.leg_reference, latest_nominal)
    finally:
      env.close()
  def test_classical_reference_is_live_while_stair_mode_is_false(
    self,
  ) -> None:
    rows = _run_camp_trajectory(stair_mode_freezes_leg_reference=True)
    for row in rows:
      self.assertFalse(bool(row['stair_mode'][0]))
      torch.testing.assert_close(
        row['leg_reference'], row['nominal_leg_targets']
      )
    # Non-vacuity: the posture map must actually be moving, otherwise a
    # frozen reference would be indistinguishable from a live one.
    self.assertFalse(
      torch.allclose(
        rows[0]['nominal_leg_targets'], rows[-1]['nominal_leg_targets']
      )
    )

  def test_classical_reference_is_held_once_stair_mode_latches(self) -> None:
    rows = _run_camp_trajectory(
      stair_mode_forced=True, stair_mode_freezes_leg_reference=True
    )
    initial = torch.tensor(
      [[float(INIT_JOINT_POS[name]) for name in LEG_JOINT_NAMES]]
    )
    for row in rows:
      self.assertTrue(bool(row['stair_mode'][0]))
      # Latched from t=0 under the trigger-off ablation, so the reference is
      # the reset pose - the pose the physics basis table perturbs about.
      torch.testing.assert_close(row['leg_reference'], initial)
    # The posture map keeps publishing (it still feeds the wheel station
    # feedforward and the observation), it simply no longer drives the legs.
    self.assertFalse(
      torch.allclose(
        rows[-1]['nominal_leg_targets'], rows[-1]['leg_reference']
      )
    )

  def test_leg_target_is_the_reference_plus_the_leg_residual(self) -> None:
    for fields in (
      {'stair_mode_freezes_leg_reference': True},
      {
        'stair_mode_forced': True,
        'stair_mode_freezes_leg_reference': True,
      },
    ):
      with self.subTest(**fields):
        rows = _run_camp_trajectory(**fields)
        for row in rows:
          torch.testing.assert_close(
            row['leg_targets'],
            row['leg_reference'] + row['applied_residual'][:, 2:],
          )

  def test_frozen_stages_keep_the_posture_map_authoritative(self) -> None:
    # With every camp field at its default the leg reference must track the
    # posture map step for step, which is what keeps Stage0-5 unchanged.
    rows = _run_camp_trajectory()
    for row in rows:
      self.assertFalse(bool(row['stair_mode'][0]))
      torch.testing.assert_close(
        row['leg_reference'], row['nominal_leg_targets']
      )
      torch.testing.assert_close(
        row['leg_targets'],
        row['nominal_leg_targets'] + row['applied_residual'][:, 2:],
      )

  def test_camp_defaults_leave_stage5_trajectories_bit_identical(self) -> None:
    baseline = _run_camp_trajectory()
    explicit = _run_camp_trajectory(
      stair_trigger_sensor_name=None,
      stair_trigger_force_n=STAIR_TRIGGER_FORCE_N,
      stair_trigger_window=STAIR_TRIGGER_WINDOW,
      stair_mode_freezes_leg_reference=False,
      stair_mode_forced=False,
    )
    for left, right in zip(baseline, explicit, strict=True):
      for key in ('leg_targets', 'nominal_leg_targets', 'applied_residual'):
        self.assertTrue(
          torch.equal(left[key], right[key]),
          f'{key} drifted when the camp fields were written explicitly.',
        )


class StairTriggerSensorTest(unittest.TestCase):
  """The trigger's force source must reproduce the C2-j3 capture layout."""

  def test_sensor_cfg_matches_the_evidence_capture(self) -> None:
    sensor = stair_trigger_sensor_cfg()
    self.assertEqual(sensor.name, STAIR_TRIGGER_SENSOR_NAME)
    self.assertEqual(sensor.fields, STAIR_TRIGGER_SENSOR_FIELDS)
    self.assertEqual(sensor.num_slots, STAIR_TRIGGER_SLOTS_PER_WHEEL)
    # A reduction would collapse the slots the metric maximises over, and a
    # global force frame would break the "component 0 is the normal force"
    # convention the frozen metric depends on.
    self.assertEqual(sensor.reduce, 'none')
    self.assertFalse(sensor.global_frame)
    self.assertEqual(sensor.primary.entity, 'robot')
    self.assertEqual(sensor.primary.mode, 'geom')
    self.assertEqual(sensor.secondary.pattern, 'terrain')

  def test_a_missing_trigger_sensor_fails_at_construction(self) -> None:
    # Silently leaving stair_mode False forever would turn the camp into an
    # un-triggered always-classical baseline, which no gate would catch.
    torch.manual_seed(2026)
    cfg = _stage5_cfg(stair_trigger_sensor_name=STAIR_TRIGGER_SENSOR_NAME)
    with self.assertRaises(ValueError):
      ManagerBasedRlEnv(cfg=cfg, device='cpu').close()

  def test_flat_ground_rolling_does_not_trigger(self) -> None:
    """Smoke check only - NOT the preregistered validity gate.

    S5B Protocol 2 registers a false-positive check under camp
    randomization covering both flat rolling and the Stage5 8x kick
    regime; that runs against the camp environment before the first
    reported training run. This test only guards the wiring: a sensor that
    reported a spurious 18 N normal-x force while rolling on the flat would
    latch stair_mode here.
    """

    rows = _run_camp_trajectory(
      steps=10,
      attach_trigger_sensor=True,
      stair_trigger_sensor_name=STAIR_TRIGGER_SENSOR_NAME,
      stair_mode_freezes_leg_reference=True,
    )
    for index, row in enumerate(rows):
      self.assertFalse(
        bool(row['stair_mode'][0]),
        f'flat rolling latched stair_mode at step {index}',
      )
      torch.testing.assert_close(
        row['leg_reference'], row['nominal_leg_targets']
      )
    # Non-vacuity: a mis-specified geom pattern would report no contacts at
    # all, and this test would then pass because the metric is identically
    # zero rather than because the trigger is selective. The wheels carry
    # the robot on the flat, so contacts and normal force must be present -
    # what keeps the metric low is the |nx| projection, not the absence of
    # force.
    self.assertGreater(max(row['contacts_found'] for row in rows), 0)
    self.assertGreater(max(row['peak_normal_force'] for row in rows), 1.0)
    self.assertLess(
      max(float(row['trigger_metric'].max()) for row in rows),
      STAIR_TRIGGER_FORCE_N,
    )


class StairProgressRewardTest(unittest.TestCase):
  """S5B: the stair-progress term MUST be contact-trigger gated."""

  def _reward_trajectory(self, steps: int = 5, **camp_fields):
    torch.manual_seed(2026)
    env = ManagerBasedRlEnv(cfg=_stage5_cfg(**camp_fields), device='cpu')
    try:
      env.reset(seed=2026)
      rows = []
      for _ in range(steps):
        env.step(torch.tensor([_ACTION_ROW], dtype=torch.float32))
        rows.append(
          (
            stair_mode_forward_progress(env).clone(),
            env.scene['robot'].data.root_link_lin_vel_w[:, 0].clone(),
          )
        )
      return rows
    finally:
      env.close()

  def test_progress_is_identically_zero_without_the_trigger(self) -> None:
    # This is the false-trigger damage bound in code: the four no-regression
    # gate runs never latch stair_mode, so the camp's progress term cannot
    # pay the policy anything during them.
    rows = self._reward_trajectory()
    for reward, _ in rows:
      self.assertEqual(float(reward[0]), 0.0)

  def test_progress_tracks_forward_speed_once_gated_on(self) -> None:
    rows = self._reward_trajectory(
      stair_mode_forced=True, stair_mode_freezes_leg_reference=True
    )
    self.assertTrue(
      any(float(velocity[0]) > 0.0 for _, velocity in rows),
      'the stack never rolled forward, so the gating is untested here',
    )
    for reward, velocity in rows:
      self.assertAlmostEqual(
        float(reward[0]), max(float(velocity[0]), 0.0), places=5
      )

  def test_backward_motion_earns_nothing(self) -> None:
    rows = self._reward_trajectory(
      stair_mode_forced=True, stair_mode_freezes_leg_reference=True
    )
    for reward, velocity in rows:
      if float(velocity[0]) < 0.0:
        self.assertEqual(float(reward[0]), 0.0)


class StairCampObservationLayoutTest(unittest.TestCase):
  """The runtime actor group must carry the registered S5B field layout."""

  def test_term_order_matches_the_registered_field_list(self) -> None:
    # `ACTOR_FIELD_NAMES` is the registered contract; proprioception is one
    # field there and seven manager terms here, so compare the tail.
    self.assertEqual(ACTOR_FIELD_NAMES[0], 'proprioception')
    self.assertEqual(tuple(ACTOR_FIELD_NAMES[1:]), STAIR_CAMP_ACTOR_TAIL_TERMS)

  def test_builder_emits_the_registered_order(self) -> None:
    base = make_hoppertrex_hybrid_env_cfg(stage=5).observations['actor'].terms
    terms = stair_camp_actor_terms(base)
    self.assertEqual(tuple(terms), STAIR_CAMP_ACTOR_TERM_ORDER)
    # Proprioception must be taken verbatim from the frozen Stage5 group.
    for name in STAIR_CAMP_PROPRIOCEPTION_TERMS:
      self.assertIs(terms[name], base[name])

  def test_builder_rejects_a_group_without_proprioception(self) -> None:
    with self.assertRaises(ValueError):
      stair_camp_actor_terms({})

  def test_actor_vector_tail_carries_the_registered_fields(self) -> None:
    """Structural end-to-end check of the concatenated actor vector."""

    torch.manual_seed(2026)
    cfg = _stage5_cfg(
      stair_mode_forced=True, stair_mode_freezes_leg_reference=True
    )
    cfg.observations['actor'].terms = stair_camp_actor_terms(
      cfg.observations['actor'].terms
    )
    # Corruption would perturb the tail and defeat an exact comparison; the
    # frozen stages' noise fields are unrelated to this layout question.
    cfg.observations['actor'].enable_corruption = False
    env = ManagerBasedRlEnv(cfg=cfg, device='cpu')
    try:
      env.reset(seed=2026)
      obs = env.step(torch.tensor([_ACTION_ROW], dtype=torch.float32))[0]
      actor = obs['actor']
      term = env.action_manager.get_term('hybrid_wheel_leg')
      # Tail widths are fixed by the contract: 9 + 2 + 4 + 4 + 6 + 1 = 26.
      self.assertGreater(actor.shape[-1], 26)
      tail = actor[:, -26:]
      phase = tail[:, :9]
      expected_phase = torch.zeros_like(phase)
      expected_phase[:, 0] = 1.0
      torch.testing.assert_close(phase, expected_phase)
      torch.testing.assert_close(tail[:, 9:11], term.controller_baseline)
      torch.testing.assert_close(tail[:, 11:15], term.leg_reference)
      torch.testing.assert_close(tail[:, 15:19], term.classical_errors)
      torch.testing.assert_close(tail[:, 19:25], term.applied_residual)
      torch.testing.assert_close(
        tail[:, 25], term.stair_mode.to(torch.float)
      )
      # The camp publishes the pose the residual acts about, which under a
      # held reference is NOT the live posture-map target.
      self.assertFalse(
        torch.allclose(term.leg_reference, term.nominal_leg_targets)
      )
    finally:
      env.close()

  def test_phase_one_hot_is_constant_idle(self) -> None:
    """S5B Protocol 1 pins the FSM off, so the channel is informationless."""

    torch.manual_seed(2026)
    env = ManagerBasedRlEnv(cfg=_stage5_cfg(), device='cpu')
    try:
      env.reset(seed=2026)
      seen = []
      for _ in range(3):
        env.step(torch.tensor([_ACTION_ROW], dtype=torch.float32))
        seen.append(stair_phase_one_hot_observation(env).clone())
      for row in seen:
        self.assertEqual(row.shape[-1], PHASE_COUNT)
        self.assertEqual(int(row.argmax(dim=-1)[0]), int(StairPhase.IDLE))
        self.assertEqual(float(row.sum()), 1.0)
    finally:
      env.close()


class StairCampRealStackEndToEndTest(unittest.TestCase):
  """Real-stack CPU end-to-end (agent_workflow traps 19/20).

  Everything below runs the QUALIFIED classical stack loaded from the frozen
  artifacts, not the unqualified PD fallback: with the fallback the robot
  cannot track the 0.07 m/s command at all and never reaches a riser, so a
  fallback run would assert nothing about the trigger.
  """

  @classmethod
  def setUpClass(cls) -> None:
    root = Path(__file__).resolve().parents[1]
    artifacts = root / 'docs' / 'experiments' / 'artifacts'
    yaw = sorted(artifacts.glob('yaw_gpu_*/yaw_calibration.json'))
    paths = {
      'HOPPERTREX_HYBRID_CONTROLLER_PATH': artifacts
      / 'c1_schedule_candidate24_1f54968_seed1'
      / 'c1_schedule.json',
      'HOPPERTREX_HYBRID_CALIBRATION_PATH': artifacts
      / 'hybrid_runtime_seed1'
      / 'velocity_calibration_seed1.json',
      'HOPPERTREX_HYBRID_POSTURE_MAP_PATH': artifacts
      / 'c1_posture_requalification_seed1'
      / 'posture_map_seed1_registered_p032.json',
      'HOPPERTREX_HYBRID_STATION_CALIBRATION_PATH': artifacts
      / 'c1_posture_requalification_seed1'
      / 'station_calibration_seed1.json',
    }
    if yaw:
      paths['HOPPERTREX_HYBRID_YAW_CALIBRATION_PATH'] = yaw[0]
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
      raise unittest.SkipTest(f'frozen artifacts missing: {missing}')
    cls._saved = {k: os.environ.get(k) for k in paths}
    for key, value in paths.items():
      os.environ[key] = str(value)

  @classmethod
  def tearDownClass(cls) -> None:
    for key, value in getattr(cls, '_saved', {}).items():
      if value is None:
        os.environ.pop(key, None)
      else:
        os.environ[key] = value

  def _drive(
    self,
    *,
    steps: int,
    initial_upper_height_m: float = 0.03,
    flat: bool = False,
    num_envs: int = 4,
  ):
    torch.manual_seed(11)
    if flat:
      # The registered tier grid excludes flat ground, so a flat camp env is
      # not a reachable curriculum state. The flat false-positive observable
      # lives on the Stage5 stack the four no-regression gates run on, so
      # that is what this variant drives: the frozen flat plane, plus the
      # camp's trigger sensor and its freeze switch.
      cfg = make_hoppertrex_hybrid_env_cfg(stage=5, play=True)
      cfg.scene.sensors = tuple(cfg.scene.sensors) + (
        stair_trigger_sensor_cfg(),
      )
      action_cfg = cfg.actions['hybrid_wheel_leg']
      action_cfg.stair_trigger_sensor_name = STAIR_TRIGGER_SENSOR_NAME
      action_cfg.stair_mode_freezes_leg_reference = True
    else:
      cfg = make_stair_camp_env_cfg(
        play=True, initial_upper_height_m=initial_upper_height_m
      )
    action_cfg = cfg.actions['hybrid_wheel_leg']
    self.assertTrue(action_cfg.controller_qualified)
    self.assertTrue(action_cfg.posture_map_qualified)
    cfg.scene.num_envs = num_envs
    if cfg.scene.terrain is not None:
      cfg.scene.terrain.num_envs = num_envs
    cfg.episode_length_s = 1.0e4
    # The Stage5 reset disturbance tips a zero-action rollout often enough
    # that no episode survives the 0.25 m approach. Removing it mirrors the
    # settle-then-drive protocol the C0/C1/C2 probes used to measure the
    # classical boundary this camp is layered on.
    reset_name = (
      'reset_root_state_with_small_disturbance'
      if flat
      else 'reset_root_to_stair_approach'
    )
    reset_cfg = cfg.events[reset_name]
    reset_cfg.params['pose_range'] = {}
    reset_cfg.params['velocity_range'] = {}
    env = ManagerBasedRlEnv(cfg=cfg, device='cpu')
    try:
      env.reset(seed=11)
      term = env.action_manager.get_term('hybrid_wheel_leg')
      twist = env.command_manager.get_term('twist')
      posture = env.command_manager.get_term('posture')

      def force(vx: float) -> None:
        for attribute in ('vel_command_b', 'vel_command_w'):
          command = getattr(twist, attribute, None)
          if command is not None:
            command[:, :] = 0.0
            command[:, 0] = vx
        for attribute in (
          'is_standing_env',
          'is_heading_env',
          'is_world_env',
          'is_forward_env',
        ):
          flags = getattr(twist, attribute, None)
          if flags is not None:
            flags[:] = False
        posture.target[:, 0] = ROOT_HEIGHT_TARGET
        posture.target[:, 1] = 0.0
        posture.command[:, 0] = ROOT_HEIGHT_TARGET
        posture.command[:, 1] = 0.0

      actions = torch.zeros(num_envs, 6)
      initial_distance = (
        None
        if flat
        else stair_camp_distance_to_riser_observation(env).clone()
      )
      reference_at_trigger = None
      peak_metric = 0.0
      for step in range(steps):
        force(0.0 if step < _SETTLE_STEPS else _APPROACH_VX_MPS)
        env.step(actions)
        data = env.scene.sensors[STAIR_TRIGGER_SENSOR_NAME].data
        peak_metric = max(
          peak_metric,
          float(
            stair_trigger_metric(
              found=data.found,
              force_contact_frame=data.force,
              normal_global=data.normal,
            ).max()
          ),
        )
        if reference_at_trigger is None and bool(term.stair_mode.any()):
          reference_at_trigger = term.leg_reference.detach().clone()
      robot = env.scene['robot']
      terrain = env.scene.terrain
      return {
        'stair_mode': term.stair_mode.clone(),
        'levels': (
          torch.zeros(num_envs, dtype=torch.long)
          if getattr(terrain, 'terrain_levels', None) is None
          else terrain.terrain_levels.clone()
        ),
        'leg_reference': term.leg_reference.detach().clone(),
        'reference_at_trigger': reference_at_trigger,
        'peak_metric': peak_metric,
        'step_height': (
          torch.zeros(num_envs, 1)
          if flat
          else stair_camp_step_height_observation(env).clone()
        ),
        'initial_distance': initial_distance,
        'final_distance': (
          None
          if flat
          else stair_camp_distance_to_riser_observation(env).clone()
        ),
        'travel': (
          robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
        ).detach()
        + (
          0.0
          if flat
          else STAIR_CAMP_RISER_OFFSET_M + STAIR_CAMP_START_OFFSET_M
        ),
      }
    finally:
      env.close()

  def test_retained_privileged_fields_are_non_degenerate_on_real_camp(self) -> None:
    result = self._drive(initial_upper_height_m=0.03, steps=520, num_envs=8)
    self.assertGreater(
      len(set(float(value) for value in result['step_height'].flatten())),
      1,
      'step_height did not vary across terrain rows',
    )
    self.assertFalse(
      torch.allclose(result['initial_distance'], result['final_distance']),
      'distance_to_riser did not vary with robot motion',
    )
    self.assertGreater(
      result['peak_metric'],
      0.0,
      'contact_force remained structurally zero through the contact rollout',
    )
  def test_rolling_into_a_riser_latches_and_freezes(self) -> None:
    result = self._drive(initial_upper_height_m=0.03, steps=520)
    self.assertGreater(
      result['peak_metric'],
      STAIR_TRIGGER_FORCE_N,
      'no contact anywhere near the trigger threshold - the robot never '
      'reached a riser, so this run asserts nothing',
    )
    stair_envs = result['levels'] > 0
    self.assertTrue(bool(stair_envs.any()), 'no stair env was drawn')
    self.assertTrue(
      bool(result['stair_mode'][stair_envs].any()),
      'rolling into a riser did not latch stair_mode',
    )
    self.assertIsNotNone(result['reference_at_trigger'])
    latched = result['stair_mode']
    torch.testing.assert_close(
      result['leg_reference'][latched],
      result['reference_at_trigger'][latched],
    )

  def test_real_runner_checkpoint_round_trip_restores_actor_and_curriculum(self) -> None:
    """Exercise the actual 256-env registered checkpoint lifecycle on CPU."""

    env_cfg = make_stair_camp_env_cfg(play=False)
    agent_cfg = hoppertrex_stair_camp_ppo_runner_cfg()
    agent_cfg.seed = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device='cpu')
    try:
      wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
      runner = HybridOnPolicyRunner(
        wrapped, asdict(agent_cfg), log_dir=None, device='cpu'
      )
      observations = wrapped.get_observations()
      actor = runner.alg.get_policy()
      output = actor(observations)
      self.assertEqual(tuple(observations['actor'].shape), (256, 52))
      self.assertEqual(tuple(observations['critic'].shape), (256, 55))
      torch.testing.assert_close(output, torch.zeros_like(output))
      torch.testing.assert_close(
        actor.distribution.std_param,
        torch.full_like(actor.distribution.std_param, 0.6),
      )

      curriculum = env.stair_camp_curriculum_state
      saved_curriculum = curriculum.state_dict()
      runner.current_learning_iteration = 999
      with tempfile.TemporaryDirectory() as temporary:
        checkpoint_path = Path(temporary) / 'model_999.pt'
        runner.save(str(checkpoint_path))
        checkpoint = torch.load(
          checkpoint_path, map_location='cpu', weights_only=False
        )
        infos = checkpoint['infos']
        self.assertEqual(checkpoint['iter'], 999)
        self.assertEqual(
          infos['stair_camp_training']['completed_updates'], 1000
        )
        self.assertEqual(
          infos['env_state']['common_step_counter'], env.common_step_counter
        )
        self.assertEqual(infos['stair_camp_curriculum'], saved_curriculum)
        self.assertEqual(
          set(infos['stair_camp_progress']),
          {
            'upper_height_m',
            'trigger_rate',
            'residual_abs_mean',
            'residual_rms',
            'residual_abs_max',
            'evaluations',
          },
        )

        with torch.no_grad():
          for module in actor.modules():
            if isinstance(module, torch.nn.Linear) and module.out_features == 6:
              module.bias.fill_(1.0)
        curriculum.evaluations = 5
        curriculum.next_evaluation_step = 7200
        runner.load(str(checkpoint_path))

      self.assertEqual(curriculum.state_dict(), saved_curriculum)
      self.assertEqual(runner.current_learning_iteration, 999)
      restored = actor(wrapped.get_observations())
      torch.testing.assert_close(restored, torch.zeros_like(restored))
    finally:
      env.close()

  def test_flat_terrain_never_latches_over_the_same_drive(self) -> None:
    """Preregistered damage bound: flat rolling must produce zero triggers.

    This is the real-stack version of the wiring check, not the registered
    envelope-transfer gate - that one additionally spans the Stage5 8x kick
    regime and runs against the camp before the first reported training run.
    """

    result = self._drive(flat=True, steps=520)
    self.assertEqual(result['levels'].tolist(), [0] * 4)
    self.assertFalse(
      bool(result['stair_mode'].any()),
      'flat rolling produced a false trigger',
    )
    # Non-vacuity: the robot has to roll PAST the plane where a riser would
    # stand, otherwise "no trigger" says nothing. `travel` is measured from
    # the spawn, which sits `STAIR_CAMP_START_OFFSET_M` short of that plane.
    self.assertGreater(
      float(result['travel'].max()), STAIR_CAMP_START_OFFSET_M
    )


class _StubTerrain:
  def __init__(self, num_envs: int, rows: int):
    self.terrain_levels = torch.zeros(num_envs, dtype=torch.long)
    self.terrain_types = torch.zeros(num_envs, dtype=torch.long)
    self.terrain_origins = torch.zeros(rows, 1, 3)
    for row in range(rows):
      self.terrain_origins[row, 0, 0] = float(row)
    self.env_origins = torch.zeros(num_envs, 3)


class _StubScene:
  def __init__(self, terrain):
    self.terrain = terrain


class _StubEnv:
  def __init__(self, num_envs: int, rows: int):
    self.num_envs = num_envs
    self.device = 'cpu'
    self.common_step_counter = 0
    self.scene = _StubScene(_StubTerrain(num_envs, rows))


class StairCampCurriculumTest(unittest.TestCase):
  """S5B Protocol 4: the evaluation-driven height band.

  Both defects this pins were real and both were silent: without the
  curriculum term the band never moved (a 1000-iteration run would spend its
  whole budget on the initial tier), and mjlab's own `max_init_terrain_level`
  draw spans [0, upper], which puts flat ground into a training distribution
  the registration explicitly excludes.
  """

  def _curriculum(self, env, *, interval=1200, upper=0.01):
    return StairCampCurriculum(env, interval, upper)

  def _window(self, curriculum, env, *, rate: float, count: int = 10):
    """Complete `count` episodes at the current upper tier with `rate`."""

    ids = torch.arange(count)
    env.scene.terrain.terrain_levels[ids] = curriculum.upper_level
    successes = int(round(rate * count))
    outcome = torch.zeros(env.num_envs)
    outcome[:successes] = 1.0
    with mock.patch.object(
      hybrid_task, 'stair_camp_climb_success', return_value=outcome
    ):
      curriculum._started = True
      curriculum._score_finished_episodes(env, ids)
    env.common_step_counter = curriculum.next_evaluation_step
    curriculum._maybe_evaluate(env)

  def test_registered_initial_state(self) -> None:
    curriculum = self._curriculum(_StubEnv(16, 16))
    self.assertEqual(curriculum.state.lower_height_m, 0.01)
    self.assertEqual(curriculum.state.upper_height_m, 0.01)
    self.assertEqual(curriculum.upper_level, 1)

  def test_tier_grid_never_assigns_flat_ground(self) -> None:
    env = _StubEnv(64, 16)
    curriculum = self._curriculum(env, upper=0.05)
    ids = torch.arange(64)
    for _ in range(20):
      curriculum._assign_levels(env, ids)
      self.assertGreaterEqual(int(env.scene.terrain.terrain_levels.min()), 1)
      self.assertLessEqual(int(env.scene.terrain.terrain_levels.max()), 5)

  def test_assigned_levels_span_the_whole_band(self) -> None:
    # Non-vacuity for the test above: a broken implementation that pinned
    # every env to the upper tier would also never assign level 0.
    env = _StubEnv(64, 16)
    curriculum = self._curriculum(env, upper=0.05)
    seen = set()
    for _ in range(20):
      curriculum._assign_levels(env, torch.arange(64))
      seen.update(env.scene.terrain.terrain_levels.tolist())
    self.assertEqual(seen, {1, 2, 3, 4, 5})

  def test_assigning_a_level_moves_the_env_origin(self) -> None:
    # The reset event that places the robot reads `scene.env_origins`, which
    # IS `terrain.env_origins`; a level change that did not update it would
    # spawn the robot on the previous episode's staircase.
    env = _StubEnv(8, 16)
    curriculum = self._curriculum(env, upper=0.05)
    curriculum._assign_levels(env, torch.arange(8))
    expected = env.scene.terrain.terrain_origins[
      env.scene.terrain.terrain_levels, env.scene.terrain.terrain_types
    ]
    torch.testing.assert_close(env.scene.terrain.env_origins, expected)

  def test_three_good_evaluations_promote_once(self) -> None:
    env = _StubEnv(16, 16)
    curriculum = self._curriculum(env)
    for _ in range(2):
      self._window(curriculum, env, rate=1.0)
      self.assertEqual(curriculum.state.upper_height_m, 0.01)
    self._window(curriculum, env, rate=1.0)
    self.assertAlmostEqual(curriculum.state.upper_height_m, 0.02)

  def test_a_bad_evaluation_resets_the_streak(self) -> None:
    env = _StubEnv(16, 16)
    curriculum = self._curriculum(env)
    self._window(curriculum, env, rate=1.0)
    self._window(curriculum, env, rate=1.0)
    self._window(curriculum, env, rate=0.7)
    self._window(curriculum, env, rate=1.0)
    self.assertEqual(curriculum.state.upper_height_m, 0.01)

  def test_an_empty_window_scores_zero_and_blocks_promotion(self) -> None:
    # Registered deviation minute: S5B is silent on empty windows. Scoring
    # 0.0 can only delay a promotion; skipping would let a policy that never
    # reaches the top tier promote on stale windows.
    env = _StubEnv(16, 16)
    curriculum = self._curriculum(env)
    self._window(curriculum, env, rate=1.0)
    self._window(curriculum, env, rate=1.0)
    env.common_step_counter = curriculum.next_evaluation_step
    curriculum._maybe_evaluate(env)
    self.assertEqual(curriculum.state.consecutive_ready_evaluations, 0)
    self.assertEqual(curriculum.state.upper_height_m, 0.01)

  def test_only_episodes_at_the_current_upper_tier_count(self) -> None:
    env = _StubEnv(16, 16)
    curriculum = self._curriculum(env, upper=0.03)
    ids = torch.arange(8)
    # Every one of these succeeded, but all sit BELOW the current upper tier.
    env.scene.terrain.terrain_levels[ids] = 1
    with mock.patch.object(
      hybrid_task,
      'stair_camp_climb_success',
      return_value=torch.ones(env.num_envs),
    ):
      curriculum._started = True
      curriculum._score_finished_episodes(env, ids)
    self.assertEqual(curriculum.episodes_at_upper, 0)
    self.assertEqual(curriculum.successes_at_upper, 0)

  def test_promotion_stops_at_the_registered_cap(self) -> None:
    env = _StubEnv(16, 16)
    curriculum = self._curriculum(env, upper=0.15)
    for _ in range(9):
      self._window(curriculum, env, rate=1.0)
    self.assertAlmostEqual(curriculum.state.upper_height_m, 0.15)

  def test_evaluation_cadence_matches_the_registered_runner(self) -> None:
    from mjlab.scripts.train import TrainConfig

    cfg = TrainConfig.from_task(STAIR_CAMP_TASK_ID)
    self.assertEqual(
      STAIR_CAMP_STEPS_PER_ITERATION, cfg.agent.num_steps_per_env
    )
    params = cfg.env.curriculum['stair_height_band'].params
    self.assertEqual(
      params['evaluation_interval_steps'],
      STAIR_CAMP_EVALUATION_INTERVAL_ITERS * cfg.agent.num_steps_per_env,
    )

  def test_registered_task_carries_the_curriculum(self) -> None:
    from mjlab.scripts.train import TrainConfig

    cfg = TrainConfig.from_task(STAIR_CAMP_TASK_ID)
    self.assertIn('stair_height_band', cfg.env.curriculum)
    self.assertEqual(
      cfg.env.curriculum['stair_height_band'].params[
        'initial_upper_height_m'
      ],
      0.01,
    )


if __name__ == '__main__':
  unittest.main()
