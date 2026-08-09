from __future__ import annotations

import copy
import json
import os
import unittest
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace
from unittest import mock

import torch
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.scripts.train import TrainConfig
from torch import nn

import hoppertrex_mjlab.tasks  # noqa: F401
from hoppertrex_mjlab.hybrid.config import STAIR_CAMP_ACTION_MASK, STAIR_CAMP_TASK_ID
from hoppertrex_mjlab.hybrid.distribution import MaskedGaussianDistribution
from hoppertrex_mjlab.hybrid.runner import (
  HybridOnPolicyRunner,
  is_stair_camp_env,
  zero_initialize_stair_camp_actor_output,
)
from hoppertrex_mjlab.hybrid.stair_camp_contract import (
  STAIR_CAMP_CANONICAL_CONTRACT_SHA256,
  STAIR_CAMP_INIT_STD,
  stair_camp_contract_hash,
  stair_camp_contract_payload,
  stair_camp_init_std,
  validate_stair_camp_progress_payload,
  validate_stair_camp_training_request,
)
from hoppertrex_mjlab.scripts.rsl_rl.adjudicate_stair_camp import (
  STAIR_CAMP_CANONICAL_CONTRACT_SHA256 as adjudicator_contract_sha256,
)
from hoppertrex_mjlab.scripts.rsl_rl.evaluate_stair_camp import (
  STAIR_CAMP_CANONICAL_CONTRACT_SHA256 as evaluator_contract_sha256,
)
from hoppertrex_mjlab.scripts.rsl_rl.train import (
  resolve_and_validate_hybrid_resume,
  validate_stair_camp_extension_checkpoint,
)
from hoppertrex_mjlab.tasks.agents import hoppertrex_stair_camp_ppo_runner_cfg
from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
  make_stair_camp_env_cfg,
  STAIR_CAMP_EVALUATION_INTERVAL_ITERS,
  STAIR_CAMP_PRIVILEGED_TERMS,
  STAIR_CAMP_STEPS_PER_ITERATION,
  STAIR_CAMP_WITHDRAWN_PRIVILEGED_TERMS,
  StairCampCurriculum,
)

ROOT = Path(__file__).resolve().parents[1]


class _Terrain:
  def __init__(self, num_envs: int, rows: int = 16):
    self.terrain_levels = torch.ones(num_envs, dtype=torch.long)
    self.terrain_types = torch.zeros(num_envs, dtype=torch.long)
    self.terrain_origins = torch.zeros(rows, 1, 3)
    self.env_origins = torch.zeros(num_envs, 3)


class _CurriculumEnv:
  def __init__(self, num_envs: int = 4):
    self.num_envs = num_envs
    self.device = 'cpu'
    self.common_step_counter = 0
    self.reset_buf = torch.zeros(num_envs, dtype=torch.bool)
    self.scene = SimpleNamespace(terrain=_Terrain(num_envs))


class StairCampRunnerLifecycleTest(unittest.TestCase):
  def test_runner_constructor_applies_zero_init_only_for_marked_camp(self) -> None:
    actor = nn.Sequential(nn.Linear(52, 128), nn.ELU(), nn.Linear(128, 6))
    env = SimpleNamespace(
      cfg=SimpleNamespace(
        stair_camp_task_id=STAIR_CAMP_TASK_ID,
        stair_camp_zero_initialize_actor_output=True,
        stair_camp_training_contract=True,
        actions={
          'hybrid_wheel_leg': SimpleNamespace(
            action_scales=(0.5, 0.3, 0.07, 0.07, 0.07, 0.07)
          )
        },
      )
    )
    train_cfg = {
      'seed': 1,
      'actor': SimpleNamespace(
        distribution_cfg={'init_std': STAIR_CAMP_INIT_STD},
      ),
    }

    def base_init(runner, *_args, **_kwargs):
      runner.alg = SimpleNamespace(get_policy=lambda: actor)

    with (
      mock.patch.object(MjlabOnPolicyRunner, '__init__', new=base_init),
      mock.patch(
        'hoppertrex_mjlab.hybrid.runner.repository_git_sha', return_value='abc'
      ),
      mock.patch(
        'hoppertrex_mjlab.hybrid.runner.bind_stair_camp_contract',
        return_value='contract',
      ),
      mock.patch(
        'hoppertrex_mjlab.hybrid.runner.stair_camp_artifact_bindings',
        return_value={'controller': 'artifact'},
      ),
    ):
      HybridOnPolicyRunner(env, train_cfg)
    output = actor(torch.randn(3, 52))
    torch.testing.assert_close(output, torch.zeros_like(output))
  def test_unique_six_output_head_is_zeroed(self) -> None:
    actor = nn.Sequential(nn.Linear(52, 128), nn.ELU(), nn.Linear(128, 6))
    head = zero_initialize_stair_camp_actor_output(actor)
    torch.testing.assert_close(head.weight, torch.zeros_like(head.weight))
    torch.testing.assert_close(head.bias, torch.zeros_like(head.bias))
    output = actor(torch.randn(5, 52))
    torch.testing.assert_close(output, torch.zeros_like(output))

  def test_ambiguous_six_output_heads_fail_closed(self) -> None:
    actor = nn.Sequential(nn.Linear(52, 6), nn.Linear(6, 6))
    with self.assertRaisesRegex(ValueError, 'exactly one'):
      zero_initialize_stair_camp_actor_output(actor)

  def test_real_distribution_keeps_the_six_output_head_the_zero_init_target(
    self,
  ) -> None:
    """Bind the zero-init guard to the REAL RSL-RL actor geometry.

    `MLPModel` sizes its final Linear as `distribution.input_dim`, not as the
    action dimension. The other zero-init tests build `nn.Sequential` stubs, so
    none of them would notice an upstream change that made `input_dim` 12
    (e.g. a per-output std parameterization): the guard would then find zero
    six-output heads and the camp would only fail at training launch. This
    test fails in CI instead.
    """

    cfg = hoppertrex_stair_camp_ppo_runner_cfg()
    distribution_cfg = dict(cfg.actor.distribution_cfg)
    distribution_cfg.pop('class_name')
    distribution = MaskedGaussianDistribution(6, **distribution_cfg)
    self.assertEqual(distribution.input_dim, 6)
    mean = distribution.deterministic_output(torch.zeros(4, distribution.input_dim))
    self.assertEqual(tuple(mean.shape), (4, 6))
    torch.testing.assert_close(mean, torch.zeros_like(mean))
    torch.testing.assert_close(
      distribution.std_param.detach().flatten(),
      torch.full((6,), STAIR_CAMP_INIT_STD),
    )

  def test_init_std_provenance_is_read_back_from_the_live_ppo_config(
    self,
  ) -> None:
    """`init_std` in the checkpoint record must be observed, not retyped.

    A literal on both the recording side and the checking side would agree
    with itself even if the run used a different exploration scale, so the
    record would certify a value no run produced.
    """

    cfg = hoppertrex_stair_camp_ppo_runner_cfg()
    self.assertEqual(stair_camp_init_std(cfg), STAIR_CAMP_INIT_STD)
    self.assertEqual(cfg.actor.distribution_cfg['init_std'], STAIR_CAMP_INIT_STD)
    drifted = hoppertrex_stair_camp_ppo_runner_cfg()
    drifted.actor.distribution_cfg['init_std'] = 0.5
    with self.assertRaisesRegex(ValueError, 'init_std drifted'):
      stair_camp_init_std(drifted)

  def test_checkpoint_actor_load_overrides_zero_initialization(self) -> None:
    runner = object.__new__(HybridOnPolicyRunner)
    runner._stair_camp = True
    runner._hybrid_loaded_infos = {}
    runner._stair_camp_loaded_completed_updates = 0
    actor = nn.Sequential(nn.Linear(52, 128), nn.ELU(), nn.Linear(128, 6))
    head = zero_initialize_stair_camp_actor_output(actor)
    runner.alg = SimpleNamespace(get_policy=lambda: actor)
    runner.env = SimpleNamespace()
    infos = {
      'stair_camp_training': {'completed_updates': 1000},
    }

    def base_load(*_args, **_kwargs):
      with torch.no_grad():
        head.weight.fill_(0.25)
        head.bias.fill_(0.5)
      return infos

    with (
      mock.patch.object(MjlabOnPolicyRunner, 'load', side_effect=base_load),
      mock.patch.object(
        HybridOnPolicyRunner,
        '_validate_stair_camp_loaded_record',
        return_value=1000,
      ),
    ):
      runner.load('model.pt', load_cfg={'actor': True})
    self.assertEqual(float(head.weight.detach().min()), 0.25)
    self.assertEqual(float(head.bias.detach().min()), 0.5)
  def test_marker_is_task_specific(self) -> None:
    camp = SimpleNamespace(
      cfg=SimpleNamespace(
        stair_camp_task_id=STAIR_CAMP_TASK_ID,
        stair_camp_zero_initialize_actor_output=True,
      )
    )
    self.assertTrue(is_stair_camp_env(camp))
    self.assertFalse(is_stair_camp_env(SimpleNamespace(cfg=SimpleNamespace())))

  def test_fresh_total_budget_remains_1000(self) -> None:
    runner = object.__new__(HybridOnPolicyRunner)
    runner._stair_camp = True
    runner._stair_camp_loaded_completed_updates = 0
    runner.current_learning_iteration = 0
    with mock.patch.object(MjlabOnPolicyRunner, 'learn') as base:
      runner.learn(1000, True)
    base.assert_called_once_with(1000, True)
    self.assertEqual(runner.current_learning_iteration, 0)

  def test_extension_runs_only_the_remaining_2000_updates(self) -> None:
    runner = object.__new__(HybridOnPolicyRunner)
    runner._stair_camp = True
    runner._stair_camp_loaded_completed_updates = 1000
    runner.current_learning_iteration = 999
    with mock.patch.object(MjlabOnPolicyRunner, 'learn') as base:
      runner.learn(3000, True)
    base.assert_called_once_with(2000, True)
    self.assertEqual(runner.current_learning_iteration, 1000)


class StairCampResumeContractTest(unittest.TestCase):
  def _cfg(self, *, seed: int = 1, target: int = 3000):
    action = SimpleNamespace(action_scales=(0.5, 0.3, 0.07, 0.07, 0.07, 0.07))
    return SimpleNamespace(
      env=SimpleNamespace(
        actions={'hybrid_wheel_leg': action},
        stair_camp_contract_sha256='contract',
      ),
      agent=SimpleNamespace(
        seed=seed,
        max_iterations=target,
        resume=True,
        actor=SimpleNamespace(
          distribution_cfg={'init_std': STAIR_CAMP_INIT_STD},
        ),
      ),
    )

  def _checkpoint(self, *, seed: int = 1):
    return {
      'iter': 999,
      'actor_state_dict': {'mlp.0.weight': torch.zeros(128, 52)},
      'critic_state_dict': {'mlp.0.weight': torch.zeros(128, 55)},
      'infos': {
        'stair_camp_training': {
          'schema_version': 1,
          'task': STAIR_CAMP_TASK_ID,
          'training_seed': seed,
          'git_sha': 'abc',
          'contract_sha256': 'contract',
          'artifact_bindings': {'controller': 'artifact'},
          'action_scales': [0.5, 0.3, 0.07, 0.07, 0.07, 0.07],
          'zero_initialized_deterministic_mean': True,
          'init_std': 0.6,
          'completed_updates': 1000,
        },
        'stair_camp_curriculum': {
          'schema_version': 1,
          'upper_height_m': 0.01,
          'evaluations': 20,
          'triggered_episodes': 0,
          'completed_episodes': 0,
          'residual_abs_sum': 0.0,
          'residual_sq_sum': 0.0,
          'residual_sample_count': 0,
          'residual_abs_max': 0.0,
        },
        'stair_camp_progress': {
          'upper_height_m': 0.01,
          'trigger_rate': 0.0,
          'residual_abs_mean': 0.0,
          'residual_rms': 0.0,
          'residual_abs_max': 0.0,
          'evaluations': 20,
        },
        'env_state': {'common_step_counter': 24000},
      },
    }

  def test_fresh_camp_does_not_require_resume(self) -> None:
    cfg = SimpleNamespace(env=SimpleNamespace(), agent=SimpleNamespace(resume=False))
    with mock.patch(
      'hoppertrex_mjlab.scripts.rsl_rl.train.validate_stair_camp_training_request',
      return_value='contract',
    ) as validate:
      self.assertIsNone(resolve_and_validate_hybrid_resume(STAIR_CAMP_TASK_ID, cfg))
    validate.assert_called_once_with(cfg.env, cfg.agent, resume=False)

  def test_valid_own_checkpoint_is_accepted_for_extension(self) -> None:
    cfg = self._cfg()
    checkpoint = self._checkpoint()
    with (
      mock.patch(
        'hoppertrex_mjlab.scripts.rsl_rl.train._repository_head',
        return_value='abc',
      ),
      mock.patch(
        'hoppertrex_mjlab.scripts.rsl_rl.train.stair_camp_contract_hash',
        return_value='contract',
      ),
      mock.patch(
        'hoppertrex_mjlab.scripts.rsl_rl.train.stair_camp_artifact_bindings',
        return_value={'controller': 'artifact'},
      ),
    ):
      validate_stair_camp_extension_checkpoint(cfg, checkpoint)

  def test_stage5_or_migration_checkpoint_is_rejected(self) -> None:
    cfg = self._cfg()
    with self.assertRaisesRegex(ValueError, 'own camp checkpoint'):
      validate_stair_camp_extension_checkpoint(
        cfg,
        {'infos': {'hybrid_stage_migration': {'target_stage': 5}}},
      )

  def test_cross_seed_and_wrong_source_budget_are_rejected(self) -> None:
    cfg = self._cfg(seed=2)
    checkpoint = self._checkpoint(seed=1)
    with self.assertRaisesRegex(ValueError, 'seed'):
      validate_stair_camp_extension_checkpoint(cfg, checkpoint)
    cfg = self._cfg(seed=1)
    checkpoint = self._checkpoint()
    checkpoint['infos']['stair_camp_training']['completed_updates'] = 900
    with (
      mock.patch(
        'hoppertrex_mjlab.scripts.rsl_rl.train._repository_head',
        return_value='abc',
      ),
      mock.patch(
        'hoppertrex_mjlab.scripts.rsl_rl.train.stair_camp_contract_hash',
        return_value='contract',
      ),
      mock.patch(
        'hoppertrex_mjlab.scripts.rsl_rl.train.stair_camp_artifact_bindings',
        return_value={'controller': 'artifact'},
      ),self.assertRaisesRegex(ValueError, '1000-update')
    ):
      validate_stair_camp_extension_checkpoint(cfg, checkpoint)


class StairCampExactCadenceTest(unittest.TestCase):
  def _curriculum(self, env: _CurriculumEnv) -> StairCampCurriculum:
    return StairCampCurriculum(env, evaluation_interval_steps=1200)

  def test_evaluates_at_exact_1200_step_boundaries_without_reset(self) -> None:
    env = _CurriculumEnv()
    curriculum = self._curriculum(env)
    for step in range(1, 1200):
      env.common_step_counter = step
      curriculum.record_step(env)
    self.assertEqual(curriculum.evaluations, 0)
    env.common_step_counter = 1200
    curriculum.record_step(env)
    self.assertEqual(curriculum.evaluations, 1)
    env.common_step_counter = 2400
    curriculum.record_step(env)
    self.assertEqual(curriculum.evaluations, 2)

  def test_terminal_episode_on_boundary_is_scored_before_evaluation(self) -> None:
    env = _CurriculumEnv()
    curriculum = self._curriculum(env)
    env.common_step_counter = 1200
    env.reset_buf[0] = True

    def score(_env, _ids):
      curriculum.episodes_at_upper = 1
      curriculum.successes_at_upper = 1

    with mock.patch.object(curriculum, '_score_finished_episodes', side_effect=score):
      curriculum.record_step(env)
    self.assertEqual(curriculum.state.consecutive_ready_evaluations, 1)

  def test_state_round_trip_matches_continuous_execution(self) -> None:
    env_a = _CurriculumEnv()
    continuous = self._curriculum(env_a)
    env_a.common_step_counter = 1200
    continuous._maybe_evaluate(env_a)
    saved = continuous.state_dict()

    env_b = _CurriculumEnv()
    resumed = self._curriculum(env_b)
    resumed.load_state_dict(saved)
    for step in (2400, 3600):
      env_a.common_step_counter = step
      env_b.common_step_counter = step
      continuous._maybe_evaluate(env_a)
      resumed._maybe_evaluate(env_b)
    self.assertEqual(resumed.state_dict(), continuous.state_dict())

  def test_state_schema_and_interval_drift_fail_closed(self) -> None:
    env = _CurriculumEnv()
    curriculum = self._curriculum(env)
    payload = curriculum.state_dict()
    payload.pop('next_evaluation_step')
    with self.assertRaisesRegex(ValueError, 'schema'):
      curriculum.load_state_dict(payload)
    payload = curriculum.state_dict()
    payload['evaluation_interval_steps'] = 1000
    with self.assertRaisesRegex(ValueError, 'interval'):
      curriculum.load_state_dict(payload)


  def test_state_rejects_type_count_grid_and_cadence_mutations_atomically(self) -> None:
    env = _CurriculumEnv()
    curriculum = self._curriculum(env)
    pristine = curriculum.state_dict()
    mutations = (
      ('schema_version', True),
      ('upper_height_m', 0.015),
      ('consecutive_ready_evaluations', 3),
      ('evaluations', -1),
      ('next_evaluation_step', 2400),
      ('last_processed_step', 1200),
      ('started', 1),
      ('triggered_episodes', 1),
      ('residual_abs_sum', float('nan')),
      ('residual_abs_sum', 1.0),
    )
    for field, value in mutations:
      payload = dict(pristine)
      payload[field] = value
      with self.subTest(field=field, value=value):
        with self.assertRaises(ValueError):
          curriculum.load_state_dict(payload)
        self.assertEqual(curriculum.state_dict(), pristine)
    extra = dict(pristine)
    extra['unexpected'] = 1
    with self.assertRaisesRegex(ValueError, 'schema'):
      curriculum.load_state_dict(extra)


class StairCampRegisteredConfigTest(unittest.TestCase):
  def test_training_request_freezes_fresh_extension_and_scalar_defaults(self) -> None:
    with mock.patch(
      'hoppertrex_mjlab.hybrid.stair_camp_contract.bind_stair_camp_contract',
      return_value=STAIR_CAMP_CANONICAL_CONTRACT_SHA256,
    ):
      fresh = TrainConfig.from_task(STAIR_CAMP_TASK_ID)
      fresh.agent.seed = 1
      self.assertEqual(
        validate_stair_camp_training_request(
          fresh.env, fresh.agent, resume=False
        ),
        STAIR_CAMP_CANONICAL_CONTRACT_SHA256,
      )
      extension = TrainConfig.from_task(STAIR_CAMP_TASK_ID)
      extension.agent.seed = 1
      extension.agent.max_iterations = 3000
      extension.agent.resume = True
      self.assertEqual(
        validate_stair_camp_training_request(
          extension.env, extension.agent, resume=True
        ),
        STAIR_CAMP_CANONICAL_CONTRACT_SHA256,
      )

      for field, value in (
        ('seed', 4),
        ('max_iterations', 999),
        ('save_interval', 99),
        ('num_steps_per_env', 23),
      ):
        cfg = TrainConfig.from_task(STAIR_CAMP_TASK_ID)
        cfg.agent.seed = 1
        setattr(cfg.agent, field, value)
        with self.subTest(field=field), self.assertRaises(ValueError):
          validate_stair_camp_training_request(
            cfg.env, cfg.agent, resume=False
          )
      cfg = TrainConfig.from_task(STAIR_CAMP_TASK_ID)
      cfg.agent.seed = 1
      cfg.env.scene.num_envs = 255
      with self.assertRaisesRegex(ValueError, '256'):
        validate_stair_camp_training_request(cfg.env, cfg.agent, resume=False)

  def test_registered_defaults_and_asymmetric_surface(self) -> None:
    cfg = TrainConfig.from_task(STAIR_CAMP_TASK_ID)
    self.assertEqual(cfg.env.scene.num_envs, 256)
    self.assertEqual(cfg.agent.max_iterations, 1000)
    self.assertEqual(cfg.agent.save_interval, 100)
    self.assertEqual(cfg.agent.num_steps_per_env, 24)
    self.assertEqual(
      tuple(cfg.agent.actor.distribution_cfg['active_mask']),
      STAIR_CAMP_ACTION_MASK,
    )
    self.assertEqual(cfg.agent.actor.distribution_cfg['init_std'], 0.6)
    actor_names = tuple(cfg.env.observations['actor'].terms)
    critic_names = tuple(cfg.env.observations['critic'].terms)
    self.assertEqual(len(actor_names), 13)
    self.assertEqual(critic_names[-3:], STAIR_CAMP_PRIVILEGED_TERMS)
    for name in STAIR_CAMP_WITHDRAWN_PRIVILEGED_TERMS:
      self.assertNotIn(name, critic_names)
    self.assertIn('stair_camp_step', cfg.env.metrics)
    params = cfg.env.curriculum['stair_height_band'].params
    self.assertEqual(
      params['evaluation_interval_steps'],
      STAIR_CAMP_EVALUATION_INTERVAL_ITERS * STAIR_CAMP_STEPS_PER_ITERATION,
    )


class StairCampContractFingerprintTest(unittest.TestCase):
  def test_contract_carries_no_machine_specific_value(self) -> None:
    """The fingerprint must be reproducible on the machine that trains.

    Binding the same configuration everywhere is the entire purpose of the
    canonical contract, so any value that changes with the checkout location
    silently turns it into a per-machine fingerprint. This regressed once:
    five action `*_source` fields plus the posture command's `source` carried
    absolute artifact paths, so the training host computed a different hash
    from the development checkout and refused to start. The artifact identity
    is bound machine-independently by the six content hashes in `artifacts`.
    """

    artifacts = ROOT / 'docs' / 'experiments' / 'artifacts'
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
      self.skipTest(f'frozen artifacts missing: {missing}')

    # The artifact env vars must be LIVE while the config is built: with them
    # unset every `*_source` field is empty, so a scan would pass vacuously
    # against exactly the defect it exists to catch.
    with mock.patch.dict(
      os.environ, {key: str(value) for key, value in paths.items()}
    ):
      env_cfg = make_stair_camp_env_cfg(play=False)
      payload = stair_camp_contract_payload(
        env_cfg, hoppertrex_stair_camp_ppo_runner_cfg()
      )

    sources = [
      value
      for value in (
        getattr(env_cfg.actions['hybrid_wheel_leg'], name, None)
        for name in (
          'controller_source',
          'posture_map_source',
          'yaw_calibration_source',
          'station_calibration_source',
        )
      )
      if isinstance(value, str) and value
    ]
    self.assertGreaterEqual(len(sources), 4, 'artifact sources were not populated')
    blob = json.dumps(payload, sort_keys=True, separators=(',', ':'))

    offenders: list[str] = []

    def walk(node: object, path: str) -> None:
      if isinstance(node, dict):
        for key, item in node.items():
          walk(item, f'{path}.{key}')
      elif isinstance(node, list):
        for index, item in enumerate(node):
          walk(item, f'{path}[{index}]')
      elif isinstance(node, str):
        windows = PureWindowsPath(node) if '\\' in node else None
        posix = PurePosixPath(node)
        if (windows is not None and windows.is_absolute()) or posix.is_absolute():
          offenders.append(f'{path}={node}')

    walk(payload, 'contract')
    self.assertEqual(offenders, [])
    for marker in (str(ROOT), ROOT.name, 'mjlab_workspace', 'worktrees'):
      self.assertNotIn(marker, blob)

  def test_all_three_registered_hash_copies_agree(self) -> None:
    """`evaluate_stair_camp` re-declares the hash to stay Torch-free.

    That is a deliberate integration constraint, but it means the registered
    fingerprint exists in three files with nothing forcing them to agree.
    """

    self.assertEqual(
      evaluator_contract_sha256, STAIR_CAMP_CANONICAL_CONTRACT_SHA256
    )
    self.assertEqual(
      adjudicator_contract_sha256, STAIR_CAMP_CANONICAL_CONTRACT_SHA256
    )

  def _cfg(self) -> TrainConfig:
    cfg = TrainConfig.from_task(STAIR_CAMP_TASK_ID)
    cfg.agent.seed = 1
    action = cfg.env.actions['hybrid_wheel_leg']
    for index, name in enumerate((
      'controller_gain_hash',
      'calibration_hash',
      'yaw_calibration_hash',
      'posture_map_hash',
      'posture_artifact_hash',
      'station_calibration_hash',
    ), start=1):
      setattr(action, name, f'{index:x}' * 64)
    return cfg

  def test_progress_snapshot_is_strict_and_state_bound(self) -> None:
    curriculum = {
      'upper_height_m': 0.03,
      'evaluations': 4,
      'triggered_episodes': 3,
      'completed_episodes': 4,
      'residual_abs_sum': 0.4,
      'residual_sq_sum': 0.06,
      'residual_sample_count': 10,
      'residual_abs_max': 0.1,
    }
    # Keep the physical max within the registered 0.070 rad authority.
    curriculum['residual_abs_max'] = 0.07
    progress = {
      'upper_height_m': 0.03,
      'trigger_rate': 0.75,
      'residual_abs_mean': 0.04,
      'residual_rms': (0.006) ** 0.5,
      'residual_abs_max': 0.07,
      'evaluations': 4,
    }
    # The square sum must also respect max^2 * count.
    curriculum['residual_sq_sum'] = 0.04
    progress['residual_rms'] = (0.004) ** 0.5
    normalized = validate_stair_camp_progress_payload(progress, curriculum)
    self.assertEqual(normalized['evaluations'], 4)

    for field, value in (
      ('trigger_rate', 0.5),
      ('upper_height_m', 0.031),
      ('residual_abs_max', 0.071),
      ('evaluations', 4.5),
    ):
      with self.subTest(field=field):
        changed = dict(progress)
        changed[field] = value
        with self.assertRaises(ValueError):
          validate_stair_camp_progress_payload(changed, curriculum)
    extra = dict(progress)
    extra['extra'] = 1
    with self.assertRaisesRegex(ValueError, 'schema'):
      validate_stair_camp_progress_payload(extra, curriculum)

  def test_seed_and_total_budget_are_outside_the_policy_contract(self) -> None:
    fresh = self._cfg()
    extension = self._cfg()
    extension.agent.seed = 3
    extension.agent.max_iterations = 3000
    self.assertEqual(
      stair_camp_contract_hash(fresh.env, fresh.agent),
      stair_camp_contract_hash(extension.env, extension.agent),
    )

  def test_runtime_subterrain_size_rewrite_is_hash_stable(self) -> None:
    cfg = self._cfg()
    expected = stair_camp_contract_hash(cfg.env, cfg.agent)
    generator = cfg.env.scene.terrain.terrain_generator
    sub_terrain = generator.sub_terrains['stair']
    self.assertNotEqual(sub_terrain.size, generator.size)

    # MjLab's TerrainGenerator performs exactly this in-place assignment while
    # constructing the environment. It is an implementation detail, not a
    # second training contract.
    sub_terrain.size = generator.size
    self.assertEqual(stair_camp_contract_hash(cfg.env, cfg.agent), expected)

    generator.size = (8.1, 8.0)
    self.assertNotEqual(stair_camp_contract_hash(cfg.env, cfg.agent), expected)
  def test_terrain_all_rewards_and_full_ppo_surface_are_digest_bound(self) -> None:
    baseline = self._cfg()
    expected = stair_camp_contract_hash(baseline.env, baseline.agent)

    mutations = (
      lambda cfg: cfg.env.events.__setitem__(
        'unregistered_friction_family', copy.deepcopy(cfg.env.events['push_robot'])
      ),
      lambda cfg: setattr(
        cfg.env.scene.terrain.terrain_generator.sub_terrains['stair'],
        'step_width',
        0.31,
      ),
      lambda cfg: setattr(cfg.env.rewards['alive'], 'weight', 0.51),
      lambda cfg: cfg.env.rewards['track_linear_velocity'].params.__setitem__(
        'std', 0.09
      ),
      lambda cfg: setattr(cfg.agent.algorithm, 'num_mini_batches', 8),
      lambda cfg: setattr(cfg.agent.actor, 'hidden_dims', (64, 64)),
      lambda cfg: cfg.env.curriculum['stair_height_band'].params.__setitem__(
        'evaluation_interval_steps', 1201
      ),
    )
    for mutate in mutations:
      with self.subTest(mutate=mutate):
        changed = copy.deepcopy(baseline)
        mutate(changed)
        self.assertNotEqual(
          stair_camp_contract_hash(changed.env, changed.agent), expected
        )


if __name__ == '__main__':
  unittest.main()
