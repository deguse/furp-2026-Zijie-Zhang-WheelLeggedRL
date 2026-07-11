#!/usr/bin/env python3
'''Diagnose Hybrid Stage1 reset and first-action behavior without training.'''

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import numpy as np
import torch

PROJECT_PATH = Path(__file__).resolve().parents[1]
SRC_PATH = Path(__file__).resolve().parents[2]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from hoppertrex_mjlab.tasks.hoppertrex_balance_task import (  # noqa: E402
  non_wheel_ground_contact,
  wheel_ground_contact,
)
from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (  # noqa: E402
  make_hoppertrex_hybrid_env_cfg,
)


SCENARIOS = ('zero', 'std', 'controller_off')
ROOT_HEIGHT_MINIMUM = 0.26
MATRIX_ENV_COUNTS = (1, 2, 4, 8, 16)


def matrix_cases() -> tuple[tuple[int, int], ...]:
  return tuple((stage, envs) for stage in (0, 1) for envs in MATRIX_ENV_COUNTS)


def first_bad_substep(
  rows: Sequence[Mapping[str, object]],
  *,
  timestep: float,
) -> int | None:
  previous_z: float | None = None
  for row in rows:
    substep = int(row['substep'])
    root_z = float(row['raw_root_z'])
    qvel_z = float(row['raw_root_qvel_z'])
    if not math.isfinite(root_z) or not math.isfinite(qvel_z):
      return substep
    if substep >= 0:
      expected_qvel = -9.81 * timestep * (substep + 1)
      if abs(qvel_z - expected_qvel) > 0.5:
        return substep
      if previous_z is not None and previous_z - root_z > 0.02:
        return substep
    previous_z = root_z
  return None


def classify_matrix(results: Sequence[Mapping[str, object]]) -> str:
  if not all(bool(result.get('finite', False)) for result in results):
    return 'non-finite'
  if any(
    float(result['reset_qz_min']) < ROOT_HEIGHT_MINIMUM
    or float(result['reset_qvz_abs_max']) > 0.5
    for result in results
  ):
    return 'invalid_reset_state'
  by_key = {
    (int(result['stage']), int(result['num_envs'])): result
    for result in results
  }
  env1_healthy = all(by_key[(stage, 1)]['first_bad_substep'] is None for stage in (0, 1))
  larger_bad = any(
    by_key[(stage, envs)]['first_bad_substep'] is not None
    for stage in (0, 1) for envs in MATRIX_ENV_COUNTS[1:]
  )
  if env1_healthy and larger_bad:
    return 'cuda_scale_dependent_dynamics'
  stage0_healthy = all(
    by_key[(0, envs)]['first_bad_substep'] is None for envs in MATRIX_ENV_COUNTS
  )
  stage1_bad = any(
    by_key[(1, envs)]['first_bad_substep'] is not None for envs in MATRIX_ENV_COUNTS
  )
  if stage0_healthy and stage1_bad:
    return 'stage_specific_startup'
  if not any(bool(result['wheel_contact_any']) for result in results):
    return 'contact_initialization_failure'
  if any(result['first_bad_substep'] is not None for result in results):
    return 'controller_startup_failure'
  return 'inconclusive'


def scenario_actions(name: str, *, num_envs: int, seed: int) -> np.ndarray:
  if name not in SCENARIOS:
    raise ValueError(f'Unsupported scenario: {name}')
  actions = np.zeros((num_envs, 6), dtype=np.float32)
  if name == 'std':
    actions[:, 0] = np.random.default_rng(seed).normal(0.0, 0.15, num_envs)
  return actions


def _finite(values: Sequence[float]) -> np.ndarray:
  array = np.asarray(values, dtype=np.float64)
  if not np.isfinite(array).all():
    raise ValueError('Startup measurements must be finite.')
  return array


def classify_startup(scenarios: Mapping[str, Mapping[str, Sequence[float]]]) -> str:
  for name in SCENARIOS:
    if name not in scenarios:
      raise ValueError(f'Missing startup scenario: {name}')
  zero = scenarios['zero']
  std = scenarios['std']
  raw_zero = _finite(zero['raw_root_z'])
  derived_zero = _finite(zero['derived_root_z'])
  raw_std = _finite(std['raw_root_z'])
  for values in scenarios.values():
    _finite(values['raw_root_z'])
    _finite(values['derived_root_z'])

  if np.max(np.abs(raw_zero - derived_zero)) > 0.05 and raw_zero[0] >= ROOT_HEIGHT_MINIMUM:
    return 'derived_state_stale'
  if all(_finite(values['raw_root_z'])[0] < ROOT_HEIGHT_MINIMUM for values in scenarios.values()):
    return 'invalid_reset_height'
  if raw_zero[-1] < ROOT_HEIGHT_MINIMUM:
    return 'controller_startup_failure'
  if raw_std[-1] < ROOT_HEIGHT_MINIMUM:
    return 'exploration_startup_failure'
  if not any(bool(value) for values in scenarios.values() for value in values['both_wheels_contact']):
    return 'contact_initialization_failure'
  for values in scenarios.values():
    would_low = values.get('would_root_too_low', ())
    if any(would_low) and min(_finite(values['derived_root_z'])) >= ROOT_HEIGHT_MINIMUM:
      return 'termination_source_mismatch'
  return 'inconclusive'


def _force_zero_command(env: object) -> None:
  term = env.command_manager.get_term('twist')
  for attribute in ('vel_command_b', 'vel_command_w'):
    getattr(term, attribute)[:, :] = 0.0


def _git_sha() -> str:
  root = Path(__file__).resolve().parents[3]
  return subprocess.run(
    ['git', 'rev-parse', 'HEAD'], cwd=root, check=True,
    capture_output=True, text=True,
  ).stdout.strip()


def _file_sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_object(path: Path) -> Mapping[str, object]:
  payload = json.loads(path.read_text(encoding='utf-8'))
  if not isinstance(payload, Mapping):
    raise ValueError(f'Expected a JSON object in {path}.')
  return payload


def _matrix_snapshot(
  env: object,
  *,
  stage: int,
  num_envs: int,
  substep: int,
  action_term: object,
  wheel_params: Mapping[str, object],
  non_wheel_params: Mapping[str, object],
) -> list[dict[str, object]]:
  robot = env.scene['robot']
  qadr = robot.data.indexing.free_joint_q_adr
  vadr = robot.data.indexing.free_joint_v_adr
  raw_z = robot.data.data.qpos[:, int(qadr[2])]
  raw_qvel_z = robot.data.data.qvel[:, int(vadr[2])]
  derived_z = robot.data.root_link_pos_w[:, 2]
  derived_vz = robot.data.root_link_lin_vel_w[:, 2]
  wheel = wheel_ground_contact(env, **wheel_params).bool()
  non_wheel = non_wheel_ground_contact(env, **non_wheel_params).bool()
  rows = []
  for env_id in range(num_envs):
    origin = env.scene.env_origins[env_id]
    values = (
      raw_z[env_id], raw_qvel_z[env_id], derived_z[env_id], derived_vz[env_id],
      origin[0], origin[1], origin[2],
      action_term.controller_baseline[env_id, 0],
      action_term.controller_baseline[env_id, 1],
      action_term.wheel_targets[env_id, 0],
      action_term.wheel_targets[env_id, 1],
    )
    finite = all(math.isfinite(float(value.item())) for value in values)
    rows.append({
      'stage': stage,
      'num_envs': num_envs,
      'env_id': env_id,
      'substep': substep,
      'elapsed_s': max(0, substep + 1) * float(env.physics_dt),
      'raw_root_z': float(raw_z[env_id].item()),
      'raw_root_qvel_z': float(raw_qvel_z[env_id].item()),
      'derived_root_z': float(derived_z[env_id].item()),
      'derived_root_velocity_z': float(derived_vz[env_id].item()),
      'terrain_origin_x': float(origin[0].item()),
      'terrain_origin_y': float(origin[1].item()),
      'terrain_origin_z': float(origin[2].item()),
      'controller_left': float(action_term.controller_baseline[env_id, 0].item()),
      'controller_right': float(action_term.controller_baseline[env_id, 1].item()),
      'wheel_target_left': float(action_term.wheel_targets[env_id, 0].item()),
      'wheel_target_right': float(action_term.wheel_targets[env_id, 1].item()),
      'both_wheels_contact': bool(wheel[env_id].item()),
      'non_wheel_contact': bool(non_wheel[env_id].item()),
      'finite': finite,
    })
  return rows


def collect_matrix_case(
  args: argparse.Namespace,
  *,
  stage: int,
  num_envs: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
  from mjlab.envs import ManagerBasedRlEnv

  cfg = make_hoppertrex_hybrid_env_cfg(
    stage=stage,
    play=False,
    controller_path=args.controller_path,
    calibration_path=args.calibration_path,
  )
  cfg.scene.num_envs = num_envs
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.num_envs = num_envs
  cfg.terminations = {}
  with args.environment_log_path.open('a', encoding='utf-8') as environment_log:
    print(f'\n===== matrix stage={stage} envs={num_envs} =====', file=environment_log)
    with redirect_stdout(environment_log), redirect_stderr(environment_log):
      env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    env.reset(seed=args.seed)
    _force_zero_command(env)
    action_term = env.action_manager.get_term('hybrid_wheel_leg')
    wheel_params = cfg.rewards['wheel_ground_contact'].params
    non_wheel_params = cfg.rewards['non_wheel_ground_contact'].params
    rows = _matrix_snapshot(
      env, stage=stage, num_envs=num_envs, substep=-1,
      action_term=action_term, wheel_params=wheel_params,
      non_wheel_params=non_wheel_params,
    )
    action = torch.zeros((num_envs, 6), device=env.device)
    env.action_manager.process_action(action)
    for substep in range(cfg.decimation):
      env.action_manager.apply_action()
      env.scene.write_data_to_sim()
      env.sim.step()
      env.scene.update(dt=env.physics_dt)
      rows.extend(_matrix_snapshot(
        env, stage=stage, num_envs=num_envs, substep=substep,
        action_term=action_term, wheel_params=wheel_params,
        non_wheel_params=non_wheel_params,
      ))
    per_env_bad = []
    for env_id in range(num_envs):
      env_rows = [row for row in rows if int(row['env_id']) == env_id]
      per_env_bad.append(first_bad_substep(env_rows, timestep=env.physics_dt))
    reset_rows = [row for row in rows if int(row['substep']) == -1]
    final_rows = [row for row in rows if int(row['substep']) == cfg.decimation - 1]
    reset_qz = np.asarray([row['raw_root_z'] for row in reset_rows], dtype=float)
    reset_qvz = np.asarray([row['raw_root_qvel_z'] for row in reset_rows], dtype=float)
    final_qz = np.asarray([row['raw_root_z'] for row in final_rows], dtype=float)
    final_qvz = np.asarray([row['raw_root_qvel_z'] for row in final_rows], dtype=float)
    result = {
      'stage': stage,
      'num_envs': num_envs,
      'reset_qz_min': float(reset_qz.min()),
      'reset_qz_mean': float(reset_qz.mean()),
      'reset_qvz_abs_max': float(np.abs(reset_qvz).max()),
      'first_bad_substep': min(
        (value for value in per_env_bad if value is not None), default=None,
      ),
      'final_qz_min': float(final_qz.min()),
      'final_qz_mean': float(final_qz.mean()),
      'final_qvz_mean': float(final_qvz.mean()),
      'wheel_contact_any': any(bool(row['both_wheels_contact']) for row in rows),
      'non_wheel_contact_any': any(bool(row['non_wheel_contact']) for row in rows),
      'finite': all(bool(row['finite']) for row in rows),
    }
    return rows, result
  finally:
    env.close()


def run_first_step_matrix(args: argparse.Namespace) -> None:
  args.output_dir.mkdir(parents=True, exist_ok=True)
  args.environment_log_path = args.output_dir / 'environment_setup.log'
  args.environment_log_path.write_text('', encoding='utf-8')
  all_rows: list[dict[str, object]] = []
  results: list[dict[str, object]] = []
  print('stage envs reset_qz reset_qvz first_bad final_qz final_qvz contact result')
  for stage, num_envs in matrix_cases():
    rows, result = collect_matrix_case(
      args, stage=stage, num_envs=num_envs,
    )
    all_rows.extend(rows)
    results.append(result)
    status = 'BAD' if result['first_bad_substep'] is not None else 'OK'
    print(
      f'{stage:>5} {num_envs:>4} {result["reset_qz_mean"]:>8.4f} '
      f'{result["reset_qvz_abs_max"]:>10.4f} '
      f'{str(result["first_bad_substep"]):>9} '
      f'{result["final_qz_mean"]:>8.4f} {result["final_qvz_mean"]:>10.4f} '
      f'{str(result["wheel_contact_any"]):>7} {status}'
    )
  csv_path = args.output_dir / 'first_step_matrix.csv'
  with csv_path.open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
    writer.writeheader()
    writer.writerows(all_rows)
  controller_payload = _json_object(args.controller_path)
  calibration_payload = _json_object(args.calibration_path)
  gpu_name = (
    torch.cuda.get_device_name(0)
    if str(args.device).startswith('cuda') and torch.cuda.is_available()
    else 'cpu'
  )
  sample_cfg = make_hoppertrex_hybrid_env_cfg(stage=1)
  summary = {
    'schema_version': 1,
    'classification': classify_matrix(results),
    'git_sha': _git_sha(),
    'seed': args.seed,
    'device': args.device,
    'gpu_name': gpu_name,
    'controller_gain_hash': controller_payload.get('gain_hash'),
    'calibration_hash': calibration_payload.get('calibration_hash'),
    'physics_timestep': sample_cfg.sim.mujoco.timestep,
    'decimation': sample_cfg.decimation,
    'solver': sample_cfg.sim.mujoco.solver,
    'integrator': sample_cfg.sim.mujoco.integrator,
    'results': results,
  }
  json_path = args.output_dir / 'first_step_matrix.json'
  json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
  print(f'[RESULT] classification={summary["classification"]}')
  print(f'[OK] CSV: {csv_path.resolve()}')
  print(f'[OK] JSON: {json_path.resolve()}')


def collect_scenario(args: argparse.Namespace, name: str) -> list[dict[str, object]]:
  from mjlab.envs import ManagerBasedRlEnv

  cfg = make_hoppertrex_hybrid_env_cfg(
    stage=1,
    play=False,
    controller_path=args.controller_path,
    calibration_path=args.calibration_path,
  )
  cfg.scene.num_envs = args.num_envs
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.num_envs = args.num_envs
  cfg.episode_length_s = 1.0e9
  action_cfg = cfg.actions['hybrid_wheel_leg']
  if name == 'controller_off':
    action_cfg.controller_gain = (0.0, 0.0, 0.0, 0.0)
  termination_cfg = dict(cfg.terminations)
  cfg.terminations = {}

  with args.environment_log_path.open('a', encoding='utf-8') as environment_log:
    print(f'\n===== scenario={name} =====', file=environment_log)
    with redirect_stdout(environment_log), redirect_stderr(environment_log):
      env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    env.reset()
    _force_zero_command(env)
    action = torch.as_tensor(
      scenario_actions(name, num_envs=args.num_envs, seed=args.seed),
      device=env.device,
    )
    action_term = env.action_manager.get_term('hybrid_wheel_leg')
    robot = env.scene['robot']
    raw_z_address = int(robot.data.indexing.free_joint_q_adr[2])
    wheel_params = cfg.rewards['wheel_ground_contact'].params
    non_wheel_params = cfg.rewards['non_wheel_ground_contact'].params
    rows: list[dict[str, object]] = []
    for step in range(args.steps):
      _force_zero_command(env)
      env.step(action)
      _force_zero_command(env)
      raw_z = robot.data.data.qpos[:, raw_z_address]
      derived_z = robot.data.root_link_pos_w[:, 2]
      wheel_contact = wheel_ground_contact(env, **wheel_params).bool()
      non_wheel_contact_value = non_wheel_ground_contact(
        env, **non_wheel_params,
      ).bool()
      would = {}
      for term_name, term in termination_cfg.items():
        if term_name == 'time_out':
          continue
        would[term_name] = term.func(env, **term.params).bool()
      gravity = robot.data.projected_gravity_b
      pitch = torch.atan2(gravity[:, 0], -gravity[:, 2])
      for env_id in range(args.num_envs):
        rows.append({
          'scenario': name,
          'seed': args.seed,
          'env_id': env_id,
          'step': step,
          'episode_length': int(env.episode_length_buf[env_id].item()),
          'raw_root_z': float(raw_z[env_id].item()),
          'derived_root_z': float(derived_z[env_id].item()),
          'root_z_difference': float((derived_z[env_id] - raw_z[env_id]).item()),
          'root_vertical_velocity': float(robot.data.root_link_lin_vel_w[env_id, 2].item()),
          'pitch': float(pitch[env_id].item()),
          'pitch_rate': float(robot.data.root_link_ang_vel_b[env_id, 1].item()),
          'requested_vx': 0.0,
          'calibrated_vx': action_cfg.velocity_command_bias,
          'controller_left': float(action_term.controller_baseline[env_id, 0].item()),
          'controller_right': float(action_term.controller_baseline[env_id, 1].item()),
          'raw_residual_balance': float(action_term.raw_action[env_id, 0].item()),
          'applied_residual_balance': float(action_term.applied_residual[env_id, 0].item()),
          'final_wheel_target_left': float(action_term.wheel_targets[env_id, 0].item()),
          'final_wheel_target_right': float(action_term.wheel_targets[env_id, 1].item()),
          'both_wheels_contact': bool(wheel_contact[env_id].item()),
          'non_wheel_contact': bool(non_wheel_contact_value[env_id].item()),
          'would_root_too_low': bool(would['root_too_low'][env_id].item()),
          'would_bad_orientation': bool(would['bad_orientation'][env_id].item()),
          'would_non_wheel_contact': bool(would['non_wheel_ground_contact'][env_id].item()),
          'would_nan': bool(would['nan_detection'][env_id].item()),
        })
    return rows
  finally:
    env.close()


def summarize(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
  measurements: dict[str, dict[str, list[object]]] = {}
  compact: dict[str, dict[str, object]] = {}
  for name in SCENARIOS:
    selected = [row for row in rows if row['scenario'] == name]
    measurements[name] = {
      key: [row[key] for row in selected]
      for key in ('raw_root_z', 'derived_root_z', 'both_wheels_contact', 'would_root_too_low')
    }
    step_summaries = []
    for step in sorted({int(row['step']) for row in selected}):
      step_rows = [row for row in selected if int(row['step']) == step]
      raw = np.asarray([row['raw_root_z'] for row in step_rows], dtype=np.float64)
      derived = np.asarray(
        [row['derived_root_z'] for row in step_rows], dtype=np.float64,
      )
      step_summaries.append({
        'step': step,
        'raw_root_z': {
          'min': float(raw.min()), 'mean': float(raw.mean()), 'max': float(raw.max()),
        },
        'derived_root_z': {
          'min': float(derived.min()),
          'mean': float(derived.mean()),
          'max': float(derived.max()),
        },
        'both_wheels_contact_rate': float(np.mean([
          bool(row['both_wheels_contact']) for row in step_rows
        ])),
        'root_too_low_rate': float(np.mean([
          bool(row['would_root_too_low']) for row in step_rows
        ])),
      })
    contact_steps = [
      int(row['step']) for row in selected if bool(row['both_wheels_contact'])
    ]
    low_steps = [
      int(row['step']) for row in selected if bool(row['would_root_too_low'])
    ]
    compact[name] = {
      'first_both_wheels_contact_step': min(contact_steps) if contact_steps else None,
      'first_root_too_low_step': min(low_steps) if low_steps else None,
      'max_raw_derived_z_difference': float(max(
        abs(float(row['raw_root_z']) - float(row['derived_root_z']))
        for row in selected
      )),
      'steps': step_summaries,
    }
  return {
    'classification': classify_startup(measurements),
    'scenarios': compact,
  }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--mode',
    choices=('continuous', 'first_step_matrix'),
    default='continuous',
  )
  parser.add_argument('--controller-path', type=Path, required=True)
  parser.add_argument('--calibration-path', type=Path, required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
  parser.add_argument('--num-envs', type=int, default=16)
  parser.add_argument('--steps', type=int, default=10)
  parser.add_argument('--seed', type=int, default=1)
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  if args.num_envs <= 0 or args.steps <= 0:
    raise ValueError('--num-envs and --steps must be positive.')
  if args.mode == 'first_step_matrix':
    run_first_step_matrix(args)
    return
  args.output_dir.mkdir(parents=True, exist_ok=True)
  args.environment_log_path = args.output_dir / 'environment_setup.log'
  args.environment_log_path.write_text('', encoding='utf-8')
  rows = []
  for name in SCENARIOS:
    print(f'[RUN] {name}: {args.num_envs} envs x {args.steps} steps')
    scenario_rows = collect_scenario(args, name)
    rows.extend(scenario_rows)
    final_rows = [row for row in scenario_rows if int(row['step']) == args.steps - 1]
    final_z = np.asarray([row['raw_root_z'] for row in final_rows], dtype=np.float64)
    print(
      f'[DONE] {name}: final raw z '
      f'min/mean/max={final_z.min():.4f}/{final_z.mean():.4f}/{final_z.max():.4f}'
    )
  csv_path = args.output_dir / 'startup_steps.csv'
  with csv_path.open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  summary = summarize(rows)
  controller_payload = _json_object(args.controller_path)
  calibration_payload = _json_object(args.calibration_path)
  summary.update({
    'schema_version': 1,
    'git_sha': _git_sha(),
    'seed': args.seed,
    'device': args.device,
    'num_envs': args.num_envs,
    'steps': args.steps,
    'controller_file_sha256': _file_sha(args.controller_path),
    'controller_gain_hash': controller_payload.get('gain_hash'),
    'calibration_file_sha256': _file_sha(args.calibration_path),
    'calibration_hash': calibration_payload.get('calibration_hash'),
  })
  json_path = args.output_dir / 'startup_summary.json'
  json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
  classification = summary['classification']
  print(f'[RESULT] classification={classification}')
  print(f'[OK] CSV: {csv_path.resolve()}')
  print(f'[OK] JSON: {json_path.resolve()}')
  print(f'[OK] Environment log: {args.environment_log_path.resolve()}')


if __name__ == '__main__':
  main()
