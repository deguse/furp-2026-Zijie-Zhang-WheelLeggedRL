#!/usr/bin/env python3
'''Diagnose Hybrid Stage1 reset and first-action behavior without training.'''

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
  scenarios: dict[str, dict[str, list[object]]] = {}
  for name in SCENARIOS:
    selected = [row for row in rows if row['scenario'] == name]
    scenarios[name] = {
      key: [row[key] for row in selected]
      for key in ('raw_root_z', 'derived_root_z', 'both_wheels_contact', 'would_root_too_low')
    }
  return {'classification': classify_startup(scenarios), 'scenarios': scenarios}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
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
  args.output_dir.mkdir(parents=True, exist_ok=True)
  rows = [row for name in SCENARIOS for row in collect_scenario(args, name)]
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


if __name__ == '__main__':
  main()
