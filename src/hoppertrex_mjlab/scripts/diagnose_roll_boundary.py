#!/usr/bin/env python3
# Non-evidentiary diagnostics for the first positive RollBoundary tier.

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mjlab
import torch
from mjlab.envs import ManagerBasedRlEnv

from hoppertrex_mjlab.hybrid.roll_pose_schedule import (
  RollPoseSchedule,
  roll_pose_schedule_candidates,
)
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb

DIAGNOSTIC_HEIGHTS_M = (0.0, 0.0025)
DEFAULT_PRE_SUBSTEPS = 8
DEFAULT_POST_SUBSTEPS = 12
SCHEDULE_DIAGNOSTIC_SCHEMA_VERSION = 1


def _git_dirty(path: Path) -> bool:
  result = subprocess.run(
    ['git', 'status', '--porcelain'], cwd=path, check=False,
    capture_output=True, text=True,
  )
  if result.returncode != 0:
    raise RuntimeError(f'Cannot inspect Git status for {path}.')
  return bool(result.stdout.strip())


def _outside_repository(path: Path) -> Path:
  output = path.resolve()
  try:
    output.relative_to(rb.REPOSITORY_PATH.resolve())
  except ValueError:
    return output
  raise ValueError('Diagnostic output must remain outside the Git checkout.')


def _validate_positive(value: int, *, name: str) -> int:
  if isinstance(value, bool) or value < 1:
    raise ValueError(f'{name} must be positive.')
  return value


def posture_grid(
  posture_payload: Mapping[str, Any], *, pitch_count: int,
) -> tuple[dict[str, float | str], ...]:
  count = _validate_positive(pitch_count, name='pitch_count')
  criteria = posture_payload.get('fit_criteria')
  envelope = posture_payload.get('training_envelope')
  if not isinstance(criteria, Mapping) or not isinstance(envelope, Mapping):
    raise TypeError('Posture artifact has no registered envelope.')
  heights_value = criteria.get('fixed_height_nodes')
  pitch_value = envelope.get('pitch')
  if (
    not isinstance(heights_value, Sequence)
    or isinstance(heights_value, (str, bytes))
    or not isinstance(pitch_value, Sequence)
    or isinstance(pitch_value, (str, bytes))
    or len(pitch_value) != 2
  ):
    raise ValueError('Posture artifact envelope is malformed.')
  heights = tuple(float(value) for value in heights_value)
  pitch_min, pitch_max = (float(value) for value in pitch_value)
  if (
    not heights
    or any(not math.isfinite(value) for value in heights)
    or not math.isfinite(pitch_min)
    or not math.isfinite(pitch_max)
    or pitch_max < pitch_min
  ):
    raise ValueError('Posture artifact envelope is not finite and ordered.')
  if count == 1:
    pitches = (0.5 * (pitch_min + pitch_max),)
  else:
    pitches = tuple(
      pitch_min + index * (pitch_max - pitch_min) / (count - 1)
      for index in range(count)
    )
  result = []
  for height in heights:
    for pitch in pitches:
      height_um = round(height * 1_000_000)
      pitch_urad = round(pitch * 1_000_000)
      result.append({
        'name': f'grid_h{height_um:06d}um_p{pitch_urad:+07d}urad',
        'height_m': height,
        'pitch_rad': pitch,
      })
  return tuple(result)


def summarize_trials(trials: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
  summaries = []
  authority_metrics = (
    'applied_residual_abs_max',
    'wheel_target_classical_path_abs_max_radps',
    'dynamic_leg_feedforward_abs_max_rad',
    'dynamic_drive_feedforward_abs_max_radps',
  )
  for height in DIAGNOSTIC_HEIGHTS_M:
    rows = [row for row in trials if float(row['stair_height_m']) == height]
    if not rows:
      raise ValueError(f'Diagnostic produced no trials at {height} m.')
    progresses = sorted(float(row['max_progress_past_face_m']) for row in rows)
    midpoint = len(progresses) // 2
    median_progress = (
      progresses[midpoint]
      if len(progresses) % 2
      else 0.5 * (progresses[midpoint - 1] + progresses[midpoint])
    )
    summary = {
      'stair_height_m': height,
      'trials': len(rows),
      'successes': sum(bool(row['success']) for row in rows),
      'bilateral_airborne_trials': sum(
        bool(row['bilateral_airborne_ever']) for row in rows
      ),
      'bilateral_unsupported_physics_substeps': sum(
        int(row['bilateral_unsupported_physics_substeps']) for row in rows
      ),
      'non_wheel_contact_trials': sum(
        bool(row['non_wheel_contact']) for row in rows
      ),
      'terminated_trials': sum(bool(row['termination']) for row in rows),
      'maximum_progress_m': max(progresses),
      'mean_progress_m': sum(progresses) / len(progresses),
      'median_progress_m': median_progress,
      'peak_pitch_rate_abs_max_radps': max(
        float(row.get('peak_pitch_rate_abs_radps', 0.0)) for row in rows
      ),
      'torque_saturation_fraction_mean': sum(
        float(row.get('torque_saturation_fraction') or 0.0) for row in rows
      ) / len(rows),
      'wheel_residual_abs_max': max(
        float(row.get('wheel_residual_abs_max', 0.0)) for row in rows
      ),
    }
    for metric in authority_metrics:
      present = [metric in row for row in rows]
      if any(present) and not all(present):
        raise ValueError(f'Diagnostic authority metric {metric} is partially missing.')
      if all(present):
        summary[metric] = max(float(row[metric]) for row in rows)
    summaries.append(summary)
  return summaries


class EventWindowCollector:
  def __init__(self, env_count: int, *, pre_substeps: int, post_substeps: int):
    self._pre = _validate_positive(pre_substeps, name='pre_substeps')
    self._post = _validate_positive(post_substeps, name='post_substeps')
    self._buffers = [deque(maxlen=self._pre) for _ in range(env_count)]
    self._pending: dict[int, tuple[dict[str, Any], int]] = {}
    self._triggered: set[int] = set()
    self.events: list[dict[str, Any]] = []

  def observe(
    self, env_id: int, sample: dict[str, Any], *, active: bool, unsupported: bool,
  ) -> None:
    if not active:
      return
    pending = self._pending.get(env_id)
    if pending is not None:
      event, remaining = pending
      event['samples'].append(sample)
      remaining -= 1
      if remaining == 0:
        del self._pending[env_id]
      else:
        self._pending[env_id] = event, remaining
      return
    if unsupported and env_id not in self._triggered:
      event = {
        'env_id': env_id,
        'trigger_substep': int(sample['substep']),
        'pre_substeps_requested': self._pre,
        'post_substeps_requested': self._post,
        'samples': [*self._buffers[env_id], sample],
      }
      self.events.append(event)
      self._triggered.add(env_id)
      self._pending[env_id] = event, self._post
      return
    self._buffers[env_id].append(sample)


def _tensor_columns(
  names: list[str], tensors: list[tuple[str, torch.Tensor]],
) -> torch.Tensor:
  columns = []
  for prefix, tensor in tensors:
    value = tensor.detach()
    if value.ndim == 1:
      value = value.unsqueeze(1)
    value = value.reshape(value.shape[0], -1)
    columns.append(value.float())
    if value.shape[1] == 1:
      names.append(prefix)
    else:
      names.extend(f'{prefix}_{index}' for index in range(value.shape[1]))
  return torch.cat(columns, dim=1)


def _event_samples(
  env: ManagerBasedRlEnv, *, heights: tuple[float, ...], face_x: torch.Tensor,
  substep: int, settle_steps: int, wheel_geom_ids: Any,
) -> tuple[list[dict[str, Any]], list[bool]]:
  robot = env.scene['robot']
  term = env.action_manager.get_term('hybrid_wheel_leg')
  left_data = env.scene[rb.LEFT_SENSOR].data
  right_data = env.scene[rb.RIGHT_SENSOR].data
  left_force = torch.linalg.vector_norm(left_data.force, dim=-1).sum(dim=-1)
  right_force = torch.linalg.vector_norm(right_data.force, dim=-1).sum(dim=-1)
  unsupported = (left_force <= 0.0) & (right_force <= 0.0)
  pitch, roll = rb._pitch_roll(robot)
  terrain_types = env.scene.terrain.terrain_types
  terrain_heights = torch.tensor(
    heights, device=env.device, dtype=robot.data.root_link_pos_w.dtype,
  )[terrain_types]
  wheel_center_x = robot.data.geom_pos_w[:, wheel_geom_ids, 0]
  horizontal_level = torch.where(
    wheel_center_x >= face_x.unsqueeze(1),
    torch.floor((wheel_center_x - face_x.unsqueeze(1)) / rb.STEP_WIDTH_M) + 1.0,
    torch.zeros_like(wheel_center_x),
  )
  inner_half_width = 0.5 * (
    rb.TERRAIN_SIZE_M[0] - 2.0 * rb.TERRAIN_BORDER_WIDTH_M
  )
  maximum_level = math.floor(
    (inner_half_width - 0.5 * rb.PLATFORM_WIDTH_M) / rb.STEP_WIDTH_M + 1.0e-9
  )
  horizontal_level.clamp_(min=0.0, max=float(maximum_level))
  horizontal_surface = horizontal_level * terrain_heights.unsqueeze(1)
  horizontal_clearance = rb.wheel_clearance_above_flat_m(env) - horizontal_surface
  names: list[str] = []
  matrix = _tensor_columns(names, [
    ('terrain_index', terrain_types),
    ('terrain_height_m', terrain_heights),
    ('progress_m', robot.data.root_link_pos_w[:, 0] - face_x),
    ('root_z_m', robot.data.root_link_pos_w[:, 2]),
    ('root_vx_mps', robot.data.root_link_lin_vel_w[:, 0]),
    ('root_vz_mps', robot.data.root_link_lin_vel_w[:, 2]),
    ('pitch_rad', pitch),
    ('roll_rad', roll),
    ('pitch_rate_radps', robot.data.root_link_ang_vel_b[:, 1]),
    ('left_force_n', left_force),
    ('right_force_n', right_force),
    ('wheel_center_x_m', wheel_center_x),
    ('horizontal_surface_z_m', horizontal_surface),
    ('horizontal_clearance_m', horizontal_clearance),
    ('wheel_speed_radps', robot.data.joint_vel[:, term._wheel_ids]),
    ('wheel_target_radps', term.wheel_targets),
    ('wheel_actuator_force_nm', robot.data.actuator_force[:, term._wheel_ids]),
    ('controller_baseline_radps', term.controller_baseline),
    ('classical_error', term.classical_errors),
    ('leg_position_rad', robot.data.joint_pos[:, term._leg_ids]),
    ('leg_velocity_radps', robot.data.joint_vel[:, term._leg_ids]),
    ('nominal_leg_target_rad', term.nominal_leg_targets),
    ('leg_reference_rad', term.leg_reference),
    ('leg_target_rad', term.leg_targets),
    ('leg_actuator_force_nm', robot.data.actuator_force[:, term._leg_ids]),
    ('applied_residual', term.applied_residual),
  ])
  values = matrix.cpu().tolist()
  control_substeps = rb.ROLL_FIRST_CONTROL_DECIMATION
  phase = 'settle' if substep <= settle_steps * control_substeps else 'drive'
  samples = []
  for row in values:
    sample = dict(zip(names, row, strict=True))
    sample.update({
      'substep': substep,
      'control_step': (substep - 1) // control_substeps,
      'phase': phase,
    })
    samples.append(sample)
  return samples, unsupported.cpu().tolist()


def _install_event_recorder(
  env: ManagerBasedRlEnv, *, heights: tuple[float, ...], settle_steps: int,
  collector: EventWindowCollector,
  base_installer=rb.install_strict_substep_support_recorder,
):
  state, restore = base_installer(env)
  inner_update = env.scene.update
  origins = env.scene.env_origins
  face_x = origins[:, 0] + rb.approach_geometry(0.0)['outer_face_x']
  counter = 0
  robot = env.scene['robot']
  wheel_geom_ids, wheel_names = robot.find_geoms(
    ('wheel_left_collision', 'wheel_right_collision'), preserve_order=True,
  )
  if tuple(wheel_names) != ('wheel_left_collision', 'wheel_right_collision'):
    raise RuntimeError('Diagnostic wheel geometry identity drifted.')

  def update(dt: float) -> None:
    nonlocal counter
    inner_update(dt)
    if not state['enabled']:
      return
    counter += 1
    samples, unsupported = _event_samples(
      env, heights=heights, face_x=face_x, substep=counter,
      settle_steps=settle_steps, wheel_geom_ids=wheel_geom_ids,
    )
    active = state['active_mask'].cpu().tolist()
    for env_id, sample in enumerate(samples):
      collector.observe(
        env_id, sample, active=bool(active[env_id]),
        unsupported=bool(unsupported[env_id]),
      )

  env.scene.update = update
  return state, restore


def _diagnostic_provenance(device: str) -> dict[str, Any]:
  mjlab_root = Path(mjlab.__file__).resolve().parents[2]
  return {
    'evidence_eligible': False,
    'promotion_eligible': False,
    'reason': 'diagnostic-only counterfactual; never RollBoundary evidence',
    'git_sha': rb._git_sha(rb.REPOSITORY_PATH),
    'mjlab_git_sha': rb._git_sha(mjlab_root),
    'device': device,
    'runtime': rb._runtime_metadata(device),
    'heights_m': list(DIAGNOSTIC_HEIGHTS_M),
    'physics_timestep_s': rb.ROLL_FIRST_PHYSICS_TIMESTEP_S,
    'control_decimation': rb.ROLL_FIRST_CONTROL_DECIMATION,
    'strict_support_scope': rb.ROLL_FIRST_SUBSTEP_SUPPORT_SCOPE,
    'wheel_residual_required_zero': True,
  }


def run_event_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
  cfg = rb.make_roll_boundary_env_cfg(
    DIAGNOSTIC_HEIGHTS_M, args.envs_per_height,
  )
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  runs = []
  original_installer = rb.install_strict_substep_support_recorder
  try:
    for card in rb.POSTURE_CARDS:
      for repeat in range(1, args.repeats + 1):
        print('[roll-boundary-diagnostic] card={} repeat={}'.format(card['name'], repeat))
        collector = EventWindowCollector(
          env.num_envs, pre_substeps=args.pre_substeps,
          post_substeps=args.post_substeps,
        )

        def installer(current_env, _collector=collector):
          return _install_event_recorder(
            current_env, heights=DIAGNOSTIC_HEIGHTS_M,
            settle_steps=args.settle_steps, collector=_collector,
            base_installer=original_installer,
          )

        rb.install_strict_substep_support_recorder = installer
        try:
          rows = rb.run_card_repeat(
            env, heights=DIAGNOSTIC_HEIGHTS_M, card=card, repeat=repeat,
            settle_steps=args.settle_steps, drive_steps=args.drive_steps,
            stable_steps=args.stable_steps, episode_wide_safety=True,
            diagnostic_continue_after_support_loss=True,
          )
        finally:
          rb.install_strict_substep_support_recorder = original_installer
        runs.append({
          'posture_card': dict(card),
          'repeat': repeat,
          'summaries': summarize_trials(rows),
          'trials': rows,
          'first_support_loss_events': collector.events,
        })
  finally:
    rb.install_strict_substep_support_recorder = original_installer
    env.close()
  return {
    'kind': 'roll_boundary_event_diagnostic',
    **_diagnostic_provenance(args.device),
    'continue_after_first_support_loss': True,
    'pre_substeps': args.pre_substeps,
    'post_substeps': args.post_substeps,
    'envs_per_height': args.envs_per_height,
    'repeats': args.repeats,
    'settle_steps': args.settle_steps,
    'drive_steps': args.drive_steps,
    'stable_steps': args.stable_steps,
    'runs': runs,
  }


def run_posture_grid(args: argparse.Namespace) -> dict[str, Any]:
  artifact = rb.frozen_artifact_paths()['posture_map_path']
  posture_payload = json.loads(artifact.read_text(encoding='utf-8-sig'))
  candidates = posture_grid(posture_payload, pitch_count=args.pitch_count)
  cfg = rb.make_roll_boundary_env_cfg(
    DIAGNOSTIC_HEIGHTS_M, args.envs_per_height,
  )
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  original_cards = rb.POSTURE_CARDS
  results = []
  try:
    for card in candidates:
      print('[roll-boundary-posture-grid] height={:.10f} pitch={:+.6f}'.format(card['height_m'], card['pitch_rad']))
      rb.POSTURE_CARDS = (card,)
      rows = []
      for repeat in range(1, args.repeats + 1):
        rows.extend(rb.run_card_repeat(
          env, heights=DIAGNOSTIC_HEIGHTS_M, card=card, repeat=repeat,
          settle_steps=args.settle_steps, drive_steps=args.drive_steps,
          stable_steps=args.stable_steps, episode_wide_safety=True,
        ))
      results.append({
        'posture_card': dict(card),
        'summaries': summarize_trials(rows),
        'trials': rows,
      })
  finally:
    rb.POSTURE_CARDS = original_cards
    env.close()
  return {
    'kind': 'roll_boundary_posture_grid_diagnostic',
    **_diagnostic_provenance(args.device),
    'matched_reset_perturbations_across_candidates': True,
    'posture_artifact': str(artifact.relative_to(rb.REPOSITORY_PATH)),
    'posture_artifact_file_sha256': rb.ARTIFACT_SPECS['posture_map_path'][1],
    'pitch_count': args.pitch_count,
    'candidate_count': len(candidates),
    'envs_per_height': args.envs_per_height,
    'repeats': args.repeats,
    'settle_steps': args.settle_steps,
    'drive_steps': args.drive_steps,
    'stable_steps': args.stable_steps,
    'candidates': results,
  }


def schedule_grid_candidates() -> tuple[dict[str, Any], ...]:
  """Return the frozen twelve schedules plus two static regression sentinels."""

  candidates = [
    {
      'name': schedule.name,
      'kind': 'position_indexed_schedule',
      'posture_card': {
        'name': schedule.name,
        'height_m': schedule.start_height_m,
        'pitch_rad': schedule.start_pitch_rad,
      },
      'schedule': schedule,
    }
    for schedule in roll_pose_schedule_candidates()
  ]
  candidates.extend((
    {
      'name': 'static_low_h290732um_p-032000urad',
      'kind': 'static_regression_sentinel',
      'posture_card': {
        'name': 'static_low_h290732um_p-032000urad',
        'height_m': 0.2907321708,
        'pitch_rad': -0.032,
      },
      'schedule': None,
    },
    {
      'name': 'static_high_h327686um_p+032000urad',
      'kind': 'static_regression_sentinel',
      'posture_card': {
        'name': 'static_high_h327686um_p+032000urad',
        'height_m': 0.3276857266,
        'pitch_rad': 0.032,
      },
      'schedule': None,
    },
  ))
  result = tuple(candidates)
  if len(result) != 14 or len({item['name'] for item in result}) != 14:
    raise RuntimeError('The frozen schedule screen must contain 14 unique candidates.')
  return result


def _schedule_candidate_definition(candidate: Mapping[str, Any]) -> dict[str, Any]:
  schedule = candidate['schedule']
  return {
    'name': str(candidate['name']),
    'kind': str(candidate['kind']),
    'posture_card': dict(candidate['posture_card']),
    'schedule': None if schedule is None else schedule.to_dict(),
  }


def run_schedule_grid(args: argparse.Namespace) -> dict[str, Any]:
  artifact = rb.frozen_artifact_paths()['posture_map_path']
  candidates = schedule_grid_candidates()
  cfg = rb.make_roll_boundary_env_cfg(
    DIAGNOSTIC_HEIGHTS_M, args.envs_per_height,
  )
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  original_cards = rb.POSTURE_CARDS
  original_installer = rb.install_strict_substep_support_recorder
  results = []
  try:
    for candidate in candidates:
      card = candidate['posture_card']
      schedule: RollPoseSchedule | None = candidate['schedule']
      print('[roll-boundary-schedule-grid] candidate={}'.format(candidate['name']))
      rb.POSTURE_CARDS = (card,)
      rows = []
      events = []
      for repeat in range(1, args.repeats + 1):
        collector = EventWindowCollector(
          env.num_envs,
          pre_substeps=args.pre_substeps,
          post_substeps=args.post_substeps,
        )

        def installer(current_env, _collector=collector):
          return _install_event_recorder(
            current_env,
            heights=DIAGNOSTIC_HEIGHTS_M,
            settle_steps=args.settle_steps,
            collector=_collector,
            base_installer=original_installer,
          )

        rb.install_strict_substep_support_recorder = installer
        try:
          rows.extend(rb.run_card_repeat(
            env,
            heights=DIAGNOSTIC_HEIGHTS_M,
            card=card,
            repeat=repeat,
            settle_steps=args.settle_steps,
            drive_steps=args.drive_steps,
            stable_steps=args.stable_steps,
            episode_wide_safety=True,
            diagnostic_continue_after_support_loss=True,
            roll_pose_schedule=schedule,
            require_pure_classical_authority=True,
          ))
        finally:
          rb.install_strict_substep_support_recorder = original_installer
        events.extend(collector.events)
      results.append({
        'candidate_definition': _schedule_candidate_definition(candidate),
        'summaries': summarize_trials(rows),
        'trials': rows,
        'first_support_loss_events': events,
      })
  finally:
    rb.POSTURE_CARDS = original_cards
    rb.install_strict_substep_support_recorder = original_installer
    env.close()
  return {
    'schema_version': SCHEDULE_DIAGNOSTIC_SCHEMA_VERSION,
    'kind': 'roll_pose_schedule_grid_diagnostic',
    **_diagnostic_provenance(args.device),
    'matched_reset_perturbations_across_candidates': True,
    'continue_after_first_support_loss': True,
    'policy_action_required_zero': True,
    'all_applied_residual_required_zero': True,
    'posture_artifact': str(artifact.relative_to(rb.REPOSITORY_PATH)),
    'posture_artifact_file_sha256': rb.ARTIFACT_SPECS['posture_map_path'][1],
    'candidate_count': len(candidates),
    'candidate_definitions': [
      _schedule_candidate_definition(candidate) for candidate in candidates
    ],
    'envs_per_height': args.envs_per_height,
    'repeats': args.repeats,
    'settle_steps': args.settle_steps,
    'drive_steps': args.drive_steps,
    'stable_steps': args.stable_steps,
    'pre_substeps': args.pre_substeps,
    'post_substeps': args.post_substeps,
    'candidates': results,
  }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description='Non-evidentiary RollBoundary diagnostics.'
  )
  parser.add_argument(
    '--mode', choices=('events', 'posture-grid', 'schedule-grid'), required=True,
  )
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--device', default='cpu')
  parser.add_argument('--envs-per-height', type=int, default=rb.OFFICIAL_ENVS_PER_HEIGHT)
  parser.add_argument('--repeats', type=int, default=rb.OFFICIAL_REPEATS)
  parser.add_argument('--settle-steps', type=int, default=rb.OFFICIAL_SETTLE_STEPS)
  parser.add_argument('--drive-steps', type=int, default=rb.OFFICIAL_DRIVE_STEPS)
  parser.add_argument('--stable-steps', type=int, default=rb.OFFICIAL_STABLE_STEPS)
  parser.add_argument('--pre-substeps', type=int, default=DEFAULT_PRE_SUBSTEPS)
  parser.add_argument('--post-substeps', type=int, default=DEFAULT_POST_SUBSTEPS)
  parser.add_argument('--pitch-count', type=int, default=5)
  parser.add_argument('--allow-dirty', action='store_true')
  args = parser.parse_args(argv)
  for name in (
    'envs_per_height', 'repeats', 'settle_steps', 'drive_steps', 'stable_steps',
    'pre_substeps', 'post_substeps', 'pitch_count',
  ):
    _validate_positive(getattr(args, name), name=name)
  return args


def main(argv: Sequence[str] | None = None) -> None:
  args = parse_args(argv)
  output = _outside_repository(args.output)
  if _git_dirty(rb.REPOSITORY_PATH) and not args.allow_dirty:
    raise RuntimeError('Diagnostic requires a clean project checkout.')
  mjlab_root = Path(mjlab.__file__).resolve().parents[2]
  if _git_dirty(mjlab_root) and not args.allow_dirty:
    raise RuntimeError('Diagnostic requires a clean MjLab checkout.')
  if args.mode == 'events':
    payload = run_event_diagnostic(args)
  elif args.mode == 'posture-grid':
    payload = run_posture_grid(args)
  else:
    payload = run_schedule_grid(args)
  output.parent.mkdir(parents=True, exist_ok=True)
  rb._atomic_write_json(output, payload)
  print(f'[roll-boundary-diagnostic] output={output}')


if __name__ == '__main__':
  main()
