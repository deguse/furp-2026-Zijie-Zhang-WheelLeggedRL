#!/usr/bin/env python3
# Non-evidentiary diagnostics for the first positive RollBoundary tier.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import deque
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import mjlab
import torch
from mjlab.envs import ManagerBasedRlEnv

from hoppertrex_mjlab.hybrid.roll_pose_schedule import (
  CONTROL_DT_S,
  INDEPENDENT_SLEW_MODE,
  POSTURE_HEIGHT_SLEW_RATE_MPS,
  POSTURE_PITCH_SLEW_RATE_RADPS,
  SYNCHRONIZED_SLEW_MODE,
  RollPoseSchedule,
  roll_pose_schedule_candidates,
)
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb

DIAGNOSTIC_HEIGHTS_M = (0.0, 0.0025)
DEFAULT_PRE_SUBSTEPS = 8
DEFAULT_POST_SUBSTEPS = 12
SCHEDULE_DIAGNOSTIC_SCHEMA_VERSION = 3
R0C_SYNC_SCHEMA_VERSION = 1
R0C_SYNC_ENVS_PER_HEIGHT = 8
R0C_SYNC_REPEATS = 1
R0C_SYNC_PASS_SUCCESSES = 7
R0C_SYNC_BASELINE_SUCCESS_RANGE = (2, 4)
R0C_SYNC_BASE_SCHEDULE_NAME = 'roll_pose_sa_cd_d030mm'
SCHEDULE_AUTHORITY_METRICS = (
  'applied_residual_abs_max',
  'wheel_target_classical_path_abs_max_radps',
  'dynamic_leg_feedforward_abs_max_rad',
  'dynamic_drive_feedforward_abs_max_radps',
)
SCHEDULE_METADATA_FIELDS = (
  'roll_pose_schedule',
  'drive_start_x_m',
  'end_distance_to_riser_m',
  'schedule_slew_mode',
  'schedule_alpha_max',
  'schedule_nominal_alpha_final',
  'schedule_applied_alpha_final',
  'schedule_applied_height_alpha_final',
  'schedule_applied_pitch_alpha_final',
  'maximum_applied_channel_alpha_gap',
  'desired_height_m_final',
  'desired_pitch_rad_final',
  'applied_height_m_final',
  'applied_pitch_rad_final',
  'maximum_height_tracking_lag_m',
  'maximum_pitch_tracking_lag_rad',
  'height_transition_completion_step',
  'pitch_transition_completion_step',
  'transition_completion_step',
  'transition_completed_before_face',
)
SCHEDULE_CONTROL_TRACE_FIELDS = (
  'control_step',
  'progress_m',
  'root_z_m',
  'root_vz_mps',
  'pitch_rad',
  'pitch_rate_radps',
  'left_vertical_normal_load_n',
  'right_vertical_normal_load_n',
  'total_vertical_normal_load_n',
  'schedule_nominal_alpha',
  'schedule_applied_alpha',
  'schedule_applied_height_alpha',
  'schedule_applied_pitch_alpha',
  'applied_height_m',
  'applied_pitch_rad',
)


def _git_dirty(path: Path) -> bool:
  result = subprocess.run(
    ['git', 'status', '--porcelain'], cwd=path, check=False,
    capture_output=True, text=True,
  )
  if result.returncode != 0:
    raise RuntimeError(f'Cannot inspect Git status for {path}.')
  return bool(result.stdout.strip())


def _git_output(path: Path, *args: str) -> bytes:
  result = subprocess.run(
    ['git', *args], cwd=path, check=False, capture_output=True,
  )
  if result.returncode != 0:
    raise RuntimeError(f'Cannot run git {" ".join(args)} in {path}.')
  return result.stdout


def _git_worktree_fingerprint(path: Path) -> str:
  digest = hashlib.sha256()
  digest.update(_git_output(path, 'rev-parse', 'HEAD'))
  digest.update(_git_output(path, 'diff', '--binary', 'HEAD'))
  untracked = _git_output(
    path, 'ls-files', '--others', '--exclude-standard', '-z',
  ).split(b'\0')
  for relative_bytes in sorted(item for item in untracked if item):
    digest.update(relative_bytes)
    candidate = path / os.fsdecode(relative_bytes)
    if candidate.is_file():
      digest.update(hashlib.sha256(candidate.read_bytes()).digest())
  return digest.hexdigest()


def _file_sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _diagnostic_source_hashes() -> dict[str, str]:
  paths = {
    'diagnose_roll_boundary.py': Path(__file__).resolve(),
    'probe_roll_boundary.py': Path(rb.__file__).resolve(),
    'roll_pose_schedule.py': (
      Path(__file__).resolve().parents[1] / 'hybrid' / 'roll_pose_schedule.py'
    ),
  }
  return {name: _file_sha256(path) for name, path in paths.items()}


def _reserve_output(path: Path) -> Path:
  temporary = path.with_name(f'.{path.name}.incomplete')
  reservation = path.with_name(f'.{path.name}.reserved')
  for candidate in (path, temporary, reservation):
    if candidate.exists():
      raise FileExistsError(f'Diagnostic output path is already occupied: {candidate}')
  path.parent.mkdir(parents=True, exist_ok=True)
  reservation.touch(exist_ok=False)
  return reservation


def _outside_repository(path: Path) -> Path:
  output = path.resolve()
  repositories = (
    rb.REPOSITORY_PATH.resolve(),
    Path(mjlab.__file__).resolve().parents[2],
  )
  for repository in repositories:
    try:
      output.relative_to(repository)
    except ValueError:
      continue
    raise ValueError('Diagnostic output must remain outside the Git checkout.')
  return output


def _validate_positive(value: int, *, name: str) -> int:
  if isinstance(value, bool) or value < 1:
    raise ValueError(f'{name} must be positive.')
  return value


def _required_finite_float(row: Mapping[str, Any], name: str) -> float:
  if (
    name not in row
    or isinstance(row[name], bool)
    or not isinstance(row[name], Real)
  ):
    raise ValueError(f'Diagnostic trial requires numeric {name}.')
  value = float(row[name])
  if not math.isfinite(value):
    raise ValueError(f'Diagnostic trial {name} must be finite.')
  return value


def _required_bool(row: Mapping[str, Any], name: str) -> bool:
  if name not in row or type(row[name]) is not bool:
    raise ValueError(f'Diagnostic trial requires boolean {name}.')
  return row[name]


def _required_int(row: Mapping[str, Any], name: str, *, minimum: int) -> int:
  if (
    name not in row
    or isinstance(row[name], bool)
    or not isinstance(row[name], Integral)
  ):
    raise ValueError(f'Diagnostic trial requires integer {name}.')
  value = int(row[name])
  if value < minimum:
    raise ValueError(f'Diagnostic trial {name} must be at least {minimum}.')
  return value


def _optional_finite_float(row: Mapping[str, Any], name: str) -> float | None:
  if name not in row:
    raise ValueError(f'Diagnostic trial requires {name}.')
  if row[name] is None:
    return None
  return _required_finite_float(row, name)


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
  authority_metrics = SCHEDULE_AUTHORITY_METRICS
  for height in DIAGNOSTIC_HEIGHTS_M:
    rows = [
      row for row in trials
      if _required_finite_float(row, 'stair_height_m') == height
    ]
    if not rows:
      raise ValueError(f'Diagnostic produced no trials at {height} m.')
    progresses = sorted(
      _required_finite_float(row, 'max_progress_past_face_m') for row in rows
    )
    pitch_rates = [
      _required_finite_float(row, 'peak_pitch_rate_abs_radps') for row in rows
    ]
    saturation = [
      _required_finite_float(row, 'torque_saturation_fraction') for row in rows
    ]
    wheel_residual = [
      _required_finite_float(row, 'wheel_residual_abs_max') for row in rows
    ]
    midpoint = len(progresses) // 2
    median_progress = (
      progresses[midpoint]
      if len(progresses) % 2
      else 0.5 * (progresses[midpoint - 1] + progresses[midpoint])
    )
    unsafe_trials = sum(
      bool(row['bilateral_airborne_ever'])
      or bool(row['non_wheel_contact'])
      or bool(row['termination'])
      for row in rows
    )
    successes = sum(bool(row['success']) for row in rows)
    summary = {
      'stair_height_m': height,
      'trials': len(rows),
      'successes': successes,
      'unsafe_trials': unsafe_trials,
      'safe_stalls': len(rows) - successes - unsafe_trials,
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
      'peak_pitch_rate_abs_max_radps': max(pitch_rates),
      'torque_saturation_fraction_mean': sum(saturation) / len(rows),
      'wheel_residual_abs_max': max(wheel_residual),
    }
    for metric in authority_metrics:
      present = [metric in row for row in rows]
      if any(present) and not all(present):
        raise ValueError(f'Diagnostic authority metric {metric} is partially missing.')
      if all(present):
        summary[metric] = max(_required_finite_float(row, metric) for row in rows)
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

  def finalize(self) -> list[dict[str, Any]]:
    if self._pending:
      raise ValueError(
        f'Event windows ended before post samples completed: {sorted(self._pending)}'
      )
    expected_samples = self._pre + 1 + self._post
    for event in self.events:
      samples = event.get('samples')
      if not isinstance(samples, list) or len(samples) != expected_samples:
        raise ValueError('Event window does not contain the requested sample count.')
      trigger = _required_int(event, 'trigger_substep', minimum=1)
      if _required_int(samples[self._pre], 'substep', minimum=1) != trigger:
        raise ValueError('Event trigger is not aligned with its sample window.')
      observed = [_required_int(sample, 'substep', minimum=1) for sample in samples]
      if observed != list(range(trigger - self._pre, trigger + self._post + 1)):
        raise ValueError('Event-window physics substeps are not consecutive.')
    return self.events


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
  support_state: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[bool]]:
  robot = env.scene['robot']
  term = env.action_manager.get_term('hybrid_wheel_leg')
  left_data = env.scene[rb.LEFT_SENSOR].data
  right_data = env.scene[rb.RIGHT_SENSOR].data
  left_force = torch.linalg.vector_norm(left_data.force, dim=-1).sum(dim=-1)
  right_force = torch.linalg.vector_norm(right_data.force, dim=-1).sum(dim=-1)
  left_vertical_load = rb.vertical_normal_load_n(
    found=left_data.found,
    force_contact_frame=left_data.force,
    normal_global=left_data.normal,
  )
  right_vertical_load = rb.vertical_normal_load_n(
    found=right_data.found,
    force_contact_frame=right_data.force,
    normal_global=right_data.normal,
  )
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
    ('left_vertical_normal_load_n', left_vertical_load),
    ('right_vertical_normal_load_n', right_vertical_load),
    ('total_vertical_normal_load_n', left_vertical_load + right_vertical_load),
    ('schedule_nominal_alpha', support_state['schedule_nominal_alpha']),
    ('schedule_applied_alpha', support_state['schedule_applied_alpha']),
    (
      'schedule_applied_height_alpha',
      support_state['schedule_applied_height_alpha'],
    ),
    (
      'schedule_applied_pitch_alpha',
      support_state['schedule_applied_pitch_alpha'],
    ),
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
  episode_control_step = (substep - 1) // control_substeps + 1
  phase = 'settle' if episode_control_step <= settle_steps else 'drive'
  drive_control_step = (
    None if phase == 'settle' else episode_control_step - settle_steps
  )
  samples = []
  for row in values:
    sample = dict(zip(names, row, strict=True))
    sample.update({
      'substep': substep,
      'episode_control_step': episode_control_step,
      'drive_control_step': drive_control_step,
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
      support_state=state,
    )
    active = state['active_mask'].cpu().tolist()
    for env_id, sample in enumerate(samples):
      collector.observe(
        env_id, sample, active=bool(active[env_id]),
        unsupported=bool(unsupported[env_id]),
      )

  env.scene.update = update
  return state, restore


def _diagnostic_provenance(
  device: str, *, allow_dirty: bool = False,
  project_dirty: bool | None = None, mjlab_dirty: bool | None = None,
) -> dict[str, Any]:
  mjlab_root = Path(mjlab.__file__).resolve().parents[2]
  project_dirty = (
    _git_dirty(rb.REPOSITORY_PATH) if project_dirty is None else project_dirty
  )
  mjlab_dirty = _git_dirty(mjlab_root) if mjlab_dirty is None else mjlab_dirty
  return {
    'evidence_eligible': False,
    'promotion_eligible': False,
    'reason': 'diagnostic-only counterfactual; never RollBoundary evidence',
    'git_sha': rb._git_sha(rb.REPOSITORY_PATH),
    'mjlab_git_sha': rb._git_sha(mjlab_root),
    'allow_dirty': bool(allow_dirty),
    'project_dirty': bool(project_dirty),
    'mjlab_dirty': bool(mjlab_dirty),
    'project_worktree_fingerprint': _git_worktree_fingerprint(rb.REPOSITORY_PATH),
    'mjlab_worktree_fingerprint': _git_worktree_fingerprint(mjlab_root),
    'source_file_sha256': _diagnostic_source_hashes(),
    'device': device,
    'runtime': rb._runtime_metadata(device),
    'heights_m': list(DIAGNOSTIC_HEIGHTS_M),
    'physics_timestep_s': rb.ROLL_FIRST_PHYSICS_TIMESTEP_S,
    'control_decimation': rb.ROLL_FIRST_CONTROL_DECIMATION,
    'strict_support_scope': rb.ROLL_FIRST_SUBSTEP_SUPPORT_SCOPE,
    'wheel_residual_required_zero': True,
  }


def _verify_provenance_unchanged(provenance: Mapping[str, Any]) -> None:
  mjlab_root = Path(mjlab.__file__).resolve().parents[2]
  current = {
    'git_sha': rb._git_sha(rb.REPOSITORY_PATH),
    'mjlab_git_sha': rb._git_sha(mjlab_root),
    'project_dirty': _git_dirty(rb.REPOSITORY_PATH),
    'mjlab_dirty': _git_dirty(mjlab_root),
    'project_worktree_fingerprint': _git_worktree_fingerprint(rb.REPOSITORY_PATH),
    'mjlab_worktree_fingerprint': _git_worktree_fingerprint(mjlab_root),
    'source_file_sha256': _diagnostic_source_hashes(),
  }
  for name, value in current.items():
    if provenance.get(name) != value:
      raise RuntimeError(f'Diagnostic provenance changed during execution: {name}.')


def _run_provenance(
  args: argparse.Namespace, provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
  if provenance is not None:
    return dict(provenance)
  return _diagnostic_provenance(
    args.device, allow_dirty=bool(getattr(args, 'allow_dirty', False)),
  )


def run_event_diagnostic(
  args: argparse.Namespace, provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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
          'first_support_loss_events': collector.finalize(),
        })
  finally:
    rb.install_strict_substep_support_recorder = original_installer
    env.close()
  return {
    'kind': 'roll_boundary_event_diagnostic',
    **_run_provenance(args, provenance),
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


def run_posture_grid(
  args: argparse.Namespace, provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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
    **_run_provenance(args, provenance),
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
      'slew_mode': INDEPENDENT_SLEW_MODE,
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
      'slew_mode': INDEPENDENT_SLEW_MODE,
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
      'slew_mode': INDEPENDENT_SLEW_MODE,
    },
  ))
  result = tuple(candidates)
  if len(result) != 14 or len({item['name'] for item in result}) != 14:
    raise RuntimeError('The frozen schedule screen must contain 14 unique candidates.')
  return result


def _r0c_sync_protocol_drift(args: argparse.Namespace) -> dict[str, tuple[Any, Any]]:
  expected = {
    'device': 'cuda:0',
    'envs_per_height': R0C_SYNC_ENVS_PER_HEIGHT,
    'repeats': R0C_SYNC_REPEATS,
    'settle_steps': rb.OFFICIAL_SETTLE_STEPS,
    'drive_steps': rb.OFFICIAL_DRIVE_STEPS,
    'stable_steps': rb.OFFICIAL_STABLE_STEPS,
    'pre_substeps': DEFAULT_PRE_SUBSTEPS,
    'post_substeps': DEFAULT_POST_SUBSTEPS,
  }
  return {
    name: (getattr(args, name), value)
    for name, value in expected.items()
    if getattr(args, name) != value
  }


def r0c_sync_candidates() -> tuple[dict[str, Any], ...]:
  """Return the frozen C0/C1 synchronized-reference rejection screen."""

  schedule = next(
    item for item in roll_pose_schedule_candidates()
    if item.name == R0C_SYNC_BASE_SCHEDULE_NAME
  )
  specifications = (
    (
      'r0c_sync_c0_independent_sa_cd_d030mm',
      'legacy_independent_slew_baseline',
      INDEPENDENT_SLEW_MODE,
    ),
    (
      'r0c_sync_c1_synchronized_sa_cd_d030mm',
      'shared_alpha_synchronized_slew',
      SYNCHRONIZED_SLEW_MODE,
    ),
  )
  result = tuple({
    'name': name,
    'kind': kind,
    'posture_card': {
      'name': name,
      'height_m': schedule.start_height_m,
      'pitch_rad': schedule.start_pitch_rad,
    },
    'schedule': schedule,
    'slew_mode': slew_mode,
  } for name, kind, slew_mode in specifications)
  if len(result) != 2 or len({item['name'] for item in result}) != 2:
    raise RuntimeError('R0c-SYNC requires two unique controller candidates.')
  return result


def _validate_r0c_sync_candidate_set(
  candidates: Sequence[Mapping[str, Any]],
) -> None:
  expected = (
    ('legacy_independent_slew_baseline', INDEPENDENT_SLEW_MODE),
    ('shared_alpha_synchronized_slew', SYNCHRONIZED_SLEW_MODE),
  )
  if len(candidates) != len(expected):
    raise ValueError('R0c-SYNC requires exactly two controller candidates.')
  for candidate, (kind, slew_mode) in zip(candidates, expected, strict=True):
    schedule = candidate.get('schedule')
    card = candidate.get('posture_card')
    if (
      candidate.get('kind') != kind
      or candidate.get('slew_mode') != slew_mode
      or not isinstance(schedule, RollPoseSchedule)
      or schedule.name != R0C_SYNC_BASE_SCHEDULE_NAME
      or not isinstance(card, Mapping)
      or card.get('name') != candidate.get('name')
      or float(card.get('height_m', math.nan)) != schedule.start_height_m
      or float(card.get('pitch_rad', math.nan)) != schedule.start_pitch_rad
    ):
      raise ValueError('R0c-SYNC candidate definitions drifted.')


def _schedule_candidate_definition(candidate: Mapping[str, Any]) -> dict[str, Any]:
  schedule = candidate['schedule']
  return {
    'name': str(candidate['name']),
    'kind': str(candidate['kind']),
    'posture_card': dict(candidate['posture_card']),
    'schedule': None if schedule is None else schedule.to_dict(),
    'slew_mode': None if schedule is None else str(candidate['slew_mode']),
  }


def _validate_schedule_candidate_set(
  candidates: Sequence[Mapping[str, Any]],
) -> None:
  names = [candidate.get('name') for candidate in candidates]
  if (
    len(candidates) != 14
    or any(not isinstance(name, str) or not name for name in names)
    or len(set(names)) != 14
  ):
    raise ValueError('Schedule diagnostic requires exactly 14 unique candidates.')
  dynamic_count = sum(
    candidate.get('kind') == 'position_indexed_schedule'
    for candidate in candidates
  )
  static_count = sum(
    candidate.get('kind') == 'static_regression_sentinel'
    for candidate in candidates
  )
  if dynamic_count != 12 or static_count != 2:
    raise ValueError('Schedule diagnostic requires twelve schedules and two sentinels.')
  for candidate in candidates:
    name = candidate['name']
    card = candidate.get('posture_card')
    schedule = candidate.get('schedule')
    if not isinstance(card, Mapping) or card.get('name') != name:
      raise ValueError(f'Schedule candidate {name} has inconsistent posture metadata.')
    if candidate['kind'] == 'position_indexed_schedule':
      if not isinstance(schedule, RollPoseSchedule) or schedule.name != name:
        raise ValueError(f'Schedule candidate {name} has an invalid schedule.')
      if candidate.get('slew_mode') != INDEPENDENT_SLEW_MODE:
        raise ValueError(f'Schedule candidate {name} changed its frozen slew mode.')
    elif schedule is not None:
      raise ValueError(f'Static schedule candidate {name} must not have a schedule.')


def _validate_schedule_control_trace(
  value: Any, *, schedule: RollPoseSchedule | None, slew_mode: str,
  drive_steps: int,
) -> None:
  if not isinstance(value, list) or not value:
    raise ValueError('Schedule control trace must be a nonempty list.')
  expected_keys = frozenset(SCHEDULE_CONTROL_TRACE_FIELDS)
  alpha_fields = (
    'schedule_nominal_alpha', 'schedule_applied_alpha',
    'schedule_applied_height_alpha', 'schedule_applied_pitch_alpha',
  )
  posture_fields = ('applied_height_m', 'applied_pitch_rad')
  previous_alphas = {field: 0.0 for field in alpha_fields}
  previous_height = None if schedule is None else schedule.start_height_m
  previous_pitch = None if schedule is None else schedule.start_pitch_rad
  if schedule is None:
    synchronized_alpha_step = None
  else:
    normalized_rates = []
    height_delta = schedule.climb_height_m - schedule.start_height_m
    pitch_delta = schedule.climb_pitch_rad - schedule.start_pitch_rad
    if height_delta != 0.0:
      normalized_rates.append(POSTURE_HEIGHT_SLEW_RATE_MPS / abs(height_delta))
    if pitch_delta != 0.0:
      normalized_rates.append(POSTURE_PITCH_SLEW_RATE_RADPS / abs(pitch_delta))
    synchronized_alpha_step = (
      1.0 if not normalized_rates else min(normalized_rates) * CONTROL_DT_S
    )
  for index, sample in enumerate(value, start=1):
    if not isinstance(sample, Mapping) or frozenset(sample) != expected_keys:
      raise ValueError('Schedule control trace schema drifted.')
    step = _required_int(sample, 'control_step', minimum=1)
    if step != index or step > drive_steps:
      raise ValueError('Schedule control trace steps must be a consecutive prefix.')
    for field in (
      'progress_m', 'root_z_m', 'root_vz_mps', 'pitch_rad',
      'pitch_rate_radps',
    ):
      _required_finite_float(sample, field)
    left = _required_finite_float(sample, 'left_vertical_normal_load_n')
    right = _required_finite_float(sample, 'right_vertical_normal_load_n')
    total = _required_finite_float(sample, 'total_vertical_normal_load_n')
    if left < 0.0 or right < 0.0 or total < 0.0:
      raise ValueError('Schedule control-trace vertical loads must be nonnegative.')
    if not math.isclose(total, left + right, rel_tol=1.0e-6, abs_tol=1.0e-5):
      raise ValueError('Schedule control-trace total vertical load is inconsistent.')
    if schedule is None:
      if any(sample[field] is not None for field in (*alpha_fields, *posture_fields)):
        raise ValueError('Static control-trace schedule fields must be null.')
      continue

    alphas = {}
    for field in alpha_fields:
      alpha = _required_finite_float(sample, field)
      if not 0.0 <= alpha <= 1.0:
        raise ValueError(f'Schedule control-trace alpha {field} left [0, 1].')
      if alpha + 1.0e-6 < previous_alphas[field]:
        raise ValueError('Schedule control-trace alpha rewound.')
      alphas[field] = alpha
    nominal_alpha = alphas['schedule_nominal_alpha']
    for field in alpha_fields[1:]:
      if alphas[field] > nominal_alpha + 1.0e-5:
        raise ValueError('Applied schedule alpha exceeded nominal progress.')
    if abs(
      alphas['schedule_applied_alpha']
      - min(
        alphas['schedule_applied_height_alpha'],
        alphas['schedule_applied_pitch_alpha'],
      )
    ) > 1.0e-5:
      raise ValueError('Joint applied alpha disagrees with its posture channels.')

    applied_height = _required_finite_float(sample, 'applied_height_m')
    applied_pitch = _required_finite_float(sample, 'applied_pitch_rad')
    expected_height = schedule.start_height_m + (
      alphas['schedule_applied_height_alpha']
      * (schedule.climb_height_m - schedule.start_height_m)
    )
    expected_pitch = schedule.start_pitch_rad + (
      alphas['schedule_applied_pitch_alpha']
      * (schedule.climb_pitch_rad - schedule.start_pitch_rad)
    )
    if not math.isclose(
      applied_height, expected_height, rel_tol=0.0, abs_tol=2.0e-6,
    ) or not math.isclose(
      applied_pitch, expected_pitch, rel_tol=0.0, abs_tol=2.0e-6,
    ):
      raise ValueError('Applied posture disagrees with its channel alpha.')
    assert previous_height is not None and previous_pitch is not None
    desired_height = schedule.start_height_m + nominal_alpha * (
      schedule.climb_height_m - schedule.start_height_m
    )
    desired_pitch = schedule.start_pitch_rad + nominal_alpha * (
      schedule.climb_pitch_rad - schedule.start_pitch_rad
    )
    if slew_mode == SYNCHRONIZED_SLEW_MODE:
      assert synchronized_alpha_step is not None
      expected_applied_alpha = min(
        nominal_alpha,
        previous_alphas['schedule_applied_alpha'] + synchronized_alpha_step,
      )
      expected_next_height = schedule.start_height_m + expected_applied_alpha * (
        schedule.climb_height_m - schedule.start_height_m
      )
      expected_next_pitch = schedule.start_pitch_rad + expected_applied_alpha * (
        schedule.climb_pitch_rad - schedule.start_pitch_rad
      )
    else:
      height_error = desired_height - previous_height
      pitch_error = desired_pitch - previous_pitch
      height_step = POSTURE_HEIGHT_SLEW_RATE_MPS * CONTROL_DT_S
      pitch_step = POSTURE_PITCH_SLEW_RATE_RADPS * CONTROL_DT_S
      expected_next_height = previous_height + min(
        max(height_error, -height_step), height_step,
      )
      expected_next_pitch = previous_pitch + min(
        max(pitch_error, -pitch_step), pitch_step,
      )
    if not math.isclose(
      applied_height, expected_next_height, rel_tol=0.0, abs_tol=2.0e-6,
    ) or not math.isclose(
      applied_pitch, expected_next_pitch, rel_tol=0.0, abs_tol=2.0e-6,
    ):
      raise ValueError('Applied posture does not follow the declared slew controller.')
    if slew_mode == SYNCHRONIZED_SLEW_MODE:
      assert synchronized_alpha_step is not None
      if (
        abs(
          alphas['schedule_applied_height_alpha']
          - alphas['schedule_applied_pitch_alpha']
        ) > 1.0e-5
        or abs(
          alphas['schedule_applied_alpha']
          - alphas['schedule_applied_height_alpha']
        ) > 1.0e-5
      ):
        raise ValueError('Synchronized control-trace channels do not share alpha.')
      if (
        alphas['schedule_applied_alpha']
        - previous_alphas['schedule_applied_alpha']
        > synchronized_alpha_step + 1.0e-5
      ):
        raise ValueError('Synchronized applied alpha exceeded its frozen slew.')
    previous_alphas.update(alphas)
    previous_height = applied_height
    previous_pitch = applied_pitch


def _validate_schedule_candidate_trials(
  candidate: Mapping[str, Any],
  trials: Sequence[Mapping[str, Any]],
  *,
  expected_repeats: int,
  expected_envs_per_height: int,
  drive_steps: int,
  expected_keys: frozenset[str] | None = None,
  require_control_trace: bool = False,
) -> frozenset[str]:
  expected_total = (
    len(DIAGNOSTIC_HEIGHTS_M) * expected_repeats * expected_envs_per_height
  )
  name = str(candidate['name'])
  if len(trials) != expected_total:
    raise ValueError(
      f'Schedule candidate {name} produced {len(trials)} trials; '
      f'expected {expected_total}.'
    )
  raw_keys = frozenset(trials[0])
  for row in trials:
    if frozenset(row) != raw_keys:
      raise ValueError(f'Schedule candidate {name} produced inconsistent trial keys.')
  if expected_keys is not None and raw_keys != expected_keys:
    raise ValueError('Static and dynamic schedule trials must use identical keys.')

  required_fields = {
    'posture_card',
    'target_height_m',
    'target_pitch_rad',
    'stair_height_m',
    'terrain_key',
    'terrain_index',
    'repeat',
    'env_id',
    'success',
    'time_to_success_s',
    'termination',
    'non_wheel_contact',
    'bilateral_airborne_ever',
    'bilateral_unsupported_physics_substeps',
    'bilateral_positive_clearance_ever',
    'actual_wheel_actuator_force_abs_max_nm',
    'wheel_residual_abs_max',
    'peak_pitch_abs_rad',
    'peak_roll_abs_rad',
    'peak_pitch_rate_abs_radps',
    'torque_saturation_fraction',
    'max_progress_past_face_m',
    'root_reset',
    'first_support_loss_progress_m',
    'left_vertical_normal_load_n_mean',
    'right_vertical_normal_load_n_mean',
    'total_vertical_normal_load_n_mean',
    'total_vertical_normal_load_n_min_control_step',
    *SCHEDULE_AUTHORITY_METRICS,
    *SCHEDULE_METADATA_FIELDS,
  }
  if require_control_trace:
    required_fields.add('control_trace')
  missing = sorted(required_fields - raw_keys)
  if missing:
    raise ValueError(
      f'Schedule candidate {name} is missing required trial fields: {missing}.'
    )

  card = candidate['posture_card']
  schedule: RollPoseSchedule | None = candidate['schedule']
  target_height = float(card['height_m'])
  target_pitch = float(card['pitch_rad'])
  slew_mode = str(candidate.get('slew_mode', INDEPENDENT_SLEW_MODE))
  env_count = len(DIAGNOSTIC_HEIGHTS_M) * expected_envs_per_height
  for row in trials:
    if row['posture_card'] != name:
      raise ValueError(f'Schedule candidate {name} trial has the wrong posture name.')
    if not isinstance(row['root_reset'], Mapping):
      raise TypeError(f'Schedule candidate {name} trial has no reset metadata.')
    observed_height = _required_finite_float(row, 'target_height_m')
    observed_pitch = _required_finite_float(row, 'target_pitch_rad')
    if not math.isclose(observed_height, target_height, rel_tol=0.0, abs_tol=1e-12):
      raise ValueError(f'Schedule candidate {name} trial has the wrong target height.')
    if not math.isclose(observed_pitch, target_pitch, rel_tol=0.0, abs_tol=1e-12):
      raise ValueError(f'Schedule candidate {name} trial has the wrong target pitch.')

    repeat = _required_int(row, 'repeat', minimum=1)
    env_id = _required_int(row, 'env_id', minimum=0)
    terrain_index = _required_int(row, 'terrain_index', minimum=0)
    if repeat > expected_repeats or env_id >= env_count:
      raise ValueError(f'Schedule candidate {name} trial identity is out of range.')
    if terrain_index >= len(DIAGNOSTIC_HEIGHTS_M):
      raise ValueError(f'Schedule candidate {name} terrain index is out of range.')
    stair_height = _required_finite_float(row, 'stair_height_m')
    expected_height = DIAGNOSTIC_HEIGHTS_M[terrain_index]
    if not math.isclose(stair_height, expected_height, rel_tol=0.0, abs_tol=1e-12):
      raise ValueError(f'Schedule candidate {name} trial height/index disagree.')
    if row['terrain_key'] != rb.terrain_key(expected_height):
      raise ValueError(f'Schedule candidate {name} trial has the wrong terrain key.')

    success = _required_bool(row, 'success')
    safety = {}
    for field in (
      'termination', 'non_wheel_contact', 'bilateral_airborne_ever',
      'bilateral_positive_clearance_ever',
    ):
      safety[field] = _required_bool(row, field)
    if success and any(
      safety[field] for field in (
        'termination', 'non_wheel_contact', 'bilateral_airborne_ever',
      )
    ):
      raise ValueError(f'Schedule candidate {name} marked an unsafe trial successful.')
    unsupported_substeps = _required_int(
      row, 'bilateral_unsupported_physics_substeps', minimum=0,
    )
    support_lost = unsupported_substeps > 0
    if support_lost != safety['bilateral_airborne_ever']:
      raise ValueError(
        f'Schedule candidate {name} support-loss boolean/count disagree.'
      )
    if support_lost and success:
      raise ValueError(f'Schedule candidate {name} promoted a 5 ms support loss.')
    if safety['bilateral_positive_clearance_ever'] and not support_lost:
      raise ValueError(
        f'Schedule candidate {name} has clearance without support loss.'
      )
    success_time = _optional_finite_float(row, 'time_to_success_s')
    if success != (success_time is not None) or (
      success_time is not None and success_time < 0.0
    ):
      raise ValueError(f'Schedule candidate {name} has inconsistent success timing.')
    first_loss_progress = _optional_finite_float(
      row, 'first_support_loss_progress_m',
    )
    if support_lost != (first_loss_progress is not None):
      raise ValueError(
        f'Schedule candidate {name} support-loss progress is inconsistent.'
      )

    for field in (
      'actual_wheel_actuator_force_abs_max_nm',
      'peak_pitch_abs_rad',
      'peak_roll_abs_rad',
      'peak_pitch_rate_abs_radps',
      'left_vertical_normal_load_n_mean',
      'right_vertical_normal_load_n_mean',
      'total_vertical_normal_load_n_mean',
      'total_vertical_normal_load_n_min_control_step',
    ):
      if _required_finite_float(row, field) < 0.0:
        raise ValueError(f'Schedule safety metric {field} must be nonnegative.')
    _required_finite_float(row, 'max_progress_past_face_m')
    saturation = _required_finite_float(row, 'torque_saturation_fraction')
    if not 0.0 <= saturation <= 1.0:
      raise ValueError('Schedule torque saturation fraction must lie in [0, 1].')
    for metric in (*SCHEDULE_AUTHORITY_METRICS, 'wheel_residual_abs_max'):
      if _required_finite_float(row, metric) != 0.0:
        raise ValueError(f'Schedule authority metric {metric} must be exactly zero.')
    trace = None
    if 'control_trace' in row:
      trace = row['control_trace']
      _validate_schedule_control_trace(
        trace, schedule=schedule, slew_mode=slew_mode,
        drive_steps=drive_steps,
      )
      if require_control_trace:
        last_trace_step = _required_int(trace[-1], 'control_step', minimum=1)
        if success:
          assert success_time is not None
          expected_success_step = round(success_time * rb.CONTROL_FREQUENCY_HZ)
          if (
            not math.isclose(
              success_time * rb.CONTROL_FREQUENCY_HZ,
              expected_success_step,
              rel_tol=0.0,
              abs_tol=1.0e-6,
            )
            or last_trace_step != expected_success_step
          ):
            raise ValueError('Successful trial control trace ended on the wrong step.')
        elif not safety['termination'] and not safety['non_wheel_contact']:
          if last_trace_step != drive_steps:
            raise ValueError('Unfinished trial control trace is not drive-complete.')

    if schedule is None:
      for field in (
        'roll_pose_schedule', 'drive_start_x_m', 'end_distance_to_riser_m',
        'schedule_slew_mode', 'schedule_alpha_max',
        'schedule_nominal_alpha_final', 'schedule_applied_alpha_final',
        'schedule_applied_height_alpha_final',
        'schedule_applied_pitch_alpha_final',
        'maximum_applied_channel_alpha_gap',
        'height_transition_completion_step',
        'pitch_transition_completion_step', 'transition_completion_step',
        'transition_completed_before_face',
      ):
        if row[field] is not None:
          raise ValueError(f'Static schedule metadata {field} must be null.')
      static_pose = {
        'desired_height_m_final': target_height,
        'desired_pitch_rad_final': target_pitch,
        'applied_height_m_final': target_height,
        'applied_pitch_rad_final': target_pitch,
        'maximum_height_tracking_lag_m': 0.0,
        'maximum_pitch_tracking_lag_rad': 0.0,
      }
      for field, expected in static_pose.items():
        if _required_finite_float(row, field) != expected:
          raise ValueError(f'Static schedule metadata {field} is inconsistent.')
      continue

    schedule_payload = row['roll_pose_schedule']
    if not isinstance(schedule_payload, Mapping) or dict(schedule_payload) != schedule.to_dict():
      raise ValueError(f'Schedule candidate {name} trial has the wrong schedule.')
    _required_finite_float(row, 'drive_start_x_m')
    end_distance = _required_finite_float(row, 'end_distance_to_riser_m')
    if end_distance != schedule.end_distance_to_riser_m:
      raise ValueError(f'Schedule candidate {name} trial has the wrong endpoint.')
    expected_slew_mode = slew_mode
    if row['schedule_slew_mode'] != expected_slew_mode:
      raise ValueError(f'Schedule candidate {name} trial has the wrong slew mode.')
    alpha_fields = (
      'schedule_alpha_max',
      'schedule_nominal_alpha_final',
      'schedule_applied_alpha_final',
      'schedule_applied_height_alpha_final',
      'schedule_applied_pitch_alpha_final',
    )
    for field in alpha_fields:
      alpha = _required_finite_float(row, field)
      if not 0.0 <= alpha <= 1.0:
        raise ValueError(f'Schedule alpha metric {field} must lie in [0, 1].')
    channel_gap = _required_finite_float(
      row, 'maximum_applied_channel_alpha_gap',
    )
    if not 0.0 <= channel_gap <= 1.0:
      raise ValueError('Schedule channel alpha gap must lie in [0, 1].')
    if expected_slew_mode == SYNCHRONIZED_SLEW_MODE and channel_gap > 1.0e-5:
      raise ValueError('Synchronized schedule channels must share applied alpha.')
    for field in (
      'desired_height_m_final', 'desired_pitch_rad_final',
      'applied_height_m_final', 'applied_pitch_rad_final',
    ):
      _required_finite_float(row, field)
    for field in (
      'maximum_height_tracking_lag_m', 'maximum_pitch_tracking_lag_rad',
    ):
      if _required_finite_float(row, field) < 0.0:
        raise ValueError(f'Schedule tracking metric {field} must be nonnegative.')
    completion_steps = {}
    for field in (
      'height_transition_completion_step',
      'pitch_transition_completion_step',
      'transition_completion_step',
    ):
      completion_step = row[field]
      if completion_step is not None and (
        isinstance(completion_step, bool)
        or not isinstance(completion_step, Integral)
        or not 1 <= int(completion_step) <= drive_steps
      ):
        raise ValueError(f'Schedule completion metric {field} is invalid.')
      completion_steps[field] = completion_step
    if (
      expected_slew_mode == SYNCHRONIZED_SLEW_MODE
      and completion_steps['height_transition_completion_step']
      != completion_steps['pitch_transition_completion_step']
    ):
      raise ValueError('Synchronized height and pitch completion steps must match.')
    _required_bool(row, 'transition_completed_before_face')
    if require_control_trace:
      assert trace is not None
      final_sample = trace[-1]
      final_pairs = {
        'schedule_nominal_alpha_final': 'schedule_nominal_alpha',
        'schedule_applied_alpha_final': 'schedule_applied_alpha',
        'schedule_applied_height_alpha_final': 'schedule_applied_height_alpha',
        'schedule_applied_pitch_alpha_final': 'schedule_applied_pitch_alpha',
        'applied_height_m_final': 'applied_height_m',
        'applied_pitch_rad_final': 'applied_pitch_rad',
      }
      for row_field, trace_field in final_pairs.items():
        if not math.isclose(
          _required_finite_float(row, row_field),
          _required_finite_float(final_sample, trace_field),
          rel_tol=0.0,
          abs_tol=2.0e-6,
        ):
          raise ValueError(f'Schedule final metric {row_field} disagrees with trace.')
      trace_nominal = [
        _required_finite_float(sample, 'schedule_nominal_alpha')
        for sample in trace
      ]
      if not math.isclose(
        _required_finite_float(row, 'schedule_alpha_max'),
        max(trace_nominal),
        rel_tol=0.0,
        abs_tol=1.0e-6,
      ):
        raise ValueError('Schedule alpha maximum disagrees with control trace.')
      trace_gaps = [
        abs(
          _required_finite_float(sample, 'schedule_applied_height_alpha')
          - _required_finite_float(sample, 'schedule_applied_pitch_alpha')
        )
        for sample in trace
      ]
      if not math.isclose(
        channel_gap, max(trace_gaps), rel_tol=0.0, abs_tol=1.0e-6,
      ):
        raise ValueError('Schedule channel alpha gap disagrees with control trace.')
      nominal_final = _required_finite_float(
        final_sample, 'schedule_nominal_alpha',
      )
      expected_desired_height = schedule.start_height_m + nominal_final * (
        schedule.climb_height_m - schedule.start_height_m
      )
      expected_desired_pitch = schedule.start_pitch_rad + nominal_final * (
        schedule.climb_pitch_rad - schedule.start_pitch_rad
      )
      if not math.isclose(
        _required_finite_float(row, 'desired_height_m_final'),
        expected_desired_height,
        rel_tol=0.0,
        abs_tol=2.0e-6,
      ) or not math.isclose(
        _required_finite_float(row, 'desired_pitch_rad_final'),
        expected_desired_pitch,
        rel_tol=0.0,
        abs_tol=2.0e-6,
      ):
        raise ValueError('Schedule desired endpoint disagrees with nominal alpha.')
      if expected_slew_mode == SYNCHRONIZED_SLEW_MODE:
        shared_completion = next((
          _required_int(sample, 'control_step', minimum=1)
          for sample in trace
          if _required_finite_float(sample, 'schedule_applied_alpha') >= 1.0
        ), None)
        for field in (
          'height_transition_completion_step',
          'pitch_transition_completion_step',
          'transition_completion_step',
        ):
          if completion_steps[field] != shared_completion:
            raise ValueError(
              'Synchronized completion metadata disagrees with shared alpha.'
            )

  rb.aggregate_trials(
    [dict(row) for row in trials],
    heights=DIAGNOSTIC_HEIGHTS_M,
    expected_repeats=expected_repeats,
    expected_envs_per_height=expected_envs_per_height,
    cards=(card,),
  )
  return raw_keys


def _identified_schedule_events(
  events: Sequence[Mapping[str, Any]], *, candidate: str, repeat: int,
) -> list[dict[str, Any]]:
  return [
    {**dict(event), 'candidate': candidate, 'repeat': repeat}
    for event in events
  ]


def _validate_schedule_events(
  candidate: Mapping[str, Any],
  trials: Sequence[Mapping[str, Any]],
  events: Sequence[Mapping[str, Any]],
  *,
  pre_substeps: int,
  post_substeps: int,
  settle_steps: int,
) -> None:
  loss_rows = {
    (
      _required_int(row, 'repeat', minimum=1),
      _required_int(row, 'env_id', minimum=0),
    ): row
    for row in trials
    if _required_int(
      row, 'bilateral_unsupported_physics_substeps', minimum=0,
    ) > 0
  }
  event_map = {}
  expected_sample_keys: frozenset[str] | None = None
  required_sample_fields = {
    'substep', 'episode_control_step', 'drive_control_step', 'phase',
    'terrain_index', 'left_force_n', 'right_force_n',
    'left_vertical_normal_load_n', 'right_vertical_normal_load_n',
    'total_vertical_normal_load_n', 'schedule_nominal_alpha',
    'schedule_applied_alpha', 'schedule_applied_height_alpha',
    'schedule_applied_pitch_alpha',
  }
  expected_count = pre_substeps + 1 + post_substeps
  for event in events:
    if not isinstance(event, Mapping):
      raise TypeError('Schedule event must be a mapping.')
    if event.get('candidate') != candidate['name']:
      raise ValueError('Schedule event has the wrong candidate identity.')
    identity = (
      _required_int(event, 'repeat', minimum=1),
      _required_int(event, 'env_id', minimum=0),
    )
    if identity in event_map:
      raise ValueError(f'Schedule event identity {identity} is duplicated.')
    if _required_int(event, 'pre_substeps_requested', minimum=1) != pre_substeps:
      raise ValueError('Schedule event pre-window length drifted.')
    if _required_int(event, 'post_substeps_requested', minimum=1) != post_substeps:
      raise ValueError('Schedule event post-window length drifted.')
    samples = event.get('samples')
    if not isinstance(samples, list) or len(samples) != expected_count:
      raise ValueError('Schedule event sample count is incomplete.')
    trigger_substep = _required_int(event, 'trigger_substep', minimum=1)
    for offset, sample in enumerate(samples):
      if not isinstance(sample, Mapping):
        raise TypeError('Schedule event sample must be a mapping.')
      sample_keys = frozenset(sample)
      if not required_sample_fields <= sample_keys:
        raise ValueError('Schedule event sample is missing required fields.')
      if expected_sample_keys is None:
        expected_sample_keys = sample_keys
      elif sample_keys != expected_sample_keys:
        raise ValueError('Schedule event sample schema is not uniform.')
      substep = _required_int(sample, 'substep', minimum=1)
      if substep != trigger_substep - pre_substeps + offset:
        raise ValueError('Schedule event substeps are not consecutive.')
      episode_step = _required_int(sample, 'episode_control_step', minimum=1)
      expected_episode_step = (
        (substep - 1) // rb.ROLL_FIRST_CONTROL_DECIMATION + 1
      )
      if episode_step != expected_episode_step:
        raise ValueError('Schedule event episode-control index is inconsistent.')
      phase = sample.get('phase')
      if phase not in ('settle', 'drive'):
        raise ValueError('Schedule event phase is invalid.')
      drive_step = sample.get('drive_control_step')
      expected_drive_step = (
        None if episode_step <= settle_steps else episode_step - settle_steps
      )
      if drive_step != expected_drive_step:
        raise ValueError('Schedule event drive-control index is inconsistent.')
      for field, raw in sample.items():
        if field in ('phase', 'drive_control_step', 'substep', 'episode_control_step'):
          continue
        if isinstance(raw, bool) or not isinstance(raw, Real) or not math.isfinite(float(raw)):
          raise ValueError(f'Schedule event field {field} must be finite numeric data.')
      left_load = _required_finite_float(sample, 'left_vertical_normal_load_n')
      right_load = _required_finite_float(sample, 'right_vertical_normal_load_n')
      total_load = _required_finite_float(sample, 'total_vertical_normal_load_n')
      if min(left_load, right_load, total_load) < 0.0 or not math.isclose(
        total_load, left_load + right_load, rel_tol=1.0e-6, abs_tol=1.0e-5,
      ):
        raise ValueError('Schedule event vertical-load decomposition is invalid.')
      for field in (
        'schedule_nominal_alpha', 'schedule_applied_alpha',
        'schedule_applied_height_alpha', 'schedule_applied_pitch_alpha',
      ):
        alpha = _required_finite_float(sample, field)
        if not 0.0 <= alpha <= 1.0:
          raise ValueError(f'Schedule event alpha {field} left [0, 1].')
    trigger = samples[pre_substeps]
    if (
      _required_int(trigger, 'substep', minimum=1) != trigger_substep
      or _required_finite_float(trigger, 'left_force_n') > 0.0
      or _required_finite_float(trigger, 'right_force_n') > 0.0
    ):
      raise ValueError('Schedule event trigger sample is not bilateral unsupported.')
    row = loss_rows.get(identity)
    if row is None:
      raise ValueError('Schedule event has no support-loss trial.')
    terrain_index = _required_int(row, 'terrain_index', minimum=0)
    if _required_finite_float(trigger, 'terrain_index') != float(terrain_index):
      raise ValueError('Schedule event terrain identity disagrees with its trial.')
    trace = row.get('control_trace')
    if trace is not None:
      if not isinstance(trace, list):
        raise TypeError('Schedule trial control trace must be a list.')
      alpha_fields = (
        'schedule_nominal_alpha', 'schedule_applied_alpha',
        'schedule_applied_height_alpha', 'schedule_applied_pitch_alpha',
      )
      for sample in samples:
        drive_step = sample['drive_control_step']
        if drive_step is None:
          expected_alphas = {field: 0.0 for field in alpha_fields}
        else:
          if not 1 <= drive_step <= len(trace):
            raise ValueError('Schedule event falls outside its control trace.')
          expected_alphas = trace[drive_step - 1]
        for field in alpha_fields:
          if not math.isclose(
            _required_finite_float(sample, field),
            _required_finite_float(expected_alphas, field),
            rel_tol=0.0,
            abs_tol=1.0e-6,
          ):
            raise ValueError('Schedule event alpha disagrees with control trace.')
    event_map[identity] = event
  if event_map.keys() != loss_rows.keys():
    missing = sorted(loss_rows.keys() - event_map.keys())
    extra = sorted(event_map.keys() - loss_rows.keys())
    raise ValueError(
      f'Schedule support-loss event coverage is incomplete: missing={missing}, extra={extra}'
    )


def _execute_schedule_candidates(
  args: argparse.Namespace,
  candidates: Sequence[Mapping[str, Any]],
  *,
  log_label: str,
  record_control_trace: bool = False,
) -> tuple[Path, list[dict[str, Any]]]:
  artifact = rb.frozen_artifact_paths()['posture_map_path']
  cfg = rb.make_roll_boundary_env_cfg(
    DIAGNOSTIC_HEIGHTS_M, args.envs_per_height,
  )
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  original_cards = rb.POSTURE_CARDS
  original_installer = rb.install_strict_substep_support_recorder
  results = []
  expected_trial_keys: frozenset[str] | None = None
  try:
    for candidate in candidates:
      card = candidate['posture_card']
      schedule: RollPoseSchedule | None = candidate['schedule']
      print(f'[{log_label}] candidate={candidate["name"]}')
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
            roll_pose_slew_mode=str(
              candidate.get('slew_mode', INDEPENDENT_SLEW_MODE)
            ),
            require_pure_classical_authority=True,
            record_diagnostic_control_trace=record_control_trace,
          ))
        finally:
          rb.install_strict_substep_support_recorder = original_installer
        events.extend(_identified_schedule_events(
          collector.finalize(), candidate=str(candidate['name']), repeat=repeat,
        ))
      trial_keys = _validate_schedule_candidate_trials(
        candidate,
        rows,
        expected_repeats=args.repeats,
        expected_envs_per_height=args.envs_per_height,
        drive_steps=args.drive_steps,
        expected_keys=expected_trial_keys,
        require_control_trace=record_control_trace,
      )
      if expected_trial_keys is None:
        expected_trial_keys = trial_keys
      _validate_schedule_events(
        candidate,
        rows,
        events,
        pre_substeps=args.pre_substeps,
        post_substeps=args.post_substeps,
        settle_steps=args.settle_steps,
      )
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
  return artifact, results


def run_schedule_grid(
  args: argparse.Namespace, provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  candidates = schedule_grid_candidates()
  _validate_schedule_candidate_set(candidates)
  artifact, results = _execute_schedule_candidates(
    args, candidates, log_label='roll-boundary-schedule-grid',
  )
  return {
    'schema_version': SCHEDULE_DIAGNOSTIC_SCHEMA_VERSION,
    'kind': 'roll_pose_schedule_grid_diagnostic',
    **_run_provenance(args, provenance),
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


def _validate_matched_r0c_resets(
  results: Sequence[Mapping[str, Any]],
) -> bool:
  if len(results) != 2:
    raise ValueError('R0c-SYNC reset matching requires exactly two results.')
  reset_maps = []
  for result in results:
    trials = result.get('trials')
    if not isinstance(trials, Sequence):
      raise TypeError('R0c-SYNC result has no trial sequence.')
    reset_map = {}
    for row in trials:
      if not isinstance(row, Mapping) or not isinstance(row.get('root_reset'), Mapping):
        raise TypeError('R0c-SYNC trial has no root-reset mapping.')
      identity = (
        _required_int(row, 'repeat', minimum=1),
        _required_int(row, 'terrain_index', minimum=0),
        _required_int(row, 'env_id', minimum=0),
      )
      if identity in reset_map:
        raise ValueError(f'R0c-SYNC duplicated reset identity {identity}.')
      reset_map[identity] = dict(row['root_reset'])
    reset_maps.append(reset_map)
  if reset_maps[0].keys() != reset_maps[1].keys():
    raise ValueError('R0c-SYNC candidate reset identities differ.')
  for identity, expected in reset_maps[0].items():
    if reset_maps[1][identity] != expected:
      raise ValueError(f'R0c-SYNC reset mismatch at identity {identity}.')
  return True


def _r0c_summary(
  result: Mapping[str, Any], height_m: float,
) -> Mapping[str, Any]:
  summaries = result.get('summaries')
  if not isinstance(summaries, Sequence):
    raise TypeError('R0c-SYNC result has no summaries.')
  matches = [
    item for item in summaries
    if isinstance(item, Mapping)
    and _required_finite_float(item, 'stair_height_m') == height_m
  ]
  if len(matches) != 1:
    raise ValueError(f'R0c-SYNC requires one summary at {height_m} m.')
  return matches[0]


def classify_r0c_sync_screen(
  results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
  by_kind = {
    str(result['candidate_definition']['kind']): result
    for result in results
  }
  expected_kinds = {
    'legacy_independent_slew_baseline',
    'shared_alpha_synchronized_slew',
  }
  if len(results) != len(expected_kinds) or set(by_kind) != expected_kinds:
    raise ValueError('R0c-SYNC result modes are incomplete or duplicated.')
  baseline_flat = _r0c_summary(
    by_kind['legacy_independent_slew_baseline'], 0.0,
  )
  baseline_step = _r0c_summary(
    by_kind['legacy_independent_slew_baseline'], 0.0025,
  )
  sync_flat = _r0c_summary(
    by_kind['shared_alpha_synchronized_slew'], 0.0,
  )
  sync_step = _r0c_summary(
    by_kind['shared_alpha_synchronized_slew'], 0.0025,
  )

  def flat_pass(summary: Mapping[str, Any]) -> bool:
    return (
      _required_int(summary, 'trials', minimum=1) == R0C_SYNC_ENVS_PER_HEIGHT
      and _required_int(summary, 'successes', minimum=0) == R0C_SYNC_ENVS_PER_HEIGHT
      and _required_int(summary, 'unsafe_trials', minimum=0) == 0
      and _required_int(summary, 'safe_stalls', minimum=0) == 0
      and _required_int(summary, 'bilateral_airborne_trials', minimum=0) == 0
      and _required_int(summary, 'bilateral_unsupported_physics_substeps', minimum=0) == 0
      and _required_int(summary, 'non_wheel_contact_trials', minimum=0) == 0
      and _required_int(summary, 'terminated_trials', minimum=0) == 0
    )

  flat_retention_passed = flat_pass(baseline_flat) and flat_pass(sync_flat)
  baseline_successes = _required_int(baseline_step, 'successes', minimum=0)
  baseline_unsafe = _required_int(baseline_step, 'unsafe_trials', minimum=0)
  baseline_reproduced = (
    _required_int(baseline_step, 'trials', minimum=1) == R0C_SYNC_ENVS_PER_HEIGHT
    and R0C_SYNC_BASELINE_SUCCESS_RANGE[0]
    <= baseline_successes
    <= R0C_SYNC_BASELINE_SUCCESS_RANGE[1]
    and baseline_unsafe == R0C_SYNC_ENVS_PER_HEIGHT - baseline_successes
    and _required_int(baseline_step, 'safe_stalls', minimum=0) == 0
    and _required_int(
      baseline_step, 'bilateral_airborne_trials', minimum=0,
    ) == baseline_unsafe
    and _required_int(
      baseline_step, 'bilateral_unsupported_physics_substeps', minimum=0,
    ) >= baseline_unsafe
    and _required_int(baseline_step, 'non_wheel_contact_trials', minimum=0) == 0
    and _required_int(baseline_step, 'terminated_trials', minimum=0) == 0
  )
  sync_trials = _required_int(sync_step, 'trials', minimum=1)
  sync_successes = _required_int(sync_step, 'successes', minimum=0)
  sync_unsafe = _required_int(sync_step, 'unsafe_trials', minimum=0)
  sync_stalls = _required_int(sync_step, 'safe_stalls', minimum=0)
  sync_airborne = _required_int(
    sync_step, 'bilateral_airborne_trials', minimum=0,
  )
  sync_unsupported_substeps = _required_int(
    sync_step, 'bilateral_unsupported_physics_substeps', minimum=0,
  )
  sync_non_wheel = _required_int(
    sync_step, 'non_wheel_contact_trials', minimum=0,
  )
  sync_terminated = _required_int(sync_step, 'terminated_trials', minimum=0)
  if (
    sync_trials != R0C_SYNC_ENVS_PER_HEIGHT
    or sync_successes + sync_unsafe + sync_stalls != sync_trials
  ):
    raise ValueError('R0c-SYNC synchronized outcome counts are inconsistent.')
  if not flat_retention_passed:
    decision = 'INVALID_FLAT_RETENTION_REGRESSION'
  elif not baseline_reproduced:
    decision = 'INVALID_BASELINE_NOT_REPRODUCED'
  elif any((
    sync_unsafe,
    sync_airborne,
    sync_unsupported_substeps,
    sync_non_wheel,
    sync_terminated,
  )):
    decision = 'SYNC_REJECTED_ADVANCE_TO_R0C_LRG'
  elif sync_successes >= R0C_SYNC_PASS_SUCCESSES:
    decision = 'SYNC_SCREEN_PASS_FORMAL_REPLICATION_REQUIRED'
  else:
    decision = 'SYNC_REJECTED_SAFE_STALL_ADVANCE_TO_R0C_LRG'
  return {
    'decision': decision,
    'promotion_eligible': False,
    'flat_retention_passed': flat_retention_passed,
    'legacy_baseline_reproduced': baseline_reproduced,
    'legacy_2p5mm_successes': baseline_successes,
    'legacy_2p5mm_unsafe_trials': baseline_unsafe,
    'legacy_success_reproduction_range': list(R0C_SYNC_BASELINE_SUCCESS_RANGE),
    'synchronized_2p5mm_successes': sync_successes,
    'synchronized_2p5mm_unsafe_trials': sync_unsafe,
    'synchronized_2p5mm_safe_stalls': sync_stalls,
    'synchronized_2p5mm_bilateral_airborne_trials': sync_airborne,
    'synchronized_2p5mm_unsupported_physics_substeps': sync_unsupported_substeps,
    'synchronized_2p5mm_non_wheel_contact_trials': sync_non_wheel,
    'synchronized_2p5mm_terminated_trials': sync_terminated,
    'screen_pass_rule': {
      'minimum_safe_successes': R0C_SYNC_PASS_SUCCESSES,
      'unsafe_trials_required': 0,
      'bilateral_unsupported_physics_substeps_required': 0,
      'bilateral_airborne_trials_required': 0,
      'non_wheel_contact_trials_required': 0,
      'terminated_trials_required': 0,
      'formal_replication_required': True,
    },
  }


def run_r0c_sync_screen(
  args: argparse.Namespace, provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  protocol_drift = _r0c_sync_protocol_drift(args)
  if protocol_drift:
    raise ValueError(f'R0c-SYNC protocol drifted: {protocol_drift}')
  candidates = r0c_sync_candidates()
  _validate_r0c_sync_candidate_set(candidates)
  artifact, results = _execute_schedule_candidates(
    args, candidates, log_label='r0c-sync', record_control_trace=True,
  )
  matched_resets = _validate_matched_r0c_resets(results)
  return {
    'schema_version': R0C_SYNC_SCHEMA_VERSION,
    'kind': 'r0c_synchronized_reference_rejection_screen',
    **_run_provenance(args, provenance),
    'matched_reset_perturbations_across_candidates': matched_resets,
    'continue_after_first_support_loss': True,
    'policy_action_required_zero': True,
    'all_applied_residual_required_zero': True,
    'experiment_variable': 'independent_vs_shared_alpha_posture_slew',
    'nominal_schedule': R0C_SYNC_BASE_SCHEDULE_NAME,
    'load_measurement': {
      'left_field': 'left_vertical_normal_load_n',
      'right_field': 'right_vertical_normal_load_n',
      'definition': 'sum(abs(contact_normal_force * global_normal_z)) over found slots',
      'control_authority': False,
    },
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
    'screen_verdict': classify_r0c_sync_screen(results),
  }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description='Non-evidentiary RollBoundary diagnostics.'
  )
  parser.add_argument(
    '--mode', choices=('events', 'posture-grid', 'schedule-grid', 'r0c-sync'), required=True,
  )
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--device', default='cpu')
  parser.add_argument('--envs-per-height', type=int)
  parser.add_argument('--repeats', type=int)
  parser.add_argument('--settle-steps', type=int, default=rb.OFFICIAL_SETTLE_STEPS)
  parser.add_argument('--drive-steps', type=int, default=rb.OFFICIAL_DRIVE_STEPS)
  parser.add_argument('--stable-steps', type=int, default=rb.OFFICIAL_STABLE_STEPS)
  parser.add_argument('--pre-substeps', type=int, default=DEFAULT_PRE_SUBSTEPS)
  parser.add_argument('--post-substeps', type=int, default=DEFAULT_POST_SUBSTEPS)
  parser.add_argument('--pitch-count', type=int, default=5)
  parser.add_argument('--allow-dirty', action='store_true')
  args = parser.parse_args(argv)
  if args.envs_per_height is None:
    args.envs_per_height = (
      R0C_SYNC_ENVS_PER_HEIGHT
      if args.mode == 'r0c-sync' else rb.OFFICIAL_ENVS_PER_HEIGHT
    )
  if args.repeats is None:
    args.repeats = R0C_SYNC_REPEATS if args.mode == 'r0c-sync' else rb.OFFICIAL_REPEATS
  for name in (
    'envs_per_height', 'repeats', 'settle_steps', 'drive_steps', 'stable_steps',
    'pre_substeps', 'post_substeps', 'pitch_count',
  ):
    _validate_positive(getattr(args, name), name=name)
  if args.mode == 'r0c-sync':
    drift = _r0c_sync_protocol_drift(args)
    if drift:
      parser.error(f'R0c-SYNC protocol is frozen; observed/expected drift: {drift}')
  return args


def main(argv: Sequence[str] | None = None) -> None:
  args = parse_args(argv)
  output = _outside_repository(args.output)
  reservation = _reserve_output(output)
  try:
    mjlab_root = Path(mjlab.__file__).resolve().parents[2]
    project_dirty = _git_dirty(rb.REPOSITORY_PATH)
    mjlab_dirty = _git_dirty(mjlab_root)
    if project_dirty and not args.allow_dirty:
      raise RuntimeError('Diagnostic requires a clean project checkout.')
    if mjlab_dirty and not args.allow_dirty:
      raise RuntimeError('Diagnostic requires a clean MjLab checkout.')
    if (
      args.mode in ('schedule-grid', 'r0c-sync')
      and torch.device(args.device).type != 'cpu'
      and args.allow_dirty
    ):
      raise RuntimeError('CUDA schedule diagnostics cannot use --allow-dirty.')
    provenance = _diagnostic_provenance(
      args.device,
      allow_dirty=args.allow_dirty,
      project_dirty=project_dirty,
      mjlab_dirty=mjlab_dirty,
    )
    if args.mode == 'events':
      payload = run_event_diagnostic(args, provenance=provenance)
    elif args.mode == 'posture-grid':
      payload = run_posture_grid(args, provenance=provenance)
    elif args.mode == 'schedule-grid':
      payload = run_schedule_grid(args, provenance=provenance)
    else:
      payload = run_r0c_sync_screen(args, provenance=provenance)
    _verify_provenance_unchanged(provenance)
    temporary = output.with_name(f'.{output.name}.incomplete')
    if not reservation.is_file() or output.exists() or temporary.exists():
      raise RuntimeError('Diagnostic output reservation changed during execution.')
    rb._atomic_write_json(output, payload)
    print(f'[roll-boundary-diagnostic] output={output}')
  finally:
    reservation.unlink(missing_ok=True)


if __name__ == '__main__':
  main()
