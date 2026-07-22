#!/usr/bin/env python3
"""Evaluate HopperTrex Hybrid v2 checkpoints with capability-driven gates."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from statistics import fmean
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

import torch

PROJECT_PATH = Path(__file__).resolve().parents[2]
SRC_PATH = Path(__file__).resolve().parents[3]
REPOSITORY_PATH = Path(__file__).resolve().parents[4]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

try:
  import hoppertrex_mjlab.tasks as tasks  # noqa: F401
except ImportError:
  import tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from hoppertrex_mjlab.hybrid.mismatch import (
  STAGE1_MISMATCH_PROFILE_VERSION,
  stage1_mismatch_audit,
  stage1_mismatch_spec,
)
from hoppertrex_mjlab.hybrid.config import HYBRID_STAGES  # noqa: E402

try:
  from .evaluate_fixed_command import (
    _force_command as _force_velocity_command,
    _run_fixed_command,
  )
  from .evaluate_fixed_yaw import _run_fixed_yaw
  from .hybrid_gate import (
    aggregate_seed_results,
    boolean_mask_on_device,
    evaluate_capability_suite,
    linear_scenario_checks,
    make_result_envelope,
    resolve_wheel_action,
    stage1_observational_improvements,
    to_deterministic_json,
    wheel_target_saturation_threshold,
  )
except ImportError:
  from scripts.rsl_rl.evaluate_fixed_command import (
    _force_command as _force_velocity_command,
    _run_fixed_command,
  )
  from scripts.rsl_rl.evaluate_fixed_yaw import _run_fixed_yaw
  from scripts.rsl_rl.hybrid_gate import (
    aggregate_seed_results,
    boolean_mask_on_device,
    evaluate_capability_suite,
    linear_scenario_checks,
    make_result_envelope,
    resolve_wheel_action,
    stage1_observational_improvements,
    to_deterministic_json,
    wheel_target_saturation_threshold,
  )
try:
  from .train import validate_hybrid_training_checkpoint
except ImportError:
  from scripts.rsl_rl.train import validate_hybrid_training_checkpoint
try:
  from hoppertrex_mjlab.tasks.hoppertrex_balance_task import (
    NON_WHEEL_GROUND_SENSOR_NAME,
    non_wheel_ground_contact,
  )
except ImportError:
  from tasks.hoppertrex_balance_task import (
    NON_WHEEL_GROUND_SENSOR_NAME,
    non_wheel_ground_contact,
  )
try:
  from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
    stage1_mismatch_event_cfg,
  )
except ImportError:
  from tasks.hoppertrex_hybrid_task import stage1_mismatch_event_cfg


HYBRID_STAGE_SUITES = {
  0: "controller",
  1: "residual",
  2: "planar",
  3: "posture",
  4: "integrated",
  5: "robust",
}
HYBRID_STAGE_TASKS = {
  stage: f"HopperTrex-Hybrid-v2-Stage{stage}"
  for stage in HYBRID_STAGE_SUITES
}
CONTROL_FREQUENCY_HZ = 50.0
STAGE1_PROFILE_COMMANDS = (-0.10, -0.07, 0.0, 0.07, 0.10)
STAGE1_PROFILE_GROUPS = len(STAGE1_PROFILE_COMMANDS) + 3
STAGE1_KICK_LIN_X = 0.04
STAGE1_KICK_PITCH_RATE = 0.06
STAGE1_HEALTHY_ERROR = 0.015
STAGE1_HEALTHY_PITCH = 0.04
STAGE1_HEALTHY_PITCH_RATE = 0.20
STAGE1_HEALTHY_HOLD_STEPS = 25
STAGE1_TRANSITION_SEGMENT_STEPS = 100
STAGE1_MIN_MEASURED_STEPS = (
  len((0.0, 0.07, 0.0, -0.07, 0.0, 0.07, -0.07))
  * STAGE1_TRANSITION_SEGMENT_STEPS
)
ZERO_RESIDUAL_STANDING_REPEATS = 2
ZERO_RESIDUAL_STANDING_METRICS = (
  "command_match_frac",
  "fast_frac",
  "late_in_band_frac",
  "late_target_band_frac",
  "terminated_event_rate",
  "non_wheel_contact_rate",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--stage", type=int, choices=tuple(HYBRID_STAGE_SUITES))
  parser.add_argument("--task", default=None)
  parser.add_argument("--checkpoint-file", default=None)
  parser.add_argument(
    "--profile",
    choices=("screen", "formal"),
    default="formal",
    help=(
      "screen is a cheap rejection run; formal enforces the promotion "
      "rollout length."
    ),
  )
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--steps", type=int, default=3000)
  parser.add_argument("--warmup-steps", type=int, default=300)
  parser.add_argument("--window-steps", type=int, default=800)
  parser.add_argument("--progress-interval", type=int, default=500)
  parser.add_argument("--episode-length-s", type=float, default=1.0e9)
  parser.add_argument(
    "--device",
    default="cuda:0" if torch.cuda.is_available() else "cpu",
  )
  parser.add_argument(
    "--scenario-file",
    type=Path,
    default=None,
    help="Evaluate pre-collected scenarios instead of running simulation.",
  )
  parser.add_argument(
    "--stage4-reference-file",
    type=Path,
    default=None,
    help=(
      "Optional Stage4 gate envelope for the Stage5 tracking-degradation "
      "check. When omitted the robust suite measures the zero-residual "
      "classical stack's fixed-command integrated tracking as the "
      "reference (Route A: no Stage4 training run exists)."
    ),
  )
  parser.add_argument(
    "--stage1-retention-file",
    type=Path,
    default=None,
    help=(
      "Required matching Stage1-B gate envelope for a live Stage2 gate. "
      "Its profile, seed, git SHA, and checkpoint SHA256 must match."
    ),
  )
  parser.add_argument(
    "--ablate-leg-residuals",
    action="store_true",
    help=(
      "Zero the four leg residual heads of the loaded policy during "
      "collection (attribution ablation: full policy vs wheels-only). "
      "Recorded in the result rollout; never valid for promotion gates."
    ),
  )
  parser.add_argument(
    "--zero-residual-standing-diagnostic",
    action="store_true",
    help=(
      "Run the pre-registered Stage5 screen standing scenario twice with "
      "the native zero policy. This is a non-promotable threshold "
      "counterfactual and does not evaluate a checkpoint gate."
    ),
  )
  parser.add_argument(
    "--diagnostic-repeats",
    type=int,
    default=ZERO_RESIDUAL_STANDING_REPEATS,
    help="Repeat count for --zero-residual-standing-diagnostic (must be 2).",
  )
  parser.add_argument(
    "--aggregate-input",
    type=Path,
    nargs=3,
    default=None,
    metavar=("SEED1_JSON", "SEED2_JSON", "SEED3_JSON"),
    help="Aggregate exactly three existing seed envelopes.",
  )
  parser.add_argument("--controller-gain-hash", default=None)
  parser.add_argument("--output", type=Path, default=None)
  return parser.parse_args(argv)


def _validate_rollout_args(args: argparse.Namespace) -> None:
  if args.num_envs <= 0:
    raise ValueError("--num-envs must be positive.")
  if args.warmup_steps < 0:
    raise ValueError("--warmup-steps must be non-negative.")
  if args.steps <= args.warmup_steps:
    raise ValueError("--steps must be greater than --warmup-steps.")
  if args.window_steps <= 0:
    raise ValueError("--window-steps must be positive.")
  if args.progress_interval <= 0:
    raise ValueError("--progress-interval must be positive.")
  measured_steps = args.steps - args.warmup_steps
  if args.profile == "formal" and args.steps < 3000:
    raise ValueError("Formal Hybrid gate requires --steps >= 3000.")
  if args.stage == 1 and args.profile == "formal" and args.num_envs < 32:
    raise ValueError("Formal Stage1-B gate requires --num-envs >= 32.")
  if args.stage == 1 and measured_steps < STAGE1_MIN_MEASURED_STEPS:
    raise ValueError(
      "Stage1 residual gate requires at least "
      f"{STAGE1_MIN_MEASURED_STEPS} measured steps after warmup so every "
      "command-transition segment and multiple kicks are evaluated."
    )


def _validate_zero_residual_standing_args(args: argparse.Namespace) -> None:
  if not args.zero_residual_standing_diagnostic:
    return
  if args.stage != 5:
    raise ValueError("Zero-residual standing diagnostic requires --stage 5.")
  if args.task not in (None, HYBRID_STAGE_TASKS[5]):
    raise ValueError("Zero-residual standing diagnostic requires the Stage5 task.")
  if args.profile != "screen":
    raise ValueError("Zero-residual standing diagnostic requires --profile screen.")
  if args.checkpoint_file is not None:
    raise ValueError("Zero-residual standing diagnostic does not take a checkpoint.")
  if args.scenario_file is not None or args.aggregate_input is not None:
    raise ValueError("Zero-residual standing diagnostic requires a live rollout.")
  if (
    args.stage4_reference_file is not None
    or args.stage1_retention_file is not None
  ):
    raise ValueError(
      "Zero-residual standing diagnostic does not take gate reference inputs."
    )
  if args.ablate_leg_residuals:
    raise ValueError("Zero-residual standing diagnostic cannot ablate a checkpoint.")
  expected = {
    "num_envs": 16,
    "steps": 1000,
    "warmup_steps": 300,
    "window_steps": 300,
    "diagnostic_repeats": ZERO_RESIDUAL_STANDING_REPEATS,
  }
  for name, value in expected.items():
    if getattr(args, name) != value:
      flag = name.replace("_", "-")
      raise ValueError(
        f"Zero-residual standing diagnostic requires --{flag} {value}."
      )


def _validate_live_scenario_coverage(
  suite: str,
  scenarios: Sequence[Mapping[str, object]],
) -> None:
  names = {str(scenario.get('name', '')) for scenario in scenarios}
  required: set[str] = set()
  if suite == 'controller':
    required.update(
      {'controller_stand', 'controller_vx_-0.070', 'controller_vx_+0.070'}
    )
  if suite == 'linear':
    required.update(
      f'linear_vx_{value:+.3f}'
      for value in (-0.07, -0.04, 0.0, 0.04, 0.07)
    )
  if suite == 'residual':
    required.update(
      f'stage1_nominal_vx_{value:+.3f}' for value in (-0.07, 0.0, 0.07)
    )
    required.update(
      f'stage1_extension_vx_{value:+.3f}' for value in (-0.10, 0.10)
    )
    required.update((
      'stage1_disturbance_recovery',
      'stage1_command_transition',
      'stage1_model_mismatch',
    ))
  if suite in ('planar', 'integrated', 'robust'):
    required.update(
      f'linear_vx_{value:+.3f}' for value in (-0.07, 0.0, 0.07)
    )
    required.update(
      f'yaw_vx_{0.0:+.3f}_wz_{yaw:+.3f}' for yaw in (-0.10, 0.10)
    )
    required.update(
      f'combo_vx_{lin:+.3f}_wz_{yaw:+.3f}'
      for lin in (-0.07, 0.07)
      for yaw in (-0.10, 0.10)
    )
  if suite in ('integrated', 'robust'):
    required.add('random_integrated')
    required.add(
      'robust_pushes' if suite == 'robust' else 'integrated_reference'
    )
  if suite == 'robust':
    required.add('stage5_recovery_center_8x')
  missing = sorted(required - names)
  posture_count = sum(
    scenario.get('kind') == 'posture' for scenario in scenarios
  )
  if suite in ('posture', 'integrated', 'robust') and posture_count < 5:
    missing.append(f'posture scenarios: expected 5, got {posture_count}')
  if suite in ('posture', 'integrated', 'robust') and posture_count >= 5:
    posture_points = {
      (
        float(scenario.get('target_height', math.nan)),
        float(scenario.get('target_pitch', math.nan)),
      )
      for scenario in scenarios
      if scenario.get('kind') == 'posture'
      and _finite_scenario_coordinate(scenario.get('target_height'))
      and _finite_scenario_coordinate(scenario.get('target_pitch'))
    }
    if len(posture_points) < 5:
      missing.append(
        f'posture scenarios: expected 5 unique finite targets, got '
        f'{len(posture_points)}'
      )
    elif not _is_center_and_four_corners(posture_points):
      missing.append('posture scenarios: expected center and four corners')
  if missing:
    raise ValueError(
      'Live Hybrid gate did not collect required scenarios: '
      + ', '.join(missing)
    )


def _survival_rate(ever_terminated: Sequence[bool]) -> float:
  if not ever_terminated:
    raise ValueError("Survival rate requires at least one environment.")
  terminated_count = sum(bool(value) for value in ever_terminated)
  return 1.0 - terminated_count / len(ever_terminated)


def _fixed_rows_to_scenarios(
  kind: str,
  rows: Sequence[dict[str, float | str]],
) -> list[dict[str, object]]:
  scenarios: list[dict[str, object]] = []
  for row in rows:
    lin_x = float(row.get("lin_x", 0.0))
    yaw = float(row.get("yaw", 0.0))
    if kind == "linear":
      name = f"linear_vx_{lin_x:+.3f}"
    elif kind == "yaw":
      name = f"yaw_vx_{lin_x:+.3f}_wz_{yaw:+.3f}"
    elif kind == "combo":
      name = f"combo_vx_{lin_x:+.3f}_wz_{yaw:+.3f}"
    else:
      raise ValueError(f"Unsupported fixed-row scenario kind: {kind}")
    scenarios.append(
      {
        "name": name,
        "kind": kind,
        "lin_x": lin_x,
        "yaw": yaw,
        "metrics": row,
      }
    )
  return scenarios


def _linear_row_to_scenario(
  suite: str,
  row: dict[str, float | str],
) -> dict[str, object]:
  lin_x = float(row.get('lin_x', 0.0))
  if suite == 'controller':
    name = (
      'controller_stand'
      if abs(lin_x) <= 1.0e-12
      else f'controller_vx_{lin_x:+.3f}'
    )
    kind = 'controller'
  else:
    name = f'linear_vx_{lin_x:+.3f}'
    kind = 'linear'
  return {
    'name': name,
    'kind': kind,
    'lin_x': lin_x,
    'metrics': row,
  }


def _extract_stage4_reference(
  envelope: Mapping[str, object],
) -> dict[str, float]:
  metrics = envelope.get("metrics")
  if not isinstance(metrics, Mapping):
    raise ValueError("Stage4 reference envelope has no metrics object.")
  ordered_names = ["integrated_reference", *sorted(str(name) for name in metrics)]
  for name in ordered_names:
    scenario = metrics.get(name)
    if not isinstance(scenario, Mapping):
      continue
    value = scenario.get("tracking_error")
    if isinstance(value, Mapping):
      value = value.get("mean")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
      return {"tracking_error": float(value)}
  raise ValueError("Stage4 reference has no finite tracking_error metric.")


def _read_json(path: Path) -> object:
  return json.loads(path.read_text(encoding="utf-8"))


def _validate_stage2_retention_gate(
  envelope: Mapping[str, object],
  *,
  requested_profile: str,
  seed: int,
  git_sha: str,
  checkpoint_file_sha256: str,
) -> dict[str, object]:
  """Validate Stage1 capability retention for the exact Stage2 checkpoint."""

  expected = {
    "suite": "residual",
    "task": HYBRID_STAGE_TASKS[1],
    "evaluation_profile": requested_profile,
    "evaluation_source": "live",
    "seed": seed,
    "git_sha": git_sha,
    "checkpoint_file_sha256": checkpoint_file_sha256,
    "stage1_profile_version": STAGE1_MISMATCH_PROFILE_VERSION,
  }
  for key, value in expected.items():
    if envelope.get(key) != value:
      raise ValueError(
        f"Stage2 retention gate {key} does not match: "
        f"{envelope.get(key)!r} != {value!r}."
      )
  if envelope.get("mismatch_profile") != stage1_mismatch_spec():
    raise ValueError("Stage2 retention gate mismatch profile does not match.")
  if envelope.get("gate_pass") is not True:
    raise ValueError("Stage2 requires a passing Stage1-B retention gate.")
  return {
    "suite": "residual",
    "task": HYBRID_STAGE_TASKS[1],
    "evaluation_profile": requested_profile,
    "seed": seed,
    "git_sha": git_sha,
    "checkpoint_file_sha256": checkpoint_file_sha256,
    "stage1_profile_version": STAGE1_MISMATCH_PROFILE_VERSION,
    "gate_pass": True,
  }


def _write_or_print(result: Mapping[str, object], output: Path | None) -> None:
  encoded = to_deterministic_json(result)
  if output is None:
    print(encoded, end="")
    return
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(encoded, encoding="utf-8")


def _git_sha() -> str:
  completed = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=REPOSITORY_PATH,
    check=True,
    capture_output=True,
    text=True,
  )
  return completed.stdout.strip()


def validate_hybrid_evaluation_checkpoint(
  task: str,
  env_cfg: object,
  checkpoint: Mapping[str, object],
  *,
  evaluation_git_sha: str,
) -> None:
  """Bind a live gate to a qualified checkpoint and training revision."""

  validate_hybrid_training_checkpoint(task, env_cfg, checkpoint)
  infos = checkpoint.get("infos")
  extension = (
    infos.get("hybrid_stage1_extension")
    if isinstance(infos, Mapping)
    else None
  )
  if task == HYBRID_STAGE_TASKS[1] and (
    not isinstance(extension, Mapping)
    or extension.get("target_profile_version")
    != STAGE1_MISMATCH_PROFILE_VERSION
  ):
    raise ValueError(
      "Stage1-B evaluation requires matching extension provenance."
    )
  training = infos.get("hybrid_training") if isinstance(infos, Mapping) else None
  if not isinstance(training, Mapping):
    raise ValueError(
      "Hybrid evaluation checkpoint is missing training provenance."
    )
  training_git_sha = training.get("git_sha")
  if training_git_sha != evaluation_git_sha:
    raise ValueError(
      "Hybrid checkpoint training git SHA does not match evaluation code: "
      f"{training_git_sha!r} != {evaluation_git_sha!r}."
    )
  stage = int(task.rsplit("Stage", 1)[1])
  action = getattr(env_cfg, "actions", {}).get("hybrid_wheel_leg")
  expected_scales = tuple(
    float(value)
    for value in getattr(
      action, "action_scales", HYBRID_STAGES[stage].action_scales
    )
  )
  training_scales = training.get("action_scales")
  active_indices = [
    index
    for index, enabled in enumerate(HYBRID_STAGES[stage].action_mask)
    if enabled
  ]
  if training_scales is None:
    if any(
      expected_scales[index] != HYBRID_STAGES[stage].action_scales[index]
      for index in active_indices
    ):
      raise ValueError(
        "Hybrid checkpoint predates action-scale provenance and cannot be "
        "evaluated under an experimental leg authority."
      )
  else:
    recorded_scales = tuple(float(value) for value in training_scales)
    if len(recorded_scales) != len(expected_scales) or any(
      recorded_scales[index] != expected_scales[index]
      for index in active_indices
    ):
      raise ValueError(
        "Hybrid checkpoint training action scales do not match evaluation."
      )


@contextlib.contextmanager
def _policy_session(
  *,
  task: str,
  checkpoint: Path | None,
  args: argparse.Namespace,
  play: bool,
  stage1_mismatch_group_count: int | None = None,
  stage1_mismatch_group_index: int | None = None,
) -> Iterator[tuple[RslRlVecEnvWrapper, Any, Any]]:
  torch.manual_seed(args.seed)
  env_cfg = load_env_cfg(task, play=play)
  if stage1_mismatch_group_index is not None:
    env_cfg.events["stage1_mild_mismatch"] = stage1_mismatch_event_cfg(
      gate_group_count=stage1_mismatch_group_count,
      gate_group_index=stage1_mismatch_group_index,
    )
    setattr(env_cfg, "stage1_profile_version", STAGE1_MISMATCH_PROFILE_VERSION)
  agent_cfg = load_rl_cfg(task)
  if hasattr(env_cfg, "seed"):
    env_cfg.seed = args.seed
  if hasattr(agent_cfg, "seed"):
    agent_cfg.seed = args.seed
  env_cfg.episode_length_s = args.episode_length_s
  env_cfg.scene.num_envs = args.num_envs
  if env_cfg.scene.terrain is not None:
    env_cfg.scene.terrain.num_envs = args.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  try:
    if checkpoint is None:
      action_dim = int(wrapped.unwrapped.action_manager.total_action_dim)

      def policy(obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
          (obs.shape[0], action_dim),
          dtype=obs.dtype,
          device=obs.device,
        )
    else:
      runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
      runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
      runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=args.device,
      )
      policy = runner.get_inference_policy(device=args.device)
      if getattr(args, "ablate_leg_residuals", False):
        base_policy = policy

        def policy(obs: torch.Tensor) -> torch.Tensor:
          actions = base_policy(obs).clone()
          actions[:, 2:6] = 0.0
          return actions
    yield wrapped, policy, env_cfg
  finally:
    wrapped.close()


def _fixed_command_args(
  args: argparse.Namespace,
  task: str,
) -> SimpleNamespace:
  return SimpleNamespace(
    task=task,
    yaw=0.0,
    num_envs=args.num_envs,
    steps=args.steps,
    warmup_steps=args.warmup_steps,
    window_steps=args.window_steps,
    device=args.device,
    stuck_speed=0.01,
    reverse_speed=-0.01,
    progress_interval=args.progress_interval,
    play_cfg=True,
    episode_length_s=args.episode_length_s,
    constant_action=None,
  )


def _fixed_yaw_args(
  args: argparse.Namespace,
  task: str,
  lin_x: float,
) -> SimpleNamespace:
  return SimpleNamespace(
    task=task,
    lin_x=lin_x,
    num_envs=args.num_envs,
    steps=args.steps,
    warmup_steps=args.warmup_steps,
    window_steps=args.window_steps,
    device=args.device,
    yaw_deadband=0.01,
    lin_deadband=0.01,
    lin_drift_speed=0.05,
    progress_interval=args.progress_interval,
    play_cfg=True,
    episode_length_s=args.episode_length_s,
  )


def _force_posture(wrapped: RslRlVecEnvWrapper, height: float, pitch: float) -> None:
  term = wrapped.unwrapped.command_manager.get_term("posture")
  command = getattr(term, "_command", None)
  if command is None:
    raise AttributeError("Posture command term does not expose _command.")
  # Gate scenarios hold static postures: snap the raw target and the shaped
  # command together so reference shaping never generates a ramp here.
  target = getattr(term, "_target", None)
  if target is not None:
    target[:, 0] = height
    target[:, 1] = pitch
  command[:, 0] = height
  command[:, 1] = pitch


def _pitch(robot_data: Any) -> torch.Tensor:
  gravity = robot_data.projected_gravity_b
  return torch.atan2(
    gravity[:, 0],
    torch.clamp(-gravity[:, 2], min=1.0e-6),
  )


def _force_velocity_commands(
  env: ManagerBasedRlEnv,
  commands: torch.Tensor,
) -> None:
  term = env.command_manager.get_term("twist")
  term.vel_command_b[:, :] = 0.0
  term.vel_command_w[:, :] = 0.0
  term.vel_command_b[:, 0] = commands
  term.vel_command_w[:, 0] = commands
  for attr in ("is_standing_env", "is_heading_env", "is_world_env", "is_forward_env"):
    value = getattr(term, attr, None)
    if value is not None:
      value[:] = False


def _apply_stage1_kick(
  wrapped: RslRlVecEnvWrapper,
  env_ids: torch.Tensor,
  *,
  kick_index: int,
) -> None:
  asset = wrapped.unwrapped.scene["robot"]
  velocity = asset.data.root_link_vel_w[env_ids].clone()
  direction = 1.0 if kick_index % 2 == 0 else -1.0
  velocity[:, 0] += direction * STAGE1_KICK_LIN_X
  velocity[:, 4] += direction * STAGE1_KICK_PITCH_RATE
  asset.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)
  wrapped.unwrapped.sim.forward()
  wrapped.unwrapped.sim.sense()


def _safe_quantile(values: torch.Tensor, quantile: float) -> float:
  return (
    torch.quantile(values, quantile).item()
    if values.numel()
    else math.nan
  )


def _settling_time_s(
  healthy: torch.Tensor,
  event_indices: Sequence[int],
) -> float:
  times: list[float] = []
  total_steps = healthy.shape[0]
  for event_number, start in enumerate(event_indices):
    stop = (
      event_indices[event_number + 1]
      if event_number + 1 < len(event_indices)
      else total_steps
    )
    if start >= stop:
      continue
    segment = healthy[start:stop]
    for env_index in range(segment.shape[1]):
      found = stop - start
      run = 0
      for offset, value in enumerate(segment[:, env_index].tolist()):
        run = run + 1 if bool(value) else 0
        if run >= STAGE1_HEALTHY_HOLD_STEPS:
          found = offset - STAGE1_HEALTHY_HOLD_STEPS + 1
          break
      times.append(found / CONTROL_FREQUENCY_HZ)
  return fmean(times) if times else math.nan


def _finite_scenario_coordinate(value: object) -> bool:
  return (
    isinstance(value, (int, float))
    and not isinstance(value, bool)
    and math.isfinite(float(value))
  )


def _is_center_and_four_corners(
  points: set[tuple[float, float]],
  *,
  tolerance: float = 1.0e-6,
) -> bool:
  heights = [point[0] for point in points]
  pitches = [point[1] for point in points]
  h_low, h_high = min(heights), max(heights)
  p_low, p_high = min(pitches), max(pitches)
  if h_high - h_low <= tolerance or p_high - p_low <= tolerance:
    return False
  expected = (
    (0.5 * (h_low + h_high), 0.5 * (p_low + p_high)),
    (h_low, p_low),
    (h_low, p_high),
    (h_high, p_low),
    (h_high, p_high),
  )
  return all(
    any(
      abs(actual_h - expected_h) <= tolerance
      and abs(actual_p - expected_p) <= tolerance
      for actual_h, actual_p in points
    )
    for expected_h, expected_p in expected
  )


def _mean_event_error_integral(
  error: torch.Tensor,
  event_indices: Sequence[int],
) -> float:
  """Return a duration-normalized mean integral over event windows."""

  integrals: list[float] = []
  for event_number, start in enumerate(event_indices):
    stop = (
      event_indices[event_number + 1]
      if event_number + 1 < len(event_indices)
      else error.shape[0]
    )
    if start >= stop:
      continue
    integrals.append(
      error[start:stop].mean(dim=1).sum().item() / CONTROL_FREQUENCY_HZ
    )
  return fmean(integrals) if integrals else math.nan


def _stage1_group_metrics(
  *,
  lin_x: torch.Tensor,
  command: torch.Tensor,
  pitch_abs: torch.Tensor,
  pitch_rate_abs: torch.Tensor,
  residual_abs: torch.Tensor,
  terminated_event_rate: float,
) -> dict[str, float]:
  error = (lin_x - command).abs()
  delta = lin_x[1:] - lin_x[:-1]
  return {
    "mean_actual_lin_x": lin_x.mean().item(),
    "mean_abs_error": error.mean().item(),
    "p95_pitch": _safe_quantile(pitch_abs.flatten(), 0.95),
    "p99_pitch_rate": _safe_quantile(pitch_rate_abs.flatten(), 0.99),
    "lin_x_delta_rms": torch.sqrt(torch.mean(torch.square(delta))).item(),
    "balance_residual_abs_mean": residual_abs.mean().item(),
    "balance_residual_abs_p95": _safe_quantile(residual_abs.flatten(), 0.95),
    "terminated_event_rate": terminated_event_rate,
  }


def _run_stage1_profile(
  *,
  wrapped: RslRlVecEnvWrapper,
  policy: Any,
  args: argparse.Namespace,
  label: str,
) -> list[dict[str, object]]:
  if args.num_envs < 2 * STAGE1_PROFILE_GROUPS:
    raise ValueError(
      "Stage1 ablation gate requires --num-envs >= "
      f"{2 * STAGE1_PROFILE_GROUPS}; got {args.num_envs}."
    )
  # Runner construction/loading may consume the global RNG. Reseed immediately
  # before reset so candidate and zero-residual LQR receive matched starts.
  torch.manual_seed(args.seed)
  wrapped.reset()
  device = torch.device(args.device)
  all_ids = torch.arange(args.num_envs, device=device)
  group_ids = [
    all_ids[all_ids.remainder(STAGE1_PROFILE_GROUPS) == index]
    for index in range(STAGE1_PROFILE_GROUPS)
  ]
  disturbance_ids = group_ids[-3]
  transition_ids = group_ids[-2]
  mismatch_ids = group_ids[-1]
  robot = wrapped.unwrapped.scene["robot"]
  commands = torch.zeros(args.num_envs, device=device)
  lin_x_samples: list[torch.Tensor] = []
  command_samples: list[torch.Tensor] = []
  pitch_samples: list[torch.Tensor] = []
  pitch_rate_samples: list[torch.Tensor] = []
  residual_samples: list[torch.Tensor] = []
  terminated_by_group = [0 for _ in range(STAGE1_PROFILE_GROUPS)]
  transition_sequence = (0.0, 0.07, 0.0, -0.07, 0.0, 0.07, -0.07)
  measured_steps = args.steps - args.warmup_steps
  transition_segment_steps = measured_steps // len(transition_sequence)
  kick_interval = max(200, min(400, args.window_steps))
  kick_steps = list(range(
    args.warmup_steps + kick_interval,
    args.steps,
    kick_interval,
  ))
  kick_relative_indices: list[int] = []
  kick_number_by_step = {step: index for index, step in enumerate(kick_steps)}
  transition_relative_indices: list[int] = []
  previous_transition_command: float | None = None

  for step in range(args.steps):
    commands.zero_()
    for command_value, ids in zip(STAGE1_PROFILE_COMMANDS, group_ids[:5]):
      commands[ids] = command_value
    measured_index = max(step - args.warmup_steps, 0)
    transition_index = min(
      measured_index // transition_segment_steps,
      len(transition_sequence) - 1,
    )
    transition_command = transition_sequence[transition_index]
    commands[transition_ids] = transition_command
    mismatch_signs = torch.where(
      torch.arange(len(mismatch_ids), device=device).remainder(2) == 0,
      torch.ones(len(mismatch_ids), device=device),
      -torch.ones(len(mismatch_ids), device=device),
    )
    commands[mismatch_ids] = 0.07 * mismatch_signs
    if (
      step >= args.warmup_steps
      and transition_command != previous_transition_command
    ):
      transition_relative_indices.append(step - args.warmup_steps)
    if step >= args.warmup_steps:
      previous_transition_command = transition_command

    _force_velocity_commands(wrapped.unwrapped, commands)
    if step in kick_number_by_step:
      kick_number = kick_number_by_step[step]
      _apply_stage1_kick(
        wrapped,
        disturbance_ids,
        kick_index=kick_number,
      )
      kick_relative_indices.append(step - args.warmup_steps)
    with torch.no_grad():
      obs = wrapped.get_observations()
      actions = policy(obs).detach()
      _obs, _reward, _done, _extras = wrapped.step(actions)
    _force_velocity_commands(wrapped.unwrapped, commands)

    terminated = boolean_mask_on_device(
      wrapped.unwrapped.reset_terminated,
      commands,
    )
    for index, ids in enumerate(group_ids):
      terminated_by_group[index] += int(terminated[ids].sum().item())

    if step >= args.warmup_steps:
      robot_data = robot.data
      wheel_action = resolve_wheel_action(wrapped.unwrapped.action_manager)
      residual = (
        actions[:, 0]
        if wheel_action.applied_residual is None
        else wheel_action.applied_residual[:, 0]
      )
      lin_x_samples.append(robot_data.root_link_lin_vel_b[:, 0].detach().cpu())
      command_samples.append(commands.detach().cpu().clone())
      pitch_samples.append(_pitch(robot_data).abs().detach().cpu())
      pitch_rate_samples.append(
        robot_data.root_link_ang_vel_b[:, 1].abs().detach().cpu()
      )
      residual_samples.append(residual.abs().detach().cpu())

    if args.progress_interval > 0 and (
      (step + 1) % args.progress_interval == 0 or step + 1 == args.steps
    ):
      print(
        f"PROGRESS stage1_profile={label} step={step + 1}/{args.steps} "
        f"terminated={sum(terminated_by_group)}",
        flush=True,
      )

  lin_x = torch.stack(lin_x_samples)
  command = torch.stack(command_samples)
  pitch_abs = torch.stack(pitch_samples)
  pitch_rate_abs = torch.stack(pitch_rate_samples)
  residual_abs = torch.stack(residual_samples)
  scenarios: list[dict[str, object]] = []
  for index, command_value in enumerate(STAGE1_PROFILE_COMMANDS):
    ids = group_ids[index].cpu()
    kind = "nominal" if abs(command_value) <= 0.070001 else "extension"
    metrics = _stage1_group_metrics(
      lin_x=lin_x[:, ids],
      command=command[:, ids],
      pitch_abs=pitch_abs[:, ids],
      pitch_rate_abs=pitch_rate_abs[:, ids],
      residual_abs=residual_abs[:, ids],
      terminated_event_rate=terminated_by_group[index] / len(ids),
    )
    scenarios.append({
      "name": f"stage1_{kind}_vx_{command_value:+.3f}",
      "kind": kind,
      "lin_x": command_value,
      "metrics": metrics,
    })

  disturbance_cpu = disturbance_ids.cpu()
  disturbance_lin = lin_x[:, disturbance_cpu]
  disturbance_command = command[:, disturbance_cpu]
  disturbance_pitch = pitch_abs[:, disturbance_cpu]
  disturbance_pitch_rate = pitch_rate_abs[:, disturbance_cpu]
  disturbance_error = (disturbance_lin - disturbance_command).abs()
  disturbance_healthy = (
    (disturbance_error <= STAGE1_HEALTHY_ERROR)
    & (disturbance_pitch <= STAGE1_HEALTHY_PITCH)
    & (disturbance_pitch_rate <= STAGE1_HEALTHY_PITCH_RATE)
  )
  disturbance_metrics = _stage1_group_metrics(
    lin_x=disturbance_lin,
    command=disturbance_command,
    pitch_abs=disturbance_pitch,
    pitch_rate_abs=disturbance_pitch_rate,
    residual_abs=residual_abs[:, disturbance_cpu],
    terminated_event_rate=terminated_by_group[-3] / len(disturbance_cpu),
  )
  disturbance_metrics.update({
    "recovery_time_s": _settling_time_s(
      disturbance_healthy,
      kick_relative_indices,
    ),
    "post_kick_error_integral": _mean_event_error_integral(
      disturbance_error,
      kick_relative_indices,
    ),
    # Statistical power audit for the improvement protocol: recovery-time
    # estimates from ~4 events moved 26% between same-seed screen runs.
    "kick_event_count": float(
      len(kick_relative_indices) * len(disturbance_cpu)
    ),
  })
  scenarios.append({
    "name": "stage1_disturbance_recovery",
    "kind": "disturbance",
    "metrics": disturbance_metrics,
  })

  transition_cpu = transition_ids.cpu()
  transition_lin = lin_x[:, transition_cpu]
  transition_command_values = command[:, transition_cpu]
  transition_pitch = pitch_abs[:, transition_cpu]
  transition_pitch_rate = pitch_rate_abs[:, transition_cpu]
  transition_error = (transition_lin - transition_command_values).abs()
  transition_healthy = (
    (transition_error <= STAGE1_HEALTHY_ERROR)
    & (transition_pitch <= STAGE1_HEALTHY_PITCH)
    & (transition_pitch_rate <= STAGE1_HEALTHY_PITCH_RATE)
  )
  overshoots: list[float] = []
  for event_number, start in enumerate(transition_relative_indices):
    stop = (
      transition_relative_indices[event_number + 1]
      if event_number + 1 < len(transition_relative_indices)
      else transition_lin.shape[0]
    )
    target = float(transition_command_values[start, 0].item())
    if abs(target) <= 1.0e-12:
      continue
    signed = math.copysign(1.0, target) * (
      transition_lin[start:stop] - target
    )
    overshoots.append(
      torch.clamp(signed, min=0.0).amax(dim=0).mean().item()
    )
  transition_metrics = _stage1_group_metrics(
    lin_x=transition_lin,
    command=transition_command_values,
    pitch_abs=transition_pitch,
    pitch_rate_abs=transition_pitch_rate,
    residual_abs=residual_abs[:, transition_cpu],
    terminated_event_rate=terminated_by_group[-2] / len(transition_cpu),
  )
  transition_metrics.update({
    "settling_time_s": _settling_time_s(
      transition_healthy,
      transition_relative_indices[1:],
    ),
    "tracking_error_integral": _mean_event_error_integral(
      transition_error,
      transition_relative_indices[1:],
    ),
    "overshoot_abs_mean": fmean(overshoots) if overshoots else math.nan,
    "transition_event_count": float(
      len(transition_relative_indices[1:]) * len(transition_cpu)
    ),
  })
  scenarios.append({
    "name": "stage1_command_transition",
    "kind": "transition",
    "metrics": transition_metrics,
  })
  mismatch_cpu = mismatch_ids.cpu()
  mismatch_metrics = _stage1_group_metrics(
    lin_x=lin_x[:, mismatch_cpu],
    command=command[:, mismatch_cpu],
    pitch_abs=pitch_abs[:, mismatch_cpu],
    pitch_rate_abs=pitch_rate_abs[:, mismatch_cpu],
    residual_abs=residual_abs[:, mismatch_cpu],
    terminated_event_rate=terminated_by_group[-1] / len(mismatch_cpu),
  )
  mismatch_parameters = stage1_mismatch_audit(
    wrapped.unwrapped
  ).select(mismatch_ids)
  scenarios.append({
    "name": "stage1_model_mismatch",
    "kind": "mismatch",
    "mismatch_parameters": mismatch_parameters,
    "metrics": mismatch_metrics,
  })
  return scenarios


def _merge_stage1_profiles(
  candidate: Sequence[Mapping[str, object]],
  baseline: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
  baseline_by_name = {_scenario_name(row): row for row in baseline}
  merged: list[dict[str, object]] = []
  for row in candidate:
    name = _scenario_name(row)
    reference = baseline_by_name.get(name)
    if reference is None:
      raise ValueError(f"Stage1 baseline is missing scenario: {name}")
    candidate_metrics = row.get("metrics")
    baseline_metrics = reference.get("metrics")
    if not isinstance(candidate_metrics, Mapping) or not isinstance(
      baseline_metrics, Mapping
    ):
      raise ValueError("Stage1 profile scenarios require metrics objects.")
    if row.get("kind") == "mismatch" and (
      row.get("mismatch_parameters") != reference.get("mismatch_parameters")
    ):
      raise ValueError(
        "Stage1 candidate and baseline mismatch parameters are not matched."
      )
    metrics = {
      f"candidate_{key}": value for key, value in candidate_metrics.items()
    }
    metrics.update(
      {f"baseline_{key}": value for key, value in baseline_metrics.items()}
    )
    merged.append({
      key: value for key, value in row.items() if key != "metrics"
    } | {"metrics": metrics})
  return merged


def _scenario_name(scenario: Mapping[str, object]) -> str:
  name = scenario.get("name")
  if not isinstance(name, str) or not name:
    raise ValueError("Stage1 profile scenario requires a non-empty name.")
  return name


def _collect_stage1_comparison(
  *,
  task: str,
  checkpoint: Path,
  args: argparse.Namespace,
) -> list[dict[str, object]]:
  with _policy_session(
    task=task,
    checkpoint=checkpoint,
    args=args,
    play=True,
    stage1_mismatch_group_count=STAGE1_PROFILE_GROUPS,
    stage1_mismatch_group_index=STAGE1_PROFILE_GROUPS - 1,
  ) as (wrapped, policy, _env_cfg):
    candidate = _run_stage1_profile(
      wrapped=wrapped,
      policy=policy,
      args=args,
      label="trained_residual",
    )
  with _policy_session(
    task=task,
    checkpoint=None,
    args=args,
    play=True,
    stage1_mismatch_group_count=STAGE1_PROFILE_GROUPS,
    stage1_mismatch_group_index=STAGE1_PROFILE_GROUPS - 1,
  ) as (wrapped, policy, _env_cfg):
    baseline = _run_stage1_profile(
      wrapped=wrapped,
      policy=policy,
      args=args,
      label="zero_residual_lqr",
    )
  return _merge_stage1_profiles(candidate, baseline)


def _run_posture_scenario(
  *,
  wrapped: RslRlVecEnvWrapper,
  policy: Any,
  args: argparse.Namespace,
  target_height: float,
  target_pitch: float,
) -> dict[str, float]:
  wrapped.reset()
  height_errors: list[torch.Tensor] = []
  pitch_errors: list[torch.Tensor] = []
  non_wheel_contacts: list[torch.Tensor] = []
  terminated_events = 0
  robot = wrapped.unwrapped.scene["robot"]
  for step in range(args.steps):
    with torch.no_grad():
      _force_velocity_command(wrapped.unwrapped, 0.0, 0.0)
      _force_posture(wrapped, target_height, target_pitch)
      obs = wrapped.get_observations()
      _obs, _reward, _done, _extras = wrapped.step(policy(obs).detach())
      _force_velocity_command(wrapped.unwrapped, 0.0, 0.0)
      _force_posture(wrapped, target_height, target_pitch)
      terminated_events += int(wrapped.unwrapped.reset_terminated.sum().item())
      if step >= args.warmup_steps:
        robot_data = robot.data
        height_errors.append(
          (robot_data.root_link_pos_w[:, 2] - target_height).detach().cpu()
        )
        pitch_errors.append((_pitch(robot_data) - target_pitch).detach().cpu())
        non_wheel_contacts.append(
          non_wheel_ground_contact(
            wrapped.unwrapped,
            NON_WHEEL_GROUND_SENSOR_NAME,
          ).detach().cpu()
        )
  height_error = torch.cat(height_errors)
  pitch_error = torch.cat(pitch_errors)
  contact = torch.cat(non_wheel_contacts).float()
  return {
    "height_rmse": torch.sqrt(torch.mean(torch.square(height_error))).item(),
    "pitch_rmse": torch.sqrt(torch.mean(torch.square(pitch_error))).item(),
    "non_wheel_contact_rate": contact.mean().item(),
    "terminated_event_rate": terminated_events / max(args.num_envs, 1),
  }


def _run_integrated_rollout(
  *,
  wrapped: RslRlVecEnvWrapper,
  policy: Any,
  args: argparse.Namespace,
  force_commands: bool,
) -> dict[str, float]:
  wrapped.reset()
  robot = wrapped.unwrapped.scene["robot"]
  posture_cfg = wrapped.unwrapped.command_manager.get_term("posture").cfg
  height = 0.5 * sum(posture_cfg.height_range)
  pitch = 0.5 * sum(posture_cfg.pitch_range)
  lin_x = 0.07
  yaw = 0.10
  errors: list[torch.Tensor] = []
  non_wheel_contacts: list[torch.Tensor] = []
  wheel_target_abses: list[torch.Tensor] = []
  terminated_events = 0
  ever_terminated = torch.zeros(
    args.num_envs,
    dtype=torch.bool,
    device=args.device,
  )
  unhealthy_streak = torch.zeros(args.num_envs, device=args.device)
  maximum_streak = torch.zeros_like(unhealthy_streak)

  for step in range(args.steps):
    with torch.no_grad():
      if force_commands:
        _force_velocity_command(wrapped.unwrapped, lin_x, yaw)
        _force_posture(wrapped, height, pitch)
      obs = wrapped.get_observations()
      _obs, _reward, done, _extras = wrapped.step(policy(obs).detach())
      if force_commands:
        _force_velocity_command(wrapped.unwrapped, lin_x, yaw)
        _force_posture(wrapped, height, pitch)
      reset_terminated = boolean_mask_on_device(
        wrapped.unwrapped.reset_terminated,
        ever_terminated,
      )
      terminated_events += int(reset_terminated.sum().item())
      ever_terminated |= reset_terminated
      if step < args.warmup_steps:
        continue

      velocity_command = wrapped.unwrapped.command_manager.get_command("twist")
      posture_command = wrapped.unwrapped.command_manager.get_command("posture")
      robot_data = robot.data
      lin_error = robot_data.root_link_lin_vel_b[:, 0] - velocity_command[:, 0]
      yaw_error = robot_data.root_link_ang_vel_b[:, 2] - velocity_command[:, 2]
      height_error = robot_data.root_link_pos_w[:, 2] - posture_command[:, 0]
      pitch_error = _pitch(robot_data) - posture_command[:, 1]
      component_errors = torch.stack(
        (lin_error, yaw_error, height_error, pitch_error),
        dim=1,
      )
      errors.append(component_errors.detach().cpu())
      non_wheel_contacts.append(
        non_wheel_ground_contact(
          wrapped.unwrapped,
          NON_WHEEL_GROUND_SENSOR_NAME,
        ).detach().cpu()
      )
      wheel_view = resolve_wheel_action(wrapped.unwrapped.action_manager)
      wheel_target_abses.append(wheel_view.wheel_targets.abs().detach().cpu())

      healthy = (
        (lin_error.abs() <= 0.06)
        & (yaw_error.abs() <= 0.08)
        & (height_error.abs() <= 0.015)
        & (pitch_error.abs() <= 0.04)
      )
      unhealthy_streak = torch.where(
        healthy | boolean_mask_on_device(done, unhealthy_streak),
        torch.zeros_like(unhealthy_streak),
        unhealthy_streak + 1.0,
      )
      maximum_streak = torch.maximum(maximum_streak, unhealthy_streak)

  error = torch.cat(errors)
  contact = torch.cat(non_wheel_contacts).float()
  wheel_target_abs = torch.cat(wheel_target_abses)
  wheel_action = resolve_wheel_action(wrapped.unwrapped.action_manager).term
  saturation_threshold = wheel_target_saturation_threshold(wheel_action)
  survival_rate = _survival_rate(ever_terminated.detach().cpu().tolist())
  return {
    "tracking_error": torch.sqrt(torch.mean(torch.sum(torch.square(error), dim=1))).item(),
    "terminated_event_rate": terminated_events / max(args.num_envs, 1),
    "survival_rate": survival_rate,
    "recovery_time_s": maximum_streak.max().item() / CONTROL_FREQUENCY_HZ,
    "non_wheel_contact_rate": contact.mean().item(),
    "wheel_saturation_ratio": (
      (wheel_target_abs >= saturation_threshold).float().mean().item()
    ),
  }


def _posture_targets(wrapped: RslRlVecEnvWrapper) -> list[tuple[float, float]]:
  cfg = wrapped.unwrapped.command_manager.get_term("posture").cfg
  h_low, h_high = (float(value) for value in cfg.height_range)
  p_low, p_high = (float(value) for value in cfg.pitch_range)
  center = (0.5 * (h_low + h_high), 0.5 * (p_low + p_high))
  return [
    center,
    (h_low, p_low),
    (h_low, p_high),
    (h_high, p_low),
    (h_high, p_high),
  ]


# Stage5 pre-registered primary metric (log section 3.8): recovery from
# the 8x Stage1 kick at the envelope-center posture. Baseline (zero
# residual) measured 0.970 s with 0.009 s run noise; the bar is 10%.
STAGE5_RECOVERY_KICK_SCALE = 8.0
STAGE5_RECOVERY_KICK_INTERVAL = 300
STAGE5_RECOVERY_KICKS = 4


def _apply_scaled_stage1_kick(
  wrapped: RslRlVecEnvWrapper,
  env_ids: torch.Tensor,
  *,
  kick_index: int,
  scale: float,
) -> None:
  asset = wrapped.unwrapped.scene["robot"]
  velocity = asset.data.root_link_vel_w[env_ids].clone()
  direction = 1.0 if kick_index % 2 == 0 else -1.0
  velocity[:, 0] += direction * scale * STAGE1_KICK_LIN_X
  velocity[:, 4] += direction * scale * STAGE1_KICK_PITCH_RATE
  asset.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)
  wrapped.unwrapped.sim.forward()
  wrapped.unwrapped.sim.sense()


def _run_recovery_scenario(
  *,
  wrapped: RslRlVecEnvWrapper,
  policy: Any,
  args: argparse.Namespace,
  target_height: float,
  target_pitch: float,
  kick_scale: float = STAGE5_RECOVERY_KICK_SCALE,
) -> dict[str, float]:
  """Hold the center posture and recover from repeated 8x kicks.

  Same healthy bands as the posture suite (commanded-relative), same kick
  mechanics as Stage1 scaled to the pre-registered magnitude, same
  settling-time estimator, so the candidate and the zero-residual baseline
  produce directly comparable recovery times.
  """

  wrapped.reset()
  device = torch.device(args.device)
  env_ids = torch.arange(args.num_envs, device=device)
  settle_steps = args.warmup_steps
  kick_steps = {
    settle_steps + index * STAGE5_RECOVERY_KICK_INTERVAL: index
    for index in range(STAGE5_RECOVERY_KICKS)
  }
  total_steps = settle_steps + STAGE5_RECOVERY_KICKS * STAGE5_RECOVERY_KICK_INTERVAL
  robot = wrapped.unwrapped.scene["robot"]
  healthy_rows: list[torch.Tensor] = []
  contacts: list[torch.Tensor] = []
  leg_residuals: list[torch.Tensor] = []
  wheel_target_abses: list[torch.Tensor] = []
  terminated_events = 0
  for step in range(total_steps):
    _force_velocity_command(wrapped.unwrapped, 0.0, 0.0)
    _force_posture(wrapped, target_height, target_pitch)
    if step in kick_steps:
      _apply_scaled_stage1_kick(
        wrapped,
        env_ids,
        kick_index=kick_steps[step],
        scale=kick_scale,
      )
    with torch.no_grad():
      obs = wrapped.get_observations()
      _obs, _reward, _done, _extras = wrapped.step(policy(obs).detach())
    _force_velocity_command(wrapped.unwrapped, 0.0, 0.0)
    _force_posture(wrapped, target_height, target_pitch)
    terminated_events += int(wrapped.unwrapped.reset_terminated.sum().item())
    if step < settle_steps:
      continue
    data = robot.data
    lin_error = data.root_link_lin_vel_b[:, 0]
    yaw_error = data.root_link_ang_vel_b[:, 2]
    height_error = data.root_link_pos_w[:, 2] - target_height
    pitch_error = _pitch(data) - target_pitch
    healthy = (
      (lin_error.abs() <= 0.06)
      & (yaw_error.abs() <= 0.08)
      & (height_error.abs() <= 0.015)
      & (pitch_error.abs() <= 0.04)
    )
    healthy_rows.append(healthy.detach().cpu())
    contacts.append(
      non_wheel_ground_contact(
        wrapped.unwrapped,
        NON_WHEEL_GROUND_SENSOR_NAME,
      ).detach().cpu()
    )
    wheel_view = resolve_wheel_action(wrapped.unwrapped.action_manager)
    applied = wheel_view.applied_residual
    if applied is not None:
      leg_residuals.append(applied[:, 2:6].detach().cpu())
    wheel_target_abses.append(wheel_view.wheel_targets.abs().detach().cpu())

  healthy_series = torch.stack(healthy_rows)
  contact = torch.stack(contacts).float()
  leg_residual = torch.cat(leg_residuals) if leg_residuals else torch.zeros(1, 4)
  wheel_target_abs = torch.cat(wheel_target_abses)
  action_cfg = wrapped.unwrapped.cfg.actions["hybrid_wheel_leg"]
  leg_scale = float(action_cfg.action_scales[2])
  leg_saturation_rate = (
    (leg_residual.abs() >= 0.99 * leg_scale).float().mean().item()
    if leg_scale > 0.0
    else 0.0
  )
  wheel_action = resolve_wheel_action(wrapped.unwrapped.action_manager).term
  wheel_saturation = wheel_target_saturation_threshold(wheel_action)
  kick_relative = [step - settle_steps for step in sorted(kick_steps)]
  return {
    "recovery_time_s": _settling_time_s(healthy_series, kick_relative),
    "kick_event_count": float(STAGE5_RECOVERY_KICKS * args.num_envs),
    "non_wheel_contact_rate": contact.mean().item(),
    "terminated_event_rate": terminated_events / max(args.num_envs, 1),
    "leg_residual_abs_mean": leg_residual.abs().mean().item(),
    "leg_residual_abs_max": leg_residual.abs().max().item(),
    "leg_residual_saturation_rate": leg_saturation_rate,
    "wheel_saturation_ratio": (
      (wheel_target_abs >= wheel_saturation).float().mean().item()
    ),
  }


def _collect_recovery_comparison(
  *,
  task: str,
  checkpoint: Path,
  args: argparse.Namespace,
) -> dict[str, float]:
  """Center-posture large-kick recovery: candidate vs zero-residual LQR."""

  center = _posture_targets_from_cfg(task)
  with _policy_session(
    task=task, checkpoint=checkpoint, args=args, play=True
  ) as (wrapped, policy, _env_cfg):
    candidate = _run_recovery_scenario(
      wrapped=wrapped,
      policy=policy,
      args=args,
      target_height=center[0],
      target_pitch=center[1],
    )
  with _policy_session(
    task=task, checkpoint=None, args=args, play=True
  ) as (wrapped, policy, _env_cfg):
    baseline = _run_recovery_scenario(
      wrapped=wrapped,
      policy=policy,
      args=args,
      target_height=center[0],
      target_pitch=center[1],
    )
  merged: dict[str, float] = {}
  for key, value in candidate.items():
    merged[f"candidate_{key}"] = value
  for key, value in baseline.items():
    merged[f"baseline_{key}"] = value
  return merged


def _collect_stage4_reference_from_baseline(
  *,
  task: str,
  args: argparse.Namespace,
) -> dict[str, float]:
  """Measure the Route-A stage4 reference from the classical stack.

  No Stage4 training run exists (Route A: stages 3/4 are classically
  closed), so the honest no-regression reference for the robust
  fixed-command tracking check is the zero-residual classical stack's
  own fixed-command integrated tracking under the same push env the
  candidate robust_pushes scenario runs in (play=False).
  """

  with _policy_session(
    task=task, checkpoint=None, args=args, play=False
  ) as (wrapped, policy, _env_cfg):
    return _run_integrated_rollout(
      wrapped=wrapped,
      policy=policy,
      args=args,
      force_commands=True,
    )


def _posture_targets_from_cfg(task: str) -> tuple[float, float]:
  env_cfg = load_env_cfg(task, play=True)
  cfg = env_cfg.commands["posture"]
  h_low, h_high = (float(value) for value in cfg.height_range)
  p_low, p_high = (float(value) for value in cfg.pitch_range)
  return (0.5 * (h_low + h_high), 0.5 * (p_low + p_high))


def _collect_scenarios(
  *,
  suite: str,
  task: str,
  checkpoint: Path | None,
  args: argparse.Namespace,
) -> list[dict[str, object]]:
  if suite == "residual":
    if checkpoint is None:
      raise ValueError("Stage1 residual comparison requires a checkpoint.")
    return _collect_stage1_comparison(
      task=task,
      checkpoint=checkpoint,
      args=args,
    )
  # Branch C channel isolation: the fine-grained tracking scenarios
  # (linear/yaw/combo/posture) are a classical qualification and are
  # collected in a clean env (play=True) judged by calibrated thresholds.
  # The robust suite's disturbance evidence is carried separately by the
  # controlled, measured recovery scenario plus the push-env robustness
  # scenarios below; it must not contaminate the calibrated tracking bands.
  play = True
  scenarios: list[dict[str, object]] = []
  with _policy_session(
    task=task,
    checkpoint=checkpoint,
    args=args,
    play=play,
  ) as (wrapped, policy, _env_cfg):
    # Pin the posture command to the envelope center while measuring the
    # velocity-tracking channel, so the calibrated pitch/tracking bands are
    # not contaminated by free-resampling posture commands (channel
    # isolation). No-op for pre-posture suites (no posture term).
    posture_center = (
      _posture_targets(wrapped)[0]
      if suite in ("posture", "integrated", "robust")
      else None
    )
    if suite in ("controller", "linear", "planar", "integrated", "robust"):
      if suite == "controller":
        linear_values = (-0.07, 0.0, 0.07)
      elif suite == "linear":
        linear_values = (-0.07, -0.04, 0.0, 0.04, 0.07)
      else:
        linear_values = (-0.07, 0.0, 0.07)
      command_args = _fixed_command_args(args, task)
      linear_rows = [
        _run_fixed_command(
          wrapped=wrapped,
          policy=policy,
          args=command_args,
          lin_x_cmd=lin_x,
          posture_target=posture_center,
        )
        for lin_x in linear_values
      ]
      for row in linear_rows:
        if (
          suite != 'controller'
          and abs(float(row.get('lin_x', 0.0))) <= 1.0e-12
        ):
          row['duration_s'] = args.steps / CONTROL_FREQUENCY_HZ
          scenarios.append(_linear_row_to_scenario(suite, row))
          continue
        row["duration_s"] = args.steps / CONTROL_FREQUENCY_HZ
        if abs(float(row["lin_x"])) <= 1.0e-12:
          scenarios.append(
            {
              "name": "controller_stand",
              "kind": "controller",
              "lin_x": 0.0,
              "metrics": row,
            }
          )
        elif suite == "controller":
          scenarios.append(
            {
              "name": f"controller_vx_{float(row['lin_x']):+.3f}",
              "kind": "controller",
              "lin_x": float(row["lin_x"]),
              "metrics": row,
            }
          )
        else:
          scenarios.extend(_fixed_rows_to_scenarios("linear", [row]))

    if suite in ("planar", "integrated", "robust"):
      yaw_rows: list[dict[str, float | str]] = []
      combo_rows: list[dict[str, float | str]] = []
      for lin_x in (0.0, -0.07, 0.07):
        yaw_args = _fixed_yaw_args(args, task, lin_x)
        for yaw in (-0.10, 0.10):
          row = _run_fixed_yaw(
            wrapped=wrapped,
            policy=policy,
            args=yaw_args,
            yaw_cmd=yaw,
            posture_target=posture_center,
          )
          (yaw_rows if lin_x == 0.0 else combo_rows).append(row)
      scenarios.extend(_fixed_rows_to_scenarios("yaw", yaw_rows))
      scenarios.extend(_fixed_rows_to_scenarios("combo", combo_rows))

    if suite in ("posture", "integrated", "robust"):
      for target_height, target_pitch in _posture_targets(wrapped):
        metrics = _run_posture_scenario(
          wrapped=wrapped,
          policy=policy,
          args=args,
          target_height=target_height,
          target_pitch=target_pitch,
        )
        scenarios.append(
          {
            "name": (
              f"posture_h_{target_height:+.4f}_p_{target_pitch:+.4f}"
            ),
            "kind": "posture",
            "target_height": target_height,
            "target_pitch": target_pitch,
            "metrics": metrics,
          }
        )

    if suite == "integrated":
      random_metrics = _run_integrated_rollout(
        wrapped=wrapped,
        policy=policy,
        args=args,
        force_commands=False,
      )
      scenarios.append(
        {
          "name": "random_integrated",
          "kind": "random",
          "metrics": random_metrics,
        }
      )
      fixed_metrics = _run_integrated_rollout(
        wrapped=wrapped,
        policy=policy,
        args=args,
        force_commands=True,
      )
      scenarios.append(
        {
          "name": (
            "robust_pushes" if suite == "robust" else "integrated_reference"
          ),
          "kind": "robust" if suite == "robust" else "reference",
          "metrics": fixed_metrics,
        }
      )
  if suite == "robust":
    # Robustness scenarios run in the training env (play=False) so the
    # stage's random pushes are active; they are judged only by the loose
    # ROBUST rules, never by the calibrated tracking bands above.
    with _policy_session(
      task=task,
      checkpoint=checkpoint,
      args=args,
      play=False,
    ) as (wrapped, policy, _env_cfg):
      random_metrics = _run_integrated_rollout(
        wrapped=wrapped,
        policy=policy,
        args=args,
        force_commands=False,
      )
      scenarios.append(
        {
          "name": "random_integrated",
          "kind": "random",
          "metrics": random_metrics,
        }
      )
      fixed_metrics = _run_integrated_rollout(
        wrapped=wrapped,
        policy=policy,
        args=args,
        force_commands=True,
      )
      scenarios.append(
        {
          "name": "robust_pushes",
          "kind": "robust",
          "metrics": fixed_metrics,
        }
      )
  return scenarios


def _scenario_file_payload(
  path: Path,
) -> tuple[list[dict[str, object]], str | None, dict[str, object]]:
  payload = _read_json(path)
  profile: str | None = None
  rollout: dict[str, object] = {}
  if isinstance(payload, Mapping):
    raw_profile = payload.get("evaluation_profile")
    profile = raw_profile if isinstance(raw_profile, str) else None
    raw_rollout = payload.get("rollout")
    if isinstance(raw_rollout, Mapping):
      rollout = dict(raw_rollout)
    payload = payload.get("scenarios")
  if not isinstance(payload, list) or not all(
    isinstance(scenario, dict) for scenario in payload
  ):
    raise ValueError("Scenario file must contain a scenario list.")
  return payload, profile, rollout


def _validate_scenario_file_profile(
  requested_profile: str,
  file_profile: str | None,
  rollout: Mapping[str, object],
) -> None:
  if requested_profile != "formal":
    return
  steps = rollout.get("steps")
  if file_profile != "formal" or not isinstance(steps, int) or steps < 3000:
    raise ValueError(
      "Formal scenario files must declare evaluation_profile='formal' "
      "and rollout.steps >= 3000."
    )


def _controller_hash(
  task: str,
  explicit_hash: str | None,
  *,
  require_loaded_match: bool = False,
) -> str:
  if require_loaded_match:
    env_cfg = load_env_cfg(task, play=True)
    action = env_cfg.actions.get('hybrid_wheel_leg')
    qualified = bool(
      action is not None
      and getattr(action, 'controller_qualified', False)
    )
    loaded_hash = (
      None
      if action is None
      else getattr(action, 'controller_gain_hash', None)
    )
    if not qualified or not isinstance(loaded_hash, str) or not loaded_hash.strip():
      raise ValueError(
        'Formal Hybrid live gate requires a qualified controller artifact '
        'loaded by the task environment.'
      )
    if explicit_hash is not None and explicit_hash != loaded_hash:
      raise ValueError(
        'Explicit controller gain hash does not match the controller '
        'loaded by the task environment.'
      )
    return loaded_hash

  if explicit_hash is not None:
    controller_hash = explicit_hash
  else:
    env_cfg = load_env_cfg(task, play=True)
    action = env_cfg.actions.get("hybrid_wheel_leg")
    controller_hash = (
      None if action is None else action.controller_gain_hash
    )
  if not isinstance(controller_hash, str) or not controller_hash.strip():
    raise ValueError(
      "Formal Hybrid gate evaluation requires a qualified controller "
      "artifact hash; provide --controller-gain-hash or configure one "
      "on the task action."
    )
  return controller_hash


def _calibration_hash(task: str, *, required: bool) -> str | None:
  env_cfg = load_env_cfg(task, play=True)
  action = env_cfg.actions.get("hybrid_wheel_leg")
  value = None if action is None else getattr(action, "calibration_hash", None)
  if required and (not isinstance(value, str) or not value.strip()):
    raise ValueError(
      "Formal Hybrid live gate requires a velocity calibration artifact "
      "loaded by the task environment."
    )
  return value if isinstance(value, str) and value.strip() else None


def _standing_diagnostic_summary(
  rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float]]:
  summary: dict[str, dict[str, float]] = {}
  for metric in ZERO_RESIDUAL_STANDING_METRICS:
    values: list[float] = []
    for row in rows:
      value = row.get(metric)
      if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
      ):
        raise ValueError(
          f"Zero-residual standing row has no finite {metric!r} metric."
        )
      values.append(float(value))
    summary[metric] = {
      "mean": fmean(values),
      "minimum": min(values),
      "maximum": max(values),
      "repeat_spread": max(values) - min(values),
    }
  return summary


def _diagnostic_check_payload(check: object) -> dict[str, object]:
  return {
    "name": getattr(check, "name"),
    "value": getattr(check, "value"),
    "operator": getattr(check, "operator"),
    "limit": getattr(check, "limit"),
    "pass": bool(getattr(check, "passed")),
    "scenario": getattr(check, "scenario"),
    "source": getattr(check, "source"),
  }


def _collect_zero_residual_standing_diagnostic(
  *,
  task: str,
  args: argparse.Namespace,
) -> list[dict[str, object]]:
  repeats: list[dict[str, object]] = []
  for repeat in range(args.diagnostic_repeats):
    with _policy_session(
      task=task,
      checkpoint=None,
      args=args,
      play=True,
    ) as (wrapped, policy, _env_cfg):
      posture_center = _posture_targets(wrapped)[0]
      row = _run_fixed_command(
        wrapped=wrapped,
        policy=policy,
        args=_fixed_command_args(args, task),
        lin_x_cmd=0.0,
        posture_target=posture_center,
      )
    row["duration_s"] = args.steps / CONTROL_FREQUENCY_HZ
    scenario = _linear_row_to_scenario("robust", row)
    checks = linear_scenario_checks(scenario)
    repeats.append(
      {
        "repeat": repeat + 1,
        "metrics": row,
        "checks": [_diagnostic_check_payload(check) for check in checks],
        "current_rules_pass": all(check.passed for check in checks),
      }
    )
  return repeats


def _zero_residual_standing_result(
  *,
  task: str,
  args: argparse.Namespace,
  git_sha: str,
  controller_gain_hash: str,
  calibration_hash: str | None,
) -> dict[str, object]:
  repeats = _collect_zero_residual_standing_diagnostic(task=task, args=args)
  metric_rows: list[Mapping[str, object]] = []
  for repeat in repeats:
    metrics = repeat.get("metrics")
    if not isinstance(metrics, Mapping):
      raise ValueError("Zero-residual standing repeat has no metrics object.")
    metric_rows.append(metrics)
  env_cfg = load_env_cfg(task, play=True)
  action = env_cfg.actions.get("hybrid_wheel_leg")
  if action is None:
    raise ValueError("Stage5 task has no hybrid_wheel_leg action config.")
  return {
    "schema_version": 1,
    "diagnostic": "hybrid_stage5_zero_residual_standing",
    "promotion_eligible": False,
    "task": task,
    "git_sha": git_sha,
    "seed": args.seed,
    "controller_gain_hash": controller_gain_hash,
    "calibration_hash": calibration_hash,
    "yaw_calibration_hash": getattr(action, "yaw_calibration_hash", None),
    "posture_map_hash": getattr(action, "posture_map_hash", None),
    "posture_artifact_hash": getattr(action, "posture_artifact_hash", None),
    "station_calibration_hash": getattr(
      action, "station_calibration_hash", None
    ),
    "action_scales": list(getattr(action, "action_scales", ())),
    "rollout": {
      "profile": args.profile,
      "num_envs": args.num_envs,
      "steps": args.steps,
      "warmup_steps": args.warmup_steps,
      "window_steps": args.window_steps,
      "episode_length_s": args.episode_length_s,
      "repeats": args.diagnostic_repeats,
      "policy": "native_zero_residual",
      "play_cfg": True,
    },
    "repeats": repeats,
    "summary": _standing_diagnostic_summary(metric_rows),
  }


def main() -> None:
  args = parse_args()
  _validate_zero_residual_standing_args(args)
  if args.aggregate_input is not None:
    results = [_read_json(path) for path in args.aggregate_input]
    if not all(isinstance(result, Mapping) for result in results):
      raise ValueError("Every aggregate input must contain a JSON object.")
    aggregate = aggregate_seed_results(
      [result for result in results if isinstance(result, Mapping)]
    )
    _write_or_print(aggregate, args.output)
    if not aggregate["gate_pass"]:
      raise SystemExit(1)
    return

  if args.stage is None:
    raise ValueError("--stage is required unless --aggregate-input is used.")
  configure_torch_backends()
  suite = HYBRID_STAGE_SUITES[args.stage]
  task = args.task or HYBRID_STAGE_TASKS[args.stage]
  checkpoint = (
    None
    if args.checkpoint_file is None
    else Path(args.checkpoint_file).expanduser().resolve()
  )
  if checkpoint is not None and not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
  if (
    checkpoint is None
    and args.stage != 0
    and args.scenario_file is None
    and not args.zero_residual_standing_diagnostic
  ):
    raise ValueError("Hybrid Stage1-5 live evaluation requires --checkpoint-file.")
  controller_gain_hash = _controller_hash(
    task,
    args.controller_gain_hash,
    require_loaded_match=args.scenario_file is None,
  )
  calibration_hash = _calibration_hash(
    task, required=args.scenario_file is None,
  )
  git_sha = _git_sha()
  if args.zero_residual_standing_diagnostic:
    _validate_rollout_args(args)
    result = _zero_residual_standing_result(
      task=task,
      args=args,
      git_sha=git_sha,
      controller_gain_hash=controller_gain_hash,
      calibration_hash=calibration_hash,
    )
    _write_or_print(result, args.output)
    return
  checkpoint_file_sha256: str | None = None
  if checkpoint is not None and args.scenario_file is None:
    checkpoint_payload = torch.load(
      checkpoint,
      map_location="cpu",
      weights_only=False,
    )
    if not isinstance(checkpoint_payload, Mapping):
      raise ValueError("Hybrid evaluation checkpoint must contain a mapping.")
    evaluation_env_cfg = load_env_cfg(task, play=True)
    validate_hybrid_evaluation_checkpoint(
      task,
      evaluation_env_cfg,
      checkpoint_payload,
      evaluation_git_sha=git_sha,
    )
    checkpoint_file_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

  retention_gate_audit: dict[str, object] | None = None
  if args.stage == 2 and args.scenario_file is None:
    if args.stage1_retention_file is None:
      raise ValueError(
        "Live Stage2 gate requires --stage1-retention-file for the exact "
        "Stage2 checkpoint."
      )
    retention_payload = _read_json(args.stage1_retention_file)
    if not isinstance(retention_payload, Mapping):
      raise ValueError("Stage1 retention file must contain a JSON object.")
    assert checkpoint_file_sha256 is not None
    retention_gate_audit = _validate_stage2_retention_gate(
      retention_payload,
      requested_profile=args.profile,
      seed=args.seed,
      git_sha=git_sha,
      checkpoint_file_sha256=checkpoint_file_sha256,
    )
    retention_gate_audit.update({
      "path": str(args.stage1_retention_file.resolve()),
      "file_sha256": hashlib.sha256(
        args.stage1_retention_file.read_bytes()
      ).hexdigest(),
    })

  scenario_rollout: dict[str, object] | None = None
  if args.scenario_file is not None:
    scenarios, scenario_profile, scenario_rollout = _scenario_file_payload(
      args.scenario_file
    )
    _validate_scenario_file_profile(
      args.profile,
      scenario_profile,
      scenario_rollout,
    )
  else:
    _validate_rollout_args(args)
    scenarios = _collect_scenarios(
      suite=suite,
      task=task,
      checkpoint=checkpoint,
      args=args,
    )
    if suite == "robust":
      if checkpoint is None:
        raise ValueError(
          "The robust suite requires a candidate checkpoint for the "
          "recovery comparison."
        )
      scenarios.append(
        {
          "name": "stage5_recovery_center_8x",
          "kind": "recovery",
          "kick_scale": STAGE5_RECOVERY_KICK_SCALE,
          "metrics": _collect_recovery_comparison(
            task=task,
            checkpoint=checkpoint,
            args=args,
          ),
        }
      )

  _validate_live_scenario_coverage(suite, scenarios)

  stage4_reference = None
  stage4_reference_audit: dict[str, object] | None = None
  if suite == "robust":
    if args.stage4_reference_file is not None:
      reference_payload = _read_json(args.stage4_reference_file)
      if not isinstance(reference_payload, Mapping):
        raise ValueError("Stage4 reference file must contain a JSON object.")
      stage4_reference = _extract_stage4_reference(reference_payload)
      stage4_reference_audit = {
        "source": "stage4_gate_file",
        "path": str(args.stage4_reference_file.resolve()),
        "file_sha256": hashlib.sha256(
          args.stage4_reference_file.read_bytes()
        ).hexdigest(),
        **stage4_reference,
      }
    elif args.scenario_file is None:
      baseline_metrics = _collect_stage4_reference_from_baseline(
        task=task,
        args=args,
      )
      stage4_reference = {
        "tracking_error": float(baseline_metrics["tracking_error"]),
      }
      stage4_reference_audit = {
        "source": "zero_residual_baseline_fixed_command",
        **baseline_metrics,
      }
    else:
      raise ValueError(
        "A robust scenario-file gate requires --stage4-reference-file."
      )

  checks = evaluate_capability_suite(
    suite,
    scenarios,
    profile=args.profile,
    stage4_reference=stage4_reference,
  )
  observations = (
    stage1_observational_improvements(scenarios)
    if suite == "residual"
    else None
  )
  result = make_result_envelope(
    suite=suite,
    task=task,
    git_sha=git_sha,
    controller_gain_hash=controller_gain_hash,
    calibration_hash=calibration_hash,
    seed=args.seed,
    checkpoint=None if checkpoint is None else str(checkpoint),
    checkpoint_file_sha256=checkpoint_file_sha256,
    scenarios=scenarios,
    checks=checks,
    stage1_profile_version=(
      STAGE1_MISMATCH_PROFILE_VERSION if suite == "residual" else None
    ),
    mismatch_profile=(stage1_mismatch_spec() if suite == "residual" else None),
    retention_gate=retention_gate_audit,
    evaluation_profile=args.profile,
    evaluation_source=(
      "scenario_file" if args.scenario_file is not None else "live"
    ),
    rollout=(
      scenario_rollout
      if scenario_rollout is not None
      else {
        "num_envs": args.num_envs,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "window_steps": args.window_steps,
        "episode_length_s": args.episode_length_s,
        "leg_residuals_ablated": bool(args.ablate_leg_residuals),
        "action_scales": list(
          load_env_cfg(task, play=True).actions[
            "hybrid_wheel_leg"
          ].action_scales
        ),
        **(
          {"stage4_reference": stage4_reference_audit}
          if stage4_reference_audit is not None
          else {}
        ),
      }
    ),
    observations=observations,
  )
  _write_or_print(result, args.output)
  if not result["gate_pass"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
