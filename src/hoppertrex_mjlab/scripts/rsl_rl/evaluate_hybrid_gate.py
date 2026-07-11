#!/usr/bin/env python3
"""Evaluate HopperTrex Hybrid v2 checkpoints with capability-driven gates."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys
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
    make_result_envelope,
    to_deterministic_json,
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
    make_result_envelope,
    to_deterministic_json,
  )
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


HYBRID_STAGE_SUITES = {
  0: "controller",
  1: "linear",
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--stage", type=int, choices=tuple(HYBRID_STAGE_SUITES))
  parser.add_argument("--task", default=None)
  parser.add_argument("--checkpoint-file", default=None)
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
    help="Stage4 gate envelope used for Stage5 tracking-degradation checks.",
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
  missing = sorted(required - names)
  posture_count = sum(
    scenario.get('kind') == 'posture' for scenario in scenarios
  )
  if suite in ('posture', 'integrated', 'robust') and posture_count < 5:
    missing.append(f'posture scenarios: expected 5, got {posture_count}')
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


@contextlib.contextmanager
def _policy_session(
  *,
  task: str,
  checkpoint: Path | None,
  args: argparse.Namespace,
  play: bool,
) -> Iterator[tuple[RslRlVecEnvWrapper, Any, Any]]:
  torch.manual_seed(args.seed)
  env_cfg = load_env_cfg(task, play=play)
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
  command[:, 0] = height
  command[:, 1] = pitch


def _pitch(robot_data: Any) -> torch.Tensor:
  gravity = robot_data.projected_gravity_b
  return torch.atan2(
    gravity[:, 0],
    torch.clamp(-gravity[:, 2], min=1.0e-6),
  )


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
  survival_rate = _survival_rate(ever_terminated.detach().cpu().tolist())
  return {
    "tracking_error": torch.sqrt(torch.mean(torch.sum(torch.square(error), dim=1))).item(),
    "terminated_event_rate": terminated_events / max(args.num_envs, 1),
    "survival_rate": survival_rate,
    "recovery_time_s": maximum_streak.max().item() / CONTROL_FREQUENCY_HZ,
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


def _collect_scenarios(
  *,
  suite: str,
  task: str,
  checkpoint: Path | None,
  args: argparse.Namespace,
) -> list[dict[str, object]]:
  play = suite != "robust"
  scenarios: list[dict[str, object]] = []
  with _policy_session(
    task=task,
    checkpoint=checkpoint,
    args=args,
    play=play,
  ) as (wrapped, policy, _env_cfg):
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

    if suite in ("integrated", "robust"):
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
  return scenarios


def _scenario_file_payload(path: Path) -> list[dict[str, object]]:
  payload = _read_json(path)
  if isinstance(payload, Mapping):
    payload = payload.get("scenarios")
  if not isinstance(payload, list) or not all(
    isinstance(scenario, dict) for scenario in payload
  ):
    raise ValueError("Scenario file must contain a scenario list.")
  return payload


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


def main() -> None:
  args = parse_args()
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
  if checkpoint is None and args.stage != 0 and args.scenario_file is None:
    raise ValueError("Hybrid Stage1-5 live evaluation requires --checkpoint-file.")
  controller_gain_hash = _controller_hash(
    task,
    args.controller_gain_hash,
    require_loaded_match=args.scenario_file is None,
  )
  calibration_hash = _calibration_hash(
    task, required=args.scenario_file is None,
  )

  if args.scenario_file is not None:
    scenarios = _scenario_file_payload(args.scenario_file)
  else:
    _validate_rollout_args(args)
    scenarios = _collect_scenarios(
      suite=suite,
      task=task,
      checkpoint=checkpoint,
      args=args,
    )

  if args.scenario_file is None:
    _validate_live_scenario_coverage(suite, scenarios)

  stage4_reference = None
  if suite == "robust":
    if args.stage4_reference_file is None:
      raise ValueError("The robust suite requires --stage4-reference-file.")
    reference_payload = _read_json(args.stage4_reference_file)
    if not isinstance(reference_payload, Mapping):
      raise ValueError("Stage4 reference file must contain a JSON object.")
    stage4_reference = _extract_stage4_reference(reference_payload)

  checks = evaluate_capability_suite(
    suite,
    scenarios,
    stage4_reference=stage4_reference,
  )
  result = make_result_envelope(
    suite=suite,
    task=task,
    git_sha=_git_sha(),
    controller_gain_hash=controller_gain_hash,
    calibration_hash=calibration_hash,
    seed=args.seed,
    checkpoint=None if checkpoint is None else str(checkpoint),
    scenarios=scenarios,
    checks=checks,
  )
  _write_or_print(result, args.output)
  if not result["gate_pass"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
