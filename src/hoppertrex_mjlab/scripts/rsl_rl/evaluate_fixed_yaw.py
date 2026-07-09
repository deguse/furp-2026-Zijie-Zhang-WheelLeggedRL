#!/usr/bin/env python3
"""Evaluate a policy under one sustained yaw command.

This catches viewer-visible Stage3 failures where the robot turns briefly, then
settles into standing balance, turns the wrong way, or drifts linearly while a
zero-linear-velocity yaw command remains enabled.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

PROJECT_PATH = Path(__file__).resolve().parents[2]
SRC_PATH = Path(__file__).resolve().parents[3]
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

WHEEL_TARGET_SATURATION = 23.9


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", required=True)
  parser.add_argument("--checkpoint-file", required=True)
  parser.add_argument("--yaw", type=float, nargs="+", default=[0.07])
  parser.add_argument("--lin-x", type=float, default=0.0)
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument("--warmup-steps", type=int, default=50)
  parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  parser.add_argument(
    "--play-cfg",
    action="store_true",
    help="Use the task play config before applying num-env overrides.",
  )
  parser.add_argument("--yaw-deadband", type=float, default=0.01)
  parser.add_argument("--lin-deadband", type=float, default=0.01)
  parser.add_argument("--lin-drift-speed", type=float, default=0.05)
  parser.add_argument(
    "--episode-length-s",
    type=float,
    default=None,
    help="Override env episode length. Use a very large value to match play/viewer.",
  )
  parser.add_argument(
    "--window-steps",
    type=int,
    default=50,
    help="Window size for per-env late-run turning/drift detection.",
  )
  parser.add_argument(
    "--progress-interval",
    type=int,
    default=1000,
    help="Print progress every N simulation steps. Set <=0 to disable.",
  )
  parser.add_argument(
    "--override-yaw-scale",
    type=float,
    default=None,
    help=(
      "Temporarily override wheel_balance yaw_scale for diagnostics. "
      "This does not modify the task registration or checkpoint."
    ),
  )
  return parser.parse_args(argv)


def _force_command(env: ManagerBasedRlEnv, lin_x: float, yaw: float) -> None:
  term = env.command_manager.get_term("twist")
  term.vel_command_b[:, :] = 0.0
  term.vel_command_w[:, :] = 0.0
  term.vel_command_b[:, 0] = lin_x
  term.vel_command_w[:, 0] = lin_x
  term.vel_command_b[:, 2] = yaw
  term.vel_command_w[:, 2] = yaw
  for attr in ("is_standing_env", "is_heading_env", "is_world_env", "is_forward_env"):
    value = getattr(term, attr, None)
    if value is not None:
      value[:] = False


def _safe_quantile(x: torch.Tensor, q: float) -> float:
  return torch.quantile(x, q).item() if x.numel() else float("nan")


def _safe_rms(x: torch.Tensor) -> float:
  return torch.sqrt(torch.mean(torch.square(x))).item() if x.numel() else float("nan")


def _safe_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
  selected = values[mask]
  return selected.mean().item() if selected.numel() else float("nan")


def _apply_yaw_scale_override(
  wrapped: RslRlVecEnvWrapper,
  override_yaw_scale: float | None,
) -> None:
  if override_yaw_scale is None:
    return
  if override_yaw_scale <= 0.0:
    raise ValueError(
      f"--override-yaw-scale must be positive, got {override_yaw_scale}."
    )

  wheel_action = wrapped.unwrapped.action_manager.get_term("wheel_balance")
  if not hasattr(wheel_action, "_yaw_scale"):
    raise AttributeError("wheel_balance action term does not expose _yaw_scale.")
  wheel_action._yaw_scale = float(override_yaw_scale)


def _yaw_tracking_health(
  *,
  yaw_by_step: torch.Tensor,
  lin_x_by_step: torch.Tensor,
  target_yaw: float,
  yaw_deadband: float,
  lin_drift_speed: float,
  window_steps: int,
  target_lin_x: float = 0.0,
  lin_deadband: float = 0.01,
) -> dict[str, float | torch.Tensor]:
  yaw = yaw_by_step.flatten()
  lin_x = lin_x_by_step.flatten()
  target_abs = abs(target_yaw)
  sign = 1.0 if target_yaw >= 0.0 else -1.0
  signed_yaw = sign * yaw

  window_steps = min(window_steps, yaw_by_step.shape[0])
  late_yaw = yaw_by_step[-window_steps:, :]
  late_lin_x = lin_x_by_step[-window_steps:, :]
  late_signed_yaw = sign * late_yaw
  late_mean_yaw = late_yaw.mean(dim=0)
  late_mean_signed_yaw = late_signed_yaw.mean(dim=0)
  late_lin_drift = late_lin_x.abs().mean(dim=0)
  lin_target_abs = abs(target_lin_x)
  lin_sign = 1.0 if target_lin_x >= 0.0 else -1.0
  signed_lin_x = lin_sign * lin_x
  late_signed_lin_x = lin_sign * late_lin_x
  late_mean_signed_lin_x = late_signed_lin_x.mean(dim=0)

  if target_abs > yaw_deadband:
    command_match = signed_yaw > yaw_deadband
    wrong_direction = signed_yaw < -yaw_deadband
    slow = signed_yaw < 0.5 * target_abs
    in_band = (signed_yaw >= 0.5 * target_abs) & (signed_yaw <= 1.5 * target_abs)
    fast = signed_yaw > 1.5 * target_abs
    late_slow_env = late_mean_signed_yaw < 0.5 * target_abs
    late_wrong_direction_env = late_mean_signed_yaw < -yaw_deadband
    late_wrong_direction_sample = late_signed_yaw < -yaw_deadband
    late_slow_sample = late_signed_yaw < 0.5 * target_abs
    late_in_band_sample = (
      (late_signed_yaw >= 0.5 * target_abs)
      & (late_signed_yaw <= 1.5 * target_abs)
    )
    late_fast_sample = late_signed_yaw > 1.5 * target_abs
  else:
    command_match = yaw.abs() <= yaw_deadband
    wrong_direction = yaw.abs() > yaw_deadband
    slow = wrong_direction
    in_band = command_match
    fast = wrong_direction
    late_slow_env = late_mean_signed_yaw.abs() > yaw_deadband
    late_wrong_direction_env = late_slow_env
    late_wrong_direction_sample = late_yaw.abs() > yaw_deadband
    late_slow_sample = late_wrong_direction_sample
    late_in_band_sample = late_yaw.abs() <= yaw_deadband
    late_fast_sample = late_wrong_direction_sample

  late_lin_drift_env = late_lin_drift > lin_drift_speed
  yaw_delta = yaw_by_step[1:, :] - yaw_by_step[:-1, :]
  late_yaw_delta = late_yaw[1:, :] - late_yaw[:-1, :]
  lin_x_delta = lin_x_by_step[1:, :] - lin_x_by_step[:-1, :]
  late_lin_x_delta = late_lin_x[1:, :] - late_lin_x[:-1, :]
  if lin_target_abs > lin_deadband:
    lin_command_match = signed_lin_x > lin_deadband
    lin_wrong_direction = signed_lin_x < -lin_deadband
    lin_slow = signed_lin_x < 0.5 * lin_target_abs
    lin_in_band = (
      (signed_lin_x >= 0.5 * lin_target_abs)
      & (signed_lin_x <= 1.5 * lin_target_abs)
    )
    lin_fast = signed_lin_x > 1.5 * lin_target_abs
    late_lin_in_band = (
      (late_signed_lin_x >= 0.5 * lin_target_abs)
      & (late_signed_lin_x <= 1.5 * lin_target_abs)
    )
    late_lin_in_band_env = (
      (late_mean_signed_lin_x >= 0.5 * lin_target_abs)
      & (late_mean_signed_lin_x <= 1.5 * lin_target_abs)
    )
  else:
    lin_command_match = lin_x.abs() <= lin_deadband
    lin_wrong_direction = lin_x.abs() > lin_deadband
    lin_slow = lin_wrong_direction
    lin_in_band = lin_command_match
    lin_fast = lin_wrong_direction
    late_lin_in_band = late_lin_x.abs() <= lin_deadband
    late_lin_in_band_env = late_lin_x.abs().mean(dim=0) <= lin_deadband

  return {
    "window_steps": float(window_steps),
    "command_match_frac": command_match.float().mean().item(),
    "wrong_direction_frac": wrong_direction.float().mean().item(),
    "slow_frac": slow.float().mean().item(),
    "slow_sample_frac": slow.float().mean().item(),
    "in_band_frac": in_band.float().mean().item(),
    "fast_frac": fast.float().mean().item(),
    "yaw_abs_error_mean": (yaw - target_yaw).abs().mean().item(),
    "yaw_abs_error_p90": _safe_quantile((yaw - target_yaw).abs(), 0.90),
    "lin_drift_abs_mean": lin_x.abs().mean().item(),
    "lin_drift_abs_p95": _safe_quantile(lin_x.abs(), 0.95),
    "late_mean_yaw": late_mean_yaw,
    "late_slow_env_frac": late_slow_env.float().mean().item(),
    "late_wrong_direction_env_frac": late_wrong_direction_env.float().mean().item(),
    "late_wrong_direction_sample_frac": late_wrong_direction_sample.float().mean().item(),
    "late_slow_sample_frac": late_slow_sample.float().mean().item(),
    "late_in_band_frac": late_in_band_sample.float().mean().item(),
    "late_fast_sample_frac": late_fast_sample.float().mean().item(),
    "late_lin_drift_env_frac": late_lin_drift_env.float().mean().item(),
    "yaw_delta_rms": _safe_rms(yaw_delta.flatten()),
    "yaw_delta_abs_p95": _safe_quantile(yaw_delta.abs().flatten(), 0.95),
    "late_yaw_delta_rms": _safe_rms(late_yaw_delta.flatten()),
    "late_yaw_delta_abs_p95": _safe_quantile(late_yaw_delta.abs().flatten(), 0.95),
    "lin_command_match_frac": lin_command_match.float().mean().item(),
    "lin_wrong_direction_frac": lin_wrong_direction.float().mean().item(),
    "lin_slow_frac": lin_slow.float().mean().item(),
    "lin_in_band_frac": lin_in_band.float().mean().item(),
    "lin_fast_frac": lin_fast.float().mean().item(),
    "late_lin_in_band_frac": late_lin_in_band.float().mean().item(),
    "late_lin_in_band_env_frac": late_lin_in_band_env.float().mean().item(),
    "lin_abs_error_mean": (lin_x - target_lin_x).abs().mean().item(),
    "lin_abs_error_p90": _safe_quantile((lin_x - target_lin_x).abs(), 0.90),
    "lin_x_delta_rms": _safe_rms(lin_x_delta.flatten()),
    "lin_x_delta_abs_p95": _safe_quantile(lin_x_delta.abs().flatten(), 0.95),
    "late_lin_x_delta_rms": _safe_rms(late_lin_x_delta.flatten()),
    "late_lin_x_delta_abs_p95": _safe_quantile(
      late_lin_x_delta.abs().flatten(),
      0.95,
    ),
    "slow_mask": slow,
    "in_band_mask": in_band,
    "fast_mask": fast,
  }


def _run_fixed_yaw(
  *,
  wrapped: RslRlVecEnvWrapper,
  policy,
  args: argparse.Namespace,
  yaw_cmd: float,
) -> dict[str, float | str]:
  wrapped.reset()
  robot = wrapped.unwrapped.scene["robot"]

  yaws: list[torch.Tensor] = []
  lin_xs: list[torch.Tensor] = []
  pitch_abses: list[torch.Tensor] = []
  pitch_rate_abses: list[torch.Tensor] = []
  wheel_target_abses: list[torch.Tensor] = []
  wheel_target_rates: list[torch.Tensor] = []
  action_abses: list[torch.Tensor] = []
  yaw_action_signs: list[torch.Tensor] = []
  raw_balance_actions: list[torch.Tensor] = []
  raw_yaw_actions: list[torch.Tensor] = []
  effective_yaw_actions: list[torch.Tensor] = []
  signed_raw_yaw_actions: list[torch.Tensor] = []
  signed_effective_yaw_actions: list[torch.Tensor] = []
  balance_components: list[torch.Tensor] = []
  yaw_components: list[torch.Tensor] = []
  signed_yaw_components: list[torch.Tensor] = []
  mapped_balance_components: list[torch.Tensor] = []
  mapped_yaw_components: list[torch.Tensor] = []
  signed_mapped_yaw_components: list[torch.Tensor] = []
  left_targets: list[torch.Tensor] = []
  right_targets: list[torch.Tensor] = []
  target_same_signs: list[torch.Tensor] = []
  target_opposite_signs: list[torch.Tensor] = []
  done_events = 0
  terminated_events = 0
  timeout_events = 0
  prev_wheel_target: torch.Tensor | None = None
  started_at = time.perf_counter()
  command_sign = 1.0 if yaw_cmd >= 0.0 else -1.0

  for step in range(args.steps):
    with torch.no_grad():
      _force_command(wrapped.unwrapped, args.lin_x, yaw_cmd)
      obs = wrapped.get_observations()
      actions = policy(obs).detach()
      _obs, _rew, done, _extras = wrapped.step(actions)
      _force_command(wrapped.unwrapped, args.lin_x, yaw_cmd)

      done_mask = done.to(dtype=torch.bool)
      done_events += int(done_mask.sum().item())
      terminated_events += int(wrapped.unwrapped.reset_terminated.sum().item())
      timeout_events += int(wrapped.unwrapped.reset_time_outs.sum().item())

      robot_data = robot.data
      wheel_action = wrapped.unwrapped.action_manager.get_term("wheel_balance")
      wheel_target = wheel_action._processed_actions.detach()
      wheel_raw_actions = getattr(wheel_action, "_raw_actions", None)
      if wheel_raw_actions is None:
        raw_balance = torch.zeros(actions.shape[0], device=actions.device)
        raw_yaw = torch.zeros_like(raw_balance)
      else:
        wheel_raw_actions = wheel_raw_actions.detach()
        raw_balance = wheel_raw_actions[:, 0]
        raw_yaw = (
          wheel_raw_actions[:, 1]
          if wheel_raw_actions.shape[1] > 1
          else torch.zeros_like(raw_balance)
        )
      balance_scale = float(getattr(wheel_action, "_balance_scale", 0.0))
      yaw_scale = float(getattr(wheel_action, "_yaw_scale", balance_scale))
      if raw_yaw.numel() and getattr(wheel_action, "_yaw_smoothing_alpha", None) is not None:
        effective_yaw_action = wheel_action._smoothed_yaw_action.detach()
      else:
        effective_yaw_action = raw_yaw
      balance_component = raw_balance * balance_scale
      yaw_component = effective_yaw_action * yaw_scale
      left_idx = int(getattr(wheel_action, "_left_idx", 0))
      right_idx = int(getattr(wheel_action, "_right_idx", 1))
      left_target = wheel_target[:, left_idx]
      right_target = wheel_target[:, right_idx]
      mapped_yaw_component = 0.5 * (left_target + right_target)
      mapped_balance_component = 0.5 * (right_target - left_target)
      if prev_wheel_target is None:
        delta_wheel_target = torch.zeros_like(wheel_target)
      else:
        delta_wheel_target = wheel_target - prev_wheel_target
      delta_wheel_target = torch.where(
        done_mask.unsqueeze(-1),
        torch.zeros_like(delta_wheel_target),
        delta_wheel_target,
      )
      prev_wheel_target = wheel_target.detach().clone()

      projected_gravity = robot_data.projected_gravity_b.detach()
      pitch = torch.atan2(
        projected_gravity[:, 0],
        torch.clamp(-projected_gravity[:, 2], min=1.0e-6),
      )
      if actions.shape[1] > 1:
        yaw_action = actions[:, 1]
      else:
        yaw_action = torch.zeros(actions.shape[0], device=actions.device)

      if step >= args.warmup_steps:
        yaws.append(robot_data.root_link_ang_vel_b[:, 2].detach().cpu())
        lin_xs.append(robot_data.root_link_lin_vel_b[:, 0].detach().cpu())
        pitch_abses.append(pitch.abs().detach().cpu())
        pitch_rate_abses.append(robot_data.root_link_ang_vel_b[:, 1].abs().detach().cpu())
        wheel_target_abses.append(torch.mean(torch.abs(wheel_target), dim=1).detach().cpu())
        wheel_target_rates.append(
          torch.mean(torch.abs(delta_wheel_target), dim=1).detach().cpu()
        )
        action_abses.append(torch.mean(torch.abs(actions), dim=1).detach().cpu())
        yaw_action_signs.append(torch.sign(yaw_action).detach().cpu())
        raw_balance_actions.append(raw_balance.detach().cpu())
        raw_yaw_actions.append(raw_yaw.detach().cpu())
        effective_yaw_actions.append(effective_yaw_action.detach().cpu())
        signed_raw_yaw_actions.append((command_sign * raw_yaw).detach().cpu())
        signed_effective_yaw_actions.append(
          (command_sign * effective_yaw_action).detach().cpu()
        )
        balance_components.append(balance_component.detach().cpu())
        yaw_components.append(yaw_component.detach().cpu())
        signed_yaw_components.append((command_sign * yaw_component).detach().cpu())
        mapped_balance_components.append(mapped_balance_component.detach().cpu())
        mapped_yaw_components.append(mapped_yaw_component.detach().cpu())
        signed_mapped_yaw_components.append(
          (command_sign * mapped_yaw_component).detach().cpu()
        )
        left_targets.append(left_target.detach().cpu())
        right_targets.append(right_target.detach().cpu())
        target_same_signs.append(
          (left_target * right_target > 0.0).float().detach().cpu()
        )
        target_opposite_signs.append(
          (left_target * right_target < 0.0).float().detach().cpu()
        )

      if args.progress_interval > 0 and (
        (step + 1) % args.progress_interval == 0 or step + 1 == args.steps
      ):
        elapsed_s = time.perf_counter() - started_at
        steps_per_s = (step + 1) / max(elapsed_s, 1.0e-6)
        print(
          f"PROGRESS yaw={yaw_cmd:+.5f} "
          f"step={step + 1}/{args.steps} "
          f"elapsed={elapsed_s:.1f}s "
          f"steps/s={steps_per_s:.1f} "
          f"terminated={terminated_events} "
          f"timeouts={timeout_events}",
          flush=True,
        )

  yaw = torch.cat(yaws)
  yaw_by_step = torch.stack(yaws)
  lin_x = torch.cat(lin_xs)
  lin_x_by_step = torch.stack(lin_xs)
  pitch_abs = torch.cat(pitch_abses)
  pitch_rate_abs = torch.cat(pitch_rate_abses)
  wheel_target_abs = torch.cat(wheel_target_abses)
  wheel_target_rate = torch.cat(wheel_target_rates)
  action_abs = torch.cat(action_abses)
  yaw_action_sign = torch.cat(yaw_action_signs)
  raw_balance_action = torch.cat(raw_balance_actions)
  raw_yaw_action = torch.cat(raw_yaw_actions)
  effective_yaw_action = torch.cat(effective_yaw_actions)
  signed_raw_yaw_action = torch.cat(signed_raw_yaw_actions)
  signed_effective_yaw_action = torch.cat(signed_effective_yaw_actions)
  balance_component = torch.cat(balance_components)
  yaw_component = torch.cat(yaw_components)
  signed_yaw_component = torch.cat(signed_yaw_components)
  mapped_balance_component = torch.cat(mapped_balance_components)
  mapped_yaw_component = torch.cat(mapped_yaw_components)
  signed_mapped_yaw_component = torch.cat(signed_mapped_yaw_components)
  left_target = torch.cat(left_targets)
  right_target = torch.cat(right_targets)
  target_same_sign = torch.cat(target_same_signs)
  target_opposite_sign = torch.cat(target_opposite_signs)
  target_error = yaw - yaw_cmd
  tracking = _yaw_tracking_health(
    yaw_by_step=yaw_by_step,
    lin_x_by_step=lin_x_by_step,
    target_yaw=yaw_cmd,
    target_lin_x=args.lin_x,
    yaw_deadband=args.yaw_deadband,
    lin_deadband=args.lin_deadband,
    lin_drift_speed=args.lin_drift_speed,
    window_steps=args.window_steps,
  )
  late_mean_yaw = tracking["late_mean_yaw"]
  assert isinstance(late_mean_yaw, torch.Tensor)
  slow_mask = tracking["slow_mask"]
  in_band_mask = tracking["in_band_mask"]
  fast_mask = tracking["fast_mask"]
  assert isinstance(slow_mask, torch.Tensor)
  assert isinstance(in_band_mask, torch.Tensor)
  assert isinstance(fast_mask, torch.Tensor)
  p95_pitch = _safe_quantile(pitch_abs, 0.95)
  p99_pitch_rate = _safe_quantile(pitch_rate_abs, 0.99)
  wheel_target_rate_rms = torch.sqrt(torch.mean(torch.square(wheel_target_rate))).item()
  wheel_saturation_ratio = (wheel_target_abs >= WHEEL_TARGET_SATURATION).float().mean().item()
  mapped_balance_abs_mean = mapped_balance_component.abs().mean().item()
  mapped_yaw_abs_mean = mapped_yaw_component.abs().mean().item()
  mapped_yaw_to_balance_abs_ratio = mapped_yaw_abs_mean / max(
    mapped_balance_abs_mean,
    1.0e-6,
  )
  mean_signed_mapped_yaw = signed_mapped_yaw_component.mean().item()
  actual_yaw_per_mapped_yaw = yaw.mul(command_sign).mean().item() / max(
    abs(mean_signed_mapped_yaw),
    1.0e-6,
  )
  mean_signed_raw_yaw = signed_raw_yaw_action.mean().item()
  actual_yaw_per_raw_yaw = yaw.mul(command_sign).mean().item() / max(
    abs(mean_signed_raw_yaw),
    1.0e-6,
  )
  wheel_action = wrapped.unwrapped.action_manager.get_term("wheel_balance")
  balance_scale = float(getattr(wheel_action, "_balance_scale", 0.0))
  yaw_scale = float(getattr(wheel_action, "_yaw_scale", balance_scale))
  yaw_smoothing_alpha = getattr(wheel_action, "_yaw_smoothing_alpha", None)
  target_slew_limit = getattr(wheel_action, "_target_slew_limit", None)

  print(f"Task: {args.task}")
  print(f"Play cfg: {args.play_cfg}")
  print(f"Episode length s: {wrapped.unwrapped.cfg.episode_length_s:.5g}")
  print(f"Fixed command: lin_x={args.lin_x:+.5f}, yaw={yaw_cmd:+.5f}")
  print(f"Wheel balance scale:   {balance_scale:.5f}")
  print(f"Wheel yaw scale:       {yaw_scale:.5f}")
  print(f"Yaw smoothing alpha:   {yaw_smoothing_alpha}")
  print(f"Target slew limit:     {target_slew_limit}")
  print(f"Samples after warmup: {yaw.numel()}")
  print(f"done_event_rate:       {done_events / max(args.num_envs, 1):.5f}")
  print(f"terminated_event_rate: {terminated_events / max(args.num_envs, 1):.5f}")
  print(f"timeout_event_rate:    {timeout_events / max(args.num_envs, 1):.5f}")
  print(f"mean actual_yaw:       {yaw.mean().item():+.5f}")
  print(f"p05 actual_yaw:        {_safe_quantile(yaw, 0.05):+.5f}")
  print(f"p50 actual_yaw:        {_safe_quantile(yaw, 0.50):+.5f}")
  print(f"p95 actual_yaw:        {_safe_quantile(yaw, 0.95):+.5f}")
  print(f"mean actual_lin_x:     {lin_x.mean().item():+.5f}")
  print(f"lin_command_match_frac:{tracking['lin_command_match_frac']:.5f}")
  print(f"lin_in_band_frac:      {tracking['lin_in_band_frac']:.5f}")
  print(f"lin_fast_frac:         {tracking['lin_fast_frac']:.5f}")
  print(f"late_lin_in_band_frac: {tracking['late_lin_in_band_frac']:.5f}")
  print(f"command_match_frac:    {tracking['command_match_frac']:.5f}")
  print(f"wrong_direction_frac:  {tracking['wrong_direction_frac']:.5f}")
  print(f"slow_frac:             {tracking['slow_frac']:.5f}")
  print(f"in_band_frac:          {tracking['in_band_frac']:.5f}")
  print(f"fast_frac:             {tracking['fast_frac']:.5f}")
  print(f"late window steps:     {int(tracking['window_steps'])}")
  print(f"late mean yaw p05:     {_safe_quantile(late_mean_yaw, 0.05):+.5f}")
  print(f"late mean yaw p50:     {_safe_quantile(late_mean_yaw, 0.50):+.5f}")
  print(f"late mean yaw p95:     {_safe_quantile(late_mean_yaw, 0.95):+.5f}")
  print(f"late_slow_env_frac:    {tracking['late_slow_env_frac']:.5f}")
  print(f"late_slow_sample_frac: {tracking['late_slow_sample_frac']:.5f}")
  print(f"late_in_band_frac:     {tracking['late_in_band_frac']:.5f}")
  print(f"late_fast_sample_frac: {tracking['late_fast_sample_frac']:.5f}")
  print(f"late_wrong_direction_env_frac: {tracking['late_wrong_direction_env_frac']:.5f}")
  print(f"late_wrong_direction_sample_frac: {tracking['late_wrong_direction_sample_frac']:.5f}")
  print(f"late_lin_drift_env_frac: {tracking['late_lin_drift_env_frac']:.5f}")
  print(f"mean |yaw error|:      {target_error.abs().mean().item():.5f}")
  print(f"p90 |yaw error|:       {_safe_quantile(target_error.abs(), 0.90):.5f}")
  print(f"mean |lin_x error|:    {tracking['lin_abs_error_mean']:.5f}")
  print(f"p90 |lin_x error|:     {tracking['lin_abs_error_p90']:.5f}")
  print(f"mean |lin_x drift|:    {lin_x.abs().mean().item():.5f}")
  print(f"p95 |lin_x drift|:     {_safe_quantile(lin_x.abs(), 0.95):.5f}")
  print(f"p95 |pitch|:           {p95_pitch:.5f}")
  print(f"p99 |pitch_rate|:      {p99_pitch_rate:.5f}")
  print(f"mean |wheel_target|:   {wheel_target_abs.mean().item():.5f}")
  print(f"wheel_saturation_ratio:{wheel_saturation_ratio:.5f}")
  print(f"wheel_target_rate_rms: {wheel_target_rate_rms:.5f}")
  print(f"yaw_delta_rms:         {tracking['yaw_delta_rms']:.5f}")
  print(f"yaw_delta_abs_p95:     {tracking['yaw_delta_abs_p95']:.5f}")
  print(f"late_yaw_delta_rms:    {tracking['late_yaw_delta_rms']:.5f}")
  print(f"late_yaw_delta_p95:    {tracking['late_yaw_delta_abs_p95']:.5f}")
  print(f"lin_x_delta_rms:       {tracking['lin_x_delta_rms']:.5f}")
  print(f"lin_x_delta_abs_p95:   {tracking['lin_x_delta_abs_p95']:.5f}")
  print(f"late_lin_x_delta_rms:  {tracking['late_lin_x_delta_rms']:.5f}")
  print(f"late_lin_x_delta_p95:  {tracking['late_lin_x_delta_abs_p95']:.5f}")
  print(f"mean |action|:         {action_abs.mean().item():.5f}")
  print(f"positive_yaw_action_frac: {(yaw_action_sign > 0.0).float().mean().item():.5f}")
  print(f"negative_yaw_action_frac: {(yaw_action_sign < 0.0).float().mean().item():.5f}")
  print(f"mean raw_balance:      {raw_balance_action.mean().item():+.5f}")
  print(f"mean |raw_balance|:    {raw_balance_action.abs().mean().item():.5f}")
  print(f"p95 |raw_balance|:     {_safe_quantile(raw_balance_action.abs(), 0.95):.5f}")
  print(f"mean raw_yaw:          {raw_yaw_action.mean().item():+.5f}")
  print(f"mean |raw_yaw|:        {raw_yaw_action.abs().mean().item():.5f}")
  print(f"p05 raw_yaw:           {_safe_quantile(raw_yaw_action, 0.05):+.5f}")
  print(f"p50 raw_yaw:           {_safe_quantile(raw_yaw_action, 0.50):+.5f}")
  print(f"p95 raw_yaw:           {_safe_quantile(raw_yaw_action, 0.95):+.5f}")
  print(f"mean signed raw_yaw:   {mean_signed_raw_yaw:+.5f}")
  print(f"mean effective_yaw:    {effective_yaw_action.mean().item():+.5f}")
  print(f"mean |effective_yaw|:  {effective_yaw_action.abs().mean().item():.5f}")
  print(f"mean signed eff_yaw:   {signed_effective_yaw_action.mean().item():+.5f}")
  print(f"mean |balance_comp|:   {balance_component.abs().mean().item():.5f}")
  print(f"mean yaw_comp:         {yaw_component.mean().item():+.5f}")
  print(f"mean |yaw_comp|:       {yaw_component.abs().mean().item():.5f}")
  print(f"mean signed yaw_comp:  {signed_yaw_component.mean().item():+.5f}")
  print(f"mean mapped_yaw_comp:  {mapped_yaw_component.mean().item():+.5f}")
  print(f"mean |mapped_yaw|:     {mapped_yaw_abs_mean:.5f}")
  print(f"mean signed mapped_yaw:{mean_signed_mapped_yaw:+.5f}")
  print(f"mean |mapped_balance|: {mapped_balance_abs_mean:.5f}")
  print(f"mapped_yaw/bal ratio:  {mapped_yaw_to_balance_abs_ratio:.5f}")
  print(f"mean left_target:      {left_target.mean().item():+.5f}")
  print(f"mean right_target:     {right_target.mean().item():+.5f}")
  print(f"target same_sign_frac: {target_same_sign.mean().item():.5f}")
  print(f"target opposite_frac:  {target_opposite_sign.mean().item():.5f}")
  print(f"actual_yaw/mapped_yaw: {actual_yaw_per_mapped_yaw:+.5f}")
  print(f"actual_yaw/raw_yaw:    {actual_yaw_per_raw_yaw:+.5f}")
  print(
    "slow/in_band/fast signed raw_yaw: "
    f"{_safe_masked_mean(signed_raw_yaw_action, slow_mask):+.5f} "
    f"{_safe_masked_mean(signed_raw_yaw_action, in_band_mask):+.5f} "
    f"{_safe_masked_mean(signed_raw_yaw_action, fast_mask):+.5f}"
  )
  print(
    "slow/in_band/fast signed eff_yaw: "
    f"{_safe_masked_mean(signed_effective_yaw_action, slow_mask):+.5f} "
    f"{_safe_masked_mean(signed_effective_yaw_action, in_band_mask):+.5f} "
    f"{_safe_masked_mean(signed_effective_yaw_action, fast_mask):+.5f}"
  )
  print(
    "slow/in_band/fast signed mapped_yaw: "
    f"{_safe_masked_mean(signed_mapped_yaw_component, slow_mask):+.5f} "
    f"{_safe_masked_mean(signed_mapped_yaw_component, in_band_mask):+.5f} "
    f"{_safe_masked_mean(signed_mapped_yaw_component, fast_mask):+.5f}"
  )
  print(
    "slow/in_band/fast |mapped_balance|: "
    f"{_safe_masked_mean(mapped_balance_component.abs(), slow_mask):.5f} "
    f"{_safe_masked_mean(mapped_balance_component.abs(), in_band_mask):.5f} "
    f"{_safe_masked_mean(mapped_balance_component.abs(), fast_mask):.5f}"
  )
  print(
    "slow/in_band/fast wheel_rate: "
    f"{_safe_masked_mean(wheel_target_rate, slow_mask):.5f} "
    f"{_safe_masked_mean(wheel_target_rate, in_band_mask):.5f} "
    f"{_safe_masked_mean(wheel_target_rate, fast_mask):.5f}"
  )

  return {
    "lin_x": args.lin_x,
    "yaw": yaw_cmd,
    "mean_actual_yaw": yaw.mean().item(),
    "command_match_frac": float(tracking["command_match_frac"]),
    "wrong_direction_frac": float(tracking["wrong_direction_frac"]),
    "slow_frac": float(tracking["slow_frac"]),
    "slow_sample_frac": float(tracking["slow_sample_frac"]),
    "in_band_frac": float(tracking["in_band_frac"]),
    "fast_frac": float(tracking["fast_frac"]),
    "late_slow_env_frac": float(tracking["late_slow_env_frac"]),
    "late_slow_sample_frac": float(tracking["late_slow_sample_frac"]),
    "late_in_band_frac": float(tracking["late_in_band_frac"]),
    "late_fast_sample_frac": float(tracking["late_fast_sample_frac"]),
    "late_wrong_direction_env_frac": float(
      tracking["late_wrong_direction_env_frac"]
    ),
    "late_wrong_direction_sample_frac": float(
      tracking["late_wrong_direction_sample_frac"]
    ),
    "late_lin_drift_env_frac": float(tracking["late_lin_drift_env_frac"]),
    "lin_command_match_frac": float(tracking["lin_command_match_frac"]),
    "lin_wrong_direction_frac": float(tracking["lin_wrong_direction_frac"]),
    "lin_slow_frac": float(tracking["lin_slow_frac"]),
    "lin_in_band_frac": float(tracking["lin_in_band_frac"]),
    "lin_fast_frac": float(tracking["lin_fast_frac"]),
    "late_lin_in_band_frac": float(tracking["late_lin_in_band_frac"]),
    "late_lin_in_band_env_frac": float(tracking["late_lin_in_band_env_frac"]),
    "lin_abs_error_mean": float(tracking["lin_abs_error_mean"]),
    "lin_abs_error_p90": float(tracking["lin_abs_error_p90"]),
    "lin_x_delta_rms": float(tracking["lin_x_delta_rms"]),
    "lin_x_delta_abs_p95": float(tracking["lin_x_delta_abs_p95"]),
    "late_lin_x_delta_rms": float(tracking["late_lin_x_delta_rms"]),
    "late_lin_x_delta_abs_p95": float(tracking["late_lin_x_delta_abs_p95"]),
    "yaw_delta_rms": float(tracking["yaw_delta_rms"]),
    "yaw_delta_abs_p95": float(tracking["yaw_delta_abs_p95"]),
    "late_yaw_delta_rms": float(tracking["late_yaw_delta_rms"]),
    "late_yaw_delta_abs_p95": float(tracking["late_yaw_delta_abs_p95"]),
    "yaw_abs_error_mean": float(tracking["yaw_abs_error_mean"]),
    "yaw_abs_error_p90": float(tracking["yaw_abs_error_p90"]),
    "lin_drift_abs_mean": float(tracking["lin_drift_abs_mean"]),
    "lin_drift_abs_p95": float(tracking["lin_drift_abs_p95"]),
    "p95_pitch": p95_pitch,
    "p99_pitch_rate": p99_pitch_rate,
    "wheel_saturation_ratio": wheel_saturation_ratio,
    "wheel_target_rate_rms": wheel_target_rate_rms,
    "signed_raw_yaw_mean": mean_signed_raw_yaw,
    "signed_mapped_yaw_mean": mean_signed_mapped_yaw,
    "mapped_yaw_abs_mean": mapped_yaw_abs_mean,
    "mapped_balance_abs_mean": mapped_balance_abs_mean,
    "slow_signed_raw_yaw_mean": _safe_masked_mean(signed_raw_yaw_action, slow_mask),
    "in_band_signed_raw_yaw_mean": _safe_masked_mean(
      signed_raw_yaw_action,
      in_band_mask,
    ),
    "fast_signed_raw_yaw_mean": _safe_masked_mean(signed_raw_yaw_action, fast_mask),
    "slow_signed_effective_yaw_mean": _safe_masked_mean(
      signed_effective_yaw_action,
      slow_mask,
    ),
    "in_band_signed_effective_yaw_mean": _safe_masked_mean(
      signed_effective_yaw_action,
      in_band_mask,
    ),
    "fast_signed_effective_yaw_mean": _safe_masked_mean(
      signed_effective_yaw_action,
      fast_mask,
    ),
    "slow_signed_mapped_yaw_mean": _safe_masked_mean(
      signed_mapped_yaw_component,
      slow_mask,
    ),
    "in_band_signed_mapped_yaw_mean": _safe_masked_mean(
      signed_mapped_yaw_component,
      in_band_mask,
    ),
    "fast_signed_mapped_yaw_mean": _safe_masked_mean(
      signed_mapped_yaw_component,
      fast_mask,
    ),
    "slow_mapped_balance_abs_mean": _safe_masked_mean(
      mapped_balance_component.abs(),
      slow_mask,
    ),
    "in_band_mapped_balance_abs_mean": _safe_masked_mean(
      mapped_balance_component.abs(),
      in_band_mask,
    ),
    "fast_mapped_balance_abs_mean": _safe_masked_mean(
      mapped_balance_component.abs(),
      fast_mask,
    ),
    "slow_wheel_target_rate_mean": _safe_masked_mean(wheel_target_rate, slow_mask),
    "in_band_wheel_target_rate_mean": _safe_masked_mean(
      wheel_target_rate,
      in_band_mask,
    ),
    "fast_wheel_target_rate_mean": _safe_masked_mean(wheel_target_rate, fast_mask),
    "target_same_sign_frac": target_same_sign.mean().item(),
    "terminated_event_rate": terminated_events / max(args.num_envs, 1),
  }


def main() -> None:
  args = parse_args()
  configure_torch_backends()

  checkpoint = Path(args.checkpoint_file)
  if not checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

  env_cfg = load_env_cfg(args.task, play=args.play_cfg)
  agent_cfg = load_rl_cfg(args.task)
  if args.episode_length_s is not None:
    env_cfg.episode_length_s = args.episode_length_s
  env_cfg.scene.num_envs = args.num_envs
  if env_cfg.scene.terrain is not None:
    env_cfg.scene.terrain.num_envs = args.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  _apply_yaw_scale_override(wrapped, args.override_yaw_scale)
  summaries: list[dict[str, float | str]] = []
  try:
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
    runner.load(
      str(checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)

    print(f"Checkpoint: {checkpoint}")
    for index, yaw_cmd in enumerate(args.yaw):
      if index:
        print("")
      summaries.append(
        _run_fixed_yaw(
          wrapped=wrapped,
          policy=policy,
          args=args,
          yaw_cmd=yaw_cmd,
        )
      )
  finally:
    wrapped.close()

  if len(summaries) > 1:
    print("")
    print(
      "SUMMARY yaw mean match wrong slow in_band fast "
      "late_slow_env late_in_band late_wrong_env late_wrong_sample "
      "late_lin_drift_env "
      "yaw_abs_err yaw_p90_abs_err lin_drift lin_p95_drift "
      "p95_pitch p99_pitch_rate wheel_sat wheel_rate yaw_delta yaw_delta_p95 "
      "signed_raw_yaw signed_mapped_yaw mapped_yaw mapped_balance "
      "same_target term"
    )
    for row in summaries:
      print(
        f"SUMMARY {row['yaw']:+.3f} {row['mean_actual_yaw']:+.4f} "
        f"{row['command_match_frac']:.3f} {row['wrong_direction_frac']:.3f} "
        f"{row['slow_frac']:.3f} {row['in_band_frac']:.3f} "
        f"{row['fast_frac']:.3f} {row['late_slow_env_frac']:.3f} "
        f"{row['late_in_band_frac']:.3f} "
        f"{row['late_wrong_direction_env_frac']:.3f} "
        f"{row['late_wrong_direction_sample_frac']:.3f} "
        f"{row['late_lin_drift_env_frac']:.3f} "
        f"{row['yaw_abs_error_mean']:.4f} {row['yaw_abs_error_p90']:.4f} "
        f"{row['lin_drift_abs_mean']:.4f} {row['lin_drift_abs_p95']:.4f} "
        f"{row['p95_pitch']:.4f} {row['p99_pitch_rate']:.4f} "
        f"{row['wheel_saturation_ratio']:.4f} "
        f"{row['wheel_target_rate_rms']:.4f} "
        f"{row['yaw_delta_rms']:.4f} {row['yaw_delta_abs_p95']:.4f} "
        f"{row['signed_raw_yaw_mean']:.4f} "
        f"{row['signed_mapped_yaw_mean']:.4f} "
        f"{row['mapped_yaw_abs_mean']:.4f} "
        f"{row['mapped_balance_abs_mean']:.4f} "
        f"{row['target_same_sign_frac']:.3f} "
        f"{row['terminated_event_rate']:.3f}"
      )


if __name__ == "__main__":
  main()
