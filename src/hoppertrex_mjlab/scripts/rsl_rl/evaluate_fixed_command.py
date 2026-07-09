#!/usr/bin/env python3
"""Evaluate a policy under one sustained velocity command.

This complements stage gates, which sample randomized commands. It is meant to
catch viewer-visible attractors such as "starts moving, then settles into
standing balance while the forward command remains enabled".
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
from assets.HopperTrex_CFG import WHEEL_JOINT_NAMES
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

REWARD_DEBUG_TERMS = (
  "track_linear_velocity",
  "lin_vel_x_sign_alignment",
  "lin_velocity_band_l2",
  "lin_velocity_delta_l2",
  "wheel_target_rate_l2",
  "action_rate_l2",
  "action_acc_l2",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", required=True)
  parser.add_argument("--checkpoint-file", required=True)
  parser.add_argument("--lin-x", type=float, nargs="+", default=[0.07])
  parser.add_argument("--yaw", type=float, default=0.0)
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument("--warmup-steps", type=int, default=50)
  parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  parser.add_argument(
    "--play-cfg",
    action="store_true",
    help="Use the task play config before applying num-env overrides.",
  )
  parser.add_argument("--stuck-speed", type=float, default=0.01)
  parser.add_argument("--reverse-speed", type=float, default=-0.01)
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
    help="Window size for per-env late-run stuck/slow detection.",
  )
  parser.add_argument(
    "--progress-interval",
    type=int,
    default=1000,
    help="Print progress every N simulation steps. Set <=0 to disable.",
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
  # Keep fixed-command diagnostics independent of standing/world/heading sampling.
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


def _safe_masked_rms(values: torch.Tensor, mask: torch.Tensor) -> float:
  selected = values[mask]
  return _safe_rms(selected) if selected.numel() else float("nan")


def _safe_masked_quantile(values: torch.Tensor, mask: torch.Tensor, q: float) -> float:
  selected = values[mask]
  return _safe_quantile(selected, q) if selected.numel() else float("nan")


def _summarize_reward_contributions(
  *,
  weighted_rewards: dict[str, list[torch.Tensor]],
  reward_weights: dict[str, float],
  term_names: tuple[str, ...] = REWARD_DEBUG_TERMS,
  main_term: str = "track_linear_velocity",
) -> dict[str, float]:
  summary: dict[str, float] = {}
  weighted_means: dict[str, float] = {}

  for name in term_names:
    samples = weighted_rewards.get(name, [])
    if samples:
      weighted_mean = torch.cat(samples).mean().item()
    else:
      weighted_mean = float("nan")
    weighted_means[name] = weighted_mean

    weight = reward_weights.get(name)
    if weight is None or weight == 0.0 or weighted_mean != weighted_mean:
      raw_mean = float("nan")
    else:
      raw_mean = weighted_mean / weight
    summary[f"{name}_raw_mean"] = raw_mean
    summary[f"{name}_weighted_mean"] = weighted_mean

  main_weighted = weighted_means.get(main_term, float("nan"))
  denominator = abs(main_weighted)
  for name in term_names:
    weighted_mean = weighted_means.get(name, float("nan"))
    if (
      weighted_mean != weighted_mean
      or main_weighted != main_weighted
      or denominator <= 1.0e-12
    ):
      relative = float("nan")
    else:
      relative = weighted_mean / denominator
    summary[f"{name}_relative_to_main"] = relative

  return summary


def _bucketed_dynamic_stats(
  *,
  slow_mask: torch.Tensor,
  in_band_mask: torch.Tensor,
  fast_mask: torch.Tensor,
  lin_x_delta: torch.Tensor,
  lin_x_delta_slow_mask: torch.Tensor,
  lin_x_delta_in_band_mask: torch.Tensor,
  lin_x_delta_fast_mask: torch.Tensor,
  wheel_target_rate: torch.Tensor,
  action_delta: torch.Tensor,
  pitch_rate_abs: torch.Tensor,
) -> dict[str, float]:
  stats: dict[str, float] = {}
  sample_masks = {
    "slow": slow_mask,
    "in_band": in_band_mask,
    "fast": fast_mask,
  }
  delta_masks = {
    "slow": lin_x_delta_slow_mask,
    "in_band": lin_x_delta_in_band_mask,
    "fast": lin_x_delta_fast_mask,
  }

  for bucket, mask in sample_masks.items():
    stats[f"{bucket}_wheel_target_rate_rms"] = _safe_masked_rms(
      wheel_target_rate,
      mask,
    )
    stats[f"{bucket}_action_delta_rms"] = _safe_masked_rms(action_delta, mask)
    stats[f"{bucket}_pitch_rate_abs_p99"] = _safe_masked_quantile(
      pitch_rate_abs,
      mask,
      0.99,
    )

  for bucket, mask in delta_masks.items():
    stats[f"{bucket}_lin_x_delta_rms"] = _safe_masked_rms(lin_x_delta, mask)
    stats[f"{bucket}_lin_x_delta_abs_p95"] = _safe_masked_quantile(
      lin_x_delta.abs(),
      mask,
      0.95,
    )

  return stats


def _late_command_health(
  *,
  late_lin_x: torch.Tensor,
  target_lin_x: float,
  stuck_speed: float,
) -> dict[str, torch.Tensor]:
  late_mean_lin_x = late_lin_x.mean(dim=0)
  late_min_lin_x = late_lin_x.min(dim=0).values
  late_max_lin_x = late_lin_x.max(dim=0).values
  late_stuck_env = late_mean_lin_x.abs() <= stuck_speed
  if target_lin_x > stuck_speed:
    late_slow_env = late_mean_lin_x < 0.5 * target_lin_x
    wrong_direction_env = late_min_lin_x < -stuck_speed
  elif target_lin_x < -stuck_speed:
    late_slow_env = late_mean_lin_x > 0.5 * target_lin_x
    wrong_direction_env = late_max_lin_x > stuck_speed
  else:
    late_slow_env = torch.zeros_like(late_stuck_env, dtype=torch.bool)
    wrong_direction_env = late_mean_lin_x.abs() > stuck_speed
  return {
    "mean_lin_x": late_mean_lin_x,
    "stuck_env": late_stuck_env,
    "slow_env": late_slow_env,
    "wrong_direction_env": wrong_direction_env,
  }


def _command_tracking_health(
  *,
  lin_x_by_step: torch.Tensor,
  target_lin_x: float,
  stuck_speed: float,
  window_steps: int,
) -> dict[str, float | torch.Tensor]:
  lin_x = lin_x_by_step.flatten()
  window_steps = min(window_steps, lin_x_by_step.shape[0])
  late_lin_x = lin_x_by_step[-window_steps:, :]
  late_health = _late_command_health(
    late_lin_x=late_lin_x,
    target_lin_x=target_lin_x,
    stuck_speed=stuck_speed,
  )

  if target_lin_x > stuck_speed:
    signed_lin_x = lin_x
    late_signed_lin_x = late_lin_x
  elif target_lin_x < -stuck_speed:
    signed_lin_x = -lin_x
    late_signed_lin_x = -late_lin_x
  else:
    signed_lin_x = -lin_x.abs()
    late_signed_lin_x = -late_lin_x.abs()

  target_abs = abs(target_lin_x)
  if target_abs > stuck_speed:
    command_match = signed_lin_x > stuck_speed
    wrong_direction = signed_lin_x < -stuck_speed
    slow = signed_lin_x < 0.5 * target_abs
    in_band = (signed_lin_x >= 0.5 * target_abs) & (signed_lin_x <= 1.5 * target_abs)
    fast = signed_lin_x > 1.5 * target_abs
    late_wrong_direction_sample = late_signed_lin_x < -stuck_speed
    late_slow_sample = late_signed_lin_x < 0.5 * target_abs
    late_in_band_sample = (
      (late_signed_lin_x >= 0.5 * target_abs)
      & (late_signed_lin_x <= 1.5 * target_abs)
    )
    late_fast_sample = late_signed_lin_x > 1.5 * target_abs
  else:
    command_match = lin_x.abs() <= stuck_speed
    wrong_direction = lin_x.abs() > stuck_speed
    slow = wrong_direction
    in_band = command_match
    fast = wrong_direction
    late_wrong_direction_sample = late_lin_x.abs() > stuck_speed
    late_slow_sample = late_wrong_direction_sample
    late_in_band_sample = late_lin_x.abs() <= stuck_speed
    late_fast_sample = late_wrong_direction_sample

  lin_x_delta = lin_x_by_step[1:, :] - lin_x_by_step[:-1, :]
  late_lin_x_delta = late_lin_x[1:, :] - late_lin_x[:-1, :]
  slow_by_step = slow.reshape(lin_x_by_step.shape)
  in_band_by_step = in_band.reshape(lin_x_by_step.shape)
  fast_by_step = fast.reshape(lin_x_by_step.shape)

  return {
    "window_steps": float(window_steps),
    "signed_lin_x_mean": signed_lin_x.mean().item(),
    "signed_lin_x_p05": _safe_quantile(signed_lin_x, 0.05),
    "signed_lin_x_p50": _safe_quantile(signed_lin_x, 0.50),
    "signed_lin_x_p95": _safe_quantile(signed_lin_x, 0.95),
    "command_match_frac": command_match.float().mean().item(),
    "wrong_direction_frac": wrong_direction.float().mean().item(),
    "slow_frac": slow.float().mean().item(),
    "slow_sample_frac": slow.float().mean().item(),
    "in_band_frac": in_band.float().mean().item(),
    "fast_frac": fast.float().mean().item(),
    "late_mean_lin_x": late_health["mean_lin_x"],
    "late_stuck_env_frac": late_health["stuck_env"].float().mean().item(),
    "late_slow_env_frac": late_health["slow_env"].float().mean().item(),
    "late_wrong_direction_env_frac": late_health["wrong_direction_env"].float().mean().item(),
    "late_wrong_direction_sample_frac": late_wrong_direction_sample.float().mean().item(),
    "late_slow_sample_frac": late_slow_sample.float().mean().item(),
    "late_in_band_frac": late_in_band_sample.float().mean().item(),
    "late_fast_sample_frac": late_fast_sample.float().mean().item(),
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
    "lin_x_delta": lin_x_delta.flatten(),
    "lin_x_delta_slow_mask": slow_by_step[1:, :].flatten(),
    "lin_x_delta_in_band_mask": in_band_by_step[1:, :].flatten(),
    "lin_x_delta_fast_mask": fast_by_step[1:, :].flatten(),
  }


def _run_fixed_command(
  *,
  wrapped: RslRlVecEnvWrapper,
  policy,
  args: argparse.Namespace,
  lin_x_cmd: float,
) -> dict[str, float | str]:
  wrapped.reset()
  robot = wrapped.unwrapped.scene["robot"]

  lin_xs: list[torch.Tensor] = []
  pitch_abses: list[torch.Tensor] = []
  pitch_rate_abses: list[torch.Tensor] = []
  wheel_target_abses: list[torch.Tensor] = []
  wheel_target_rates: list[torch.Tensor] = []
  action_abses: list[torch.Tensor] = []
  action_deltas: list[torch.Tensor] = []
  action_signs: list[torch.Tensor] = []
  done_events = 0
  terminated_events = 0
  timeout_events = 0
  prev_wheel_target: torch.Tensor | None = None
  prev_actions: torch.Tensor | None = None
  started_at = time.perf_counter()
  reward_manager = getattr(wrapped.unwrapped, "reward_manager", None)
  reward_term_names = list(getattr(reward_manager, "active_terms", []))
  reward_term_indices = {
    name: reward_term_names.index(name)
    for name in REWARD_DEBUG_TERMS
    if name in reward_term_names
  }
  reward_weights = {
    name: reward_manager.get_term_cfg(name).weight
    for name in reward_term_indices
  } if reward_manager is not None else {}
  weighted_rewards = {name: [] for name in REWARD_DEBUG_TERMS}

  wheel_joint_ids = torch.tensor(
    [list(robot.joint_names).index(name) for name in WHEEL_JOINT_NAMES],
    device=args.device,
    dtype=torch.long,
  )
  del wheel_joint_ids

  for step in range(args.steps):
    with torch.no_grad():
      _force_command(wrapped.unwrapped, lin_x_cmd, args.yaw)
      obs = wrapped.get_observations()
      actions = policy(obs).detach()
      _obs, _rew, done, _extras = wrapped.step(actions)
      _force_command(wrapped.unwrapped, lin_x_cmd, args.yaw)

      done_mask = done.to(dtype=torch.bool)
      done_events += int(done_mask.sum().item())
      terminated_events += int(wrapped.unwrapped.reset_terminated.sum().item())
      timeout_events += int(wrapped.unwrapped.reset_time_outs.sum().item())

      robot_data = robot.data
      if prev_actions is None:
        delta_actions = torch.zeros_like(actions)
      else:
        delta_actions = actions - prev_actions
      delta_actions = torch.where(
        done_mask.unsqueeze(-1),
        torch.zeros_like(delta_actions),
        delta_actions,
      )
      prev_actions = actions.detach().clone()

      wheel_action = wrapped.unwrapped.action_manager.get_term("wheel_balance")
      wheel_target = wheel_action._processed_actions.detach()
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

      if step >= args.warmup_steps:
        lin_xs.append(robot_data.root_link_lin_vel_b[:, 0].detach().cpu())
        pitch_abses.append(pitch.abs().detach().cpu())
        pitch_rate_abses.append(robot_data.root_link_ang_vel_b[:, 1].abs().detach().cpu())
        wheel_target_abses.append(torch.mean(torch.abs(wheel_target), dim=1).detach().cpu())
        wheel_target_rates.append(
          torch.mean(torch.abs(delta_wheel_target), dim=1).detach().cpu()
        )
        action_abses.append(torch.abs(actions[:, 0]).detach().cpu())
        action_deltas.append(
          torch.mean(torch.abs(delta_actions), dim=1).detach().cpu()
        )
        action_signs.append(torch.sign(actions[:, 0]).detach().cpu())
        if reward_manager is not None:
          step_reward = reward_manager._step_reward.detach()
          for name, term_idx in reward_term_indices.items():
            weighted_rewards[name].append(step_reward[:, term_idx].cpu())

      if args.progress_interval > 0 and (
        (step + 1) % args.progress_interval == 0 or step + 1 == args.steps
      ):
        elapsed_s = time.perf_counter() - started_at
        steps_per_s = (step + 1) / max(elapsed_s, 1.0e-6)
        print(
          f"PROGRESS lin_x={lin_x_cmd:+.5f} "
          f"step={step + 1}/{args.steps} "
          f"elapsed={elapsed_s:.1f}s "
          f"steps/s={steps_per_s:.1f} "
          f"terminated={terminated_events} "
          f"timeouts={timeout_events}",
          flush=True,
        )

  lin_x = torch.cat(lin_xs)
  lin_x_by_step = torch.stack(lin_xs)
  pitch_abs = torch.cat(pitch_abses)
  pitch_rate_abs = torch.cat(pitch_rate_abses)
  wheel_target_abs = torch.cat(wheel_target_abses)
  wheel_target_rate = torch.cat(wheel_target_rates)
  action_abs = torch.cat(action_abses)
  action_delta = torch.cat(action_deltas)
  action_sign = torch.cat(action_signs)

  forward = lin_x > args.stuck_speed
  stuck = lin_x.abs() <= args.stuck_speed
  reverse = lin_x < args.reverse_speed
  target_error = lin_x - lin_x_cmd
  window_steps = min(args.window_steps, lin_x_by_step.shape[0])
  tracking = _command_tracking_health(
    lin_x_by_step=lin_x_by_step,
    target_lin_x=lin_x_cmd,
    stuck_speed=args.stuck_speed,
    window_steps=window_steps,
  )
  late_mean_lin_x = tracking["late_mean_lin_x"]
  slow_mask = tracking["slow_mask"]
  in_band_mask = tracking["in_band_mask"]
  fast_mask = tracking["fast_mask"]
  lin_x_delta = tracking["lin_x_delta"]
  lin_x_delta_slow_mask = tracking["lin_x_delta_slow_mask"]
  lin_x_delta_in_band_mask = tracking["lin_x_delta_in_band_mask"]
  lin_x_delta_fast_mask = tracking["lin_x_delta_fast_mask"]
  assert isinstance(slow_mask, torch.Tensor)
  assert isinstance(in_band_mask, torch.Tensor)
  assert isinstance(fast_mask, torch.Tensor)
  assert isinstance(lin_x_delta, torch.Tensor)
  assert isinstance(lin_x_delta_slow_mask, torch.Tensor)
  assert isinstance(lin_x_delta_in_band_mask, torch.Tensor)
  assert isinstance(lin_x_delta_fast_mask, torch.Tensor)
  ever_stuck_env = (lin_x_by_step.abs() <= args.stuck_speed).any(dim=0)
  mostly_stuck_env = (lin_x_by_step.abs() <= args.stuck_speed).float().mean(dim=0) > 0.25
  mean_lin_x = lin_x.mean().item()
  p95_pitch = _safe_quantile(pitch_abs, 0.95)
  p99_pitch_rate = _safe_quantile(pitch_rate_abs, 0.99)
  wheel_target_rate_rms = _safe_rms(wheel_target_rate)
  action_delta_rms = _safe_rms(action_delta)
  reward_contributions = _summarize_reward_contributions(
    weighted_rewards=weighted_rewards,
    reward_weights=reward_weights,
  )
  dynamic_stats = _bucketed_dynamic_stats(
    slow_mask=slow_mask,
    in_band_mask=in_band_mask,
    fast_mask=fast_mask,
    lin_x_delta=lin_x_delta,
    lin_x_delta_slow_mask=lin_x_delta_slow_mask,
    lin_x_delta_in_band_mask=lin_x_delta_in_band_mask,
    lin_x_delta_fast_mask=lin_x_delta_fast_mask,
    wheel_target_rate=wheel_target_rate,
    action_delta=action_delta,
    pitch_rate_abs=pitch_rate_abs,
  )

  print(f"Task: {args.task}")
  print(f"Play cfg: {args.play_cfg}")
  print(f"Episode length s: {wrapped.unwrapped.cfg.episode_length_s:.5g}")
  print(f"Fixed command: lin_x={lin_x_cmd:+.5f}, yaw={args.yaw:+.5f}")
  print(f"Samples after warmup: {lin_x.numel()}")
  print(f"done_event_rate:       {done_events / max(args.num_envs, 1):.5f}")
  print(f"terminated_event_rate: {terminated_events / max(args.num_envs, 1):.5f}")
  print(f"timeout_event_rate:    {timeout_events / max(args.num_envs, 1):.5f}")
  print(f"mean actual_lin_x:     {mean_lin_x:+.5f}")
  print(f"p05 actual_lin_x:      {_safe_quantile(lin_x, 0.05):+.5f}")
  print(f"p50 actual_lin_x:      {_safe_quantile(lin_x, 0.50):+.5f}")
  print(f"p95 actual_lin_x:      {_safe_quantile(lin_x, 0.95):+.5f}")
  print(f"signed mean lin_x:     {tracking['signed_lin_x_mean']:+.5f}")
  print(f"signed p05 lin_x:      {tracking['signed_lin_x_p05']:+.5f}")
  print(f"command_match_frac:    {tracking['command_match_frac']:.5f}")
  print(f"wrong_direction_frac:  {tracking['wrong_direction_frac']:.5f}")
  print(f"slow_frac:             {tracking['slow_frac']:.5f}")
  print(f"in_band_frac:          {tracking['in_band_frac']:.5f}")
  print(f"fast_frac:             {tracking['fast_frac']:.5f}")
  print(f"forward_frac:          {forward.float().mean().item():.5f}")
  print(f"stuck_frac:            {stuck.float().mean().item():.5f}")
  print(f"reverse_frac:          {reverse.float().mean().item():.5f}")
  print(f"late window steps:     {window_steps}")
  print(f"late mean lin_x p05:   {_safe_quantile(late_mean_lin_x, 0.05):+.5f}")
  print(f"late mean lin_x p50:   {_safe_quantile(late_mean_lin_x, 0.50):+.5f}")
  print(f"late mean lin_x p95:   {_safe_quantile(late_mean_lin_x, 0.95):+.5f}")
  print(f"late_stuck_env_frac:   {tracking['late_stuck_env_frac']:.5f}")
  print(f"late_slow_env_frac:    {tracking['late_slow_env_frac']:.5f}")
  print(f"late_slow_sample_frac: {tracking['late_slow_sample_frac']:.5f}")
  print(f"late_in_band_frac:     {tracking['late_in_band_frac']:.5f}")
  print(f"late_fast_sample_frac: {tracking['late_fast_sample_frac']:.5f}")
  print(f"late_wrong_direction_env_frac: {tracking['late_wrong_direction_env_frac']:.5f}")
  print(f"late_wrong_direction_sample_frac: {tracking['late_wrong_direction_sample_frac']:.5f}")
  print(f"ever_stuck_env_frac:   {ever_stuck_env.float().mean().item():.5f}")
  print(f"mostly_stuck_env_frac: {mostly_stuck_env.float().mean().item():.5f}")
  print(f"mean |lin_x error|:    {target_error.abs().mean().item():.5f}")
  print(f"p90 |lin_x error|:     {_safe_quantile(target_error.abs(), 0.90):.5f}")
  print(f"p95 |pitch|:           {p95_pitch:.5f}")
  print(f"p99 |pitch_rate|:      {p99_pitch_rate:.5f}")
  print(f"mean |wheel_target|:   {wheel_target_abs.mean().item():.5f}")
  print(f"wheel_target_rate_rms: {wheel_target_rate_rms:.5f}")
  print(f"action_delta_rms:      {action_delta_rms:.5f}")
  print(f"lin_x_delta_rms:       {tracking['lin_x_delta_rms']:.5f}")
  print(f"lin_x_delta_abs_p95:   {tracking['lin_x_delta_abs_p95']:.5f}")
  print(f"late_lin_x_delta_rms:  {tracking['late_lin_x_delta_rms']:.5f}")
  print(f"late_lin_x_delta_p95:  {tracking['late_lin_x_delta_abs_p95']:.5f}")
  print(f"mean |action|:         {action_abs.mean().item():.5f}")
  print(f"positive_action_frac:  {(action_sign > 0.0).float().mean().item():.5f}")
  print(f"negative_action_frac:  {(action_sign < 0.0).float().mean().item():.5f}")
  print(f"slow mean |action|:    {_safe_masked_mean(action_abs, slow_mask):.5f}")
  print(f"in_band mean |action|: {_safe_masked_mean(action_abs, in_band_mask):.5f}")
  print(f"fast mean |action|:    {_safe_masked_mean(action_abs, fast_mask):.5f}")
  print(
    f"slow wheel_rate mean:  {_safe_masked_mean(wheel_target_rate, slow_mask):.5f}"
  )
  print(
    f"in_band wheel_rate mean: {_safe_masked_mean(wheel_target_rate, in_band_mask):.5f}"
  )
  print(
    f"fast wheel_rate mean:  {_safe_masked_mean(wheel_target_rate, fast_mask):.5f}"
  )
  print("dynamic bucket stats:")
  for bucket in ("slow", "in_band", "fast"):
    print(
      f"  {bucket}: "
      f"lin_delta_rms={dynamic_stats[f'{bucket}_lin_x_delta_rms']:.5f} "
      f"lin_delta_p95={dynamic_stats[f'{bucket}_lin_x_delta_abs_p95']:.5f} "
      f"action_delta_rms={dynamic_stats[f'{bucket}_action_delta_rms']:.5f} "
      f"wheel_rate_rms={dynamic_stats[f'{bucket}_wheel_target_rate_rms']:.5f} "
      f"pitch_rate_p99={dynamic_stats[f'{bucket}_pitch_rate_abs_p99']:.5f}"
    )
  print("reward contribution debug:")
  print("  term raw_mean weighted_mean relative_to_track_linear_velocity")
  for name in REWARD_DEBUG_TERMS:
    print(
      f"  {name} "
      f"{reward_contributions[f'{name}_raw_mean']:.5f} "
      f"{reward_contributions[f'{name}_weighted_mean']:.5f} "
      f"{reward_contributions[f'{name}_relative_to_main']:.5f}"
    )

  summary = {
    "lin_x": lin_x_cmd,
    "mean_actual_lin_x": mean_lin_x,
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
    "lin_x_delta_rms": float(tracking["lin_x_delta_rms"]),
    "lin_x_delta_abs_p95": float(tracking["lin_x_delta_abs_p95"]),
    "late_lin_x_delta_rms": float(tracking["late_lin_x_delta_rms"]),
    "late_lin_x_delta_abs_p95": float(tracking["late_lin_x_delta_abs_p95"]),
    "slow_action_abs_mean": _safe_masked_mean(action_abs, slow_mask),
    "in_band_action_abs_mean": _safe_masked_mean(action_abs, in_band_mask),
    "fast_action_abs_mean": _safe_masked_mean(action_abs, fast_mask),
    "slow_wheel_target_rate_mean": _safe_masked_mean(wheel_target_rate, slow_mask),
    "in_band_wheel_target_rate_mean": _safe_masked_mean(
      wheel_target_rate,
      in_band_mask,
    ),
    "fast_wheel_target_rate_mean": _safe_masked_mean(wheel_target_rate, fast_mask),
    "mean_abs_error": target_error.abs().mean().item(),
    "p90_abs_error": _safe_quantile(target_error.abs(), 0.90),
    "p95_pitch": p95_pitch,
    "p99_pitch_rate": p99_pitch_rate,
    "wheel_target_rate_rms": wheel_target_rate_rms,
    "action_delta_rms": action_delta_rms,
    "terminated_event_rate": terminated_events / max(args.num_envs, 1),
  }
  summary.update(dynamic_stats)
  summary.update(reward_contributions)
  return summary


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
    for index, lin_x_cmd in enumerate(args.lin_x):
      if index:
        print("")
      summaries.append(
        _run_fixed_command(
          wrapped=wrapped,
          policy=policy,
          args=args,
          lin_x_cmd=lin_x_cmd,
        )
      )
  finally:
    wrapped.close()

  if len(summaries) > 1:
    print("")
    print(
      "SUMMARY lin_x mean match wrong slow in_band fast late_slow_env "
      "late_in_band late_wrong_env late_wrong_sample mean_abs_err p90_abs_err "
      "p95_pitch p99_pitch_rate wheel_rate action_delta lin_delta "
      "lin_delta_p95 term"
    )
    for row in summaries:
      print(
        f"SUMMARY {row['lin_x']:+.3f} {row['mean_actual_lin_x']:+.4f} "
        f"{row['command_match_frac']:.3f} {row['wrong_direction_frac']:.3f} "
        f"{row['slow_frac']:.3f} {row['in_band_frac']:.3f} "
        f"{row['fast_frac']:.3f} {row['late_slow_env_frac']:.3f} "
        f"{row['late_in_band_frac']:.3f} "
        f"{row['late_wrong_direction_env_frac']:.3f} "
        f"{row['late_wrong_direction_sample_frac']:.3f} "
        f"{row['mean_abs_error']:.4f} {row['p90_abs_error']:.4f} "
        f"{row['p95_pitch']:.4f} {row['p99_pitch_rate']:.4f} "
        f"{row['wheel_target_rate_rms']:.4f} "
        f"{row['action_delta_rms']:.4f} "
        f"{row['lin_x_delta_rms']:.4f} {row['lin_x_delta_abs_p95']:.4f} "
        f"{row['terminated_event_rate']:.3f}"
      )


if __name__ == "__main__":
  main()
