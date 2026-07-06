#!/usr/bin/env python3
"""Evaluate a policy under one sustained velocity command.

This complements stage gates, which sample randomized commands. It is meant to
catch viewer-visible attractors such as "starts moving, then settles into
standing balance while the forward command remains enabled".
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch

PROJECT_PATH = Path(__file__).resolve().parents[2]
if str(PROJECT_PATH) not in sys.path:
  sys.path.insert(0, str(PROJECT_PATH))

import tasks  # noqa: F401
from assets.HopperTrex_CFG import WHEEL_JOINT_NAMES
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


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
    late_wrong_direction_sample = late_signed_lin_x < -stuck_speed
    late_slow_sample = late_signed_lin_x < 0.5 * target_abs
  else:
    command_match = lin_x.abs() <= stuck_speed
    wrong_direction = lin_x.abs() > stuck_speed
    slow = wrong_direction
    late_wrong_direction_sample = late_lin_x.abs() > stuck_speed
    late_slow_sample = late_wrong_direction_sample

  return {
    "window_steps": float(window_steps),
    "signed_lin_x_mean": signed_lin_x.mean().item(),
    "signed_lin_x_p05": _safe_quantile(signed_lin_x, 0.05),
    "signed_lin_x_p50": _safe_quantile(signed_lin_x, 0.50),
    "signed_lin_x_p95": _safe_quantile(signed_lin_x, 0.95),
    "command_match_frac": command_match.float().mean().item(),
    "wrong_direction_frac": wrong_direction.float().mean().item(),
    "slow_frac": slow.float().mean().item(),
    "late_mean_lin_x": late_health["mean_lin_x"],
    "late_stuck_env_frac": late_health["stuck_env"].float().mean().item(),
    "late_slow_env_frac": late_health["slow_env"].float().mean().item(),
    "late_wrong_direction_env_frac": late_health["wrong_direction_env"].float().mean().item(),
    "late_wrong_direction_sample_frac": late_wrong_direction_sample.float().mean().item(),
    "late_slow_sample_frac": late_slow_sample.float().mean().item(),
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
  action_signs: list[torch.Tensor] = []
  done_events = 0
  terminated_events = 0
  timeout_events = 0
  prev_wheel_target: torch.Tensor | None = None

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
        action_signs.append(torch.sign(actions[:, 0]).detach().cpu())

  lin_x = torch.cat(lin_xs)
  lin_x_by_step = torch.stack(lin_xs)
  pitch_abs = torch.cat(pitch_abses)
  pitch_rate_abs = torch.cat(pitch_rate_abses)
  wheel_target_abs = torch.cat(wheel_target_abses)
  wheel_target_rate = torch.cat(wheel_target_rates)
  action_abs = torch.cat(action_abses)
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
  ever_stuck_env = (lin_x_by_step.abs() <= args.stuck_speed).any(dim=0)
  mostly_stuck_env = (lin_x_by_step.abs() <= args.stuck_speed).float().mean(dim=0) > 0.25
  mean_lin_x = lin_x.mean().item()
  p95_pitch = _safe_quantile(pitch_abs, 0.95)
  p99_pitch_rate = _safe_quantile(pitch_rate_abs, 0.99)
  wheel_target_rate_rms = torch.sqrt(torch.mean(torch.square(wheel_target_rate))).item()

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
  print(f"mean |action|:         {action_abs.mean().item():.5f}")
  print(f"positive_action_frac:  {(action_sign > 0.0).float().mean().item():.5f}")
  print(f"negative_action_frac:  {(action_sign < 0.0).float().mean().item():.5f}")

  return {
    "lin_x": lin_x_cmd,
    "mean_actual_lin_x": mean_lin_x,
    "command_match_frac": float(tracking["command_match_frac"]),
    "wrong_direction_frac": float(tracking["wrong_direction_frac"]),
    "slow_frac": float(tracking["slow_frac"]),
    "late_slow_env_frac": float(tracking["late_slow_env_frac"]),
    "late_wrong_direction_env_frac": float(
      tracking["late_wrong_direction_env_frac"]
    ),
    "late_wrong_direction_sample_frac": float(
      tracking["late_wrong_direction_sample_frac"]
    ),
    "mean_abs_error": target_error.abs().mean().item(),
    "p90_abs_error": _safe_quantile(target_error.abs(), 0.90),
    "p95_pitch": p95_pitch,
    "p99_pitch_rate": p99_pitch_rate,
    "wheel_target_rate_rms": wheel_target_rate_rms,
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
      "SUMMARY lin_x mean match wrong slow late_slow_env "
      "late_wrong_env late_wrong_sample mean_abs_err p90_abs_err "
      "p95_pitch p99_pitch_rate wheel_rate term"
    )
    for row in summaries:
      print(
        f"SUMMARY {row['lin_x']:+.3f} {row['mean_actual_lin_x']:+.4f} "
        f"{row['command_match_frac']:.3f} {row['wrong_direction_frac']:.3f} "
        f"{row['slow_frac']:.3f} {row['late_slow_env_frac']:.3f} "
        f"{row['late_wrong_direction_env_frac']:.3f} "
        f"{row['late_wrong_direction_sample_frac']:.3f} "
        f"{row['mean_abs_error']:.4f} {row['p90_abs_error']:.4f} "
        f"{row['p95_pitch']:.4f} {row['p99_pitch_rate']:.4f} "
        f"{row['wheel_target_rate_rms']:.4f} {row['terminated_event_rate']:.3f}"
      )


if __name__ == "__main__":
  main()
