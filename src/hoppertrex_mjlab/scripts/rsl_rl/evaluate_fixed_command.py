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


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", required=True)
  parser.add_argument("--checkpoint-file", required=True)
  parser.add_argument("--lin-x", type=float, default=0.07)
  parser.add_argument("--yaw", type=float, default=0.0)
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument("--warmup-steps", type=int, default=50)
  parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--stuck-speed", type=float, default=0.01)
  parser.add_argument("--reverse-speed", type=float, default=-0.01)
  parser.add_argument(
    "--window-steps",
    type=int,
    default=50,
    help="Window size for per-env late-run stuck/slow detection.",
  )
  return parser.parse_args()


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


def main() -> None:
  args = parse_args()
  configure_torch_backends()

  checkpoint = Path(args.checkpoint_file)
  if not checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

  env_cfg = load_env_cfg(args.task)
  agent_cfg = load_rl_cfg(args.task)
  env_cfg.scene.num_envs = args.num_envs
  if env_cfg.scene.terrain is not None:
    env_cfg.scene.terrain.num_envs = args.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
  runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)

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

  try:
    robot = wrapped.unwrapped.scene["robot"]
    wheel_joint_ids = torch.tensor(
      [list(robot.joint_names).index(name) for name in WHEEL_JOINT_NAMES],
      device=args.device,
      dtype=torch.long,
    )
    del wheel_joint_ids

    for step in range(args.steps):
      with torch.no_grad():
        _force_command(wrapped.unwrapped, args.lin_x, args.yaw)
        obs = wrapped.get_observations()
        actions = policy(obs).detach()
        obs, _rew, done, _extras = wrapped.step(actions)
        _force_command(wrapped.unwrapped, args.lin_x, args.yaw)

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
  finally:
    wrapped.close()

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
  target_error = lin_x - args.lin_x
  window_steps = min(args.window_steps, lin_x_by_step.shape[0])
  late_lin_x = lin_x_by_step[-window_steps:, :]
  late_mean_lin_x = late_lin_x.mean(dim=0)
  late_min_lin_x = late_lin_x.min(dim=0).values
  late_stuck_env = late_mean_lin_x.abs() <= args.stuck_speed
  late_slow_env = late_mean_lin_x < 0.5 * args.lin_x
  late_reverse_env = late_min_lin_x < args.reverse_speed
  ever_stuck_env = (lin_x_by_step.abs() <= args.stuck_speed).any(dim=0)
  mostly_stuck_env = (lin_x_by_step.abs() <= args.stuck_speed).float().mean(dim=0) > 0.25

  print(f"Task: {args.task}")
  print(f"Checkpoint: {checkpoint}")
  print(f"Fixed command: lin_x={args.lin_x:+.5f}, yaw={args.yaw:+.5f}")
  print(f"Samples after warmup: {lin_x.numel()}")
  print(f"done_event_rate:       {done_events / max(args.num_envs, 1):.5f}")
  print(f"terminated_event_rate: {terminated_events / max(args.num_envs, 1):.5f}")
  print(f"timeout_event_rate:    {timeout_events / max(args.num_envs, 1):.5f}")
  print(f"mean actual_lin_x:     {lin_x.mean().item():+.5f}")
  print(f"p05 actual_lin_x:      {_safe_quantile(lin_x, 0.05):+.5f}")
  print(f"p50 actual_lin_x:      {_safe_quantile(lin_x, 0.50):+.5f}")
  print(f"p95 actual_lin_x:      {_safe_quantile(lin_x, 0.95):+.5f}")
  print(f"forward_frac:          {forward.float().mean().item():.5f}")
  print(f"stuck_frac:            {stuck.float().mean().item():.5f}")
  print(f"reverse_frac:          {reverse.float().mean().item():.5f}")
  print(f"late window steps:     {window_steps}")
  print(f"late mean lin_x p05:   {_safe_quantile(late_mean_lin_x, 0.05):+.5f}")
  print(f"late mean lin_x p50:   {_safe_quantile(late_mean_lin_x, 0.50):+.5f}")
  print(f"late mean lin_x p95:   {_safe_quantile(late_mean_lin_x, 0.95):+.5f}")
  print(f"late_stuck_env_frac:   {late_stuck_env.float().mean().item():.5f}")
  print(f"late_slow_env_frac:    {late_slow_env.float().mean().item():.5f}")
  print(f"late_reverse_env_frac: {late_reverse_env.float().mean().item():.5f}")
  print(f"ever_stuck_env_frac:   {ever_stuck_env.float().mean().item():.5f}")
  print(f"mostly_stuck_env_frac: {mostly_stuck_env.float().mean().item():.5f}")
  print(f"mean |lin_x error|:    {target_error.abs().mean().item():.5f}")
  print(f"p90 |lin_x error|:     {_safe_quantile(target_error.abs(), 0.90):.5f}")
  print(f"p95 |pitch|:           {_safe_quantile(pitch_abs, 0.95):.5f}")
  print(f"p99 |pitch_rate|:      {_safe_quantile(pitch_rate_abs, 0.99):.5f}")
  print(f"mean |wheel_target|:   {wheel_target_abs.mean().item():.5f}")
  print(f"wheel_target_rate_rms: {torch.sqrt(torch.mean(torch.square(wheel_target_rate))).item():.5f}")
  print(f"mean |action|:         {action_abs.mean().item():.5f}")
  print(f"positive_action_frac:  {(action_sign > 0.0).float().mean().item():.5f}")
  print(f"negative_action_frac:  {(action_sign < 0.0).float().mean().item():.5f}")


if __name__ == "__main__":
  main()
