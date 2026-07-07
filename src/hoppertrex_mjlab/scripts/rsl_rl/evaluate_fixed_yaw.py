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
if str(PROJECT_PATH) not in sys.path:
  sys.path.insert(0, str(PROJECT_PATH))

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


def _yaw_tracking_health(
  *,
  yaw_by_step: torch.Tensor,
  lin_x_by_step: torch.Tensor,
  target_yaw: float,
  yaw_deadband: float,
  lin_drift_speed: float,
  window_steps: int,
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

  if target_abs > yaw_deadband:
    command_match = signed_yaw > yaw_deadband
    wrong_direction = signed_yaw < -yaw_deadband
    slow = signed_yaw < 0.5 * target_abs
    late_slow_env = late_mean_signed_yaw < 0.5 * target_abs
    late_wrong_direction_env = late_mean_signed_yaw < -yaw_deadband
    late_wrong_direction_sample = late_signed_yaw < -yaw_deadband
    late_slow_sample = late_signed_yaw < 0.5 * target_abs
  else:
    command_match = yaw.abs() <= yaw_deadband
    wrong_direction = yaw.abs() > yaw_deadband
    slow = wrong_direction
    late_slow_env = late_mean_signed_yaw.abs() > yaw_deadband
    late_wrong_direction_env = late_slow_env
    late_wrong_direction_sample = late_yaw.abs() > yaw_deadband
    late_slow_sample = late_wrong_direction_sample

  late_lin_drift_env = late_lin_drift > lin_drift_speed

  return {
    "window_steps": float(window_steps),
    "command_match_frac": command_match.float().mean().item(),
    "wrong_direction_frac": wrong_direction.float().mean().item(),
    "slow_frac": slow.float().mean().item(),
    "yaw_abs_error_mean": (yaw - target_yaw).abs().mean().item(),
    "yaw_abs_error_p90": _safe_quantile((yaw - target_yaw).abs(), 0.90),
    "lin_drift_abs_mean": lin_x.abs().mean().item(),
    "lin_drift_abs_p95": _safe_quantile(lin_x.abs(), 0.95),
    "late_mean_yaw": late_mean_yaw,
    "late_slow_env_frac": late_slow_env.float().mean().item(),
    "late_wrong_direction_env_frac": late_wrong_direction_env.float().mean().item(),
    "late_wrong_direction_sample_frac": late_wrong_direction_sample.float().mean().item(),
    "late_slow_sample_frac": late_slow_sample.float().mean().item(),
    "late_lin_drift_env_frac": late_lin_drift_env.float().mean().item(),
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
  done_events = 0
  terminated_events = 0
  timeout_events = 0
  prev_wheel_target: torch.Tensor | None = None
  started_at = time.perf_counter()

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
  target_error = yaw - yaw_cmd
  tracking = _yaw_tracking_health(
    yaw_by_step=yaw_by_step,
    lin_x_by_step=lin_x_by_step,
    target_yaw=yaw_cmd,
    yaw_deadband=args.yaw_deadband,
    lin_drift_speed=args.lin_drift_speed,
    window_steps=args.window_steps,
  )
  late_mean_yaw = tracking["late_mean_yaw"]
  assert isinstance(late_mean_yaw, torch.Tensor)
  p95_pitch = _safe_quantile(pitch_abs, 0.95)
  p99_pitch_rate = _safe_quantile(pitch_rate_abs, 0.99)
  wheel_target_rate_rms = torch.sqrt(torch.mean(torch.square(wheel_target_rate))).item()
  wheel_saturation_ratio = (wheel_target_abs >= WHEEL_TARGET_SATURATION).float().mean().item()

  print(f"Task: {args.task}")
  print(f"Play cfg: {args.play_cfg}")
  print(f"Episode length s: {wrapped.unwrapped.cfg.episode_length_s:.5g}")
  print(f"Fixed command: lin_x={args.lin_x:+.5f}, yaw={yaw_cmd:+.5f}")
  print(f"Samples after warmup: {yaw.numel()}")
  print(f"done_event_rate:       {done_events / max(args.num_envs, 1):.5f}")
  print(f"terminated_event_rate: {terminated_events / max(args.num_envs, 1):.5f}")
  print(f"timeout_event_rate:    {timeout_events / max(args.num_envs, 1):.5f}")
  print(f"mean actual_yaw:       {yaw.mean().item():+.5f}")
  print(f"p05 actual_yaw:        {_safe_quantile(yaw, 0.05):+.5f}")
  print(f"p50 actual_yaw:        {_safe_quantile(yaw, 0.50):+.5f}")
  print(f"p95 actual_yaw:        {_safe_quantile(yaw, 0.95):+.5f}")
  print(f"command_match_frac:    {tracking['command_match_frac']:.5f}")
  print(f"wrong_direction_frac:  {tracking['wrong_direction_frac']:.5f}")
  print(f"slow_frac:             {tracking['slow_frac']:.5f}")
  print(f"late window steps:     {int(tracking['window_steps'])}")
  print(f"late mean yaw p05:     {_safe_quantile(late_mean_yaw, 0.05):+.5f}")
  print(f"late mean yaw p50:     {_safe_quantile(late_mean_yaw, 0.50):+.5f}")
  print(f"late mean yaw p95:     {_safe_quantile(late_mean_yaw, 0.95):+.5f}")
  print(f"late_slow_env_frac:    {tracking['late_slow_env_frac']:.5f}")
  print(f"late_slow_sample_frac: {tracking['late_slow_sample_frac']:.5f}")
  print(f"late_wrong_direction_env_frac: {tracking['late_wrong_direction_env_frac']:.5f}")
  print(f"late_wrong_direction_sample_frac: {tracking['late_wrong_direction_sample_frac']:.5f}")
  print(f"late_lin_drift_env_frac: {tracking['late_lin_drift_env_frac']:.5f}")
  print(f"mean |yaw error|:      {target_error.abs().mean().item():.5f}")
  print(f"p90 |yaw error|:       {_safe_quantile(target_error.abs(), 0.90):.5f}")
  print(f"mean |lin_x drift|:    {lin_x.abs().mean().item():.5f}")
  print(f"p95 |lin_x drift|:     {_safe_quantile(lin_x.abs(), 0.95):.5f}")
  print(f"p95 |pitch|:           {p95_pitch:.5f}")
  print(f"p99 |pitch_rate|:      {p99_pitch_rate:.5f}")
  print(f"mean |wheel_target|:   {wheel_target_abs.mean().item():.5f}")
  print(f"wheel_saturation_ratio:{wheel_saturation_ratio:.5f}")
  print(f"wheel_target_rate_rms: {wheel_target_rate_rms:.5f}")
  print(f"mean |action|:         {action_abs.mean().item():.5f}")
  print(f"positive_yaw_action_frac: {(yaw_action_sign > 0.0).float().mean().item():.5f}")
  print(f"negative_yaw_action_frac: {(yaw_action_sign < 0.0).float().mean().item():.5f}")

  return {
    "yaw": yaw_cmd,
    "mean_actual_yaw": yaw.mean().item(),
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
    "late_lin_drift_env_frac": float(tracking["late_lin_drift_env_frac"]),
    "yaw_abs_error_mean": float(tracking["yaw_abs_error_mean"]),
    "yaw_abs_error_p90": float(tracking["yaw_abs_error_p90"]),
    "lin_drift_abs_mean": float(tracking["lin_drift_abs_mean"]),
    "lin_drift_abs_p95": float(tracking["lin_drift_abs_p95"]),
    "p95_pitch": p95_pitch,
    "p99_pitch_rate": p99_pitch_rate,
    "wheel_saturation_ratio": wheel_saturation_ratio,
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
      "SUMMARY yaw mean match wrong slow late_slow_env "
      "late_wrong_env late_wrong_sample late_lin_drift_env "
      "yaw_abs_err yaw_p90_abs_err lin_drift lin_p95_drift "
      "p95_pitch p99_pitch_rate wheel_sat wheel_rate term"
    )
    for row in summaries:
      print(
        f"SUMMARY {row['yaw']:+.3f} {row['mean_actual_yaw']:+.4f} "
        f"{row['command_match_frac']:.3f} {row['wrong_direction_frac']:.3f} "
        f"{row['slow_frac']:.3f} {row['late_slow_env_frac']:.3f} "
        f"{row['late_wrong_direction_env_frac']:.3f} "
        f"{row['late_wrong_direction_sample_frac']:.3f} "
        f"{row['late_lin_drift_env_frac']:.3f} "
        f"{row['yaw_abs_error_mean']:.4f} {row['yaw_abs_error_p90']:.4f} "
        f"{row['lin_drift_abs_mean']:.4f} {row['lin_drift_abs_p95']:.4f} "
        f"{row['p95_pitch']:.4f} {row['p99_pitch_rate']:.4f} "
        f"{row['wheel_saturation_ratio']:.4f} "
        f"{row['wheel_target_rate_rms']:.4f} "
        f"{row['terminated_event_rate']:.3f}"
      )


if __name__ == "__main__":
  main()
