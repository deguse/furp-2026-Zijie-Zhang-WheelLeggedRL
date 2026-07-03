#!/usr/bin/env python3
"""Diagnose whether a turn policy uses yaw command sign correctly."""

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
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


DEFAULT_TASK = "Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-v0"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("task", nargs="?", default=DEFAULT_TASK)
  parser.add_argument("--checkpoint-file", required=True)
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  parser.add_argument(
    "--detail-groups",
    action="store_true",
    help="Also print forward/backward by positive/negative yaw quadrant groups.",
  )
  parser.add_argument(
    "--slew-cap",
    type=float,
    default=None,
    help="Optional wheel target slew cap used to report near-cap fractions.",
  )
  return parser.parse_args()


def _print_group(
  name: str,
  mask: torch.Tensor,
  data: dict[str, torch.Tensor],
  *,
  slew_cap: float | None,
) -> None:
  count = int(mask.sum().item())
  if count == 0:
    print(f"{name}: no samples")
    return

  cmd_yaw = data["cmd_yaw"][mask]
  cmd_lin_x = data["cmd_lin_x"][mask]
  raw_action_balance = data["raw_action_balance"][mask]
  raw_action_yaw = data["raw_action_yaw"][mask]
  clipped_action_balance = data["clipped_action_balance"][mask]
  clipped_action_yaw = data["clipped_action_yaw"][mask]
  effective_action_yaw = data["effective_action_yaw"][mask]
  delta_clip_balance = data["delta_clip_balance"][mask]
  delta_clip_yaw = data["delta_clip_yaw"][mask]
  delta_effective_yaw = data["delta_effective_yaw"][mask]
  delta_left_target = data["delta_left_target"][mask]
  delta_right_target = data["delta_right_target"][mask]
  delta_wheel_target = data["delta_wheel_target"][mask]
  actual_yaw = data["actual_yaw"][mask]
  actual_lin_x = data["actual_lin_x"][mask]
  cmd_action = cmd_yaw * effective_action_yaw
  cmd_actual = cmd_yaw * actual_yaw
  cmd_lin_actual = cmd_lin_x * actual_lin_x
  lin_error = actual_lin_x - cmd_lin_x
  delta_wheel_target_abs = delta_wheel_target.abs()

  print(f"\n{name}: n={count}")
  print(f"  mean cmd_lin_x:      {cmd_lin_x.mean().item():+.5f}")
  print(f"  mean cmd_yaw:        {cmd_yaw.mean().item():+.5f}")
  print(f"  mean raw_balance:    {raw_action_balance.mean().item():+.5f}")
  print(f"  mean |raw_balance|:  {raw_action_balance.abs().mean().item():+.5f}")
  print(f"  mean clip_balance:   {clipped_action_balance.mean().item():+.5f}")
  print(f"  mean |clip_balance|: {clipped_action_balance.abs().mean().item():+.5f}")
  print(f"  mean raw_yaw:        {raw_action_yaw.mean().item():+.5f}")
  print(f"  mean |raw_yaw|:      {raw_action_yaw.abs().mean().item():+.5f}")
  print(f"  mean clip_yaw:       {clipped_action_yaw.mean().item():+.5f}")
  print(f"  mean |clip_yaw|:     {clipped_action_yaw.abs().mean().item():+.5f}")
  print(f"  mean effective_yaw:  {effective_action_yaw.mean().item():+.5f}")
  print(f"  mean |effective_yaw|:{effective_action_yaw.abs().mean().item():+.5f}")
  print(f"  mean |d_clip_bal|:   {delta_clip_balance.abs().mean().item():+.5f}")
  print(f"  mean |d_clip_yaw|:   {delta_clip_yaw.abs().mean().item():+.5f}")
  print(f"  mean |d_eff_yaw|:    {delta_effective_yaw.abs().mean().item():+.5f}")
  print(f"  mean |d_left_tgt|:   {delta_left_target.abs().mean().item():+.5f}")
  print(f"  mean |d_right_tgt|:  {delta_right_target.abs().mean().item():+.5f}")
  print(f"  mean |d_wheel_tgt|:  {delta_wheel_target.mean().item():+.5f}")
  print(f"  p95 |d_left_tgt|:    {torch.quantile(delta_left_target.abs(), 0.95).item():+.5f}")
  print(f"  p95 |d_right_tgt|:   {torch.quantile(delta_right_target.abs(), 0.95).item():+.5f}")
  print(f"  p95 |d_wheel_tgt|:   {torch.quantile(delta_wheel_target, 0.95).item():+.5f}")
  print(f"  max |d_left_tgt|:    {delta_left_target.abs().max().item():+.5f}")
  print(f"  max |d_right_tgt|:   {delta_right_target.abs().max().item():+.5f}")
  print(f"  max |d_wheel_tgt|:   {delta_wheel_target.max().item():+.5f}")
  if slew_cap is not None:
    near_cap = delta_wheel_target_abs >= max(slew_cap - 1.0e-4, 0.0)
    print(f"  slew cap frac:       {near_cap.float().mean().item():.3f}")
  print(f"  mean actual_yaw:     {actual_yaw.mean().item():+.5f}")
  print(f"  mean actual_lin_x:   {actual_lin_x.mean().item():+.5f}")
  print(f"  mean lin_x error:    {lin_error.mean().item():+.5f}")
  print(f"  mean |lin_x error|:  {lin_error.abs().mean().item():+.5f}")
  print(f"  p95 |lin_x error|:   {torch.quantile(lin_error.abs(), 0.95).item():+.5f}")
  print(f"  p05 actual_lin_x:    {torch.quantile(actual_lin_x, 0.05).item():+.5f}")
  print(f"  min actual_lin_x:    {actual_lin_x.min().item():+.5f}")
  print(f"  reverse lin_x frac:  {(actual_lin_x < 0.0).float().mean().item():.3f}")
  print(f"  hard reverse frac:   {(actual_lin_x < -0.03).float().mean().item():.3f}")
  print(f"  lin sign match:      {(cmd_lin_actual > 0).float().mean().item():.3f}")
  print(f"  clip sign match:     {(cmd_action > 0).float().mean().item():.3f}")
  print(f"  actual sign match:   {(cmd_actual > 0).float().mean().item():.3f}")
  print(f"  yaw_sign_alignment:  {(cmd_actual / torch.clamp(cmd_yaw.square(), min=1.0e-6)).clamp(-1.0, 1.0).mean().item():+.5f}")


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
  runner.load(
    str(checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(device=args.device)

  cmd_yaws: list[torch.Tensor] = []
  cmd_lin_xs: list[torch.Tensor] = []
  raw_action_balances: list[torch.Tensor] = []
  raw_action_yaws: list[torch.Tensor] = []
  clipped_action_balances: list[torch.Tensor] = []
  clipped_action_yaws: list[torch.Tensor] = []
  effective_action_yaws: list[torch.Tensor] = []
  delta_clip_balances: list[torch.Tensor] = []
  delta_clip_yaws: list[torch.Tensor] = []
  delta_effective_yaws: list[torch.Tensor] = []
  delta_left_targets: list[torch.Tensor] = []
  delta_right_targets: list[torch.Tensor] = []
  delta_wheel_targets: list[torch.Tensor] = []
  actual_yaws: list[torch.Tensor] = []
  actual_lin_xs: list[torch.Tensor] = []
  prev_clip_balance: torch.Tensor | None = None
  prev_clip_yaw: torch.Tensor | None = None
  prev_effective_yaw: torch.Tensor | None = None
  prev_wheel_target: torch.Tensor | None = None

  try:
    obs = wrapped.get_observations()
    for _ in range(args.steps):
      with torch.no_grad():
        cmd = wrapped.unwrapped.command_manager.get_command("twist").detach()
        actions = policy(obs).detach()
        actions_clipped = torch.clamp(actions, -1.0, 1.0)
        action_dim = actions_clipped.shape[1]
        clipped_balance = actions_clipped[:, 0]
        clipped_yaw = (
          actions_clipped[:, 1]
          if action_dim > 1
          else torch.zeros_like(clipped_balance)
        )
        obs, _rew, _done, _extras = wrapped.step(actions)
        robot_data = wrapped.unwrapped.scene["robot"].data
        wheel_action = wrapped.unwrapped.action_manager.get_term("wheel_balance")
        if action_dim > 1 and getattr(wheel_action, "_yaw_smoothing_alpha", None) is not None:
          effective_yaw = wheel_action._smoothed_yaw_action
        elif action_dim > 1:
          effective_yaw = clipped_yaw
        else:
          effective_yaw = torch.zeros_like(clipped_balance)
        wheel_target = wheel_action._processed_actions.detach()
        if prev_clip_balance is None:
          delta_clip_balance = torch.zeros_like(clipped_balance)
          delta_clip_yaw = torch.zeros_like(clipped_yaw)
          delta_effective_yaw = torch.zeros_like(effective_yaw)
          delta_wheel_target = torch.zeros_like(wheel_target)
        else:
          delta_clip_balance = clipped_balance - prev_clip_balance
          delta_clip_yaw = clipped_yaw - prev_clip_yaw
          assert prev_effective_yaw is not None
          delta_effective_yaw = effective_yaw - prev_effective_yaw
          assert prev_wheel_target is not None
          delta_wheel_target = wheel_target - prev_wheel_target
        done_mask = _done.to(dtype=torch.bool)
        delta_clip_balance = torch.where(
          done_mask, torch.zeros_like(delta_clip_balance), delta_clip_balance
        )
        delta_clip_yaw = torch.where(
          done_mask, torch.zeros_like(delta_clip_yaw), delta_clip_yaw
        )
        delta_effective_yaw = torch.where(
          done_mask, torch.zeros_like(delta_effective_yaw), delta_effective_yaw
        )
        delta_wheel_target = torch.where(
          done_mask.unsqueeze(-1),
          torch.zeros_like(delta_wheel_target),
          delta_wheel_target,
        )
        prev_clip_balance = clipped_balance.detach().clone()
        prev_clip_yaw = clipped_yaw.detach().clone()
        prev_effective_yaw = effective_yaw.detach().clone()
        prev_wheel_target = wheel_target.detach().clone()
        actual_yaw = robot_data.root_link_ang_vel_b[:, 2].detach()
        actual_lin_x = robot_data.root_link_lin_vel_b[:, 0].detach()

      cmd_yaws.append(cmd[:, 2].cpu())
      cmd_lin_xs.append(cmd[:, 0].cpu())
      raw_action_balances.append(actions[:, 0].cpu())
      raw_action_yaws.append(
        (actions[:, 1] if actions.shape[1] > 1 else torch.zeros_like(actions[:, 0])).cpu()
      )
      clipped_action_balances.append(clipped_balance.cpu())
      clipped_action_yaws.append(clipped_yaw.cpu())
      effective_action_yaws.append(effective_yaw.detach().cpu())
      delta_clip_balances.append(delta_clip_balance.detach().cpu())
      delta_clip_yaws.append(delta_clip_yaw.detach().cpu())
      delta_effective_yaws.append(delta_effective_yaw.detach().cpu())
      delta_left_targets.append(delta_wheel_target[:, 0].detach().cpu())
      delta_right_targets.append(delta_wheel_target[:, 1].detach().cpu())
      delta_wheel_targets.append(torch.mean(torch.abs(delta_wheel_target), dim=1).cpu())
      actual_yaws.append(actual_yaw.cpu())
      actual_lin_xs.append(actual_lin_x.cpu())

    data = {
      "cmd_yaw": torch.cat(cmd_yaws),
      "cmd_lin_x": torch.cat(cmd_lin_xs),
      "raw_action_balance": torch.cat(raw_action_balances),
      "raw_action_yaw": torch.cat(raw_action_yaws),
      "clipped_action_balance": torch.cat(clipped_action_balances),
      "clipped_action_yaw": torch.cat(clipped_action_yaws),
      "effective_action_yaw": torch.cat(effective_action_yaws),
      "delta_clip_balance": torch.cat(delta_clip_balances),
      "delta_clip_yaw": torch.cat(delta_clip_yaws),
      "delta_effective_yaw": torch.cat(delta_effective_yaws),
      "delta_left_target": torch.cat(delta_left_targets),
      "delta_right_target": torch.cat(delta_right_targets),
      "delta_wheel_target": torch.cat(delta_wheel_targets),
      "actual_yaw": torch.cat(actual_yaws),
      "actual_lin_x": torch.cat(actual_lin_xs),
    }
    pos = data["cmd_yaw"] > 0
    neg = data["cmd_yaw"] < 0
    forward = data["cmd_lin_x"] > 0.01
    backward = data["cmd_lin_x"] < -0.01

    print(f"Task: {args.task}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Samples: {data['cmd_yaw'].numel()}")
    _print_group("cmd_yaw > 0", pos, data, slew_cap=args.slew_cap)
    _print_group("cmd_yaw < 0", neg, data, slew_cap=args.slew_cap)
    _print_group("cmd_lin_x > 0.01", forward, data, slew_cap=args.slew_cap)
    _print_group("cmd_lin_x < -0.01", backward, data, slew_cap=args.slew_cap)
    if args.detail_groups:
      _print_group("cmd_lin_x > 0.01 and cmd_yaw > 0", forward & pos, data, slew_cap=args.slew_cap)
      _print_group("cmd_lin_x > 0.01 and cmd_yaw < 0", forward & neg, data, slew_cap=args.slew_cap)
      _print_group("cmd_lin_x < -0.01 and cmd_yaw > 0", backward & pos, data, slew_cap=args.slew_cap)
      _print_group("cmd_lin_x < -0.01 and cmd_yaw < 0", backward & neg, data, slew_cap=args.slew_cap)
    _print_group("all", torch.ones_like(pos, dtype=torch.bool), data, slew_cap=args.slew_cap)
  finally:
    wrapped.close()


if __name__ == "__main__":
  main()
