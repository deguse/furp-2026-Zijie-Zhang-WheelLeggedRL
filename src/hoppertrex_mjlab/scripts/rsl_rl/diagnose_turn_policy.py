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
from assets.HopperTrex_CFG import WHEEL_JOINT_NAMES
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
  residual_action_balance = data["residual_action_balance"][mask]
  feedforward_action_balance = data["feedforward_action_balance"][mask]
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
  raw_leg_action_abs = data.get("raw_leg_action_abs")
  delta_raw_leg_action_abs = data.get("delta_raw_leg_action_abs")
  pitch_proxy = data["pitch_proxy"][mask]
  pitch_rate = data["pitch_rate"][mask]
  wheel_target_abs = data["wheel_target_abs"][mask]
  wheel_force_abs = data["wheel_force_abs"][mask]
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
  if feedforward_action_balance.abs().max().item() > 0.0:
    print(f"  mean residual_bal:   {residual_action_balance.mean().item():+.5f}")
    print(f"  mean ff_balance:     {feedforward_action_balance.mean().item():+.5f}")
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
  print(f"  mean |wheel_tgt|:    {wheel_target_abs.mean().item():+.5f}")
  print(f"  p95 |wheel_tgt|:     {torch.quantile(wheel_target_abs, 0.95).item():+.5f}")
  print(f"  target sat frac:     {(wheel_target_abs >= 23.9).float().mean().item():.3f}")
  print(f"  mean |wheel_force|:  {wheel_force_abs.mean().item():+.5f}")
  print(f"  p95 |wheel_force|:   {torch.quantile(wheel_force_abs, 0.95).item():+.5f}")
  print(f"  mean |pitch_proxy|:  {pitch_proxy.abs().mean().item():+.5f}")
  print(f"  p95 |pitch_proxy|:   {torch.quantile(pitch_proxy.abs(), 0.95).item():+.5f}")
  print(f"  mean |pitch_rate|:   {pitch_rate.abs().mean().item():+.5f}")
  print(f"  p95 |pitch_rate|:    {torch.quantile(pitch_rate.abs(), 0.95).item():+.5f}")
  if raw_leg_action_abs is not None and delta_raw_leg_action_abs is not None:
    group_raw_leg_action_abs = raw_leg_action_abs[mask]
    group_delta_raw_leg_action_abs = delta_raw_leg_action_abs[mask]
    print(f"  mean |raw_leg|:      {group_raw_leg_action_abs.mean().item():+.5f}")
    print(f"  p95 |raw_leg|:       {torch.quantile(group_raw_leg_action_abs, 0.95).item():+.5f}")
    print(f"  mean |d_raw_leg|:    {group_delta_raw_leg_action_abs.mean().item():+.5f}")
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
  residual_action_balances: list[torch.Tensor] = []
  feedforward_action_balances: list[torch.Tensor] = []
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
  wheel_target_abses: list[torch.Tensor] = []
  wheel_force_abses: list[torch.Tensor] = []
  pitch_proxies: list[torch.Tensor] = []
  pitch_rates: list[torch.Tensor] = []
  actual_yaws: list[torch.Tensor] = []
  actual_lin_xs: list[torch.Tensor] = []
  raw_leg_action_abses: list[torch.Tensor] = []
  delta_raw_leg_action_abses: list[torch.Tensor] = []
  prev_clip_balance: torch.Tensor | None = None
  prev_clip_yaw: torch.Tensor | None = None
  prev_effective_yaw: torch.Tensor | None = None
  prev_wheel_target: torch.Tensor | None = None
  prev_raw_leg_action: torch.Tensor | None = None

  try:
    obs = wrapped.get_observations()
    has_leg_assist = "leg_assist_pos" in wrapped.unwrapped.action_manager.active_terms
    robot = wrapped.unwrapped.scene["robot"]
    wheel_joint_ids = torch.tensor(
      [list(robot.joint_names).index(name) for name in WHEEL_JOINT_NAMES],
      device=args.device,
      dtype=torch.long,
    )
    for _ in range(args.steps):
      with torch.no_grad():
        cmd = wrapped.unwrapped.command_manager.get_command("twist").detach()
        actions = policy(obs).detach()
        obs, _rew, _done, _extras = wrapped.step(actions)
        robot_data = robot.data
        wheel_action = wrapped.unwrapped.action_manager.get_term("wheel_balance")
        wheel_raw_actions = wheel_action._raw_actions.detach()
        residual_actions = getattr(wheel_action, "_residual_actions", None)
        feedforward_actions = getattr(wheel_action, "_feedforward_actions", None)
        if has_leg_assist:
          leg_action = wrapped.unwrapped.action_manager.get_term("leg_assist_pos")
          raw_leg_action = leg_action._raw_actions.detach()
        else:
          raw_leg_action = None
        wheel_action_dim = wheel_raw_actions.shape[1]
        clipped_balance = wheel_raw_actions[:, 0]
        residual_balance = (
          residual_actions[:, 0].detach()
          if residual_actions is not None
          else torch.zeros_like(clipped_balance)
        )
        feedforward_balance = (
          feedforward_actions[:, 0].detach()
          if feedforward_actions is not None
          else torch.zeros_like(clipped_balance)
        )
        clipped_yaw = (
          wheel_raw_actions[:, 1]
          if wheel_action_dim > 1
          else torch.zeros_like(clipped_balance)
        )
        if wheel_action_dim > 1 and getattr(wheel_action, "_yaw_smoothing_alpha", None) is not None:
          effective_yaw = wheel_action._smoothed_yaw_action
        elif wheel_action_dim > 1:
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
        if raw_leg_action is None:
          raw_leg_action_abs = torch.zeros_like(clipped_balance)
          delta_raw_leg_action_abs = torch.zeros_like(clipped_balance)
        elif prev_raw_leg_action is None:
          raw_leg_action_abs = torch.mean(torch.abs(raw_leg_action), dim=1)
          delta_raw_leg_action_abs = torch.zeros_like(raw_leg_action_abs)
        else:
          raw_leg_action_abs = torch.mean(torch.abs(raw_leg_action), dim=1)
          delta_raw_leg_action = raw_leg_action - prev_raw_leg_action
          delta_raw_leg_action_abs = torch.mean(torch.abs(delta_raw_leg_action), dim=1)
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
        delta_raw_leg_action_abs = torch.where(
          done_mask,
          torch.zeros_like(delta_raw_leg_action_abs),
          delta_raw_leg_action_abs,
        )
        prev_clip_balance = clipped_balance.detach().clone()
        prev_clip_yaw = clipped_yaw.detach().clone()
        prev_effective_yaw = effective_yaw.detach().clone()
        prev_wheel_target = wheel_target.detach().clone()
        prev_raw_leg_action = (
          raw_leg_action.detach().clone() if raw_leg_action is not None else None
        )
        actual_yaw = robot_data.root_link_ang_vel_b[:, 2].detach()
        actual_lin_x = robot_data.root_link_lin_vel_b[:, 0].detach()
        projected_gravity = robot_data.projected_gravity_b.detach()
        pitch_proxy = torch.atan2(
          projected_gravity[:, 0],
          torch.clamp(-projected_gravity[:, 2], min=1.0e-6),
        )
        pitch_rate = robot_data.root_link_ang_vel_b[:, 1].detach()
        wheel_target_abs = torch.mean(torch.abs(wheel_target), dim=1)
        wheel_force_abs = torch.mean(
          torch.abs(robot_data.qfrc_actuator[:, wheel_joint_ids]), dim=1
        )

      cmd_yaws.append(cmd[:, 2].cpu())
      cmd_lin_xs.append(cmd[:, 0].cpu())
      raw_action_balances.append(clipped_balance.cpu())
      residual_action_balances.append(residual_balance.cpu())
      feedforward_action_balances.append(feedforward_balance.cpu())
      raw_action_yaws.append(clipped_yaw.cpu())
      clipped_action_balances.append(clipped_balance.cpu())
      clipped_action_yaws.append(clipped_yaw.cpu())
      effective_action_yaws.append(effective_yaw.detach().cpu())
      delta_clip_balances.append(delta_clip_balance.detach().cpu())
      delta_clip_yaws.append(delta_clip_yaw.detach().cpu())
      delta_effective_yaws.append(delta_effective_yaw.detach().cpu())
      delta_left_targets.append(delta_wheel_target[:, 0].detach().cpu())
      delta_right_targets.append(delta_wheel_target[:, 1].detach().cpu())
      delta_wheel_targets.append(torch.mean(torch.abs(delta_wheel_target), dim=1).cpu())
      wheel_target_abses.append(wheel_target_abs.detach().cpu())
      wheel_force_abses.append(wheel_force_abs.detach().cpu())
      pitch_proxies.append(pitch_proxy.detach().cpu())
      pitch_rates.append(pitch_rate.detach().cpu())
      if has_leg_assist:
        raw_leg_action_abses.append(raw_leg_action_abs.detach().cpu())
        delta_raw_leg_action_abses.append(delta_raw_leg_action_abs.detach().cpu())
      actual_yaws.append(actual_yaw.cpu())
      actual_lin_xs.append(actual_lin_x.cpu())

    data = {
      "cmd_yaw": torch.cat(cmd_yaws),
      "cmd_lin_x": torch.cat(cmd_lin_xs),
      "raw_action_balance": torch.cat(raw_action_balances),
      "residual_action_balance": torch.cat(residual_action_balances),
      "feedforward_action_balance": torch.cat(feedforward_action_balances),
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
      "wheel_target_abs": torch.cat(wheel_target_abses),
      "wheel_force_abs": torch.cat(wheel_force_abses),
      "pitch_proxy": torch.cat(pitch_proxies),
      "pitch_rate": torch.cat(pitch_rates),
      "actual_yaw": torch.cat(actual_yaws),
      "actual_lin_x": torch.cat(actual_lin_xs),
    }
    if has_leg_assist:
      data["raw_leg_action_abs"] = torch.cat(raw_leg_action_abses)
      data["delta_raw_leg_action_abs"] = torch.cat(delta_raw_leg_action_abses)
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
