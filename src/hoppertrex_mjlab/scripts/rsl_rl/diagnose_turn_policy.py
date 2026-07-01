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
  return parser.parse_args()


def _print_group(name: str, mask: torch.Tensor, data: dict[str, torch.Tensor]) -> None:
  count = int(mask.sum().item())
  if count == 0:
    print(f"{name}: no samples")
    return

  cmd_yaw = data["cmd_yaw"][mask]
  action_balance = data["action_balance"][mask]
  action_yaw = data["action_yaw"][mask]
  actual_yaw = data["actual_yaw"][mask]
  actual_lin_x = data["actual_lin_x"][mask]
  cmd_action = cmd_yaw * action_yaw
  cmd_actual = cmd_yaw * actual_yaw

  print(f"\n{name}: n={count}")
  print(f"  mean cmd_yaw:        {cmd_yaw.mean().item():+.5f}")
  print(f"  mean action_balance: {action_balance.mean().item():+.5f}")
  print(f"  mean |act_balance|:  {action_balance.abs().mean().item():+.5f}")
  print(f"  mean action_yaw:     {action_yaw.mean().item():+.5f}")
  print(f"  mean |action_yaw|:   {action_yaw.abs().mean().item():+.5f}")
  print(f"  mean actual_yaw:     {actual_yaw.mean().item():+.5f}")
  print(f"  mean actual_lin_x:   {actual_lin_x.mean().item():+.5f}")
  print(f"  action sign match:   {(cmd_action > 0).float().mean().item():.3f}")
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
  action_balances: list[torch.Tensor] = []
  action_yaws: list[torch.Tensor] = []
  actual_yaws: list[torch.Tensor] = []
  actual_lin_xs: list[torch.Tensor] = []

  try:
    obs = wrapped.get_observations()
    for _ in range(args.steps):
      with torch.no_grad():
        cmd = wrapped.unwrapped.command_manager.get_command("twist").detach()
        actions = policy(obs).detach()
        if agent_cfg.clip_actions is not None:
          actions_for_stats = torch.clamp(actions, -agent_cfg.clip_actions, agent_cfg.clip_actions)
        else:
          actions_for_stats = actions
        obs, _rew, _done, _extras = wrapped.step(actions)
        robot_data = wrapped.unwrapped.scene["robot"].data
        actual_yaw = robot_data.root_link_ang_vel_b[:, 2].detach()
        actual_lin_x = robot_data.root_link_lin_vel_b[:, 0].detach()

      cmd_yaws.append(cmd[:, 2].cpu())
      action_balances.append(actions_for_stats[:, 0].cpu())
      action_yaws.append(actions_for_stats[:, 1].cpu())
      actual_yaws.append(actual_yaw.cpu())
      actual_lin_xs.append(actual_lin_x.cpu())

    data = {
      "cmd_yaw": torch.cat(cmd_yaws),
      "action_balance": torch.cat(action_balances),
      "action_yaw": torch.cat(action_yaws),
      "actual_yaw": torch.cat(actual_yaws),
      "actual_lin_x": torch.cat(actual_lin_xs),
    }
    pos = data["cmd_yaw"] > 0
    neg = data["cmd_yaw"] < 0

    print(f"Task: {args.task}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Samples: {data['cmd_yaw'].numel()}")
    _print_group("cmd_yaw > 0", pos, data)
    _print_group("cmd_yaw < 0", neg, data)
    _print_group("all", pos | neg, data)
  finally:
    wrapped.close()


if __name__ == "__main__":
  main()
