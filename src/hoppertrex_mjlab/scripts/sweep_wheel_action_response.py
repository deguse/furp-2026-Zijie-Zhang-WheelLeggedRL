#!/usr/bin/env python3
"""Sweep fixed wheel actions and report the resulting body-frame x velocity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_PATH = Path(__file__).resolve().parents[1]
if str(PROJECT_PATH) not in sys.path:
  sys.path.insert(0, str(PROJECT_PATH))

import tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default="Mjlab-HopperTrex-Balance-Robust-L2-v0")
  parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--settle-steps", type=int, default=20)
  parser.add_argument("--measure-steps", type=int, default=80)
  parser.add_argument(
    "--disable-reset-disturbance",
    action="store_true",
    help="Remove robust reset pose/velocity disturbance for cleaner action-response tests.",
  )
  parser.add_argument(
    "--actions",
    type=float,
    nargs="+",
    default=(-0.8, -0.4, 0.0, 0.4, 0.8),
    help="Wheel balance action values to test.",
  )
  return parser.parse_args()


def _run_action(env: ManagerBasedRlEnv, value: float, settle_steps: int, measure_steps: int) -> None:
  obs, _ = env.reset()
  del obs

  action_shape = env.action_space.shape
  if len(action_shape) != 2:
    raise ValueError(f"Expected batched action space shape, got {action_shape}")

  actions = torch.zeros(action_shape, device=env.device)
  actions[:, 0] = value

  lin_xs: list[torch.Tensor] = []
  pitch_proxies: list[torch.Tensor] = []
  pitch_rates: list[torch.Tensor] = []
  terminated_total = 0
  timeout_total = 0
  wheel_left = torch.zeros(env.num_envs, device=env.device)
  wheel_right = torch.zeros(env.num_envs, device=env.device)

  for step in range(settle_steps + measure_steps):
    _, _rewards, terminated, time_outs, _ = env.step(actions)
    robot_data = env.scene["robot"].data
    wheel_action = env.action_manager.get_term("wheel_balance")
    wheel_target = wheel_action._processed_actions.detach()
    wheel_left = wheel_target[:, 0]
    wheel_right = wheel_target[:, 1]
    terminated_total += int(terminated.sum().item())
    timeout_total += int(time_outs.sum().item())
    if step >= settle_steps:
      projected_gravity = robot_data.projected_gravity_b.detach()
      pitch_proxy = torch.atan2(
        projected_gravity[:, 0],
        torch.clamp(-projected_gravity[:, 2], min=1.0e-6),
      )
      lin_xs.append(robot_data.root_link_lin_vel_b[:, 0].detach().cpu())
      pitch_proxies.append(pitch_proxy.detach().cpu())
      pitch_rates.append(robot_data.root_link_ang_vel_b[:, 1].detach().cpu())

  lin_x = torch.cat(lin_xs)
  pitch_proxy = torch.cat(pitch_proxies)
  pitch_rate = torch.cat(pitch_rates)
  print(
    f"action={value:+.3f} "
    f"left_tgt={wheel_left.mean().item():+.3f} "
    f"right_tgt={wheel_right.mean().item():+.3f} "
    f"mean_lin_x={lin_x.mean().item():+.5f} "
    f"p05_lin_x={torch.quantile(lin_x, 0.05).item():+.5f} "
    f"p95_lin_x={torch.quantile(lin_x, 0.95).item():+.5f} "
    f"p95_pitch={torch.quantile(pitch_proxy.abs(), 0.95).item():+.5f} "
    f"mean_pitch_rate={pitch_rate.mean().item():+.5f} "
    f"p95_abs_pitch_rate={torch.quantile(pitch_rate.abs(), 0.95).item():+.5f} "
    f"terminated={terminated_total} "
    f"timeouts={timeout_total}"
  )


def main() -> None:
  args = parse_args()
  configure_torch_backends()

  cfg = load_env_cfg(args.task)
  cfg.scene.num_envs = args.num_envs
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.num_envs = args.num_envs
  if args.disable_reset_disturbance:
    cfg.events.pop("reset_root_state_with_small_disturbance", None)

  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    print(f"Task: {args.task}")
    print(f"Action space: {env.action_space}")
    for value in args.actions:
      _run_action(env, value, args.settle_steps, args.measure_steps)
  finally:
    env.close()


if __name__ == "__main__":
  main()
