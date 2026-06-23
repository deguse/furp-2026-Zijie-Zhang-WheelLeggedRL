#!/usr/bin/env python3
"""Step a HopperTrex MjLab environment with zero actions."""

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


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default="Mjlab-HopperTrex-Balance-v0")
  parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--num_envs", type=int, default=1)
  parser.add_argument("--max_steps", type=int, default=200)
  parser.add_argument("--play", action="store_true", help="Use the play environment config.")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  cfg = load_env_cfg(args.task, play=args.play)
  cfg.scene.num_envs = args.num_envs
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.num_envs = args.num_envs

  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    obs, _ = env.reset()
    del obs
    print(f"[INFO] observation_space={env.observation_space}")
    print(f"[INFO] action_space={env.action_space}")
    action_shape = env.action_space.shape
    for step in range(args.max_steps):
      actions = torch.zeros(action_shape, device=env.device)
      _, rewards, terminated, time_outs, _ = env.step(actions)
      if step == 0 or (step + 1) % 50 == 0:
        print(
          f"[INFO] step={step + 1} reward_mean={rewards.mean().item():.4f} "
          f"terminated={int(terminated.sum().item())} "
          f"timeouts={int(time_outs.sum().item())}"
        )
    print(f"[INFO] Reached max_steps={args.max_steps}.")
  finally:
    env.close()


if __name__ == "__main__":
  main()

