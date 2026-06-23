#!/usr/bin/env python3
"""Run random HopperTrex actions through the MjLab environment."""

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
  parser.add_argument("--steps", type=int, default=200)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = args.num_envs
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.num_envs = args.num_envs

  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    env.reset()
    for step in range(args.steps):
      action = 2.0 * torch.rand(env.action_space.shape, device=env.device) - 1.0
      _, reward, terminated, time_outs, _ = env.step(action)
      if step == 0 or (step + 1) % 50 == 0:
        print(
          f"[INFO] step={step + 1} reward_mean={reward.mean().item():.4f} "
          f"terminated={int(terminated.sum().item())} "
          f"timeouts={int(time_outs.sum().item())}"
        )
  finally:
    env.close()


if __name__ == "__main__":
  main()

