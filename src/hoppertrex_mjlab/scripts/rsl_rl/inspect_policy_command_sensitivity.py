#!/usr/bin/env python3
"""Inspect how a trained policy action changes when command observations change."""

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


DEFAULT_TASK = "Mjlab-HopperTrex-Balance-SlowSpeed-Easy-BackwardOnly-LinSign-ObsScale-v0"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("task", nargs="?", default=DEFAULT_TASK)
  parser.add_argument("--checkpoint-file", required=True)
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  parser.add_argument(
    "--cmd-lin-x",
    type=float,
    nargs="+",
    default=(-0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08),
    help="Unscaled command x values to inject into observation.",
  )
  parser.add_argument(
    "--cmd-x-obs-index",
    type=int,
    default=9,
    help="Actor observation index for scaled command x. Current layout: base lin 0:3, ang 3:6, gravity 6:9, command x 9.",
  )
  parser.add_argument(
    "--cmd-x-scale",
    type=float,
    default=20.0,
    help="Observation scale applied to command x in slow-speed obs-scale tasks.",
  )
  return parser.parse_args()


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
    obs = wrapped.get_observations()
    base_obs = obs.clone()

    print(f"Task: {args.task}")
    print(f"Checkpoint: {checkpoint}")
    print(
      "cmd_lin_x scaled_cmd_x mean_action p05_action p95_action "
      "positive_frac negative_frac"
    )
    for cmd_lin_x in args.cmd_lin_x:
      test_obs = base_obs.clone()
      scaled = cmd_lin_x * args.cmd_x_scale
      test_obs[:, args.cmd_x_obs_index] = scaled
      with torch.no_grad():
        action = policy(test_obs).detach()[:, 0].cpu()
      print(
        f"{cmd_lin_x:+.5f} "
        f"{scaled:+.5f} "
        f"{action.mean().item():+.5f} "
        f"{torch.quantile(action, 0.05).item():+.5f} "
        f"{torch.quantile(action, 0.95).item():+.5f} "
        f"{(action > 0.0).float().mean().item():.3f} "
        f"{(action < 0.0).float().mean().item():.3f}"
      )
  finally:
    wrapped.close()


if __name__ == "__main__":
  main()
