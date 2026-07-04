#!/usr/bin/env python3
"""Sweep slow-speed feedforward action parameters without training."""

from __future__ import annotations

import argparse
import itertools
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


DEFAULT_TASK = (
  "Mjlab-HopperTrex-Balance-SlowSpeed-Easy-BackwardOnly-LinSign-ObsScale-Strict-"
  "Feedforward-v0"
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default=DEFAULT_TASK)
  parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--settle-steps", type=int, default=20)
  parser.add_argument("--measure-steps", type=int, default=200)
  parser.add_argument("--gains", type=float, nargs="+", default=(0.4, 0.6, 0.8, 1.0, 1.2))
  parser.add_argument("--clips", type=float, nargs="+", default=(0.06, 0.08, 0.10, 0.12))
  parser.add_argument("--residual-scales", type=float, nargs="+", default=(0.0, 0.05, 0.10, 0.15))
  parser.add_argument(
    "--residual-action",
    type=float,
    default=0.0,
    help="Constant policy residual action before residual scaling.",
  )
  return parser.parse_args()


def _quantile(value: torch.Tensor, q: float) -> float:
  return torch.quantile(value, q).item()


def _run_combo(
  env: ManagerBasedRlEnv,
  *,
  gain: float,
  clip: float,
  residual_scale: float,
  residual_action: float,
  settle_steps: int,
  measure_steps: int,
) -> dict[str, float]:
  obs, _ = env.reset()
  del obs

  wheel_action = env.action_manager.get_term("wheel_balance")
  if not hasattr(wheel_action, "_command_gain"):
    raise TypeError(
      "Task wheel_balance action does not expose feedforward parameters. "
      "Use a Strict-Feedforward task."
    )
  wheel_action._command_gain = gain
  wheel_action._feedforward_clip = clip
  wheel_action._residual_scale = residual_scale

  action_shape = env.action_space.shape
  actions = torch.zeros(action_shape, device=env.device)
  actions[:, 0] = residual_action

  cmd_lin_xs: list[torch.Tensor] = []
  raw_balances: list[torch.Tensor] = []
  ff_balances: list[torch.Tensor] = []
  lin_xs: list[torch.Tensor] = []
  pitch_proxies: list[torch.Tensor] = []
  pitch_rates: list[torch.Tensor] = []
  wheel_forces: list[torch.Tensor] = []
  terminated_total = 0
  timeout_total = 0

  robot = env.scene["robot"]
  wheel_joint_ids = torch.tensor(
    [list(robot.joint_names).index(name) for name in ("wheel_left", "wheel_right")],
    device=env.device,
    dtype=torch.long,
  )

  for step in range(settle_steps + measure_steps):
    _, _rewards, terminated, time_outs, _ = env.step(actions)
    terminated_total += int(terminated.sum().item())
    timeout_total += int(time_outs.sum().item())
    if step < settle_steps:
      continue

    robot_data = robot.data
    cmd = env.command_manager.get_command("twist")
    assert cmd is not None
    projected_gravity = robot_data.projected_gravity_b.detach()
    pitch_proxy = torch.atan2(
      projected_gravity[:, 0],
      torch.clamp(-projected_gravity[:, 2], min=1.0e-6),
    )
    cmd_lin_xs.append(cmd[:, 0].detach().cpu())
    raw_balances.append(wheel_action._raw_actions[:, 0].detach().cpu())
    ff_balances.append(wheel_action._feedforward_actions[:, 0].detach().cpu())
    lin_xs.append(robot_data.root_link_lin_vel_b[:, 0].detach().cpu())
    pitch_proxies.append(pitch_proxy.detach().cpu())
    pitch_rates.append(robot_data.root_link_ang_vel_b[:, 1].detach().cpu())
    wheel_forces.append(
      torch.mean(torch.abs(robot_data.qfrc_actuator[:, wheel_joint_ids]), dim=1)
      .detach()
      .cpu()
    )

  cmd_lin_x = torch.cat(cmd_lin_xs)
  raw_balance = torch.cat(raw_balances)
  ff_balance = torch.cat(ff_balances)
  lin_x = torch.cat(lin_xs)
  pitch_proxy = torch.cat(pitch_proxies)
  pitch_rate = torch.cat(pitch_rates)
  wheel_force = torch.cat(wheel_forces)
  lin_error = lin_x - cmd_lin_x
  sign_match = (cmd_lin_x * lin_x > 0.0).float().mean().item()
  return {
    "gain": gain,
    "clip": clip,
    "residual_scale": residual_scale,
    "residual_action": residual_action,
    "mean_cmd": cmd_lin_x.mean().item(),
    "mean_raw": raw_balance.mean().item(),
    "mean_ff": ff_balance.mean().item(),
    "mean_lin_x": lin_x.mean().item(),
    "mean_abs_error": lin_error.abs().mean().item(),
    "p95_abs_error": _quantile(lin_error.abs(), 0.95),
    "sign_match": sign_match,
    "p95_pitch": _quantile(pitch_proxy.abs(), 0.95),
    "p95_pitch_rate": _quantile(pitch_rate.abs(), 0.95),
    "mean_wheel_force": wheel_force.mean().item(),
    "terminated": float(terminated_total),
    "timeouts": float(timeout_total),
  }


def _score(row: dict[str, float]) -> float:
  return (
    row["mean_abs_error"]
    + 0.25 * row["p95_abs_error"]
    + 0.20 * row["p95_pitch"]
    + 0.05 * row["p95_pitch_rate"]
    + 0.20 * max(0.0, 0.80 - row["sign_match"])
  )


def main() -> None:
  args = parse_args()
  configure_torch_backends()

  cfg = load_env_cfg(args.task)
  cfg.scene.num_envs = args.num_envs
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.num_envs = args.num_envs

  rows: list[dict[str, float]] = []
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    for gain, clip, residual_scale in itertools.product(
      args.gains,
      args.clips,
      args.residual_scales,
    ):
      row = _run_combo(
        env,
        gain=gain,
        clip=clip,
        residual_scale=residual_scale,
        residual_action=args.residual_action,
        settle_steps=args.settle_steps,
        measure_steps=args.measure_steps,
      )
      row["score"] = _score(row)
      rows.append(row)
  finally:
    env.close()

  rows.sort(key=lambda item: item["score"])
  print(
    "score gain clip residual mean_raw mean_ff mean_lin_x mean_abs_err "
    "p95_abs_err sign_match p95_pitch p95_pitch_rate mean_force term"
  )
  for row in rows:
    print(
      f"{row['score']:.4f} "
      f"{row['gain']:.3f} "
      f"{row['clip']:.3f} "
      f"{row['residual_scale']:.3f} "
      f"{row['mean_raw']:+.4f} "
      f"{row['mean_ff']:+.4f} "
      f"{row['mean_lin_x']:+.4f} "
      f"{row['mean_abs_error']:.4f} "
      f"{row['p95_abs_error']:.4f} "
      f"{row['sign_match']:.3f} "
      f"{row['p95_pitch']:.4f} "
      f"{row['p95_pitch_rate']:.4f} "
      f"{row['mean_wheel_force']:.4f} "
      f"{int(row['terminated'])}"
    )


if __name__ == "__main__":
  main()
