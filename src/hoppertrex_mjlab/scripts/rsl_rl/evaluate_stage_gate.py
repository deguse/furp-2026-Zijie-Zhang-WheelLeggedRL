#!/usr/bin/env python3
"""Evaluate a scratch curriculum checkpoint with stage-specific gates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

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


STAGE_TASKS = {
  0: "Mjlab-HopperTrex-Scratch-Stage0-Balance-v0",
  1: "Mjlab-HopperTrex-Scratch-Stage1-SmallForward-v0",
  2: "Mjlab-HopperTrex-Scratch-Stage2-BidirLin-v0",
  3: "Mjlab-HopperTrex-Scratch-Stage3-YawOnly-v0",
  4: "Mjlab-HopperTrex-Scratch-Stage4-SmallLinSmallYaw-v0",
  5: "Mjlab-HopperTrex-Scratch-Stage5-FullLinFullYaw-v0",
  6: "Mjlab-HopperTrex-Scratch-Stage6-PushNoise-v0",
  8: "Mjlab-HopperTrex-Scratch-Stage8-LegAssistSafe-v0",
}

DEG = math.pi / 180.0
WHEEL_TARGET_SATURATION = 23.9


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--stage", type=int, required=True, choices=(0, 1, 2, 3, 4, 5, 6, 8))
  parser.add_argument("--task", default=None, help="Override the default scratch task for the stage.")
  parser.add_argument("--checkpoint-file", required=True)
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--lin-deadband", type=float, default=0.01)
  parser.add_argument("--yaw-deadband", type=float, default=0.01)
  parser.add_argument("--safe-pitch-abs", type=float, default=0.08)
  parser.add_argument("--safe-pitch-rate-abs", type=float, default=0.8)
  parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
  return parser.parse_args()


def _safe_mean(value: torch.Tensor) -> float:
  return value.mean().item() if value.numel() else float("nan")


def _safe_rms(value: torch.Tensor) -> float:
  return torch.sqrt(torch.mean(torch.square(value))).item() if value.numel() else float("nan")


def _safe_quantile(value: torch.Tensor, q: float) -> float:
  return torch.quantile(value, q).item() if value.numel() else float("nan")


def _safe_frac(mask: torch.Tensor) -> float:
  return mask.float().mean().item() if mask.numel() else float("nan")


def _collect_rollout(
  task: str,
  checkpoint: Path,
  *,
  num_envs: int,
  steps: int,
  device: str,
) -> dict[str, torch.Tensor | float | int]:
  env_cfg = load_env_cfg(task)
  agent_cfg = load_rl_cfg(task)
  env_cfg.scene.num_envs = num_envs
  if env_cfg.scene.terrain is not None:
    env_cfg.scene.terrain.num_envs = num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)

  cmd_lin_xs: list[torch.Tensor] = []
  cmd_yaws: list[torch.Tensor] = []
  actual_lin_xs: list[torch.Tensor] = []
  actual_yaws: list[torch.Tensor] = []
  pitch_proxies: list[torch.Tensor] = []
  pitch_rates: list[torch.Tensor] = []
  wheel_target_abses: list[torch.Tensor] = []
  delta_wheel_target_abses: list[torch.Tensor] = []
  wheel_force_abses: list[torch.Tensor] = []
  raw_leg_action_abses: list[torch.Tensor] = []
  done_events = 0
  terminated_events = 0
  timeout_events = 0
  prev_wheel_target: torch.Tensor | None = None

  try:
    obs = wrapped.get_observations()
    robot = wrapped.unwrapped.scene["robot"]
    has_leg_assist = "leg_assist_pos" in wrapped.unwrapped.action_manager.active_terms
    wheel_joint_ids = torch.tensor(
      [list(robot.joint_names).index(name) for name in WHEEL_JOINT_NAMES],
      device=device,
      dtype=torch.long,
    )

    for _ in range(steps):
      with torch.no_grad():
        cmd = wrapped.unwrapped.command_manager.get_command("twist").detach()
        actions = policy(obs).detach()
        obs, _rew, done, _extras = wrapped.step(actions)
        done_mask = done.to(dtype=torch.bool)
        terminated_mask = wrapped.unwrapped.reset_terminated.to(dtype=torch.bool)
        timeout_mask = wrapped.unwrapped.reset_time_outs.to(dtype=torch.bool)
        done_events += int(done_mask.sum().item())
        terminated_events += int(terminated_mask.sum().item())
        timeout_events += int(timeout_mask.sum().item())

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
        pitch_proxy = torch.atan2(
          projected_gravity[:, 0],
          torch.clamp(-projected_gravity[:, 2], min=1.0e-6),
        )
        wheel_force_abs = torch.mean(
          torch.abs(robot_data.qfrc_actuator[:, wheel_joint_ids]),
          dim=1,
        )

        if has_leg_assist:
          leg_action = wrapped.unwrapped.action_manager.get_term("leg_assist_pos")
          raw_leg_action_abs = torch.mean(torch.abs(leg_action._raw_actions.detach()), dim=1)
        else:
          raw_leg_action_abs = torch.zeros_like(pitch_proxy)

      cmd_lin_xs.append(cmd[:, 0].detach().cpu())
      cmd_yaws.append(cmd[:, 2].detach().cpu())
      actual_lin_xs.append(robot_data.root_link_lin_vel_b[:, 0].detach().cpu())
      actual_yaws.append(robot_data.root_link_ang_vel_b[:, 2].detach().cpu())
      pitch_proxies.append(pitch_proxy.detach().cpu())
      pitch_rates.append(robot_data.root_link_ang_vel_b[:, 1].detach().cpu())
      wheel_target_abses.append(torch.mean(torch.abs(wheel_target), dim=1).detach().cpu())
      delta_wheel_target_abses.append(
        torch.mean(torch.abs(delta_wheel_target), dim=1).detach().cpu()
      )
      wheel_force_abses.append(wheel_force_abs.detach().cpu())
      raw_leg_action_abses.append(raw_leg_action_abs.detach().cpu())
  finally:
    wrapped.close()

  return {
    "cmd_lin_x": torch.cat(cmd_lin_xs),
    "cmd_yaw": torch.cat(cmd_yaws),
    "actual_lin_x": torch.cat(actual_lin_xs),
    "actual_yaw": torch.cat(actual_yaws),
    "pitch_proxy": torch.cat(pitch_proxies),
    "pitch_rate": torch.cat(pitch_rates),
    "wheel_target_abs": torch.cat(wheel_target_abses),
    "delta_wheel_target_abs": torch.cat(delta_wheel_target_abses),
    "wheel_force_abs": torch.cat(wheel_force_abses),
    "raw_leg_action_abs": torch.cat(raw_leg_action_abses),
    "done_events": done_events,
    "terminated_events": terminated_events,
    "timeout_events": timeout_events,
    "num_envs": num_envs,
    "steps": steps,
  }


def _metrics(
  data: dict[str, torch.Tensor | float | int],
  lin_deadband: float,
  yaw_deadband: float,
  safe_pitch_abs: float,
  safe_pitch_rate_abs: float,
) -> dict[str, float]:
  cmd_lin_x = data["cmd_lin_x"]
  cmd_yaw = data["cmd_yaw"]
  actual_lin_x = data["actual_lin_x"]
  actual_yaw = data["actual_yaw"]
  pitch_proxy = data["pitch_proxy"]
  pitch_rate = data["pitch_rate"]
  wheel_target_abs = data["wheel_target_abs"]
  delta_wheel_target_abs = data["delta_wheel_target_abs"]
  wheel_force_abs = data["wheel_force_abs"]
  raw_leg_action_abs = data["raw_leg_action_abs"]
  assert isinstance(cmd_lin_x, torch.Tensor)
  assert isinstance(cmd_yaw, torch.Tensor)
  assert isinstance(actual_lin_x, torch.Tensor)
  assert isinstance(actual_yaw, torch.Tensor)
  assert isinstance(pitch_proxy, torch.Tensor)
  assert isinstance(pitch_rate, torch.Tensor)
  assert isinstance(wheel_target_abs, torch.Tensor)
  assert isinstance(delta_wheel_target_abs, torch.Tensor)
  assert isinstance(wheel_force_abs, torch.Tensor)
  assert isinstance(raw_leg_action_abs, torch.Tensor)

  lin_active = cmd_lin_x.abs() > lin_deadband
  forward = cmd_lin_x > lin_deadband
  backward = cmd_lin_x < -lin_deadband
  yaw_active = cmd_yaw.abs() > yaw_deadband
  yaw_left = cmd_yaw > yaw_deadband
  yaw_right = cmd_yaw < -yaw_deadband
  zero_cmd = (cmd_lin_x.abs() <= lin_deadband) & (cmd_yaw.abs() <= yaw_deadband)
  zero_lin = cmd_lin_x.abs() <= lin_deadband
  safe_posture = (
    (pitch_proxy.abs() < safe_pitch_abs)
    & (pitch_rate.abs() < safe_pitch_rate_abs)
  )
  unsafe_forward = forward & (~safe_posture) & (actual_lin_x > lin_deadband)

  lin_error = actual_lin_x - cmd_lin_x
  yaw_error = actual_yaw - cmd_yaw
  done_events = int(data["done_events"])
  terminated_events = int(data["terminated_events"])
  timeout_events = int(data["timeout_events"])
  num_envs = int(data["num_envs"])
  steps = int(data["steps"])

  return {
    "samples": float(cmd_lin_x.numel()),
    "done_events": float(done_events),
    "terminated_events": float(terminated_events),
    "timeout_events": float(timeout_events),
    "fall_event_rate": terminated_events / max(num_envs, 1),
    "timeout_event_rate": timeout_events / max(num_envs, 1),
    "done_sample_rate": done_events / max(num_envs * steps, 1),
    "terminated_sample_rate": terminated_events / max(num_envs * steps, 1),
    "timeout_sample_rate": timeout_events / max(num_envs * steps, 1),
    "mean_actual_lin_x": actual_lin_x.mean().item(),
    "mean_actual_lin_x_backward": _safe_mean(actual_lin_x[backward]),
    "mean_actual_lin_x_forward": _safe_mean(actual_lin_x[forward]),
    "drift_when_zero_cmd": _safe_mean(actual_lin_x[zero_cmd].abs()),
    "lin_drift_when_zero_lin": _safe_mean(actual_lin_x[zero_lin].abs()),
    "unsafe_forward_ratio": _safe_frac(unsafe_forward[forward]),
    "lin_sign_match": _safe_frac((cmd_lin_x[lin_active] * actual_lin_x[lin_active]) > 0.0),
    "forward_sign_match": _safe_frac((cmd_lin_x[forward] * actual_lin_x[forward]) > 0.0),
    "backward_sign_match": _safe_frac((cmd_lin_x[backward] * actual_lin_x[backward]) > 0.0),
    "yaw_sign_match": _safe_frac((cmd_yaw[yaw_active] * actual_yaw[yaw_active]) > 0.0),
    "yaw_left_sign_match": _safe_frac((cmd_yaw[yaw_left] * actual_yaw[yaw_left]) > 0.0),
    "yaw_right_sign_match": _safe_frac((cmd_yaw[yaw_right] * actual_yaw[yaw_right]) > 0.0),
    "lin_abs_error_mean": _safe_mean(lin_error[lin_active].abs()),
    "lin_abs_error_p50": _safe_quantile(lin_error[lin_active].abs(), 0.50),
    "lin_abs_error_p90": _safe_quantile(lin_error[lin_active].abs(), 0.90),
    "yaw_abs_error_mean": _safe_mean(yaw_error[yaw_active].abs()),
    "yaw_abs_error_p50": _safe_quantile(yaw_error[yaw_active].abs(), 0.50),
    "yaw_abs_error_p90": _safe_quantile(yaw_error[yaw_active].abs(), 0.90),
    "pitch_rms": _safe_rms(pitch_proxy),
    "pitch_abs_p95": _safe_quantile(pitch_proxy.abs(), 0.95),
    "pitch_rate_abs_p95": _safe_quantile(pitch_rate.abs(), 0.95),
    "wheel_saturation_ratio": _safe_frac(wheel_target_abs >= WHEEL_TARGET_SATURATION),
    "wheel_target_rate_rms": _safe_rms(delta_wheel_target_abs),
    "wheel_force_abs_mean": wheel_force_abs.mean().item(),
    "raw_leg_abs_mean": raw_leg_action_abs.mean().item(),
    "raw_leg_abs_p95": _safe_quantile(raw_leg_action_abs, 0.95),
    "n_forward": float(forward.sum().item()),
    "n_backward": float(backward.sum().item()),
    "n_yaw_left": float(yaw_left.sum().item()),
    "n_yaw_right": float(yaw_right.sum().item()),
    "n_zero_cmd": float(zero_cmd.sum().item()),
  }


def _is_number(value: float) -> bool:
  return not math.isnan(value)


def _le(metrics: dict[str, float], name: str, limit: float) -> tuple[bool, str]:
  value = metrics[name]
  passed = _is_number(value) and value <= limit
  return passed, f"{name}={value:.5f} <= {limit:.5f}"


def _ge(metrics: dict[str, float], name: str, limit: float) -> tuple[bool, str]:
  value = metrics[name]
  passed = _is_number(value) and value >= limit
  return passed, f"{name}={value:.5f} >= {limit:.5f}"


def _lt(metrics: dict[str, float], name: str, limit: float) -> tuple[bool, str]:
  value = metrics[name]
  passed = _is_number(value) and value < limit
  return passed, f"{name}={value:.5f} < {limit:.5f}"


def _stage_checks(stage: int, metrics: dict[str, float]) -> list[tuple[bool, str]]:
  common_safety = [
    _le(metrics, "fall_event_rate", 0.05 if stage >= 6 else 0.01),
    _le(metrics, "wheel_saturation_ratio", 0.30 if stage >= 6 else 0.20),
  ]

  if stage == 0:
    return [
      _le(metrics, "fall_event_rate", 0.01),
      _le(metrics, "pitch_rms", 5.0 * DEG),
      _le(metrics, "pitch_abs_p95", 10.0 * DEG),
      _le(metrics, "drift_when_zero_cmd", 0.05),
      _le(metrics, "wheel_saturation_ratio", 0.10),
    ]
  if stage == 1:
    checks = common_safety + [
      _ge(metrics, "forward_sign_match", 0.85),
      _ge(metrics, "mean_actual_lin_x_forward", 0.0),
      _le(metrics, "lin_abs_error_mean", 0.07),
      _le(metrics, "pitch_abs_p95", 0.20),
      _le(metrics, "unsafe_forward_ratio", 0.15),
    ]
    if _is_number(metrics["drift_when_zero_cmd"]):
      checks.append(_le(metrics, "drift_when_zero_cmd", 0.055))
    return checks
  if stage == 2:
    return common_safety + [
      _ge(metrics, "lin_sign_match", 0.90),
      _ge(metrics, "backward_sign_match", 0.85),
      _lt(metrics, "mean_actual_lin_x_backward", 0.0),
      _le(metrics, "lin_abs_error_mean", 0.07),
      _le(metrics, "lin_abs_error_p90", 0.14),
      _le(metrics, "pitch_abs_p95", 0.22),
    ]
  if stage == 3:
    return common_safety + [
      _ge(metrics, "yaw_sign_match", 0.90),
      _le(metrics, "yaw_abs_error_mean", 0.07),
      _le(metrics, "lin_drift_when_zero_lin", 0.05),
      _le(metrics, "pitch_abs_p95", 0.22),
    ]
  if stage == 4:
    return common_safety + [
      _ge(metrics, "lin_sign_match", 0.88),
      _ge(metrics, "backward_sign_match", 0.85),
      _ge(metrics, "yaw_sign_match", 0.85),
      _lt(metrics, "mean_actual_lin_x_backward", 0.0),
      _le(metrics, "pitch_abs_p95", 0.25),
    ]
  if stage == 5:
    return common_safety + [
      _ge(metrics, "lin_sign_match", 0.90),
      _ge(metrics, "backward_sign_match", 0.88),
      _ge(metrics, "yaw_sign_match", 0.88),
      _lt(metrics, "mean_actual_lin_x_backward", 0.0),
      _le(metrics, "lin_abs_error_p90", 0.16),
      _le(metrics, "yaw_abs_error_p90", 0.16),
      _le(metrics, "pitch_abs_p95", 0.30),
      _le(metrics, "wheel_saturation_ratio", 0.25),
    ]
  if stage == 6:
    return common_safety + [
      _ge(metrics, "lin_sign_match", 0.85),
      _ge(metrics, "backward_sign_match", 0.80),
      _ge(metrics, "yaw_sign_match", 0.80),
      _lt(metrics, "mean_actual_lin_x_backward", 0.0),
      _le(metrics, "pitch_abs_p95", 0.35),
    ]
  if stage == 8:
    return common_safety + [
      _ge(metrics, "lin_sign_match", 0.90),
      _ge(metrics, "backward_sign_match", 0.85),
      _lt(metrics, "mean_actual_lin_x_backward", 0.0),
      _le(metrics, "raw_leg_abs_mean", 0.25),
      _le(metrics, "raw_leg_abs_p95", 0.60),
    ]
  raise ValueError(f"Unsupported stage: {stage}")


def _soft_score(stage: int, metrics: dict[str, float]) -> float:
  score = 0.0
  score += 2.0 * metrics["done_sample_rate"]
  score += 0.5 * metrics["wheel_saturation_ratio"]
  score += 0.5 * metrics["pitch_rms"]
  if _is_number(metrics["lin_abs_error_mean"]):
    score += metrics["lin_abs_error_mean"]
  if _is_number(metrics["yaw_abs_error_mean"]):
    score += metrics["yaw_abs_error_mean"]
  if _is_number(metrics["lin_sign_match"]):
    score += max(0.0, 1.0 - metrics["lin_sign_match"])
  if _is_number(metrics["backward_sign_match"]):
    score += 1.5 * max(0.0, 1.0 - metrics["backward_sign_match"])
  if _is_number(metrics["yaw_sign_match"]):
    score += max(0.0, 1.0 - metrics["yaw_sign_match"])
  if stage == 8:
    score += 0.25 * metrics["raw_leg_abs_mean"]
  return score


def main() -> None:
  args = parse_args()
  configure_torch_backends()

  checkpoint = Path(args.checkpoint_file)
  if not checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

  task = args.task or STAGE_TASKS[args.stage]
  data = _collect_rollout(
    task,
    checkpoint,
    num_envs=args.num_envs,
    steps=args.steps,
    device=args.device,
  )
  metrics = _metrics(
    data,
    args.lin_deadband,
    args.yaw_deadband,
    args.safe_pitch_abs,
    args.safe_pitch_rate_abs,
  )
  checks = _stage_checks(args.stage, metrics)
  gate_pass = all(passed for passed, _ in checks)
  soft_score = _soft_score(args.stage, metrics)
  result: dict[str, Any] = {
    "stage": args.stage,
    "task": task,
    "checkpoint": str(checkpoint),
    "gate_pass": gate_pass,
    "soft_score": soft_score,
    "metrics": metrics,
    "checks": [
      {"pass": passed, "detail": detail}
      for passed, detail in checks
    ],
  }

  if args.json:
    print(json.dumps(result, indent=2, sort_keys=True))
    return

  print(f"Stage: {args.stage}")
  print(f"Task: {task}")
  print(f"Checkpoint: {checkpoint}")
  print(f"Gate: {'PASS' if gate_pass else 'FAIL'}")
  print(f"Soft score: {soft_score:.5f}  (lower is better)")
  print("\nHard checks:")
  for passed, detail in checks:
    print(f"  [{'PASS' if passed else 'FAIL'}] {detail}")
  print("\nKey metrics:")
  for name in (
    "fall_event_rate",
    "timeout_event_rate",
    "done_sample_rate",
    "terminated_sample_rate",
    "timeout_sample_rate",
    "lin_sign_match",
    "forward_sign_match",
    "backward_sign_match",
    "yaw_sign_match",
    "mean_actual_lin_x_backward",
    "lin_abs_error_mean",
    "lin_abs_error_p90",
    "yaw_abs_error_mean",
    "yaw_abs_error_p90",
    "drift_when_zero_cmd",
    "lin_drift_when_zero_lin",
    "unsafe_forward_ratio",
    "pitch_rms",
    "pitch_abs_p95",
    "wheel_saturation_ratio",
    "wheel_target_rate_rms",
    "raw_leg_abs_mean",
    "raw_leg_abs_p95",
  ):
    print(f"  {name}: {metrics[name]:.5f}")


if __name__ == "__main__":
  main()
