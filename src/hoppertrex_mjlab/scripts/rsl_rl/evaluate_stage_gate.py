#!/usr/bin/env python3
"""Evaluate a scratch curriculum checkpoint with stage-specific gates."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from dataclasses import asdict
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import torch

PROJECT_PATH = Path(__file__).resolve().parents[2]
SRC_PATH = Path(__file__).resolve().parents[3]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

try:
  import hoppertrex_mjlab.tasks as tasks  # noqa: F401
except ImportError:
  import tasks  # noqa: F401
from assets.HopperTrex_CFG import WHEEL_JOINT_NAMES
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
try:
  from .evaluate_fixed_command import _run_fixed_command
  from .evaluate_fixed_yaw import _run_fixed_yaw
except ImportError:
  from scripts.rsl_rl.evaluate_fixed_command import _run_fixed_command
  from scripts.rsl_rl.evaluate_fixed_yaw import _run_fixed_yaw
try:
  from hoppertrex_mjlab.tasks.hoppertrex_balance_task import (
    CLEAN_SUPPORT_MAX_TILT_XY,
    CLEAN_SUPPORT_MIN_HEIGHT,
    NON_WHEEL_GROUND_SENSOR_NAME,
    ROOT_HEIGHT_HARD_MIN,
    WHEEL_GROUND_SENSOR_NAME,
    clean_wheel_support,
    non_wheel_ground_contact,
    wheel_ground_contact,
    wheel_support_posture,
  )
except ImportError:
  from tasks.hoppertrex_balance_task import (
    CLEAN_SUPPORT_MAX_TILT_XY,
    CLEAN_SUPPORT_MIN_HEIGHT,
    NON_WHEEL_GROUND_SENSOR_NAME,
    ROOT_HEIGHT_HARD_MIN,
    WHEEL_GROUND_SENSOR_NAME,
    clean_wheel_support,
    non_wheel_ground_contact,
    wheel_ground_contact,
    wheel_support_posture,
  )


STAGE_TASKS = {
  0: "Mjlab-HopperTrex-Scratch-Stage0-Balance-v0",
  1: "Mjlab-HopperTrex-Scratch-Stage1-SmallForward-v0",
  2: "Mjlab-HopperTrex-Scratch-Stage2-BidirLinSmoothSlew6RewardBalance-v0",
  3: "Mjlab-HopperTrex-Scratch-Stage3-YawOnlyMediumAlignedSmooth-v0",
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
  parser.add_argument(
    "--play-cfg",
    action="store_true",
    help="Use the task play config before applying num-env overrides.",
  )
  parser.add_argument(
    "--episode-length-s",
    type=float,
    default=None,
    help="Override env episode length. Use a large value for viewer-like checks.",
  )
  parser.add_argument("--lin-deadband", type=float, default=0.01)
  parser.add_argument("--yaw-deadband", type=float, default=0.01)
  parser.add_argument("--safe-pitch-abs", type=float, default=0.08)
  parser.add_argument("--safe-pitch-rate-abs", type=float, default=0.8)
  parser.add_argument(
    "--skip-fixed-command-promotion",
    action="store_true",
    help="For stage 2 debugging only: skip fixed-command promotion checks.",
  )
  parser.add_argument(
    "--skip-fixed-yaw-promotion",
    action="store_true",
    help="For stage 3 debugging only: skip fixed-yaw promotion checks.",
  )
  parser.add_argument(
    "--skip-fixed-combo-promotion",
    action="store_true",
    help="For stage 4/5 debugging only: skip fixed lin+yaw promotion checks.",
  )
  parser.add_argument(
    "--fixed-command-lin-x",
    type=float,
    nargs="+",
    default=[-0.07, 0.07],
    help="Stage 2 fixed-command velocities for promotion checks.",
  )
  parser.add_argument("--fixed-command-num-envs", type=int, default=16)
  parser.add_argument("--fixed-command-steps", type=int, default=3000)
  parser.add_argument("--fixed-command-warmup-steps", type=int, default=300)
  parser.add_argument("--fixed-command-window-steps", type=int, default=800)
  parser.add_argument("--fixed-command-progress-interval", type=int, default=500)
  parser.add_argument("--fixed-command-episode-length-s", type=float, default=1.0e9)
  parser.add_argument(
    "--fixed-yaw",
    type=float,
    nargs="+",
    default=[-0.07, 0.07],
    help="Stage 3 fixed yaw rates for promotion checks.",
  )
  parser.add_argument("--fixed-yaw-num-envs", type=int, default=16)
  parser.add_argument("--fixed-yaw-steps", type=int, default=3000)
  parser.add_argument("--fixed-yaw-warmup-steps", type=int, default=300)
  parser.add_argument("--fixed-yaw-window-steps", type=int, default=800)
  parser.add_argument("--fixed-yaw-progress-interval", type=int, default=500)
  parser.add_argument("--fixed-yaw-episode-length-s", type=float, default=1.0e9)
  parser.add_argument("--fixed-yaw-lin-drift-speed", type=float, default=0.05)
  parser.add_argument(
    "--fixed-combo-lin-x",
    type=float,
    nargs="+",
    default=[-0.05, 0.05],
    help="Stage 4/5 fixed linear velocities for combined promotion checks.",
  )
  parser.add_argument(
    "--fixed-combo-yaw",
    type=float,
    nargs="+",
    default=[-0.05, 0.05],
    help="Stage 4/5 fixed yaw rates for combined promotion checks.",
  )
  parser.add_argument("--fixed-combo-num-envs", type=int, default=16)
  parser.add_argument("--fixed-combo-steps", type=int, default=2500)
  parser.add_argument("--fixed-combo-warmup-steps", type=int, default=300)
  parser.add_argument("--fixed-combo-window-steps", type=int, default=600)
  parser.add_argument("--fixed-combo-progress-interval", type=int, default=500)
  parser.add_argument("--fixed-combo-episode-length-s", type=float, default=1.0e9)
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


def _promotion_output_context(json_output: bool):
  if json_output:
    return contextlib.redirect_stdout(sys.stderr)
  return contextlib.nullcontext()


def _collect_rollout(
  task: str,
  checkpoint: Path,
  *,
  num_envs: int,
  steps: int,
  device: str,
  play_cfg: bool,
  episode_length_s: float | None,
) -> dict[str, torch.Tensor | float | int]:
  env_cfg = load_env_cfg(task, play=play_cfg)
  agent_cfg = load_rl_cfg(task)
  if episode_length_s is not None:
    env_cfg.episode_length_s = episode_length_s
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
  clean_supports: list[torch.Tensor] = []
  wheel_contacts: list[torch.Tensor] = []
  non_wheel_contacts: list[torch.Tensor] = []
  root_heights: list[torch.Tensor] = []
  support_postures: list[torch.Tensor] = []
  done_events = 0
  terminated_events = 0
  timeout_events = 0
  termination_term_events: dict[str, int] = {}
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
        term_dones = getattr(wrapped.unwrapped.termination_manager, "_term_dones", {})
        for name, term_done in term_dones.items():
          if name not in termination_term_events:
            termination_term_events[name] = 0
          termination_term_events[name] += int(term_done.to(dtype=torch.bool).sum().item())

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
        clean_support = clean_wheel_support(
          env=wrapped.unwrapped,
          wheel_sensor_name=WHEEL_GROUND_SENSOR_NAME,
          non_wheel_sensor_name=NON_WHEEL_GROUND_SENSOR_NAME,
          minimum_height=CLEAN_SUPPORT_MIN_HEIGHT,
          max_tilt_xy=CLEAN_SUPPORT_MAX_TILT_XY,
        )
        wheel_contact = wheel_ground_contact(
          env=wrapped.unwrapped,
          sensor_name=WHEEL_GROUND_SENSOR_NAME,
        )
        non_wheel_contact = non_wheel_ground_contact(
          env=wrapped.unwrapped,
          sensor_name=NON_WHEEL_GROUND_SENSOR_NAME,
        )
        support_posture = wheel_support_posture(
          env=wrapped.unwrapped,
          wheel_sensor_name=WHEEL_GROUND_SENSOR_NAME,
          minimum_height=CLEAN_SUPPORT_MIN_HEIGHT,
          max_tilt_xy=CLEAN_SUPPORT_MAX_TILT_XY,
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
      clean_supports.append(clean_support.detach().cpu())
      wheel_contacts.append(wheel_contact.detach().cpu())
      non_wheel_contacts.append(non_wheel_contact.detach().cpu())
      root_heights.append(robot_data.root_link_pos_w[:, 2].detach().cpu())
      support_postures.append(support_posture.detach().cpu())
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
    "clean_support": torch.cat(clean_supports),
    "wheel_contact": torch.cat(wheel_contacts),
    "non_wheel_contact": torch.cat(non_wheel_contacts),
    "root_height": torch.cat(root_heights),
    "support_posture": torch.cat(support_postures),
    "done_events": done_events,
    "terminated_events": terminated_events,
    "timeout_events": timeout_events,
    "termination_term_events": termination_term_events,
    "num_envs": num_envs,
    "steps": steps,
    "episode_length_s": float(env_cfg.episode_length_s),
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
  clean_support = data["clean_support"]
  wheel_contact = data["wheel_contact"]
  non_wheel_contact = data["non_wheel_contact"]
  root_height = data["root_height"]
  support_posture = data["support_posture"]
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
  assert isinstance(clean_support, torch.Tensor)
  assert isinstance(wheel_contact, torch.Tensor)
  assert isinstance(non_wheel_contact, torch.Tensor)
  assert isinstance(root_height, torch.Tensor)
  assert isinstance(support_posture, torch.Tensor)

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
  support_safe = support_posture > 0.5
  support_posture_safe = support_safe & (pitch_rate.abs() < safe_pitch_rate_abs)
  unsafe_support_forward = (
    forward & (~support_posture_safe) & (actual_lin_x > lin_deadband)
  )

  lin_error = actual_lin_x - cmd_lin_x
  yaw_error = actual_yaw - cmd_yaw
  done_events = int(data["done_events"])
  terminated_events = int(data["terminated_events"])
  timeout_events = int(data["timeout_events"])
  termination_term_events = data["termination_term_events"]
  assert isinstance(termination_term_events, dict)
  num_envs = int(data["num_envs"])
  steps = int(data["steps"])

  metrics = {
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
    "unsafe_support_forward_ratio": _safe_frac(unsafe_support_forward[forward]),
    "clean_support_forward_frac": _safe_mean(clean_support[forward]),
    "support_posture_forward_frac": _safe_mean(support_posture[forward]),
    "non_wheel_contact_forward_frac": _safe_mean(non_wheel_contact[forward]),
    "root_height_below_clean_forward_frac": _safe_frac(
      root_height[forward] < CLEAN_SUPPORT_MIN_HEIGHT
    ),
    "root_height_below_hard_forward_frac": _safe_frac(
      root_height[forward] < ROOT_HEIGHT_HARD_MIN
    ),
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
    "pitch_abs_p99": _safe_quantile(pitch_proxy.abs(), 0.99),
    "pitch_abs_max": pitch_proxy.abs().max().item() if pitch_proxy.numel() else float("nan"),
    "pitch_rate_abs_p95": _safe_quantile(pitch_rate.abs(), 0.95),
    "pitch_rate_abs_p99": _safe_quantile(pitch_rate.abs(), 0.99),
    "pitch_rate_abs_max": (
      pitch_rate.abs().max().item() if pitch_rate.numel() else float("nan")
    ),
    "wheel_saturation_ratio": _safe_frac(wheel_target_abs >= WHEEL_TARGET_SATURATION),
    "wheel_target_rate_rms": _safe_rms(delta_wheel_target_abs),
    "wheel_force_abs_mean": wheel_force_abs.mean().item(),
    "clean_support_frac": _safe_mean(clean_support),
    "support_posture_frac": _safe_mean(support_posture),
    "wheel_contact_frac": _safe_mean(wheel_contact),
    "non_wheel_contact_frac": _safe_mean(non_wheel_contact),
    "root_height_min": root_height.min().item() if root_height.numel() else float("nan"),
    "root_height_p05": _safe_quantile(root_height, 0.05),
    "root_height_below_clean_frac": _safe_frac(root_height < CLEAN_SUPPORT_MIN_HEIGHT),
    "root_height_below_hard_frac": _safe_frac(root_height < ROOT_HEIGHT_HARD_MIN),
    "raw_leg_abs_mean": raw_leg_action_abs.mean().item(),
    "raw_leg_abs_p95": _safe_quantile(raw_leg_action_abs, 0.95),
    "n_forward": float(forward.sum().item()),
    "n_backward": float(backward.sum().item()),
    "n_yaw_left": float(yaw_left.sum().item()),
    "n_yaw_right": float(yaw_right.sum().item()),
    "n_zero_cmd": float(zero_cmd.sum().item()),
  }
  for name, count in termination_term_events.items():
    metrics[f"termination_{name}_events"] = float(count)
    metrics[f"termination_{name}_event_rate"] = count / max(num_envs, 1)
  return metrics


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


def _fixed_key(row: dict[str, float | str], metric_name: str) -> str:
  return f"fixed_{float(row['lin_x']):+.3f}_{metric_name}"


def _fixed_ge(
  row: dict[str, float | str],
  metric_name: str,
  limit: float,
) -> tuple[bool, str]:
  value = float(row[metric_name])
  passed = _is_number(value) and value >= limit
  return passed, f"{_fixed_key(row, metric_name)}={value:.5f} >= {limit:.5f}"


def _fixed_le(
  row: dict[str, float | str],
  metric_name: str,
  limit: float,
) -> tuple[bool, str]:
  value = float(row[metric_name])
  passed = _is_number(value) and value <= limit
  return passed, f"{_fixed_key(row, metric_name)}={value:.5f} <= {limit:.5f}"


def _fixed_yaw_key(row: dict[str, float | str], metric_name: str) -> str:
  return f"fixed_yaw_{float(row['yaw']):+.3f}_{metric_name}"


def _fixed_yaw_ge(
  row: dict[str, float | str],
  metric_name: str,
  limit: float,
) -> tuple[bool, str]:
  value = float(row[metric_name])
  passed = _is_number(value) and value >= limit
  return passed, f"{_fixed_yaw_key(row, metric_name)}={value:.5f} >= {limit:.5f}"


def _fixed_yaw_le(
  row: dict[str, float | str],
  metric_name: str,
  limit: float,
) -> tuple[bool, str]:
  value = float(row[metric_name])
  passed = _is_number(value) and value <= limit
  return passed, f"{_fixed_yaw_key(row, metric_name)}={value:.5f} <= {limit:.5f}"


def _fixed_yaw_signed_mean_ge_fraction(
  row: dict[str, float | str],
  fraction: float,
) -> tuple[bool, str]:
  target = float(row["yaw"])
  target_abs = abs(target)
  sign = 1.0 if target >= 0.0 else -1.0
  value = sign * float(row["mean_actual_yaw"])
  limit = fraction * target_abs
  passed = _is_number(value) and value >= limit
  return (
    passed,
    f"{_fixed_yaw_key(row, 'signed_mean_actual_yaw')}={value:.5f} >= {limit:.5f}",
  )


def _fixed_combo_key(row: dict[str, float | str], metric_name: str) -> str:
  return (
    f"combo_{float(row['lin_x']):+.3f}_{float(row['yaw']):+.3f}_{metric_name}"
  )


def _fixed_combo_ge(
  row: dict[str, float | str],
  metric_name: str,
  limit: float,
  *,
  source_metric_name: str | None = None,
) -> tuple[bool, str]:
  source_metric_name = source_metric_name or metric_name
  value = float(row[source_metric_name])
  passed = _is_number(value) and value >= limit
  return passed, f"{_fixed_combo_key(row, metric_name)}={value:.5f} >= {limit:.5f}"


def _fixed_combo_le(
  row: dict[str, float | str],
  metric_name: str,
  limit: float,
  *,
  source_metric_name: str | None = None,
) -> tuple[bool, str]:
  source_metric_name = source_metric_name or metric_name
  value = float(row[source_metric_name])
  passed = _is_number(value) and value <= limit
  return passed, f"{_fixed_combo_key(row, metric_name)}={value:.5f} <= {limit:.5f}"


def _stage2_fixed_command_checks(
  summaries: list[dict[str, float | str]],
) -> list[tuple[bool, str]]:
  checks: list[tuple[bool, str]] = []
  for row in summaries:
    checks.extend(
      [
        _fixed_ge(row, "command_match_frac", 0.90),
        _fixed_le(row, "late_slow_env_frac", 0.10),
        _fixed_le(row, "late_wrong_direction_env_frac", 0.10),
        _fixed_ge(row, "in_band_frac", 0.70),
        _fixed_le(row, "fast_frac", 0.25),
        _fixed_ge(row, "late_in_band_frac", 0.80),
        _fixed_ge(row, "target_band_frac", 0.70),
        _fixed_ge(row, "late_target_band_frac", 0.80),
        _fixed_ge(row, "signed_speed_ratio_mean", 0.75),
        _fixed_le(row, "signed_speed_ratio_mean", 1.25),
        _fixed_le(row, "lin_x_delta_rms", 0.035),
        _fixed_le(row, "lin_x_delta_abs_p95", 0.070),
        _fixed_le(row, "late_lin_x_delta_rms", 0.035),
        _fixed_le(row, "late_lin_x_delta_abs_p95", 0.070),
        _fixed_le(row, "mean_abs_error", 0.06),
        _fixed_le(row, "p95_pitch", 0.08),
        _fixed_le(row, "p99_pitch_rate", 0.90),
        _fixed_le(row, "terminated_event_rate", 0.01),
      ]
    )
  return checks


def _stage45_fixed_combo_checks(
  summaries: list[dict[str, float | str]],
) -> list[tuple[bool, str]]:
  checks: list[tuple[bool, str]] = []
  for row in summaries:
    checks.extend(
      [
        _fixed_combo_ge(row, "lin_command_match_frac", 0.85),
        _fixed_combo_le(row, "lin_wrong_direction_frac", 0.10),
        _fixed_combo_ge(row, "lin_in_band_frac", 0.70),
        _fixed_combo_le(row, "lin_fast_frac", 0.30),
        _fixed_combo_ge(row, "late_lin_in_band_frac", 0.70),
        _fixed_combo_le(row, "lin_abs_error_mean", 0.07),
        _fixed_combo_le(row, "lin_abs_error_p90", 0.12),
        _fixed_combo_le(row, "lin_x_delta_rms", 0.045),
        _fixed_combo_le(row, "lin_x_delta_abs_p95", 0.090),
        _fixed_combo_le(row, "late_lin_x_delta_rms", 0.045),
        _fixed_combo_le(row, "late_lin_x_delta_abs_p95", 0.090),
        _fixed_combo_ge(
          row,
          "yaw_command_match_frac",
          0.85,
          source_metric_name="command_match_frac",
        ),
        _fixed_combo_le(
          row,
          "yaw_wrong_direction_frac",
          0.10,
          source_metric_name="wrong_direction_frac",
        ),
        _fixed_combo_ge(
          row,
          "yaw_in_band_frac",
          0.65,
          source_metric_name="in_band_frac",
        ),
        _fixed_combo_le(
          row,
          "yaw_fast_frac",
          0.30,
          source_metric_name="fast_frac",
        ),
        _fixed_combo_ge(
          row,
          "yaw_late_in_band_frac",
          0.65,
          source_metric_name="late_in_band_frac",
        ),
        _fixed_combo_le(row, "yaw_abs_error_mean", 0.08),
        _fixed_combo_le(row, "yaw_abs_error_p90", 0.12),
        _fixed_combo_le(row, "yaw_delta_rms", 0.045),
        _fixed_combo_le(row, "yaw_delta_abs_p95", 0.090),
        _fixed_combo_le(row, "late_yaw_delta_rms", 0.045),
        _fixed_combo_le(row, "late_yaw_delta_abs_p95", 0.090),
        _fixed_combo_le(row, "p95_pitch", 0.12),
        _fixed_combo_le(row, "p99_pitch_rate", 0.95),
        _fixed_combo_le(row, "wheel_saturation_ratio", 0.20),
        _fixed_combo_le(row, "terminated_event_rate", 0.01),
      ]
    )
  return checks


def _stage3_fixed_yaw_checks(
  summaries: list[dict[str, float | str]],
) -> list[tuple[bool, str]]:
  checks: list[tuple[bool, str]] = []
  for row in summaries:
    checks.extend(
      [
        _fixed_yaw_ge(row, "command_match_frac", 0.90),
        _fixed_yaw_signed_mean_ge_fraction(row, 0.50),
        _fixed_yaw_le(row, "late_slow_env_frac", 0.10),
        _fixed_yaw_le(row, "late_wrong_direction_env_frac", 0.10),
        _fixed_yaw_le(row, "late_lin_drift_env_frac", 0.10),
        _fixed_yaw_ge(row, "in_band_frac", 0.70),
        _fixed_yaw_le(row, "fast_frac", 0.25),
        _fixed_yaw_ge(row, "late_in_band_frac", 0.70),
        _fixed_yaw_le(row, "yaw_delta_rms", 0.035),
        _fixed_yaw_le(row, "yaw_delta_abs_p95", 0.080),
        _fixed_yaw_le(row, "late_yaw_delta_rms", 0.035),
        _fixed_yaw_le(row, "late_yaw_delta_abs_p95", 0.080),
        _fixed_yaw_le(row, "yaw_abs_error_mean", 0.07),
        _fixed_yaw_le(row, "yaw_abs_error_p90", 0.10),
        _fixed_yaw_le(row, "lin_drift_abs_mean", 0.05),
        _fixed_yaw_le(row, "p95_pitch", 0.10),
        _fixed_yaw_le(row, "p99_pitch_rate", 0.90),
        _fixed_yaw_le(row, "wheel_saturation_ratio", 0.20),
        _fixed_yaw_le(row, "terminated_event_rate", 0.01),
      ]
    )
  return checks


def _collect_fixed_command_summaries(
  task: str,
  checkpoint: Path,
  *,
  lin_x_values: list[float],
  num_envs: int,
  steps: int,
  warmup_steps: int,
  window_steps: int,
  progress_interval: int,
  episode_length_s: float,
  device: str,
) -> list[dict[str, float | str]]:
  env_cfg = load_env_cfg(task, play=True)
  agent_cfg = load_rl_cfg(task)
  env_cfg.episode_length_s = episode_length_s
  env_cfg.scene.num_envs = num_envs
  if env_cfg.scene.terrain is not None:
    env_cfg.scene.terrain.num_envs = num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  summaries: list[dict[str, float | str]] = []
  try:
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(
      str(checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    policy = runner.get_inference_policy(device=device)
    fixed_args = SimpleNamespace(
      task=task,
      yaw=0.0,
      num_envs=num_envs,
      steps=steps,
      warmup_steps=warmup_steps,
      window_steps=window_steps,
      device=device,
      stuck_speed=0.01,
      reverse_speed=-0.01,
      progress_interval=progress_interval,
      play_cfg=True,
      episode_length_s=episode_length_s,
      constant_action=None,
    )
    for lin_x_cmd in lin_x_values:
      summaries.append(
        _run_fixed_command(
          wrapped=wrapped,
          policy=policy,
          args=fixed_args,
          lin_x_cmd=lin_x_cmd,
        )
      )
  finally:
    wrapped.close()
  return summaries


def _collect_fixed_yaw_summaries(
  task: str,
  checkpoint: Path,
  *,
  lin_x_values: list[float],
  yaw_values: list[float],
  num_envs: int,
  steps: int,
  warmup_steps: int,
  window_steps: int,
  progress_interval: int,
  episode_length_s: float,
  lin_drift_speed: float,
  device: str,
) -> list[dict[str, float | str]]:
  env_cfg = load_env_cfg(task, play=True)
  agent_cfg = load_rl_cfg(task)
  env_cfg.episode_length_s = episode_length_s
  env_cfg.scene.num_envs = num_envs
  if env_cfg.scene.terrain is not None:
    env_cfg.scene.terrain.num_envs = num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  summaries: list[dict[str, float | str]] = []
  try:
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(
      str(checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    policy = runner.get_inference_policy(device=device)
    for lin_x_cmd in lin_x_values:
      fixed_args = SimpleNamespace(
        task=task,
        lin_x=lin_x_cmd,
        num_envs=num_envs,
        steps=steps,
        warmup_steps=warmup_steps,
        window_steps=window_steps,
        device=device,
        yaw_deadband=0.01,
        lin_deadband=0.01,
        lin_drift_speed=lin_drift_speed,
        progress_interval=progress_interval,
        play_cfg=True,
        episode_length_s=episode_length_s,
      )
      for yaw_cmd in yaw_values:
        summaries.append(
          _run_fixed_yaw(
            wrapped=wrapped,
            policy=policy,
            args=fixed_args,
            yaw_cmd=yaw_cmd,
          )
        )
  finally:
    wrapped.close()
  return summaries


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
      _le(metrics, "pitch_abs_p99", 0.16),
      _le(metrics, "pitch_rate_abs_p99", 1.00),
      _le(metrics, "wheel_target_rate_rms", 12.0),
      _le(metrics, "unsafe_forward_ratio", 0.15),
      _le(metrics, "unsafe_support_forward_ratio", 0.15),
      _ge(metrics, "support_posture_frac", 0.85),
      _le(metrics, "root_height_below_hard_frac", 0.01),
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
    play_cfg=args.play_cfg,
    episode_length_s=args.episode_length_s,
  )
  metrics = _metrics(
    data,
    args.lin_deadband,
    args.yaw_deadband,
    args.safe_pitch_abs,
    args.safe_pitch_rate_abs,
  )
  checks = _stage_checks(args.stage, metrics)
  fixed_summaries: list[dict[str, float | str]] = []
  fixed_checks: list[tuple[bool, str]] = []
  fixed_yaw_summaries: list[dict[str, float | str]] = []
  fixed_yaw_checks: list[tuple[bool, str]] = []
  fixed_combo_summaries: list[dict[str, float | str]] = []
  fixed_combo_checks: list[tuple[bool, str]] = []
  if args.stage == 2 and not args.skip_fixed_command_promotion:
    with _promotion_output_context(args.json):
      fixed_summaries = _collect_fixed_command_summaries(
        task,
        checkpoint,
        lin_x_values=args.fixed_command_lin_x,
        num_envs=args.fixed_command_num_envs,
        steps=args.fixed_command_steps,
        warmup_steps=args.fixed_command_warmup_steps,
        window_steps=args.fixed_command_window_steps,
        progress_interval=args.fixed_command_progress_interval,
        episode_length_s=args.fixed_command_episode_length_s,
        device=args.device,
      )
    fixed_checks = _stage2_fixed_command_checks(fixed_summaries)
    checks.extend(fixed_checks)
  if args.stage == 3 and not args.skip_fixed_yaw_promotion:
    with _promotion_output_context(args.json):
      fixed_yaw_summaries = _collect_fixed_yaw_summaries(
        task,
        checkpoint,
        lin_x_values=[0.0],
        yaw_values=args.fixed_yaw,
        num_envs=args.fixed_yaw_num_envs,
        steps=args.fixed_yaw_steps,
        warmup_steps=args.fixed_yaw_warmup_steps,
        window_steps=args.fixed_yaw_window_steps,
        progress_interval=args.fixed_yaw_progress_interval,
        episode_length_s=args.fixed_yaw_episode_length_s,
        lin_drift_speed=args.fixed_yaw_lin_drift_speed,
        device=args.device,
      )
    fixed_yaw_checks = _stage3_fixed_yaw_checks(fixed_yaw_summaries)
    checks.extend(fixed_yaw_checks)
  if args.stage in (4, 5) and not args.skip_fixed_combo_promotion:
    with _promotion_output_context(args.json):
      fixed_combo_summaries = _collect_fixed_yaw_summaries(
        task,
        checkpoint,
        lin_x_values=args.fixed_combo_lin_x,
        yaw_values=args.fixed_combo_yaw,
        num_envs=args.fixed_combo_num_envs,
        steps=args.fixed_combo_steps,
        warmup_steps=args.fixed_combo_warmup_steps,
        window_steps=args.fixed_combo_window_steps,
        progress_interval=args.fixed_combo_progress_interval,
        episode_length_s=args.fixed_combo_episode_length_s,
        lin_drift_speed=args.fixed_yaw_lin_drift_speed,
        device=args.device,
      )
    fixed_combo_checks = _stage45_fixed_combo_checks(fixed_combo_summaries)
    checks.extend(fixed_combo_checks)
  gate_pass = all(passed for passed, _ in checks)
  soft_score = _soft_score(args.stage, metrics)
  result: dict[str, Any] = {
    "stage": args.stage,
    "task": task,
    "checkpoint": str(checkpoint),
    "play_cfg": args.play_cfg,
    "episode_length_s": data["episode_length_s"],
    "gate_pass": gate_pass,
    "soft_score": soft_score,
    "metrics": metrics,
    "fixed_command_summaries": fixed_summaries,
    "fixed_yaw_summaries": fixed_yaw_summaries,
    "fixed_combo_summaries": fixed_combo_summaries,
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
  print(f"Play cfg: {args.play_cfg}")
  print(f"Episode length s: {data['episode_length_s']:.5g}")
  print(f"Gate: {'PASS' if gate_pass else 'FAIL'}")
  print(f"Soft score: {soft_score:.5f}  (lower is better)")
  print("\nHard checks:")
  for passed, detail in checks:
    print(f"  [{'PASS' if passed else 'FAIL'}] {detail}")
  if args.stage == 2 and args.skip_fixed_command_promotion:
    print("\n[WARN] Stage 2 fixed-command promotion checks were skipped.")
  if args.stage == 3 and args.skip_fixed_yaw_promotion:
    print("\n[WARN] Stage 3 fixed-yaw promotion checks were skipped.")
  if args.stage in (4, 5) and args.skip_fixed_combo_promotion:
    print("\n[WARN] Stage 4/5 fixed lin+yaw promotion checks were skipped.")
  elif fixed_summaries:
    print("\nFixed-command promotion summaries:")
    print(
      "  lin_x mean ratio match wrong slow in_band target_band fast "
      "late_slow_env late_in_band late_target_band late_wrong_env mean_abs_err "
      "p95_pitch p99_pitch_rate action_delta lin_delta lin_delta_p95 term"
    )
    for row in fixed_summaries:
      print(
        f"  {row['lin_x']:+.3f} {row['mean_actual_lin_x']:+.4f} "
        f"{row['signed_speed_ratio_mean']:.3f} "
        f"{row['command_match_frac']:.3f} {row['wrong_direction_frac']:.3f} "
        f"{row['slow_frac']:.3f} {row['in_band_frac']:.3f} "
        f"{row['target_band_frac']:.3f} {row['fast_frac']:.3f} "
        f"{row['late_slow_env_frac']:.3f} {row['late_in_band_frac']:.3f} "
        f"{row['late_target_band_frac']:.3f} "
        f"{row['late_wrong_direction_env_frac']:.3f} "
        f"{row['mean_abs_error']:.4f} {row['p95_pitch']:.4f} "
        f"{row['p99_pitch_rate']:.4f} {row['action_delta_rms']:.4f} "
        f"{row['lin_x_delta_rms']:.4f} "
        f"{row['lin_x_delta_abs_p95']:.4f} "
        f"{row['terminated_event_rate']:.3f}"
      )
  if fixed_yaw_summaries:
    print("\nFixed-yaw promotion summaries:")
    print(
      "  yaw mean match wrong slow in_band fast late_slow_env late_in_band "
      "late_wrong_env late_lin_drift_env yaw_abs_err yaw_p90_abs_err "
      "lin_drift p95_pitch p99_pitch_rate wheel_sat yaw_delta "
      "yaw_delta_p95 term"
    )
    for row in fixed_yaw_summaries:
      print(
        f"  {row['yaw']:+.3f} {row['mean_actual_yaw']:+.4f} "
        f"{row['command_match_frac']:.3f} {row['wrong_direction_frac']:.3f} "
        f"{row['slow_frac']:.3f} {row['in_band_frac']:.3f} "
        f"{row['fast_frac']:.3f} {row['late_slow_env_frac']:.3f} "
        f"{row['late_in_band_frac']:.3f} "
        f"{row['late_wrong_direction_env_frac']:.3f} "
        f"{row['late_lin_drift_env_frac']:.3f} "
        f"{row['yaw_abs_error_mean']:.4f} {row['yaw_abs_error_p90']:.4f} "
        f"{row['lin_drift_abs_mean']:.4f} {row['p95_pitch']:.4f} "
        f"{row['p99_pitch_rate']:.4f} {row['wheel_saturation_ratio']:.4f} "
        f"{row['yaw_delta_rms']:.4f} {row['yaw_delta_abs_p95']:.4f} "
        f"{row['terminated_event_rate']:.3f}"
      )
  if fixed_combo_summaries:
    print("\nFixed lin+yaw promotion summaries:")
    print(
      "  lin_x yaw lin_match lin_band lin_fast late_lin_band lin_err "
      "lin_delta yaw_match yaw_band yaw_fast late_yaw_band yaw_err "
      "yaw_delta p95_pitch p99_pitch_rate wheel_sat term"
    )
    for row in fixed_combo_summaries:
      print(
        f"  {row['lin_x']:+.3f} {row['yaw']:+.3f} "
        f"{row['lin_command_match_frac']:.3f} "
        f"{row['lin_in_band_frac']:.3f} {row['lin_fast_frac']:.3f} "
        f"{row['late_lin_in_band_frac']:.3f} "
        f"{row['lin_abs_error_mean']:.4f} {row['lin_x_delta_rms']:.4f} "
        f"{row['command_match_frac']:.3f} {row['in_band_frac']:.3f} "
        f"{row['fast_frac']:.3f} {row['late_in_band_frac']:.3f} "
        f"{row['yaw_abs_error_mean']:.4f} {row['yaw_delta_rms']:.4f} "
        f"{row['p95_pitch']:.4f} {row['p99_pitch_rate']:.4f} "
        f"{row['wheel_saturation_ratio']:.4f} "
        f"{row['terminated_event_rate']:.3f}"
      )
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
    "unsafe_support_forward_ratio",
    "clean_support_forward_frac",
    "support_posture_forward_frac",
    "non_wheel_contact_forward_frac",
    "root_height_below_clean_forward_frac",
    "root_height_below_hard_forward_frac",
    "pitch_rms",
    "pitch_abs_p95",
    "pitch_abs_p99",
    "pitch_abs_max",
    "pitch_rate_abs_p95",
    "pitch_rate_abs_p99",
    "pitch_rate_abs_max",
    "wheel_saturation_ratio",
    "wheel_target_rate_rms",
    "clean_support_frac",
    "support_posture_frac",
    "wheel_contact_frac",
    "non_wheel_contact_frac",
    "root_height_min",
    "root_height_p05",
    "root_height_below_clean_frac",
    "root_height_below_hard_frac",
    "raw_leg_abs_mean",
    "raw_leg_abs_p95",
    "termination_time_out_events",
    "termination_time_out_event_rate",
    "termination_bad_orientation_events",
    "termination_bad_orientation_event_rate",
    "termination_root_too_low_events",
    "termination_root_too_low_event_rate",
    "termination_non_wheel_ground_contact_events",
    "termination_non_wheel_ground_contact_event_rate",
    "termination_nan_detection_events",
    "termination_nan_detection_event_rate",
  ):
    value = metrics.get(name)
    if value is not None:
      print(f"  {name}: {value:.5f}")


if __name__ == "__main__":
  main()
