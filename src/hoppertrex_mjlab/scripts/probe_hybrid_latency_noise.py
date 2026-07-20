"""Latency/noise tolerance probe for the classical stack (sim-to-real k.0).

Sweeps control-loop action delay and sensor-noise tiers over the frozen
Stage5 classical stack (zero residual by default; optionally the trained
candidate) and measures standing/tracking/kick-recovery health per cell.

The output JSON is the measured tolerance envelope that the hardware
requirements section of the sim-to-real runbook consumes: it answers "how
much loop latency and how much IMU/encoder noise can the deployed stack
absorb before the qualification scenarios degrade" BEFORE any real-robot
parameter is known, so bus/compute/IMU choices can be made against data.

Noise tiers are scan inputs, not thresholds: magnitudes are annotated with
their datasheet-class provenance and replaced by measured values once the
real IMU exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_PATH = Path(__file__).resolve().parents[1]
SRC_PATH = Path(__file__).resolve().parents[2]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

try:
  import hoppertrex_mjlab.tasks as tasks  # noqa: E402,F401
  from hoppertrex_mjlab.scripts.probe_hybrid_posture_balance import (
    _force_commands,
    _pitch,
  )
  from hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate import (
    CONTROL_FREQUENCY_HZ,
    HYBRID_STAGE_TASKS,
    STAGE1_KICK_LIN_X,
    STAGE1_KICK_PITCH_RATE,
  )
  from hoppertrex_mjlab.tasks.hoppertrex_balance_task import (
    NON_WHEEL_GROUND_SENSOR_NAME,
    non_wheel_ground_contact,
  )
  from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
    hybrid_provenance_lines,
  )
except ImportError:
  import tasks  # noqa: E402,F401
  from scripts.probe_hybrid_posture_balance import (  # type: ignore[no-redef]
    _force_commands,
    _pitch,
  )
  from scripts.rsl_rl.evaluate_hybrid_gate import (  # type: ignore[no-redef]
    CONTROL_FREQUENCY_HZ,
    HYBRID_STAGE_TASKS,
    STAGE1_KICK_LIN_X,
    STAGE1_KICK_PITCH_RATE,
  )
  from tasks.hoppertrex_balance_task import (  # type: ignore[no-redef]
    NON_WHEEL_GROUND_SENSOR_NAME,
    non_wheel_ground_contact,
  )
  from tasks.hoppertrex_hybrid_task import (  # type: ignore[no-redef]
    hybrid_provenance_lines,
  )

from dataclasses import asdict  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import (  # noqa: E402
  load_env_cfg,
  load_rl_cfg,
  load_runner_cls,
)

# Scan tiers. Magnitudes are datasheet-class orders for consumer MEMS IMUs
# (BMI088/MPU-6050 class gyro/attitude noise after filtering) and joint
# encoder velocity differentiation; they parameterize the sweep and carry
# no pass/fail meaning of their own.
NOISE_TIERS: dict[str, dict[str, float]] = {
  "none": {
    "pitch_std": 0.0,
    "pitch_rate_std": 0.0,
    "vx_std": 0.0,
    "wheel_vel_std": 0.0,
  },
  "encoder": {
    "pitch_std": 0.0,
    "pitch_rate_std": 0.0,
    "vx_std": 0.0,
    "wheel_vel_std": 0.05,
  },
  "mems_imu": {
    "pitch_std": 0.005,
    "pitch_rate_std": 0.02,
    "vx_std": 0.01,
    "wheel_vel_std": 0.05,
  },
  "mems_imu_2x": {
    "pitch_std": 0.010,
    "pitch_rate_std": 0.04,
    "vx_std": 0.02,
    "wheel_vel_std": 0.10,
  },
}
DEFAULT_DELAYS = (0, 1, 2, 3, 4)
TRACKING_VX = 0.07


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default=HYBRID_STAGE_TASKS[5])
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument(
    "--delays",
    type=int,
    nargs="+",
    default=list(DEFAULT_DELAYS),
    help="Action delay values in 50 Hz control steps (20 ms each).",
  )
  parser.add_argument(
    "--noise-tiers",
    nargs="+",
    default=list(NOISE_TIERS),
    choices=list(NOISE_TIERS),
  )
  parser.add_argument(
    "--checkpoint-file",
    type=Path,
    default=None,
    help=(
      "Optional trained Stage5 candidate. When set, every cell runs twice "
      "(zero residual and candidate) so the probe also answers whether the "
      "residual policy helps or hurts under latency/noise."
    ),
  )
  parser.add_argument("--settle-steps", type=int, default=150)
  parser.add_argument("--tracking-steps", type=int, default=300)
  parser.add_argument("--kicks-per-cell", type=int, default=4)
  parser.add_argument("--kick-interval", type=int, default=300)
  parser.add_argument("--fit-output", type=Path, default=None)
  return parser.parse_args(argv)


def _git_sha() -> str:
  return subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=Path(__file__).resolve().parents[2],
    text=True,
  ).strip()


def _cell_env_cfg(
  task: str,
  *,
  num_envs: int,
  delay_steps: int,
  tier: str,
  seed: int,
  policy_obs_noise: bool,
):
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = num_envs
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.num_envs = num_envs
  action = cfg.actions["hybrid_wheel_leg"]
  action.action_delay_steps = int(delay_steps)
  noise = NOISE_TIERS[tier]
  action.sensor_noise_pitch_std = noise["pitch_std"]
  action.sensor_noise_pitch_rate_std = noise["pitch_rate_std"]
  action.sensor_noise_vx_std = noise["vx_std"]
  action.sensor_noise_wheel_vel_std = noise["wheel_vel_std"]
  action.sensor_noise_seed = seed
  if policy_obs_noise and tier != "none":
    # The candidate consumes observations through the manager pipeline; the
    # training-time corruption fields are the honest noise model for it.
    cfg.observations["actor"].enable_corruption = True
  return cfg


def _run_tracking_cell(
  env: ManagerBasedRlEnv,
  policy,
  *,
  settle_steps: int,
  tracking_steps: int,
  center: tuple[float, float],
) -> dict[str, float]:
  """Standing window then a vx step, both at the center posture."""

  observations, _ = env.reset()
  height, pitch = center
  pitch_rows: list[torch.Tensor] = []
  vx_error_rows: list[torch.Tensor] = []
  stand_vx_rows: list[torch.Tensor] = []
  terminated = 0
  robot = env.scene["robot"]
  phases = (
    (settle_steps, 0.0, None),
    (tracking_steps, 0.0, stand_vx_rows),
    (settle_steps, TRACKING_VX, None),
    (tracking_steps, TRACKING_VX, vx_error_rows),
  )
  for steps, vx, sink in phases:
    for _ in range(steps):
      _force_commands(env, vx=vx, height=height, pitch=pitch)
      with torch.no_grad():
        observations, _reward, _term, _trunc, _extras = env.step(
          policy(observations)
        )
      _force_commands(env, vx=vx, height=height, pitch=pitch)
      terminated += int(env.reset_terminated.sum().item())
      if sink is None:
        continue
      body_vx = robot.data.root_link_lin_vel_b[:, 0].detach()
      if vx == 0.0:
        sink.append(body_vx.abs().cpu())
      else:
        sink.append((body_vx - vx).abs().cpu())
      pitch_rows.append(_pitch(robot.data).abs().detach().cpu())
  pitch_all = torch.cat(pitch_rows)
  tracking_error = torch.cat(vx_error_rows)
  standing_vx = torch.cat(stand_vx_rows)
  return {
    "standing_abs_vx_mean": standing_vx.mean().item(),
    "tracking_abs_error_mean": tracking_error.mean().item(),
    "tracking_abs_error_p95": tracking_error.quantile(0.95).item(),
    "pitch_abs_p95": pitch_all.quantile(0.95).item(),
    "terminated_events": float(terminated),
  }


def _run_policy_kick_cell(
  env: ManagerBasedRlEnv,
  policy,
  *,
  center: tuple[float, float],
  kicks: int,
  kick_interval: int,
  settle_steps: int,
) -> dict[str, float]:
  """Policy-aware mirror of run_kick_cell (which drives zero actions only)."""

  observations, _ = env.reset()
  height, pitch = center
  robot = env.scene["robot"]
  env_ids = torch.arange(env.num_envs, device=env.device)
  kick_steps = {
    settle_steps + index * kick_interval: index for index in range(kicks)
  }
  total_steps = settle_steps + kicks * kick_interval
  healthy_rows: list[torch.Tensor] = []
  lin_rows: list[torch.Tensor] = []
  contact_rows: list[torch.Tensor] = []
  terminated = 0
  for step in range(total_steps):
    _force_commands(env, vx=0.0, height=height, pitch=pitch)
    if step in kick_steps:
      _apply_stage1_kick(env, env_ids, kick_index=kick_steps[step])
    with torch.no_grad():
      observations, _reward, _term, _trunc, _extras = env.step(
        policy(observations)
      )
    _force_commands(env, vx=0.0, height=height, pitch=pitch)
    terminated += int(env.reset_terminated.sum().item())
    if step < settle_steps:
      continue
    data = robot.data
    lin = data.root_link_lin_vel_b[:, 0]
    healthy = (
      (lin.abs() <= 0.06)
      & ((data.root_link_pos_w[:, 2] - height).abs() <= 0.015)
      & ((_pitch(data) - pitch).abs() <= 0.04)
    )
    healthy_rows.append(healthy.detach().cpu())
    lin_rows.append(lin.abs().detach().cpu())
    contact_rows.append(
      non_wheel_ground_contact(env, NON_WHEEL_GROUND_SENSOR_NAME)
      .detach()
      .cpu()
    )
  healthy_all = torch.stack(healthy_rows)
  streak = torch.zeros(env.num_envs)
  longest = torch.zeros(env.num_envs)
  for row in healthy_all:
    streak = torch.where(row, torch.zeros_like(streak), streak + 1.0)
    longest = torch.maximum(longest, streak)
  return {
    "recovery_time_s": longest.max().item() / CONTROL_FREQUENCY_HZ,
    "post_kick_lin_x_abs_max": torch.cat(lin_rows).max().item(),
    "kick_event_count": float(kicks * env.num_envs),
    "terminated_events": float(terminated),
    "non_wheel_contact_rate": (
      torch.cat(contact_rows).float().mean().item()
    ),
  }


def _apply_stage1_kick(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  *,
  kick_index: int,
) -> None:
  robot = env.scene["robot"]
  velocity = robot.data.root_link_vel_w[env_ids].clone()
  direction = 1.0 if kick_index % 2 == 0 else -1.0
  velocity[:, 0] += direction * STAGE1_KICK_LIN_X
  velocity[:, 4] += direction * STAGE1_KICK_PITCH_RATE
  robot.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)
  env.sim.forward()
  env.sim.sense()


def _zero_policy(action_dim: int):
  def policy(observations) -> torch.Tensor:
    actor = observations["actor"]
    return torch.zeros(
      (actor.shape[0], action_dim),
      dtype=actor.dtype,
      device=actor.device,
    )

  return policy


def _candidate_policy(task: str, env, checkpoint: Path, device: str):
  agent_cfg = load_rl_cfg(task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  return runner.get_inference_policy(device=device)


def probe_payload(
  *,
  cells: list[dict[str, object]],
  action_cfg,
  source_probe: dict[str, object],
  checkpoint: Path | None,
) -> dict[str, object]:
  reference = next(
    (
      cell
      for cell in cells
      if cell["policy"] == "zero_residual"
      and cell["delay_steps"] == 0
      and cell["noise_tier"] == "none"
    ),
    None,
  )
  knees: dict[str, object] = {}
  for tier in {str(cell["noise_tier"]) for cell in cells}:
    tier_cells = sorted(
      (
        cell
        for cell in cells
        if cell["policy"] == "zero_residual"
        and cell["noise_tier"] == tier
      ),
      key=lambda cell: int(cell["delay_steps"]),  # type: ignore[arg-type]
    )
    knee = None
    for cell in tier_cells:
      degraded = cell["terminated_events"] > 0 or (
        reference is not None
        and float(cell["recovery_time_s"])
        > 2.0 * float(reference["recovery_time_s"])
      )
      if degraded:
        knee = int(cell["delay_steps"])
        break
    knees[tier] = knee
  payload: dict[str, object] = {
    "schema_version": 1,
    "probe": "hybrid_latency_noise_tolerance",
    "noise_tiers": NOISE_TIERS,
    "noise_tier_provenance": (
      "datasheet-class magnitudes (consumer MEMS IMU attitude/gyro noise, "
      "encoder velocity differentiation); scan inputs, not thresholds"
    ),
    "control_step_ms": 1000.0 / CONTROL_FREQUENCY_HZ,
    "controller_gain_hash": action_cfg.controller_gain_hash,
    "controller_qualified": bool(action_cfg.controller_qualified),
    "calibration_hash": action_cfg.calibration_hash,
    "yaw_calibration_hash": action_cfg.yaw_calibration_hash,
    "posture_map_hash": action_cfg.posture_map_hash,
    "station_calibration_hash": action_cfg.station_calibration_hash,
    "checkpoint": None if checkpoint is None else str(checkpoint),
    "checkpoint_file_sha256": (
      None
      if checkpoint is None
      else hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    ),
    "source_probe": source_probe,
    "zero_residual_delay_knee_by_tier": knees,
    "cells": cells,
  }
  return payload


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  checkpoint = (
    None
    if args.checkpoint_file is None
    else args.checkpoint_file.expanduser().resolve()
  )
  if checkpoint is not None and not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

  preview_cfg = load_env_cfg(args.task, play=True)
  for line in hybrid_provenance_lines(preview_cfg):
    print(line)
  posture_cfg = preview_cfg.commands["posture"]
  center = (
    0.5 * (posture_cfg.height_range[0] + posture_cfg.height_range[1]),
    0.5 * (posture_cfg.pitch_range[0] + posture_cfg.pitch_range[1]),
  )
  action_cfg = preview_cfg.actions["hybrid_wheel_leg"]

  policies = ["zero_residual"] + ([] if checkpoint is None else ["candidate"])
  cells: list[dict[str, object]] = []
  print(
    f"{'policy':>13} {'delay':>5} {'tier':>12} {'recov_s':>8} "
    f"{'trk_p95':>8} {'p95_pitch':>9} {'term':>5}"
  )
  for policy_name in policies:
    for tier in args.noise_tiers:
      for delay in args.delays:
        cfg = _cell_env_cfg(
          args.task,
          num_envs=args.num_envs,
          delay_steps=delay,
          tier=tier,
          seed=args.seed,
          policy_obs_noise=policy_name == "candidate",
        )
        torch.manual_seed(args.seed)
        env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
        try:
          if policy_name == "candidate":
            assert checkpoint is not None
            policy = _candidate_policy(
              args.task, env, checkpoint, args.device
            )
          else:
            policy = _zero_policy(
              int(env.action_manager.total_action_dim)
            )
          tracking = _run_tracking_cell(
            env,
            policy,
            settle_steps=args.settle_steps,
            tracking_steps=args.tracking_steps,
            center=center,
          )
          kick = _run_policy_kick_cell(
            env,
            policy,
            center=center,
            kicks=args.kicks_per_cell,
            kick_interval=args.kick_interval,
            settle_steps=args.settle_steps,
          )
        finally:
          env.close()
        cell: dict[str, object] = {
          "policy": policy_name,
          "delay_steps": int(delay),
          "delay_ms": 1000.0 * delay / CONTROL_FREQUENCY_HZ,
          "noise_tier": tier,
          **tracking,
          "recovery_time_s": float(kick["recovery_time_s"]),
          "post_kick_lin_x_abs_max": float(
            kick["post_kick_lin_x_abs_max"]
          ),
          "kick_event_count": float(kick["kick_event_count"]),
          "terminated_events": float(
            tracking["terminated_events"] + kick["terminated_events"]
          ),
          "non_wheel_contact_rate": float(kick["non_wheel_contact_rate"]),
        }
        cells.append(cell)
        print(
          f"{policy_name:>13} {delay:>5d} {tier:>12} "
          f"{cell['recovery_time_s']:>8.3f} "
          f"{cell['tracking_abs_error_p95']:>8.4f} "
          f"{cell['pitch_abs_p95']:>9.4f} "
          f"{cell['terminated_events']:>5.0f}"
        )

  if args.fit_output is None:
    return
  payload = probe_payload(
    cells=cells,
    action_cfg=action_cfg,
    source_probe={
      "git_sha": _git_sha(),
      "task": args.task,
      "device": args.device,
      "num_envs": args.num_envs,
      "seed": args.seed,
      "settle_steps": args.settle_steps,
      "tracking_steps": args.tracking_steps,
      "kicks_per_cell": args.kicks_per_cell,
      "kick_interval": args.kick_interval,
      "delays": [int(value) for value in args.delays],
      "noise_tiers": list(args.noise_tiers),
      "tracking_vx": TRACKING_VX,
    },
    checkpoint=checkpoint,
  )
  args.fit_output.parent.mkdir(parents=True, exist_ok=True)
  args.fit_output.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  print(f"[probe] wrote {args.fit_output}")


if __name__ == "__main__":
  main()
