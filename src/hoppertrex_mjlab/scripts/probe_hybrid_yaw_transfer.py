#!/usr/bin/env python3
"""Sweep fixed yaw-head actions on a Hybrid stage and report body yaw transfer.

The Stage2 planar screen failed on yaw in-band/smoothness rules while the
policy only used ~35% of its yaw authority. This probe bypasses the policy,
holds the yaw residual at fixed values, and measures how much body yaw rate
each wheel differential actually buys on this platform. The resulting
transfer curve decides whether the in-band shortfall is a reward-equilibrium
problem (trainable) or a wheel-slip physics wall (task/gate recalibration).

With ``--fit-output`` the sweep additionally fits the Stage 2.0 yaw
feedforward map: the measured (body yaw rate, wheel differential) pairs are
pinned at (0, 0), monotonized outward from zero, and written as a
controller-bound yaw calibration artifact.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_PATH = Path(__file__).resolve().parents[1]
SRC_PATH = Path(__file__).resolve().parents[2]
REPOSITORY_PATH = Path(__file__).resolve().parents[3]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

try:
  import hoppertrex_mjlab.tasks as tasks  # noqa: E402,F401
  from hoppertrex_mjlab.hybrid.yaw_calibration import (
    validate_yaw_breakpoints,
    yaw_calibration_artifact,
  )
  from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
    hybrid_provenance_lines,
  )
except ImportError:
  import tasks  # noqa: E402,F401
  from hybrid.yaw_calibration import (  # type: ignore[no-redef]
    validate_yaw_breakpoints,
    yaw_calibration_artifact,
  )
  from tasks.hoppertrex_hybrid_task import (  # type: ignore[no-redef]
    hybrid_provenance_lines,
  )
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402


# The transfer sweep defaults to the historical diagnostic amplitudes; the
# fit sweep densifies both signs so the piecewise-linear map has enough knots
# across the stiction knee near zero.
DEFAULT_YAW_ACTIONS = (-1.0, -0.75, -0.5, -0.35, 0.35, 0.5, 0.75, 1.0)
FIT_DEFAULT_YAW_ACTIONS = (
  -1.0, -0.85, -0.7, -0.55, -0.4, -0.25, -0.15,
  0.15, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default="HopperTrex-Hybrid-v2-Stage2")
  parser.add_argument(
    "--device",
    default="cuda:0" if torch.cuda.is_available() else "cpu",
  )
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--settle-steps", type=int, default=50)
  parser.add_argument("--measure-steps", type=int, default=150)
  parser.add_argument(
    "--yaw-actions",
    type=float,
    nargs="+",
    default=None,
    help="Yaw-head action values in [-1, 1] to hold fixed.",
  )
  parser.add_argument(
    "--fit-output",
    type=Path,
    default=None,
    help=(
      "Fit the measured transfer into a yaw calibration artifact and write "
      "it to this JSON path. Requires a controller artifact with a gain hash."
    ),
  )
  parser.add_argument(
    "--probe-yaw-scale",
    type=float,
    default=1.0,
    help=(
      "Yaw action scale used for the probe env, decoupled from the training "
      "residual scale so the sweep can reach the full nominal differential."
    ),
  )
  return parser.parse_args(argv)


def _force_zero_twist_command(env: ManagerBasedRlEnv) -> None:
  term = env.command_manager.get_term("twist")
  for attribute in ("vel_command_b", "vel_command_w"):
    command = getattr(term, attribute)
    command[:, :] = 0.0
  for attribute in (
    "is_standing_env",
    "is_heading_env",
    "is_world_env",
    "is_forward_env",
  ):
    value = getattr(term, attribute, None)
    if value is not None:
      value[:] = False


def _run_value(
  env: ManagerBasedRlEnv,
  value: float,
  settle_steps: int,
  measure_steps: int,
) -> dict[str, float]:
  env.reset()
  actions = torch.zeros(
    (env.num_envs, env.action_space.shape[-1]),
    device=env.device,
  )
  actions[:, 1] = value
  term = env.action_manager.get_term("hybrid_wheel_leg")

  yaw_rates: list[torch.Tensor] = []
  lin_speeds: list[torch.Tensor] = []
  pitches: list[torch.Tensor] = []
  mapped_yaws: list[torch.Tensor] = []
  terminated_total = 0
  for step in range(settle_steps + measure_steps):
    _force_zero_twist_command(env)
    _obs, _rewards, terminated, _time_outs, _extras = env.step(actions)
    terminated_total += int(terminated.sum().item())
    if step < settle_steps:
      continue
    data = env.scene["robot"].data
    yaw_rates.append(data.root_link_ang_vel_b[:, 2].detach().cpu())
    lin_speeds.append(data.root_link_lin_vel_b[:, 0].abs().detach().cpu())
    projected_gravity = data.projected_gravity_b
    pitches.append(
      torch.atan2(
        projected_gravity[:, 0],
        torch.clamp(-projected_gravity[:, 2], min=1.0e-6),
      )
      .abs()
      .detach()
      .cpu()
    )
    wheel = term.wheel_targets.detach().cpu()
    mapped_yaws.append(0.5 * (wheel[:, 0] + wheel[:, 1]))

  yaw = torch.stack(yaw_rates)
  mapped = torch.stack(mapped_yaws)
  mean_yaw = float(yaw.mean().item())
  mean_mapped = float(mapped.mean().item())
  transfer = mean_yaw / mean_mapped if abs(mean_mapped) > 1.0e-9 else float("nan")
  yaw_delta = yaw[1:, :] - yaw[:-1, :]
  return {
    "yaw_action": value,
    "mean_mapped_yaw": mean_mapped,
    "mean_body_yaw": mean_yaw,
    "body_yaw_std": float(yaw.std().item()),
    "yaw_delta_rms": float(torch.sqrt(torch.mean(torch.square(yaw_delta))).item()),
    "transfer": transfer,
    "lin_speed_abs_mean": float(torch.stack(lin_speeds).mean().item()),
    "pitch_abs_p95": float(torch.quantile(torch.stack(pitches), 0.95).item()),
    "terminated_events": float(terminated_total),
  }


def fit_yaw_breakpoints(
  samples: list[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
  """Fit measured (body_yaw_rate, wheel_differential) pairs into a map.

  Pins (0, 0), sorts by yaw rate, drops near-duplicate rates, and enforces
  the sign/monotone structure the calibration contract requires by
  monotonizing outward from zero: measured differentials on the positive
  branch may only grow, on the negative branch only shrink. Small
  noise-driven inversions are absorbed instead of rejected because the
  probe measures a physically monotone plant.
  """

  cleaned: list[tuple[float, float]] = [(0.0, 0.0)]
  for rate, differential in samples:
    rate = float(rate)
    differential = float(differential)
    if abs(rate) <= 1.0e-9:
      continue
    # The differential must share the sign of the rate it produced; a probe
    # row that violates that is measurement garbage, not a knot.
    if rate > 0.0 and differential <= 0.0:
      continue
    if rate < 0.0 and differential >= 0.0:
      continue
    cleaned.append((rate, differential))
  cleaned.sort(key=lambda pair: pair[0])

  deduplicated: list[tuple[float, float]] = []
  for rate, differential in cleaned:
    if deduplicated and rate - deduplicated[-1][0] <= 1.0e-9:
      continue
    deduplicated.append((rate, differential))

  zero_index = next(
    index for index, (rate, _) in enumerate(deduplicated) if rate == 0.0
  )
  monotone = list(deduplicated)
  for index in range(zero_index + 1, len(monotone)):
    rate, differential = monotone[index]
    monotone[index] = (rate, max(differential, monotone[index - 1][1]))
  for index in range(zero_index - 1, -1, -1):
    rate, differential = monotone[index]
    monotone[index] = (rate, min(differential, monotone[index + 1][1]))

  return validate_yaw_breakpoints(monotone)


def _git_sha() -> str:
  completed = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=REPOSITORY_PATH,
    check=True,
    capture_output=True,
    text=True,
  )
  return completed.stdout.strip()


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  yaw_actions = args.yaw_actions
  if yaw_actions is None:
    yaw_actions = (
      FIT_DEFAULT_YAW_ACTIONS if args.fit_output else DEFAULT_YAW_ACTIONS
    )
  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = args.num_envs
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.num_envs = args.num_envs
  action_cfg = cfg.actions["hybrid_wheel_leg"]
  if args.fit_output is not None:
    if not action_cfg.controller_gain_hash:
      raise SystemExit(
        "--fit-output requires a controller artifact with a gain hash; the "
        "local PD fallback cannot anchor a yaw calibration. Set "
        "HOPPERTREX_HYBRID_CONTROLLER_PATH."
      )
    scales = list(action_cfg.action_scales)
    scales[1] = float(args.probe_yaw_scale)
    action_cfg.action_scales = tuple(scales)
    # The map is fit against the raw plant; a preloaded feedforward would
    # contaminate the measured differential if any yaw command leaked in.
    action_cfg.yaw_feedforward_breakpoints = (
      (-1.0, 0.0), (0.0, 0.0), (1.0, 0.0),
    )
    action_cfg.yaw_calibration_qualified = False
  for line in hybrid_provenance_lines(cfg):
    print(line)

  rows: list[dict[str, float]] = []
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    header = (
      f"{'action':>8} {'mapped_yaw':>11} {'body_yaw':>9} {'std':>7} "
      f"{'delta_rms':>10} {'transfer':>9} {'|lin_x|':>8} "
      f"{'p95|pitch|':>11} {'term':>5}"
    )
    print(header)
    for value in yaw_actions:
      row = _run_value(env, value, args.settle_steps, args.measure_steps)
      rows.append(row)
      print(
        f"{row['yaw_action']:>+8.3f} {row['mean_mapped_yaw']:>+11.4f} "
        f"{row['mean_body_yaw']:>+9.4f} {row['body_yaw_std']:>7.4f} "
        f"{row['yaw_delta_rms']:>10.4f} {row['transfer']:>+9.4f} "
        f"{row['lin_speed_abs_mean']:>8.4f} "
        f"{row['pitch_abs_p95']:>11.4f} {row['terminated_events']:>5.0f}"
      )
  finally:
    env.close()

  if args.fit_output is None:
    return

  terminated = sum(row["terminated_events"] for row in rows)
  if terminated > 0:
    raise SystemExit(
      f"Refusing to fit a yaw calibration from a sweep with {terminated:.0f} "
      "terminations; the measured transfer is contaminated by falls."
    )
  breakpoints = fit_yaw_breakpoints(
    [(row["mean_body_yaw"], row["mean_mapped_yaw"]) for row in rows]
  )
  payload = yaw_calibration_artifact(
    controller_gain_hash=action_cfg.controller_gain_hash,
    breakpoints=breakpoints,
    source_probe={
      "git_sha": _git_sha(),
      "task": args.task,
      "device": args.device,
      "num_envs": args.num_envs,
      "settle_steps": args.settle_steps,
      "measure_steps": args.measure_steps,
      "probe_yaw_scale": float(args.probe_yaw_scale),
      "yaw_actions": [float(value) for value in yaw_actions],
      "controller_qualified": bool(action_cfg.controller_qualified),
      "rows": rows,
    },
  )
  args.fit_output.parent.mkdir(parents=True, exist_ok=True)
  args.fit_output.write_text(
    json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
  )
  print(f"[probe] yaw calibration written: {args.fit_output}")
  print(f"[probe] yaw_calibration_hash={payload['yaw_calibration_hash']}")
  print(f"[probe] breakpoints={payload['breakpoints']}")


if __name__ == "__main__":
  main()
