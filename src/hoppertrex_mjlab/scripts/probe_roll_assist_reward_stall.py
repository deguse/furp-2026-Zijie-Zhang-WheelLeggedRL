#!/usr/bin/env python3
"""Measure B from the final 3 s of a safe zero-residual Hnext stall."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv

from hoppertrex_mjlab.hybrid.roll_assist import load_roll_boundary_verdict
from hoppertrex_mjlab.scripts.probe_roll_boundary import (
  COMMAND_VX_MPS,
  LEFT_SENSOR,
  POSTURE_CARDS,
  RIGHT_SENSOR,
  _force_commands,
  _reset_to_approach,
  bilateral_airborne,
  make_roll_boundary_env_cfg,
  wheel_contact,
)
from hoppertrex_mjlab.tasks.hoppertrex_balance_task import (
  NON_WHEEL_GROUND_SENSOR_NAME,
  non_wheel_ground_contact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--roll-boundary", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--smoke", action="store_true")
  args = parser.parse_args(argv)
  if args.output.exists():
    parser.error(f"Refusing to overwrite stall artifact: {args.output}")
  if not args.smoke and args.device != "cuda:0":
    parser.error("Formal reward stall is pinned to cuda:0.")
  return args


def _git_sha() -> str:
  repository = Path(__file__).resolve().parents[3]
  return subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
    capture_output=True, text=True,
  ).stdout.strip()


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite stall artifact: {args.output}")
  verdict = load_roll_boundary_verdict(
    args.roll_boundary, expected_git_sha=_git_sha()
  )
  hnext = float(verdict["hnext_m"])
  cfg = make_roll_boundary_env_cfg((hnext,), 1)
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  window_steps = 2 if args.smoke else 150
  drive_steps = 5 if args.smoke else 500
  settle_steps = 2 if args.smoke else 100
  card = POSTURE_CARDS[0]
  try:
    _types, _face, _cross, _reset = _reset_to_approach(
      env, root_height=float(card["height_m"]), card_name=str(card["name"]),
      repeat=1, height_count=1,
    )
    actions = torch.zeros((1, 6), device=env.device)
    inherited_positive_rates = []
    full_terminations = full_contacts = full_airborne_steps = 0
    window_terminations = window_contacts = window_airborne_steps = 0
    def step(vx: float, record: bool) -> None:
      nonlocal full_terminations, full_contacts, full_airborne_steps
      nonlocal window_terminations, window_contacts, window_airborne_steps
      active = torch.ones(1, dtype=torch.bool, device=env.device)
      _force_commands(env, active=active, vx=vx, height=float(card["height_m"]),
                      pitch=float(card["pitch_rad"]))
      _obs, _reward, terminated, _timeout, _extra = env.step(actions)
      left, right = wheel_contact(env, LEFT_SENSOR), wheel_contact(env, RIGHT_SENSOR)
      airborne = bilateral_airborne(left, right)
      non_wheel = non_wheel_ground_contact(env, NON_WHEEL_GROUND_SENSOR_NAME).bool()
      terminated_count = int(terminated.sum().item())
      contact_count = int(non_wheel.sum().item())
      airborne_count = int(airborne.sum().item())
      full_terminations += terminated_count
      full_contacts += contact_count
      full_airborne_steps += airborne_count
      if record:
        window_terminations += terminated_count
        window_contacts += contact_count
        window_airborne_steps += airborne_count
        manager = env.reward_manager
        positive = torch.clamp(manager._step_reward, min=0.0).sum(dim=1)
        inherited_positive_rates.append(float(positive[0].item()))
    for _ in range(settle_steps):
      step(0.0, False)
    for index in range(drive_steps):
      step(COMMAND_VX_MPS, index >= drive_steps - window_steps)
  finally:
    env.close()
  payload = {
    "schema_version": 1, "kind": "roll_assist_zero_residual_stall",
    "evidence_eligible": not args.smoke, "stair_height_m": hnext,
    "protocol": {"policy_action": [0.0] * 6, "wheel_residual_exact_zero": True,
                 "measurement_window_s": 3.0, "height_role": "Hnext",
                 "command_vx_mps": COMMAND_VX_MPS},
    "safety": {
      "scope": "final_3s_measurement_window",
      "terminations": window_terminations,
      "non_wheel_contacts": window_contacts,
      "bilateral_airborne": window_airborne_steps,
    },
    "full_rollout_safety": {
      "scope": "post_reset_settle_and_drive",
      "terminations": full_terminations,
      "non_wheel_contacts": full_contacts,
      "bilateral_airborne": full_airborne_steps,
    },
    "measurement": {
      "inherited_positive_reward_rate": sum(inherited_positive_rates) / len(inherited_positive_rates),
      "samples": len(inherited_positive_rates),
    },
  }
  if not args.smoke and (
    full_terminations or full_contacts or full_airborne_steps
  ):
    raise RuntimeError("Hnext stall rollout was unsafe; do not train RollAssist.")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  temporary = args.output.with_name(f".{args.output.name}.incomplete")
  if temporary.exists():
    raise FileExistsError(f"Stale reward-stall temporary output: {temporary}")
  try:
    temporary.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
  finally:
    if temporary.exists():
      temporary.unlink()
  print(f"[roll-assist] stall={args.output}")
  print(f"[roll-assist] B={payload['measurement']['inherited_positive_reward_rate']}")


if __name__ == "__main__":
  main()
