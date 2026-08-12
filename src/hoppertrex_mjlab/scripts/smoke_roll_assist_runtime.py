#!/usr/bin/env python3
"""CPU-only RollAssist runtime smoke; never evidence eligible."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv

from hoppertrex_mjlab.hybrid.runner import zero_initialize_actor_output
from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
  ROLL_ASSIST_FLAT_ENVS,
  ROLL_ASSIST_SETTLE_STEPS,
  make_stair_roll_assist_env_cfg,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cpu")
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  if args.device != "cpu":
    raise ValueError("RollAssist runtime smoke is pinned to CPU and is never evidence.")
  if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite smoke output: {args.output}")
  cfg = make_stair_roll_assist_env_cfg(play=False)
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    observation = env.reset()[0]
    actor_obs = observation["actor"]
    critic_obs = observation["critic"]
    actor = torch.nn.Sequential(
      torch.nn.Linear(34, 128), torch.nn.ELU(), torch.nn.Linear(128, 6)
    ).to(env.device)
    zero_initialize_actor_output(actor, label="RollAssistSmoke")
    actions = actor(actor_obs)
    command_before = env.command_manager.get_term("twist").command.clone()
    _observation, _reward, _terminated, _timeout, _extras = env.step(actions)
    action_term = env.action_manager.get_term("hybrid_wheel_leg")
    terrain = env.scene.terrain
    payload = {
      "schema_version": 1,
      "kind": "roll_assist_cpu_runtime_smoke",
      "evidence_eligible": False,
      "device": args.device,
      "actor_shape": list(actor_obs.shape),
      "critic_shape": list(critic_obs.shape),
      "action_shape": list(actions.shape),
      "deterministic_action_abs_max": float(actions.abs().max().item()),
      "wheel_residual_abs_max": float(
        action_term.applied_residual[:, :2].abs().max().item()
      ),
      "dynamic_stair_maneuver": action_term.cfg.dynamic_stair_maneuver is not None,
      "contact_trigger_sensor": action_term.cfg.stair_trigger_sensor_name,
      "reference_freeze": action_term.cfg.stair_mode_freezes_leg_reference,
      "flat_slots": int((terrain.terrain_types == 0).sum().item()),
      "stair_slots": int((terrain.terrain_types == 1).sum().item()),
      "stair_command_before_first_step_abs_max": float(
        command_before[ROLL_ASSIST_FLAT_ENVS:, 0].abs().max().item()
      ),
      "settle_steps": ROLL_ASSIST_SETTLE_STEPS,
    }
  finally:
    env.close()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  temporary = args.output.with_name(f".{args.output.name}.incomplete")
  try:
    temporary.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
  finally:
    if temporary.exists():
      temporary.unlink()
  print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
