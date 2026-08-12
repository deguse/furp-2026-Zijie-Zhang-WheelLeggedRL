#!/usr/bin/env python3
"""Formal RollAssist checkpoint envelope and paired R0/R1 live evaluator."""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.config import flat

import hoppertrex_mjlab.tasks  # noqa: F401
from hoppertrex_mjlab.hybrid.roll_assist import (
  ROLL_ASSIST_ACTION_MASK,
  ROLL_ASSIST_ACTION_SCALES,
  ROLL_ASSIST_TASK_ID,
  ROLL_ASSIST_TRAINING_INFO_KEY,
  canonical_json_sha256,
  continuation_gate,
  file_sha256,
  final_expansion_gate,
  load_roll_boundary_verdict,
  validate_reward_calibration,
  validate_roll_assist_training_record,
)
from hoppertrex_mjlab.hybrid.runner import HybridOnPolicyRunner, repository_git_sha
from hoppertrex_mjlab.scripts.evaluate_hybrid_c1_flat_gate import (
  REGISTERED_CAPS,
  aggregate_candidate,
  evaluation_cells,
)
from hoppertrex_mjlab.scripts.probe_roll_boundary import (
  POSTURE_CARDS,
  aggregate_trials,
  make_roll_boundary_env_cfg,
  run_card_repeat,
)
from hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate import (
  _collect_scenarios,
  _collect_stage4_reference_from_baseline,
  _posture_targets_from_cfg,
  _run_recovery_scenario,
)
from hoppertrex_mjlab.scripts.rsl_rl.hybrid_gate import (
  evaluate_capability_suite,
)

SCHEMA_VERSION = 1
CHECKPOINT_KIND = "roll_assist_checkpoint_envelope"
EVALUATION_KIND = "roll_assist_evaluation"
FORMAL_DEVICE = "cuda:0"
STAGE5_TASK_ID = "HopperTrex-Hybrid-v2-Stage5"
FORMAL_ENVS_PER_CELL = 16
FORMAL_REPEATS = 3
FORMAL_SETTLE_STEPS = 100
FORMAL_DRIVE_STEPS = 500
FORMAL_STABLE_STEPS = 25
FLAT_EVALUATION_SIZE_M = (16.0, 16.0)
REPOSITORY_PATH = Path(__file__).resolve().parents[3]


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
  if path.exists():
    raise FileExistsError(f"Refusing to overwrite RollAssist output: {path}")
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.incomplete")
  if temporary.exists():
    raise FileExistsError(f"Stale RollAssist temporary output: {temporary}")
  try:
    temporary.write_text(
      json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
      encoding="utf-8",
    )
    temporary.replace(path)
  finally:
    if temporary.exists():
      temporary.unlink()


def _git_sha() -> str:
  return subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_PATH, check=True,
    capture_output=True, text=True,
  ).stdout.strip()


def _read(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8-sig"))
  if not isinstance(value, dict):
    raise TypeError(f"{path} must contain a JSON object.")
  return value


def validate_checkpoint_reward_binding(
  training: Mapping[str, Any], reward_calibration_path: Path
) -> dict[str, Any]:
  """Bind evaluation to the exact reward-calibration bytes in the checkpoint."""

  path = reward_calibration_path.expanduser().resolve()
  if not path.is_file() or file_sha256(path) != training.get(
    "reward_calibration_sha256"
  ):
    raise ValueError("RollAssist checkpoint/evaluator reward-calibration bytes drifted.")
  payload = _read(path)
  return validate_reward_calibration(
    payload, expected_roll_boundary_sha256=str(training.get("r0_sha256", ""))
  )


def checkpoint_envelope(checkpoint_path: Path) -> dict[str, Any]:
  path = checkpoint_path.expanduser().resolve()
  checkpoint = torch.load(path, map_location="cpu", weights_only=False)
  if not isinstance(checkpoint, Mapping):
    raise TypeError("RollAssist checkpoint root must be a mapping.")
  infos = checkpoint.get("infos")
  if not isinstance(infos, Mapping):
    raise TypeError("RollAssist checkpoint has no infos mapping.")
  training = infos.get(ROLL_ASSIST_TRAINING_INFO_KEY)
  if not isinstance(training, Mapping):
    raise TypeError("RollAssist checkpoint has no training provenance.")
  completed = validate_roll_assist_training_record(
    training,
    git_sha=_git_sha(),
    r0_sha256=str(training.get("r0_sha256", "")),
    reward_calibration_sha256=str(training.get("reward_calibration_sha256", "")),
  )
  if training.get("update25_curriculum_decided") is not True:
    raise ValueError("RollAssist checkpoint has no frozen update-25 decision.")
  iteration = checkpoint.get("iter")
  if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration + 1 != completed:
    raise ValueError("RollAssist checkpoint iteration/update count drifted.")
  curriculum = infos.get("roll_assist_curriculum")
  progress = infos.get("roll_assist_progress")
  env_state = infos.get("env_state")
  if not all(isinstance(value, Mapping) for value in (curriculum, progress, env_state)):
    raise TypeError("RollAssist checkpoint is missing curriculum/progress/env state.")
  if (
    curriculum.get("decision_made") is not True
    or float(training["active_height_m"]) != float(progress.get("active_height_m"))
  ):
    raise ValueError("RollAssist training/curriculum active-height binding drifted.")
  common_step = env_state.get("common_step_counter")
  if isinstance(common_step, bool) or not isinstance(common_step, int):
    raise TypeError("RollAssist checkpoint common step is invalid.")
  if common_step != completed * 24:
    raise ValueError("RollAssist checkpoint common step/update count drifted.")
  payload = {
    "schema_version": SCHEMA_VERSION,
    "kind": CHECKPOINT_KIND,
    "checkpoint_file": str(path),
    "checkpoint_file_sha256": file_sha256(path),
    "completed_updates": completed,
    "iteration": iteration,
    "training": dict(training),
    "curriculum": dict(curriculum),
    "progress": dict(progress),
    "common_step_counter": common_step,
  }
  payload["envelope_sha256"] = canonical_json_sha256(payload)
  return payload


def validate_checkpoint_envelope(
  payload: Mapping[str, Any], *, verify_file: bool = True
) -> dict[str, Any]:
  expected = {
    "schema_version", "kind", "checkpoint_file", "checkpoint_file_sha256",
    "completed_updates", "iteration", "training", "curriculum", "progress",
    "common_step_counter", "envelope_sha256",
  }
  if set(payload) != expected:
    raise ValueError("RollAssist checkpoint envelope schema drifted.")
  unsigned = dict(payload)
  observed_hash = unsigned.pop("envelope_sha256")
  if observed_hash != canonical_json_sha256(unsigned):
    raise ValueError("RollAssist checkpoint envelope hash drifted.")
  if payload["schema_version"] != SCHEMA_VERSION or payload["kind"] != CHECKPOINT_KIND:
    raise ValueError("RollAssist checkpoint envelope kind/version drifted.")
  training = payload["training"]
  if not isinstance(training, Mapping):
    raise TypeError("RollAssist checkpoint envelope training must be an object.")
  completed = validate_roll_assist_training_record(
    training,
    git_sha=_git_sha(),
    r0_sha256=str(training.get("r0_sha256", "")),
    reward_calibration_sha256=str(training.get("reward_calibration_sha256", "")),
  )
  if (
    payload["completed_updates"] != completed
    or payload["iteration"] + 1 != completed
    or payload["common_step_counter"] != completed * 24
  ):
    raise ValueError("RollAssist checkpoint envelope update count drifted.")
  if not isinstance(payload["curriculum"], Mapping) or not isinstance(
    payload["progress"], Mapping
  ):
    raise TypeError("RollAssist checkpoint envelope runtime state must be objects.")
  if (
    payload["curriculum"].get("decision_made") is not True
    or float(payload["progress"].get("active_height_m", -1.0))
    != float(training["active_height_m"])
  ):
    raise ValueError("RollAssist envelope runtime/training state drifted.")
  if verify_file:
    path = Path(str(payload["checkpoint_file"])).resolve()
    if not path.is_file() or file_sha256(path) != payload["checkpoint_file_sha256"]:
      raise ValueError("RollAssist checkpoint envelope file bytes drifted.")
  return dict(payload)


def screen_checkpoint_envelope(
  checkpoint: Mapping[str, Any], screen: Mapping[str, Any]
) -> dict[str, Any]:
  """Attach rejection-only K=3 status without score ranking."""

  if screen.get("kind") == EVALUATION_KIND:
    if screen.get("profile") != "screen" or screen.get("evidence_eligible") is not False:
      raise ValueError("RollAssist K=3 input is not a rejection-only screen.")
    evaluation_checkpoint = screen.get("checkpoint")
    if not isinstance(evaluation_checkpoint, Mapping):
      raise TypeError("RollAssist screen evaluation has no checkpoint envelope.")
    validated_evaluation_checkpoint = validate_checkpoint_envelope(
      evaluation_checkpoint, verify_file=False
    )
    validated_input_checkpoint = validate_checkpoint_envelope(
      checkpoint, verify_file=False
    )
    if validated_evaluation_checkpoint != validated_input_checkpoint:
      raise ValueError("RollAssist screen and checkpoint envelopes differ.")
    screen = screen.get("screen")
    if not isinstance(screen, Mapping):
      raise TypeError("RollAssist screen evaluation has no checks object.")
  allowed = {"flat_retention_passed", "hpass_retained", "hnext_safe", "wheel_residual_exact_zero"}
  if set(screen) != allowed or not all(isinstance(value, bool) for value in screen.values()):
    raise ValueError("RollAssist K=3 screen schema drifted.")
  envelope = validate_checkpoint_envelope(checkpoint, verify_file=False)
  return {
    "schema_version": SCHEMA_VERSION,
    "kind": "roll_assist_k3_screen",
    "checkpoint_file": envelope["checkpoint_file"],
    "checkpoint_file_sha256": envelope["checkpoint_file_sha256"],
    "completed_updates": envelope["completed_updates"],
    "passed": all(screen.values()),
    "checks": dict(screen),
  }


def _load_policy(
  checkpoint: Mapping[str, Any], *, device: str
) -> tuple[Any, Any]:
  envelope = validate_checkpoint_envelope(checkpoint, verify_file=True)
  env_cfg = load_env_cfg(ROLL_ASSIST_TASK_ID, play=True)
  env_cfg.scene.num_envs = 1
  if env_cfg.scene.terrain is not None:
    env_cfg.scene.terrain.num_envs = 1
  agent_cfg = load_rl_cfg(ROLL_ASSIST_TASK_ID)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(ROLL_ASSIST_TASK_ID)
  if runner_cls is not HybridOnPolicyRunner:
    wrapped.close()
    raise RuntimeError("RollAssist registered runner drifted.")
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  infos = runner.load(
    str(envelope["checkpoint_file"]),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  actual = infos.get(ROLL_ASSIST_TRAINING_INFO_KEY)
  if actual != envelope["training"]:
    wrapped.close()
    raise RuntimeError("Loaded RollAssist checkpoint differs from its envelope.")
  policy = runner.get_inference_policy(device=device)
  probe = policy(wrapped.get_observations())
  if tuple(probe.shape) != (1, 6):
    wrapped.close()
    raise RuntimeError("RollAssist actor does not emit six actions.")
  wrapped.close()
  return runner, policy


def _evaluate_stair_arm(
  *,
  policy: Any,
  height: float,
  device: str,
  baseline: bool,
  profile: str,
) -> dict[str, Any]:
  if profile == "smoke":
    envs, repeats, settle, drive, stable = 1, 1, 2, 5, 2
  elif profile == "screen":
    envs, repeats, settle, drive, stable = (
      FORMAL_ENVS_PER_CELL, 1, FORMAL_SETTLE_STEPS,
      FORMAL_DRIVE_STEPS, FORMAL_STABLE_STEPS,
    )
  elif profile == "formal":
    envs, repeats, settle, drive, stable = (
      FORMAL_ENVS_PER_CELL, FORMAL_REPEATS, FORMAL_SETTLE_STEPS,
      FORMAL_DRIVE_STEPS, FORMAL_STABLE_STEPS,
    )
  else:
    raise ValueError(f"Unknown RollAssist evaluation profile: {profile!r}")
  mask = (False,) * 6 if baseline else ROLL_ASSIST_ACTION_MASK
  cfg = make_roll_boundary_env_cfg(
    (height,), envs,
    residual_mask=mask,
    action_scales=ROLL_ASSIST_ACTION_SCALES,
  )
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    trials: list[dict[str, Any]] = []
    for card in POSTURE_CARDS:
      for repeat in range(1, repeats + 1):
        trials.extend(run_card_repeat(
          env,
          heights=(height,),
          card=card,
          repeat=repeat,
          settle_steps=settle,
          drive_steps=drive,
          stable_steps=stable,
          policy=None if baseline else policy,
          wheel_residual_exact_zero=True,
          episode_wide_safety=True,
        ))
  finally:
    env.close()
  cells, repeat_cells = aggregate_trials(
    trials,
    heights=(height,),
    expected_repeats=repeats,
    expected_envs_per_height=envs,
  )
  return {
    "height_m": height,
    "mode": "zero_residual" if baseline else "roll_assist",
    "cells": cells,
    "repeat_cells": repeat_cells,
    "trials": trials,
  }


def _make_flat_env_cfg(num_envs: int) -> Any:
  cfg = load_env_cfg(ROLL_ASSIST_TASK_ID, play=True)
  cfg.seed = 1
  cfg.auto_reset = True
  cfg.scene.num_envs = num_envs
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      seed=1,
      curriculum=True,
      size=FLAT_EVALUATION_SIZE_M,
      num_rows=1,
      num_cols=1,
      difficulty_range=(0.0, 0.0),
      sub_terrains={"flat_zero_height": flat(proportion=1.0)},
    ),
    max_init_terrain_level=0,
    num_envs=num_envs,
  )
  cfg.roll_assist_flat_env_count = num_envs
  cfg.scene.terrain.num_envs = num_envs
  cfg.curriculum = {}
  cfg.metrics = {}
  cfg.episode_length_s = 1.0e9
  reset = cfg.events.get("reset_root_to_roll_assist")
  if reset is None:
    raise RuntimeError("RollAssist flat evaluator reset is missing.")
  reset.params["flat_env_count"] = num_envs
  reset.params["x_offset_from_origin_m"] = 0.0
  twist = cfg.commands["twist"]
  posture = cfg.commands["posture"]
  twist.flat_env_count = num_envs
  posture.flat_env_count = num_envs
  action = cfg.actions["hybrid_wheel_leg"]
  action.action_mask = ROLL_ASSIST_ACTION_MASK
  action.action_scales = ROLL_ASSIST_ACTION_SCALES
  action.__post_init__()
  return cfg


def _run_flat_cell(
  env: ManagerBasedRlEnv,
  *,
  policy: Any,
  height: float,
  pitch: float,
  vx: float,
  settle_steps: int,
  measure_steps: int,
) -> dict[str, float]:
  env.reset()
  observation = env.get_observations()
  robot = env.scene["robot"]
  action_term = env.action_manager.get_term("hybrid_wheel_leg")
  heights: list[torch.Tensor] = []
  pitches: list[torch.Tensor] = []
  pitch_rates: list[torch.Tensor] = []
  lin_x: list[torch.Tensor] = []
  wheel_rate_sq: list[torch.Tensor] = []
  previous_wheel_targets: torch.Tensor | None = None
  terminated_total = 0
  non_wheel_total = 0
  for step in range(settle_steps + measure_steps):
    from hoppertrex_mjlab.scripts.evaluate_hybrid_c1_flat_gate import _force_commands
    _force_commands(env, vx=vx, height=height, pitch=pitch)
    observation = env.get_observations()
    with torch.inference_mode():
      actions = policy(observation)
    observation, _rewards, terminated, _timeouts, _extras = env.step(actions)
    _force_commands(env, vx=vx, height=height, pitch=pitch)
    terminated_total += int(terminated.sum().item())
    non_wheel_total += int(
      env.termination_manager.get_term("non_wheel_ground_contact").sum().item()
    )
    if float(action_term.applied_residual[:, :2].abs().max().item()) != 0.0:
      raise RuntimeError("RollAssist flat evaluator observed a wheel residual.")
    wheel_targets = action_term.wheel_targets.detach().clone()
    if step < settle_steps:
      previous_wheel_targets = wheel_targets
      continue
    data = robot.data
    gravity = data.projected_gravity_b
    actual_pitch = torch.atan2(
      gravity[:, 0], torch.clamp(-gravity[:, 2], min=1.0e-6)
    )
    heights.append(data.root_link_pos_w[:, 2].detach().cpu())
    pitches.append(actual_pitch.detach().cpu())
    pitch_rates.append(data.root_link_ang_vel_b[:, 1].abs().detach().cpu())
    lin_x.append(data.root_link_lin_vel_b[:, 0].detach().cpu())
    if previous_wheel_targets is not None:
      wheel_rate_sq.append(
        torch.sum(torch.square(wheel_targets - previous_wheel_targets), dim=1).cpu()
      )
    previous_wheel_targets = wheel_targets
  height_error = torch.stack(heights) - height
  pitch_error = torch.stack(pitches) - pitch
  pitch_rate_abs = torch.stack(pitch_rates)
  lin = torch.stack(lin_x)
  wheel_rate = torch.stack(wheel_rate_sq)
  return {
    "target_height": float(height),
    "target_pitch": float(pitch),
    "vx_command": float(vx),
    "height_rmse": float(torch.sqrt(torch.mean(torch.square(height_error))).item()),
    "pitch_rmse": float(torch.sqrt(torch.mean(torch.square(pitch_error))).item()),
    "pitch_error_abs_p95": float(torch.quantile(pitch_error.abs(), 0.95).item()),
    "pitch_rate_abs_p99": float(torch.quantile(pitch_rate_abs, 0.99).item()),
    "mean_actual_lin_x": float(lin.mean().item()),
    "velocity_error_abs": float(abs(lin.mean().item() - vx)),
    "wheel_target_rate_rms": float(torch.sqrt(torch.mean(wheel_rate)).item()),
    "non_wheel_contact_rate": float(non_wheel_total > 0),
    "terminated_events": float(terminated_total),
    "safety_window_steps": float(settle_steps + measure_steps),
  }


def _roll_assist_stage5_env_cfg(env_cfg: Any) -> None:
  """Evaluate RollAssist actor bytes in the original flat Stage5 domains."""

  action = env_cfg.actions["hybrid_wheel_leg"]
  action.action_mask = ROLL_ASSIST_ACTION_MASK
  action.action_scales = ROLL_ASSIST_ACTION_SCALES
  action.dynamic_stair_maneuver = None
  action.dynamic_stair_request_command_name = None
  action.dynamic_stair_left_sensor_name = None
  action.dynamic_stair_right_sensor_name = None
  action.stair_trigger_sensor_name = None
  action.stair_mode_freezes_leg_reference = False
  action.stair_mode_forced = False
  action.__post_init__()


@contextlib.contextmanager
def _roll_assist_stage5_session(
  *, checkpoint: Path | None, args: SimpleNamespace, play: bool
) -> Any:
  """Stage5 flat session with RollAssist leg rights and RollAssist actor bytes."""

  env_cfg = load_env_cfg(STAGE5_TASK_ID, play=play)
  _roll_assist_stage5_env_cfg(env_cfg)
  agent_cfg = load_rl_cfg(ROLL_ASSIST_TASK_ID)
  env_cfg.seed = args.seed
  agent_cfg.seed = args.seed
  env_cfg.episode_length_s = args.episode_length_s
  env_cfg.scene.num_envs = args.num_envs
  if env_cfg.scene.terrain is not None:
    env_cfg.scene.terrain.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  owner = None
  try:
    if checkpoint is None:
      action_dim = int(wrapped.unwrapped.action_manager.total_action_dim)

      def policy(obs: Any) -> torch.Tensor:
        return torch.zeros(
          (obs.shape[0], action_dim), dtype=obs.dtype, device=obs.device
        )
    else:
      owner = HybridOnPolicyRunner(
        wrapped, asdict(agent_cfg), device=args.device
      )
      owner.load(
        str(checkpoint), load_cfg={"actor": True}, strict=True,
        map_location=args.device,
      )
      policy = owner.get_inference_policy(device=args.device)
    yield wrapped, policy, env_cfg
  finally:
    wrapped.close()


def _collect_roll_assist_stage5_scenarios(
  *, checkpoint: Path, args: SimpleNamespace
) -> list[dict[str, object]]:
  """Use the existing Stage5 scenario functions with a RollAssist actor session."""

  # _collect_scenarios centralizes many private calls around _policy_session.
  # Temporarily replace only that module-global session within this synchronous
  # call; no process/thread concurrency is used by the formal evaluator.
  import hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate as gate

  original = gate._policy_session

  @contextlib.contextmanager
  def session(**kwargs: Any) -> Any:
    with _roll_assist_stage5_session(
      checkpoint=kwargs.get("checkpoint"), args=kwargs["args"],
      play=kwargs["play"],
    ) as values:
      yield values

  gate._policy_session = session
  try:
    return _collect_scenarios(
      suite="robust", task=STAGE5_TASK_ID,
      checkpoint=checkpoint, args=args,
    )
  finally:
    gate._policy_session = original


def _hybrid_gate_args(*, device: str, profile: str) -> SimpleNamespace:
  screen = profile == "screen"
  return SimpleNamespace(
    seed=1,
    num_envs=16,
    steps=1000 if screen else 3000,
    warmup_steps=300,
    window_steps=300 if screen else 800,
    progress_interval=250 if screen else 500,
    episode_length_s=1.0e9,
    device=device,
    ablate_leg_residuals=False,
  )


def _roll_assist_recovery_comparison(
  checkpoint_path: Path, *, args: SimpleNamespace
) -> dict[str, float]:
  center = _posture_targets_from_cfg(STAGE5_TASK_ID)
  with _roll_assist_stage5_session(
    checkpoint=checkpoint_path, args=args, play=True
  ) as (wrapped, policy, _env_cfg):
    candidate = _run_recovery_scenario(
      wrapped=wrapped, policy=policy, args=args,
      target_height=center[0], target_pitch=center[1],
    )
  with _roll_assist_stage5_session(
    checkpoint=None, args=args, play=True
  ) as (wrapped, policy, _env_cfg):
    baseline = _run_recovery_scenario(
      wrapped=wrapped, policy=policy, args=args,
      target_height=center[0], target_pitch=center[1],
    )
  return {
    **{f"candidate_{key}": value for key, value in candidate.items()},
    **{f"baseline_{key}": value for key, value in baseline.items()},
  }


def _evaluate_stage5_retention(
  checkpoint_path: Path, *, device: str, profile: str
) -> dict[str, Any]:
  """Run the existing Stage5 no-regression suite against the same actor bytes."""

  if profile == "smoke":
    return {
      "profile": "smoke",
      "passed": False,
      "evidence_eligible": False,
      "reason": "smoke_uses_short_flat_and_stair_rollouts_only",
    }
  args = _hybrid_gate_args(device=device, profile=profile)
  scenarios = _collect_roll_assist_stage5_scenarios(
    checkpoint=checkpoint_path, args=args
  )
  scenarios.append({
    "name": "stage5_recovery_center_8x",
    "kind": "recovery",
    "kick_scale": 8.0,
    "metrics": _roll_assist_recovery_comparison(
      checkpoint_path, args=args
    ),
  })
  baseline = _collect_stage4_reference_from_baseline(
    task=STAGE5_TASK_ID,
    args=args,
  )
  stage4_reference = {"tracking_error": float(baseline["tracking_error"])}
  checks = evaluate_capability_suite(
    "robust", scenarios, profile=profile,
    stage4_reference=stage4_reference,
  )
  return {
    "profile": profile,
    "scenarios": scenarios,
    "checks": [
      {
        "name": check.name,
        "value": check.value,
        "operator": check.operator,
        "limit": check.limit,
        "pass": check.passed,
        "scenario": check.scenario,
        "source": check.source,
      }
      for check in checks
    ],
    "passed": all(check.passed for check in checks),
    "stage4_reference": stage4_reference,
  }


def _evaluate_flat_retention(policy: Any, *, device: str, profile: str) -> dict[str, Any]:
  num_envs = 1 if profile == "smoke" else FORMAL_ENVS_PER_CELL
  settle = 2 if profile == "smoke" else 100
  measure = 3 if profile == "smoke" else 200
  cells = evaluation_cells(0.05)
  if profile == "smoke":
    cells = cells[:1]
  cfg = _make_flat_env_cfg(num_envs)
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    rows = [
      _run_flat_cell(
        env, policy=policy, height=height, pitch=pitch, vx=vx,
        settle_steps=settle, measure_steps=measure,
      )
      for height, pitch, vx in cells
    ]
  finally:
    env.close()
  verdict = aggregate_candidate(rows, REGISTERED_CAPS)
  return {"cells": rows, **verdict}


def _successes_by_card(arm: Mapping[str, Any]) -> list[int]:
  cells = arm.get("cells")
  if not isinstance(cells, Sequence):
    raise TypeError("RollAssist stair arm cells must be a sequence.")
  by_name = {str(cell["posture_card"]): int(cell["successes"]) for cell in cells}
  return [by_name[str(card["name"])] for card in POSTURE_CARDS]


def _trials_by_pair(
  candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> tuple[list[float], list[float]]:
  def keyed(arm: Mapping[str, Any]) -> dict[tuple[str, int, int], Mapping[str, Any]]:
    rows = arm.get("trials")
    if not isinstance(rows, Sequence):
      raise TypeError("RollAssist stair arm trials must be a sequence.")
    keyed_rows = {
      (str(row["posture_card"]), int(row["repeat"]), int(row["env_id"])): row
      for row in rows
    }
    if len(keyed_rows) != len(rows):
      raise ValueError("RollAssist paired arm contains duplicate trial keys.")
    return keyed_rows
  cand = keyed(candidate)
  base = keyed(baseline)
  if set(cand) != set(base):
    raise ValueError("RollAssist paired resets/trials drifted.")
  expected_pairs = FORMAL_ENVS_PER_CELL * FORMAL_REPEATS * len(POSTURE_CARDS)
  if len(cand) != expected_pairs:
    raise ValueError(
      f"Formal RollAssist pairing requires {expected_pairs} trials per arm."
    )
  order = sorted(cand)
  return (
    [float(cand[key]["max_progress_past_face_m"]) for key in order],
    [float(base[key]["max_progress_past_face_m"]) for key in order],
  )


def evaluate(
  *,
  checkpoint: Mapping[str, Any],
  roll_boundary_path: Path,
  reward_calibration_path: Path,
  device: str,
  profile: str,
) -> dict[str, Any]:
  if profile in ("formal", "screen") and device != FORMAL_DEVICE:
    raise ValueError("Evidence-bearing RollAssist evaluation is pinned to cuda:0.")
  envelope = validate_checkpoint_envelope(checkpoint, verify_file=True)
  verdict = load_roll_boundary_verdict(
    roll_boundary_path, expected_git_sha=_git_sha()
  )
  training = envelope["training"]
  if training["r0_sha256"] != verdict["file_sha256"]:
    raise ValueError("RollAssist checkpoint/evaluator R0 binding drifted.")
  if training["git_sha"] != verdict["git_sha"]:
    raise ValueError("RollAssist checkpoint and R0 Git bindings differ.")
  validate_checkpoint_reward_binding(training, reward_calibration_path)
  owner, policy = _load_policy(envelope, device=device)
  if profile == "formal" and envelope["completed_updates"] < 51:
    raise ValueError("Formal RollAssist evaluation requires a trained K=3 checkpoint.")
  del owner
  hpass = float(verdict["hpass_m"])
  hnext = float(verdict["hnext_m"])
  stage5_retention = _evaluate_stage5_retention(
    Path(str(envelope["checkpoint_file"])), device=device, profile=profile
  )
  flat_result = _evaluate_flat_retention(policy, device=device, profile=profile)
  hpass_candidate = _evaluate_stair_arm(
    policy=policy, height=hpass, device=device, baseline=False, profile=profile
  )
  hnext_candidate = _evaluate_stair_arm(
    policy=policy, height=hnext, device=device, baseline=False, profile=profile
  )
  hnext_baseline = _evaluate_stair_arm(
    policy=policy, height=hnext, device=device, baseline=True, profile=profile
  )
  candidate_progress, baseline_progress = _trials_by_pair(
    hnext_candidate, hnext_baseline
  )
  hnext_cells = hnext_candidate["cells"]
  unsafe = {
    "terminations": sum(int(cell["terminated_trials"]) for cell in hnext_cells),
    "non_wheel_contacts": sum(int(cell["non_wheel_contact_trials"]) for cell in hnext_cells),
    "bilateral_airborne": sum(int(cell["bilateral_airborne_trials"]) for cell in hnext_cells),
  }
  candidate_successes = sum(_successes_by_card(hnext_candidate))
  baseline_successes = sum(_successes_by_card(hnext_baseline))
  wheel_residual_abs_max = max(
    float(row["wheel_residual_abs_max"])
    for arm in (hpass_candidate, hnext_candidate)
    for row in arm["trials"]
  )
  if profile != "formal":
    continuation = None
    final = None
  else:
    continuation = continuation_gate(
      flat_retention_passed=bool(stage5_retention["passed"]),
      hpass_card_successes=_successes_by_card(hpass_candidate),
      hnext_terminations=unsafe["terminations"],
      hnext_non_wheel_contacts=unsafe["non_wheel_contacts"],
      hnext_bilateral_airborne=unsafe["bilateral_airborne"],
      wheel_residual_abs_max=wheel_residual_abs_max,
      hnext_candidate_successes=candidate_successes,
      hnext_baseline_successes=baseline_successes,
      paired_candidate_progress=candidate_progress,
      paired_baseline_progress=baseline_progress,
    )
    final = final_expansion_gate(
      hnext_card_successes=_successes_by_card(hnext_candidate),
      safety_gate_passed=all(value == 0 for value in unsafe.values()),
      wheel_residual_abs_max=wheel_residual_abs_max,
    )
  screen = None
  if profile == "screen":
    screen = {
      "flat_retention_passed": bool(stage5_retention["passed"]),
      "hpass_retained": all(
        value >= FORMAL_ENVS_PER_CELL - 1
        for value in _successes_by_card(hpass_candidate)
      ),
      "hnext_safe": all(value == 0 for value in unsafe.values()),
      "wheel_residual_exact_zero": wheel_residual_abs_max == 0.0,
    }
  payload = {
    "schema_version": SCHEMA_VERSION,
    "kind": EVALUATION_KIND,
    "evidence_eligible": profile == "formal",
    "profile": profile,
    "task": ROLL_ASSIST_TASK_ID,
    "git_sha": repository_git_sha(),
    "device": device,
    "checkpoint": envelope,
    "roll_boundary_file": str(roll_boundary_path.resolve()),
    "roll_boundary_sha256": verdict["file_sha256"],
    "reward_calibration_file": str(reward_calibration_path.resolve()),
    "reward_calibration_sha256": file_sha256(reward_calibration_path.resolve()),
    "hpass_m": hpass,
    "hnext_m": hnext,
    "stage5_retention": stage5_retention,
    "flat_c1_diagnostic": flat_result,
    "hpass_candidate": hpass_candidate,
    "hnext_candidate": hnext_candidate,
    "hnext_zero_residual": hnext_baseline,
    "paired_candidate_progress": candidate_progress,
    "paired_baseline_progress": baseline_progress,
    "wheel_residual_abs_max": wheel_residual_abs_max,
    "hnext_unsafe": unsafe,
    "screen": screen,
    "continuation": continuation,
    "final": final,
    # Recovery retention is still enforced by the inherited Stage5 robust gate,
    # but this evaluator does not yet collect paired per-reset recovery-time
    # vectors.  Fail closed on the separate "recovery is faster" paper claim.
    "recovery_claim": {
      "eligible": False,
      "classification": "RECOVERY_CLAIM_NOT_EVALUATED",
      "reason": "paired_recovery_time_bootstrap_not_implemented",
    },
  }
  payload["evaluation_sha256"] = canonical_json_sha256(payload)
  return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="command", required=True)
  envelope = sub.add_parser("checkpoint-envelope")
  envelope.add_argument("--checkpoint-file", type=Path, required=True)
  envelope.add_argument("--output", type=Path, required=True)
  validate = sub.add_parser("validate-checkpoint")
  validate.add_argument("--checkpoint-envelope", type=Path, required=True)
  validate.add_argument("--verify-file", action="store_true")
  validate.add_argument("--output", type=Path, required=True)
  screen = sub.add_parser("screen-envelope")
  screen.add_argument("--checkpoint-envelope", type=Path, required=True)
  screen.add_argument("--screen-json", type=Path, required=True)
  screen.add_argument("--output", type=Path, required=True)
  live = sub.add_parser("live")
  live.add_argument("--checkpoint-envelope", type=Path, required=True)
  live.add_argument("--roll-boundary", type=Path, required=True)
  live.add_argument("--reward-calibration", type=Path, required=True)
  live.add_argument("--device", default=FORMAL_DEVICE)
  live.add_argument("--profile", choices=("formal", "screen", "smoke"), default="formal")
  live.add_argument("--output", type=Path, required=True)
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  if args.command == "checkpoint-envelope":
    result = checkpoint_envelope(args.checkpoint_file)
  elif args.command == "validate-checkpoint":
    result = validate_checkpoint_envelope(
      _read(args.checkpoint_envelope), verify_file=args.verify_file
    )
  elif args.command == "screen-envelope":
    result = screen_checkpoint_envelope(
      _read(args.checkpoint_envelope), _read(args.screen_json)
    )
  else:
    result = evaluate(
      checkpoint=_read(args.checkpoint_envelope),
      roll_boundary_path=args.roll_boundary,
      reward_calibration_path=args.reward_calibration,
      device=args.device,
      profile=args.profile,
    )
  _atomic_write(args.output, result)
  print(f"[roll-assist] kind={result.get('kind')}")
  print(f"[roll-assist] output={args.output}")


if __name__ == "__main__":
  main()
