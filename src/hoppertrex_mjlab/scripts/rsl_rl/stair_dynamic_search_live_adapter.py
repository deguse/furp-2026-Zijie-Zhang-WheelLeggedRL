# ruff: noqa: TRY004
"""Delayed-import MjLab adapter for the v3 feedforward screen/CEM search."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

TRIGGER_QUALIFICATION_PATH_ENV = "HOPPERTREX_DYNAMIC_STAIR_TRIGGER_QUALIFICATION_PATH"
STAGE5_CHECKPOINT_PATH_ENV = "HOPPERTREX_DYNAMIC_STAIR_STAGE5_CHECKPOINT_PATH"
SETTLE_STEPS = 100
DRIVE_STEPS = 500
STABLE_STEPS = 25
CROSS_DISTANCE_M = 0.40


def _trigger_qualification(expected_sha256: object) -> dict[str, object]:
  value = os.environ.get(TRIGGER_QUALIFICATION_PATH_ENV)
  if not value:
    raise ValueError(
      f"Set {TRIGGER_QUALIFICATION_PATH_ENV} to the live per-wheel qualification JSON."
    )
  path = Path(value).resolve()
  payload_bytes = path.read_bytes()
  digest = hashlib.sha256(payload_bytes).hexdigest()
  if digest != expected_sha256:
    raise ValueError("Per-wheel qualification file SHA256 does not match request.")
  payload = json.loads(payload_bytes.decode("utf-8"))
  if not isinstance(payload, Mapping):
    raise ValueError("Per-wheel qualification JSON must contain an object.")
  # The search core owns the exact schema/rule validation.  The adapter only
  # binds the bytes and returns the evidence section it measured live.
  evidence = payload.get("qualification", payload)
  if not isinstance(evidence, Mapping):
    raise ValueError("Per-wheel qualification evidence is missing.")
  result = dict(evidence)
  result["evidence_sha256"] = digest
  return result


def _stage5_checkpoint_path(expected_sha256: object) -> Path:
  if (
    not isinstance(expected_sha256, str)
    or len(expected_sha256) != 64
    or any(char not in "0123456789abcdef" for char in expected_sha256)
  ):
    raise ValueError("Expected Stage5 checkpoint SHA256 is invalid.")
  value = os.environ.get(STAGE5_CHECKPOINT_PATH_ENV)
  if not value:
    raise ValueError(f"Set {STAGE5_CHECKPOINT_PATH_ENV} to the selected Stage5 checkpoint.")
  path = Path(value).resolve()
  if not path.is_file():
    raise FileNotFoundError(f"Stage5 checkpoint does not exist: {path}.")
  digest = hashlib.sha256(path.read_bytes()).hexdigest()
  if digest != expected_sha256:
    raise ValueError("Stage5 policy checkpoint SHA256 does not match search bindings.")
  return path


def _load_stage5_policy(
  env: Any, *, expected_sha256: object, device: str
) -> tuple[Any, Any, Any]:
  """Expand only zero input columns and load the selected Stage5 actor."""

  from dataclasses import asdict

  import torch
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg

  from hoppertrex_mjlab.hybrid.stair_dynamic import DYNAMIC_STAIR_TASK_ID
  from hoppertrex_mjlab.hybrid.stair_dynamic_contract import (
    DYNAMIC_STAIR_ACTOR_WIDTH,
    DYNAMIC_STAIR_STAGE5_ACTOR_WIDTH,
  )
  from hoppertrex_mjlab.scripts.rsl_rl.migrate_stage5_to_stair_dynamic import (
    expand_observation_input,
  )

  path = _stage5_checkpoint_path(expected_sha256)
  loaded = torch.load(path, map_location="cpu", weights_only=False)
  if not isinstance(loaded, Mapping):
    raise ValueError("Stage5 checkpoint root must be a mapping.")
  actor_state = loaded.get("actor_state_dict")
  if not isinstance(actor_state, Mapping):
    raise ValueError("Stage5 checkpoint actor_state_dict is missing.")
  expanded_actor, _first_layer = expand_observation_input(
    actor_state,
    source_width=DYNAMIC_STAIR_STAGE5_ACTOR_WIDTH,
    target_width=DYNAMIC_STAIR_ACTOR_WIDTH,
    name="Stage5 search actor",
  )
  agent_cfg = load_rl_cfg(DYNAMIC_STAIR_TASK_ID)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
  runner.alg.load(
    {"actor_state_dict": expanded_actor},
    {"actor": True},
    strict=True,
  )
  policy = runner.get_inference_policy(device=device)
  return wrapped, runner, policy


def _dynamic_danger(action: Any, terminated: Any, timed_out: Any):
  """Treat every FSM/actuator unsafe latch as danger, not only env reset."""

  return (
    terminated.bool()
    | timed_out.bool()
    | (action.dynamic_traversal_mode == 3)
    | (action.dynamic_abort_code != 0)
    | action.dynamic_episode_unsafe.bool()
    | action.dynamic_target_saturation.bool()
  )


def _maneuver(family: str):
  from hoppertrex_mjlab.hybrid.stair_dynamic import (
    DynamicLiftMode,
    DynamicStairManeuver,
  )

  return DynamicStairManeuver(
    lift_mode=DynamicLiftMode(family),
    split_amplitude_rad=0.035,
    lift_amplitude_rad=0.045,
    trailing_delay_s=0.20,
    drive_feedforward_radps=1.0,
    source="live-search-candidate",
  )


def _configure_env(*, family: str, num_envs: int, device: str):
  from mjlab.envs import ManagerBasedRlEnv

  from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
    make_stair_dynamic_env_cfg,
  )

  cfg = make_stair_dynamic_env_cfg(
    play=True,
    initial_upper_height_m=0.01,
    dynamic_maneuver=_maneuver(family),
  )
  cfg.seed = 1
  cfg.scene.num_envs = num_envs
  if cfg.scene.terrain is None:
    raise RuntimeError("StairDynamic search requires generated terrain.")
  cfg.scene.terrain.num_envs = num_envs
  for name in ("stair_request", "twist", "posture"):
    command = cfg.commands[name]
    if hasattr(command, "flat_env_count"):
      command.flat_env_count = 0
  reset = cfg.events.get("reset_root_to_stair_dynamic")
  if reset is None:
    raise RuntimeError("StairDynamic search reset event is missing.")
  reset.params["flat_env_count"] = 0
  curriculum = cfg.curriculum.get("stair_dynamic_height")
  if curriculum is None:
    raise RuntimeError("StairDynamic search curriculum is missing.")
  curriculum.params["flat_env_count"] = 0
  cfg.metrics = {}
  cfg.episode_length_s = 20.0
  return ManagerBasedRlEnv(cfg=cfg, device=device)


def _force_commands(env: Any, *, vx: float, stair_request: bool) -> None:
  from hoppertrex_mjlab.scripts.rsl_rl.evaluate_fixed_command import (
    _force_command,
    _force_static_posture,
  )
  from hoppertrex_mjlab.tasks.hoppertrex_balance_task import ROOT_HEIGHT_TARGET

  _force_command(env, vx, 0.0)
  _force_static_posture(env, (ROOT_HEIGHT_TARGET, 0.0))
  request = env.command_manager.get_term("stair_request")
  request._command[:, 0] = float(stair_request)


def _score_batch(
  *,
  family: str,
  candidates: Sequence[Sequence[float]],
  replicates: int,
  device: str,
  expected_stage5_checkpoint_sha256: object,
  roll_only: bool = False,
) -> list[dict[str, float | int]]:
  import torch

  if replicates != 8 or not candidates:
    raise ValueError("Live search requires exactly eight replicas per candidate.")
  num_candidates = len(candidates)
  num_envs = num_candidates * replicates
  env = _configure_env(family=family, num_envs=num_envs, device=device)
  try:
    wrapped, policy_owner, policy = _load_stage5_policy(
      env,
      expected_sha256=expected_stage5_checkpoint_sha256,
      device=device,
    )
    # Keep both objects alive for the bound policy and observation TensorDict.
    _ = policy_owner
    action = env.action_manager.get_term("hybrid_wheel_leg")
    parameter_rows = torch.tensor(candidates, device=env.device, dtype=torch.float)
    parameter_rows = parameter_rows.repeat_interleave(replicates, dim=0)
    action.set_dynamic_candidate_parameters(parameter_rows)
    # This is a long deterministic evaluation, not training.  Without
    # no_grad(), autograd retains one policy graph per step.  Releasing the
    # resulting 600-step chain overflows the native Windows stack (0xC00000FD).
    with torch.no_grad():
      for _ in range(SETTLE_STEPS):
        _force_commands(env, vx=0.0, stair_request=False)
        observations = wrapped.get_observations()
        env.step(policy(observations))
      start_x = env.scene["robot"].data.root_link_pos_w[:, 0].clone()
      max_progress = torch.zeros(num_envs, device=env.device)
      stable = torch.zeros(num_envs, device=env.device, dtype=torch.long)
      success = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
      unsafe = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
      peak_pitch = torch.zeros(num_envs, device=env.device)
      energy = torch.zeros(num_envs, device=env.device)
      smoothness = torch.zeros(num_envs, device=env.device)
      previous_ff = torch.zeros(num_envs, 4, device=env.device)
      request = not roll_only
      for _ in range(DRIVE_STEPS):
        _force_commands(env, vx=0.07, stair_request=request)
        observations = wrapped.get_observations()
        _obs, _reward, terminated, timed_out, _extras = env.step(policy(observations))
        danger = _dynamic_danger(action, terminated, timed_out)
        # Danger remains disqualifying even if a trial had crossed/stabilized
        # earlier in the rollout; success must never hide a later ABORT.
        unsafe.logical_or_(danger)
        active = ~unsafe & ~success
        root_x = env.scene["robot"].data.root_link_pos_w[:, 0]
        progress = root_x - start_x
        valid = active
        max_progress.copy_(
          torch.where(valid, torch.maximum(max_progress, progress), max_progress)
        )
        gravity = env.scene["robot"].data.projected_gravity_b
        pitch = torch.atan2(
          gravity[:, 0], torch.clamp(-gravity[:, 2], min=1.0e-6)
        ).abs()
        peak_pitch.copy_(
          torch.where(valid, torch.maximum(peak_pitch, pitch), peak_pitch)
        )
        crossed = valid & (max_progress >= CROSS_DISTANCE_M)
        stable.copy_(
          torch.where(
            crossed & (pitch <= 0.10),
            stable + 1,
            torch.where(valid, torch.zeros_like(stable), stable),
          )
        )
        success.logical_or_(valid & (stable >= STABLE_STEPS))
        ff = action.dynamic_leg_feedforward.detach()
        energy += valid.to(torch.float) * (
          torch.square(action.wheel_targets.detach()).sum(dim=1)
          + torch.square(ff).sum(dim=1)
        )
        smoothness += valid.to(torch.float) * torch.square(
          ff - previous_ff
        ).sum(dim=1)
        previous_ff.copy_(ff)
      scores: list[dict[str, float | int]] = []
      for candidate_index in range(num_candidates):
        begin = candidate_index * replicates
        end = begin + replicates
        scores.append(
          {
            "safe_successes": int(
              (success[begin:end] & ~unsafe[begin:end]).sum().item()
            ),
            "median_progress": float(max_progress[begin:end].median().item()),
            "peak_pitch": float(peak_pitch[begin:end].max().item()),
            "energy": float(energy[begin:end].mean().item()),
            "target_smoothness": float(
              smoothness[begin:end].mean().item()
            ),
            "unsafe_trials": int(unsafe[begin:end].sum().item()),
          }
        )
      return scores
  finally:
    env.close()


def collect(request: Mapping[str, object]) -> Mapping[str, object]:
  """Execute one search request; imports MjLab only inside this function."""

  if request.get("schema_version") != 1:
    raise ValueError("Unsupported StairDynamic search request schema.")
  kind = request.get("kind")
  device = str(request.get("device", "cuda:0"))
  if kind == "family_screen":
    if request.get("replicates") != 8:
      raise ValueError("Family screen requires eight replicas.")
    default = [[0.035, 0.045, 0.20, 1.0]]
    scores = {
      "roll_only": _score_batch(
        family="alternating",
        candidates=default,
        replicates=8,
        device=device,
        expected_stage5_checkpoint_sha256=request.get(
          "expected_stage5_checkpoint_sha256"
        ),
        roll_only=True,
      )[0],
      "synchronized": _score_batch(
        family="synchronized",
        candidates=default,
        replicates=8,
        device=device,
        expected_stage5_checkpoint_sha256=request.get(
          "expected_stage5_checkpoint_sha256"
        ),
      )[0],
      "alternating": _score_batch(
        family="alternating",
        candidates=default,
        replicates=8,
        device=device,
        expected_stage5_checkpoint_sha256=request.get(
          "expected_stage5_checkpoint_sha256"
        ),
      )[0],
    }
    return {
      "scores": scores,
      "trigger_qualification": _trigger_qualification(
        request.get("expected_trigger_qualification_sha256")
      ),
    }
  if kind == "cem_batch":
    family = request.get("family")
    if family not in ("synchronized", "alternating"):
      raise ValueError("CEM family is invalid.")
    candidates = request.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
      raise ValueError("CEM candidates must be a sequence.")
    return {
      "scores": _score_batch(
        family=str(family),
        candidates=candidates,
        replicates=int(request.get("replicates", -1)),
        device=device,
        expected_stage5_checkpoint_sha256=request.get(
          "expected_stage5_checkpoint_sha256"
        ),
      )
    }
  raise ValueError(f"Unsupported StairDynamic search request kind: {kind!r}")


__all__ = [
  "STAGE5_CHECKPOINT_PATH_ENV",
  "TRIGGER_QUALIFICATION_PATH_ENV",
  "collect",
]
