"""Separate C1 pilot task and runner; frozen Hybrid ladder remains unchanged."""
from __future__ import annotations

import dataclasses
import enum
import functools
import os
from pathlib import Path

import torch
from mjlab.rl import MjlabOnPolicyRunner

from hoppertrex_mjlab.hybrid.config import HYBRID_STAGES
from hoppertrex_mjlab.hybrid.recovery_pilot import (
  ACTION_SCALES, ACTION_STD, ARTIFACTS, INFO_KEY, PROTOCOL_HASH,
  SCHEDULE_HASH, TASK_ID, TRAIN_SEEDS, digest, identity, require_approval,
  validate_checkpoint_record, verify_artifacts,
)
from hoppertrex_mjlab.hybrid.runner import zero_initialize_actor_output
from hoppertrex_mjlab.tasks.agents.hoppertrex_balance_rsl_rl_ppo import hoppertrex_hybrid_ppo_runner_cfg
from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import make_hoppertrex_hybrid_env_cfg

REPO = Path(__file__).resolve().parents[3]


def make_env_cfg(*, play: bool):
  # Explicit paths, never inherit a previous campaign's environment overrides.
  verify_artifacts(REPO)
  kwargs = {key + "_path": REPO / value[0] for key, value in ARTIFACTS.items()}
  override = os.environ.pop("HOPPERTREX_HYBRID_LEG_RESIDUAL_SCALE", None)
  try:
    cfg = make_hoppertrex_hybrid_env_cfg(stage=5, play=play, **kwargs)
  finally:
    if override is not None:
      os.environ["HOPPERTREX_HYBRID_LEG_RESIDUAL_SCALE"] = override
  cfg.scene.num_envs = 32 if play else 256
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.num_envs = cfg.scene.num_envs
  cfg.seed = 11
  if play:
    cfg.episode_length_s = 1.0e9
  cfg.recovery_pilot_task = TASK_ID
  cfg.recovery_pilot_mode = "evaluation" if play else "unapproved"
  action = cfg.actions["hybrid_wheel_leg"]
  if action.controller_gain_hash != SCHEDULE_HASH or tuple(action.action_scales) != ACTION_SCALES:
    raise ValueError("C1 pilot controller/action contract mismatch")
  if not all((action.controller_qualified, action.yaw_calibration_qualified,
              action.posture_map_qualified, action.station_calibration_qualified)):
    raise ValueError("C1 pilot five-artifact stack is not qualified")
  if cfg.decimation != 4 or cfg.sim.mujoco.timestep != 0.005:
    raise ValueError("Pilot physics/control cadence drifted")
  if not play:
    push = cfg.events["push_robot"]
    if push.params["velocity_range"] != {"x": (-0.32, 0.32), "pitch": (-0.48, 0.48)}:
      raise ValueError("Stage5 push consumer changed")
    if push.interval_range_s != HYBRID_STAGES[5].push_interval_s:
      raise ValueError("Push interval changed")
  return cfg


def make_runner_cfg():
  cfg = hoppertrex_hybrid_ppo_runner_cfg((True,) * 6)
  cfg.experiment_name = "hoppertrex_recovery_c1_pilot_v1"
  cfg.seed = 11
  cfg.resume = False
  cfg.num_steps_per_env = 24
  cfg.max_iterations = 1000
  cfg.save_interval = 100
  cfg.actor.distribution_cfg["std_type"] = "scalar"
  return cfg


def normalized(value):
  """Machine-independent complete configuration, including callable identity."""
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    return {f.name: normalized(getattr(value, f.name)) for f in dataclasses.fields(value)}
  if isinstance(value, dict):
    return {str(k): normalized(v) for k, v in value.items()}
  if isinstance(value, (tuple, list)):
    return [normalized(v) for v in value]
  if isinstance(value, slice):
    return {"slice": [value.start, value.stop, value.step]}
  if isinstance(value, Path):
    return normalized(str(value))
  if isinstance(value, str):
    return value.replace(str(REPO), "<REPO>").replace(REPO.as_posix(), "<REPO>").replace("\\", "/")
  if value is None or isinstance(value, (int, float, bool)):
    return value
  if isinstance(value, enum.Enum):
    return normalized(value.value)
  if isinstance(value, functools.partial):
    return {"callable": normalized(value.func), "args": normalized(value.args), "kwargs": normalized(value.keywords)}
  if callable(value):
    return {"callable": value.__module__ + "." + value.__qualname__}
  if hasattr(value, "tolist"):
    return normalized(value.tolist())
  raise TypeError(f"Unsupported config field: {type(value)}")


def configuration():
  train, play, agent = make_env_cfg(play=False), make_env_cfg(play=True), make_runner_cfg()
  return {"protocol_sha256": PROTOCOL_HASH, "train_env": normalized(train),
          "play_env": normalized(play), "agent": normalized(agent),
          "initial_std_override": list(ACTION_STD),
          "randomization_audit": {
            "training_events": list(train.events), "play_events": list(play.events),
            "claim": "Stage5 configured events only; randomization_level alone is not an implementation"}}


class RecoveryPilotRunner(MjlabOnPolicyRunner):
  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    self._pilot_cfg = env.unwrapped.cfg
    if getattr(self._pilot_cfg, "recovery_pilot_task", None) != TASK_ID:
      raise ValueError("Recovery runner requires its dedicated task")
    self._mode = self._pilot_cfg.recovery_pilot_mode
    self._seed = int(train_cfg["seed"])
    self._identity = identity(REPO, __import__("subprocess").check_output(
      ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
      local_check=self._mode == "cpu_smoke")
    self._config_hash = digest(configuration())
    if self._mode == "training":
      if device != "cuda:0" or self._seed not in TRAIN_SEEDS or train_cfg.get("resume"):
        raise ValueError("Only fresh CUDA pilot seeds 11/12/13 can train")
      require_approval(Path(self._pilot_cfg.recovery_pilot_campaign), self._identity)
      expected = dataclasses.asdict(make_runner_cfg())
      expected["seed"] = self._seed
      if normalized(train_cfg) != normalized(expected) or env.num_envs != 256:
        raise ValueError("Training configuration differs from frozen pilot")
    super().__init__(env, train_cfg, log_dir, device)
    actor = self.alg.get_policy()
    zero_initialize_actor_output(actor, label="C1 Recovery Pilot")
    std = actor.state_dict().get("distribution.std_param")
    if std is None or tuple(std.shape) != (6,):
      raise ValueError("Expected six independent Gaussian standard deviations")
    with torch.no_grad():
      actor.distribution.std_param.copy_(torch.tensor(ACTION_STD, device=device))
    self._base_log = self.logger.log
    self.logger.log = self._log_and_save

  def _log_and_save(self, *args, **kwargs):
    self._base_log(*args, **kwargs)
    actor = self.alg.get_policy()
    if any(not torch.isfinite(p).all() for p in actor.parameters()):
      raise ValueError("Non-finite actor parameters")
    if (actor.distribution.std_param <= 0).any():
      raise ValueError("Non-positive exploration standard deviation")
    if (self.current_learning_iteration + 1) % 100 == 0 and self.logger.log_dir:
      self.save(str(Path(self.logger.log_dir) / f"model_{self.current_learning_iteration}.pt"))

  def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    if self.current_learning_iteration != 0:
      raise ValueError("Pilot does not resume or extend runs")
    if self._mode == "training":
      if num_learning_iterations != 1000 or init_at_random_ep_len:
        raise ValueError("Pilot training is exactly 1000 updates")
    elif self._mode == "cpu_smoke":
      if num_learning_iterations != 2 or self.device != "cpu":
        raise ValueError("CPU implementation smoke is exactly two updates")
    else:
      raise ValueError("Use the approved pilot wrapper, not generic train.py")
    return super().learn(num_learning_iterations, init_at_random_ep_len)

  def save(self, path, infos=None):
    completed = int(self.current_learning_iteration) + 1
    if completed % 100 != 0 or self._mode != "training":
      return  # Base RSL-RL's zero-based save calls are not our completed-update grid.
    if Path(path).exists():
      previous = torch.load(path, map_location="cpu", weights_only=True).get("infos", {}).get(INFO_KEY, {})
      if previous.get("completed_updates") != completed or previous.get("git_sha") != self._identity["git_sha"]:
        raise FileExistsError("Existing checkpoint is not the duplicate final save")
      return  # Final base save duplicates the completed-update hook, no overwrite.
    record = {**self._identity, "training_seed": self._seed,
      "completed_updates": completed, "config_sha256": self._config_hash,
      "initial_std": list(ACTION_STD), "zero_initialized_deterministic_mean": True}
    super().save(path, {**(infos or {}), INFO_KEY: record})

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    if self._mode != "evaluation":
      raise ValueError("Fresh pilot training must not load a checkpoint")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    record = payload.get("infos", {}).get(INFO_KEY, {})
    validate_checkpoint_record(record, self._identity, record.get("training_seed"), self._config_hash)
    if payload.get("iter") != 999:
      raise ValueError("Only final model_999 is eligible")
    self.alg.load(payload, {"actor": True}, strict)
    return payload["infos"]
