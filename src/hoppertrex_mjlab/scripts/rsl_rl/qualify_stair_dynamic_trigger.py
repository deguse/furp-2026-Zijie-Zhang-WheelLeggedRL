#!/usr/bin/env python3
# ruff: noqa: TRY004
"""Live-qualify the Hybrid-v3 per-wheel loaded-contact trigger.

The contract and mockable backend seam are pure Python. Torch and MjLab are
loaded only by :class:`LiveBackend` when ``collect`` runs in the machine room.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from hoppertrex_mjlab.hybrid.stair_dynamic import (
  DYNAMIC_STAIR_TASK_ID,
  DYNAMIC_STAIR_TRIGGER_FORCE_N,
  DYNAMIC_STAIR_TRIGGER_WINDOW,
)
from hoppertrex_mjlab.scripts.rsl_rl import stair_camp_live_adapter as camp_live
from hoppertrex_mjlab.scripts.rsl_rl.search_stair_dynamic import (
  validate_trigger_qualification,
)

QUALIFICATION_SCHEMA_VERSION = 1
QUALIFICATION_KIND = "stair_dynamic_per_wheel_trigger_qualification"
EVALUATION_SEED = 1
SINGLE_RISER_HEIGHT_M = 0.01
SINGLE_RISER_NUM_ENVS = 16
SINGLE_RISER_SETTLE_STEPS = 100
SINGLE_RISER_DRIVE_STEPS = 500
METRIC_NAME = "abs(F0*nx)"
LEFT_SENSOR_NAME = "stair_dynamic_left_contact"
RIGHT_SENSOR_NAME = "stair_dynamic_right_contact"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class QualificationBackend(Protocol):
  def runtime_evidence(self) -> Mapping[str, object]: ...
  def sensor_identity_evidence(self) -> Mapping[str, object]: ...
  def run_single_riser(self) -> Mapping[str, object]: ...
  def run_false_positive(
    self, domain: str, request: camp_live.GateRequest
  ) -> Mapping[str, object]: ...


def _mapping(value: object, name: str) -> Mapping[str, object]:
  if not isinstance(value, Mapping):
    raise ValueError(f"{name} must be a mapping.")
  return value


def _sequence(value: object, name: str) -> Sequence[object]:
  if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
    raise ValueError(f"{name} must be a sequence.")
  return value


def _integer(value: object, name: str, minimum: int = 0) -> int:
  if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
    raise ValueError(f"{name} must be an integer >= {minimum}.")
  return int(value)


def _finite(value: object, name: str, minimum: float = 0.0) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"{name} must be finite.")
  result = float(value)
  if not math.isfinite(result) or result < minimum:
    raise ValueError(f"{name} must be finite and >= {minimum}.")
  return result


def _digest(value: object, name: str) -> str:
  if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
    raise ValueError(f"{name} must be lowercase SHA256.")
  return value


def evidence_sha256(evidence: Mapping[str, object]) -> str:
  """Hash canonical raw live evidence; this is independently reproducible."""
  try:
    encoded = json.dumps(
      dict(evidence), sort_keys=True, separators=(",", ":"),
      ensure_ascii=True, allow_nan=False,
    ).encode("ascii")
  except (TypeError, ValueError) as exc:
    raise ValueError("Live evidence must be finite JSON data.") from exc
  return hashlib.sha256(encoded).hexdigest()


def _normalize_runtime(value: object) -> dict[str, object]:
  row = _mapping(value, "runtime evidence")
  if set(row) != {"task", "evaluation_seed", "device", "git_sha"}:
    raise ValueError("Runtime evidence schema drifted.")
  device, git_sha = row["device"], row["git_sha"]
  if row["task"] != DYNAMIC_STAIR_TASK_ID or row["evaluation_seed"] != 1:
    raise ValueError("Runtime task or evaluation seed drifted.")
  if (
    not isinstance(device, str) or not device or device != device.strip()
    or not isinstance(git_sha, str) or _GIT_SHA_RE.fullmatch(git_sha) is None
  ):
    raise ValueError("Runtime device or Git SHA is invalid.")
  return dict(row)


def _expected_sensor_identity() -> dict[str, object]:
  common: dict[str, object] = {
    "primary_mode": "geom", "primary_entity": "robot",
    "secondary_mode": "body", "secondary_pattern": "terrain",
    "fields": ["found", "force", "normal"], "reduce": "none", "num_slots": 8,
  }
  return {
    "left": {**common, "sensor_name": LEFT_SENSOR_NAME,
             "primary_pattern": "wheel_left_collision",
             "action_binding": LEFT_SENSOR_NAME},
    "right": {**common, "sensor_name": RIGHT_SENSOR_NAME,
              "primary_pattern": "wheel_right_collision",
              "action_binding": RIGHT_SENSOR_NAME},
    "objects_distinct": True,
  }


def _normalize_sensor_identity(value: object) -> dict[str, object]:
  row = _mapping(value, "sensor identity evidence")
  expected = _expected_sensor_identity()
  if dict(row) != expected:
    raise ValueError("Registered left/right StairDynamic sensor identity drifted.")
  return expected


def _force_window(value: object, name: str) -> list[float]:
  row = _sequence(value, name)
  if len(row) != DYNAMIC_STAIR_TRIGGER_WINDOW:
    raise ValueError(f"{name} must contain exactly three samples.")
  return [_finite(item, f"{name}[{i}]") for i, item in enumerate(row)]


def _trigger_event(
  value: object, index: int, num_envs: int, drive_steps: int
) -> dict[str, object]:
  row = _mapping(value, f"events[{index}]")
  expected = {
    "env_id", "side", "first_qualifying_step", "loaded_rising_step",
    "sensor_force_window_n", "fsm_force_window_n",
    "preceding_sensor_force_n", "loaded_was_false_before",
  }
  if set(row) != expected:
    raise ValueError(f"events[{index}] schema drifted.")
  env_id = _integer(row["env_id"], f"events[{index}].env_id")
  side = row["side"]
  first = _integer(row["first_qualifying_step"], "first_qualifying_step", 2)
  rising = _integer(row["loaded_rising_step"], "loaded_rising_step", 2)
  if env_id >= num_envs or side not in ("left", "right"):
    raise ValueError("Trigger event identity is invalid.")
  if first != rising or rising >= drive_steps:
    raise ValueError("Loaded-contact did not rise on the exact third sample.")
  sensor = _force_window(row["sensor_force_window_n"], "sensor force window")
  fsm = _force_window(row["fsm_force_window_n"], "FSM force window")
  if any(force < DYNAMIC_STAIR_TRIGGER_FORCE_N for force in sensor):
    raise ValueError("Live trigger window contains a sub-threshold force.")
  if any(
    not math.isclose(a, b, rel_tol=0.0, abs_tol=1.0e-5)
    for a, b in zip(sensor, fsm, strict=True)
  ):
    raise ValueError("FSM force does not match its per-wheel sensor.")
  before = row["preceding_sensor_force_n"]
  before = None if before is None else _finite(before, "preceding force")
  if first == 2:
    if before is not None:
      raise ValueError("First possible trigger cannot have a preceding sample.")
  elif before is None or before >= DYNAMIC_STAIR_TRIGGER_FORCE_N:
    raise ValueError("Evidence does not identify the first force window.")
  if row["loaded_was_false_before"] is not True:
    raise ValueError("Loaded-contact evidence is not a rising edge.")
  return {
    "env_id": env_id, "side": side, "first_qualifying_step": first,
    "loaded_rising_step": rising, "sensor_force_window_n": sensor,
    "fsm_force_window_n": fsm, "preceding_sensor_force_n": before,
    "loaded_was_false_before": True,
  }


def _normalize_single_riser(value: object) -> dict[str, object]:
  row = _mapping(value, "single-riser evidence")
  expected = {
    "height_m", "num_envs", "settle_steps", "drive_steps", "step_index_base",
    "reset_events", "events", "trace_sha256",
  }
  if set(row) != expected:
    raise ValueError("Single-riser evidence schema drifted.")
  height = _finite(row["height_m"], "height_m")
  num_envs = _integer(row["num_envs"], "num_envs", 1)
  settle = _integer(row["settle_steps"], "settle_steps")
  drive = _integer(row["drive_steps"], "drive_steps", 1)
  if (
    not math.isclose(height, SINGLE_RISER_HEIGHT_M, abs_tol=1.0e-12)
    or num_envs != SINGLE_RISER_NUM_ENVS
    or settle != SINGLE_RISER_SETTLE_STEPS
    or drive != SINGLE_RISER_DRIVE_STEPS
    or row["step_index_base"] != 0
    or row["reset_events"] != 0
  ):
    raise ValueError("Single-riser protocol drifted or reset unexpectedly.")
  events: list[dict[str, object]] = []
  seen: set[tuple[int, str]] = set()
  for index, raw_event in enumerate(_sequence(row["events"], "trigger events")):
    event = _trigger_event(raw_event, index, num_envs, drive)
    key = (int(event["env_id"]), str(event["side"]))
    if key in seen:
      raise ValueError("Single-riser evidence repeats a wheel rising edge.")
    seen.add(key)
    events.append(event)
  if {str(event["side"]) for event in events} != {"left", "right"}:
    raise ValueError("The 1 cm live run did not detect both wheel triggers.")
  return {
    "height_m": height, "num_envs": num_envs, "settle_steps": settle,
    "drive_steps": drive, "step_index_base": 0, "reset_events": 0,
    "events": events,
    "trace_sha256": _digest(row["trace_sha256"], "riser trace_sha256"),
  }


def _formal_request(domain: str) -> camp_live.GateRequest:
  requests = camp_live._formal_pretraining_gate_requests()
  names = {
    "camp_flat_rolling": "velocity_gate_passed",
    "stage5_kick": "stage5_gate_passed",
  }
  if domain not in names:
    raise ValueError(f"Unknown trigger qualification domain: {domain!r}.")
  return requests[names[domain]]


def _expected_events(domain: str, request: camp_live.GateRequest) -> int:
  if domain == "camp_flat_rolling":
    return request.num_envs * request.steps * len(request.commands)
  return request.minimum_kick_events


def _normalize_false_positive(
  domain: str, value: object, request: camp_live.GateRequest
) -> dict[str, object]:
  row = _mapping(value, f"{domain} FP evidence")
  expected = {
    "events", "sample_events", "left_false_positives",
    "right_false_positives", "left_max_streak", "right_max_streak",
    "left_peak_metric_n", "right_peak_metric_n", "trace_sha256",
  }
  if set(row) not in (expected, expected | {"protocol"}):
    raise ValueError(f"{domain} false-positive evidence schema drifted.")
  events = _integer(row["events"], f"{domain}.events", 1)
  samples = _integer(row["sample_events"], f"{domain}.sample_events", 1)
  if events != _expected_events(domain, request):
    raise ValueError(f"{domain} event count is not the formal binding.")
  if domain == "camp_flat_rolling" and samples != events:
    raise ValueError("Flat rolling sample accounting is incomplete.")
  left_fp = _integer(row["left_false_positives"], "left_false_positives")
  right_fp = _integer(row["right_false_positives"], "right_false_positives")
  streaks = {
    side: _integer(row[f"{side}_max_streak"], f"{side}_max_streak")
    for side in ("left", "right")
  }
  if any(value > DYNAMIC_STAIR_TRIGGER_WINDOW for value in streaks.values()):
    raise ValueError("False-positive streak exceeded its saturation bound.")
  if left_fp + right_fp == 0 and any(
    value >= DYNAMIC_STAIR_TRIGGER_WINDOW for value in streaks.values()
  ):
    raise ValueError("A trigger window was hidden from false-positive counts.")
  protocol = camp_live.make_trigger_false_positive_check(
    domain=domain, events=events,
    stair_mode_false_positives=left_fp + right_fp,
  )
  if "protocol" in row and dict(_mapping(row["protocol"], "FP protocol")) != protocol:
    raise ValueError(f"{domain} formal FP protocol does not verify.")
  return {
    "events": events, "protocol": protocol, "sample_events": samples,
    "left_false_positives": left_fp, "right_false_positives": right_fp,
    "left_max_streak": streaks["left"],
    "right_max_streak": streaks["right"],
    "left_peak_metric_n": _finite(row["left_peak_metric_n"], "left peak"),
    "right_peak_metric_n": _finite(row["right_peak_metric_n"], "right peak"),
    "trace_sha256": _digest(row["trace_sha256"], f"{domain} trace_sha256"),
  }


def _normalize_evidence(evidence: Mapping[str, object]) -> dict[str, object]:
  if set(evidence) != {
    "runtime", "sensor_identity", "single_riser_1cm",
    "false_positive_checks",
  }:
    raise ValueError("Raw live evidence schema drifted.")
  checks = _mapping(evidence["false_positive_checks"], "FP checks")
  if set(checks) != {"camp_flat_rolling", "stage5_kick"}:
    raise ValueError("Both formal false-positive checks are required.")
  return {
    "runtime": _normalize_runtime(evidence["runtime"]),
    "sensor_identity": _normalize_sensor_identity(evidence["sensor_identity"]),
    "single_riser_1cm": _normalize_single_riser(evidence["single_riser_1cm"]),
    "false_positive_checks": {
      domain: _normalize_false_positive(
        domain, checks[domain], _formal_request(domain)
      )
      for domain in ("camp_flat_rolling", "stage5_kick")
    },
  }


def _qualification(evidence: Mapping[str, object]) -> dict[str, object]:
  riser = _mapping(evidence["single_riser_1cm"], "single riser")
  sides = {
    str(_mapping(row, "trigger event")["side"])
    for row in _sequence(riser["events"], "trigger events")
  }
  checks = _mapping(evidence["false_positive_checks"], "FP checks")
  counts: dict[str, int] = {}
  for domain in ("camp_flat_rolling", "stage5_kick"):
    row = _mapping(checks[domain], domain)
    counts[domain] = sum(
      int(row[f"{side}_false_positives"]) for side in ("left", "right")
    )
  result = {
    "metric": METRIC_NAME,
    "threshold_n": DYNAMIC_STAIR_TRIGGER_FORCE_N,
    "window": DYNAMIC_STAIR_TRIGGER_WINDOW,
    "left_sensor_identity": True,
    "right_sensor_identity": True,
    "left_live_detected": "left" in sides,
    "right_live_detected": "right" in sides,
    "flat_false_positives": counts["camp_flat_rolling"],
    "kick_false_positives": counts["stage5_kick"],
    "evidence_sha256": evidence_sha256(evidence),
  }
  return validate_trigger_qualification(result)


def build_document(evidence: Mapping[str, object]) -> dict[str, object]:
  """Create the exact qualification consumed by ``search_stair_dynamic``."""
  normalized = _normalize_evidence(evidence)
  return {
    "schema_version": QUALIFICATION_SCHEMA_VERSION,
    "kind": QUALIFICATION_KIND,
    "task": DYNAMIC_STAIR_TASK_ID,
    "evidence": normalized,
    "qualification": _qualification(normalized),
  }


def verify_document(value: object) -> dict[str, object]:
  """Recompute all verdict fields and the digest from saved raw evidence."""
  row = _mapping(value, "qualification document")
  if set(row) != {"schema_version", "kind", "task", "evidence", "qualification"}:
    raise ValueError("Qualification document schema drifted.")
  if (
    row["schema_version"] != QUALIFICATION_SCHEMA_VERSION
    or row["kind"] != QUALIFICATION_KIND
    or row["task"] != DYNAMIC_STAIR_TASK_ID
  ):
    raise ValueError("Qualification document identity drifted.")
  rebuilt = build_document(_mapping(row["evidence"], "raw evidence"))
  if dict(row) != rebuilt:
    raise ValueError("Qualification summary or evidence digest does not verify.")
  return dict(rebuilt["qualification"])


def collect_with_backend(backend: QualificationBackend) -> dict[str, object]:
  """Collect sensor, timing, flat and kick evidence via a mockable backend."""
  # Fail cheap before running either formal 3000-step false-positive protocol.
  runtime = dict(backend.runtime_evidence())
  identity = dict(backend.sensor_identity_evidence())
  riser = dict(backend.run_single_riser())
  _normalize_runtime(runtime)
  _normalize_sensor_identity(identity)
  _normalize_single_riser(riser)
  checks = {
    domain: dict(backend.run_false_positive(domain, _formal_request(domain)))
    for domain in ("camp_flat_rolling", "stage5_kick")
  }
  return build_document({
    "runtime": runtime,
    "sensor_identity": identity,
    "single_riser_1cm": riser,
    "false_positive_checks": checks,
  })


class _RawTriggerTrackingWrapper:
  """Thin interceptor used by the existing formal flat/kick rollout helpers."""
  def __init__(self, wrapped: Any, torch: Any) -> None:
    self._wrapped, self._torch = wrapped, torch
    count, device = int(wrapped.unwrapped.num_envs), wrapped.unwrapped.device
    self._streak = torch.zeros(count, 2, dtype=torch.long, device=device)
    self._latched = torch.zeros(count, 2, dtype=torch.bool, device=device)
    self._max_streak = torch.zeros(2, dtype=torch.long, device=device)
    self._peak = torch.zeros(2, dtype=torch.float, device=device)
    self._false_positives = torch.zeros(2, dtype=torch.long, device=device)
    self.sample_events = 0
    self._trace = hashlib.sha256()

  def __getattr__(self, name: str) -> Any:
    return getattr(self._wrapped, name)

  @property
  def unwrapped(self) -> Any:
    return self._wrapped.unwrapped

  def _request_off(self) -> None:
    command = self.unwrapped.command_manager.get_term("stair_request")._command
    if tuple(command.shape) != (self.unwrapped.num_envs, 1):
      raise RuntimeError("StairDynamic request command is unavailable.")
    command[:, 0] = 0.0

  def _metric(self) -> Any:
    trigger = importlib.import_module("hoppertrex_mjlab.hybrid.stair_trigger")
    values = []
    for name in (LEFT_SENSOR_NAME, RIGHT_SENSOR_NAME):
      data = self.unwrapped.scene.sensors[name].data
      values.append(trigger.stair_trigger_metric(
        found=data.found, force_contact_frame=data.force,
        normal_global=data.normal,
      ))
    return self._torch.stack(values, dim=1)

  def _record(self, metric: Any) -> None:
    cpu = metric.detach().to("cpu", dtype=self._torch.float32).contiguous()
    self._trace.update(struct.pack("<II", int(metric.shape[0]), 2))
    self._trace.update(cpu.numpy().tobytes(order="C"))
    hit = metric >= DYNAMIC_STAIR_TRIGGER_FORCE_N
    self._streak.copy_(self._torch.where(
      hit,
      self._torch.clamp(self._streak + 1, max=DYNAMIC_STAIR_TRIGGER_WINDOW),
      self._torch.zeros_like(self._streak),
    ))
    fired = self._streak >= DYNAMIC_STAIR_TRIGGER_WINDOW
    rising = fired & ~self._latched
    self._false_positives += rising.sum(dim=0)
    self._latched.logical_or_(fired)
    self._max_streak = self._torch.maximum(
      self._max_streak, self._streak.amax(dim=0)
    )
    self._peak = self._torch.maximum(self._peak, metric.amax(dim=0))
    self.sample_events += int(metric.shape[0])

  def reset(self) -> Any:
    result = self._wrapped.reset()
    self._streak.zero_()
    self._latched.zero_()
    self._request_off()
    return result

  def step(self, actions: Any) -> Any:
    self._request_off()
    self._record(self._metric())
    result = self._wrapped.step(actions)
    reset = self.unwrapped.reset_buf.bool()
    if bool(reset.any()):
      self._streak[reset] = 0
      self._latched[reset] = False
    self._request_off()
    return result

  def summary(self, events: int) -> dict[str, object]:
    return {
      "events": events, "sample_events": self.sample_events,
      "left_false_positives": int(self._false_positives[0].item()),
      "right_false_positives": int(self._false_positives[1].item()),
      "left_max_streak": int(self._max_streak[0].item()),
      "right_max_streak": int(self._max_streak[1].item()),
      "left_peak_metric_n": float(self._peak[0].item()),
      "right_peak_metric_n": float(self._peak[1].item()),
      "trace_sha256": self._trace.hexdigest(),
    }


class LiveBackend:
  """Delayed-import MjLab backend; never instantiated by local pure tests."""
  def __init__(self, device: str) -> None:
    if not isinstance(device, str) or not device or device != device.strip():
      raise ValueError("device must be a non-empty exact string.")
    self.device = device

  @staticmethod
  def _registry() -> Any:
    importlib.import_module("hoppertrex_mjlab.tasks")
    return importlib.import_module("mjlab.tasks.registry")

  def _registered(self) -> tuple[Any, Any]:
    registry = self._registry()
    cfg = registry.load_env_cfg(DYNAMIC_STAIR_TASK_ID, play=True)
    agent = registry.load_rl_cfg(DYNAMIC_STAIR_TASK_ID)
    task = importlib.import_module("hoppertrex_mjlab.tasks.hoppertrex_hybrid_task")
    task.validate_stair_dynamic_observation_contract(cfg)
    if getattr(cfg, "stair_dynamic_training_contract", None) is not False:
      raise RuntimeError("Registered StairDynamic play config carries training state.")
    return cfg, agent

  def runtime_evidence(self) -> Mapping[str, object]:
    runner = importlib.import_module("hoppertrex_mjlab.hybrid.runner")
    return {
      "task": DYNAMIC_STAIR_TASK_ID, "evaluation_seed": EVALUATION_SEED,
      "device": self.device, "git_sha": runner.repository_git_sha(),
    }

  def sensor_identity_evidence(self) -> Mapping[str, object]:
    cfg, _ = self._registered()
    sensors = {sensor.name: sensor for sensor in cfg.scene.sensors}
    action = cfg.actions["hybrid_wheel_leg"]
    def identity(side: str, name: str) -> dict[str, object]:
      sensor = sensors.get(name)
      if sensor is None:
        raise RuntimeError(f"Registered {side} wheel sensor is missing.")
      return {
        "sensor_name": sensor.name, "primary_mode": sensor.primary.mode,
        "primary_pattern": sensor.primary.pattern,
        "primary_entity": sensor.primary.entity,
        "secondary_mode": sensor.secondary.mode,
        "secondary_pattern": sensor.secondary.pattern,
        "fields": list(sensor.fields), "reduce": sensor.reduce,
        "num_slots": sensor.num_slots,
        "action_binding": getattr(action, f"dynamic_stair_{side}_sensor_name"),
      }
    return {
      "left": identity("left", LEFT_SENSOR_NAME),
      "right": identity("right", RIGHT_SENSOR_NAME),
      "objects_distinct": sensors.get(LEFT_SENSOR_NAME)
      is not sensors.get(RIGHT_SENSOR_NAME),
    }

  @staticmethod
  def _set_flat_count(cfg: Any, count: int) -> None:
    for command in cfg.commands.values():
      if hasattr(command, "flat_env_count"):
        command.flat_env_count = count
    reset = cfg.events.get("reset_root_to_stair_dynamic")
    if reset is None:
      raise RuntimeError("Registered StairDynamic reset event is missing.")
    reset.params["flat_env_count"] = count
    reset.params["x_offset_from_origin_m"] = 0.0
    curriculum = cfg.curriculum.get("stair_dynamic_height")
    if curriculum is not None:
      curriculum.params["flat_env_count"] = count

  def _configure(self, num_envs: int, flat: bool) -> tuple[Any, Any]:
    cfg, agent = self._registered()
    cfg.seed = EVALUATION_SEED
    cfg.scene.num_envs = num_envs
    cfg.events.pop("push_robot", None)
    cfg.metrics = {}
    if flat:
      terrain = importlib.import_module("mjlab.terrains")
      terrain_cfg = importlib.import_module("mjlab.terrains.config")
      cfg.scene.terrain = terrain.TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=terrain.TerrainGeneratorCfg(
          seed=1, curriculum=True,
          size=camp_live.FLAT_EVALUATION_TERRAIN_SIZE_M,
          num_rows=1, num_cols=1, difficulty_range=(0.0, 0.0),
          sub_terrains={"flat_zero_height": terrain_cfg.flat(proportion=1.0)},
        ),
        max_init_terrain_level=0, num_envs=num_envs,
      )
      cfg.curriculum = {}
      cfg.episode_length_s = camp_live.EPISODE_LENGTH_S
      self._set_flat_count(cfg, num_envs)
    else:
      terrain = cfg.scene.terrain
      if terrain is None or terrain.terrain_generator is None:
        raise RuntimeError("Registered StairDynamic terrain is unavailable.")
      terrain.num_envs = num_envs
      terrain.max_init_terrain_level = 1
      terrain.terrain_generator.seed = 1
      self._set_flat_count(cfg, 0)
      curriculum = cfg.curriculum.get("stair_dynamic_height")
      if curriculum is None:
        raise RuntimeError("Registered StairDynamic curriculum is missing.")
      curriculum.params["initial_upper_height_m"] = SINGLE_RISER_HEIGHT_M
    return cfg, agent

  @contextmanager
  def _flat_session(self, num_envs: int) -> Any:
    torch = importlib.import_module("torch")
    env_cls = importlib.import_module("mjlab.envs").ManagerBasedRlEnv
    wrapper_cls = importlib.import_module("mjlab.rl").RslRlVecEnvWrapper
    cfg, agent = self._configure(num_envs, flat=True)
    tracker = _RawTriggerTrackingWrapper(
      wrapper_cls(env_cls(cfg=cfg, device=self.device),
                  clip_actions=agent.clip_actions), torch
    )
    try:
      yield tracker
    finally:
      tracker.close()

  @staticmethod
  def _zero_policy(torch: Any) -> Any:
    def policy(observations: Any) -> Any:
      actor = camp_live._observation_group(observations, "actor")
      return torch.zeros((actor.shape[0], 6), dtype=actor.dtype, device=actor.device)
    return policy

  def run_false_positive(
    self, domain: str, request: camp_live.GateRequest
  ) -> Mapping[str, object]:
    torch = importlib.import_module("torch")
    fixed = importlib.import_module(
      "hoppertrex_mjlab.scripts.rsl_rl.evaluate_fixed_command")
    hybrid = importlib.import_module(
      "hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate")
    balance = importlib.import_module(
      "hoppertrex_mjlab.tasks.hoppertrex_balance_task")
    base = camp_live._MjLabBackend._rollout_args(request, self.device)
    policy = self._zero_policy(torch)
    with self._flat_session(request.num_envs) as tracker:
      if domain == "camp_flat_rolling":
        args = hybrid._fixed_command_args(base, DYNAMIC_STAIR_TASK_ID)
        posture = (float(balance.ROOT_HEIGHT_TARGET), 0.0)
        for vx, yaw in request.commands:
          if abs(yaw) > 1.0e-12:
            raise RuntimeError("Formal flat trigger check commands yaw.")
          fixed._run_fixed_command(
            wrapped=tracker, policy=policy, args=args,
            lin_x_cmd=vx, posture_target=posture)
        events = request.num_envs * request.steps * len(request.commands)
        if tracker.sample_events != events:
          raise RuntimeError("Flat trigger sample accounting is incomplete.")
      elif domain == "stage5_kick":
        metrics = hybrid._run_recovery_scenario(
          wrapped=tracker, policy=policy, args=base,
          target_height=float(balance.ROOT_HEIGHT_TARGET), target_pitch=0.0,
          kick_scale=float(request.kick_scale))
        events = round(float(metrics["kick_event_count"]))
      else:
        raise ValueError(f"Unknown false-positive domain: {domain!r}.")
      return tracker.summary(events)

  @staticmethod
  def _force_commands(env: Any, vx: float, request: bool) -> None:
    fixed = importlib.import_module(
      "hoppertrex_mjlab.scripts.rsl_rl.evaluate_fixed_command")
    balance = importlib.import_module(
      "hoppertrex_mjlab.tasks.hoppertrex_balance_task")
    fixed._force_command(env, vx, 0.0)
    fixed._force_static_posture(env, (float(balance.ROOT_HEIGHT_TARGET), 0.0))
    env.command_manager.get_term("stair_request")._command[:, 0] = float(request)

  @staticmethod
  def _raw_metric(env: Any, torch: Any) -> Any:
    trigger = importlib.import_module("hoppertrex_mjlab.hybrid.stair_trigger")
    rows = []
    for name in (LEFT_SENSOR_NAME, RIGHT_SENSOR_NAME):
      data = env.scene.sensors[name].data
      rows.append(trigger.stair_trigger_metric(
        found=data.found, force_contact_frame=data.force,
        normal_global=data.normal,
      ))
    return torch.stack(rows, dim=1)

  def run_single_riser(self) -> Mapping[str, object]:
    torch = importlib.import_module("torch")
    env_cls = importlib.import_module("mjlab.envs").ManagerBasedRlEnv
    cfg, _ = self._configure(SINGLE_RISER_NUM_ENVS, flat=False)
    env = env_cls(cfg=cfg, device=self.device)
    trace = hashlib.sha256()
    try:
      env.reset()
      terrain = env.scene.terrain
      if terrain is None or bool((terrain.terrain_levels != 1).any()):
        raise RuntimeError("Live trigger rollout is not entirely on 1 cm row 1.")
      zeros = torch.zeros(SINGLE_RISER_NUM_ENVS, 6, device=env.device)
      for _ in range(SINGLE_RISER_SETTLE_STEPS):
        self._force_commands(env, 0.0, False)
        env.step(zeros)
      action = env.action_manager.get_term("hybrid_wheel_leg")
      if bool(action.dynamic_loaded_contact.any()):
        raise RuntimeError("Loaded-contact latched before the 1 cm drive.")

      history = [[[] for _ in range(SINGLE_RISER_NUM_ENVS)] for _ in range(2)]
      fsm_history = [[[] for _ in range(SINGLE_RISER_NUM_ENVS)] for _ in range(2)]
      streak = torch.zeros(
        SINGLE_RISER_NUM_ENVS, 2, dtype=torch.long, device=env.device)
      first = torch.full_like(streak, -1)
      previous_loaded = action.dynamic_loaded_contact.clone()
      events: list[dict[str, object]] = []
      seen: set[tuple[int, int]] = set()
      reset_events = 0
      for step in range(SINGLE_RISER_DRIVE_STEPS):
        self._force_commands(env, 0.07, True)
        raw = self._raw_metric(env, torch)
        cpu = raw.detach().to("cpu", dtype=torch.float32).contiguous()
        trace.update(struct.pack("<II", SINGLE_RISER_NUM_ENVS, 2))
        trace.update(cpu.numpy().tobytes(order="C"))
        hit = raw >= DYNAMIC_STAIR_TRIGGER_FORCE_N
        streak = torch.where(
          hit, torch.clamp(streak + 1, max=DYNAMIC_STAIR_TRIGGER_WINDOW),
          torch.zeros_like(streak))
        newly = (streak >= DYNAMIC_STAIR_TRIGGER_WINDOW) & (first < 0)
        first[newly] = step
        for env_id in range(SINGLE_RISER_NUM_ENVS):
          for side_index in range(2):
            history[side_index][env_id].append(float(raw[env_id, side_index].item()))

        env.step(zeros)
        fsm_force = action.dynamic_contact_force.clone()
        loaded = action.dynamic_loaded_contact.clone()
        for env_id in range(SINGLE_RISER_NUM_ENVS):
          for side_index, side in enumerate(("left", "right")):
            fsm_history[side_index][env_id].append(
              float(fsm_force[env_id, side_index].item()))
            key = (env_id, side_index)
            rising = bool(loaded[env_id, side_index].item()) and not bool(
              previous_loaded[env_id, side_index].item())
            if not rising or key in seen:
              continue
            seen.add(key)
            sensor_values = history[side_index][env_id]
            fsm_values = fsm_history[side_index][env_id]
            events.append({
              "env_id": env_id, "side": side,
              "first_qualifying_step": int(first[env_id, side_index].item()),
              "loaded_rising_step": step,
              "sensor_force_window_n": sensor_values[-3:],
              "fsm_force_window_n": fsm_values[-3:],
              "preceding_sensor_force_n": (
                None if len(sensor_values) == 3 else sensor_values[-4]),
              "loaded_was_false_before": True,
            })
        previous_loaded = loaded
        reset_events += int(env.reset_buf.sum().item())
      return {
        "height_m": SINGLE_RISER_HEIGHT_M,
        "num_envs": SINGLE_RISER_NUM_ENVS,
        "settle_steps": SINGLE_RISER_SETTLE_STEPS,
        "drive_steps": SINGLE_RISER_DRIVE_STEPS,
        "step_index_base": 0, "reset_events": reset_events,
        "events": events, "trace_sha256": trace.hexdigest(),
      }
    finally:
      env.close()


def _read_document(path: Path) -> Mapping[str, object]:
  return camp_live._read_json_mapping(path)


def _write_document(document: Mapping[str, object], output: Path | None) -> None:
  # Reuse the existing hard-link publication: atomic and non-overwriting.
  camp_live._write_output(document, output)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest="command", required=True)
  collect_parser = commands.add_parser("collect", help="Run live qualification.")
  collect_parser.add_argument("--device", default="cuda:0")
  collect_parser.add_argument("--output", type=Path, required=True)
  verify_parser = commands.add_parser("verify", help="Reverify saved evidence.")
  verify_parser.add_argument("--input", type=Path, required=True)
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  if args.command == "collect":
    _write_document(collect_with_backend(LiveBackend(args.device)), args.output)
  elif args.command == "verify":
    qualification = verify_document(_read_document(args.input))
    _write_document({"qualification": qualification}, None)
  else:  # pragma: no cover
    raise AssertionError(f"Unhandled command: {args.command!r}.")
  return 0


__all__ = [
  "EVALUATION_SEED",
  "LEFT_SENSOR_NAME",
  "METRIC_NAME",
  "QUALIFICATION_KIND",
  "QUALIFICATION_SCHEMA_VERSION",
  "RIGHT_SENSOR_NAME",
  "SINGLE_RISER_DRIVE_STEPS",
  "SINGLE_RISER_HEIGHT_M",
  "SINGLE_RISER_NUM_ENVS",
  "SINGLE_RISER_SETTLE_STEPS",
  "LiveBackend",
  "QualificationBackend",
  "build_document",
  "collect_with_backend",
  "evidence_sha256",
  "main",
  "parse_args",
  "verify_document",
]

if __name__ == "__main__":
  raise SystemExit(main())
