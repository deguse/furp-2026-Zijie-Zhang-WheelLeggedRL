# ruff: noqa: TRY004
"""Runtime and checkpoint contract for HopperTrex Hybrid-v3 StairDynamic."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .config import DEFAULT_ACTION_SCALES
from .stair_dynamic import (
  DYNAMIC_STAIR_CONTRACT_SCHEMA_VERSION,
  DYNAMIC_STAIR_FEEDFORWARD_LIMIT_RAD,
  DYNAMIC_STAIR_PPO_LEG_SCALE_RAD,
  DYNAMIC_STAIR_TASK_ID,
  dynamic_maneuver_payload,
)

DYNAMIC_STAIR_ACTOR_TERMS = (
  "base_lin_vel",
  "base_ang_vel",
  "projected_gravity",
  "velocity_command",
  "posture_command",
  "joint_pos",
  "joint_vel",
  "controller_baseline",
  "applied_residual",
  "stair_request",
  "phase_one_hot",
  "loaded_contact",
  "lead_side",
  "leg_feedforward",
)
DYNAMIC_STAIR_STAGE5_PREFIX_TERMS = DYNAMIC_STAIR_ACTOR_TERMS[:9]
DYNAMIC_STAIR_ACTOR_TAIL_TERMS = DYNAMIC_STAIR_ACTOR_TERMS[9:]
DYNAMIC_STAIR_CRITIC_TAIL_TERMS = (
  "step_height",
  "distance_to_next_riser",
  "left_contact_force",
  "right_contact_force",
)
DYNAMIC_STAIR_TERM_WIDTHS = {
  "base_lin_vel": 3,
  "base_ang_vel": 3,
  "projected_gravity": 3,
  "velocity_command": 3,
  "posture_command": 2,
  "joint_pos": 6,
  "joint_vel": 6,
  "controller_baseline": 2,
  "applied_residual": 6,
  "stair_request": 1,
  "phase_one_hot": 9,
  "loaded_contact": 2,
  "lead_side": 2,
  "leg_feedforward": 4,
  "step_height": 1,
  "distance_to_next_riser": 1,
  "left_contact_force": 1,
  "right_contact_force": 1,
}
DYNAMIC_STAIR_STAGE5_ACTOR_WIDTH = 34
DYNAMIC_STAIR_ACTOR_WIDTH = 52
DYNAMIC_STAIR_CRITIC_WIDTH = 56
DYNAMIC_STAIR_NUM_ENVS = 256
DYNAMIC_STAIR_FLAT_ENVS = 64
DYNAMIC_STAIR_STAIR_ENVS = 192
DYNAMIC_STAIR_PROBE_UPDATES = 100
DYNAMIC_STAIR_EXTENSION_TOTAL_UPDATES = 500
DYNAMIC_STAIR_SAVE_INTERVAL = 25
DYNAMIC_STAIR_STEPS_PER_ITERATION = 24
DYNAMIC_STAIR_TRAINING_INFO_KEY = "stair_dynamic_training"
DYNAMIC_STAIR_CURRICULUM_INFO_KEY = "stair_dynamic_curriculum"
DYNAMIC_STAIR_PROGRESS_INFO_KEY = "stair_dynamic_progress"
DYNAMIC_STAIR_MIGRATION_INFO_KEY = "stair_dynamic_migration"
DYNAMIC_STAIR_ACTION_MASK = (True, True, True, True, True, True)
DYNAMIC_STAIR_ACTION_SCALES = DEFAULT_ACTION_SCALES
DYNAMIC_STAIR_WITHDRAWN_CRITIC_TERMS = ("friction", "randomization_parameters")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[/\\]")


def _portable_text(value: str) -> str:
  candidate = (
    PureWindowsPath(value)
    if ("\\" in value or _WINDOWS_DRIVE_RE.match(value))
    else PurePosixPath(value)
  )
  return candidate.name if candidate.is_absolute() else value


def _plain(value: Any) -> Any:
  if value.__class__.__name__ == "SceneEntityCfg":
    names = (
      "joint_names",
      "body_names",
      "geom_names",
      "site_names",
      "actuator_names",
      "tendon_names",
      "camera_names",
      "light_names",
      "material_names",
      "pair_names",
    )
    return {
      "name": getattr(value, "name", None),
      **{
        name: _plain(getattr(value, name, None))
        for name in names
        if getattr(value, name, None) is not None
      },
      "preserve_order": bool(getattr(value, "preserve_order", False)),
    }
  if is_dataclass(value):
    return _plain(asdict(value))
  if isinstance(value, Enum):
    return value.value
  if isinstance(value, Path):
    return value.name if value.is_absolute() else str(value)
  if isinstance(value, slice):
    return {
      "slice": [_plain(value.start), _plain(value.stop), _plain(value.step)]
    }
  if type(value).__module__.split(".", 1)[0] == "numpy":
    converter = getattr(value, "tolist", None)
    if callable(converter):
      return _plain(converter())
    scalar = getattr(value, "item", None)
    if callable(scalar):
      return _plain(scalar())
  if isinstance(value, Mapping):
    return {str(key): _plain(item) for key, item in value.items()}
  if isinstance(value, (tuple, list)):
    return [_plain(item) for item in value]
  if isinstance(value, str):
    return _portable_text(value)
  if value is None or isinstance(value, (bool, int)):
    return value
  if isinstance(value, float):
    if not math.isfinite(value):
      raise ValueError("Dynamic stair contract forbids non-finite floats.")
    return value
  if callable(value):
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not module or not qualname:
      raise ValueError("Dynamic stair contract callable has no stable identity.")
    return f"{module}.{qualname}"
  if hasattr(value, "__dict__"):
    return {
      str(key): _plain(item)
      for key, item in vars(value).items()
      if not str(key).startswith("_")
    }
  raise TypeError(f"Unsupported dynamic stair contract value: {type(value).__name__}")


def _field(value: object, name: str, default: Any = None) -> Any:
  if isinstance(value, Mapping):
    return value.get(name, default)
  return getattr(value, name, default)


def _callable_id(value: object) -> str:
  module = getattr(value, "__module__", None)
  name = getattr(value, "__qualname__", None)
  if not isinstance(module, str) or not isinstance(name, str):
    raise ValueError("Dynamic stair contract callable has no stable identity.")
  return f"{module}.{name}"


def _dynamic_terrain_payload(env_cfg: object) -> dict[str, Any]:
  scene = _field(env_cfg, "scene")
  terrain = _field(scene, "terrain")
  generator = _field(terrain, "terrain_generator")
  sub_terrains = _field(generator, "sub_terrains", {})
  stair = sub_terrains.get("stair") if isinstance(sub_terrains, Mapping) else None
  if terrain is None or generator is None or stair is None:
    raise ValueError("StairDynamic terrain declaration is missing.")
  return {
    "terrain_type": _field(terrain, "terrain_type"),
    "max_init_terrain_level": int(_field(terrain, "max_init_terrain_level", -1)),
    "generator": {
      "curriculum": bool(_field(generator, "curriculum", False)),
      "size": _plain(_field(generator, "size")),
      "num_rows": int(_field(generator, "num_rows", -1)),
      "num_cols": int(_field(generator, "num_cols", -1)),
      "difficulty_range": _plain(_field(generator, "difficulty_range")),
      "stair": {
        # Omit sub-terrain ``size``: MjLab rewrites it from generator.size.
        "proportion": float(_field(stair, "proportion", math.nan)),
        "border_width": float(_field(stair, "border_width", math.nan)),
        "step_height_range": _plain(_field(stair, "step_height_range")),
        "step_width": float(_field(stair, "step_width", math.nan)),
        "platform_width": float(_field(stair, "platform_width", math.nan)),
        "holes": bool(_field(stair, "holes", False)),
      },
    },
  }


def _dynamic_sensor_payload(env_cfg: object) -> list[dict[str, Any]]:
  scene = _field(env_cfg, "scene")
  sensors = _field(scene, "sensors", ())
  result = []
  for sensor in sensors:
    name = _field(sensor, "name")
    if name not in ("stair_dynamic_left_contact", "stair_dynamic_right_contact"):
      continue
    primary = _field(sensor, "primary")
    secondary = _field(sensor, "secondary")
    result.append(
      {
        "name": name,
        "primary": {
          "mode": _field(primary, "mode"),
          "pattern": _plain(_field(primary, "pattern")),
          "entity": _field(primary, "entity"),
        },
        "secondary": {
          "mode": _field(secondary, "mode"),
          "pattern": _plain(_field(secondary, "pattern")),
        },
        "fields": _plain(_field(sensor, "fields")),
        "reduce": _field(sensor, "reduce"),
        "num_slots": int(_field(sensor, "num_slots", -1)),
        "global_frame": bool(_field(sensor, "global_frame", False)),
      }
    )
  if [item["name"] for item in result] != [
    "stair_dynamic_left_contact",
    "stair_dynamic_right_contact",
  ]:
    raise ValueError("StairDynamic per-wheel sensor declaration drifted.")
  return result


def _dynamic_event_payload(env_cfg: object) -> dict[str, Any]:
  events = _field(env_cfg, "events", {})
  reset = events.get("reset_root_to_stair_dynamic") if isinstance(events, Mapping) else None
  push = events.get("push_robot") if isinstance(events, Mapping) else None
  if reset is None or push is None:
    raise ValueError("StairDynamic training reset/push events are missing.")
  reset_params = _field(reset, "params", {})
  push_params = _field(push, "params", {})
  return {
    "reset": {
      "function": _callable_id(_field(reset, "func")),
      "mode": _field(reset, "mode"),
      "root_height": float(reset_params["root_height"]),
      "pose_range": _plain(reset_params["pose_range"]),
      "velocity_range": _plain(reset_params["velocity_range"]),
      "flat_env_count": int(reset_params["flat_env_count"]),
    },
    "push": {
      "function": _callable_id(_field(push, "func")),
      "mode": _field(push, "mode"),
      "interval_range_s": _plain(_field(push, "interval_range_s")),
      "velocity_range": _plain(push_params["velocity_range"]),
      "flat_env_count": int(push_params["flat_env_count"]),
    },
  }


def validate_dynamic_stair_observation_layout(
  actor_terms: Mapping[str, object],
  critic_terms: Mapping[str, object],
) -> None:
  """Validate exact term order, widths, and privileged isolation."""

  actor_names = tuple(actor_terms)
  critic_names = tuple(critic_terms)
  if actor_names != DYNAMIC_STAIR_ACTOR_TERMS:
    raise ValueError("StairDynamic actor term order drifted.")
  if critic_names != actor_names + DYNAMIC_STAIR_CRITIC_TAIL_TERMS:
    raise ValueError("StairDynamic critic must append exactly four privileged terms.")
  if any(name in critic_names for name in DYNAMIC_STAIR_WITHDRAWN_CRITIC_TERMS):
    raise ValueError("StairDynamic withdrawn critic fields must stay absent.")
  actor_width = sum(DYNAMIC_STAIR_TERM_WIDTHS[name] for name in actor_names)
  critic_width = sum(DYNAMIC_STAIR_TERM_WIDTHS[name] for name in critic_names)
  prefix_width = sum(
    DYNAMIC_STAIR_TERM_WIDTHS[name]
    for name in DYNAMIC_STAIR_STAGE5_PREFIX_TERMS
  )
  if prefix_width != DYNAMIC_STAIR_STAGE5_ACTOR_WIDTH:
    raise ValueError("StairDynamic Stage5 observation prefix width drifted.")
  if actor_width != DYNAMIC_STAIR_ACTOR_WIDTH:
    raise ValueError("StairDynamic actor width must be 52.")
  if critic_width != DYNAMIC_STAIR_CRITIC_WIDTH:
    raise ValueError("StairDynamic critic width must be 56.")


def dynamic_stair_artifact_bindings(env_cfg: object) -> dict[str, str]:
  actions = _field(env_cfg, "actions", {})
  action = actions.get("hybrid_wheel_leg") if isinstance(actions, Mapping) else None
  if action is None:
    raise ValueError("StairDynamic hybrid action config is missing.")
  maneuver = _field(action, "dynamic_stair_maneuver")
  bindings = {
    "controller_gain_hash": _field(action, "controller_gain_hash"),
    "calibration_hash": _field(action, "calibration_hash"),
    "yaw_calibration_hash": _field(action, "yaw_calibration_hash"),
    "posture_map_hash": _field(action, "posture_map_hash"),
    "posture_artifact_hash": _field(action, "posture_artifact_hash"),
    "station_calibration_hash": _field(action, "station_calibration_hash"),
    "dynamic_maneuver_hash": _field(maneuver, "maneuver_hash"),
  }
  result: dict[str, str] = {}
  for name, value in bindings.items():
    if not isinstance(value, str) or not value:
      raise ValueError(f"StairDynamic artifact binding {name} is missing.")
    result[name] = value
  return result


def dynamic_stair_contract_payload(env_cfg: object, agent_cfg: object) -> dict[str, Any]:
  """Return the canonical training contract without machine-specific paths."""

  task = _field(env_cfg, "stair_dynamic_task_id")
  if task != DYNAMIC_STAIR_TASK_ID:
    raise ValueError("StairDynamic task marker is missing or invalid.")
  observations = _field(env_cfg, "observations", {})
  actor = observations.get("actor") if isinstance(observations, Mapping) else None
  critic = observations.get("critic") if isinstance(observations, Mapping) else None
  actor_terms = _field(actor, "terms", {})
  critic_terms = _field(critic, "terms", {})
  if not isinstance(actor_terms, Mapping) or not isinstance(critic_terms, Mapping):
    raise ValueError("StairDynamic observation groups are missing.")
  validate_dynamic_stair_observation_layout(actor_terms, critic_terms)
  actions = _field(env_cfg, "actions", {})
  action = actions.get("hybrid_wheel_leg") if isinstance(actions, Mapping) else None
  if action is None:
    raise ValueError("StairDynamic action config is missing.")
  if tuple(_field(action, "action_mask", ())) != DYNAMIC_STAIR_ACTION_MASK:
    raise ValueError("StairDynamic runtime action mask drifted.")
  if tuple(float(v) for v in _field(action, "action_scales", ())) != DYNAMIC_STAIR_ACTION_SCALES:
    raise ValueError("StairDynamic action scales drifted.")
  maneuver = _field(action, "dynamic_stair_maneuver")
  if maneuver is None:
    raise ValueError("StairDynamic requires a qualified maneuver artifact.")
  if _field(env_cfg, "stair_dynamic_maneuver_qualified") is not True:
    raise ValueError("StairDynamic maneuver has not passed CEM/live qualification.")
  actor_cfg = _field(agent_cfg, "actor", {})
  distribution = _field(actor_cfg, "distribution_cfg", {})
  if tuple(_field(distribution, "active_mask", ())) != DYNAMIC_STAIR_ACTION_MASK:
    raise ValueError("StairDynamic PPO active mask drifted.")
  return {
    "schema_version": DYNAMIC_STAIR_CONTRACT_SCHEMA_VERSION,
    "task": task,
    "observations": {
      "actor_terms": list(DYNAMIC_STAIR_ACTOR_TERMS),
      "critic_terms": list(
        DYNAMIC_STAIR_ACTOR_TERMS + DYNAMIC_STAIR_CRITIC_TAIL_TERMS
      ),
      "widths": {
        "stage5_prefix": DYNAMIC_STAIR_STAGE5_ACTOR_WIDTH,
        "actor": DYNAMIC_STAIR_ACTOR_WIDTH,
        "critic": DYNAMIC_STAIR_CRITIC_WIDTH,
      },
    },
    "control": {
      "action_mask": list(DYNAMIC_STAIR_ACTION_MASK),
      "action_scales": list(DYNAMIC_STAIR_ACTION_SCALES),
      "ppo_leg_scale_rad": DYNAMIC_STAIR_PPO_LEG_SCALE_RAD,
      "feedforward_limit_rad": DYNAMIC_STAIR_FEEDFORWARD_LIMIT_RAD,
      "maneuver_hash": _field(maneuver, "maneuver_hash"),
      "maneuver_payload": dynamic_maneuver_payload(
        maneuver,
        bindings=_field(maneuver, "bindings", {}),
      ),
    },
    "environment": {
      "num_envs": int(_field(_field(env_cfg, "scene"), "num_envs", -1)),
      "episode_length_s": float(_field(env_cfg, "episode_length_s", math.nan)),
      "decimation": int(_field(env_cfg, "decimation", -1)),
      "sim_timestep": float(
        _field(_field(_field(env_cfg, "sim"), "mujoco"), "timestep", math.nan)
      ),
      "terrain": _dynamic_terrain_payload(env_cfg),
      "sensors": _dynamic_sensor_payload(env_cfg),
      "events": _dynamic_event_payload(env_cfg),
      "commands": _plain(_field(env_cfg, "commands", {})),
      "rewards": _plain(_field(env_cfg, "rewards", {})),
      "curriculum": _plain(_field(env_cfg, "curriculum", {})),
    },
    "ppo": {
      "num_steps_per_env": int(_field(agent_cfg, "num_steps_per_env", -1)),
      "save_interval": int(_field(agent_cfg, "save_interval", -1)),
      "actor": _plain(actor_cfg),
      "critic": _plain(_field(agent_cfg, "critic", {})),
      "algorithm": _plain(_field(agent_cfg, "algorithm", {})),
    },
    "artifacts": dynamic_stair_artifact_bindings(env_cfg),
  }


def dynamic_stair_contract_hash(env_cfg: object, agent_cfg: object) -> str:
  encoded = json.dumps(
    dynamic_stair_contract_payload(env_cfg, agent_cfg),
    sort_keys=True,
    separators=(",", ":"),
  ).encode("ascii")
  return hashlib.sha256(encoded).hexdigest()


def bind_dynamic_stair_contract(env_cfg: object, agent_cfg: object) -> str:
  contract = dynamic_stair_contract_hash(env_cfg, agent_cfg)
  existing = _field(env_cfg, "stair_dynamic_contract_sha256")
  if existing not in (None, contract):
    raise ValueError("StairDynamic environment carries a conflicting contract hash.")
  env_cfg.stair_dynamic_contract_sha256 = contract
  return contract


def validate_dynamic_stair_progress_payload(
  payload: Mapping[str, object],
  curriculum_state: Mapping[str, object],
) -> dict[str, float | int]:
  """Validate the compact progress snapshot stored beside curriculum state."""

  expected = {
    "upper_height_m",
    "consecutive_ready_evaluations",
    "evaluations",
    "completed_stair_episodes",
    "successful_stair_episodes",
    "stair_success_rate",
  }
  if not isinstance(payload, Mapping) or set(payload) != expected:
    raise ValueError("StairDynamic progress schema drifted.")

  def integer(name: str) -> int:
    value = payload[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
      raise ValueError(f"StairDynamic progress {name} must be non-negative integer.")
    return int(value)

  def number(name: str) -> float:
    value = payload[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
      raise ValueError(f"StairDynamic progress {name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
      raise ValueError(f"StairDynamic progress {name} must be finite.")
    return result

  upper = number("upper_height_m")
  ready = integer("consecutive_ready_evaluations")
  evaluations = integer("evaluations")
  completed = integer("completed_stair_episodes")
  successful = integer("successful_stair_episodes")
  rate = number("stair_success_rate")
  if (
    successful > completed
    or not 0.0 <= rate <= 1.0
    or not math.isclose(
      rate,
      successful / max(completed, 1),
      rel_tol=0.0,
      abs_tol=1.0e-12,
    )
    or upper != float(curriculum_state.get("upper_height_m", math.nan))
    or ready != curriculum_state.get("consecutive_ready_evaluations")
    or evaluations != curriculum_state.get("evaluations")
    or completed != curriculum_state.get("completed_stair_episodes")
    or successful != curriculum_state.get("successful_stair_episodes")
  ):
    raise ValueError("StairDynamic progress and curriculum state disagree.")
  return {
    "upper_height_m": upper,
    "consecutive_ready_evaluations": ready,
    "evaluations": evaluations,
    "completed_stair_episodes": completed,
    "successful_stair_episodes": successful,
    "stair_success_rate": rate,
  }


def validate_dynamic_stair_training_request(
  env_cfg: object,
  agent_cfg: object,
  *,
  resume: bool,
) -> None:
  if _field(env_cfg, "stair_dynamic_task_id") != DYNAMIC_STAIR_TASK_ID:
    raise ValueError("StairDynamic task marker is missing.")
  if _field(env_cfg, "stair_dynamic_training_contract") is not True:
    raise ValueError("StairDynamic training marker is missing.")
  if _field(env_cfg, "stair_dynamic_maneuver_qualified") is not True:
    raise ValueError("StairDynamic training requires the qualified maneuver artifact.")
  if int(_field(agent_cfg, "seed", -1)) != 1:
    raise ValueError("StairDynamic currently permits only training seed 1.")
  if int(_field(_field(env_cfg, "scene"), "num_envs", -1)) != DYNAMIC_STAIR_NUM_ENVS:
    raise ValueError("StairDynamic training requires 256 environments.")
  if int(_field(agent_cfg, "num_steps_per_env", -1)) != DYNAMIC_STAIR_STEPS_PER_ITERATION:
    raise ValueError("StairDynamic rollout length must be 24.")
  if int(_field(agent_cfg, "save_interval", -1)) != DYNAMIC_STAIR_SAVE_INTERVAL:
    raise ValueError("StairDynamic save interval must be 25.")
  total = int(_field(agent_cfg, "max_iterations", -1))
  if total not in (
    DYNAMIC_STAIR_PROBE_UPDATES,
    DYNAMIC_STAIR_EXTENSION_TOTAL_UPDATES,
  ):
    raise ValueError("StairDynamic total budget must be 100 or 500 updates.")
  if not resume:
    raise ValueError("StairDynamic must load a validated Stage5 migration checkpoint.")
  bind_dynamic_stair_contract(env_cfg, agent_cfg)


__all__ = [
  "DYNAMIC_STAIR_ACTION_MASK",
  "DYNAMIC_STAIR_ACTION_SCALES",
  "DYNAMIC_STAIR_ACTOR_TAIL_TERMS",
  "DYNAMIC_STAIR_ACTOR_TERMS",
  "DYNAMIC_STAIR_ACTOR_WIDTH",
  "DYNAMIC_STAIR_CRITIC_TAIL_TERMS",
  "DYNAMIC_STAIR_CRITIC_WIDTH",
  "DYNAMIC_STAIR_CURRICULUM_INFO_KEY",
  "DYNAMIC_STAIR_EXTENSION_TOTAL_UPDATES",
  "DYNAMIC_STAIR_FLAT_ENVS",
  "DYNAMIC_STAIR_MIGRATION_INFO_KEY",
  "DYNAMIC_STAIR_NUM_ENVS",
  "DYNAMIC_STAIR_PROBE_UPDATES",
  "DYNAMIC_STAIR_PROGRESS_INFO_KEY",
  "DYNAMIC_STAIR_SAVE_INTERVAL",
  "DYNAMIC_STAIR_STAGE5_ACTOR_WIDTH",
  "DYNAMIC_STAIR_STAGE5_PREFIX_TERMS",
  "DYNAMIC_STAIR_STAIR_ENVS",
  "DYNAMIC_STAIR_STEPS_PER_ITERATION",
  "DYNAMIC_STAIR_TERM_WIDTHS",
  "DYNAMIC_STAIR_TRAINING_INFO_KEY",
  "bind_dynamic_stair_contract",
  "dynamic_stair_artifact_bindings",
  "dynamic_stair_contract_hash",
  "dynamic_stair_contract_payload",
  "validate_dynamic_stair_observation_layout",
  "validate_dynamic_stair_progress_payload",
  "validate_dynamic_stair_training_request",
]
