"""Immutable contract and checkpoint provenance for the S5B StairCamp."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, asdict, fields, is_dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from hoppertrex_mjlab.hybrid.config import (
  STAIR_CAMP_ACTION_MASK,
  STAIR_CAMP_FAILURE_LADDER_VARIANT,
  STAIR_CAMP_FAILURE_LQR_GAIN_SCALE,
  STAIR_CAMP_LEG_RESIDUAL_SCALE,
  STAIR_CAMP_LQR_ALPHA05_TASK_ID,
  STAIR_CAMP_TASK_ID,
  STAIR_CAMP_TASK_IDS,
)
from hoppertrex_mjlab.hybrid.stair_trigger import (
  STAIR_TRIGGER_FORCE_N,
  STAIR_TRIGGER_WINDOW,
)

STAIR_CAMP_CONTRACT_SCHEMA_VERSION = 1
STAIR_CAMP_CANONICAL_CONTRACT_SHA256 = (
  "1d4b18db32e48b3ae8803e385a032203bdddc7f8198da9679f519bc8947190cb"
)
STAIR_CAMP_LQR_ALPHA05_CONTRACT_SHA256 = (
  "17428b449a7da2def8609001ce82c989120462f5b289163b85dc9c2971449de6"
)
STAIR_CAMP_EXPECTED_CONTRACT_SHA256 = {
  STAIR_CAMP_TASK_ID: STAIR_CAMP_CANONICAL_CONTRACT_SHA256,
  STAIR_CAMP_LQR_ALPHA05_TASK_ID: STAIR_CAMP_LQR_ALPHA05_CONTRACT_SHA256,
}
STAIR_CAMP_TRAINING_SEEDS = (1, 2, 3)
STAIR_CAMP_FRESH_UPDATES = 1000
STAIR_CAMP_EXTENSION_TOTAL_UPDATES = 3000
STAIR_CAMP_NUM_ENVS = 256
STAIR_CAMP_SAVE_INTERVAL = 100
STAIR_CAMP_INIT_STD = 0.6
STAIR_CAMP_STEPS_PER_ITERATION = 24
STAIR_CAMP_TRAINING_INFO_KEY = "stair_camp_training"
STAIR_CAMP_CURRICULUM_INFO_KEY = "stair_camp_curriculum"
STAIR_CAMP_PROGRESS_INFO_KEY = "stair_camp_progress"
STAIR_CAMP_EXPECTED_ACTOR_TERMS = (
  "base_lin_vel",
  "base_ang_vel",
  "projected_gravity",
  "velocity_command",
  "posture_command",
  "joint_pos",
  "joint_vel",
  "phase_one_hot",
  "classical_wheel_baseline",
  "nominal_leg_targets",
  "classical_errors",
  "previous_residual",
  "stair_mode",
)
STAIR_CAMP_EXPECTED_CRITIC_TAIL = (
  "step_height",
  "distance_to_riser",
  "contact_force",
)
STAIR_CAMP_WITHDRAWN_CRITIC_TERMS = (
  "friction",
  "randomization_parameters",
)
STAIR_CAMP_ACTOR_WIDTH = 52
STAIR_CAMP_CRITIC_WIDTH = 55
STAIR_CAMP_EXPECTED_TERM_WIDTHS = {
  "base_lin_vel": 3,
  "base_ang_vel": 3,
  "projected_gravity": 3,
  "velocity_command": 3,
  "posture_command": 2,
  "joint_pos": 6,
  "joint_vel": 6,
  "phase_one_hot": 9,
  "classical_wheel_baseline": 2,
  "nominal_leg_targets": 4,
  "classical_errors": 4,
  "previous_residual": 6,
  "stair_mode": 1,
  "step_height": 1,
  "distance_to_riser": 1,
  "contact_force": 1,
}


def _plain(value: Any) -> Any:
  if is_dataclass(value):
    return asdict(value)
  if isinstance(value, Mapping):
    return dict(value)
  return value


def _field(source: Any, name: str, default: Any = None) -> Any:
  if isinstance(source, Mapping):
    return source.get(name, default)
  return getattr(source, name, default)


def _finite_float(value: Any, *, name: str) -> float:
  result = float(value)
  if not math.isfinite(result):
    raise ValueError(f"{name} must be finite.")
  return result


def _callable_name(value: object, *, name: str) -> str:
  module = getattr(value, "__module__", None)
  qualified = getattr(value, "__qualname__", None)
  if not isinstance(module, str) or not isinstance(qualified, str):
    raise ValueError(f"{name} must have a stable qualified name.")
  return f"{module}.{qualified}"


# A drive-letter path written with FORWARD slashes ("D:/x/y.json") contains no
# backslash and is not POSIX-absolute, so without this pattern it would pass
# straight through the normalizer and re-introduce machine-specific text into
# the digest (audit finding: currently unreachable, hardened proactively).
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[/\\]")


def _portable_path_text(text: str) -> str:
  """Reduce an absolute filesystem path to its bare file name.

  The whole point of the canonical contract is to bind the SAME configuration
  on every machine that runs the camp, so a value that cannot survive moving
  between machines does not belong in it. Absolute paths are machine-specific
  by construction: the training host and this development checkout differ in
  drive letter and directory layout, so leaving them in made the registered
  fingerprint reproducible only on the machine that computed it. The artifact
  identity that actually matters is already bound, machine-independently, by
  the six content hashes in the `artifacts` section - the paths were pure
  redundancy carrying a portability defect.
  """

  candidate = (
    PureWindowsPath(text)
    if ("\\" in text or _WINDOWS_DRIVE_RE.match(text))
    else PurePosixPath(text)
  )
  if not candidate.is_absolute():
    return text
  return candidate.name


def _contract_value(value: Any, *, name: str) -> Any:
  """Convert config values to strict, machine-independent JSON data."""

  if isinstance(value, str):
    return _portable_path_text(value)
  if value is None or isinstance(value, (bool, int)):
    return value
  if isinstance(value, float):
    return _finite_float(value, name=name)
  if isinstance(value, Path):
    return _portable_path_text(str(value))
  if isinstance(value, slice):
    return {
      "slice": [
        _contract_value(value.start, name=f"{name}.start"),
        _contract_value(value.stop, name=f"{name}.stop"),
        _contract_value(value.step, name=f"{name}.step"),
      ]
    }
  if is_dataclass(value):
    return _contract_value(asdict(value), name=name)
  if type(value).__module__.split(".", 1)[0] == "numpy":
    converter = getattr(value, "tolist", None)
    if callable(converter):
      return _contract_value(converter(), name=name)
    scalar = getattr(value, "item", None)
    if callable(scalar):
      return _contract_value(scalar(), name=name)
  if isinstance(value, Mapping):
    result: dict[str, Any] = {}
    for key, item in value.items():
      if not isinstance(key, str):
        raise ValueError(f"{name} contains a non-string mapping key.")
      result[key] = _contract_value(item, name=f"{name}.{key}")
    return result
  if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
    return [
      _contract_value(item, name=f"{name}[{index}]")
      for index, item in enumerate(value)
    ]
  if callable(value):
    return _callable_name(value, name=name)
  raise ValueError(f"{name} contains unsupported config type {type(value).__name__}.")


def _terrain_generator_contract_value(generator: object) -> dict[str, object]:
  """Serialize the terrain declaration without MjLab's runtime size rewrite.

  ``TerrainGenerator`` assigns the generator tile size into every sub-terrain
  config in-place while the environment is constructed. The sub-terrain
  dataclass default is therefore visible before construction and the generated
  tile size is visible afterwards, even though both declarations produce the
  exact same terrain. Contract hashing happens in train preflight *and* in the
  runner, so that implementation detail must not create two hashes for one
  immutable configuration.

  The generator's own ``size`` remains digest-bound. Only each redundant
  sub-terrain ``size`` is normalized to its dataclass declaration default--the
  value present in the preregistered payload--because MjLab unconditionally
  overwrites that field before generation.
  """

  serialized = _contract_value(generator, name="terrain.generator")
  if not isinstance(serialized, dict):
    raise ValueError("StairCamp terrain generator contract must be a mapping.")
  sub_terrains = getattr(generator, "sub_terrains", None)
  serialized_sub_terrains = serialized.get("sub_terrains")
  if not isinstance(sub_terrains, Mapping) or not isinstance(
    serialized_sub_terrains, dict
  ):
    raise ValueError("StairCamp terrain generator sub-terrains are malformed.")
  if set(serialized_sub_terrains) != set(sub_terrains):
    raise ValueError("StairCamp terrain generator sub-terrain schema drifted.")
  for term_name, term in sub_terrains.items():
    if not is_dataclass(term):
      raise ValueError(
        f"StairCamp sub-terrain {term_name!r} must be a dataclass config."
      )
    size_field = next((field for field in fields(term) if field.name == "size"), None)
    if size_field is None or size_field.default is MISSING:
      raise ValueError(
        f"StairCamp sub-terrain {term_name!r} has no stable size default."
      )
    serialized_term = serialized_sub_terrains.get(term_name)
    if not isinstance(serialized_term, dict) or "size" not in serialized_term:
      raise ValueError(
        f"StairCamp sub-terrain {term_name!r} contract is malformed."
      )
    serialized_term["size"] = _contract_value(
      size_field.default,
      name=f"terrain.generator.sub_terrains.{term_name}.size_default",
    )
  return serialized


def _term_contract(term: object, *, name: str) -> dict[str, object]:
  return {
    "function": _callable_name(getattr(term, "func", None), name=f"{name}.func"),
    "weight": _finite_float(getattr(term, "weight", math.nan), name=f"{name}.weight"),
    "params": _contract_value(
      dict(getattr(term, "params", {}) or {}), name=f"{name}.params"
    ),
  }


def validate_stair_camp_progress_payload(
  progress: Mapping[str, object],
  curriculum: Mapping[str, object],
) -> dict[str, float | int]:
  """Validate the machine-readable progress snapshot against full state."""

  expected = {
    "upper_height_m",
    "trigger_rate",
    "residual_abs_mean",
    "residual_rms",
    "residual_abs_max",
    "evaluations",
  }
  if not isinstance(progress, Mapping) or set(progress) != expected:
    raise ValueError("StairCamp progress snapshot schema drifted.")
  if not isinstance(curriculum, Mapping):
    raise ValueError("StairCamp progress requires curriculum state.")

  def number(source: Mapping[str, object], field: str) -> float:
    value = source.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
      raise ValueError(f"StairCamp progress {field} must be numeric.")
    return _finite_float(value, name=f"progress {field}")

  upper = number(progress, "upper_height_m")
  trigger_rate = number(progress, "trigger_rate")
  residual_mean = number(progress, "residual_abs_mean")
  residual_rms = number(progress, "residual_rms")
  residual_max = number(progress, "residual_abs_max")
  evaluations_float = number(progress, "evaluations")
  evaluations = int(evaluations_float)
  if evaluations_float != evaluations or evaluations < 0:
    raise ValueError("StairCamp progress evaluations must be a non-negative integer.")
  upper_level = upper / 0.01
  if (
    upper < 0.01
    or upper > 0.15
    or abs(upper_level - round(upper_level)) > 1.0e-9
    or not 0.0 <= trigger_rate <= 1.0
    or min(residual_mean, residual_rms, residual_max) < 0.0
    or residual_mean > residual_rms + 1.0e-9
    or residual_rms > residual_max + 1.0e-9
    or residual_max > STAIR_CAMP_LEG_RESIDUAL_SCALE + 1.0e-6
  ):
    raise ValueError("StairCamp progress snapshot values are inconsistent.")

  curriculum_upper = number(curriculum, "upper_height_m")
  curriculum_evaluations = number(curriculum, "evaluations")
  triggered = number(curriculum, "triggered_episodes")
  completed = number(curriculum, "completed_episodes")
  residual_abs_sum = number(curriculum, "residual_abs_sum")
  residual_sq_sum = number(curriculum, "residual_sq_sum")
  residual_count_float = number(curriculum, "residual_sample_count")
  residual_count = int(residual_count_float)
  curriculum_max = number(curriculum, "residual_abs_max")
  if any(
    value != int(value)
    for value in (curriculum_evaluations, triggered, completed, residual_count_float)
  ):
    raise ValueError("StairCamp progress state counters must be integers.")
  if min(triggered, completed, residual_count) < 0 or triggered > completed:
    raise ValueError("StairCamp progress state counters are invalid.")
  denominator = max(residual_count, 1)
  expected_trigger_rate = triggered / max(completed, 1)
  expected_mean = residual_abs_sum / denominator
  expected_rms = math.sqrt(residual_sq_sum / denominator)
  comparisons = (
    (upper, curriculum_upper),
    (evaluations_float, curriculum_evaluations),
    (trigger_rate, expected_trigger_rate),
    (residual_mean, expected_mean),
    (residual_rms, expected_rms),
    (residual_max, curriculum_max),
  )
  if any(
    not math.isclose(actual, expected_value, rel_tol=1.0e-9, abs_tol=1.0e-12)
    for actual, expected_value in comparisons
  ):
    raise ValueError("StairCamp progress snapshot does not match curriculum state.")
  return {
    "upper_height_m": upper,
    "trigger_rate": trigger_rate,
    "residual_abs_mean": residual_mean,
    "residual_rms": residual_rms,
    "residual_abs_max": residual_max,
    "evaluations": evaluations,
  }


def stair_camp_artifact_bindings(env_cfg: object) -> dict[str, str]:
  """Return the exact classical artifacts consumed by a camp environment."""

  actions = getattr(env_cfg, "actions", {})
  action = actions.get("hybrid_wheel_leg") if isinstance(actions, Mapping) else None
  if action is None:
    raise ValueError("StairCamp environment has no hybrid_wheel_leg action.")
  names = (
    "controller_gain_hash",
    "calibration_hash",
    "yaw_calibration_hash",
    "posture_map_hash",
    "posture_artifact_hash",
    "station_calibration_hash",
  )
  result: dict[str, str] = {}
  for name in names:
    value = getattr(action, name, None)
    if not isinstance(value, str) or not value.strip():
      raise ValueError(f"StairCamp artifact binding {name} is missing.")
    result[name] = value
  return result


def stair_camp_contract_payload(
  env_cfg: object,
  agent_cfg: object,
) -> dict[str, object]:
  """Build the canonical training/evaluation contract, excluding total budget.

  ``max_iterations`` is deliberately excluded: 1000 and the one registered
  extension target 3000 are two budgets of the same policy/environment contract.
  """

  task_id = getattr(env_cfg, "stair_camp_task_id", None)
  if task_id not in STAIR_CAMP_TASK_IDS:
    raise ValueError("Environment is not marked as a registered StairCamp task.")
  expected_gain_scale = (
    STAIR_CAMP_FAILURE_LQR_GAIN_SCALE
    if task_id == STAIR_CAMP_LQR_ALPHA05_TASK_ID
    else 1.0
  )
  actions = getattr(env_cfg, "actions", {})
  action = actions.get("hybrid_wheel_leg") if isinstance(actions, Mapping) else None
  if action is None:
    raise ValueError("StairCamp environment has no hybrid_wheel_leg action.")
  observations = getattr(env_cfg, "observations", {})
  actor_group = observations.get("actor") if isinstance(observations, Mapping) else None
  critic_group = observations.get("critic") if isinstance(observations, Mapping) else None
  actor_terms = tuple(getattr(actor_group, "terms", {}))
  critic_terms = tuple(getattr(critic_group, "terms", {}))
  if actor_terms != STAIR_CAMP_EXPECTED_ACTOR_TERMS:
    raise ValueError("StairCamp actor term order does not match the frozen contract.")
  if critic_terms != actor_terms + STAIR_CAMP_EXPECTED_CRITIC_TAIL:
    raise ValueError("StairCamp critic term order does not match the frozen contract.")

  actor_width = sum(STAIR_CAMP_EXPECTED_TERM_WIDTHS[name] for name in actor_terms)
  critic_width = sum(STAIR_CAMP_EXPECTED_TERM_WIDTHS[name] for name in critic_terms)
  if actor_width != STAIR_CAMP_ACTOR_WIDTH or critic_width != STAIR_CAMP_CRITIC_WIDTH:
    raise ValueError("StairCamp observation width contract is internally inconsistent.")

  agent = _plain(agent_cfg)
  actor = _plain(_field(agent, "actor", {}))
  critic = _plain(_field(agent, "critic", {}))
  distribution = _plain(_field(actor, "distribution_cfg", {}))
  algorithm = _plain(_field(agent, "algorithm", {}))
  scene = getattr(env_cfg, "scene", None)
  terrain = getattr(scene, "terrain", None)
  generator = getattr(terrain, "terrain_generator", None)
  if generator is None:
    raise ValueError("StairCamp training terrain generator is missing.")
  curriculum = getattr(env_cfg, "curriculum", {})
  if not isinstance(curriculum, Mapping) or tuple(curriculum) != ("stair_height_band",):
    raise ValueError("StairCamp curriculum terms drifted from the frozen contract.")
  curriculum_term = curriculum["stair_height_band"]
  curriculum_params = dict(getattr(curriculum_term, "params", {}) or {})
  rewards = getattr(env_cfg, "rewards", {})
  if not isinstance(rewards, Mapping):
    raise ValueError("StairCamp rewards must be a mapping.")

  payload: dict[str, object] = {
    "schema_version": STAIR_CAMP_CONTRACT_SCHEMA_VERSION,
    "task": task_id,
    "interface": {
      "actor_terms": list(actor_terms),
      "critic_terms": list(critic_terms),
      "term_widths": {
        name: STAIR_CAMP_EXPECTED_TERM_WIDTHS[name] for name in critic_terms
      },
      "actor_width": actor_width,
      "critic_width": critic_width,
      "withdrawn_critic_terms": list(STAIR_CAMP_WITHDRAWN_CRITIC_TERMS),
    },
    "environment": {
      "action_config": _contract_value(action, name="environment.action"),
      "observation_groups": [
        {
          "name": name,
          "config": _contract_value(group, name=f"observations.{name}"),
        }
        for name, group in observations.items()
      ],
      "commands": [
        {
          "name": name,
          "config": _contract_value(term, name=f"commands.{name}"),
        }
        for name, term in getattr(env_cfg, "commands", {}).items()
      ],
      "events": [
        {
          "name": name,
          "config": _contract_value(term, name=f"events.{name}"),
        }
        for name, term in getattr(env_cfg, "events", {}).items()
      ],
      "terminations": [
        {
          "name": name,
          "config": _contract_value(term, name=f"terminations.{name}"),
        }
        for name, term in getattr(env_cfg, "terminations", {}).items()
      ],
      "sensors": _contract_value(
        tuple(getattr(scene, "sensors", ())), name="environment.sensors"
      ),
    },
    "action": {
      "mask": [bool(value) for value in getattr(action, "action_mask", ())],
      "scales": [
        _finite_float(value, name="action scale")
        for value in getattr(action, "action_scales", ())
      ],
      "leg_scale_rad": STAIR_CAMP_LEG_RESIDUAL_SCALE,
      "trigger_force_n": _finite_float(
        getattr(action, "stair_trigger_force_n", math.nan),
        name="trigger force",
      ),
      "trigger_window": int(getattr(action, "stair_trigger_window", -1)),
      "trigger_sensor": getattr(action, "stair_trigger_sensor_name", None),
      "freeze_at_trigger": bool(
        getattr(action, "stair_mode_freezes_leg_reference", False)
      ),
      "lqr_gain_scale": _finite_float(
        getattr(action, "stair_mode_lqr_gain_scale", 1.0),
        name="stair-mode LQR gain scale",
      ),
    },
    "terrain": {
      "entity_type": getattr(terrain, "terrain_type", None),
      "max_init_terrain_level": int(
        getattr(terrain, "max_init_terrain_level", -1)
      ),
      "generator": _terrain_generator_contract_value(generator),
      "episode_length_s": _finite_float(
        getattr(env_cfg, "episode_length_s", math.nan),
        name="episode length",
      ),
      "num_envs": int(getattr(scene, "num_envs", -1)),
      "decimation": int(getattr(env_cfg, "decimation", -1)),
      "sim_dt": _finite_float(
        getattr(
          getattr(getattr(env_cfg, "sim", None), "mujoco", None),
          "timestep",
          math.nan,
        ),
        name="sim dt",
      ),
      "auto_reset": bool(getattr(env_cfg, "auto_reset", False)),
      "finite_horizon": bool(getattr(env_cfg, "is_finite_horizon", False)),
      "scale_rewards_by_dt": bool(
        getattr(env_cfg, "scale_rewards_by_dt", False)
      ),
    },
    "rewards": [
      {"name": name, **_term_contract(term, name=f"rewards.{name}")}
      for name, term in rewards.items()
    ],
    "curriculum": {
      "function": _callable_name(
        getattr(curriculum_term, "func", None), name="curriculum.func"
      ),
      "params": _contract_value(curriculum_params, name="curriculum.params"),
    },
    "ppo": {
      "num_steps_per_env": int(_field(agent, "num_steps_per_env", -1)),
      "save_interval": int(_field(agent, "save_interval", -1)),
      "obs_groups": _contract_value(
        _field(agent, "obs_groups", {}), name="ppo.obs_groups"
      ),
      "clip_actions": _contract_value(
        _field(agent, "clip_actions", None), name="ppo.clip_actions"
      ),
      "runner_class": _field(agent, "class_name", None),
      "init_std": _finite_float(
        _field(distribution, "init_std", math.nan), name="init std"
      ),
      "active_mask": [bool(value) for value in _field(distribution, "active_mask", ())],
      "actor": _contract_value(actor, name="ppo.actor"),
      "critic": _contract_value(critic, name="ppo.critic"),
      "algorithm": _contract_value(algorithm, name="ppo.algorithm"),
    },
    "artifacts": stair_camp_artifact_bindings(env_cfg),
  }

  action_contract = payload["action"]
  assert isinstance(action_contract, dict)
  if tuple(action_contract["mask"]) != STAIR_CAMP_ACTION_MASK:
    raise ValueError("StairCamp environment action mask drifted.")
  if tuple(payload["ppo"]["active_mask"]) != STAIR_CAMP_ACTION_MASK:  # type: ignore[index]
    raise ValueError("StairCamp PPO active mask drifted.")
  if action_contract["trigger_force_n"] != STAIR_TRIGGER_FORCE_N:
    raise ValueError("StairCamp trigger force drifted.")
  if action_contract["trigger_window"] != STAIR_TRIGGER_WINDOW:
    raise ValueError("StairCamp trigger window drifted.")
  if action_contract["lqr_gain_scale"] != expected_gain_scale:
    raise ValueError("StairCamp task/LQR gain-scale binding drifted.")
  return payload


def stair_camp_contract_hash(env_cfg: object, agent_cfg: object) -> str:
  payload = stair_camp_contract_payload(env_cfg, agent_cfg)
  encoded = json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def expected_stair_camp_contract_hash(task_id: object) -> str:
  """Return the exact primary or isolated failure-rung contract hash."""

  try:
    return STAIR_CAMP_EXPECTED_CONTRACT_SHA256[task_id]  # type: ignore[index]
  except (KeyError, TypeError) as exc:
    raise ValueError("Unknown StairCamp task contract.") from exc


def bind_stair_camp_contract(env_cfg: object, agent_cfg: object) -> str:
  """Bind runtime config to its exact preregistered StairCamp task hash."""

  task_id = getattr(env_cfg, "stair_camp_task_id", None)
  expected = expected_stair_camp_contract_hash(task_id)
  contract_sha256 = stair_camp_contract_hash(env_cfg, agent_cfg)
  if contract_sha256 != expected:
    raise ValueError(
      "StairCamp config/artifact contract drifted from the preregistered hash."
    )
  existing = getattr(env_cfg, "stair_camp_contract_sha256", None)
  if existing not in (None, contract_sha256):
    raise ValueError("StairCamp environment carries a conflicting contract hash.")
  env_cfg.stair_camp_contract_sha256 = contract_sha256
  return contract_sha256


def stair_camp_init_std(agent_cfg: object) -> float:
  """Return the OBSERVED exploration init_std, pinned to the registered value.

  Provenance records must certify what the run actually used, not a literal
  retyped alongside the check. Reading the value back from the live PPO
  configuration is what makes the recorded `init_std` evidence rather than an
  assertion, and pinning it here keeps the registered 0.6 fail-closed.
  """

  actor = _field(agent_cfg, "actor")
  distribution = _field(actor, "distribution_cfg", {})
  observed = _finite_float(
    _field(distribution, "init_std", math.nan), name="init std"
  )
  if observed != STAIR_CAMP_INIT_STD:
    raise ValueError("StairCamp exploration init_std drifted from 0.6.")
  return observed


def validate_stair_camp_training_request(
  env_cfg: object,
  agent_cfg: object,
  *,
  resume: bool,
) -> str:
  """Validate the immutable fresh/extension surface before runner launch."""

  def exact_int(source: object, name: str) -> int:
    value = _field(source, name)
    if isinstance(value, bool) or not isinstance(value, int):
      raise ValueError(f"StairCamp {name} must be an integer.")  # noqa: TRY004
    return value

  if resume is not True and resume is not False:
    raise ValueError("StairCamp resume must be a boolean.")
  task_id = getattr(env_cfg, "stair_camp_task_id", None)
  if task_id not in STAIR_CAMP_TASK_IDS:
    raise ValueError("StairCamp task marker is missing.")
  failure_variant = getattr(env_cfg, "stair_camp_failure_ladder_variant", None)
  if task_id == STAIR_CAMP_LQR_ALPHA05_TASK_ID:
    if failure_variant != STAIR_CAMP_FAILURE_LADDER_VARIANT:
      raise ValueError("StairCamp alpha=0.5 failure-rung marker is missing.")
  elif failure_variant is not None:
    raise ValueError("Primary StairCamp must not carry a failure-rung marker.")
  if getattr(env_cfg, "stair_camp_zero_initialize_actor_output", None) is not True:
    raise ValueError("StairCamp deterministic-mean zero-init marker is missing.")
  if getattr(env_cfg, "stair_camp_training_contract", None) is not True:
    raise ValueError("StairCamp training-contract marker is missing.")
  if (
    getattr(env_cfg, "stair_camp_contract_schema_version", None)
    != STAIR_CAMP_CONTRACT_SCHEMA_VERSION
  ):
    raise ValueError("StairCamp environment contract schema drifted.")

  seed = exact_int(agent_cfg, "seed")
  target = exact_int(agent_cfg, "max_iterations")
  if task_id == STAIR_CAMP_LQR_ALPHA05_TASK_ID:
    if resume:
      raise ValueError("StairCamp alpha=0.5 failure rung must be a fresh retrain.")
    if seed != 1:
      raise ValueError("StairCamp alpha=0.5 failure rung is restricted to seed 1.")
    if target not in (STAIR_CAMP_FRESH_UPDATES, STAIR_CAMP_EXTENSION_TOTAL_UPDATES):
      raise ValueError(
        "StairCamp alpha=0.5 failure rung must use final main budget 1000 or 3000."
      )
  else:
    if seed not in STAIR_CAMP_TRAINING_SEEDS:
      raise ValueError("StairCamp training seed must be one of {1, 2, 3}.")
    expected_target = (
      STAIR_CAMP_EXTENSION_TOTAL_UPDATES if resume else STAIR_CAMP_FRESH_UPDATES
    )
    if target != expected_target:
      raise ValueError(
        f"StairCamp {'extension' if resume else 'fresh'} total budget must be "
        f"exactly {expected_target} updates."
      )
  if exact_int(agent_cfg, "save_interval") != STAIR_CAMP_SAVE_INTERVAL:
    raise ValueError("StairCamp save interval drifted from 100 updates.")
  if (
    exact_int(agent_cfg, "num_steps_per_env")
    != STAIR_CAMP_STEPS_PER_ITERATION
  ):
    raise ValueError("StairCamp rollout length drifted from 24 steps.")
  scene = getattr(env_cfg, "scene", None)
  if exact_int(scene, "num_envs") != STAIR_CAMP_NUM_ENVS:
    raise ValueError("StairCamp training requires exactly 256 environments.")

  actor = _field(agent_cfg, "actor")
  distribution = _field(actor, "distribution_cfg", {})
  stair_camp_init_std(agent_cfg)
  if tuple(_field(distribution, "active_mask", ())) != STAIR_CAMP_ACTION_MASK:
    raise ValueError("StairCamp PPO active mask drifted.")
  return bind_stair_camp_contract(env_cfg, agent_cfg)


__all__ = [
  "STAIR_CAMP_ACTOR_WIDTH",
  "STAIR_CAMP_CANONICAL_CONTRACT_SHA256",
  "STAIR_CAMP_EXPECTED_CONTRACT_SHA256",
  "STAIR_CAMP_LQR_ALPHA05_CONTRACT_SHA256",
  "STAIR_CAMP_CONTRACT_SCHEMA_VERSION",
  "STAIR_CAMP_CRITIC_WIDTH",
  "STAIR_CAMP_CURRICULUM_INFO_KEY",
  "STAIR_CAMP_EXPECTED_ACTOR_TERMS",
  "STAIR_CAMP_EXPECTED_CRITIC_TAIL",
  "STAIR_CAMP_EXPECTED_TERM_WIDTHS",
  "STAIR_CAMP_EXTENSION_TOTAL_UPDATES",
  "STAIR_CAMP_FRESH_UPDATES",
  "STAIR_CAMP_INIT_STD",
  "STAIR_CAMP_NUM_ENVS",
  "STAIR_CAMP_PROGRESS_INFO_KEY",
  "STAIR_CAMP_SAVE_INTERVAL",
  "STAIR_CAMP_STEPS_PER_ITERATION",
  "STAIR_CAMP_TRAINING_INFO_KEY",
  "STAIR_CAMP_TRAINING_SEEDS",
  "STAIR_CAMP_WITHDRAWN_CRITIC_TERMS",
  "bind_stair_camp_contract",
  "expected_stair_camp_contract_hash",
  "stair_camp_artifact_bindings",
  "stair_camp_contract_hash",
  "stair_camp_init_std",
  "stair_camp_contract_payload",
  "validate_stair_camp_progress_payload",
  "validate_stair_camp_training_request",
]
