"""RSL-RL runner safeguards for the Hybrid v2 checkpoint chain."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from mjlab.rl import MjlabOnPolicyRunner
from torch import nn

from hoppertrex_mjlab.hybrid.config import STAIR_CAMP_TASK_IDS
from hoppertrex_mjlab.hybrid.stair_camp_contract import (
  STAIR_CAMP_CONTRACT_SCHEMA_VERSION,
  STAIR_CAMP_CURRICULUM_INFO_KEY,
  STAIR_CAMP_PROGRESS_INFO_KEY,
  STAIR_CAMP_TRAINING_INFO_KEY,
  bind_stair_camp_contract,
  expected_stair_camp_contract_hash,
  stair_camp_artifact_bindings,
  stair_camp_init_std,
  validate_stair_camp_progress_payload,
)
from hoppertrex_mjlab.hybrid.stair_dynamic import DYNAMIC_STAIR_TASK_ID
from hoppertrex_mjlab.hybrid.stair_dynamic_contract import (
  DYNAMIC_STAIR_CONTRACT_SCHEMA_VERSION,
  DYNAMIC_STAIR_CURRICULUM_INFO_KEY,
  DYNAMIC_STAIR_MIGRATION_INFO_KEY,
  DYNAMIC_STAIR_PROGRESS_INFO_KEY,
  DYNAMIC_STAIR_TRAINING_INFO_KEY,
  bind_dynamic_stair_contract,
  dynamic_stair_artifact_bindings,
  validate_dynamic_stair_progress_payload,
)

REPOSITORY_PATH = Path(__file__).resolve().parents[3]


def repository_git_sha() -> str:
  """Return the code revision used by the current training process."""

  completed = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=REPOSITORY_PATH,
    check=True,
    capture_output=True,
    text=True,
  )
  return completed.stdout.strip()


def merge_hybrid_checkpoint_infos(
  loaded_infos: Mapping[str, Any] | None,
  save_infos: Mapping[str, Any] | None,
) -> dict[str, Any]:
  """Preserve bootstrap/migration provenance across later checkpoint saves."""

  return {**dict(loaded_infos or {}), **dict(save_infos or {})}


def _unwrapped_env(env: object) -> object:
  return getattr(env, "unwrapped", env)


def hybrid_action_scales_from_env(env: object) -> list[float]:
  """Read the six applied scales from the real wrapped environment config."""

  unwrapped = _unwrapped_env(env)
  env_cfg = getattr(unwrapped, "cfg", None)
  actions = getattr(env_cfg, "actions", {})
  action = actions.get("hybrid_wheel_leg") if isinstance(actions, dict) else None
  scales = getattr(action, "action_scales", ())
  if len(scales) != 6:
    raise ValueError("Hybrid runner expected six environment action scales.")
  return [float(value) for value in scales]


def is_stair_camp_env(env: object) -> bool:
  cfg = getattr(_unwrapped_env(env), "cfg", None)
  return (
    getattr(cfg, "stair_camp_task_id", None) in STAIR_CAMP_TASK_IDS
    and getattr(cfg, "stair_camp_zero_initialize_actor_output", False) is True
  )


def is_stair_dynamic_env(env: object) -> bool:
  cfg = getattr(_unwrapped_env(env), "cfg", None)
  return getattr(cfg, "stair_dynamic_task_id", None) == DYNAMIC_STAIR_TASK_ID


def zero_initialize_stair_camp_actor_output(actor: nn.Module) -> nn.Linear:
  """Zero the unique six-output actor head and return it."""

  candidates = [
    module
    for module in actor.modules()
    if isinstance(module, nn.Linear)
    and module.out_features == 6
    and module.bias is not None
    and tuple(module.bias.shape) == (6,)
  ]
  if len(candidates) != 1:
    raise ValueError(
      "Expected exactly one six-output StairCamp actor head, "
      f"found {len(candidates)}."
    )
  head = candidates[0]
  with torch.no_grad():
    head.weight.zero_()
    head.bias.zero_()
  return head


def _runner_train_cfg(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> object:
  if len(args) >= 2:
    return args[1]
  cfg = kwargs.get("train_cfg")
  if cfg is None:
    raise ValueError("Hybrid runner requires a train_cfg argument.")
  return cfg


def _cfg_field(cfg: object, name: str, default: Any = None) -> Any:
  if isinstance(cfg, Mapping):
    return cfg.get(name, default)
  return getattr(cfg, name, default)


def _camp_curriculum_from_env(env: object) -> Any | None:
  return getattr(_unwrapped_env(env), "stair_camp_curriculum_state", None)


def _dynamic_curriculum_from_env(env: object) -> Any | None:
  return getattr(_unwrapped_env(env), "stair_dynamic_curriculum_state", None)


def stair_dynamic_effective_load_cfg(
  checkpoint: Mapping[str, Any],
  requested: Mapping[str, Any] | None,
) -> dict[str, Any] | Mapping[str, Any] | None:
  """Keep the v3 runner's fresh optimizer for a migration-only checkpoint."""

  infos = checkpoint.get("infos")
  if not isinstance(infos, Mapping):
    return requested
  migration = infos.get(DYNAMIC_STAIR_MIGRATION_INFO_KEY)
  training = infos.get(DYNAMIC_STAIR_TRAINING_INFO_KEY)
  if migration is not None and not isinstance(migration, Mapping):
    raise ValueError("StairDynamic migration provenance is malformed.")
  if training is not None and not isinstance(training, Mapping):
    raise ValueError("StairDynamic training provenance is malformed.")
  if not isinstance(migration, Mapping) or training is not None:
    return requested
  if requested is None:
    effective: dict[str, Any] = {
      "actor": True,
      "critic": True,
      "optimizer": False,
      "iteration": True,
      "rnd": True,
    }
  else:
    effective = dict(requested)
    effective["optimizer"] = False
  return effective


class HybridOnPolicyRunner(MjlabOnPolicyRunner):
  """Hybrid runner with provenance and exact StairCamp extension semantics."""

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    self._hybrid_loaded_infos: dict[str, Any] = {}
    self._hybrid_training_git_sha = repository_git_sha()
    env = args[0] if args else kwargs.get("env")
    if env is None:
      raise ValueError("Hybrid runner requires an environment.")
    train_cfg = _runner_train_cfg(args, kwargs)
    self._hybrid_action_scales = hybrid_action_scales_from_env(env)
    self._stair_camp = is_stair_camp_env(env)
    self._stair_camp_training_env = False
    self._stair_camp_task_id: str | None = None
    self._stair_camp_training_seed: int | None = None
    self._stair_camp_contract_hash: str | None = None
    self._stair_camp_artifacts: dict[str, str] | None = None
    self._stair_camp_init_std: float | None = None
    self._stair_camp_loaded_completed_updates = 0
    self._stair_dynamic = is_stair_dynamic_env(env)
    self._stair_dynamic_training_env = False
    self._stair_dynamic_training_seed: int | None = None
    self._stair_dynamic_contract_hash: str | None = None
    self._stair_dynamic_artifacts: dict[str, str] | None = None
    self._stair_dynamic_maneuver_bindings: dict[str, str] | None = None
    self._stair_dynamic_maneuver_hash: str | None = None
    self._stair_dynamic_loaded_completed_updates = 0
    if self._stair_camp:
      env_cfg = _unwrapped_env(env).cfg
      self._stair_camp_task_id = str(env_cfg.stair_camp_task_id)
      self._stair_camp_training_env = (
        getattr(env_cfg, "stair_camp_training_contract", False) is True
      )
      self._stair_camp_training_seed = int(_cfg_field(train_cfg, "seed", -1))
      if self._stair_camp_training_seed < 0:
        raise ValueError("StairCamp runner requires a non-negative training seed.")
      if self._stair_camp_training_env:
        self._stair_camp_contract_hash = bind_stair_camp_contract(
          env_cfg, train_cfg
        )
      else:
        expected_contract = expected_stair_camp_contract_hash(
          self._stair_camp_task_id
        )
        existing_contract = getattr(env_cfg, "stair_camp_contract_sha256", None)
        if existing_contract not in (None, expected_contract):
          raise ValueError("StairCamp evaluation contract marker drifted.")
        env_cfg.stair_camp_contract_sha256 = expected_contract
        self._stair_camp_contract_hash = expected_contract
      self._stair_camp_artifacts = stair_camp_artifact_bindings(env_cfg)
      self._stair_camp_init_std = stair_camp_init_std(train_cfg)
    if self._stair_dynamic:
      env_cfg = _unwrapped_env(env).cfg
      self._stair_dynamic_training_env = (
        getattr(env_cfg, "stair_dynamic_training_contract", False) is True
      )
      self._stair_dynamic_training_seed = int(_cfg_field(train_cfg, "seed", -1))
      if self._stair_dynamic_training_seed != 1:
        raise ValueError("StairDynamic runner currently permits only seed 1.")
      if getattr(env_cfg, "stair_dynamic_maneuver_qualified", False) is not True:
        raise ValueError("StairDynamic runner requires the qualified maneuver artifact.")
      if self._stair_dynamic_training_env:
        self._stair_dynamic_contract_hash = bind_dynamic_stair_contract(
          env_cfg, train_cfg
        )
      self._stair_dynamic_artifacts = dynamic_stair_artifact_bindings(env_cfg)
      bindings = getattr(env_cfg, "stair_dynamic_maneuver_bindings", None)
      if not isinstance(bindings, Mapping):
        raise ValueError("StairDynamic maneuver bindings are missing.")
      self._stair_dynamic_maneuver_bindings = dict(bindings)
      if bindings.get("git_sha") != self._hybrid_training_git_sha:
        raise ValueError("StairDynamic maneuver Git binding does not match.")
      action = env_cfg.actions["hybrid_wheel_leg"]
      self._stair_dynamic_maneuver_hash = str(
        action.dynamic_stair_maneuver.maneuver_hash
      )
    super().__init__(*args, **kwargs)
    if self._stair_camp:
      zero_initialize_stair_camp_actor_output(self.alg.get_policy())

  def _stair_camp_training_record(self) -> dict[str, object]:
    completed_updates = int(self.current_learning_iteration) + 1
    return {
      "schema_version": STAIR_CAMP_CONTRACT_SCHEMA_VERSION,
      "task": self._stair_camp_task_id,
      "training_seed": self._stair_camp_training_seed,
      "git_sha": self._hybrid_training_git_sha,
      "contract_sha256": self._stair_camp_contract_hash,
      "artifact_bindings": dict(self._stair_camp_artifacts or {}),
      "action_scales": list(self._hybrid_action_scales),
      "zero_initialized_deterministic_mean": True,
      "init_std": self._stair_camp_init_std,
      "completed_updates": completed_updates,
    }

  def _stair_dynamic_training_record(self) -> dict[str, object]:
    completed_updates = int(self.current_learning_iteration) + 1
    bindings = self._stair_dynamic_maneuver_bindings or {}
    return {
      "schema_version": DYNAMIC_STAIR_CONTRACT_SCHEMA_VERSION,
      "task": DYNAMIC_STAIR_TASK_ID,
      "training_seed": self._stair_dynamic_training_seed,
      "git_sha": self._hybrid_training_git_sha,
      "contract_sha256": self._stair_dynamic_contract_hash,
      "artifact_bindings": dict(self._stair_dynamic_artifacts or {}),
      "action_scales": list(self._hybrid_action_scales),
      "maneuver_sha256": self._stair_dynamic_maneuver_hash,
      "source_stage5_checkpoint_sha256": bindings.get(
        "stage5_checkpoint_sha256"
      ),
      "source_stage5_gate_sha256": bindings.get(
        "stage5_formal_gate_sha256"
      ),
      "stage5_prefix_preserved_and_new_columns_zero": True,
      "completed_updates": completed_updates,
    }

  def _validate_stair_dynamic_loaded_record(
    self,
    record: Mapping[str, Any],
  ) -> int:
    expected = set(self._stair_dynamic_training_record())
    if set(record) != expected:
      raise ValueError("StairDynamic checkpoint training provenance schema drifted.")

    def exact_int(name: str, minimum: int = 0) -> int:
      value = record.get(name)
      if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"StairDynamic checkpoint {name} is invalid.")
      return int(value)

    if exact_int("schema_version") != DYNAMIC_STAIR_CONTRACT_SCHEMA_VERSION:
      raise ValueError("StairDynamic checkpoint schema version drifted.")
    if record.get("task") != DYNAMIC_STAIR_TASK_ID:
      raise ValueError("StairDynamic checkpoint task does not match.")
    if exact_int("training_seed") != self._stair_dynamic_training_seed:
      raise ValueError("StairDynamic checkpoint seed does not match.")
    if record.get("git_sha") != self._hybrid_training_git_sha:
      raise ValueError("StairDynamic checkpoint Git SHA does not match.")
    if (
      self._stair_dynamic_training_env
      and record.get("contract_sha256") != self._stair_dynamic_contract_hash
    ):
      raise ValueError("StairDynamic checkpoint contract hash does not match.")
    contract = record.get("contract_sha256")
    if not isinstance(contract, str) or len(contract) != 64:
      raise ValueError("StairDynamic checkpoint contract SHA is invalid.")
    if record.get("artifact_bindings") != self._stair_dynamic_artifacts:
      raise ValueError("StairDynamic checkpoint artifact bindings do not match.")
    if record.get("action_scales") != self._hybrid_action_scales:
      raise ValueError("StairDynamic checkpoint action scales do not match.")
    bindings = self._stair_dynamic_maneuver_bindings or {}
    if (
      record.get("maneuver_sha256") != self._stair_dynamic_maneuver_hash
      or record.get("source_stage5_checkpoint_sha256")
      != bindings.get("stage5_checkpoint_sha256")
      or record.get("source_stage5_gate_sha256")
      != bindings.get("stage5_formal_gate_sha256")
      or record.get("stage5_prefix_preserved_and_new_columns_zero") is not True
    ):
      raise ValueError("StairDynamic checkpoint transfer provenance drifted.")
    return exact_int("completed_updates", 1)

  def _validate_stair_dynamic_migration_record(
    self,
    record: Mapping[str, Any],
  ) -> None:
    bindings = self._stair_dynamic_maneuver_bindings or {}
    if (
      record.get("source_task") != "HopperTrex-Hybrid-v2-Stage5"
      or record.get("target_task") != DYNAMIC_STAIR_TASK_ID
      or record.get("source_seed") != 1
      or record.get("source_completed_updates") != 100
      or record.get("source_actor_width") != 34
      or record.get("target_actor_width") != 52
      or record.get("source_critic_width") != 34
      or record.get("target_critic_width") != 56
      or record.get("source_checkpoint_sha256")
      != bindings.get("stage5_checkpoint_sha256")
      or record.get("source_gate_sha256")
      != bindings.get("stage5_formal_gate_sha256")
    ):
      raise ValueError("StairDynamic Stage5 migration provenance does not match.")

  def _validate_stair_camp_loaded_record(
    self,
    record: Mapping[str, Any],
  ) -> int:
    expected = set(self._stair_camp_training_record())
    if set(record) != expected:
      raise ValueError("StairCamp checkpoint training provenance schema drifted.")

    def exact_int(name: str, *, minimum: int = 0) -> int:
      value = record.get(name)
      if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"StairCamp checkpoint {name} is invalid.")
      return value

    if exact_int("schema_version") != STAIR_CAMP_CONTRACT_SCHEMA_VERSION:
      raise ValueError("StairCamp checkpoint schema version drifted.")
    if record.get("task") != self._stair_camp_task_id:
      raise ValueError("StairCamp checkpoint task does not match.")
    if exact_int("training_seed") != self._stair_camp_training_seed:
      raise ValueError("StairCamp checkpoint seed does not match.")
    if record.get("git_sha") != self._hybrid_training_git_sha:
      raise ValueError("StairCamp checkpoint Git SHA does not match.")
    if record.get("contract_sha256") != self._stair_camp_contract_hash:
      raise ValueError("StairCamp checkpoint contract hash does not match.")
    if record.get("artifact_bindings") != self._stair_camp_artifacts:
      raise ValueError("StairCamp checkpoint artifact bindings do not match.")
    if record.get("action_scales") != self._hybrid_action_scales:
      raise ValueError("StairCamp checkpoint action scales do not match.")
    init_std = record.get("init_std")
    if (
      record.get("zero_initialized_deterministic_mean") is not True
      or isinstance(init_std, bool)
      or not isinstance(init_std, (int, float))
      or float(init_std) != self._stair_camp_init_std
    ):
      raise ValueError("StairCamp checkpoint initialization provenance drifted.")
    return exact_int("completed_updates", minimum=1)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    requested_load_cfg = load_cfg
    full_training_load = load_cfg is None or bool(load_cfg.get("iteration"))
    if self._stair_dynamic and full_training_load:
      checkpoint = torch.load(path, weights_only=False, map_location="cpu")
      if not isinstance(checkpoint, Mapping):
        raise ValueError("StairDynamic checkpoint root must be a mapping.")
      load_cfg = stair_dynamic_effective_load_cfg(checkpoint, load_cfg)
    infos = super().load(path, load_cfg, strict, map_location)
    self._hybrid_loaded_infos = dict(infos or {})
    if self._stair_camp:
      record = self._hybrid_loaded_infos.get(STAIR_CAMP_TRAINING_INFO_KEY)
      if not isinstance(record, Mapping):
        raise ValueError("StairCamp checkpoint is missing training provenance.")
      completed = self._validate_stair_camp_loaded_record(record)
      self._stair_camp_loaded_completed_updates = completed
      full_training_load = requested_load_cfg is None or bool(
        requested_load_cfg.get("iteration")
      )
      if full_training_load:
        if int(self.current_learning_iteration) + 1 != completed:
          raise ValueError("StairCamp checkpoint iteration/update count drifted.")
        curriculum_payload = self._hybrid_loaded_infos.get(
          STAIR_CAMP_CURRICULUM_INFO_KEY
        )
        curriculum = _camp_curriculum_from_env(self.env)
        if curriculum is None or not hasattr(curriculum, "load_state_dict"):
          raise ValueError("StairCamp environment has no restorable curriculum state.")
        if not isinstance(curriculum_payload, Mapping):
          raise ValueError("StairCamp checkpoint is missing curriculum state.")
        progress_payload = self._hybrid_loaded_infos.get(
          STAIR_CAMP_PROGRESS_INFO_KEY
        )
        if not isinstance(progress_payload, Mapping):
          raise ValueError("StairCamp checkpoint is missing progress state.")
        validate_stair_camp_progress_payload(progress_payload, curriculum_payload)
        curriculum.load_state_dict(curriculum_payload)
        unwrapped = _unwrapped_env(self.env)
        common_step = int(getattr(unwrapped, "common_step_counter", -1))
        if (
          common_step < 0
          or curriculum.last_processed_step > common_step
          or common_step >= curriculum.next_evaluation_step
        ):
          raise ValueError("StairCamp environment/curriculum step state drifted.")
        # The physics state is intentionally not checkpointed by MjLab. Start
        # the extension with fresh episodes sampled from the restored band.
        self.env.reset()
    if getattr(self, "_stair_dynamic", False):
      training_record = self._hybrid_loaded_infos.get(
        DYNAMIC_STAIR_TRAINING_INFO_KEY
      )
      migration_record = self._hybrid_loaded_infos.get(
        DYNAMIC_STAIR_MIGRATION_INFO_KEY
      )
      full_training_load = requested_load_cfg is None or bool(
        requested_load_cfg.get("iteration")
      )
      if isinstance(training_record, Mapping):
        if migration_record is not None and not isinstance(migration_record, Mapping):
          raise ValueError("StairDynamic migration provenance is malformed.")
        completed = self._validate_stair_dynamic_loaded_record(training_record)
        self._stair_dynamic_loaded_completed_updates = completed
        if full_training_load:
          if int(self.current_learning_iteration) + 1 != completed:
            raise ValueError(
              "StairDynamic checkpoint iteration/update count drifted."
            )
          curriculum_payload = self._hybrid_loaded_infos.get(
            DYNAMIC_STAIR_CURRICULUM_INFO_KEY
          )
          progress_payload = self._hybrid_loaded_infos.get(
            DYNAMIC_STAIR_PROGRESS_INFO_KEY
          )
          curriculum = _dynamic_curriculum_from_env(self.env)
          if curriculum is None or not hasattr(curriculum, "load_state_dict"):
            raise ValueError(
              "StairDynamic environment has no restorable curriculum state."
            )
          if not isinstance(curriculum_payload, Mapping):
            raise ValueError("StairDynamic checkpoint is missing curriculum state.")
          if not isinstance(progress_payload, Mapping):
            raise ValueError("StairDynamic checkpoint is missing progress state.")
          validate_dynamic_stair_progress_payload(
            progress_payload,
            curriculum_payload,
          )
          curriculum.load_state_dict(curriculum_payload)
          unwrapped = _unwrapped_env(self.env)
          common_step = int(getattr(unwrapped, "common_step_counter", -1))
          if (
            common_step < 0
            or curriculum.last_processed_step > common_step
            or common_step >= curriculum.next_evaluation_step
          ):
            raise ValueError(
              "StairDynamic environment/curriculum step state drifted."
            )
          self.env.reset()
      elif isinstance(migration_record, Mapping):
        self._validate_stair_dynamic_migration_record(migration_record)
        self._stair_dynamic_loaded_completed_updates = 0
        if full_training_load:
          if int(self.current_learning_iteration) != 0:
            raise ValueError("StairDynamic migration must start at iteration zero.")
          common_step = int(
            getattr(_unwrapped_env(self.env), "common_step_counter", -1)
          )
          if common_step != 0:
            raise ValueError("StairDynamic migration must reset env step state.")
          self.env.reset()
      else:
        raise ValueError(
          "StairDynamic accepts only its Stage5 migration or its own checkpoint."
        )
    return infos

  def learn(
    self,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = False,
  ) -> None:
    dynamic = getattr(self, "_stair_dynamic", False)
    if not self._stair_camp and not dynamic:
      return super().learn(num_learning_iterations, init_at_random_ep_len)
    target_updates = int(num_learning_iterations)
    completed = (
      self._stair_dynamic_loaded_completed_updates
      if dynamic
      else self._stair_camp_loaded_completed_updates
    )
    label = "StairDynamic" if dynamic else "StairCamp"
    if target_updates <= completed:
      raise ValueError(
        f"{label} total iteration target must exceed completed updates."
      )
    # RSL-RL stores the last zero-based update index. Use the completed count
    # as the next index so an N-update checkpoint resumes at update index N.
    self.current_learning_iteration = completed
    remaining = target_updates - completed
    super().learn(remaining, init_at_random_ep_len)

  def save(self, path: str, infos: dict | None = None) -> None:
    current_infos: dict[str, Any] = {
      **dict(infos or {}),
      "hybrid_training": {
        "git_sha": self._hybrid_training_git_sha,
        "action_scales": self._hybrid_action_scales,
      },
    }
    if self._stair_camp:
      current_infos[STAIR_CAMP_TRAINING_INFO_KEY] = (
        self._stair_camp_training_record()
      )
      curriculum = _camp_curriculum_from_env(self.env)
      if curriculum is None or not hasattr(curriculum, "state_dict"):
        raise ValueError("StairCamp runner cannot save without curriculum state.")
      curriculum_state = curriculum.state_dict()
      progress = validate_stair_camp_progress_payload(
        curriculum.progress_snapshot(), curriculum_state
      )
      current_infos[STAIR_CAMP_CURRICULUM_INFO_KEY] = curriculum_state
      current_infos[STAIR_CAMP_PROGRESS_INFO_KEY] = progress
    if getattr(self, "_stair_dynamic", False):
      current_infos[DYNAMIC_STAIR_TRAINING_INFO_KEY] = (
        self._stair_dynamic_training_record()
      )
      curriculum = _dynamic_curriculum_from_env(self.env)
      if curriculum is None or not hasattr(curriculum, "state_dict"):
        raise ValueError(
          "StairDynamic runner cannot save without curriculum state."
        )
      curriculum_state = curriculum.state_dict()
      progress = validate_dynamic_stair_progress_payload(
        curriculum.progress_snapshot(),
        curriculum_state,
      )
      current_infos[DYNAMIC_STAIR_CURRICULUM_INFO_KEY] = curriculum_state
      current_infos[DYNAMIC_STAIR_PROGRESS_INFO_KEY] = progress
    merged = merge_hybrid_checkpoint_infos(
      self._hybrid_loaded_infos,
      current_infos,
    )
    super().save(path, merged)


__all__ = [
  "HybridOnPolicyRunner",
  "hybrid_action_scales_from_env",
  "is_stair_camp_env",
  "is_stair_dynamic_env",
  "merge_hybrid_checkpoint_infos",
  "repository_git_sha",
  "zero_initialize_stair_camp_actor_output",
]
