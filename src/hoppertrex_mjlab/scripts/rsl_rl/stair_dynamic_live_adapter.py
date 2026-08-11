# ruff: noqa: TRY004
"""Live MjLab adapter for Hybrid-v3 StairDynamic evaluation.

The pure evaluator owns the signed protocol and output schema. This module
bridges it to the registered task while reusing StairCamp dependency loading,
checkpoint byte verification, formal gate implementations, gate normalization,
and pre-reset safety accounting. Importing this module imports neither Torch
nor MjLab; collect loads them only after validating the signed request.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import math

# Reusing these private seams is intentional: the four formal gates must keep
# one implementation rather than drift into a v3 copy.
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from types import MethodType
from typing import Any, Protocol

from hoppertrex_mjlab.scripts.rsl_rl import evaluate_stair_dynamic as evaluator
from hoppertrex_mjlab.scripts.rsl_rl import stair_camp_live_adapter as stair_camp

LIVE_ADAPTER_SCHEMA_VERSION = 1
ACTOR_OBSERVATION_WIDTH = 52
CRITIC_OBSERVATION_WIDTH = 56
ACTION_WIDTH = 6
EVALUATION_SOURCE = "mjlab_rsl_rl_stair_dynamic_live_adapter"

_ABORT_REASONS = {
  1: "non_wheel_contact",
  2: "actuator_limit",
  3: "orientation_limit",
  4: "backward_progress",
  5: "contact_timeout",
  6: "trail_contact_timeout",
  7: "cross_timeout",
  8: "target_saturation",
}


class RolloutBackend(Protocol):
  """Mockable seam around the heavy live implementation."""

  def metadata(self) -> Mapping[str, object]: ...

  def run_stair_suite(
    self,
    protocol: evaluator.StairEvaluationProtocol,
    descriptor: evaluator.AblationDescriptor,
  ) -> Sequence[Mapping[str, object]]: ...

  def run_gate(
    self, request: stair_camp.GateRequest
  ) -> Mapping[str, object]: ...


def _checkpoint_binding(checkpoint: Mapping[str, object]) -> Mapping[str, object]:
  """Return trained provenance or the honest zero-update runtime binding."""

  kind = checkpoint.get("kind")
  if kind == evaluator.CHECKPOINT_ENVELOPE_KIND:
    return stair_camp._mapping(checkpoint.get("training"), name="training")
  if kind == evaluator.MIGRATION_CHECKPOINT_ENVELOPE_KIND:
    binding = stair_camp._mapping(
      checkpoint.get("runtime_binding"), name="runtime_binding"
    )
    if binding.get("completed_updates") != 0 or "training" in checkpoint:
      raise ValueError("Migration envelope must remain an honest zero-update input.")
    return binding
  raise ValueError("Unsupported StairDynamic checkpoint envelope kind.")


def _validate_loaded_checkpoint_infos(
  checkpoint: Mapping[str, object], infos: Mapping[str, object]
) -> None:
  """Bind runner-returned provenance to either accepted envelope kind."""

  kind = checkpoint.get("kind")
  if kind == evaluator.CHECKPOINT_ENVELOPE_KIND:
    actual = evaluator.validate_training_info(
      stair_camp._mapping(
        infos.get("stair_dynamic_training"),
        name="infos.stair_dynamic_training",
      )
    )
    if actual != _checkpoint_binding(checkpoint):
      raise RuntimeError("Loaded training provenance differs from envelope.")
    return
  if kind == evaluator.MIGRATION_CHECKPOINT_ENVELOPE_KIND:
    if "stair_dynamic_training" in infos:
      raise RuntimeError("Zero-update migration fabricated training provenance.")
    actual = stair_camp._mapping(
      infos.get("stair_dynamic_migration"),
      name="infos.stair_dynamic_migration",
    )
    expected = stair_camp._mapping(
      checkpoint.get("migration"), name="checkpoint.migration"
    )
    if dict(actual) != dict(expected):
      raise RuntimeError("Loaded migration provenance differs from envelope.")
    evaluator.validate_zero_update_runtime_binding(
      _checkpoint_binding(checkpoint)
    )
    return
  raise RuntimeError("Loaded checkpoint envelope kind is unsupported.")


def _metadata(backend: RolloutBackend) -> dict[str, object]:
  metadata = dict(
    stair_camp._mapping(backend.metadata(), name="backend metadata")
  )
  required = {
    "actor_observation_width": ACTOR_OBSERVATION_WIDTH,
    "critic_observation_width": CRITIC_OBSERVATION_WIDTH,
    "action_width": ACTION_WIDTH,
    "stage5_actor_adapter_used": False,
  }
  for name, expected in required.items():
    if metadata.get(name) != expected:
      raise ValueError(
        f"StairDynamic backend metadata {name} must be {expected!r}."
      )
  stair_camp._json_safe(metadata, path="adapter_metadata")
  return metadata


def apply_policy_ablation(
  policy: Callable[[Any], Any], descriptor: Mapping[str, object]
) -> Callable[[Any], Any]:
  """Apply exactly the evaluator-registered PPO-head ablation."""

  name = descriptor.get("name")
  if not isinstance(name, str):
    raise ValueError("StairDynamic ablation has no name.")
  registered = evaluator.resolve_ablation(name)
  if dict(descriptor) != registered.to_dict():
    raise ValueError("StairDynamic ablation descriptor drifted.")
  indices = tuple(registered.zero_action_indices)
  if not indices:
    return policy

  def ablated_policy(observations: Any) -> Any:
    actions = policy(observations).clone()
    shape = getattr(actions, "shape", ())
    if len(shape) != 2 or int(shape[-1]) != ACTION_WIDTH:
      raise RuntimeError("StairDynamic policy did not emit [B, 6] actions.")
    for index in indices:
      actions[..., index] = 0.0
    return actions

  return ablated_policy


def _disable_feedforward(action_term: Any) -> None:
  """Keep the FSM live but zero its outputs for registered ablations."""

  original = getattr(action_term, "_update_dynamic_stair", None)
  if not callable(original):
    raise RuntimeError("StairDynamic action exposes no FSM update seam.")
  if getattr(action_term, "_stair_dynamic_feedforward_disabled", False):
    raise RuntimeError("Feedforward ablation was installed twice.")

  def update_without_feedforward(self: Any, *args: Any, **kwargs: Any) -> Any:
    result = original(*args, **kwargs)
    self._dynamic_leg_feedforward.zero_()
    self._dynamic_drive_feedforward.zero_()
    return result

  action_term._update_dynamic_stair = MethodType(
    update_without_feedforward, action_term
  )
  action_term._stair_dynamic_feedforward_disabled = True


def assert_policy_interface(observations: Any, *, action_width: int) -> None:
  """Fail closed unless the actual live interface is exactly 52/56/6."""

  actor = stair_camp._observation_group(observations, "actor")
  critic = stair_camp._observation_group(observations, "critic")
  if len(actor.shape) != 2 or int(actor.shape[-1]) != ACTOR_OBSERVATION_WIDTH:
    raise RuntimeError("Live StairDynamic actor width is not 52.")
  if len(critic.shape) != 2 or int(critic.shape[-1]) != CRITIC_OBSERVATION_WIDTH:
    raise RuntimeError("Live StairDynamic critic width is not 56.")
  if action_width != ACTION_WIDTH:
    raise RuntimeError("Live StairDynamic action width is not six.")


def _collect_validated(
  request: Mapping[str, object], backend: RolloutBackend
) -> dict[str, object]:
  suite = str(request["suite"])
  result: dict[str, object] = {
    "request_sha256": request["request_sha256"],
    "evaluation_source": EVALUATION_SOURCE,
  }
  if suite == evaluator.RETENTION_SUITE:
    gate_requests = stair_camp._gate_requests(
      {"domain": "flat", "gate_bindings": request["gate_bindings"]}
    )
    result["gates"] = [
      stair_camp._normalize_gate_outcome(
        stair_camp._mapping(
          backend.run_gate(gate_request),
          name=f"{gate_request.name} outcome",
        ),
        gate_request,
      )
      for gate_request in gate_requests
    ]
  else:
    protocol = evaluator.protocol_for(suite)
    ablation = stair_camp._mapping(request["ablation"], name="ablation")
    descriptor = evaluator.resolve_ablation(str(ablation["name"]))
    raw = stair_camp._sequence(
      backend.run_stair_suite(protocol, descriptor), name="stair trials"
    )
    result["trials"] = [
      dict(stair_camp._mapping(value, name=f"stair trials[{index}]"))
      for index, value in enumerate(raw)
    ]
  result["adapter_metadata"] = _metadata(backend)
  evaluator.finalize_collection(request, result)
  return result


def collect_with_backend(
  request: Mapping[str, object], backend: RolloutBackend
) -> dict[str, object]:
  """Run a signed request through a supplied mock or real backend."""

  normalized = evaluator.validate_evaluation_request(request)
  return _collect_validated(normalized, backend)


def _load_live_dependencies() -> Any:
  return stair_camp._load_live_dependencies()


def collect(request: Mapping[str, object]) -> Mapping[str, object]:
  """Execute one evaluator-signed live collection."""

  normalized = evaluator.validate_evaluation_request(request)
  dependencies = _load_live_dependencies()
  return _collect_validated(
    normalized, _DynamicMjLabBackend(normalized, dependencies)
  )


def _k3_protocol() -> evaluator.StairEvaluationProtocol:
  protocol = evaluator.K3_SCREEN_PROTOCOL
  return evaluator.StairEvaluationProtocol(
    suite="k3-screen",
    terrain=str(protocol.terrain),
    heights_m=(float(evaluator.PRIMARY_HEIGHT_M),),
    risers_per_trial=1,
    num_envs_per_height=int(protocol.num_envs_per_cell),
    repeats=int(protocol.repeats),
    stable_steps=int(protocol.stable_steps),
    minimum_successes=int(evaluator.K3_MIN_SUCCESSES),
    primary_height_m=float(evaluator.PRIMARY_HEIGHT_M),
    profile="screen",
    evidence_eligible=False,
  )


def _validate_k3_inputs(
  checkpoint_envelope: Mapping[str, object],
  budget_updates: int,
  device: str,
) -> tuple[dict[str, object], int, str]:
  checkpoint = evaluator.validate_checkpoint_envelope(
    checkpoint_envelope, verify_file=True
  )
  if (
    isinstance(budget_updates, bool)
    or not isinstance(budget_updates, int)
    or budget_updates not in evaluator.REGISTERED_BUDGETS
  ):
    raise ValueError("K=3 budget must be the registered 100 or 500 updates.")
  if not isinstance(device, str) or not device.strip():
    raise ValueError("K=3 device must be a non-empty string.")
  return checkpoint, int(budget_updates), device


def _k3_backend_request(
  checkpoint: Mapping[str, object], device: str
) -> dict[str, object]:
  descriptor = evaluator.resolve_ablation("full")
  protocol = _k3_protocol()
  return {
    "task": evaluator.DYNAMIC_STAIR_TASK_ID,
    "suite": "k3-screen",
    "protocol": protocol.to_dict(),
    "checkpoint": dict(checkpoint),
    "ablation": descriptor.to_dict(),
    "device": device,
    "evaluation_seed": evaluator.REGISTERED_EVALUATION_SEED,
  }


def _collect_k3_validated(
  checkpoint: Mapping[str, object],
  budget_updates: int,
  backend: RolloutBackend,
) -> dict[str, object]:
  smoke_bindings = stair_camp._evaluator().gate_bindings_for_profile("smoke")
  gate_requests = stair_camp._gate_requests(
    {
      "domain": "flat",
      "gate_bindings": {
        name: binding.to_dict() for name, binding in smoke_bindings.items()
      },
    }
  )
  gate_rows = [
    stair_camp._normalize_gate_outcome(
      stair_camp._mapping(
        backend.run_gate(request), name=f"{request.name} outcome"
      ),
      request,
    )
    for request in gate_requests
  ]
  normalized_gates, gate_passes = stair_camp._evaluator()._normalize_gate_results(
    gate_rows, profile="smoke"
  )
  false_positives = {
    str(row["name"]): int(row["stair_mode_false_positives"])
    for row in normalized_gates
  }

  protocol = _k3_protocol()
  descriptor = evaluator.resolve_ablation("full")
  backend_config = getattr(backend, "config", None)
  if isinstance(backend_config, dict):
    backend_config["domain"] = "stairs"
  raw_trials = stair_camp._sequence(
    backend.run_stair_suite(protocol, descriptor), name="K=3 stair trials"
  )
  _trials, rows = evaluator.normalize_trials(
    raw_trials, protocol=protocol, ablation=descriptor
  )
  if len(rows) != 1:
    raise RuntimeError("K=3 screen did not produce exactly one 1 cm row.")
  row = rows[0]
  height_row = {
    "height_m": row["height_m"],
    "trials": row["trials"],
    "successes": row["successes"],
    "terminations": row["terminations"],
    "non_wheel_contacts": row["non_wheel_contacts"],
    # v3 requests the FSM from approach; that activation is not a false
    # positive. Flat/kick false positives are accounted by the four gates.
    "stair_mode_false_positives": 0,
  }
  candidate = evaluator.make_k3_screen_candidate(
    checkpoint_envelope=checkpoint,
    budget_updates=budget_updates,
    gate_passes=gate_passes,
    gate_stair_mode_false_positives=false_positives,
    height_row=height_row,
  )
  if candidate["profile"] != "screen" or candidate["evidence_eligible"] is not False:
    raise RuntimeError("K=3 rejection-only evidence was mislabeled as formal.")
  return candidate


def collect_k3_with_backend(
  checkpoint_envelope: Mapping[str, object],
  budget_updates: int,
  device: str,
  backend: RolloutBackend,
) -> dict[str, object]:
  """Pure seam for one rejection-only K=3 screen candidate."""

  checkpoint, budget, _device = _validate_k3_inputs(
    checkpoint_envelope, budget_updates, device
  )
  return _collect_k3_validated(checkpoint, budget, backend)


def collect_k3(
  checkpoint_envelope: Mapping[str, object],
  budget_updates: int,
  device: str,
) -> dict[str, object]:
  """Run smoke retention plus 1 cm x 16 x 1 rejection-only screen."""

  checkpoint, budget, normalized_device = _validate_k3_inputs(
    checkpoint_envelope, budget_updates, device
  )
  dependencies = _load_live_dependencies()
  backend = _DynamicMjLabBackend(
    _k3_backend_request(checkpoint, normalized_device), dependencies
  )
  backend.config["domain"] = "flat"
  return _collect_k3_validated(checkpoint, budget, backend)


class _DynamicPreResetEvidenceMetric(stair_camp._LivePreResetEvidenceMetric):
  """Augment reused safety evidence with v3 FSM and action snapshots."""

  def __init__(self, cfg: Any, env: Any) -> None:
    super().__init__(cfg, env)
    torch = self._torch
    self.last_phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.last_loaded = torch.zeros(
      env.num_envs, 2, dtype=torch.bool, device=env.device
    )
    self.last_lead = torch.zeros_like(self.last_phase)
    self.last_traversal = torch.zeros_like(self.last_phase)
    self.last_abort_code = torch.zeros_like(self.last_phase)
    self.last_step_index = torch.zeros_like(self.last_phase)
    self.last_recover_stable = torch.zeros_like(self.last_phase)
    self.last_feedforward = torch.zeros(
      env.num_envs, 4, dtype=torch.float, device=env.device
    )
    self.last_residual = torch.zeros(
      env.num_envs, ACTION_WIDTH, dtype=torch.float, device=env.device
    )
    self.last_projected_gravity = torch.zeros(
      env.num_envs, 3, dtype=torch.float, device=env.device
    )
    self.last_pitch_rate = torch.zeros(
      env.num_envs, dtype=torch.float, device=env.device
    )

  def _action(self) -> Any:
    return self._env.action_manager.get_term("hybrid_wheel_leg")

  def _mode(self) -> Any:
    phase = getattr(self._action(), "dynamic_phase", None)
    if phase is None:
      raise RuntimeError("StairDynamic action exposes no dynamic phase.")
    return phase != 0

  def reset(self, env_ids: Any = None) -> None:
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    for value in (
      self.last_phase,
      self.last_loaded,
      self.last_lead,
      self.last_traversal,
      self.last_abort_code,
      self.last_step_index,
      self.last_recover_stable,
      self.last_feedforward,
      self.last_residual,
      self.last_projected_gravity,
      self.last_pitch_rate,
    ):
      value[ids] = 0

  def __call__(
    self,
    env: Any,
    non_wheel_contact_func: Callable[..., Any],
    sensor_name: str,
  ) -> Any:
    result = super().__call__(env, non_wheel_contact_func, sensor_name)
    action = self._action()
    self.last_phase.copy_(action.dynamic_phase)
    self.last_loaded.copy_(action.dynamic_loaded_contact)
    self.last_lead.copy_(action.dynamic_lead_side)
    self.last_traversal.copy_(action.dynamic_traversal_mode)
    self.last_abort_code.copy_(action._dynamic_abort_code)
    self.last_step_index.copy_(action.dynamic_step_index)
    self.last_recover_stable.copy_(action._dynamic_recover_stable)
    self.last_feedforward.copy_(action.dynamic_leg_feedforward)
    self.last_residual.copy_(action.applied_residual)
    robot = env.scene["robot"].data
    self.last_projected_gravity.copy_(robot.projected_gravity_b)
    self.last_pitch_rate.copy_(robot.root_link_ang_vel_b[:, 1])
    return result


def _platform_width_for_risers(
  risers: int,
  *,
  terrain_length_m: float,
  border_width_m: float,
  step_width_m: float,
) -> float:
  if isinstance(risers, bool) or risers < 1:
    raise ValueError("StairDynamic riser count must be positive.")
  platform = (
    float(terrain_length_m)
    - 2.0 * float(border_width_m)
    - 2.0 * int(risers) * float(step_width_m)
  )
  if not math.isfinite(platform) or platform <= 0.0:
    raise ValueError("Requested riser geometry does not fit the terrain tile.")
  # MjLab computes int((size - borders - platform) / (2 * step_width)).
  # Step decimal arithmetic can otherwise turn an exact N into N - epsilon.
  return math.nextafter(platform, -math.inf)


class _DynamicMjLabBackend(stair_camp._MjLabBackend):
  """v3 environment/policy adapter reusing StairCamp formal gate methods."""

  def __init__(self, request: Mapping[str, object], dependencies: Any) -> None:
    self.request = dict(request)
    self.config = {
      **self.request,
      "domain": (
        "flat"
        if self.request["suite"] == evaluator.RETENTION_SUITE
        else "stairs"
      ),
    }
    self.deps = dependencies
    self.dynamic_contract = importlib.import_module(
      "hoppertrex_mjlab.hybrid.stair_dynamic_contract"
    )
    self.device = str(self.request["device"])
    self.evaluation_seed = int(self.request["evaluation_seed"])
    self.descriptor = stair_camp._mapping(
      self.request["ablation"], name="ablation"
    )
    self._policy_owner: Any | None = None
    self._base_policy: Callable[[Any], Any] | None = None
    self._policy: Callable[[Any], Any] | None = None
    self._session_records: list[dict[str, object]] = []

  @property
  def checkpoint(self) -> Mapping[str, object]:
    return stair_camp._mapping(self.request["checkpoint"], name="checkpoint")

  @property
  def runtime_binding(self) -> Mapping[str, object]:
    return _checkpoint_binding(self.checkpoint)

  def metadata(self) -> Mapping[str, object]:
    return {
      "schema_version": LIVE_ADAPTER_SCHEMA_VERSION,
      "adapter": "stair_dynamic_live_adapter",
      "task": self.request["task"],
      "suite": self.request["suite"],
      "checkpoint_kind": self.checkpoint["kind"],
      "completed_updates": self.runtime_binding["completed_updates"],
      "actor_observation_width": ACTOR_OBSERVATION_WIDTH,
      "critic_observation_width": CRITIC_OBSERVATION_WIDTH,
      "action_width": ACTION_WIDTH,
      "stage5_actor_adapter_used": False,
      "ablation": self.descriptor["name"],
      "sessions": list(self._session_records),
    }

  def _registered_configs(self, *, play: bool) -> tuple[Any, Any]:
    registry = self.deps.registry_module
    task = str(self.request["task"])
    env_cfg = registry.load_env_cfg(task, play=play)
    contract_cfg = env_cfg if not play else registry.load_env_cfg(task, play=False)
    agent_cfg = registry.load_rl_cfg(task)
    if play:
      if getattr(env_cfg, "stair_dynamic_training_contract", None) is not False:
        raise RuntimeError("Registered StairDynamic play marker is not false.")
      if getattr(contract_cfg, "stair_dynamic_training_contract", None) is not True:
        raise RuntimeError("Canonical StairDynamic training marker is not true.")
    elif getattr(env_cfg, "stair_dynamic_training_contract", None) is not True:
      raise RuntimeError("Registered StairDynamic training marker is not true.")

    binding = self.runtime_binding
    agent_cfg.seed = int(binding["training_seed"])
    for candidate in (env_cfg, contract_cfg):
      if getattr(candidate, "stair_dynamic_task_id", None) != task:
        raise RuntimeError("Registry did not return the StairDynamic task.")
      if getattr(candidate, "stair_dynamic_maneuver_qualified", None) is not True:
        raise RuntimeError(
          "Qualified maneuver unavailable; set "
          "HOPPERTREX_DYNAMIC_STAIR_MANEUVER_PATH to the bound artifact."
        )
      self.dynamic_contract.validate_dynamic_stair_observation_layout(
        candidate.observations["actor"].terms,
        candidate.observations["critic"].terms,
      )
      action = candidate.actions["hybrid_wheel_leg"]
      maneuver = action.dynamic_stair_maneuver
      if maneuver.maneuver_hash != binding["maneuver_sha256"]:
        raise RuntimeError("Registered maneuver hash differs from checkpoint.")
      bindings = getattr(candidate, "stair_dynamic_maneuver_bindings", None)
      if not isinstance(bindings, Mapping):
        raise RuntimeError("Registered maneuver bindings are unavailable.")
      if (
        bindings.get("stage5_checkpoint_sha256")
        != binding["source_stage5_checkpoint_sha256"]
        or bindings.get("stage5_formal_gate_sha256")
        != binding["source_stage5_gate_sha256"]
      ):
        raise RuntimeError("Registered maneuver Stage5 provenance drifted.")

    actual_contract = self.dynamic_contract.dynamic_stair_contract_hash(
      contract_cfg, agent_cfg
    )
    if actual_contract != binding["contract_sha256"]:
      raise RuntimeError("Runtime config does not match the checkpoint contract.")
    contract_artifacts = self.dynamic_contract.dynamic_stair_artifact_bindings(
      contract_cfg
    )
    evaluation_artifacts = self.dynamic_contract.dynamic_stair_artifact_bindings(
      env_cfg
    )
    if (
      contract_artifacts != binding["artifact_bindings"]
      or evaluation_artifacts != contract_artifacts
    ):
      raise RuntimeError("Runtime artifacts do not match checkpoint bindings.")
    if self.deps.runner_module.repository_git_sha() != binding["git_sha"]:
      raise RuntimeError("Live Git SHA does not match checkpoint binding.")
    return env_cfg, agent_cfg


  def _verify_checkpoint_bytes(self) -> Any:
    path = super()._verify_checkpoint_bytes()
    if self.checkpoint["kind"] == evaluator.MIGRATION_CHECKPOINT_ENVELOPE_KIND:
      loaded = self.deps.torch.load(
        path, map_location="cpu", weights_only=False
      )
      checkpoint = stair_camp._mapping(loaded, name="migration checkpoint")
      evaluator._validate_loaded_zero_update_network(
        checkpoint,
        stair_camp._mapping(
          self.checkpoint["migration"], name="checkpoint.migration"
        ),
      )
    return path

  def _load_policy(self) -> Callable[[Any], Any]:
    if self._policy is not None:
      return self._policy
    torch = self.deps.torch
    self.deps.torch_utils_module.configure_torch_backends()
    torch.manual_seed(self.evaluation_seed)
    env_cfg, agent_cfg = self._registered_configs(play=True)
    env_cfg.scene.num_envs = 1
    if env_cfg.scene.terrain is not None:
      env_cfg.scene.terrain.num_envs = 1
    env = self.deps.env_module.ManagerBasedRlEnv(cfg=env_cfg, device=self.device)
    wrapped = self.deps.rl_module.RslRlVecEnvWrapper(
      env, clip_actions=agent_cfg.clip_actions
    )
    try:
      observations = wrapped.get_observations()
      assert_policy_interface(
        observations,
        action_width=int(wrapped.unwrapped.action_manager.total_action_dim),
      )
      runner_cls = self.deps.registry_module.load_runner_cls(
        str(self.request["task"])
      )
      if runner_cls is None:
        raise RuntimeError("StairDynamic has no registered runner class.")
      runner = runner_cls(wrapped, asdict(agent_cfg), device=self.device)
      infos = stair_camp._mapping(
        runner.load(
          str(self._verify_checkpoint_bytes()),
          load_cfg={"actor": True},
          strict=True,
          map_location=self.device,
        ),
        name="checkpoint infos",
      )
      _validate_loaded_checkpoint_infos(self.checkpoint, infos)
      base_policy = runner.get_inference_policy(device=self.device)
      actions = base_policy(observations)
      if len(actions.shape) != 2 or int(actions.shape[-1]) != ACTION_WIDTH:
        raise RuntimeError("Loaded StairDynamic actor does not emit six actions.")
      self._policy_owner = runner
      self._base_policy = base_policy
      self._policy = apply_policy_ablation(base_policy, self.descriptor)
    finally:
      wrapped.close()
    return self._policy


  def _stair_terrain_cfg(
    self, *, height_m: float, risers: int, num_envs: int
  ) -> Any:
    task = self.deps.task_module
    platform_width = _platform_width_for_risers(
      risers,
      terrain_length_m=task.DYNAMIC_STAIR_TERRAIN_SIZE_M[0],
      border_width_m=task.DYNAMIC_STAIR_TERRAIN_BORDER_WIDTH_M,
      step_width_m=task.DYNAMIC_STAIR_STEP_WIDTH_M,
    )
    stair = self.deps.terrain_config_module.pyramid_stairs(
      proportion=1.0,
      step_height_range=(height_m, height_m),
      step_width=task.DYNAMIC_STAIR_STEP_WIDTH_M,
      platform_width=platform_width,
      border_width=task.DYNAMIC_STAIR_TERRAIN_BORDER_WIDTH_M,
    )
    return self.deps.terrain_module.TerrainEntityCfg(
      terrain_type="generator",
      terrain_generator=self.deps.terrain_module.TerrainGeneratorCfg(
        seed=self.evaluation_seed,
        curriculum=True,
        size=task.DYNAMIC_STAIR_TERRAIN_SIZE_M,
        num_rows=1,
        num_cols=1,
        difficulty_range=(0.0, 0.0),
        sub_terrains={"stair": stair},
      ),
      max_init_terrain_level=0,
      num_envs=num_envs,
    )

  def _evaluation_env_cfg(
    self,
    *,
    domain: str,
    cells: tuple[float, ...],
    num_envs: int,
    pushes: bool,
    risers_per_trial: int | None = None,
  ) -> tuple[Any, Any]:
    if domain not in ("flat", "stairs"):
      raise RuntimeError("StairDynamic supports only flat or stairs.")
    if pushes and domain != "flat":
      raise RuntimeError("Pushes are restricted to Stage5 retention.")
    if domain == "stairs" and (
      len(cells) != 1 or risers_per_trial not in (1, 3)
    ):
      raise RuntimeError("StairDynamic live stair geometry is not registered.")

    env_cfg, agent_cfg = self._registered_configs(play=True)
    if getattr(env_cfg, "stair_dynamic_training_contract", None) is not False:
      raise RuntimeError("Evaluation env inherited a training marker.")
    events = getattr(env_cfg, "events", None)
    if not isinstance(events, dict) or "push_robot" in events:
      raise RuntimeError("Registered play event surface drifted.")
    if pushes:
      camp_task = stair_camp._evaluator().STAIR_CAMP_TASK_ID
      camp_training = self.deps.registry_module.load_env_cfg(
        camp_task, play=False
      )
      events["push_robot"] = stair_camp._validated_stage5_push_event(
        camp_training, self.deps.task_module
      )

    env_cfg.seed = self.evaluation_seed
    env_cfg.scene.num_envs = num_envs
    if domain == "flat":
      env_cfg.scene.terrain = super()._terrain_cfg(
        domain="flat", cells=(0.0,), num_envs=num_envs
      )
    else:
      env_cfg.scene.terrain = self._stair_terrain_cfg(
        height_m=float(cells[0]),
        risers=int(risers_per_trial),
        num_envs=num_envs,
      )

    flat_env_count = num_envs if domain == "flat" else 0
    for name in ("twist", "posture", "stair_request"):
      command = env_cfg.commands[name]
      if not hasattr(command, "flat_env_count"):
        raise RuntimeError(f"Command {name!r} has no flat/stair split.")
      command.flat_env_count = flat_env_count
    reset = events.get("reset_root_to_stair_dynamic")
    if reset is None:
      raise RuntimeError("StairDynamic evaluation reset is missing.")
    reset.params["flat_env_count"] = flat_env_count

    env_cfg.curriculum = {}
    env_cfg.metrics = {
      stair_camp._LIVE_EVIDENCE_TERM_NAME: (
        self.deps.manager_module.MetricsTermCfg(
          func=_DynamicPreResetEvidenceMetric,
          params={
            "non_wheel_contact_func": (
              self.deps.balance_task_module.non_wheel_ground_contact
            ),
            "sensor_name": (
              self.deps.balance_task_module.NON_WHEEL_GROUND_SENSOR_NAME
            ),
          },
          reduce="last",
        )
      )
    }
    if domain == "flat":
      env_cfg.episode_length_s = stair_camp.EPISODE_LENGTH_S
    height_term = env_cfg.observations["critic"].terms["step_height"]
    height_term.func = stair_camp._live_step_height_observation
    height_term.params = {
      "cell_values": (0.0,) if domain == "flat" else cells
    }
    return env_cfg, agent_cfg

  @contextlib.contextmanager
  def _session(
    self,
    *,
    domain: str,
    cells: tuple[float, ...],
    num_envs: int,
    pushes: bool,
    purpose: str,
    risers_per_trial: int | None = None,
  ) -> Any:
    self._policy_for_rollout()
    env_cfg, agent_cfg = self._evaluation_env_cfg(
      domain=domain,
      cells=cells,
      num_envs=num_envs,
      pushes=pushes,
      risers_per_trial=risers_per_trial,
    )
    env = self.deps.env_module.ManagerBasedRlEnv(cfg=env_cfg, device=self.device)
    if getattr(env.cfg, "stair_dynamic_training_contract", None) is not False:
      raise RuntimeError("Constructed evaluation env has a training marker.")
    wrapped = self.deps.rl_module.RslRlVecEnvWrapper(
      env, clip_actions=agent_cfg.clip_actions
    )
    stair_camp._live_evidence(wrapped.unwrapped)
    tracker = stair_camp._SafetyTrackingWrapper(wrapped, self.deps)
    try:
      observations = wrapped.get_observations()
      assert_policy_interface(
        observations,
        action_width=int(wrapped.unwrapped.action_manager.total_action_dim),
      )
      terrain = wrapped.unwrapped.scene.terrain
      if terrain is None or terrain.terrain_types is None:
        raise RuntimeError("Generated terrain is unavailable.")
      counts = self.deps.torch.bincount(terrain.terrain_types, minlength=1)
      if int(counts.sum().item()) != num_envs or len(counts) != 1:
        raise RuntimeError("Generated terrain assignment drifted.")
      action = wrapped.unwrapped.action_manager.get_term("hybrid_wheel_leg")
      if bool(self.descriptor.get("disable_feedforward")):
        _disable_feedforward(action)
      self._session_records.append(
        {
          "purpose": purpose,
          "domain": domain,
          "terrain": (
            "flat_generated_zero_height"
            if domain == "flat"
            else str(self.request["protocol"]["terrain"])
          ),
          "num_envs": num_envs,
          "cells": list(cells),
          "risers_per_trial": risers_per_trial,
          "pushes_enabled": pushes,
          "training_contract": False,
          "pre_reset_evidence_term": stair_camp._LIVE_EVIDENCE_TERM_NAME,
        }
      )
      yield stair_camp._LiveSession(
        wrapped=wrapped, tracker=tracker, env_cfg=env_cfg
      )
    finally:
      wrapped.close()

  @staticmethod
  def _force_stair_request(env: Any, enabled: Any) -> None:
    term = env.command_manager.get_term("stair_request")
    command = getattr(term, "_command", None)
    if command is None or tuple(command.shape) != (env.num_envs, 1):
      raise RuntimeError("Stair request does not expose [B, 1] state.")
    command[:, 0] = enabled.to(device=command.device, dtype=command.dtype)

  def _run_stair_repeat(
    self,
    *,
    session: Any,
    protocol: evaluator.StairEvaluationProtocol,
    descriptor: evaluator.AblationDescriptor,
    height_m: float,
    repeat_index: int,
  ) -> list[dict[str, object]]:
    torch = self.deps.torch
    wrapped = session.wrapped
    env = wrapped.unwrapped
    wrapped.reset()
    action = env.action_manager.get_term("hybrid_wheel_leg")
    dt = float(action.cfg.dynamic_stair_control_dt)
    if not math.isfinite(dt) or dt <= 0.0:
      raise RuntimeError("StairDynamic control dt is invalid.")
    max_steps = math.ceil(float(session.env_cfg.episode_length_s) / dt)
    if max_steps < protocol.stable_steps:
      raise RuntimeError("Episode is shorter than the stability gate.")
    num_envs = int(env.num_envs)
    posture = self._posture_center(session.env_cfg)
    maneuver = action.cfg.dynamic_stair_maneuver
    thresholds = tuple(
      float(maneuver.first_cross_m)
      + index * float(maneuver.next_cross_m)
      for index in range(protocol.risers_per_trial)
    )
    start_x = env.scene["robot"].data.root_link_pos_w[:, 0].clone()

    active = torch.ones(num_envs, dtype=torch.bool, device=env.device)
    success = torch.zeros_like(active)
    terminated_ever = torch.zeros_like(active)
    contact_ever = torch.zeros_like(active)
    timeout_ever = torch.zeros_like(active)
    horizon_ever = torch.zeros_like(active)
    dynamic_seen = torch.zeros_like(active)
    lead_code = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    abort_code = torch.zeros_like(lead_code)
    max_step_index = torch.zeros_like(lead_code)
    stable_steps = torch.zeros_like(lead_code)
    phase_counts = torch.zeros(
      num_envs,
      len(evaluator.PHASE_NAMES),
      dtype=torch.long,
      device=env.device,
    )
    previous_phase = action.dynamic_phase.clone()
    previous_loaded = action.dynamic_loaded_contact.clone()
    previous_step = action.dynamic_step_index.clone()
    left_trigger = torch.full(
      (num_envs,), math.nan, dtype=torch.float, device=env.device
    )
    right_trigger = torch.full_like(left_trigger, math.nan)
    recover_start = torch.full_like(left_trigger, math.nan)
    recovery_times: list[list[float]] = [[] for _ in range(num_envs)]
    wheel_sumsq = torch.zeros_like(left_trigger)
    leg_sumsq = torch.zeros_like(left_trigger)
    wheel_samples = torch.zeros_like(lead_code)
    leg_samples = torch.zeros_like(lead_code)
    wheel_max = torch.zeros_like(left_trigger)
    leg_max = torch.zeros_like(left_trigger)
    feedforward_max = torch.zeros_like(left_trigger)
    peak_pitch = torch.zeros_like(left_trigger)
    peak_roll = torch.zeros_like(left_trigger)

    physical_cross_time = torch.full(
      (num_envs, protocol.risers_per_trial),
      math.nan,
      dtype=torch.float,
      device=env.device,
    )
    physical_recovered = torch.zeros(
      num_envs,
      protocol.risers_per_trial,
      dtype=torch.bool,
      device=env.device,
    )
    physical_stable = torch.zeros(
      num_envs,
      protocol.risers_per_trial,
      dtype=torch.long,
      device=env.device,
    )
    physical_times: list[list[float]] = [[] for _ in range(num_envs)]
    force_request_false = bool(descriptor.force_stair_request_false)

    for step in range(max_steps):
      was_active = active.clone()
      self._force_commands(
        wrapped, vx=float(maneuver.approach_vx), yaw=0.0, posture=posture
      )
      requested = (
        torch.zeros_like(was_active) if force_request_false else was_active
      )
      self._force_stair_request(env, requested)
      actions = self._policy_actions(wrapped)
      evidence = stair_camp._live_evidence(env)
      before_sequence = evidence.sequence
      wrapped.step(actions)
      evidence = stair_camp._live_evidence(env)
      if evidence.sequence != before_sequence + 1:
        raise RuntimeError("Pre-reset evidence did not run exactly once.")
      if not isinstance(evidence, _DynamicPreResetEvidenceMetric):
        raise RuntimeError("StairDynamic evidence metric type drifted.")

      phase = evidence.last_phase
      loaded = evidence.last_loaded
      traversal = evidence.last_traversal
      current_step = evidence.last_step_index
      for phase_code in range(len(evaluator.PHASE_NAMES)):
        phase_counts[:, phase_code] += (
          was_active & (phase == phase_code)
        ).long()

      now_s = (step + 1) * dt
      left_edge = (
        was_active
        & loaded[:, 0]
        & ~previous_loaded[:, 0]
        & torch.isnan(left_trigger)
      )
      right_edge = (
        was_active
        & loaded[:, 1]
        & ~previous_loaded[:, 1]
        & torch.isnan(right_trigger)
      )
      left_trigger[left_edge] = now_s
      right_trigger[right_edge] = now_s
      first_lead = was_active & (lead_code == 0) & (evidence.last_lead != 0)
      lead_code[first_lead] = evidence.last_lead[first_lead]
      dynamic_seen.logical_or_(was_active & (traversal == 2))
      has_abort_code = was_active & (evidence.last_abort_code > 0)
      abort_code[has_abort_code] = evidence.last_abort_code[has_abort_code]
      max_step_index.copy_(torch.maximum(max_step_index, current_step))

      residual = evidence.last_residual
      wheel = residual[:, :2]
      leg = residual[:, 2:]
      wheel_sumsq += was_active.float() * torch.square(wheel).sum(dim=1)
      leg_sumsq += was_active.float() * torch.square(leg).sum(dim=1)
      wheel_samples += was_active.long() * 2
      leg_samples += was_active.long() * 4
      wheel_max.copy_(
        torch.maximum(
          wheel_max,
          torch.where(was_active, wheel.abs().max(dim=1).values, 0.0),
        )
      )
      leg_max.copy_(
        torch.maximum(
          leg_max,
          torch.where(was_active, leg.abs().max(dim=1).values, 0.0),
        )
      )
      feedforward_max.copy_(
        torch.maximum(
          feedforward_max,
          torch.where(
            was_active,
            evidence.last_feedforward.abs().max(dim=1).values,
            0.0,
          ),
        )
      )
      gravity = evidence.last_projected_gravity
      pitch = torch.atan2(
        gravity[:, 0], torch.clamp(-gravity[:, 2], min=1.0e-6)
      ).abs()
      roll = torch.atan2(
        -gravity[:, 1], torch.clamp(-gravity[:, 2], min=1.0e-6)
      ).abs()
      peak_pitch.copy_(
        torch.maximum(peak_pitch, torch.where(was_active, pitch, 0.0))
      )
      peak_roll.copy_(
        torch.maximum(peak_roll, torch.where(was_active, roll, 0.0))
      )

      entered_recover = was_active & (phase == 6) & (previous_phase != 6)
      recover_start[entered_recover] = step * dt
      advanced = was_active & (current_step > previous_step)
      for index in torch.nonzero(advanced, as_tuple=False).squeeze(-1).tolist():
        if torch.isnan(recover_start[index]):
          raise RuntimeError("Riser completed without RECOVER timing.")
        recovery_times[index].append(
          float(now_s - float(recover_start[index].item()))
        )
        recover_start[index] = math.nan

      progress = evidence.last_root_x - start_x
      posture_stable = (
        (pitch <= 0.10)
        & (roll <= 0.10)
        & (evidence.last_pitch_rate.abs() <= 0.5)
      )
      for riser_index, threshold in enumerate(thresholds):
        crossed = was_active & (progress >= threshold)
        first_cross = crossed & torch.isnan(
          physical_cross_time[:, riser_index]
        )
        physical_cross_time[first_cross, riser_index] = now_s
        pending = crossed & ~physical_recovered[:, riser_index]
        physical_stable[:, riser_index] = torch.where(
          pending & posture_stable,
          physical_stable[:, riser_index] + 1,
          torch.where(
            pending,
            torch.zeros_like(physical_stable[:, riser_index]),
            physical_stable[:, riser_index],
          ),
        )
        recovered = pending & (
          physical_stable[:, riser_index] >= protocol.stable_steps
        )
        physical_recovered[:, riser_index].logical_or_(recovered)
        for index in (
          torch.nonzero(recovered, as_tuple=False).squeeze(-1).tolist()
        ):
          physical_times[index].append(
            float(
              now_s
              - float(physical_cross_time[index, riser_index].item())
            )
          )

      terminated = evidence.last_terminated
      non_wheel = evidence.last_contact
      timed_out = evidence.last_reset & ~terminated
      terminated_ever.logical_or_(was_active & terminated)
      contact_ever.logical_or_(was_active & non_wheel)
      timeout_ever.logical_or_(was_active & timed_out)
      abort_now = was_active & (
        (traversal == 3) | terminated | non_wheel | timed_out
      )
      if force_request_false:
        completed = physical_recovered.all(dim=1)
        success_now = was_active & completed & ~abort_now
        current_stable = physical_stable[:, -1]
      else:
        completed = current_step >= protocol.risers_per_trial
        success_now = was_active & completed & ~abort_now
        current_stable = evidence.last_recover_stable
      success.logical_or_(success_now)
      stable_steps.copy_(torch.maximum(stable_steps, current_stable))
      active &= ~(success_now | abort_now)
      previous_phase.copy_(phase)
      previous_loaded.copy_(loaded)
      previous_step.copy_(current_step)
      if not bool(active.any()):
        break

    if bool(active.any()):
      horizon_ever.logical_or_(active)
      active.zero_()

    trials: list[dict[str, object]] = []
    for index in range(num_envs):
      is_success = bool(success[index].item())
      mode = (
        "DYNAMIC"
        if is_success and bool(dynamic_seen[index].item())
        else "ROLL"
        if is_success
        else "ABORT"
      )
      if force_request_false:
        completed_steps = len(physical_times[index])
        recoveries = physical_times[index]
      else:
        completed_steps = int(max_step_index[index].item())
        recoveries = recovery_times[index]
      completed_steps = min(completed_steps, protocol.risers_per_trial)
      recoveries = list(recoveries[:completed_steps])
      if len(recoveries) != completed_steps:
        raise RuntimeError("Recovery timing is incomplete.")

      if mode == "ROLL":
        lead_side = "NONE"
        left_time: float | None = None
        right_time: float | None = None
      else:
        lead_value = int(lead_code[index].item())
        lead_side = {0: "NONE", 1: "LEFT", 2: "RIGHT"}.get(lead_value)
        if lead_side is None:
          raise RuntimeError("Live lead-side code is invalid.")
        left_value = float(left_trigger[index].item())
        right_value = float(right_trigger[index].item())
        left_time = None if math.isnan(left_value) else left_value
        right_time = None if math.isnan(right_value) else right_value

      reason: str | None = None
      if mode == "ABORT":
        code = int(abort_code[index].item())
        if code in _ABORT_REASONS:
          reason = _ABORT_REASONS[code]
        elif bool(contact_ever[index].item()):
          reason = "non_wheel_contact"
        elif bool(terminated_ever[index].item()):
          reason = "environment_termination"
        elif bool(timeout_ever[index].item()):
          reason = "episode_timeout"
        elif bool(horizon_ever[index].item()):
          reason = "rollout_horizon_exhausted"
        else:
          reason = "dynamic_stair_abort"

      wheel_count = max(int(wheel_samples[index].item()), 1)
      leg_count = max(int(leg_samples[index].item()), 1)
      stable_value = int(stable_steps[index].item())
      if is_success:
        stable_value = max(stable_value, protocol.stable_steps)
      trials.append(
        {
          "height_m": float(height_m),
          "env_index": index,
          "repeat_index": repeat_index,
          "success": is_success,
          "traversal_mode": mode,
          "lift_mode": maneuver.lift_mode.value,
          "lead_side": lead_side,
          "left_trigger_time_s": left_time,
          "right_trigger_time_s": right_time,
          "phase_durations_s": {
            phase_name: float(phase_counts[index, code].item() * dt)
            for code, phase_name in enumerate(evaluator.PHASE_NAMES)
          },
          "wheel_ppo_rms": math.sqrt(
            float(wheel_sumsq[index].item()) / wheel_count
          ),
          "wheel_ppo_max_abs": float(wheel_max[index].item()),
          "leg_ppo_rms": math.sqrt(
            float(leg_sumsq[index].item()) / leg_count
          ),
          "leg_ppo_max_abs": float(leg_max[index].item()),
          "feedforward_max_abs_rad": float(feedforward_max[index].item()),
          "peak_abs_pitch_rad": float(peak_pitch[index].item()),
          "peak_abs_roll_rad": float(peak_roll[index].item()),
          "steps_completed": completed_steps,
          "step_recovery_times_s": recoveries,
          "stable_steps": stable_value,
          "terminated": bool(terminated_ever[index].item()),
          "non_wheel_contact": bool(contact_ever[index].item()),
          "abort_reason": reason,
        }
      )
    return trials

  def run_stair_suite(
    self,
    protocol: evaluator.StairEvaluationProtocol,
    descriptor: evaluator.AblationDescriptor,
  ) -> Sequence[Mapping[str, object]]:
    if self.config["domain"] != "stairs":
      raise RuntimeError("Stair suite requested from retention backend.")
    if descriptor.to_dict() != dict(self.descriptor):
      raise RuntimeError("Backend ablation differs from signed request.")
    trials: list[dict[str, object]] = []
    for height_m in protocol.heights_m:
      with self._session(
        domain="stairs",
        cells=(float(height_m),),
        num_envs=protocol.num_envs_per_height,
        pushes=False,
        purpose=f"{protocol.suite}_{height_m:.3f}m",
        risers_per_trial=protocol.risers_per_trial,
      ) as session:
        for repeat_index in range(protocol.repeats):
          trials.extend(
            self._run_stair_repeat(
              session=session,
              protocol=protocol,
              descriptor=descriptor,
              height_m=float(height_m),
              repeat_index=repeat_index,
            )
          )
    return trials



def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest="command", required=True)
  live = commands.add_parser("collect", help="Run one signed formal request.")
  live.add_argument("--request", type=Path, required=True)
  live.add_argument("--output", type=Path, required=True)
  k3 = commands.add_parser("collect-k3", help="Run one rejection-only K=3 screen.")
  k3.add_argument("--checkpoint-envelope", type=Path, required=True)
  k3.add_argument("--budget-updates", type=int, choices=(100, 500), required=True)
  k3.add_argument("--device", default="cuda:0")
  k3.add_argument("--output", type=Path, required=True)
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  if args.command == "collect":
    result = collect(stair_camp._read_json_mapping(args.request))
  elif args.command == "collect-k3":
    result = collect_k3(
      stair_camp._read_json_mapping(args.checkpoint_envelope),
      args.budget_updates,
      args.device,
    )
  else:  # pragma: no cover
    raise AssertionError(f"Unhandled command: {args.command}")
  stair_camp._write_output(result, args.output)
  return 0


__all__ = [
  "ACTION_WIDTH",
  "ACTOR_OBSERVATION_WIDTH",
  "CRITIC_OBSERVATION_WIDTH",
  "EVALUATION_SOURCE",
  "LIVE_ADAPTER_SCHEMA_VERSION",
  "RolloutBackend",
  "apply_policy_ablation",
  "assert_policy_interface",
  "collect",
  "collect_k3",
  "collect_k3_with_backend",
  "collect_with_backend",
  "main",
  "parse_args",
]


if __name__ == "__main__":
  raise SystemExit(main())
