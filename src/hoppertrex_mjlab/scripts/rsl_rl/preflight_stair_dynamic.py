#!/usr/bin/env python3
# ruff: noqa: TRY004
"""Fast static registry preflight for Hybrid-v3 StairDynamic.

This allocates no MjLab environment and runs no rollout.  It resolves the
qualified maneuver and classical artifacts from the registered task, computes
the canonical runtime contract, and emits the exact evaluator expectation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from hoppertrex_mjlab.hybrid.stair_dynamic import (
  DYNAMIC_STAIR_TASK_ID,
  validate_dynamic_maneuver_bindings,
)
from hoppertrex_mjlab.hybrid.stair_dynamic_contract import (
  DYNAMIC_STAIR_ACTION_SCALES,
  DYNAMIC_STAIR_ACTOR_WIDTH,
  DYNAMIC_STAIR_CRITIC_WIDTH,
  dynamic_stair_artifact_bindings,
  dynamic_stair_contract_hash,
)
from hoppertrex_mjlab.scripts.rsl_rl.evaluate_stair_dynamic import (
  checkpoint_expectation_from_mapping,
)
from hoppertrex_mjlab.scripts.rsl_rl.search_stair_dynamic import (
  validate_trigger_qualification,
)

PREFLIGHT_SCHEMA_VERSION = 1


def _expectation_payload(
  *,
  git_sha: str,
  contract_sha256: str,
  artifact_bindings: Mapping[str, str],
  maneuver_sha256: str,
  maneuver_bindings: Mapping[str, str],
  completed_updates: int | None,
) -> dict[str, object]:
  if artifact_bindings.get("dynamic_maneuver_hash") != maneuver_sha256:
    raise ValueError("Runtime artifact and maneuver hashes disagree.")
  if maneuver_bindings.get("git_sha") != git_sha:
    raise ValueError("Qualified maneuver Git SHA differs from the checkout.")
  payload = {
    "git_sha": git_sha,
    "contract_sha256": contract_sha256,
    "artifact_bindings": dict(artifact_bindings),
    "maneuver_sha256": maneuver_sha256,
    "source_stage5_checkpoint_sha256": maneuver_bindings.get(
      "stage5_checkpoint_sha256"
    ),
    "source_stage5_gate_sha256": maneuver_bindings.get(
      "stage5_formal_gate_sha256"
    ),
    "completed_updates": completed_updates,
  }
  expectation = checkpoint_expectation_from_mapping(payload)
  return {
    "git_sha": expectation.git_sha,
    "contract_sha256": expectation.contract_sha256,
    "artifact_bindings": dict(expectation.artifact_bindings or {}),
    "maneuver_sha256": expectation.maneuver_sha256,
    "source_stage5_checkpoint_sha256": (
      expectation.source_stage5_checkpoint_sha256
    ),
    "source_stage5_gate_sha256": expectation.source_stage5_gate_sha256,
    "completed_updates": expectation.completed_updates,
  }



def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _registered_classical_bindings() -> tuple[str, dict[str, str]]:
  from mjlab.tasks.registry import load_env_cfg

  import hoppertrex_mjlab.tasks  # noqa: F401
  from hoppertrex_mjlab.hybrid.runner import repository_git_sha

  env_cfg = load_env_cfg("HopperTrex-Hybrid-v2-Stage5", play=False)
  action = env_cfg.actions.get("hybrid_wheel_leg")
  names = (
    "controller_gain_hash",
    "calibration_hash",
    "yaw_calibration_hash",
    "posture_map_hash",
    "posture_artifact_hash",
    "station_calibration_hash",
  )
  values = {name: getattr(action, name, None) for name in names}
  if any(not isinstance(value, str) or not value for value in values.values()):
    raise ValueError("Registered Stage5 classical artifact bindings are incomplete.")
  return repository_git_sha(), {name: str(value) for name, value in values.items()}


def _validate_stage5_search_source(
  checkpoint_path: Path,
  gate_path: Path,
  *,
  checkpoint_sha256: str,
) -> None:
  """Reuse the migration validator before the Stage5 policy enters CEM."""

  import torch

  from hoppertrex_mjlab.scripts.rsl_rl.migrate_stage5_to_stair_dynamic import (
    validate_stage5_source,
  )

  checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  try:
    gate = json.loads(gate_path.read_text(encoding="utf-8-sig"))
  except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise ValueError("Stage5 formal gate is not valid JSON.") from exc
  if not isinstance(checkpoint, Mapping) or not isinstance(gate, Mapping):
    raise ValueError("Stage5 checkpoint/gate roots must be mappings.")
  validate_stage5_source(
    checkpoint, gate, source_checkpoint_sha256=checkpoint_sha256
  )


def collect_search_bindings(
  *,
  stage5_checkpoint: Path,
  stage5_gate: Path,
  trigger_qualification: Path,
) -> dict[str, str]:
  """Build the exact CEM binding object without allocating an environment."""

  for name, path in (
    ("Stage5 checkpoint", stage5_checkpoint),
    ("Stage5 formal gate", stage5_gate),
    ("per-wheel trigger qualification", trigger_qualification),
  ):
    if not path.is_file():
      raise FileNotFoundError(f"{name} does not exist: {path}.")
  qualification_bytes = trigger_qualification.read_bytes()
  qualification_sha = hashlib.sha256(qualification_bytes).hexdigest()
  try:
    payload = json.loads(qualification_bytes.decode("utf-8-sig"))
  except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise ValueError("Per-wheel trigger qualification is not valid JSON.") from exc
  if not isinstance(payload, Mapping):
    raise TypeError("Per-wheel trigger qualification must be a JSON object.")
  from hoppertrex_mjlab.scripts.rsl_rl.qualify_stair_dynamic_trigger import (
    verify_document,
  )

  evidence = verify_document(payload)
  normalized_evidence = dict(evidence)
  normalized_evidence["evidence_sha256"] = qualification_sha
  validate_trigger_qualification(normalized_evidence)
  checkpoint_sha256 = _file_sha256(stage5_checkpoint)
  _validate_stage5_search_source(
    stage5_checkpoint, stage5_gate, checkpoint_sha256=checkpoint_sha256
  )
  git_sha, classical = _registered_classical_bindings()
  bindings = {
    "git_sha": git_sha,
    "stage5_checkpoint_sha256": checkpoint_sha256,
    "stage5_formal_gate_sha256": _file_sha256(stage5_gate),
    "per_wheel_trigger_qualification_sha256": qualification_sha,
    **classical,
  }
  return validate_dynamic_maneuver_bindings(bindings)


def _registered_per_wheel_sensor_names(env_cfg: object) -> tuple[str, str]:
  """Validate and return the two registered sensor cfg names, without an env."""

  from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
    DYNAMIC_STAIR_LEFT_SENSOR_NAME,
    DYNAMIC_STAIR_RIGHT_SENSOR_NAME,
    validate_stair_dynamic_observation_contract,
  )

  # Reuse the task's canonical validator: it checks each cfg name, primary geom
  # pattern/entity, fields, and reduction mode.  ``scene.sensors`` is an
  # iterable tuple of ContactSensorCfg, not a name->cfg mapping.
  validate_stair_dynamic_observation_contract(env_cfg)
  sensors = getattr(getattr(env_cfg, "scene", None), "sensors", ())
  names = tuple(getattr(sensor, "name", None) for sensor in sensors)
  required = (DYNAMIC_STAIR_LEFT_SENSOR_NAME, DYNAMIC_STAIR_RIGHT_SENSOR_NAME)
  if any(name not in names for name in required):  # pragma: no cover - validator owns detail
    raise ValueError("Registered StairDynamic per-wheel sensors are missing.")
  return required


def collect_runtime_preflight(
  *,
  completed_updates: int | None,
) -> dict[str, object]:
  """Load task configs only; do not instantiate ManagerBasedRlEnv."""

  if completed_updates is not None and (
    isinstance(completed_updates, bool) or completed_updates < 0
  ):
    raise ValueError("completed_updates must be null or a non-negative integer.")
  # Import registration and MjLab only after the pure CLI contract is accepted.
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  import hoppertrex_mjlab.tasks  # noqa: F401
  from hoppertrex_mjlab.hybrid.runner import repository_git_sha

  env_cfg = load_env_cfg(DYNAMIC_STAIR_TASK_ID, play=False)
  agent_cfg = load_rl_cfg(DYNAMIC_STAIR_TASK_ID)
  if (
    getattr(env_cfg, "stair_dynamic_task_id", None) != DYNAMIC_STAIR_TASK_ID
    or getattr(env_cfg, "stair_dynamic_training_contract", None) is not True
    or getattr(env_cfg, "stair_dynamic_maneuver_qualified", None) is not True
  ):
    raise ValueError("Registered StairDynamic task is not training/qualified v3.")
  action = env_cfg.actions.get("hybrid_wheel_leg")
  maneuver = getattr(action, "dynamic_stair_maneuver", None)
  maneuver_sha = getattr(maneuver, "maneuver_hash", None)
  maneuver_bindings = getattr(env_cfg, "stair_dynamic_maneuver_bindings", None)
  if (
    not isinstance(maneuver_sha, str)
    or not isinstance(maneuver_bindings, Mapping)
  ):
    raise ValueError("Registered StairDynamic maneuver provenance is missing.")
  expectation = _expectation_payload(
    git_sha=repository_git_sha(),
    contract_sha256=dynamic_stair_contract_hash(env_cfg, agent_cfg),
    artifact_bindings=dynamic_stair_artifact_bindings(env_cfg),
    maneuver_sha256=maneuver_sha,
    maneuver_bindings=maneuver_bindings,
    completed_updates=completed_updates,
  )
  required_sensors = _registered_per_wheel_sensor_names(env_cfg)
  result = {
    "schema_version": PREFLIGHT_SCHEMA_VERSION,
    "kind": "stair_dynamic_runtime_preflight",
    "status": "pass",
    "classification": "STAIR_DYNAMIC_STATIC_PREFLIGHT_PASS",
    "task": DYNAMIC_STAIR_TASK_ID,
    "simulation_started": False,
    "policy_interface": {
      "actor_observation_width": DYNAMIC_STAIR_ACTOR_WIDTH,
      "critic_observation_width": DYNAMIC_STAIR_CRITIC_WIDTH,
      "action_width": len(DYNAMIC_STAIR_ACTION_SCALES),
      "action_scales": list(DYNAMIC_STAIR_ACTION_SCALES),
    },
    "per_wheel_sensors": list(required_sensors),
    "expectation": expectation,
  }
  return result


def _atomic_json_no_clobber(payload: Mapping[str, object], output: Path) -> None:
  if output.exists():
    raise FileExistsError(f"Refusing to overwrite {output}.")
  output.parent.mkdir(parents=True, exist_ok=True)
  temporary = output.with_name(f".{output.name}.incomplete.{uuid.uuid4().hex}")
  encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
  try:
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
      stream.write(encoded)
      stream.flush()
      os.fsync(stream.fileno())
    os.link(temporary, output)
  finally:
    temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest="command", required=True)
  runtime = commands.add_parser("runtime-expectation")
  runtime.add_argument(
    "--completed-updates",
    type=int,
    help="0 for migration evaluation; omit to accept a trained checkpoint budget.",
  )
  runtime.add_argument("--output", type=Path, required=True)
  search = commands.add_parser("search-bindings")
  search.add_argument("--stage5-checkpoint", type=Path, required=True)
  search.add_argument("--stage5-gate", type=Path, required=True)
  search.add_argument("--trigger-qualification", type=Path, required=True)
  search.add_argument("--output", type=Path, required=True)
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  if args.command == "runtime-expectation":
    result = collect_runtime_preflight(completed_updates=args.completed_updates)
    expectation = result["expectation"]
    if not isinstance(expectation, Mapping):  # pragma: no cover - internal invariant
      raise TypeError("StairDynamic preflight expectation must be a mapping.")
    _atomic_json_no_clobber(expectation, args.output)
    print("[PASS] StairDynamic static registry preflight (no simulation)")
  elif args.command == "search-bindings":
    bindings = collect_search_bindings(
      stage5_checkpoint=args.stage5_checkpoint,
      stage5_gate=args.stage5_gate,
      trigger_qualification=args.trigger_qualification,
    )
    _atomic_json_no_clobber(bindings, args.output)
    print("[PASS] StairDynamic search bindings resolved (no simulation)")
  else:  # pragma: no cover
    raise AssertionError(f"Unhandled command: {args.command}")
  return 0


__all__ = [
  "PREFLIGHT_SCHEMA_VERSION",
  "collect_runtime_preflight",
  "collect_search_bindings",
  "main",
  "parse_args",
]


if __name__ == "__main__":
  raise SystemExit(main())
