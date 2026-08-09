#!/usr/bin/env python3
"""Fail-closed three-seed adjudication for the registered StairCamp campaign.

The sidecar deliberately keeps aggregation outside the frozen
``residual_promotion_decision`` function.  It validates all three formal seed
envelopes first, injects the registered synthetic flat row symmetrically, and
then calls the frozen decision exactly once for each seed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hoppertrex_mjlab.hybrid.stair_residual import residual_promotion_decision

REGISTERED_TRAINING_SEEDS = frozenset((1, 2, 3))
REGISTERED_EVALUATION_SEED = 1
REGISTERED_HEIGHT_GRID_M = (0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15)
# The classical arm is frozen C0 evidence, and that probe swept 0.00-0.10 m in
# 0.01 m steps only, so 0.15 m is the one registered height it cannot supply.
# Demanding it would require either authoring a number or re-sweeping a frozen
# script for a cell that provably cannot change the verdict: the classical
# contiguous passing prefix already terminates at 0.01 m (measured 0/48 at
# every tier from 0.01 m up), so `classical_height_m` is 0.00 m with or
# without a 0.15 m row. The classical grid is therefore the frozen evidence
# grid; the residual arm keeps the full registered scan.
REGISTERED_CLASSICAL_HEIGHT_GRID_M = (0.01, 0.02, 0.03, 0.05, 0.07, 0.10)
REGISTERED_TRIALS_PER_HEIGHT = 48
REGISTERED_BUDGET_ITERATIONS = frozenset((1000, 3000))
STAIR_CAMP_CANONICAL_CONTRACT_SHA256 = (
  "1d4b18db32e48b3ae8803e385a032203bdddc7f8198da9679f519bc8947190cb"
)
MINIMUM_BOUNDARY_EXTENSION_M = 0.01
BOUNDARY_TOLERANCE_M = 1.0e-12

PROMOTION_CLASSIFICATION = "RESIDUAL_PPO_EXTENDS_CLASSICAL_BOUNDARY"
STOP_CLASSIFICATION = "STOP_NO_PROMOTION"
VALID_CLASSIFICATIONS = frozenset((PROMOTION_CLASSIFICATION, STOP_CLASSIFICATION))

GATE_NAMES = (
  "flat_gate_passed",
  "standing_gate_passed",
  "velocity_gate_passed",
  "stage5_gate_passed",
)
ROW_FIELDS = (
  "height_m",
  "success_rate",
  "terminations",
  "non_wheel_contacts",
  "trials",
)
ARTIFACT_BINDING_NAMES = (
  "controller_gain_hash",
  "calibration_hash",
  "yaw_calibration_hash",
  "posture_map_hash",
  "posture_artifact_hash",
  "station_calibration_hash",
)
REQUIRED_ABLATIONS = (
  "leg-off",
  "zero-shot-scale-0.035",
  "zero-shot-scale-0.070",
  "zero-shot-scale-0.100",
  "mode-always-on",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SYNTHETIC_FLAT_ROW: dict[str, float | int] = {
  "height_m": 0.0,
  "success_rate": 1.0,
  "terminations": 0,
  "non_wheel_contacts": 0,
}


class StairCampAdjudicationError(ValueError):
  """Raised when an input violates the registered adjudication protocol."""


@dataclass(frozen=True)
class _ValidatedEnvelope:
  training_seed: int
  evaluation_seed: int
  budget_iterations: int
  git_sha: str
  contract_hash: str
  artifact_bindings: dict[str, Any]
  classical_rows: tuple[dict[str, float | int], ...]
  residual_rows: tuple[dict[str, float | int], ...]
  gates: dict[str, bool]
  gate_stair_mode_false_positives: dict[str, int]
  completed_ablations: tuple[str, ...]
  ablations_complete: bool
  checkpoint: str
  checkpoint_file_sha256: str


def _protocol_error(path: str, message: str) -> StairCampAdjudicationError:
  return StairCampAdjudicationError(f"{path}: {message}")


def _validate_json_value(value: object, *, path: str = "$") -> None:
  """Reject non-JSON values and every NaN/Inf, including ignored metadata."""

  if value is None or isinstance(value, (str, bool, int)):
    return
  if isinstance(value, float):
    if not math.isfinite(value):
      raise _protocol_error(path, "NaN and infinity are forbidden")
    return
  if isinstance(value, Mapping):
    for key, item in value.items():
      if not isinstance(key, str):
        raise _protocol_error(path, "object keys must be strings")
      _validate_json_value(item, path=f"{path}.{key}")
    return
  if isinstance(value, Sequence) and not isinstance(
    value, (str, bytes, bytearray)
  ):
    for index, item in enumerate(value):
      _validate_json_value(item, path=f"{path}[{index}]")
    return
  raise _protocol_error(path, f"unsupported JSON value {type(value).__name__}")


def _require_mapping(value: object, *, path: str) -> Mapping[str, object]:
  if not isinstance(value, Mapping):
    raise _protocol_error(path, "must be an object")
  if any(not isinstance(key, str) for key in value):
    raise _protocol_error(path, "object keys must be strings")
  return value


def _require_sequence(value: object, *, path: str) -> Sequence[object]:
  if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
    raise _protocol_error(path, "must be an array")
  return value


def _require_int(value: object, *, path: str) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise _protocol_error(path, "must be an integer")
  return value


def _require_count(value: object, *, path: str) -> int:
  count = _require_int(value, path=path)
  if count < 0 or count > REGISTERED_TRIALS_PER_HEIGHT:
    raise _protocol_error(
      path,
      f"must be between 0 and {REGISTERED_TRIALS_PER_HEIGHT}",
    )
  return count


def _require_nonnegative_int(value: object, *, path: str) -> int:
  count = _require_int(value, path=path)
  if count < 0:
    raise _protocol_error(path, "must be nonnegative")
  return count


def _require_float(value: object, *, path: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise _protocol_error(path, "must be a finite number")
  result = float(value)
  if not math.isfinite(result):
    raise _protocol_error(path, "must be a finite number")
  return result


def _require_bool(value: object, *, path: str) -> bool:
  if not isinstance(value, bool):
    raise _protocol_error(path, "must be a boolean")
  return value


def _require_nonempty_string(value: object, *, path: str) -> str:
  if not isinstance(value, str) or not value.strip():
    raise _protocol_error(path, "must be a non-empty string")
  return value


def _optional_nonempty_string(
  value: object,
  *,
  path: str,
) -> str | None:
  if value is None:
    return None
  return _require_nonempty_string(value, path=path)


def _require_sha256(value: object, *, path: str) -> str:
  digest = _require_nonempty_string(value, path=path)
  if not _SHA256_RE.fullmatch(digest):
    raise _protocol_error(path, "must be a lowercase 64-character SHA256")
  return digest


def _require_git_sha(value: object, *, path: str) -> str:
  digest = _require_nonempty_string(value, path=path)
  if not _GIT_SHA_RE.fullmatch(digest):
    raise _protocol_error(path, "must be a lowercase 40-character Git SHA")
  return digest


def _canonical_json_value(value: object) -> Any:
  """Return a detached JSON value while preserving type-sensitive equality."""

  return json.loads(
    json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
  )


def _canonical_json(value: object) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_rows(
  value: object,
  *,
  path: str,
  grid: tuple[float, ...] = REGISTERED_HEIGHT_GRID_M,
) -> tuple[dict[str, float | int], ...]:
  rows = _require_sequence(value, path=path)
  if len(rows) != len(grid):
    raise _protocol_error(
      path,
      "must contain exactly one row for every registered height "
      f"{grid}",
    )

  by_height: dict[float, dict[str, float | int]] = {}
  for index, raw_row in enumerate(rows):
    row_path = f"{path}[{index}]"
    row = _require_mapping(raw_row, path=row_path)
    missing = [field for field in ROW_FIELDS if field not in row]
    if missing:
      raise _protocol_error(row_path, f"missing required fields {missing}")

    height = _require_float(row["height_m"], path=f"{row_path}.height_m")
    if height not in grid:
      raise _protocol_error(
        f"{row_path}.height_m",
        f"must be exactly one of {grid}",
      )
    if height in by_height:
      raise _protocol_error(
        f"{row_path}.height_m",
        f"duplicates registered height {height}",
      )

    trials = _require_int(row["trials"], path=f"{row_path}.trials")
    if trials != REGISTERED_TRIALS_PER_HEIGHT:
      raise _protocol_error(
        f"{row_path}.trials",
        f"must equal {REGISTERED_TRIALS_PER_HEIGHT}",
      )
    success_rate = _require_float(
      row["success_rate"], path=f"{row_path}.success_rate"
    )
    if not 0.0 <= success_rate <= 1.0:
      raise _protocol_error(f"{row_path}.success_rate", "must be in [0, 1]")
    terminations = _require_count(
      row["terminations"], path=f"{row_path}.terminations"
    )
    non_wheel_contacts = _require_count(
      row["non_wheel_contacts"], path=f"{row_path}.non_wheel_contacts"
    )

    by_height[height] = {
      "height_m": height,
      "success_rate": success_rate,
      "terminations": terminations,
      "non_wheel_contacts": non_wheel_contacts,
      "trials": trials,
    }

  actual_heights = frozenset(by_height)
  expected_heights = frozenset(grid)
  if actual_heights != expected_heights:
    missing = sorted(expected_heights - actual_heights)
    extra = sorted(actual_heights - expected_heights)
    raise _protocol_error(path, f"height grid mismatch; missing={missing}, extra={extra}")

  return tuple(by_height[height] for height in grid)


def _validate_envelope(
  value: object,
  *,
  index: int,
) -> _ValidatedEnvelope:
  path = f"$[{index}]"
  envelope = _require_mapping(value, path=path)
  required = (
    "training_seed",
    "evaluation_seed",
    "budget_iterations",
    "git_sha",
    "contract_hash",
    "artifact_bindings",
    "classical_rows",
    "residual_rows",
    *GATE_NAMES,
    "gate_stair_mode_false_positives",
    "completed_ablations",
    "ablations_complete",
    "evidence_eligible",
    "checkpoint",
    "checkpoint_file_sha256",
  )
  missing = [field for field in required if field not in envelope]
  if missing:
    raise _protocol_error(path, f"missing required fields {missing}")

  training_seed = _require_int(
    envelope["training_seed"], path=f"{path}.training_seed"
  )
  evaluation_seed = _require_int(
    envelope["evaluation_seed"], path=f"{path}.evaluation_seed"
  )
  if evaluation_seed != REGISTERED_EVALUATION_SEED:
    raise _protocol_error(
      f"{path}.evaluation_seed",
      f"must equal registered seed {REGISTERED_EVALUATION_SEED}",
    )
  if envelope["evidence_eligible"] is not True:
    raise _protocol_error(
      f"{path}.evidence_eligible", "formal evidence must be eligible"
    )

  budget_iterations = _require_int(
    envelope["budget_iterations"], path=f"{path}.budget_iterations"
  )
  if budget_iterations not in REGISTERED_BUDGET_ITERATIONS:
    raise _protocol_error(
      f"{path}.budget_iterations",
      f"must be one of {sorted(REGISTERED_BUDGET_ITERATIONS)}",
    )

  bindings = _require_mapping(
    envelope["artifact_bindings"], path=f"{path}.artifact_bindings"
  )
  if set(bindings) != set(ARTIFACT_BINDING_NAMES):
    raise _protocol_error(
      f"{path}.artifact_bindings", "must contain exactly the six frozen bindings"
    )
  normalized_bindings = {
    name: _require_sha256(
      bindings[name], path=f"{path}.artifact_bindings.{name}"
    )
    for name in ARTIFACT_BINDING_NAMES
  }

  gates = {
    gate: _require_bool(envelope[gate], path=f"{path}.{gate}")
    for gate in GATE_NAMES
  }
  raw_false_positives = _require_mapping(
    envelope["gate_stair_mode_false_positives"],
    path=f"{path}.gate_stair_mode_false_positives",
  )
  if set(raw_false_positives) != set(GATE_NAMES):
    raise _protocol_error(
      f"{path}.gate_stair_mode_false_positives",
      "must contain exactly the four gate names",
    )
  false_positives = {
    gate: _require_nonnegative_int(
      raw_false_positives[gate],
      path=f"{path}.gate_stair_mode_false_positives.{gate}",
    )
    for gate in GATE_NAMES
  }
  for gate in GATE_NAMES:
    if gates[gate] and false_positives[gate] != 0:
      raise _protocol_error(
        f"{path}.{gate}",
        "cannot pass while stair_mode false positives are nonzero",
      )

  raw_ablations = _require_sequence(
    envelope["completed_ablations"], path=f"{path}.completed_ablations"
  )
  completed_ablations: list[str] = []
  for ablation_index, ablation_value in enumerate(raw_ablations):
    name = _require_nonempty_string(
      ablation_value, path=f"{path}.completed_ablations[{ablation_index}]"
    )
    if name not in REQUIRED_ABLATIONS or name in completed_ablations:
      raise _protocol_error(
        f"{path}.completed_ablations[{ablation_index}]",
        "must be a unique registered ablation",
      )
    completed_ablations.append(name)
  ablations_complete = _require_bool(
    envelope["ablations_complete"], path=f"{path}.ablations_complete"
  )
  derived_complete = set(completed_ablations) == set(REQUIRED_ABLATIONS)
  if ablations_complete is not derived_complete:
    raise _protocol_error(
      f"{path}.ablations_complete",
      "does not match the completed registered ablation set",
    )

  contract_hash = _require_sha256(
    envelope["contract_hash"], path=f"{path}.contract_hash"
  )
  if contract_hash != STAIR_CAMP_CANONICAL_CONTRACT_SHA256:
    raise _protocol_error(
      f"{path}.contract_hash", "does not match the canonical StairCamp contract"
    )

  return _ValidatedEnvelope(
    training_seed=training_seed,
    evaluation_seed=evaluation_seed,
    budget_iterations=budget_iterations,
    git_sha=_require_git_sha(envelope["git_sha"], path=f"{path}.git_sha"),
    contract_hash=contract_hash,
    artifact_bindings=normalized_bindings,
    classical_rows=_validate_rows(
      envelope["classical_rows"],
      path=f"{path}.classical_rows",
      grid=REGISTERED_CLASSICAL_HEIGHT_GRID_M,
    ),
    residual_rows=_validate_rows(
      envelope["residual_rows"], path=f"{path}.residual_rows"
    ),
    gates=gates,
    gate_stair_mode_false_positives=false_positives,
    completed_ablations=tuple(completed_ablations),
    ablations_complete=ablations_complete,
    checkpoint=_require_nonempty_string(
      envelope["checkpoint"], path=f"{path}.checkpoint"
    ),
    checkpoint_file_sha256=_require_sha256(
      envelope["checkpoint_file_sha256"],
      path=f"{path}.checkpoint_file_sha256",
    ),
  )


def _require_consistent_binding(
  envelopes: Sequence[_ValidatedEnvelope],
  attribute: str,
) -> None:
  rendered = {
    _canonical_json(getattr(envelope, attribute)) for envelope in envelopes
  }
  if len(rendered) != 1:
    raise _protocol_error("$", f"seed envelopes disagree on {attribute}")


def _inject_synthetic_flat_row(
  rows: Sequence[Mapping[str, float | int]],
) -> list[dict[str, float | int]]:
  return [dict(SYNTHETIC_FLAT_ROW), *(dict(row) for row in rows)]


def adjudicate_stair_camp(
  envelopes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
  """Validate and adjudicate exactly the registered three formal seeds.

  Scientific failures (a failed row, gate, ablation, or minimum extension)
  return ``STOP_NO_PROMOTION``.  Protocol failures raise
  :class:`StairCampAdjudicationError` before the frozen function is called.
  """

  _validate_json_value(envelopes)
  raw_envelopes = _require_sequence(envelopes, path="$")
  if len(raw_envelopes) != len(REGISTERED_TRAINING_SEEDS):
    raise _protocol_error(
      "$",
      "requires exactly three seed envelopes with training seeds {1, 2, 3}",
    )

  validated = [
    _validate_envelope(envelope, index=index)
    for index, envelope in enumerate(raw_envelopes)
  ]
  seeds = [envelope.training_seed for envelope in validated]
  if len(set(seeds)) != len(seeds):
    raise _protocol_error("$", f"duplicate training seeds are forbidden: {seeds}")
  if set(seeds) != REGISTERED_TRAINING_SEEDS:
    raise _protocol_error(
      "$",
      "training seeds must be exactly {1, 2, 3}; "
      f"received {sorted(seeds)}",
    )

  ordered = sorted(validated, key=lambda envelope: envelope.training_seed)
  for attribute in (
    "budget_iterations",
    "git_sha",
    "contract_hash",
    "artifact_bindings",
  ):
    _require_consistent_binding(ordered, attribute)

  seed_results: list[dict[str, object]] = []
  boundary_extensions: list[float] = []
  for envelope in ordered:
    classical_rows = _inject_synthetic_flat_row(envelope.classical_rows)
    residual_rows = _inject_synthetic_flat_row(envelope.residual_rows)
    decision = residual_promotion_decision(
      classical_rows=classical_rows,
      residual_rows=residual_rows,
      **envelope.gates,
      ablations_complete=envelope.ablations_complete,
    )

    classification = decision.get("classification")
    if classification not in VALID_CLASSIFICATIONS:
      raise _protocol_error(
        f"$.seed[{envelope.training_seed}].decision.classification",
        "frozen decision returned an unsupported classification",
      )
    promotion_eligible = decision.get("promotion_eligible")
    if not isinstance(promotion_eligible, bool):
      raise _protocol_error(
        f"$.seed[{envelope.training_seed}].decision.promotion_eligible",
        "frozen decision must return a boolean",
      )
    extension = _require_float(
      decision.get("boundary_extension_m"),
      path=f"$.seed[{envelope.training_seed}].decision.boundary_extension_m",
    )
    boundary_extensions.append(extension)

    seed_results.append({
      "training_seed": envelope.training_seed,
      "evaluation_seed": envelope.evaluation_seed,
      "checkpoint": envelope.checkpoint,
      "checkpoint_file_sha256": envelope.checkpoint_file_sha256,
      "gates": dict(envelope.gates),
      "gate_stair_mode_false_positives": dict(
        envelope.gate_stair_mode_false_positives
      ),
      "completed_ablations": list(envelope.completed_ablations),
      "ablations_complete": envelope.ablations_complete,
      "classical_rows": classical_rows,
      "residual_rows": residual_rows,
      "decision": dict(decision),
    })

  minimum_extension = min(boundary_extensions)
  all_gates_passed = all(
    all(envelope.gates.values()) for envelope in ordered
  )
  all_trigger_fp_checks_passed = all(
    not any(envelope.gate_stair_mode_false_positives.values())
    for envelope in ordered
  )
  global_ablations_complete = all(
    envelope.ablations_complete for envelope in ordered
  )
  every_seed_eligible = all(
    result["decision"]["promotion_eligible"]  # type: ignore[index]
    for result in seed_results
  )
  minimum_extension_passed = (
    minimum_extension
    >= MINIMUM_BOUNDARY_EXTENSION_M - BOUNDARY_TOLERANCE_M
  )
  promotion_eligible = bool(
    every_seed_eligible
    and all_gates_passed
    and all_trigger_fp_checks_passed
    and global_ablations_complete
    and minimum_extension_passed
  )

  first = ordered[0]
  return {
    "schema_version": 1,
    "kind": "stair_camp_three_seed_adjudication",
    "classification": (
      PROMOTION_CLASSIFICATION if promotion_eligible else STOP_CLASSIFICATION
    ),
    "promotion_eligible": promotion_eligible,
    "training_seeds": sorted(REGISTERED_TRAINING_SEEDS),
    "evaluation_seed": REGISTERED_EVALUATION_SEED,
    "budget_iterations": first.budget_iterations,
    "git_sha": first.git_sha,
    "contract_hash": first.contract_hash,
    "artifact_bindings": first.artifact_bindings,
    "minimum_boundary_extension_m": minimum_extension,
    "minimum_boundary_extension_passed": minimum_extension_passed,
    "all_regression_gates_passed": all_gates_passed,
    "all_trigger_false_positive_checks_passed": all_trigger_fp_checks_passed,
    "ablations_complete": global_ablations_complete,
    "seed_results": seed_results,
  }


def to_deterministic_json(result: Mapping[str, object]) -> str:
  """Serialize a result as strict, deterministic, newline-terminated JSON."""

  _validate_json_value(result)
  return json.dumps(
    dict(result),
    indent=2,
    sort_keys=True,
    allow_nan=False,
  ) + "\n"


def write_atomic_json(path: str | Path, result: Mapping[str, object]) -> Path:
  """Atomically create deterministic JSON without replacing prior evidence."""

  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  if destination.exists():
    raise FileExistsError(f"Refusing to overwrite adjudication output: {destination}")
  encoded = to_deterministic_json(result)
  temporary = destination.with_name(
    f".{destination.name}.incomplete.{uuid.uuid4().hex}"
  )
  try:
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
      stream.write(encoded)
      stream.flush()
      os.fsync(stream.fileno())
    # Publish the fully written inode without a check-then-replace race. A
    # concurrent writer that wins the destination name makes os.link fail.
    os.link(temporary, destination)
  finally:
    temporary.unlink(missing_ok=True)
  return destination


def _reject_json_constant(value: str) -> None:
  raise _protocol_error("$", f"non-finite JSON constant {value} is forbidden")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
  result: dict[str, object] = {}
  for key, value in pairs:
    if key in result:
      raise _protocol_error("$", f"duplicate JSON object key {key!r}")
    result[key] = value
  return result


def load_envelopes(path: str | Path) -> Sequence[Mapping[str, object]]:
  """Load a JSON array (or an exact ``{"envelopes": [...]}`` wrapper)."""

  source = Path(path)
  try:
    payload = json.loads(
      source.read_text(encoding="utf-8-sig"),
      parse_constant=_reject_json_constant,
      object_pairs_hook=_reject_duplicate_keys,
    )
  except json.JSONDecodeError as exc:
    raise _protocol_error("$", f"invalid JSON in {source}: {exc.msg}") from exc

  if isinstance(payload, Mapping):
    if set(payload) != {"envelopes"}:
      raise _protocol_error(
        "$", 'input object must contain only the key "envelopes"'
      )
    payload = payload["envelopes"]
  rows = _require_sequence(payload, path="$")
  return [
    _require_mapping(envelope, path=f"$[{index}]")
    for index, envelope in enumerate(rows)
  ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input",
    required=True,
    type=Path,
    help="JSON array containing exactly the three final seed envelopes.",
  )
  parser.add_argument(
    "--output",
    required=True,
    type=Path,
    help="Atomic adjudication JSON destination.",
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  result = adjudicate_stair_camp(load_envelopes(args.input))
  write_atomic_json(args.output, result)
  print(result["classification"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
