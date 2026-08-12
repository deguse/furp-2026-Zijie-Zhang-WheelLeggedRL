#!/usr/bin/env python3
"""Adjudicate RollAssist K=3, continuation, or final expansion envelopes."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hoppertrex_mjlab.hybrid.roll_assist import (
  build_extension_authorization,
  canonical_json_sha256,
  continuation_gate,
  file_sha256,
  final_expansion_gate,
  newest_passer,
)


def _read(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8-sig"))
  if not isinstance(value, dict):
    raise TypeError(f"{path} must contain a JSON object.")
  return value


K3_CHECK_FIELDS = {
  "flat_retention_passed", "hpass_retained", "hnext_safe",
  "wheel_residual_exact_zero",
}
K3_CANDIDATE_FIELDS = {
  "screen_envelope_file", "screen_envelope_sha256", "checkpoint_file",
  "checkpoint_file_sha256", "completed_updates", "passed", "checks",
}
K3_SELECTION_FIELDS = {
  "schema_version", "kind", "classification", "selection_rule",
  "candidates", "selected",
}


def _sha256_text(value: Any, *, name: str) -> str:
  if (
    not isinstance(value, str)
    or len(value) != 64
    or any(character not in "0123456789abcdef" for character in value)
  ):
    raise ValueError(f"{name} must be a lowercase SHA256.")
  return value


def validate_k3_selection(
  payload: Mapping[str, Any], *, verify_screen_files: bool = True
) -> dict[str, Any]:
  """Validate canonical ordering, latest-three grid, and byte-bound screens."""

  if set(payload) != K3_SELECTION_FIELDS:
    raise ValueError("RollAssist K=3 selection schema drifted.")
  if (
    payload.get("schema_version") != 1
    or payload.get("kind") != "roll_assist_k3_selection"
    or payload.get("selection_rule") != "newest_passer_rejection_only"
  ):
    raise ValueError("RollAssist K=3 selection identity drifted.")
  raw_candidates = payload.get("candidates")
  if not isinstance(raw_candidates, list) or len(raw_candidates) != 3:
    raise ValueError("RollAssist K=3 selection requires exactly three candidates.")
  candidates: list[dict[str, Any]] = []
  for raw in raw_candidates:
    if not isinstance(raw, Mapping) or set(raw) != K3_CANDIDATE_FIELDS:
      raise ValueError("RollAssist K=3 candidate schema drifted.")
    candidate = dict(raw)
    updates = candidate.get("completed_updates")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 1:
      raise ValueError("RollAssist K=3 candidate update is invalid.")
    if not isinstance(candidate.get("passed"), bool):
      raise TypeError("RollAssist K=3 candidate pass status must be boolean.")
    checks = candidate.get("checks")
    if (
      not isinstance(checks, Mapping)
      or set(checks) != K3_CHECK_FIELDS
      or not all(isinstance(value, bool) for value in checks.values())
      or candidate["passed"] is not all(checks.values())
    ):
      raise ValueError("RollAssist K=3 candidate checks drifted.")
    _sha256_text(candidate.get("checkpoint_file_sha256"), name="checkpoint SHA")
    screen_sha = _sha256_text(
      candidate.get("screen_envelope_sha256"), name="screen-envelope SHA"
    )
    checkpoint_path = Path(str(candidate.get("checkpoint_file", ""))).resolve()
    if verify_screen_files and (
      not checkpoint_path.is_file()
      or file_sha256(checkpoint_path) != candidate["checkpoint_file_sha256"]
    ):
      raise ValueError("RollAssist K=3 checkpoint bytes drifted.")
    screen_path = Path(str(candidate.get("screen_envelope_file", ""))).resolve()
    if verify_screen_files:
      if not screen_path.is_file() or file_sha256(screen_path) != screen_sha:
        raise ValueError("RollAssist K=3 screen-envelope bytes drifted.")
      screen = _read(screen_path)
      expected_screen = {
        "schema_version": 1,
        "kind": "roll_assist_k3_screen",
        "checkpoint_file": candidate["checkpoint_file"],
        "checkpoint_file_sha256": candidate["checkpoint_file_sha256"],
        "completed_updates": updates,
        "passed": candidate["passed"],
        "checks": dict(checks),
      }
      if screen != expected_screen:
        raise ValueError("RollAssist K=3 screen and selection candidate differ.")
    candidates.append(candidate)
  updates = [int(candidate["completed_updates"]) for candidate in candidates]
  if updates != sorted(updates):
    raise ValueError("RollAssist K=3 candidates are not strictly update-ordered.")
  if len({candidate["checkpoint_file_sha256"] for candidate in candidates}) != 3:
    raise ValueError("RollAssist K=3 must bind three distinct checkpoint bytes.")
  expected_selected = newest_passer(candidates)
  selected = payload.get("selected")
  classification = payload.get("classification")
  if expected_selected is None:
    if selected is not None or classification != "ROLL_ASSIST_K3_NO_PASSER":
      raise ValueError("RollAssist K=3 no-passer classification drifted.")
  elif (
    not isinstance(selected, Mapping)
    or dict(selected) != dict(expected_selected)
    or classification != "ROLL_ASSIST_K3_PASSER_SELECTED"
  ):
    raise ValueError("RollAssist K=3 newest-passer selection drifted.")
  return dict(payload)


def select_k3(screen_paths: list[Path]) -> dict[str, Any]:
  """Build an ordered, byte-bound K=3 rejection-only selection."""

  if len(screen_paths) != 3:
    raise ValueError("select-k3 requires exactly three envelopes.")
  candidates: list[dict[str, Any]] = []
  for source in screen_paths:
    path = source.expanduser().resolve()
    envelope = _read(path)
    if (
      envelope.get("kind") != "roll_assist_k3_screen"
      or envelope.get("schema_version") != 1
    ):
      raise ValueError("select-k3 requires canonical RollAssist screen envelopes.")
    candidates.append({
      "screen_envelope_file": str(path),
      "screen_envelope_sha256": file_sha256(path),
      "checkpoint_file": envelope.get("checkpoint_file"),
      "checkpoint_file_sha256": envelope.get("checkpoint_file_sha256"),
      "completed_updates": envelope.get("completed_updates"),
      "passed": envelope.get("passed"),
      "checks": envelope.get("checks"),
    })
  if len({str(item["checkpoint_file_sha256"]) for item in candidates}) != 3:
    raise ValueError("RollAssist K=3 must reference three distinct checkpoint bytes.")
  candidates.sort(key=lambda item: int(item["completed_updates"]))
  selected = newest_passer(candidates)
  result = {
    "schema_version": 1,
    "kind": "roll_assist_k3_selection",
    "classification": (
      "ROLL_ASSIST_K3_PASSER_SELECTED" if selected else "ROLL_ASSIST_K3_NO_PASSER"
    ),
    "selection_rule": "newest_passer_rejection_only",
    "candidates": candidates,
    "selected": selected,
  }
  return validate_k3_selection(result, verify_screen_files=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="phase", required=True)
  select = sub.add_parser("select-k3")
  select.add_argument("--checkpoint-envelope", type=Path, action="append", required=True)
  select.add_argument("--output", type=Path, required=True)
  validate = sub.add_parser("validate-k3")
  validate.add_argument("--selection", type=Path, required=True)
  validate.add_argument("--verify-screen-files", action="store_true")
  continuation = sub.add_parser("continuation")
  continuation.add_argument("--evidence", type=Path, required=True)
  continuation.add_argument("--selected-checkpoint", type=Path)
  continuation.add_argument("--selected-completed-updates", type=int)
  continuation.add_argument("--target-total-updates", type=int)
  continuation.add_argument("--output", type=Path, required=True)
  final = sub.add_parser("final")
  final.add_argument("--evidence", type=Path, required=True)
  final.add_argument("--output", type=Path, required=True)
  return parser.parse_args(argv)


CONTINUATION_FIELDS = {
  "flat_retention_passed", "hpass_card_successes", "hnext_terminations",
  "hnext_non_wheel_contacts", "hnext_bilateral_airborne",
  "wheel_residual_abs_max", "hnext_candidate_successes",
  "hnext_baseline_successes", "paired_candidate_progress",
  "paired_baseline_progress",
}


def _formal_continuation(payload: Mapping[str, Any]) -> Mapping[str, Any]:
  if set(payload) == CONTINUATION_FIELDS:
    return payload
  if (
    payload.get("kind") != "roll_assist_evaluation"
    or payload.get("profile") != "formal"
    or payload.get("evidence_eligible") is not True
  ):
    raise ValueError("Continuation authorization requires formal eligible evaluator evidence.")
  unsigned = dict(payload)
  observed = unsigned.pop("evaluation_sha256", None)
  if observed != canonical_json_sha256(unsigned):
    raise ValueError("Formal RollAssist evaluation hash drifted.")
  final = payload.get("final")
  if not isinstance(final, Mapping):
    raise TypeError("Formal RollAssist evaluation has no final gate.")
  if final.get("passed") is True:
    raise ValueError("RollAssist Hnext already passed; extension must stop immediately.")
  continuation = payload.get("continuation")
  if not isinstance(continuation, Mapping):
    raise TypeError("Formal RollAssist evaluation has no continuation gate.")
  return continuation


def adjudicate_continuation(payload: Mapping[str, Any]) -> dict[str, Any]:
  evidence = _formal_continuation(payload)
  # Evaluator envelopes carry the already-adjudicated continuation object;
  # raw input remains supported for a narrow unit/CLI boundary.
  if set(evidence) == CONTINUATION_FIELDS:
    return continuation_gate(**evidence)
  expected = {"authorized", "classification", "checks", "paired_progress_bootstrap"}
  if set(evidence) != expected:
    raise ValueError("Continuation evidence schema drifted.")
  if evidence.get("authorized") is not True or evidence.get("classification") != "ROLL_ASSIST_EXTEND_BLOCK":
    raise ValueError("Formal evaluator did not authorize another RollAssist block.")
  return dict(evidence)


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  if args.phase == "validate-k3":
    result = validate_k3_selection(
      _read(args.selection), verify_screen_files=args.verify_screen_files
    )
    print(f"[roll-assist] classification={result['classification']}")
    print(f"[roll-assist] validated_selection={args.selection}")
    return
  if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite adjudication: {args.output}")
  if args.phase == "select-k3":
    result = select_k3(args.checkpoint_envelope)
  elif args.phase == "continuation":
    result = adjudicate_continuation(_read(args.evidence))
    extension_args = (
      args.selected_checkpoint,
      args.selected_completed_updates,
      args.target_total_updates,
    )
    if any(value is not None for value in extension_args):
      if not all(value is not None for value in extension_args):
        raise ValueError("Continuation authorization requires all checkpoint/update arguments.")
      if result["authorized"] is not True:
        raise ValueError("Rejected continuation evidence cannot authorize training.")
      source_evidence = _read(args.evidence)
      if source_evidence.get("kind") != "roll_assist_evaluation":
        raise ValueError(
          "Training authorization requires the full formal evaluator envelope."
        )
      checkpoint = source_evidence.get("checkpoint")
      if not isinstance(checkpoint, Mapping):
        raise TypeError("Formal evidence has no checkpoint envelope.")
      if (
        checkpoint.get("checkpoint_file_sha256")
        != file_sha256(args.selected_checkpoint.resolve())
        or checkpoint.get("completed_updates") != args.selected_completed_updates
      ):
        raise ValueError("Formal evidence and selected checkpoint differ.")
      selected = args.selected_checkpoint.resolve()
      if not selected.is_file():
        raise FileNotFoundError(f"Selected RollAssist checkpoint does not exist: {selected}")
      result = build_extension_authorization(
        selected_checkpoint_file=selected,
        selected_checkpoint_sha256=file_sha256(selected),
        selected_completed_updates=args.selected_completed_updates,
        target_total_updates=args.target_total_updates,
        continuation_evidence_sha256=file_sha256(args.evidence.resolve()),
      )
  else:
    evidence = _read(args.evidence)
    expected = {"hnext_card_successes", "safety_gate_passed", "wheel_residual_abs_max"}
    if set(evidence) != expected:
      raise ValueError("Final evidence schema drifted.")
    result = final_expansion_gate(**evidence)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  temporary = args.output.with_name(f".{args.output.name}.incomplete")
  try:
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
  finally:
    if temporary.exists():
      temporary.unlink()
  print(f"[roll-assist] classification={result['classification']}")
  print(f"[roll-assist] output={args.output}")


if __name__ == "__main__":
  main()
