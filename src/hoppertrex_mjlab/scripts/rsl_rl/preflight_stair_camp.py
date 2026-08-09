#!/usr/bin/env python3
"""Fail-closed preflight for the registered S5B StairCamp campaign.

The frozen C2-j3 arrays are replayed read-only with the exact runtime trigger
metric.  This module does not re-run any Stage0--5 or C2 simulation evidence.
It also validates the two fresh live false-positive checks that must pass before
PPO training starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

SRC_PATH = Path(__file__).resolve().parents[3]
if str(SRC_PATH) not in sys.path:
  sys.path.insert(0, str(SRC_PATH))

from hoppertrex_mjlab.hybrid.stair_trigger import (  # noqa: E402
  STAIR_TRIGGER_FORCE_N,
  STAIR_TRIGGER_WINDOW,
  stair_trigger_metric,
  update_stair_trigger,
)

PREFLIGHT_SCHEMA_VERSION = 1
FROZEN_C2_SHA256SUMS_SHA256 = (
  "8a11fa89914ccf22e47a7ed46adc8d22353ff1c5812a509fd300e7a6b032d29f"
)
FROZEN_C2_RESULT_NAME = "c2_innovation_detector_qualification.json"
FROZEN_C2_CELL_COUNT = 18
FROZEN_C2_PAIRS_PER_CELL = 16
FROZEN_C2_TICKS = 500
FROZEN_C2_CONTACT_SLOTS = 16
FROZEN_C2_TOTAL_PAIRS = FROZEN_C2_CELL_COUNT * FROZEN_C2_PAIRS_PER_CELL
FROZEN_C2_ALLOWED_CLASSIFICATIONS = frozenset(
  (
    "INNOVATION_DETECTOR_QUALIFIED",
    "C2_INNOVATION_DETECTOR_UNQUALIFIED_STOP",
  )
)
FROZEN_C2_SOURCE_GIT_SHA = "f1d58e38e125dd6cf3a6f90c377f9d9736b9c39b"
FROZEN_C2_MJLAB_GIT_SHA = "43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6"
PASS_CLASSIFICATION = "STAIR_CAMP_PREFLIGHT_PASS"
STOP_CLASSIFICATION = "STOP_NO_PROMOTION"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class StairCampPreflightError(ValueError):
  """Raised when input provenance or protocol shape is invalid."""


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _strict_json(path: Path) -> Any:
  def reject_constant(value: str) -> None:
    raise StairCampPreflightError(f"Non-finite JSON constant {value} is forbidden.")

  def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
      if key in result:
        raise StairCampPreflightError(f"Duplicate JSON key {key!r} is forbidden.")
      result[key] = value
    return result

  try:
    return json.loads(
      path.read_text(encoding="utf-8-sig"),
      parse_constant=reject_constant,
      object_pairs_hook=reject_duplicates,
    )
  except json.JSONDecodeError as exc:
    raise StairCampPreflightError(f"Invalid JSON in {path}: {exc.msg}") from exc


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
  if not isinstance(value, Mapping):
    raise StairCampPreflightError(f"{name} must be an object.")
  return value


def _sequence(value: object, *, name: str) -> Sequence[object]:
  if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
    raise StairCampPreflightError(f"{name} must be an array.")
  return value


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise StairCampPreflightError(f"{name} must be an integer.")
  if value < minimum:
    raise StairCampPreflightError(f"{name} must be at least {minimum}.")
  return value


def _finite(value: object, *, name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise StairCampPreflightError(f"{name} must be numeric.")
  result = float(value)
  if not math.isfinite(result):
    raise StairCampPreflightError(f"{name} must be finite.")
  return result


def _parse_sha256s(path: Path) -> dict[str, str]:
  records: dict[str, str] = {}
  for line_number, raw in enumerate(
    path.read_text(encoding="ascii").splitlines(), start=1
  ):
    if not raw:
      raise StairCampPreflightError("SHA256SUMS.txt contains a blank line.")
    pieces = raw.split("  ", maxsplit=1)
    if len(pieces) != 2 or not _SHA256_RE.fullmatch(pieces[0]):
      raise StairCampPreflightError(
        f"Malformed SHA256SUMS.txt line {line_number}."
      )
    name = pieces[1]
    if (
      name in records
      or Path(name).name != name
      or name in (".", "..", "SHA256SUMS.txt")
    ):
      raise StairCampPreflightError("SHA256SUMS.txt contains an unsafe filename.")
    records[name] = pieces[0]
  return records


def _directory_hash_snapshot(root: Path) -> dict[str, str]:
  files = sorted(path for path in root.iterdir() if path.is_file())
  return {path.name: _sha256(path) for path in files}


def _replay_trigger_arrays(
  *,
  found: np.ndarray,
  force: np.ndarray,
  normal: np.ndarray,
  impact_steps: np.ndarray,
  post_impact_steps: int,
) -> dict[str, object]:
  """Replay one cell with the deployed Torch metric/state machine."""

  if found.ndim != 3:
    raise StairCampPreflightError("C2 found array must be [ticks,pairs,slots].")
  ticks, pairs, slots = found.shape
  expected_vector = (ticks, pairs, slots, 3)
  if force.shape != expected_vector or normal.shape != expected_vector:
    raise StairCampPreflightError("C2 force/normal array shapes do not match found.")
  if impact_steps.shape != (pairs,):
    raise StairCampPreflightError("C2 impact_steps shape does not match pairs.")
  if found.dtype != np.float32 or force.dtype != np.float32 or normal.dtype != np.float32:
    raise StairCampPreflightError("C2 trigger arrays must retain frozen float32 dtype.")
  if impact_steps.dtype.kind not in "iu":
    raise StairCampPreflightError("C2 impact_steps must be integral.")
  if not (
    np.isfinite(found).all()
    and np.isfinite(force).all()
    and np.isfinite(normal).all()
  ):
    raise StairCampPreflightError("C2 trigger arrays contain NaN or infinity.")
  if np.any(found < 0.0) or np.any(found != np.floor(found)):
    raise StairCampPreflightError("C2 found values must be nonnegative match counts.")
  if post_impact_steps < 1:
    raise StairCampPreflightError("C2 post-impact window must be positive.")
  if np.any(impact_steps < 0) or np.any(impact_steps >= ticks):
    raise StairCampPreflightError("C2 impact step lies outside the capture.")

  found_tensor = torch.from_numpy(found)
  force_tensor = torch.from_numpy(force)
  normal_tensor = torch.from_numpy(normal)
  latched = torch.zeros(pairs, dtype=torch.bool)
  streak = torch.zeros(pairs, dtype=torch.int64)
  first_trigger = torch.full((pairs,), -1, dtype=torch.int64)
  for tick in range(ticks):
    metric = stair_trigger_metric(
      found=found_tensor[tick],
      force_contact_frame=force_tensor[tick],
      normal_global=normal_tensor[tick],
    )
    previous = latched
    latched, streak = update_stair_trigger(
      latched=latched,
      streak=streak,
      metric=metric,
      threshold=STAIR_TRIGGER_FORCE_N,
      window=STAIR_TRIGGER_WINDOW,
    )
    rising = latched & ~previous
    first_trigger[(first_trigger < 0) & rising] = tick

  impact = torch.from_numpy(impact_steps.astype(np.int64, copy=False))
  pre_impact = (first_trigger >= 0) & (first_trigger < impact)
  deadline = torch.clamp(impact + post_impact_steps, max=ticks - 1)
  detected = (first_trigger >= impact) & (first_trigger <= deadline)
  delays = first_trigger[detected] - impact[detected]
  return {
    "pairs": pairs,
    "detections": int(detected.sum().item()),
    "pre_impact_triggers": int(pre_impact.sum().item()),
    "missing_or_late": int((~detected & ~pre_impact).sum().item()),
    "minimum_delay_steps": None if not len(delays) else int(delays.min().item()),
    "maximum_delay_steps": None if not len(delays) else int(delays.max().item()),
  }


def replay_frozen_c2_trigger(
  root: str | Path,
  *,
  expected_manifest_sha256: str = FROZEN_C2_SHA256SUMS_SHA256,
  expected_cells: int = FROZEN_C2_CELL_COUNT,
  expected_pairs_per_cell: int = FROZEN_C2_PAIRS_PER_CELL,
  expected_ticks: int = FROZEN_C2_TICKS,
  expected_slots: int = FROZEN_C2_CONTACT_SLOTS,
) -> dict[str, object]:
  """Replay and prove that every frozen input file remained byte-identical."""

  directory = Path(root).resolve(strict=True)
  if not directory.is_dir():
    raise StairCampPreflightError("Frozen C2 replay path is not a directory.")
  sums_path = directory / "SHA256SUMS.txt"
  if not sums_path.is_file():
    raise StairCampPreflightError("Frozen C2 directory has no SHA256SUMS.txt.")
  if _sha256(sums_path) != expected_manifest_sha256:
    raise StairCampPreflightError("Frozen C2 SHA256SUMS.txt binding drifted.")
  expected_hashes = _parse_sha256s(sums_path)
  before = _directory_hash_snapshot(directory)
  expected_names = set(expected_hashes) | {"SHA256SUMS.txt"}
  if set(before) != expected_names:
    raise StairCampPreflightError("Frozen C2 directory file set drifted.")
  for name, expected_hash in expected_hashes.items():
    if before.get(name) != expected_hash:
      raise StairCampPreflightError(f"Frozen C2 file hash drifted: {name}.")

  result_payload = _mapping(
    _strict_json(directory / FROZEN_C2_RESULT_NAME), name="C2 result"
  )
  if (
    result_payload.get("classification") not in FROZEN_C2_ALLOWED_CLASSIFICATIONS
    or result_payload.get("evidence_eligible") is not True
    or result_payload.get("git_sha") != FROZEN_C2_SOURCE_GIT_SHA
    or result_payload.get("mjlab_git_sha") != FROZEN_C2_MJLAB_GIT_SHA
  ):
    raise StairCampPreflightError("Frozen C2 result provenance is not eligible.")
  if _exact_int(
    result_payload.get("completed_cell_count"), name="completed_cell_count"
  ) != expected_cells:
    raise StairCampPreflightError("Frozen C2 completed cell count drifted.")
  if _exact_int(
    result_payload.get("completed_pair_count"), name="completed_pair_count"
  ) != expected_cells * expected_pairs_per_cell:
    raise StairCampPreflightError("Frozen C2 completed pair count drifted.")
  protocol = _mapping(result_payload.get("protocol"), name="C2 protocol")
  post_steps = _exact_int(
    protocol.get("post_impact_steps"), name="post_impact_steps", minimum=1
  )
  cells = _sequence(result_payload.get("cells"), name="C2 cells")
  if len(cells) != expected_cells:
    raise StairCampPreflightError("Frozen C2 result does not contain every cell.")

  total_detections = 0
  total_pre = 0
  total_missing = 0
  cell_results: list[dict[str, object]] = []
  for index, raw_cell in enumerate(cells):
    cell = _mapping(raw_cell, name=f"C2 cells[{index}]")
    raw_name = f"cell_{index:02d}.npz"
    if cell.get("raw_file") != raw_name:
      raise StairCampPreflightError("Frozen C2 raw file ordering drifted.")
    if cell.get("raw_sha256") != expected_hashes.get(raw_name):
      raise StairCampPreflightError("Frozen C2 result/raw hash binding drifted.")
    raw_path = directory / raw_name
    with np.load(raw_path, allow_pickle=False) as raw:
      required = {
        "stair_contact_found",
        "stair_contact_force_contact_frame",
        "stair_contact_normal_global",
        "impact_steps",
        "stair_riser_contact",
      }
      if not required.issubset(raw.files):
        raise StairCampPreflightError("Frozen C2 trigger arrays are missing.")
      found = raw["stair_contact_found"]
      force = raw["stair_contact_force_contact_frame"]
      normal = raw["stair_contact_normal_global"]
      impacts = raw["impact_steps"]
      truth = raw["stair_riser_contact"]
    if found.shape != (
      expected_ticks,
      expected_pairs_per_cell,
      expected_slots,
    ):
      raise StairCampPreflightError("Frozen C2 found shape drifted.")
    if truth.shape != (expected_ticks, expected_pairs_per_cell) or truth.dtype != np.bool_:
      raise StairCampPreflightError("Frozen C2 riser-contact truth shape drifted.")
    expected_impacts = np.argmax(truth, axis=0)
    if np.any(~truth.any(axis=0)) or not np.array_equal(impacts, expected_impacts):
      raise StairCampPreflightError("Frozen C2 impact truth does not reproduce.")
    recorded_impacts = _sequence(cell.get("impact_steps"), name="cell impact_steps")
    if list(map(int, impacts)) != list(recorded_impacts):
      raise StairCampPreflightError("Frozen C2 result impact steps drifted.")
    replay = _replay_trigger_arrays(
      found=found,
      force=force,
      normal=normal,
      impact_steps=impacts,
      post_impact_steps=post_steps,
    )
    if replay["pairs"] != expected_pairs_per_cell:
      raise StairCampPreflightError("Frozen C2 pair width drifted.")
    total_detections += int(replay["detections"])
    total_pre += int(replay["pre_impact_triggers"])
    total_missing += int(replay["missing_or_late"])
    cell_results.append({"cell_index": index, "raw_sha256": before[raw_name], **replay})

  after = _directory_hash_snapshot(directory)
  if after != before:
    raise StairCampPreflightError("Frozen C2 bytes changed during read-only replay.")
  expected_total = expected_cells * expected_pairs_per_cell
  passed = (
    total_detections == expected_total
    and total_pre == 0
    and total_missing == 0
  )
  result: dict[str, object] = {
    "schema_version": PREFLIGHT_SCHEMA_VERSION,
    "kind": "stair_camp_c2_trigger_replay",
    "classification": PASS_CLASSIFICATION if passed else STOP_CLASSIFICATION,
    "passed": passed,
    "threshold_n": STAIR_TRIGGER_FORCE_N,
    "window_steps": STAIR_TRIGGER_WINDOW,
    "frozen_directory": str(directory),
    "sha256s_file_sha256": before["SHA256SUMS.txt"],
    "input_hashes_before": before,
    "input_hashes_after": after,
    "files_unchanged": True,
    "completed_cells": expected_cells,
    "completed_pairs": expected_total,
    "detections": total_detections,
    "pre_impact_triggers": total_pre,
    "missing_or_late": total_missing,
    "cell_results": cell_results,
  }
  _validate_json(result)
  return result


def validate_live_false_positive_result(
  payload: object,
  *,
  expected_domain: str,
) -> dict[str, object]:
  result = _mapping(payload, name=f"{expected_domain} false-positive result")
  required = {
    "schema_version",
    "kind",
    "domain",
    "threshold_n",
    "window_steps",
    "events",
    "stair_mode_false_positives",
    "completed",
  }
  if set(result) != required:
    raise StairCampPreflightError(
      f"{expected_domain} false-positive result schema drifted."
    )
  if result.get("kind") != "stair_camp_trigger_false_positive_check":
    raise StairCampPreflightError("False-positive result kind drifted.")
  if result.get("domain") != expected_domain:
    raise StairCampPreflightError("False-positive result domain drifted.")
  if _exact_int(result.get("schema_version"), name="schema_version") != 1:
    raise StairCampPreflightError("False-positive result schema is unsupported.")
  if _finite(result.get("threshold_n"), name="threshold_n") != STAIR_TRIGGER_FORCE_N:
    raise StairCampPreflightError("False-positive trigger threshold drifted.")
  if _exact_int(result.get("window_steps"), name="window_steps", minimum=1) != STAIR_TRIGGER_WINDOW:
    raise StairCampPreflightError("False-positive trigger window drifted.")
  events = _exact_int(result.get("events"), name="events", minimum=1)
  false_positives = _exact_int(
    result.get("stair_mode_false_positives"),
    name="stair_mode_false_positives",
  )
  if false_positives > events:
    raise StairCampPreflightError("False-positive count exceeds events.")
  if result.get("completed") is not True:
    raise StairCampPreflightError("False-positive live check did not complete.")
  return {
    "domain": expected_domain,
    "events": events,
    "stair_mode_false_positives": false_positives,
    "passed": false_positives == 0,
  }


def finalize_preflight(
  c2_replay: object,
  flat_rolling: object,
  stage5_kick: object,
) -> dict[str, object]:
  replay = _mapping(c2_replay, name="C2 replay")
  if (
    replay.get("kind") != "stair_camp_c2_trigger_replay"
    or replay.get("passed") is not True
    or replay.get("classification") != PASS_CLASSIFICATION
    or replay.get("files_unchanged") is not True
    or _exact_int(replay.get("completed_pairs"), name="completed_pairs")
    != FROZEN_C2_TOTAL_PAIRS
    or _exact_int(replay.get("detections"), name="detections")
    != FROZEN_C2_TOTAL_PAIRS
    or _exact_int(replay.get("pre_impact_triggers"), name="pre_impact_triggers")
    != 0
  ):
    raise StairCampPreflightError("C2 trigger replay is not the registered pass.")
  flat = validate_live_false_positive_result(
    flat_rolling, expected_domain="camp_flat_rolling"
  )
  kick = validate_live_false_positive_result(
    stage5_kick, expected_domain="stage5_kick"
  )
  passed = bool(flat["passed"] and kick["passed"])
  result = {
    "schema_version": PREFLIGHT_SCHEMA_VERSION,
    "kind": "stair_camp_training_preflight",
    "classification": PASS_CLASSIFICATION if passed else STOP_CLASSIFICATION,
    "training_authorized": passed,
    "trigger_replay": {
      "completed_pairs": replay["completed_pairs"],
      "detections": replay["detections"],
      "pre_impact_triggers": replay["pre_impact_triggers"],
      "files_unchanged": replay["files_unchanged"],
      "sha256s_file_sha256": replay.get("sha256s_file_sha256"),
    },
    "false_positive_checks": [flat, kick],
  }
  _validate_json(result)
  return result


def _validate_json(value: object, *, path: str = "$") -> None:
  if value is None or isinstance(value, (str, bool, int)):
    return
  if isinstance(value, float):
    if not math.isfinite(value):
      raise StairCampPreflightError(f"{path} contains NaN or infinity.")
    return
  if isinstance(value, Mapping):
    for key, item in value.items():
      if not isinstance(key, str):
        raise StairCampPreflightError(f"{path} has a non-string key.")
      _validate_json(item, path=f"{path}.{key}")
    return
  if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
    for index, item in enumerate(value):
      _validate_json(item, path=f"{path}[{index}]")
    return
  raise StairCampPreflightError(f"{path} is not JSON-safe.")


def deterministic_json(payload: Mapping[str, object]) -> str:
  _validate_json(payload)
  return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"


def write_new_atomic(path: str | Path, payload: Mapping[str, object]) -> Path:
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  if destination.exists():
    raise FileExistsError(f"Refusing to overwrite {destination}.")
  temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
  try:
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
      stream.write(deterministic_json(payload))
      stream.flush()
      os.fsync(stream.fileno())
    os.link(temporary, destination)
  finally:
    temporary.unlink(missing_ok=True)
  return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  subparsers = parser.add_subparsers(dest="command", required=True)
  replay = subparsers.add_parser("replay-c2")
  replay.add_argument("--input-dir", type=Path, required=True)
  replay.add_argument("--output", type=Path, required=True)
  final = subparsers.add_parser("finalize")
  final.add_argument("--c2-replay", type=Path, required=True)
  final.add_argument("--flat-fp", type=Path, required=True)
  final.add_argument("--stage5-kick-fp", type=Path, required=True)
  final.add_argument("--output", type=Path, required=True)
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  if args.command == "replay-c2":
    result = replay_frozen_c2_trigger(args.input_dir)
  else:
    result = finalize_preflight(
      _strict_json(args.c2_replay),
      _strict_json(args.flat_fp),
      _strict_json(args.stage5_kick_fp),
    )
  write_new_atomic(args.output, result)
  print(result["classification"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
