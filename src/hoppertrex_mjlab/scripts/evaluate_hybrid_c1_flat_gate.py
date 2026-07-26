#!/usr/bin/env python3
"""Evaluate all 27 registered Q/R candidates through the C1 flat safety gate.

C1 requires a schedule artifact whose selection record proves every
registered Q/R candidate was evaluated on flat ground. This evaluator
closes that requirement in one machine-room run:

1. Verify the nine-node identification collection (SHA-256 per node,
   registered coordinates, schedule-v2 state definition, artifact bindings
   against the loaded runtime).
2. Offline-fit all 27 registered Q/R candidates at all nine nodes with the
   registered identification path; abort if any node of any candidate is
   not a rank-four, NRMSE-qualified, fallback-free LQR.
3. Drive a Stage3 play env (zero residual, flat ground) once per candidate
   by swapping the schedule gain grid in place, over the nine registered
   nodes at vx=0 plus +/-vx spot checks at the center and extreme corners
   - the same cell protocol as the frozen compensated requalification.
4. Gate each candidate: zero terminations, zero non-wheel contact, and the
   three registered performance caps derived as REGISTERED_FLOOR_MARGIN
   times the measured floors of the frozen compensated qualification
   artifact (never authored numbers). wheel_target_rate is a registered
   selection metric but has no frozen floor, so it ranks and never gates.
5. Write the selection record (the flat-gate evidence file) and the full
   per-cell evaluation detail, then self-check the record through
   validate_flat_gate_selection before reporting PASS.

The evaluator never builds the final schedule artifact: the formal node
fit at the selected Q/R and build_hybrid_controller_schedule run offline
on the development side, against this evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

PROJECT_PATH = Path(__file__).resolve().parents[1]
SRC_PATH = Path(__file__).resolve().parents[2]
REPOSITORY_PATH = Path(__file__).resolve().parents[3]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

try:
  import hoppertrex_mjlab.tasks as tasks  # noqa: E402,F401
  from hoppertrex_mjlab.tasks.hoppertrex_balance_task import (
    NON_WHEEL_GROUND_SENSOR_NAME,
    non_wheel_ground_contact,
  )
  from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
    hybrid_provenance_lines,
  )
except ImportError:
  import tasks  # noqa: E402,F401
  from tasks.hoppertrex_balance_task import (  # type: ignore[no-redef]
    NON_WHEEL_GROUND_SENSOR_NAME,
    non_wheel_ground_contact,
  )
  from tasks.hoppertrex_hybrid_task import (  # type: ignore[no-redef]
    hybrid_provenance_lines,
  )
from hoppertrex_mjlab.hybrid.controller_schedule import (  # noqa: E402
  AFFINE_SCHEDULE_STATE_DEFINITION,
  ControllerSchedule,
  REGISTERED_HEIGHT_NODES,
  SELECTION_METRICS,
  qr_candidate_grid,
  validate_flat_gate_selection,
)
from hoppertrex_mjlab.hybrid.identification import (  # noqa: E402
  CONTROLLER_STATE_NAMES,
  NOMINAL_WHEEL_RADIUS_M,
  anchor_gains_to_incumbent,
  identify_controller,
)
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

REGISTERED_PITCH_NODES = (-0.032, 0.0, 0.032)
REGISTERED_FLOOR_MARGIN = 1.5
EXPECTED_COLLECTION_GIT_SHA = "e54bd1a604b08b634821d88ce3a53a0f2fe66724"
EXPECTED_MJLAB_GIT_SHA = "43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6"
EXPECTED_COLLECTION_ZIP_SHA256 = (
  "364590b8d9f2f5c66fdaac2b3fa124ee914236e33f6fc47e31e75f64d53c72e2"
)
EXPECTED_COMPENSATED_SHA256 = (
  "c003192963b257c8d497ffd347be2cd60695c5ce8653932403709d8193c88e55"
)
EXPECTED_BINDINGS = {
  "controller_gain_hash": (
    "8fee25a0339dd1e99127cbed912941dc3ad8ef2030ce49a0d310d1563cb87d98"
  ),
  "velocity_calibration_hash": (
    "f62648b57bd17a3503bcbdbf58f349f91fcd8de8ef0cf04551c200401233ed01"
  ),
  "posture_artifact_hash": (
    "3b96fd3dae66ad781b5b875c74184db101c42da02c53dfcc40a5137a6b5de11a"
  ),
  "station_calibration_hash": (
    "c00e859b3093b4812d54799253accdaeb99171a2cf4028b08bc39e68eaaa7d8a"
  ),
}
EXPECTED_POSTURE_MAP_HASH = (
  "4289fb286c6a76a2aca2652d6bcc40acb1bf9c1f70b779a47ceff65c4dca3513"
)
EXPECTED_COLLECTION_PROTOCOL = {
  "height_nodes_m": list(REGISTERED_HEIGHT_NODES),
  "pitch_nodes_rad": list(REGISTERED_PITCH_NODES),
  "num_envs": 32,
  "steps": 2500,
  "warmup_steps": 250,
  "hold_steps": 5,
  "balance_amplitude": 0.35,
  "heldout_fraction": 0.20,
}
EXPECTED_ARRAY_SHAPES = {
  "states": (64000, 4),
  "inputs": (64000, 1),
  "next_states": (64000, 4),
  "heldout_states": (16000, 4),
  "heldout_inputs": (16000, 1),
  "heldout_next_states": (16000, 4),
}
REGISTERED_FLOORS = {
  "velocity_error_floor": 0.0069367592222988605,
  "p95_pitch_floor": 0.015581169165670872,
  "p99_pitch_rate_floor": 0.2190857082605362,
}
REGISTERED_CAPS = {
  "worst_velocity_error": 0.01040513883344829,
  "p95_pitch": 0.023371753748506308,
  "p99_pitch_rate": 0.3286285623908043,
}
FORMAL_TASK = "HopperTrex-Hybrid-v2-Stage3"
FORMAL_DEVICE = "cuda:0"
FORMAL_NUM_ENVS = 16
FORMAL_SETTLE_STEPS = 100
FORMAL_MEASURE_STEPS = 200
FORMAL_VX_CHECK = 0.05
FORMAL_SEED = 1
NODE_STEMS = tuple(
  f"node_h{h_index}_p{p_index}" for h_index in range(3) for p_index in range(3)
)
REQUIRED_ARRAYS = (
  "states",
  "inputs",
  "next_states",
  "heldout_states",
  "heldout_inputs",
  "heldout_next_states",
)
IDENTIFICATION_PD_GAIN = (8.0, 1.0, 3.0, 0.2)
NRMSE_LIMIT = 0.15
COMMAND_ERROR_GAIN_DIRECTION = np.asarray(
  (0.0, 0.0, 1.0, 1.0 / NOMINAL_WHEEL_RADIUS_M), dtype=np.float64
)
MINIMUM_INCUMBENT_COMMAND_GAIN_RATIO = 0.70


def _read_json(path: Path) -> dict[str, object]:
  """Read JSON emitted by either PowerShell 5.1 or normal UTF-8 tools."""

  payload = json.loads(path.read_text(encoding="utf-8-sig"))
  if not isinstance(payload, dict):
    raise TypeError(f"JSON root must be an object: {path}")
  return payload


def _is_number(value: object) -> bool:
  return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_equal(actual: object, expected: object, name: str) -> None:
  if actual != expected:
    raise ValueError(f"{name} mismatch: expected {expected!r}, got {actual!r}.")


def _prepare_output_dir(output_dir: Path) -> None:
  if output_dir.exists():
    if not output_dir.is_dir():
      raise ValueError(f"Output path is not a directory: {output_dir}")
    if any(output_dir.iterdir()):
      raise ValueError(f"Refusing to overwrite non-empty output: {output_dir}")
    return
  output_dir.mkdir(parents=True)


def _parse_sha256s(path: Path) -> dict[str, str]:
  entries: dict[str, str] = {}
  for line in path.read_text(encoding="ascii").splitlines():
    match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\]+)", line)
    if match is None:
      raise ValueError(f"Malformed SHA256SUMS line: {line!r}")
    digest, name = match.groups()
    if name in entries:
      raise ValueError(f"Duplicate SHA256SUMS entry: {name}")
    entries[name] = digest
  return entries


def _verify_zip_entries(
  zip_path: Path,
  nodes_dir: Path,
  expected_files: set[str],
) -> None:
  """Bind every extracted input byte to the already SHA-pinned ZIP."""

  expected_entries = {
    f"{nodes_dir.name}/{name}": name for name in expected_files
  }
  try:
    with zipfile.ZipFile(zip_path) as archive:
      entries: dict[str, zipfile.ZipInfo] = {}
      for info in archive.infolist():
        normalized = info.filename.replace("\\", "/")
        if info.is_dir() or normalized.endswith("/"):
          raise ValueError("Frozen nine-node ZIP contains a directory entry.")
        if normalized in entries:
          raise ValueError(f"Frozen nine-node ZIP has duplicate entry: {normalized}")
        entries[normalized] = info
      if set(entries) != set(expected_entries):
        raise ValueError("Frozen nine-node ZIP entry set mismatch.")
      for entry_name, local_name in expected_entries.items():
        if archive.read(entries[entry_name]) != (nodes_dir / local_name).read_bytes():
          raise ValueError(f"Frozen ZIP entry mismatch: {local_name}")
  except zipfile.BadZipFile as exc:
    raise ValueError("Frozen nine-node ZIP is invalid.") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--nodes-dir", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
    "--compensated-qualification",
    type=Path,
    required=True,
    help="Frozen compensated posture qualification JSON (floor source).",
  )
  parser.add_argument("--mjlab-git-sha", required=True)
  parser.add_argument("--task", default="HopperTrex-Hybrid-v2-Stage3")
  parser.add_argument(
    "--device",
    default="cuda:0" if torch.cuda.is_available() else "cpu",
  )
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--settle-steps", type=int, default=100)
  parser.add_argument("--measure-steps", type=int, default=200)
  parser.add_argument("--vx-check", type=float, default=0.05)
  return parser.parse_args(argv)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
  completed = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=REPOSITORY_PATH,
    check=True,
    capture_output=True,
    text=True,
  )
  return completed.stdout.strip()


def _require_full_sha(value: str, name: str) -> str:
  if re.fullmatch(r"[0-9a-f]{40}", value) is None:
    raise ValueError(f"{name} must be a full lowercase Git SHA.")
  return value


def load_verified_nodes(
  nodes_dir: Path,
  *,
  expected_zip_sha256: str = EXPECTED_COLLECTION_ZIP_SHA256,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
  """Verify the frozen ZIP, collection protocol, hashes, and node arrays."""

  nodes_dir = nodes_dir.resolve()
  if not nodes_dir.is_dir():
    raise ValueError(f"Nine-node collection directory is missing: {nodes_dir}")
  zip_path = nodes_dir.parent / f"{nodes_dir.name}.zip"
  if not zip_path.is_file():
    raise ValueError(f"Frozen nine-node ZIP is missing: {zip_path}")
  if _sha256(zip_path) != expected_zip_sha256:
    raise ValueError("Frozen nine-node ZIP SHA256 mismatch.")

  expected_files = {
    "protocol_note.json",
    *(
      f"{stem}.{suffix}"
      for stem in NODE_STEMS
      for suffix in ("json", "log", "npz")
    ),
  }
  archive_files = expected_files | {"SHA256SUMS.txt"}
  checksum_path = nodes_dir / "SHA256SUMS.txt"
  _verify_zip_entries(zip_path, nodes_dir, archive_files)
  checksums = _parse_sha256s(checksum_path)
  if set(checksums) != expected_files:
    raise ValueError("SHA256SUMS does not contain the exact 28 frozen files.")
  actual_files = {path.name for path in nodes_dir.iterdir() if path.is_file()}
  if actual_files != expected_files | {"SHA256SUMS.txt"}:
    raise ValueError("Nine-node directory contains missing or unexpected files.")
  for name, expected in checksums.items():
    if _sha256(nodes_dir / name) != expected:
      raise ValueError(f"SHA256SUMS mismatch: {name}")

  protocol_path = nodes_dir / "protocol_note.json"
  protocol = _read_json(protocol_path)
  _require_equal(
    protocol.get("kind"),
    "c1_scheduled_identification_collection",
    "collection kind",
  )
  _require_equal(
    protocol.get("git_sha"), EXPECTED_COLLECTION_GIT_SHA, "collection Git SHA"
  )
  _require_equal(
    protocol.get("mjlab_git_sha"),
    EXPECTED_MJLAB_GIT_SHA,
    "collection MjLab SHA",
  )
  _require_equal(protocol.get("seed"), 1, "collection seed")
  _require_equal(protocol.get("device"), "cuda:0", "collection device")
  _require_equal(
    protocol.get("protocol"),
    EXPECTED_COLLECTION_PROTOCOL,
    "collection protocol",
  )
  _require_equal(protocol.get("bindings"), EXPECTED_BINDINGS, "artifact bindings")
  for field, expected in (
    ("evidence_eligible", True),
    ("promotion_eligible", False),
    ("training_eligible", False),
    ("checkpoint", None),
    ("yaw_calibration_hash", None),
    (
      "next_step",
      "offline_node_fit_then_registered_27_candidate_flat_gate",
    ),
  ):
    _require_equal(protocol.get(field), expected, f"collection {field}")

  node_records = protocol.get("nodes")
  if not isinstance(node_records, list) or len(node_records) != 9:
    raise ValueError("Collection protocol must register exactly nine nodes.")
  if not all(isinstance(item, dict) for item in node_records):
    raise TypeError("Collection node records must be JSON objects.")
  records_by_stem = {str(item.get("stem")): item for item in node_records}
  if len(records_by_stem) != 9 or set(records_by_stem) != set(NODE_STEMS):
    raise ValueError("Collection protocol node stems do not match the grid.")

  nodes: dict[str, dict[str, object]] = {}
  for stem in NODE_STEMS:
    record = records_by_stem[stem]
    h_index = int(stem[6])
    p_index = int(stem[9])
    expected_height = REGISTERED_HEIGHT_NODES[h_index]
    expected_pitch = REGISTERED_PITCH_NODES[p_index]
    for field, expected in (
      ("height_m", expected_height),
      ("pitch_rad", expected_pitch),
      ("valid_sample_count", 80000),
      ("discarded_sample_count", 0),
    ):
      _require_equal(record.get(field), expected, f"{stem} record {field}")
    equilibrium = record.get("equilibrium_pitch_rad")
    if not _is_number(equilibrium) or not math.isfinite(float(equilibrium)):
      raise ValueError(f"{stem} record equilibrium pitch must be finite.")

    npz_path = nodes_dir / f"{stem}.npz"
    metadata_path = nodes_dir / f"{stem}.json"
    log_path = nodes_dir / f"{stem}.log"
    record_hashes = {
      "npz_sha256": (npz_path, f"{stem}.npz"),
      "metadata_sha256": (metadata_path, f"{stem}.json"),
      "log_sha256": (log_path, f"{stem}.log"),
    }
    for field, (path, checksum_name) in record_hashes.items():
      digest = record.get(field)
      if digest != checksums[checksum_name] or _sha256(path) != digest:
        raise ValueError(f"Node {field} mismatch: {stem}")

    metadata = _read_json(metadata_path)
    metadata_protocol = {
      "git_sha": EXPECTED_COLLECTION_GIT_SHA,
      "device": "cuda:0",
      "seed": 1,
      "num_envs": 32,
      "steps": 2500,
      "warmup_steps": 250,
      "hold_steps": 5,
      "balance_amplitude": 0.35,
      "heldout_fraction": 0.20,
      "height_command": expected_height,
      "pitch_command": expected_pitch,
      "state_definition_version": "hybrid_lqr_equilibrium_pitch_v2",
      "state_names": list(CONTROLLER_STATE_NAMES),
      "input_name": "actual_signed_balance_wheel_velocity_target",
      "wheel_radius": NOMINAL_WHEEL_RADIUS_M,
      "valid_sample_count": 80000,
      "discarded_sample_count": 0,
      "calibration_hash": EXPECTED_BINDINGS["velocity_calibration_hash"],
      "posture_artifact_hash": EXPECTED_BINDINGS["posture_artifact_hash"],
      "station_calibration_hash": EXPECTED_BINDINGS[
        "station_calibration_hash"
      ],
    }
    for field, expected in metadata_protocol.items():
      _require_equal(metadata.get(field), expected, f"{stem} metadata {field}")
    controller = metadata.get("controller")
    if not isinstance(controller, dict):
      raise TypeError(f"{stem} controller metadata must be an object.")
    _require_equal(controller.get("qualified"), True, f"{stem} controller qualified")
    _require_equal(
      controller.get("gain_hash"),
      EXPECTED_BINDINGS["controller_gain_hash"],
      f"{stem} controller gain hash",
    )
    if not math.isclose(
      float(metadata.get("equilibrium_pitch", math.nan)),
      float(equilibrium),
      rel_tol=0.0,
      abs_tol=1.0e-15,
    ):
      raise ValueError(f"{stem} equilibrium pitch does not match protocol.")

    with np.load(npz_path, allow_pickle=False) as data:
      missing = [name for name in REQUIRED_ARRAYS if name not in data]
      if missing:
        raise ValueError(f"{stem} missing arrays: {', '.join(missing)}")
      if set(data.files) != set(REQUIRED_ARRAYS):
        raise ValueError(f"{stem} contains unexpected arrays.")
      arrays = {name: np.asarray(data[name]) for name in REQUIRED_ARRAYS}
    for name, array in arrays.items():
      if array.shape != EXPECTED_ARRAY_SHAPES[name]:
        raise ValueError(
          f"{stem} {name} shape mismatch: {array.shape} "
          f"!= {EXPECTED_ARRAY_SHAPES[name]}"
        )
      if not np.issubdtype(array.dtype, np.number) or not np.all(
        np.isfinite(array)
      ):
        raise ValueError(f"{stem} {name} must contain finite numeric values.")
    nodes[stem] = {
      "arrays": arrays,
      "metadata": metadata,
      "npz_sha256": record["npz_sha256"],
      "metadata_sha256": record["metadata_sha256"],
      "log_sha256": record["log_sha256"],
    }
  return protocol, nodes


def fit_all_candidates(
  nodes: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
  """Fit every candidate at every node; abort on any disqualified node."""

  fitted: list[dict[str, object]] = []
  for index, candidate in enumerate(qr_candidate_grid()):
    gains = np.zeros((3, 3, 4), dtype=np.float64)
    node_facts: dict[str, dict[str, object]] = {}
    models = []
    raw_gains = []
    affine = all(
      node.get("metadata", {}).get("state_definition_version")
      == AFFINE_SCHEDULE_STATE_DEFINITION
      for node in nodes.values()
    )
    for stem in NODE_STEMS:
      arrays = nodes[stem]["arrays"]
      design = identify_controller(
        arrays["states"],
        arrays["inputs"],
        arrays["next_states"],
        heldout_states=arrays["heldout_states"],
        heldout_inputs=arrays["heldout_inputs"],
        heldout_next_states=arrays["heldout_next_states"],
        q_diag=tuple(candidate["q_diag"]),
        r_diag=tuple(candidate["r_diag"]),
        pd_gain=IDENTIFICATION_PD_GAIN,
        nrmse_limit=NRMSE_LIMIT,
      )
      if (
        design.controller_type != "lqr"
        or design.fallback_reasons
        or design.controllability_rank != 4
        or not math.isfinite(design.heldout_nrmse.maximum)
        or design.heldout_nrmse.maximum > NRMSE_LIMIT
      ):
        raise ValueError(
          f"Candidate {index} node {stem} is not a qualified LQR: "
          f"rank={design.controllability_rank}, "
          f"nrmse={design.heldout_nrmse.maximum}, "
          f"fallback={design.fallback_reasons}"
        )
      h_index = int(stem[6])
      p_index = int(stem[9])
      raw_gain = np.asarray(design.gain, dtype=np.float64).reshape(4)
      gains[h_index, p_index] = raw_gain
      if affine:
        models.append(design.model)
        raw_gains.append(raw_gain.reshape(1, 4))
      node_facts[stem] = {
        "controllability_rank": int(design.controllability_rank),
        "max_nrmse": float(design.heldout_nrmse.maximum),
        "gain": raw_gain.tolist(),
        "raw_gain": raw_gain.tolist(),
        "fallback_reasons": list(design.fallback_reasons),
      }
    anchor_alpha = 1.0
    if affine:
      incumbent_gains = {
        tuple(float(value) for value in node["metadata"]["controller"]["gain"])
        for node in nodes.values()
      }
      if len(incumbent_gains) != 1:
        raise ValueError("Affine nodes must share one incumbent feedback gain.")
      incumbent_gain = np.asarray(next(iter(incumbent_gains)), dtype=np.float64)
      anchored = anchor_gains_to_incumbent(
        models,
        np.asarray(raw_gains, dtype=np.float64),
        incumbent_gain,
        protected_gain_direction=COMMAND_ERROR_GAIN_DIRECTION,
        minimum_protected_gain_ratio=MINIMUM_INCUMBENT_COMMAND_GAIN_RATIO,
      )
      anchor_alpha = anchored.alpha
      for node_index, stem in enumerate(NODE_STEMS):
        h_index = int(stem[6])
        p_index = int(stem[9])
        deployed = anchored.gains[node_index].reshape(4)
        command_gain_ratio = float(
          abs(deployed @ COMMAND_ERROR_GAIN_DIRECTION)
          / abs(incumbent_gain @ COMMAND_ERROR_GAIN_DIRECTION)
        )
        gains[h_index, p_index] = deployed
        node_facts[stem].update(
          {
            "gain": deployed.tolist(),
            "anchor_alpha": anchor_alpha,
            "raw_closed_loop_spectral_radius": anchored.raw_spectral_radius[
              node_index
            ],
            "incumbent_closed_loop_spectral_radius": (
              anchored.incumbent_spectral_radius[node_index]
            ),
            "deployed_closed_loop_spectral_radius": (
              anchored.deployed_spectral_radius[node_index]
            ),
            "command_gain_ratio": command_gain_ratio,
            "minimum_command_gain_ratio": (
              MINIMUM_INCUMBENT_COMMAND_GAIN_RATIO
            ),
          }
        )
    fitted.append(
      {
        "index": index,
        "q_diag": [float(value) for value in candidate["q_diag"]],
        "r_diag": [float(value) for value in candidate["r_diag"]],
        "gains": gains,
        "anchor_alpha": anchor_alpha,
        "node_facts": node_facts,
      }
    )
  if len(fitted) != 27:
    raise ValueError("Registered Q/R grid did not produce exactly 27 candidates.")
  return fitted


def fit_qualification_summary(
  fitted: list[dict[str, object]],
) -> dict[str, int | float]:
  facts = [
    fact
    for candidate in fitted
    for fact in candidate["node_facts"].values()
  ]
  if len(fitted) != 27 or len(facts) != 243:
    raise ValueError("Flat gate requires exactly 27 candidates and 243 node fits.")
  summary: dict[str, int | float] = {
    "candidate_count": len(fitted),
    "node_fit_count": len(facts),
    "minimum_controllability_rank": min(
      int(fact["controllability_rank"]) for fact in facts
    ),
    "maximum_heldout_nrmse": max(float(fact["max_nrmse"]) for fact in facts),
    "fallback_count": sum(
      bool(fact["fallback_reasons"]) for fact in facts
    ),
  }
  if (
    summary["minimum_controllability_rank"] != 4
    or float(summary["maximum_heldout_nrmse"]) > NRMSE_LIMIT
    or summary["fallback_count"] != 0
  ):
    raise ValueError(f"Node-fit qualification failed: {summary}")
  return summary


def load_registered_floors(
  compensated_path: Path,
  *,
  expected_controller_gain_hash: str | None,
  expected_file_sha256: str = EXPECTED_COMPENSATED_SHA256,
) -> dict[str, float]:
  """Derive the flat-gate caps from the frozen compensated floors."""

  if _sha256(compensated_path) != expected_file_sha256:
    raise ValueError("Frozen compensated qualification SHA256 mismatch.")
  payload = _read_json(compensated_path)
  if payload.get("kind") != "posture_balance_qualification":
    raise ValueError("Floor source is not a posture balance qualification.")
  if not all(
    payload.get(field) is True
    for field in (
      "controller_qualified",
      "posture_map_qualified",
      "station_calibration_qualified",
    )
  ):
    raise ValueError("Floor source is not the compensated qualification.")
  if (
    expected_controller_gain_hash is not None
    and payload.get("controller_gain_hash") != expected_controller_gain_hash
  ):
    raise ValueError("Floor source controller binding mismatch.")
  expected_floor_bindings = {
    "calibration_hash": EXPECTED_BINDINGS["velocity_calibration_hash"],
    "posture_artifact_hash": EXPECTED_BINDINGS["posture_artifact_hash"],
    "posture_map_hash": EXPECTED_POSTURE_MAP_HASH,
    "station_calibration_hash": EXPECTED_BINDINGS["station_calibration_hash"],
  }
  for field, expected in expected_floor_bindings.items():
    _require_equal(payload.get(field), expected, f"floor source {field}")
  _require_equal(
    payload.get("source_probe"),
    {
      "device": "cuda:0",
      "git_sha": "c2d1ffe224549462a7858a082772a400e74b6f86",
      "height_points": 3,
      "measure_steps": 200,
      "num_envs": 16,
      "pitch_points": 3,
      "settle_steps": 100,
      "task": FORMAL_TASK,
      "vx_check": FORMAL_VX_CHECK,
    },
    "floor source probe",
  )
  grid_cells = payload.get("grid_cells")
  vx_cells = payload.get("vx_checks")
  if not isinstance(grid_cells, list) or not isinstance(vx_cells, list):
    raise TypeError("Floor source cells must be lists.")
  cells = grid_cells + vx_cells
  expected_cells = evaluation_cells(FORMAL_VX_CHECK)
  if len(cells) != len(expected_cells):
    raise ValueError("Floor source must contain the registered 15 cells.")
  for cell, (height, pitch, vx) in zip(cells, expected_cells, strict=True):
    if not isinstance(cell, dict):
      raise TypeError("Floor source cells must be JSON objects.")
    for field, expected in (
      ("target_height", height),
      ("target_pitch", pitch),
      ("vx_command", vx),
    ):
      if not math.isclose(
        float(cell.get(field, math.nan)),
        expected,
        rel_tol=0.0,
        abs_tol=1.0e-12,
      ):
        raise ValueError(f"Floor source cell order mismatch for {field}.")
    if (
      float(cell["terminated_events"]) != 0.0
      or float(cell["non_wheel_contact_rate"]) != 0.0
    ):
      raise ValueError("Floor source contains an unsafe cell.")
  floors = {
    "velocity_error_floor": max(
      abs(float(cell["mean_actual_lin_x"]) - float(cell["vx_command"]))
      for cell in cells
    ),
    "p95_pitch_floor": max(
      float(cell["pitch_error_abs_p95"]) for cell in cells
    ),
    "p99_pitch_rate_floor": max(
      float(cell["pitch_rate_abs_p99"]) for cell in cells
    ),
  }
  for name, value in floors.items():
    if not math.isfinite(value) or value <= 0.0:
      raise ValueError(f"Floor {name} must be finite and positive.")
    if not math.isclose(
      value, REGISTERED_FLOORS[name], rel_tol=0.0, abs_tol=1.0e-12
    ):
      raise ValueError(f"Frozen floor drifted from preregistration: {name}")
  return floors


def registered_caps(floors: dict[str, float]) -> dict[str, float]:
  caps = {
    "worst_velocity_error": (
      REGISTERED_FLOOR_MARGIN * floors["velocity_error_floor"]
    ),
    "p95_pitch": REGISTERED_FLOOR_MARGIN * floors["p95_pitch_floor"],
    "p99_pitch_rate": (
      REGISTERED_FLOOR_MARGIN * floors["p99_pitch_rate_floor"]
    ),
  }
  for name, value in caps.items():
    if not math.isclose(
      value, REGISTERED_CAPS[name], rel_tol=0.0, abs_tol=1.0e-12
    ):
      raise ValueError(f"Derived cap drifted from preregistration: {name}")
  return caps


def candidate_schedule(
  gains: np.ndarray,
  equilibrium: np.ndarray,
  q_diag: list[float],
  r_diag: list[float],
  equilibrium_input: np.ndarray | None = None,
) -> ControllerSchedule:
  """Directly construct an unqualified candidate schedule for evaluation.

  parse_controller_schedule intentionally rejects schedules without a
  flat-gate selection record; the gate evaluator is the one component
  that must run candidates before that record exists, so it constructs
  the runtime dataclass itself and never writes these to disk.
  """

  equilibrium_array = np.asarray(equilibrium, dtype=np.float64)
  if equilibrium_array.shape == (3, 3):
    equilibrium_array = np.stack(
      (
        equilibrium_array,
        np.zeros_like(equilibrium_array),
        np.zeros_like(equilibrium_array),
        np.zeros_like(equilibrium_array),
      ),
      axis=2,
    )
  if equilibrium_array.shape != (3, 3, 4):
    raise ValueError("Candidate equilibrium must have shape (3, 3) or (3, 3, 4).")
  input_array = (
    np.zeros((3, 3), dtype=np.float64)
    if equilibrium_input is None
    else np.asarray(equilibrium_input, dtype=np.float64)
  )
  if input_array.shape != (3, 3):
    raise ValueError("Candidate equilibrium input must have shape (3, 3).")
  return ControllerSchedule(
    height_nodes=REGISTERED_HEIGHT_NODES,
    pitch_nodes=REGISTERED_PITCH_NODES,
    gains=np.asarray(gains, dtype=np.float64),
    equilibrium_state=equilibrium_array,
    equilibrium_input=input_array,
    schedule_hash="candidate-under-flat-gate-evaluation",
    q_diag=tuple(q_diag),  # type: ignore[arg-type]
    r_diag=tuple(r_diag),  # type: ignore[arg-type]
    anchor_alpha=1.0,
    bindings={},
    source="flat-gate-candidate",
  )


def _force_commands(
  env: ManagerBasedRlEnv,
  *,
  vx: float,
  height: float,
  pitch: float,
) -> None:
  twist = env.command_manager.get_term("twist")
  for attribute in ("vel_command_b", "vel_command_w"):
    command = getattr(twist, attribute)
    command[:, :] = 0.0
    command[:, 0] = vx
  for attribute in (
    "is_standing_env",
    "is_heading_env",
    "is_world_env",
    "is_forward_env",
  ):
    value = getattr(twist, attribute, None)
    if value is not None:
      value[:] = False
  posture = env.command_manager.get_term("posture")
  command = getattr(posture, "_command", None)
  if command is None:
    raise AttributeError("Posture command term does not expose _command.")
  target = getattr(posture, "_target", None)
  if target is not None:
    target[:, 0] = height
    target[:, 1] = pitch
  command[:, 0] = height
  command[:, 1] = pitch


def _pitch(robot_data: object) -> torch.Tensor:
  gravity = robot_data.projected_gravity_b
  return torch.atan2(
    gravity[:, 0],
    torch.clamp(-gravity[:, 2], min=1.0e-6),
  )


def run_cell(
  env: ManagerBasedRlEnv,
  *,
  height: float,
  pitch: float,
  vx: float,
  settle_steps: int,
  measure_steps: int,
) -> dict[str, float]:
  """Hold one command cell under zero residual and measure it."""

  env.reset()
  actions = torch.zeros(
    (env.num_envs, env.action_space.shape[-1]),
    device=env.device,
  )
  robot = env.scene["robot"]
  action_term = env.action_manager.get_term("hybrid_wheel_leg")
  heights: list[torch.Tensor] = []
  pitches: list[torch.Tensor] = []
  pitch_rates: list[torch.Tensor] = []
  lin_x: list[torch.Tensor] = []
  safety_contacts: list[torch.Tensor] = []
  wheel_rate_sq: list[torch.Tensor] = []
  previous_wheel_targets: torch.Tensor | None = None
  terminated_total = 0
  for step in range(settle_steps + measure_steps):
    _force_commands(env, vx=vx, height=height, pitch=pitch)
    _obs, _rewards, terminated, _time_outs, _extras = env.step(actions)
    _force_commands(env, vx=vx, height=height, pitch=pitch)
    terminated_total += int(terminated.sum().item())
    safety_contacts.append(
      non_wheel_ground_contact(env, NON_WHEEL_GROUND_SENSOR_NAME)
      .detach()
      .cpu()
    )
    wheel_targets = action_term.wheel_targets.detach().clone()
    if step < settle_steps:
      previous_wheel_targets = wheel_targets
      continue
    data = robot.data
    heights.append(data.root_link_pos_w[:, 2].detach().cpu())
    pitches.append(_pitch(data).detach().cpu())
    pitch_rates.append(data.root_link_ang_vel_b[:, 1].abs().detach().cpu())
    lin_x.append(data.root_link_lin_vel_b[:, 0].detach().cpu())
    if previous_wheel_targets is not None:
      wheel_rate_sq.append(
        torch.sum(
          torch.square(wheel_targets - previous_wheel_targets), dim=1
        ).cpu()
      )
    previous_wheel_targets = wheel_targets

  height_error = torch.stack(heights) - height
  pitch_error = torch.stack(pitches) - pitch
  pitch_rate_abs = torch.stack(pitch_rates)
  lin = torch.stack(lin_x)
  contact = torch.stack(safety_contacts).float()
  wheel_rate = torch.stack(wheel_rate_sq)
  return {
    "target_height": float(height),
    "target_pitch": float(pitch),
    "vx_command": float(vx),
    "height_rmse": float(
      torch.sqrt(torch.mean(torch.square(height_error))).item()
    ),
    "pitch_rmse": float(
      torch.sqrt(torch.mean(torch.square(pitch_error))).item()
    ),
    "pitch_error_abs_p95": float(
      torch.quantile(pitch_error.abs(), 0.95).item()
    ),
    "pitch_rate_abs_p99": float(torch.quantile(pitch_rate_abs, 0.99).item()),
    "mean_actual_lin_x": float(lin.mean().item()),
    "velocity_error_abs": float(abs(lin.mean().item() - vx)),
    "wheel_target_rate_rms": float(
      torch.sqrt(torch.mean(wheel_rate)).item()
    ),
    "non_wheel_contact_rate": float(contact.mean().item()),
    "terminated_events": float(terminated_total),
    "safety_window_steps": float(settle_steps + measure_steps),
  }


def evaluation_cells(vx_check: float) -> list[tuple[float, float, float]]:
  """Nine registered nodes at vx=0 plus center/corner +/-vx spot checks."""

  cells = [
    (height, pitch, 0.0)
    for height in REGISTERED_HEIGHT_NODES
    for pitch in REGISTERED_PITCH_NODES
  ]
  spots = (
    (REGISTERED_HEIGHT_NODES[1], REGISTERED_PITCH_NODES[1]),
    (REGISTERED_HEIGHT_NODES[0], REGISTERED_PITCH_NODES[0]),
    (REGISTERED_HEIGHT_NODES[2], REGISTERED_PITCH_NODES[2]),
  )
  for height, pitch in spots:
    for sign in (1.0, -1.0):
      cells.append((height, pitch, sign * vx_check))
  return cells


def aggregate_candidate(
  cells: list[dict[str, float]],
  caps: dict[str, float],
) -> dict[str, object]:
  """Reduce one candidate's cells to the registered metrics and verdict."""

  safety_clean = all(
    float(cell["terminated_events"]) == 0.0
    and float(cell["non_wheel_contact_rate"]) == 0.0
    for cell in cells
  )
  metrics = {
    "worst_velocity_error": max(
      float(cell["velocity_error_abs"]) for cell in cells
    ),
    "p95_pitch": max(float(cell["pitch_error_abs_p95"]) for cell in cells),
    "p99_pitch_rate": max(
      float(cell["pitch_rate_abs_p99"]) for cell in cells
    ),
    "wheel_target_rate": max(
      float(cell["wheel_target_rate_rms"]) for cell in cells
    ),
  }
  passed = (
    safety_clean
    and metrics["worst_velocity_error"] <= caps["worst_velocity_error"]
    and metrics["p95_pitch"] <= caps["p95_pitch"]
    and metrics["p99_pitch_rate"] <= caps["p99_pitch_rate"]
  )
  return {**metrics, "safety_clean": safety_clean, "flat_gate_passed": passed}


def select_best(evaluated: list[dict[str, object]]) -> int:
  """The registered rule: lexicographic best over SELECTION_METRICS."""

  passed = [
    index
    for index, candidate in enumerate(evaluated)
    if candidate["flat_gate_passed"]
  ]
  if not passed:
    raise ValueError("No Q/R candidate passed the flat safety gate.")
  return min(
    passed,
    key=lambda index: tuple(
      float(evaluated[index][metric]) for metric in SELECTION_METRICS
    ),
  )


def _validate_formal_args(args: argparse.Namespace) -> None:
  expected = {
    "task": FORMAL_TASK,
    "device": FORMAL_DEVICE,
    "num_envs": FORMAL_NUM_ENVS,
    "settle_steps": FORMAL_SETTLE_STEPS,
    "measure_steps": FORMAL_MEASURE_STEPS,
    "vx_check": FORMAL_VX_CHECK,
  }
  for name, value in expected.items():
    _require_equal(getattr(args, name), value, f"formal argument --{name}")
  _require_equal(
    args.mjlab_git_sha, EXPECTED_MJLAB_GIT_SHA, "formal MjLab Git SHA"
  )


def write_evaluation_outputs(
  output_dir: Path,
  *,
  git_sha: str,
  mjlab_git_sha: str,
  collection_protocol: dict[str, object],
  bindings: dict[str, object],
  floors: dict[str, float],
  caps: dict[str, float],
  run_protocol: dict[str, object],
  fit_summary: dict[str, int | float],
  evaluated: list[dict[str, object]],
  detail: list[dict[str, object]],
) -> dict[str, object]:
  """Write complete evidence for either selected or all-failed outcomes."""

  if len(evaluated) != 27 or len(detail) != 27:
    raise ValueError("Output requires 27 completed candidate evaluations.")
  if any(len(candidate.get("cells", ())) != 15 for candidate in detail):
    raise ValueError("Output requires all 15 cells for every candidate.")
  passed_count = sum(
    1 for candidate in evaluated if candidate["flat_gate_passed"]
  )
  selected_index = select_best(evaluated) if passed_count else None
  common = {
    "git_sha": git_sha,
    "mjlab_git_sha": mjlab_git_sha,
    "collection_git_sha": collection_protocol.get("git_sha"),
    "collection_zip_sha256": EXPECTED_COLLECTION_ZIP_SHA256,
    "compensated_qualification_sha256": EXPECTED_COMPENSATED_SHA256,
    "bindings": dict(bindings),
    "floors": floors,
    "floor_margin": REGISTERED_FLOOR_MARGIN,
    "caps": caps,
    "evidence_eligible": True,
    "promotion_eligible": False,
    "training_eligible": False,
    "checkpoint": None,
    "yaw_calibration_hash": None,
  }
  detail_payload = {
    "schema_version": 1,
    "kind": "c1_flat_gate_evaluation_detail",
    **common,
    "protocol": run_protocol,
    "fit_qualification": fit_summary,
    "candidates": detail,
    "passed_candidate_count": passed_count,
    "selected_candidate_index": selected_index,
    "next_step": (
      "DOWNLOAD_FOR_OFFLINE_SCHEDULE_BUILD" if passed_count else "STOP"
    ),
  }
  detail_path = output_dir / "flat_gate_evaluation_detail.json"
  detail_path.write_text(
    json.dumps(
      detail_payload, indent=2, sort_keys=True, allow_nan=False
    )
    + "\n",
    encoding="utf-8",
  )

  selection_path = output_dir / "flat_gate_selection.json"
  selection_sha256: str | None = None
  if selected_index is not None:
    selected = evaluated[selected_index]
    selection = {
      "schema_version": 1,
      "kind": "c1_flat_gate_selection",
      "status": "flat_gate_selected",
      **common,
      "evaluated_candidates": evaluated,
      "passed_candidate_count": passed_count,
      "selected_candidate_index": selected_index,
      "evaluation_detail_path": detail_path.name,
      "evaluation_detail_sha256": _sha256(detail_path),
      "next_step": "DOWNLOAD_FOR_OFFLINE_SCHEDULE_BUILD",
    }
    selection_path.write_text(
      json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + "\n",
      encoding="utf-8",
    )
    selection_sha256 = _sha256(selection_path)
    validate_flat_gate_selection(
      {**selection, "evaluation_artifact_sha256": selection_sha256},
      selected_q_diag=tuple(float(v) for v in selected["q_diag"]),
      selected_r_diag=tuple(float(v) for v in selected["r_diag"]),
    )

  adjudication = {
    "schema_version": 1,
    "kind": "c1_flat_gate_adjudication",
    **common,
    "classification": (
      "C1_FLAT_GATE_SELECTED"
      if selected_index is not None
      else "NO_QR_CANDIDATE_PASSED_FLAT_GATE"
    ),
    "completed_candidate_count": len(evaluated),
    "completed_node_fit_count": int(fit_summary["node_fit_count"]),
    "passed_candidate_count": passed_count,
    "selected_candidate_index": selected_index,
    "selected_q_diag": (
      evaluated[selected_index]["q_diag"] if selected_index is not None else None
    ),
    "selected_r_diag": (
      evaluated[selected_index]["r_diag"] if selected_index is not None else None
    ),
    "evaluation_detail_path": detail_path.name,
    "evaluation_detail_sha256": _sha256(detail_path),
    "selection_path": selection_path.name if selected_index is not None else None,
    "selection_sha256": selection_sha256,
    "next_step": (
      "DOWNLOAD_FOR_OFFLINE_SCHEDULE_BUILD"
      if selected_index is not None
      else "STOP"
    ),
  }
  adjudication_path = output_dir / "flat_gate_adjudication.json"
  adjudication_path.write_text(
    json.dumps(adjudication, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
  )
  return adjudication


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  _validate_formal_args(args)
  mjlab_git_sha = _require_full_sha(args.mjlab_git_sha, "--mjlab-git-sha")
  git_sha = _require_full_sha(_git_sha(), "repository HEAD")
  nodes_dir = args.nodes_dir.resolve()
  output_dir = args.output_dir.resolve()
  _prepare_output_dir(output_dir)
  if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise RuntimeError("Formal C1 flat gate requires CUDA device 0.")

  protocol, nodes = load_verified_nodes(nodes_dir)
  print(f"[gate] verified nine-node collection at {nodes_dir}")

  fitted = fit_all_candidates(nodes)
  fit_summary = fit_qualification_summary(fitted)
  print("[gate] all 27 candidates produced nine qualified LQR nodes")

  equilibrium = np.zeros((3, 3), dtype=np.float64)
  for stem in NODE_STEMS:
    metadata = nodes[stem]["metadata"]
    equilibrium[int(stem[6]), int(stem[9])] = float(
      metadata["equilibrium_pitch"]
    )

  cfg = load_env_cfg(args.task, play=True)
  cfg.seed = FORMAL_SEED
  cfg.scene.num_envs = args.num_envs
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.num_envs = args.num_envs
  action_cfg = cfg.actions["hybrid_wheel_leg"]
  for line in hybrid_provenance_lines(cfg):
    print(line)
  bindings = protocol["bindings"]
  runtime_bindings = {
    "controller_gain_hash": action_cfg.controller_gain_hash,
    "velocity_calibration_hash": action_cfg.calibration_hash,
    "posture_artifact_hash": getattr(
      action_cfg, "posture_artifact_hash", None
    ),
    "station_calibration_hash": getattr(
      action_cfg, "station_calibration_hash", None
    ),
  }
  for key, runtime_value in runtime_bindings.items():
    if bindings.get(key) != runtime_value:
      raise ValueError(
        f"Collection binding {key} does not match the loaded runtime."
      )
  if not (
    action_cfg.controller_qualified
    and action_cfg.posture_map_qualified
    and getattr(action_cfg, "station_calibration_qualified", False)
  ):
    raise ValueError("Flat gate requires the fully qualified classical stack.")

  floors = load_registered_floors(
    args.compensated_qualification.resolve(),
    expected_controller_gain_hash=action_cfg.controller_gain_hash,
  )
  caps = registered_caps(floors)
  print(f"[gate] floors={floors}")
  print(f"[gate] caps(margin {REGISTERED_FLOOR_MARGIN})={caps}")

  action_cfg.controller_schedule = candidate_schedule(
    fitted[0]["gains"],
    equilibrium,
    fitted[0]["q_diag"],
    fitted[0]["r_diag"],
  )
  cells = evaluation_cells(args.vx_check)
  evaluated: list[dict[str, object]] = []
  detail: list[dict[str, object]] = []
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    action_term = env.action_manager.get_term("hybrid_wheel_leg")
    if not getattr(action_term, "_schedule_enabled", False):
      raise RuntimeError("Action term did not enter schedule mode.")
    for candidate in fitted:
      action_term._schedule_gains.copy_(
        torch.tensor(
          candidate["gains"],
          device=action_term._schedule_gains.device,
          dtype=action_term._schedule_gains.dtype,
        )
      )
      candidate_cells: list[dict[str, float]] = []
      for height, pitch, vx in cells:
        candidate_cells.append(
          run_cell(
            env,
            height=height,
            pitch=pitch,
            vx=vx,
            settle_steps=args.settle_steps,
            measure_steps=args.measure_steps,
          )
        )
      verdict = aggregate_candidate(candidate_cells, caps)
      evaluated.append(
        {
          "index": candidate["index"],
          "q_diag": candidate["q_diag"],
          "r_diag": candidate["r_diag"],
          "anchor_alpha": candidate["anchor_alpha"],
          **{metric: verdict[metric] for metric in SELECTION_METRICS},
          "flat_gate_passed": verdict["flat_gate_passed"],
        }
      )
      detail.append(
        {
          "index": candidate["index"],
          "q_diag": candidate["q_diag"],
          "r_diag": candidate["r_diag"],
          "anchor_alpha": candidate["anchor_alpha"],
          "node_facts": candidate["node_facts"],
          "cells": candidate_cells,
          **verdict,
        }
      )
      print(
        f"[gate] candidate {candidate['index']:>2} "
        f"pass={verdict['flat_gate_passed']} "
        f"vel={verdict['worst_velocity_error']:.5f} "
        f"p95={verdict['p95_pitch']:.5f} "
        f"p99={verdict['p99_pitch_rate']:.5f} "
        f"rate={verdict['wheel_target_rate']:.5f}"
      )
  finally:
    env.close()

  run_protocol = {
    "task": args.task,
    "device": args.device,
    "seed": FORMAL_SEED,
    "num_envs": args.num_envs,
    "settle_steps": args.settle_steps,
    "measure_steps": args.measure_steps,
    "vx_check": float(args.vx_check),
    "cells_per_candidate": len(cells),
    "cell_order": [
      {"height_m": height, "pitch_rad": pitch, "vx_m_s": vx}
      for height, pitch, vx in cells
    ],
    "candidate_order": list(range(27)),
    "environment_process_count": 1,
    "environment_reused_across_candidates": True,
    "reset_before_each_cell": True,
    "wheel_target_rate_unit": "rms_delta_per_control_tick",
    "scope": "qr_selection_screen_not_cstar_formal_validation",
  }
  adjudication = write_evaluation_outputs(
    output_dir,
    git_sha=git_sha,
    mjlab_git_sha=mjlab_git_sha,
    collection_protocol=protocol,
    bindings=bindings,
    floors=floors,
    caps=caps,
    run_protocol=run_protocol,
    fit_summary=fit_summary,
    evaluated=evaluated,
    detail=detail,
  )
  print(
    f"[gate] passed {adjudication['passed_candidate_count']}/27 candidates"
  )
  print(f"[gate] classification={adjudication['classification']}")
  print(
    f"[gate] selected index {adjudication['selected_candidate_index']}"
  )
  print(f"[gate] output written: {output_dir}")


if __name__ == "__main__":
  main()
