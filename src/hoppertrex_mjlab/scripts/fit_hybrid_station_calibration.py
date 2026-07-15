#!/usr/bin/env python3
"""Fit the Stage 3.0 station-keeping calibration from a posture probe JSON.

Input is the qualification JSON written by ``probe_hybrid_posture_balance
--fit-output`` on an UNCOMPENSATED run (zero residual, no station artifact
active): its vx=0 grid cells measure the steady-state drift per commanded
pitch. The probe established that the drift is affine in the commanded
pitch and independent of the commanded height, so this fitter averages the
drift across heights per pitch level (refusing if the spread contradicts
height independence) and stores the resulting (pitch, drift) breakpoints
with bindings to both the controller gain hash and the posture map hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_PATH = Path(__file__).resolve().parents[1]
SRC_PATH = Path(__file__).resolve().parents[2]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

try:
  from hoppertrex_mjlab.hybrid.station_calibration import (
    station_calibration_artifact,
  )
except ImportError:
  from hybrid.station_calibration import (  # type: ignore[no-redef]
    station_calibration_artifact,
  )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input",
    type=Path,
    required=True,
    help="Uncompensated posture_balance_qualification JSON.",
  )
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--max-height-spread",
    type=float,
    default=0.01,
    help=(
      "Refuse to average if measured drift varies more than this across "
      "heights at one pitch level (m/s); the probe measured <0.001."
    ),
  )
  return parser.parse_args(argv)


def _validated_qualification(payload: object) -> dict[str, object]:
  if not isinstance(payload, dict):
    raise ValueError("Qualification input must be a JSON object.")
  if payload.get("kind") != "posture_balance_qualification":
    raise ValueError("Input is not a posture_balance_qualification JSON.")
  if payload.get("schema_version") != 1:
    raise ValueError("Qualification schema_version must be 1.")
  if payload.get("controller_qualified") is not True:
    raise ValueError(
      "Station fitting requires a qualification run on the qualified LQR."
    )
  if payload.get("posture_map_qualified") is not True:
    raise ValueError(
      "Station fitting requires a qualification run on a qualified posture map."
    )
  if payload.get("station_calibration_qualified"):
    raise ValueError(
      "Input qualification already ran with station compensation active; "
      "fit from an uncompensated probe instead."
    )
  for key in ("controller_gain_hash", "posture_map_hash"):
    value = payload.get(key)
    if not isinstance(value, str) or len(value) != 64:
      raise ValueError(f"Qualification must record a 64-character {key}.")
  cells = payload.get("grid_cells")
  if not isinstance(cells, list) or not cells:
    raise ValueError("Qualification must contain measured grid cells.")
  return payload


def fit_station_breakpoints(
  cells: list[dict[str, float]],
  *,
  max_height_spread: float,
) -> list[list[float]]:
  """Average vx=0 drift across heights per pitch level."""

  drift_by_pitch: dict[float, list[float]] = {}
  for cell in cells:
    if float(cell.get("vx_command", 0.0)) != 0.0:
      continue
    if float(cell.get("terminated_events", 0.0)) != 0.0:
      raise ValueError(
        "Station fitting requires termination-free grid cells; the drift "
        "of a falling posture is not a calibration input."
      )
    pitch = float(cell["target_pitch"])
    drift_by_pitch.setdefault(pitch, []).append(
      float(cell["mean_actual_lin_x"])
    )
  if len(drift_by_pitch) < 3:
    raise ValueError(
      "Station fitting requires at least three distinct pitch levels."
    )
  breakpoints: list[list[float]] = []
  for pitch in sorted(drift_by_pitch):
    drifts = drift_by_pitch[pitch]
    spread = max(drifts) - min(drifts)
    if spread > max_height_spread:
      raise ValueError(
        f"Drift at pitch {pitch:+.4f} varies {spread:.4f} m/s across "
        "heights; the height-independence assumption does not hold, so an "
        "averaged pitch-only map would hide a real dependence."
      )
    breakpoints.append([pitch, sum(drifts) / len(drifts)])
  return breakpoints


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  input_path = args.input.resolve()
  raw = input_path.read_bytes()
  payload = _validated_qualification(json.loads(raw.decode("utf-8")))
  breakpoints = fit_station_breakpoints(
    payload["grid_cells"],
    max_height_spread=args.max_height_spread,
  )
  source_probe = {
    "kind": "posture_balance_qualification",
    "path": str(input_path),
    "input_sha256": hashlib.sha256(raw).hexdigest(),
    "probe": dict(payload.get("source_probe") or {}),
  }
  artifact = station_calibration_artifact(
    controller_gain_hash=str(payload["controller_gain_hash"]),
    posture_map_hash=str(payload["posture_map_hash"]),
    breakpoints=breakpoints,
    source_probe=source_probe,
  )
  output_path = args.output.resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(
    json.dumps(artifact, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  print(f"Wrote station-keeping calibration: {output_path}")
  print(f"station_calibration_hash: {artifact['station_calibration_hash']}")
  for pitch, drift in artifact["breakpoints"]:
    print(f"  pitch {pitch:+.4f} -> drift {drift:+.4f} m/s")


if __name__ == "__main__":
  main()
