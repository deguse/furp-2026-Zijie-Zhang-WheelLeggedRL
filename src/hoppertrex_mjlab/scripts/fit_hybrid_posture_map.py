"""Fit the Hybrid v2 two-leg posture map from static sweep NPZ data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hoppertrex_mjlab.hybrid.posture import (
  fit_posture_map,
  posture_map_to_dict,
  select_feasible_samples,
  training_envelope,
)


REQUIRED_ARRAYS = (
  "heights",
  "pitches",
  "joint_positions",
  "non_wheel_contact",
  "joint_lower",
  "joint_upper",
  "actuator_load_fraction",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, required=True, help="Input sweep NPZ.")
  parser.add_argument("--output", type=Path, required=True, help="Output JSON.")
  parser.add_argument("--joint-margin", type=float, default=0.10)
  parser.add_argument("--load-limit", type=float, default=0.80)
  parser.add_argument("--inward-fraction", type=float, default=0.10)
  parser.add_argument("--pitch-limit", type=float, default=0.08)
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  input_path = args.input.resolve()
  with np.load(input_path, allow_pickle=False) as data:
    missing = [name for name in REQUIRED_ARRAYS if name not in data]
    if missing:
      raise ValueError(f"Input NPZ is missing arrays: {', '.join(missing)}")
    arrays = {name: data[name] for name in REQUIRED_ARRAYS}

  feasible = select_feasible_samples(
    non_wheel_contact=arrays["non_wheel_contact"],
    joint_positions=arrays["joint_positions"],
    joint_lower=arrays["joint_lower"],
    joint_upper=arrays["joint_upper"],
    actuator_load_fraction=arrays["actuator_load_fraction"],
    joint_margin_fraction=args.joint_margin,
    actuator_load_limit=args.load_limit,
  )
  envelope = training_envelope(
    heights=arrays["heights"],
    pitches=arrays["pitches"],
    feasible=feasible,
    inward_fraction=args.inward_fraction,
    pitch_limit=args.pitch_limit,
  )
  posture_map = fit_posture_map(
    arrays["heights"][feasible],
    arrays["pitches"][feasible],
    arrays["joint_positions"][feasible],
  )
  payload = posture_map_to_dict(
    posture_map,
    envelope,
    feasible_sample_count=int(np.count_nonzero(feasible)),
    total_sample_count=int(feasible.size),
  )
  payload["source_npz"] = str(input_path)

  output_path = args.output.resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  print(f"Wrote two-leg posture map: {output_path}")


if __name__ == "__main__":
  main()
