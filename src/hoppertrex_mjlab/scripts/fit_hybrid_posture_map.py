"""Fit the Hybrid v2 two-leg posture map from static sweep NPZ data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hoppertrex_mjlab.hybrid.posture import (
  LEG_JOINT_NAMES,
  fit_posture_map,
  posture_artifact_hash,
  posture_map_to_dict,
  select_feasible_samples,
  training_envelope,
  training_envelope_from_sweep_grid,
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


def validated_sweep_metadata(input_path: Path) -> dict[str, object]:
  metadata_path = input_path.with_suffix('.json')
  if not metadata_path.is_file():
    raise FileNotFoundError(
      f'Posture sweep metadata sidecar is missing: {metadata_path}'
    )
  payload = json.loads(metadata_path.read_text(encoding='utf-8'))
  if not isinstance(payload, dict) or payload.get('schema_version') != 1:
    raise ValueError('Posture sweep metadata must use schema_version 1.')
  if tuple(payload.get('joint_names', ())) != LEG_JOINT_NAMES:
    raise ValueError('Posture sweep metadata has unexpected leg joint names.')
  if not isinstance(payload.get('git_sha'), str) or not payload['git_sha']:
    raise ValueError('Posture sweep metadata must record a git SHA.')
  controller = payload.get('controller')
  if (
    not isinstance(controller, dict)
    or controller.get('type') != 'lqr'
    or controller.get('qualified') is not True
    or not isinstance(controller.get('gain_hash'), str)
    or not controller['gain_hash']
  ):
    raise ValueError(
      'Formal posture fitting requires sweep metadata from a qualified LQR.'
    )
  calibration = payload.get('calibration')
  if (
    not isinstance(calibration, dict)
    or not isinstance(calibration.get('hash'), str)
    or not calibration['hash']
  ):
    raise ValueError(
      'Formal posture fitting requires a qualified velocity calibration.'
    )
  point_count = payload.get('point_count')
  if not isinstance(point_count, int) or point_count <= 0:
    raise ValueError('Posture sweep metadata must record a positive point_count.')
  return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, required=True, help="Input sweep NPZ.")
  parser.add_argument("--output", type=Path, required=True, help="Output JSON.")
  parser.add_argument("--joint-margin", type=float, default=0.10)
  parser.add_argument(
    "--joint-margin-rad",
    type=float,
    default=None,
    help=(
      "Absolute joint margin in rad; overrides the fraction-of-range "
      "--joint-margin. Evidence-based value 0.12 (2026-07-15 diagnosis: "
      "the 0.10 fraction demanded 0.279 rad on the knees and rejected the "
      "nominal standing posture at 0.245 rad from its limit)."
    ),
  )
  parser.add_argument("--load-limit", type=float, default=0.80)
  parser.add_argument("--inward-fraction", type=float, default=0.10)
  parser.add_argument("--pitch-limit", type=float, default=0.08)
  parser.add_argument(
    "--pitch-half-span",
    type=float,
    default=None,
    help=(
      "Height-priority inscription: fix the pitch half-width (rad) and "
      "maximize the height span inside the verified hull, requiring the "
      "pitch range to contain zero. Use when the feasible band is "
      "diagonal and the uniform inscription crushes the height span "
      "(measured on the 2026-07-19 expanded sweep: uniform 3.5 cm vs "
      "height-priority ~6.5 cm)."
    ),
  )
  parser.add_argument(
    "--height-floor",
    type=float,
    default=None,
    help=(
      "Exclude sweep samples below this height (m) before envelope "
      "inscription. Balance-probe authority over static feasibility: "
      "the 2026-07-19 expanded sweep accepted height 0.265 statically "
      "but the qualified LQR fell there in 4/5 pitch cells."
    ),
  )
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  input_path = args.input.resolve()
  sweep_metadata = validated_sweep_metadata(input_path)
  with np.load(input_path, allow_pickle=False) as data:
    missing = [name for name in REQUIRED_ARRAYS if name not in data]
    if missing:
      raise ValueError(f"Input NPZ is missing arrays: {', '.join(missing)}")
    arrays = {name: data[name] for name in REQUIRED_ARRAYS}
    sweep_coordinates = {
      name: data[name]
      for name in ("hip_offsets", "knee_offsets")
      if name in data
    }
  if sweep_metadata['point_count'] != int(arrays['heights'].size):
    raise ValueError(
      'Posture sweep metadata point_count does not match the NPZ arrays.'
    )
  if len(sweep_coordinates) == 1:
    raise ValueError(
      "Posture sweep must contain both hip_offsets and knee_offsets."
    )

  feasible = select_feasible_samples(
    non_wheel_contact=arrays["non_wheel_contact"],
    joint_positions=arrays["joint_positions"],
    joint_lower=arrays["joint_lower"],
    joint_upper=arrays["joint_upper"],
    actuator_load_fraction=arrays["actuator_load_fraction"],
    joint_margin_fraction=args.joint_margin,
    actuator_load_limit=args.load_limit,
    joint_margin_rad=args.joint_margin_rad,
  )
  if sweep_coordinates:
    envelope = training_envelope_from_sweep_grid(
      heights=arrays["heights"],
      pitches=arrays["pitches"],
      feasible=feasible,
      first_coordinates=sweep_coordinates["hip_offsets"],
      second_coordinates=sweep_coordinates["knee_offsets"],
      inward_fraction=args.inward_fraction,
      pitch_limit=args.pitch_limit,
      pitch_half_span=args.pitch_half_span,
      height_floor=args.height_floor,
    )
  else:
    if args.pitch_half_span is not None:
      raise ValueError(
        "--pitch-half-span requires a sweep NPZ with grid coordinates."
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
  payload["fit_criteria"] = {
    "joint_margin_fraction": args.joint_margin,
    "joint_margin_rad": args.joint_margin_rad,
    "actuator_load_limit": args.load_limit,
    "inward_fraction": args.inward_fraction,
    "pitch_limit": args.pitch_limit,
    "pitch_half_span": args.pitch_half_span,
    "height_floor": args.height_floor,
  }

  payload['source_sweep'] = {
    'git_sha': sweep_metadata['git_sha'],
    'seed': sweep_metadata.get('seed'),
    'controller_gain_hash': sweep_metadata['controller']['gain_hash'],
    'calibration_hash': sweep_metadata['calibration']['hash'],
  }
  payload["posture_artifact_hash"] = posture_artifact_hash(payload)

  output_path = args.output.resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  print(f"Wrote two-leg posture map: {output_path}")


if __name__ == "__main__":
  main()
