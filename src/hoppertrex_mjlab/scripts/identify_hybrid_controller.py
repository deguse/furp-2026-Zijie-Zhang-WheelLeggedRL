"""Identify and qualify the Hybrid v2 wheel-balance controller from NPZ data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hoppertrex_mjlab.hybrid.identification import (
  controller_design_to_dict,
  identify_controller,
)


REQUIRED_ARRAYS = (
  "states",
  "inputs",
  "next_states",
  "heldout_states",
  "heldout_inputs",
  "heldout_next_states",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, required=True, help="Input sweep NPZ.")
  parser.add_argument("--output", type=Path, required=True, help="Output JSON.")
  parser.add_argument(
    "--q-diag",
    type=float,
    nargs=4,
    default=(20.0, 2.0, 4.0, 0.5),
    metavar=("PITCH", "PITCH_RATE", "VX_ERROR", "WHEEL_ERROR"),
  )
  parser.add_argument("--r", type=float, default=1.0, help="Scalar input cost.")
  parser.add_argument(
    "--pd-gain",
    type=float,
    nargs=4,
    default=(8.0, 1.0, 3.0, 0.2),
    metavar=("PITCH", "PITCH_RATE", "VX_ERROR", "WHEEL_ERROR"),
  )
  parser.add_argument(
    "--nrmse-limit",
    type=float,
    default=0.15,
    help="Maximum held-out one-step range-normalized RMSE.",
  )
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  input_path = args.input.resolve()
  with np.load(input_path, allow_pickle=False) as data:
    missing = [name for name in REQUIRED_ARRAYS if name not in data]
    if missing:
      raise ValueError(f"Input NPZ is missing arrays: {', '.join(missing)}")
    arrays = {name: data[name] for name in REQUIRED_ARRAYS}

  design = identify_controller(
    arrays["states"],
    arrays["inputs"],
    arrays["next_states"],
    heldout_states=arrays["heldout_states"],
    heldout_inputs=arrays["heldout_inputs"],
    heldout_next_states=arrays["heldout_next_states"],
    q_diag=args.q_diag,
    r_diag=(args.r,),
    pd_gain=args.pd_gain,
    nrmse_limit=args.nrmse_limit,
  )
  payload = controller_design_to_dict(design)
  payload["source_npz"] = str(input_path)

  output_path = args.output.resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  print(f"Wrote {design.controller_type.upper()} controller: {output_path}")


if __name__ == "__main__":
  main()
