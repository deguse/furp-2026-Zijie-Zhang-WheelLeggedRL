"""Build a frozen, hash-bound deployable classical stair maneuver artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hoppertrex_mjlab.hybrid.stair_classical import (
    ARM_DISTANCE_M,
    MANEUVER_ARTIFACT_TYPE,
    _canonical_hash,
    parse_stair_maneuver,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def build_maneuver(
    parameters: dict[str, Any],
    detector_result: dict[str, Any],
    bindings: dict[str, Any],
) -> dict[str, Any]:
    selected = detector_result.get("selected")
    if not isinstance(selected, dict) or not selected.get("qualification", {}).get(
        "qualified"
    ):
        raise ValueError("Contact detector must be formally qualified.")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": MANEUVER_ARTIFACT_TYPE,
        "known_step_height": None,
        "arm_distance_m": ARM_DISTANCE_M,
        "control_frequency_hz": 50.0,
        "parameters": parameters,
        "contact_detector": selected["contact_detector"],
        "bindings": bindings,
        "optimization_protocol": {
            "method": "cem",
            "population": 256,
            "elite_fraction": 0.10,
            "iterations": 20,
            "seed": 1,
            "smoothing": 0.25,
            "minimum_std_fraction": 0.01,
            "training_heights_m": [0.01, 0.03, 0.05, 0.07, 0.09],
            "development_seeds": [1, 2, 3],
        },
    }
    payload["maneuver_hash"] = _canonical_hash(payload)
    parse_stair_maneuver(payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = build_maneuver(
        _object(args.parameters),
        _object(args.detector),
        _object(args.bindings),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
    )
    print(f"Wrote classical stair maneuver: {args.output.resolve()}")
    print(f"maneuver_hash={payload['maneuver_hash']}")


if __name__ == "__main__":
    main()
