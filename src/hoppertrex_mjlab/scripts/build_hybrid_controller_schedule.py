"""Build and validate one 3x3 gain-scheduled LQR artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from hoppertrex_mjlab.hybrid.controller_schedule import (
    SCHEDULE_ARTIFACT_TYPE,
    SCHEDULE_STATE_DEFINITION,
    canonical_hash,
    parse_controller_schedule,
)
from hoppertrex_mjlab.hybrid.identification import (
    CONTROLLER_STATE_NAMES,
    NOMINAL_WHEEL_RADIUS_M,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_schedule(manifest: dict[str, Any], base: Path) -> dict[str, Any]:
    heights = [float(value) for value in manifest["height_nodes"]]
    pitches = [float(value) for value in manifest["pitch_nodes"]]
    registered = manifest.get("nodes")
    if not isinstance(registered, list) or len(registered) != 9:
        raise ValueError("Schedule manifest must register exactly nine nodes.")
    by_coordinate = {
        (float(item["height_m"]), float(item["pitch_rad"])): item for item in registered
    }
    rows: list[list[dict[str, Any]]] = []
    for height in heights:
        row: list[dict[str, Any]] = []
        for pitch in pitches:
            item = by_coordinate.get((height, pitch))
            if item is None:
                raise ValueError(
                    f"Missing controller node at height={height}, pitch={pitch}."
                )
            controller_path = (base / str(item["controller_path"])).resolve()
            controller = json.loads(controller_path.read_text(encoding="utf-8"))
            gain = np.asarray(controller.get("gain"), dtype=np.float64).reshape(-1)
            if gain.shape != (4,):
                raise ValueError(
                    f"Node gain must contain four values: {controller_path}"
                )
            row.append(
                {
                    "controller_type": controller.get("controller_type"),
                    "gain": gain.tolist(),
                    "equilibrium_pitch": float(item["equilibrium_pitch"]),
                    "model": controller.get("model"),
                    "controllability_rank": controller.get("controllability_rank"),
                    "heldout_one_step_nrmse": controller.get("heldout_one_step_nrmse"),
                    "fallback_reasons": controller.get("fallback_reasons"),
                    "source_npz": controller.get("source_npz"),
                    "controller_file_sha256": _sha256(controller_path),
                }
            )
        rows.append(row)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": SCHEDULE_ARTIFACT_TYPE,
        "state_names": list(CONTROLLER_STATE_NAMES),
        "state_construction": {
            "state_definition_version": SCHEDULE_STATE_DEFINITION,
            "wheel_radius": NOMINAL_WHEEL_RADIUS_M,
        },
        "height_nodes": heights,
        "pitch_nodes": pitches,
        "q_diag": [float(value) for value in manifest["q_diag"]],
        "r_diag": [float(value) for value in manifest["r_diag"]],
        "bindings": dict(manifest["bindings"]),
        "selection": dict(manifest["selection"]),
        "collection_protocol": {
            "num_envs": 32,
            "steps": 2500,
            "warmup_steps": 250,
            "hold_steps": 5,
            "heldout_fraction": 0.20,
        },
        "nodes": rows,
    }
    payload["schedule_hash"] = canonical_hash(payload, hash_field="schedule_hash")
    parse_controller_schedule(payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("Schedule manifest must contain a JSON object.")
    payload = build_schedule(manifest, manifest_path.parent)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
    )
    print(f"Wrote qualified controller schedule: {output}")
    print(f"schedule_hash={payload['schedule_hash']}")


if __name__ == "__main__":
    main()
