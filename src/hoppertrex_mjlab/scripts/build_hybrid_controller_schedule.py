"""Build and validate one 3x3 gain-scheduled LQR artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def _source_metadata(
    controller: dict[str, Any], controller_path: Path
) -> tuple[Path, dict[str, Any]]:
    source = controller.get("source_npz")
    if not isinstance(source, str) or not source:
        raise ValueError(f"Node controller has no source_npz: {controller_path}")
    source_path = Path(source).expanduser()
    if not source_path.is_absolute():
        source_path = controller_path.parent / source_path
    source_path = source_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Node source NPZ does not exist: {source_path}")
    sidecar = source_path.with_suffix(".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError(f"Node source sidecar must be a JSON object: {sidecar}")
    return source_path, metadata


def _validated_selection(selection: Any, base: Path) -> dict[str, Any]:
    if not isinstance(selection, dict):
        raise TypeError("Schedule manifest selection must be a JSON object.")
    evidence_source = selection.get("evaluation_artifact_path")
    expected_hash = selection.get("evaluation_artifact_sha256")
    if not isinstance(evidence_source, str) or not evidence_source:
        raise ValueError("Schedule selection requires evaluation_artifact_path.")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError("Schedule selection requires evaluation_artifact_sha256.")
    evidence_path = (base / evidence_source).resolve()
    if not evidence_path.is_file():
        raise FileNotFoundError(
            f"Flat-gate evaluation artifact missing: {evidence_path}"
        )
    actual_hash = _sha256(evidence_path)
    if actual_hash != expected_hash:
        raise ValueError("Flat-gate evaluation artifact SHA256 mismatch.")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise TypeError("Flat-gate evaluation artifact must be a JSON object.")
    result = dict(evidence)
    result["evaluation_artifact_sha256"] = actual_hash
    return result


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
            source_npz, metadata = _source_metadata(controller, controller_path)
            source_metadata = source_npz.with_suffix(".json")
            source_npz_sha256 = _sha256(source_npz)
            source_metadata_sha256 = _sha256(source_metadata)
            if controller.get("source_npz_sha256") != source_npz_sha256:
                raise ValueError(f"Node source NPZ changed after fitting: {source_npz}")
            if controller.get("source_metadata_sha256") != source_metadata_sha256:
                raise ValueError(
                    f"Node source metadata changed after fitting: {source_metadata}"
                )
            gain = np.asarray(controller.get("gain"), dtype=np.float64).reshape(-1)
            if gain.shape != (4,):
                raise ValueError(
                    f"Node gain must contain four values: {controller_path}"
                )
            if tuple(float(value) for value in controller.get("q_diag", ())) != tuple(
                float(value) for value in manifest["q_diag"]
            ):
                raise ValueError(f"Node Q diagonal mismatch: {controller_path}")
            if tuple(float(value) for value in controller.get("r_diag", ())) != tuple(
                float(value) for value in manifest["r_diag"]
            ):
                raise ValueError(f"Node R diagonal mismatch: {controller_path}")
            if controller.get("state_construction", {}).get(
                "state_definition_version"
            ) != SCHEDULE_STATE_DEFINITION:
                raise ValueError(
                    f"Node state definition is not schedule-v2: {controller_path}"
                )
            if metadata.get("state_definition_version") != SCHEDULE_STATE_DEFINITION:
                raise ValueError(f"Node sidecar is not schedule-v2: {source_npz}")
            expected_protocol = {
                "num_envs": 32,
                "steps": 2500,
                "warmup_steps": 250,
                "hold_steps": 5,
                "heldout_fraction": 0.20,
            }
            for key, expected in expected_protocol.items():
                if metadata.get(key) != expected:
                    raise ValueError(
                        f"Node collection protocol mismatch for {key}: {source_npz}"
                    )
            if float(metadata.get("height_command", math.nan)) != height:
                raise ValueError(f"Node height metadata mismatch: {source_npz}")
            if float(metadata.get("pitch_command", math.nan)) != pitch:
                raise ValueError(f"Node pitch metadata mismatch: {source_npz}")
            if float(metadata.get("equilibrium_pitch", math.nan)) != float(
                item["equilibrium_pitch"]
            ):
                raise ValueError(f"Node equilibrium metadata mismatch: {source_npz}")
            bindings = manifest["bindings"]
            if metadata.get("controller", {}).get("gain_hash") != bindings[
                "identification_controller_gain_hash"
            ]:
                raise ValueError(
                    f"Node source controller binding mismatch: {source_npz}"
                )
            if metadata.get("calibration_hash") != bindings[
                "identification_calibration_hash"
            ]:
                raise ValueError(
                    f"Node source calibration binding mismatch: {source_npz}"
                )
            if metadata.get("posture_artifact_hash") != bindings[
                "posture_artifact_hash"
            ]:
                raise ValueError(f"Node posture binding mismatch: {source_npz}")
            row.append(
                {
                    "controller_type": controller.get("controller_type"),
                    "gain": gain.tolist(),
                    "equilibrium_pitch": float(item["equilibrium_pitch"]),
                    "model": controller.get("model"),
                    "controllability_rank": controller.get("controllability_rank"),
                    "heldout_one_step_nrmse": controller.get("heldout_one_step_nrmse"),
                    "fallback_reasons": controller.get("fallback_reasons"),
                    "source_npz": str(source_npz),
                    "source_npz_sha256": source_npz_sha256,
                    "source_metadata_sha256": source_metadata_sha256,
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
        "selection": _validated_selection(manifest.get("selection"), base),
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
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote qualified controller schedule: {output}")
    print(f"schedule_hash={payload['schedule_hash']}")


if __name__ == "__main__":
    main()
