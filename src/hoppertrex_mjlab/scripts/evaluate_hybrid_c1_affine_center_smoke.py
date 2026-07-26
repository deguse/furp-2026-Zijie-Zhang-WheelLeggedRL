#!/usr/bin/env python3
"""Run the incumbent-controlled C1 affine center smoke before a full flat gate."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_PATH = Path(__file__).resolve().parents[1]
SRC_PATH = Path(__file__).resolve().parents[2]
for path in (PROJECT_PATH, SRC_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hoppertrex_mjlab.tasks as tasks  # noqa: E402,F401
from hoppertrex_mjlab.hybrid.controller_schedule import (  # noqa: E402
    AFFINE_SCHEDULE_STATE_DEFINITION,
    REGISTERED_HEIGHT_NODES,
)
from hoppertrex_mjlab.scripts.evaluate_hybrid_c1_flat_gate import (  # noqa: E402
    EXPECTED_ARRAY_SHAPES,
    NODE_STEMS,
    REQUIRED_ARRAYS,
    aggregate_candidate,
    candidate_schedule,
    fit_all_candidates,
    fit_qualification_summary,
    load_registered_floors,
    registered_caps,
    run_cell,
)
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

FORMAL_TASK = "HopperTrex-Hybrid-v2-Stage3"
FORMAL_DEVICE = "cuda:0"
FORMAL_NUM_ENVS = 16
FORMAL_SETTLE_STEPS = 100
FORMAL_MEASURE_STEPS = 200
FORMAL_VX_CHECK = 0.05
FORMAL_CENTER_HEIGHT = REGISTERED_HEIGHT_NODES[1]
DELTA_MEAN_MATCH_TOLERANCE = 1.0e-12


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compensated-qualification", type=Path, required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--mjlab-git-sha", required=True)
    parser.add_argument("--task", default=FORMAL_TASK)
    parser.add_argument("--device", default=FORMAL_DEVICE)
    parser.add_argument("--num-envs", type=int, default=FORMAL_NUM_ENVS)
    parser.add_argument("--settle-steps", type=int, default=FORMAL_SETTLE_STEPS)
    parser.add_argument("--measure-steps", type=int, default=FORMAL_MEASURE_STEPS)
    parser.add_argument("--vx-check", type=float, default=FORMAL_VX_CHECK)
    return parser.parse_args(argv)


def _validate_formal_args(args: argparse.Namespace) -> None:
    actual = (
        args.task,
        args.device,
        args.num_envs,
        args.settle_steps,
        args.measure_steps,
        args.vx_check,
    )
    expected = (
        FORMAL_TASK,
        FORMAL_DEVICE,
        FORMAL_NUM_ENVS,
        FORMAL_SETTLE_STEPS,
        FORMAL_MEASURE_STEPS,
        FORMAL_VX_CHECK,
    )
    if actual != expected:
        raise ValueError("Affine center smoke arguments are frozen.")
    for name in ("git_sha", "mjlab_git_sha"):
        if re.fullmatch(r"[0-9a-f]{40}", getattr(args, name)) is None:
            raise ValueError(f"{name} must be a full lowercase Git SHA.")


def load_affine_nodes(nodes_dir: Path) -> dict[str, dict[str, object]]:
    """Load exact centered arrays and their affine equilibrium provenance."""

    nodes: dict[str, dict[str, object]] = {}
    incumbent: tuple[float, ...] | None = None
    for stem in NODE_STEMS:
        npz_path = nodes_dir / f"{stem}.npz"
        metadata_path = nodes_dir / f"{stem}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("state_definition_version") != AFFINE_SCHEDULE_STATE_DEFINITION:
            raise ValueError(f"{stem} is not an affine-v3 node.")
        state = np.asarray(metadata.get("equilibrium_state"), dtype=np.float64)
        control = np.asarray(metadata.get("equilibrium_input"), dtype=np.float64)
        if state.shape != (4,) or control.shape != (1,):
            raise ValueError(f"{stem} has invalid affine equilibrium dimensions.")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(control)):
            raise ValueError(f"{stem} affine equilibrium must be finite.")
        gain = tuple(float(value) for value in metadata["controller"]["gain"])
        if len(gain) != 4 or not all(math.isfinite(value) for value in gain):
            raise ValueError(f"{stem} incumbent gain is invalid.")
        if incumbent is None:
            incumbent = gain
        elif incumbent != gain:
            raise ValueError("Affine nodes do not share one incumbent gain.")
        with np.load(npz_path, allow_pickle=False) as data:
            if set(data.files) != set(REQUIRED_ARRAYS):
                raise ValueError(f"{stem} identification arrays are incomplete.")
            arrays = {name: np.asarray(data[name]) for name in REQUIRED_ARRAYS}
        for name, array in arrays.items():
            if array.shape != EXPECTED_ARRAY_SHAPES[name] or not np.all(
                np.isfinite(array)
            ):
                raise ValueError(f"{stem} {name} is invalid.")
        combined_state = np.concatenate((arrays["states"], arrays["heldout_states"]))
        combined_input = np.concatenate((arrays["inputs"], arrays["heldout_inputs"]))
        recorded_state_mean = np.asarray(
            metadata.get("delta_state_mean"), dtype=np.float64
        )
        recorded_input_mean = np.asarray(
            metadata.get("delta_input_mean"), dtype=np.float64
        )
        if recorded_state_mean.shape != (4,) or recorded_input_mean.shape != (1,):
            raise ValueError(f"{stem} delta mean provenance is incomplete.")
        if not np.allclose(
            np.mean(combined_state, axis=0),
            recorded_state_mean,
            rtol=0.0,
            atol=DELTA_MEAN_MATCH_TOLERANCE,
        ):
            raise ValueError(f"{stem} delta state mean does not match metadata.")
        if not np.allclose(
            np.mean(combined_input, axis=0),
            recorded_input_mean,
            rtol=0.0,
            atol=DELTA_MEAN_MATCH_TOLERANCE,
        ):
            raise ValueError(f"{stem} delta input mean does not match metadata.")
        nodes[stem] = {"arrays": arrays, "metadata": metadata}
    return nodes


def center_cells(vx_check: float = FORMAL_VX_CHECK) -> list[tuple[float, float, float]]:
    return [
        (FORMAL_CENTER_HEIGHT, 0.0, 0.0),
        (FORMAL_CENTER_HEIGHT, 0.0, vx_check),
        (FORMAL_CENTER_HEIGHT, 0.0, -vx_check),
    ]


def _equilibrium_grids(
    nodes: dict[str, dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = np.zeros((3, 3, 4), dtype=np.float64)
    control = np.zeros((3, 3), dtype=np.float64)
    for stem in NODE_STEMS:
        h_index, p_index = int(stem[6]), int(stem[9])
        metadata = nodes[stem]["metadata"]
        state[h_index, p_index] = metadata["equilibrium_state"]
        control[h_index, p_index] = float(metadata["equilibrium_input"][0])
    incumbent = np.asarray(
        nodes[NODE_STEMS[0]]["metadata"]["controller"]["gain"], dtype=np.float64
    )
    return state, control, incumbent


def _collection_git_sha(nodes: dict[str, dict[str, object]]) -> str:
    git_shas = {str(node["metadata"].get("git_sha")) for node in nodes.values()}
    if len(git_shas) != 1:
        raise ValueError("Affine nodes do not share one collection Git SHA.")
    git_sha = next(iter(git_shas))
    if re.fullmatch(r"[0-9a-f]{40}", git_sha) is None:
        raise ValueError("Affine collection Git SHA must be full lowercase hex.")
    return git_sha


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_formal_args(args)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite affine center smoke: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("Formal affine center smoke requires CUDA.")
    nodes = load_affine_nodes(args.nodes_dir.resolve())
    collection_git_sha = _collection_git_sha(nodes)
    fitted = fit_all_candidates(nodes)
    fit_summary = fit_qualification_summary(fitted)
    equilibrium_state, equilibrium_input, incumbent_gain = _equilibrium_grids(nodes)
    cfg = load_env_cfg(args.task, play=True)
    cfg.seed = 1
    cfg.scene.num_envs = args.num_envs
    if cfg.scene.terrain is not None:
        cfg.scene.terrain.num_envs = args.num_envs
    action_cfg = cfg.actions["hybrid_wheel_leg"]
    first_metadata = nodes[NODE_STEMS[0]]["metadata"]
    runtime_bindings = {
        "controller_gain_hash": action_cfg.controller_gain_hash,
        "velocity_calibration_hash": action_cfg.calibration_hash,
        "posture_artifact_hash": action_cfg.posture_artifact_hash,
        "station_calibration_hash": action_cfg.station_calibration_hash,
    }
    node_bindings = {
        "controller_gain_hash": first_metadata["controller"]["gain_hash"],
        "velocity_calibration_hash": first_metadata["calibration_hash"],
        "posture_artifact_hash": first_metadata["posture_artifact_hash"],
        "station_calibration_hash": first_metadata["station_calibration_hash"],
    }
    if runtime_bindings != node_bindings:
        raise ValueError("Affine node bindings do not match the loaded runtime.")
    floors = load_registered_floors(
        args.compensated_qualification.resolve(),
        expected_controller_gain_hash=action_cfg.controller_gain_hash,
    )
    caps = registered_caps(floors)
    zero_state = np.zeros((3, 3, 4), dtype=np.float64)
    zero_input = np.zeros((3, 3), dtype=np.float64)
    incumbent_grid = np.broadcast_to(incumbent_gain, (3, 3, 4)).copy()
    action_cfg.controller_schedule = candidate_schedule(
        incumbent_grid, zero_state, [20.0, 2.0, 4.0, 0.5], [1.0], zero_input
    )
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    try:
        action_term = env.action_manager.get_term("hybrid_wheel_leg")
        cells = center_cells(args.vx_check)
        incumbent_cells = [
            run_cell(
                env,
                height=height,
                pitch=pitch,
                vx=vx,
                settle_steps=args.settle_steps,
                measure_steps=args.measure_steps,
            )
            for height, pitch, vx in cells
        ]
        incumbent_verdict = aggregate_candidate(incumbent_cells, caps)
        if not incumbent_verdict["flat_gate_passed"]:
            raise RuntimeError(
                "Legacy incumbent failed the same-session center control."
            )
        action_term._schedule_equilibrium_state.copy_(
            torch.tensor(equilibrium_state, device=env.device, dtype=torch.float)
        )
        action_term._schedule_equilibrium_input.copy_(
            torch.tensor(equilibrium_input, device=env.device, dtype=torch.float)
        )
        action_term._schedule_gains.copy_(
            torch.tensor(incumbent_grid, device=env.device, dtype=torch.float)
        )
        affine_incumbent_cells = [
            run_cell(
                env,
                height=height,
                pitch=pitch,
                vx=vx,
                settle_steps=args.settle_steps,
                measure_steps=args.measure_steps,
            )
            for height, pitch, vx in cells
        ]
        affine_incumbent_verdict = aggregate_candidate(affine_incumbent_cells, caps)
        if not affine_incumbent_verdict["flat_gate_passed"]:
            raise RuntimeError(
                "Affine incumbent failed zero-blend equivalence at center."
            )
        print("[affine-smoke] affine incumbent alpha=0 pass=True")
        candidates = []
        for candidate in fitted:
            action_term._schedule_gains.copy_(
                torch.tensor(candidate["gains"], device=env.device, dtype=torch.float)
            )
            cell_results = [
                run_cell(
                    env,
                    height=height,
                    pitch=pitch,
                    vx=vx,
                    settle_steps=args.settle_steps,
                    measure_steps=args.measure_steps,
                )
                for height, pitch, vx in cells
            ]
            verdict = aggregate_candidate(cell_results, caps)
            candidates.append(
                {
                    "index": candidate["index"],
                    "q_diag": candidate["q_diag"],
                    "r_diag": candidate["r_diag"],
                    "anchor_alpha": candidate["anchor_alpha"],
                    "node_facts": candidate["node_facts"],
                    "cells": cell_results,
                    **verdict,
                }
            )
            print(
                f"[affine-smoke] candidate {candidate['index']:>2} alpha={candidate['anchor_alpha']:.4f} pass={verdict['flat_gate_passed']}"
            )
    finally:
        env.close()
    passed = sum(item["flat_gate_passed"] for item in candidates)
    classification = (
        "AFFINE_CENTER_SMOKE_HAS_CANDIDATES"
        if passed
        else "AFFINE_CENTER_SMOKE_NO_CANDIDATE_STOP"
    )
    payload = {
        "schema_version": 1,
        "kind": "c1_affine_center_smoke",
        "classification": classification,
        "incumbent": {"cells": incumbent_cells, **incumbent_verdict},
        "affine_incumbent": {
            "anchor_alpha": 0.0,
            "cells": affine_incumbent_cells,
            **affine_incumbent_verdict,
        },
        "candidates": candidates,
        "passed_candidate_count": passed,
        "caps": caps,
        "fit_qualification": fit_summary,
        "evidence_eligible": True,
        "promotion_eligible": False,
        "training_eligible": False,
        "checkpoint": None,
        "next_step": "DOWNLOAD_FOR_REVIEW" if passed else "STOP",
    }
    payload["git_sha"] = args.git_sha
    payload["collection_git_sha"] = collection_git_sha
    payload["mjlab_git_sha"] = args.mjlab_git_sha
    payload["bindings"] = node_bindings
    payload["completed_candidate_count"] = 27
    payload["completed_node_fit_count"] = 243
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"[affine-smoke] classification={classification} passed={passed}/27")


if __name__ == "__main__":
    main()
