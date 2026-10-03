"""Cross-check one ellipsoid wheel-collision candidate on both physics backends.

This development-only rejection probe changes no visual, inertia, actuator,
controller, reset, command, or verdict setting. It replaces exactly the two
wheel cylinder collision geoms with axis-preserving ellipsoids, runs the
reviewed C0 exact resets in MJWarp, then repeats them with the native closed-loop
backend probe. No shape parameter is exposed and no grid is permitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from mjlab.envs import ManagerBasedRlEnv

from hoppertrex_mjlab.hybrid.wheel_collision_proxy import (
    WHEEL_COLLISION_GEOM_NAMES,
    make_ellipsoid_wheel_spec,
)
from hoppertrex_mjlab.scripts import diagnose_roll_boundary as diag
from hoppertrex_mjlab.scripts import probe_r0c_support_transfer_shadow as shadow_probe
from hoppertrex_mjlab.scripts import probe_r0c_yaw_feedback as lateral
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb
from hoppertrex_mjlab.scripts.probe_r0c_native_contact_replay import (
    _model_equivalence_report,
)
from hoppertrex_mjlab.scripts.probe_r0c_native_full_rollout import (
    _run_native_rollout,
)
from hoppertrex_mjlab.scripts.probe_r0c_native_full_rollout import (
    _validate_rows as _validate_native_rows,
)

SCHEMA_VERSION = 1
HEIGHTS_M = (0.0, 0.0025)
_FULL_WHEEL_NAMES = tuple(f"robot/{name}" for name in WHEEL_COLLISION_GEOM_NAMES)
_UNCHANGED_MODEL_FIELDS = (
    "body_mass",
    "body_inertia",
    "body_ipos",
    "body_iquat",
    "body_pos",
    "body_quat",
    "jnt_type",
    "jnt_bodyid",
    "jnt_qposadr",
    "jnt_dofadr",
    "jnt_pos",
    "jnt_axis",
    "jnt_range",
    "dof_damping",
    "dof_frictionloss",
    "dof_armature",
    "actuator_trntype",
    "actuator_trnid",
    "actuator_dyntype",
    "actuator_gaintype",
    "actuator_biastype",
    "actuator_dynprm",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_ctrlrange",
    "actuator_forcerange",
    "actuator_gear",
    "geom_bodyid",
    "geom_pos",
    "geom_quat",
    "geom_friction",
    "geom_solref",
    "geom_solimp",
    "geom_margin",
    "geom_gap",
    "geom_condim",
    "geom_contype",
    "geom_conaffinity",
    "geom_priority",
    "sensor_type",
    "sensor_objtype",
    "sensor_objid",
    "sensor_dim",
    "sensor_adr",
    "sensor_cutoff",
    "sensor_noise",
)


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=rb.REPOSITORY_PATH,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_snapshot(
    model: mujoco.MjModel,
    wheel_names: Sequence[str],
) -> dict[str, Any]:
    wheel_ids = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in wheel_names
    )
    if any(geom_id < 0 for geom_id in wheel_ids):
        raise ValueError("Model contract could not resolve both wheel geoms.")
    return {
        "dimensions": {
            name: int(getattr(model, name))
            for name in ("nq", "nv", "nu", "na", "nbody", "ngeom", "njnt", "nsensor")
        },
        "names": bytes(model.names),
        "wheel_ids": wheel_ids,
        "geom_type": np.asarray(model.geom_type).copy(),
        "geom_size": np.asarray(model.geom_size).copy(),
        "unchanged": {
            name: np.asarray(getattr(model, name)).copy()
            for name in _UNCHANGED_MODEL_FIELDS
        },
        "options": {
            "timestep": float(model.opt.timestep),
            "gravity": np.asarray(model.opt.gravity).copy(),
            "integrator": int(model.opt.integrator),
            "cone": int(model.opt.cone),
            "solver": int(model.opt.solver),
            "iterations": int(model.opt.iterations),
            "tolerance": float(model.opt.tolerance),
            "disableflags": int(model.opt.disableflags),
            "enableflags": int(model.opt.enableflags),
        },
    }


def _compare_model_contract(
    baseline: Mapping[str, Any],
    proxy: mujoco.MjModel,
    wheel_names: Sequence[str],
) -> dict[str, Any]:
    observed = _model_snapshot(proxy, wheel_names)
    if baseline["dimensions"] != observed["dimensions"]:
        raise RuntimeError("Ellipsoid proxy changed model dimensions.")
    if baseline["names"] != observed["names"]:
        raise RuntimeError("Ellipsoid proxy changed model names/order.")
    if baseline["wheel_ids"] != observed["wheel_ids"]:
        raise RuntimeError("Ellipsoid proxy changed wheel geom identities.")
    if baseline["options"].keys() != observed["options"].keys():
        raise RuntimeError("Ellipsoid proxy option schema changed.")
    for name in baseline["options"]:
        before = baseline["options"][name]
        after = observed["options"][name]
        if isinstance(before, np.ndarray):
            equal = np.array_equal(before, after)
        else:
            equal = before == after
        if not equal:
            raise RuntimeError(f"Ellipsoid proxy changed model option {name}.")
    for name, before in baseline["unchanged"].items():
        if not np.array_equal(before, observed["unchanged"][name]):
            raise RuntimeError(f"Ellipsoid proxy unexpectedly changed {name}.")

    wheel_ids = tuple(int(value) for value in baseline["wheel_ids"])
    type_diffs = tuple(
        int(value)
        for value in np.flatnonzero(
            baseline["geom_type"] != observed["geom_type"]
        )
    )
    size_diffs = tuple(
        int(value)
        for value in np.flatnonzero(
            np.any(baseline["geom_size"] != observed["geom_size"], axis=1)
        )
    )
    if set(type_diffs) != set(wheel_ids) or set(size_diffs) != set(wheel_ids):
        raise RuntimeError(
            "Ellipsoid proxy changed geometry outside the two wheel collision geoms."
        )

    changes = []
    for name, geom_id in zip(wheel_names, wheel_ids, strict=True):
        before_type = int(baseline["geom_type"][geom_id])
        after_type = int(observed["geom_type"][geom_id])
        before_size = baseline["geom_size"][geom_id]
        after_size = observed["geom_size"][geom_id]
        expected_size = np.asarray(
            (before_size[0], before_size[0], before_size[1]), dtype=np.float64
        )
        if before_type != int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            raise RuntimeError("Baseline wheel collision geom is not a cylinder.")
        if after_type != int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            raise RuntimeError("Proxy wheel collision geom is not an ellipsoid.")
        if not np.array_equal(after_size, expected_size):
            raise RuntimeError("Ellipsoid proxy did not preserve radius/half-width.")
        changes.append(
            {
                "name": name,
                "geom_id": geom_id,
                "before_type": "cylinder",
                "after_type": "ellipsoid",
                "before_size": before_size.tolist(),
                "after_size": after_size.tolist(),
            }
        )
    return {
        "only_wheel_geom_type_and_size_changed": True,
        "all_model_options_unchanged": True,
        "all_inertial_joint_actuator_contact_sensor_fields_unchanged": True,
        "changes": changes,
    }


def _backend_qualifies(summaries: Sequence[Mapping[str, Any]]) -> bool:
    if len(summaries) != 2:
        return False
    for height in HEIGHTS_M:
        row = next(
            (item for item in summaries if float(item["stair_height_m"]) == height),
            None,
        )
        if row is None:
            return False
        if (
            int(row["trials"]) != 8
            or int(row["successes"]) != 8
            or int(row.get("geometric_successes_ignoring_support", -1)) != 8
            or int(row["bilateral_unsupported_physics_substeps"]) != 0
            or int(row["non_wheel_contact_trials"]) != 0
            or int(row["terminated_trials"]) != 0
        ):
            return False
    return True


def _classification(rows: Sequence[Mapping[str, Any]]) -> dict[int, tuple[bool, ...]]:
    return {
        int(row["env_id"]): (
            bool(row["success"]),
            bool(row["geometric_success_ignoring_support"]),
            bool(row["bilateral_airborne_ever"]),
            bool(row["non_wheel_contact"]),
            bool(row["termination"]),
        )
        for row in rows
    }


def _build_proxy_cfg() -> Any:
    cfg = rb.make_roll_boundary_env_cfg(
        HEIGHTS_M,
        diag.R0C_SYNC_ENVS_PER_HEIGHT,
    )
    cfg.scene.entities["robot"].spec_fn = make_ellipsoid_wheel_spec
    return cfg


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.source_result.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("kind") != "r0c_synchronized_reference_rejection_screen":
        raise ValueError("Source result is not an R0c-SYNC artifact.")
    if source.get("matched_reset_perturbations_across_candidates") is not True:
        raise ValueError("Source R0c-SYNC artifact does not certify matched resets.")
    candidate = lateral._candidate_map()["c0"]
    source_resets = lateral._source_reset_map(source, "c0")

    baseline_env = ManagerBasedRlEnv(
        cfg=rb.make_roll_boundary_env_cfg(
            HEIGHTS_M, diag.R0C_SYNC_ENVS_PER_HEIGHT
        ),
        device=args.device,
    )
    try:
        baseline_snapshot = _model_snapshot(
            baseline_env.sim.mj_model, _FULL_WHEEL_NAMES
        )
    finally:
        baseline_env.close()

    original_cards = rb.POSTURE_CARDS
    rb.POSTURE_CARDS = (candidate["posture_card"],)
    proxy_env = ManagerBasedRlEnv(cfg=_build_proxy_cfg(), device=args.device)
    try:
        geometry_contract = _compare_model_contract(
            baseline_snapshot,
            proxy_env.sim.mj_model,
            _FULL_WHEEL_NAMES,
        )
        model_equivalence = _model_equivalence_report(proxy_env)
        mjwarp_rows = rb.run_card_repeat(
            proxy_env,
            heights=HEIGHTS_M,
            card=candidate["posture_card"],
            repeat=1,
            settle_steps=rb.OFFICIAL_SETTLE_STEPS,
            drive_steps=rb.OFFICIAL_DRIVE_STEPS,
            stable_steps=rb.OFFICIAL_STABLE_STEPS,
            episode_wide_safety=True,
            diagnostic_continue_after_support_loss=True,
            roll_pose_schedule=candidate["schedule"],
            roll_pose_slew_mode=str(candidate["slew_mode"]),
            require_pure_classical_authority=True,
            root_reset_override=shadow_probe._reset_override(source_resets),
            command_vx_mps=0.07,
        )
        lateral._validate_resets(mjwarp_rows, source_resets)
        native_rows = _run_native_rollout(
            proxy_env,
            candidate=candidate,
            source_resets=source_resets,
        )
        model = proxy_env.sim.mj_model
        model_metadata = {
            "timestep_s": float(model.opt.timestep),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "na": int(model.na),
        }
    finally:
        proxy_env.close()
        rb.POSTURE_CARDS = original_cards

    _validate_native_rows(native_rows)
    mjwarp_summaries = diag.summarize_trials(mjwarp_rows)
    native_summaries = diag.summarize_trials(native_rows)
    mjwarp_qualified = _backend_qualifies(mjwarp_summaries)
    native_qualified = _backend_qualifies(native_summaries)
    classification_match = _classification(mjwarp_rows) == _classification(native_rows)
    acceptance = {
        "mjwarp_flat_and_step_strict_8_of_8": mjwarp_qualified,
        "native_flat_and_step_strict_8_of_8": native_qualified,
        "cross_backend_trial_classification_exact_match": classification_match,
        "ready_to_propose_geometry_contract": (
            mjwarp_qualified and native_qualified and classification_match
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "r0c_ellipsoid_collision_dual_backend_development_probe",
        "evidence_eligible": False,
        "promotion_eligible": False,
        "formal_protocol_modified": False,
        "reason": "single no-sweep collision representation rejection test",
        "git_sha": _git_value("rev-parse", "HEAD"),
        "project_dirty": bool(_git_value("status", "--porcelain")),
        "device": args.device,
        "source_result": str(source_path),
        "source_result_sha256": _sha256(source_path),
        "matched_reviewed_resets": True,
        "strict_verdict_modified": False,
        "controller_modified": False,
        "geometry_contract": geometry_contract,
        "model": model_metadata,
        "model_equivalence": model_equivalence,
        "candidate": {
            "name": "axis_preserving_wheel_ellipsoid",
            "shape_parameter_sweep": False,
            "visual_geometry_modified": False,
            "rigid_body_inertia_modified": False,
        },
        "protocol": {
            "heights_m": list(HEIGHTS_M),
            "envs_per_height": diag.R0C_SYNC_ENVS_PER_HEIGHT,
            "settle_steps": rb.OFFICIAL_SETTLE_STEPS,
            "drive_steps": rb.OFFICIAL_DRIVE_STEPS,
            "stable_steps": rb.OFFICIAL_STABLE_STEPS,
            "command_vx_mps": 0.07,
        },
        "acceptance": acceptance,
        "mjwarp": {
            "summaries": mjwarp_summaries,
            "trials": mjwarp_rows,
        },
        "native": {
            "summaries": native_summaries,
            "trials": native_rows,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Refusing to overwrite ellipsoid probe output: {args.output}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_probe(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rb._atomic_write_json(output, payload)
    print(f"[ellipsoid-collision] output={output}")


if __name__ == "__main__":
    main()
