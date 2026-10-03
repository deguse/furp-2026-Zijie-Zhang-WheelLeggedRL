"""Shadow-check bilateral load allocation on all captured R0c failure states.

The probe applies no action. It reconstructs each of the fourteen reviewed
MJWarp zero-force pre-states in native MuJoCo, computes native contact points
and Jacobians, and asks whether a 5 N bilateral support margin, total static
weight, and zero roll moment fit within rated and peak motor torque envelopes.
This is a necessary feasibility test for torque/WBC development, not evidence
that a closed-loop controller already succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from mjlab.envs import ManagerBasedRlEnv

from hoppertrex_mjlab.assets.HopperTrex_CFG import (
    DM_J6248P_PEAK_TORQUE,
    DM_J6248P_RATED_TORQUE,
    RMD_L_9025_35T_PEAK_TORQUE,
    RMD_L_9025_35T_RATED_TORQUE,
)
from hoppertrex_mjlab.hybrid.support_load_allocator import (
    SupportLoadAllocatorConfig,
    allocate_support_loads,
)
from hoppertrex_mjlab.scripts import diagnose_roll_boundary as diag
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb
from hoppertrex_mjlab.scripts.probe_r0c_native_contact_replay import (
    _load_native_state,
    _native_wheel_forces,
    _terrain_geom_ids,
    _wheel_geom_ids,
)

SCHEMA_VERSION = 1
HEIGHTS_M = (0.0, 0.0025)
MINIMUM_SUPPORT_MARGIN_N = 5.0
_SIDE_JOINTS = {
    "left": ("thigh_left_01", "knee_left", "wheel_left"),
    "right": ("thigh_right_01", "knee_right", "wheel_right"),
}
_RATED_LIMITS_NM = (
    DM_J6248P_RATED_TORQUE,
    DM_J6248P_RATED_TORQUE,
    RMD_L_9025_35T_RATED_TORQUE,
)
_PEAK_LIMITS_NM = (
    DM_J6248P_PEAK_TORQUE,
    DM_J6248P_PEAK_TORQUE,
    RMD_L_9025_35T_PEAK_TORQUE,
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


def _joint_dof_ids(model: mujoco.MjModel, names: Sequence[str]) -> tuple[int, ...]:
    result = []
    for name in names:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{name}"
        )
        if joint_id < 0:
            raise ValueError(f"Could not resolve joint {name!r}.")
        result.append(int(model.jnt_dofadr[joint_id]))
    return tuple(result)


def _positive_contact_point(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    wheel_geom_id: int,
    terrain_geom_ids: frozenset[int],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    points = []
    weights = []
    contacts = []
    wrench = np.zeros(6, dtype=np.float64)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        if contact.geom1 == wheel_geom_id:
            other = int(contact.geom2)
        elif contact.geom2 == wheel_geom_id:
            other = int(contact.geom1)
        else:
            continue
        if other not in terrain_geom_ids:
            continue
        wrench.fill(0.0)
        mujoco.mj_contactForce(model, data, contact_id, wrench)
        force = float(np.linalg.norm(wrench[:3]))
        contacts.append(
            {
                "contact_id": contact_id,
                "other_geom_id": other,
                "force_norm_n": force,
                "distance_m": float(contact.dist),
                "position_m": np.asarray(contact.pos).tolist(),
            }
        )
        if force > 0.0:
            points.append(np.asarray(contact.pos).copy())
            weights.append(force)
    if not points:
        raise RuntimeError("Native replay side has no positive wheel-terrain contact.")
    return np.average(np.stack(points), axis=0, weights=np.asarray(weights)), contacts


def _vertical_contact_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    wheel_geom_id: int,
    point: np.ndarray,
) -> np.ndarray:
    jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
    body_id = int(model.geom_bodyid[wheel_geom_id])
    mujoco.mj_jac(
        model,
        data,
        jacobian_position,
        jacobian_rotation,
        point,
        body_id,
    )
    return jacobian_position[2].copy()


def _force_interval(
    bias: np.ndarray,
    vertical_jacobian: np.ndarray,
    dof_ids: Sequence[int],
    torque_limits_nm: Sequence[float],
) -> tuple[float, float]:
    if len(dof_ids) != len(torque_limits_nm):
        raise ValueError("Torque-limit and DOF counts differ.")
    lower = 0.0
    upper = math.inf
    for dof_id, limit in zip(dof_ids, torque_limits_nm, strict=True):
        coefficient = float(vertical_jacobian[dof_id])
        offset = float(bias[dof_id])
        if not math.isfinite(limit) or limit <= 0.0:
            raise ValueError("Torque limits must be finite and positive.")
        if abs(coefficient) <= 1.0e-12:
            if abs(offset) > limit:
                return math.inf, -math.inf
            continue
        roots = sorted(((offset - limit) / coefficient, (offset + limit) / coefficient))
        lower = max(lower, roots[0])
        upper = min(upper, roots[1])
    return max(0.0, lower), upper


def _actuator_torques(
    data: mujoco.MjData,
    left_jacobian: np.ndarray,
    right_jacobian: np.ndarray,
    left_load_n: float,
    right_load_n: float,
) -> np.ndarray:
    return (
        np.asarray(data.qfrc_bias).copy()
        - left_jacobian * left_load_n
        - right_jacobian * right_load_n
    )


def _envelope_result(
    *,
    name: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_jacobian: np.ndarray,
    right_jacobian: np.ndarray,
    left_dofs: tuple[int, ...],
    right_dofs: tuple[int, ...],
    torque_limits: tuple[float, ...],
    track_width_m: float,
    weight_n: float,
) -> dict[str, Any]:
    left_interval = _force_interval(
        data.qfrc_bias, left_jacobian, left_dofs, torque_limits
    )
    right_interval = _force_interval(
        data.qfrc_bias, right_jacobian, right_dofs, torque_limits
    )
    minimum = max(
        MINIMUM_SUPPORT_MARGIN_N,
        left_interval[0],
        right_interval[0],
    )
    maximum = min(left_interval[1], right_interval[1])
    if not math.isfinite(maximum) or maximum < minimum:
        return {
            "name": name,
            "feasible": False,
            "left_force_interval_n": list(left_interval),
            "right_force_interval_n": list(right_interval),
            "reason": "no shared per-side force interval",
        }
    config = SupportLoadAllocatorConfig(
        track_width_m=track_width_m,
        minimum_contact_load_n=minimum,
        maximum_contact_load_n=maximum,
        total_load_weight=1.0 / weight_n**2,
        roll_moment_weight=1.0 / (0.5 * track_width_m * weight_n) ** 2,
    )
    allocation = allocate_support_loads(
        config,
        desired_total_load_n=weight_n,
        desired_roll_moment_nm=0.0,
        left_in_contact=True,
        right_in_contact=True,
    )
    torques = _actuator_torques(
        data,
        left_jacobian,
        right_jacobian,
        allocation.left_load_n,
        allocation.right_load_n,
    )
    actuator_dofs = (*left_dofs, *right_dofs)
    actuator_limits = (*torque_limits, *torque_limits)
    values = [float(torques[dof_id]) for dof_id in actuator_dofs]
    utilization = [
        abs(value) / limit
        for value, limit in zip(values, actuator_limits, strict=True)
    ]
    feasible = (
        allocation.bilateral_minimum_satisfied
        and max(utilization) <= 1.0 + 1.0e-9
        and abs(allocation.total_load_residual_n) <= 1.0e-6
        and abs(allocation.roll_moment_residual_nm) <= 1.0e-6
    )
    return {
        "name": name,
        "feasible": feasible,
        "left_force_interval_n": list(left_interval),
        "right_force_interval_n": list(right_interval),
        "shared_force_bounds_n": [minimum, maximum],
        "allocation": {
            "left_load_n": allocation.left_load_n,
            "right_load_n": allocation.right_load_n,
            "achieved_total_load_n": allocation.achieved_total_load_n,
            "achieved_roll_moment_nm": allocation.achieved_roll_moment_nm,
            "bilateral_minimum_satisfied": allocation.bilateral_minimum_satisfied,
        },
        "actuator_dof_ids": list(actuator_dofs),
        "actuator_torques_nm": values,
        "actuator_torque_limits_nm": list(actuator_limits),
        "maximum_torque_utilization": max(utilization),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.source_result.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("captured_event_count") != 14:
        raise ValueError("Support-load shadow requires all fourteen captured events.")
    if source.get("native_during_step_bilateral_zero_count") != 0:
        raise ValueError("Source replay does not certify native support at every event.")

    env = ManagerBasedRlEnv(
        cfg=rb.make_roll_boundary_env_cfg(
            HEIGHTS_M, diag.R0C_SYNC_ENVS_PER_HEIGHT
        ),
        device=args.device,
    )
    try:
        model = env.sim.mj_model
        wheel_ids = _wheel_geom_ids(model)
        terrain_ids = _terrain_geom_ids(model)
        left_dofs = _joint_dof_ids(model, _SIDE_JOINTS["left"])
        right_dofs = _joint_dof_ids(model, _SIDE_JOINTS["right"])
        weight_n = float(np.sum(model.body_mass) * abs(model.opt.gravity[2]))
        rows = []
        for replay in source["replays"]:
            capture = replay["capture"]
            data = mujoco.MjData(model)
            _load_native_state(model, data, capture)
            mujoco.mj_forward(model, data)
            support = _native_wheel_forces(
                model, data, wheel_ids, terrain_ids
            )
            if bool(support["bilateral_zero_force"]):
                raise RuntimeError("Native source state unexpectedly lost support.")
            left_point, left_contacts = _positive_contact_point(
                model, data, wheel_ids[0], terrain_ids
            )
            right_point, right_contacts = _positive_contact_point(
                model, data, wheel_ids[1], terrain_ids
            )
            left_jacobian = _vertical_contact_jacobian(
                model, data, wheel_ids[0], left_point
            )
            right_jacobian = _vertical_contact_jacobian(
                model, data, wheel_ids[1], right_point
            )
            track_width = abs(
                float(data.geom_xpos[wheel_ids[0], 1])
                - float(data.geom_xpos[wheel_ids[1], 1])
            )
            rated = _envelope_result(
                name="rated",
                model=model,
                data=data,
                left_jacobian=left_jacobian,
                right_jacobian=right_jacobian,
                left_dofs=left_dofs,
                right_dofs=right_dofs,
                torque_limits=_RATED_LIMITS_NM,
                track_width_m=track_width,
                weight_n=weight_n,
            )
            peak = _envelope_result(
                name="peak",
                model=model,
                data=data,
                left_jacobian=left_jacobian,
                right_jacobian=right_jacobian,
                left_dofs=left_dofs,
                right_dofs=right_dofs,
                torque_limits=_PEAK_LIMITS_NM,
                track_width_m=track_width,
                weight_n=weight_n,
            )
            rows.append(
                {
                    "env_id": int(replay["env_id"]),
                    "event_index_within_env": int(
                        replay["event_index_within_env"]
                    ),
                    "event_substep": int(capture["event_substep"]),
                    "track_width_m": track_width,
                    "robot_weight_n": weight_n,
                    "native_current_left_force_n": float(
                        support["left_force_norm_n"]
                    ),
                    "native_current_right_force_n": float(
                        support["right_force_norm_n"]
                    ),
                    "left_contact_point_m": left_point.tolist(),
                    "right_contact_point_m": right_point.tolist(),
                    "left_contacts": left_contacts,
                    "right_contacts": right_contacts,
                    "rated_envelope": rated,
                    "peak_envelope": peak,
                }
            )
    finally:
        env.close()

    rated_count = sum(bool(row["rated_envelope"]["feasible"]) for row in rows)
    peak_count = sum(bool(row["peak_envelope"]["feasible"]) for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "r0c_support_load_allocation_shadow_feasibility_probe",
        "evidence_eligible": False,
        "promotion_eligible": False,
        "formal_protocol_modified": False,
        "controller_modified": False,
        "actuator_modified": False,
        "git_sha": _git_value("rev-parse", "HEAD"),
        "project_dirty": bool(_git_value("status", "--porcelain")),
        "device": args.device,
        "source_result": str(source_path),
        "source_result_sha256": _sha256(source_path),
        "event_count": len(rows),
        "minimum_support_margin_n": MINIMUM_SUPPORT_MARGIN_N,
        "rated_feasible_event_count": rated_count,
        "peak_feasible_event_count": peak_count,
        "all_events_rated_feasible": rated_count == len(rows),
        "all_events_peak_feasible": peak_count == len(rows),
        "scope_limit": (
            "static qdd=0 vertical-load feasibility at captured pre-states; "
            "not a closed-loop success claim"
        ),
        "events": rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Refusing to overwrite support-load output: {args.output}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_probe(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rb._atomic_write_json(output, payload)
    print(f"[support-load-feasibility] output={output}")


if __name__ == "__main__":
    main()
