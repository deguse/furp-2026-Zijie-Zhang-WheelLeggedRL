"""Evaluate one explicit bilateral-load effort controller on both R0c backends.

Development-only: only the six actuator control laws become direct effort in a
probe-local config. Exact resets, contacts, physics, and strict 5 ms verdict
remain unchanged. No parameter sweep is exposed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import types
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from mjlab.actuator import BuiltinMotorActuatorCfg
from mjlab.envs import ManagerBasedRlEnv

from hoppertrex_mjlab.assets.HopperTrex_CFG import (
    DM_J6248P_PEAK_TORQUE,
    DM_J6248P_RATED_TORQUE,
    LEG_JOINT_NAMES,
    RMD_L_9025_35T_PEAK_TORQUE,
    WHEEL_JOINT_NAMES,
)
from hoppertrex_mjlab.hybrid.support_transfer import (
    SupportTransferConfig,
    SupportTransferPhase,
)
from hoppertrex_mjlab.hybrid.whole_body_effort import (
    WholeBodyEffortConfig,
    WholeBodyEffortResult,
    compute_whole_body_effort,
)
from hoppertrex_mjlab.scripts import diagnose_roll_boundary as diag
from hoppertrex_mjlab.scripts import probe_r0c_support_transfer_shadow as shadow_probe
from hoppertrex_mjlab.scripts import probe_r0c_yaw_feedback as lateral
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb
from hoppertrex_mjlab.scripts.probe_r0c_native_contact_replay import (
    _model_equivalence_report,
    _native_wheel_forces,
    _terrain_geom_ids,
    _wheel_geom_ids,
)
from hoppertrex_mjlab.scripts.probe_r0c_native_full_rollout import (
    TrialAccumulator,
    _apply_controller_packet,
    _finite_or_none,
    _native_data_from_controller_state,
    _native_non_wheel_contacts,
    _non_wheel_geom_ids,
    _root_reset_row,
    _sync_native_state_to_controller,
)
from hoppertrex_mjlab.tasks.hoppertrex_balance_task import (
    BAD_ORIENTATION_LIMIT_ANGLE,
    ROOT_HEIGHT_HARD_MIN,
)

SCHEMA_VERSION = 2
HEIGHTS_M = (0.0, 0.0025)
CONTROL_DECIMATION = 4
EXPECTED_SOURCE_SHA256 = "dd8826dcb1abd77ddeb1c68d2bef1406d74b045365d49915d672379e74267c14"
_CAUSAL_FILE_PATHS = (
    "src/hoppertrex_mjlab/assets/HopperTrex_CFG.py",
    "src/hoppertrex_mjlab/deploy/hal.py",
    "src/hoppertrex_mjlab/deploy/safety.py",
    "src/hoppertrex_mjlab/deploy/motor_protocols.py",
    "src/hoppertrex_mjlab/hybrid/support_load_allocator.py",
    "src/hoppertrex_mjlab/hybrid/support_transfer.py",
    "src/hoppertrex_mjlab/hybrid/whole_body_effort.py",
    "src/hoppertrex_mjlab/tasks/hoppertrex_balance_task.py",
    "src/hoppertrex_mjlab/tasks/hoppertrex_hybrid_task.py",
    "src/hoppertrex_mjlab/scripts/probe_roll_boundary.py",
    "src/hoppertrex_mjlab/scripts/diagnose_roll_boundary.py",
    "src/hoppertrex_mjlab/scripts/probe_r0c_native_contact_replay.py",
    "src/hoppertrex_mjlab/scripts/probe_r0c_native_full_rollout.py",
    "src/hoppertrex_mjlab/scripts/probe_r0c_support_transfer_shadow.py",
    "src/hoppertrex_mjlab/scripts/probe_r0c_yaw_feedback.py",
    "src/hoppertrex_mjlab/scripts/probe_r0c_effort_wbc.py",
)
_ALLOWED_ACTUATOR_ARRAYS = (
    "actuator_actearly", "actuator_actlimited", "actuator_biastype",
    "actuator_ctrllimited", "actuator_dyntype", "actuator_forcelimited",
    "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange",
    "actuator_forcerange", "actuator_dynprm", "actuator_gaintype",
    "actuator_gear", "actuator_group", "actuator_trnid", "actuator_trntype",
)
_UNCHANGED_MODEL_ARRAYS = (
    "body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos",
    "body_quat", "jnt_type", "jnt_bodyid", "jnt_qposadr", "jnt_dofadr",
    "jnt_pos", "jnt_axis", "jnt_range", "dof_damping", "dof_frictionloss",
    "dof_armature", "geom_type", "geom_bodyid", "geom_pos", "geom_quat",
    "geom_size", "geom_friction", "geom_solref", "geom_solimp",
    "geom_margin", "geom_gap", "geom_condim", "geom_contype",
    "geom_conaffinity", "geom_priority", "sensor_type", "sensor_objtype",
    "sensor_objid", "sensor_dim", "sensor_adr", "sensor_cutoff", "sensor_noise",
)
_OPTION_FIELDS = (
    "timestep", "integrator", "cone", "jacobian", "solver", "iterations",
    "ls_iterations", "tolerance", "ls_tolerance", "disableflags", "enableflags",
)


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=rb.REPOSITORY_PATH, check=True,
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _causal_file_manifest() -> dict[str, dict[str, int | str]]:
    manifest: dict[str, dict[str, int | str]] = {}
    for relative in _CAUSAL_FILE_PATHS:
        path = rb.REPOSITORY_PATH / relative
        if not path.is_file():
            raise FileNotFoundError(f"Causal artifact input is missing: {path}")
        manifest[relative] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return manifest


def _assert_causal_manifest_unchanged(
    before: Mapping[str, Any], after: Mapping[str, Any],
) -> None:
    if before == after:
        return
    changed = sorted(
        key for key in set(before) | set(after)
        if before.get(key) != after.get(key)
    )
    raise RuntimeError(
        "Causal files changed while the effort-WBC probe was running: "
        + ", ".join(changed)
    )


def _validate_full_run_sanity_artifact(
    path: Path,
    *,
    expected_sha256: str,
    source_sha256: str,
    current_manifest: Mapping[str, Any],
    current_git_sha: str,
) -> dict[str, Any]:
    normalized_sha = expected_sha256.strip().lower()
    if len(normalized_sha) != 64 or any(
        character not in "0123456789abcdef" for character in normalized_sha
    ):
        raise ValueError("Validated sanity SHA256 must contain 64 hex digits.")
    actual_sha = _sha256(path)
    if actual_sha != normalized_sha:
        raise ValueError(
            f"Validated sanity SHA256 mismatch: expected {normalized_sha}, "
            f"got {actual_sha}."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = (
        (payload.get("schema_version") == SCHEMA_VERSION, "schema version"),
        (
            payload.get("kind")
            == "r0c_effort_wbc_dual_backend_development_probe",
            "artifact kind",
        ),
        (payload.get("mode") == "sanity", "sanity mode"),
        (payload.get("source_result_sha256") == source_sha256, "source SHA256"),
        (payload.get("git_sha") == current_git_sha, "Git HEAD"),
        (
            payload.get("acceptance", {}).get("all_required_checks_pass") is True,
            "passing acceptance",
        ),
        (
            payload.get("causal_provenance_gate_pass") is True,
            "causal provenance gate",
        ),
        (
            payload.get("causal_file_manifest") == current_manifest,
            "current causal manifest",
        ),
        (
            payload.get("causal_file_manifest_pre") == current_manifest,
            "pre-run causal manifest",
        ),
        (
            payload.get("causal_file_manifest_post") == current_manifest,
            "post-run causal manifest",
        ),
    )
    failed = [label for passed, label in checks if not passed]
    if failed:
        raise ValueError(
            "Validated sanity artifact failed the full-run gate: "
            + ", ".join(failed)
        )
    return {
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "schema_version": SCHEMA_VERSION,
        "source_result_sha256": source_sha256,
        "git_sha": current_git_sha,
        "causal_manifest_verified": True,
        "all_required_checks_pass": True,
    }


def _dirty_snapshot_sha256() -> str:
    digest = hashlib.sha256()
    digest.update(subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=rb.REPOSITORY_PATH,
        check=True, capture_output=True,
    ).stdout)
    entries = subprocess.run(
        ["git", "status", "--porcelain", "-z"], cwd=rb.REPOSITORY_PATH,
        check=True, capture_output=True,
    ).stdout.split(b"\0")
    for entry in sorted(value for value in entries if value.startswith(b"?? ")):
        relative = entry[3:].decode("utf-8")
        path = rb.REPOSITORY_PATH / relative
        if path.is_file():
            digest.update(relative.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _make_effort_cfg(envs_per_height: int) -> Any:
    cfg = rb.make_roll_boundary_env_cfg(HEIGHTS_M, envs_per_height)
    robot_cfg = cfg.scene.entities["robot"]
    if robot_cfg.articulation is None:
        raise RuntimeError("HopperTrex unexpectedly has no articulation config.")
    robot_cfg.articulation.actuators = (
        BuiltinMotorActuatorCfg(
            target_names_expr=LEG_JOINT_NAMES, effort_limit=DM_J6248P_PEAK_TORQUE,
            armature=0.02, frictionloss=0.0, viscous_damping=0.0,
        ),
        BuiltinMotorActuatorCfg(
            target_names_expr=WHEEL_JOINT_NAMES,
            effort_limit=RMD_L_9025_35T_PEAK_TORQUE,
            armature=0.005, frictionloss=0.0, viscous_damping=0.0,
        ),
    )
    return cfg


def _model_snapshot(model: mujoco.MjModel) -> dict[str, Any]:
    return {
        "dimensions": {name: int(getattr(model, name)) for name in
                       ("nq", "nv", "nu", "na", "nbody", "njnt", "ngeom", "nsensor")},
        "names": bytes(model.names),
        "gravity": np.asarray(model.opt.gravity).copy(),
        "options": {name: getattr(model.opt, name) for name in _OPTION_FIELDS},
        "unchanged": {name: np.asarray(getattr(model, name)).copy()
                      for name in _UNCHANGED_MODEL_ARRAYS},
        "actuator": {name: np.asarray(getattr(model, name)).copy()
                     for name in _ALLOWED_ACTUATOR_ARRAYS},
    }


def _max_diff(left: np.ndarray, right: np.ndarray) -> float:
    return 0.0 if left.size == 0 else float(np.max(
        np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _compare_actuator_only_contract(
    baseline: Mapping[str, Any], effort_model: mujoco.MjModel,
) -> dict[str, Any]:
    candidate = _model_snapshot(effort_model)
    if candidate["dimensions"] != baseline["dimensions"]:
        raise RuntimeError("Effort candidate changed model dimensions.")
    if candidate["names"] != baseline["names"]:
        raise RuntimeError("Effort candidate changed names/order.")
    if not np.array_equal(candidate["gravity"], baseline["gravity"]):
        raise RuntimeError("Effort candidate changed gravity.")
    if candidate["options"] != baseline["options"]:
        raise RuntimeError("Effort candidate changed model options.")
    unchanged_report = {}
    for name in _UNCHANGED_MODEL_ARRAYS:
        before, after = baseline["unchanged"][name], candidate["unchanged"][name]
        equal = bool(np.array_equal(before, after))
        unchanged_report[name] = {"equal": equal, "maximum_abs_difference": _max_diff(before, after)}
        if not equal:
            raise RuntimeError(f"Effort candidate changed non-actuator field {name}.")
    actuator_report, changed = {}, []
    for name in _ALLOWED_ACTUATOR_ARRAYS:
        before, after = baseline["actuator"][name], candidate["actuator"][name]
        equal = bool(np.array_equal(before, after))
        actuator_report[name] = {"equal": equal, "maximum_abs_difference": _max_diff(before, after)}
        if not equal:
            changed.append(name)
    expected = np.asarray([
        [-97.0, 97.0], [-97.0, 97.0], [-5.8, 5.8],
        [-97.0, 97.0], [-97.0, 97.0], [-5.8, 5.8],
    ])
    if not np.array_equal(effort_model.actuator_ctrlrange, expected):
        raise RuntimeError("Effort ctrl ranges differ from physical peaks.")
    if not np.array_equal(effort_model.actuator_forcerange, expected):
        raise RuntimeError("Effort force ranges differ from physical peaks.")
    if not np.array_equal(effort_model.actuator_gainprm[:, 0], np.ones(6)):
        raise RuntimeError("Effort motor gain is not one.")
    if not np.array_equal(effort_model.actuator_biasprm,
                          np.zeros_like(effort_model.actuator_biasprm)):
        raise RuntimeError("Effort motor has a hidden bias.")
    if not changed:
        raise RuntimeError("No actuator-law change was observed.")
    return {
        "only_actuator_arrays_may_differ": True,
        "all_model_options_unchanged": True,
        "all_geometry_contact_inertia_joint_sensor_arrays_unchanged": True,
        "unchanged_arrays": unchanged_report,
        "actuator_arrays": actuator_report,
        "changed_actuator_fields": changed,
        "expected_effort_ctrl_ranges_verified": True,
    }


def _name_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    result = mujoco.mj_name2id(model, object_type, f"robot/{name}")
    if result < 0:
        raise ValueError(f"Could not resolve robot/{name}.")
    return int(result)


@dataclass(frozen=True)
class _ModelIndices:
    leg_dof_ids: tuple[int, int, int, int]
    wheel_dof_ids: tuple[int, int]
    wheel_body_ids: tuple[int, int]
    roll_dof_id: int


def _model_indices(model: mujoco.MjModel) -> _ModelIndices:
    def joint_dof(name: str) -> int:
        jid = _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(model.jnt_dofadr[jid])
    leg = tuple(joint_dof(name) for name in LEG_JOINT_NAMES)
    wheel_jids = tuple(_name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                       for name in WHEEL_JOINT_NAMES)
    free = np.flatnonzero(model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))
    if len(free) != 1:
        raise ValueError("Effort controller requires one free root joint.")
    return _ModelIndices(
        leg_dof_ids=leg,  # type: ignore[arg-type]
        wheel_dof_ids=tuple(int(model.jnt_dofadr[jid]) for jid in wheel_jids),  # type: ignore[arg-type]
        wheel_body_ids=tuple(int(model.jnt_bodyid[jid]) for jid in wheel_jids),  # type: ignore[arg-type]
        roll_dof_id=int(model.jnt_dofadr[int(free[0])]) + 3,
    )


def _vertical_body_jacobian(
    model: mujoco.MjModel, data: mujoco.MjData, body_id: int,
) -> np.ndarray:
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jac(model, data, jacp, jacr, np.asarray(data.xpos[body_id]), body_id)
    return jacp[2].copy()

@dataclass
class _EffortMetrics:
    maximum_abs_leg_torque_nm: float = 0.0
    maximum_abs_wheel_torque_nm: float = 0.0
    leg_saturated_samples: int = 0
    wheel_saturated_samples: int = 0
    wheel_above_rated_samples: int = 0
    wheel_total_samples: int = 0
    wheel_peak_current_run: list[int] = field(default_factory=lambda: [0, 0])
    wheel_peak_max_run: list[int] = field(default_factory=lambda: [0, 0])
    minimum_target_left_load_n: float = math.inf
    minimum_target_right_load_n: float = math.inf
    maximum_total_load_residual_n: float = 0.0
    maximum_roll_moment_residual_nm: float = 0.0
    traces: list[dict[str, Any]] = field(default_factory=list)

    def observe(
        self, result: WholeBodyEffortResult, *, control_step: int,
        physics_substep: int, active: bool,
    ) -> None:
        self.maximum_abs_leg_torque_nm = max(
            self.maximum_abs_leg_torque_nm,
            float(np.max(np.abs(result.leg_effort_nm))))
        self.maximum_abs_wheel_torque_nm = max(
            self.maximum_abs_wheel_torque_nm,
            float(np.max(np.abs(result.wheel_effort_nm))))
        self.leg_saturated_samples += int(np.sum(result.leg_saturated))
        self.wheel_saturated_samples += int(np.sum(result.wheel_saturated))
        self.wheel_above_rated_samples += int(np.sum(result.wheel_above_rated))
        self.wheel_total_samples += 2
        for index, above in enumerate(result.wheel_above_rated):
            self.wheel_peak_current_run[index] = (
                self.wheel_peak_current_run[index] + 1 if bool(above) else 0)
            self.wheel_peak_max_run[index] = max(
                self.wheel_peak_max_run[index], self.wheel_peak_current_run[index])
        allocation = result.allocation
        self.minimum_target_left_load_n = min(
            self.minimum_target_left_load_n, allocation.left_load_n)
        self.minimum_target_right_load_n = min(
            self.minimum_target_right_load_n, allocation.right_load_n)
        self.maximum_total_load_residual_n = max(
            self.maximum_total_load_residual_n,
            abs(allocation.total_load_residual_n))
        self.maximum_roll_moment_residual_nm = max(
            self.maximum_roll_moment_residual_nm,
            abs(allocation.roll_moment_residual_nm))
        self.traces.append({
            "control_step": control_step,
            "physics_substep_within_control": physics_substep,
            "phase": "settle" if control_step <= rb.OFFICIAL_SETTLE_STEPS else "drive",
            "active": active,
            "support_transfer_constraint": "BOTH_REQUIRED",
            "left_target_load_n": allocation.left_load_n,
            "right_target_load_n": allocation.right_load_n,
            "total_load_residual_n": allocation.total_load_residual_n,
            "roll_moment_residual_nm": allocation.roll_moment_residual_nm,
            "roll_inertia_kg_m2": result.roll_inertia_kg_m2,
            "desired_roll_moment_unclamped_nm": result.desired_roll_moment_unclamped_nm,
            "desired_roll_moment_nm": result.desired_roll_moment_nm,
            "leg_kp": result.leg_kp.tolist(),
            "leg_kd": result.leg_kd.tolist(),
            "inverse_effort_nm": result.inverse_effort_nm.tolist(),
            "leg_pd_effort_nm": result.leg_pd_effort_nm.tolist(),
            "wheel_drive_effort_nm": result.wheel_drive_effort_nm.tolist(),
            "raw_leg_effort_nm": result.raw_leg_effort_nm.tolist(),
            "raw_wheel_effort_nm": result.raw_wheel_effort_nm.tolist(),
            "leg_effort_nm": result.leg_effort_nm.tolist(),
            "wheel_effort_nm": result.wheel_effort_nm.tolist(),
            "leg_saturated": result.leg_saturated.tolist(),
            "wheel_saturated": result.wheel_saturated.tolist(),
            "wheel_above_rated": result.wheel_above_rated.tolist(),
            "qp_bilateral_minimum_satisfied": allocation.bilateral_minimum_satisfied,
        })

    def summary(self) -> dict[str, Any]:
        return {
            "maximum_abs_leg_torque_nm": self.maximum_abs_leg_torque_nm,
            "maximum_abs_wheel_torque_nm": self.maximum_abs_wheel_torque_nm,
            "leg_saturated_samples": self.leg_saturated_samples,
            "wheel_saturated_samples": self.wheel_saturated_samples,
            "wheel_above_rated_samples": self.wheel_above_rated_samples,
            "wheel_total_samples": self.wheel_total_samples,
            "wheel_above_rated_fraction": (
                self.wheel_above_rated_samples / self.wheel_total_samples
                if self.wheel_total_samples else 0.0),
            "wheel_peak_max_consecutive_substeps": self.wheel_peak_max_run,
            "wheel_peak_max_consecutive_duration_s": [
                run * rb.ROLL_FIRST_PHYSICS_TIMESTEP_S
                for run in self.wheel_peak_max_run],
            "metric_aggregation_window": (
                "fixed_settle_plus_drive_all_physics_substeps_including_inactive"
            ),
            "controller_samples": len(self.traces),
            "active_controller_samples": sum(
                int(bool(row["active"])) for row in self.traces
            ),
            "minimum_target_left_load_n": _finite_or_none(self.minimum_target_left_load_n),
            "minimum_target_right_load_n": _finite_or_none(self.minimum_target_right_load_n),
            "maximum_total_load_residual_n": self.maximum_total_load_residual_n,
            "maximum_roll_moment_residual_nm": self.maximum_roll_moment_residual_nm,
            "control_trace": self.traces,
        }


class _WholeBodyEffortAdapter:
    """Replace one action-term instance's apply phase with 200 Hz effort."""

    def __init__(self, env: ManagerBasedRlEnv, config: WholeBodyEffortConfig) -> None:
        self.env, self.config = env, config
        self.model = env.sim.mj_model
        self.term = env.action_manager.get_term("hybrid_wheel_leg")
        if int(self.term._delay_steps) != 0:
            raise ValueError("Effort candidate does not permit delayed outer targets.")
        self.robot = env.scene["robot"]
        self.indices = _model_indices(self.model)
        self.native_data = [mujoco.MjData(self.model) for _ in range(env.num_envs)]
        self.original_apply = self.term.apply_actions
        self.call_index = 0
        self.metrics = [_EffortMetrics() for _ in range(env.num_envs)]
        self.active_mask = np.ones(env.num_envs, dtype=bool)
        self._held_leg_targets: np.ndarray | None = None
        self._held_wheel_targets: np.ndarray | None = None

    def install(self) -> None:
        adapter = self
        def apply_effort(_term: Any) -> None:
            adapter.apply()
        self.term.apply_actions = types.MethodType(apply_effort, self.term)

    def restore(self) -> None:
        self.term.apply_actions = self.original_apply

    def reset_metrics(self) -> None:
        self.call_index = 0
        self.metrics = [_EffortMetrics() for _ in range(self.env.num_envs)]
        self.active_mask = np.ones(self.env.num_envs, dtype=bool)
        self._held_leg_targets = None
        self._held_wheel_targets = None

    def set_active_mask(self, active: torch.Tensor | np.ndarray) -> None:
        values = (active.detach().cpu().numpy() if isinstance(active, torch.Tensor)
                  else np.asarray(active))
        if values.shape != (self.env.num_envs,):
            raise ValueError("Effort adapter active-mask shape mismatch.")
        self.active_mask = values.astype(bool, copy=True)

    def apply(self) -> None:
        self.call_index += 1
        substep = (self.call_index - 1) % CONTROL_DECIMATION + 1
        control_step = (self.call_index - 1) // CONTROL_DECIMATION + 1
        qpos = self.env.sim.data.qpos.detach().cpu().numpy()
        qvel = self.env.sim.data.qvel.detach().cpu().numpy()
        times = self.env.sim.data.time.detach().cpu().numpy()
        leg_targets = self.term._leg_targets.detach().cpu().numpy().copy()
        wheel_targets = self.term._wheel_targets.detach().cpu().numpy().copy()
        if substep == 1:
            self._held_leg_targets, self._held_wheel_targets = (
                leg_targets.copy(), wheel_targets.copy())
        elif (self._held_leg_targets is None or self._held_wheel_targets is None
              or not np.array_equal(leg_targets, self._held_leg_targets)
              or not np.array_equal(wheel_targets, self._held_wheel_targets)):
            raise RuntimeError("50 Hz outer targets changed inside a 5 ms quartet.")
        leg_positions = self.robot.data.joint_pos[:, self.term._leg_ids].detach().cpu().numpy()
        leg_velocities = self.robot.data.joint_vel[:, self.term._leg_ids].detach().cpu().numpy()
        wheel_velocities = self.robot.data.joint_vel[:, self.term._wheel_ids].detach().cpu().numpy()
        gravity = self.robot.data.projected_gravity_b.detach().cpu().numpy()
        root_ang_vel = self.robot.data.root_link_ang_vel_b.detach().cpu().numpy()
        roll = np.arctan2(-gravity[:, 1], np.maximum(-gravity[:, 2], 1.0e-6))
        roll_rate = root_ang_vel[:, 0]
        leg_effort = np.zeros((self.env.num_envs, 4), dtype=np.float64)
        wheel_effort = np.zeros((self.env.num_envs, 2), dtype=np.float64)
        for env_id, data in enumerate(self.native_data):
            data.qpos[:], data.qvel[:], data.time = qpos[env_id], qvel[env_id], float(times[env_id])
            mujoco.mj_forward(self.model, data)
            mass = np.zeros((self.model.nv, self.model.nv), dtype=np.float64)
            mujoco.mj_fullM(self.model, mass, data.qM)
            left_jac = _vertical_body_jacobian(self.model, data, self.indices.wheel_body_ids[0])
            right_jac = _vertical_body_jacobian(self.model, data, self.indices.wheel_body_ids[1])
            track = abs(float(data.xpos[self.indices.wheel_body_ids[0], 1])
                        - float(data.xpos[self.indices.wheel_body_ids[1], 1]))
            result = compute_whole_body_effort(
                self.config, mass_matrix=mass, bias_effort=np.asarray(data.qfrc_bias),
                left_vertical_jacobian=left_jac, right_vertical_jacobian=right_jac,
                leg_dof_ids=self.indices.leg_dof_ids,
                wheel_dof_ids=self.indices.wheel_dof_ids,
                roll_dof_id=self.indices.roll_dof_id, track_width_m=track,
                roll_rad=float(roll[env_id]), roll_rate_rad_s=float(roll_rate[env_id]),
                leg_positions_rad=leg_positions[env_id],
                leg_velocities_rad_s=leg_velocities[env_id],
                leg_position_targets_rad=leg_targets[env_id],
                wheel_velocities_rad_s=wheel_velocities[env_id],
                wheel_velocity_targets_rad_s=wheel_targets[env_id])
            leg_effort[env_id], wheel_effort[env_id] = result.leg_effort_nm, result.wheel_effort_nm
            self.metrics[env_id].observe(
                result, control_step=control_step, physics_substep=substep,
                active=bool(self.active_mask[env_id]))
        if not np.isfinite(leg_effort).all() or not np.isfinite(wheel_effort).all():
            raise RuntimeError("Effort adapter generated non-finite torque.")
        dtype = self.robot.data.joint_effort_target.dtype
        self.robot.set_joint_effort_target(
            torch.as_tensor(leg_effort, device=self.env.device, dtype=dtype),
            joint_ids=self.term._leg_ids)
        self.robot.set_joint_effort_target(
            torch.as_tensor(wheel_effort, device=self.env.device, dtype=dtype),
            joint_ids=self.term._wheel_ids)


def _selected_resets(
    source_resets: Mapping[tuple[int, float], Any], *, sanity: bool,
) -> tuple[int, list[Any], list[tuple[int, float]]]:
    if sanity:
        identities = [(0, 0.0), (8, 0.0025)]
        return 1, [source_resets[key] for key in identities], identities
    identities = list(source_resets)
    return (diag.R0C_SYNC_ENVS_PER_HEIGHT,
            shadow_probe._reset_override(source_resets), identities)


def _validate_selected_resets(
    rows: Sequence[Mapping[str, Any]], reset_override: Sequence[Any],
) -> None:
    if len(rows) != len(reset_override):
        raise ValueError("Effort probe produced the wrong reset count.")
    for row, expected in zip(rows, reset_override, strict=True):
        if row["root_reset"] != expected:
            raise ValueError("Effort probe exact reset drifted from source.")


def _attach_effort_metrics(
    rows: Sequence[dict[str, Any]], adapter: _WholeBodyEffortAdapter,
) -> None:
    for row in rows:
        row["whole_body_effort"] = adapter.metrics[int(row["env_id"])].summary()


def _attach_support_shadow(
    rows: Sequence[dict[str, Any]], shadow: shadow_probe.SupportTransferShadow,
) -> None:
    for row in rows:
        env_id = int(row["env_id"])
        state = shadow.states[env_id]
        row["support_transfer_shadow"] = {
            "final_phase": state.phase.name,
            "lead_side": state.lead_side.name,
            "abort_reason": state.abort_reason,
            "travel_since_trigger_m": state.travel_since_trigger_m,
            "phase_substeps": {phase.name: shadow.phase_counts[env_id][int(phase)]
                               for phase in SupportTransferPhase},
            "first_bilateral_zero_after_trigger": shadow.first_bilateral_zero[env_id],
            "bilateral_zero_after_trigger_substeps": shadow.bilateral_zero_after_trigger[env_id],
            "transitions": shadow.transitions[env_id],
        }


def _run_mjwarp_effort(
    env: ManagerBasedRlEnv, *, adapter: _WholeBodyEffortAdapter,
    candidate: Mapping[str, Any], reset_override: Sequence[Any],
) -> list[dict[str, Any]]:
    adapter.reset_metrics()
    shadow = shadow_probe.SupportTransferShadow(
        env, SupportTransferConfig(control_dt_s=rb.ROLL_FIRST_PHYSICS_TIMESTEP_S))
    shadow.install()
    try:
        rows = rb.run_card_repeat(
            env, heights=HEIGHTS_M, card=candidate["posture_card"], repeat=1,
            settle_steps=rb.OFFICIAL_SETTLE_STEPS,
            drive_steps=rb.OFFICIAL_DRIVE_STEPS,
            stable_steps=rb.OFFICIAL_STABLE_STEPS,
            episode_wide_safety=True,
            diagnostic_continue_after_support_loss=True,
            roll_pose_schedule=candidate["schedule"],
            roll_pose_slew_mode=str(candidate["slew_mode"]),
            require_pure_classical_authority=True,
            root_reset_override=reset_override, command_vx_mps=0.07)
    finally:
        shadow.restore()
    _validate_selected_resets(rows, reset_override)
    _attach_effort_metrics(rows, adapter)
    _attach_support_shadow(rows, shadow)
    return rows


def _controller_packet(env: ManagerBasedRlEnv) -> dict[str, np.ndarray]:
    env.action_manager.apply_action()
    env.scene.write_data_to_sim()
    packet = {name: getattr(env.sim.data, name).detach().cpu().numpy().copy()
              for name in ("ctrl", "qfrc_applied", "xfrc_applied")}
    if not all(np.isfinite(values).all() for values in packet.values()):
        raise RuntimeError("Native effort controller packet is non-finite.")
    return packet


def _run_native_effort(
    env: ManagerBasedRlEnv, *, adapter: _WholeBodyEffortAdapter,
    candidate: Mapping[str, Any], reset_override: Sequence[Any],
) -> list[dict[str, Any]]:
    card, schedule = candidate["posture_card"], candidate["schedule"]
    terrain_types, face_x, cross_x, reset = rb._reset_to_approach(
        env, root_height=float(card["height_m"]), card_name=str(card["name"]),
        repeat=1, height_count=len(HEIGHTS_M), reset_override=list(reset_override))
    native_data = _native_data_from_controller_state(env)
    model = env.sim.mj_model
    wheel_geom_ids, terrain_geom_ids = _wheel_geom_ids(model), _terrain_geom_ids(model)
    non_wheel_geom_ids = _non_wheel_geom_ids(model)
    robot = env.scene["robot"]
    term = env.action_manager.get_term("hybrid_wheel_leg")
    wheel_joint_ids = term._wheel_ids
    actions = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    accumulators = [TrialAccumulator() for _ in range(env.num_envs)]
    for env_id, accumulator in enumerate(accumulators):
        accumulator.max_progress_m = float(reset["x_relative_to_face_m"][env_id])
    schedule_state = rb.make_roll_pose_schedule_state(
        schedule, robot.data.root_link_pos_w[:, 0],
        slew_mode=str(candidate["slew_mode"]))
    adapter.reset_metrics()
    global_control_step = 0
    controller_forward_calls = 0

    def control_step(vx: float, drive_index: int | None) -> None:
        nonlocal active, global_control_step, controller_forward_calls
        global_control_step += 1
        was_active = active.clone()
        adapter.set_active_mask(was_active)
        phase = "settle" if drive_index is None else "drive"
        _sync_native_state_to_controller(env, native_data)
        controller_forward_calls += 1
        schedule_output = rb.roll_pose_schedule_step(
            schedule, schedule_state, root_x_m=robot.data.root_link_pos_w[:, 0],
            face_x_m=face_x, active_mask=was_active,
            drive_active=drive_index is not None, dt=rb.ROLL_POSE_CONTROL_DT_S)
        rb._force_commands(
            env, active=was_active, vx=vx,
            height=schedule_output.applied_height_m,
            pitch=schedule_output.applied_pitch_rad)
        previous = term._previous_wheel_targets.detach().clone()
        env.action_manager.process_action(actions)
        classical = torch.clamp(
            previous + torch.clamp(
                term.controller_baseline - previous,
                -term.cfg.wheel_slew_limit, term.cfg.wheel_slew_limit),
            -term.cfg.wheel_velocity_limit, term.cfg.wheel_velocity_limit)
        if float((term.wheel_targets - classical).abs().max()) != 0.0:
            raise RuntimeError("Native effort probe changed outer wheel authority.")
        if (float(term.dynamic_leg_feedforward.abs().max()) != 0.0
                or float(term.dynamic_drive_feedforward.abs().max()) != 0.0
                or float(term.applied_residual.abs().max()) != 0.0):
            raise RuntimeError("Native effort probe observed non-classical authority.")
        for physics_substep in range(1, CONTROL_DECIMATION + 1):
            if physics_substep > 1:
                _sync_native_state_to_controller(env, native_data)
                controller_forward_calls += 1
            packet = _controller_packet(env)
            for env_id, data in enumerate(native_data):
                _apply_controller_packet(data, packet, env_id)
                pre_root_z, pre_root_vz = float(data.qpos[2]), float(data.qvel[2])
                mujoco.mj_step(model, data)
                if not bool(was_active[env_id]):
                    continue
                support = _native_wheel_forces(model, data, wheel_geom_ids, terrain_geom_ids)
                accumulators[env_id].observe_substep_support(
                    left_force_n=float(support["left_force_norm_n"]),
                    right_force_n=float(support["right_force_norm_n"]),
                    phase=phase, control_step=global_control_step,
                    physics_substep=physics_substep, time_s=float(data.time))
                if bool(support["bilateral_zero_force"]):
                    accumulators[env_id].support_events[-1].update({
                        "pre_root_z_m": pre_root_z, "pre_root_vz_mps": pre_root_vz,
                        "post_root_z_m": float(data.qpos[2]),
                        "post_root_vz_mps": float(data.qvel[2]),
                        "native_contact_count": int(data.ncon),
                        "wheel_contacts": support["contacts"],
                    })

        endpoint_support, endpoint_non_wheel = [], []
        for data in native_data:
            mujoco.mj_forward(model, data)
            endpoint_support.append(
                _native_wheel_forces(model, data, wheel_geom_ids, terrain_geom_ids))
            endpoint_non_wheel.append(_native_non_wheel_contacts(
                model, data, non_wheel_geom_ids, terrain_geom_ids))
        _sync_native_state_to_controller(env, native_data)
        controller_forward_calls += 1
        pitch, roll = rb._pitch_roll(robot)
        pitch_rate = robot.data.root_link_ang_vel_b[:, 1]
        projected_gravity = robot.data.projected_gravity_b
        root_pos = robot.data.root_link_pos_w
        progress = root_pos[:, 0] - face_x
        joint_speed = robot.data.joint_vel[:, wheel_joint_ids].detach()
        wheel_target = term.wheel_targets.detach()
        finite_state = (
            torch.isfinite(root_pos).all(dim=1)
            & torch.isfinite(robot.data.root_link_quat_w).all(dim=1)
            & torch.isfinite(robot.data.root_link_lin_vel_w).all(dim=1)
            & torch.isfinite(robot.data.root_link_ang_vel_w).all(dim=1)
            & torch.isfinite(robot.data.joint_pos).all(dim=1)
            & torch.isfinite(robot.data.joint_vel).all(dim=1))
        tilt_cosine = torch.clamp(-projected_gravity[:, 2], -1.0, 1.0)
        bad_orientation = torch.acos(tilt_cosine) > BAD_ORIENTATION_LIMIT_ANGLE
        root_too_low = root_pos[:, 2] < ROOT_HEIGHT_HARD_MIN
        for env_id, accumulator in enumerate(accumulators):
            if not bool(was_active[env_id]):
                continue
            endpoint = endpoint_support[env_id]
            accumulator.observe_endpoint_support(
                left_force_n=float(endpoint["left_force_norm_n"]),
                right_force_n=float(endpoint["right_force_norm_n"]),
                phase=phase, control_step=global_control_step,
                time_s=float(native_data[env_id].time))
            contacts = endpoint_non_wheel[env_id]
            if contacts:
                accumulator.non_wheel_contact = True
                if accumulator.non_wheel_event is None:
                    accumulator.non_wheel_event = {
                        "phase": phase, "control_step": global_control_step,
                        "time_s": float(native_data[env_id].time), "contacts": contacts}
            reasons = []
            if bool(bad_orientation[env_id]):
                reasons.append("bad_orientation")
            if bool(root_too_low[env_id]):
                reasons.append("root_too_low")
            if not bool(finite_state[env_id]):
                reasons.append("non_finite_state")
            if reasons:
                accumulator.termination = True
                for reason in reasons:
                    if reason not in accumulator.termination_reasons:
                        accumulator.termination_reasons.append(reason)
            accumulator.peak_pitch_abs_rad = max(
                accumulator.peak_pitch_abs_rad, abs(float(pitch[env_id])))
            accumulator.peak_roll_abs_rad = max(
                accumulator.peak_roll_abs_rad, abs(float(roll[env_id])))
            accumulator.peak_pitch_rate_abs_radps = max(
                accumulator.peak_pitch_rate_abs_radps,
                abs(float(pitch_rate[env_id])))
            if drive_index is not None:
                accumulator.max_progress_m = max(
                    accumulator.max_progress_m, float(progress[env_id]))
                accumulator.max_wheel_target_abs_radps = max(
                    accumulator.max_wheel_target_abs_radps,
                    float(wheel_target[env_id].abs().max()))
                accumulator.max_wheel_speed_abs_radps = max(
                    accumulator.max_wheel_speed_abs_radps,
                    float(joint_speed[env_id].abs().max()))
                posture_ok = (
                    abs(float(pitch[env_id])) <= rb.PITCH_LIMIT_RAD
                    and abs(float(roll[env_id])) <= rb.ROLL_LIMIT_RAD
                    and abs(float(pitch_rate[env_id])) <= rb.PITCH_RATE_LIMIT_RADPS)
                crossed = float(root_pos[env_id, 0]) >= float(cross_x[env_id])
                strict_valid = not (
                    accumulator.support_failed or accumulator.non_wheel_contact
                    or accumulator.termination)
                geometric_valid = not (
                    accumulator.non_wheel_contact or accumulator.termination)
                accumulator.strict_stable_steps = (
                    accumulator.strict_stable_steps + 1
                    if strict_valid and crossed and posture_ok else 0)
                accumulator.geometric_stable_steps = (
                    accumulator.geometric_stable_steps + 1
                    if geometric_valid and crossed and posture_ok else 0)
                if (not accumulator.strict_success
                        and accumulator.strict_stable_steps >= rb.OFFICIAL_STABLE_STEPS):
                    accumulator.strict_success = True
                    accumulator.strict_success_step = drive_index + 1
                if (not accumulator.geometric_success
                        and accumulator.geometric_stable_steps >= rb.OFFICIAL_STABLE_STEPS):
                    accumulator.geometric_success = True
                    accumulator.geometric_success_step = drive_index + 1
            accumulator.active = not (
                accumulator.termination or accumulator.non_wheel_contact
                or accumulator.strict_success)
            active[env_id] = accumulator.active

    for _ in range(rb.OFFICIAL_SETTLE_STEPS):
        control_step(0.0, None)
    for drive_index in range(rb.OFFICIAL_DRIVE_STEPS):
        control_step(0.07, drive_index)
    rows = []
    for env_id, terrain_type in enumerate(terrain_types.detach().cpu().tolist()):
        accumulator = accumulators[env_id]
        effort = adapter.metrics[env_id].summary()
        wheel_total = int(effort["wheel_total_samples"])
        wheel_saturated = int(effort["wheel_saturated_samples"])
        rows.append({
            "env_id": env_id,
            "stair_height_m": float(HEIGHTS_M[terrain_type]),
            "success": accumulator.strict_success,
            "geometric_success_ignoring_support": accumulator.geometric_success,
            "safe_stall": (not accumulator.strict_success
                           and not accumulator.support_failed
                           and not accumulator.non_wheel_contact
                           and not accumulator.termination),
            "unsafe": (accumulator.support_failed or accumulator.non_wheel_contact
                       or accumulator.termination),
            "bilateral_airborne_ever": accumulator.support_failed,
            "bilateral_unsupported_physics_substeps": accumulator.unsupported_physics_substeps,
            "bilateral_unsupported_endpoint_samples": accumulator.unsupported_endpoint_samples,
            "bilateral_unsupported_max_consecutive_physics_substeps": accumulator.unsupported_max_run,
            "left_contact_ever": accumulator.left_contact_ever,
            "right_contact_ever": accumulator.right_contact_ever,
            "left_unloaded_physics_substeps": accumulator.left_unloaded_physics_substeps,
            "right_unloaded_physics_substeps": accumulator.right_unloaded_physics_substeps,
            "non_wheel_contact": accumulator.non_wheel_contact,
            "termination": accumulator.termination,
            "termination_reasons": accumulator.termination_reasons,
            "success_step": accumulator.strict_success_step,
            "geometric_success_step": accumulator.geometric_success_step,
            "max_progress_past_face_m": accumulator.max_progress_m,
            "peak_pitch_abs_rad": accumulator.peak_pitch_abs_rad,
            "peak_roll_abs_rad": accumulator.peak_roll_abs_rad,
            "peak_pitch_rate_abs_radps": accumulator.peak_pitch_rate_abs_radps,
            "minimum_native_left_force_n": _finite_or_none(accumulator.minimum_left_force_n),
            "minimum_native_right_force_n": _finite_or_none(accumulator.minimum_right_force_n),
            "minimum_native_total_force_n": _finite_or_none(accumulator.minimum_total_force_n),
            "torque_saturation_fraction": (
                wheel_saturated / wheel_total if wheel_total else 0.0),
            "wheel_target_abs_max_radps": accumulator.max_wheel_target_abs_radps,
            "wheel_speed_abs_max_radps": accumulator.max_wheel_speed_abs_radps,
            "wheel_residual_abs_max": 0.0,
            "applied_residual_abs_max": 0.0,
            "wheel_target_classical_path_abs_max_radps": 0.0,
            "dynamic_leg_feedforward_abs_max_rad": 0.0,
            "dynamic_drive_feedforward_abs_max_radps": 0.0,
            "support_events": accumulator.support_events,
            "non_wheel_event": accumulator.non_wheel_event,
            "root_reset": _root_reset_row(reset, env_id),
            "final_native_time_s": float(native_data[env_id].time),
            "final_qpos": native_data[env_id].qpos.tolist(),
            "final_qvel": native_data[env_id].qvel.tolist(),
            "controller_mirror_forward_calls": controller_forward_calls,
            "mjwarp_dynamics_steps": 0,
            "whole_body_effort": effort,
            "support_transfer_monitor": {
                "constraint": "BOTH_REQUIRED",
                "minimum_load_n": adapter.config.minimum_side_load_n,
                "single_support_commands": 0},
        })
    _validate_selected_resets(rows, reset_override)
    return rows


def _row_unsafe(row: Mapping[str, Any]) -> bool:
    if "unsafe" in row:
        return bool(row["unsafe"])
    return bool(
        row["bilateral_airborne_ever"]
        or row["non_wheel_contact"]
        or row["termination"]
    )


def _classification(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, float], Any]:
    return {(int(row["env_id"]), float(row["stair_height_m"])):
            (bool(row["success"]),
             bool(row["geometric_success_ignoring_support"]), _row_unsafe(row))
            for row in rows}


def _backend_acceptance(
    rows: Sequence[Mapping[str, Any]], *, expected_per_height: int,
) -> dict[str, bool]:
    summaries = diag.summarize_trials(rows)
    result = {}
    for height, label in ((0.0, "flat"), (0.0025, "step")):
        summary = next(row for row in summaries if row["stair_height_m"] == height)
        result[f"{label}_strict_all"] = int(summary["successes"]) == expected_per_height
        result[f"{label}_geometric_all"] = (
            int(summary["geometric_successes_ignoring_support"]) == expected_per_height)
    result["zero_bilateral_unsupported_substeps"] = not any(
        int(row["bilateral_unsupported_physics_substeps"]) > 0 for row in rows)
    result["zero_non_wheel_contacts"] = not any(
        bool(row["non_wheel_contact"]) for row in rows)
    result["zero_terminations"] = not any(bool(row["termination"]) for row in rows)
    result["leg_within_rated_limit"] = all(
        float(row["whole_body_effort"]["maximum_abs_leg_torque_nm"])
        <= DM_J6248P_RATED_TORQUE + 1.0e-9 for row in rows)
    result["wheel_within_peak_limit"] = all(
        float(row["whole_body_effort"]["maximum_abs_wheel_torque_nm"])
        <= RMD_L_9025_35T_PEAK_TORQUE + 1.0e-9 for row in rows)
    return result


def _first_failure_classification(
    backends: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any] | None:
    for backend, rows in backends:
        for row in rows:
            if bool(row["success"]):
                continue
            effort, traces = row["whole_body_effort"], row["whole_body_effort"]["control_trace"]
            if any(not bool(sample["qp_bilateral_minimum_satisfied"])
                   for sample in traces):
                category = "qp_or_load_constraint_infeasible"
            elif int(effort["leg_saturated_samples"]) > 0:
                category = "leg_rated_torque_saturation"
            elif int(effort["wheel_saturated_samples"]) > 0:
                category = "wheel_peak_torque_saturation"
            elif bool(row["bilateral_airborne_ever"]):
                category = "contact_geometry_cannot_realize_target_load"
            else:
                category = "control_sign_or_5ms_phase_error"
            return {
                "backend": backend, "env_id": int(row["env_id"]),
                "stair_height_m": float(row["stair_height_m"]),
                "category": category,
                "first_support_event": (
                    row.get("support_events", [None])[0]
                    if row.get("support_events")
                    else None
                ),
            }
    return None


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.source_result.resolve()
    source_sha = _sha256(source_path)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            f"R0c source SHA256 drifted: expected {EXPECTED_SOURCE_SHA256}, got {source_sha}.")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("kind") != "r0c_synchronized_reference_rejection_screen":
        raise ValueError("Source result is not an R0c-SYNC artifact.")
    if source.get("matched_reset_perturbations_across_candidates") is not True:
        raise ValueError("Source artifact does not certify matched resets.")
    causal_manifest_pre = _causal_file_manifest()
    current_git_sha = _git_value("rev-parse", "HEAD")
    validated_sanity_gate = None
    if not args.sanity:
        sanity_path = getattr(args, "validated_sanity_artifact", None)
        sanity_sha = getattr(args, "validated_sanity_sha256", None)
        if sanity_path is None or sanity_sha is None:
            raise ValueError(
                "Full mode requires a validated sanity artifact and SHA256."
            )
        validated_sanity_gate = _validate_full_run_sanity_artifact(
            sanity_path.resolve(),
            expected_sha256=sanity_sha,
            source_sha256=source_sha,
            current_manifest=causal_manifest_pre,
            current_git_sha=current_git_sha,
        )
    candidate = lateral._candidate_map()["c0"]
    source_resets = lateral._source_reset_map(source, "c0")
    envs_per_height, reset_override, source_identities = _selected_resets(
        source_resets, sanity=args.sanity)
    baseline_env = ManagerBasedRlEnv(
        cfg=rb.make_roll_boundary_env_cfg(HEIGHTS_M, envs_per_height),
        device=args.device)
    try:
        baseline_snapshot = _model_snapshot(baseline_env.sim.mj_model)
    finally:
        baseline_env.close()
    original_cards = rb.POSTURE_CARDS
    rb.POSTURE_CARDS = (candidate["posture_card"],)
    env = ManagerBasedRlEnv(cfg=_make_effort_cfg(envs_per_height), device=args.device)
    adapter = _WholeBodyEffortAdapter(env, WholeBodyEffortConfig())
    adapter.install()
    try:
        actuator_contract = _compare_actuator_only_contract(
            baseline_snapshot, env.sim.mj_model)
        model_equivalence = _model_equivalence_report(env)
        mjwarp_rows = _run_mjwarp_effort(
            env, adapter=adapter, candidate=candidate,
            reset_override=reset_override)
        native_rows = _run_native_effort(
            env, adapter=adapter, candidate=candidate,
            reset_override=reset_override)
        model_metadata = {
            "timestep_s": float(env.sim.mj_model.opt.timestep),
            "nq": int(env.sim.mj_model.nq), "nv": int(env.sim.mj_model.nv),
            "nu": int(env.sim.mj_model.nu), "na": int(env.sim.mj_model.na)}
    finally:
        adapter.restore()
        env.close()
        rb.POSTURE_CARDS = original_cards
    causal_manifest_post = _causal_file_manifest()
    _assert_causal_manifest_unchanged(
        causal_manifest_pre, causal_manifest_post,
    )
    expected = 1 if args.sanity else diag.R0C_SYNC_ENVS_PER_HEIGHT
    mjwarp_acceptance = _backend_acceptance(mjwarp_rows, expected_per_height=expected)
    native_acceptance = _backend_acceptance(native_rows, expected_per_height=expected)
    classification_match = _classification(mjwarp_rows) == _classification(native_rows)
    all_checks = (all(mjwarp_acceptance.values())
                  and all(native_acceptance.values()) and classification_match)
    first_failure = _first_failure_classification(
        (("mjwarp", mjwarp_rows), ("native", native_rows)))
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "r0c_effort_wbc_dual_backend_development_probe",
        "mode": "sanity" if args.sanity else "full",
        "evidence_eligible": False, "promotion_eligible": False,
        "formal_protocol_modified": False, "strict_verdict_modified": False,
        "shape_parameter_sweep": False, "controller_parameter_sweep": False,
        "git_sha": current_git_sha,
        "project_dirty": bool(_git_value("status", "--porcelain")),
        "dirty_snapshot_sha256": _dirty_snapshot_sha256(),
        "causal_file_manifest": causal_manifest_pre,
        "causal_file_manifest_pre": causal_manifest_pre,
        "causal_file_manifest_post": causal_manifest_post,
        "causal_provenance_gate_pass": True,
        "validated_sanity_gate": validated_sanity_gate,
        "device": args.device, "source_result": str(source_path),
        "source_result_sha256": source_sha,
        "source_reset_identities": [list(identity) for identity in source_identities],
        "matched_reviewed_resets": True,
        "controller": asdict(WholeBodyEffortConfig()),
        "protocol": {
            "heights_m": list(HEIGHTS_M), "envs_per_height": expected,
            "settle_control_steps": rb.OFFICIAL_SETTLE_STEPS,
            "drive_control_steps": rb.OFFICIAL_DRIVE_STEPS,
            "stable_control_steps": rb.OFFICIAL_STABLE_STEPS,
            "control_dt_s": rb.ROLL_POSE_CONTROL_DT_S,
            "physics_dt_s": rb.ROLL_FIRST_PHYSICS_TIMESTEP_S,
            "control_decimation": CONTROL_DECIMATION,
            "effort_metric_aggregation": {
                "window": "fixed_settle_plus_drive",
                "control_steps": (
                    rb.OFFICIAL_SETTLE_STEPS + rb.OFFICIAL_DRIVE_STEPS
                ),
                "physics_substeps_per_control_step": CONTROL_DECIMATION,
                "expected_controller_samples_per_trial": (
                    (rb.OFFICIAL_SETTLE_STEPS + rb.OFFICIAL_DRIVE_STEPS)
                    * CONTROL_DECIMATION
                ),
                "includes_inactive_post_classification_samples": True,
                "active_field_is_annotation_only": True,
            },
            "command_vx_mps": 0.07, "episode_wide_support_safety": True,
        },
        "model": model_metadata,
        "actuator_contract": actuator_contract,
        "model_equivalence": model_equivalence,
        "acceptance": {
            "mjwarp": mjwarp_acceptance, "native": native_acceptance,
            "cross_backend_trial_classification_exact_match": classification_match,
            "all_required_checks_pass": all_checks,
        },
        "first_failure": first_failure,
        "mjwarp": {"summaries": diag.summarize_trials(mjwarp_rows),
                   "trials": mjwarp_rows},
        "native": {"summaries": diag.summarize_trials(native_rows),
                   "trials": native_rows},
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument(
        "--sanity", action="store_true",
        help="Run source env 0 flat and env 8 stair before the full run.")
    parser.add_argument(
        "--validated-sanity-artifact",
        type=Path,
        help="Passing schema-v2 sanity artifact required for full mode.",
    )
    parser.add_argument(
        "--validated-sanity-sha256",
        help="Expected SHA256 of --validated-sanity-artifact.",
    )
    args = parser.parse_args(argv)
    gate_values = (
        args.validated_sanity_artifact,
        args.validated_sanity_sha256,
    )
    if args.sanity and any(value is not None for value in gate_values):
        parser.error("Sanity mode must not consume a prior sanity artifact.")
    if not args.sanity and any(value is None for value in gate_values):
        parser.error(
            "Full mode requires --validated-sanity-artifact and "
            "--validated-sanity-sha256."
        )
    if args.output.exists():
        parser.error(f"Refusing to overwrite effort-WBC output: {args.output}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_probe(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rb._atomic_write_json(output, payload)
    print(f"[effort-wbc] mode={payload['mode']} output={output}")
    print("[effort-wbc] all_required_checks_pass="
          f"{payload['acceptance']['all_required_checks_pass']}")


if __name__ == "__main__":
    main()


