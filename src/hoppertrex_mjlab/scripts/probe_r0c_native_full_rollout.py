"""Run the reviewed R0c exact resets as a native-MuJoCo closed loop.

This development-only backend qualification probe reuses the existing MjLab
command and HybridWheelLegAction implementations as a controller mirror, but
never calls MJWarp dynamics. At every 20 ms control step it synchronizes native
state into the mirror, obtains the exact actuator ctrl vector, and advances
four 5 ms steps with native MuJoCo. Contact, safety, and traversal verdicts are
computed from the native trajectory.

A successful artifact can justify proposing a formal backend change; it does
not itself change the RollBoundary protocol or become promotion evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

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
from hoppertrex_mjlab.tasks.hoppertrex_balance_task import (
    BAD_ORIENTATION_LIMIT_ANGLE,
    NON_WHEEL_GROUND_GEOMS,
    ROOT_HEIGHT_HARD_MIN,
)

SCHEMA_VERSION = 1
HEIGHTS_M = (0.0, 0.0025)
CONTROL_DECIMATION = 4
_BRIDGE_STATE_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "qacc_warmstart",
    "qfrc_applied",
    "xfrc_applied",
    "ctrl",
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


def _non_wheel_geom_ids(model: mujoco.MjModel) -> frozenset[int]:
    ids = []
    for name in NON_WHEEL_GROUND_GEOMS:
        geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{name}"
        )
        if geom_id < 0:
            raise ValueError(f"Native rollout could not resolve geom {name!r}.")
        ids.append(geom_id)
    if len(set(ids)) != len(NON_WHEEL_GROUND_GEOMS):
        raise ValueError("Native rollout non-wheel geom identities are not unique.")
    return frozenset(ids)


def _native_non_wheel_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    non_wheel_geom_ids: frozenset[int],
    terrain_geom_ids: frozenset[int],
) -> list[dict[str, Any]]:
    contacts = []
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        pair = {int(contact.geom1), int(contact.geom2)}
        robot_matches = pair & non_wheel_geom_ids
        terrain_matches = pair & terrain_geom_ids
        if not robot_matches or not terrain_matches:
            continue
        robot_geom_id = next(iter(robot_matches))
        terrain_geom_id = next(iter(terrain_matches))
        contacts.append(
            {
                "contact_id": contact_id,
                "robot_geom_id": robot_geom_id,
                "robot_geom_name": mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom_id
                ),
                "terrain_geom_id": terrain_geom_id,
                "terrain_geom_name": mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, terrain_geom_id
                ),
                "distance_m": float(contact.dist),
            }
        )
    return contacts


def _update_unsupported_run(
    unsupported: bool,
    current_run: int,
    maximum_run: int,
) -> tuple[int, int]:
    current_run = current_run + 1 if unsupported else 0
    return current_run, max(maximum_run, current_run)


@dataclass
class TrialAccumulator:
    active: bool = True
    support_failed: bool = False
    termination: bool = False
    non_wheel_contact: bool = False
    strict_success: bool = False
    geometric_success: bool = False
    strict_success_step: int | None = None
    geometric_success_step: int | None = None
    strict_stable_steps: int = 0
    geometric_stable_steps: int = 0
    unsupported_physics_substeps: int = 0
    unsupported_endpoint_samples: int = 0
    unsupported_current_run: int = 0
    unsupported_max_run: int = 0
    left_unloaded_physics_substeps: int = 0
    right_unloaded_physics_substeps: int = 0
    left_contact_ever: bool = False
    right_contact_ever: bool = False
    max_progress_m: float = -math.inf
    peak_pitch_abs_rad: float = 0.0
    peak_roll_abs_rad: float = 0.0
    peak_pitch_rate_abs_radps: float = 0.0
    minimum_left_force_n: float = math.inf
    minimum_right_force_n: float = math.inf
    minimum_total_force_n: float = math.inf
    torque_saturated_wheel_samples: int = 0
    torque_wheel_samples: int = 0
    max_wheel_target_abs_radps: float = 0.0
    max_wheel_speed_abs_radps: float = 0.0
    support_events: list[dict[str, Any]] = field(default_factory=list)
    non_wheel_event: dict[str, Any] | None = None
    termination_reasons: list[str] = field(default_factory=list)

    def observe_substep_support(
        self,
        *,
        left_force_n: float,
        right_force_n: float,
        phase: str,
        control_step: int,
        physics_substep: int,
        time_s: float,
    ) -> None:
        left = left_force_n > 0.0
        right = right_force_n > 0.0
        unsupported = not left and not right
        self.left_contact_ever |= left
        self.right_contact_ever |= right
        self.minimum_left_force_n = min(self.minimum_left_force_n, left_force_n)
        self.minimum_right_force_n = min(self.minimum_right_force_n, right_force_n)
        self.minimum_total_force_n = min(
            self.minimum_total_force_n, left_force_n + right_force_n
        )
        if phase == "drive":
            self.left_unloaded_physics_substeps += int(not left)
            self.right_unloaded_physics_substeps += int(not right)
        self.unsupported_current_run, self.unsupported_max_run = (
            _update_unsupported_run(
                unsupported,
                self.unsupported_current_run,
                self.unsupported_max_run,
            )
        )
        if not unsupported:
            return
        self.support_failed = True
        self.unsupported_physics_substeps += 1
        self.support_events.append(
            {
                "sample_kind": "native_during_step",
                "phase": phase,
                "control_step": control_step,
                "physics_substep_within_control": physics_substep,
                "time_s": time_s,
                "left_force_norm_n": left_force_n,
                "right_force_norm_n": right_force_n,
            }
        )

    def observe_endpoint_support(
        self,
        *,
        left_force_n: float,
        right_force_n: float,
        phase: str,
        control_step: int,
        time_s: float,
    ) -> None:
        left = left_force_n > 0.0
        right = right_force_n > 0.0
        self.left_contact_ever |= left
        self.right_contact_ever |= right
        if left or right:
            return
        self.support_failed = True
        self.unsupported_endpoint_samples += 1
        self.support_events.append(
            {
                "sample_kind": "native_integrated_endpoint",
                "phase": phase,
                "control_step": control_step,
                "time_s": time_s,
                "left_force_norm_n": left_force_n,
                "right_force_norm_n": right_force_n,
            }
        )


def _native_data_from_controller_state(
    env: ManagerBasedRlEnv,
) -> list[mujoco.MjData]:
    model = env.sim.mj_model
    fields = {
        name: getattr(env.sim.data, name).detach().cpu().numpy().copy()
        for name in _BRIDGE_STATE_FIELDS
    }
    times = env.sim.data.time.detach().cpu().numpy().copy()
    native_data = []
    for env_id in range(env.num_envs):
        data = mujoco.MjData(model)
        for name, values in fields.items():
            getattr(data, name)[:] = values[env_id]
        data.time = float(times[env_id])
        mujoco.mj_forward(model, data)
        native_data.append(data)
    return native_data


def _sync_native_state_to_controller(
    env: ManagerBasedRlEnv,
    native_data: Sequence[mujoco.MjData],
) -> None:
    if len(native_data) != env.num_envs:
        raise ValueError("Native/controller environment counts differ.")
    for name in _BRIDGE_STATE_FIELDS:
        target = getattr(env.sim.data, name)
        values = np.stack([np.asarray(getattr(data, name)) for data in native_data])
        target[:] = torch.as_tensor(
            values,
            device=env.device,
            dtype=target.dtype,
        )
    env.sim.data.time[:] = torch.as_tensor(
        [data.time for data in native_data],
        device=env.device,
        dtype=env.sim.data.time.dtype,
    )
    env.sim.forward()
    env.sim.sense()


def _controller_packet(
    env: ManagerBasedRlEnv,
    actions: torch.Tensor,
) -> dict[str, np.ndarray]:
    env.action_manager.process_action(actions)
    env.action_manager.apply_action()
    env.scene.write_data_to_sim()
    return {
        name: getattr(env.sim.data, name).detach().cpu().numpy().copy()
        for name in ("ctrl", "qfrc_applied", "xfrc_applied")
    }


def _apply_controller_packet(
    data: mujoco.MjData,
    packet: Mapping[str, np.ndarray],
    env_id: int,
) -> None:
    data.ctrl[:] = packet["ctrl"][env_id]
    data.qfrc_applied[:] = packet["qfrc_applied"][env_id]
    data.xfrc_applied[:] = packet["xfrc_applied"][env_id]


def _root_reset_row(reset: Mapping[str, torch.Tensor], env_id: int) -> dict[str, Any]:
    return {
        "x_relative_to_face_m": float(reset["x_relative_to_face_m"][env_id]),
        "y_relative_to_center_m": float(
            reset["y_relative_to_center_m"][env_id]
        ),
        "root_height_m": float(reset["root_height_m"][env_id]),
        "root_quaternion_wxyz": reset["root_quaternion_wxyz"][env_id]
        .detach()
        .cpu()
        .tolist(),
        "root_linear_velocity_mps": reset["root_linear_velocity_mps"][env_id]
        .detach()
        .cpu()
        .tolist(),
        "root_angular_velocity_radps": reset[
            "root_angular_velocity_radps"
        ][env_id]
        .detach()
        .cpu()
        .tolist(),
        "leg_joint_position_rad": reset["leg_joint_position_rad"][env_id]
        .detach()
        .cpu()
        .tolist(),
        "leg_joint_velocity_radps": reset[
            "leg_joint_velocity_radps"
        ][env_id]
        .detach()
        .cpu()
        .tolist(),
    }


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _run_native_rollout(
    env: ManagerBasedRlEnv,
    *,
    candidate: Mapping[str, Any],
    source_resets: Mapping[tuple[int, float], Any],
) -> list[dict[str, Any]]:
    card = candidate["posture_card"]
    schedule = candidate["schedule"]
    terrain_types, face_x, cross_x, reset = rb._reset_to_approach(
        env,
        root_height=float(card["height_m"]),
        card_name=str(card["name"]),
        repeat=1,
        height_count=len(HEIGHTS_M),
        reset_override=shadow_probe._reset_override(source_resets),
    )
    native_data = _native_data_from_controller_state(env)
    model = env.sim.mj_model
    wheel_geom_ids = _wheel_geom_ids(model)
    terrain_geom_ids = _terrain_geom_ids(model)
    non_wheel_geom_ids = _non_wheel_geom_ids(model)
    robot = env.scene["robot"]
    term = env.action_manager.get_term("hybrid_wheel_leg")
    wheel_joint_ids = term._wheel_ids
    actions = torch.zeros(
        (env.num_envs, env.action_space.shape[-1]), device=env.device
    )
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    accumulators = [TrialAccumulator() for _ in range(env.num_envs)]
    for env_id, accumulator in enumerate(accumulators):
        accumulator.max_progress_m = float(reset["x_relative_to_face_m"][env_id])

    schedule_state = rb.make_roll_pose_schedule_state(
        schedule,
        robot.data.root_link_pos_w[:, 0],
        slew_mode=str(candidate["slew_mode"]),
    )
    controller_forward_calls = 0
    global_control_step = 0

    def control_step(vx: float, drive_index: int | None) -> None:
        nonlocal active, controller_forward_calls, global_control_step
        global_control_step += 1
        was_active = active.clone()
        phase = "settle" if drive_index is None else "drive"
        schedule_output = rb.roll_pose_schedule_step(
            schedule,
            schedule_state,
            root_x_m=robot.data.root_link_pos_w[:, 0],
            face_x_m=face_x,
            active_mask=was_active,
            drive_active=drive_index is not None,
            dt=rb.ROLL_POSE_CONTROL_DT_S,
        )
        rb._force_commands(
            env,
            active=was_active,
            vx=vx,
            height=schedule_output.applied_height_m,
            pitch=schedule_output.applied_pitch_rad,
        )
        previous_wheel_targets = term._previous_wheel_targets.detach().clone()
        packet = _controller_packet(env, actions)
        classical_delta = torch.clamp(
            term.controller_baseline - previous_wheel_targets,
            -term.cfg.wheel_slew_limit,
            term.cfg.wheel_slew_limit,
        )
        classical_target = torch.clamp(
            previous_wheel_targets + classical_delta,
            -term.cfg.wheel_velocity_limit,
            term.cfg.wheel_velocity_limit,
        )
        if float((term.wheel_targets - classical_target).abs().max()) != 0.0:
            raise RuntimeError("Native rollout controller mirror changed wheel authority.")
        if float(term.dynamic_leg_feedforward.abs().max()) != 0.0:
            raise RuntimeError("Native rollout enabled dynamic leg feedforward.")
        if float(term.dynamic_drive_feedforward.abs().max()) != 0.0:
            raise RuntimeError("Native rollout enabled dynamic drive feedforward.")
        if float(term.applied_residual.abs().max()) != 0.0:
            raise RuntimeError("Native rollout observed nonzero residual authority.")
        if not all(np.isfinite(values).all() for values in packet.values()):
            raise RuntimeError("Native rollout controller packet is non-finite.")

        for physics_substep in range(1, CONTROL_DECIMATION + 1):
            for env_id, data in enumerate(native_data):
                _apply_controller_packet(data, packet, env_id)
                pre_root_z = float(data.qpos[2])
                pre_root_vz = float(data.qvel[2])
                mujoco.mj_step(model, data)
                if not bool(was_active[env_id]):
                    continue
                support = _native_wheel_forces(
                    model, data, wheel_geom_ids, terrain_geom_ids
                )
                accumulators[env_id].observe_substep_support(
                    left_force_n=float(support["left_force_norm_n"]),
                    right_force_n=float(support["right_force_norm_n"]),
                    phase=phase,
                    control_step=global_control_step,
                    physics_substep=physics_substep,
                    time_s=float(data.time),
                )
                if bool(support["bilateral_zero_force"]):
                    # mj_step leaves geom_xpos/contact at the state whose
                    # constraints generated this step, while qpos/qvel are the
                    # integrated endpoint. Keep both sides of the transition.
                    accumulators[env_id].support_events[-1].update(
                        {
                            "pre_root_z_m": pre_root_z,
                            "pre_root_vz_mps": pre_root_vz,
                            "pre_wheel_center_z_m": [
                                float(data.geom_xpos[geom_id, 2])
                                for geom_id in wheel_geom_ids
                            ],
                            "post_root_z_m": float(data.qpos[2]),
                            "post_root_vz_mps": float(data.qvel[2]),
                            "native_contact_count": int(data.ncon),
                            "wheel_contacts": support["contacts"],
                        }
                    )

        endpoint_support = []
        endpoint_non_wheel = []
        for data in native_data:
            mujoco.mj_forward(model, data)
            endpoint_support.append(
                _native_wheel_forces(
                    model, data, wheel_geom_ids, terrain_geom_ids
                )
            )
            endpoint_non_wheel.append(
                _native_non_wheel_contacts(
                    model, data, non_wheel_geom_ids, terrain_geom_ids
                )
            )
        _sync_native_state_to_controller(env, native_data)
        controller_forward_calls += 1

        pitch, roll = rb._pitch_roll(robot)
        pitch_rate = robot.data.root_link_ang_vel_b[:, 1]
        projected_gravity = robot.data.projected_gravity_b
        root_pos = robot.data.root_link_pos_w
        progress = root_pos[:, 0] - face_x
        joint_speed = robot.data.joint_vel[:, wheel_joint_ids].detach()
        wheel_target = term.wheel_targets.detach()
        _torque, saturated = rb.model_wheel_torque(wheel_target, joint_speed)
        finite_state = (
            torch.isfinite(root_pos).all(dim=1)
            & torch.isfinite(robot.data.root_link_quat_w).all(dim=1)
            & torch.isfinite(robot.data.root_link_lin_vel_w).all(dim=1)
            & torch.isfinite(robot.data.root_link_ang_vel_w).all(dim=1)
            & torch.isfinite(robot.data.joint_pos).all(dim=1)
            & torch.isfinite(robot.data.joint_vel).all(dim=1)
        )
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
                phase=phase,
                control_step=global_control_step,
                time_s=float(native_data[env_id].time),
            )
            if bool(endpoint["bilateral_zero_force"]):
                data = native_data[env_id]
                accumulator.support_events[-1].update(
                    {
                        "root_z_m": float(data.qpos[2]),
                        "root_vz_mps": float(data.qvel[2]),
                        "wheel_center_z_m": [
                            float(data.geom_xpos[geom_id, 2])
                            for geom_id in wheel_geom_ids
                        ],
                        "native_contact_count": int(data.ncon),
                        "wheel_contacts": endpoint["contacts"],
                    }
                )
            contacts = endpoint_non_wheel[env_id]
            if contacts:
                accumulator.non_wheel_contact = True
                if accumulator.non_wheel_event is None:
                    accumulator.non_wheel_event = {
                        "phase": phase,
                        "control_step": global_control_step,
                        "time_s": float(native_data[env_id].time),
                        "contacts": contacts,
                    }
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
                accumulator.peak_pitch_abs_rad, abs(float(pitch[env_id]))
            )
            accumulator.peak_roll_abs_rad = max(
                accumulator.peak_roll_abs_rad, abs(float(roll[env_id]))
            )
            accumulator.peak_pitch_rate_abs_radps = max(
                accumulator.peak_pitch_rate_abs_radps,
                abs(float(pitch_rate[env_id])),
            )
            if drive_index is not None:
                accumulator.max_progress_m = max(
                    accumulator.max_progress_m, float(progress[env_id])
                )
                accumulator.max_wheel_target_abs_radps = max(
                    accumulator.max_wheel_target_abs_radps,
                    float(wheel_target[env_id].abs().max()),
                )
                accumulator.max_wheel_speed_abs_radps = max(
                    accumulator.max_wheel_speed_abs_radps,
                    float(joint_speed[env_id].abs().max()),
                )
                accumulator.torque_saturated_wheel_samples += int(
                    saturated[env_id].sum()
                )
                accumulator.torque_wheel_samples += int(saturated.shape[1])
                posture_ok = (
                    abs(float(pitch[env_id])) <= rb.PITCH_LIMIT_RAD
                    and abs(float(roll[env_id])) <= rb.ROLL_LIMIT_RAD
                    and abs(float(pitch_rate[env_id]))
                    <= rb.PITCH_RATE_LIMIT_RADPS
                )
                crossed = float(root_pos[env_id, 0]) >= float(cross_x[env_id])
                strict_valid = not (
                    accumulator.support_failed
                    or accumulator.non_wheel_contact
                    or accumulator.termination
                )
                geometric_valid = not (
                    accumulator.non_wheel_contact or accumulator.termination
                )
                accumulator.strict_stable_steps = (
                    accumulator.strict_stable_steps + 1
                    if strict_valid and crossed and posture_ok
                    else 0
                )
                accumulator.geometric_stable_steps = (
                    accumulator.geometric_stable_steps + 1
                    if geometric_valid and crossed and posture_ok
                    else 0
                )
                if (
                    not accumulator.strict_success
                    and accumulator.strict_stable_steps >= rb.OFFICIAL_STABLE_STEPS
                ):
                    accumulator.strict_success = True
                    accumulator.strict_success_step = drive_index + 1
                if (
                    not accumulator.geometric_success
                    and accumulator.geometric_stable_steps
                    >= rb.OFFICIAL_STABLE_STEPS
                ):
                    accumulator.geometric_success = True
                    accumulator.geometric_success_step = drive_index + 1

            accumulator.active = not (
                accumulator.termination
                or accumulator.non_wheel_contact
                or accumulator.strict_success
            )
            active[env_id] = accumulator.active

    for _ in range(rb.OFFICIAL_SETTLE_STEPS):
        control_step(0.0, None)
    for drive_index in range(rb.OFFICIAL_DRIVE_STEPS):
        control_step(0.07, drive_index)

    rows = []
    for env_id, terrain_type in enumerate(terrain_types.detach().cpu().tolist()):
        accumulator = accumulators[env_id]
        torque_fraction = (
            accumulator.torque_saturated_wheel_samples
            / accumulator.torque_wheel_samples
            if accumulator.torque_wheel_samples
            else 0.0
        )
        rows.append(
            {
                "env_id": env_id,
                "stair_height_m": float(HEIGHTS_M[terrain_type]),
                "success": accumulator.strict_success,
                "geometric_success_ignoring_support": accumulator.geometric_success,
                "safe_stall": not accumulator.strict_success
                and not accumulator.support_failed
                and not accumulator.non_wheel_contact
                and not accumulator.termination,
                "unsafe": accumulator.support_failed
                or accumulator.non_wheel_contact
                or accumulator.termination,
                "bilateral_airborne_ever": accumulator.support_failed,
                "bilateral_unsupported_physics_substeps": (
                    accumulator.unsupported_physics_substeps
                ),
                "bilateral_unsupported_endpoint_samples": (
                    accumulator.unsupported_endpoint_samples
                ),
                "bilateral_unsupported_max_consecutive_physics_substeps": (
                    accumulator.unsupported_max_run
                ),
                "left_contact_ever": accumulator.left_contact_ever,
                "right_contact_ever": accumulator.right_contact_ever,
                "left_unloaded_physics_substeps": (
                    accumulator.left_unloaded_physics_substeps
                ),
                "right_unloaded_physics_substeps": (
                    accumulator.right_unloaded_physics_substeps
                ),
                "non_wheel_contact": accumulator.non_wheel_contact,
                "termination": accumulator.termination,
                "termination_reasons": accumulator.termination_reasons,
                "success_step": accumulator.strict_success_step,
                "geometric_success_step": accumulator.geometric_success_step,
                "max_progress_past_face_m": accumulator.max_progress_m,
                "peak_pitch_abs_rad": accumulator.peak_pitch_abs_rad,
                "peak_roll_abs_rad": accumulator.peak_roll_abs_rad,
                "peak_pitch_rate_abs_radps": (
                    accumulator.peak_pitch_rate_abs_radps
                ),
                "minimum_native_left_force_n": _finite_or_none(
                    accumulator.minimum_left_force_n
                ),
                "minimum_native_right_force_n": _finite_or_none(
                    accumulator.minimum_right_force_n
                ),
                "minimum_native_total_force_n": _finite_or_none(
                    accumulator.minimum_total_force_n
                ),
                "torque_saturation_fraction": torque_fraction,
                "wheel_target_abs_max_radps": (
                    accumulator.max_wheel_target_abs_radps
                ),
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
            }
        )
    lateral._validate_resets(rows, source_resets)
    return rows


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 2 * diag.R0C_SYNC_ENVS_PER_HEIGHT:
        raise RuntimeError("Native rollout did not produce exactly sixteen trials.")
    for row in rows:
        strict_success = bool(row["success"])
        geometric_success = bool(row["geometric_success_ignoring_support"])
        unsafe = bool(row["unsafe"])
        if strict_success and (not geometric_success or unsafe):
            raise RuntimeError("Native rollout strict-success invariants failed.")
        if bool(row["bilateral_airborne_ever"]) != (
            int(row["bilateral_unsupported_physics_substeps"]) > 0
            or int(row["bilateral_unsupported_endpoint_samples"]) > 0
        ):
            raise RuntimeError("Native rollout support counters disagree.")
        for key in diag.SCHEDULE_AUTHORITY_METRICS:
            if float(row[key]) != 0.0:
                raise RuntimeError(f"Native rollout observed authority in {key}.")
        if int(row["mjwarp_dynamics_steps"]) != 0:
            raise RuntimeError("Native rollout accidentally used MJWarp dynamics.")


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.source_result.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("kind") != "r0c_synchronized_reference_rejection_screen":
        raise ValueError("Source result is not an R0c-SYNC artifact.")
    if source.get("matched_reset_perturbations_across_candidates") is not True:
        raise ValueError("Source R0c-SYNC artifact does not certify matched resets.")

    candidate = lateral._candidate_map()["c0"]
    source_resets = lateral._source_reset_map(source, "c0")
    cfg = rb.make_roll_boundary_env_cfg(
        HEIGHTS_M,
        diag.R0C_SYNC_ENVS_PER_HEIGHT,
    )
    original_cards = rb.POSTURE_CARDS
    rb.POSTURE_CARDS = (candidate["posture_card"],)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    try:
        model_equivalence = _model_equivalence_report(env)
        rows = _run_native_rollout(
            env,
            candidate=candidate,
            source_resets=source_resets,
        )
        model = env.sim.mj_model
        model_metadata = {
            "timestep_s": float(model.opt.timestep),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "na": int(model.na),
        }
    finally:
        env.close()
        rb.POSTURE_CARDS = original_cards
    _validate_rows(rows)
    summaries = diag.summarize_trials(rows)

    flat = next(row for row in summaries if row["stair_height_m"] == 0.0)
    step = next(row for row in summaries if row["stair_height_m"] == 0.0025)
    qualification = {
        "flat_strict_8_of_8": int(flat["successes"]) == 8,
        "flat_geometric_8_of_8": int(flat["geometric_successes_ignoring_support"])
        == 8,
        "step_strict_8_of_8": int(step["successes"]) == 8,
        "step_geometric_8_of_8": int(
            step["geometric_successes_ignoring_support"]
        )
        == 8,
        "zero_native_bilateral_unsupported_substeps": sum(
            int(row["bilateral_unsupported_physics_substeps"]) for row in rows
        )
        == 0,
        "zero_native_bilateral_unsupported_endpoints": sum(
            int(row["bilateral_unsupported_endpoint_samples"]) for row in rows
        )
        == 0,
        "zero_non_wheel_contacts": not any(
            bool(row["non_wheel_contact"]) for row in rows
        ),
        "zero_terminations": not any(bool(row["termination"]) for row in rows),
    }
    qualification["ready_to_propose_formal_backend_change"] = all(
        qualification.values()
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "r0c_native_full_rollout_development_probe",
        "evidence_eligible": False,
        "promotion_eligible": False,
        "formal_protocol_modified": False,
        "reason": "closed-loop native backend qualification before formal proposal",
        "git_sha": _git_value("rev-parse", "HEAD"),
        "project_dirty": bool(_git_value("status", "--porcelain")),
        "device": args.device,
        "source_result": str(source_path),
        "source_result_sha256": _sha256(source_path),
        "matched_reviewed_resets": True,
        "candidate": {
            "key": "c0",
            "name": candidate["name"],
            "posture_card": candidate["posture_card"],
            "schedule": candidate["schedule"].to_dict(),
            "slew_mode": candidate["slew_mode"],
        },
        "protocol": {
            "heights_m": list(HEIGHTS_M),
            "envs_per_height": diag.R0C_SYNC_ENVS_PER_HEIGHT,
            "settle_control_steps": rb.OFFICIAL_SETTLE_STEPS,
            "drive_control_steps": rb.OFFICIAL_DRIVE_STEPS,
            "stable_control_steps": rb.OFFICIAL_STABLE_STEPS,
            "control_dt_s": rb.ROLL_POSE_CONTROL_DT_S,
            "physics_dt_s": float(model_metadata["timestep_s"]),
            "control_decimation": CONTROL_DECIMATION,
            "command_vx_mps": 0.07,
            "episode_wide_support_safety": True,
            "strict_verdict_modified": False,
        },
        "backend": {
            "dynamics": "native_mujoco_mj_step",
            "controller": "existing_mjlab_hybrid_action_term",
            "controller_measurement": "mjwarp_forward_only_from_native_state",
            "contact_verdict": "native_mujoco_wheel_terrain_contacts",
            "mjwarp_dynamics_steps": 0,
        },
        "model": model_metadata,
        "model_equivalence": model_equivalence,
        "qualification": qualification,
        "summaries": summaries,
        "trials": rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Refusing to overwrite native rollout output: {args.output}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_probe(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rb._atomic_write_json(output, payload)
    print(f"[native-full-rollout] output={output}")


if __name__ == "__main__":
    main()
