"""Replay MJWarp zero-support substeps in native MuJoCo.

The probe captures the dynamic state at the beginning of every 5 ms substep
that MJWarp reports as bilateral zero wheel force.  It then advances the same
state and control through the same host ``MjModel`` with native MuJoCo.

MjLab contact sensors observed inside ``scene.update()`` describe the
constraint solve performed before that substep's integration, while qpos and
qvel already contain the integrated endpoint.  The native replay therefore
records both quantities explicitly:

* ``during_step_contact``: immediately after ``mj_step`` and before any extra
  ``mj_forward``; this is the force comparison matching the strict recorder.
* ``integrated_endpoint_contact``: after forwarding the integrated endpoint;
  this is a secondary geometric/contact-persistence check.

The probe never changes controller authority or the strict RollBoundary
verdict.  Its artifacts are development diagnostics, not promotion evidence.
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
import torch
from mjlab.envs import ManagerBasedRlEnv

from hoppertrex_mjlab.scripts import diagnose_roll_boundary as diag
from hoppertrex_mjlab.scripts import probe_r0c_support_transfer_shadow as shadow_probe
from hoppertrex_mjlab.scripts import probe_r0c_yaw_feedback as lateral
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb

SCHEMA_VERSION = 2
HEIGHTS_M = (0.0, 0.0025)
_STATE_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "qacc_warmstart",
    "qfrc_applied",
    "xfrc_applied",
    "ctrl",
)
_MODEL_EQUIVALENCE_FIELDS = (
    "geom_pos",
    "geom_quat",
    "geom_size",
    "geom_type",
    "geom_friction",
    "geom_solref",
    "geom_solimp",
    "geom_margin",
    "geom_gap",
    "geom_condim",
    "geom_priority",
    "geom_contype",
    "geom_conaffinity",
    "body_mass",
    "body_inertia",
    "body_pos",
    "body_quat",
    "dof_damping",
    "dof_frictionloss",
    "dof_armature",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
    "actuator_ctrlrange",
)
_MODEL_HOST_ATOL = 1.0e-6


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


def _model_equivalence_report(env: ManagerBasedRlEnv) -> dict[str, Any]:
    """Prove that every MJWarp world uses the host model being replayed."""

    bridge_model = env.sim.model
    host_model = env.sim.mj_model
    fields: dict[str, dict[str, Any]] = {}
    for name in _MODEL_EQUIVALENCE_FIELDS:
        wrapped = getattr(bridge_model, name)
        tensor = wrapped.torch() if hasattr(wrapped, "torch") else wrapped
        batched = np.asarray(tensor.detach().cpu(), dtype=np.float64)
        host = np.asarray(getattr(host_model, name), dtype=np.float64)
        if batched.shape == host.shape:
            reference = batched
            world_delta = 0.0
        elif (
            batched.ndim == host.ndim + 1
            and batched.shape[0] == env.num_envs
            and batched.shape[1:] == host.shape
        ):
            reference = batched[0]
            world_delta = float(np.max(np.abs(batched - reference)))
        else:
            raise RuntimeError(
                f"Unexpected model-field shape for {name}: "
                f"MJWarp {batched.shape}, host {host.shape}."
            )
        host_delta = (
            float(np.max(np.abs(reference - host))) if host.size else 0.0
        )
        fields[name] = {
            "mjwarp_shape": list(batched.shape),
            "host_shape": list(host.shape),
            "max_abs_delta_across_worlds": world_delta,
            "max_abs_delta_host_vs_world0": host_delta,
        }

    world_max = max(
        value["max_abs_delta_across_worlds"] for value in fields.values()
    )
    host_max = max(
        value["max_abs_delta_host_vs_world0"] for value in fields.values()
    )
    if world_max != 0.0:
        raise RuntimeError(
            "Native replay is invalid because MJWarp model parameters differ "
            f"across worlds (max delta {world_max})."
        )
    if host_max > _MODEL_HOST_ATOL:
        raise RuntimeError(
            "Native replay is invalid because the host MjModel differs from "
            f"MJWarp world 0 (max delta {host_max})."
        )
    return {
        "all_worlds_identical": True,
        "host_matches_world0_within_float32_roundoff": True,
        "host_absolute_tolerance": _MODEL_HOST_ATOL,
        "max_abs_delta_across_worlds": world_max,
        "max_abs_delta_host_vs_world0": host_max,
        "fields": fields,
    }


class UnsupportedSubstepCapture:
    """Capture every MJWarp bilateral-zero-force physics substep."""

    def __init__(self, env: ManagerBasedRlEnv):
        if env.device != "cpu":
            raise ValueError("Native contact replay capture is CPU-only.")
        self.env = env
        self.original_update = env.scene.update
        self.previous: dict[str, torch.Tensor] | None = None
        self.captures: list[list[dict[str, Any]]] = [
            [] for _ in range(env.num_envs)
        ]
        self.substep = 0

    def install(self) -> None:
        self.env.scene.update = self.update

    def restore(self) -> None:
        self.env.scene.update = self.original_update

    def _snapshot(self) -> dict[str, torch.Tensor]:
        data = self.env.sim.data
        return {
            name: getattr(data, name).clone()
            for name in _STATE_FIELDS
        } | {"time": data.time.clone()}

    def update(self, dt: float) -> None:
        self.original_update(dt)
        current = self._snapshot()
        left_data = self.env.scene[rb.LEFT_SENSOR].data
        right_data = self.env.scene[rb.RIGHT_SENSOR].data
        left = (left_data.force.square().sum(dim=-1) > 0.0).any(dim=-1)
        right = (right_data.force.square().sum(dim=-1) > 0.0).any(dim=-1)
        unsupported = ~left & ~right
        if self.previous is not None:
            for env_id in torch.nonzero(unsupported, as_tuple=False).flatten().tolist():
                previous = self.previous
                self.captures[env_id].append(
                    {
                        "env_id": env_id,
                        "event_substep": self.substep,
                        "substep_dt_s": float(dt),
                        "pre": {
                            name: previous[name][env_id].tolist()
                            for name in _STATE_FIELDS
                        },
                        "pre_time_s": float(previous["time"][env_id]),
                        "event_ctrl": current["ctrl"][env_id].tolist(),
                        "event_time_s": float(current["time"][env_id]),
                        # sensordata/constraint force belongs to the solve at
                        # the start of this substep, not endpoint qpos/qvel.
                        "warp_during_step_contact": {
                            "left_force_norm_n": float(
                                torch.linalg.vector_norm(
                                    left_data.force[env_id], dim=-1
                                ).sum()
                            ),
                            "right_force_norm_n": float(
                                torch.linalg.vector_norm(
                                    right_data.force[env_id], dim=-1
                                ).sum()
                            ),
                            "bilateral_zero_force": True,
                        },
                        "warp_integrated_endpoint": {
                            "qpos": current["qpos"][env_id].tolist(),
                            "qvel": current["qvel"][env_id].tolist(),
                        },
                    }
                )
        self.previous = current
        self.substep += 1


def _wheel_geom_ids(model: mujoco.MjModel) -> tuple[int, int]:
    ids = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{name}")
        for name in ("wheel_left_collision", "wheel_right_collision")
    )
    if any(value < 0 for value in ids):
        raise ValueError("Native replay could not resolve both wheel geoms.")
    return ids  # type: ignore[return-value]


def _terrain_geom_ids(model: mujoco.MjModel) -> frozenset[int]:
    terrain_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "terrain"
    )
    if terrain_body_id < 0:
        raise ValueError("Native replay could not resolve the terrain body.")
    ids = frozenset(
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == terrain_body_id
    )
    if not ids:
        raise ValueError("Native replay found no terrain geoms.")
    return ids


def _native_wheel_forces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    wheel_geom_ids: tuple[int, int],
    terrain_geom_ids: frozenset[int],
) -> dict[str, Any]:
    """Read wheel-terrain forces with the same pairing as RollBoundary sensors."""

    totals = [0.0, 0.0]
    contact_counts = [0, 0]
    contacts = []
    wrench = np.zeros(6, dtype=np.float64)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        for side, wheel_id in enumerate(wheel_geom_ids):
            if contact.geom1 == wheel_id:
                other_geom_id = int(contact.geom2)
            elif contact.geom2 == wheel_id:
                other_geom_id = int(contact.geom1)
            else:
                continue
            if other_geom_id not in terrain_geom_ids:
                continue
            wrench.fill(0.0)
            mujoco.mj_contactForce(model, data, contact_id, wrench)
            force_norm = float(np.linalg.norm(wrench[:3]))
            totals[side] += force_norm
            contact_counts[side] += 1
            contacts.append(
                {
                    "side": "left" if side == 0 else "right",
                    "contact_id": contact_id,
                    "other_geom_id": other_geom_id,
                    "other_geom_name": mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, other_geom_id
                    ),
                    "force_norm_n": force_norm,
                    "distance_m": float(contact.dist),
                }
            )
    return {
        "left_force_norm_n": totals[0],
        "right_force_norm_n": totals[1],
        "left_contact_count": contact_counts[0],
        "right_contact_count": contact_counts[1],
        "bilateral_zero_force": totals[0] == 0.0 and totals[1] == 0.0,
        "contacts": contacts,
    }


def _load_native_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    capture: Mapping[str, Any],
) -> None:
    pre = capture["pre"]
    data.qpos[:] = np.asarray(pre["qpos"], dtype=np.float64)
    data.qvel[:] = np.asarray(pre["qvel"], dtype=np.float64)
    data.act[:] = np.asarray(pre.get("act", []), dtype=np.float64)
    data.qacc_warmstart[:] = np.asarray(
        pre["qacc_warmstart"], dtype=np.float64
    )
    data.qfrc_applied[:] = np.asarray(
        pre.get("qfrc_applied", np.zeros(model.nv)), dtype=np.float64
    )
    data.xfrc_applied[:] = np.asarray(
        pre.get("xfrc_applied", np.zeros((model.nbody, 6))), dtype=np.float64
    )
    data.ctrl[:] = np.asarray(capture["event_ctrl"], dtype=np.float64)
    data.time = float(capture["pre_time_s"])


def _native_replay(
    model: mujoco.MjModel,
    capture: Mapping[str, Any],
) -> dict[str, Any]:
    wheel_ids = _wheel_geom_ids(model)
    terrain_ids = _terrain_geom_ids(model)

    # Use a separate data object for the start-state diagnostic so mj_forward
    # cannot alter the warm-start state used by the actual transition replay.
    start_data = mujoco.MjData(model)
    _load_native_state(model, start_data, capture)
    mujoco.mj_forward(model, start_data)
    start_contact = _native_wheel_forces(
        model, start_data, wheel_ids, terrain_ids
    )

    step_data = mujoco.MjData(model)
    _load_native_state(model, step_data, capture)
    mujoco.mj_step(model, step_data)
    # MuJoCo leaves contacts and constraint forces for the solve that produced
    # this integration step.  This is the strict, same-phase comparison.
    during_step_contact = _native_wheel_forces(
        model, step_data, wheel_ids, terrain_ids
    )
    post_qpos = step_data.qpos.copy()
    post_qvel = step_data.qvel.copy()
    post_time_s = float(step_data.time)

    # Forward only after recording the during-step solve.  This second view
    # reports contact at the already integrated endpoint.
    mujoco.mj_forward(model, step_data)
    endpoint_contact = _native_wheel_forces(
        model, step_data, wheel_ids, terrain_ids
    )

    warp_endpoint = capture["warp_integrated_endpoint"]
    qpos_error = np.asarray(warp_endpoint["qpos"], dtype=np.float64) - post_qpos
    qvel_error = np.asarray(warp_endpoint["qvel"], dtype=np.float64) - post_qvel
    pre_ctrl = np.asarray(capture["pre"]["ctrl"], dtype=np.float64)
    event_ctrl = np.asarray(capture["event_ctrl"], dtype=np.float64)
    event_dt = float(capture["event_time_s"]) - float(capture["pre_time_s"])

    return {
        "model_timestep_s": float(model.opt.timestep),
        "captured_event_dt_s": event_dt,
        "event_ctrl_change_abs_max": float(np.max(np.abs(event_ctrl - pre_ctrl))),
        "start_state_contact": start_contact,
        "during_step_contact": during_step_contact,
        "integrated_endpoint_contact": endpoint_contact,
        "integrated_endpoint": {
            "time_s": post_time_s,
            "qpos": post_qpos.tolist(),
            "qvel": post_qvel.tolist(),
            "warp_native_qpos_abs_max_m_or_rad": float(np.max(np.abs(qpos_error))),
            "warp_native_qvel_abs_max_mps_or_radps": float(
                np.max(np.abs(qvel_error))
            ),
        },
    }


def _validate_capture_counts(
    captures: Sequence[Sequence[Mapping[str, Any]]],
    trials_by_env: Mapping[int, Mapping[str, Any]],
) -> None:
    expected = {
        env_id: int(trial["bilateral_unsupported_physics_substeps"])
        for env_id, trial in trials_by_env.items()
    }
    observed = {env_id: len(items) for env_id, items in enumerate(captures)}
    mismatches = {
        env_id: {"expected": expected.get(env_id, 0), "observed": count}
        for env_id, count in observed.items()
        if count != expected.get(env_id, 0)
    }
    if mismatches:
        raise RuntimeError(f"Native replay capture-count mismatch: {mismatches}.")


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.source_result.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    candidate = lateral._candidate_map()["c0"]
    source_resets = lateral._source_reset_map(source, "c0")
    cfg = rb.make_roll_boundary_env_cfg(
        HEIGHTS_M,
        diag.R0C_SYNC_ENVS_PER_HEIGHT,
    )
    original_cards = rb.POSTURE_CARDS
    rb.POSTURE_CARDS = (candidate["posture_card"],)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    capture = UnsupportedSubstepCapture(env)
    try:
        model_equivalence = _model_equivalence_report(env)
        capture.install()
        rows = rb.run_card_repeat(
            env,
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
        host_model = env.sim.mj_model
    finally:
        capture.restore()
        env.close()
        rb.POSTURE_CARDS = original_cards
    lateral._validate_resets(rows, source_resets)

    trials_by_env = {int(row["env_id"]): row for row in rows}
    _validate_capture_counts(capture.captures, trials_by_env)
    replays = []
    for env_id, items in enumerate(capture.captures):
        for event_index, item in enumerate(items):
            replay = _native_replay(host_model, item)
            replays.append(
                {
                    "env_id": env_id,
                    "event_index_within_env": event_index,
                    "stair_height_m": float(
                        trials_by_env[env_id]["stair_height_m"]
                    ),
                    "strict_unsupported_substeps_in_trial": int(
                        trials_by_env[env_id][
                            "bilateral_unsupported_physics_substeps"
                        ]
                    ),
                    "capture": item,
                    "native_replay": replay,
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "r0c_native_contact_transition_replay_development_probe",
        "evidence_eligible": False,
        "promotion_eligible": False,
        "reason": "same-phase native replay of every MJWarp zero-support substep",
        "git_sha": _git_value("rev-parse", "HEAD"),
        "project_dirty": bool(_git_value("status", "--porcelain")),
        "device": args.device,
        "source_result": str(source_path),
        "source_result_sha256": _sha256(source_path),
        "matched_reviewed_resets": True,
        "strict_verdict_modified": False,
        "contact_pairing": "wheel collision geom against terrain body only",
        "sensor_timing": (
            "MJLab scene.update contact force and native during_step_contact both "
            "refer to the solve at the substep start; endpoint contact is secondary"
        ),
        "model": {
            "timestep_s": float(host_model.opt.timestep),
            "nq": int(host_model.nq),
            "nv": int(host_model.nv),
            "nu": int(host_model.nu),
            "na": int(host_model.na),
        },
        "model_equivalence": model_equivalence,
        "captured_trial_count": len(
            {int(row["env_id"]) for row in replays}
        ),
        "captured_event_count": len(replays),
        "native_during_step_bilateral_zero_count": sum(
            bool(row["native_replay"]["during_step_contact"]["bilateral_zero_force"])
            for row in replays
        ),
        "native_endpoint_bilateral_zero_count": sum(
            bool(
                row["native_replay"]["integrated_endpoint_contact"][
                    "bilateral_zero_force"
                ]
            )
            for row in replays
        ),
        "replays": replays,
        "summaries": diag.summarize_trials(rows),
        "trials": rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Refusing to overwrite native replay output: {args.output}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_probe(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rb._atomic_write_json(output, payload)
    print(f"[native-contact-replay] output={output}")


if __name__ == "__main__":
    main()
