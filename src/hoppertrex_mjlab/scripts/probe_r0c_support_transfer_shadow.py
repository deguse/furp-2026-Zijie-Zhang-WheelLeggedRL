"""Exact-reset 5 ms shadow probe for make-before-break support transfer.

The probe grants no actuator authority.  It replays the reviewed R0c C0 resets,
feeds per-side riser/vertical contact truth into the simulator-independent
support-transfer state machine, and records every phase transition alongside
the unchanged strict RollBoundary verdict.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mjlab.envs import ManagerBasedRlEnv

from hoppertrex_mjlab.hybrid.stair_trigger import stair_trigger_metric
from hoppertrex_mjlab.hybrid.support_transfer import (
    SupportTransferConfig,
    SupportTransferPhase,
    SupportTransferSensors,
    SupportTransferState,
    support_transfer_step,
)
from hoppertrex_mjlab.scripts import diagnose_roll_boundary as diag
from hoppertrex_mjlab.scripts import probe_r0c_yaw_feedback as lateral
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb

SCHEMA_VERSION = 1
HEIGHTS_M = (0.0, 0.0025)


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=rb.REPOSITORY_PATH,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _force_values(env: ManagerBasedRlEnv, sensor_name: str) -> tuple[Any, Any]:
    data = env.scene[sensor_name].data
    riser = stair_trigger_metric(
        found=data.found,
        force_contact_frame=data.force,
        normal_global=data.normal,
    )
    vertical = rb.vertical_normal_load_n(
        found=data.found,
        force_contact_frame=data.force,
        normal_global=data.normal,
    )
    return riser, vertical


class SupportTransferShadow:
    """CPU-only scalar-reference adapter called after every 5 ms scene update."""

    def __init__(self, env: ManagerBasedRlEnv, config: SupportTransferConfig):
        if env.device != "cpu":
            raise ValueError("Support-transfer scalar shadow probe is pinned to CPU.")
        if abs(float(env.physics_dt) - config.control_dt_s) > 1.0e-12:
            raise ValueError("Support-transfer shadow cadence must equal physics_dt.")
        self.env = env
        self.config = config
        self.states = [SupportTransferState() for _ in range(env.num_envs)]
        self.substep = 0
        self.transitions: list[list[dict[str, Any]]] = [[] for _ in range(env.num_envs)]
        self.phase_counts = [
            [0 for _ in SupportTransferPhase] for _ in range(env.num_envs)
        ]
        self.first_bilateral_zero: list[dict[str, Any] | None] = [
            None for _ in range(env.num_envs)
        ]
        self.bilateral_zero_after_trigger = [0 for _ in range(env.num_envs)]
        self.previous_root_x = env.scene["robot"].data.root_link_pos_w[:, 0].tolist()
        self.original_update = env.scene.update

    def install(self) -> None:
        self.env.scene.update = self.update

    def restore(self) -> None:
        self.env.scene.update = self.original_update

    def update(self, dt: float) -> None:
        self.original_update(dt)
        left_data = self.env.scene[rb.LEFT_SENSOR].data
        right_data = self.env.scene[rb.RIGHT_SENSOR].data
        left_riser, left_vertical = _force_values(self.env, rb.LEFT_SENSOR)
        right_riser, right_vertical = _force_values(self.env, rb.RIGHT_SENSOR)
        left_contact = (left_data.force.square().sum(dim=-1) > 0.0).any(dim=-1)
        right_contact = (right_data.force.square().sum(dim=-1) > 0.0).any(dim=-1)
        left_riser_values = left_riser.tolist()
        right_riser_values = right_riser.tolist()
        left_vertical_values = left_vertical.tolist()
        right_vertical_values = right_vertical.tolist()
        left_contact_values = left_contact.tolist()
        right_contact_values = right_contact.tolist()
        root_x_values = self.env.scene["robot"].data.root_link_pos_w[:, 0].tolist()
        forward_increments = [
            float(current - previous)
            for current, previous in zip(
                root_x_values, self.previous_root_x, strict=True
            )
        ]
        self.previous_root_x = root_x_values
        for env_id, state in enumerate(self.states):
            sensors = SupportTransferSensors(
                left_riser_force_n=float(left_riser_values[env_id]),
                right_riser_force_n=float(right_riser_values[env_id]),
                left_vertical_force_n=float(left_vertical_values[env_id]),
                right_vertical_force_n=float(right_vertical_values[env_id]),
                forward_increment_m=forward_increments[env_id],
            )
            _target, next_state = support_transfer_step(
                self.config,
                state,
                sensors,
                stair_request=True,
            )
            self.phase_counts[env_id][int(next_state.phase)] += 1
            bilateral_zero = (
                not left_contact_values[env_id] and not right_contact_values[env_id]
            )
            trigger_seen = next_state.left_riser_seen or next_state.right_riser_seen
            if bilateral_zero and trigger_seen:
                self.bilateral_zero_after_trigger[env_id] += 1
                if self.first_bilateral_zero[env_id] is None:
                    self.first_bilateral_zero[env_id] = {
                        "substep": self.substep,
                        "phase": next_state.phase.name,
                        "lead_side": next_state.lead_side.name,
                        "root_x_m": float(root_x_values[env_id]),
                        "left_riser_force_n": sensors.left_riser_force_n,
                        "right_riser_force_n": sensors.right_riser_force_n,
                        "left_vertical_force_n": sensors.left_vertical_force_n,
                        "right_vertical_force_n": sensors.right_vertical_force_n,
                    }
            if next_state.phase != state.phase:
                self.transitions[env_id].append(
                    {
                        "substep": self.substep,
                        "from_phase": state.phase.name,
                        "to_phase": next_state.phase.name,
                        "lead_side": next_state.lead_side.name,
                        "root_x_m": float(root_x_values[env_id]),
                        "left_riser_force_n": sensors.left_riser_force_n,
                        "right_riser_force_n": sensors.right_riser_force_n,
                        "left_vertical_force_n": sensors.left_vertical_force_n,
                        "right_vertical_force_n": sensors.right_vertical_force_n,
                        "abort_reason": next_state.abort_reason,
                    }
                )
            self.states[env_id] = next_state
        self.substep += 1


def _reset_override(
    source_resets: Mapping[tuple[int, float], Any],
) -> list[Any]:
    result = []
    for env_id in range(2 * diag.R0C_SYNC_ENVS_PER_HEIGHT):
        matches = [
            reset
            for (source_env_id, _height), reset in source_resets.items()
            if source_env_id == env_id
        ]
        if len(matches) != 1:
            raise ValueError("Source R0c resets do not map one-to-one by env id.")
        result.append(matches[0])
    return result


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.source_result.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    candidate = lateral._candidate_map()["c0"]
    source_resets = lateral._source_reset_map(source, "c0")
    config = SupportTransferConfig(control_dt_s=rb.ROLL_FIRST_PHYSICS_TIMESTEP_S)
    cfg = rb.make_roll_boundary_env_cfg(
        HEIGHTS_M,
        diag.R0C_SYNC_ENVS_PER_HEIGHT,
    )
    original_cards = rb.POSTURE_CARDS
    rb.POSTURE_CARDS = (candidate["posture_card"],)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    shadow = SupportTransferShadow(env, config)
    try:
        shadow.install()
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
            root_reset_override=_reset_override(source_resets),
            command_vx_mps=0.07,
        )
    finally:
        shadow.restore()
        env.close()
        rb.POSTURE_CARDS = original_cards
    lateral._validate_resets(rows, source_resets)

    for row in rows:
        env_id = int(row["env_id"])
        state = shadow.states[env_id]
        row["support_transfer_shadow"] = {
            "final_phase": state.phase.name,
            "lead_side": state.lead_side.name,
            "abort_reason": state.abort_reason,
            "travel_since_trigger_m": state.travel_since_trigger_m,
            "phase_substeps": {
                phase.name: shadow.phase_counts[env_id][int(phase)]
                for phase in SupportTransferPhase
            },
            "first_bilateral_zero_after_trigger": shadow.first_bilateral_zero[env_id],
            "bilateral_zero_after_trigger_substeps": shadow.bilateral_zero_after_trigger[
                env_id
            ],
            "transitions": shadow.transitions[env_id],
        }

    stair_rows = [row for row in rows if float(row["stair_height_m"]) > 0.0]
    phase_counts = {
        phase.name: sum(
            row["support_transfer_shadow"]["final_phase"] == phase.name
            for row in stair_rows
        )
        for phase in SupportTransferPhase
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "r0c_support_transfer_shadow_development_probe",
        "evidence_eligible": False,
        "promotion_eligible": False,
        "reason": "direct-contact scalar shadow; zero actuator authority",
        "git_sha": _git_value("rev-parse", "HEAD"),
        "project_dirty": bool(_git_value("status", "--porcelain")),
        "device": args.device,
        "source_result": str(source_path),
        "candidate_key": "c0",
        "matched_reviewed_resets": True,
        "actuator_authority": {
            "wheel": False,
            "leg": False,
            "strict_verdict_modified": False,
        },
        "config": {name: getattr(config, name) for name in config.__dataclass_fields__},
        "stair_final_phase_counts": phase_counts,
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
        parser.error(f"Refusing to overwrite shadow output: {args.output}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_probe(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rb._atomic_write_json(output, payload)
    print(f"[support-transfer-shadow] output={output}")


if __name__ == "__main__":
    main()
