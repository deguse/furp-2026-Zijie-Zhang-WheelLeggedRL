"""Exact-reset causal probe for the bounded per-side leg force filter.

This rejected development-only probe uses the verified 5 ms support-transfer
shadow to apply leg-position admittance offsets.  Wheel targets and all
residuals remain exactly unchanged so the result isolates leg force
redistribution.  It is retained for negative-result reproducibility and must
not be connected to the formal controller.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv

from hoppertrex_mjlab.hybrid.stair_dynamic import LeadSide
from hoppertrex_mjlab.hybrid.support_force_filter import (
    SupportForceFilterConfig,
    SupportForceFilterState,
    support_force_filter_step,
)
from hoppertrex_mjlab.hybrid.support_transfer import (
    SupportTransferConfig,
    SupportTransferPhase,
    SupportTransferTargets,
)
from hoppertrex_mjlab.scripts import diagnose_roll_boundary as diag
from hoppertrex_mjlab.scripts import probe_r0c_support_transfer_shadow as shadow_probe
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


def _roles(
    shadow: shadow_probe.SupportTransferShadow, env_id: int
) -> SupportTransferTargets:
    state = shadow.states[env_id]
    if state.phase == SupportTransferPhase.LEAD_GUARD:
        hold = LeadSide.RIGHT if state.lead_side == LeadSide.LEFT else LeadSide.LEFT
        compliant = state.lead_side
        active = True
    elif state.phase == SupportTransferPhase.TRAIL_TRANSFER:
        hold = state.lead_side
        compliant = (
            LeadSide.RIGHT if state.lead_side == LeadSide.LEFT else LeadSide.LEFT
        )
        active = True
    else:
        hold = LeadSide.NONE
        compliant = LeadSide.NONE
        active = state.phase == SupportTransferPhase.BOTH_SUPPORTED
    return SupportTransferTargets(
        phase=state.phase,
        lead_side=state.lead_side,
        hold_support_side=hold,
        compliant_side=compliant,
        active=active,
        abort=state.phase == SupportTransferPhase.ABORT,
    )


class LegForceFilterAdapter:
    """Apply scalar-reference force offsets before each physics substep."""

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        shadow: shadow_probe.SupportTransferShadow,
        config: SupportForceFilterConfig,
    ):
        self.env = env
        self.shadow = shadow
        self.config = config
        self.term = env.action_manager.get_term("hybrid_wheel_leg")
        self.original_apply = self.term.apply_actions
        self.states = [SupportForceFilterState() for _ in range(env.num_envs)]
        self.max_abs_amplitude = torch.zeros((env.num_envs, 2), device=env.device)
        self.active_substeps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

    def install(self) -> None:
        self.term.apply_actions = types.MethodType(self.apply, self.term)

    def restore(self) -> None:
        self.term.apply_actions = self.original_apply

    def apply(self, _term: Any) -> None:
        left_vertical = rb.wheel_vertical_normal_load_n(
            self.env,
            rb.LEFT_SENSOR,
        ).tolist()
        right_vertical = rb.wheel_vertical_normal_load_n(
            self.env,
            rb.RIGHT_SENSOR,
        ).tolist()
        offsets = []
        for env_id, state in enumerate(self.states):
            roles = _roles(self.shadow, env_id)
            output, next_state = support_force_filter_step(
                self.config,
                state,
                roles,
                left_vertical_force_n=float(left_vertical[env_id]),
                right_vertical_force_n=float(right_vertical[env_id]),
            )
            self.states[env_id] = next_state
            offsets.append(output.leg_offsets_rad)
            self.max_abs_amplitude[env_id, 0] = max(
                self.max_abs_amplitude[env_id, 0], abs(output.left_amplitude_rad)
            )
            self.max_abs_amplitude[env_id, 1] = max(
                self.max_abs_amplitude[env_id, 1], abs(output.right_amplitude_rad)
            )
            self.active_substeps[env_id] += int(roles.active)
        offset_tensor = torch.tensor(
            offsets,
            dtype=self.term._leg_targets.dtype,
            device=self.term._leg_targets.device,
        )
        nominal = self.term._leg_targets.clone()
        self.term._leg_targets.add_(offset_tensor)
        try:
            self.original_apply()
        finally:
            self.term._leg_targets.copy_(nominal)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.source_result.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    candidate = lateral._candidate_map()["c0"]
    source_resets = lateral._source_reset_map(source, "c0")
    transfer_config = SupportTransferConfig(
        control_dt_s=rb.ROLL_FIRST_PHYSICS_TIMESTEP_S,
    )
    filter_config = SupportForceFilterConfig(
        control_dt_s=rb.ROLL_FIRST_PHYSICS_TIMESTEP_S,
    )
    if not math.isclose(
        transfer_config.control_dt_s,
        filter_config.control_dt_s,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError("Support state and force filter cadences differ.")
    cfg = rb.make_roll_boundary_env_cfg(
        HEIGHTS_M,
        diag.R0C_SYNC_ENVS_PER_HEIGHT,
    )
    original_cards = rb.POSTURE_CARDS
    rb.POSTURE_CARDS = (candidate["posture_card"],)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    shadow = shadow_probe.SupportTransferShadow(env, transfer_config)
    adapter = LegForceFilterAdapter(env, shadow, filter_config)
    try:
        shadow.install()
        adapter.install()
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
    finally:
        adapter.restore()
        shadow.restore()
        env.close()
        rb.POSTURE_CARDS = original_cards
    lateral._validate_resets(rows, source_resets)

    max_amplitude = adapter.max_abs_amplitude.cpu().tolist()
    active_substeps = adapter.active_substeps.cpu().tolist()
    for row in rows:
        env_id = int(row["env_id"])
        row["support_force_filter"] = {
            "max_abs_left_amplitude_rad": max_amplitude[env_id][0],
            "max_abs_right_amplitude_rad": max_amplitude[env_id][1],
            "active_physics_substeps": active_substeps[env_id],
            "final_state": {
                name: getattr(adapter.states[env_id], name)
                for name in adapter.states[env_id].__dataclass_fields__
            },
            "support_transfer_final_phase": shadow.states[env_id].phase.name,
            "support_transfer_transitions": shadow.transitions[env_id],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "r0c_support_force_filter_development_probe",
        "evidence_eligible": False,
        "promotion_eligible": False,
        "reason": "direct-contact causal leg-authority probe",
        "git_sha": _git_value("rev-parse", "HEAD"),
        "project_dirty": bool(_git_value("status", "--porcelain")),
        "device": args.device,
        "source_result": str(source_path),
        "candidate_key": "c0",
        "matched_reviewed_resets": True,
        "actuator_authority": {
            "wheel_target_modified": False,
            "leg_position_admittance": True,
            "residual_required_zero": True,
            "strict_verdict_modified": False,
        },
        "support_transfer_config": {
            name: getattr(transfer_config, name)
            for name in transfer_config.__dataclass_fields__
        },
        "support_force_filter_config": {
            name: getattr(filter_config, name)
            for name in filter_config.__dataclass_fields__
        },
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
        parser.error(f"Refusing to overwrite support-force output: {args.output}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_probe(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rb._atomic_write_json(output, payload)
    print(f"[support-force-filter] output={output}")


if __name__ == "__main__":
    main()
