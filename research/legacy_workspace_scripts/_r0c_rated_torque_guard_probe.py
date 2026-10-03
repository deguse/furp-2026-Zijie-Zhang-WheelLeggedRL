from __future__ import annotations

import argparse
import json
import math
import types
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv

from hoppertrex_mjlab.assets.HopperTrex_CFG import (
    RMD_L_9025_35T_RATED_TORQUE,
    WHEEL_VELOCITY_DAMPING,
)
from hoppertrex_mjlab.scripts import diagnose_roll_boundary as diag
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb
from hoppertrex_mjlab.scripts import probe_r0c_yaw_feedback as lateral

HEIGHTS_M = (0.0, 0.0025)
TRIGGER_N = 5.0
HOLD_S = 0.60


def side_metric(sensor):
    data = sensor.data
    magnitude = (data.force[..., 0] * data.normal[..., 0]).abs()
    return torch.where(data.found > 0, magnitude, torch.zeros_like(magnitude)).amax(dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-result', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    source_path = args.source_result.resolve()
    source = json.loads(source_path.read_text(encoding='utf-8'))
    candidate = lateral._candidate_map()['c0']
    source_resets = lateral._source_reset_map(source, 'c0')
    reset_override = []
    for env_id in range(2 * diag.R0C_SYNC_ENVS_PER_HEIGHT):
        matches = [reset for (sid, _height), reset in source_resets.items() if sid == env_id]
        if len(matches) != 1:
            raise RuntimeError(f'env {env_id} reset mismatch')
        reset_override.append(matches[0])

    cfg = rb.make_roll_boundary_env_cfg(HEIGHTS_M, diag.R0C_SYNC_ENVS_PER_HEIGHT)
    original_cards = rb.POSTURE_CARDS
    rb.POSTURE_CARDS = (candidate['posture_card'],)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    try:
        term = env.action_manager.get_term('hybrid_wheel_leg')
        robot = env.scene['robot']
        original_apply = term.apply_actions
        hold_substeps = round(HOLD_S / float(env.physics_dt))
        if not math.isclose(hold_substeps * float(env.physics_dt), HOLD_S, abs_tol=1e-12):
            raise RuntimeError('hold duration not exact')
        timers = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.long)
        trigger_count = torch.zeros_like(timers)
        guarded_substeps = torch.zeros_like(timers)
        max_target_error = RMD_L_9025_35T_RATED_TORQUE / WHEEL_VELOCITY_DAMPING

        def guarded_apply(self) -> None:
            left = side_metric(env.scene[rb.LEFT_SENSOR])
            right = side_metric(env.scene[rb.RIGHT_SENSOR])
            hit = torch.stack((left >= TRIGGER_N, right >= TRIGGER_N), dim=1)
            rising = hit & (timers == 0)
            trigger_count.add_(rising.long())
            timers.copy_(torch.where(hit, torch.full_like(timers, hold_substeps), torch.clamp(timers - 1, min=0)))
            guard = timers > 0
            guarded_substeps.add_(guard.long())

            desired = self._wheel_targets.clone()
            actual = robot.data.joint_vel[:, self._wheel_ids]
            rated_target = actual + torch.clamp(
                desired - actual, -max_target_error, max_target_error
            )
            self._wheel_targets.copy_(torch.where(guard, rated_target, desired))
            try:
                original_apply()
            finally:
                self._wheel_targets.copy_(desired)

        term.apply_actions = types.MethodType(guarded_apply, term)
        rows = rb.run_card_repeat(
            env,
            heights=HEIGHTS_M,
            card=candidate['posture_card'],
            repeat=1,
            settle_steps=rb.OFFICIAL_SETTLE_STEPS,
            drive_steps=rb.OFFICIAL_DRIVE_STEPS,
            stable_steps=rb.OFFICIAL_STABLE_STEPS,
            episode_wide_safety=True,
            diagnostic_continue_after_support_loss=True,
            roll_pose_schedule=candidate['schedule'],
            roll_pose_slew_mode=str(candidate['slew_mode']),
            require_pure_classical_authority=True,
            record_diagnostic_control_trace=False,
            root_reset_override=reset_override,
            command_vx_mps=0.07,
        )
    finally:
        env.close()
        rb.POSTURE_CARDS = original_cards

    lateral._validate_resets(rows, source_resets)
    payload = {
        'kind': 'r0c_rated_torque_contact_guard_development_probe',
        'evidence_eligible': False,
        'promotion_eligible': False,
        'source_result': str(source_path),
        'candidate_key': 'c0',
        'matched_reviewed_resets': True,
        'controller': {
            'trigger_metric': 'per-side max |contact_frame_F0 * global_normal_x|',
            'trigger_n': TRIGGER_N,
            'hold_s': HOLD_S,
            'wheel_torque_limit_nm': RMD_L_9025_35T_RATED_TORQUE,
            'velocity_actuator_damping': WHEEL_VELOCITY_DAMPING,
            'max_target_error_radps': max_target_error,
            'leg_action': 'unchanged',
            'purpose': 'single causal probe; direct contact truth is not deployable',
        },
        'trigger_count_per_env_side': trigger_count.cpu().tolist(),
        'guarded_substeps_per_env_side': guarded_substeps.cpu().tolist(),
        'summaries': diag.summarize_trials(rows),
        'trials': rows,
    }
    rb._atomic_write_json(args.output.resolve(), payload)
    for summary in payload['summaries']:
        print(json.dumps(summary, sort_keys=True))
    print(f'output={args.output.resolve()}')


if __name__ == '__main__':
    main()
