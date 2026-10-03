import hashlib, json, os
from collections import Counter
import torch
from hoppertrex_mjlab.scripts.rsl_rl.stair_dynamic_search_live_adapter import (
    _configure_env, _load_stage5_policy, _force_commands, _dynamic_danger,
    SETTLE_STEPS, DRIVE_STEPS,
)

checkpoint = os.environ['HOPPERTREX_DYNAMIC_STAIR_STAGE5_CHECKPOINT_PATH']
digest = hashlib.sha256(open(checkpoint, 'rb').read()).hexdigest()
code_names = {0:'none',1:'non_wheel_contact',2:'actuator_limit',3:'orientation_limit',4:'backward_progress',5:'contact_timeout',6:'trail_contact_timeout',7:'cross_timeout',8:'target_saturation'}
result = {}
for family in ('synchronized','alternating'):
    env = _configure_env(family=family, num_envs=8, device='cpu')
    try:
        wrapped, owner, policy = _load_stage5_policy(env, expected_sha256=digest, device='cpu')
        action = env.action_manager.get_term('hybrid_wheel_leg')
        params = torch.tensor([[.035,.045,.20,1.0]], device=env.device).repeat(8,1)
        action.set_dynamic_candidate_parameters(params)
        with torch.no_grad():
            for _ in range(SETTLE_STEPS):
                _force_commands(env, vx=0.0, stair_request=False)
                env.step(policy(wrapped.get_observations()))
            start_x = env.scene['robot'].data.root_link_pos_w[:,0].clone()
            seen = torch.zeros(8, dtype=torch.bool, device=env.device)
            first = [None] * 8
            max_progress = torch.zeros(8, device=env.device)
            for step in range(DRIVE_STEPS):
                _force_commands(env, vx=.07, stair_request=True)
                _, _, terminated, timed_out, _ = env.step(policy(wrapped.get_observations()))
                progress = env.scene['robot'].data.root_link_pos_w[:,0] - start_x
                max_progress = torch.maximum(max_progress, progress)
                danger = _dynamic_danger(action, terminated, timed_out)
                new = danger & ~seen
                for idx in torch.nonzero(new).flatten().tolist():
                    code = int(action.dynamic_abort_code[idx].item())
                    first[idx] = {
                        'env': idx,
                        'step': step,
                        'abort_code': code,
                        'abort_name': code_names.get(code, 'unknown'),
                        'phase': int(action.dynamic_phase[idx].item()),
                        'traversal_mode': int(action.dynamic_traversal_mode[idx].item()),
                        'target_saturation': bool(action.dynamic_target_saturation[idx].item()),
                        'episode_unsafe': bool(action.dynamic_episode_unsafe[idx].item()),
                        'terminated': bool(terminated[idx].item()),
                        'timed_out': bool(timed_out[idx].item()),
                        'progress_m': float(progress[idx].item()),
                        'max_progress_m': float(max_progress[idx].item()),
                    }
                seen |= danger
            counts = Counter(row['abort_name'] if row else 'no_danger' for row in first)
            result[family] = {'counts': dict(counts), 'first_danger': first}
    finally:
        env.close()
print('DIAGNOSTIC_JSON_START')
print(json.dumps(result, indent=2, sort_keys=True))
