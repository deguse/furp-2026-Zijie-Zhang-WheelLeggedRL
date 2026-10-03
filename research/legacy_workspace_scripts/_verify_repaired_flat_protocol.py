from __future__ import annotations
import sys,json
from pathlib import Path
import torch
REPO=Path(r'D:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL\.worktrees\p2-classical-upper-bound')
for x in (REPO/'src',REPO/'src'/'hoppertrex_mjlab'):
 if str(x) not in sys.path:sys.path.insert(0,str(x))
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb
from mjlab.envs import ManagerBasedRlEnv
r=[]
for card in rb.POSTURE_CARDS:
 cfg=rb.make_roll_boundary_env_cfg((0.0,),16);env=ManagerBasedRlEnv(cfg=cfg,device='cpu')
 try:
  rows=rb.run_card_repeat(env,heights=(0.0,),card=card,repeat=1,settle_steps=100,drive_steps=500,stable_steps=25)
  c=rb._cell_summary(rows);r.append({'card':card['name'],'successes':c['successes'],'trials':c['trials'],'bilateral_airborne_trials':c['bilateral_airborne_trials'],'terminations':c['terminated_trials'],'non_wheel_contacts':c['non_wheel_contact_trials'],'passed':c['passed'],'max_progress_min_m':min(x['max_progress_past_face_m'] for x in rows),'max_progress_max_m':max(x['max_progress_past_face_m'] for x in rows),'pitch_rate_peak_max':max(x['peak_pitch_rate_abs_radps'] for x in rows)})
 finally:env.close()
print(json.dumps(r,indent=2));Path(r'D:\mjlab_workspace\repaired_flat_protocol.json').write_text(json.dumps(r,indent=2)+'\n')
