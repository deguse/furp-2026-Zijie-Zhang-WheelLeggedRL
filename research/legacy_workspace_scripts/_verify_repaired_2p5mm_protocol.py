from __future__ import annotations
import sys,json
from pathlib import Path
REPO=Path(r'D:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL\.worktrees\p2-classical-upper-bound')
for x in (REPO/'src',REPO/'src'/'hoppertrex_mjlab'):
 if str(x) not in sys.path:sys.path.insert(0,str(x))
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb
from mjlab.envs import ManagerBasedRlEnv
out=[]
for card in rb.POSTURE_CARDS:
 cfg=rb.make_roll_boundary_env_cfg((0.0025,),16);env=ManagerBasedRlEnv(cfg=cfg,device='cpu')
 try:
  rows=rb.run_card_repeat(env,heights=(0.0025,),card=card,repeat=1,settle_steps=100,drive_steps=500,stable_steps=25)
  out.append({'card':card['name'],'successes':sum(r['success'] for r in rows),'trials':len(rows),'bilateral_airborne_trials':sum(r['bilateral_airborne_ever'] for r in rows),'terminations':sum(r['termination'] for r in rows),'non_wheel_contacts':sum(r['non_wheel_contact'] for r in rows),'wheel_residual_abs_max':max(r['wheel_residual_abs_max'] for r in rows),'max_progress_min_m':min(r['max_progress_past_face_m'] for r in rows),'max_progress_max_m':max(r['max_progress_past_face_m'] for r in rows)})
 finally:env.close()
print(json.dumps(out,indent=2));Path(r'D:\mjlab_workspace\repaired_2p5mm_protocol.json').write_text(json.dumps({'evidence_eligible':False,'rows':out},indent=2)+'\n')
