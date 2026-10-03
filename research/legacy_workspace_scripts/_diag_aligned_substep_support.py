from __future__ import annotations
import sys,json,collections
from pathlib import Path
import torch
REPO=Path(r'D:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL\.worktrees\p2-classical-upper-bound')
for x in (REPO/'src',REPO/'src'/'hoppertrex_mjlab'):
 if str(x) not in sys.path:sys.path.insert(0,str(x))
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb
from mjlab.envs import ManagerBasedRlEnv

def substep_support(env,name):
 d=env.scene[name].data
 h=torch.linalg.vector_norm(d.force_history,dim=-1)>0
 # [B, slots, H] -> [B,H]
 return torch.any(h,dim=1)
def run(card):
 cfg=rb.make_roll_boundary_env_cfg((0.0,),16);env=ManagerBasedRlEnv(cfg=cfg,device='cpu')
 try:
  rb._reset_to_approach(env,root_height=float(card['height_m']),card_name=str(card['name']),repeat=1,height_count=1);a=torch.zeros((16,6));dist=collections.Counter();cycles=0;affected=torch.zeros(16,dtype=torch.bool);worst=0;examples=[]
  for step in range(600):
   active=torch.ones(16,dtype=torch.bool);rb._force_commands(env,active=active,vx=0 if step<100 else rb.COMMAND_VX_MPS,height=float(card['height_m']),pitch=float(card['pitch_rad']));env.step(a)
   if step<100:continue
   l=substep_support(env,rb.LEFT_SENSOR);r=substep_support(env,rb.RIGHT_SENSOR);unsupported=~l&~r
   for e in range(16):
    pattern=unsupported[e].tolist();count=sum(pattern);dist[count]+=1;cycles+=1
    affected[e]|=count>0
    # max consecutive within interval
    cur=mx=0
    for v in pattern:cur=cur+1 if v else 0;mx=max(mx,cur)
    worst=max(worst,mx)
    if count and len(examples)<20:examples.append({'step':step-100,'env':e,'unsupported_oldest_to_newest':pattern,'left_force_support':l[e].tolist(),'right_force_support':r[e].tolist()})
  return {'card':card['name'],'control_cycles':cycles,'cycles_by_unsupported_substep_count':dict(sorted(dist.items())),'affected_trials':int(affected.sum()),'max_consecutive_unsupported_physics_substeps_within_cycle':worst,'examples':examples}
 finally:env.close()
r=[run(c) for c in rb.POSTURE_CARDS];print(json.dumps(r,indent=2));Path(r'D:\mjlab_workspace\aligned_substep_support.json').write_text(json.dumps(r,indent=2)+'\n')
