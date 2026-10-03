from __future__ import annotations
import sys,json
from pathlib import Path
import torch
REPO=Path(r'D:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL\.worktrees\p2-classical-upper-bound')
for x in (REPO/'src',REPO/'src'/'hoppertrex_mjlab'):
 if str(x) not in sys.path:sys.path.insert(0,str(x))
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb
from mjlab.envs import ManagerBasedRlEnv

def current(env,n):
 d=env.scene[n].data;return torch.any(d.found.reshape(d.found.shape[0],-1)>0,-1)
def interval(env,n):
 d=env.scene[n].data
 f=current(env,n)
 assert d.force_history is not None
 hist=torch.linalg.vector_norm(d.force_history,dim=-1)>0
 return f|torch.any(hist.reshape(hist.shape[0],-1),dim=-1)
def run(card):
 cfg=rb.make_roll_boundary_env_cfg((0.0,),16)
 for s in cfg.scene.sensors:
  if s.name in (rb.LEFT_SENSOR,rb.RIGHT_SENSOR):s.history_length=cfg.decimation
 env=ManagerBasedRlEnv(cfg=cfg,device='cpu')
 try:
  rb._reset_to_approach(env,root_height=float(card['height_m']),card_name=str(card['name']),repeat=1,height_count=1);a=torch.zeros((16,6));cur=[];inter=[]
  for step in range(600):
   active=torch.ones(16,dtype=torch.bool);rb._force_commands(env,active=active,vx=0 if step<100 else rb.COMMAND_VX_MPS,height=float(card['height_m']),pitch=float(card['pitch_rad']));env.step(a)
   lc,rc=current(env,rb.LEFT_SENSOR),current(env,rb.RIGHT_SENSOR);li,ri=interval(env,rb.LEFT_SENSOR),interval(env,rb.RIGHT_SENSOR)
   if step>=100:cur.append((~lc&~rc).cpu());inter.append((~li&~ri).cpu())
  cur=torch.stack(cur);inter=torch.stack(inter)
  def mr(a):
   m=0
   for e in range(16):
    x=0
    for v in a[:,e]:x=x+1 if v else 0;m=max(m,x)
   return m
  return {'card':card['name'],'current_sample_airborne_steps':int(cur.sum()),'current_sample_airborne_trials':int(cur.any(0).sum()),'current_sample_max_run':mr(cur),'whole_control_interval_unsupported_steps':int(inter.sum()),'whole_control_interval_unsupported_trials':int(inter.any(0).sum()),'whole_control_interval_max_run':mr(inter)}
 finally:env.close()
r=[run(c) for c in rb.POSTURE_CARDS]
print(json.dumps(r,indent=2));Path(r'D:\mjlab_workspace\contact_history_support.json').write_text(json.dumps(r,indent=2)+'\n')
