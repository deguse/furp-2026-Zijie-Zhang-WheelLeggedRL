from __future__ import annotations
import sys,json
from pathlib import Path
import torch
REPO=Path(r'D:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL\.worktrees\p2-classical-upper-bound')
for x in (REPO/'src',REPO/'src'/'hoppertrex_mjlab'):
 if str(x) not in sys.path:sys.path.insert(0,str(x))
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb
from mjlab.envs import ManagerBasedRlEnv

def wc(env,n):
 f=env.scene[n].data.found;return bool(torch.any(f.reshape(f.shape[0],-1)>0,-1)[0])
cfg=rb.make_roll_boundary_env_cfg((0.0,),1);cfg.decimation=1
env=ManagerBasedRlEnv(cfg=cfg,device='cpu')
try:
 card=rb.POSTURE_CARDS[0];rb._reset_to_approach(env,root_height=float(card['height_m']),card_name=str(card['name']),repeat=1,height_count=1);robot=env.scene['robot'];a=torch.zeros((1,6));rows=[]
 # 400 physics steps settle = 2s, then 2000 drive =10s; same controller gets recalculated each 5ms (different cadence!) but diagnostic only
 for s in range(2400):
  active=torch.ones(1,dtype=torch.bool);vx=0 if s<400 else rb.COMMAND_VX_MPS;rb._force_commands(env,active=active,vx=vx,height=float(card['height_m']),pitch=float(card['pitch_rad']));env.step(a);lf=wc(env,rb.LEFT_SENSOR);rf=wc(env,rb.RIGHT_SENSOR);rows.append({'phase':'settle' if s<400 else 'drive','step':s if s<400 else s-400,'air':not lf and not rf,'lf':lf,'rf':rf,'z':float(robot.data.root_link_pos_w[0,2]),'vz':float(robot.data.root_link_lin_vel_w[0,2])})
 drive=[r for r in rows if r['phase']=='drive'];runs=[];st=None
 for i,r in enumerate(drive+[{'air':False}]):
  if r['air'] and st is None:st=i
  elif not r['air'] and st is not None:runs.append((st,i-st));st=None
 res={'air_steps':sum(r['air'] for r in drive),'runs':runs,'maxrun':max([x[1] for x in runs],default=0),'dt':env.step_dt,'vzmin':min(r['vz'] for r in drive),'vzmax':max(r['vz'] for r in drive)}
 print(json.dumps(res,indent=2));Path(r'D:\mjlab_workspace\airborne_physics_substeps.json').write_text(json.dumps({'summary':res,'rows':rows},indent=2)+'\n')
finally:env.close()
