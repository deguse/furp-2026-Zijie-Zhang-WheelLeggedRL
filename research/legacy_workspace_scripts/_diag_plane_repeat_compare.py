from __future__ import annotations
import sys,json
from pathlib import Path
import torch
REPO=Path(r'D:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL\.worktrees\p2-classical-upper-bound')
for x in (REPO/'src',REPO/'src'/'hoppertrex_mjlab'):
 if str(x) not in sys.path:sys.path.insert(0,str(x))
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb
from mjlab.envs import ManagerBasedRlEnv
from mjlab.terrains import TerrainEntityCfg,TerrainGeneratorCfg
from mjlab.terrains.config import flat

def wc(env,n):
 f=env.scene[n].data.found;return torch.any(f.reshape(f.shape[0],-1)>0,-1)
def one(repeat,card):
 cfg=rb.make_roll_boundary_env_cfg((0.0,),16);cfg.scene.terrain=TerrainEntityCfg(terrain_type='generator',terrain_generator=TerrainGeneratorCfg(seed=1,curriculum=True,size=rb.TERRAIN_SIZE_M,num_rows=1,num_cols=1,difficulty_range=(0,0),sub_terrains={rb.terrain_key(0):flat(proportion=1.0)}),max_init_terrain_level=0,num_envs=16)
 env=ManagerBasedRlEnv(cfg=cfg,device='cpu')
 try:
  types,face,cross,reset=rb._reset_to_approach(env,root_height=float(card['height_m']),card_name=str(card['name']),repeat=repeat,height_count=1);robot=env.scene['robot'];a=torch.zeros((16,6),device=env.device);air=[];progress=[];success=torch.zeros(16,dtype=torch.bool);stable=torch.zeros(16,dtype=torch.long)
  for s in range(600):
   active=torch.ones(16,dtype=torch.bool,device=env.device);vx=0 if s<100 else rb.COMMAND_VX_MPS;rb._force_commands(env,active=active,vx=vx,height=float(card['height_m']),pitch=float(card['pitch_rad']));env.step(a);lf,rf=wc(env,rb.LEFT_SENSOR),wc(env,rb.RIGHT_SENSOR);aa=~lf&~rf
   if s>=100:
    air.append(aa.cpu());p=robot.data.root_link_pos_w[:,0]-face;progress.append(p.cpu());pitch,roll=rb._pitch_roll(robot);ok=(~aa)&(p>=cross-face)&(pitch.abs()<=rb.PITCH_LIMIT_RAD)&(roll.abs()<=rb.ROLL_LIMIT_RAD)&(robot.data.root_link_ang_vel_b[:,1].abs()<=rb.PITCH_RATE_LIMIT_RADPS);stable=torch.where(ok,stable+1,torch.zeros_like(stable));success|=stable>=rb.OFFICIAL_STABLE_STEPS
  air=torch.stack(air);progress=torch.stack(progress)
  return {'card':card['name'],'repeat':repeat,'air_trials':int(air.any(0).sum()),'air_steps':int(air.sum()),'successes_ignoring_air_safety':int(success.sum()),'max_progress_min':float(progress.max(0).values.min()),'max_progress_max':float(progress.max())}
 finally:env.close()
r=[]
for c in rb.POSTURE_CARDS:
 for repeat in (1,2,3):r.append(one(repeat,c))
print(json.dumps(r,indent=2));Path(r'D:\mjlab_workspace\plane_repeat_compare.json').write_text(json.dumps(r,indent=2)+'\n')
