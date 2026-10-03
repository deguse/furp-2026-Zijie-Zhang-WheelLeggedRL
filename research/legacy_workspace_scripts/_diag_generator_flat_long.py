from __future__ import annotations
import sys, json
from pathlib import Path
import torch
REPO=Path(r'D:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL\.worktrees\p2-classical-upper-bound')
for x in (REPO/'src',REPO/'src'/'hoppertrex_mjlab'):
 if str(x) not in sys.path:sys.path.insert(0,str(x))
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb
from mjlab.envs import ManagerBasedRlEnv
from mjlab.terrains import TerrainEntityCfg

def sv(env,n):
 d=env.scene[n].data; f=d.found; F=d.force
 return torch.any(f.reshape(f.shape[0],-1)>0,-1),torch.linalg.vector_norm(F.reshape(F.shape[0],-1,3),dim=-1).sum(-1)
def run(kind):
 cfg=rb.make_roll_boundary_env_cfg((0.0,),16)
 if kind=='plane': cfg.scene.terrain=TerrainEntityCfg(terrain_type='plane',env_spacing=2.5,num_envs=16)
 env=ManagerBasedRlEnv(cfg=cfg,device='cpu')
 try:
  if kind=='generator':
   rb._reset_to_approach(env,root_height=float(rb.POSTURE_CARDS[0]['height_m']),card_name='envelope_center',repeat=1,height_count=1)
  else:
   env.reset(); robot=env.scene['robot']; roots=robot.data.default_root_state.clone(); roots[:,:3]+=env.scene.env_origins; roots[:,2]=float(rb.POSTURE_CARDS[0]['height_m']); roots[:,7:13]=0; robot.write_root_state_to_sim(roots);env.sim.forward();env.sim.sense()
  robot=env.scene['robot']; actions=torch.zeros((16,6),device=env.device); air=[]; force=[]; z=[]; vz=[]; pr=[]
  for step in range(2100):
   active=torch.ones(16,dtype=torch.bool,device=env.device)
   vx=0 if step<100 else rb.COMMAND_VX_MPS
   rb._force_commands(env,active=active,vx=vx,height=float(rb.POSTURE_CARDS[0]['height_m']),pitch=float(rb.POSTURE_CARDS[0]['pitch_rad']))
   env.step(actions); lf,lF=sv(env,rb.LEFT_SENSOR);rf,rF=sv(env,rb.RIGHT_SENSOR)
   air.append((~lf&~rf).cpu());force.append((lF+rF).cpu());z.append(robot.data.root_link_pos_w[:,2].cpu());vz.append(robot.data.root_link_lin_vel_w[:,2].cpu());pr.append(robot.data.root_link_ang_vel_b[:,1].cpu())
  air=torch.stack(air);force=torch.stack(force);z=torch.stack(z);vz=torch.stack(vz);pr=torch.stack(pr)
  def maxrun(a):
   out=[]
   for e in range(16):
    cur=mx=0
    for v in a[:,e]:cur=cur+1 if v else 0;mx=max(mx,cur)
    out.append(mx)
   return out
  return {'kind':kind,'settle_air_steps':int(air[:100].sum()),'drive_air_steps':int(air[100:].sum()),'settle_max_run':max(maxrun(air[:100])),'drive_max_run':max(maxrun(air[100:])),'air_trials':int(air[100:].any(0).sum()),'root_z_drive_min':float(z[100:].min()),'root_z_drive_max':float(z[100:].max()),'root_vz_drive_min':float(vz[100:].min()),'root_vz_drive_max':float(vz[100:].max()),'pitch_rate_abs_drive_max':float(pr[100:].abs().max()),'contact_force_median':float(force[100:][~air[100:]].median()),'root_x_delta_mean':float((robot.data.root_link_pos_w[:,0]-robot.data.root_link_pos_w[:,0]*0).mean())}
 finally:env.close()
res=[run('generator')]
print(json.dumps(res,indent=2))
Path(r'D:\mjlab_workspace\generator_flat_long.json').write_text(json.dumps(res,indent=2)+'\n')
