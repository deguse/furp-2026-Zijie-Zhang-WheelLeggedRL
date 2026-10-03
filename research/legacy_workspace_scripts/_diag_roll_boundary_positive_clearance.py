from __future__ import annotations
import sys,json
from pathlib import Path
import torch
REPO=Path(r'D:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL\.worktrees\p2-classical-upper-bound')
for x in (REPO/'src',REPO/'src'/'hoppertrex_mjlab'):
 if str(x) not in sys.path:sys.path.insert(0,str(x))
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb
from mjlab.envs import ManagerBasedRlEnv

def sv(env,n):
 d=env.scene[n].data
 return torch.any(d.found.reshape(d.found.shape[0],-1)>0,-1)
def run(h):
 cfg=rb.make_roll_boundary_env_cfg((h,),16);env=ManagerBasedRlEnv(cfg=cfg,device='cpu')
 try:
  card=rb.POSTURE_CARDS[0];types,face,cross,reset=rb._reset_to_approach(env,root_height=float(card['height_m']),card_name=str(card['name']),repeat=1,height_count=1)
  robot=env.scene['robot'];term=env.action_manager.get_term('hybrid_wheel_leg');actions=torch.zeros((16,6),device=env.device)
  ids=[robot.find_geoms(n,preserve_order=True)[0][0] for n in ('wheel_left_collision','wheel_right_collision')]
  air=[];clr=[];progress=[];x=[];vz=[];contacts=[]
  for s in range(600):
   active=torch.ones(16,dtype=torch.bool,device=env.device);vx=0 if s<100 else rb.COMMAND_VX_MPS
   rb._force_commands(env,active=active,vx=vx,height=float(card['height_m']),pitch=float(card['pitch_rad']));env.step(actions)
   lf,rf=sv(env,rb.LEFT_SENSOR),sv(env,rb.RIGHT_SENSOR);a=~lf&~rf;gp=robot.data.geom_pose_w[:,ids,:3]
   air.append(a.cpu());clr.append((gp[:,:,2]-rb.NOMINAL_WHEEL_RADIUS_M).cpu());progress.append((robot.data.root_link_pos_w[:,0]-face).cpu());vz.append(robot.data.root_link_lin_vel_w[:,2].cpu())
  air=torch.stack(air)[100:];clr=torch.stack(clr)[100:];progress=torch.stack(progress)[100:];vz=torch.stack(vz)[100:]
  def maxrun(a):
   z=[]
   for e in range(16):
    c=m=0
    for v in a[:,e]:c=c+1 if v else 0;m=max(m,c)
    z.append(m)
   return z
  # event relative to whether the robot is before/at/after first face
  pre=progress < 0; near=(progress>=0)&(progress<rb.CROSS_DEPTH_M);post=progress>=rb.CROSS_DEPTH_M
  return {'height_m':h,'air_steps':int(air.sum()),'air_trials':int(air.any(0).sum()),'max_air_run':max(maxrun(air)),'air_pre_face':int((air&pre).sum()),'air_between_face_and_cross':int((air&near).sum()),'air_post_cross':int((air&post).sum()),'max_progress_m':float(progress.max()),'min_progress_m':float(progress.min()),'max_clearance_mm':float(clr.max()*1000),'root_vz_abs_max':float(vz.abs().max())}
 finally:env.close()
r=[run(0.0025),run(0.005)]
print(json.dumps(r,indent=2))
Path(r'D:\mjlab_workspace\roll_boundary_positive_clearance.json').write_text(json.dumps(r,indent=2)+'\n')
