from __future__ import annotations
import sys,json
from pathlib import Path
from collections import Counter
import torch
REPO=Path(r'D:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL\.worktrees\p2-classical-upper-bound')
for x in (REPO/'src',REPO/'src'/'hoppertrex_mjlab'):
 if str(x) not in sys.path: sys.path.insert(0,str(x))
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb
from mjlab.envs import ManagerBasedRlEnv
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.config import flat
CARDS=rb.POSTURE_CARDS
N=4; SETTLE=100; DRIVE=200
WHEEL_GEOMS=('wheel_left_collision','wheel_right_collision')
LEG_JOINTS=('thigh_left_01','thigh_right_01','knee_left','knee_right')
WHEEL_JOINTS=('wheel_left','wheel_right')

def make_cfg(terrain='box'):
 cfg=rb.make_roll_boundary_env_cfg((0.0,),N)
 if terrain=='plane': cfg.scene.terrain=TerrainEntityCfg(terrain_type='plane',env_spacing=2.5,num_envs=N)
 elif terrain=='box':
  cfg.scene.terrain=TerrainEntityCfg(terrain_type='generator',terrain_generator=TerrainGeneratorCfg(seed=rb.SEED,curriculum=False,size=rb.TERRAIN_SIZE_M,num_rows=1,num_cols=1,difficulty_range=(0,0),sub_terrains={'flat':flat(proportion=1.0,size=rb.TERRAIN_SIZE_M)}),max_init_terrain_level=0,num_envs=N)
 elif terrain!='pyramid': raise ValueError(terrain)
 return cfg

def qtarget(env,card):
 term=env.action_manager.get_term('hybrid_wheel_leg')
 feat=torch.tensor([1.,float(card['height_m']),float(card['pitch_rad'])],device=env.device)
 return feat@term._posture_coefficients

def reset(env,card,mode='root_only',repeat=1):
 env.reset(); robot=env.scene['robot']; origins=env.scene.env_origins
 pert=rb.reset_perturbations(slots=N,card_name=str(card['name']),repeat=repeat).to(env.device)
 roots=robot.data.default_root_state.clone(); geom=rb.approach_geometry(0.0)
 generated=getattr(env.scene.terrain,'terrain_types',None) is not None
 roots[:,0]=origins[:,0]+(geom['start_x'] if generated else 0.0)+pert[:,0]
 roots[:,1]=origins[:,1]+pert[:,1]; roots[:,2]=float(card['height_m']); roots[:,7:13]=0
 roots[:,7]=pert[:,2]; roots[:,11]=pert[:,3]
 if mode in ('joint_root','joint_root_nojitter','posture_reset'):
  qt=qtarget(env,card); pos=robot.data.default_joint_pos.clone(); vel=torch.zeros_like(pos)
  legids,_=robot.find_joints(LEG_JOINTS,preserve_order=True); pos[:,legids]=qt
  robot.write_joint_state_to_sim(pos,vel)
 if mode=='posture_reset':
  half=.5*float(card['pitch_rad']); roots[:,3]=torch.cos(torch.tensor(half)); roots[:,4]=0; roots[:,5]=torch.sin(torch.tensor(half)); roots[:,6]=0
 if mode in ('joint_root_nojitter','posture_reset'):
  roots[:,7:13]=0; roots[:,0]=origins[:,0]+(geom['start_x'] if generated else 0.0); roots[:,1]=origins[:,1]
 robot.write_root_state_to_sim(roots); env.sim.forward(); env.sim.sense()

def support(env,name):
 d=env.scene[name].data
 found=torch.any(d.found.reshape(N,-1)>0,dim=-1)
 force=torch.linalg.vector_norm(d.force,dim=-1)
 return found,torch.max(force,dim=1).values,torch.sum(force,dim=1)

def clearance(env):
 robot=env.scene['robot']; gids,_=robot.find_geoms(WHEEL_GEOMS,preserve_order=True)
 pos=robot.data.geom_pos_w[:,gids,:]; mat=robot.data.geom_pose_w[:,gids,7:].reshape(N,2,3,3) if False else None
 quat=robot.data.geom_quat_w[:,gids,:]
 # Local cylinder z-axis world-z component from quaternion rotation matrix.
 w,x,y,z=quat.unbind(-1); axis_z=1.0-2.0*(x.square()+y.square())
 extent=.1*torch.sqrt(torch.clamp(1-axis_z.square(),min=0))+.018*axis_z.abs()
 return pos[:,:,2]-extent

def run_variant(terrain,card,reset_mode,command_mode='step',repeat=1):
 cfg=make_cfg(terrain); env=ManagerBasedRlEnv(cfg=cfg,device='cpu')
 try:
  reset(env,card,reset_mode,repeat); robot=env.scene['robot']; term=env.action_manager.get_term('hybrid_wheel_leg')
  legids,_=robot.find_joints(LEG_JOINTS,preserve_order=True); wheelids,_=robot.find_joints(WHEEL_JOINTS,preserve_order=True)
  actids,_=robot.find_actuators((*LEG_JOINTS,*WHEEL_JOINTS),preserve_order=True)
  actions=torch.zeros((N,6),device=env.device); events=[]; totals=Counter(); affected=torch.zeros(N,dtype=torch.bool)
  max_gap=torch.full((N,),-1e9); min_gap=torch.full((N,),1e9); max_abs_wheel_force=torch.zeros(N); max_abs_leg_force=torch.zeros(N); max_pitchrate=torch.zeros(N); max_vz=torch.zeros(N)
  original_update=env.scene.update; subidx={'v':0}; phase={'drive':False,'cycle':-1}
  def hooked(dt):
   original_update(dt); i=subidx['v']%cfg.decimation; subidx['v']+=1
   if not phase['drive']: return
   lf,lmax,lsum=support(env,rb.LEFT_SENSOR); rf,rmax,rsum=support(env,rb.RIGHT_SENSOR); air=~lf&~rf
   gap=clearance(env).detach().cpu(); af=robot.data.actuator_force[:,actids].detach().cpu(); p,_=rb._pitch_roll(robot); pr=robot.data.root_link_ang_vel_b[:,1].detach().cpu(); vz=robot.data.root_link_lin_vel_w[:,2].detach().cpu()
   max_gap.copy_(torch.maximum(max_gap,gap.max(1).values)); min_gap.copy_(torch.minimum(min_gap,gap.min(1).values)); max_abs_leg_force.copy_(torch.maximum(max_abs_leg_force,af[:,:4].abs().max(1).values)); max_abs_wheel_force.copy_(torch.maximum(max_abs_wheel_force,af[:,4:].abs().max(1).values)); max_pitchrate.copy_(torch.maximum(max_pitchrate,pr.abs())); max_vz.copy_(torch.maximum(max_vz,vz.abs()))
   totals['physics_substeps']+=N; totals['air_substeps']+=int(air.sum()); totals['positive_gap_air_substeps']+=int((air.cpu()&(gap.min(1).values>0)).sum()); affected.logical_or_(air.cpu())
   for e in torch.nonzero(air,as_tuple=False).squeeze(-1).tolist():
    if len(events)<80: events.append({'cycle':phase['cycle'],'substep':i,'env':e,'clearance_m':gap[e].tolist(),'found':[bool(lf[e]),bool(rf[e])],'force_max_n':[float(lmax[e]),float(rmax[e])],'force_sum_n':[float(lsum[e]),float(rsum[e])],'root_z_m':float(robot.data.root_link_pos_w[e,2]),'root_vz_mps':float(vz[e]),'pitch_rad':float(p[e]),'pitch_rate_radps':float(pr[e]),'leg_q':robot.data.joint_pos[e,legids].detach().cpu().tolist(),'leg_qdot':robot.data.joint_vel[e,legids].detach().cpu().tolist(),'leg_target':term.leg_targets[e].detach().cpu().tolist(),'wheel_target':term.wheel_targets[e].detach().cpu().tolist(),'wheel_speed':robot.data.joint_vel[e,wheelids].detach().cpu().tolist(),'actuator_force_order':list((*LEG_JOINTS,*WHEEL_JOINTS)),'actuator_force_nm':af[e].tolist()})
  env.scene.update=hooked
  for step in range(SETTLE+DRIVE):
   active=torch.ones(N,dtype=torch.bool); vx=0 if (command_mode=='zero' or step<SETTLE) else rb.COMMAND_VX_MPS
   rb._force_commands(env,active=active,vx=vx,height=float(card['height_m']),pitch=float(card['pitch_rad'])); phase['drive']=step>=SETTLE; phase['cycle']=step-SETTLE; env.step(actions)
  return {'terrain':terrain,'card':card['name'],'reset_mode':reset_mode,'command_mode':command_mode,'control_hz':50,'physics_dt_s':.005,'drive_control_cycles':DRIVE*N,'physics_substeps':totals['physics_substeps'],'air_substeps':totals['air_substeps'],'positive_gap_air_substeps':totals['positive_gap_air_substeps'],'affected_trials':int(affected.sum()),'max_clearance_m':float(max_gap.max()),'min_clearance_m':float(min_gap.min()),'max_actual_wheel_actuator_force_nm':float(max_abs_wheel_force.max()),'max_actual_leg_actuator_force_nm':float(max_abs_leg_force.max()),'max_abs_pitch_rate_radps':float(max_pitchrate.max()),'max_abs_root_vz_mps':float(max_vz.max()),'events':events}
 finally: env.close()

def main():
 x=run_variant('plane',CARDS[0],'posture_reset','step')
 Path(r'D:\mjlab_workspace\strict_flat_plane.json').write_text(json.dumps(x,indent=2)+'\n')
 print(json.dumps({k:v for k,v in x.items() if k!='events'},indent=2),flush=True)
if __name__=='__main__': main()





