from __future__ import annotations
import json, math, sys
from pathlib import Path
import torch

REPO=Path(r"D:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL\.worktrees\p2-classical-upper-bound")
for x in (REPO/'src', REPO/'src'/'hoppertrex_mjlab'):
    if str(x) not in sys.path: sys.path.insert(0,str(x))
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb
from mjlab.envs import ManagerBasedRlEnv

OUT=Path(r"D:\mjlab_workspace\roll_boundary_flat_drive_diagnostic.json")
HEIGHTS=(0.0,)
ENVS=16
SETTLE=100
DRIVE=500

def sensor_values(env,name):
    d=env.scene[name].data
    f=d.found
    force=d.force
    normal=d.normal
    found=torch.any(f.reshape(f.shape[0],-1)>0,dim=-1)
    force_slots=force.reshape(force.shape[0],-1,3)
    # contact frame component 0 is normal force for this unreduced sensor
    normal_force=force_slots[...,0].abs().sum(dim=-1)
    force_norm=torch.linalg.vector_norm(force_slots,dim=-1).sum(dim=-1)
    return found,normal_force,force_norm

def quant(v):
    vals=torch.as_tensor(v,dtype=torch.float64)
    return {'min':float(vals.min()),'mean':float(vals.mean()),'max':float(vals.max())}

def run_card(card):
    cfg=rb.make_roll_boundary_env_cfg(HEIGHTS,ENVS)
    env=ManagerBasedRlEnv(cfg=cfg,device='cpu')
    try:
        types,face,cross,reset=rb._reset_to_approach(env,root_height=float(card['height_m']),card_name=str(card['name']),repeat=1,height_count=1)
        robot=env.scene['robot']
        term=env.action_manager.get_term('hybrid_wheel_leg')
        wheel_ids=term._wheel_ids
        actions=torch.zeros((env.num_envs,env.action_space.shape[-1]),device=env.device)
        geom_ids=[]
        for name in ('wheel_left_collision','wheel_right_collision'):
            ids,names=robot.find_geoms(name,preserve_order=True)
            assert names==[name] and len(ids)==1,(name,ids,names)
            geom_ids.append(ids[0])
        rows=[]
        def sample(phase,idx,vx):
            active=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
            rb._force_commands(env,active=active,vx=vx,height=float(card['height_m']),pitch=float(card['pitch_rad']))
            obs,reward,terminated,timeouts,extras=env.step(actions)
            lf,ln,lmag=sensor_values(env,rb.LEFT_SENSOR)
            rf,rn,rmag=sensor_values(env,rb.RIGHT_SENSOR)
            gp=robot.data.geom_pose_w[:,geom_ids,:3]
            gv=robot.data.geom_vel_w[:,geom_ids,:3]
            pitch,roll=rb._pitch_roll(robot)
            speed=robot.data.joint_vel[:,wheel_ids].detach()
            target=term.wheel_targets.detach()
            torque,sat=rb.model_wheel_torque(target,speed)
            for e in range(env.num_envs):
                rows.append({
                  'phase':phase,'step':idx,'env_id':e,
                  'left_found':bool(lf[e]),'right_found':bool(rf[e]),'airborne':bool((~lf&~rf)[e]),
                  'left_normal_force_sum_n':float(ln[e]),'right_normal_force_sum_n':float(rn[e]),
                  'left_force_norm_sum_n':float(lmag[e]),'right_force_norm_sum_n':float(rmag[e]),
                  'left_wheel_center_z_m':float(gp[e,0,2]),'right_wheel_center_z_m':float(gp[e,1,2]),
                  'left_wheel_center_vz_mps':float(gv[e,0,2]),'right_wheel_center_vz_mps':float(gv[e,1,2]),
                  'left_wheel_bottom_clearance_m':float(gp[e,0,2]-rb.NOMINAL_WHEEL_RADIUS_M),
                  'right_wheel_bottom_clearance_m':float(gp[e,1,2]-rb.NOMINAL_WHEEL_RADIUS_M),
                  'root_x_relative_face_m':float(robot.data.root_link_pos_w[e,0]-face[e]),
                  'root_z_m':float(robot.data.root_link_pos_w[e,2]),
                  'root_vx_mps':float(robot.data.root_link_lin_vel_w[e,0]),
                  'root_vz_mps':float(robot.data.root_link_lin_vel_w[e,2]),
                  'pitch_rad':float(pitch[e]),'roll_rad':float(roll[e]),
                  'pitch_rate_radps':float(robot.data.root_link_ang_vel_b[e,1]),
                  'left_wheel_target_radps':float(target[e,0]),'right_wheel_target_radps':float(target[e,1]),
                  'left_wheel_speed_radps':float(speed[e,0]),'right_wheel_speed_radps':float(speed[e,1]),
                  'left_model_torque_nm':float(torque[e,0]),'right_model_torque_nm':float(torque[e,1]),
                  'left_torque_saturated':bool(sat[e,0]),'right_torque_saturated':bool(sat[e,1]),
                  'terminated':bool(terminated[e]),'timeout':bool(timeouts[e]),
                })
        for i in range(SETTLE): sample('settle',i,0.0)
        for i in range(DRIVE): sample('drive',i,rb.COMMAND_VX_MPS)
        # summarize exact drive event runs per env
        per=[]
        for e in range(env.num_envs):
            rr=[r for r in rows if r['phase']=='drive' and r['env_id']==e]
            runs=[]; start=None
            for i,r in enumerate(rr+[{'airborne':False}]):
                if r['airborne'] and start is None:start=i
                elif not r['airborne'] and start is not None:runs.append((start,i-start));start=None
            air=[r for r in rr if r['airborne']]
            per.append({
              'env_id':e,'first_drive_airborne_step':None if not air else air[0]['step'],
              'airborne_steps':len(air),'airborne_runs':runs,'max_airborne_run_steps':max([x[1] for x in runs],default=0),
              'max_root_vz_mps':max(r['root_vz_mps'] for r in rr),'min_root_vz_mps':min(r['root_vz_mps'] for r in rr),
              'max_left_bottom_clearance_m':max(r['left_wheel_bottom_clearance_m'] for r in rr),
              'max_right_bottom_clearance_m':max(r['right_wheel_bottom_clearance_m'] for r in rr),
              'max_pitch_rate_abs_radps':max(abs(r['pitch_rate_radps']) for r in rr),
              'max_progress_m':max(r['root_x_relative_face_m'] for r in rr),
              'torque_saturation_fraction':sum((r['left_torque_saturated']+r['right_torque_saturated']) for r in rr)/(2*len(rr)),
            })
        return {'card':card,'reset':{k:v.cpu().tolist() for k,v in reset.items()},'per_env':per,'rows':rows}
    finally: env.close()

runs=[run_card(c) for c in rb.POSTURE_CARDS]
out={'schema_version':1,'purpose':'non-evidence local diagnosis; safety disabled; flat 0 mm; no early reset','device':'cpu','settle_steps':SETTLE,'drive_steps':DRIVE,'command_vx_mps':rb.COMMAND_VX_MPS,'runs':runs}
OUT.write_text(json.dumps(out,indent=2)+'\n')
print(OUT)
for run in runs:
    print('\nCARD',run['card']['name'])
    for e in run['per_env']:
        print(e)
