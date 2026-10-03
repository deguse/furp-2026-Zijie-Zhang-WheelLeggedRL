import json,math,numpy as np,mujoco,pathlib
ROOT=pathlib.Path(r'D:\mjlab_workspace'); ART=pathlib.Path(r'D:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL\.worktrees\p2-classical-upper-bound\docs\experiments\artifacts')
schedule=json.loads((ART/'c1_schedule_candidate24_1f54968_seed1/c1_schedule.json').read_text());post=json.loads((ART/'c1_posture_requalification_seed1/posture_map_seed1_registered_p032.json').read_text());station=json.loads((ART/'c1_posture_requalification_seed1/station_calibration_seed1.json').read_text());cal=json.loads((ART/'hybrid_runtime_seed1/velocity_calibration_seed1.json').read_text())
H=np.array(schedule['height_nodes']);P=np.array(schedule['pitch_nodes']);nodes=schedule['nodes'];G=np.array([[n['gain'] for n in r] for r in nodes]);E=np.array([[n['equilibrium_state'] for n in r] for r in nodes]);U=np.array([[n['equilibrium_input'] for n in r] for r in nodes]);C=np.array(post['coefficients'])
def bil(x,y,a):
 i=np.clip(np.searchsorted(H,x,side='right'),1,len(H)-1);j=np.clip(np.searchsorted(P,y,side='right'),1,len(P)-1);i0=i-1;j0=j-1;wx=(np.clip(x,H[0],H[-1])-H[i0])/(H[i]-H[i0]);wy=(np.clip(y,P[0],P[-1])-P[j0])/(P[j]-P[j0]);return (1-wx)*((1-wy)*a[i0,j0]+wy*a[i0,j]) + wx*((1-wy)*a[i,j0]+wy*a[i,j])
def run(kind='box',h=.3092089487,p=.016,vx=.07,steps=2400):
 m=mujoco.MjModel.from_xml_path(str(ROOT/f'native_{kind}.xml'));m.opt.timestep=.005;m.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST;m.opt.cone=mujoco.mjtCone.mjCONE_ELLIPTIC;m.opt.iterations=50;m.opt.ls_iterations=20;m.opt.impratio=10;d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0)
 qleg=np.array([1,h,p])@C;d.qpos[0:3]=[0,0,h];d.qpos[3:7]=[math.cos(p/2),0,math.sin(p/2),0];d.qpos[[7,10,8,11]]=qleg;d.qvel[:]=0;mujoco.mj_forward(m,d)
 names=('robot/thigh_left_01','robot/thigh_right_01','robot/knee_left','robot/knee_right','robot/wheel_left','robot/wheel_right');qids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names];dids=[m.jnt_dofadr[i] for i in qids];aids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in names];wg=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ('robot/wheel_left_collision','robot/wheel_right_collision')]
 events=[];air=0;prev=np.zeros(2);bp=np.array(station['breakpoints']);station_v=np.interp(p,bp[:,0],bp[:,1]);scale=cal['velocity_command_scale'];bias=cal['velocity_command_bias']
 for s in range(steps):
  if s%4==0:
   cmd=0 if s<400 else vx;calvx=scale*cmd+bias-station_v;mat=np.zeros(9);mujoco.mju_quat2Mat(mat,d.qpos[3:7]);mat=mat.reshape(3,3);grav=mat.T@np.array([0,0,-1]);pitch=math.atan2(grav[0],max(-grav[2],1e-6));pr=d.qvel[4];vbody=mat.T@d.qvel[:3];wspd=d.qvel[dids[4:]];state=np.array([pitch,pr,vbody[0]-calvx,.5*(wspd[1]-wspd[0])-calvx/.1]);control=float(bil(h,p,U))-np.sum((state-bil(h,p,E))*bil(h,p,G));desired=np.array([-control,control]);prev=np.clip(prev+np.clip(desired-prev,-6,6),-12,12);d.ctrl[aids[:4]]=qleg;d.ctrl[aids[4:]]=prev
  mujoco.mj_step(m,d);hit=[False,False];forces=[0.,0.]
  for ci in range(d.ncon):
   c=d.contact[ci]
   if c.efc_address<0:continue
   for k,g in enumerate(wg):
    if g==c.geom[0] or g==c.geom[1]:f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f);hit[k]=True;forces[k]+=abs(f[0])
  if s>=400 and not any(hit):
   air+=1
   if len(events)<20:events.append({'substep':s-400,'z':float(d.qpos[2]),'vz':float(d.qvel[2]),'ctrl':d.ctrl.tolist(),'actuator_force':d.actuator_force.tolist(),'forces':forces})
 return {'kind':kind,'air_substeps':air,'events':events}
for k in ('plane','box'):
 x=run(k);(ROOT/f'native_{k}_result.json').write_text(json.dumps(x,indent=2));print(k,x['air_substeps'])
