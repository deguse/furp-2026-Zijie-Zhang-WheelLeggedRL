from __future__ import annotations
import sys,json
from pathlib import Path
REPO=Path(r'D:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL\.worktrees\p2-classical-upper-bound')
for x in (REPO/'src',REPO/'src'/'hoppertrex_mjlab'):
 if str(x) not in sys.path:sys.path.insert(0,str(x))
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb
from mjlab.envs import ManagerBasedRlEnv
from mjlab.terrains import TerrainEntityCfg

def run(kind):
 cfg=rb.make_roll_boundary_env_cfg((0.0,),1)
 if kind=='plane':cfg.scene.terrain=TerrainEntityCfg(terrain_type='plane',env_spacing=2.5,num_envs=1)
 env=ManagerBasedRlEnv(cfg=cfg,device='cpu')
 try:
  m=env.sim.mj_model
  names=[]
  for i in range(m.ngeom):
   g=m.geom(i); n=g.name
   if 'terrain' in n or 'wheel' in n:
    names.append({'id':i,'name':n,'type':int(m.geom_type[i]),'friction':[float(x) for x in m.geom_friction[i]],'solref':[float(x) for x in m.geom_solref[i]],'solimp':[float(x) for x in m.geom_solimp[i]],'condim':int(m.geom_condim[i])})
  return {'kind':kind,'geoms':names}
 finally:env.close()
r=[run('plane'),run('generator')]
print(json.dumps(r,indent=2))
