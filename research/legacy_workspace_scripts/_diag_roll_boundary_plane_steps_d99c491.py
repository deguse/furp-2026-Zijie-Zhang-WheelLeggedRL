import json
import pathlib
from mjlab.envs import ManagerBasedRlEnv
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.config import flat, pyramid_stairs
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb

out=pathlib.Path(r"D:\mjlab_workspace\roll_boundary_plane_steps_d99c491.json")
heights=(0.0,0.0025,0.005)
results=[]
for label, zero_kind in (("finite_box_zero","box"),("infinite_plane_zero","plane")):
  cfg=rb.make_roll_boundary_env_cfg(heights,4)
  if zero_kind=="plane":
    sub=rb.roll_boundary_sub_terrains(heights)
    sub[rb.terrain_key(0.0)]=flat(proportion=1.0,size=rb.TERRAIN_SIZE_M)
    # BoxFlatTerrainCfg always emits finite box. Replace its function at instance level is impossible;
    # use a diagnostic subclass that emits one MuJoCo plane at z=0.
    from dataclasses import dataclass
    import mujoco, numpy as np
    from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
    @dataclass(kw_only=True)
    class PlaneCfg(SubTerrainCfg):
      def function(self,difficulty,spec,rng):
        del difficulty,rng
        geom=spec.body("terrain").add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE,size=(0,0,1),pos=(self.size[0]/2,self.size[1]/2,0))
        return TerrainOutput(origin=np.array((self.size[0]/2,self.size[1]/2,0.0)),geometries=[TerrainGeometry(geom=geom,color=(.5,.5,.5,1))])
    sub[rb.terrain_key(0.0)]=PlaneCfg(proportion=1.0,size=rb.TERRAIN_SIZE_M)
    cfg.scene.terrain=TerrainEntityCfg(terrain_type="generator",terrain_generator=TerrainGeneratorCfg(
      seed=rb.SEED,curriculum=True,size=rb.TERRAIN_SIZE_M,num_rows=1,num_cols=len(heights),difficulty_range=(0,0),sub_terrains=sub),
      max_init_terrain_level=0,num_envs=len(heights)*4)
  env=ManagerBasedRlEnv(cfg=cfg,device="cpu")
  rows=[]
  try:
    for card in rb.POSTURE_CARDS:
      rows+=rb.run_card_repeat(env,heights=heights,card=card,repeat=1,
        settle_steps=rb.OFFICIAL_SETTLE_STEPS,drive_steps=rb.OFFICIAL_DRIVE_STEPS,stable_steps=rb.OFFICIAL_STABLE_STEPS)
  finally: env.close()
  results.append({"label":label,"evidence_eligible":False,"trials":rows})
payload={"kind":"zero-cell-plane-ablation","evidence_eligible":False,"git_sha":rb._git_sha(rb.REPOSITORY_PATH),"variants":results}
out.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
for result in results:
 print(result["label"])
 for card in rb.POSTURE_CARDS:
  for h in heights:
   rs=[t for t in result["trials"] if t["posture_card"]==card["name"] and t["stair_height_m"]==h]
   print(card["name"],h,"success",sum(t["success"] for t in rs),"air",sum(t["bilateral_airborne_ever"] for t in rs),"progress",[round(t["max_progress_past_face_m"],4) for t in rs])
print("output",out)
