import json
import pathlib
import mujoco
from mjlab.envs import ManagerBasedRlEnv
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb

out=pathlib.Path(r"D:\mjlab_workspace\roll_boundary_spherewheel_d99c491.json")
heights=(0.0,0.0025,0.005)
results=[]
for label, use_sphere in (("cylinder",False),("sagittal_sphere_proxy",True)):
  cfg=rb.make_roll_boundary_env_cfg(heights,4)
  if use_sphere:
    orig=cfg.scene.entities["robot"].spec_fn
    def sphere_spec(orig=orig):
      spec=orig()
      for name in ("wheel_left_collision","wheel_right_collision"):
        geom=spec.geom(name)
        geom.type=mujoco.mjtGeom.mjGEOM_SPHERE
        geom.size[:]=(.1,0.0,0.0)
      return spec
    cfg.scene.entities["robot"].spec_fn=sphere_spec
  env=ManagerBasedRlEnv(cfg=cfg,device="cpu")
  rows=[]
  try:
    model=env.sim.mj_model
    geom_types={name:int(model.geom_type[model.geom("robot/"+name).id]) for name in ("wheel_left_collision","wheel_right_collision")}
    for card in rb.POSTURE_CARDS:
      rows+=rb.run_card_repeat(env,heights=heights,card=card,repeat=1,
        settle_steps=rb.OFFICIAL_SETTLE_STEPS,drive_steps=rb.OFFICIAL_DRIVE_STEPS,stable_steps=rb.OFFICIAL_STABLE_STEPS)
  finally: env.close()
  results.append({"label":label,"evidence_eligible":False,"geom_types":geom_types,"trials":rows})
payload={"kind":"wheel_collision_shape_ablation","evidence_eligible":False,"git_sha":rb._git_sha(rb.REPOSITORY_PATH),"variants":results}
out.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
for result in results:
 print(result["label"],result["geom_types"])
 for card in rb.POSTURE_CARDS:
  for h in heights:
   rs=[t for t in result["trials"] if t["posture_card"]==card["name"] and t["stair_height_m"]==h]
   print(card["name"],h,"success",sum(t["success"] for t in rs),"air",sum(t["bilateral_airborne_ever"] for t in rs),"substeps",sum(t["bilateral_unsupported_physics_substeps"] for t in rs),"progress",[round(t["max_progress_past_face_m"],4) for t in rs])
print("output",out)
