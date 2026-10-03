import json
import pathlib
from mjlab.envs import ManagerBasedRlEnv
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb
out = pathlib.Path(r"D:\mjlab_workspace\roll_boundary_repeat4_d99c491.json")
heights=(0.0,0.0025,0.005)
cfg=rb.make_roll_boundary_env_cfg(heights,4)
env=ManagerBasedRlEnv(cfg=cfg,device="cpu")
rows=[]
try:
  for card in rb.POSTURE_CARDS:
    print("card",card["name"])
    rows += rb.run_card_repeat(env,heights=heights,card=card,repeat=1,
      settle_steps=rb.OFFICIAL_SETTLE_STEPS,drive_steps=rb.OFFICIAL_DRIVE_STEPS,
      stable_steps=rb.OFFICIAL_STABLE_STEPS)
finally:
  env.close()
payload={"kind":"cpu_repeat4_diagnostic","evidence_eligible":False,
 "git_sha":rb._git_sha(rb.REPOSITORY_PATH),"heights_m":heights,"trials":rows}
out.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
for card in rb.POSTURE_CARDS:
 for h in heights:
  r=[t for t in rows if t["posture_card"]==card["name"] and t["stair_height_m"]==h]
  print(card["name"],h,"n",len(r),"success",sum(t["success"] for t in r),"air",sum(t["bilateral_airborne_ever"] for t in r),"substeps",sum(t["bilateral_unsupported_physics_substeps"] for t in r),"progress",[round(t["max_progress_past_face_m"],4) for t in r])
print("output",out)
