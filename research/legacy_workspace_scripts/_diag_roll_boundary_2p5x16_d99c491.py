import json,pathlib
from mjlab.envs import ManagerBasedRlEnv
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb
out=pathlib.Path(r"D:\mjlab_workspace\roll_boundary_2p5x16_d99c491.json")
heights=(0.0,0.0025)
cfg=rb.make_roll_boundary_env_cfg(heights,16)
env=ManagerBasedRlEnv(cfg=cfg,device="cpu")
rows=[]
try:
 for card in rb.POSTURE_CARDS:
  print("card",card["name"])
  rows += rb.run_card_repeat(env,heights=heights,card=card,repeat=1,
    settle_steps=rb.OFFICIAL_SETTLE_STEPS,drive_steps=rb.OFFICIAL_DRIVE_STEPS,stable_steps=rb.OFFICIAL_STABLE_STEPS)
finally: env.close()
payload={"kind":"cpu_2p5x16_diagnostic","evidence_eligible":False,"git_sha":rb._git_sha(rb.REPOSITORY_PATH),"trials":rows}
out.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
for card in rb.POSTURE_CARDS:
 for h in heights:
  rs=[t for t in rows if t["posture_card"]==card["name"] and t["stair_height_m"]==h]
  print(card["name"],h,"success",sum(t["success"] for t in rs),"air",sum(t["bilateral_airborne_ever"] for t in rs),"substeps",sum(t["bilateral_unsupported_physics_substeps"] for t in rs),"termination",sum(t["termination"] for t in rs),"nonwheel",sum(t["non_wheel_contact"] for t in rs))
print("output",out)
