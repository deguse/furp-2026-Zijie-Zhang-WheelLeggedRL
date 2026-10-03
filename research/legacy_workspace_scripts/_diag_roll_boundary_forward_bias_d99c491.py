import json,pathlib,types,torch
from mjlab.envs import ManagerBasedRlEnv
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb
out=pathlib.Path(r"D:\mjlab_workspace\roll_boundary_forward_bias_d99c491.json")
heights=(0.0,0.0025,0.005)
results=[]
for bias in (0.0,0.5,1.0,2.0):
 cfg=rb.make_roll_boundary_env_cfg(heights,4)
 env=ManagerBasedRlEnv(cfg=cfg,device="cpu")
 rows=[]
 try:
  term=env.action_manager.get_term("hybrid_wheel_leg")
  original=term.process_actions
  if bias:
   def process(self, actions, original=original, bias=bias):
    original(actions)
    self._controller_baseline[:,0] -= bias
    self._controller_baseline[:,1] += bias
    self._wheel_targets[:,0] = torch.clamp(self._wheel_targets[:,0]-bias,-self.cfg.wheel_velocity_limit,self.cfg.wheel_velocity_limit)
    self._wheel_targets[:,1] = torch.clamp(self._wheel_targets[:,1]+bias,-self.cfg.wheel_velocity_limit,self.cfg.wheel_velocity_limit)
    self._previous_wheel_targets[:] = self._wheel_targets
   term.process_actions=types.MethodType(process,term)
  for card in rb.POSTURE_CARDS:
   rows+=rb.run_card_repeat(env,heights=heights,card=card,repeat=1,
    settle_steps=rb.OFFICIAL_SETTLE_STEPS,drive_steps=rb.OFFICIAL_DRIVE_STEPS,stable_steps=rb.OFFICIAL_STABLE_STEPS)
 finally: env.close()
 results.append({"forward_wheel_target_bias_radps":bias,"evidence_eligible":False,"trials":rows})
payload={"kind":"forward_bias_ablation","evidence_eligible":False,"reason":"counterfactual diagnostic only","git_sha":rb._git_sha(rb.REPOSITORY_PATH),"variants":results}
out.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
for result in results:
 print("bias",result["forward_wheel_target_bias_radps"])
 for card in rb.POSTURE_CARDS:
  for h in heights:
   rs=[t for t in result["trials"] if t["posture_card"]==card["name"] and t["stair_height_m"]==h]
   print(card["name"],h,"success",sum(t["success"] for t in rs),"air",sum(t["bilateral_airborne_ever"] for t in rs),"sub",sum(t["bilateral_unsupported_physics_substeps"] for t in rs),"progress",[round(t["max_progress_past_face_m"],4) for t in rs])
print("output",out)
