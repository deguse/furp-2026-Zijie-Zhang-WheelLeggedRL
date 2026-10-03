from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
from mjlab.actuator import BuiltinPdActuatorCfg
from mjlab.envs import ManagerBasedRlEnv
from hoppertrex_mjlab.assets.HopperTrex_CFG import (
 LEG_JOINT_NAMES,LEG_POSITION_STIFFNESS,LEG_POSITION_DAMPING,DM_J6248P_PEAK_TORQUE,
 HOPPERTREX_WHEEL_ACTUATOR_CFG,
)
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb
from hoppertrex_mjlab.scripts import diagnose_roll_boundary as diag
from hoppertrex_mjlab.scripts import probe_r0c_yaw_feedback as lateral

HEIGHTS=(0.0,0.0025)
def git(*a):return subprocess.run(['git',*a],cwd=rb.REPOSITORY_PATH,check=True,capture_output=True,text=True).stdout.strip()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source-result',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 source=json.loads(a.source_result.read_text());cand=lateral._candidate_map()['c0'];resets=lateral._source_reset_map(source,'c0')
 cfg=rb.make_roll_boundary_env_cfg(HEIGHTS,diag.R0C_SYNC_ENVS_PER_HEIGHT)
 robot_cfg=cfg.scene.entities['robot']
 robot_cfg.articulation.actuators=(BuiltinPdActuatorCfg(target_names_expr=LEG_JOINT_NAMES,stiffness=LEG_POSITION_STIFFNESS,damping=LEG_POSITION_DAMPING,effort_limit=DM_J6248P_PEAK_TORQUE,armature=0.02,frictionloss=0.0,viscous_damping=0.0),HOPPERTREX_WHEEL_ACTUATOR_CFG)
 old=rb.POSTURE_CARDS;rb.POSTURE_CARDS=(cand['posture_card'],);env=ManagerBasedRlEnv(cfg=cfg,device='cpu')
 try:
  rows=rb.run_card_repeat(env,heights=HEIGHTS,card=cand['posture_card'],repeat=1,settle_steps=rb.OFFICIAL_SETTLE_STEPS,drive_steps=rb.OFFICIAL_DRIVE_STEPS,stable_steps=rb.OFFICIAL_STABLE_STEPS,episode_wide_safety=True,diagnostic_continue_after_support_loss=True,roll_pose_schedule=cand['schedule'],roll_pose_slew_mode=str(cand['slew_mode']),require_pure_classical_authority=True,root_reset_override=__import__('hoppertrex_mjlab.scripts.probe_r0c_support_transfer_shadow',fromlist=['_reset_override'])._reset_override(resets),command_vx_mps=0.07)
  layout=[{'type':type(x).__name__,'targets':list(x.target_names)} for x in env.scene['robot'].actuators]
 finally:env.close();rb.POSTURE_CARDS=old
 lateral._validate_resets(rows,resets)
 payload={'kind':'r0c_builtin_pd_zero_velocity_equivalence_development_probe','evidence_eligible':False,'promotion_eligible':False,'git_sha':git('rev-parse','HEAD'),'project_dirty':bool(git('status','--porcelain')),'source_result':str(a.source_result.resolve()),'matched_reviewed_resets':True,'leg_actuator':{'type':'BuiltinPdActuatorCfg','stiffness':LEG_POSITION_STIFFNESS,'damping':LEG_POSITION_DAMPING,'effort_limit':DM_J6248P_PEAK_TORQUE,'velocity_target':'exact_zero'},'runtime_actuator_layout':layout,'summaries':[{'stair_height_m':h,'trials':len(g),'successes':sum(bool(r['success']) for r in g),'geometric_successes_ignoring_support':sum(bool(r.get('geometric_success_ignoring_support',False)) for r in g),'bilateral_unsupported_physics_substeps':sum(int(r['bilateral_unsupported_physics_substeps']) for r in g),'unsafe_trials':sum(bool(r['bilateral_airborne_ever']) or bool(r['non_wheel_contact']) or r['termination'] is not None for r in g),'safe_stalls':sum((not bool(r['success'])) and (not bool(r['bilateral_airborne_ever'])) and (not bool(r['non_wheel_contact'])) and r['termination'] is None for r in g)} for h in HEIGHTS for g in [[r for r in rows if float(r['stair_height_m'])==h]]],'trials':rows}
 rb._atomic_write_json(a.output.resolve(),payload);print('output='+str(a.output.resolve()))
if __name__=='__main__':main()
