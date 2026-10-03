import json
import pathlib

from mjlab.envs import ManagerBasedRlEnv
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb

out = pathlib.Path(r"D:\mjlab_workspace\roll_boundary_control_ablation_d99c491.json")
heights = (0.0, 0.0025, 0.005)
variants = [
    ("frozen_c1", None),
    ("wheel_speed_feedback_zero", 0.0),
]
all_results = []
for label, wheel_gain in variants:
    cfg = rb.make_roll_boundary_env_cfg(heights, 1)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        term = env.action_manager.get_term("hybrid_wheel_leg")
        original = term._schedule_gains.clone()
        if wheel_gain is not None:
            term._schedule_gains[..., 3] = wheel_gain
        rows = rb.run_card_repeat(
            env,
            heights=heights,
            card=rb.POSTURE_CARDS[0],
            repeat=1,
            settle_steps=rb.OFFICIAL_SETTLE_STEPS,
            drive_steps=rb.OFFICIAL_DRIVE_STEPS,
            stable_steps=rb.OFFICIAL_STABLE_STEPS,
        )
        all_results.append({
            "label": label,
            "controller_changed": wheel_gain is not None,
            "evidence_eligible": False,
            "schedule_gain_original": original.tolist(),
            "schedule_gain_used": term._schedule_gains.tolist(),
            "trials": rows,
        })
    finally:
        env.close()
payload = {
    "kind": "diagnostic_control_ablation",
    "evidence_eligible": False,
    "reason": "counterfactual controller diagnostic; not a protocol candidate",
    "git_sha": rb._git_sha(rb.REPOSITORY_PATH),
    "heights_m": heights,
    "variants": all_results,
}
out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
for result in all_results:
    print(result["label"])
    for t in result["trials"]:
        print({k: t[k] for k in (
            "stair_height_m", "success", "bilateral_unsupported_physics_substeps",
            "max_progress_past_face_m", "peak_pitch_abs_rad", "wheel_target_forward_radps_mean",
            "wheel_speed_forward_radps_mean", "torque_saturation_fraction",
        )})
print(f"output={out}")
