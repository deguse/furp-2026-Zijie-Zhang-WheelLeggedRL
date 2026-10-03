import json
import pathlib

import torch
from mjlab.envs import ManagerBasedRlEnv
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb

out = pathlib.Path(r"D:\mjlab_workspace\roll_boundary_trace_0_2p5_d99c491_v2.json")
heights = (0.0, 0.0025)
cfg = rb.make_roll_boundary_env_cfg(heights, 1)
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
trace = []
original_installer = rb.install_strict_substep_support_recorder


def traced_installer(e):
    state, restore = original_installer(e)
    inner = e.scene.update
    substep = {"i": 0}

    def update(dt):
        inner(dt)
        if not state["enabled"]:
            return
        substep["i"] += 1
        robot = e.scene["robot"]
        term = e.action_manager.get_term("hybrid_wheel_leg")
        left_sensor = e.scene[rb.LEFT_SENSOR].data
        right_sensor = e.scene[rb.RIGHT_SENSOR].data
        left_force = torch.linalg.vector_norm(left_sensor.force, dim=-1).sum(dim=-1)
        right_force = torch.linalg.vector_norm(right_sensor.force, dim=-1).sum(dim=-1)
        pitch, roll = rb._pitch_roll(robot)
        clearance = rb.wheel_clearance_above_flat_m(e)
        face = e.scene.env_origins[:, 0] + rb.approach_geometry(0.0)["outer_face_x"]
        for env_id in range(e.num_envs):
            trace.append({
                "substep": substep["i"],
                "env_id": env_id,
                "terrain_type": int(e.scene.terrain.terrain_types[env_id]),
                "progress_m": float(robot.data.root_link_pos_w[env_id, 0] - face[env_id]),
                "root_z_m": float(robot.data.root_link_pos_w[env_id, 2]),
                "root_vx_mps": float(robot.data.root_link_lin_vel_w[env_id, 0]),
                "root_vz_mps": float(robot.data.root_link_lin_vel_w[env_id, 2]),
                "pitch_rad": float(pitch[env_id]),
                "roll_rad": float(roll[env_id]),
                "pitch_rate_radps": float(robot.data.root_link_ang_vel_b[env_id, 1]),
                "left_force_n": float(left_force[env_id]),
                "right_force_n": float(right_force[env_id]),
                "left_found": int(left_sensor.found[env_id].sum()),
                "right_found": int(right_sensor.found[env_id].sum()),
                "left_min_dist_m": float(left_sensor.dist[env_id].min()),
                "right_min_dist_m": float(right_sensor.dist[env_id].min()),
                "left_contact_pos": left_sensor.pos[env_id, 0].tolist(),
                "right_contact_pos": right_sensor.pos[env_id, 0].tolist(),
                "wheel_clearance_ground_m": clearance[env_id].tolist(),
                "wheel_speed_radps": robot.data.joint_vel[env_id, term._wheel_ids].tolist(),
                "wheel_target_radps": term.wheel_targets[env_id].tolist(),
                "wheel_actuator_force_nm": robot.data.actuator_force[env_id, term._wheel_ids].tolist(),
            })

    e.scene.update = update
    return state, restore


rb.install_strict_substep_support_recorder = traced_installer
try:
    rows = rb.run_card_repeat(
        env,
        heights=heights,
        card=rb.POSTURE_CARDS[0],
        repeat=1,
        settle_steps=rb.OFFICIAL_SETTLE_STEPS,
        drive_steps=rb.OFFICIAL_DRIVE_STEPS,
        stable_steps=rb.OFFICIAL_STABLE_STEPS,
    )
finally:
    rb.install_strict_substep_support_recorder = original_installer
    env.close()
payload = {"evidence_eligible": False, "heights_m": heights, "rows": rows, "trace": trace}
out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(f"trace={out} samples={len(trace)}")
for env_id, height in enumerate(heights):
    samples = [x for x in trace if x["env_id"] == env_id]
    airborne = [x for x in samples if x["left_force_n"] == 0 and x["right_force_n"] == 0]
    print(f"env={env_id} h={height} samples={len(samples)} airborne={len(airborne)}")
    for item in airborne[:10]:
        print(item)
