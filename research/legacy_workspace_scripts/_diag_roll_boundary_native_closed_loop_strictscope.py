"""Run the exact MjLab C1/action stack around native MuJoCo physics."""

from __future__ import annotations

import json
import pathlib
import subprocess
from typing import Any

import mujoco
import numpy as np
import torch
from _diag_roll_boundary_native_replay_strictscope import (
  DRIVE_STEPS,
  HEIGHT_M,
  SETTLE_STEPS,
  _jsonable,
  _native_contact,
  _sha256_arrays,
  _summary,
)
from mjlab.envs import ManagerBasedRlEnv

from hoppertrex_mjlab.scripts import probe_roll_boundary as rb

SOURCE = pathlib.Path(
  r"D:\mjlab_workspace\roll_boundary_native_replay_strictscope_20260814.json"
)
OUTPUT = pathlib.Path(
  r"D:\mjlab_workspace\roll_boundary_native_closed_loop_strictscope_20260814.json"
)


def _copy_native_state_to_shadow(env: ManagerBasedRlEnv, data: mujoco.MjData) -> None:
  env.sim.data.qpos[0].copy_(torch.as_tensor(data.qpos, device=env.device))
  env.sim.data.qvel[0].copy_(torch.as_tensor(data.qvel, device=env.device))
  if env.sim.mj_model.na:
    env.sim.data.act[0].copy_(torch.as_tensor(data.act, device=env.device))
  if env.sim.mj_model.nmocap:
    env.sim.data.mocap_pos[0].copy_(torch.as_tensor(data.mocap_pos, device=env.device))
    env.sim.data.mocap_quat[0].copy_(torch.as_tensor(data.mocap_quat, device=env.device))
  env.sim.forward()
  env.scene.update(dt=env.physics_dt)


def _run_native_card(card: dict[str, float | str]) -> dict[str, Any]:
  cfg = rb.make_roll_boundary_env_cfg((HEIGHT_M,), 1)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    _terrain_types, face_x_tensor, cross_x_tensor, reset = rb._reset_to_approach(
      env,
      root_height=float(card["height_m"]),
      card_name=str(card["name"]),
      repeat=1,
      height_count=1,
    )
    face_x = float(face_x_tensor[0])
    cross_x = float(cross_x_tensor[0])
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    initial_qpos = env.sim.data.qpos[0].cpu().numpy().copy()
    initial_qvel = env.sim.data.qvel[0].cpu().numpy().copy()
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    if model.na:
      data.act[:] = env.sim.data.act[0].cpu().numpy()
    if model.nmocap:
      data.mocap_pos[:] = env.sim.data.mocap_pos[0].cpu().numpy()
      data.mocap_quat[:] = env.sim.data.mocap_quat[0].cpu().numpy()
    mujoco.mj_forward(model, data)

    wheel_joint_ids = tuple(
      model.joint(name).id for name in ("robot/wheel_left", "robot/wheel_right")
    )
    wheel_dof_ids = tuple(int(model.jnt_dofadr[joint_id]) for joint_id in wheel_joint_ids)
    wheel_actuator_ids = tuple(
      model.actuator(name).id for name in ("robot/wheel_left", "robot/wheel_right")
    )
    wheel_geom_ids = tuple(
      model.geom(name).id
      for name in ("robot/wheel_left_collision", "robot/wheel_right_collision")
    )
    terrain_body_id = model.body("terrain").id
    actions = torch.zeros((1, env.action_space.shape[-1]), device=env.device)
    active = torch.ones(1, dtype=torch.bool, device=env.device)
    controls: list[np.ndarray] = []
    samples: list[dict[str, Any]] = []

    for control_step in range(SETTLE_STEPS + DRIVE_STEPS):
      _copy_native_state_to_shadow(env, data)
      rb._force_commands(
        env,
        active=active,
        vx=0.0 if control_step < SETTLE_STEPS else rb.COMMAND_VX_MPS,
        height=float(card["height_m"]),
        pitch=float(card["pitch_rad"]),
      )
      env.action_manager.process_action(actions)
      for _ in range(rb.ROLL_FIRST_CONTROL_DECIMATION):
        _copy_native_state_to_shadow(env, data)
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        control = env.sim.data.ctrl[0].cpu().numpy().copy()
        controls.append(control)
        data.ctrl[:] = control
        mujoco.mj_step(model, data)
        contact, force = _native_contact(
          model, data, wheel_geom_ids, terrain_body_id,
        )
        samples.append({
          "index": len(samples),
          "root_x": float(data.qpos[0]),
          "root_z": float(data.qpos[2]),
          "left": contact[0],
          "right": contact[1],
          "left_force": force[0],
          "right_force": force[1],
          "wheel_speed_forward_radps": float(
            0.5 * (data.qvel[wheel_dof_ids[1]] - data.qvel[wheel_dof_ids[0]])
          ),
          "wheel_ctrl": [float(data.ctrl[actuator_id]) for actuator_id in wheel_actuator_ids],
        })

    return {
      "posture_card": dict(card),
      "face_x_m": face_x,
      "cross_x_m": cross_x,
      "root_reset": {key: _jsonable(value) for key, value in reset.items()},
      "initial_state_sha256": _sha256_arrays([initial_qpos, initial_qvel]),
      "native_control_trace_sha256": _sha256_arrays(controls),
      "native_closed_loop": _summary(samples, face_x, cross_x),
    }
  finally:
    env.close()


def main() -> None:
  if OUTPUT.exists():
    raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
  source = json.loads(SOURCE.read_text(encoding="utf-8"))
  source_by_card = {
    item["posture_card"]["name"]: item for item in source["cards"]
  }
  cards = []
  for card in rb.POSTURE_CARDS:
    result = _run_native_card(dict(card))
    previous = source_by_card[result["posture_card"]["name"]]
    if result["initial_state_sha256"] != previous["initial_state_sha256"]:
      raise RuntimeError("Native closed-loop reset differs from the paired MJWarp reset.")
    result["warp"] = previous["warp"]
    cards.append(result)
    print(json.dumps({
      "posture_card": result["posture_card"]["name"],
      "warp": result["warp"],
      "native_closed_loop": result["native_closed_loop"],
    }, indent=2))

  repo = pathlib.Path(rb.REPOSITORY_PATH)
  git_sha = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True,
  ).stdout.strip()
  dirty = bool(subprocess.run(
    ["git", "status", "--porcelain"], cwd=repo, check=True, capture_output=True, text=True,
  ).stdout.strip())
  payload = {
    "kind": "roll_boundary_native_closed_loop_backend_diagnostic",
    "evidence_eligible": False,
    "reason": (
      "Local CPU backend isolation only. Native physics is closed around the exact MjLab "
      "C1 action and actuator stack, but this is not a formal CUDA qualification."
    ),
    "git_sha": git_sha,
    "worktree_dirty": dirty,
    "height_m": HEIGHT_M,
    "physics_dt_s": rb.ROLL_FIRST_PHYSICS_TIMESTEP_S,
    "control_decimation": rb.ROLL_FIRST_CONTROL_DECIMATION,
    "settle_steps": SETTLE_STEPS,
    "drive_steps": DRIVE_STEPS,
    "cards": cards,
  }
  OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
  print(f"output={OUTPUT}")


if __name__ == "__main__":
  main()
