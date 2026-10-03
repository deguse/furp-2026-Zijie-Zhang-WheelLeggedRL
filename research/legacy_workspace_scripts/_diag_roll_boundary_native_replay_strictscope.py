"""Paired MJWarp/native MuJoCo replay for the 2.5 mm RollBoundary cell."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
from typing import Any

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from hoppertrex_mjlab.scripts import probe_roll_boundary as rb

OUTPUT = pathlib.Path(
  r"D:\mjlab_workspace\roll_boundary_native_replay_strictscope_20260814.json"
)
HEIGHT_M = 0.0025
SETTLE_STEPS = rb.OFFICIAL_SETTLE_STEPS
DRIVE_STEPS = rb.OFFICIAL_DRIVE_STEPS
SUBSTEPS = rb.ROLL_FIRST_CONTROL_DECIMATION


def _numpy(value: torch.Tensor) -> np.ndarray:
  return value.detach().cpu().numpy().copy()


def _jsonable(value: Any) -> Any:
  if isinstance(value, torch.Tensor):
    return _numpy(value).tolist()
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  return value


def _sha256_arrays(values: list[np.ndarray]) -> str:
  digest = hashlib.sha256()
  for value in values:
    array = np.ascontiguousarray(value)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
  return digest.hexdigest()


def _summary(samples: list[dict[str, Any]], face_x: float, cross_x: float) -> dict[str, Any]:
  drive_start = SETTLE_STEPS * SUBSTEPS
  drive = samples[drive_start:]
  unsupported = [sample for sample in samples if not sample["left"] and not sample["right"]]
  drive_unsupported = [
    sample for sample in drive if not sample["left"] and not sample["right"]
  ]
  return {
    "total_physics_substeps": len(samples),
    "settle_unsupported_substeps": len(unsupported) - len(drive_unsupported),
    "drive_unsupported_substeps": len(drive_unsupported),
    "first_drive_unsupported_substep": (
      None if not drive_unsupported else drive_unsupported[0]["index"] - drive_start
    ),
    "max_progress_past_face_m": max(sample["root_x"] - face_x for sample in drive),
    "final_progress_past_face_m": drive[-1]["root_x"] - face_x,
    "crossed_success_line": any(sample["root_x"] >= cross_x for sample in drive),
    "max_root_z_m": max(sample["root_z"] for sample in samples),
    "min_root_z_m": min(sample["root_z"] for sample in samples),
  }


def _event_window(
  warp_samples: list[dict[str, Any]], native_samples: list[dict[str, Any]], radius: int = 4,
) -> list[dict[str, Any]]:
  drive_start = SETTLE_STEPS * SUBSTEPS
  first = next(
    (
      sample["index"] for sample in warp_samples[drive_start:]
      if not sample["left"] and not sample["right"]
    ),
    None,
  )
  if first is None:
    return []
  start, stop = max(drive_start, first - radius), min(len(warp_samples), first + radius + 1)
  return [
    {
      "index": index,
      "drive_substep": index - drive_start,
      "warp": warp_samples[index],
      "native": native_samples[index],
    }
    for index in range(start, stop)
  ]


def _native_contact(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  wheel_geom_ids: tuple[int, int],
  terrain_body_id: int,
) -> tuple[tuple[bool, bool], tuple[float, float]]:
  forces = [0.0, 0.0]
  for contact_id in range(data.ncon):
    contact = data.contact[contact_id]
    geom_a, geom_b = int(contact.geom[0]), int(contact.geom[1])
    for side, wheel_geom_id in enumerate(wheel_geom_ids):
      if wheel_geom_id not in (geom_a, geom_b):
        continue
      other = geom_b if geom_a == wheel_geom_id else geom_a
      if int(model.geom_bodyid[other]) != terrain_body_id or contact.efc_address < 0:
        continue
      wrench = np.zeros(6, dtype=np.float64)
      mujoco.mj_contactForce(model, data, contact_id, wrench)
      forces[side] += abs(float(wrench[0]))
  return (forces[0] > 0.0, forces[1] > 0.0), (forces[0], forces[1])


def _run_card(card: dict[str, float | str]) -> dict[str, Any]:
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
    initial = {
      "qpos": _numpy(env.sim.data.qpos[0]),
      "qvel": _numpy(env.sim.data.qvel[0]),
      "ctrl": _numpy(env.sim.data.ctrl[0]),
      "time": float(env.sim.data.time[0]),
    }
    if env.sim.mj_model.na:
      initial["act"] = _numpy(env.sim.data.act[0])
    if env.sim.mj_model.nmocap:
      initial["mocap_pos"] = _numpy(env.sim.data.mocap_pos[0])
      initial["mocap_quat"] = _numpy(env.sim.data.mocap_quat[0])

    control_trace: list[np.ndarray] = []
    warp_samples: list[dict[str, Any]] = []
    original_step = env.sim.step
    original_update = env.scene.update
    robot = env.scene["robot"]
    model = env.sim.mj_model
    wheel_joint_ids = tuple(
      model.joint(name).id for name in ("robot/wheel_left", "robot/wheel_right")
    )
    wheel_dof_ids = tuple(int(model.jnt_dofadr[joint_id]) for joint_id in wheel_joint_ids)
    wheel_actuator_ids = tuple(
      model.actuator(name).id for name in ("robot/wheel_left", "robot/wheel_right")
    )

    def record_step() -> None:
      control_trace.append(_numpy(env.sim.data.ctrl[0]))
      original_step()

    def record_update(dt: float) -> None:
      original_update(dt)
      left_force = float(torch.linalg.vector_norm(
        env.scene[rb.LEFT_SENSOR].data.force[0].reshape(-1, 3), dim=-1,
      ).sum())
      right_force = float(torch.linalg.vector_norm(
        env.scene[rb.RIGHT_SENSOR].data.force[0].reshape(-1, 3), dim=-1,
      ).sum())
      qvel = env.sim.data.qvel[0]
      warp_samples.append({
        "index": len(warp_samples),
        "root_x": float(robot.data.root_link_pos_w[0, 0]),
        "root_z": float(robot.data.root_link_pos_w[0, 2]),
        "left": left_force > 0.0,
        "right": right_force > 0.0,
        "left_force": left_force,
        "right_force": right_force,
        "wheel_speed_forward_radps": float(
          0.5 * (qvel[wheel_dof_ids[1]] - qvel[wheel_dof_ids[0]])
        ),
        "wheel_ctrl": [float(env.sim.data.ctrl[0, actuator_id]) for actuator_id in wheel_actuator_ids],
      })

    env.sim.step = record_step
    env.scene.update = record_update
    actions = torch.zeros((1, env.action_space.shape[-1]), device=env.device)
    active = torch.ones(1, dtype=torch.bool, device=env.device)
    terminated = False
    try:
      for control_step in range(SETTLE_STEPS + DRIVE_STEPS):
        rb._force_commands(
          env,
          active=active,
          vx=0.0 if control_step < SETTLE_STEPS else rb.COMMAND_VX_MPS,
          height=float(card["height_m"]),
          pitch=float(card["pitch_rad"]),
        )
        _observation, _reward, done, timeout, _extras = env.step(actions)
        if bool(done[0]) or bool(timeout[0]):
          terminated = True
          break
    finally:
      env.sim.step = original_step
      env.scene.update = original_update

    if len(control_trace) != len(warp_samples):
      raise RuntimeError("MJWarp control and state traces have different lengths.")

    native = mujoco.MjData(model)
    native.qpos[:] = initial["qpos"]
    native.qvel[:] = initial["qvel"]
    native.ctrl[:] = initial["ctrl"]
    native.time = initial["time"]
    if model.na:
      native.act[:] = initial["act"]
    if model.nmocap:
      native.mocap_pos[:] = initial["mocap_pos"]
      native.mocap_quat[:] = initial["mocap_quat"]
    mujoco.mj_forward(model, native)

    wheel_geom_ids = tuple(
      model.geom(name).id
      for name in ("robot/wheel_left_collision", "robot/wheel_right_collision")
    )
    terrain_body_id = model.body("terrain").id
    native_samples: list[dict[str, Any]] = []
    for index, control in enumerate(control_trace):
      native.ctrl[:] = control
      mujoco.mj_step(model, native)
      contact, force = _native_contact(model, native, wheel_geom_ids, terrain_body_id)
      native_samples.append({
        "index": index,
        "root_x": float(native.qpos[0]),
        "root_z": float(native.qpos[2]),
        "left": contact[0],
        "right": contact[1],
        "left_force": force[0],
        "right_force": force[1],
        "wheel_speed_forward_radps": float(
          0.5 * (native.qvel[wheel_dof_ids[1]] - native.qvel[wheel_dof_ids[0]])
        ),
        "wheel_ctrl": [float(native.ctrl[actuator_id]) for actuator_id in wheel_actuator_ids],
      })

    return {
      "posture_card": dict(card),
      "terminated_early": terminated,
      "face_x_m": face_x,
      "cross_x_m": cross_x,
      "root_reset": {
        key: _jsonable(value)
        for key, value in reset.items()
      },
      "initial_state_sha256": _sha256_arrays([initial["qpos"], initial["qvel"]]),
      "control_trace_sha256": _sha256_arrays(control_trace),
      "warp": _summary(warp_samples, face_x, cross_x),
      "native_open_loop_replay": _summary(native_samples, face_x, cross_x),
      "first_warp_unsupported_window": _event_window(warp_samples, native_samples),
    }
  finally:
    env.close()


def main() -> None:
  if OUTPUT.exists():
    raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
  repo = pathlib.Path(rb.REPOSITORY_PATH)
  git_sha = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True,
  ).stdout.strip()
  dirty = bool(subprocess.run(
    ["git", "status", "--porcelain"], cwd=repo, check=True, capture_output=True, text=True,
  ).stdout.strip())
  cards = []
  for card in rb.POSTURE_CARDS:
    result = _run_card(dict(card))
    cards.append(result)
    print(json.dumps({
      "posture_card": result["posture_card"]["name"],
      "warp": result["warp"],
      "native_open_loop_replay": result["native_open_loop_replay"],
    }, indent=2))
  payload = {
    "kind": "roll_boundary_native_open_loop_replay",
    "evidence_eligible": False,
    "reason": (
      "Local CPU backend isolation only; native uses the exact MJWarp-generated open-loop "
      "actuator control trace and is not a native closed-loop qualification."
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
  print(json.dumps({
    card["posture_card"]["name"]: {
      "warp": card["warp"],
      "native_open_loop_replay": card["native_open_loop_replay"],
    }
    for card in cards
  }, indent=2))
  print(f"output={OUTPUT}")


if __name__ == "__main__":
  main()
