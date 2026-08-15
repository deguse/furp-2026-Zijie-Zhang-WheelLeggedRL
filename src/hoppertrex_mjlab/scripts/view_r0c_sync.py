#!/usr/bin/env python3
"""Inspect one frozen R0c-SYNC case in MjLab's live Viser viewer."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mjlab
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer.base import VerbosityLevel
from mjlab.viewer.viser.viewer import UpdateReason, ViserPlayViewer

from hoppertrex_mjlab.scripts import diagnose_roll_boundary as diag
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb

EXPECTED_RESULT_SHA256 = "dd8826dcb1abd77ddeb1c68d2bef1406d74b045365d49915d672379e74267c14"
EXPECTED_RESULT_GIT_SHA = "a31a60aea08b142d48d3ac2d0523a20aa9dad3c5"
R0C_HEIGHTS_M = (0.0, 0.0025)
CANDIDATE_KINDS = {
  "c0": "legacy_independent_slew_baseline",
  "c1": "shared_alpha_synchronized_slew",
}


@dataclass(frozen=True)
class R0cViewCase:
  result_path: Path
  payload: Mapping[str, Any]
  candidate: Mapping[str, Any]
  result: Mapping[str, Any]
  trial: Mapping[str, Any]
  event: Mapping[str, Any] | None
  candidate_key: str
  env_id: int


@dataclass(frozen=True)
class TriggerSnapshot:
  root_state: torch.Tensor
  joint_pos: torch.Tensor
  joint_vel: torch.Tensor
  physics_substep: int
  episode_control_step: int
  drive_control_step: int | None
  phase: str
  progress_m: float
  root_z_m: float
  root_vz_mps: float
  pitch_rad: float
  left_force_n: float
  right_force_n: float
  left_vertical_load_n: float
  right_vertical_load_n: float
  left_clearance_m: float
  right_clearance_m: float
  applied_alpha: float


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(path: Path, *args: str) -> str:
  result = subprocess.run(
    ["git", *args], cwd=path, check=False, capture_output=True, text=True,
  )
  if result.returncode != 0:
    raise RuntimeError(
      f"git {' '.join(args)} failed in {path}: {result.stderr.strip()}"
    )
  return result.stdout.strip()


def _require_finite_sequence(value: Any, *, length: int, name: str) -> list[float]:
  if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
    raise TypeError(f"{name} must be a numeric sequence.")
  result = [float(item) for item in value]
  if len(result) != length or not all(math.isfinite(item) for item in result):
    raise ValueError(f"{name} must contain {length} finite values.")
  return result


def _validate_reset(reset: Any) -> Mapping[str, Any]:
  if not isinstance(reset, Mapping):
    raise TypeError("Selected trial has no root_reset mapping.")
  for name in ("x_relative_to_face_m", "y_relative_to_center_m", "root_height_m"):
    value = float(reset[name])
    if not math.isfinite(value):
      raise ValueError(f"root_reset.{name} must be finite.")
  _require_finite_sequence(
    reset["root_linear_velocity_mps"], length=3,
    name="root_reset.root_linear_velocity_mps",
  )
  _require_finite_sequence(
    reset["root_angular_velocity_radps"], length=3,
    name="root_reset.root_angular_velocity_radps",
  )
  _require_finite_sequence(
    reset["root_quaternion_wxyz"], length=4,
    name="root_reset.root_quaternion_wxyz",
  )
  _require_finite_sequence(
    reset["leg_joint_position_rad"], length=4,
    name="root_reset.leg_joint_position_rad",
  )
  _require_finite_sequence(
    reset["leg_joint_velocity_radps"], length=4,
    name="root_reset.leg_joint_velocity_radps",
  )
  return reset


def load_view_case(
  result_path: Path, *, candidate_key: str, env_id: int,
  enforce_file_hash: bool = True,
) -> R0cViewCase:
  path = result_path.resolve()
  if enforce_file_hash and _sha256(path) != EXPECTED_RESULT_SHA256:
    raise ValueError("R0c-SYNC result SHA256 does not match the reviewed artifact.")
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, Mapping):
    raise TypeError("R0c-SYNC result must be a JSON object.")
  if payload.get("kind") != "r0c_synchronized_reference_rejection_screen":
    raise ValueError("Input is not an R0c-SYNC result.")
  if payload.get("git_sha") != EXPECTED_RESULT_GIT_SHA:
    raise ValueError("R0c-SYNC result project SHA is not the reviewed run.")
  if payload.get("project_dirty") is not False or payload.get("mjlab_dirty") is not False:
    raise ValueError("R0c-SYNC result provenance is dirty.")
  if payload.get("matched_reset_perturbations_across_candidates") is not True:
    raise ValueError("R0c-SYNC result does not prove matched C0/C1 resets.")
  if candidate_key not in CANDIDATE_KINDS:
    raise ValueError(f"Unknown R0c-SYNC candidate: {candidate_key}.")
  if env_id not in range(8, 16):
    raise ValueError("R0c-SYNC 2.5 mm env-id must be in [8, 15].")

  runtime_candidates = {
    str(candidate["candidate_definition"]["kind"]): candidate
    for candidate in payload["candidates"]
  }
  kind = CANDIDATE_KINDS[candidate_key]
  result = runtime_candidates.get(kind)
  if result is None:
    raise ValueError(f"R0c-SYNC result has no {candidate_key} arm.")
  if not isinstance(result, Mapping):
    raise TypeError(f"R0c-SYNC {candidate_key} arm must be a mapping.")
  candidate = next(
    item for item in diag.r0c_sync_candidates()
    if str(item["kind"]) == kind
  )
  if result.get("candidate_definition") != diag._schedule_candidate_definition(candidate):
    raise ValueError("R0c-SYNC candidate definition drifted.")
  matches = [
    row for row in result["trials"]
    if int(row["repeat"]) == 1
    and int(row["terrain_index"]) == 1
    and int(row["env_id"]) == env_id
  ]
  if len(matches) != 1:
    raise ValueError("Selected R0c-SYNC trial is missing or duplicated.")
  trial = matches[0]
  _validate_reset(trial.get("root_reset"))
  events = [
    event for event in result["first_support_loss_events"]
    if int(event["repeat"]) == 1 and int(event["env_id"]) == env_id
  ]
  unsupported = int(trial["bilateral_unsupported_physics_substeps"])
  if len(events) != (1 if unsupported > 0 else 0):
    raise ValueError("Selected trial event coverage is inconsistent.")
  return R0cViewCase(
    result_path=path,
    payload=payload,
    candidate=candidate,
    result=result,
    trial=trial,
    event=events[0] if events else None,
    candidate_key=candidate_key,
    env_id=env_id,
  )


def counterfactual_crossing_counts(result: Mapping[str, Any]) -> dict[int, int]:
  """Diagnostic-only counts under total unsupported-substep allowances."""
  rows = [
    row for row in result["trials"]
    if float(row["stair_height_m"]) == 0.0025
  ]
  counts = {}
  for allowance in (0, 1, 2, 4):
    counts[allowance] = sum(
      float(row["max_progress_past_face_m"]) >= rb.CROSS_DEPTH_M
      and not bool(row["termination"])
      and not bool(row["non_wheel_contact"])
      and int(row["bilateral_unsupported_physics_substeps"]) <= allowance
      for row in rows
    )
  return counts


def validate_runtime_checkout(case: R0cViewCase, *, device: str) -> None:
  if device != "cuda:0":
    raise ValueError("Reviewed R0c-SYNC Viser replay is pinned to cuda:0.")
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; use the machine-room checkout.")
  project = rb.REPOSITORY_PATH
  mjlab_root = Path(mjlab.__file__).resolve().parents[2]
  if _git_output(project, "status", "--porcelain"):
    raise RuntimeError("R0c-SYNC Viser replay requires a clean project checkout.")
  if _git_output(mjlab_root, "status", "--porcelain"):
    raise RuntimeError("R0c-SYNC Viser replay requires a clean MjLab checkout.")
  if _git_output(mjlab_root, "rev-parse", "HEAD") != case.payload["mjlab_git_sha"]:
    raise RuntimeError("MjLab SHA differs from the reviewed R0c-SYNC run.")
  ancestor = subprocess.run(
    ["git", "merge-base", "--is-ancestor", EXPECTED_RESULT_GIT_SHA, "HEAD"],
    cwd=project, check=False,
  )
  if ancestor.returncode != 0:
    raise RuntimeError("Reviewed R0c-SYNC result SHA is not an ancestor of HEAD.")
  dynamics_paths = (
    "src/hoppertrex_mjlab/scripts/diagnose_roll_boundary.py",
    "src/hoppertrex_mjlab/scripts/probe_roll_boundary.py",
    "src/hoppertrex_mjlab/hybrid/roll_pose_schedule.py",
  )
  drift = subprocess.run(
    ["git", "diff", "--quiet", EXPECTED_RESULT_GIT_SHA, "HEAD", "--", *dynamics_paths],
    cwd=project, check=False,
  )
  if drift.returncode != 0:
    raise RuntimeError("R0c-SYNC dynamics sources changed since the reviewed run.")


def _reset_values_close(
  observed: Mapping[str, Any], expected: Mapping[str, Any], *, atol: float = 1.0e-7,
) -> bool:
  if observed.keys() != expected.keys():
    return False
  for name, expected_value in expected.items():
    observed_value = observed[name]
    if isinstance(expected_value, list):
      if not isinstance(observed_value, list) or len(observed_value) != len(expected_value):
        return False
      if not all(
        math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)
        for left, right in zip(observed_value, expected_value, strict=True)
      ):
        return False
    elif not math.isclose(
      float(observed_value), float(expected_value), rel_tol=0.0, abs_tol=atol,
    ):
      return False
  return True


def _reset_row(reset: Mapping[str, torch.Tensor], env_id: int) -> dict[str, Any]:
  return {
    "x_relative_to_face_m": float(reset["x_relative_to_face_m"][env_id]),
    "y_relative_to_center_m": float(reset["y_relative_to_center_m"][env_id]),
    "root_height_m": float(reset["root_height_m"][env_id]),
    "root_linear_velocity_mps": [
      float(value) for value in reset["root_linear_velocity_mps"][env_id].tolist()
    ],
    "root_angular_velocity_radps": [
      float(value) for value in reset["root_angular_velocity_radps"][env_id].tolist()
    ],
    "root_quaternion_wxyz": [
      float(value) for value in reset["root_quaternion_wxyz"][env_id].tolist()
    ],
    "leg_joint_position_rad": [
      float(value) for value in reset["leg_joint_position_rad"][env_id].tolist()
    ],
    "leg_joint_velocity_radps": [
      float(value) for value in reset["leg_joint_velocity_radps"][env_id].tolist()
    ],
  }


class SupportLossObserver:
  def __init__(self, env: ManagerBasedRlEnv, *, env_id: int) -> None:
    self.env = env
    self.env_id = env_id
    self.robot = env.scene["robot"]
    self.wheel_geom_ids, wheel_names = self.robot.find_geoms(
      ("wheel_left_collision", "wheel_right_collision"), preserve_order=True,
    )
    if tuple(wheel_names) != ("wheel_left_collision", "wheel_right_collision"):
      raise RuntimeError("R0c-SYNC viewer wheel geometry identity drifted.")
    self._original_update = env.scene.update
    self.face_x = torch.zeros(env.num_envs, device=env.device)
    self.schedule_alpha = torch.zeros(env.num_envs, device=env.device)
    self.enabled = False
    self.auto_freeze = True
    self.reset()
    env.scene.update = self._update

  def reset(self) -> None:
    self.enabled = False
    self.physics_substep = 0
    self.control_step = 0
    self.drive_control_step: int | None = None
    self.phase = "settle"
    self.total_unsupported = 0
    self.current_run = 0
    self.max_run = 0
    self.first_snapshot: TriggerSnapshot | None = None
    self.snapshot_displayed = False
    self.completed = False

  def set_context(
    self, *, control_step: int, drive_control_step: int | None,
    phase: str, schedule_alpha: torch.Tensor,
  ) -> None:
    self.control_step = control_step
    self.drive_control_step = drive_control_step
    self.phase = phase
    self.schedule_alpha.copy_(schedule_alpha)
    self.enabled = True

  def _update(self, dt: float) -> None:
    self._original_update(dt)
    if not self.enabled:
      return
    self.physics_substep += 1
    left_data = self.env.scene[rb.LEFT_SENSOR].data
    right_data = self.env.scene[rb.RIGHT_SENSOR].data
    left_force = torch.linalg.vector_norm(left_data.force, dim=-1).sum(dim=-1)
    right_force = torch.linalg.vector_norm(right_data.force, dim=-1).sum(dim=-1)
    unsupported = bool((
      (left_force[self.env_id] <= 0.0)
      & (right_force[self.env_id] <= 0.0)
    ).item())
    if not unsupported:
      self.current_run = 0
      return
    self.total_unsupported += 1
    self.current_run += 1
    self.max_run = max(self.max_run, self.current_run)
    if self.first_snapshot is not None:
      return

    left_load = rb.vertical_normal_load_n(
      found=left_data.found,
      force_contact_frame=left_data.force,
      normal_global=left_data.normal,
    )
    right_load = rb.vertical_normal_load_n(
      found=right_data.found,
      force_contact_frame=right_data.force,
      normal_global=right_data.normal,
    )
    wheel_center_x = self.robot.data.geom_pos_w[:, self.wheel_geom_ids, 0]
    terrain_types = self.env.scene.terrain.terrain_types
    terrain_height = torch.tensor(
      R0C_HEIGHTS_M, device=self.env.device,
      dtype=self.robot.data.root_link_pos_w.dtype,
    )[terrain_types]
    horizontal_level = torch.where(
      wheel_center_x >= self.face_x.unsqueeze(1),
      torch.floor((wheel_center_x - self.face_x.unsqueeze(1)) / rb.STEP_WIDTH_M) + 1.0,
      torch.zeros_like(wheel_center_x),
    )
    horizontal_surface = horizontal_level.clamp(min=0.0) * terrain_height.unsqueeze(1)
    clearance = rb.wheel_clearance_above_flat_m(self.env) - horizontal_surface
    pitch, _roll = rb._pitch_roll(self.robot)
    root_state = torch.cat((
      self.robot.data.root_link_pos_w,
      self.robot.data.root_link_quat_w,
      self.robot.data.root_link_lin_vel_w,
      self.robot.data.root_link_ang_vel_w,
    ), dim=1)
    i = self.env_id
    self.first_snapshot = TriggerSnapshot(
      root_state=root_state.detach().clone(),
      joint_pos=self.robot.data.joint_pos.detach().clone(),
      joint_vel=self.robot.data.joint_vel.detach().clone(),
      physics_substep=self.physics_substep,
      episode_control_step=self.control_step,
      drive_control_step=self.drive_control_step,
      phase=self.phase,
      progress_m=float(self.robot.data.root_link_pos_w[i, 0] - self.face_x[i]),
      root_z_m=float(self.robot.data.root_link_pos_w[i, 2]),
      root_vz_mps=float(self.robot.data.root_link_lin_vel_w[i, 2]),
      pitch_rad=float(pitch[i]),
      left_force_n=float(left_force[i]),
      right_force_n=float(right_force[i]),
      left_vertical_load_n=float(left_load[i]),
      right_vertical_load_n=float(right_load[i]),
      left_clearance_m=float(clearance[i, 0]),
      right_clearance_m=float(clearance[i, 1]),
      applied_alpha=float(self.schedule_alpha[i]),
    )

  def restore_first_snapshot(self) -> None:
    snapshot = self.first_snapshot
    if snapshot is None:
      raise RuntimeError("No R0c-SYNC support-loss snapshot is available.")
    self.enabled = False
    self.robot.write_root_state_to_sim(snapshot.root_state)
    self.robot.write_joint_state_to_sim(snapshot.joint_pos, snapshot.joint_vel)
    self.env.sim.forward()
    self.env.sim.sense()
    self.snapshot_displayed = True

  def close(self) -> None:
    self.enabled = False
    self.env.scene.update = self._original_update


class R0cViewerPolicy:
  def __init__(
    self, env: ManagerBasedRlEnv, *, case: R0cViewCase,
    observer: SupportLossObserver,
  ) -> None:
    self.env = env
    self.case = case
    self.observer = observer
    self.card = case.candidate["posture_card"]
    self.schedule = case.candidate["schedule"]
    self.slew_mode = str(case.candidate["slew_mode"])
    self.actions = torch.zeros(
      (env.num_envs, env.action_space.shape[-1]), device=env.device,
    )
    original_cards = rb.POSTURE_CARDS
    rb.POSTURE_CARDS = (self.card,)
    try:
      terrain_types, face_x, _cross_x, reset = rb._reset_to_approach(
        env,
        root_height=float(self.card["height_m"]),
        card_name=str(self.card["name"]),
        repeat=1,
        height_count=len(R0C_HEIGHTS_M),
      )
    finally:
      rb.POSTURE_CARDS = original_cards
    if int(terrain_types[case.env_id]) != 1:
      raise RuntimeError("Selected R0c-SYNC viewer env is not on 2.5 mm terrain.")
    artifact_reset = dict(case.trial["root_reset"])
    if not _reset_values_close(_reset_row(reset, case.env_id), artifact_reset):
      raise RuntimeError("Live R0c-SYNC reset does not match the artifact tolerance.")
    self.face_x = face_x.detach().clone()
    observer.face_x.copy_(self.face_x)
    robot = env.scene["robot"]
    selected = torch.tensor([case.env_id], dtype=torch.int64, device=env.device)
    selected_root = torch.cat((
      robot.data.root_link_pos_w[case.env_id : case.env_id + 1],
      robot.data.root_link_quat_w[case.env_id : case.env_id + 1],
      robot.data.root_link_lin_vel_w[case.env_id : case.env_id + 1],
      robot.data.root_link_ang_vel_w[case.env_id : case.env_id + 1],
    ), dim=1).clone()
    selected_root[0, 0] = face_x[case.env_id] + float(
      artifact_reset["x_relative_to_face_m"]
    )
    selected_root[0, 1] = env.scene.env_origins[case.env_id, 1] + float(
      artifact_reset["y_relative_to_center_m"]
    )
    selected_root[0, 2] = float(artifact_reset["root_height_m"])
    selected_root[0, 3:7] = torch.tensor(
      artifact_reset["root_quaternion_wxyz"],
      device=env.device, dtype=selected_root.dtype,
    )
    selected_root[0, 7:10] = torch.tensor(
      artifact_reset["root_linear_velocity_mps"],
      device=env.device, dtype=selected_root.dtype,
    )
    selected_root[0, 10:13] = torch.tensor(
      artifact_reset["root_angular_velocity_radps"],
      device=env.device, dtype=selected_root.dtype,
    )
    selected_joint_pos = robot.data.joint_pos[case.env_id : case.env_id + 1].clone()
    selected_joint_vel = robot.data.joint_vel[case.env_id : case.env_id + 1].clone()
    leg_ids = env.action_manager.get_term("hybrid_wheel_leg")._leg_ids
    selected_joint_pos[:, leg_ids] = torch.tensor(
      artifact_reset["leg_joint_position_rad"],
      device=env.device, dtype=selected_joint_pos.dtype,
    )
    selected_joint_vel[:, leg_ids] = torch.tensor(
      artifact_reset["leg_joint_velocity_radps"],
      device=env.device, dtype=selected_joint_vel.dtype,
    )
    robot.write_root_state_to_sim(selected_root, env_ids=selected)
    robot.write_joint_state_to_sim(
      selected_joint_pos, selected_joint_vel, env_ids=selected,
    )
    env.sim.forward()
    env.sim.sense()
    self.initial_root_state = torch.cat((
      robot.data.root_link_pos_w,
      robot.data.root_link_quat_w,
      robot.data.root_link_lin_vel_w,
      robot.data.root_link_ang_vel_w,
    ), dim=1).detach().clone()
    self.initial_joint_pos = robot.data.joint_pos.detach().clone()
    self.initial_joint_vel = robot.data.joint_vel.detach().clone()
    self.active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    self.reset()

  def reset(self) -> None:
    robot = self.env.scene["robot"]
    robot.write_root_state_to_sim(self.initial_root_state)
    robot.write_joint_state_to_sim(self.initial_joint_pos, self.initial_joint_vel)
    self.env.sim.forward()
    self.env.sim.sense()
    self.schedule_state = rb.make_roll_pose_schedule_state(
      self.schedule,
      robot.data.root_link_pos_w[:, 0],
      slew_mode=self.slew_mode,
    )
    self.control_step = 0
    self.observer.reset()
    self.observer.face_x.copy_(self.face_x)
    rb._force_commands(
      self.env, active=self.active, vx=0.0,
      height=float(self.card["height_m"]), pitch=float(self.card["pitch_rad"]),
    )

  def __call__(self, _observation: Any) -> torch.Tensor:
    if self.control_step >= rb.OFFICIAL_SETTLE_STEPS + rb.OFFICIAL_DRIVE_STEPS:
      self.observer.enabled = False
      self.observer.completed = True
      self.actions.zero_()
      return self.actions
    drive_index = (
      None
      if self.control_step < diag.rb.OFFICIAL_SETTLE_STEPS
      else self.control_step - diag.rb.OFFICIAL_SETTLE_STEPS
    )
    schedule_output = rb.roll_pose_schedule_step(
      self.schedule,
      self.schedule_state,
      root_x_m=self.env.scene["robot"].data.root_link_pos_w[:, 0],
      face_x_m=self.face_x,
      active_mask=self.active,
      drive_active=drive_index is not None,
      dt=rb.ROLL_POSE_CONTROL_DT_S,
    )
    self.observer.set_context(
      control_step=self.control_step + 1,
      drive_control_step=None if drive_index is None else drive_index + 1,
      phase="settle" if drive_index is None else "drive",
      schedule_alpha=schedule_output.applied_alpha,
    )
    rb._force_commands(
      self.env,
      active=self.active,
      vx=0.0 if drive_index is None else rb.COMMAND_VX_MPS,
      height=schedule_output.applied_height_m,
      pitch=schedule_output.applied_pitch_rad,
    )
    self.control_step += 1
    self.actions.zero_()
    return self.actions


class R0cSyncViserViewer(ViserPlayViewer):
  def __init__(
    self, env: ManagerBasedRlEnv, policy: R0cViewerPolicy,
    *, case: R0cViewCase, observer: SupportLossObserver,
  ) -> None:
    super().__init__(
      env, policy, frame_rate=60.0, verbosity=VerbosityLevel.INFO,
    )
    self.case = case
    self.observer = observer
    self._is_paused = True
    self._completion_paused = False

  def setup(self) -> None:
    super().setup()
    with self._server.gui.add_folder("R0c-SYNC inspection"):
      self._r0c_status = self._server.gui.add_html("")
      self._freeze_checkbox = self._server.gui.add_checkbox(
        "Freeze on first 5 ms support loss", initial_value=True,
      )

      @self._freeze_checkbox.on_update
      def _(_) -> None:
        self.observer.auto_freeze = bool(self._freeze_checkbox.value)

    self._update_r0c_status()

  def reset_environment(self) -> None:
    self.observer.enabled = False
    self._completion_paused = False
    super().reset_environment()
    self._update_r0c_status()

  def sync_env_to_viewer(self) -> None:
    if (
      self.observer.auto_freeze
      and self.observer.first_snapshot is not None
      and not self.observer.snapshot_displayed
    ):
      with self._sim_lock:
        self.observer.restore_first_snapshot()
      self.pause()
      self._sync_ui_state()
      self._pending_update_reasons.add(UpdateReason.ACTION)
      self._scene.request_update()
    elif self.observer.completed and not self._completion_paused:
      self.pause()
      self._completion_paused = True
      self._sync_ui_state()
    super().sync_env_to_viewer()
    self._update_r0c_status()

  def _update_r0c_status(self) -> None:
    if not hasattr(self, "_r0c_status"):
      return
    trial = self.case.trial
    artifact_losses = int(trial["bilateral_unsupported_physics_substeps"])
    max_progress_mm = 1_000.0 * float(trial["max_progress_past_face_m"])
    reached = max_progress_mm >= 1_000.0 * rb.CROSS_DEPTH_M
    snapshot = self.observer.first_snapshot
    if snapshot is None:
      live = "Waiting for first raw support loss."
    else:
      live = (
        f"<strong>Frozen raw event:</strong> substep {snapshot.physics_substep}, "
        f"drive step {snapshot.drive_control_step}, "
        f"progress {1_000.0 * snapshot.progress_m:+.1f} mm, "
        f"left/right load {snapshot.left_vertical_load_n:.1f}/"
        f"{snapshot.right_vertical_load_n:.1f} N, "
        f"clearance {1_000.0 * snapshot.left_clearance_m:+.2f}/"
        f"{1_000.0 * snapshot.right_clearance_m:+.2f} mm, "
        f"alpha {snapshot.applied_alpha:.3f}."
      )
    strict = (
      '<span style="color:#e74c3c;"><strong>STRICT REJECT</strong></span>'
      if artifact_losses > 0 else "STRICT SAFE"
    )
    self._r0c_status.content = (
      '<div style="font-size:0.85em;line-height:1.4;padding:0 0.5em 0.5em 0.5em;">'
      f"<strong>Case:</strong> {html.escape(self.case.candidate_key.upper())}, "
      f"env {self.case.env_id}<br/>"
      f"<strong>Artifact:</strong> {strict}; {artifact_losses} raw 5 ms "
      f"unsupported substeps<br/>"
      f"<strong>Success line:</strong> {'reached' if reached else 'not reached'}; "
      f"max progress {max_progress_mm:+.1f} mm<br/>"
      f"{live}<br/>"
      "<em>Reset Environment + Play replays the exact matched reset. "
      "This viewer does not change the formal verdict.</em></div>"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="View one reviewed R0c-SYNC case in live 3D Viser.",
  )
  parser.add_argument("--result", type=Path, required=True)
  parser.add_argument("--candidate", choices=tuple(CANDIDATE_KINDS), default="c1")
  parser.add_argument("--env-id", type=int, choices=range(8, 16), default=14)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  args = parse_args(argv)
  case = load_view_case(
    args.result, candidate_key=args.candidate, env_id=args.env_id,
  )
  validate_runtime_checkout(case, device=args.device)
  counts = counterfactual_crossing_counts(case.result)
  print(
    "[r0c-sync-viewer] diagnostic-only crossing counts by total raw "
    f"unsupported-substep allowance: {counts}"
  )
  print(
    f"[r0c-sync-viewer] selected={case.candidate_key} env={case.env_id} "
    f"artifact_losses={case.trial['bilateral_unsupported_physics_substeps']}"
  )
  cfg = rb.make_roll_boundary_env_cfg(
    R0C_HEIGHTS_M, diag.R0C_SYNC_ENVS_PER_HEIGHT,
  )
  cfg.viewer.env_idx = args.env_id
  cfg.viewer.max_extra_envs = 0
  cfg.viewer.distance = 1.4
  cfg.viewer.azimuth = 90.0
  cfg.viewer.elevation = 0.0
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  observer = SupportLossObserver(env, env_id=args.env_id)
  try:
    policy = R0cViewerPolicy(env, case=case, observer=observer)
    viewer = R0cSyncViserViewer(
      env, policy, case=case, observer=observer,
    )
    print("[r0c-sync-viewer] open the Viser URL, then click Play.")
    viewer.run()
  finally:
    observer.close()
    env.close()


if __name__ == "__main__":
  main()