"""Position-indexed classical posture schedules."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

CONTROL_DT_S = 0.02
POSTURE_HEIGHT_SLEW_RATE_MPS = 0.01215
POSTURE_PITCH_SLEW_RATE_RADPS = 0.07755
REGISTERED_HEIGHTS_M = (0.2907321708, 0.3092089487, 0.3276857266)
REGISTERED_PITCH_RANGE_RAD = (-0.032, 0.032)
SCHEDULE_END_DISTANCES_M = (0.030, 0.015, 0.0)
START_POSES = (
  ("a", REGISTERED_HEIGHTS_M[0], -0.032),
  ("b", REGISTERED_HEIGHTS_M[0], -0.016),
)
CLIMB_POSES = (
  ("c", REGISTERED_HEIGHTS_M[1], 0.032),
  ("d", REGISTERED_HEIGHTS_M[2], 0.032),
)


@dataclass(frozen=True)
class RollPoseSchedule:
  """One frozen start-to-climb posture instruction."""

  name: str
  start_height_m: float
  start_pitch_rad: float
  climb_height_m: float
  climb_pitch_rad: float
  end_distance_to_riser_m: float

  def __post_init__(self) -> None:
    values = (
      self.start_height_m, self.start_pitch_rad,
      self.climb_height_m, self.climb_pitch_rad,
      self.end_distance_to_riser_m,
    )
    if not self.name or any(not math.isfinite(value) for value in values):
      raise ValueError("Roll-pose schedule values must be named and finite.")
    height_min, height_max = REGISTERED_HEIGHTS_M[0], REGISTERED_HEIGHTS_M[-1]
    pitch_min, pitch_max = REGISTERED_PITCH_RANGE_RAD
    if any(not height_min <= value <= height_max for value in (
      self.start_height_m, self.climb_height_m,
    )):
      raise ValueError("Roll-pose height lies outside the registered envelope.")
    if any(not pitch_min <= value <= pitch_max for value in (
      self.start_pitch_rad, self.climb_pitch_rad,
    )):
      raise ValueError("Roll-pose pitch lies outside the registered envelope.")
    if not 0.0 <= self.end_distance_to_riser_m <= 0.030:
      raise ValueError("Roll-pose completion distance lies outside [0, 0.03] m.")

  def to_dict(self) -> dict[str, float | str]:
    return {
      "name": self.name,
      "start_height_m": self.start_height_m,
      "start_pitch_rad": self.start_pitch_rad,
      "climb_height_m": self.climb_height_m,
      "climb_pitch_rad": self.climb_pitch_rad,
      "end_distance_to_riser_m": self.end_distance_to_riser_m,
    }


@dataclass
class RollPoseScheduleState:
  drive_started: torch.Tensor
  drive_start_x_m: torch.Tensor
  max_forward_progress_m: torch.Tensor
  applied_height_m: torch.Tensor
  applied_pitch_rad: torch.Tensor


@dataclass(frozen=True)
class RollPoseScheduleOutput:
  alpha: torch.Tensor
  desired_height_m: torch.Tensor
  desired_pitch_rad: torch.Tensor
  applied_height_m: torch.Tensor
  applied_pitch_rad: torch.Tensor
  distance_to_riser_m: torch.Tensor


def roll_pose_schedule_candidates() -> tuple[RollPoseSchedule, ...]:
  """Return the fixed 2 x 2 x 3 diagnostic candidate grid."""

  candidates = []
  for start_name, start_height, start_pitch in START_POSES:
    for climb_name, climb_height, climb_pitch in CLIMB_POSES:
      for end_distance in SCHEDULE_END_DISTANCES_M:
        end_mm = round(end_distance * 1_000)
        candidates.append(RollPoseSchedule(
          name=f"roll_pose_s{start_name}_c{climb_name}_d{end_mm:03d}mm",
          start_height_m=start_height,
          start_pitch_rad=start_pitch,
          climb_height_m=climb_height,
          climb_pitch_rad=climb_pitch,
          end_distance_to_riser_m=end_distance,
        ))
  result = tuple(candidates)
  if len(result) != 12 or len({candidate.name for candidate in result}) != 12:
    raise RuntimeError("The frozen schedule grid must contain 12 unique candidates.")
  return result


def make_roll_pose_schedule_state(
  schedule: RollPoseSchedule,
  root_x_m: torch.Tensor,
) -> RollPoseScheduleState:
  """Initialize a schedule in its settle posture without starting drive."""

  if root_x_m.ndim != 1:
    raise ValueError("Roll-pose root_x_m must be a one-dimensional tensor.")
  return RollPoseScheduleState(
    drive_started=torch.zeros_like(root_x_m, dtype=torch.bool),
    drive_start_x_m=root_x_m.detach().clone(),
    max_forward_progress_m=torch.zeros_like(root_x_m),
    applied_height_m=torch.full_like(root_x_m, schedule.start_height_m),
    applied_pitch_rad=torch.full_like(root_x_m, schedule.start_pitch_rad),
  )


def _validate_step_inputs(
  state: RollPoseScheduleState,
  root_x_m: torch.Tensor,
  face_x_m: torch.Tensor,
  active_mask: torch.Tensor,
) -> None:
  shape = root_x_m.shape
  if root_x_m.ndim != 1 or face_x_m.shape != shape or active_mask.shape != shape:
    raise ValueError("Roll-pose step tensors must share one-dimensional shape.")
  if active_mask.dtype != torch.bool:
    raise TypeError("Roll-pose active_mask must be boolean.")
  for value in (
    state.drive_started,
    state.drive_start_x_m,
    state.max_forward_progress_m,
    state.applied_height_m,
    state.applied_pitch_rad,
  ):
    if value.shape != shape:
      raise ValueError("Roll-pose state shape drifted.")


def roll_pose_schedule_step(
  schedule: RollPoseSchedule,
  state: RollPoseScheduleState,
  *,
  root_x_m: torch.Tensor,
  face_x_m: torch.Tensor,
  active_mask: torch.Tensor,
  drive_active: bool,
  dt: float = CONTROL_DT_S,
) -> RollPoseScheduleOutput:
  """Advance one tick while preventing pose rewind when the robot rolls back."""

  _validate_step_inputs(state, root_x_m, face_x_m, active_mask)
  if not math.isfinite(dt) or dt <= 0.0:
    raise ValueError("Roll-pose dt must be finite and positive.")
  if drive_active:
    newly_started = active_mask & ~state.drive_started
    state.drive_start_x_m.copy_(torch.where(
      newly_started, root_x_m, state.drive_start_x_m,
    ))
    state.max_forward_progress_m.copy_(torch.where(
      newly_started,
      torch.zeros_like(state.max_forward_progress_m),
      state.max_forward_progress_m,
    ))
    state.drive_started.logical_or_(newly_started)
    raw_progress = root_x_m - state.drive_start_x_m
    state.max_forward_progress_m.copy_(torch.where(
      active_mask & state.drive_started,
      torch.maximum(state.max_forward_progress_m, raw_progress),
      state.max_forward_progress_m,
    ))

  required_progress = (
    face_x_m - schedule.end_distance_to_riser_m - state.drive_start_x_m
  )
  if bool(torch.any(state.drive_started & (required_progress <= 0.0))):
    raise ValueError("Roll-pose transition endpoint must remain ahead of drive start.")
  safe_required = torch.clamp(required_progress, min=torch.finfo(root_x_m.dtype).eps)
  alpha = torch.where(
    state.drive_started,
    torch.clamp(state.max_forward_progress_m / safe_required, 0.0, 1.0),
    torch.zeros_like(root_x_m),
  )
  desired_height = schedule.start_height_m + alpha * (
    schedule.climb_height_m - schedule.start_height_m
  )
  desired_pitch = schedule.start_pitch_rad + alpha * (
    schedule.climb_pitch_rad - schedule.start_pitch_rad
  )

  height_step = POSTURE_HEIGHT_SLEW_RATE_MPS * dt
  pitch_step = POSTURE_PITCH_SLEW_RATE_RADPS * dt
  next_height = state.applied_height_m + torch.clamp(
    desired_height - state.applied_height_m, -height_step, height_step,
  )
  next_pitch = state.applied_pitch_rad + torch.clamp(
    desired_pitch - state.applied_pitch_rad, -pitch_step, pitch_step,
  )
  state.applied_height_m.copy_(torch.where(
    active_mask, next_height, state.applied_height_m,
  ))
  state.applied_pitch_rad.copy_(torch.where(
    active_mask, next_pitch, state.applied_pitch_rad,
  ))
  return RollPoseScheduleOutput(
    alpha=alpha,
    desired_height_m=desired_height,
    desired_pitch_rad=desired_pitch,
    applied_height_m=state.applied_height_m.clone(),
    applied_pitch_rad=state.applied_pitch_rad.clone(),
    distance_to_riser_m=face_x_m - root_x_m,
  )
