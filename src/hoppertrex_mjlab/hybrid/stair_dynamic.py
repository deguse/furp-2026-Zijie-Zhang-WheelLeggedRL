"""CTBC-inspired dynamic stair primitives for HopperTrex Hybrid v3.

This module is deliberately independent of MjLab.  It is the scalar reference
used by deployment, artifact validation, CEM tuning, and Torch parity tests.
The archived Hybrid-v2 stair controller remains in :mod:`stair_classical`;
this module reuses its phase values and CEM score rather than reviving the
falsified three-channel contact detector.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any

from .stair_classical import CONTROL_DT_S, PHASE_COUNT, StairPhase

DYNAMIC_STAIR_TASK_ID = "HopperTrex-Hybrid-v3-StairDynamic"
DYNAMIC_MANEUVER_ARTIFACT_TYPE = "dynamic_stair_maneuver"
DYNAMIC_MANEUVER_SCHEMA_VERSION = 1
DYNAMIC_STAIR_CONTRACT_SCHEMA_VERSION = 1
DYNAMIC_STAIR_APPROACH_VX_MPS = 0.07
DYNAMIC_STAIR_FEEDFORWARD_LIMIT_RAD = 0.070
DYNAMIC_STAIR_PPO_LEG_SCALE_RAD = 0.035
DYNAMIC_STAIR_LIFT_DURATION_S = 0.6
DYNAMIC_STAIR_RECOVER_DURATION_S = 0.5
DYNAMIC_STAIR_PRELOAD_DURATION_S = 0.4
DYNAMIC_STAIR_APPROACH_DURATION_S = 0.2
DYNAMIC_STAIR_CONTACT_TIMEOUT_S = 5.0
DYNAMIC_STAIR_TRAIL_CONTACT_TIMEOUT_S = 1.5
DYNAMIC_STAIR_CROSS_TIMEOUT_S = 1.5
DYNAMIC_STAIR_FIRST_CROSS_M = 0.40
DYNAMIC_STAIR_NEXT_CROSS_M = 0.30
DYNAMIC_STAIR_RECOVER_STABLE_STEPS = 25
DYNAMIC_STAIR_TRIGGER_FORCE_N = 18.0
DYNAMIC_STAIR_TRIGGER_WINDOW = 3
DYNAMIC_STAIR_TIME_EPS_S = 1.0e-7
DYNAMIC_STAIR_CEM_LOWER = (0.0, 0.02, 0.0, 0.0)
DYNAMIC_STAIR_CEM_UPPER = (0.07, 0.07, 0.4, 2.0)
DYNAMIC_STAIR_CEM_POPULATION = 32
DYNAMIC_STAIR_CEM_ITERATIONS = 5
DYNAMIC_STAIR_CEM_SEED = 1
DYNAMIC_STAIR_CEM_REPLICATES = 8
DYNAMIC_MANEUVER_REQUIRED_BINDINGS = (
  "git_sha",
  "stage5_checkpoint_sha256",
  "stage5_formal_gate_sha256",
  "per_wheel_trigger_qualification_sha256",
  "controller_gain_hash",
  "calibration_hash",
  "yaw_calibration_hash",
  "posture_map_hash",
  "posture_artifact_hash",
  "station_calibration_hash",
)

# Numerical Jacobian directions measured from the checked-in HopperTrex MJCF
# at INIT_JOINT_POS.  Each two-joint vector is normalized so max(abs(v)) == 1.
# Joint order everywhere in the Hybrid action is
# (left thigh, right thigh, left knee, right knee).
LEFT_FORWARD_BASIS = (1.0, 0.74800799)
RIGHT_FORWARD_BASIS = (-1.0, -0.74895033)
LEFT_LIFT_BASIS = (-0.67582901, 1.0)
RIGHT_LIFT_BASIS = (0.67825934, -1.0)


class LeadSide(IntEnum):
  """Wheel identity used by the dynamic stair reflex."""

  NONE = 0
  LEFT = 1
  RIGHT = 2


class DynamicLiftMode(StrEnum):
  """Nominal leg-motion family screened before PPO training."""

  SYNCHRONIZED = "synchronized"
  ALTERNATING = "alternating"


class StairTraversalMode(StrEnum):
  """Observed per-riser behavior, never a height classifier."""

  NONE = "NONE"
  ROLL = "ROLL"
  DYNAMIC = "DYNAMIC"
  ABORT = "ABORT"


@dataclass(frozen=True)
class DynamicStairManeuver:
  """Frozen nominal stair instruction consumed by simulation and deployment."""

  lift_mode: DynamicLiftMode
  split_amplitude_rad: float
  lift_amplitude_rad: float
  trailing_delay_s: float
  drive_feedforward_radps: float
  split_basis_left: tuple[float, float] = LEFT_FORWARD_BASIS
  split_basis_right: tuple[float, float] = RIGHT_FORWARD_BASIS
  lift_basis_left: tuple[float, float] = LEFT_LIFT_BASIS
  lift_basis_right: tuple[float, float] = RIGHT_LIFT_BASIS
  approach_vx: float = DYNAMIC_STAIR_APPROACH_VX_MPS
  approach_duration_s: float = DYNAMIC_STAIR_APPROACH_DURATION_S
  preload_duration_s: float = DYNAMIC_STAIR_PRELOAD_DURATION_S
  lift_duration_s: float = DYNAMIC_STAIR_LIFT_DURATION_S
  recover_duration_s: float = DYNAMIC_STAIR_RECOVER_DURATION_S
  contact_timeout_s: float = DYNAMIC_STAIR_CONTACT_TIMEOUT_S
  trail_contact_timeout_s: float = DYNAMIC_STAIR_TRAIL_CONTACT_TIMEOUT_S
  cross_timeout_s: float = DYNAMIC_STAIR_CROSS_TIMEOUT_S
  first_cross_m: float = DYNAMIC_STAIR_FIRST_CROSS_M
  next_cross_m: float = DYNAMIC_STAIR_NEXT_CROSS_M
  recover_stable_steps: int = DYNAMIC_STAIR_RECOVER_STABLE_STEPS
  trigger_force_n: float = DYNAMIC_STAIR_TRIGGER_FORCE_N
  trigger_window: int = DYNAMIC_STAIR_TRIGGER_WINDOW
  maneuver_hash: str = ""
  bindings: Mapping[str, str] | None = None
  source: str = "memory"

  def __post_init__(self) -> None:
    scalar_bounds = {
      "split_amplitude_rad": (self.split_amplitude_rad, 0.0, 0.07),
      "lift_amplitude_rad": (self.lift_amplitude_rad, 0.02, 0.07),
      "trailing_delay_s": (self.trailing_delay_s, 0.0, 0.4),
      "drive_feedforward_radps": (self.drive_feedforward_radps, 0.0, 2.0),
    }
    for name, (value, lower, upper) in scalar_bounds.items():
      if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must stay within [{lower}, {upper}].")
    positive = {
      "approach_vx": self.approach_vx,
      "approach_duration_s": self.approach_duration_s,
      "preload_duration_s": self.preload_duration_s,
      "lift_duration_s": self.lift_duration_s,
      "recover_duration_s": self.recover_duration_s,
      "contact_timeout_s": self.contact_timeout_s,
      "trail_contact_timeout_s": self.trail_contact_timeout_s,
      "cross_timeout_s": self.cross_timeout_s,
      "first_cross_m": self.first_cross_m,
      "next_cross_m": self.next_cross_m,
      "trigger_force_n": self.trigger_force_n,
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in positive.values()):
      raise ValueError("Dynamic stair timing, progress, and trigger values must be positive.")
    if self.recover_stable_steps < 1 or self.trigger_window < 1:
      raise ValueError("Dynamic stair counters must be positive.")
    for name in (
      "split_basis_left",
      "split_basis_right",
      "lift_basis_left",
      "lift_basis_right",
    ):
      basis = tuple(float(value) for value in getattr(self, name))
      if len(basis) != 2 or any(not math.isfinite(value) for value in basis):
        raise ValueError(f"{name} must contain two finite values.")
      if not math.isclose(max(abs(value) for value in basis), 1.0, abs_tol=1e-8):
        raise ValueError(f"{name} must be max-norm normalized.")
    if self.maneuver_hash and (
      len(self.maneuver_hash) != 64
      or any(char not in "0123456789abcdef" for char in self.maneuver_hash)
    ):
      raise ValueError("Dynamic stair maneuver_hash must be lowercase SHA256.")
    if self.bindings is not None and (
      not self.bindings
      or any(not isinstance(key, str) or not isinstance(value, str) or not value for key, value in self.bindings.items())
    ):
      raise ValueError("Dynamic stair maneuver bindings must be non-empty strings.")

  @property
  def cross_distance_m(self) -> tuple[float, float]:
    return (self.first_cross_m, self.next_cross_m)


@dataclass(frozen=True)
class DynamicStairState:
  """Scalar reference state for one robot."""

  phase: StairPhase = StairPhase.IDLE
  phase_elapsed_s: float = 0.0
  step_progress_m: float = 0.0
  step_index: int = 0
  preferred_side: LeadSide = LeadSide.LEFT
  lead_side: LeadSide = LeadSide.NONE
  left_trigger_streak: int = 0
  right_trigger_streak: int = 0
  left_loaded_contact: bool = False
  right_loaded_contact: bool = False
  # Phase-local timestamp of the trailing wheel's first qualified contact.
  # ``None`` means the trailing wheel has not yet triggered.  The frozen
  # trailing delay is measured from this edge, not from lead-lift entry.
  trail_contact_elapsed_s: float | None = None
  recover_stable_steps: int = 0
  traversal_mode: StairTraversalMode = StairTraversalMode.NONE
  abort_reason: str | None = None


@dataclass(frozen=True)
class DynamicStairSensors:
  """Deployable inputs used by the scalar FSM."""

  progress_delta_m: float
  left_force_n: float
  right_force_n: float
  stable: bool = False
  non_wheel_contact: bool = False
  actuator_limit: bool = False
  orientation_limit: bool = False


@dataclass(frozen=True)
class DynamicStairTargets:
  """Nominal outputs composed underneath the six-dimensional PPO feedback."""

  vx: float
  drive_feedforward_radps: float
  leg_feedforward: tuple[float, float, float, float]
  phase: StairPhase
  step_index: int
  lead_side: LeadSide
  left_loaded_contact: bool
  right_loaded_contact: bool
  traversal_mode: StairTraversalMode
  abort_reason: str | None

  def phase_one_hot(self) -> tuple[float, ...]:
    return tuple(float(index == int(self.phase)) for index in range(PHASE_COUNT))

  def lead_side_one_hot(self) -> tuple[float, float]:
    return (
      float(self.lead_side == LeadSide.LEFT),
      float(self.lead_side == LeadSide.RIGHT),
    )


def half_cosine_bump(elapsed_s: float, duration_s: float) -> float:
  """Unit bump with zero value and velocity at both endpoints."""

  if not math.isfinite(elapsed_s) or not math.isfinite(duration_s) or duration_s <= 0.0:
    raise ValueError("Half-cosine bump inputs must be finite and duration positive.")
  if elapsed_s <= 0.0 or elapsed_s >= duration_s:
    return 0.0
  return 0.5 * (1.0 - math.cos(2.0 * math.pi * elapsed_s / duration_s))


def update_loaded_contact(
  *,
  streak: int,
  latched: bool,
  force_n: float,
  threshold_n: float,
  window: int,
  active: bool,
) -> tuple[int, bool]:
  """Advance one side of the candidate 18 N x 3 loaded-contact rule."""

  if streak < 0 or window < 1:
    raise ValueError("Loaded-contact counters are invalid.")
  if not all(math.isfinite(value) for value in (force_n, threshold_n)):
    raise ValueError("Loaded-contact forces must be finite.")
  if threshold_n <= 0.0:
    raise ValueError("Loaded-contact threshold must be positive.")
  if not active:
    return 0, False
  hit = force_n >= threshold_n
  next_streak = min(window, streak + 1) if hit else 0
  return next_streak, latched or next_streak >= window


def choose_lead_side(
  *,
  left_loaded: bool,
  right_loaded: bool,
  left_force_n: float,
  right_force_n: float,
  preferred_side: LeadSide,
) -> LeadSide:
  """Choose the first loaded wheel with a deterministic simultaneous tie-break."""

  if left_loaded and not right_loaded:
    return LeadSide.LEFT
  if right_loaded and not left_loaded:
    return LeadSide.RIGHT
  if not left_loaded and not right_loaded:
    return LeadSide.NONE
  if left_force_n > right_force_n:
    return LeadSide.LEFT
  if right_force_n > left_force_n:
    return LeadSide.RIGHT
  if preferred_side not in (LeadSide.LEFT, LeadSide.RIGHT):
    raise ValueError("A simultaneous tie requires a concrete preferred side.")
  return preferred_side


def _write_side(
  target: list[float],
  side: LeadSide,
  basis: Sequence[float],
  amplitude: float,
) -> None:
  if side == LeadSide.LEFT:
    target[0] += amplitude * float(basis[0])
    target[2] += amplitude * float(basis[1])
  elif side == LeadSide.RIGHT:
    target[1] += amplitude * float(basis[0])
    target[3] += amplitude * float(basis[1])
  else:
    raise ValueError("Leg feedforward requires LEFT or RIGHT.")


def dynamic_leg_feedforward(
  maneuver: DynamicStairManeuver,
  state: DynamicStairState,
) -> tuple[float, float, float, float]:
  """Compose pre-split and lift offsets for the current phase."""

  if state.phase in (StairPhase.IDLE, StairPhase.DONE, StairPhase.ABORT):
    return (0.0, 0.0, 0.0, 0.0)
  split_fraction = 0.0
  if state.phase == StairPhase.PRELOAD:
    split_fraction = min(1.0, state.phase_elapsed_s / maneuver.preload_duration_s)
  elif state.phase in (
    StairPhase.CONTACT_WAIT,
    StairPhase.LEAD_LIFT,
    StairPhase.TRAIL_LIFT,
  ):
    split_fraction = 1.0
  elif state.phase == StairPhase.RECOVER:
    split_fraction = max(0.0, 1.0 - state.phase_elapsed_s / maneuver.recover_duration_s)

  lead = state.lead_side if state.lead_side != LeadSide.NONE else state.preferred_side
  trail = LeadSide.RIGHT if lead == LeadSide.LEFT else LeadSide.LEFT
  target = [0.0, 0.0, 0.0, 0.0]
  if split_fraction:
    lead_basis = maneuver.split_basis_left if lead == LeadSide.LEFT else maneuver.split_basis_right
    trail_basis = maneuver.split_basis_left if trail == LeadSide.LEFT else maneuver.split_basis_right
    _write_side(target, lead, lead_basis, maneuver.split_amplitude_rad * split_fraction)
    _write_side(target, trail, trail_basis, -maneuver.split_amplitude_rad * split_fraction)

  if state.phase == StairPhase.LEAD_LIFT:
    lift = half_cosine_bump(state.phase_elapsed_s, maneuver.lift_duration_s)
    if maneuver.lift_mode == DynamicLiftMode.SYNCHRONIZED:
      for side in (LeadSide.LEFT, LeadSide.RIGHT):
        basis = maneuver.lift_basis_left if side == LeadSide.LEFT else maneuver.lift_basis_right
        _write_side(target, side, basis, maneuver.lift_amplitude_rad * lift)
    else:
      basis = maneuver.lift_basis_left if lead == LeadSide.LEFT else maneuver.lift_basis_right
      _write_side(target, lead, basis, maneuver.lift_amplitude_rad * lift)
  elif (
    state.phase == StairPhase.TRAIL_LIFT
    and maneuver.lift_mode == DynamicLiftMode.ALTERNATING
  ):
    lift = half_cosine_bump(state.phase_elapsed_s, maneuver.lift_duration_s)
    basis = maneuver.lift_basis_left if trail == LeadSide.LEFT else maneuver.lift_basis_right
    _write_side(target, trail, basis, maneuver.lift_amplitude_rad * lift)

  bounded = tuple(
    max(-DYNAMIC_STAIR_FEEDFORWARD_LIMIT_RAD, min(DYNAMIC_STAIR_FEEDFORWARD_LIMIT_RAD, value))
    for value in target
  )
  if any(not math.isfinite(value) for value in bounded):
    raise ValueError("Composed dynamic leg feedforward is non-finite.")
  return bounded  # type: ignore[return-value]


def _targets(
  maneuver: DynamicStairManeuver,
  state: DynamicStairState,
) -> DynamicStairTargets:
  active = state.phase not in (StairPhase.IDLE, StairPhase.DONE, StairPhase.ABORT)
  drive = maneuver.drive_feedforward_radps if state.phase in (
    StairPhase.LEAD_LIFT,
    StairPhase.TRAIL_LIFT,
  ) else 0.0
  if state.phase == StairPhase.RECOVER:
    drive = maneuver.drive_feedforward_radps * max(
      0.0, 1.0 - state.phase_elapsed_s / maneuver.recover_duration_s
    )
  return DynamicStairTargets(
    vx=maneuver.approach_vx if active else 0.0,
    drive_feedforward_radps=drive,
    leg_feedforward=dynamic_leg_feedforward(maneuver, state),
    phase=state.phase,
    step_index=state.step_index,
    lead_side=state.lead_side,
    left_loaded_contact=state.left_loaded_contact,
    right_loaded_contact=state.right_loaded_contact,
    traversal_mode=state.traversal_mode,
    abort_reason=state.abort_reason,
  )


def dynamic_stair_step(
  maneuver: DynamicStairManeuver,
  state: DynamicStairState,
  sensors: DynamicStairSensors,
  *,
  stair_request: bool,
  dt: float = CONTROL_DT_S,
) -> tuple[DynamicStairTargets, DynamicStairState]:
  """Advance one scalar CTBC-inspired stair-control tick."""

  values = (sensors.progress_delta_m, sensors.left_force_n, sensors.right_force_n, dt)
  if any(not math.isfinite(value) for value in values) or dt <= 0.0:
    raise ValueError("Dynamic stair step inputs must be finite and dt positive.")
  if not stair_request:
    reset = DynamicStairState(preferred_side=state.preferred_side)
    return _targets(maneuver, reset), reset

  next_state = replace(
    state,
    phase_elapsed_s=state.phase_elapsed_s + dt,
    step_progress_m=state.step_progress_m + sensors.progress_delta_m,
  )
  if state.phase in (StairPhase.IDLE, StairPhase.DONE):
    next_state = DynamicStairState(
      phase=StairPhase.APPROACH,
      preferred_side=state.preferred_side,
      step_index=state.step_index,
    )

  active_trigger = next_state.phase in (
    StairPhase.APPROACH,
    StairPhase.PRELOAD,
    StairPhase.CONTACT_WAIT,
    StairPhase.LEAD_LIFT,
  )
  left_streak, left_loaded = update_loaded_contact(
    streak=next_state.left_trigger_streak,
    latched=next_state.left_loaded_contact,
    force_n=sensors.left_force_n,
    threshold_n=maneuver.trigger_force_n,
    window=maneuver.trigger_window,
    active=active_trigger,
  )
  right_streak, right_loaded = update_loaded_contact(
    streak=next_state.right_trigger_streak,
    latched=next_state.right_loaded_contact,
    force_n=sensors.right_force_n,
    threshold_n=maneuver.trigger_force_n,
    window=maneuver.trigger_window,
    active=active_trigger,
  )
  if active_trigger:
    next_state = replace(
      next_state,
      left_trigger_streak=left_streak,
      right_trigger_streak=right_streak,
      left_loaded_contact=left_loaded,
      right_loaded_contact=right_loaded,
    )

  abort_reason = None
  if sensors.non_wheel_contact:
    abort_reason = "non_wheel_contact"
  elif sensors.actuator_limit:
    abort_reason = "actuator_limit"
  elif sensors.orientation_limit:
    abort_reason = "orientation_limit"
  elif next_state.step_progress_m < -0.10:
    abort_reason = "backward_progress"
  if abort_reason is not None:
    next_state = replace(
      next_state,
      phase=StairPhase.ABORT,
      phase_elapsed_s=0.0,
      traversal_mode=StairTraversalMode.ABORT,
      abort_reason=abort_reason,
    )
    return _targets(maneuver, next_state), next_state

  phase = next_state.phase
  crossed = next_state.step_progress_m >= (
    maneuver.first_cross_m if next_state.step_index == 0 else maneuver.next_cross_m
  )
  if phase in (StairPhase.APPROACH, StairPhase.PRELOAD, StairPhase.CONTACT_WAIT):
    lead = choose_lead_side(
      left_loaded=next_state.left_loaded_contact,
      right_loaded=next_state.right_loaded_contact,
      left_force_n=sensors.left_force_n,
      right_force_n=sensors.right_force_n,
      preferred_side=next_state.preferred_side,
    )
    if crossed and lead == LeadSide.NONE:
      next_state = replace(
        next_state,
        phase=StairPhase.RECOVER,
        phase_elapsed_s=0.0,
        traversal_mode=StairTraversalMode.ROLL,
      )
    elif lead != LeadSide.NONE:
      trail_loaded = (
        next_state.right_loaded_contact
        if lead == LeadSide.LEFT
        else next_state.left_loaded_contact
      )
      next_state = replace(
        next_state,
        phase=StairPhase.LEAD_LIFT,
        phase_elapsed_s=0.0,
        lead_side=lead,
        trail_contact_elapsed_s=0.0 if trail_loaded else None,
        traversal_mode=StairTraversalMode.DYNAMIC,
      )
    elif phase == StairPhase.APPROACH and next_state.phase_elapsed_s + DYNAMIC_STAIR_TIME_EPS_S >= maneuver.approach_duration_s:
      next_state = replace(next_state, phase=StairPhase.PRELOAD, phase_elapsed_s=0.0)
    elif phase == StairPhase.PRELOAD and next_state.phase_elapsed_s + DYNAMIC_STAIR_TIME_EPS_S >= maneuver.preload_duration_s:
      next_state = replace(next_state, phase=StairPhase.CONTACT_WAIT, phase_elapsed_s=0.0)
    elif phase == StairPhase.CONTACT_WAIT and next_state.phase_elapsed_s + DYNAMIC_STAIR_TIME_EPS_S >= maneuver.contact_timeout_s:
      next_state = replace(
        next_state,
        phase=StairPhase.ABORT,
        phase_elapsed_s=0.0,
        traversal_mode=StairTraversalMode.ABORT,
        abort_reason="contact_timeout",
      )
  elif phase == StairPhase.LEAD_LIFT:
    if maneuver.lift_mode == DynamicLiftMode.SYNCHRONIZED:
      if next_state.phase_elapsed_s + DYNAMIC_STAIR_TIME_EPS_S >= maneuver.lift_duration_s:
        next_state = replace(next_state, phase=StairPhase.TRAIL_LIFT, phase_elapsed_s=0.0)
    else:
      trail_loaded = (
        next_state.right_loaded_contact
        if next_state.lead_side == LeadSide.LEFT
        else next_state.left_loaded_contact
      )
      trail_edge = next_state.trail_contact_elapsed_s
      if trail_loaded and trail_edge is None:
        trail_edge = next_state.phase_elapsed_s
        next_state = replace(next_state, trail_contact_elapsed_s=trail_edge)
      ready_time = (
        math.inf
        if trail_edge is None
        else max(
          maneuver.lift_duration_s,
          trail_edge + maneuver.trailing_delay_s,
        )
      )
      if next_state.phase_elapsed_s + DYNAMIC_STAIR_TIME_EPS_S >= ready_time:
        next_state = replace(next_state, phase=StairPhase.TRAIL_LIFT, phase_elapsed_s=0.0)
      elif (
        trail_edge is None
        and next_state.phase_elapsed_s + DYNAMIC_STAIR_TIME_EPS_S
        >= maneuver.lift_duration_s + maneuver.trail_contact_timeout_s
      ):
        next_state = replace(
          next_state,
          phase=StairPhase.ABORT,
          phase_elapsed_s=0.0,
          traversal_mode=StairTraversalMode.ABORT,
          abort_reason="trail_contact_timeout",
        )
  elif phase == StairPhase.TRAIL_LIFT:
    required_lift = (
      maneuver.lift_duration_s
      if maneuver.lift_mode == DynamicLiftMode.ALTERNATING
      else 0.0
    )
    if crossed and next_state.phase_elapsed_s + DYNAMIC_STAIR_TIME_EPS_S >= required_lift:
      next_state = replace(next_state, phase=StairPhase.RECOVER, phase_elapsed_s=0.0)
    elif next_state.phase_elapsed_s + DYNAMIC_STAIR_TIME_EPS_S >= required_lift + maneuver.cross_timeout_s:
      next_state = replace(
        next_state,
        phase=StairPhase.ABORT,
        phase_elapsed_s=0.0,
        traversal_mode=StairTraversalMode.ABORT,
        abort_reason="cross_timeout",
      )
  elif phase == StairPhase.RECOVER:
    stable_steps = next_state.recover_stable_steps + 1 if sensors.stable else 0
    next_state = replace(next_state, recover_stable_steps=stable_steps)
    if (
      next_state.phase_elapsed_s + DYNAMIC_STAIR_TIME_EPS_S >= maneuver.recover_duration_s
      and stable_steps >= maneuver.recover_stable_steps
    ):
      preferred = (
        LeadSide.RIGHT
        if next_state.preferred_side == LeadSide.LEFT
        else LeadSide.LEFT
      )
      next_state = DynamicStairState(
        phase=StairPhase.APPROACH,
        step_index=next_state.step_index + 1,
        preferred_side=preferred,
      )

  return _targets(maneuver, next_state), next_state


def _canonical_hash(payload: Mapping[str, Any]) -> str:
  body = {key: value for key, value in payload.items() if key != "maneuver_hash"}
  encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
  return hashlib.sha256(encoded).hexdigest()


def validate_dynamic_maneuver_bindings(
  bindings: Mapping[str, str],
) -> dict[str, str]:
  """Validate exact Git, Stage5, gate, trigger, and classical bindings."""

  if set(bindings) != set(DYNAMIC_MANEUVER_REQUIRED_BINDINGS):
    raise ValueError("Dynamic stair maneuver binding schema drifted.")
  normalized = {str(key): str(value) for key, value in bindings.items()}
  for name, value in normalized.items():
    expected_length = 40 if name == "git_sha" else 64
    if len(value) != expected_length or any(
      char not in "0123456789abcdef" for char in value
    ):
      raise ValueError(
        f"Dynamic stair maneuver binding {name} must be lowercase hex."
      )
  return normalized


def dynamic_maneuver_payload(
  maneuver: DynamicStairManeuver,
  *,
  bindings: Mapping[str, str],
) -> dict[str, Any]:
  """Serialize one fully bound maneuver and compute its canonical hash."""

  payload: dict[str, Any] = {
    "schema_version": DYNAMIC_MANEUVER_SCHEMA_VERSION,
    "artifact_type": DYNAMIC_MANEUVER_ARTIFACT_TYPE,
    "task": DYNAMIC_STAIR_TASK_ID,
    "known_step_height": None,
    "supported_terrain": "regular_uniform_front",
    "parameters": {
      "lift_mode": maneuver.lift_mode.value,
      "split_amplitude_rad": maneuver.split_amplitude_rad,
      "lift_amplitude_rad": maneuver.lift_amplitude_rad,
      "trailing_delay_s": maneuver.trailing_delay_s,
      "drive_feedforward_radps": maneuver.drive_feedforward_radps,
      "approach_vx": maneuver.approach_vx,
      "approach_duration_s": maneuver.approach_duration_s,
      "preload_duration_s": maneuver.preload_duration_s,
      "lift_duration_s": maneuver.lift_duration_s,
      "recover_duration_s": maneuver.recover_duration_s,
      "contact_timeout_s": maneuver.contact_timeout_s,
      "trail_contact_timeout_s": maneuver.trail_contact_timeout_s,
      "cross_timeout_s": maneuver.cross_timeout_s,
      "first_cross_m": maneuver.first_cross_m,
      "next_cross_m": maneuver.next_cross_m,
      "recover_stable_steps": maneuver.recover_stable_steps,
    },
    "trigger": {
      "metric": "abs(F0*nx)",
      "threshold_n": maneuver.trigger_force_n,
      "window": maneuver.trigger_window,
      "qualification": "per-wheel-live-required",
    },
    "kinematic_bases": {
      "split_left": list(maneuver.split_basis_left),
      "split_right": list(maneuver.split_basis_right),
      "lift_left": list(maneuver.lift_basis_left),
      "lift_right": list(maneuver.lift_basis_right),
    },
    "cem": {
      "parameter_order": [
        "split_amplitude_rad",
        "lift_amplitude_rad",
        "trailing_delay_s",
        "drive_feedforward_radps",
      ],
      "lower": list(DYNAMIC_STAIR_CEM_LOWER),
      "upper": list(DYNAMIC_STAIR_CEM_UPPER),
      "population": DYNAMIC_STAIR_CEM_POPULATION,
      "iterations": DYNAMIC_STAIR_CEM_ITERATIONS,
      "seed": DYNAMIC_STAIR_CEM_SEED,
      "replicates": DYNAMIC_STAIR_CEM_REPLICATES,
      "feedback_policy": "stage5_seed1_100_selected_deterministic_mean",
      "observation_adapter": "34_to_52_zero_appended_columns",
    },
    "bindings": validate_dynamic_maneuver_bindings(bindings),
  }
  payload["maneuver_hash"] = _canonical_hash(payload)
  return payload


def parse_dynamic_maneuver(
  payload: Mapping[str, Any],
  *,
  source: str = "memory",
) -> DynamicStairManeuver:
  """Fail-closed parser for a CEM-selected v3 maneuver artifact."""

  if payload.get("schema_version") != DYNAMIC_MANEUVER_SCHEMA_VERSION:
    raise ValueError("Dynamic stair maneuver schema_version is unsupported.")
  if payload.get("artifact_type") != DYNAMIC_MANEUVER_ARTIFACT_TYPE:
    raise ValueError("Dynamic stair maneuver artifact_type is invalid.")
  if payload.get("task") != DYNAMIC_STAIR_TASK_ID:
    raise ValueError("Dynamic stair maneuver task binding is invalid.")
  if payload.get("known_step_height") is not None:
    raise ValueError("Dynamic stair maneuver may not depend on known step height.")
  if payload.get("supported_terrain") != "regular_uniform_front":
    raise ValueError("Dynamic stair maneuver terrain envelope drifted.")
  if payload.get("maneuver_hash") != _canonical_hash(payload):
    raise ValueError("Dynamic stair maneuver hash does not match its contents.")
  parameters = payload.get("parameters")
  trigger = payload.get("trigger")
  bases = payload.get("kinematic_bases")
  bindings = payload.get("bindings")
  if not all(isinstance(value, Mapping) for value in (parameters, trigger, bases, bindings)):
    raise TypeError("Dynamic stair maneuver sections must be mappings.")
  assert isinstance(parameters, Mapping)
  assert isinstance(trigger, Mapping)
  assert isinstance(bases, Mapping)
  assert isinstance(bindings, Mapping)
  validated_bindings = validate_dynamic_maneuver_bindings(bindings)
  if trigger.get("metric") != "abs(F0*nx)" or trigger.get("qualification") != "per-wheel-live-required":
    raise ValueError("Dynamic stair trigger contract drifted.")
  maneuver = DynamicStairManeuver(
    lift_mode=DynamicLiftMode(str(parameters["lift_mode"])),
    split_amplitude_rad=float(parameters["split_amplitude_rad"]),
    lift_amplitude_rad=float(parameters["lift_amplitude_rad"]),
    trailing_delay_s=float(parameters["trailing_delay_s"]),
    drive_feedforward_radps=float(parameters["drive_feedforward_radps"]),
    split_basis_left=tuple(float(value) for value in bases["split_left"]),
    split_basis_right=tuple(float(value) for value in bases["split_right"]),
    lift_basis_left=tuple(float(value) for value in bases["lift_left"]),
    lift_basis_right=tuple(float(value) for value in bases["lift_right"]),
    approach_vx=float(parameters["approach_vx"]),
    approach_duration_s=float(parameters["approach_duration_s"]),
    preload_duration_s=float(parameters["preload_duration_s"]),
    lift_duration_s=float(parameters["lift_duration_s"]),
    recover_duration_s=float(parameters["recover_duration_s"]),
    contact_timeout_s=float(parameters["contact_timeout_s"]),
    trail_contact_timeout_s=float(parameters["trail_contact_timeout_s"]),
    cross_timeout_s=float(parameters["cross_timeout_s"]),
    first_cross_m=float(parameters["first_cross_m"]),
    next_cross_m=float(parameters["next_cross_m"]),
    recover_stable_steps=int(parameters["recover_stable_steps"]),
    trigger_force_n=float(trigger["threshold_n"]),
    trigger_window=int(trigger["window"]),
    maneuver_hash=str(payload["maneuver_hash"]),
    bindings=validated_bindings,
    source=source,
  )
  # Round-trip guards the declared fixed CEM protocol too.
  cem = payload.get("cem")
  expected_cem = dynamic_maneuver_payload(maneuver, bindings=maneuver.bindings or {})["cem"]
  if cem != expected_cem:
    raise ValueError("Dynamic stair maneuver CEM protocol drifted.")
  return maneuver


def load_dynamic_maneuver(path: Path) -> DynamicStairManeuver:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, Mapping):
    raise TypeError("Dynamic stair maneuver file must contain an object.")
  return parse_dynamic_maneuver(payload, source=str(path.resolve()))


__all__ = [
  "DYNAMIC_MANEUVER_ARTIFACT_TYPE",
  "DYNAMIC_MANEUVER_REQUIRED_BINDINGS",
  "DYNAMIC_MANEUVER_SCHEMA_VERSION",
  "DYNAMIC_STAIR_CEM_ITERATIONS",
  "DYNAMIC_STAIR_CEM_LOWER",
  "DYNAMIC_STAIR_CEM_POPULATION",
  "DYNAMIC_STAIR_CEM_REPLICATES",
  "DYNAMIC_STAIR_CEM_SEED",
  "DYNAMIC_STAIR_CEM_UPPER",
  "DYNAMIC_STAIR_CONTRACT_SCHEMA_VERSION",
  "DYNAMIC_STAIR_FEEDFORWARD_LIMIT_RAD",
  "DYNAMIC_STAIR_PPO_LEG_SCALE_RAD",
  "DYNAMIC_STAIR_TASK_ID",
  "DYNAMIC_STAIR_TIME_EPS_S",
  "DynamicLiftMode",
  "DynamicStairManeuver",
  "DynamicStairSensors",
  "DynamicStairState",
  "DynamicStairTargets",
  "LeadSide",
  "StairTraversalMode",
  "choose_lead_side",
  "dynamic_leg_feedforward",
  "dynamic_maneuver_payload",
  "dynamic_stair_step",
  "half_cosine_bump",
  "load_dynamic_maneuver",
  "parse_dynamic_maneuver",
  "update_loaded_contact",
  "validate_dynamic_maneuver_bindings",
]
