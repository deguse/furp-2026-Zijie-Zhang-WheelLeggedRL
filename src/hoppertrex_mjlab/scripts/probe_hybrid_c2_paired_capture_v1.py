"""C2 paired flat/stair capture on candidate-24 gain-scheduled LQR stack.

Prerequisite: C1 schedule artifact (schedule_hash 8fe8548b...) must be frozen
and deployed via HOPPERTREX_HYBRID_CONTROLLER_PATH. Outputs stall_causal_v2.json
compatible schema for detector fitting.

Observational only: no checkpoint, training, promotion, or PPO launch.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_PATH = Path(__file__).resolve().parents[1]
SRC_PATH = Path(__file__).resolve().parents[2]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

try:
  from hoppertrex_mjlab.hybrid.stair_classical import (
    CONTROL_DT_S,
    contact_detector_wheel_reference_radps,
  )
  from hoppertrex_mjlab.scripts import probe_hybrid_stair_height as stair
  from hoppertrex_mjlab.scripts import probe_hybrid_stall_causal_v2 as causal
  from hoppertrex_mjlab.scripts import probe_hybrid_stall_diagnostic as stall
  from hoppertrex_mjlab.tasks.hoppertrex_balance_task import (
    NON_WHEEL_GROUND_SENSOR_NAME,
    WHEEL_GROUND_GEOMS,
    non_wheel_ground_contact,
  )
  from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
    hybrid_provenance_lines,
  )
except ImportError:
  from hybrid.stair_classical import (  # type: ignore[no-redef]
    CONTROL_DT_S,
    contact_detector_wheel_reference_radps,
  )
  from tasks.hoppertrex_balance_task import (  # type: ignore[no-redef]
    NON_WHEEL_GROUND_SENSOR_NAME,
    WHEEL_GROUND_GEOMS,
    non_wheel_ground_contact,
  )
  from tasks.hoppertrex_hybrid_task import (  # type: ignore[no-redef]
    hybrid_provenance_lines,
  )

  from scripts import probe_hybrid_stair_height as stair  # type: ignore[no-redef]
  from scripts import probe_hybrid_stall_causal_v2 as causal  # type: ignore[no-redef]
  from scripts import probe_hybrid_stall_diagnostic as stall  # type: ignore[no-redef]

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import (
  BuiltinSensorCfg,
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
)

DIAGNOSTIC_HEIGHTS_M = (0.0, 0.01)
DIAGNOSTIC_SENSOR_NAME = "wheel_terrain_causal_capture"
IMU_ACCELEROMETER_NAME = "chassis_imu_accel"
IMU_ACCELEROMETER_SCENE_NAME = f"robot/{IMU_ACCELEROMETER_NAME}"
DIAGNOSTIC_SENSOR_SLOTS_PER_WHEEL = 8
DIAGNOSTIC_SENSOR_FIELDS = (
  "found", "force", "dist", "pos", "normal", "tangent"
)

CARD_NAME = "envelope_center"
CARD_HEIGHT_M = stall.CARD_HEIGHT_M
COMMAND_CELLS = (
  {"name": "pitch_zero", "pitch_rad": 0.0, "vx_mps": 0.07},
  {"name": "fast_lean_0p032", "pitch_rad": -0.032, "vx_mps": 0.10},
)
BASELINE_CELL_NAME = "pitch_zero"

OFFICIAL_ENVS_PER_HEIGHT = 16
OFFICIAL_SETTLE_STEPS = 200
OFFICIAL_DRIVE_STEPS = 500
OFFICIAL_PRE_IMPACT_STEPS = 25
OFFICIAL_POST_IMPACT_STEPS = 75
OFFICIAL_STABLE_STEPS = 25
SMOKE_ENVS_PER_HEIGHT = 1
SMOKE_SETTLE_STEPS = 2
SMOKE_DRIVE_STEPS = 8
SMOKE_PRE_IMPACT_STEPS = 1
SMOKE_POST_IMPACT_STEPS = 1
SMOKE_STABLE_STEPS = 2
SMOKE_CELL_NAMES = (BASELINE_CELL_NAME,)

RISER_MIN_ABS_NORMAL_X = 0.25
RISER_FACE_X_TOLERANCE_M = 0.02
RISER_MIN_NORMAL_FORCE_N = 1.0
FLAT_CONTROL_SUCCESS_RATE = 0.90
GRAVITY_MPS2 = 9.81
CLASSIFICATIONS = ("ANALYSIS_READY", "INVALID_CAPTURE")

# Minimal series fields for detector fitting
DETECTOR_SERIES_FIELDS = (
  "pitch_rate_radps",
  "wheel_speed_error_radps",
  "body_deceleration_mps2",
)
DETECTOR_ATTEMPT_FIELDS = DETECTOR_SERIES_FIELDS + ("detector_active",)
DETECTOR_SIGNAL_SCHEMA = "deployment_attempt_v2"

C1_SCHEDULE_HASH = (
  "8fe8548bca85978c164bbd7de39d2d6463cdfd8d7ab91796cf57696b0f64e203"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--smoke",
    action="store_true",
    help="CPU interface smoke; incomplete and never evidence eligible.",
  )
  args = parser.parse_args(argv)
  if not args.smoke and args.device != "cuda:0":
    parser.error("The official protocol is pinned to --device cuda:0.")
  return args


def protocol_for_mode(smoke: bool, device: str) -> dict[str, Any]:
  cells = (
    tuple(cell for cell in COMMAND_CELLS if cell["name"] in SMOKE_CELL_NAMES)
    if smoke
    else COMMAND_CELLS
  )
  envs_per_height = (
    SMOKE_ENVS_PER_HEIGHT if smoke else OFFICIAL_ENVS_PER_HEIGHT
  )
  pre_impact_steps = (
    SMOKE_PRE_IMPACT_STEPS if smoke else OFFICIAL_PRE_IMPACT_STEPS
  )
  post_impact_steps = (
    SMOKE_POST_IMPACT_STEPS if smoke else OFFICIAL_POST_IMPACT_STEPS
  )
  return {
    "heights_m": DIAGNOSTIC_HEIGHTS_M,
    "command_cells": cells,
    "envs_per_height": envs_per_height,
    "settle_steps": SMOKE_SETTLE_STEPS if smoke else OFFICIAL_SETTLE_STEPS,
    "drive_steps": SMOKE_DRIVE_STEPS if smoke else OFFICIAL_DRIVE_STEPS,
    "pre_impact_steps": pre_impact_steps,
    "post_impact_steps": post_impact_steps,
    "stable_steps": SMOKE_STABLE_STEPS if smoke else OFFICIAL_STABLE_STEPS,
    "control_dt_s": CONTROL_DT_S,
    "detector_signal_schema": DETECTOR_SIGNAL_SCHEMA,
    "detector_activation": "stair_attempt_start",
    "detector_series_fields": DETECTOR_SERIES_FIELDS,
    "detector_attempt_fields": DETECTOR_ATTEMPT_FIELDS,
    "detector_series_samples": (
      SMOKE_DRIVE_STEPS if smoke else OFFICIAL_DRIVE_STEPS
    ),
    "expected_capture_count": len(cells) * envs_per_height,
    "evidence_eligible": (not smoke) and device == "cuda:0",
  }


def _capture_provenance(cfg: Any, device: str) -> dict[str, Any]:
  """Provenance mapping consumed by the wrapper via top-level payload keys.

  Must stay a dict: the payload construction spreads it with ``**`` and the
  machine-room wrapper hard-fails unless git_sha/mjlab_git_sha/
  calibration_hash/posture_artifact_hash/station_calibration_hash exist.
  """

  action_cfg = cfg.actions["hybrid_wheel_leg"]
  mjlab_root = Path(mjlab.__file__).resolve().parents[2]
  return {
    "git_sha": stair._git_sha(stair.REPOSITORY_PATH),
    "mjlab_git_sha": stair._git_sha(mjlab_root),
    "runtime": stair._runtime_metadata(device),
    "controller_gain_hash": action_cfg.controller_gain_hash,
    "calibration_hash": action_cfg.calibration_hash,
    "posture_artifact_hash": action_cfg.posture_artifact_hash,
    "station_calibration_hash": action_cfg.station_calibration_hash,
  }


def make_causal_env_cfg(heights: tuple[float, ...], envs_per_height: int):
  """Build probe terrain and append an independent contact sensor."""

  cfg = stair.make_stair_env_cfg(heights, envs_per_height)
  sensor = ContactSensorCfg(
    name=DIAGNOSTIC_SENSOR_NAME,
    primary=ContactMatch(
      mode="geom", pattern=WHEEL_GROUND_GEOMS, entity="robot"
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=DIAGNOSTIC_SENSOR_FIELDS,
    reduce="none",
    num_slots=DIAGNOSTIC_SENSOR_SLOTS_PER_WHEEL,
    history_length=4,
  )
  accelerometer = BuiltinSensorCfg(
    name=IMU_ACCELEROMETER_NAME,
    sensor_type="accelerometer",
    obj=ObjRef(type="site", name="imu", entity="robot"),
  )
  cfg.scene.sensors = tuple(cfg.scene.sensors) + (sensor, accelerometer)
  return cfg


def paired_environment_ids(
  terrain_types: torch.Tensor,
  *,
  flat_type: int = 0,
  stair_type: int = 1,
) -> list[dict[str, int]]:
  """Pair flat and stair envs by stable within-terrain slot index."""

  flat_ids = torch.nonzero(
    terrain_types == flat_type, as_tuple=False
  ).squeeze(-1)
  stair_ids = torch.nonzero(
    terrain_types == stair_type, as_tuple=False
  ).squeeze(-1)
  if len(flat_ids) != len(stair_ids) or len(flat_ids) == 0:
    raise ValueError("Flat and stair terrain types need equal nonzero counts.")
  return [
    {
      "slot": slot,
      "flat_env_id": int(flat_id),
      "stair_env_id": int(stair_id),
    }
    for slot, (flat_id, stair_id) in enumerate(zip(flat_ids, stair_ids))
  ]


def riser_contact_mask(
  *,
  found: torch.Tensor,
  force_contact_frame: torch.Tensor,
  pos_global: torch.Tensor,
  normal_global: torch.Tensor,
  outer_face_x: torch.Tensor,
) -> torch.Tensor:
  """Select contacts suitable for anchoring first-riser impact time.

  Delegates to the machine-room-proven C0 implementation so the riser
  criterion cannot drift between the C0 and C2 capture families. A local
  transcription of it diverged three ways (float ``found`` used in a bitwise
  and, force vector norm instead of the contact-frame normal component, no
  shape guard) and crashed the 2026-07-29 CPU smoke.
  """

  return causal.riser_contact_mask(
    found=found,
    force_contact_frame=force_contact_frame,
    pos_global=pos_global,
    normal_global=normal_global,
    outer_face_x=outer_face_x,
  )


def riser_contact_mask_over_time(
  *,
  found: torch.Tensor,
  force_contact_frame: torch.Tensor,
  pos_global: torch.Tensor,
  normal_global: torch.Tensor,
  outer_face_x: torch.Tensor,
) -> torch.Tensor:
  """Apply the riser criterion to an ``[env, step, slot]`` history.

  The C0 criterion is written for one instant, i.e. ``[env, slot]`` with
  ``outer_face_x`` of shape ``(env,)``; handing it a time axis makes its
  ``outer_face_x[:, None]`` broadcast against the step axis instead of the
  slot axis. Folding steps into the env axis keeps the criterion byte-exact
  while giving each (env, step) row its own face position. With a single
  stair env the wrong shape broadcasts silently, which is how this survived
  the first CPU smoke.
  """

  num_envs, num_steps, num_slots = found.shape
  mask = riser_contact_mask(
    found=found.reshape(num_envs * num_steps, num_slots),
    force_contact_frame=force_contact_frame.reshape(
      num_envs * num_steps, num_slots, 3
    ),
    pos_global=pos_global.reshape(num_envs * num_steps, num_slots, 3),
    normal_global=normal_global.reshape(num_envs * num_steps, num_slots, 3),
    outer_face_x=outer_face_x.repeat_interleave(num_steps),
  )
  return mask.reshape(num_envs, num_steps, num_slots)


def first_riser_impact_step(
  sensor_history: dict[str, torch.Tensor],
  *,
  stair_env_ids: torch.Tensor,
  outer_face_x: torch.Tensor,
) -> torch.Tensor:
  """Find the first drive step where each stair env has riser contact.

  ``sensor_history`` columns are stacked ``[step, env, slot]``; the riser
  criterion is env-major, so the stair envs are selected on axis 1 and then
  transposed to ``[env, step, slot]``.
  """

  def _stair_major(field: str) -> torch.Tensor:
    return sensor_history[field][:, stair_env_ids].transpose(0, 1)

  riser_mask = riser_contact_mask_over_time(
    found=_stair_major("found"),
    force_contact_frame=_stair_major("force"),
    pos_global=_stair_major("pos"),
    normal_global=_stair_major("normal"),
    outer_face_x=outer_face_x[stair_env_ids],
  )
  any_riser_contact = riser_mask.any(dim=-1)
  has_impact = any_riser_contact.any(dim=-1)
  first_impact = torch.full(
    (len(stair_env_ids),), -1, dtype=torch.long, device=riser_mask.device
  )
  first_impact[has_impact] = (
    any_riser_contact[has_impact].to(torch.long).argmax(dim=-1)
  )
  return first_impact


def extract_detector_series(
  env: ManagerBasedRlEnv,
  samples: dict[str, torch.Tensor],
  *,
  env_id: int,
  start_index: int,
  count: int,
) -> dict[str, list[float]]:
  """Extract minimal series fields for detector fitting.

  ``samples`` columns are stacked as ``[step, env]`` (see ``_stack_samples``),
  so the step slice comes first and the env index second. Every field must be
  read from the recorded rollout, never from live ``robot.data`` at write-out
  time (that would be a single post-rollout instant, not a series).
  """

  del env
  end = start_index + count
  series: dict[str, list[float]] = {}
  for field in DETECTOR_SERIES_FIELDS:
    column = samples[field][start_index:end, env_id].detach().cpu()
    series[field] = [float(value) for value in column.tolist()]
  if "detector_active" in samples:
    column = samples["detector_active"][start_index:end, env_id].detach().cpu()
    series["detector_active"] = [bool(value) for value in column.tolist()]
  return series


def make_flat_attempts(
  env: ManagerBasedRlEnv,
  samples: dict[str, torch.Tensor],
  *,
  cell_name: str,
  flat_env_ids: torch.Tensor,
  terminated_ever: torch.Tensor,
  contact_ever: torch.Tensor,
) -> list[dict[str, Any]]:
  """Serialize full flat attempts without consulting stair-side state."""

  recorded_steps = int(next(iter(samples.values())).shape[0])
  return [
    {
      "cell_name": cell_name,
      "slot": int(slot),
      "flat_env_id": int(env_id),
      "terminated": bool(terminated_ever[env_id].item()),
      "non_wheel_contact": bool(contact_ever[env_id].item()),
      "recorded_steps": recorded_steps,
      "series": extract_detector_series(
        env,
        samples,
        env_id=int(env_id),
        start_index=0,
        count=recorded_steps,
      ),
    }
    for slot, env_id in enumerate(flat_env_ids.tolist())
  ]


def make_paired_capture(
  env: ManagerBasedRlEnv,
  samples: dict[str, torch.Tensor],
  *,
  slot: int,
  flat_env_id: int,
  stair_env_id: int,
  impact_step: int,
  protocol: dict[str, Any],
) -> dict[str, Any]:
  """Build one paired capture around the impact time anchor.

  The window must fit entirely inside the recorded drive steps: the fitter
  takes ``protocol.pre_impact_steps`` as the impact index for every capture,
  so a clamped or truncated window would silently feed post-impact samples in
  as the pre-impact baseline and measure detection latency from the wrong
  tick. A capture that does not fit is returned as ``valid=False`` with the
  reason recorded (the C0 producer raises; keeping the artifact and marking
  the row lets the rest of an expensive GPU session survive).
  """

  pre = int(protocol["pre_impact_steps"])
  post = int(protocol["post_impact_steps"])
  recorded_steps = int(next(iter(samples.values())).shape[0])
  start = impact_step - pre
  count = pre + 1 + post
  capture: dict[str, Any] = {
    "slot": slot,
    "flat_env_id": flat_env_id,
    "stair_env_id": stair_env_id,
    "impact_step": impact_step,
    "recorded_steps": recorded_steps,
  }
  if start < 0:
    return capture | {
      "valid": False,
      "invalid_reason": "impact_lacks_pre_impact_history",
      "aligned_series": None,
    }
  if start + count > recorded_steps:
    return capture | {
      "valid": False,
      "invalid_reason": "impact_lacks_post_impact_history",
      "aligned_series": None,
    }
  flat_series = extract_detector_series(
    env, samples, env_id=flat_env_id, start_index=start, count=count
  )
  stair_series = extract_detector_series(
    env, samples, env_id=stair_env_id, start_index=start, count=count
  )
  for series in (flat_series, stair_series):
    for values in series.values():
      if not causal._finite_values(values):
        return capture | {
          "valid": False,
          "invalid_reason": "non_finite_series",
          "aligned_series": None,
        }
  return capture | {
    "valid": True,
    "invalid_reason": None,
    "attempt_series": {
      "flat": extract_detector_series(
        env,
        samples,
        env_id=flat_env_id,
        start_index=0,
        count=recorded_steps,
      ),
      "stair": extract_detector_series(
        env,
        samples,
        env_id=stair_env_id,
        start_index=0,
        count=recorded_steps,
      ),
    },
    "aligned_series": {
      "flat": flat_series,
      "stair": stair_series,
    },
  }


def _stack_samples(
  samples: dict[str, list[torch.Tensor]],
) -> dict[str, torch.Tensor]:
  if not samples or any(not values for values in samples.values()):
    raise RuntimeError("Causal capture recorded no drive samples.")
  return {key: torch.stack(values) for key, values in samples.items()}


def body_forward_deceleration(
  specific_force_x: torch.Tensor,
  projected_gravity_x: torch.Tensor,
) -> torch.Tensor:
  """Convert a body-frame accelerometer sample to forward deceleration."""

  forward_acceleration = (
    specific_force_x + GRAVITY_MPS2 * projected_gravity_x
  )
  return torch.clamp(-forward_acceleration, min=0.0)


def flat_floor_protocol(device: str) -> dict[str, Any]:
  """Return the frozen, non-evidence C2-i flat-floor protocol."""

  return {
    "protocol_name": "hybrid_c2_flat_floor_v1",
    "capture_scope": "flat_only",
    "command_cells": COMMAND_CELLS,
    "envs_per_cell": OFFICIAL_ENVS_PER_HEIGHT,
    "settle_steps": OFFICIAL_SETTLE_STEPS,
    "drive_steps": OFFICIAL_DRIVE_STEPS,
    "stable_steps": OFFICIAL_STABLE_STEPS,
    "control_dt_s": CONTROL_DT_S,
    "seed": 1,
    "device": device,
    "detector_signal_schema": DETECTOR_SIGNAL_SCHEMA,
    "detector_activation": "wheel_odometry_progress_lt_0p35m",
    "detector_series_fields": DETECTOR_SERIES_FIELDS,
    "detector_attempt_fields": DETECTOR_ATTEMPT_FIELDS,
    "detector_series_samples": OFFICIAL_DRIVE_STEPS,
    "expected_flat_attempt_count": (
      len(COMMAND_CELLS) * OFFICIAL_ENVS_PER_HEIGHT
    ),
    "evidence_eligible": False,
    "detector_fit_eligible": False,
    "promotion_eligible": False,
    "training_eligible": False,
  }


FLAT_FLOOR_THRESHOLDS = {
  "pitch_rate_radps": (0.02, 0.04, 0.06, 0.08, 0.10),
  "wheel_speed_error_radps": (0.10, 0.20, 0.30, 0.50, 1.00),
  "body_deceleration_mps2": (0.5, 1.0, 2.0, 3.0, 5.0),
}


def _floor_feature_summary(
  attempts: list[dict[str, Any]],
) -> dict[str, Any]:
  active_values: dict[str, list[float]] = {
    field: [] for field in DETECTOR_SERIES_FIELDS
  }
  for attempt in attempts:
    series = attempt["series"]
    mask = np.asarray(series["detector_active"])
    if mask.dtype != np.bool_ or mask.shape != (OFFICIAL_DRIVE_STEPS,):
      raise ValueError("Flat floor mask must be Boolean and 500 ticks long.")
    if not np.any(mask):
      raise ValueError("Flat floor capture requires a nonempty active mask.")
    for field in DETECTOR_SERIES_FIELDS:
      values = np.asarray(series[field], dtype=np.float64)
      if values.shape != mask.shape or not np.all(np.isfinite(values)):
        raise ValueError("Flat floor detector series must be finite and complete.")
      if field == "body_deceleration_mps2" and np.any(values < 0.0):
        raise ValueError("Flat floor body deceleration must be nonnegative.")
      feature = values
      if field == "pitch_rate_radps":
        feature = np.abs(np.diff(values, prepend=values[0]))
      elif field == "wheel_speed_error_radps":
        feature = np.abs(values)
      active_values[field].extend(float(value) for value in feature[mask])

  features: dict[str, Any] = {}
  for field, values in active_values.items():
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
      raise ValueError("Flat floor summary requires active detector samples.")
    features[field] = {
      "active_tick_count": int(array.size),
      "max": float(np.max(array)),
      "p99_9": float(np.quantile(array, 0.999)),
    }
  return features


def flat_floor_summary(attempts: list[dict[str, Any]]) -> dict[str, Any]:
  """Summarize frozen-cell active ticks without fitting a detector."""

  expected_cells = {cell["name"] for cell in COMMAND_CELLS}
  actual_cells = {str(attempt.get("cell_name")) for attempt in attempts}
  if actual_cells != expected_cells:
    raise ValueError("Flat floor attempts must contain both frozen cells.")

  by_cell: dict[str, Any] = {}
  for cell_name in sorted(expected_cells):
    cell_attempts = [
      attempt for attempt in attempts if attempt.get("cell_name") == cell_name
    ]
    if len(cell_attempts) != OFFICIAL_ENVS_PER_HEIGHT:
      raise ValueError("Each flat floor cell requires exactly 16 attempts.")
    if any(
      attempt.get("terminated") is not False
      or attempt.get("non_wheel_contact") is not False
      or int(attempt.get("recorded_steps", -1)) != OFFICIAL_DRIVE_STEPS
      for attempt in cell_attempts
    ):
      raise ValueError("Flat floor attempt health or length is invalid.")
    by_cell[cell_name] = {"features": _floor_feature_summary(cell_attempts)}

  overall_features = _floor_feature_summary(attempts)
  covered = True
  for field, feature in overall_features.items():
    maximum = float(feature["max"])
    table = [
      {
        "threshold": threshold,
        "strictly_above_overall_max": bool(threshold > maximum),
      }
      for threshold in FLAT_FLOOR_THRESHOLDS[field]
    ]
    feature["fixed_grid_thresholds"] = table
    covered = covered and any(
      row["strictly_above_overall_max"] for row in table
    )
  return {
    "classification": (
      "FLOOR_GRID_COVERED" if covered else "FLOOR_GRID_UNCOVERED_STOP"
    ),
    "cells": by_cell,
    "overall": {"features": overall_features},
  }


def run_cell(
  env: ManagerBasedRlEnv,
  *,
  heights: tuple[float, ...],
  cell: dict[str, Any],
  protocol: dict[str, Any],
  flat_only: bool = False,
) -> tuple[
  list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
  """Run one cell and return health, paired captures, and flat attempts."""

  terrain_types, cross_x, reset_metadata = stair._reset_to_approach(
    env,
    root_height=CARD_HEIGHT_M,
    card_name=CARD_NAME,
    repeat=1,
  )
  if int(terrain_types.max().item()) >= len(heights):
    raise RuntimeError("Terrain type index exceeds diagnostic heights.")
  pairs = paired_environment_ids(terrain_types)
  stair_env_ids = torch.tensor(
    [pair["stair_env_id"] for pair in pairs],
    device=env.device,
    dtype=torch.long,
  )

  robot = env.scene["robot"]
  action_term = env.action_manager.get_term("hybrid_wheel_leg")
  wheel_ids = action_term._wheel_ids
  if len(wheel_ids) != 2:
    raise RuntimeError("Causal capture requires exactly two wheel joints.")
  sensor = None if flat_only else env.scene.sensors[DIAGNOSTIC_SENSOR_NAME]
  accelerometer = env.scene.sensors[IMU_ACCELEROMETER_SCENE_NAME]
  actions = torch.zeros(
    (env.num_envs, env.action_space.shape[-1]), device=env.device
  )
  outer_face_x = cross_x - stair.CROSS_DEPTH_M

  alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  terminated_ever = torch.zeros_like(alive)
  timeout_ever = torch.zeros_like(alive)
  contact_ever = torch.zeros_like(alive)
  success = torch.zeros_like(alive)
  stable = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  max_progress = reset_metadata["x_relative_to_face_m"].clone()
  detector_progress = torch.zeros(
    env.num_envs, dtype=torch.float, device=env.device
  )
  detector_active = torch.ones_like(alive)

  samples: dict[str, list[torch.Tensor]] = {
    field: [] for field in DETECTOR_ATTEMPT_FIELDS
  }
  sensor_history: dict[str, list[torch.Tensor]] = {
    field: [] for field in DIAGNOSTIC_SENSOR_FIELDS
  }
  pitch_cmd = float(cell["pitch_rad"])
  vx_cmd = float(cell["vx_mps"])

  def _step(vx: float, drive_index: int | None) -> None:
    was_alive = alive.clone()
    stair._force_commands(
      env, vx=vx, height=CARD_HEIGHT_M, pitch=pitch_cmd
    )
    _obs, _reward, terminated, timeouts, _extras = env.step(actions)
    stair._force_commands(
      env, vx=vx, height=CARD_HEIGHT_M, pitch=pitch_cmd
    )

    ended = was_alive & (terminated | timeouts)
    terminated_ever.logical_or_(was_alive & terminated)
    timeout_ever.logical_or_(was_alive & timeouts)
    active = was_alive & ~ended
    alive.logical_and_(~ended)
    direct_contact = non_wheel_ground_contact(
      env, NON_WHEEL_GROUND_SENSOR_NAME
    ).bool()
    termination_contact = env.termination_manager.get_term(
      "non_wheel_ground_contact"
    )
    contact = stair.merge_contact_observations(
      direct_contact, termination_contact
    )
    contact_ever.copy_(
      stair.update_contact_history(contact_ever, contact, was_alive)
    )
    progress = robot.data.root_link_pos_w[:, 0] - outer_face_x
    max_progress.copy_(
      stair.update_valid_max_progress(max_progress, progress, active)
    )
    if drive_index is None:
      return

    stable.copy_(
      torch.where(
        active & ~contact & (robot.data.root_link_pos_w[:, 0] >= cross_x),
        stable + 1,
        torch.zeros_like(stable),
      )
    )
    success.logical_or_(
      active & ~contact & (stable >= int(protocol["stable_steps"]))
    )

    # Record the detector's deployment inputs directly. Reconstructing pitch
    # rate from pitch or using the closed-loop wheel target changes the online
    # detector semantics and cannot qualify a deployable artifact.
    wheel_velocity = robot.data.joint_vel[:, wheel_ids]
    signed_wheel_speed = stall.signed_balance_channel(wheel_velocity)
    detector_progress.add_(
      signed_wheel_speed * action_term.cfg.wheel_radius * CONTROL_DT_S
    )
    station_drift = float(
      np.interp(
        pitch_cmd,
        action_term._station_pitch.detach().cpu().numpy(),
        action_term._station_drift.detach().cpu().numpy(),
      )
    )
    wheel_reference = contact_detector_wheel_reference_radps(
      command_vx=vx,
      velocity_command_scale=action_term.cfg.velocity_command_scale,
      velocity_command_bias=action_term.cfg.velocity_command_bias,
      station_drift_mps=station_drift,
      wheel_radius=action_term.cfg.wheel_radius,
    )
    samples["pitch_rate_radps"].append(
      robot.data.root_link_ang_vel_b[:, 1].clone()
    )
    samples["wheel_speed_error_radps"].append(
      (signed_wheel_speed - wheel_reference).clone()
    )
    samples["body_deceleration_mps2"].append(
      body_forward_deceleration(
        accelerometer.data[:, 0],
        robot.data.projected_gravity_b[:, 0],
      ).clone()
    )
    detector_active.logical_and_(detector_progress < 0.35)
    samples["detector_active"].append(detector_active.clone())
    if sensor is not None:
      # ContactData exposes fields as attributes, not by subscript.
      for field in DIAGNOSTIC_SENSOR_FIELDS:
        sensor_history[field].append(getattr(sensor.data, field).clone())

  # Settle at ZERO command velocity, as the registered C0 v2 producer does
  # (probe_hybrid_stall_causal_v2.run_cell). Settling at the command velocity
  # travels 200 steps x 0.07 m/s / 50 Hz = 0.28 m > the 0.25 m start offset,
  # so the riser is struck during the unrecorded settle and the "pre-impact"
  # window would actually hold deep post-impact samples -- while still
  # satisfying every wrapper check.
  for _step_index in range(int(protocol["settle_steps"])):
    _step(0.0, None)
  for drive_step in range(int(protocol["drive_steps"])):
    _step(vx_cmd, drive_step)

  stacked = _stack_samples(samples)

  flat_env_ids = torch.tensor(
    [pair["flat_env_id"] for pair in pairs], device=env.device
  )
  flat_attempts = make_flat_attempts(
    env,
    stacked,
    cell_name=str(cell["name"]),
    flat_env_ids=flat_env_ids,
    terminated_ever=terminated_ever,
    contact_ever=contact_ever,
  )

  first_impact = (
    torch.full(
      (len(stair_env_ids),), -1, dtype=torch.long, device=env.device
    )
    if flat_only
    else first_riser_impact_step(
      _stack_samples(sensor_history),
      stair_env_ids=stair_env_ids,
      outer_face_x=outer_face_x,
    )
  )

  paired_captures = [
    make_paired_capture(
      env,
      stacked,
      slot=pair["slot"],
      flat_env_id=pair["flat_env_id"],
      stair_env_id=pair["stair_env_id"],
      impact_step=int(first_impact[idx].item()),
      protocol=protocol,
    )
    for idx, pair in enumerate(pairs)
    if first_impact[idx] >= 0
  ]

  flat_terminated = terminated_ever[flat_env_ids].sum().item()
  flat_contact = contact_ever[flat_env_ids].sum().item()
  flat_success_count = success[flat_env_ids].sum().item()
  flat_success_rate = flat_success_count / len(flat_env_ids)
  # Stair-side health is diagnostic only (a stalling stair env is the expected
  # C0 outcome, not a failure), but without it a mid-rollout termination or
  # reset would be undetectable after the fact.
  stair_terminated = terminated_ever[stair_env_ids].sum().item()
  stair_contact = contact_ever[stair_env_ids].sum().item()
  invalid_reasons = sorted(
    {
      str(capture["invalid_reason"])
      for capture in paired_captures
      if not capture["valid"]
    }
  )

  trials = [
    {
      "cell_name": cell["name"],
      "total_envs": int(env.num_envs),
      "flat_envs": len(flat_env_ids),
      "stair_envs": len(stair_env_ids),
      "flat_terminated": int(flat_terminated),
      "flat_non_wheel_contact": int(flat_contact),
      "flat_success": int(flat_success_count),
      "flat_success_rate": float(flat_success_rate),
      "stair_terminated": int(stair_terminated),
      "stair_non_wheel_contact": int(stair_contact),
      "stair_envs_without_impact": int((first_impact < 0).sum().item()),
      "paired_captures": len(paired_captures),
      "valid_paired_captures": sum(
        1 for capture in paired_captures if capture["valid"]
      ),
      "invalid_capture_reasons": invalid_reasons,
      "recorded_drive_steps": int(next(iter(stacked.values())).shape[0]),
    }
  ]
  return trials, paired_captures, flat_attempts


def _require_schedule_hash(action_term: Any) -> str:
  """Resolve the C1 schedule hash from the real action-term surface.

  The runtime term exposes the schedule only as ``cfg.controller_schedule``;
  it has no ``controller_schedule_hash`` attribute (2026-07-29 machine-room
  preflight crash).
  """

  schedule = action_term.cfg.controller_schedule
  if schedule is None:
    raise RuntimeError(
      "C2 capture requires controller_schedule_hash (C1 schedule artifact)."
    )
  schedule_hash = str(schedule.schedule_hash)
  if schedule_hash != C1_SCHEDULE_HASH:
    raise ValueError(
      f"C2 requires schedule_hash {C1_SCHEDULE_HASH}, got {schedule_hash}."
    )
  return schedule_hash


def classify_capture(
  *,
  protocol: dict[str, Any],
  trials: list[dict[str, Any]],
  captures: list[dict[str, Any]],
) -> str:
  """Classify only a complete, healthy official capture as analysis-ready."""

  if protocol != protocol_for_mode(smoke=False, device="cuda:0"):
    return "INVALID_CAPTURE"
  expected_cells = len(COMMAND_CELLS)
  expected_captures = 32
  expected_drive_steps = OFFICIAL_DRIVE_STEPS
  valid_capture_count = sum(capture.get("valid") is True for capture in captures)
  captures_complete = all(
    capture.get("valid") is True
    and isinstance(capture.get("attempt_series"), dict)
    and isinstance(capture.get("aligned_series"), dict)
    and all(
      isinstance(capture["attempt_series"].get(side), dict)
      and isinstance(capture["attempt_series"][side].get("detector_active"), list)
      and len(capture["attempt_series"][side]["detector_active"])
      == expected_drive_steps
      and all(
        isinstance(capture["attempt_series"][side].get(field), list)
        and len(capture["attempt_series"][side][field]) == expected_drive_steps
        and all(
          math.isfinite(float(value))
          for value in capture["attempt_series"][side][field]
        )
        for field in DETECTOR_SERIES_FIELDS
      )
      for side in ("flat", "stair")
    )
    for capture in captures
  )
  flat_control_passed = bool(trials) and all(
    float(trial["flat_success_rate"]) >= FLAT_CONTROL_SUCCESS_RATE
    for trial in trials
  )
  trials_complete = len(trials) == expected_cells and all(
    int(trial["recorded_drive_steps"]) == expected_drive_steps
    and int(trial["stair_terminated"]) == 0
    and int(trial["stair_envs_without_impact"]) == 0
    and int(trial["paired_captures"]) == int(protocol["envs_per_height"])
    and int(trial["valid_paired_captures"]) == int(protocol["envs_per_height"])
    and int(trial["flat_terminated"]) == 0
    and int(trial["flat_non_wheel_contact"]) == 0
    for trial in trials
  )
  if (
    flat_control_passed
    and trials_complete
    and len(captures) == expected_captures
    and valid_capture_count == expected_captures
    and captures_complete
  ):
    return "ANALYSIS_READY"
  return "INVALID_CAPTURE"


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  if args.output.exists():
    raise FileExistsError(f"Output already exists: {args.output}")

  if not torch.cuda.is_available() and not args.smoke:
    raise RuntimeError("Official C2 capture requires CUDA.")

  protocol = protocol_for_mode(args.smoke, args.device)
  cfg = make_causal_env_cfg(
    DIAGNOSTIC_HEIGHTS_M, int(protocol["envs_per_height"])
  )
  cfg.seed = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)

  try:
    action_term = env.action_manager.get_term("hybrid_wheel_leg")
    schedule_hash = _require_schedule_hash(action_term)

    provenance = _capture_provenance(cfg, args.device)
    for line in hybrid_provenance_lines(cfg):
      print(line)
    all_trials = []
    all_captures = []
    cells = protocol["command_cells"]
    for cell in cells:
      trials, captures, _flat_attempts = run_cell(
        env,
        heights=DIAGNOSTIC_HEIGHTS_M,
        cell=cell,
        protocol=protocol,
      )
      all_trials.extend(trials)
      all_captures.extend(captures)
  finally:
    env.close()

  flat_success_rates = [t["flat_success_rate"] for t in all_trials]
  flat_control_passed = all(
    rate >= FLAT_CONTROL_SUCCESS_RATE for rate in flat_success_rates
  )
  # Only captures whose window fits and whose series are finite count toward
  # the classification; invalid rows stay in the artifact for provenance.
  valid_capture_count = sum(1 for c in all_captures if c["valid"])
  invalid_capture_count = len(all_captures) - valid_capture_count

  classification = classify_capture(
    protocol=protocol,
    trials=all_trials,
    captures=all_captures,
  )

  payload = {
    "schema_version": 1,
    "probe": "hybrid_c2_paired_capture_v1",
    "classification": classification,
    "evidence_eligible": bool(protocol["evidence_eligible"]),
    "promotion_eligible": False,
    "training_eligible": False,
    "checkpoint": None,
    "yaw_calibration_hash": None,
    # The env cfg carries no task id; the registered task identity is the
    # stair module's TASK (pinned by contract test and by the wrapper).
    "task": stair.TASK,
    "seed": int(cfg.seed),
    "device": args.device,
    "smoke": args.smoke,
    **provenance,
    "controller_schedule_hash": schedule_hash,
    "protocol": protocol,
    "trials": all_trials,
    "paired_captures": all_captures,
    "flat_control_passed": flat_control_passed,
    "valid_capture_count": valid_capture_count,
    "invalid_capture_count": invalid_capture_count,
  }

  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
    json.dumps(payload, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
  )
  print(
    f"[c2-paired-capture] classification={classification} "
    f"captures={valid_capture_count} cells={len(cells)}"
  )


if __name__ == "__main__":
  main()
