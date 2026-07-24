"""Capture paired flat/stair evidence around first wheel-riser impact.

The first P2 stall diagnostic used absolute slip and saturation thresholds.
Its formal flat controls already crossed those thresholds, so this follow-up
records paired evidence without assigning a single physical cause.

Observational only: no checkpoint, training, promotion, or P3 launch.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

PROJECT_PATH = Path(__file__).resolve().parents[1]
SRC_PATH = Path(__file__).resolve().parents[2]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

try:
  from hoppertrex_mjlab.hybrid.identification import NOMINAL_WHEEL_RADIUS_M
  from hoppertrex_mjlab.scripts import probe_hybrid_stair_height as stair
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
  from hybrid.identification import NOMINAL_WHEEL_RADIUS_M  # type: ignore[no-redef]
  from tasks.hoppertrex_balance_task import (  # type: ignore[no-redef]
    NON_WHEEL_GROUND_SENSOR_NAME,
    WHEEL_GROUND_GEOMS,
    non_wheel_ground_contact,
  )
  from tasks.hoppertrex_hybrid_task import (  # type: ignore[no-redef]
    hybrid_provenance_lines,
  )

  from scripts import probe_hybrid_stair_height as stair  # type: ignore[no-redef]
  from scripts import probe_hybrid_stall_diagnostic as stall  # type: ignore[no-redef]

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactMatch, ContactSensorCfg

DIAGNOSTIC_HEIGHTS_M = (0.0, 0.01)
DIAGNOSTIC_SENSOR_NAME = "wheel_terrain_causal_capture"
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

# Time-anchor selector, not a mechanism threshold. On 2026-07-24 the CPU
# interface probe observed max |normal_x| ~=0.106 flat and ~=0.446 at 1 cm.
RISER_MIN_ABS_NORMAL_X = 0.25
RISER_FACE_X_TOLERANCE_M = 0.02
RISER_MIN_NORMAL_FORCE_N = 1.0
FLAT_CONTROL_SUCCESS_RATE = 0.90
CLASSIFICATIONS = ("ANALYSIS_READY", "INVALID_CAPTURE")

SERIES_FIELDS = (
  "progress_past_face_m", "root_height_m", "pitch_rad", "body_vx_mps",
  "wheel_target_radps", "wheel_speed_radps", "model_torque_abs_nm",
  "torque_saturated", "wheel_slip_mps", "wheel_contact_count",
  "max_abs_contact_normal_x", "total_normal_force_n",
  "total_tangential_force_n", "riser_contact_count",
  "riser_normal_force_n", "riser_tangential_force_n",
  "riser_tangential_normal_ratio",
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
  return {
    "heights_m": DIAGNOSTIC_HEIGHTS_M,
    "command_cells": cells,
    "envs_per_height": (
      SMOKE_ENVS_PER_HEIGHT if smoke else OFFICIAL_ENVS_PER_HEIGHT
    ),
    "settle_steps": SMOKE_SETTLE_STEPS if smoke else OFFICIAL_SETTLE_STEPS,
    "drive_steps": SMOKE_DRIVE_STEPS if smoke else OFFICIAL_DRIVE_STEPS,
    "pre_impact_steps": (
      SMOKE_PRE_IMPACT_STEPS if smoke else OFFICIAL_PRE_IMPACT_STEPS
    ),
    "post_impact_steps": (
      SMOKE_POST_IMPACT_STEPS if smoke else OFFICIAL_POST_IMPACT_STEPS
    ),
    "stable_steps": SMOKE_STABLE_STEPS if smoke else OFFICIAL_STABLE_STEPS,
    "evidence_eligible": (not smoke) and device == "cuda:0",
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
  cfg.scene.sensors = tuple(cfg.scene.sensors) + (sensor,)
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
  """Select contacts suitable for anchoring first-riser impact time."""

  expected_shape = (*found.shape, 3)
  for name, value in (
    ("force_contact_frame", force_contact_frame),
    ("pos_global", pos_global),
    ("normal_global", normal_global),
  ):
    if tuple(value.shape) != expected_shape:
      raise ValueError(f"{name} must have shape {expected_shape}.")
  if outer_face_x.shape != found.shape[:1]:
    raise ValueError("outer_face_x must have shape (num_envs,).")
  face_error = (pos_global[..., 0] - outer_face_x[:, None]).abs()
  normal_force = force_contact_frame[..., 0].abs()
  return (
    found.bool()
    & (normal_global[..., 0].abs() >= RISER_MIN_ABS_NORMAL_X)
    & (face_error <= RISER_FACE_X_TOLERANCE_M)
    & (normal_force >= RISER_MIN_NORMAL_FORCE_N)
  )


def _finite_values(values: Iterable[float]) -> bool:
  return all(math.isfinite(float(value)) for value in values)


def _contact_snapshot(
  *,
  sensor: Any,
  env_id: int,
  mask: torch.Tensor,
  outer_face_x: float,
) -> list[dict[str, Any]]:
  data = sensor.data
  assert data.found is not None
  assert data.force is not None
  assert data.dist is not None
  assert data.pos is not None
  assert data.normal is not None
  assert data.tangent is not None
  found_ids = torch.nonzero(
    data.found[env_id] > 0, as_tuple=False
  ).squeeze(-1)
  rows: list[dict[str, Any]] = []
  slots_per_primary = int(sensor.cfg.num_slots)
  for slot_id_tensor in found_ids:
    slot_id = int(slot_id_tensor.item())
    primary_index = slot_id // slots_per_primary
    force = data.force[env_id, slot_id]
    rows.append({
      "slot_id": slot_id,
      "primary_index": primary_index,
      "primary_name": sensor.primary_names[primary_index],
      "selected_as_riser": bool(mask[env_id, slot_id].item()),
      "found_count": float(data.found[env_id, slot_id].item()),
      "force_contact_frame_n": force.detach().cpu().tolist(),
      "normal_force_abs_n": float(force[0].abs().item()),
      "tangential_force_norm_n": float(
        torch.linalg.vector_norm(force[1:]).item()
      ),
      "distance_m": float(data.dist[env_id, slot_id].item()),
      "position_global_m": data.pos[env_id, slot_id].detach().cpu().tolist(),
      "position_from_face_x_m": float(
        data.pos[env_id, slot_id, 0].item() - outer_face_x
      ),
      "normal_global": data.normal[env_id, slot_id].detach().cpu().tolist(),
      "tangent_global": data.tangent[env_id, slot_id].detach().cpu().tolist(),
    })
  return rows


def build_aligned_series(
  samples: dict[str, torch.Tensor],
  *,
  flat_env_id: int,
  stair_env_id: int,
  impact_step: int,
  pre_steps: int,
  post_steps: int,
) -> dict[str, Any]:
  """Build a columnar same-time flat/stair capture around stair impact."""

  start = impact_step - pre_steps
  stop = impact_step + post_steps
  if start < 0:
    raise ValueError("Impact lacks requested pre-impact history.")
  first = next(iter(samples.values()))
  if stop >= first.shape[0]:
    raise ValueError("Impact lacks requested post-impact history.")
  relative_steps = list(range(-pre_steps, post_steps + 1))
  flat: dict[str, list[float]] = {}
  stair_values: dict[str, list[float]] = {}
  delta: dict[str, list[float]] = {}
  for field in SERIES_FIELDS:
    values = samples[field][start : stop + 1]
    flat_column = values[:, flat_env_id].detach().cpu().tolist()
    stair_column = values[:, stair_env_id].detach().cpu().tolist()
    flat[field] = [float(value) for value in flat_column]
    stair_values[field] = [float(value) for value in stair_column]
    delta[field] = [
      float(stair_value - flat_value)
      for flat_value, stair_value in zip(flat_column, stair_column)
    ]
  return {
    "relative_steps": relative_steps,
    "relative_time_s": [
      step / stair.CONTROL_FREQUENCY_HZ for step in relative_steps
    ],
    "flat": flat,
    "stair": stair_values,
    "stair_minus_flat": delta,
  }


def summarize_aligned_series(series: dict[str, Any]) -> dict[str, Any]:
  """Summarize pre and impact/post values without causal labeling."""

  relative_steps = list(series["relative_steps"])
  pre_ids = [index for index, step in enumerate(relative_steps) if step < 0]
  post_ids = [index for index, step in enumerate(relative_steps) if step >= 0]
  summary: dict[str, Any] = {}
  for field in SERIES_FIELDS:
    field_summary: dict[str, Any] = {}
    for source in ("flat", "stair", "stair_minus_flat"):
      values = list(series[source][field])
      field_summary[source] = {
        "pre_mean": (
          sum(values[index] for index in pre_ids) / len(pre_ids)
          if pre_ids
          else None
        ),
        "impact": values[relative_steps.index(0)],
        "impact_and_post_mean": (
          sum(values[index] for index in post_ids) / len(post_ids)
        ),
      }
    summary[field] = field_summary
  return summary


def _stack_samples(
  samples: dict[str, list[torch.Tensor]],
) -> dict[str, torch.Tensor]:
  if not samples or any(not values for values in samples.values()):
    raise RuntimeError("Causal capture recorded no drive samples.")
  return {key: torch.stack(values) for key, values in samples.items()}


def run_cell(
  env: ManagerBasedRlEnv,
  *,
  heights: tuple[float, ...],
  cell: dict[str, Any],
  protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  """Run one command cell and return trial rows plus paired captures."""

  terrain_types, cross_x, reset_metadata = stair._reset_to_approach(
    env,
    root_height=CARD_HEIGHT_M,
    card_name=CARD_NAME,
    repeat=1,
  )
  if int(terrain_types.max().item()) >= len(heights):
    raise RuntimeError("Terrain type index exceeds diagnostic heights.")
  pairs = paired_environment_ids(terrain_types)
  pair_by_env = {
    pair["flat_env_id"]: pair["slot"] for pair in pairs
  } | {
    pair["stair_env_id"]: pair["slot"] for pair in pairs
  }
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
  sensor = env.scene.sensors[DIAGNOSTIC_SENSOR_NAME]
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
  first_impact_steps = torch.full(
    (env.num_envs,), -1, dtype=torch.long, device=env.device
  )
  impact_slots: dict[int, list[dict[str, Any]]] = {}
  samples: dict[str, list[torch.Tensor]] = {
    field: [] for field in SERIES_FIELDS
  }
  samples["valid_sample"] = []
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

    data = sensor.data
    assert data.found is not None
    assert data.force is not None
    assert data.pos is not None
    assert data.normal is not None
    found = data.found > 0
    riser = riser_contact_mask(
      found=found,
      force_contact_frame=data.force,
      pos_global=data.pos,
      normal_global=data.normal,
      outer_face_x=outer_face_x,
    )
    normal_force = data.force[..., 0].abs() * found
    tangential_force = torch.linalg.vector_norm(
      data.force[..., 1:], dim=-1
    ) * found
    riser_normal_force = (normal_force * riser).sum(dim=-1)
    riser_tangential_force = (tangential_force * riser).sum(dim=-1)
    riser_ratio = torch.where(
      riser_normal_force > 1.0e-9,
      riser_tangential_force / riser_normal_force,
      torch.zeros_like(riser_normal_force),
    )

    wheel_velocity = robot.data.joint_vel[:, wheel_ids]
    wheel_target = action_term.wheel_targets
    torque, saturated = stall.model_wheel_torque(
      wheel_target, wheel_velocity
    )
    forward_target = stall.signed_balance_channel(wheel_target)
    forward_speed = stall.signed_balance_channel(wheel_velocity)
    body_vx = robot.data.root_link_lin_vel_b[:, 0]
    step_values = {
      "progress_past_face_m": progress,
      "root_height_m": robot.data.root_link_pos_w[:, 2],
      "pitch_rad": stall._pitch_from_gravity(robot),
      "body_vx_mps": body_vx,
      "wheel_target_radps": forward_target,
      "wheel_speed_radps": forward_speed,
      "model_torque_abs_nm": torque.abs().mean(dim=-1),
      "torque_saturated": saturated.any(dim=-1).float(),
      "wheel_slip_mps": (
        forward_speed * NOMINAL_WHEEL_RADIUS_M - body_vx
      ).abs(),
      "wheel_contact_count": found.sum(dim=-1).float(),
      "max_abs_contact_normal_x": torch.where(
        found,
        data.normal[..., 0].abs(),
        torch.zeros_like(data.normal[..., 0]),
      ).max(dim=-1).values,
      "total_normal_force_n": normal_force.sum(dim=-1),
      "total_tangential_force_n": tangential_force.sum(dim=-1),
      "riser_contact_count": riser.sum(dim=-1).float(),
      "riser_normal_force_n": riser_normal_force,
      "riser_tangential_force_n": riser_tangential_force,
      "riser_tangential_normal_ratio": riser_ratio,
    }
    for field, value in step_values.items():
      samples[field].append(value.detach().clone())
    samples["valid_sample"].append(active.detach().clone())

    new_impact = active & (first_impact_steps < 0) & riser.any(dim=-1)
    for env_id_tensor in stair_env_ids[new_impact[stair_env_ids]]:
      env_id = int(env_id_tensor.item())
      first_impact_steps[env_id] = drive_index
      impact_slots[env_id] = _contact_snapshot(
        sensor=sensor,
        env_id=env_id,
        mask=riser,
        outer_face_x=float(outer_face_x[env_id].item()),
      )

  for _ in range(int(protocol["settle_steps"])):
    _step(0.0, None)
  for drive_index in range(int(protocol["drive_steps"])):
    _step(vx_cmd, drive_index)

  stacked = _stack_samples(samples)
  terrain_types_cpu = terrain_types.detach().cpu().tolist()
  trials: list[dict[str, Any]] = []
  for env_id, terrain_type in enumerate(terrain_types_cpu):
    trials.append({
      "cell": str(cell["name"]),
      "pitch_rad": pitch_cmd,
      "vx_mps": vx_cmd,
      "terrain_slot": int(pair_by_env[env_id]),
      "stair_height_m": float(heights[terrain_type]),
      "env_id": env_id,
      "success": bool(success[env_id].item()),
      "terminated": bool(terminated_ever[env_id].item()),
      "timeout": bool(timeout_ever[env_id].item()),
      "non_wheel_contact": bool(contact_ever[env_id].item()),
      "max_progress_past_face_m": float(max_progress[env_id].item()),
      "first_riser_impact_step": (
        int(first_impact_steps[env_id].item())
        if heights[terrain_type] > 0.0 and first_impact_steps[env_id] >= 0
        else None
      ),
      "root_reset": {
        "x_relative_to_face_m": float(
          reset_metadata["x_relative_to_face_m"][env_id].item()
        ),
        "y_relative_to_center_m": float(
          reset_metadata["y_relative_to_center_m"][env_id].item()
        ),
        "root_height_m": float(
          reset_metadata["root_height_m"][env_id].item()
        ),
        "root_linear_velocity_mps": (
          reset_metadata["root_linear_velocity_mps"][env_id]
          .detach().cpu().tolist()
        ),
        "root_angular_velocity_radps": (
          reset_metadata["root_angular_velocity_radps"][env_id]
          .detach().cpu().tolist()
        ),
      },
    })

  paired_captures: list[dict[str, Any]] = []
  pre_steps = int(protocol["pre_impact_steps"])
  post_steps = int(protocol["post_impact_steps"])
  for pair in pairs:
    flat_env_id = pair["flat_env_id"]
    stair_env_id = pair["stair_env_id"]
    impact_step = int(first_impact_steps[stair_env_id].item())
    invalid_reasons: list[str] = []
    series: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    if impact_step < 0:
      invalid_reasons.append("missing_first_riser_impact")
    else:
      start = impact_step - pre_steps
      stop = impact_step + post_steps
      if start < 0:
        invalid_reasons.append("insufficient_pre_impact_history")
      if stop >= int(protocol["drive_steps"]):
        invalid_reasons.append("insufficient_post_impact_history")
      if not invalid_reasons:
        valid_window = stacked["valid_sample"][start : stop + 1]
        if not bool(valid_window[:, flat_env_id].all()):
          invalid_reasons.append("flat_partner_ended_during_capture")
        if not bool(valid_window[:, stair_env_id].all()):
          invalid_reasons.append("stair_trial_ended_during_capture")
      if not invalid_reasons:
        series = build_aligned_series(
          stacked,
          flat_env_id=flat_env_id,
          stair_env_id=stair_env_id,
          impact_step=impact_step,
          pre_steps=pre_steps,
          post_steps=post_steps,
        )
        values = (
          value
          for source in ("flat", "stair", "stair_minus_flat")
          for field in SERIES_FIELDS
          for value in series[source][field]
        )
        if not _finite_values(values):
          invalid_reasons.append("non_finite_capture_value")
          series = None
        else:
          summary = summarize_aligned_series(series)
    paired_captures.append({
      "cell": str(cell["name"]),
      "terrain_slot": int(pair["slot"]),
      "flat_env_id": flat_env_id,
      "stair_env_id": stair_env_id,
      "impact_step": impact_step if impact_step >= 0 else None,
      "impact_time_s": (
        impact_step / stair.CONTROL_FREQUENCY_HZ
        if impact_step >= 0 else None
      ),
      "valid": not invalid_reasons,
      "invalid_reasons": invalid_reasons,
      "impact_contact_slots": impact_slots.get(stair_env_id, []),
      "aligned_series": series,
      "aligned_summary": summary,
    })
  return trials, paired_captures


def aggregate_cells(
  trials: list[dict[str, Any]],
  captures: list[dict[str, Any]],
  *,
  command_cells: tuple[dict[str, Any], ...],
  expected_pairs: int,
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for command_cell in command_cells:
    name = str(command_cell["name"])
    cell_trials = [trial for trial in trials if trial["cell"] == name]
    flat = [trial for trial in cell_trials if trial["stair_height_m"] == 0.0]
    stairs = [trial for trial in cell_trials if trial["stair_height_m"] > 0.0]
    cell_captures = [capture for capture in captures if capture["cell"] == name]
    if len(flat) != expected_pairs or len(stairs) != expected_pairs:
      raise ValueError(f"Unexpected trial count for command cell {name}.")
    if len(cell_captures) != expected_pairs:
      raise ValueError(f"Unexpected capture count for command cell {name}.")
    valid = [capture for capture in cell_captures if capture["valid"]]
    impact_steps = [
      int(capture["impact_step"])
      for capture in valid
      if capture["impact_step"] is not None
    ]
    delta_post_means: dict[str, float] = {}
    if valid:
      for field in SERIES_FIELDS:
        delta_post_means[field] = sum(
          float(
            capture["aligned_summary"][field]["stair_minus_flat"]
            ["impact_and_post_mean"]
          )
          for capture in valid
        ) / len(valid)
    rows.append({
      "cell": name,
      "pitch_rad": float(command_cell["pitch_rad"]),
      "vx_mps": float(command_cell["vx_mps"]),
      "expected_pairs": expected_pairs,
      "valid_capture_pairs": len(valid),
      "invalid_capture_pairs": expected_pairs - len(valid),
      "flat_success_rate": sum(bool(row["success"]) for row in flat) / len(flat),
      "flat_terminated_trials": sum(bool(row["terminated"]) for row in flat),
      "flat_timeout_trials": sum(bool(row["timeout"]) for row in flat),
      "flat_non_wheel_contact_trials": sum(
        bool(row["non_wheel_contact"]) for row in flat
      ),
      "stair_success_rate": sum(bool(row["success"]) for row in stairs) / len(stairs),
      "stair_terminated_trials": sum(bool(row["terminated"]) for row in stairs),
      "stair_timeout_trials": sum(bool(row["timeout"]) for row in stairs),
      "stair_non_wheel_contact_trials": sum(
        bool(row["non_wheel_contact"]) for row in stairs
      ),
      "impact_step_min": min(impact_steps) if impact_steps else None,
      "impact_step_max": max(impact_steps) if impact_steps else None,
      "mean_stair_minus_flat_impact_and_post": delta_post_means,
    })
  return rows


def classify_capture(
  cells: list[dict[str, Any]],
  captures: list[dict[str, Any]],
  *,
  expected_cells: int,
  expected_pairs_per_cell: int,
) -> dict[str, Any]:
  """Validate capture completeness without assigning a physical cause."""

  reasons: list[str] = []
  if len(cells) != expected_cells:
    reasons.append("unexpected_cell_count")
  if len(captures) != expected_cells * expected_pairs_per_cell:
    reasons.append("unexpected_pair_count")
  invalid_pairs = [
    f"{capture['cell']}:slot{capture['terrain_slot']}"
    for capture in captures
    if not capture["valid"]
  ]
  if invalid_pairs:
    reasons.append("invalid_aligned_pairs")
  invalid_flat_cells = [
    str(cell["cell"])
    for cell in cells
    if float(cell["flat_success_rate"]) < FLAT_CONTROL_SUCCESS_RATE
    or int(cell["flat_terminated_trials"]) != 0
    or int(cell["flat_timeout_trials"]) != 0
    or int(cell["flat_non_wheel_contact_trials"]) != 0
  ]
  if invalid_flat_cells:
    reasons.append("invalid_flat_control")
  return {
    "classification": "INVALID_CAPTURE" if reasons else "ANALYSIS_READY",
    "invalid_reasons": reasons,
    "invalid_pairs": invalid_pairs,
    "invalid_flat_cells": invalid_flat_cells,
    "single_cause_label": None,
  }


def build_payload(
  *,
  trials: list[dict[str, Any]],
  captures: list[dict[str, Any]],
  cells: list[dict[str, Any]],
  verdict: dict[str, Any] | None,
  action_cfg: Any,
  protocol: dict[str, Any],
  device: str,
  sensor: Any,
) -> dict[str, Any]:
  mjlab_root = Path(mjlab.__file__).resolve().parents[2]
  return {
    "schema_version": 1,
    "probe": "hybrid_p2_stall_causal_capture_v2",
    "evidence_eligible": bool(protocol["evidence_eligible"]),
    "promotion_eligible": False,
    "training_eligible": False,
    "classification": (
      None if verdict is None else verdict["classification"]
    ),
    "single_cause_label": None,
    "invalid_reasons": [] if verdict is None else verdict["invalid_reasons"],
    "invalid_pairs": [] if verdict is None else verdict["invalid_pairs"],
    "invalid_flat_cells": (
      [] if verdict is None else verdict["invalid_flat_cells"]
    ),
    "task": stair.TASK,
    "seed": stair.SEED,
    "git_sha": stair._git_sha(stair.REPOSITORY_PATH),
    "mjlab_git_sha": stair._git_sha(mjlab_root),
    "device": device,
    "runtime": stair._runtime_metadata(device),
    "checkpoint": None,
    "checkpoint_file_sha256": None,
    "controller_gain_hash": action_cfg.controller_gain_hash,
    "calibration_hash": action_cfg.calibration_hash,
    "yaw_calibration_hash": action_cfg.yaw_calibration_hash,
    "posture_map_hash": action_cfg.posture_map_hash,
    "posture_artifact_hash": action_cfg.posture_artifact_hash,
    "station_calibration_hash": action_cfg.station_calibration_hash,
    "action_scales": list(action_cfg.action_scales),
    "protocol": {
      "diagnostic_card": {"name": CARD_NAME, "height_m": CARD_HEIGHT_M},
      "command_cells": [dict(cell) for cell in protocol["command_cells"]],
      "baseline_cell": BASELINE_CELL_NAME,
      "paired_resets_across_cells": True,
      "paired_flat_stair_by_terrain_slot": True,
      "flat_sample_time_basis": "same absolute drive step as stair impact",
      "environment_seed": stair.SEED,
      "terrain": "pyramid_stairs",
      "step_width_m": stair.STEP_WIDTH_M,
      "heights_m": list(protocol["heights_m"]),
      "envs_per_height": int(protocol["envs_per_height"]),
      "settle_steps": int(protocol["settle_steps"]),
      "drive_steps": int(protocol["drive_steps"]),
      "pre_impact_steps": int(protocol["pre_impact_steps"]),
      "post_impact_steps": int(protocol["post_impact_steps"]),
      "aligned_sample_count": (
        int(protocol["pre_impact_steps"])
        + int(protocol["post_impact_steps"])
        + 1
      ),
      "control_frequency_hz": stair.CONTROL_FREQUENCY_HZ,
      "commanded_yaw_rate": 0.0,
      "policy_action": [0.0] * 6,
      "classifications": list(CLASSIFICATIONS),
      "classification_scope": (
        "capture validity only; no friction-only, torque-only, or "
        "drive-collapse causal label"
      ),
      "flat_control_success_rate": FLAT_CONTROL_SUCCESS_RATE,
      "riser_contact_selector": {
        "purpose": "first-impact time anchor only",
        "min_abs_normal_x": RISER_MIN_ABS_NORMAL_X,
        "face_x_tolerance_m": RISER_FACE_X_TOLERANCE_M,
        "min_normal_force_n": RISER_MIN_NORMAL_FORCE_N,
      },
      "contact_sensor": {
        "name": DIAGNOSTIC_SENSOR_NAME,
        "primary_names": list(sensor.primary_names),
        "secondary": "terrain body",
        "fields": list(DIAGNOSTIC_SENSOR_FIELDS),
        "reduce": "none",
        "num_slots_per_wheel": DIAGNOSTIC_SENSOR_SLOTS_PER_WHEEL,
        "history_length_substeps": 4,
        "force_frame": "contact frame",
        "position_normal_tangent_frame": "global frame",
      },
      "series_fields": list(SERIES_FIELDS),
      "wheel_model": {
        "radius_m": NOMINAL_WHEEL_RADIUS_M,
        "forward_channel": "0.5 * (right - left)",
        "torque_source": (
          "actuator model damping*(target-actual) clipped to peak; "
          "not a torque sensor"
        ),
      },
    },
    "cells": cells,
    "paired_captures": captures,
    "trials": trials,
  }


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite output: {args.output}")
  protocol = protocol_for_mode(args.smoke, args.device)
  heights = tuple(float(value) for value in protocol["heights_m"])
  cfg = make_causal_env_cfg(heights, int(protocol["envs_per_height"]))
  for line in hybrid_provenance_lines(cfg):
    print(line)
  action_cfg = cfg.actions["hybrid_wheel_leg"]
  if protocol["evidence_eligible"]:
    required = {
      "controller": action_cfg.controller_qualified,
      "velocity calibration": action_cfg.calibration_hash,
      "posture map": action_cfg.posture_map_qualified,
      "posture artifact hash": action_cfg.posture_artifact_hash,
      "station calibration": action_cfg.station_calibration_qualified,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
      raise ValueError(
        "Official causal capture lacks: " + ", ".join(missing)
      )
    if action_cfg.yaw_calibration_hash is not None:
      raise ValueError(
        "Official zero-yaw causal capture must not load a yaw artifact."
      )

  trials: list[dict[str, Any]] = []
  captures: list[dict[str, Any]] = []
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    sensor = env.scene.sensors[DIAGNOSTIC_SENSOR_NAME]
    for cell in protocol["command_cells"]:
      print(f"[stall-causal-v2] cell={cell['name']}")
      cell_trials, cell_captures = run_cell(
        env,
        heights=heights,
        cell=cell,
        protocol=protocol,
      )
      trials.extend(cell_trials)
      captures.extend(cell_captures)
  finally:
    env.close()

  cells = aggregate_cells(
    trials,
    captures,
    command_cells=tuple(protocol["command_cells"]),
    expected_pairs=int(protocol["envs_per_height"]),
  )
  verdict = (
    classify_capture(
      cells,
      captures,
      expected_cells=len(protocol["command_cells"]),
      expected_pairs_per_cell=int(protocol["envs_per_height"]),
    )
    if protocol["evidence_eligible"]
    else None
  )
  payload = build_payload(
    trials=trials,
    captures=captures,
    cells=cells,
    verdict=verdict,
    action_cfg=action_cfg,
    protocol=protocol,
    device=args.device,
    sensor=sensor,
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
    encoding="utf-8",
  )
  print(f"[stall-causal-v2] output={args.output}")
  print(f"[stall-causal-v2] classification={payload['classification']}")


if __name__ == "__main__":
  main()
