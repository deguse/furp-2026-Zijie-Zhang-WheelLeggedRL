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

RISER_MIN_ABS_NORMAL_X = 0.25
RISER_FACE_X_TOLERANCE_M = 0.02
RISER_MIN_NORMAL_FORCE_N = 1.0
FLAT_CONTROL_SUCCESS_RATE = 0.90
CLASSIFICATIONS = ("ANALYSIS_READY", "INVALID_CAPTURE")

# Minimal series fields for detector fitting
DETECTOR_SERIES_FIELDS = (
  "pitch_rad", "body_vx_mps", "wheel_target_radps", "wheel_speed_radps"
)

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

  abs_normal_x = normal_global[..., 0].abs()
  near_face = (pos_global[..., 0] - outer_face_x.unsqueeze(-1)).abs()
  force_magnitude = torch.linalg.vector_norm(
    force_contact_frame, dim=-1
  )
  return (
    found
    & (abs_normal_x >= RISER_MIN_ABS_NORMAL_X)
    & (near_face <= RISER_FACE_X_TOLERANCE_M)
    & (force_magnitude >= RISER_MIN_NORMAL_FORCE_N)
  )


def first_riser_impact_step(
  sensor_history: dict[str, torch.Tensor],
  *,
  stair_env_ids: torch.Tensor,
  outer_face_x: torch.Tensor,
) -> torch.Tensor:
  """Find the first drive step where each stair env has riser contact."""

  found = sensor_history["found"]
  num_envs, num_steps, num_slots = found.shape
  stair_slice = found[stair_env_ids]
  force = sensor_history["force"][stair_env_ids]
  pos = sensor_history["pos"][stair_env_ids]
  normal = sensor_history["normal"][stair_env_ids]
  riser_mask = riser_contact_mask(
    found=stair_slice,
    force_contact_frame=force,
    pos_global=pos,
    normal_global=normal,
    outer_face_x=outer_face_x[stair_env_ids].unsqueeze(-1).unsqueeze(-1),
  )
  any_riser_contact = riser_mask.any(dim=-1)
  has_impact = any_riser_contact.any(dim=-1)
  first_impact = torch.full(
    (len(stair_env_ids),), -1, dtype=torch.long, device=found.device
  )
  first_impact[has_impact] = any_riser_contact[has_impact].to(torch.long).argmax(dim=-1)
  return first_impact


def extract_detector_series(
  env: ManagerBasedRlEnv,
  samples: dict[str, torch.Tensor],
  *,
  env_id: int,
  start_index: int,
  count: int,
) -> dict[str, list[float]]:
  """Extract minimal series fields for detector fitting."""

  robot = env.scene["robot"]
  action_term = env.action_manager.get_term("hybrid_wheel_leg")
  wheel_ids = action_term._wheel_ids
  end = start_index + count
  pitch = samples["pitch"][env_id, start_index:end].cpu().numpy()
  body_vx = robot.data.root_lin_vel_w[env_id, start_index:end, 0].cpu().numpy()
  wheel_speed = (
    robot.data.joint_vel[env_id, start_index:end, wheel_ids]
    .mean(dim=-1)
    .cpu()
    .numpy()
  )
  wheel_target = samples["wheel_target"][env_id, start_index:end].cpu().numpy()
  return {
    "pitch_rad": pitch.tolist(),
    "body_vx_mps": body_vx.tolist(),
    "wheel_speed_radps": wheel_speed.tolist(),
    "wheel_target_radps": wheel_target.tolist(),
  }


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
  """Build one paired capture around the impact time anchor."""

  pre = int(protocol["pre_impact_steps"])
  post = int(protocol["post_impact_steps"])
  start = max(0, impact_step - pre)
  count = pre + 1 + post
  flat_series = extract_detector_series(
    env, samples, env_id=flat_env_id, start_index=start, count=count
  )
  stair_series = extract_detector_series(
    env, samples, env_id=stair_env_id, start_index=start, count=count
  )
  return {
    "slot": slot,
    "flat_env_id": flat_env_id,
    "stair_env_id": stair_env_id,
    "impact_step": impact_step,
    "valid": True,
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

  samples: dict[str, list[torch.Tensor]] = {}
  samples["pitch"] = []
  samples["wheel_target"] = []
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

    samples["pitch"].append(robot.data.root_quat_w[:, :].clone())
    command_outputs = action_term.processed_actions
    if command_outputs.ndim == 2 and command_outputs.shape[-1] >= 4:
      wheel_target = command_outputs[:, 3:5].mean(dim=-1)
    else:
      wheel_target = torch.zeros(env.num_envs, device=env.device)
    samples["wheel_target"].append(wheel_target)
    for field in DIAGNOSTIC_SENSOR_FIELDS:
      sensor_history[field].append(sensor.data[field].clone())

  for step in range(int(protocol["settle_steps"])):
    _step(vx_cmd, None)
  for drive_step in range(int(protocol["drive_steps"])):
    _step(vx_cmd, drive_step)

  stacked = _stack_samples(samples)
  quat_w = stacked["pitch"]
  pitch = torch.atan2(
    2.0 * (quat_w[..., 0] * quat_w[..., 2] + quat_w[..., 3] * quat_w[..., 1]),
    1.0 - 2.0 * (quat_w[..., 1] ** 2 + quat_w[..., 2] ** 2),
  )
  stacked["pitch"] = pitch
  sensor_stacked = _stack_samples(sensor_history)

  first_impact = first_riser_impact_step(
    sensor_stacked,
    stair_env_ids=stair_env_ids,
    outer_face_x=outer_face_x,
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

  flat_env_ids = torch.tensor(
    [pair["flat_env_id"] for pair in pairs], device=env.device
  )
  flat_terminated = terminated_ever[flat_env_ids].sum().item()
  flat_contact = contact_ever[flat_env_ids].sum().item()
  flat_success_count = success[flat_env_ids].sum().item()
  flat_success_rate = flat_success_count / len(flat_env_ids)

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
      "paired_captures": len(paired_captures),
    }
  ]
  return trials, paired_captures


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  if args.output.exists():
    raise FileExistsError(f"Output already exists: {args.output}")

  if not torch.cuda.is_available():
    if not args.smoke:
      raise RuntimeError("Official C2 capture requires CUDA.")

  protocol = protocol_for_mode(args.smoke, args.device)
  cfg = make_causal_env_cfg(
    DIAGNOSTIC_HEIGHTS_M, int(protocol["envs_per_height"])
  )
  cfg.seed = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)

  try:
    action_term = env.action_manager.get_term("hybrid_wheel_leg")
    if not hasattr(action_term, "controller_schedule_hash"):
      raise RuntimeError(
        "C2 capture requires controller_schedule_hash (C1 schedule artifact)."
      )
    schedule_hash = str(action_term.controller_schedule_hash)
    if schedule_hash != C1_SCHEDULE_HASH:
      raise ValueError(
        f"C2 requires schedule_hash {C1_SCHEDULE_HASH}, got {schedule_hash}."
      )

    provenance = _capture_provenance(cfg, args.device)
    for line in hybrid_provenance_lines(cfg):
      print(line)
    all_trials = []
    all_captures = []
    cells = protocol["command_cells"]
    for cell in cells:
      trials, captures = run_cell(
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
  flat_terminated_total = sum(t["flat_terminated"] for t in all_trials)
  flat_contact_total = sum(t["flat_non_wheel_contact"] for t in all_trials)
  valid_capture_count = len(all_captures)

  if (
    flat_control_passed
    and flat_terminated_total == 0
    and flat_contact_total == 0
    and valid_capture_count > 0
  ):
    classification = "ANALYSIS_READY"
  else:
    classification = "INVALID_CAPTURE"

  payload = {
    "schema_version": 1,
    "probe": "hybrid_c2_paired_capture_v1",
    "classification": classification,
    "evidence_eligible": bool(protocol["evidence_eligible"]),
    "promotion_eligible": False,
    "training_eligible": False,
    "checkpoint": None,
    "yaw_calibration_hash": None,
    "task": str(cfg.name),
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
