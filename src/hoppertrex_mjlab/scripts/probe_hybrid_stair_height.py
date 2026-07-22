#!/usr/bin/env python3
"""Measure the zero-residual classical stack's first-step height boundary.

The official P2 k.0 protocol uses fixed pyramid stairs, two pre-registered
posture cards, and zero policy actions. It is a physical allocation probe,
not a gate or training entrypoint. A short ``--smoke`` mode only exercises
the mechanics on CPU and is always marked ineligible as evidence.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

PROJECT_PATH = Path(__file__).resolve().parents[1]
SRC_PATH = Path(__file__).resolve().parents[2]
REPOSITORY_PATH = Path(__file__).resolve().parents[3]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

try:
  import hoppertrex_mjlab.tasks as tasks  # noqa: E402,F401
  from hoppertrex_mjlab.tasks.hoppertrex_balance_task import (  # noqa: E402
    NON_WHEEL_GROUND_SENSOR_NAME,
    non_wheel_ground_contact,
  )
  from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (  # noqa: E402
    hybrid_provenance_lines,
  )
except ImportError:
  import tasks  # noqa: E402,F401
  from tasks.hoppertrex_balance_task import (  # type: ignore[no-redef]
    NON_WHEEL_GROUND_SENSOR_NAME,
    non_wheel_ground_contact,
  )
  from tasks.hoppertrex_hybrid_task import (  # type: ignore[no-redef]
    hybrid_provenance_lines,
  )

import mjlab  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg  # noqa: E402
from mjlab.terrains.config import pyramid_stairs  # noqa: E402


TASK = "HopperTrex-Hybrid-v2-Stage5"
SEED = 1
HEIGHTS_M = tuple(round(index * 0.01, 2) for index in range(11))
POSTURE_CARDS = (
  {
    "name": "envelope_center",
    "height_m": 0.3092089487,
    "pitch_rad": 0.016,
  },
  {
    "name": "high_zero_pitch",
    "height_m": 0.3276857266,
    "pitch_rad": 0.0,
  },
)
TERRAIN_SIZE_M = (8.0, 8.0)
TERRAIN_BORDER_WIDTH_M = 1.0
STEP_WIDTH_M = 0.30
PLATFORM_WIDTH_M = 3.0
START_OFFSET_M = 0.25
CROSS_DEPTH_M = 0.15
RESET_X_JITTER_M = 0.02
RESET_Y_JITTER_M = 0.03
RESET_VX_JITTER_MPS = 0.01
RESET_PITCH_RATE_JITTER_RADPS = 0.02
COMMAND_VX_MPS = 0.07
CONTROL_FREQUENCY_HZ = 50.0
OFFICIAL_ENVS_PER_HEIGHT = 16
OFFICIAL_REPEATS = 3
OFFICIAL_SETTLE_STEPS = 100
OFFICIAL_DRIVE_STEPS = 500
OFFICIAL_STABLE_STEPS = 25
SUCCESS_RATE_LIMIT = 0.90
CLASSIFICATIONS = (
  "CLASSICAL_DEATH_HEIGHT_BRACKETED",
  "EXTEND_SWEEP_BEFORE_P3",
  "STOP_FOR_VARIANCE_ANALYSIS",
  "INVALID_FLAT_CONTROL_STOP",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--smoke",
    action="store_true",
    help="Run a one-env CPU mechanics check; never valid research evidence.",
  )
  args = parser.parse_args(argv)
  if not args.smoke and args.device != "cuda:0":
    parser.error("The official protocol is pinned to --device cuda:0.")
  return args


def protocol_for_mode(smoke: bool) -> dict[str, Any]:
  if smoke:
    return {
      "heights_m": (0.0,),
      "envs_per_height": 1,
      "repeats": 1,
      "settle_steps": 2,
      "drive_steps": 5,
      "stable_steps": 2,
      "evidence_eligible": False,
    }
  return {
    "heights_m": HEIGHTS_M,
    "envs_per_height": OFFICIAL_ENVS_PER_HEIGHT,
    "repeats": OFFICIAL_REPEATS,
    "settle_steps": OFFICIAL_SETTLE_STEPS,
    "drive_steps": OFFICIAL_DRIVE_STEPS,
    "stable_steps": OFFICIAL_STABLE_STEPS,
    "evidence_eligible": True,
  }


def approach_geometry(origin_x: float) -> dict[str, float]:
  """Return the first-riser geometry relative to one terrain origin."""

  inner_half_width = 0.5 * (
    TERRAIN_SIZE_M[0] - 2.0 * TERRAIN_BORDER_WIDTH_M
  )
  outer_face_x = float(origin_x) - inner_half_width
  return {
    "outer_face_x": outer_face_x,
    "start_x": outer_face_x - START_OFFSET_M,
    "cross_x": outer_face_x + CROSS_DEPTH_M,
  }


def make_stair_env_cfg(heights: tuple[float, ...], envs_per_height: int):
  if not heights or envs_per_height < 1:
    raise ValueError("Stair probe needs heights and positive envs_per_height.")
  if any(height < 0.0 for height in heights):
    raise ValueError("Stair heights must be non-negative.")

  cfg = load_env_cfg(TASK, play=True)
  cfg.seed = SEED
  num_envs = len(heights) * envs_per_height
  cfg.scene.num_envs = num_envs
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      seed=SEED,
      curriculum=True,
      size=TERRAIN_SIZE_M,
      num_rows=1,
      num_cols=len(heights),
      difficulty_range=(0.0, 0.0),
      sub_terrains={
        f"stair_{int(round(height * 100)):02d}cm": pyramid_stairs(
          proportion=1.0,
          step_height_range=(height, height),
          step_width=STEP_WIDTH_M,
          platform_width=PLATFORM_WIDTH_M,
          border_width=TERRAIN_BORDER_WIDTH_M,
        )
        for height in heights
      },
    ),
    max_init_terrain_level=0,
    num_envs=num_envs,
  )
  cfg.episode_length_s = 1.0e9
  return cfg


def _force_commands(
  env: ManagerBasedRlEnv,
  *,
  vx: float,
  height: float,
  pitch: float,
) -> None:
  twist = env.command_manager.get_term("twist")
  for attribute in ("vel_command_b", "vel_command_w"):
    command = getattr(twist, attribute)
    command[:, :] = 0.0
    command[:, 0] = vx
  for attribute in (
    "is_standing_env",
    "is_heading_env",
    "is_world_env",
    "is_forward_env",
  ):
    value = getattr(twist, attribute, None)
    if value is not None:
      value[:] = False

  posture = env.command_manager.get_term("posture")
  command = getattr(posture, "_command", None)
  if command is None:
    raise AttributeError("Posture command term does not expose _command.")
  target = getattr(posture, "_target", None)
  if target is not None:
    target[:, 0] = height
    target[:, 1] = pitch
  command[:, 0] = height
  command[:, 1] = pitch


def reset_perturbations(
  *,
  slots: int,
  card_name: str,
  repeat: int,
) -> torch.Tensor:
  """Return reproducible per-slot reset perturbations on CPU."""

  if slots < 1 or repeat < 1:
    raise ValueError("Reset perturbations need positive slots and repeat.")
  card_names = [str(card["name"]) for card in POSTURE_CARDS]
  if card_name not in card_names:
    raise ValueError(f"Unknown posture card: {card_name}")
  card_index = card_names.index(card_name)
  generator = torch.Generator(device="cpu")
  generator.manual_seed(SEED * 10_000 + card_index * 100 + repeat)
  unit = 2.0 * torch.rand((slots, 4), generator=generator) - 1.0
  scales = torch.tensor([
    RESET_X_JITTER_M,
    RESET_Y_JITTER_M,
    RESET_VX_JITTER_MPS,
    RESET_PITCH_RATE_JITTER_RADPS,
  ])
  return unit * scales


def _reset_to_approach(
  env: ManagerBasedRlEnv,
  *,
  root_height: float,
  card_name: str,
  repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
  env.reset()
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_types is None:
    raise RuntimeError("Stair probe requires generated terrain types.")
  origins = env.scene.env_origins
  geometry = approach_geometry(0.0)
  terrain_types = terrain.terrain_types.clone()
  counts = torch.bincount(terrain_types, minlength=int(terrain_types.max()) + 1)
  if not torch.all(counts == counts[0]):
    raise RuntimeError("Stair heights do not have equal environment counts.")
  slots_per_height = int(counts[0].item())
  slot_ids = torch.empty_like(terrain_types)
  for terrain_type in range(len(counts)):
    env_ids = torch.nonzero(
      terrain_types == terrain_type, as_tuple=False
    ).squeeze(-1)
    slot_ids[env_ids] = torch.arange(len(env_ids), device=env.device)
  perturbations = reset_perturbations(
    slots=slots_per_height,
    card_name=card_name,
    repeat=repeat,
  ).to(env.device)
  reset_values = perturbations[slot_ids]
  robot = env.scene["robot"]
  root_states = robot.data.default_root_state.clone()
  root_states[:, 0] = (
    origins[:, 0] + geometry["start_x"] + reset_values[:, 0]
  )
  root_states[:, 1] = origins[:, 1] + reset_values[:, 1]
  root_states[:, 2] = float(root_height)
  root_states[:, 7:13] = 0.0
  root_states[:, 7] = reset_values[:, 2]
  root_states[:, 11] = reset_values[:, 3]
  robot.write_root_state_to_sim(root_states)
  env.sim.forward()
  env.sim.sense()
  cross_x = origins[:, 0] + geometry["cross_x"]
  outer_face_x = origins[:, 0] + geometry["outer_face_x"]
  reset_metadata = {
    "x_relative_to_face_m": root_states[:, 0] - outer_face_x,
    "y_relative_to_center_m": root_states[:, 1] - origins[:, 1],
    "root_height_m": root_states[:, 2],
    "root_linear_velocity_mps": root_states[:, 7:10],
    "root_angular_velocity_radps": root_states[:, 10:13],
  }
  return terrain_types, cross_x, reset_metadata


def merge_contact_observations(
  direct_contact: torch.Tensor,
  termination_contact: torch.Tensor,
) -> torch.Tensor:
  """Preserve contact that triggered an in-step automatic reset."""

  return direct_contact.bool() | termination_contact.bool()


def update_valid_max_progress(
  maximum: torch.Tensor,
  progress: torch.Tensor,
  valid: torch.Tensor,
) -> torch.Tensor:
  """Update progress only from states that still belong to the trial."""

  return torch.where(valid, torch.maximum(maximum, progress), maximum)


def update_contact_history(
  history: torch.Tensor,
  contact: torch.Tensor,
  was_active: torch.Tensor,
) -> torch.Tensor:
  """Record contact from trials that were active before the step."""

  return history | (was_active & contact)


def run_card_repeat(
  env: ManagerBasedRlEnv,
  *,
  heights: tuple[float, ...],
  card: dict[str, float | str],
  repeat: int,
  settle_steps: int,
  drive_steps: int,
  stable_steps: int,
) -> list[dict[str, Any]]:
  terrain_types, cross_x, reset_metadata = _reset_to_approach(
    env,
    root_height=float(card["height_m"]),
    card_name=str(card["name"]),
    repeat=repeat,
  )
  if int(terrain_types.max().item()) >= len(heights):
    raise RuntimeError("Terrain type index exceeds the frozen height table.")

  robot = env.scene["robot"]
  action_dim = env.action_space.shape[-1]
  actions = torch.zeros((env.num_envs, action_dim), device=env.device)
  terminated_ever = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  contact_ever = torch.zeros_like(terminated_ever)
  success = torch.zeros_like(terminated_ever)
  stable = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  success_step = torch.full(
    (env.num_envs,), -1, dtype=torch.long, device=env.device
  )
  max_progress = reset_metadata["x_relative_to_face_m"].clone()

  def _step(vx: float, drive_index: int | None) -> None:
    was_active = ~success & ~terminated_ever
    _force_commands(
      env,
      vx=vx,
      height=float(card["height_m"]),
      pitch=float(card["pitch_rad"]),
    )
    _obs, _reward, terminated, _timeouts, _extras = env.step(actions)
    _force_commands(
      env,
      vx=vx,
      height=float(card["height_m"]),
      pitch=float(card["pitch_rad"]),
    )
    terminated_ever.logical_or_(was_active & terminated)
    active = was_active & ~terminated
    direct_contact = non_wheel_ground_contact(
      env, NON_WHEEL_GROUND_SENSOR_NAME
    ).bool()
    termination_contact = env.termination_manager.get_term(
      "non_wheel_ground_contact"
    )
    contact = merge_contact_observations(
      direct_contact, termination_contact
    )
    contact_ever.copy_(update_contact_history(contact_ever, contact, was_active))
    progress = robot.data.root_link_pos_w[:, 0] - (
      cross_x - CROSS_DEPTH_M
    )
    max_progress.copy_(update_valid_max_progress(max_progress, progress, active))
    if drive_index is None:
      return
    stable.copy_(
      torch.where(
        active & ~contact & (robot.data.root_link_pos_w[:, 0] >= cross_x),
        stable + 1,
        torch.zeros_like(stable),
      )
    )
    newly_successful = active & ~contact & (stable >= stable_steps) & ~success
    success_step[newly_successful] = drive_index + 1
    success.logical_or_(newly_successful)

  for _ in range(settle_steps):
    _step(0.0, None)
  for drive_index in range(drive_steps):
    _step(COMMAND_VX_MPS, drive_index)

  rows: list[dict[str, Any]] = []
  terrain_types_cpu = terrain_types.detach().cpu().tolist()
  for env_id, terrain_type in enumerate(terrain_types_cpu):
    step = int(success_step[env_id].item())
    rows.append({
      "posture_card": str(card["name"]),
      "target_height_m": float(card["height_m"]),
      "target_pitch_rad": float(card["pitch_rad"]),
      "stair_height_m": float(heights[terrain_type]),
      "repeat": int(repeat),
      "env_id": env_id,
      "success": bool(success[env_id].item()),
      "time_to_success_s": (
        None if step < 0 else step / CONTROL_FREQUENCY_HZ
      ),
      "terminated": bool(terminated_ever[env_id].item()),
      "non_wheel_contact": bool(contact_ever[env_id].item()),
      "max_progress_past_face_m": float(max_progress[env_id].item()),
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
        "root_linear_velocity_mps": [
          float(value) for value in
          reset_metadata["root_linear_velocity_mps"][env_id].tolist()
        ],
        "root_angular_velocity_radps": [
          float(value) for value in
          reset_metadata["root_angular_velocity_radps"][env_id].tolist()
        ],
      },
    })
  return rows


def _cell_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
  if not rows:
    raise ValueError("A stair cell cannot be empty.")
  successes = sum(bool(row["success"]) for row in rows)
  terminated = sum(bool(row["terminated"]) for row in rows)
  contacts = sum(bool(row["non_wheel_contact"]) for row in rows)
  success_rate = successes / len(rows)
  termination_rate = terminated / len(rows)
  contact_rate = contacts / len(rows)
  return {
    "trials": len(rows),
    "successes": successes,
    "success_rate": success_rate,
    "terminated_trials": terminated,
    "termination_rate": termination_rate,
    "non_wheel_contact_trials": contacts,
    "non_wheel_contact_rate": contact_rate,
    "passed": (
      success_rate >= SUCCESS_RATE_LIMIT
      and termination_rate == 0.0
      and contact_rate == 0.0
    ),
  }


def aggregate_trials(
  trials: list[dict[str, Any]],
  *,
  heights: tuple[float, ...],
  cards: tuple[dict[str, float | str], ...] = POSTURE_CARDS,
  expected_repeats: int | None = None,
  expected_envs_per_height: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  aggregate_groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
  repeat_groups: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
  for row in trials:
    card = str(row["posture_card"])
    height = float(row["stair_height_m"])
    repeat = int(row["repeat"])
    aggregate_groups[(card, height)].append(row)
    repeat_groups[(card, height, repeat)].append(row)

  cells: list[dict[str, Any]] = []
  repeat_cells: list[dict[str, Any]] = []
  for card in cards:
    card_name = str(card["name"])
    for height in heights:
      key = (card_name, float(height))
      if key not in aggregate_groups:
        raise ValueError(f"Missing stair cell: {key}")
      if expected_repeats is not None and expected_envs_per_height is not None:
        expected_trials = expected_repeats * expected_envs_per_height
        if len(aggregate_groups[key]) != expected_trials:
          raise ValueError(
            f"Stair cell {key} has {len(aggregate_groups[key])} trials; "
            f"expected {expected_trials}."
          )
      cells.append({
        "posture_card": card_name,
        "stair_height_m": float(height),
        **_cell_summary(aggregate_groups[key]),
      })
      repeat_keys = sorted(
        key3 for key3 in repeat_groups if key3[:2] == key
      )
      if expected_repeats is not None:
        observed_repeats = [repeat_key[2] for repeat_key in repeat_keys]
        if observed_repeats != list(range(1, expected_repeats + 1)):
          raise ValueError(
            f"Stair cell {key} has repeats {observed_repeats}; "
            f"expected 1-{expected_repeats}."
          )
      for repeat_key in repeat_keys:
        rows = repeat_groups[repeat_key]
        if (
          expected_envs_per_height is not None
          and len(rows) != expected_envs_per_height
        ):
          raise ValueError(
            f"Stair repeat {repeat_key} has {len(rows)} trials; "
            f"expected {expected_envs_per_height}."
          )
        env_ids = [int(row["env_id"]) for row in rows]
        if len(env_ids) != len(set(env_ids)):
          raise ValueError(f"Stair repeat {repeat_key} has duplicate env_ids.")
        repeat_cells.append({
          "posture_card": card_name,
          "stair_height_m": float(height),
          "repeat": repeat_key[2],
          **_cell_summary(rows),
        })
  return cells, repeat_cells


def classify_results(
  cells: list[dict[str, Any]],
  repeat_cells: list[dict[str, Any]],
  *,
  heights: tuple[float, ...],
  cards: tuple[dict[str, float | str], ...] = POSTURE_CARDS,
) -> dict[str, Any]:
  by_card = {
    str(card["name"]): sorted(
      (cell for cell in cells if cell["posture_card"] == card["name"]),
      key=lambda cell: float(cell["stair_height_m"]),
    )
    for card in cards
  }
  flat_valid = all(bool(rows[0]["passed"]) for rows in by_card.values())

  mixed_repeat = False
  for card in cards:
    card_name = str(card["name"])
    for height in heights:
      values = {
        bool(cell["passed"])
        for cell in repeat_cells
        if cell["posture_card"] == card_name
        and math.isclose(float(cell["stair_height_m"]), height)
      }
      mixed_repeat = mixed_repeat or len(values) > 1

  card_summaries: list[dict[str, Any]] = []
  non_monotonic = False
  for card in cards:
    card_name = str(card["name"])
    rows = by_card[card_name]
    flags = [bool(row["passed"]) for row in rows]
    seen_failure = False
    monotonic = True
    for flag in flags:
      if not flag:
        seen_failure = True
      elif seen_failure:
        monotonic = False
    non_monotonic = non_monotonic or not monotonic
    passing = [float(row["stair_height_m"]) for row in rows if row["passed"]]
    failing = [float(row["stair_height_m"]) for row in rows if not row["passed"]]
    card_summaries.append({
      "posture_card": card_name,
      "monotonic": monotonic,
      "max_passing_height_m": max(passing) if passing else None,
      "first_failing_height_m": min(failing) if failing else None,
    })

  both_fail_height = next(
    (
      height
      for height in heights
      if all(
        not next(
          bool(cell["passed"])
          for cell in by_card[str(card["name"])]
          if math.isclose(float(cell["stair_height_m"]), height)
        )
        for card in cards
      )
    ),
    None,
  )
  both_pass_max = all(
    bool(rows[-1]["passed"]) for rows in by_card.values()
  )

  if not flat_valid:
    classification = "INVALID_FLAT_CONTROL_STOP"
  elif mixed_repeat or non_monotonic:
    classification = "STOP_FOR_VARIANCE_ANALYSIS"
  elif both_pass_max:
    classification = "EXTEND_SWEEP_BEFORE_P3"
  elif both_fail_height is not None:
    classification = "CLASSICAL_DEATH_HEIGHT_BRACKETED"
  elif any(
    summary["first_failing_height_m"] is None
    for summary in card_summaries
  ):
    # At least one card has not failed inside the registered sweep. Extend the
    # sweep before choosing a P3 height; this is not variance evidence.
    classification = "EXTEND_SWEEP_BEFORE_P3"
  else:
    raise RuntimeError("Unable to classify complete monotonic stair results.")

  return {
    "classification": classification,
    "flat_control_valid": flat_valid,
    "mixed_repeat": mixed_repeat,
    "non_monotonic": non_monotonic,
    "posture_cards": card_summaries,
    "p3_candidate_height_m": (
      both_fail_height
      if classification == "CLASSICAL_DEATH_HEIGHT_BRACKETED"
      else None
    ),
  }


def _git_sha(root: Path) -> str:
  return subprocess.check_output(
    ["git", "-c", f"safe.directory={root}", "rev-parse", "HEAD"],
    cwd=root,
    text=True,
  ).strip()


def build_payload(
  *,
  trials: list[dict[str, Any]],
  cells: list[dict[str, Any]],
  repeat_cells: list[dict[str, Any]],
  verdict: dict[str, Any] | None,
  action_cfg: Any,
  protocol: dict[str, Any],
  device: str,
) -> dict[str, Any]:
  mjlab_root = Path(mjlab.__file__).resolve().parents[2]
  return {
    "schema_version": 1,
    "probe": "hybrid_p2_classical_stair_height",
    "evidence_eligible": bool(protocol["evidence_eligible"]),
    "promotion_eligible": False,
    "training_eligible": False,
    "classification": None if verdict is None else verdict["classification"],
    "p3_candidate_height_m": (
      None if verdict is None else verdict["p3_candidate_height_m"]
    ),
    "task": TASK,
    "seed": SEED,
    "git_sha": _git_sha(REPOSITORY_PATH),
    "mjlab_git_sha": _git_sha(mjlab_root),
    "device": device,
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
      "terrain": "pyramid_stairs",
      "terrain_size_m": list(TERRAIN_SIZE_M),
      "terrain_border_width_m": TERRAIN_BORDER_WIDTH_M,
      "step_width_m": STEP_WIDTH_M,
      "platform_width_m": PLATFORM_WIDTH_M,
      "heights_m": list(protocol["heights_m"]),
      "environment_seed": SEED,
      "envs_per_height": int(protocol["envs_per_height"]),
      "repeats": int(protocol["repeats"]),
      "settle_steps": int(protocol["settle_steps"]),
      "drive_steps": int(protocol["drive_steps"]),
      "stable_steps": int(protocol["stable_steps"]),
      "settle_duration_s": (
        int(protocol["settle_steps"]) / CONTROL_FREQUENCY_HZ
      ),
      "maximum_drive_duration_s": (
        int(protocol["drive_steps"]) / CONTROL_FREQUENCY_HZ
      ),
      "required_stable_duration_s": (
        int(protocol["stable_steps"]) / CONTROL_FREQUENCY_HZ
      ),
      "command_vx_mps": COMMAND_VX_MPS,
      "success_rate_limit": SUCCESS_RATE_LIMIT,
      "posture_cards": [dict(card) for card in POSTURE_CARDS],
      "policy_action": [0.0] * 6,
      "commanded_yaw_rate": 0.0,
      "root_reset": {
        "reference": "first_riser_outer_face",
        "start_offset_outside_m": START_OFFSET_M,
        "success_line_inside_m": CROSS_DEPTH_M,
        "root_height_source": "posture_card.height_m",
        "nominal_root_linear_velocity_mps": [0.0, 0.0, 0.0],
        "nominal_root_angular_velocity_radps": [0.0, 0.0, 0.0],
        "x_jitter_abs_m": RESET_X_JITTER_M,
        "y_jitter_abs_m": RESET_Y_JITTER_M,
        "vx_jitter_abs_mps": RESET_VX_JITTER_MPS,
        "pitch_rate_jitter_abs_radps": RESET_PITCH_RATE_JITTER_RADPS,
        "perturbation_seed_formula": "SEED*10000 + card_index*100 + repeat",
      },
      "success_definition": {
        "cross_first_riser": True,
        "continuous_steps_past_success_line": int(protocol["stable_steps"]),
        "termination_rate_required": 0.0,
        "non_wheel_contact_rate_required": 0.0,
      },
    },
    "cells": cells,
    "repeat_cells": repeat_cells,
    "trials": trials,
    "verdict": verdict,
  }


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite output: {args.output}")
  protocol = protocol_for_mode(args.smoke)
  heights = tuple(float(value) for value in protocol["heights_m"])
  cfg = make_stair_env_cfg(heights, int(protocol["envs_per_height"]))
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
      raise ValueError("Official stair probe lacks: " + ", ".join(missing))
    if action_cfg.yaw_calibration_hash is not None:
      raise ValueError("Official zero-yaw probe must not load a yaw artifact.")

  trials: list[dict[str, Any]] = []
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    for card in POSTURE_CARDS:
      for repeat in range(1, int(protocol["repeats"]) + 1):
        print(f"[stair] card={card['name']} repeat={repeat}")
        trials.extend(run_card_repeat(
          env,
          heights=heights,
          card=card,
          repeat=repeat,
          settle_steps=int(protocol["settle_steps"]),
          drive_steps=int(protocol["drive_steps"]),
          stable_steps=int(protocol["stable_steps"]),
        ))
  finally:
    env.close()

  cells, repeat_cells = aggregate_trials(
    trials,
    heights=heights,
    expected_repeats=int(protocol["repeats"]),
    expected_envs_per_height=int(protocol["envs_per_height"]),
  )
  verdict = (
    classify_results(cells, repeat_cells, heights=heights)
    if protocol["evidence_eligible"]
    else None
  )
  payload = build_payload(
    trials=trials,
    cells=cells,
    repeat_cells=repeat_cells,
    verdict=verdict,
    action_cfg=action_cfg,
    protocol=protocol,
    device=args.device,
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
    json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
  )
  print(f"[stair] output={args.output}")
  print(f"[stair] classification={payload['classification']}")


if __name__ == "__main__":
  main()
