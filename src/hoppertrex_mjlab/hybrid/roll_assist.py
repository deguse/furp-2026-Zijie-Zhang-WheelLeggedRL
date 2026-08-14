"""Fail-closed contract and protocol helpers for leg-only StairRollAssist."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

ROLL_ASSIST_TASK_ID = "HopperTrex-Hybrid-v2-StairRollAssist"
ROLL_ASSIST_ACTION_MASK = (False, False, True, True, True, True)
ROLL_ASSIST_ACTION_SCALES = (0.5, 0.3, 0.035, 0.035, 0.035, 0.035)
ROLL_ASSIST_ACTOR_TERMS = (
  "base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_command",
  "posture_command", "joint_pos", "joint_vel", "controller_baseline",
  "applied_residual",
)
ROLL_ASSIST_CRITIC_TAIL = (
  "step_height", "distance_to_riser", "left_contact_force", "right_contact_force",
)
ROLL_ASSIST_TERM_WIDTHS = {
  "base_lin_vel": 3, "base_ang_vel": 3, "projected_gravity": 3,
  "velocity_command": 3, "posture_command": 2, "joint_pos": 6,
  "joint_vel": 6, "controller_baseline": 2, "applied_residual": 6,
  "step_height": 1, "distance_to_riser": 1, "left_contact_force": 1,
  "right_contact_force": 1,
}
ROLL_ASSIST_ACTOR_WIDTH = 34
ROLL_ASSIST_CRITIC_WIDTH = 38
ROLL_ASSIST_NUM_ENVS = 256
ROLL_ASSIST_FLAT_ENVS = 64
ROLL_ASSIST_STAIR_ENVS = 192
ROLL_ASSIST_STEPS_PER_UPDATE = 24
ROLL_ASSIST_INITIAL_UPDATES = 100
ROLL_ASSIST_MAX_UPDATES = 500
ROLL_ASSIST_SAVE_INTERVAL = 25
ROLL_ASSIST_SWITCH_UPDATE = 25
ROLL_ASSIST_SETTLE_STEPS = 100
ROLL_ASSIST_HEIGHT_STEP_M = 0.0025
ROLL_ASSIST_COMMAND_VX_MPS = 0.07
ROLL_ASSIST_STAIR_POSTURE_HEIGHT_M = 0.3092089487
ROLL_ASSIST_STAIR_POSTURE_PITCH_RAD = 0.016
ROLL_ASSIST_ONLINE_SUCCESS_RATE = 0.80
ROLL_ASSIST_FORMAL_CELL_SUCCESSES = 44
ROLL_ASSIST_BOOTSTRAP_SAMPLES = 10_000
ROLL_ASSIST_TRAINING_INFO_KEY = "roll_assist_training"
ROLL_ASSIST_CURRICULUM_INFO_KEY = "roll_assist_curriculum"
ROLL_ASSIST_PROGRESS_INFO_KEY = "roll_assist_progress"
ROLL_ASSIST_CHECKPOINT_SCHEMA_VERSION = 1
ROLL_ASSIST_EXTENSION_AUTHORIZATION_SCHEMA_VERSION = 1
REWARD_CALIBRATION_SCHEMA_VERSION = 1
R0_PROBE_NAME = "hoppertrex_roll_boundary_r0"
R0_TRAINABLE_CLASSIFICATION = "CLASSICAL_CROLL_BRACKETED"
ROLL_ASSIST_CONTROLLER_SCHEDULE_HASH = (
  "8fe8548bca85978c164bbd7de39d2d6463cdfd8d7ab91796cf57696b0f64e203"
)
ROLL_FIRST_ARTIFACT_SPECS = {
  "controller_path": (
    "docs/experiments/artifacts/c1_schedule_candidate24_1f54968_seed1/c1_schedule.json",
    "9b21125e7cc48be3ea61e12a67171a855892ad3ced1f54b3176ed979e76224ec",
  ),
  "calibration_path": (
    "docs/experiments/artifacts/hybrid_runtime_seed1/velocity_calibration_seed1.json",
    "ef002d0d622725509b47c8ff40d8af658fd42f705bdeac67ac35bae4458f889d",
  ),
  "yaw_calibration_path": (
    "docs/experiments/artifacts/yaw_gpu_3f8a9330b88fa6129d05ce42ac3a8cc835295a6f_seed1/yaw_calibration.json",
    "123122e75955468dfc475d86ac3f9160b428720fd8e1b90ab614bc1bc0749765",
  ),
  "posture_map_path": (
    "docs/experiments/artifacts/c1_posture_requalification_seed1/posture_map_seed1_registered_p032.json",
    "b8e627f85b53d21dd8d9c26edbe2943151d9bcf9e5864ff998ede5f909118e23",
  ),
  "station_calibration_path": (
    "docs/experiments/artifacts/c1_posture_requalification_seed1/station_calibration_seed1.json",
    "f22a9b66f734004ff14b6586a22a991d527f360806bbbdefe096e9f0474db72a",
  ),
}
# R0 and R1 are compared only under this shared contact model.  Keeping the
# override local to the roll-first mainline preserves the historical Stage0--5
# and withdrawn stair-campaign physics contracts.
ROLL_FIRST_WHEEL_CONTACT_SOLREF = (0.020, 1.0)
ROLL_FIRST_WHEEL_CONTACT_SOLIMP = (0.90, 0.95, 0.001)
ROLL_FIRST_PHYSICS_TIMESTEP_S = 0.005
ROLL_FIRST_CONTROL_DECIMATION = 4
ROLL_FIRST_SUBSTEP_SUPPORT_SCOPE = "post_reset_settle_through_success"
ROLL_FIRST_TERRAIN_PROTOCOL = "flat_box_at_zero_else_pyramid_stairs"
ROLL_FIRST_RESET_JOINT_STATE = "registered_posture_map_absolute_targets"
ROLL_FIRST_RESET_ORIENTATION = "posture_card_pitch_quaternion"
ROLL_FIRST_POSTURE_CARDS = (
  {"name": "envelope_center", "height_m": 0.3092089487, "pitch_rad": 0.016},
  {"name": "high_zero_pitch", "height_m": 0.3276857266, "pitch_rad": 0.0},
)
ROLL_FIRST_TASK = "HopperTrex-Hybrid-v2-Stage5"
ROLL_FIRST_MJLAB_GIT_SHA = "43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6"
ROLL_FIRST_ENVS_PER_HEIGHT = 16
ROLL_FIRST_REPEATS = 3
ROLL_FIRST_CELL_PASS_SUCCESSES = 44
ROLL_FIRST_FORMAL_CAP_M = 0.030
ROLL_FIRST_FORMAL_SWEEP_MAXIMA_M = (0.010, 0.020, ROLL_FIRST_FORMAL_CAP_M)
ROLL_FIRST_CONTROL_FREQUENCY_HZ = 50.0
ROLL_FIRST_SETTLE_STEPS = 100
ROLL_FIRST_DRIVE_STEPS = 500
ROLL_FIRST_STABLE_STEPS = 25


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
  encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
  return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def roll_first_artifact_paths(repository_path: Path) -> dict[str, Path]:
  """Resolve and byte-verify the frozen final-C1 artifact stack."""

  resolved = {}
  for name, (relative, expected) in ROLL_FIRST_ARTIFACT_SPECS.items():
    path = (repository_path / relative).resolve()
    if not path.is_file():
      raise FileNotFoundError(f"Missing roll-first artifact {name}: {path}")
    observed = file_sha256(path)
    if observed != expected:
      raise ValueError(
        f"Roll-first artifact {name} SHA drifted: {observed} != {expected}."
      )
    resolved[name] = path
  return resolved


def _finite(value: Any, *, name: str) -> float:
  if isinstance(value, bool):
    raise TypeError(f"{name} must be numeric.")
  result = float(value)
  if not math.isfinite(result):
    raise ValueError(f"{name} must be finite.")
  return result


def _exact_int(value: Any, *, name: str, minimum: int = 0) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise TypeError(f"{name} must be an integer.")
  if value < minimum:
    raise ValueError(f"{name} must be >= {minimum}.")
  return value


def validate_height_pair(hpass_m: float, hnext_m: float) -> tuple[float, float]:
  hpass, hnext = _finite(hpass_m, name="Hpass"), _finite(hnext_m, name="Hnext")
  if hpass <= 0.0:
    raise ValueError("RollAssist requires a positive classical Hpass.")
  if not math.isclose(hnext - hpass, ROLL_ASSIST_HEIGHT_STEP_M, rel_tol=0.0, abs_tol=1e-12):
    raise ValueError("RollAssist permits exactly Hpass -> Hpass + 2.5 mm.")
  for name, value in (("Hpass", hpass), ("Hnext", hnext)):
    level = value / ROLL_ASSIST_HEIGHT_STEP_M
    if not math.isclose(level, round(level), rel_tol=0.0, abs_tol=1e-9):
      raise ValueError(f"{name} is not on the 2.5 mm grid.")
  return hpass, hnext


def _roll_boundary_protocol_shape(
  protocol: Mapping[str, Any],
) -> tuple[tuple[float, ...], tuple[str, ...]]:
  heights_value = protocol.get("heights_m")
  if (
    not isinstance(heights_value, Sequence)
    or isinstance(heights_value, (str, bytes))
  ):
    raise TypeError("RollBoundary protocol has no height grid.")
  heights = tuple(
    _finite(value, name="RollBoundary height") for value in heights_value
  )
  if len(heights) < 2 or not math.isclose(heights[0], 0.0, abs_tol=1e-12):
    raise ValueError("RollBoundary evidence must start with flat and include a next height.")
  if any(
    not math.isclose(
      right - left, ROLL_ASSIST_HEIGHT_STEP_M, rel_tol=0.0, abs_tol=1e-12
    )
    for left, right in pairwise(heights)
  ):
    raise ValueError("RollBoundary evidence is not on the 2.5 mm grid.")
  if heights[-1] > ROLL_FIRST_FORMAL_CAP_M + 1e-12:
    raise ValueError("RollBoundary evidence exceeds the registered formal cap.")
  if not any(
    math.isclose(heights[-1], maximum, rel_tol=0.0, abs_tol=1e-12)
    for maximum in ROLL_FIRST_FORMAL_SWEEP_MAXIMA_M
  ):
    raise ValueError("RollBoundary evidence has an unregistered formal sweep maximum.")
  if not math.isclose(
    _finite(protocol.get("height_step_m"), name="RollBoundary height step"),
    ROLL_ASSIST_HEIGHT_STEP_M,
    rel_tol=0.0,
    abs_tol=1e-12,
  ):
    raise ValueError("RollBoundary height-step contract drifted.")
  for field, expected in (
    ("physics_timestep_s", ROLL_FIRST_PHYSICS_TIMESTEP_S),
    ("control_frequency_hz", ROLL_FIRST_CONTROL_FREQUENCY_HZ),
  ):
    if not math.isclose(
      _finite(protocol.get(field), name=f"RollBoundary {field}"),
      expected,
      rel_tol=0.0,
      abs_tol=1e-12,
    ):
      raise ValueError("RollBoundary cadence contract drifted.")
  if (
    _exact_int(protocol.get("control_decimation"), name="control_decimation")
    != ROLL_FIRST_CONTROL_DECIMATION
  ):
    raise ValueError("RollBoundary cadence contract drifted.")
  for field, expected in (
    ("settle_steps", ROLL_FIRST_SETTLE_STEPS),
    ("drive_steps", ROLL_FIRST_DRIVE_STEPS),
    ("stable_steps", ROLL_FIRST_STABLE_STEPS),
  ):
    if _exact_int(protocol.get(field), name=field) != expected:
      raise ValueError("RollBoundary timing contract drifted.")

  expected_terrain_keys = [
    f"stair_{round(height * 1_000_000):06d}um" for height in heights
  ]
  if protocol.get("terrain_keys") != expected_terrain_keys:
    raise ValueError("RollBoundary terrain-key contract drifted.")
  if not math.isclose(
    _finite(protocol.get("formal_cap_m"), name="RollBoundary formal cap"),
    ROLL_FIRST_FORMAL_CAP_M,
    rel_tol=0.0,
    abs_tol=1e-12,
  ):
    raise ValueError("RollBoundary formal cap drifted.")
  if (
    _exact_int(protocol.get("envs_per_height"), name="envs_per_height")
    != ROLL_FIRST_ENVS_PER_HEIGHT
    or _exact_int(protocol.get("repeats"), name="repeats") != ROLL_FIRST_REPEATS
    or _exact_int(protocol.get("cell_pass_successes"), name="cell_pass_successes")
    != ROLL_FIRST_CELL_PASS_SUCCESSES
    or _exact_int(protocol.get("cell_trials"), name="cell_trials")
    != ROLL_FIRST_ENVS_PER_HEIGHT * ROLL_FIRST_REPEATS
  ):
    raise ValueError("RollBoundary sampling contract drifted.")

  cards_value = protocol.get("posture_cards")
  if (
    not isinstance(cards_value, Sequence)
    or isinstance(cards_value, (str, bytes))
    or len(cards_value) != len(ROLL_FIRST_POSTURE_CARDS)
  ):
    raise ValueError("RollBoundary posture-card contract drifted.")
  names = []
  for observed, expected in zip(cards_value, ROLL_FIRST_POSTURE_CARDS, strict=True):
    if not isinstance(observed, Mapping) or observed.get("name") != expected["name"]:
      raise ValueError("RollBoundary posture-card identity drifted.")
    for field in ("height_m", "pitch_rad"):
      if not math.isclose(
        _finite(observed.get(field), name=f"posture card {field}"),
        float(expected[field]),
        rel_tol=0.0,
        abs_tol=1e-12,
      ):
        raise ValueError("RollBoundary posture-card values drifted.")
    names.append(str(expected["name"]))
  return heights, tuple(names)


def _roll_boundary_cell_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
  successes = sum(row["success"] for row in rows)
  terminations = sum(row["termination"] for row in rows)
  non_wheel = sum(row["non_wheel_contact"] for row in rows)
  airborne = sum(row["bilateral_airborne_ever"] for row in rows)
  count = len(rows)
  return {
    "trials": count,
    "successes": successes,
    "success_rate": successes / count,
    "terminated_trials": terminations,
    "termination_rate": terminations / count,
    "non_wheel_contact_trials": non_wheel,
    "non_wheel_contact_rate": non_wheel / count,
    "bilateral_airborne_trials": airborne,
    "bilateral_airborne_rate": airborne / count,
    "passed": (
      successes >= ROLL_FIRST_CELL_PASS_SUCCESSES
      and terminations == 0
      and non_wheel == 0
      and airborne == 0
    ),
  }


def _recompute_roll_boundary_evidence(
  payload: Mapping[str, Any], protocol: Mapping[str, Any],
) -> dict[str, Any]:
  heights, card_names = _roll_boundary_protocol_shape(protocol)
  trials_value = payload.get("trials")
  if not isinstance(trials_value, list):
    raise TypeError("RollBoundary verdict has no raw trial evidence.")
  expected_total = (
    len(card_names) * len(heights) * ROLL_FIRST_REPEATS * ROLL_FIRST_ENVS_PER_HEIGHT
  )
  if len(trials_value) != expected_total:
    raise ValueError("RollBoundary raw trial count drifted.")

  groups: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
  repeats: dict[tuple[str, float, int], list[Mapping[str, Any]]] = {}
  bool_fields = (
    "success", "termination", "non_wheel_contact", "bilateral_airborne_ever",
  )
  success_time_min = ROLL_FIRST_STABLE_STEPS / ROLL_FIRST_CONTROL_FREQUENCY_HZ
  success_time_max = ROLL_FIRST_DRIVE_STEPS / ROLL_FIRST_CONTROL_FREQUENCY_HZ
  for row in trials_value:
    if not isinstance(row, Mapping):
      raise TypeError("RollBoundary trial must be an object.")
    name = row.get("posture_card")
    height = _finite(row.get("stair_height_m"), name="trial stair height")
    repeat = _exact_int(row.get("repeat"), name="trial repeat", minimum=1)
    _exact_int(row.get("env_id"), name="trial env_id")
    if name not in card_names or height not in heights or repeat > ROLL_FIRST_REPEATS:
      raise ValueError("RollBoundary trial identity is outside the protocol.")
    height_index = heights.index(height)
    if _exact_int(row.get("terrain_index"), name="trial terrain_index") != height_index:
      raise ValueError("RollBoundary trial terrain index drifted.")
    if row.get("terrain_key") != f"stair_{round(height * 1_000_000):06d}um":
      raise ValueError("RollBoundary trial terrain key drifted.")
    expected_card = next(
      card for card in ROLL_FIRST_POSTURE_CARDS if card["name"] == name
    )
    if not math.isclose(
      _finite(row.get("target_height_m"), name="trial target height"),
      float(expected_card["height_m"]),
      rel_tol=0.0,
      abs_tol=1e-12,
    ) or not math.isclose(
      _finite(row.get("target_pitch_rad"), name="trial target pitch"),
      float(expected_card["pitch_rad"]),
      rel_tol=0.0,
      abs_tol=1e-12,
    ):
      raise ValueError("RollBoundary trial posture target drifted.")
    for field in bool_fields:
      if not isinstance(row.get(field), bool):
        raise TypeError(f"RollBoundary trial {field} must be boolean.")
    unsupported = _exact_int(
      row.get("bilateral_unsupported_physics_substeps"),
      name="bilateral unsupported physics substeps",
    )
    success = bool(row["success"])
    time_to_success = row.get("time_to_success_s")
    if success:
      success_time = _finite(time_to_success, name="time_to_success_s")
      if not (
        success_time_min - 1e-12 <= success_time <= success_time_max + 1e-12
        and math.isclose(
          success_time * ROLL_FIRST_CONTROL_FREQUENCY_HZ,
          round(success_time * ROLL_FIRST_CONTROL_FREQUENCY_HZ),
          rel_tol=0.0,
          abs_tol=1e-9,
        )
      ):
        raise ValueError("RollBoundary success time is outside the control-step grid.")
    elif time_to_success is not None:
      raise ValueError("Failed RollBoundary trial retained a success time.")
    if unsupported > 0 and (
      row["bilateral_airborne_ever"] is not True or success
    ):
      raise ValueError("RollBoundary substep failure was not fail-closed.")
    if success and any(
      bool(row[field])
      for field in ("termination", "non_wheel_contact", "bilateral_airborne_ever")
    ):
      raise ValueError("Unsafe RollBoundary trial was marked successful.")
    if _finite(row.get("wheel_residual_abs_max"), name="wheel residual") != 0.0:
      raise ValueError("RollBoundary raw trial used a nonzero wheel residual.")
    groups.setdefault((str(name), height), []).append(row)
    repeats.setdefault((str(name), height, repeat), []).append(row)

  expected_env_ids = set(range(len(heights) * ROLL_FIRST_ENVS_PER_HEIGHT))
  for name in card_names:
    for repeat in range(1, ROLL_FIRST_REPEATS + 1):
      ids = [
        int(row["env_id"])
        for height in heights
        for row in repeats.get((name, height, repeat), [])
      ]
      if len(ids) != len(expected_env_ids) or set(ids) != expected_env_ids:
        raise ValueError("RollBoundary repeat env ids do not cover the vector batch.")

  cells = []
  repeat_cells = []
  for name in card_names:
    for height in heights:
      rows = groups.get((name, height), [])
      if len(rows) != ROLL_FIRST_ENVS_PER_HEIGHT * ROLL_FIRST_REPEATS:
        raise ValueError("RollBoundary raw cell count drifted.")
      cells.append({
        "posture_card": name,
        "stair_height_m": height,
        **_roll_boundary_cell_summary(rows),
      })
      for repeat in range(1, ROLL_FIRST_REPEATS + 1):
        repeat_rows = repeats.get((name, height, repeat), [])
        if len(repeat_rows) != ROLL_FIRST_ENVS_PER_HEIGHT:
          raise ValueError("RollBoundary raw repeat count drifted.")
        env_ids = [int(row["env_id"]) for row in repeat_rows]
        if len(env_ids) != len(set(env_ids)):
          raise ValueError("RollBoundary raw repeat contains duplicate env ids.")
        repeat_cells.append({
          "posture_card": name,
          "stair_height_m": height,
          "repeat": repeat,
          **_roll_boundary_cell_summary(repeat_rows),
        })
  if payload.get("cells") != cells or payload.get("repeat_cells") != repeat_cells:
    raise ValueError("RollBoundary summaries disagree with raw trials.")

  by_key = {
    (str(cell["posture_card"]), float(cell["stair_height_m"])): cell
    for cell in cells
  }
  common = [all(bool(by_key[(name, height)]["passed"]) for name in card_names)
            for height in heights]
  non_monotonic = False
  for flags in [common] + [
    [bool(by_key[(name, height)]["passed"]) for height in heights]
    for name in card_names
  ]:
    seen_failure = False
    for passed in flags:
      if not passed:
        seen_failure = True
      elif seen_failure:
        non_monotonic = True
  pass_index = 0
  for index, passed in enumerate(common):
    if not passed:
      break
    pass_index = index
  flat_valid = common[0]
  hpass = heights[pass_index] if flat_valid else None
  hfail = None if all(common) else heights[pass_index + 1]
  unsafe = False
  if hfail is not None:
    unsafe = any(
      int(by_key[(name, hfail)][field]) > 0
      for name in card_names
      for field in (
        "terminated_trials", "non_wheel_contact_trials", "bilateral_airborne_trials",
      )
    )
  if not flat_valid:
    classification = "INVALID_FLAT_CONTROL_STOP"
  elif non_monotonic:
    classification = "NON_MONOTONIC_STOP"
  elif hfail is not None and math.isclose(hpass or 0.0, 0.0, abs_tol=1e-12):
    classification = "NO_POSITIVE_CLASSICAL_CROLL"
  elif hfail is not None and unsafe:
    classification = "NEXT_HEIGHT_UNSAFE_STOP"
  elif hfail is not None:
    classification = R0_TRAINABLE_CLASSIFICATION
  elif heights[-1] >= ROLL_FIRST_FORMAL_CAP_M - 1e-12:
    classification = "CLASSICAL_CROLL_AT_LEAST_CAP"
  else:
    classification = "EXTEND_ROLL_BOUNDARY_SWEEP"
  verdict = {
    "classification": classification,
    "flat_control_valid": flat_valid,
    "non_monotonic": non_monotonic,
    "max_common_passing_height_m": hpass,
    "first_non_common_height_m": hfail,
    "croll_bracket_m": None if hpass is None else [hpass, hfail],
    "next_height_unsafe": unsafe,
    "training_eligible": classification == R0_TRAINABLE_CLASSIFICATION,
    "common_height_results": [
      {"stair_height_m": height, "both_cards_passed": passed}
      for height, passed in zip(heights, common, strict=True)
    ],
  }
  if payload.get("verdict") != verdict:
    raise ValueError("RollBoundary verdict disagrees with raw trials.")
  for field in (
    "classification", "training_eligible", "max_common_passing_height_m",
    "first_non_common_height_m", "croll_bracket_m",
  ):
    if payload.get(field) != verdict[field]:
      raise ValueError(f"RollBoundary top-level {field} disagrees with raw trials.")
  return verdict


def load_roll_boundary_verdict(
  path: Path, *, expected_git_sha: str | None = None
) -> dict[str, Any]:
  """Load a formal R0 result and accept only the safe trainable branch."""

  source = path.resolve()
  try:
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
  except (OSError, json.JSONDecodeError) as exc:
    raise ValueError("RollBoundary verdict is not valid JSON.") from exc
  if not isinstance(payload, dict):
    raise TypeError("RollBoundary verdict must be an object.")
  if (
    payload.get("probe") != R0_PROBE_NAME
    or payload.get("schema_version") != 1
    or payload.get("task") != ROLL_FIRST_TASK
    or payload.get("promotion_eligible") is not False
  ):
    raise ValueError("Unsupported RollBoundary verdict.")
  if payload.get("evidence_eligible") is not True or payload.get("training_eligible") is not True:
    raise ValueError("RollBoundary verdict does not authorize RollAssist training.")
  if payload.get("classification") != R0_TRAINABLE_CLASSIFICATION:
    raise ValueError("RollBoundary classification is not the safe bracket branch.")
  if payload.get("seed") != 1 or payload.get("device") != "cuda:0":
    raise ValueError("RollBoundary verdict is not the formal seed1 CUDA protocol.")
  git_sha = payload.get("git_sha")
  if not isinstance(git_sha, str) or len(git_sha) != 40:
    raise ValueError("RollBoundary verdict has no valid Git binding.")
  if expected_git_sha is not None and git_sha != expected_git_sha:
    raise ValueError("RollBoundary verdict Git SHA differs from the current checkout.")
  if payload.get("mjlab_git_sha") != ROLL_FIRST_MJLAB_GIT_SHA:
    raise ValueError("RollBoundary verdict MjLab SHA differs from the frozen checkout.")
  runtime = payload.get("runtime")
  if (
    not isinstance(runtime, Mapping)
    or runtime.get("device") != "cuda:0"
    or runtime.get("cuda_available") is not True
    or not isinstance(runtime.get("gpu_name"), str)
    or not runtime["gpu_name"].strip()
  ):
    raise ValueError("RollBoundary verdict lacks formal CUDA runtime provenance.")
  schedule_hash = payload.get("controller_schedule_hash")
  if schedule_hash != ROLL_ASSIST_CONTROLLER_SCHEDULE_HASH:
    raise ValueError("RollBoundary verdict does not bind the frozen C1 schedule.")
  protocol = payload.get("protocol")
  if not isinstance(protocol, Mapping):
    raise TypeError("RollBoundary verdict has no formal protocol contract.")
  if protocol.get("terrain") != ROLL_FIRST_TERRAIN_PROTOCOL:
    raise ValueError("RollBoundary verdict predates the corrected zero-height terrain.")
  if protocol.get("strict_physics_substep_support_required") is not True:
    raise ValueError("RollBoundary verdict did not enforce strict 5 ms support.")
  if protocol.get("strict_physics_substep_support_scope") != ROLL_FIRST_SUBSTEP_SUPPORT_SCOPE:
    raise ValueError("RollBoundary verdict did not cover settle-through-success support.")
  safety = protocol.get("safety")
  if (
    not isinstance(safety, Mapping)
    or safety.get("termination_trials_required") != 0
    or safety.get("non_wheel_contact_trials_required") != 0
    or safety.get("bilateral_airborne_trials_required") != 0
    or safety.get("terminal_state_latched_before_reset") is not True
  ):
    raise ValueError("RollBoundary verdict support safety contract drifted.")
  try:
    observed_solref = tuple(
      _finite(value, name="wheel_contact_solref")
      for value in protocol["wheel_contact_solref"]
    )
    observed_solimp = tuple(
      _finite(value, name="wheel_contact_solimp")
      for value in protocol["wheel_contact_solimp"]
    )
  except (KeyError, TypeError) as exc:
    raise ValueError("RollBoundary verdict has no valid wheel-contact contract.") from exc
  if (
    observed_solref != ROLL_FIRST_WHEEL_CONTACT_SOLREF
    or observed_solimp != ROLL_FIRST_WHEEL_CONTACT_SOLIMP
  ):
    raise ValueError("RollBoundary verdict wheel-contact model drifted.")
  root_reset = protocol.get("root_reset")
  if (
    not isinstance(root_reset, Mapping)
    or root_reset.get("joint_state") != ROLL_FIRST_RESET_JOINT_STATE
    or root_reset.get("orientation") != ROLL_FIRST_RESET_ORIENTATION
  ):
    raise ValueError("RollBoundary verdict predates the posture-consistent reset.")
  hpass, hnext = validate_height_pair(
    payload.get("max_common_passing_height_m"),
    payload.get("first_non_common_height_m"),
  )
  bracket = payload.get("croll_bracket_m")
  if not isinstance(bracket, list) or len(bracket) != 2:
    raise ValueError("RollBoundary verdict has no two-value bracket.")
  if not all(math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
             for value, expected in zip(bracket, (hpass, hnext), strict=True)):
    raise ValueError("RollBoundary bracket fields disagree.")
  verdict = payload.get("verdict")
  if not isinstance(verdict, Mapping) or verdict.get("next_height_unsafe") is not False:
    raise ValueError("RollBoundary next height is not a confirmed safe failure.")
  action_mask = payload.get("action_mask")
  if action_mask != [False] * 6 or payload.get("checkpoint") is not None:
    raise ValueError("RollBoundary baseline was not zero-residual classical control.")
  _recompute_roll_boundary_evidence(payload, protocol)
  return {
    "path": str(source), "file_sha256": file_sha256(source),
    "git_sha": payload.get("git_sha"), "controller_schedule_hash": payload.get("controller_schedule_hash"),
    "hpass_m": hpass, "hnext_m": hnext,
  }


def reward_weights(baseline_positive_reward_rate: float) -> tuple[float, float]:
  b = _finite(baseline_positive_reward_rate, name="baseline_positive_reward_rate")
  if b <= 0.0:
    raise ValueError("Reward calibration B must be positive.")
  return 2.0 * b / ROLL_ASSIST_COMMAND_VX_MPS, 2.0 * b


def build_reward_calibration(*, baseline_positive_reward_rate: float,
                             source_stall_sha256: str,
                             roll_boundary_sha256: str) -> dict[str, Any]:
  progress, success = reward_weights(baseline_positive_reward_rate)
  for name, value in (("source_stall_sha256", source_stall_sha256),
                      ("roll_boundary_sha256", roll_boundary_sha256)):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
      raise ValueError(f"{name} must be lowercase SHA256.")
  payload = {
    "schema_version": REWARD_CALIBRATION_SCHEMA_VERSION,
    "kind": "roll_assist_reward_calibration",
    "baseline_positive_reward_rate": float(baseline_positive_reward_rate),
    "measurement_window_s": 3.0, "command_vx_mps": ROLL_ASSIST_COMMAND_VX_MPS,
    "progress_weight": progress, "success_weight": success,
    "source_stall_sha256": source_stall_sha256,
    "roll_boundary_sha256": roll_boundary_sha256,
    "safe_zero_residual_stall": True,
  }
  payload["calibration_sha256"] = canonical_json_sha256(payload)
  return payload


def validate_reward_calibration(payload: Mapping[str, Any], *, expected_roll_boundary_sha256: str | None = None) -> dict[str, Any]:
  expected_fields = {
    "schema_version", "kind", "baseline_positive_reward_rate", "measurement_window_s",
    "command_vx_mps", "progress_weight", "success_weight", "source_stall_sha256",
    "roll_boundary_sha256", "safe_zero_residual_stall", "calibration_sha256",
  }
  if set(payload) != expected_fields:
    raise ValueError("Reward-calibration schema drifted.")
  if _exact_int(payload["schema_version"], name="schema_version", minimum=1) != REWARD_CALIBRATION_SCHEMA_VERSION:
    raise ValueError("Unsupported reward-calibration schema.")
  if payload["kind"] != "roll_assist_reward_calibration" or payload["safe_zero_residual_stall"] is not True:
    raise ValueError("Reward calibration is not a safe zero-residual stall.")
  if not math.isclose(_finite(payload["measurement_window_s"], name="measurement_window_s"), 3.0):
    raise ValueError("Reward calibration window must be the final 3 seconds.")
  b = _finite(payload["baseline_positive_reward_rate"], name="B")
  progress, success = reward_weights(b)
  if not math.isclose(_finite(payload["command_vx_mps"], name="command_vx_mps"), ROLL_ASSIST_COMMAND_VX_MPS):
    raise ValueError("Reward calibration command drifted.")
  if not math.isclose(_finite(payload["progress_weight"], name="progress_weight"), progress, rel_tol=0.0, abs_tol=1e-12):
    raise ValueError("Reward progress weight is not 2B/0.07.")
  if not math.isclose(_finite(payload["success_weight"], name="success_weight"), success, rel_tol=0.0, abs_tol=1e-12):
    raise ValueError("Reward success weight is not 2B.")
  for name in ("source_stall_sha256", "roll_boundary_sha256"):
    value = payload[name]
    if (
      not isinstance(value, str)
      or len(value) != 64
      or any(character not in "0123456789abcdef" for character in value)
    ):
      raise ValueError(f"{name} must be lowercase SHA256.")
  unsigned = dict(payload)
  observed_hash = str(unsigned.pop("calibration_sha256"))
  if observed_hash != canonical_json_sha256(unsigned):
    raise ValueError("Reward-calibration hash drifted.")
  if expected_roll_boundary_sha256 is not None and payload["roll_boundary_sha256"] != expected_roll_boundary_sha256:
    raise ValueError("Reward calibration is bound to another R0 verdict.")
  return dict(payload)

@dataclass
class RollAssistCurriculumState:
  """Two-level curriculum with immutable cumulative update-25 evidence."""

  hpass_m: float
  hnext_m: float
  switched_to_hnext: bool = False
  decision_made: bool = False
  update25_success_rate: float | None = None
  update25_safe: bool | None = None
  completed_stair_episodes: int = 0
  successful_stair_episodes: int = 0
  termination_episodes: int = 0
  non_wheel_contact_episodes: int = 0
  bilateral_airborne_episodes: int = 0

  def __post_init__(self) -> None:
    self.hpass_m, self.hnext_m = validate_height_pair(self.hpass_m, self.hnext_m)

  @property
  def active_height_m(self) -> float:
    return self.hnext_m if self.switched_to_hnext else self.hpass_m

  @property
  def online_success_rate(self) -> float:
    return self.successful_stair_episodes / max(self.completed_stair_episodes, 1)

  def record_completed_episodes(
    self,
    *,
    completed: int,
    successes: int,
    terminations: int,
    non_wheel_contacts: int,
    bilateral_airborne: int,
  ) -> None:
    """Accumulate pre-reset Hpass episode outcomes until the update-25 decision."""

    if self.decision_made:
      return
    counts = {
      "completed": _exact_int(completed, name="completed"),
      "successes": _exact_int(successes, name="successes"),
      "terminations": _exact_int(terminations, name="terminations"),
      "non_wheel_contacts": _exact_int(non_wheel_contacts, name="non_wheel_contacts"),
      "bilateral_airborne": _exact_int(bilateral_airborne, name="bilateral_airborne"),
    }
    if any(value > counts["completed"] for name, value in counts.items() if name != "completed"):
      raise ValueError("RollAssist episode counters cannot exceed completed episodes.")
    self.completed_stair_episodes += counts["completed"]
    self.successful_stair_episodes += counts["successes"]
    self.termination_episodes += counts["terminations"]
    self.non_wheel_contact_episodes += counts["non_wheel_contacts"]
    self.bilateral_airborne_episodes += counts["bilateral_airborne"]
    if self.successful_stair_episodes > self.completed_stair_episodes:
      raise ValueError("RollAssist cumulative successes exceed completed episodes.")

  def evaluate_update25(
    self,
    *,
    completed_updates: int,
    success_rate: float | None = None,
    terminations: int | None = None,
    non_wheel_contacts: int | None = None,
    bilateral_airborne: int | None = None,
  ) -> bool:
    """Freeze the one permitted height decision from cumulative episode evidence.

    The optional explicit values retain a pure/unit-test interface. Runtime code
    omits them and therefore cannot accidentally substitute the currently active
    episodes for the full updates 0--24 window.
    """

    if completed_updates != ROLL_ASSIST_SWITCH_UPDATE:
      raise ValueError("RollAssist curriculum decision is permitted only at update 25.")
    if self.decision_made:
      raise ValueError("RollAssist update-25 curriculum decision is immutable.")
    supplied = (success_rate, terminations, non_wheel_contacts, bilateral_airborne)
    if any(value is not None for value in supplied):
      if not all(value is not None for value in supplied):
        raise ValueError("Explicit update-25 evidence must provide all four values.")
      rate = _finite(success_rate, name="success_rate")
      unsafe = (
        _exact_int(terminations, name="terminations"),
        _exact_int(non_wheel_contacts, name="non_wheel_contacts"),
        _exact_int(bilateral_airborne, name="bilateral_airborne"),
      )
    else:
      if self.completed_stair_episodes < 1:
        raise ValueError("RollAssist update-25 gate has no completed stair episodes.")
      rate = self.online_success_rate
      unsafe = (
        self.termination_episodes,
        self.non_wheel_contact_episodes,
        self.bilateral_airborne_episodes,
      )
    if rate < 0.0 or rate > 1.0:
      raise ValueError("RollAssist success rate must be in [0, 1].")
    safe = unsafe == (0, 0, 0)
    self.switched_to_hnext = safe and rate >= ROLL_ASSIST_ONLINE_SUCCESS_RATE
    self.decision_made = True
    self.update25_success_rate = rate
    self.update25_safe = safe
    return self.switched_to_hnext

  def state_dict(self) -> dict[str, Any]:
    return {
      "schema_version": 2,
      "hpass_m": self.hpass_m,
      "hnext_m": self.hnext_m,
      "switched_to_hnext": self.switched_to_hnext,
      "decision_made": self.decision_made,
      "update25_success_rate": self.update25_success_rate,
      "update25_safe": self.update25_safe,
      "completed_stair_episodes": self.completed_stair_episodes,
      "successful_stair_episodes": self.successful_stair_episodes,
      "termination_episodes": self.termination_episodes,
      "non_wheel_contact_episodes": self.non_wheel_contact_episodes,
      "bilateral_airborne_episodes": self.bilateral_airborne_episodes,
    }

  @classmethod
  def from_state_dict(cls, payload: Mapping[str, Any]) -> RollAssistCurriculumState:
    expected = {
      "schema_version", "hpass_m", "hnext_m", "switched_to_hnext",
      "decision_made", "update25_success_rate", "update25_safe",
      "completed_stair_episodes", "successful_stair_episodes",
      "termination_episodes", "non_wheel_contact_episodes",
      "bilateral_airborne_episodes",
    }
    if set(payload) != expected or payload.get("schema_version") != 2:
      raise ValueError("RollAssist curriculum state schema drifted.")
    state = cls(
      _finite(payload["hpass_m"], name="hpass_m"),
      _finite(payload["hnext_m"], name="hnext_m"),
    )
    for name in ("switched_to_hnext", "decision_made"):
      if not isinstance(payload[name], bool):
        raise TypeError(f"{name} must be boolean.")
      setattr(state, name, payload[name])
    for name in (
      "completed_stair_episodes", "successful_stair_episodes",
      "termination_episodes", "non_wheel_contact_episodes",
      "bilateral_airborne_episodes",
    ):
      setattr(state, name, _exact_int(payload[name], name=name))
    if state.successful_stair_episodes > state.completed_stair_episodes:
      raise ValueError("RollAssist restored successes exceed completed episodes.")
    for name in (
      "termination_episodes", "non_wheel_contact_episodes",
      "bilateral_airborne_episodes",
    ):
      if getattr(state, name) > state.completed_stair_episodes:
        raise ValueError(f"RollAssist restored {name} exceeds completed episodes.")
    rate = payload["update25_success_rate"]
    state.update25_success_rate = (
      None if rate is None else _finite(rate, name="update25_success_rate")
    )
    safe = payload["update25_safe"]
    if safe is not None and not isinstance(safe, bool):
      raise TypeError("update25_safe must be boolean or null.")
    state.update25_safe = safe
    if state.decision_made:
      if state.update25_success_rate is None or state.update25_safe is None:
        raise ValueError("RollAssist decision lacks frozen update-25 evidence.")
      expected_switch = (
        state.update25_safe
        and state.update25_success_rate >= ROLL_ASSIST_ONLINE_SUCCESS_RATE
      )
      if state.switched_to_hnext != expected_switch:
        raise ValueError("RollAssist restored switch contradicts update-25 evidence.")
    elif state.switched_to_hnext or state.update25_success_rate is not None or state.update25_safe is not None:
      raise ValueError("RollAssist undecided state contains decision evidence.")
    return state


def validate_roll_assist_training_record(
  record: Mapping[str, Any],
  *,
  git_sha: str,
  r0_sha256: str,
  reward_calibration_sha256: str,
  action_scales: Sequence[float] = ROLL_ASSIST_ACTION_SCALES,
) -> int:
  """Validate the tensor-free provenance embedded in a RollAssist checkpoint."""

  expected = {
    "schema_version", "task", "training_seed", "git_sha", "r0_sha256",
    "reward_calibration_sha256", "action_scales", "wheel_residual_exact_zero",
    "zero_initialized_deterministic_mean", "completed_updates",
    "update25_curriculum_decided", "active_height_m",
  }
  if set(record) != expected:
    raise ValueError("RollAssist checkpoint training schema drifted.")
  if _exact_int(record["schema_version"], name="schema_version", minimum=1) != ROLL_ASSIST_CHECKPOINT_SCHEMA_VERSION:
    raise ValueError("RollAssist checkpoint schema version drifted.")
  if record["task"] != ROLL_ASSIST_TASK_ID or record["training_seed"] != 1:
    raise ValueError("RollAssist checkpoint task or seed drifted.")
  if not isinstance(git_sha, str) or len(git_sha) != 40:
    raise ValueError("Expected RollAssist Git SHA is invalid.")
  for name, value in (
    ("r0_sha256", r0_sha256),
    ("reward_calibration_sha256", reward_calibration_sha256),
  ):
    if (
      not isinstance(value, str)
      or len(value) != 64
      or any(character not in "0123456789abcdef" for character in value)
    ):
      raise ValueError(f"Expected {name} is not lowercase SHA256.")
  if record["git_sha"] != git_sha:
    raise ValueError("RollAssist checkpoint Git SHA drifted.")
  if record["r0_sha256"] != r0_sha256:
    raise ValueError("RollAssist checkpoint R0 binding drifted.")
  if record["reward_calibration_sha256"] != reward_calibration_sha256:
    raise ValueError("RollAssist checkpoint reward binding drifted.")
  if not isinstance(record["action_scales"], Sequence) or isinstance(
    record["action_scales"], (str, bytes)
  ):
    raise TypeError("RollAssist checkpoint action scales must be a sequence.")
  if tuple(float(value) for value in record["action_scales"]) != tuple(action_scales):
    raise ValueError("RollAssist checkpoint action scales drifted.")
  if (
    record["wheel_residual_exact_zero"] is not True
    or record["zero_initialized_deterministic_mean"] is not True
  ):
    raise ValueError("RollAssist checkpoint lost its initialization/mask provenance.")
  if record["update25_curriculum_decided"] is not True:
    raise ValueError("RollAssist checkpoint precedes the immutable update-25 decision.")
  active_height = _finite(record["active_height_m"], name="active_height_m")
  active_level = active_height / ROLL_ASSIST_HEIGHT_STEP_M
  if not math.isclose(active_level, round(active_level), rel_tol=0.0, abs_tol=1e-9):
    raise ValueError("RollAssist checkpoint active height is off the 2.5 mm grid.")
  if active_height <= 0.0:
    raise ValueError("RollAssist checkpoint active height must be positive.")
  completed = _exact_int(record["completed_updates"], name="completed_updates", minimum=1)
  # RSL-RL saves after zero-based iterations 0, 25, 50, ... and also at the
  # final iteration.  The embedded completed-update counts are consequently
  # 1, 26, 51, 76, ... plus the permitted 100-update block endpoints.
  if (
    completed > ROLL_ASSIST_MAX_UPDATES
    or not (
      completed % ROLL_ASSIST_SAVE_INTERVAL == 1
      or completed % ROLL_ASSIST_INITIAL_UPDATES == 0
    )
  ):
    raise ValueError("RollAssist checkpoint is not on the actual RSL-RL save grid.")
  return completed


def build_extension_authorization(
  *,
  selected_checkpoint_file: Path,
  selected_checkpoint_sha256: str,
  selected_completed_updates: int,
  target_total_updates: int,
  continuation_evidence_sha256: str,
) -> dict[str, Any]:
  """Bind one passing selected checkpoint to exactly one next 100-update block."""

  selected = _exact_int(
    selected_completed_updates, name="selected_completed_updates", minimum=51
  )
  target = _exact_int(target_total_updates, name="target_total_updates", minimum=1)
  if not (
    selected % ROLL_ASSIST_SAVE_INTERVAL == 1
    or selected % ROLL_ASSIST_INITIAL_UPDATES == 0
  ):
    raise ValueError("Selected RollAssist checkpoint is not on the actual save grid.")
  if target != selected + ROLL_ASSIST_INITIAL_UPDATES or target > ROLL_ASSIST_MAX_UPDATES:
    raise ValueError("RollAssist extension must add exactly 100 updates up to 500.")
  resolved = selected_checkpoint_file.expanduser().resolve()
  for name, value in (
    ("selected_checkpoint_sha256", selected_checkpoint_sha256),
    ("continuation_evidence_sha256", continuation_evidence_sha256),
  ):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
      raise ValueError(f"{name} must be lowercase SHA256.")
  payload = {
    "schema_version": ROLL_ASSIST_EXTENSION_AUTHORIZATION_SCHEMA_VERSION,
    "kind": "roll_assist_extension_authorization",
    "classification": "ROLL_ASSIST_EXTEND_BLOCK",
    "selected_checkpoint_file": str(resolved),
    "selected_checkpoint_sha256": selected_checkpoint_sha256,
    "selected_completed_updates": selected,
    "target_total_updates": target,
    "block_updates": ROLL_ASSIST_INITIAL_UPDATES,
    "continuation_evidence_sha256": continuation_evidence_sha256,
  }
  payload["authorization_sha256"] = canonical_json_sha256(payload)
  return payload


def validate_extension_authorization(payload: Mapping[str, Any]) -> dict[str, Any]:
  expected = {
    "schema_version", "kind", "classification", "selected_checkpoint_file",
    "selected_checkpoint_sha256", "selected_completed_updates",
    "target_total_updates", "block_updates", "continuation_evidence_sha256",
    "authorization_sha256",
  }
  if set(payload) != expected:
    raise ValueError("RollAssist extension authorization schema drifted.")
  unsigned = dict(payload)
  observed_hash = unsigned.pop("authorization_sha256")
  if observed_hash != canonical_json_sha256(unsigned):
    raise ValueError("RollAssist extension authorization hash drifted.")
  if (
    payload["schema_version"] != ROLL_ASSIST_EXTENSION_AUTHORIZATION_SCHEMA_VERSION
    or payload["kind"] != "roll_assist_extension_authorization"
    or payload["classification"] != "ROLL_ASSIST_EXTEND_BLOCK"
    or payload["block_updates"] != ROLL_ASSIST_INITIAL_UPDATES
  ):
    raise ValueError("RollAssist extension authorization contract drifted.")
  rebuilt = build_extension_authorization(
    selected_checkpoint_file=Path(str(payload["selected_checkpoint_file"])),
    selected_checkpoint_sha256=str(payload["selected_checkpoint_sha256"]),
    selected_completed_updates=payload["selected_completed_updates"],
    target_total_updates=payload["target_total_updates"],
    continuation_evidence_sha256=str(payload["continuation_evidence_sha256"]),
  )
  if rebuilt != dict(payload):
    raise ValueError("RollAssist extension authorization is non-canonical.")
  return dict(payload)


def paired_bootstrap_lower_bound(candidate: Sequence[float], baseline: Sequence[float],
                                 *, samples: int = ROLL_ASSIST_BOOTSTRAP_SAMPLES,
                                 seed: int = 1, confidence: float = 0.95) -> dict[str, float | int]:
  candidate_array = np.asarray(candidate, dtype=np.float64)
  baseline_array = np.asarray(baseline, dtype=np.float64)
  if candidate_array.ndim != 1 or candidate_array.shape != baseline_array.shape or candidate_array.size == 0:
    raise ValueError("Paired bootstrap requires nonempty equal-length vectors.")
  if not np.all(np.isfinite(candidate_array)) or not np.all(np.isfinite(baseline_array)):
    raise ValueError("Paired bootstrap input must be finite.")
  if samples != ROLL_ASSIST_BOOTSTRAP_SAMPLES or seed != 1 or not math.isclose(confidence, 0.95):
    raise ValueError("Formal RollAssist bootstrap is pinned to seed1, 10,000 samples, 95%.")
  if candidate_array.size != 96:
    raise ValueError("Formal RollAssist bootstrap requires exactly 96 paired trials.")
  differences = candidate_array - baseline_array
  rng = np.random.default_rng(seed)
  indices = rng.integers(0, differences.size, size=(samples, differences.size))
  means = differences[indices].mean(axis=1)
  return {
    "pairs": int(differences.size), "samples": samples, "seed": seed,
    "mean_delta_m": float(differences.mean()),
    "lower_95_m": float(np.quantile(means, 0.025)),
    "upper_95_m": float(np.quantile(means, 0.975)),
  }


def continuation_gate(*, flat_retention_passed: bool, hpass_card_successes: Sequence[int],
                      hnext_terminations: int, hnext_non_wheel_contacts: int,
                      hnext_bilateral_airborne: int, wheel_residual_abs_max: float,
                      hnext_candidate_successes: int, hnext_baseline_successes: int,
                      paired_candidate_progress: Sequence[float],
                      paired_baseline_progress: Sequence[float]) -> dict[str, Any]:
  if len(hpass_card_successes) != 2:
    raise ValueError("RollAssist Hpass gate requires exactly two posture cards.")
  bootstrap = paired_bootstrap_lower_bound(paired_candidate_progress, paired_baseline_progress)
  checks = {
    "flat_retention": flat_retention_passed is True,
    "hpass_retained": all(_exact_int(value, name="hpass_successes") >= ROLL_ASSIST_FORMAL_CELL_SUCCESSES
                           for value in hpass_card_successes),
    "hnext_safe": all(_exact_int(value, name="unsafe_count") == 0 for value in (
      hnext_terminations, hnext_non_wheel_contacts, hnext_bilateral_airborne,
    )),
    "wheel_residual_exact_zero": _finite(wheel_residual_abs_max, name="wheel_residual_abs_max") == 0.0,
    "positive_success_evidence": (
      _exact_int(hnext_candidate_successes, name="candidate_successes") > 0
      and hnext_candidate_successes > _exact_int(hnext_baseline_successes, name="baseline_successes")
    ),
    "paired_progress_positive": float(bootstrap["lower_95_m"]) > 0.0,
  }
  passed = all(checks.values())
  return {
    "authorized": passed,
    "classification": "ROLL_ASSIST_EXTEND_BLOCK" if passed else "ROLL_ASSIST_REJECT_EXTENSION",
    "checks": checks, "paired_progress_bootstrap": bootstrap,
  }


def final_expansion_gate(*, hnext_card_successes: Sequence[int], safety_gate_passed: bool,
                         wheel_residual_abs_max: float) -> dict[str, Any]:
  if len(hnext_card_successes) != 2:
    raise ValueError("Final RollAssist gate requires exactly two posture cards.")
  passed = (
    safety_gate_passed is True
    and all(_exact_int(value, name="hnext_successes") >= ROLL_ASSIST_FORMAL_CELL_SUCCESSES
            for value in hnext_card_successes)
    and _finite(wheel_residual_abs_max, name="wheel_residual_abs_max") == 0.0
  )
  return {
    "passed": passed,
    "classification": "ROLL_ASSIST_BOUNDARY_EXPANDED" if passed else "ROLL_ASSIST_NO_EXPANSION",
  }


def newest_passer(checkpoints: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
  """Select the newest of K=3 rejection-only envelopes; never score-rank."""

  if len(checkpoints) != 3:
    raise ValueError("RollAssist checkpoint selection requires exactly K=3.")
  iterations = [
    _exact_int(
      checkpoint.get("completed_updates"), name="completed_updates", minimum=1
    )
    for checkpoint in checkpoints
  ]
  if len(set(iterations)) != 3:
    raise ValueError("RollAssist K=3 checkpoints must have distinct updates.")
  newest = max(iterations)
  if newest % ROLL_ASSIST_INITIAL_UPDATES == 0 and newest >= ROLL_ASSIST_INITIAL_UPDATES:
    expected = {newest - 49, newest - 24, newest}
  elif newest % ROLL_ASSIST_SAVE_INTERVAL == 1 and newest >= 151:
    expected = {newest - 50, newest - 25, newest}
  else:
    raise ValueError("RollAssist K=3 newest checkpoint is not a completed block endpoint.")
  if set(iterations) != expected:
    raise ValueError("RollAssist K=3 is not the exact latest three RSL-RL saves.")
  for checkpoint in checkpoints:
    if not isinstance(checkpoint.get("passed"), bool):
      raise TypeError("RollAssist K=3 checkpoint pass status must be boolean.")
  passing = [item for item in checkpoints if item["passed"] is True]
  return max(passing, key=lambda item: int(item["completed_updates"])) if passing else None


__all__ = [name for name in globals() if name.startswith("ROLL_ASSIST_")] + [
  "RollAssistCurriculumState", "build_reward_calibration", "canonical_json_sha256",
  "continuation_gate", "file_sha256", "final_expansion_gate", "load_roll_boundary_verdict",
  "newest_passer", "paired_bootstrap_lower_bound", "reward_weights",
  "roll_first_artifact_paths",
  "build_extension_authorization", "validate_extension_authorization",
  "validate_height_pair", "validate_reward_calibration",
  "validate_roll_assist_training_record",
]
