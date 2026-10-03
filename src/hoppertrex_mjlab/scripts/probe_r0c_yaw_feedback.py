"""Non-evidentiary exact-reset probe for bounded lateral feedback.

This probe reuses the rejected R0c-SYNC C0/C1 schedules and their deterministic
16-environment reset construction. It can opt into bounded yaw feedback and/or
roll-to-leg differential feedback, keeps every residual exactly zero, and
records deployable IMU/wheel signals plus simulation-only phase/load truth.
Results are development evidence only; they cannot promote a controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mjlab.envs import ManagerBasedRlEnv

from hoppertrex_mjlab.hybrid.roll_feedback import (
  LEFT_LIFT_BASIS,
  RIGHT_LIFT_BASIS,
  validate_roll_feedback_parameters,
)
from hoppertrex_mjlab.scripts import diagnose_roll_boundary as diag
from hoppertrex_mjlab.scripts import probe_roll_boundary as rb

SCHEMA_VERSION = 1
HEIGHTS_M = (0.0, 0.0025)
CANDIDATE_KEYS = ("c0", "c1")


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_value(*args: str) -> str:
  result = subprocess.run(
    ["git", *args], cwd=rb.REPOSITORY_PATH, check=True,
    capture_output=True, text=True,
  )
  return result.stdout.strip()


def _candidate_map() -> dict[str, Mapping[str, Any]]:
  candidates = diag.r0c_sync_candidates()
  if len(candidates) != 2:
    raise RuntimeError("R0c yaw probe requires the frozen two-candidate set.")
  return dict(zip(CANDIDATE_KEYS, candidates, strict=True))


def _source_reset_map(
  payload: Mapping[str, Any], candidate_key: str,
) -> dict[tuple[int, float], Any]:
  candidate = _candidate_map()[candidate_key]
  definition = diag._schedule_candidate_definition(candidate)
  arms = [
    arm for arm in payload.get("candidates", [])
    if arm.get("candidate_definition") == definition
  ]
  if len(arms) != 1:
    raise ValueError(f"Source result does not contain exactly one {candidate_key} arm.")
  result = {}
  for row in arms[0].get("trials", []):
    identity = (int(row["env_id"]), float(row["stair_height_m"]))
    if identity in result:
      raise ValueError("Source result contains duplicate reset identities.")
    result[identity] = row["root_reset"]
  if len(result) != 16:
    raise ValueError("Source R0c arm must contain 16 exact reset rows.")
  return result


def _validate_resets(
  rows: Sequence[Mapping[str, Any]], expected: Mapping[tuple[int, float], Any],
) -> dict[tuple[int, float], Any]:
  observed = {
    (int(row["env_id"]), float(row["stair_height_m"])): row["root_reset"]
    for row in rows
  }
  if observed.keys() != expected.keys():
    raise ValueError("Probe reset identities differ from the reviewed R0c result.")
  mismatches = [key for key in observed if observed[key] != expected[key]]
  if mismatches:
    raise ValueError(f"Probe exact resets drifted for identities: {mismatches}")
  return observed


def _trace_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
  step_rows = [row for row in rows if float(row["stair_height_m"]) > 0.0]
  traces = [sample for row in step_rows for sample in row["control_trace"]]
  if not traces:
    raise ValueError("Yaw feedback probe produced no lateral control trace.")

  def maximum_abs(name: str) -> float:
    return max(abs(float(sample[name])) for sample in traces)

  return {
    "step_max_abs_roll_rad": maximum_abs("measured_roll_rad"),
    "step_max_abs_roll_rate_radps": maximum_abs("measured_roll_rate_radps"),
    "step_max_abs_roll_feedback_amplitude_rad": maximum_abs(
      "roll_feedback_amplitude_rad"
    ),
    "step_max_abs_yaw_rad": maximum_abs("yaw_rad"),
    "step_max_abs_measured_yaw_rate_radps": maximum_abs(
      "measured_yaw_rate_radps"
    ),
    "step_max_abs_yaw_heading_error_rad": maximum_abs(
      "yaw_heading_error_rad"
    ),
    "step_max_abs_yaw_feedback_radps": maximum_abs("yaw_feedback_radps"),
    "step_max_abs_yaw_differential_radps": maximum_abs(
      "yaw_differential_radps"
    ),
    "step_max_abs_forward_wheel_speed_difference_radps": maximum_abs(
      "forward_wheel_speed_difference_radps"
    ),
    "step_max_abs_wheel_center_x_difference_m": maximum_abs(
      "wheel_center_x_difference_m"
    ),
  }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
  source_path = args.source_result.resolve()
  source = json.loads(source_path.read_text(encoding="utf-8"))
  if source.get("kind") != "r0c_synchronized_reference_rejection_screen":
    raise ValueError("Source result is not an R0c-SYNC artifact.")
  if source.get("matched_reset_perturbations_across_candidates") is not True:
    raise ValueError("Source R0c-SYNC artifact does not certify matched resets.")

  candidates = _candidate_map()
  results = []
  reset_reference: dict[tuple[int, float], Any] | None = None
  for candidate_key in args.candidate:
    candidate = candidates[candidate_key]
    source_resets = _source_reset_map(source, candidate_key)
    reset_override = []
    for env_id in range(2 * diag.R0C_SYNC_ENVS_PER_HEIGHT):
      matches = [
        reset for (source_env_id, _height), reset in source_resets.items()
        if source_env_id == env_id
      ]
      if len(matches) != 1:
        raise ValueError("Source R0c reset rows do not map one-to-one by env id.")
      reset_override.append(matches[0])
    for command_vx in args.command_vx:
      for rate_kp in args.kp:
        for heading_kp in args.heading_kp:
          print(
            f"[r0c-lateral-probe] candidate={candidate_key} "
            f"vx={command_vx:g} rate_kp={rate_kp:g} heading_kp={heading_kp:g} "
            f"roll_kp={args.roll_kp:g} roll_kd={args.roll_kd:g} "
            f"roll_limit={args.roll_limit_mrad:g}mrad"
          )
          cfg = rb.make_roll_boundary_env_cfg(
            HEIGHTS_M,
            diag.R0C_SYNC_ENVS_PER_HEIGHT,
            yaw_feedback_kp=rate_kp,
            yaw_heading_feedback_kp=heading_kp,
            yaw_heading_error_limit_rad=args.heading_error_limit,
            roll_feedback_kp=args.roll_kp,
            roll_feedback_kd=args.roll_kd,
            roll_feedback_max_amplitude_rad=args.roll_limit_mrad * 1.0e-3,
          )
          action_cfg = cfg.actions["hybrid_wheel_leg"]
          if not action_cfg.yaw_calibration_qualified:
            raise RuntimeError("Yaw feedback probe requires the qualified yaw map.")
          if not math.isclose(
            action_cfg.yaw_feedback_kp, rate_kp, rel_tol=0.0, abs_tol=0.0,
          ):
            raise RuntimeError(
              "Yaw-rate gain override did not reach the action config."
            )
          if not math.isclose(
            action_cfg.yaw_heading_feedback_kp,
            heading_kp,
            rel_tol=0.0,
            abs_tol=0.0,
          ):
            raise RuntimeError(
              "Yaw-heading gain override did not reach the action config."
            )
          for name, expected in (
            ("roll_feedback_kp", args.roll_kp),
            ("roll_feedback_kd", args.roll_kd),
            (
              "roll_feedback_max_amplitude_rad",
              args.roll_limit_mrad * 1.0e-3,
            ),
          ):
            if not math.isclose(
              getattr(action_cfg, name), expected, rel_tol=0.0, abs_tol=0.0
            ):
              raise RuntimeError(f"{name} override did not reach the action config.")
          original_cards = rb.POSTURE_CARDS
          rb.POSTURE_CARDS = (candidate["posture_card"],)
          env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
          try:
            rows = rb.run_card_repeat(
              env,
              heights=HEIGHTS_M,
              card=candidate["posture_card"],
              repeat=1,
              settle_steps=rb.OFFICIAL_SETTLE_STEPS,
              drive_steps=rb.OFFICIAL_DRIVE_STEPS,
              stable_steps=rb.OFFICIAL_STABLE_STEPS,
              episode_wide_safety=True,
              diagnostic_continue_after_support_loss=True,
              roll_pose_schedule=candidate["schedule"],
              roll_pose_slew_mode=str(candidate["slew_mode"]),
              require_pure_classical_authority=True,
              record_diagnostic_control_trace=True,
              record_lateral_control_trace=True,
              root_reset_override=reset_override,
              command_vx_mps=command_vx,
            )
          finally:
            env.close()
            rb.POSTURE_CARDS = original_cards
          observed_resets = _validate_resets(rows, source_resets)
          if reset_reference is None:
            reset_reference = observed_resets
          elif observed_resets != reset_reference:
            raise ValueError(
              "Yaw feedback candidates did not preserve matched resets."
            )
          results.append({
            "candidate_key": candidate_key,
            "candidate_definition": diag._schedule_candidate_definition(candidate),
            "command_vx_mps": command_vx,
            "yaw_feedback_kp": rate_kp,
            "yaw_heading_feedback_kp": heading_kp,
            "yaw_heading_error_limit_rad": args.heading_error_limit,
            "roll_feedback_kp": args.roll_kp,
            "roll_feedback_kd": args.roll_kd,
            "roll_feedback_max_amplitude_rad": args.roll_limit_mrad * 1.0e-3,
            "roll_feedback_lift_basis": {
              "left": list(LEFT_LIFT_BASIS),
              "right": list(RIGHT_LIFT_BASIS),
              "composition": (
                "left=+a*left_lift_basis; right=-a*right_lift_basis"
              ),
            },
            "yaw_feedback_authority": {
              "formula": (
                "clip(yaw_feedforward + rate_kp * yaw_rate_error + "
                "heading_kp * bounded_integral(yaw_rate_error), "
                "calibrated envelope)"
              ),
              "residual_required_zero": True,
              "direct_load_feedback": False,
              "simulation_phase_feedback": False,
            },
            "roll_feedback_authority": {
              "formula": (
                "a=clip(roll_kp*roll + roll_kd*roll_rate, +/-limit); "
                "identified differential lift-basis composition"
              ),
              "residual_required_zero": True,
              "direct_load_feedback": False,
              "simulation_phase_feedback": False,
            },
            "summaries": diag.summarize_trials(rows),
            "lateral_trace_summary": _trace_summary(rows),
            "trials": rows,
          })
  return {
    "schema_version": SCHEMA_VERSION,
    "kind": "r0c_lateral_feedback_development_probe",
    "evidence_eligible": False,
    "promotion_eligible": False,
    "reason": "dirty-tree-capable exact-reset development probe; requires a clean preregistered screen",
    "git_sha": _git_value("rev-parse", "HEAD"),
    "project_dirty": bool(_git_value("status", "--porcelain")),
    "device": args.device,
    "source_result": str(source_path),
    "source_result_sha256": _sha256(source_path),
    "matched_reviewed_resets_across_all_candidates": True,
    "heights_m": list(HEIGHTS_M),
    "envs_per_height": diag.R0C_SYNC_ENVS_PER_HEIGHT,
    "repeats": 1,
    "settle_steps": rb.OFFICIAL_SETTLE_STEPS,
    "drive_steps": rb.OFFICIAL_DRIVE_STEPS,
    "stable_steps": rb.OFFICIAL_STABLE_STEPS,
    "candidate_count": len(results),
    "candidates": results,
  }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source-result", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cpu")
  parser.add_argument(
    "--candidate", choices=CANDIDATE_KEYS, nargs="+", default=list(CANDIDATE_KEYS),
  )
  parser.add_argument("--command-vx", type=float, nargs="+", default=[0.07])
  parser.add_argument("--kp", type=float, nargs="+", default=[0.0])
  parser.add_argument("--heading-kp", type=float, nargs="+", default=[0.0])
  parser.add_argument("--heading-error-limit", type=float, default=0.12)
  parser.add_argument("--roll-kp", type=float, default=0.0)
  parser.add_argument("--roll-kd", type=float, default=0.0)
  parser.add_argument("--roll-limit-mrad", type=float, default=0.0)
  args = parser.parse_args(argv)
  if len(set(args.candidate)) != len(args.candidate):
    parser.error("--candidate values must be unique.")
  for name, values in (
    ("--command-vx", args.command_vx),
    ("--kp", args.kp),
    ("--heading-kp", args.heading_kp),
  ):
    if len(set(values)) != len(values):
      parser.error(f"{name} values must be unique.")
    if any(
      not math.isfinite(value)
      or (value <= 0.0 if name == "--command-vx" else value < 0.0)
      for value in values
    ):
      parser.error(f"{name} values are outside their finite non-negative domain.")
  if (
    not math.isfinite(args.heading_error_limit)
    or args.heading_error_limit <= 0.0
  ):
    parser.error("--heading-error-limit must be finite and positive.")
  try:
    validate_roll_feedback_parameters(
      kp=args.roll_kp,
      kd=args.roll_kd,
      max_amplitude_rad=args.roll_limit_mrad * 1.0e-3,
    )
  except ValueError as exc:
    parser.error(str(exc))
  return args


def main(argv: Sequence[str] | None = None) -> None:
  args = parse_args(argv)
  output = args.output.resolve()
  if output.exists():
    raise FileExistsError(f"Probe output already exists: {output}")
  payload = run_probe(args)
  output.parent.mkdir(parents=True, exist_ok=True)
  rb._atomic_write_json(output, payload)
  print(f"[r0c-lateral-probe] output={output}")


if __name__ == "__main__":
  main()

