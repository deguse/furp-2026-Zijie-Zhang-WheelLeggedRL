"""Run a single-seed coarse/fine velocity calibration sweep."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable
from hoppertrex_mjlab.hybrid.calibration import (
  calibration_artifact, candidate_from_envelope, fine_grid, score_candidate,
)

COARSE_GRID = tuple((s, b) for s in (0.80, 0.86, 0.92) for b in (-0.016, -0.012, -0.008))
TASK = "HopperTrex-Hybrid-v2-Stage0"

def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--controller", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--work-dir", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--steps", type=int, default=600)
  parser.add_argument("--warmup-steps", type=int, default=150)
  parser.add_argument("--window-steps", type=int, default=300)
  return parser.parse_args()

def _unique_grid(values: Iterable[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
  return tuple(dict.fromkeys((round(s, 10), round(b, 10)) for s, b in values))

def _candidate_manifest(args: argparse.Namespace, *, gain_hash: str, scale: float, bias: float) -> dict[str, object]:
  candidate_hash = calibration_artifact(
    controller_gain_hash=gain_hash, scale=scale, bias=bias,
    seed=args.seed, candidates=[],
  )["calibration_hash"]
  return {"schema_version": 1, "task": TASK, "controller_gain_hash": gain_hash,
    "calibration_hash": candidate_hash,
    "seed": args.seed, "device": args.device, "num_envs": args.num_envs,
    "steps": args.steps, "warmup_steps": args.warmup_steps,
    "window_steps": args.window_steps, "scale": scale, "bias": bias}

def _is_reusable_candidate(manifest_path: Path, result_path: Path, expected: dict[str, object]) -> bool:
  try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    envelope = json.loads(result_path.read_text(encoding="utf-8"))
    if manifest != expected:
      return False
    if (envelope.get("schema_version") != 2 or envelope.get("suite") != "controller"
        or envelope.get("evaluation_profile") != "screen"
        or envelope.get("task") != TASK or envelope.get("seed") != expected["seed"]
        or envelope.get("controller_gain_hash") != expected["controller_gain_hash"]
        or envelope.get("calibration_hash") != expected["calibration_hash"]):
      return False
    candidate = candidate_from_envelope(envelope,
      scale=float(expected["scale"]), bias=float(expected["bias"]))
    duration = float(expected["steps"]) / 50.0
    return all(abs(float(row.get("duration_s", duration)) - duration) <= 1.0e-9
      for row in candidate.scenarios)
  except (OSError, ValueError, TypeError, json.JSONDecodeError):
    return False

def _run_candidate(args: argparse.Namespace, *, gain_hash: str, scale: float, bias: float, label: str) -> dict[str, object]:
  candidate_dir = args.work_dir / label
  candidate_dir.mkdir(parents=True, exist_ok=True)
  calibration_path = candidate_dir / "calibration.json"
  result_path = candidate_dir / "gate.json"
  log_path = candidate_dir / "gate.log"
  manifest_path = candidate_dir / "manifest.json"
  candidate_artifact = calibration_artifact(
    controller_gain_hash=gain_hash, scale=scale, bias=bias, seed=args.seed,
    candidates=[],
  )
  manifest = _candidate_manifest(args, gain_hash=gain_hash, scale=scale, bias=bias)
  reusable = _is_reusable_candidate(manifest_path, result_path, manifest)
  calibration_path.write_text(
    json.dumps(candidate_artifact, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(f"CANDIDATE {label}: scale={scale:.5f} bias={bias:+.5f}", flush=True)
  if reusable:
    print(f"REUSING {label}: {result_path}", flush=True)
  else:
    result_path.unlink(missing_ok=True)
    env = os.environ.copy()
    env["HOPPERTREX_HYBRID_CONTROLLER_PATH"] = str(args.controller.resolve())
    env["HOPPERTREX_HYBRID_CALIBRATION_PATH"] = str(calibration_path.resolve())
    command = [sys.executable, "-u", "-m", "hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate",
      "--stage", "0", "--profile", "screen",
      "--seed", str(args.seed), "--device", args.device,
      "--num-envs", str(args.num_envs), "--steps", str(args.steps),
      "--warmup-steps", str(args.warmup_steps), "--window-steps", str(args.window_steps),
      "--progress-interval", "200", "--episode-length-s", "1.0e9",
      "--controller-gain-hash", gain_hash, "--output", str(result_path)]
    with log_path.open("w", encoding="utf-8") as log:
      process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1)
      assert process.stdout is not None
      for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
      return_code = process.wait()
    if return_code not in (0, 1):
      raise RuntimeError(
        f"execution_error: {label} exited {return_code}; see {log_path}"
      )
    if not result_path.is_file():
      raise RuntimeError(f"execution_error: {label} exited {return_code} without JSON; see {log_path}")
    if not _is_reusable_candidate(manifest_path, result_path, manifest):
      raise RuntimeError(
        f"execution_error: {label} produced an incompatible gate JSON; see {result_path}"
      )
  envelope = json.loads(result_path.read_text(encoding="utf-8"))
  candidate = candidate_from_envelope(envelope, scale=scale, bias=bias)
  scored = score_candidate(candidate)
  record = {"label": label, "scale": scale, "bias": bias,
    "accepted": scored.accepted,
    "score": scored.score if scored.accepted else None,
    "rejection_reasons": list(scored.rejection_reasons),
    "scenarios": [dict(row) for row in candidate.scenarios]}
  print(f"{'ACCEPTED' if scored.accepted else 'REJECTED'} {label}: score={scored.score:.6f}", flush=True)
  return record

def main() -> None:
  args = parse_args()
  controller = json.loads(args.controller.read_text(encoding="utf-8"))
  gain_hash = str(controller.get("gain_hash", ""))
  if len(gain_hash) != 64:
    raise ValueError("Controller artifact has no valid gain_hash.")
  args.work_dir.mkdir(parents=True, exist_ok=True)
  records: list[dict[str, object]] = []
  for index, (scale, bias) in enumerate(_unique_grid(((1.0, 0.0), *COARSE_GRID))):
    records.append(_run_candidate(args, gain_hash=gain_hash, scale=scale, bias=bias, label=f"coarse_{index:02d}"))
  accepted = [row for row in records if row["accepted"]]
  if not accepted:
    raise RuntimeError("calibration_candidate_rejected: no coarse candidate is stable.")
  coarse_best = min(accepted, key=lambda row: (row["score"], row["scale"], row["bias"]))
  existing = {(row["scale"], row["bias"]) for row in records}
  for index, (scale, bias) in enumerate(fine_grid(float(coarse_best["scale"]), float(coarse_best["bias"]))):
    if (scale, bias) not in existing:
      records.append(_run_candidate(args, gain_hash=gain_hash, scale=scale, bias=bias, label=f"fine_{index:02d}"))
  accepted = [row for row in records if row["accepted"]]
  best = min(accepted, key=lambda row: (row["score"], row["scale"], row["bias"]))
  ranked = sorted(records, key=lambda row: (
    not row["accepted"],
    row["score"] if row["score"] is not None else float("inf"),
    row["scale"], row["bias"],
  ))
  artifact = calibration_artifact(controller_gain_hash=gain_hash,
    scale=float(best["scale"]), bias=float(best["bias"]), seed=args.seed,
    candidates=ranked)
  artifact.update(selection_status="short_sweep_only", stage0_probe_passed=False)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
  print(f"SELECTED scale={best['scale']:.5f} bias={best['bias']:+.5f} score={best['score']:.6f}")
  print(f"Wrote calibration: {args.output}")

if __name__ == "__main__":
  main()
