"""Frozen C1 recovery pilot contracts. No simulator imports in this module."""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

TASK_ID = "HopperTrex-Recovery-C1-Pilot-v1"
INFO_KEY = "recovery_c1_pilot"
MJLAB_SHA = "43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6"
SCHEDULE_HASH = "8fe8548bca85978c164bbd7de39d2d6463cdfd8d7ab91796cf57696b0f64e203"
TRAIN_SEEDS = (11, 12, 13)
DEV_SEEDS = (211, 212, 213, 214)
TEST_SEEDS = (911, 912, 913, 914)
SCALES = (4, 8, 12)
ACTION_SCALES = (0.5, 0.3, 0.035, 0.035, 0.035, 0.035)
ACTION_STD = (0.15, 0.10, 0.05, 0.05, 0.05, 0.05)
ARTIFACTS = {
 "controller": ("docs/experiments/artifacts/c1_schedule_candidate24_1f54968_seed1/c1_schedule.json", "9b21125e7cc48be3ea61e12a67171a855892ad3ced1f54b3176ed979e76224ec"),
 "calibration": ("docs/experiments/artifacts/hybrid_runtime_seed1/velocity_calibration_seed1.json", "ef002d0d622725509b47c8ff40d8af658fd42f705bdeac67ac35bae4458f889d"),
 "yaw_calibration": ("docs/experiments/artifacts/yaw_gpu_3f8a9330b88fa6129d05ce42ac3a8cc835295a6f_seed1/yaw_calibration.json", "123122e75955468dfc475d86ac3f9160b428720fd8e1b90ab614bc1bc0749765"),
 "posture_map": ("docs/experiments/artifacts/c1_posture_requalification_seed1/posture_map_seed1_registered_p032.json", "b8e627f85b53d21dd8d9c26edbe2943151d9bcf9e5864ff998ede5f909118e23"),
 "station_calibration": ("docs/experiments/artifacts/c1_posture_requalification_seed1/station_calibration_seed1.json", "f22a9b66f734004ff14b6586a22a991d527f360806bbbdefe096e9f0474db72a"),
}
PROTOCOL = {
 "schema_version": 1, "task": TASK_ID, "start_date": "2026-10-07",
 "review_date": "2026-10-20", "timezone": "Asia/Shanghai",
 "training_seeds": list(TRAIN_SEEDS), "development_seeds": list(DEV_SEEDS),
 "test_seeds": list(TEST_SEEDS), "kick_scales": list(SCALES),
 "kick_unit": {"world_vx_mps": 0.04, "world_pitch_rate_radps": 0.06},
 "control_hz": 50, "physics_dt_s": 0.005, "decimation": 4,
 "settle_steps": 300, "observe_steps": 300, "healthy_hold_steps": 25,
 "healthy_bands": [0.06, 0.08, 0.015, 0.04],
 "failure_penalty_s": 6.0, "eval_envs": 32, "trials_per_scale": 128,
 "training_envs": 256, "updates": 1000, "steps_per_update": 24,
 "save_every_completed_updates": 100, "selection": "final_1000_only",
 "action_scales": list(ACTION_SCALES), "initial_std": list(ACTION_STD),
 "fresh_actor_critic_optimizer": True, "zero_output_head": True,
 "primary_scale": 8, "continuation_mean_improvement": 0.10,
 "continuation_positive_seeds": 2,
 "gpu_seconds": {"total": 36000, "baseline": 7200, "train": 21600, "evaluate": 7200, "per_seed": 7200},
 "reset_policy": "one_kick_per_reset; any reset/contact/termination latches failure",
 "retention": "registered C1 15-cell caps plus existing integrated capability suite; no threshold changes",
 "artifact_sha256": {k: v[1] for k, v in ARTIFACTS.items()},
 "historical_e6_online_reproduction": False,
}


def canonical(value: Any) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
  return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


PROTOCOL_HASH = digest(PROTOCOL)


def file_sha(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      h.update(chunk)
  return h.hexdigest()


def read_json(path: Path) -> dict:
  value = json.loads(path.read_text(encoding="utf-8-sig"))
  if not isinstance(value, dict):
    raise ValueError(f"Expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temp = path.with_name(path.name + ".tmp")
  temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
  os.replace(temp, path)


def git(repo: Path, *args: str) -> str:
  return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def verify_artifacts(repo: Path) -> dict[str, str]:
  actual = {}
  for key, (relative, expected) in ARTIFACTS.items():
    p = repo / relative
    if not p.is_file() or file_sha(p) != expected:
      raise ValueError(f"Frozen C1 artifact missing or changed: {key}: {p}")
    actual[key] = expected
  return actual


def identity(repo: Path, expected_sha: str, *, local_check: bool = False) -> dict:
  head = git(repo, "rev-parse", "HEAD")
  if len(expected_sha) != 40 or head != expected_sha:
    raise ValueError("Expected full release SHA does not match HEAD")
  dirty = git(repo, "status", "--porcelain")
  if dirty and not local_check:
    raise ValueError("Pilot requires a clean project checkout")
  companion = repo.parent / "mjlab-main"
  if git(companion, "rev-parse", "HEAD") != MJLAB_SHA or git(companion, "status", "--porcelain"):
    raise ValueError("Companion mjlab revision/worktree differs from the frozen runtime")
  return {"task": TASK_ID, "git_sha": head, "mjlab_git_sha": MJLAB_SHA,
          "protocol_sha256": PROTOCOL_HASH, "artifacts": verify_artifacts(repo),
          "local_check": local_check, "dirty": bool(dirty)}


def require_identity(actual: dict, expected: dict) -> None:
  for key in ("task", "git_sha", "mjlab_git_sha", "protocol_sha256", "artifacts"):
    if actual.get(key) != expected.get(key):
      raise ValueError(f"Provenance mismatch: {key}")
  if actual.get("local_check") is not False or actual.get("dirty") is not False:
    raise ValueError("Diagnostic/dirty records cannot authorize formal runs")


def trial_metrics(healthy: list[bool], *, terminated: bool = False,
                  contact: bool = False, reset: bool = False) -> dict:
  if len(healthy) != 300 or any(type(x) is not bool for x in healthy):
    raise ValueError("Recovery trace must contain exactly 300 booleans")
  run = 0
  onset = None
  for i, ok in enumerate(healthy):
    run = run + 1 if ok else 0
    if run == 25 and onset is None:
      onset = i - 24
  hard_failure = terminated or contact or reset
  success = onset is not None and not hard_failure
  return {"success": success, "recovery_time_s": onset / 50 if success else None,
          "penalized_recovery_s": onset / 50 if success else 6.0,
          "terminated": bool(terminated), "non_wheel_contact": bool(contact),
          "automatic_reset": bool(reset), "timeout": onset is None}


def trial_id(split: str, scale: int, seed: int, env_id: int) -> str:
  return f"{split}/scale{scale}/reset{seed}/env{env_id:02d}"


def validate_trials(rows: list[dict], split: str, scales: tuple = SCALES) -> None:
  seeds = DEV_SEEDS if split == "development" else TEST_SEEDS if split == "test" else ()
  expected = {trial_id(split, s, seed, i) for s in scales for seed in seeds for i in range(32)}
  ids = [r["pairing_id"] for r in rows]
  if not expected or len(ids) != len(set(ids)) or set(ids) != expected:
    raise ValueError("Missing, duplicate, or unexpected trial pairing IDs")
  for r in rows:
    for flag in ("success", "terminated", "non_wheel_contact", "automatic_reset", "timeout"):
      if type(r.get(flag)) is not bool:
        raise ValueError("Trial flags must be booleans")
    if r.get("sign") != (1 if r["env_id"] < 16 else -1):
      raise ValueError("Unbalanced or wrong kick direction")
    if r["pairing_id"] != trial_id(split, r["scale"], r["reset_seed"], r["env_id"]):
      raise ValueError("Pairing ID and metadata disagree")
    if not isinstance(r.get("initial_state_sha256"), str) or len(r["initial_state_sha256"]) != 64:
      raise ValueError("Initial state binding missing")
    t = r["penalized_recovery_s"]
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or not 0 <= t <= 6:
      raise ValueError("Invalid recovery time")
    if r["success"] and r.get("recovery_time_s") != t:
      raise ValueError("Successful recovery time and score disagree")
    if not r["success"] and r.get("recovery_time_s") is not None:
      raise ValueError("Failed trial must not carry a success time")
    if not r["success"] and t != 6:
      raise ValueError("Failure must receive the full penalty")
    if r["success"] and (r["terminated"] or r["non_wheel_contact"] or r["automatic_reset"] or r["timeout"]):
      raise ValueError("Failure cannot be counted as success")


def summarize(rows: list[dict]) -> dict:
  result = {}
  for scale in sorted({r["scale"] for r in rows}):
    part = [r for r in rows if r["scale"] == scale]
    values = sorted(r["penalized_recovery_s"] for r in part)
    result[str(scale)] = {"trials": len(part), "mean_penalized_recovery_s": statistics.fmean(values),
      "successes": sum(r["success"] for r in part),
      "hard_failures": sum(r["terminated"] or r["non_wheel_contact"] or r["automatic_reset"] for r in part),
      "p95_penalized_s": values[math.ceil(0.95 * len(values)) - 1],
      "unique_initial_states": len({r["initial_state_sha256"] for r in part})}
  return result


def compare(baseline: list[dict], candidate: list[dict]) -> dict:
  validate_trials(baseline, "test")
  validate_trials(candidate, "test")
  left = {r["pairing_id"]: r for r in baseline}
  for row in candidate:
    b = left[row["pairing_id"]]
    for k in ("initial_state_sha256", "sign", "scale", "reset_seed", "env_id"):
      if row[k] != b[k]:
        raise ValueError(f"Unpaired trial {row['pairing_id']}: {k}")
  bs, cs = summarize(baseline), summarize(candidate)
  return {s: {"baseline": bs[s], "candidate": cs[s],
    "fractional_improvement": (bs[s]["mean_penalized_recovery_s"] - cs[s]["mean_penalized_recovery_s"]) / bs[s]["mean_penalized_recovery_s"] if bs[s]["mean_penalized_recovery_s"] > 0 else None}
    for s in bs}


def validate_checkpoint_record(record: dict, expected: dict, seed: int, config_hash: str) -> None:
  require_identity(record, expected)
  if seed not in TRAIN_SEEDS or record.get("training_seed") != seed:
    raise ValueError("Checkpoint training seed mismatch")
  if record.get("completed_updates") != 1000 or record.get("config_sha256") != config_hash:
    raise ValueError("Checkpoint is not the completed fixed-budget candidate")
  if record.get("zero_initialized_deterministic_mean") is not True or record.get("initial_std") != list(ACTION_STD):
    raise ValueError("Fresh initialization provenance is invalid")


def seal(directory: Path, metadata: dict) -> dict:
  if (directory / "manifest.json").exists():
    raise FileExistsError("Already sealed")
  files = {p.relative_to(directory).as_posix(): file_sha(p)
           for p in sorted(directory.rglob("*")) if p.is_file()}
  result = {**metadata, "files": files, "complete": True}
  write_json(directory / "manifest.json", result)
  return result


def verify_sealed(directory: Path) -> dict:
  m = read_json(directory / "manifest.json")
  if m.get("complete") is not True or not m.get("files"):
    raise ValueError("Incomplete result directory")
  actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
  if actual != set(m["files"]) | {"manifest.json"}:
    raise ValueError("Result file set differs from manifest")
  for rel, sha in m["files"].items():
    target = (directory / rel).resolve()
    if not target.is_relative_to(directory.resolve()) or file_sha(target) != sha:
      raise ValueError(f"Result checksum mismatch: {rel}")
  return m


def require_approval(campaign: Path, expected: dict) -> dict:
  baseline = campaign / "baseline"
  manifest = verify_sealed(baseline)
  require_identity(manifest, expected)
  summary = read_json(baseline / "summary.json")
  if summary.get("baseline_eligible") is not True or manifest.get("device") != "cuda:0":
    raise ValueError("CUDA baseline did not qualify")
  viewer = campaign / "baseline_viewer"
  require_identity(verify_sealed(viewer), expected)
  if read_json(viewer / "summary.json").get("viewer_session_completed") is not True:
    raise ValueError("Baseline viewer session is incomplete")
  approval = read_json(campaign / "authorizations/baseline_review.json")
  require_identity(approval, expected)
  if (approval.get("baseline_manifest_sha256") != file_sha(baseline / "manifest.json")
      or approval.get("viewer_manifest_sha256") != file_sha(viewer / "manifest.json")
      or approval.get("viewer_verdict") != "PASS"
      or not str(approval.get("user_feedback", "")).strip()
      or approval.get("review_kind") != "self_review_with_user_viewer"):
    raise ValueError("Missing/mismatched user viewer and baseline review")
  return approval


def budget_remaining(ledger: dict, bucket: str, seed: int | None) -> float:
  entries = ledger.get("entries", [])
  if not isinstance(entries, list):
    raise ValueError("Malformed budget entries")
  for entry in entries:
    seconds = entry.get("seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
      raise ValueError("Malformed budget duration")
    if entry.get("bucket") not in ("baseline", "train", "evaluate"):
      raise ValueError("Malformed budget bucket")
  used = sum(x["seconds"] for x in entries)
  bucket_used = sum(x["seconds"] for x in entries if x["bucket"] == bucket)
  values = [36000 - used, PROTOCOL["gpu_seconds"][bucket] - bucket_used]
  if bucket == "train":
    values.append(7200 - sum(x["seconds"] for x in entries if x.get("seed") == seed and x["bucket"] == "train"))
  return max(0.0, min(values))


class BudgetLease:
  """Exclusive, persistent fail-closed lease. Crash leaves an active reservation."""
  def __init__(self, campaign: Path, bucket: str, seed: int | None):
    self.path = campaign / "gpu_budget.json"
    self.lock = campaign / ".gpu_budget.lock"
    self.bucket, self.seed = bucket, seed

  def __enter__(self):
    self.path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
      self.ledger = read_json(self.path) if self.path.exists() else {"protocol_sha256": PROTOCOL_HASH, "entries": [], "active": None}
      if self.ledger.get("protocol_sha256") != PROTOCOL_HASH or self.ledger.get("active"):
        raise ValueError("Budget ledger drift/stale active run: review required")
      self.allowed = budget_remaining(self.ledger, self.bucket, self.seed)
      if self.allowed <= 0:
        raise ValueError("GPU budget exhausted")
      self.started = time.monotonic()
      self.ledger["active"] = {"bucket": self.bucket, "seed": self.seed,
                               "started_unix": time.time(), "reserved_seconds": self.allowed}
      write_json(self.path, self.ledger)
      return self
    except BaseException:
      self.lock.unlink(missing_ok=True)
      raise

  def __exit__(self, exc_type, exc, tb):
    cleanup = getattr(self, "on_exit", None)
    if cleanup is not None:
      cleanup()
    self.ledger["entries"].append({"bucket": self.bucket, "seed": self.seed,
      "seconds": max(0.0, time.monotonic() - self.started), "completed_without_supervisor_error": exc_type is None})
    self.ledger["active"] = None
    write_json(self.path, self.ledger)
    self.lock.unlink(missing_ok=True)


def aggregate_comparisons(summaries: list[dict]) -> dict:
  if len(summaries) != 3 or {s.get("training_seed") for s in summaries} != set(TRAIN_SEEDS):
    raise ValueError("Exactly seeds 11/12/13 are required; missing seeds mean incomplete evidence")
  primary, weak = [], []
  hard_clean = True
  for summary in summaries:
    comparison = summary["comparison"]
    if set(comparison) != {"4", "8", "12"}:
      raise ValueError("Pressure-test results must not be dropped")
    primary.append(comparison["8"]["fractional_improvement"])
    weak.append(comparison["4"]["fractional_improvement"])
    for scale in ("4", "8"):
      for arm in ("baseline", "candidate"):
        x = comparison[scale][arm]
        if x["trials"] != 128:
          raise ValueError("Wrong trial count")
        hard_clean &= x["hard_failures"] == 0
  defined = all(x is not None and math.isfinite(x) for x in primary + weak)
  mean_primary = statistics.fmean(primary) if defined else None
  mean_weak = statistics.fmean(weak) if defined else None
  checks = {"primary_defined": defined, "hard_failure_free": hard_clean,
            "retention_passed": all(s.get("retention_passed") is True for s in summaries),
            "two_positive_seeds": defined and sum(x > 0 for x in primary) >= 2,
            "mean_primary_at_least_10pct": defined and mean_primary >= .1,
            "weak_condition_no_mean_regression": defined and mean_weak >= 0}
  return {"training_seeds": list(TRAIN_SEEDS), "primary_improvements": primary,
          "mean_primary_improvement": mean_primary, "mean_weak_improvement": mean_weak,
          "checks": checks, "numerical_continuation_criteria_passed": all(checks.values()),
          "decision": "REQUIRES_REVIEW_AND_USER_VIEWER" if all(checks.values()) else "DO_NOT_AUTO_EXTEND",
          "claim": "Pilot investment screen, not statistical proof or hardware validation"}
