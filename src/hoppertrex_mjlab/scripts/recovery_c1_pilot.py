"""Public, standard-library supervisor for the frozen C1 recovery pilot."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timedelta, timezone
import os
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from pathlib import Path

from hoppertrex_mjlab.hybrid.recovery_pilot import (
  BudgetLease, PROTOCOL, TRAIN_SEEDS, file_sha, identity, read_json,
  require_approval, require_identity, seal, verify_sealed, write_json,
)

REPO = Path(__file__).resolve().parents[3]


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--phase", choices=("Validate", "Baseline", "Train", "Evaluate", "Package"), required=True)
  p.add_argument("--expected-git-sha", required=True)
  p.add_argument("--campaign-root", type=Path, required=True)
  p.add_argument("--seed", type=int, choices=TRAIN_SEEDS)
  p.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
  p.add_argument("--run-directory", type=Path, help="A sealed result under campaign-root to package")
  p.add_argument("--local-check", action="store_true", help="CPU Validate only, never training-eligible")
  p.add_argument("--viewer", action="store_true", help="After Baseline/Evaluate: separate budgeted 3000-step Viser session")
  p.add_argument("--cpu-smoke", action="store_true", help="CPU Validate only: two implementation-test updates")
  a = p.parse_args(argv)
  if (a.local_check or a.cpu_smoke or a.device == "cpu") and (a.phase != "Validate" or a.device != "cpu"):
    p.error("CPU/local checks are restricted to Validate")
  if a.phase in ("Train", "Evaluate") and a.seed is None:
    p.error("Train/Evaluate require one explicit seed")
  if a.phase not in ("Train", "Evaluate") and a.seed is not None:
    p.error("This phase does not take a training seed")
  if a.viewer and a.phase not in ("Baseline", "Evaluate"):
    p.error("Viewer is only available after Baseline/Evaluate")
  if a.phase == "Package" and a.run_directory is None:
    p.error("Package requires run-directory")
  return a


def run_name(phase, seed):
  return {"Validate": "validate", "Baseline": "baseline", "Train": f"train_seed{seed}",
          "Evaluate": f"evaluate_seed{seed}"}[phase]


def require_prior_review(campaign, seed, expected):
  if seed == 11:
    return
  previous = seed - 1
  directory = campaign / f"evaluate_seed{previous}"
  require_identity(verify_sealed(directory), expected)
  review = read_json(campaign / f"authorizations/seed{previous}_review.json")
  require_identity(review, expected)
  if review.get("evaluation_manifest_sha256") != file_sha(directory / "manifest.json") or review.get("decision") != "CONTINUE":
    raise ValueError("Previous seed has not been reviewed for continuation")
  viewer = campaign / f"evaluate_seed{previous}_viewer"
  require_identity(verify_sealed(viewer), expected)
  if (review.get("viewer_manifest_sha256") != file_sha(viewer / "manifest.json")
      or review.get("viewer_verdict") != "PASS" or not str(review.get("user_feedback", "")).strip()):
    raise ValueError("Previous seed viewer confirmation is missing")


def package_run(campaign, directory, expected):
  directory = directory.resolve()
  if not directory.is_relative_to(campaign) or directory == campaign or directory.name.startswith("."):
    raise ValueError("Package only an explicit complete run below campaign-root")
  m = verify_sealed(directory)
  require_identity(m, expected)
  destination = campaign / (directory.name + ".zip")
  if destination.exists():
    raise FileExistsError(f"Package exists: {destination}")
  temp = campaign / ("." + destination.name + ".incomplete")
  if temp.exists():
    raise FileExistsError(temp)
  with zipfile.ZipFile(temp, "x", compression=zipfile.ZIP_DEFLATED) as archive:
    for p in sorted(directory.rglob("*")):
      if p.is_file():
        archive.write(p, directory.name + "/" + p.relative_to(directory).as_posix())
  # Verify compressed bytes before making the package visible.
  with zipfile.ZipFile(temp) as archive:
    if archive.testzip() is not None:
      raise ValueError("ZIP CRC verification failed")
  temp.rename(destination)
  checksum = file_sha(destination)
  destination.with_suffix(".zip.sha256").write_text(checksum + "  " + destination.name + "\n", encoding="utf-8")
  return destination


def stop_child(proc):
  if proc.poll() is not None:
    return
  if os.name == "nt":
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)
  else:
    proc.kill()
  proc.wait(timeout=30)


def launch(args):
  campaign = args.campaign_root.resolve()
  if campaign == REPO or campaign.is_relative_to(REPO) or REPO.is_relative_to(campaign):
    raise ValueError("Campaign must be a separate directory, not the checkout or its ancestor")
  if args.device == "cuda:0" and args.phase != "Package":
    today = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
    if not PROTOCOL["start_date"] <= today <= PROTOCOL["review_date"]:
      raise ValueError("Outside the approved pilot calendar window; user review required")
  ident = identity(REPO, args.expected_git_sha, local_check=args.local_check)
  campaign.mkdir(parents=True, exist_ok=True)
  binding = campaign / "campaign.json"
  if binding.exists():
    previous = read_json(binding)
    if previous != ident:
      raise ValueError("Campaign is bound to another release or diagnostic mode")
  else:
    write_json(binding, ident)
  if args.phase == "Package":
    print("PACKAGE=" + str(package_run(campaign, args.run_directory, ident)))
    return
  if args.phase == "Baseline":
    require_identity(verify_sealed(campaign / "validate"), ident)
    if read_json(campaign / "validate/runtime.json")["device"] != "cuda:0":
      raise ValueError("M0 CUDA validation is required")
  if args.phase in ("Train", "Evaluate"):
    require_approval(campaign, ident)
  if args.phase == "Evaluate":
    require_identity(verify_sealed(campaign / f"train_seed{args.seed}"), ident)
  if args.phase == "Train":
    require_prior_review(campaign, args.seed, ident)
  name = run_name(args.phase, args.seed)
  if args.viewer:
    require_identity(verify_sealed(campaign / name), ident)
  final = campaign / (name + "_viewer" if args.viewer else name)
  if final.exists():
    raise FileExistsError(f"Refusing to overwrite {final}")
  # Even failed attempts cannot be silently rerun in the same campaign.
  attempts = list(campaign.glob("." + final.name + ".incomplete.*"))
  if attempts:
    raise ValueError("A failed/incomplete attempt exists; diagnose it before an explicitly reviewed retry")
  temporary = campaign / ("." + final.name + ".incomplete." + uuid.uuid4().hex)
  temporary.mkdir()
  write_json(temporary / "protocol.json", PROTOCOL)
  bucket = "train" if args.phase == "Train" else "evaluate" if args.phase == "Evaluate" else "baseline"
  gpu = args.device == "cuda:0"
  lease_context = BudgetLease(campaign, bucket, args.seed) if gpu else contextlib.nullcontext()
  env = os.environ.copy()
  env["PYTHONPATH"] = os.pathsep.join((str(REPO / "src"), str(REPO / "src/hoppertrex_mjlab")))
  env["PYTHONDONTWRITEBYTECODE"] = "1"
  env["PYTHONUNBUFFERED"] = "1"
  env["TORCHINDUCTOR_CACHE_DIR"] = str(campaign / ".cache/torch")
  proc = None
  try:
    with lease_context as lease:
      token = uuid.uuid4().hex
      if gpu:
        lease.ledger["active"]["token"] = token
        write_json(lease.path, lease.ledger)
      request = {"phase": args.phase, "seed": args.seed, "output": str(temporary),
        "campaign": str(campaign), "device": args.device, "expected_sha": args.expected_git_sha,
        "local_check": args.local_check, "cpu_smoke": args.cpu_smoke, "lease_token": token, "viewer": args.viewer}
      write_json(temporary / "request.json", request)
      command = [sys.executable, "-B", "-m", "hoppertrex_mjlab.scripts.recovery_c1_runtime", "--request", str(temporary / "request.json")]
      timeout = lease.allowed if gpu else 900
      with (temporary / "console.log").open("w", encoding="utf-8") as console:
        proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=console, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        print(f"RUNNING={temporary}\nPID={proc.pid}\nLIMIT_SECONDS={timeout:.1f}", flush=True)
        start = time.monotonic()
        if gpu:
          lease.on_exit = lambda: stop_child(proc)
        deadline = lease.started + timeout if gpu else start + timeout
        while proc.poll() is None:
          if time.monotonic() >= deadline:
            stop_child(proc)
            raise TimeoutError("Fixed budget expired; incomplete files preserved")
          time.sleep(0.5)
        if proc.returncode != 0:
          raise RuntimeError(f"Worker exited {proc.returncode}; see {temporary / 'console.log'}")
      metadata = read_json(temporary / "worker_result.json")
      metadata["wall_seconds"] = time.monotonic() - start
    if gpu:
      shutil.copyfile(campaign / "gpu_budget.json", temporary / "budget_snapshot.json")
    seal(temporary, metadata)
    if temporary.resolve().parent != campaign or final.resolve().parent != campaign:
      raise ValueError("Resolved result paths escaped the campaign directory")
    temporary.rename(final)
    print("RESULT=" + str(final))
    print("No automatic next phase. Return this result for review.")
  except BaseException as exc:
    if proc is not None:
      stop_child(proc)
    write_json(temporary / "failure.json", {"type": type(exc).__name__, "message": str(exc),
      "training_authorized_by_failure": False, "complete": False})
    raise


def main():
  launch(parse_args())


if __name__ == "__main__":
  main()
