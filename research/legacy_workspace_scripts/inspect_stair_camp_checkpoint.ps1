[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$Checkpoint,
  [string]$Repo = 'D:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL\.worktrees\p2-classical-upper-bound'
)

# Read-only diagnostic for StairCamp intermediate checkpoints. It neither
# loads an environment nor touches CUDA/checkpoint bytes. (Codex: 2026-08-11)
$ErrorActionPreference = 'Stop'
$CheckpointPath = (Resolve-Path -LiteralPath $Checkpoint).Path
$RepoPath = (Resolve-Path -LiteralPath $Repo).Path
$Python = Join-Path $RepoPath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
  throw "Repository Python is missing: $Python"
}

$Payload = @'
import hashlib
import json
import pathlib
import sys

import torch

checkpoint_path = pathlib.Path(sys.argv[1]).resolve()
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
if not isinstance(checkpoint, dict):
    raise TypeError("checkpoint must be a mapping")
infos = checkpoint.get("infos")
if not isinstance(infos, dict):
    raise TypeError("checkpoint infos must be a mapping")
training = infos["stair_camp_training"]
curriculum = infos["stair_camp_curriculum"]
progress = infos["stair_camp_progress"]
env_state = infos["env_state"]
completed = int(training["completed_updates"])
iteration = int(checkpoint["iter"])
common_step = int(env_state["common_step_counter"])
if iteration + 1 != completed:
    raise ValueError("zero-based checkpoint iteration is inconsistent")
if common_step != completed * 24:
    raise ValueError("common_step_counter is inconsistent")
output = {
    "schema_version": 1,
    "kind": "stair_camp_intermediate_checkpoint_diagnostic",
    "checkpoint": str(checkpoint_path),
    "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    "task": training["task"],
    "git_sha": training["git_sha"],
    "contract_sha256": training["contract_sha256"],
    "training_seed": int(training["training_seed"]),
    "checkpoint_iteration": iteration,
    "completed_updates": completed,
    "common_step_counter": common_step,
    "evaluations": int(curriculum["evaluations"]),
    "upper_height_m": float(curriculum["upper_height_m"]),
    "consecutive_ready_evaluations": int(curriculum["consecutive_ready_evaluations"]),
    "next_evaluation_step": int(curriculum["next_evaluation_step"]),
    "episodes_at_upper_current_window": int(curriculum["episodes_at_upper"]),
    "successes_at_upper_current_window": int(curriculum["successes_at_upper"]),
    "completed_episodes": int(curriculum["completed_episodes"]),
    "triggered_episodes": int(curriculum["triggered_episodes"]),
    "trigger_rate": float(progress["trigger_rate"]),
    "residual_abs_mean": float(progress["residual_abs_mean"]),
    "residual_rms": float(progress["residual_rms"]),
    "residual_abs_max": float(progress["residual_abs_max"]),
}
print(json.dumps(output, indent=2, sort_keys=True))
'@

$PayloadPath = Join-Path ([System.IO.Path]::GetTempPath()) (
  'inspect_stair_camp_' + [System.Guid]::NewGuid().ToString('N') + '.py'
)
[System.IO.File]::WriteAllText(
  $PayloadPath, $Payload, [System.Text.UTF8Encoding]::new($false)
)
try {
  & $Python $PayloadPath $CheckpointPath
  if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint inspection failed with exit code $LASTEXITCODE"
  }
} finally {
  Remove-Item -LiteralPath $PayloadPath -Force -ErrorAction SilentlyContinue
}
