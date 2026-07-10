# Hybrid v2 Remote Experiment Protocol

Date: 2026-07-10

This protocol separates commands that are valid now from commands that require
remote-only checkpoint or artifact paths. No checkpoint path or result is
invented in this document.

## Current Status

- Hybrid v2 training has not started.
- `model_122.pt` and std-reset `model_24.pt` are pending remote path and run
  metadata resolution.
- No current three-seed Stage2 gate JSON exists for either candidate.
- No qualified controller artifact, posture-map artifact, GPU result, or Viser
  verdict has been recorded.

The machine-readable candidate state is in
`experiments/hybrid_v2/stage2_candidates.json`.

## Required Legacy Stage2 Evaluation

| Field | Required value |
| --- | --- |
| task | `Mjlab-HopperTrex-Scratch-Stage2-BidirLinSmoothSlew6RewardBalancePrecisionCenter-v0` |
| candidates | `model_122.pt`, std-reset `model_24.pt` |
| fixed linear commands | `-0.07`, `+0.07 m/s` |
| fixed-command environments | 16 |
| fixed-command steps | 3000 |
| warmup | 300 |
| late window | 800 |
| evaluation seeds | 1, 2, 3 |
| episode length for fixed checks | `1.0e9 s` |

Both command directions must pass for all three evaluation seeds. Until then,
neither candidate is promoted.

## Checkout Preflight

Run these from the repository root on the remote machine:

```powershell
git status --short --branch
git rev-parse HEAD
python src/hoppertrex_mjlab/scripts/rsl_rl/evaluate_stage_gate.py --help
```

The evaluator help must list `--seed`, `--fixed-command-lin-x`,
`--fixed-command-num-envs`, `--fixed-command-steps`,
`--fixed-command-warmup-steps`, and `--fixed-command-window-steps`.

Do not evaluate from a dirty worktree. Record the exact SHA in the experiment
manifest before running a gate.

## Resolve Candidate Metadata First

For each filename, locate the exact run directory without assuming that a
same-named checkpoint is unique:

```powershell
Get-ChildItem src/hoppertrex_mjlab/logs/rsl_rl -Filter model_122.pt -File -Recurse
Get-ChildItem src/hoppertrex_mjlab/logs/rsl_rl -Filter model_24.pt -File -Recurse
```

For every match, inspect the adjacent run configuration and record:

- absolute checkpoint path,
- task ID,
- training seed,
- source checkpoint,
- whether action std was reset,
- run directory,
- checkpoint iteration, and
- repository SHA if stored.

Do not run the gate until one unambiguous checkpoint has been selected for each
candidate.

## Validated Gate Runner

The following PowerShell function requires an existing checkpoint file, so it
cannot generate a command with a placeholder path:

```powershell
function Invoke-HopperTrexStage2Gate {
  param(
    [Parameter(Mandatory)]
    [ValidateSet("model_122", "std_reset_model_24")]
    [string] $Candidate,

    [Parameter(Mandatory)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string] $Checkpoint,

    [Parameter(Mandatory)]
    [ValidateSet(1, 2, 3)]
    [int] $Seed,

    [Parameter(Mandatory)]
    [string] $OutputDirectory
  )

  $python = "python"
  $script = "src/hoppertrex_mjlab/scripts/rsl_rl/evaluate_stage_gate.py"
  $task = "Mjlab-HopperTrex-Scratch-Stage2-BidirLinSmoothSlew6RewardBalancePrecisionCenter-v0"
  $output = Join-Path $OutputDirectory "$Candidate.seed$Seed.json"
  New-Item -ItemType Directory -Force $OutputDirectory | Out-Null

  $json = & $python $script `
    --stage 2 `
    --task $task `
    --checkpoint-file (Resolve-Path -LiteralPath $Checkpoint).Path `
    --seed $Seed `
    --fixed-command-lin-x -0.07 0.07 `
    --fixed-command-num-envs 16 `
    --fixed-command-steps 3000 `
    --fixed-command-warmup-steps 300 `
    --fixed-command-window-steps 800 `
    --fixed-command-progress-interval 0 `
    --fixed-command-episode-length-s 1.0e9 `
    --json

  if ($LASTEXITCODE -ne 0) {
    throw "Stage2 evaluator failed for $Candidate seed $Seed."
  }
  $result = $json | ConvertFrom-Json
  $json | Set-Content -LiteralPath $output -Encoding utf8
  if (-not $result.gate_pass) {
    Write-Warning "$Candidate seed $Seed failed the Stage2 gate."
  }
  return $result
}
```

Call the function only after assigning the real path discovered in the previous
step. Run seeds 1, 2, and 3 separately for each candidate. Keep the six raw JSON
files even when a gate fails. In JSON mode the evaluator suppresses collector
progress output, so stdout remains one parseable JSON document.

## Result Recording

For each gate output, verify:

- `task` exactly matches `PrecisionCenter`;
- `seed` matches the requested evaluation seed;
- `checkpoint` resolves to the selected file;
- both fixed-command summaries exist;
- each summary used 16 environments, 3000 steps, 300 warmup steps, and an
  800-step late window through the recorded invocation;
- all hard checks are present; and
- `gate_pass` is not inferred from reward or a partial scenario.

After all three seeds:

- promote only if all three `gate_pass` values are true;
- report per-metric mean, population standard deviation, and failure count;
- retain failed candidates as historical baselines; and
- perform Viser review on any candidate that passes numerically.

If Viser shows sustained rocking, pulses, or drift, add a metric and rerun all
three seeds before promotion.

## Hybrid v2 Remote Order

1. Collect small-disturbance identification data around the nominal standing
   posture at 50 Hz.
2. Run `identify_hybrid_controller.py`; retain LQR only with rank four and
   held-out NRMSE at most 15%, otherwise retain the explicitly labelled PD
   artifact.
3. Run the 60-second Stage0 controller suite for three seeds and Viser.
4. Collect the static two-leg joint sweep and run
   `fit_hybrid_posture_map.py`.
5. Confirm the generated posture artifact reports an
   `all_feasible_grid_rectangle`; rejected non-rectangular sweeps must be
   recollected or narrowed.
6. Train a single-seed, 100-iteration Stage1 probe.
7. Continue to three training seeds only after no structural failure appears.
8. Use the qualified posture artifact before any Stage3 probe.

The planned training caps remain 3000 iterations for Hybrid Stage1-3 and 5000
for Stage4, with Stage5 and the classical-controller/pure-PPO/Hybrid ablations
performed only after Stage4 promotion.
