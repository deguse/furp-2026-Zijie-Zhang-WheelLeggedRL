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

## Local CPU Smoke

The local machine can run MuJoCo configuration checks, short identification
rollouts, and the two-leg/four-joint posture sweep. It does not need an NVIDIA
GPU for these checks. From the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path
$artifactRoot = "experiments/hybrid_v2/local_smoke"
New-Item -ItemType Directory -Force $artifactRoot | Out-Null

python -m hoppertrex_mjlab.scripts.collect_hybrid_identification `
  --output "$artifactRoot/identification_smoke.npz" `
  --device cpu `
  --num-envs 2 `
  --steps 12 `
  --warmup-steps 2 `
  --hold-steps 2 `
  --progress-interval 12 `
  --seed 1

python -m hoppertrex_mjlab.scripts.collect_hybrid_posture_sweep `
  --output "$artifactRoot/posture_sweep_smoke.npz" `
  --allow-unqualified-controller `
  --device cpu `
  --hip-range -0.02 0.02 `
  --knee-range -0.02 0.02 `
  --hip-points 2 `
  --knee-points 2 `
  --ramp-steps 2 `
  --settle-steps 2 `
  --sample-steps 3 `
  --progress-interval 7 `
  --seed 1
```

The second command scans symmetric coordinates over the physical joints
`thigh_left_01`, `thigh_right_01`, `knee_left`, and `knee_right`. The
`--allow-unqualified-controller` result is a smoke artifact only. In
particular, a short sample may exceed the formal 80% actuator-load limit and
must not be used to define the training envelope.

## Remote Hybrid v2 Bootstrap

Use one clean checkout and keep all generated files under a SHA-specific
artifact directory:

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path
$sha = git rev-parse --short=12 HEAD
$artifactRoot = "experiments/hybrid_v2/artifacts/$sha"
New-Item -ItemType Directory -Force $artifactRoot | Out-Null
```

Before collecting artifacts, verify the codex/hybrid-v2 branch, record the
full HEAD SHA, and confirm that the selected Python can import MjLab, NumPy,
SciPy, and PyTorch. The repository uses an editable sibling mjlab-main; a
checkout without that sibling or an equivalent installed MjLab environment is
not ready for remote execution.

### 1. Identification data

The collector applies deterministic PRBS excitation to the balance residual
head and records the actual post-slew, post-saturation signed wheel target:

```powershell
$identification = "$artifactRoot/identification_seed1.npz"
python -m hoppertrex_mjlab.scripts.collect_hybrid_identification `
  --output $identification `
  --device cuda:0 `
  --num-envs 32 `
  --steps 2500 `
  --warmup-steps 250 `
  --hold-steps 5 `
  --balance-amplitude 0.35 `
  --heldout-fraction 0.20 `
  --progress-interval 250 `
  --seed 1
```

Keep both the NPZ and its same-named JSON sidecar. The state order in the
artifact must be:

```text
pitch, pitch_rate, vx_error, signed_wheel_speed_error
```

### 2. Identify and qualify the controller

```powershell
$controller = "$artifactRoot/controller_seed1.json"
python -m hoppertrex_mjlab.scripts.identify_hybrid_controller `
  --input $identification `
  --output $controller `
  --q-diag 20.0 2.0 4.0 0.5 `
  --r 1.0 `
  --pd-gain 8.0 1.0 3.0 0.2 `
  --nrmse-limit 0.15

$controllerPayload = Get-Content -LiteralPath $controller -Raw |
  ConvertFrom-Json
if ($controllerPayload.controller_type -ne "lqr") {
  throw "Identification produced labelled PD fallback; do not start Stage1."
}
```

The current executable training path requires a qualified LQR: controllability
rank four and held-out one-step NRMSE no greater than 15%. The identification
script still emits an explicitly labelled PD artifact when either check fails,
but that PD artifact is not accepted by the formal posture collector or Stage1
bootstrap. A PD can enter training only after a separate Stage0 behavioral
qualification-provenance mechanism is implemented; it must never be reported
as LQR.

Set the qualified controller for every following process:

```powershell
$env:HOPPERTREX_HYBRID_CONTROLLER_PATH = (
  Resolve-Path -LiteralPath $controller
).Path
$gainHash = $controllerPayload.gain_hash
```

### 3. Stage0 controller gate

Stage0 has no PPO checkpoint. Run each evaluation seed independently and keep
failed JSON files:

```powershell
$stage0Root = "$artifactRoot/stage0_gate"
New-Item -ItemType Directory -Force $stage0Root | Out-Null
foreach ($seed in 1, 2, 3) {
  python -m hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate `
    --stage 0 `
    --seed $seed `
    --device cuda:0 `
    --num-envs 16 `
    --steps 3000 `
    --warmup-steps 300 `
    --window-steps 800 `
    --progress-interval 500 `
    --episode-length-s 1.0e9 `
    --controller-gain-hash $gainHash `
    --output "$stage0Root/seed$seed.json"
  if ($LASTEXITCODE -ne 0) {
    Write-Warning "Stage0 seed $seed failed; retain its JSON."
  }
}

python -m hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate `
  --aggregate-input `
    "$stage0Root/seed1.json" `
    "$stage0Root/seed2.json" `
    "$stage0Root/seed3.json" `
  --output "$stage0Root/aggregate.json"
if ($LASTEXITCODE -ne 0) {
  throw "Stage0 did not pass all three seeds; do not collect a formal sweep."
}
```

The aggregate must report suite `controller`, task
`HopperTrex-Hybrid-v2-Stage0`, seeds `[1, 2, 3]`, the same gain hash, and
`gate_pass: true`. Complete the Viser inspection before promotion.

### 4. Formal two-leg posture sweep

This is the same sweep that can run locally, but the formal run uses the
qualified controller, a 7 by 7 grid, and full settling windows:

```powershell
$postureSweep = "$artifactRoot/posture_sweep_seed1.npz"
python -m hoppertrex_mjlab.scripts.collect_hybrid_posture_sweep `
  --output $postureSweep `
  --controller-path $controller `
  --device cuda:0 `
  --hip-range -0.18 0.18 `
  --knee-range -0.18 0.18 `
  --hip-points 7 `
  --knee-points 7 `
  --ramp-steps 100 `
  --settle-steps 200 `
  --sample-steps 100 `
  --progress-interval 100 `
  --seed 1

$postureMap = "$artifactRoot/posture_map_seed1.json"
python -m hoppertrex_mjlab.scripts.fit_hybrid_posture_map `
  --input $postureSweep `
  --output $postureMap `
  --joint-margin 0.10 `
  --load-limit 0.80 `
  --inward-fraction 0.10 `
  --pitch-limit 0.08

$env:HOPPERTREX_HYBRID_POSTURE_MAP_PATH = (
  Resolve-Path -LiteralPath $postureMap
).Path
```

Inspect the sweep sidecar for zero invalid points and review the fitted map's
feasible count and shrunken height/pitch envelope. For collector artifacts, the
fitter selects an all-feasible rectangle in the two-leg joint sweep and then
inscribes the command range inside the convex hull of its measured height/pitch
points. Stage3 must not start unless the map is qualified and its feasible
envelope is non-empty. The fitter requires the same-named JSON sidecar and
rejects smoke sweeps collected with the unqualified fallback controller. The
fitted map records the source controller gain hash; Stage3-5 reject it when a
different controller is loaded.

### 5. Create the Stage1 training origin

The bootstrap creates a fresh six-output policy, zeros every actor output head,
sets the per-dimension std to `[0.15, 0.10, 0.05, 0.05, 0.05, 0.05]`, clears
optimizer state, and records controller provenance:

```powershell
$bootstrapRun = "hybrid_v2_stage1_bootstrap_seed1"
python -m hoppertrex_mjlab.scripts.rsl_rl.bootstrap_hybrid_stage1 `
  --controller-path $controller `
  --seed 1 `
  --device cuda:0 `
  --output-run $bootstrapRun
```

Confirm that both files exist before training:

```powershell
$bootstrapRoot = "src/hoppertrex_mjlab/logs/rsl_rl/hoppertrex_balance/$bootstrapRun"
Get-Item `
  "$bootstrapRoot/model_0.pt", `
  "$bootstrapRoot/bootstrap_provenance.json"
```

### 6. Stage1 100-iteration probe

```powershell
python src/hoppertrex_mjlab/scripts/rsl_rl/train.py `
  HopperTrex-Hybrid-v2-Stage1 `
  --env.scene.num-envs 256 `
  --agent.max-iterations 100 `
  --agent.save-interval 25 `
  --agent.seed 1 `
  --agent.resume True `
  --agent.load-run ".*hybrid_v2_stage1_bootstrap_seed1.*" `
  --agent.load-checkpoint "model_0.pt" `
  --agent.run-name "hybrid_v2_stage1_probe_seed1"
```

The training entry point rejects Hybrid Stage1-5 when the qualified controller
environment variable is missing. Stage3-5 additionally require the qualified
posture-map variable, and Stage0 cannot be launched as PPO training.

Do not launch the three-seed or 3000-iteration run merely because training
starts. First evaluate the probe with the Stage1 linear suite, inspect it in
Viser, and convert any persistent oscillation, pulse, or drift into a recorded
metric. Only then promote Stage1 to the full training budget.
