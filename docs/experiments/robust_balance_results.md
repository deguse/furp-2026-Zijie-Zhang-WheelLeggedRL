# HopperTrex Robust Balance Results

## 2026-06-27 - Robust L1 Passed

### Summary

The fixed-leg two-wheel balance task has passed the first robust stationary balance stage. The policy balances with zero velocity command and small reset disturbances while keeping support on the two main wheels only.

Current conclusion:

```text
Fixed legs + zero command + small reset disturbance -> stable two-wheel stationary balance.
```

### Robust L1 Task

Task id:

```text
Mjlab-HopperTrex-Balance-Robust-v0
```

Reset disturbances:

```text
roll/pitch: ±2 deg
root x velocity: ±0.05 m/s
root roll/pitch angular velocity: ±0.10 rad/s
```

Not included:

```text
no forward velocity command
no lateral velocity command
no yaw command
no leg action
no terrain
no continuous push force
```

### Observed Acceptance

Training metrics from the passed robust run included:

```text
Mean episode length: 500.00
Episode_Termination/root_too_low: 0.0000
Episode_Termination/non_wheel_ground_contact: 0.0000
Episode_Termination/bad_orientation: 0.0000
Episode_Termination/nan_detection: 0.0000
Episode_Reward/wheel_ground_contact: about 0.95
Episode_Reward/clean_wheel_support: about 3.81 / 4.00
```

Viewer acceptance:

```text
only the two main wheels support the robot
thigh/calf/chassis do not touch the ground
reset tests recover to upright balance
no low-posture non-wheel support
```

### Runs

| Stage | Run name | Seed | Status | Notes |
| --- | --- | --- | --- | --- |
| Robust L1 | `robust_init_seed1` | 1 | Passed | Metrics and viewer passed. |
| Robust L1 | `robust_init_seed2` | 2 | Passed | User-reported passed. |
| Robust L1 | `robust_init_seed3` | 3 | Passed | User-reported passed. |

### Checkpoint Archival

Checkpoints should not be committed to Git. Archive them on the lab PC that produced the training run:

```powershell
cd C:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL
New-Item -ItemType Directory -Force -Path C:\mjlab_workspace\trained_models | Out-Null

$run = Get-ChildItem src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance -Directory |
  Where-Object { $_.Name -like "*robust_init_seed1*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$ckpt = Get-ChildItem $run.FullName -Filter "model_*.pt" |
  Sort-Object { [int]($_.BaseName -replace "model_","") } -Descending |
  Select-Object -First 1

Copy-Item $ckpt.FullName "C:\mjlab_workspace\trained_models\robust_l1_seed1_$($ckpt.BaseName)_20260627.pt"
Copy-Item (Join-Path $run.FullName "params\agent.yaml") "C:\mjlab_workspace\trained_models\robust_l1_seed1_agent_20260627.yaml"
Copy-Item (Join-Path $run.FullName "params\env.yaml") "C:\mjlab_workspace\trained_models\robust_l1_seed1_env_20260627.yaml"
```

Repeat the same command for `seed2` and `seed3` by changing the run-name filter and destination names.

### Playback Command

```powershell
uv run python src\hoppertrex_mjlab\scripts\rsl_rl\play.py Mjlab-HopperTrex-Balance-Robust-v0 --agent trained --checkpoint-file "<checkpoint-path>" --num-envs 1 --device cuda:0
```

## Next Stage - Robust L2

Task id:

```text
Mjlab-HopperTrex-Balance-Robust-L2-v0
```

Reset disturbances:

```text
roll/pitch: ±5 deg
root x velocity: ±0.10 m/s
root roll/pitch angular velocity: ±0.20 rad/s
```

Training should resume from the corresponding robust L1 checkpoint:

```powershell
uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-Robust-L2-v0 --env.scene.num-envs 256 --agent.max-iterations 1000 --agent.save-interval 50 --agent.seed 1 --agent.resume True --agent.load-run ".*robust_init_seed1.*" --agent.load-checkpoint "model_999.pt" --agent.algorithm.learning-rate 3.0e-4 --agent.algorithm.entropy-coef 0.002 --agent.run-name robust_l2_seed1
```

Success criteria:

```text
Mean episode length close to 500
Episode_Termination/non_wheel_ground_contact = 0
Episode_Termination/root_too_low = 0
Episode_Termination/bad_orientation near 0
Episode_Reward/clean_wheel_support close to 4
Episode_Reward/wheel_ground_contact close to 1
viewer confirms only wheel support
```

