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

## 2026-06-27 - Robust L2 Passed

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

Not included:

```text
no forward velocity command
no lateral velocity command
no yaw command
no leg action
no terrain
no continuous push force
```

### Runs

| Stage | Run name | Seed | Status | Notes |
| --- | --- | --- | --- | --- |
| Robust L2 | `robust_l2_seed1` | 1 | Passed | Metrics and viewer passed. |
| Robust L2 | `robust_l2_seed2` | 2 | Passed | Metrics and viewer passed. |
| Robust L2 | `robust_l2_seed3` | 3 | Passed | Metrics and viewer passed. |

### Observed Acceptance

The three L2 seeds reached the expected acceptance region:

```text
Mean episode length: 500.00
Episode_Termination/non_wheel_ground_contact: 0.0000
Episode_Termination/root_too_low: 0.0000
Episode_Termination/bad_orientation: 0.0000
Episode_Reward/clean_wheel_support: about 3.8 / 4.0
Episode_Reward/wheel_ground_contact: about 0.95
```

Viewer validation passed for all three seeds:

```text
reset recovery works
only main-wheel support observed
no thigh/calf/chassis support
no low-posture contact solution
```

### L2 Checkpoint Archival

Run this on each lab PC that produced the corresponding L2 run:

```powershell
cd C:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL
New-Item -ItemType Directory -Force -Path C:\mjlab_workspace\trained_models | Out-Null

$seed = 1
$runName = "robust_l2_seed$seed"
$run = Get-ChildItem src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance -Directory |
  Where-Object { $_.Name -like "*$runName*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$ckpt = Get-ChildItem $run.FullName -Filter "model_*.pt" |
  Sort-Object { [int]($_.BaseName -replace "model_","") } -Descending |
  Select-Object -First 1

Copy-Item $ckpt.FullName "C:\mjlab_workspace\trained_models\robust_l2_seed${seed}_$($ckpt.BaseName)_20260627.pt"
Copy-Item (Join-Path $run.FullName "params\agent.yaml") "C:\mjlab_workspace\trained_models\robust_l2_seed${seed}_agent_20260627.yaml"
Copy-Item (Join-Path $run.FullName "params\env.yaml") "C:\mjlab_workspace\trained_models\robust_l2_seed${seed}_env_20260627.yaml"
```

Change `$seed` to `2` or `3` for the other runs.

## 2026-06-27 - Push Recovery L3 Passed

Task id:

```text
Mjlab-HopperTrex-Balance-Push-L3-v0
```

Alias:

```text
hoppertrex-balance-push-l3-v0
```

Reset disturbances are inherited from L2:

```text
roll/pitch: ±5 deg
root x velocity: ±0.10 m/s
root roll/pitch angular velocity: ±0.20 rad/s
```

Interval push disturbance:

```text
interval: every 2.0-4.0 s, independently per environment
x velocity kick: ±0.15 m/s
pitch rate kick: ±0.25 rad/s
```

Not included:

```text
no y/z/roll/yaw push
no external wrench
no terrain
no leg action
no velocity tracking command
```

Training should resume from the corresponding robust L2 checkpoint. The command below selects the latest checkpoint automatically:

```powershell
cd C:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL

$seed = 1
$l2RunName = "robust_l2_seed$seed"
$run = Get-ChildItem src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance -Directory |
  Where-Object { $_.Name -like "*$l2RunName*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$ckpt = Get-ChildItem $run.FullName -Filter "model_*.pt" |
  Sort-Object { [int]($_.BaseName -replace "model_","") } -Descending |
  Select-Object -First 1

uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-Push-L3-v0 --env.scene.num-envs 256 --agent.max-iterations 1000 --agent.save-interval 50 --agent.seed $seed --agent.resume True --agent.load-run ".*$l2RunName.*" --agent.load-checkpoint "$($ckpt.Name)" --agent.algorithm.learning-rate 3.0e-4 --agent.algorithm.entropy-coef 0.002 --agent.run-name "push_l3_seed$seed"
```

Success criteria:

```text
Mean episode length close to 500
Episode_Termination/non_wheel_ground_contact = 0
Episode_Termination/root_too_low = 0
Episode_Termination/bad_orientation near 0
Episode_Reward/clean_wheel_support > 3.5
Episode_Reward/wheel_ground_contact > 0.9
viewer confirms push recovery and only wheel support
```

### Runs

| Stage | Run name | Seed | Status | Notes |
| --- | --- | --- | --- | --- |
| Push L3 | `push_l3_seed1` | 1 | Passed | Metrics and viewer passed. |
| Push L3 | `push_l3_seed2` | 2 | Passed | Metrics and viewer passed. |
| Push L3 | `push_l3_seed3` | 3 | Passed | Metrics and viewer passed. |

### Observed Acceptance

The three L3 seeds passed the target checks:

```text
Mean episode length: 500.00
Episode_Termination/non_wheel_ground_contact: 0.0000
Episode_Termination/root_too_low: 0.0000
Episode_Termination/bad_orientation: 0.0000
Episode_Reward/clean_wheel_support: > 3.5
Episode_Reward/wheel_ground_contact: > 0.9
```

Viewer validation passed for all three seeds:

```text
reset recovery works
interval push recovery works
only main-wheel support observed
no thigh/calf/chassis support
no low-posture contact solution
```

### L3 Checkpoint Archival

Run this on each lab PC that produced the corresponding L3 run:

```powershell
cd C:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL
New-Item -ItemType Directory -Force -Path C:\mjlab_workspace\trained_models | Out-Null

$seed = 1
$runName = "push_l3_seed$seed"
$run = Get-ChildItem src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance -Directory |
  Where-Object { $_.Name -like "*$runName*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$ckpt = Get-ChildItem $run.FullName -Filter "model_*.pt" |
  Sort-Object { [int]($_.BaseName -replace "model_","") } -Descending |
  Select-Object -First 1

Copy-Item $ckpt.FullName "C:\mjlab_workspace\trained_models\push_l3_seed${seed}_$($ckpt.BaseName)_20260627.pt"
Copy-Item (Join-Path $run.FullName "params\agent.yaml") "C:\mjlab_workspace\trained_models\push_l3_seed${seed}_agent_20260627.yaml"
Copy-Item (Join-Path $run.FullName "params\env.yaml") "C:\mjlab_workspace\trained_models\push_l3_seed${seed}_env_20260627.yaml"
```

Change `$seed` to `2` or `3` for the other runs.

## 2026-06-27 - Low-Speed Balance Initial Results

Task id:

```text
Mjlab-HopperTrex-Balance-SlowSpeed-v0
```

Alias:

```text
hoppertrex-balance-slow-speed-v0
```

Reset disturbances are inherited from L2:

```text
roll/pitch: ±5 deg
root x velocity: ±0.10 m/s
root roll/pitch angular velocity: ±0.20 rad/s
```

Command range:

```text
lin_vel_x: -0.10 to 0.10 m/s
lin_vel_y: 0.0
ang_vel_z: 0.0
standing commands: 20%
```

Not included:

```text
no yaw command
no lateral command
no interval push
no terrain
no leg action
```

Training should resume from the corresponding Push L3 checkpoint. The command below selects the latest checkpoint automatically:

```powershell
cd C:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL

$seed = 1
$l3RunName = "push_l3_seed$seed"
$run = Get-ChildItem src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance -Directory |
  Where-Object { $_.Name -like "*$l3RunName*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$ckpt = Get-ChildItem $run.FullName -Filter "model_*.pt" |
  Sort-Object { [int]($_.BaseName -replace "model_","") } -Descending |
  Select-Object -First 1

uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-SlowSpeed-v0 --env.scene.num-envs 256 --agent.max-iterations 1000 --agent.save-interval 50 --agent.seed $seed --agent.resume True --agent.load-run ".*$l3RunName.*" --agent.load-checkpoint "$($ckpt.Name)" --agent.algorithm.learning-rate 3.0e-4 --agent.algorithm.entropy-coef 0.002 --agent.run-name "slow_speed_seed$seed"
```

Success criteria:

```text
Mean episode length close to 500
Episode_Termination/non_wheel_ground_contact = 0
Episode_Termination/root_too_low = 0
Episode_Termination/bad_orientation near 0
Episode_Reward/clean_wheel_support > 3.3
Episode_Reward/wheel_ground_contact > 0.85
Episode_Reward/track_linear_velocity improves during training
Metrics/twist/error_vel_xy decreases from the initial phase
viewer confirms slow forward/backward motion with only wheel support
```

### Runs

| Stage | Run name | Seed | Status | Notes |
| --- | --- | --- | --- | --- |
| SlowSpeed | `slow_speed_seed1` | 1 | Passed safety, best tracking | Moves forward/backward and rebalances. |
| SlowSpeed | `slow_speed_seed2` | 2 | Passed safety, weak tracking | Moves forward but may reverse to recover balance. |
| SlowSpeed | `slow_speed_seed3` | 3 | Passed safety, medium tracking | Moves forward/backward and rebalances. |

### Observed Acceptance

All three SlowSpeed seeds preserved clean two-wheel support:

```text
Mean episode length: about 496-500
Episode_Termination/non_wheel_ground_contact: 0.0000
Episode_Termination/root_too_low: 0.0000
Episode_Termination/bad_orientation: 0.0000
Episode_Reward/clean_wheel_support: about 3.83-3.85
Episode_Reward/wheel_ground_contact: about 0.96
```

Velocity tracking differed by seed:

```text
seed1: track_linear_velocity about 1.28, error_vel_xy about 0.057
seed2: track_linear_velocity about 0.64, error_vel_xy about 0.120
seed3: track_linear_velocity about 0.94, error_vel_xy about 0.084
```

Viewer notes:

```text
seed1 and seed3 move and then rebalance
seed2 moves but can reverse for balance recovery
no seed showed non-wheel support as the main solution
```

## Next Stage - SlowSpeed Easy Curriculum

Task id:

```text
Mjlab-HopperTrex-Balance-SlowSpeed-Easy-v0
```

Alias:

```text
hoppertrex-balance-slow-speed-easy-v0
```

Purpose:

```text
Reduce command difficulty and strengthen velocity tracking so the policy learns command direction more cleanly before returning to ±0.10 m/s.
```

Command range:

```text
lin_vel_x: -0.05 to 0.05 m/s
lin_vel_y: 0.0
ang_vel_z: 0.0
standing commands: 10%
```

Reward changes compared with SlowSpeed-v0:

```text
track_linear_velocity weight: 3.0
track_linear_velocity std: 0.08
lin_vel_xy_l2: -0.001
```

Training should resume from the corresponding SlowSpeed checkpoint:

```powershell
cd C:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL

$seed = 1
$prevRunName = "slow_speed_seed$seed"
$run = Get-ChildItem src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance -Directory |
  Where-Object { $_.Name -like "*$prevRunName*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$ckpt = Get-ChildItem $run.FullName -Filter "model_*.pt" |
  Sort-Object { [int]($_.BaseName -replace "model_","") } -Descending |
  Select-Object -First 1

uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-SlowSpeed-Easy-v0 --env.scene.num-envs 256 --agent.max-iterations 1000 --agent.save-interval 50 --agent.seed $seed --agent.resume True --agent.load-run ".*$prevRunName.*" --agent.load-checkpoint "$($ckpt.Name)" --agent.algorithm.learning-rate 3.0e-4 --agent.algorithm.entropy-coef 0.002 --agent.run-name "slow_speed_easy_seed$seed"
```

Success criteria:

```text
Mean episode length close to 500
Episode_Termination/non_wheel_ground_contact = 0
Episode_Termination/root_too_low = 0
Episode_Termination/bad_orientation near 0
Episode_Reward/track_linear_velocity > 1.3
Metrics/twist/error_vel_xy < 0.04
viewer confirms forward commands mostly move forward and backward commands mostly move backward
```

## Next Stage - Turn L4

Task id:

```text
Mjlab-HopperTrex-Balance-Turn-L4-v0
```

Alias:

```text
hoppertrex-balance-turn-l4-v0
```

Purpose:

```text
Move beyond 1D coupled wheel control by adding differential wheel control.
The first target is in-place yaw tracking while keeping the fixed-leg, clean
two-wheel support behavior from the robust balance stages.
```

Important change:

```text
SlowSpeed-v0 action dimension: 1
Turn-L4-v0 action dimension: 2
```

The new wheel action maps policy outputs as:

```text
action[0] = pitch balance / forward-backward wheel channel
action[1] = yaw channel
left wheel  = -balance + yaw
right wheel = +balance + yaw
```

Because the actor input/output dimensions change, old 1D checkpoints should
not be used with normal `--agent.resume True`. Train Turn L4 from scratch first,
or add a dedicated policy migration script later.

Command range:

```text
lin_vel_x: 0.0
lin_vel_y: 0.0
ang_vel_z: -0.30 to 0.30 rad/s
standing commands: 20%
```

Reward additions compared with Robust L2:

```text
track_angular_velocity weight: 2.0
track_angular_velocity std: 0.25
```

Smoke test:

```powershell
cd C:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL

uv run python src\hoppertrex_mjlab\scripts\zero_agent.py --task Mjlab-HopperTrex-Balance-Turn-L4-v0 --device cuda:0 --num_envs 1 --max_steps 100
```

Training:

```powershell
uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-Turn-L4-v0 --env.scene.num-envs 256 --agent.max-iterations 1000 --agent.save-interval 50 --agent.seed 1 --agent.algorithm.learning-rate 3.0e-4 --agent.algorithm.entropy-coef 0.002 --agent.run-name turn_l4_seed1
```

Run seed2/seed3 by changing only `--agent.seed` and `--agent.run-name`.

Success criteria:

```text
Mean episode length close to 500
Episode_Termination/non_wheel_ground_contact = 0
Episode_Termination/root_too_low = 0
Episode_Termination/bad_orientation near 0
Episode_Reward/clean_wheel_support > 3.5
Episode_Reward/wheel_ground_contact > 0.9
Episode_Reward/track_angular_velocity rises during training
Metrics/twist/error_vel_yaw decreases from the initial phase
viewer confirms left/right turning with only the two main wheels touching
```

If Turn L4 passes, the next task should combine low-speed forward/backward
commands with yaw commands. Do not add terrain or leg motion before in-place
turning is stable.
