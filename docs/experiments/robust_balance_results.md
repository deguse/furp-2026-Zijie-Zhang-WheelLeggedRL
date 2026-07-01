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

## Follow-up - Turn L4 Track

Task id:

```text
Mjlab-HopperTrex-Balance-Turn-L4-Track-v0
```

Alias:

```text
hoppertrex-balance-turn-l4-track-v0
```

Reason:

```text
The first Turn L4 seed1 run learned clean standing and preserved two-wheel
support, but yaw tracking plateaued. By iteration 999/1000:

Mean episode length: 500
Episode_Termination/non_wheel_ground_contact: 0
Episode_Termination/root_too_low: 0
Episode_Termination/bad_orientation: 0
Episode_Reward/clean_wheel_support: about 3.82
Episode_Reward/track_angular_velocity: about 1.01
Metrics/twist/error_vel_yaw: about 0.147
Mean action std: about 0.01
```

Interpretation:

```text
The policy chose the safe local optimum: stand cleanly and avoid using the new
yaw channel aggressively. Safety passed, but turning did not improve after the
early phase.
```

Changes compared with Turn L4:

```text
standing commands: 20% -> 5%
track_angular_velocity weight: 2.0 -> 5.0
track_angular_velocity std: 0.25 -> 0.18
lin_vel_xy_l2: -0.02 -> -0.005
wheel_vel_l2: -5e-4 -> -2e-4
action_rate_l2: -0.01 -> -0.003
```

The action space remains 2D:

```text
action[0] = pitch balance / forward-backward wheel channel
action[1] = yaw channel
```

Smoke test:

```powershell
cd C:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL

uv run python src\hoppertrex_mjlab\scripts\zero_agent.py --task Mjlab-HopperTrex-Balance-Turn-L4-Track-v0 --device cuda:0 --num_envs 1 --max_steps 100
```

Training:

```powershell
uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-Turn-L4-Track-v0 --env.scene.num-envs 256 --agent.max-iterations 1000 --agent.save-interval 50 --agent.seed 1 --agent.algorithm.learning-rate 3.0e-4 --agent.algorithm.entropy-coef 0.003 --agent.run-name turn_l4_track_seed1
```

Run seed2/seed3 by changing only `--agent.seed` and `--agent.run-name`.

Success criteria:

```text
Mean episode length close to 500
Episode_Termination/non_wheel_ground_contact = 0
Episode_Termination/root_too_low = 0
Episode_Termination/bad_orientation near 0
Episode_Reward/clean_wheel_support > 3.3
Episode_Reward/wheel_ground_contact > 0.85
Episode_Reward/track_angular_velocity clearly above Turn L4 baseline
Metrics/twist/error_vel_yaw < 0.10, ideally < 0.08
viewer confirms visible left/right in-place turning without non-wheel contact
```

If this still plateaus with clean standing but weak turning, the next likely
step is not more iterations; it is either a smaller yaw command curriculum or a
partial policy migration from the best 1D balance model into the 2D actor.

## Follow-up - Turn L4 Track v2

Task id:

```text
Mjlab-HopperTrex-Balance-Turn-L4-Track-v2
```

Alias:

```text
hoppertrex-balance-turn-l4-track-v2
```

Reason:

```text
The first Track probe improved the yaw reward but traded away stability and did
not reduce yaw error.

At iteration 149/150:
Mean episode length: 481.06
Episode_Termination/bad_orientation: 0.1111
Episode_Termination/non_wheel_ground_contact: 0
Episode_Reward/clean_wheel_support: 3.5027
Episode_Reward/wheel_ground_contact: 0.8757
Episode_Reward/track_angular_velocity: 1.4654
Metrics/twist/error_vel_yaw: 0.1741
Mean action std: 0.11
```

Interpretation:

```text
Track-v1 made the policy willing to move, but it likely encouraged excessive
body angular motion or oscillation rather than clean yaw command tracking.
Track-v2 keeps the stronger turning objective but softens the reward and
restores more action smoothness.
```

Changes compared with Track-v1:

```text
track_angular_velocity weight: 5.0 -> 4.0
track_angular_velocity std: 0.18 -> 0.22
standing commands: 5% unchanged
lin_vel_xy_l2: -0.005 unchanged
wheel_vel_l2: -2e-4 -> -3e-4
action_rate_l2: -0.003 -> -0.006
```

Probe command:

```powershell
cd C:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL

uv run python src\hoppertrex_mjlab\scripts\zero_agent.py --task Mjlab-HopperTrex-Balance-Turn-L4-Track-v2 --device cuda:0 --num_envs 1 --max_steps 100

uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-Turn-L4-Track-v2 --env.scene.num-envs 256 --agent.max-iterations 150 --agent.save-interval 50 --agent.seed 1 --agent.algorithm.learning-rate 3.0e-4 --agent.algorithm.entropy-coef 0.003 --agent.run-name turn_l4_track_v2_probe_seed1
```

Probe acceptance:

```text
Mean episode length >= 495
Episode_Termination/bad_orientation near 0
Episode_Termination/non_wheel_ground_contact = 0
Episode_Reward/wheel_ground_contact > 0.90
Episode_Reward/clean_wheel_support > 3.5
Episode_Reward/track_angular_velocity > 1.2
Metrics/twist/error_vel_yaw < 0.14
```

Stop rule:

```text
Do not run to 1000 if the 150-iteration probe has error_vel_yaw > 0.14,
bad_orientation increasing, or wheel_ground_contact below 0.90.
```

## Follow-up - 1D Balance to 2D Turn Migration

Reason:

```text
Turn-L4 learned clean standing but weak yaw tracking. Track-v1/v2 increased
yaw reward but introduced bad_orientation and did not reliably reduce
Metrics/twist/error_vel_yaw. The next step is warm-starting the 2D turning
policy from an existing 1D fixed-leg balance checkpoint instead of continuing
small reward edits.
```

Migration script:

```text
src/hoppertrex_mjlab/scripts/rsl_rl/migrate_balance_1d_to_turn_2d.py
```

Default target task:

```text
Mjlab-HopperTrex-Balance-Turn-L4-Track-v2
```

Default source checkpoint search priority:

```text
push_l3_seed{seed}
robust_l2_seed{seed}
robust_init_seed{seed}
clean_wheel_seed{seed}
slow_speed_seed{seed}
```

The script validates that the source checkpoint is from the current 1D
fixed-leg wheel policy:

```text
source actor/critic observation input: 25
source actor action output: 1
target actor/critic observation input: 26
target actor action output: 2
```

Generate migrated checkpoint:

```powershell
cd C:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL

uv run python src\hoppertrex_mjlab\scripts\rsl_rl\migrate_balance_1d_to_turn_2d.py --seed 1 --output-run migrated_turn_l4_track_v2_seed1
```

If the correct source checkpoint is archived outside the normal logs, pass it
explicitly:

```powershell
uv run python src\hoppertrex_mjlab\scripts\rsl_rl\migrate_balance_1d_to_turn_2d.py --seed 1 --source-checkpoint C:\mjlab_workspace\trained_models\your_best_1d_model.pt --output-run migrated_turn_l4_track_v2_seed1
```

Probe training from migrated checkpoint:

```powershell
uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-Turn-L4-Track-v2 --env.scene.num-envs 256 --agent.max-iterations 150 --agent.save-interval 50 --agent.seed 1 --agent.resume True --agent.load-run ".*migrated_turn_l4_track_v2_seed1.*" --agent.load-checkpoint "model_0.pt" --agent.algorithm.learning-rate 3.0e-4 --agent.algorithm.entropy-coef 0.003 --agent.run-name turn_l4_migrated_probe_seed1
```

Continue only if the 150-iteration probe meets:

```text
Mean episode length >= 495
Episode_Termination/bad_orientation near 0
Episode_Termination/non_wheel_ground_contact = 0
Episode_Reward/wheel_ground_contact > 0.90
Episode_Reward/clean_wheel_support > 3.5
Episode_Reward/track_angular_velocity > 1.2
Metrics/twist/error_vel_yaw < 0.14
```

## Follow-up - Turn L4 Easy Curriculum

Reason:

```text
The migrated Track-v2 probe improved from the early phase but still failed the
150-iteration gate:

Mean episode length: 481.32
Episode_Termination/bad_orientation: 0.0769
Episode_Reward/wheel_ground_contact: 0.8305
Episode_Reward/clean_wheel_support: 3.3209
Episode_Reward/track_angular_velocity: 1.4335
Metrics/twist/error_vel_yaw: 0.1839
```

Interpretation:

```text
Warm start helped recover balance, but the ±0.30 rad/s yaw command remains too
hard. The next step is a smaller yaw command curriculum before returning to
Track-v2.
```

Task id:

```text
Mjlab-HopperTrex-Balance-Turn-L4-Easy-v0
```

Alias:

```text
hoppertrex-balance-turn-l4-easy-v0
```

Changes compared with Track-v2:

```text
ang_vel_z range: ±0.30 -> ±0.10 rad/s
standing commands: 5% -> 10%
track_angular_velocity weight: 4.0 -> 3.0
track_angular_velocity std: 0.22 -> 0.20
lin_vel_xy_l2: -0.005 unchanged
wheel_vel_l2: -3e-4 unchanged
action_rate_l2: -0.006 unchanged
```

Use the existing migrated checkpoint; no new migration is required because
Turn-L4-Easy keeps the same 2D action and 26D observation shapes.

Smoke test:

```powershell
uv run python src\hoppertrex_mjlab\scripts\zero_agent.py --task Mjlab-HopperTrex-Balance-Turn-L4-Easy-v0 --device cuda:0 --num_envs 1 --max_steps 100
```

Probe training:

```powershell
uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-Turn-L4-Easy-v0 --env.scene.num-envs 256 --agent.max-iterations 150 --agent.save-interval 50 --agent.seed 1 --agent.resume True --agent.load-run ".*migrated_turn_l4_track_v2_seed1.*" --agent.load-checkpoint "model_0.pt" --agent.algorithm.learning-rate 3.0e-4 --agent.algorithm.entropy-coef 0.003 --agent.run-name turn_l4_easy_migrated_probe_seed1
```

Probe acceptance:

```text
Mean episode length >= 495
Episode_Termination/bad_orientation near 0
Episode_Termination/non_wheel_ground_contact = 0
Episode_Reward/wheel_ground_contact > 0.90
Episode_Reward/clean_wheel_support > 3.5
Metrics/twist/error_vel_yaw < 0.07 for the ±0.10 rad/s range
viewer confirms visible slow left/right in-place turning
```

If Easy passes, continue it to 500 iterations and then add a middle curriculum
stage at `ang_vel_z=±0.20` before returning to Track-v2.

## Follow-up - Turn L4 Easy LowYawScale

Reason:

```text
Turn-L4-Easy also failed the probe gate:

Mean episode length: 490.80
Episode_Termination/bad_orientation: 0.0000
Episode_Reward/wheel_ground_contact: 0.8767
Episode_Reward/clean_wheel_support: 3.5069
Episode_Reward/track_angular_velocity: 1.3362
Metrics/twist/error_vel_yaw: 0.1484
```

Direct fixed-wheel diagnostics showed that same-signed wheel targets can
produce yaw, so the yaw mapping direction is not the main issue. The issue is
that the yaw action channel uses the same `12 rad/s` scale as the balance
channel, which is too sensitive for a `±0.10 rad/s` yaw target.

Task id:

```text
Mjlab-HopperTrex-Balance-Turn-L4-Easy-LowYawScale-v0
```

Alias:

```text
hoppertrex-balance-turn-l4-easy-low-yaw-scale-v0
```

Changes compared with Turn-L4-Easy:

```text
balance action scale: 12.0
yaw action scale: 2.0
ang_vel_z range: ±0.10 rad/s
standing commands: 10%
track_angular_velocity weight: 3.0
track_angular_velocity std: 0.20
```

Smoke test:

```powershell
uv run python src\hoppertrex_mjlab\scripts\zero_agent.py --task Mjlab-HopperTrex-Balance-Turn-L4-Easy-LowYawScale-v0 --device cuda:0 --num_envs 1 --max_steps 100
```

Probe training:

```powershell
uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-Turn-L4-Easy-LowYawScale-v0 --env.scene.num-envs 256 --agent.max-iterations 150 --agent.save-interval 50 --agent.seed 1 --agent.resume True --agent.load-run ".*migrated_turn_l4_track_v2_seed1.*" --agent.load-checkpoint "model_0.pt" --agent.algorithm.learning-rate 3.0e-4 --agent.algorithm.entropy-coef 0.003 --agent.run-name turn_l4_easy_lowyaw_migrated_probe_seed1
```

Probe acceptance:

```text
Mean episode length >= 495
Episode_Termination/bad_orientation near 0
Episode_Termination/non_wheel_ground_contact = 0
Episode_Reward/wheel_ground_contact > 0.90
Episode_Reward/clean_wheel_support > 3.5
Metrics/twist/error_vel_yaw < 0.07
viewer confirms slow left/right in-place turning rather than front/back rocking
```

If LowYawScale fails, stop prioritizing in-place yaw and switch to a
`SlowSpeedTurn` task where yaw happens while the robot is already rolling.

## Follow-up - Turn L4 SignYaw

Reason:

```text
LowYawScale seed1 passed scalar metrics, but manual viewer testing with
ang_vel_z=+0.1 and ang_vel_z=-0.1 suggested both commands can produce the same
visual yaw direction. This means scalar error can pass while the policy does
not reliably learn command sign.
```

Task id:

```text
Mjlab-HopperTrex-Balance-Turn-L4-SignYaw-v0
```

Alias:

```text
hoppertrex-balance-turn-l4-sign-yaw-v0
```

Changes compared with LowYawScale:

```text
Yaw command samples only ±0.10 rad/s, not values near 0.
standing commands: 0%
yaw_scale: 2.0
adds yaw_sign_alignment reward with weight 2.0
```

The sign reward is positive when `command_yaw * actual_yaw > 0` and negative
when the policy turns the wrong way.

Probe training:

```powershell
uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-Turn-L4-SignYaw-v0 --env.scene.num-envs 256 --agent.max-iterations 150 --agent.save-interval 50 --agent.seed 1 --agent.resume True --agent.load-run ".*turn_l4_easy_lowyaw_migrated_seed1.*" --agent.load-checkpoint "model_499.pt" --agent.algorithm.learning-rate 2.0e-4 --agent.algorithm.entropy-coef 0.003 --agent.run-name turn_l4_sign_yaw_probe_seed1
```

If `model_499.pt` is not the latest checkpoint name, select the latest model
from `turn_l4_easy_lowyaw_migrated_seed1` dynamically.

Acceptance:

```text
Mean episode length >= 495
bad_orientation near 0
non_wheel_ground_contact = 0
wheel_ground_contact > 0.90
clean_wheel_support > 3.5
yaw_sign_alignment clearly positive
viewer confirms +0.1 and -0.1 turn in opposite directions
```

Do not continue to seed2/3 until the sign test passes in viewer.

## Next Stage - SlowSpeedTurn

Reason:

```text
SignYaw preserved balance and contact quality but did not learn reliable yaw
command sign. yaw_sign_alignment stayed near zero and error_vel_yaw remained
too high for a ±0.10 rad/s binary yaw task. This suggests pure in-place yaw is
not the right next target for the current fixed-leg ordinary-wheel setup.
```

Task id:

```text
Mjlab-HopperTrex-Balance-SlowSpeedTurn-v0
```

Alias:

```text
hoppertrex-balance-slow-speed-turn-v0
```

Purpose:

```text
Train turning while the robot is already rolling forward. This should give the
wheels a cleaner contact condition than pure point-turn yaw.
```

Task settings:

```text
lin_vel_x: 0.03 to 0.08 m/s
lin_vel_y: 0.0
ang_vel_z: -0.10 to 0.10 rad/s
standing commands: 0%
balance_scale: 12.0
yaw_scale: 2.0
track_linear_velocity weight/std: 2.0 / 0.08
track_angular_velocity weight/std: 2.0 / 0.20
```

Probe training:

```powershell
uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-v0 --env.scene.num-envs 256 --agent.max-iterations 150 --agent.save-interval 50 --agent.seed 1 --agent.resume True --agent.load-run ".*turn_l4_easy_lowyaw_migrated_seed1.*" --agent.load-checkpoint "model_499.pt" --agent.algorithm.learning-rate 2.0e-4 --agent.algorithm.entropy-coef 0.003 --agent.run-name slow_speed_turn_probe_seed1
```

Acceptance:

```text
Mean episode length >= 495
bad_orientation near 0
non_wheel_ground_contact = 0
wheel_ground_contact > 0.90
clean_wheel_support > 3.5
viewer confirms forward rolling arcs
+0.10 and -0.10 yaw commands curve in opposite directions
```

## Next Stage - SlowSpeedTurn Sign

Reason:

```text
SlowSpeedTurn metrics passed, but viewer showed +0.10 and -0.10 yaw commands
still curving to the same side. This means average yaw tracking metrics are not
enough: the policy can preserve balance and reduce mean yaw error while not
using the command sign correctly.
```

Task id:

```text
Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-v0
```

Alias:

```text
hoppertrex-balance-slow-speed-turn-sign-v0
```

Change from SlowSpeedTurn:

```text
Command uses BinarySlowSpeedTurnCommand.
lin_vel_x remains 0.03 to 0.08 m/s.
ang_vel_z is sampled only as -0.10 or +0.10 rad/s.
yaw_sign_alignment reward is enabled with weight 4.0.
```

Probe training from the latest SlowSpeedTurn checkpoint:

```powershell
$srcRunName = "slow_speed_turn_probe_seed1"
$srcRun = Get-ChildItem src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance -Directory |
  Where-Object { $_.Name -like "*$srcRunName*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$srcCkpt = Get-ChildItem $srcRun.FullName -Filter "model_*.pt" |
  Sort-Object { [int]($_.BaseName -replace "model_","") } -Descending |
  Select-Object -First 1

uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-v0 --env.scene.num-envs 256 --agent.max-iterations 150 --agent.save-interval 50 --agent.seed 1 --agent.resume True --agent.load-run ".*$srcRunName.*" --agent.load-checkpoint "$($srcCkpt.Name)" --agent.algorithm.learning-rate 2.0e-4 --agent.algorithm.entropy-coef 0.003 --agent.run-name slow_speed_turn_sign_probe_seed1
```

Acceptance:

```text
Mean episode length >= 495
non_wheel_ground_contact = 0
wheel_ground_contact > 0.90
clean_wheel_support > 3.5
yaw_sign_alignment clearly positive, target > 0.5
viewer confirms +0.10 and -0.10 yaw commands curve in opposite directions
```

Stop rule:

```text
If yaw_sign_alignment stays near 0 or viewer still shows same-side curvature,
stop this branch and diagnose action/measurement sign directly with scripted
constant action rollouts before further PPO training.
```

## SlowSpeedTurn Sign Probe Result

Observed terminal metrics:

```text
run name: slow_speed_turn_sign_probe_seed1
Mean episode length: 500.00
bad_orientation: 0.0000
non_wheel_ground_contact: 0.0000
wheel_ground_contact: 0.9599
clean_wheel_support: 3.8395
track_linear_velocity: 1.3762
track_angular_velocity: 1.0533
Metrics/twist/error_vel_xy: 0.0437
Metrics/twist/error_vel_yaw: 0.1067
Episode_Reward/yaw_sign_alignment: -0.0615
```

Decision:

```text
Do not continue this run. Safety/contact remain good, but yaw sign failed.
The negative yaw_sign_alignment confirms the viewer observation that the policy
does not reliably use the sign of the yaw command.
```

Fixed-action physics check:

```text
yaw_action < 0 produced negative actual yaw rate.
yaw_action > 0 produced positive actual yaw rate.
Therefore the low-level differential wheel action sign is not flipped. The
failure is in the learned policy behavior, not the direct wheel/yaw mapping.
```

Next diagnostic:

```powershell
$runName = "slow_speed_turn_sign_probe_seed1"
$run = Get-ChildItem src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance -Directory |
  Where-Object { $_.Name -like "*$runName*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$ckpt = Get-ChildItem $run.FullName -Filter "model_*.pt" |
  Sort-Object { [int]($_.BaseName -replace "model_","") } -Descending |
  Select-Object -First 1

uv run python src\hoppertrex_mjlab\scripts\rsl_rl\diagnose_turn_policy.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-v0 --checkpoint-file "$($ckpt.FullName)" --num-envs 256 --steps 500 --device cuda:0
```

Interpretation:

```text
If action_yaw has the same sign for cmd_yaw > 0 and cmd_yaw < 0, the actor is
ignoring the yaw command sign.

If action_yaw changes sign but actual_yaw does not, inspect wheel saturation,
contact asymmetry, or observation/action timing.

If both action_yaw and actual_yaw change sign in this script but viewer still
looks same-side, inspect viewer command override and camera/world-frame visual
interpretation.
```

## Next Stage - SlowSpeedTurn Sign ObsScale

Reason:

```text
The pulled checkpoint slow_speed_turn_sign_probe_seed1/model_797.pt showed that
the actor outputs negative yaw action for both positive and negative yaw
commands:

cmd_yaw > 0: mean action_yaw = -1.72265
cmd_yaw < 0: mean action_yaw = -2.17278

Direct fixed-action checks showed yaw_action sign maps correctly to actual yaw
rate, so the low-level differential wheel sign is not flipped. The failure is
that the actor ignores or underuses the yaw command sign.
```

Reference-project comparison:

```text
jaykorea/Isaac-RL-Two-wheel-Legged-Bot feeds velocity commands through a scaled
observation term. Its yaw command range is much larger (about +/-2.0 to +/-2.5)
and then scaled by 0.25, so the policy sees a yaw-command signal around +/-0.5.

Our Sign task used a true yaw command of only +/-0.10 and fed it to the actor
without observation scaling. With obs_normalization=False, this is likely too
small compared with other state features and the migrated actor can settle into
a safe fixed-yaw bias.
```

New task:

```text
Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-v0
alias: hoppertrex-balance-slow-speed-turn-sign-obs-scale-v0
```

Change from `SlowSpeedTurn-Sign`:

```text
True command is unchanged:
lin_vel_x = 0.03 to 0.08 m/s
ang_vel_z = -0.10 or +0.10 rad/s

Observation command is scaled only for actor/critic input:
lin_vel_x obs scale = 10.0
lin_vel_y obs scale = 1.0
ang_vel_z obs scale = 10.0

Therefore actor sees yaw command as -1.0 or +1.0, while rewards and metrics
still use the true +/-0.10 rad/s command.
```

Probe training:

```powershell
$srcRunName = "slow_speed_turn_sign_probe_seed1"
$srcRun = Get-ChildItem src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance -Directory |
  Where-Object { $_.Name -like "*$srcRunName*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$srcCkpt = Get-ChildItem $srcRun.FullName -Filter "model_*.pt" |
  Sort-Object { [int]($_.BaseName -replace "model_","") } -Descending |
  Select-Object -First 1

uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-v0 --env.scene.num-envs 256 --agent.max-iterations 150 --agent.save-interval 50 --agent.seed 1 --agent.resume True --agent.load-run ".*$srcRunName.*" --agent.load-checkpoint "$($srcCkpt.Name)" --agent.algorithm.learning-rate 2.0e-4 --agent.algorithm.entropy-coef 0.003 --agent.run-name slow_speed_turn_sign_obs_scale_probe_seed1
```

Acceptance:

```text
Mean episode length >= 495
non_wheel_ground_contact = 0
wheel_ground_contact > 0.90
clean_wheel_support > 3.5
yaw_sign_alignment > 0.5
diagnose_turn_policy.py shows mean action_yaw changes sign between cmd_yaw > 0
and cmd_yaw < 0
viewer confirms +0.10 and -0.10 yaw commands curve in opposite directions
```

Stop rule:

```text
If mean action_yaw still has the same sign for both command groups, do not
continue training. At that point the next branch should not be more PPO on the
same policy; it should change the action/control structure or open limited leg
control.
```

## SlowSpeedTurn Sign ObsScale Probe Result

Observed diagnostic:

```text
run name: slow_speed_turn_sign_obs_scale_probe_seed1
checkpoint: model_946.pt

cmd_yaw > 0: mean action_yaw = -1.41167
cmd_yaw < 0: mean action_yaw = -3.58762
all: yaw_sign_alignment = +0.02197
```

Observed terminal metrics:

```text
Mean episode length: 454.33
wheel_ground_contact: 0.8806
clean_wheel_support: 3.5196
bad_orientation: 0.3333
Episode_Reward/yaw_sign_alignment: 0.3337
Metrics/twist/error_vel_yaw: 0.0958
```

Decision:

```text
Stop this branch. Observation scaling made the command visible, but the actor
still outputs negative yaw action for both positive and negative commands. It
also degraded safety/contact quality. Do not continue from
slow_speed_turn_sign_probe_seed1 or slow_speed_turn_sign_obs_scale_probe_seed1.
```

## Next Stage - Reset Yaw Head

Reason:

```text
The yaw output head has a strong learned negative bias. Further fine-tuning
from failed Sign/ObsScale checkpoints reinforces the bad yaw habit instead of
crossing through zero. Keep the useful 2D balance/slow-speed actor features,
but reset the final yaw output row to neutral before training Sign-ObsScale.
```

New script:

```text
src/hoppertrex_mjlab/scripts/rsl_rl/reset_turn_yaw_head.py
```

Script behavior:

```text
Default target task:
Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-v0

Default source run:
slow_speed_turn_probe_seed{seed}

Output run:
reset_yaw_head_sign_obs_scale_seed{seed}

Checkpoint changes:
copy actor hidden layers from source
copy action[0] balance output row from source
zero action[1] yaw output row: mlp.4.weight[1, :] = 0, mlp.4.bias[1] = 0
copy action[0] std from source
set action[1] yaw std to 1.0
keep fresh target critic and optimizer
set iter = 0
```

Create reset checkpoint:

```powershell
uv run python src\hoppertrex_mjlab\scripts\rsl_rl\reset_turn_yaw_head.py --seed 1 --source-run "slow_speed_turn_probe_seed1" --output-run reset_yaw_head_sign_obs_scale_seed1 --force
```

Diagnose reset checkpoint before training:

```powershell
uv run python src\hoppertrex_mjlab\scripts\rsl_rl\diagnose_turn_policy.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-v0 --checkpoint-file src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance\reset_yaw_head_sign_obs_scale_seed1\model_0.pt --num-envs 256 --steps 100 --device cuda:0
```

Expected reset diagnostic:

```text
cmd_yaw > 0: mean action_yaw near 0
cmd_yaw < 0: mean action_yaw near 0
```

Probe training:

```powershell
uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-v0 --env.scene.num-envs 256 --agent.max-iterations 150 --agent.save-interval 50 --agent.seed 1 --agent.resume True --agent.load-run ".*reset_yaw_head_sign_obs_scale_seed1.*" --agent.load-checkpoint "model_0.pt" --agent.algorithm.learning-rate 2.0e-4 --agent.algorithm.entropy-coef 0.006 --agent.run-name slow_speed_turn_sign_obs_scale_reset_probe_seed1
```

Acceptance:

```text
Mean episode length >= 495
non_wheel_ground_contact = 0
wheel_ground_contact > 0.90
clean_wheel_support > 3.5
yaw_sign_alignment > 0.5
diagnose_turn_policy.py shows action_yaw changes sign:
  cmd_yaw > 0: mean action_yaw > 0
  cmd_yaw < 0: mean action_yaw < 0
viewer confirms +0.10 and -0.10 yaw commands curve in opposite directions
```

Stop rule:

```text
If reset-yaw-head training still produces same-sign action_yaw for both command
groups, stop PPO on this fixed-leg 2D action branch. Next change must be
control structure or limited leg control, not more reward/scale tuning.
```

## SlowSpeedTurn Sign ObsScale From Push L3 - Seed1 Result

Status:

```text
First yaw-sign success. Viewer confirms +0.10 and -0.10 yaw commands curve in
opposite directions.
```

Best branch:

```text
slow_speed_turn_sign_obs_scale_from_push_l3_seed1
```

Diagnostic summary:

```text
cmd_yaw > 0:
  mean action_yaw: +1.15879
  mean actual_yaw: +0.17093
  action sign match: 0.872
  actual sign match: 0.917
  yaw_sign_alignment: +0.69401

cmd_yaw < 0:
  mean action_yaw: -1.27761
  mean actual_yaw: -0.12037
  action sign match: 0.876
  actual sign match: 0.826
  yaw_sign_alignment: +0.53133

all:
  yaw_sign_alignment: +0.61376
```

Caveat:

```text
Yaw direction is correct, but contact quality is still marginal:
wheel_ground_contact was around 0.82 and clean_wheel_support around 3.28 in the
150-iteration probe. Do not continue the same run blindly.
```

## Next Stage - SlowSpeedTurn Sign ObsScale Safe

Goal:

```text
Keep the learned positive/negative yaw sign behavior, while shifting PPO pressure
back toward clean two-wheel contact.
```

New task:

```text
Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-Safe-v0
alias: hoppertrex-balance-slow-speed-turn-sign-obs-scale-safe-v0
```

Reward changes from `Sign-ObsScale`:

```text
clean_wheel_support: 4.0 -> 6.0
wheel_ground_contact: 1.0 -> 2.0
non_wheel_ground_contact: -6.0 -> -8.0
track_angular_velocity: 2.0 -> 1.0
yaw_sign_alignment: 4.0 -> 1.5
```

Training command:

```powershell
cd C:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL

$srcRunName = "slow_speed_turn_sign_obs_scale_from_push_l3_seed1"
$srcRun = Get-ChildItem src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance -Directory |
  Where-Object { $_.Name -like "*$srcRunName*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$srcCkpt = Get-ChildItem $srcRun.FullName -Filter "model_*.pt" |
  Sort-Object { [int]($_.BaseName -replace "model_","") } -Descending |
  Select-Object -First 1

uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-Safe-v0 --env.scene.num-envs 256 --agent.max-iterations 150 --agent.save-interval 50 --agent.seed 1 --agent.resume True --agent.load-run ".*$srcRunName.*" --agent.load-checkpoint "$($srcCkpt.Name)" --agent.algorithm.learning-rate 1.0e-4 --agent.algorithm.entropy-coef 0.003 --agent.run-name slow_speed_turn_sign_obs_scale_safe_seed1
```

Diagnosis after training:

```powershell
$runName = "slow_speed_turn_sign_obs_scale_safe_seed1"
$run = Get-ChildItem src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance -Directory |
  Where-Object { $_.Name -like "*$runName*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$ckpt = Get-ChildItem $run.FullName -Filter "model_*.pt" |
  Sort-Object { [int]($_.BaseName -replace "model_","") } -Descending |
  Select-Object -First 1

uv run python src\hoppertrex_mjlab\scripts\rsl_rl\diagnose_turn_policy.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-Safe-v0 --checkpoint-file "$($ckpt.FullName)" --num-envs 256 --steps 500 --device cuda:0
```

Acceptance:

```text
Mean episode length >= 495
non_wheel_ground_contact = 0
bad_orientation near 0
wheel_ground_contact > 0.90
clean_wheel_support > 3.5
yaw_sign_alignment > 0.5
cmd_yaw > 0 mean action_yaw > 0
cmd_yaw < 0 mean action_yaw < 0
viewer still shows opposite-direction arcs
```

Stop rule:

```text
If Safe loses yaw sign, stop immediately and keep the previous Sign-ObsScale
checkpoint as best. If yaw sign remains but contact does not improve by 150
iterations, do not extend to 500; the next change should be task/control design,
not a longer run.
```

## SlowSpeedTurn Sign ObsScale Safe-v1 Result

Observed at the end of the Safe-v1 run:

```text
run name: slow_speed_turn_sign_obs_scale_safe_seed1
Mean episode length: 491.84
clean_wheel_support: 5.3186
wheel_ground_contact: 1.7732
non_wheel_ground_contact: 0.0000
root_too_low: 0.0000
bad_orientation: 0.0769
track_angular_velocity: 0.4771
Metrics/twist/error_vel_yaw: 0.0822
```

Post-training diagnostic:

```text
cmd_yaw > 0:
  mean action_yaw: +1.64835
  mean actual_yaw: +0.03251
  action sign match: 0.987
  actual sign match: 0.769
  yaw_sign_alignment: +0.25703

cmd_yaw < 0:
  mean action_yaw: -2.15925
  mean actual_yaw: -0.01837
  action sign match: 0.988
  actual sign match: 0.580
  yaw_sign_alignment: +0.13963

all:
  yaw_sign_alignment: +0.19416
```

Interpretation:

```text
Safe-v1 improved clean contact, but it over-regularized the turn behavior.
The policy still outputs the correct yaw action sign, yet actual yaw becomes too
small. Do not continue Safe-v1.
```

## Next Stage - SlowSpeedTurn Sign ObsScale SafeV2

Reason:

```text
Safe-v1 moved too far toward "stand cleanly and barely turn". SafeV2 is a middle
ground: keep more yaw pressure than Safe-v1, but still put more weight on clean
wheel contact than the original Sign-ObsScale task.
```

New task:

```text
Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-v0
alias: hoppertrex-balance-slow-speed-turn-sign-obs-scale-safe-v2-v0
```

Reward changes:

```text
Original Sign-ObsScale:
  clean_wheel_support = 4.0
  wheel_ground_contact = 1.0
  non_wheel_ground_contact = -6.0
  track_angular_velocity = 2.0
  yaw_sign_alignment = 4.0

Safe-v1:
  clean_wheel_support = 6.0
  wheel_ground_contact = 2.0
  non_wheel_ground_contact = -8.0
  track_angular_velocity = 1.0
  yaw_sign_alignment = 1.5

SafeV2:
  clean_wheel_support = 5.0
  wheel_ground_contact = 1.5
  non_wheel_ground_contact = -7.0
  track_angular_velocity = 1.5
  yaw_sign_alignment = 2.5
```

Training command:

```powershell
cd C:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL

$srcRunName = "slow_speed_turn_sign_obs_scale_from_push_l3_seed1"
$srcRun = Get-ChildItem src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance -Directory |
  Where-Object { $_.Name -like "*$srcRunName*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$srcCkpt = Get-ChildItem $srcRun.FullName -Filter "model_*.pt" |
  Sort-Object { [int]($_.BaseName -replace "model_","") } -Descending |
  Select-Object -First 1

uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-v0 --env.scene.num-envs 256 --agent.max-iterations 150 --agent.save-interval 50 --agent.seed 1 --agent.resume True --agent.load-run ".*$srcRunName.*" --agent.load-checkpoint "$($srcCkpt.Name)" --agent.algorithm.learning-rate 1.0e-4 --agent.algorithm.entropy-coef 0.003 --agent.run-name slow_speed_turn_sign_obs_scale_safe_v2_seed1
```

Diagnosis command:

```powershell
$runName = "slow_speed_turn_sign_obs_scale_safe_v2_seed1"
$run = Get-ChildItem src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance -Directory |
  Where-Object { $_.Name -like "*$runName*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$ckpt = Get-ChildItem $run.FullName -Filter "model_*.pt" |
  Sort-Object { [int]($_.BaseName -replace "model_","") } -Descending |
  Select-Object -First 1

uv run python src\hoppertrex_mjlab\scripts\rsl_rl\diagnose_turn_policy.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-v0 --checkpoint-file "$($ckpt.FullName)" --num-envs 256 --steps 500 --device cuda:0
```

Acceptance:

```text
yaw_sign_alignment clearly above Safe-v1, ideally > 0.45
wheel_ground_contact > original Sign-ObsScale, ideally > 0.90 normalized
clean_wheel_support > 3.5
non_wheel_ground_contact = 0
bad_orientation near 0
viewer still shows opposite-direction arcs
```

Stop rule:

```text
Do not train past 150 if SafeV2 still collapses actual yaw below the original
Sign-ObsScale model. In that case the next useful change is action/control
structure or command curriculum, not another weight-only patch.
```
