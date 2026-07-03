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

## Next Stage - SafeV2 YawScale3

Reason:

```text
SafeV2 is the current best compromise, but diagnose output shows the yaw policy
often asks for action values beyond +/-1. The task action term clips actions to
[-1, 1], so increasing yaw reward further is unlikely to help. The next test
should increase yaw actuator authority, not reward pressure.
```

New task:

```text
Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale3-v0
alias: hoppertrex-balance-slow-speed-turn-sign-obs-scale-safe-v2-yaw-scale3-v0
```

Only change from SafeV2:

```text
yaw_scale: 2.0 -> 3.0
```

Unchanged:

```text
clean_wheel_support = 5.0
wheel_ground_contact = 1.5
non_wheel_ground_contact = -7.0
track_angular_velocity = 1.5
yaw_sign_alignment = 2.5
lin_vel_x = (0.03, 0.08)
ang_vel_z = +/-0.10
```

Updated diagnosis note:

```text
diagnose_turn_policy.py now reports both raw policy actions and clipped actions.
Use clip_yaw / clip_balance when reasoning about what the action term actually
receives.
```

Zero-training diagnostic from SafeV2 checkpoint:

```powershell
cd C:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL

$srcRunName = "slow_speed_turn_sign_obs_scale_safe_v2_seed1"
$srcRun = Get-ChildItem src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance -Directory |
  Where-Object { $_.Name -like "*$srcRunName*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$srcCkpt = Get-ChildItem $srcRun.FullName -Filter "model_*.pt" |
  Sort-Object { [int]($_.BaseName -replace "model_","") } -Descending |
  Select-Object -First 1

uv run python src\hoppertrex_mjlab\scripts\rsl_rl\diagnose_turn_policy.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale3-v0 --checkpoint-file "$($srcCkpt.FullName)" --num-envs 256 --steps 500 --device cuda:0
```

Zero-training viewer check:

```powershell
uv run python src\hoppertrex_mjlab\scripts\rsl_rl\play.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale3-v0 --agent trained --checkpoint-file "$($srcCkpt.FullName)" --num-envs 1 --device cuda:0
```

Fine-tune only if zero-training check improves yaw without breaking contact:

```powershell
uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale3-v0 --env.scene.num-envs 256 --agent.max-iterations 100 --agent.save-interval 25 --agent.seed 1 --agent.resume True --agent.load-run ".*$srcRunName.*" --agent.load-checkpoint "$($srcCkpt.Name)" --agent.algorithm.learning-rate 5.0e-5 --agent.algorithm.entropy-coef 0.002 --agent.run-name slow_speed_turn_sign_obs_scale_safe_v2_yawscale3_seed1
```

Decision rule:

```text
If zero-training YawScale3 improves actual_yaw and viewer remains clean, do the
100-iteration fine-tune. If it causes wobble, non-wheel contact, or bad
orientation, stop and try yaw_scale=2.5 instead of continuing training.
```

## Next Stage - SafeV2 YawScale3 Smooth

Observation:

```text
YawScale3 can turn and stays safe, but viewer motion has slight mechanical
stutter. The issue is no longer "cannot turn"; it is control smoothness.
```

Reason:

```text
Do not add PID first. The policy already closes the balance loop and the wheels
are velocity actuators. Adding an external PID risks fighting the learned policy.
Instead, smooth only the yaw action channel inside the training environment, so
the policy can adapt to the filter.
```

New task:

```text
Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale3-Smooth-v0
alias: hoppertrex-balance-slow-speed-turn-sign-obs-scale-safe-v2-yaw-scale3-smooth-v0
```

Only change from YawScale3:

```text
yaw_smoothing_alpha = 0.65
smoothed_yaw = 0.65 * previous_smoothed_yaw + 0.35 * current_clipped_yaw
```

Unchanged:

```text
balance action is not smoothed
yaw_scale = 3.0
reward = SafeV2 reward
command = low-speed forward + binary +/-0.10 yaw
observation/action dimensions unchanged
```

Zero-training diagnostic from YawScale3 checkpoint:

```powershell
cd C:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL

$srcRunName = "slow_speed_turn_sign_obs_scale_safe_v2_yawscale3_seed1"
$srcRun = Get-ChildItem src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance -Directory |
  Where-Object { $_.Name -like "*$srcRunName*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$srcCkpt = Get-ChildItem $srcRun.FullName -Filter "model_*.pt" |
  Sort-Object { [int]($_.BaseName -replace "model_","") } -Descending |
  Select-Object -First 1

uv run python src\hoppertrex_mjlab\scripts\rsl_rl\diagnose_turn_policy.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale3-Smooth-v0 --checkpoint-file "$($srcCkpt.FullName)" --num-envs 256 --steps 500 --device cuda:0
```

Zero-training viewer:

```powershell
uv run python src\hoppertrex_mjlab\scripts\rsl_rl\play.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale3-Smooth-v0 --agent trained --checkpoint-file "$($srcCkpt.FullName)" --num-envs 1 --device cuda:0
```

Fine-tune only if smoothing does not destabilize the viewer:

```powershell
uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale3-Smooth-v0 --env.scene.num-envs 256 --agent.max-iterations 100 --agent.save-interval 25 --agent.seed 1 --agent.resume True --agent.load-run ".*$srcRunName.*" --agent.load-checkpoint "$($srcCkpt.Name)" --agent.algorithm.learning-rate 5.0e-5 --agent.algorithm.entropy-coef 0.001 --agent.run-name slow_speed_turn_sign_obs_scale_safe_v2_yawscale3_smooth_seed1
```

Acceptance:

```text
viewer motion is visibly smoother
actual_yaw remains near +/-0.10
non_wheel_ground_contact = 0
bad_orientation = 0 or near 0
no obvious delay-induced wobble
```

Stop rule:

```text
If yaw smoothing introduces lag, wobble, or weaker turning, do not keep training
this branch. Use the unsmoothed YawScale3 checkpoint as current best and test a
lighter smoothing alpha such as 0.45.
```

## Next Stage - SafeV2 YawScale3 SmoothV2

Observation:

```text
Smooth can turn and remains safe, but the policy is still not directly rewarded
for reducing the rate of the effective yaw command that actually reaches the
wheel action term. Existing action_rate_l2 uses raw policy action.
```

New task:

```text
Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale3-SmoothV2-v0
alias: hoppertrex-balance-slow-speed-turn-sign-obs-scale-safe-v2-yaw-scale3-smooth-v2-v0
```

Only change from Smooth:

```text
effective_yaw_rate_l2 weight = -0.03
effective_yaw_rate_l2 = (smoothed_yaw[t] - smoothed_yaw[t-1])^2
```

Unchanged:

```text
balance action is not smoothed
yaw_scale = 3.0
yaw_smoothing_alpha = 0.65
SafeV2 reward terms remain unchanged
observation/action dimensions unchanged
```

Updated diagnostic:

```text
diagnose_turn_policy.py now reports:
mean |d_clip_bal|
mean |d_clip_yaw|
mean |d_eff_yaw|

Use mean |d_eff_yaw| as the quantitative smoothness metric.
```

Baseline diagnostic from current Smooth checkpoint:

```powershell
cd C:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL

$srcRunName = "slow_speed_turn_sign_obs_scale_safe_v2_yawscale3_smooth_seed1"
$srcRun = Get-ChildItem src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance -Directory |
  Where-Object { $_.Name -like "*$srcRunName*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$srcCkpt = Get-ChildItem $srcRun.FullName -Filter "model_*.pt" |
  Sort-Object { [int]($_.BaseName -replace "model_","") } -Descending |
  Select-Object -First 1

uv run python src\hoppertrex_mjlab\scripts\rsl_rl\diagnose_turn_policy.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale3-SmoothV2-v0 --checkpoint-file "$($srcCkpt.FullName)" --num-envs 256 --steps 500 --device cuda:0
```

Fine-tune:

```powershell
uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale3-SmoothV2-v0 --env.scene.num-envs 256 --agent.max-iterations 100 --agent.save-interval 25 --agent.seed 1 --agent.resume True --agent.load-run ".*$srcRunName.*" --agent.load-checkpoint "$($srcCkpt.Name)" --agent.algorithm.learning-rate 3.0e-5 --agent.algorithm.entropy-coef 0.0005 --agent.run-name slow_speed_turn_sign_obs_scale_safe_v2_yawscale3_smooth_v2_seed1
```

Acceptance:

```text
viewer motion is visibly smoother
actual_yaw stays around +/-0.075 to +/-0.12
yaw_sign_alignment >= 0.45
non_wheel_ground_contact = 0
root_too_low = 0
bad_orientation = 0 or near 0
clean_wheel_support >= 4.0 / 5.0
wheel_ground_contact >= 1.2 / 1.5
mean |d_eff_yaw| drops by roughly 15% versus Smooth baseline
```

Stop rule:

```text
If SmoothV2 weakens turning, adds visible lag, or fails to reduce d_eff_yaw,
keep the current Smooth checkpoint as best. Do not increase smoothness penalty;
next test should be a lighter smoothing alpha such as 0.45.
```

## Next Stage - SafeV2 YawScale3 Smooth WheelRate

Observation:

```text
SmoothV2 did not reduce effective yaw rate enough. Diagnosis also shows the
larger high-frequency term is the balance channel, which is amplified by the
12.0 balance wheel scale. Directly smoothing balance is risky because balance
must remain fast for recovery.
```

New task:

```text
Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale3-Smooth-WheelRate-v0
alias: hoppertrex-balance-slow-speed-turn-sign-obs-scale-safe-v2-yaw-scale3-smooth-wheel-rate-v0
```

Only change from Smooth:

```text
wheel_target_rate_l2 weight = -5.0e-4
wheel_target_rate_l2 = sum((processed_wheel_target[t] - processed_wheel_target[t-1])^2)
```

Unchanged:

```text
balance action is not filtered
yaw_scale = 3.0
yaw_smoothing_alpha = 0.65
SafeV2 reward terms remain unchanged
effective_yaw_rate_l2 is not used
observation/action dimensions unchanged
```

Updated diagnostic:

```text
diagnose_turn_policy.py now reports:
mean |d_left_tgt|
mean |d_right_tgt|
mean |d_wheel_tgt|
p95 |d_left_tgt|
p95 |d_right_tgt|
p95 |d_wheel_tgt|
max |d_left_tgt|
max |d_right_tgt|
max |d_wheel_tgt|

Use mean and p95 |d_wheel_tgt| as the processed wheel target smoothness metrics.
```

Baseline diagnostic from current Smooth checkpoint:

```powershell
cd C:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL

$srcRunName = "slow_speed_turn_sign_obs_scale_safe_v2_yawscale3_smooth_seed1"
$srcRun = Get-ChildItem src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance -Directory |
  Where-Object { $_.Name -like "*$srcRunName*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$srcCkpt = Get-ChildItem $srcRun.FullName -Filter "model_*.pt" |
  Sort-Object { [int]($_.BaseName -replace "model_","") } -Descending |
  Select-Object -First 1

uv run python src\hoppertrex_mjlab\scripts\rsl_rl\diagnose_turn_policy.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale3-Smooth-WheelRate-v0 --checkpoint-file "$($srcCkpt.FullName)" --num-envs 256 --steps 500 --device cuda:0
```

Fine-tune:

```powershell
uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale3-Smooth-WheelRate-v0 --env.scene.num-envs 256 --agent.max-iterations 100 --agent.save-interval 25 --agent.seed 1 --agent.resume True --agent.load-run ".*$srcRunName.*" --agent.load-checkpoint "$($srcCkpt.Name)" --agent.algorithm.learning-rate 3.0e-5 --agent.algorithm.entropy-coef 0.0005 --agent.run-name slow_speed_turn_sign_obs_scale_safe_v2_yawscale3_smooth_wheelrate_seed1
```

Acceptance:

```text
viewer motion is visibly smoother
actual_yaw stays around +/-0.075 to +/-0.12
yaw_sign_alignment >= 0.45
non_wheel_ground_contact = 0
root_too_low = 0
bad_orientation = 0 or near 0
clean_wheel_support >= 4.0 / 5.0
wheel_ground_contact >= 1.2 / 1.5
mean |d_wheel_tgt| drops versus Smooth baseline
p95 |d_wheel_tgt| drops versus Smooth baseline
```

Stop rule:

```text
If WheelRate weakens balance recovery or yaw tracking, stop this branch and keep
the current Smooth checkpoint. Do not directly low-pass filter balance action
unless a future experiment explicitly retrains with that delay.
```

## Next Stage - SafeV2 YawScale3 Smooth LowForward

Observation:

```text
WheelRate did not reduce mean/p95 wheel target jumps and slightly weakened
positive yaw. The remaining stutter is likely caused by conflict between forward
tracking, turning, and fixed-leg balance recovery.
```

New task:

```text
Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale3-Smooth-LowForward-v0
alias: hoppertrex-balance-slow-speed-turn-sign-obs-scale-safe-v2-yaw-scale3-smooth-low-forward-v0
```

Only change from Smooth:

```text
lin_vel_x = (0.015, 0.05)
```

Unchanged:

```text
ang_vel_z = +/-0.10
yaw_scale = 3.0
yaw_smoothing_alpha = 0.65
SafeV2 reward terms remain unchanged
wheel_target_rate_l2 is not used
effective_yaw_rate_l2 is not used
observation/action dimensions unchanged
```

Baseline diagnostic from current Smooth checkpoint:

```powershell
cd C:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL

$srcRunName = "slow_speed_turn_sign_obs_scale_safe_v2_yawscale3_smooth_seed1"
$srcRun = Get-ChildItem src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance -Directory |
  Where-Object { $_.Name -like "*$srcRunName*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$srcCkpt = Get-ChildItem $srcRun.FullName -Filter "model_*.pt" |
  Sort-Object { [int]($_.BaseName -replace "model_","") } -Descending |
  Select-Object -First 1

uv run python src\hoppertrex_mjlab\scripts\rsl_rl\diagnose_turn_policy.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale3-Smooth-LowForward-v0 --checkpoint-file "$($srcCkpt.FullName)" --num-envs 256 --steps 500 --device cuda:0
```

Fine-tune:

```powershell
uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale3-Smooth-LowForward-v0 --env.scene.num-envs 256 --agent.max-iterations 100 --agent.save-interval 25 --agent.seed 1 --agent.resume True --agent.load-run ".*$srcRunName.*" --agent.load-checkpoint "$($srcCkpt.Name)" --agent.algorithm.learning-rate 3.0e-5 --agent.algorithm.entropy-coef 0.0005 --agent.run-name slow_speed_turn_sign_obs_scale_safe_v2_yawscale3_smooth_lowforward_seed1
```

Acceptance:

```text
viewer motion is visibly smoother
actual_yaw positive remains around +0.09 or higher
actual_yaw negative remains around -0.08 or lower
yaw_sign_alignment >= 0.45
overall actual sign match >= 0.82
non_wheel_ground_contact = 0
bad_orientation = 0 or near 0
mean/p95 |d_wheel_tgt| should not get worse
```

Stop rule:

```text
If LowForward does not improve viewer smoothness, keep the current Smooth
checkpoint as best. If it improves smoothness but tracks forward too slowly,
use this as a curriculum stage before gradually returning lin_vel_x to
(0.03, 0.08).
```

Weight adjustment rule:

```text
First run weight = -5.0e-4.
If d_wheel_tgt does not drop and yaw/balance remain healthy, try -1.0e-3.
If actual_yaw drops, actual sign match drops, or recovery becomes too dull, try
-2.5e-4 instead.
```

Minimum directional acceptance:

```text
cmd_yaw > 0 actual_yaw should not fall clearly below +0.09
cmd_yaw < 0 actual_yaw should not rise clearly above -0.08
overall actual sign match should not fall below 0.82
yaw_sign_alignment should not fall below 0.45
```

## Next Stage - SafeV2 YawScale3 Smooth MidForward

Decision after LowForward:

```text
LowForward is the current best candidate by safety and yaw tracking. It improves
actual_yaw and yaw_sign_alignment compared with Smooth, but it does not reduce
p95 wheel target spikes. Therefore the next step is not stronger wheel-rate
penalty. Use LowForward as a curriculum checkpoint and gradually restore forward
speed.
```

New task:

```text
Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale3-Smooth-MidForward-v0
alias: hoppertrex-balance-slow-speed-turn-sign-obs-scale-safe-v2-yaw-scale3-smooth-mid-forward-v0
```

Only change from LowForward:

```text
lin_vel_x = (0.02, 0.065)
```

Unchanged:

```text
ang_vel_z = +/-0.10
yaw_scale = 3.0
yaw_smoothing_alpha = 0.65
SafeV2 reward terms remain unchanged
wheel_target_rate_l2 is not used
effective_yaw_rate_l2 is not used
observation/action dimensions unchanged
```

Probe rule:

```text
Resume from the latest LowForward checkpoint.
Run only 100 iterations first.
Diagnose before viewer.
Continue only if yaw tracking and contact metrics remain healthy.
```

Acceptance:

```text
viewer is not worse than LowForward
actual_yaw positive remains around +0.09 or higher
actual_yaw negative remains around -0.08 or lower
yaw_sign_alignment >= 0.50 preferred, >= 0.45 minimum
overall actual sign match >= 0.84 preferred, >= 0.82 minimum
non_wheel_ground_contact = 0
bad_orientation = 0 or near 0
mean/p95 |d_wheel_tgt| should not get worse by more than a few percent
```

Stop rule:

```text
If MidForward makes viewer stutter worse or weakens yaw tracking, keep
LowForward as best and do not restore forward speed further. If it passes, the
next curriculum step is returning to the original lin_vel_x range (0.03, 0.08)
from the MidForward checkpoint with another short probe.
```

## Next Stage - MidForward StableRate

Observation:

```text
The remaining visual stutter matches the control mechanism: policy target jumps
are large, the velocity actuator has limited torque, the wheel actuator saturates,
and the balance channel then recovers posture. This looks like "push, saturate,
recover" rather than a purely viewer-side issue.
```

Why not continue WheelRate:

```text
Global wheel_target_rate_l2 penalizes target changes even when the robot needs a
fast balance correction. That can make the policy too dull and weaken yaw or
recovery.
```

New task:

```text
Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale3-Smooth-MidForward-StableRate-v0
alias: hoppertrex-balance-slow-speed-turn-sign-obs-scale-safe-v2-yaw-scale3-smooth-mid-forward-stable-rate-v0
```

Only change from MidForward:

```text
Add stable_wheel_target_rate_l2 with weight -7.5e-4.
The penalty is active only when clean_wheel_support is true.
```

Interpretation:

```text
When the robot is upright, high enough, on both wheels, and has no non-wheel
contact, encourage smoother final wheel targets. When it is not in a clean
support state, do not penalize aggressive wheel target changes, so balance
recovery remains available.
```

Probe rule:

```text
Resume from the latest MidForward checkpoint.
Run only 100 iterations first.
Diagnose before viewer.
Do not continue if yaw tracking weakens or contact safety worsens.
```

Acceptance:

```text
viewer should show less "push, saturate, recover" stutter
actual_yaw positive remains around +0.09 or higher
actual_yaw negative remains around -0.08 or lower
yaw_sign_alignment >= 0.50 preferred, >= 0.45 minimum
overall actual sign match >= 0.84 preferred, >= 0.82 minimum
non_wheel_ground_contact = 0
bad_orientation = 0 or near 0
mean/p95 |d_wheel_tgt| should improve or at least not worsen
```

Stop rule:

```text
If StableRate does not improve viewer smoothness or makes yaw weaker, keep
MidForward as best. Do not increase this penalty blindly; the next alternative
would be a target-rate limiter inside the action term and retraining with that
limiter from the start of the turn curriculum.
```

## Next Stage - MidForward Slew6

Decision after StableRate:

```text
StableRate did not materially reduce mean/p95 wheel target jumps and slightly
weakened negative yaw. The remaining stutter is better explained as a hard
control-path issue: wheel velocity targets jump, the torque-limited velocity
actuator saturates, then the policy recovers posture. Reward-only smoothing is
not reliable enough for this failure mode.
```

New task:

```text
Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale3-Smooth-MidForward-Slew6-v0
alias: hoppertrex-balance-slow-speed-turn-sign-obs-scale-safe-v2-yaw-scale3-smooth-mid-forward-slew6-v0
```

Only change from MidForward:

```text
Final left/right wheel velocity targets are slew-rate limited to +/-6 rad/s per
policy step.
```

Unchanged:

```text
lin_vel_x = (0.02, 0.065)
ang_vel_z = +/-0.10
yaw_scale = 3.0
yaw_smoothing_alpha = 0.65
SafeV2 reward terms remain unchanged
observation/action dimensions unchanged
```

Interpretation:

```text
This does not low-pass the raw policy output. It clamps the final velocity
targets sent to the velocity actuator, directly attacking the target jump that
causes torque saturation. Balance may feel slightly delayed, so this must be
tested as a short probe first.
```

Probe rule:

```text
Resume from the latest MidForward checkpoint.
Run only 100 iterations first.
Diagnose before viewer.
If stability is poor, stop immediately and keep MidForward as best.
```

Acceptance:

```text
viewer should show less "push, saturate, recover" stutter
max |d_wheel_tgt| should be around 6 because of the limiter
mean/p95 |d_wheel_tgt| should clearly drop
actual_yaw positive remains around +0.09 or higher
actual_yaw negative remains around -0.08 or lower
yaw_sign_alignment >= 0.45 minimum
overall actual sign match >= 0.82 minimum
non_wheel_ground_contact = 0
bad_orientation = 0 or near 0
```

Stop rule:

```text
If Slew6 makes balance recovery too dull or yaw tracking collapses, do not tune
reward further on this branch. Try a weaker limiter such as Slew8, or return to
MidForward as the practical best fixed-leg turn policy.
```

## Next Stage - Slew6 YawScale2p5

Decision after Slew6:

```text
Slew6 successfully removes the wheel target spikes:
mean |d_wheel_tgt| drops to about 5.39, p95/max are capped at 6.0. Viewer feedback
also reports slight improvement. However actual_yaw overshoots the +/-0.10
command, reaching roughly +0.136 / -0.142, so yaw authority is now too high for
the slew-limited action path.
```

New task:

```text
Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale2p5-Smooth-MidForward-Slew6-v0
alias: hoppertrex-balance-slow-speed-turn-sign-obs-scale-safe-v2-yaw-scale2p5-smooth-mid-forward-slew6-v0
```

Only change from Slew6:

```text
yaw_scale = 2.5
```

Unchanged:

```text
target_slew_limit = 6.0 rad/s per policy step
lin_vel_x = (0.02, 0.065)
ang_vel_z = +/-0.10
yaw_smoothing_alpha = 0.65
SafeV2 reward terms remain unchanged
observation/action dimensions unchanged
```

Probe rule:

```text
Resume from the latest Slew6 checkpoint.
Run only 100 iterations first.
Diagnose before viewer.
```

Acceptance:

```text
mean/p95/max |d_wheel_tgt| stay near Slew6 levels
actual_yaw moves closer to +/-0.10 than Slew6
yaw_sign_alignment remains >= 0.55 preferred, >= 0.45 minimum
actual sign match remains >= 0.88 preferred, >= 0.82 minimum
non_wheel_ground_contact = 0
bad_orientation = 0 or near 0
viewer is at least as smooth as Slew6 and less over-aggressive
```

Stop rule:

```text
If YawScale2p5 becomes too weak or sluggish, keep Slew6 as the current smoothness
best and try an intermediate yaw_scale=2.75 only if the viewer clearly prefers
Slew6 smoothness over MidForward.
```

## Next Stage - VarYawNoBack

Observation:

```text
YawScale2p5-Slew6 improves smoothness and reduces yaw overshoot, but viewer
feedback still reports non-smooth turning, occasional backward recovery steps,
and asymmetric left/right turn quality.
```

Search/code notes:

```text
Public wheel-legged RL references commonly rely on curriculum, action limiting,
and leg/wheel coordination rather than only action-rate rewards. One searched
snippet explicitly notes excluding wheel actions from action-rate penalties to
preserve recovery flexibility. In this project, hard target slew limiting was
more effective than reward-only wheel-rate penalties, so the next step should
adjust command distribution and backward motion, not increase action penalties.
```

New task:

```text
Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale2p5-Smooth-MidForward-Slew6-VarYawNoBack-v0
alias: hoppertrex-balance-slow-speed-turn-sign-obs-scale-safe-v2-yaw-scale2p5-smooth-mid-forward-slew6-var-yaw-no-back-v0
```

Changes from YawScale2p5-Slew6:

```text
yaw command magnitude is sampled from 0.04 to 0.10 rad/s with random sign,
instead of fixed +/-0.10
add light backward_lin_vel_x_l2 penalty with weight -0.6
```

Unchanged:

```text
target_slew_limit = 6.0 rad/s per policy step
yaw_scale = 2.5
lin_vel_x = (0.02, 0.065)
yaw_smoothing_alpha = 0.65
SafeV2 contact/upright rewards remain unchanged
observation/action dimensions unchanged
```

Rationale:

```text
Fixed binary +/-0.10 yaw encourages a bang-bang left/right policy. Variable yaw
magnitude forces the policy to learn proportional turning. The backward penalty
is light and only active when forward command is positive, so it discourages the
observed backward recovery step without banning balance recovery.
```

Diagnostic additions:

```text
diagnose_turn_policy.py now reports mean cmd_lin_x, p05/min actual_lin_x,
reverse lin_x fraction, and hard reverse fraction.
```

Probe rule:

```text
Resume from the latest YawScale2p5-Slew6 checkpoint.
Run only 100 iterations first.
Diagnose before viewer.
```

Acceptance:

```text
viewer turning should feel smoother or at least not worse
reverse lin_x fraction and hard reverse fraction should not increase
actual_yaw should remain sign-correct for both directions
mean/p95/max |d_wheel_tgt| stay near Slew6 levels
non_wheel_ground_contact = 0
bad_orientation = 0 or near 0
```

Stop rule:

```text
If VarYawNoBack makes yaw weak, increases backward steps, or worsens viewer
quality, stop this fixed-leg wheel-only branch. The next meaningful upgrade is
limited leg involvement or explicit body lean/height control, not more reward
micro-tuning.
```

## Result Summary - 2026-07-02 Fixed-Leg Slow Turn

Current fixed-leg best:

```text
task: Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale2p5-Smooth-MidForward-Slew6-v0
run: slow_speed_turn_sign_obs_scale_safe_v2_yawscale2p5_smooth_midforward_slew6_seed1
checkpoint tested: model_892.pt
status: current best fixed-leg slow-turn checkpoint
```

Why this is current best:

```text
Slew6 caps final wheel target changes at 6 rad/s per policy step, eliminating
the large 15-24 rad/s target spikes that caused "push, saturate, recover"
stutter. YawScale2p5 reduces the over-aggressive yaw response seen with
YawScale3/Slew6.
```

Latest diagnostic for current best:

```text
actual_yaw:
  cmd_yaw > 0: +0.11243
  cmd_yaw < 0: -0.11995

wheel target jumps:
  mean |d_wheel_tgt|: 5.45900
  p95 |d_wheel_tgt|: 6.00000
  max |d_wheel_tgt|: 6.00000

sign/stability:
  actual sign match: 0.918
  yaw_sign_alignment: 0.65151
```

Known unresolved issue:

```text
The policy still uses backward/forward wheel motion to recover pitch balance
during turns:
  reverse lin_x frac: 0.319
  hard reverse frac: 0.147
  min actual_lin_x: -0.37041

This is interpreted as a structural limitation of fixed-leg, wheel-only control:
the same wheels must handle yaw tracking, forward velocity tracking, and inverted
pendulum balance recovery.
```

Rejected / not promoted:

```text
VarYawNoBack:
  task: Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale2p5-Smooth-MidForward-Slew6-VarYawNoBack-v0
  run: slow_speed_turn_sign_obs_scale_safe_v2_yawscale2p5_smooth_midforward_slew6_varyaw_noback_seed1
  status: not promoted

Reason:
  variable yaw made left/right yaw more proportional, but did not materially fix
  backward recovery:
    reverse lin_x frac: 0.312
    hard reverse frac: 0.135
  It also weakened fixed +/-0.10 yaw tracking compared with current best.
```

Forward/backward status:

```text
SlowSpeed was already trained before turning work:
  slow_speed_seed1: passed safety, best tracking
  slow_speed_seed2: passed safety, weak tracking
  slow_speed_seed3: passed safety, medium tracking

The task can move forward/backward while preserving clean two-wheel support, but
tracking is not perfect and weak seeds may reverse to recover balance. It should
be treated as partially solved under fixed legs, not as a fully smooth locomotion
controller.
```

Decision:

```text
Stop fixed-leg reward/action micro-tuning for smooth turning. Archive
YawScale2p5-Slew6 as the best fixed-leg slow-turn checkpoint. The next meaningful
stage is limited leg assist / body lean assist so the robot can absorb turn
disturbances without relying only on wheel backtracking.
```

## Next Stage - SlowSpeedTurn Slew6 Push Combined Probe

Purpose:

```text
Before moving to limited leg assist, run one explicit combined validation task:
low-speed forward turning plus light interval push recovery. This checks whether
the single-task capabilities can coexist in one fixed-leg wheel-only policy.
```

New task:

```text
Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale2p5-Smooth-MidForward-Slew6-Push-v0
alias: hoppertrex-balance-slow-speed-turn-sign-obs-scale-safe-v2-yaw-scale2p5-smooth-mid-forward-slew6-push-v0
```

Changes from current best YawScale2p5-Slew6:

```text
Add interval velocity-kick push:
  interval: 3.0-5.0 s
  x velocity kick: +/-0.08 m/s
  pitch rate kick: +/-0.12 rad/s

Unchanged:
  lin_vel_x = (0.02, 0.065)
  ang_vel_z = +/-0.10
  yaw_scale = 2.5
  yaw_smoothing_alpha = 0.65
  target_slew_limit = 6.0 rad/s per policy step
  fixed legs only
  no external wrench
  no terrain
```

Probe rule:

```text
Resume from the current best fixed-leg slow-turn checkpoint:
slow_speed_turn_sign_obs_scale_safe_v2_yawscale2p5_smooth_midforward_slew6_seed1/model_892.pt

Run only 100 iterations first. Diagnose before viewer. Do not continue to 300+
unless safety and yaw direction both pass.
```

Training command:

```powershell
cd C:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL

$srcRunName = "slow_speed_turn_sign_obs_scale_safe_v2_yawscale2p5_smooth_midforward_slew6_seed1"

$srcRun = Get-ChildItem src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance -Directory |
  Where-Object { $_.Name -like "*$srcRunName*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$srcCkpt = Get-ChildItem $srcRun.FullName -Filter "model_892.pt" |
  Select-Object -First 1

if ($null -eq $srcCkpt) {
  throw "Expected best checkpoint model_892.pt was not found in $($srcRun.FullName)"
}

uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale2p5-Smooth-MidForward-Slew6-Push-v0 --env.scene.num-envs 256 --agent.max-iterations 100 --agent.save-interval 25 --agent.seed 1 --agent.resume True --agent.load-run ".*$srcRunName.*" --agent.load-checkpoint "model_892.pt" --agent.algorithm.learning-rate 3.0e-5 --agent.algorithm.entropy-coef 0.0005 --agent.run-name slow_speed_turn_slew6_push_probe_seed1
```

Acceptance:

```text
Mean episode length >= 495
non_wheel_ground_contact = 0
root_too_low = 0
bad_orientation = 0 or near 0
actual_yaw remains sign-correct for both directions
actual sign match >= 0.85
yaw_sign_alignment >= 0.50
viewer confirms forward slow turning through light push without non-wheel support
```

Manual push viewer:

```text
Use the project-local play_with_manual_push.py script for demonstration-only
manual push buttons. This avoids modifying the upstream mjlab-main viewer.
The buttons apply one-shot root velocity kicks to the selected env; they do not
change the velocity command sliders.
```

```powershell
$ckpt = "C:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL\src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance\2026-07-03_18-37-54_slow_speed_turn_slew6_push_probe_seed1\model_991.pt"

uv run python src\hoppertrex_mjlab\scripts\rsl_rl\play_with_manual_push.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale2p5-Smooth-MidForward-Slew6-Push-v0 --agent trained --checkpoint-file "$ckpt" --num-envs 1 --device cuda:0 --viewer viser
```

Stop rule:

```text
If the probe fails safety or yaw direction, stop and do not run 500/1000.
If safety passes but push collapses turning, reduce push to x +/-0.05 and pitch
rate +/-0.08 before another probe. Only after this combined task passes should
the project move to bidirectional combined validation or limited leg assist.
```

## Result Summary - 2026-07-03 SlowSpeedTurn Slew6 Push

Probe result:

```text
task: Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale2p5-Smooth-MidForward-Slew6-Push-v0
run: slow_speed_turn_slew6_push_probe_seed1
checkpoint tested: model_991.pt
status: passed as first fixed-leg combined forward-turn + light-push baseline
```

Safety:

```text
Mean episode length: 500
non_wheel_ground_contact: 0
bad_orientation: 0
root_too_low: 0
```

Diagnostic summary:

```text
cmd_yaw > 0:
  actual_yaw: +0.12049
  actual sign match: 0.930
  yaw_sign_alignment: 0.67995

cmd_yaw < 0:
  actual_yaw: -0.11464
  actual sign match: 0.910
  yaw_sign_alignment: 0.63580

all:
  actual sign match: 0.919
  yaw_sign_alignment: 0.65538
  mean |d_wheel_tgt|: 5.47129
  p95 |d_wheel_tgt|: 6.00000
  max |d_wheel_tgt|: 6.00000
```

Remaining limitation:

```text
Forward-turn + light-push passes, but the policy still uses backward velocity
as part of pitch recovery:
  reverse lin_x frac: 0.321
  hard reverse frac: 0.151
  min actual_lin_x: -0.46024

This means the combined forward-turn baseline is usable, but it is not yet a
complete forward/backward/turn/push controller.
```

Decision:

```text
Do not continue forward-only reward micro-tuning. The next probe should add
negative lin_vel_x commands while keeping the same fixed-leg action structure,
Slew6 target limiting, yaw_scale=2.5, and light push event. This tests whether
the current policy can unify forward, reverse, turning, and push recovery before
moving to limited leg assist.
```

## Next Stage - SlowSpeedTurn Bidir Slew6 Push Probe

Purpose:

```text
Validate a minimal combined fixed-leg controller:
  forward + backward command tracking
  left/right low-speed turning
  light interval push recovery

This is a diagnostic bridge between the current wheel-only baseline and the
planned limited leg assist stage.
```

New task:

```text
Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale2p5-Smooth-Bidir-Slew6-Push-v0
alias: hoppertrex-balance-slow-speed-turn-sign-obs-scale-safe-v2-yaw-scale2p5-smooth-bidir-slew6-push-v0
```

Changes from passed forward-turn + push baseline:

```text
lin_vel_x range: (-0.05, 0.065)

Unchanged:
  ang_vel_z = +/-0.10
  yaw_scale = 2.5
  yaw_smoothing_alpha = 0.65
  target_slew_limit = 6.0 rad/s per policy step
  push interval = 3.0-5.0 s
  x velocity kick = +/-0.08 m/s
  pitch rate kick = +/-0.12 rad/s
  fixed legs only
  no external wrench
```

Probe rule:

```text
Resume from the passed combined push checkpoint:
slow_speed_turn_slew6_push_probe_seed1/model_991.pt

Run only 100 iterations first. Diagnose before viewer. Do not run 300/500 until
safety, yaw sign, and forward/backward command groups are checked.
```

Training command:

```powershell
cd C:\mjlab_workspace\furp-2026-Zijie-Zhang-WheelLeggedRL

$srcRunName = "slow_speed_turn_slew6_push_probe_seed1"

$srcRun = Get-ChildItem src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance -Directory |
  Where-Object { $_.Name -like "*$srcRunName*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$srcCkpt = Get-ChildItem $srcRun.FullName -Filter "model_991.pt" |
  Select-Object -First 1

if ($null -eq $srcCkpt) {
  throw "Expected checkpoint model_991.pt was not found in $($srcRun.FullName)"
}

uv run python src\hoppertrex_mjlab\scripts\rsl_rl\train.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale2p5-Smooth-Bidir-Slew6-Push-v0 --env.scene.num-envs 256 --agent.max-iterations 100 --agent.save-interval 25 --agent.seed 1 --agent.resume True --agent.load-run ".*$srcRunName.*" --agent.load-checkpoint "model_991.pt" --agent.algorithm.learning-rate 5.0e-5 --agent.algorithm.entropy-coef 0.001 --agent.run-name slow_speed_turn_bidir_slew6_push_probe_seed1
```

Diagnostic command:

```powershell
$runName = "slow_speed_turn_bidir_slew6_push_probe_seed1"

$run = Get-ChildItem src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance -Directory |
  Where-Object { $_.Name -like "*$runName*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$ckpt = Get-ChildItem $run.FullName -Filter "model_*.pt" |
  Sort-Object { [int]($_.BaseName -replace "model_","") } -Descending |
  Select-Object -First 1

uv run python src\hoppertrex_mjlab\scripts\rsl_rl\diagnose_turn_policy.py Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale2p5-Smooth-Bidir-Slew6-Push-v0 --checkpoint-file "$($ckpt.FullName)" --num-envs 256 --steps 500 --device cuda:0 --detail-groups --slew-cap 6.0
```

Diagnostic focus:

```text
Use the normal groups for headline pass/fail:
  cmd_yaw > 0
  cmd_yaw < 0
  cmd_lin_x > 0.01
  cmd_lin_x < -0.01

Use --detail-groups to check the four combined cases:
  forward + positive yaw
  forward + negative yaw
  backward + positive yaw
  backward + negative yaw

Use slew cap frac to detect whether the policy is constantly hitting the Slew6
limit instead of producing naturally smooth wheel targets.
```

Acceptance:

```text
Mean episode length >= 495
non_wheel_ground_contact = 0
root_too_low = 0
bad_orientation = 0 or near 0
actual sign match >= 0.85
yaw_sign_alignment >= 0.50
cmd_lin_x > 0.01 group: mean actual_lin_x should be positive
cmd_lin_x < -0.01 group: mean actual_lin_x should be negative or clearly lower
lin sign match >= 0.65 as an initial bidirectional probe target
four detail groups should keep yaw sign correct and avoid non-wheel support
slew cap frac should not become materially worse than the forward-turn baseline
viewer shows no thigh/calf/chassis support
```

Stop rule:

```text
If reverse commands collapse balance or yaw direction, stop and do not run long
training. If bidirectional command tracking fails but safety/yaw remain good,
record the failure explicitly and move to limited leg assist rather than more
fixed-leg reward-only tuning.
```
