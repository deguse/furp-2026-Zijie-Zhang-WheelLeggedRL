# HopperTrex Model Inventory - 2026-07-03

## Keep / Baseline

### Robust L2 balance
Task: `Mjlab-HopperTrex-Balance-Robust-L2-v0`  
Run: `2026-06-27_10-49-06_robust_l2_seed1`  
Checkpoint: `model_1997.pt`  
Status: keep, clean fixed-leg balance source.

### Push L3
Task: `Mjlab-HopperTrex-Balance-Push-L3-v0`  
Run: `2026-06-27_11-32-43_push_l3_seed1`  
Checkpoint: `model_2996.pt`  
Status: keep, push recovery source.

### Fixed-leg forward slow-turn best
Task: `Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale2p5-Smooth-MidForward-Slew6-v0`  
Run: `2026-07-02_20-42-32_slow_speed_turn_sign_obs_scale_safe_v2_yawscale2p5_smooth_midforward_slew6_seed1`  
Checkpoint: `model_892.pt`  
Status: keep, fixed-leg forward slow-turn baseline.

### Fixed-leg forward slow-turn + push
Task: `Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale2p5-Smooth-MidForward-Slew6-Push-v0`  
Run: `2026-07-03_18-37-54_slow_speed_turn_slew6_push_probe_seed1`  
Checkpoint: `model_991.pt`  
Status: keep, fixed-leg forward turn + push baseline.

## Failed / Do Not Continue

### SlowSpeed Easy LinSign from Push L3
Task: `Mjlab-HopperTrex-Balance-SlowSpeed-Easy-LinSign-v0`  
Run: `2026-07-03_21-00-43_slow_speed_easy_linsign_from_push_l3_seed1`  
Checkpoint: `model_3145.pt`  
Status: failed, do not continue.

Evidence:
```text
cmd_lin_x < -0.01
mean actual_lin_x: +0.02902
lin sign match: 0.226
```

### SlowSpeed Easy LinSign LegAssist from Robust L2
Task: `Mjlab-HopperTrex-Balance-SlowSpeed-Easy-LinSign-LegAssist-v0`  
Run: `2026-07-03_21-18-23_slow_speed_easy_linsign_legassist_from_robust_l2_seed1`  
Checkpoint: `model_149.pt`  
Status: failed, do not continue.

Evidence:
```text
Mean episode length: 52.43
Episode_Termination/bad_orientation: 4.4348
Episode_Reward/wheel_ground_contact: 0.0077
cmd_lin_x > 0.01 mean actual_lin_x: -0.02717
cmd_lin_x > 0.01 lin sign match: 0.330
```

### Bidir fixed-leg turn probes
Runs:
- `2026-07-03_19-46-42_slow_speed_turn_bidir_slew6_push_probe_seed1`
- `2026-07-03_19-58-22_slow_speed_turn_bidir_slew6_push_linsign_probe_seed1`
- `2026-07-03_20-12-37_slow_speed_turn_bidir_lowyaw_slew6_linsignstrong_probe_seed1`

Status: failed, reverse x tracking not solved.

### Limited Leg Assist Safe probe
Task: `Mjlab-HopperTrex-Balance-SlowSpeed-Easy-LinSign-LegAssistSafe-v0`  
Source: `2026-06-27_10-49-06_robust_l2_seed1/model_1997.pt`  
Migration run: `migrated_robust_l2_to_slow_speed_easy_linsign_legassist_safe_seed1`  
Attach run: `slow_speed_easy_linsign_legassist_safe_attach_seed1`  
Speed run: `slow_speed_easy_linsign_legassist_safe_seed1`
Checkpoint: `model_198.pt`
Status: failed, do not continue.

Evidence:
```text
cmd_lin_x < -0.01
mean actual_lin_x: +0.01464
lin sign match: 0.327

cmd_lin_x > 0.01
lin sign match degraded from fixed-leg control 0.807 to 0.687
```

## Current Next

### Fixed-leg command/action sign sanity
Status: pending.

Use only fixed legs. Check action-to-velocity sign directly, then train separate
ForwardOnly and BackwardOnly sanity tasks before returning to any combined task.
