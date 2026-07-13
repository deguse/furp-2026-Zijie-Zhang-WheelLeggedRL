# Project Operating Notes

## HopperTrex RL Stage Transfer Rule

Stage transitions must not blindly resume from the previous stage checkpoint.
Before training a new curriculum stage, inspect the actor action std and reset it
when the previous stage has collapsed exploration.

For HopperTrex fixed-leg wheel-only stages, the Stage0 -> Stage1 failure mode was:

- Stage0 balance trained a stable policy but collapsed `distribution.std_param`
  / mean action std to a very small value.
- Direct Stage1 resume kept the policy trapped in a balance-only attractor.
- Reward-only changes produced a false tradeoff: stronger velocity caused
  `bad_orientation`, stronger safety caused weak/no forward motion.
- The actual unlock was migrating from the Stage0 checkpoint with action std
  reset, e.g. `migrate_1d_to_slow_speed.py --action-std 0.15`, then training
  with enough entropy for Stage1 exploration.

Required checklist for every new HopperTrex curriculum stage:

1. Print or inspect the checkpoint `distribution.std_param` / training
   `Mean action std`.
2. If std is already collapsed, create a migrated checkpoint with an explicit
   action std reset instead of direct `--agent.resume=True` continuation.
3. Use short probe training first, then gate and viewer.
4. A short default gate is not enough for stage transfer. The default training
   env uses short episodes, while play/viewer uses effectively infinite
   episodes. Before advancing stages, run a viewer-equivalent long fixed-command
   diagnostic with `evaluate_fixed_command.py --play-cfg --episode-length-s
   1000000000` and fixed commands for every active direction. If long-horizon
   behavior disagrees with the short gate, the stage has not passed.
5. In Viser, velocity command sliders only override the selected environment
   when the command GUI checkbox is enabled. If it is disabled, the command term
   keeps resampling and may sample standing commands. Do not use that as proof
   of policy failure; confirm with fixed-command diagnostics.
6. Before entering the next stage, or whenever a checkpoint becomes a key
   candidate for rollback/comparison, explicitly remind the user to preserve it
   in the workspace and record the task id, run directory, checkpoint filename,
   gate output, fixed-command output, and viewer verdict.
7. Before giving any command for `train.py`, `play.py`, gate scripts, or
   diagnostics, validate the CLI flags against that exact script's `--help` in
   the current checkout. Do not assume common aliases such as `--device` work
   across scripts; use the names exposed by the actual entrypoint.
8. Do not keep changing reward terms until command sampling, action sign,
   action std, and gate metrics have all been checked.
9. Hybrid gate runs must label `--profile screen` or use the default formal
   profile. Screen results are rejection-only and must never be aggregated or
   used for stage promotion; formal live gates require at least 3000 steps.
10. Hybrid Stage2-5 transitions are adjacent-only. Use
    `migrate_hybrid_stage.py` for `N -> N+1`; do not bypass checkpoint
    provenance, controller/calibration binding, or collapsed-std preflight.
