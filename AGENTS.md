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
4. Do not keep changing reward terms until command sampling, action sign,
   action std, and gate metrics have all been checked.

