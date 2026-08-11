# Project Operating Notes

## User-Mandated Collaboration Rules

These rules govern all work on this project and must be followed before
starting implementation, investigation, or operational guidance.

1. **Search and reuse before building anything.** Inspect the repository's
   existing code, scripts, documentation, logs, and prior conclusions first.
   For external facts, tooling, known issues, and established solutions,
   prefer the Tavily Hikari MCP search tools and cross-check important claims
   against multiple independent sources plus the local implementation or
   actual runtime output. Do not recreate an existing solution or spend time
   re-solving work that is already available.
2. **Do not make simple work complicated.** Use the shortest adequate and
   reliable solution for simple tasks. Do not introduce unnecessary wrappers,
   launchers, abstractions, agents, preflights, or multi-step procedures when a
   direct answer, existing command, or small parameter change solves the task.
   Simplicity must not come at the expense of correctness.
3. **Analyze ambiguity first; ask only at consequential decisions.** First use
   the available context, repository evidence, runtime output, and search to
   resolve uncertainty independently. Do not ask the user reflexively whenever
   a problem appears. Ask before a materially consequential choice, including
   experiment direction, reward/protocol/training changes, costly or
   irreversible actions, or alternatives with substantive tradeoffs. When
   asking, state the confirmed facts, the unresolved decision, the options,
   and the recommended choice.

(Codex: 2026-08-11, recorded verbatim in substance after explicit user approval.)

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
11. Probe before PPO (Hybrid v3 k.0/k.1 discipline). Before assigning any
    control channel to the learned residual, run a zero-residual physical
    probe of that channel under the qualified classical layer. If the probe
    shows the channel is classically tractable (monotone, first-order,
    constant-gain), the classical layer must own it — calibrated feedforward
    or scheduled gains — and PPO keeps only a residual margin around it.
    Every gate threshold must cite a measured noise floor from such a probe
    (see `Rule.source` in `hybrid_gate.py`); a threshold below the physical
    floor is unsatisfiable by construction, not strict. Corollary: a trained
    residual that holds a large, constant output on some channel is not
    "learning the skill" — it is compensating for a misallocated channel,
    and the fix is reallocation to the classical layer, not more training
    (falsified twice on 44a44b1: Stage2 yaw at a constant 0.35 action, gate
    yaw_delta_rms limit 0.035 below the measured 0.064 floor).
