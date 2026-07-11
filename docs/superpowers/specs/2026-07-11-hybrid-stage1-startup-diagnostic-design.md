# Hybrid Stage1 Startup Diagnostic Design

## Goal

Diagnose the all-environment `root_too_low` startup failure without training,
changing controller behavior, lowering safety thresholds, or hiding the failure
behind a longer termination grace period.

## Rollout

Run Hybrid Stage1 for ten control steps with termination-driven reset disabled.
Compute the original termination predicates as observations. Compare three
deterministic scenarios: zero residual, seed-fixed balance residual sampled at
std 0.15, and controller-off zero wheel command.

## Evidence

Write one CSV row per scenario, environment, and step. Record raw MuJoCo root
qpos z, derived root-link z, vertical velocity, pitch and pitch rate, command,
controller baseline, raw and applied residual, wheel targets, contacts, and the
termination predicates that would have fired.

Write a JSON summary with code and artifact provenance, per-step z statistics,
first contact and termination steps, maximum raw/derived z disagreement, and a
conservative classification. Supported classifications are
`derived_state_stale`, `invalid_reset_height`, `controller_startup_failure`,
`exploration_startup_failure`, `contact_initialization_failure`,
`termination_source_mismatch`, and `inconclusive`.

## Safety

The diagnostic does not learn, persist a policy, modify artifacts, or claim a
gate result. The reset grace experiment introduced by commit `32cb602` is
removed because the machine-room two-iteration probe falsified it: every
environment still terminated as soon as the grace ended.

## Acceptance

Unit tests cover deterministic actions, classification boundaries, non-finite
input rejection, summary provenance, and restoration of the original Hybrid
root-height termination. A CPU integration test covers environment reset/step
and artifact writing. The machine-room run uses seed 1, 16 environments, and 10
steps and produces CSV, JSON, and a retained console log.
