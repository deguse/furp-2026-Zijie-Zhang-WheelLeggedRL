# HopperTrex Hybrid v2 Design Audit

Date: 2026-07-10

Status: local implementation audit complete; remote controller/posture data
collection, checkpoint gates, GPU training, and Viser review are not complete.

## Evidence Policy

Current conclusions use this evidence order:

1. Current code and task registration.
2. Checkpoint metadata and exact run provenance.
3. Git status and commit SHA.
4. Complete gate JSON from the matching checkout.
5. User-observed Viser behavior.
6. `D:\mjlab_workspace\handover.md` only as a historical lead.

The handover has not been kept current. Its reported passes and failures are
historical claims until the corresponding checkpoint metadata and current gate
output are available.

## Robot And Runtime Facts

- HopperTrex has two physical legs with four controlled leg joints:
  `thigh_left_01`, `thigh_right_01`, `knee_left`, and `knee_right`.
- The wheel joints are `wheel_left` and `wheel_right`.
- Hybrid v2 currently targets flat-ground MjLab simulation at 50 Hz.
- Wheels remain velocity-controlled and the four joints of the two legs remain
  position-controlled.
- Sim-to-real, hardware I/O, QP, and WBC are outside the current baseline.

## Current State

| Item | Current fact |
| --- | --- |
| Legacy Stage0-8 | Registrations and behavior remain available; only shared evaluator/metric fixes were made. |
| Legacy Stage2 candidates | `model_122.pt` and std-reset `model_24.pt` are unresolved and have no current full gate result. |
| Hybrid code | Stage0-5 tasks, six-dimensional action, identification, posture fitting, migration, and capability gates are implemented locally. |
| Controller artifact | No remote identified and qualified artifact has been recorded. Local fallback is explicitly labelled unqualified PD. |
| Posture artifact | No remote static-sweep artifact has been recorded. Local fallback is a fixed initial posture. |
| Hybrid training | Not started. No GPU result, promoted checkpoint, or Viser verdict exists. |

## MDP Audit

### Current code facts

The actor interface is invariant across Hybrid Stage0-5 and has 34 values:

| Observation | Dimension |
| --- | ---: |
| base linear velocity | 3 |
| base angular velocity | 3 |
| projected gravity | 3 |
| velocity command `[vx, vy, yaw]` | 3 |
| posture command `[height, pitch]` | 2 |
| six joint positions, with wheel positions zeroed | 6 |
| six joint velocities | 6 |
| controller wheel baseline | 2 |
| applied, scaled, masked residual | 6 |

The actor does not observe the raw output of disabled heads. It observes the
residual after clipping, scaling, and the stage mask. Actor observations are
corrupted during training; critic observations use the same terms without
corruption.

The balance controller state is:

```text
[pitch, pitch_rate, vx_error, signed_wheel_speed_error]
```

This state is compact and physically interpretable. It also contains correlated
velocity signals, so controller qualification must rely on independently
excited identification data and held-out prediction rather than training fit.

### Decision

The stable observation contract is preferable to the legacy stage-dependent
contract because checkpoints can move between capabilities without changing
network dimensions. Keep it for Hybrid v2.

## Action And Controller Audit

The fixed policy action order is:

```text
[
  wheel_balance_residual,
  wheel_yaw_residual,
  left_thigh_residual,
  right_thigh_residual,
  left_knee_residual,
  right_knee_residual,
]
```

The first two outputs are composed as:

```text
left_wheel_residual  = -balance + yaw
right_wheel_residual = +balance + yaw
```

The four remaining outputs are residual targets for the four joints of the two
legs. All outputs are clipped, stage-masked, and scaled. Final wheel targets
receive slew and velocity limits; final leg targets receive joint-position
limits.

Newly enabled actor rows are zeroed during checkpoint migration. Exploration
standard deviations are restored per capability:

```text
balance: 0.15
yaw:     0.10
legs:    0.05 each
```

Identification fits `x[k+1] = A x[k] + B u[k]`. LQR is used only when the
four-state model has controllability rank four and held-out one-step NRMSE is at
most 15%. Otherwise the artifact and runtime remain explicitly labelled `pd`.
No fallback may be reported as LQR.

### Training blocker: controller provenance

The no-file runtime fallback is useful for CPU tests but is not a research
controller artifact: it has no gain hash and is marked unqualified. Stage0
remote qualification must use a serialized LQR or explicit-PD artifact with its
gain hash. Stage1 training must not begin from the local fallback.

### Resolved locally: disabled-head reward leakage

The inherited MjLab action-rate and action-acceleration terms operate on raw
policy output. Hybrid v2 now replaces both functions, while preserving their
configured weights, with terms computed from the current and two previous
`applied_residual` tensors. Those tensors are clipped, stage-masked, and scaled
before entering either smoothness reward.

Focused tests inject large changes into raw disabled dimensions and verify that
only applied residual history contributes. Per-head migration remains necessary
when a capability is enabled, but disabled heads no longer receive a direct
smoothness penalty before that transition.

## Posture Audit

`PostureCommandCfg` supplies `[target_height, target_pitch]`. Sweep samples are
accepted only with:

- no non-wheel contact,
- at least 10% joint-range margin on all four leg joints, and
- actuator load below 80% on all four leg joints.

The sampled height and pitch ranges are each shrunk inward by 10%, with
`abs(target_pitch) <= 0.08 rad`. An affine local map predicts the nominal four
joint targets from `[1, height, pitch]`; PPO adds four small joint residuals.

### Resolved locally: rectangular-envelope validity

The artifact generator maps samples to their height-pitch grid and selects the
largest axis-aligned rectangle for which every sampled grid point is feasible.
The rectangle must span at least two heights and two pitches; scattered sets
without a complete `2 x 2` interpolation cell are rejected. The verified range
is then shrunk inward by 10% and capped at `abs(pitch) <= 0.08 rad`.

The JSON records `all_feasible_grid_rectangle` and its grid shape. Runtime
loading rejects posture artifacts without that verification record. The remote
static sweep and its qualified artifact are still pending, so Stage3 cannot
start yet even though the local construction error is fixed.

## Reward Audit

### Legacy code facts

The legacy factory combines contact/support, upright, height, orientation,
velocity tracking, sign alignment, command bands, overspeed, wheel-target
smoothness, action smoothness, and stage-specific shaping. Stage2
`PrecisionCenter` adds center-speed pressure on top of the existing safety,
direction, band, delta, overspeed, and tail terms.

This reward family is reproducible but difficult to reason about because a large
constructor selects many overlapping terms through boolean variants. Reward
improvement alone is not evidence that a physical capability passed.

### Historical claims

The stale handover and experiment inventory report:

- clean fixed-leg balance and robust L1/L2 passed,
- Push L3 passed,
- slow bidirectional tracking remained seed-sensitive,
- wheel-only fixed-leg control struggled to combine pitch recovery, linear
  direction, and yaw, and
- repeated reward micro-tuning did not solve reverse tracking.

These claims motivate the classical-controller plus residual-PPO direction, but
they do not replace the pending current gate on `model_122.pt` and `model_24.pt`.

### Hybrid decision

Hybrid v2 reuses established safety/contact and tracking rewards to limit
behavioral churn. When posture commands are enabled, fixed flat-orientation and
fixed-height terms are replaced by command-relative height and pitch errors.
Promotion is determined by external capability metrics, not mean reward.

Do not resume unbounded reward tuning if both legacy Stage2 candidates fail.
Record their failure modes and retain them as historical pure-PPO baselines.

## Curriculum Audit

| Stage | Enabled policy outputs | Required capability |
| --- | --- | --- |
| 0 | none | controller-only standing and `vx = +/-0.07 m/s` for 60 s |
| 1 | balance residual | no-regression at `vx = 0, +/-0.07 m/s` plus measurable improvement over zero-residual LQR during kicks or command transitions; `+/-0.10 m/s` is boundary-only |
| 2 | balance and yaw residuals | axis and combined planar commands |
| 3 | all four leg-joint residuals plus wheel residuals | posture center and verified boundaries |
| 4 | all six outputs | fixed and random integrated commands |
| 5 | all six outputs | robust level 2 plus 3-5 s velocity kicks |

This capability order is reasonable: it qualifies the non-learned baseline
first, adds wheel objectives separately, then adds the two-leg posture map, and
only then introduces integrated randomization. Stage transitions must use the
per-head migration tool and a 100-iteration single-seed probe before long runs.

## Gate Audit

Gates are capability-suite driven rather than keyed only by a numeric stage.
The shared metric decisions use inclusive threshold comparisons, and a missing
or non-finite metric fails. Result envelopes include task, git SHA, controller
gain hash, seed, checkpoint, metrics, and every individual decision. Exactly
three unique seeds are required for Hybrid aggregate reports.

The legacy evaluator remains CLI compatible and now accepts `--seed`. Its
Stage2 fixed-command checks reuse the same linear capability rules as Hybrid v2.

Known gate semantics retained for compatibility:

- legacy `terminated_event_rate` is terminated events divided by environment
  count, not unique terminated environments;
- Stage5 survival uses the fraction of unique environments that never
  terminated; and
- a recorded git SHA is auditable only when the worktree is clean.

Every promotion requires all fixed long-horizon scenarios, three evaluation
seeds, and Viser review. Persistent oscillation, command pulses, or drift seen
in Viser must first become a reproducible metric and threshold.

Stage1 uses a matched ablation rather than an absolute low-speed gate. Candidate
and zero-residual LQR use the same seed and profile. The gate rejects nominal
tracking, pitch, or oscillation regressions, bounds residual authority, and
requires at least one 10% improvement in a kick, transition, or boundary metric.
Boundary commands are diagnostic evidence and are not added to the training
range; a boundary rollout with termination or excessive pitch cannot supply the
required improvement evidence. Offline scenario files receive the same
complete-coverage validation as live rollouts, so a partial scenario list
cannot bypass a Stage1-5 gate.

## Audit Decision

Do not append more legacy Stage2 training now. Run the pending three-seed,
3000-step `PrecisionCenter` gates when both real remote checkpoint paths are
resolved.

Hybrid v2 is ready for remote data collection and Stage0 qualification, but PPO
training has not started. A hashed, explicit controller artifact and passing
Stage0 suite are required before Stage1. A remote verified posture artifact is
required before Stage3. QP/WBC remains future work and is not part of the
present mainline.
