# Hybrid v2 Stage0 Frozen Baseline

Date: 2026-07-11

This document freezes the first reproducible Hybrid v2 baseline. Large binary
artifacts, checkpoints, and complete logs remain on the machine-room server;
only their provenance, relative paths, and acceptance results are tracked in
Git.

## Status

| Item | Result |
| --- | --- |
| Calibrated Stage0 gate, seeds 1/2/3 | PASS |
| Calibrated Stage0 aggregate gate | PASS |
| Stage0 Viser inspection | PASS |
| Posture sweep | 21/49 feasible samples |
| Posture map | Qualified local map |
| Hybrid PPO | Not started |

## Immutable Provenance

- Stage0 code SHA: `de4ba075ff8b`
- Posture sweep code SHA: `8ad0e3fdb134`
- Controller gain hash: `8fee25a0339dd1e99127cbed912941dc3ad8ef2030ce49a0d310d1563cb87d98`
- Velocity calibration hash: `f62648b57bd17a3503bcbdbf58f349f91fcd8de8ef0cf04551c200401233ed01`
- Posture map hash: `ed17d232500284a4cc6da01ef137dc59181344c8aec18a2c595b021cab6fde24`

The qualified velocity calibration uses scale `0.86` and bias `-0.012 m/s`.
The controller and calibration artifacts are separate: the calibration is
bound to the controller gain hash and does not replace or rename it.

## Server Artifact Manifest

Paths are relative to the machine-room repository root.

- `experiments/hybrid_v2/artifacts/de4ba075ff8b/stage0_gate_calibrated/seed1.json`
- `experiments/hybrid_v2/artifacts/de4ba075ff8b/stage0_gate_calibrated/seed2.json`
- `experiments/hybrid_v2/artifacts/de4ba075ff8b/stage0_gate_calibrated/seed3.json`
- `experiments/hybrid_v2/artifacts/de4ba075ff8b/stage0_gate_calibrated/aggregate.json`
- `experiments/hybrid_v2/artifacts/8ad0e3fdb134/posture_sweep_seed1.npz`
- `experiments/hybrid_v2/artifacts/8ad0e3fdb134/posture_map_seed1.json`

The corresponding `.log` files remain beside the JSON files on the server.
Identification NPZ files, posture-sweep NPZ files, training checkpoints, Viser
recordings, and full logs are intentionally not committed to Git.

## Stage0 Visual Acceptance

Static balance, forward motion, reverse motion, and return to zero were normal.
There was no abnormal jitter, geometry penetration, posture discontinuity, or
periodic wheel-speed pulse. Return to a zero command was stable.

Mild, bounded fore-aft active-balancing motion was visible and is accepted for
this two-wheel inverted-pendulum baseline. A later policy fails visual review if
this motion grows into a persistent limit cycle, wheel-speed pulse, contact
violation, or clear tracking regression.

## Posture Map Scope

- Feasible samples: `21/49`
- Verified rectangle: `7 x 3`
- Height envelope: `[0.2992646, 0.3090443] m`
- Pitch envelope: `[0.0116717, 0.08] rad`

This is a narrow, positive-pitch-biased local safety map, not a large-range or
zero-centered posture controller. Stage1 keeps both legs fixed and does not use
posture commands, so this limitation does not block its probe. Stage3 and later
must explicitly load this posture map, or another qualified replacement bound
to the same controller and calibration provenance.

## Stage1 Entry Rule

Stage1 may begin only from a bootstrap checkpoint generated on the merged
`master` SHA with explicit controller and calibration paths. Development uses
seed 1, 256 environments, and a 100-iteration probe with checkpoints at 25, 50,
75, and 100 iterations. No three-seed or 3000-iteration training run is
authorized by this baseline freeze. Stage1 must demonstrate residual value
against a matched zero-residual LQR in disturbance or command-transition
metrics; reproducing nominal low-speed tracking alone is not a Stage1 result.
