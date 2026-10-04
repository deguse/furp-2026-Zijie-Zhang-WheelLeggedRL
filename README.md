# HopperTrex: Traceable Hybrid Wheel-Leg Control

FURP 2026 - Faculty of Science and Engineering, University of Nottingham Ningbo China

The project integrates an identified classical controller, calibrated references, bounded residual PPO, staged evaluation and artifact provenance. Its contribution is a testable engineering system, not a new PPO or physics algorithm.

## Representative evidence

| Experiment | Result | Scope |
|---|---|---|
| Stage5 recovery | 1.0130 s to 0.8872 s; 12.42% improvement | Formal, one training seed, 128 disturbance events per arm |
| Leg-masked evaluation | 3.82% improvement against its own matched baseline | Same checkpoint; not an exact causal contribution percentage |
| Later C1 qualification | 15/15 flat-control cells passed | Separate controller version from the recovery experiment |
| Formal RollBoundary | Flat 96/96; 2.5 mm 18/96 | Failed strict support qualification; no safe positive stair capability established |



## System components

- Robot assets, task observations/actions, resets and curriculum configuration.
- Identified feedback, velocity/yaw calibration, posture mapping and reference shaping.
- Fixed six-channel residual interface with capability masks, bounds and migration checks.
- Gate records bound to source revisions, controller artifacts and checkpoints.
- Portable classical control and deployment interfaces with mocks and safety supervision.

The 2026-10-03 research release preserves the latest `codex/p2-classical-upper-bound` work, including previously uncommitted R0c diagnostics and their tests. Historical branches and experiment SHAs remain available. Inclusion is not a claim that diagnostic features passed physical qualification.

## Limitations

Simulation-only. No independent-seed generalization, reliable stair climbing, USB-CAN integration or real closed-loop hardware result is claimed. Contact/backend and effort-control investigations remain development diagnostics. No new training or physics experiments were run for closeout.

## Contribution and reproducibility

Zijie Zhang: research questions, experiment design, integration, diagnosis and evidence review. Implementation was collaborative and AI-assisted; sole authorship of every code path is not claimed.

The public [evidence bundle](research/evidence_bundle_2026-09-29.zip) contains core records, two checkpoints, frozen diagnostic source bindings and scripts for rebuilding the reported metrics. [SHA-256 manifest](research/evidence_manifest.json). The original local closeout remains immutable.

## Install and verify

See [reproducibility instructions](docs/REPRODUCIBILITY.md) for the pinned Windows/Python 3.11 environment and two-repository installation. Run `python -B scripts/reproduce_closeout.py --output ../closeout_verify` for an offline, standard-library-only verification of the numbers. The output directory must be new.

| Directory | Contents |
|---|---|
| `src/hoppertrex_mjlab` | Robot assets, tasks, classical/residual controllers, deployment contracts and evaluation tools |
| `tests` | Software and separately identified physics-runtime tests |
| `scripts` | Version-bound launchers and the safe installation/verification entrypoints |
| `research` | Frozen public evidence and historical workspace diagnostic source |
| `docs` | Experiment records and technical runbooks |
| `validation` | Publication-time software and integrity checks |

[Third-party notices and reuse scope](docs/THIRD_PARTY_NOTICES.md). This is a research-source release, not a hardware-ready product.
