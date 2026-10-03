# HopperTrex: Traceable Hybrid Wheel-Leg Control

FURP 2026 - Faculty of Science and Engineering, University of Nottingham Ningbo China

## Closeout materials

- [Showcase poster - A0 landscape](FURP_Showcase.pdf)
- [Technical report](FURP_Report.pdf)

The project integrates an identified classical controller, calibrated references, bounded residual PPO, staged evaluation and artifact provenance. Its contribution is a testable engineering system, not a new PPO or physics algorithm.

## Representative evidence

| Experiment | Result | Scope |
|---|---|---|
| Stage5 recovery | 1.0130 s to 0.8872 s; 12.42% improvement | Formal, one training seed, 128 disturbance events per arm |
| Leg-masked evaluation | 3.82% improvement against its own matched baseline | Same checkpoint; not an exact causal contribution percentage |
| Later C1 qualification | 15/15 flat-control cells passed | Separate controller version from the recovery experiment |
| Formal RollBoundary | Flat 96/96; 2.5 mm 18/96 | Failed strict support qualification; no safe positive stair capability established |

The Stage5 comparison uses its historical classical stack, not the later C1 gain schedule. Formal gates can produce valid negative results. A completed run or a working interface does not imply a qualified capability.

## System components

- Robot assets, task observations/actions, resets and curriculum configuration.
- Identified feedback, velocity/yaw calibration, posture mapping and reference shaping.
- Fixed six-channel residual interface with capability masks, bounds and migration checks.
- Gate records bound to source revisions, controller artifacts and checkpoints.
- Portable classical control and deployment interfaces with mocks and safety supervision.

Latest research development resides on branch `codex/p2-classical-upper-bound`. The local closeout snapshot preserves the existing uncommitted R0c diagnostics; it is not a clean software release. Historical experiments retain their own revisions.

## Limitations

Simulation-only. No independent-seed generalization, reliable stair climbing, USB-CAN integration or real closed-loop hardware result is claimed. Contact/backend and effort-control investigations remain development diagnostics. No new training or physics experiments were run for closeout.

## Contribution and reproducibility

Zijie Zhang: research questions, experiment design, integration, diagnosis and evidence review. Implementation was collaborative and AI-assisted; sole authorship of every code path is not claimed.

The local closeout package at `D:/mjlab_workspace/closeout_2026-09-29` contains the evidence matrix, core records/checkpoints, source snapshot, SHA256 manifest and scripts that regenerate the displayed numbers and figures without simulation. It excludes caches, virtual environments and credentials. No automatic public upload, push or branch merge was performed.
