# Hybrid P1 Leg Mainline

## P1.1 Expanded Posture Envelope

Status: **PASS** (seed 1, RTX machine-room rollout, code `6e1f03c` plus
the height-floor fitter at `5c20e8e`). This is a development qualification,
not a three-seed promotion result.

The expanded static sweep contained 121 cells. After the measured dynamic
height floor (`height >= 0.28 m`) and height-priority inscription, the
qualified command rectangle is:

```text
height = [0.2907321708, 0.3276857266] m
pitch  = [0.0000000000, 0.0320000000] rad
legacy coefficient map_hash = 8849ce39ff24b3342376dbae9c62d658c01288ad8c2b71dcd2ec20741b19a2f1
```

The uncompensated 25-cell balance probe had zero terminations and zero
non-wheel contacts, but measured worst steady drift `0.0393466 m/s`.
The station calibration reduced the compensated worst drift to
`0.0066133 m/s` (about 83%, below the `0.015 m/s` qualification limit).
The compensated grid also recorded:

```text
worst height RMSE     = 0.0008473 m
worst pitch RMSE      = 0.0061633 rad
worst pitch-rate p99  = 0.22465 rad/s
terminations          = 0
non-wheel contact     = 0
vx +/-0.05 error      = about 0.0019-0.0048 m/s
```

Canonical machine-room evidence (stored outside Git):

| Artifact | SHA-256 |
| --- | --- |
| `posture_map_seed1_floor028.json` | `eefe72151b1ada42d2e95ffe582dafbe11f6616e46703f71d243831d911a69a6` |
| `balance_uncompensated_floor028_seed1.json` | `d90fc874811fb171d5fb8d76f0b12416e771ba58cb6a42b49db5218099d78623` |
| `station_calibration_floor028_seed1.json` | `6a325c209bac17904276c5f37b030ecdfc327a6ffa9fc7248a3e3cab535dacdf` |
| `balance_compensated_floor028_seed1.json` | `ec566af60aa17968c3a82eba5a17045242a2853567496d1f39c60f6b9b85eccb` |

The legacy `map_hash` covers fitted coefficients only. For P1.2, the same
measured data was migrated offline into a full-envelope identity without a
new simulation run:

```text
posture_artifact_hash = 0d54fca78b38a880678d0ee69964ac86cb18e1a1f62a0ee716a4715071687ad3
station_calibration_hash = a4d805ce87fff2ef786c740ff366d24833e4c1162a9f70740cc1941dbeaf004a
full-hash posture file SHA-256 = 7408ff66f97619d0c89324b653d1fea1c41a8ff8d4bb73b4539bfe9cb24cf2af
full-hash station file SHA-256 = 551974d1d8f785c299d0a7b15e3caf61af45634f92aea65790eb166055582cb9
repository posture JSON SHA-256 = 593f3e9770bbb4b7afd74b0c6fa935603146c968c487aafd28e56ac500dbc5e0
repository station JSON SHA-256 = 7b7e84bbbd660680ca0f7a0799bfc83199a132ea2a1770a5f74478b32dc17c3a
```

Legacy Stage5 artifacts remain valid under coefficient-only binding. New P1
artifacts bind both `map_hash` and `posture_artifact_hash`; a station artifact
from another command envelope is rejected even when coefficients are equal.

## P1.2 Leg Residual Authority

Status: **RUN, STOP_FOR_ANALYSIS** (seed 1, RTX machine-room rollout,
code `136dc00`).

Pre-registered matrix:

```text
leg residual scale = {0.035, 0.070, 0.100} rad
seed = 1
training = 256 envs x 100 iterations
save interval = 25; screen newest K=3 checkpoints
recovery = center posture, 8x kick, 32 envs x 4 kicks = 128 events
```

Before training, the zero-residual recovery is repeated twice to measure the
run noise. Every scale must report retention and robust safety screens,
matched full-policy recovery, evaluation-time leg ablation, residual
magnitude and saturation, terminations, and non-wheel contact.

The matrix stopped for analysis after all three rows. Five checkpoints passed
the retention screen, but all five failed the robust screen on the same four
correlated zero-speed standing metrics. No failure involved a termination or
non-wheel contact. Because no checkpoint passed both screens, the protocol
correctly did not enter the 128-event recovery or evaluation-time leg-ablation
probe. It did not select a scale, launch more seeds, or add iterations.

The five P1.2 candidate checkpoint files are no longer available. The local
`D:\mjlab_workspace\model_99.pt` file, when present, has SHA-256
`48a053f0d1acaa799b105c20bce29d5f4a738da36fd9117d4bdf5f03f2c63937`.
It is the frozen formal Stage5 candidate, not one of the P1.2 candidates, and
must not be relabelled or used to reconstruct the P1.2 matrix.

Machine-room entry point:

```powershell
& .\scripts\run_hybrid_leg_authority_seed1.ps1
```

This command is historical only. Do not rerun the matrix to replace the lost
checkpoint files.

## P2 k.0 Classical Stair Height Probe

Status: **FORMAL GPU RUN, CLASSICAL_DEATH_HEIGHT_BRACKETED** on branch
`codex/p2-stair-probe` at `9edb8b7` (seed 1, RTX machine room,
MjLab `43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6`).

The zero-residual classical stack is evaluated on MjLab `pyramid_stairs` at
heights 0.00-0.10 m in 0.01 m increments with a fixed 0.30 m step width. Each
height uses 16 seed-controlled reset trials and three repeats for each of two posture cards:
the envelope center `(0.3092089487 m, 0.016 rad)` and the high posture
`(0.3276857266 m, 0 rad)`. Runs start on flat ground outside the first riser,
settle for 2 s, command `+0.07 m/s` for at most 10 s, and require 0.5 s beyond
the first riser with no termination or non-wheel contact. Reset x/y and small
forward/pitch-rate perturbations are generated reproducibly from seed 1 and
saved in every trial row; their registered bounds are 0.02 m, 0.03 m,
0.01 m/s, and 0.02 rad/s respectively.

The probe is a non-promotable allocation measurement. It loads no checkpoint
or yaw artifact, applies six zero policy actions, reports both posture cards
separately, and never starts P3. If only one posture card fails within the
registered sweep, the result is `EXTEND_SWEEP_BEFORE_P3` until both cards have
a common failure height; it is not variance evidence. The machine-room entry
point is:

```powershell
& .\scripts\run_hybrid_p2_stair_height_probe.ps1
```

The corrected formal run measured the same monotonic hard cliff on both
pre-registered posture cards: every flat-control trial passed, every trial at
0.01-0.10 m failed to cross the first riser, and no trial terminated or made
non-wheel contact. The first common failure height and recorded P3 candidate
is therefore 0.01 m. The first retained GPU run at `f4a4d4b` had already
completed all 1056 trials, but its wrapper rejected every trial because a
float64-exact `1.0e-9` root-height assertion was tighter than float32 GPU
state read-back. Commit `9edb8b7` widened only that assertion to
`5.0e-8` m; it did not change the terrain, reset, command, success, or
classification protocol.

## P2 k.0 Stall-Mechanism Diagnostic

Status: **READY, NOT RUN** on `codex/p2-stair-probe`. This is an
observational follow-up to the 1 cm hard cliff, not a gate and not a P3
promotion.

The diagnostic compares eight paired command cells on flat ground and the
0.01 m stair. It keeps the envelope-center height, sweeps qualified and
diagnostic-only lean-in pitches, and adds two 0.10 m/s cells that remain inside
the velocity-calibration domain but outside the Stage5 task range. Each cell
uses the same 16 seed-controlled resets, settles for 4 s, drives for 10 s, and
records the final 3 s stall window. The recorded channels are the signed
forward wheel target and wheel speed using `0.5 * (right - left)`, modeled
actuator torque and saturation, wheel slip, body velocity, pitch error, and
progress toward the riser.

The fixed classifications distinguish a viable classical card, friction-
limited wheel spin, torque-saturated stationary stall, balance-loop drive-
target collapse, a mixed mechanism, and an invalid flat baseline. Candidate
cells must also pass their paired flat control. The run loads no checkpoint or
yaw artifact and cannot authorize training or promotion. Machine-room entry
point after the branch is committed and pulled:

```powershell
& .\scripts\run_hybrid_p2_stall_diagnostic.ps1
```
