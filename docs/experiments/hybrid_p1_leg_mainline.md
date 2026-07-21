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

Status: **READY, NOT RUN**.

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
magnitude and saturation, terminations, and non-wheel contact. The matrix
stops for analysis after all three rows; it does not auto-promote the best
observed row or launch more seeds/iterations.

Machine-room entry point:

```powershell
& .\scripts\run_hybrid_leg_authority_seed1.ps1
```
