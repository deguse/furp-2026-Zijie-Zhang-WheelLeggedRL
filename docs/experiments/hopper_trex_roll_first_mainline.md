# HopperTrex roll-first stair mainline

Status: implemented protocol; no formal RollBoundary or RollAssist result is claimed by this document.

## Evidence and scope

The control allocation is deliberately narrower than the earlier StairCamp/StairDynamic route:

1. R0 measures direct wheel roll-over under the frozen final C1 classical stack.
2. R1, only after a positive safe R0 bracket, grants PPO the four leg residuals while the classical controller permanently owns both wheel channels.
3. Contact-triggered lifting, synchronized/alternating feedforward, a jumping/landing FSM, online height classification, multi-seed claims, and real-hardware claims remain Future Work.

The literature supports the existence of distinct nearby modes, not a universal height threshold:

- [Li et al., Balancing Control and Pose Optimization](https://arxiv.org/abs/2109.09934): posture optimization plus wheel drive can roll wheel-legged platforms over obstacles and multi-step terrain.
- [Chamorro et al., Ascento stair climbing](https://arxiv.org/abs/2402.06143): dynamic, contact-rich wheel-leg climbing of 15 cm stairs; it does not prove HopperTrex capability.
- [CTBC](https://arxiv.org/abs/2509.02986): contact-triggered guided leg motion plus RL, retained only as Future Work here.
- [Ascento jumping robot](https://arxiv.org/abs/2005.11435): jumping requires purpose-built mechanics and separate qualification, also retained as Future Work.

No cited result establishes a universal `1 cm` or fixed `h/r` boundary. Every HopperTrex boundary is reported as a measured interval.

## R0: RollBoundary

Entrypoints:

- `src/hoppertrex_mjlab/scripts/probe_roll_boundary.py`
- `scripts/run_roll_boundary.ps1`

The first sweep is `0, 2.5, 5.0, 7.5, 10.0 mm`, represented by integer-micrometre terrain keys. It uses seed 1, two registered posture cards, 16 environments per height and three repeats. Residuals are identically zero; dynamic stair control, contact triggers, reference freeze, leg feedforward, and drive feedforward are disabled.

### Flat-control qualification correction (2026-08-14)

The first formal seed-1 result at `0588158` is archived as
`INVALID_FLAT_CONTROL_STOP`, not as `Croll = 0`: 476/480 trials latched at
least one bilateral no-support sample. A 50 Hz-control / 5 ms-physics
root-cause audit found four independent contributors:

- the `0 mm` `pyramid_stairs` cell compiled to 21 adjacent stair boxes
  (plus four border boxes), only `2 micrometres` in half-thickness, rather
  than one flat support surface;
- the probe overwrote only the root state while leaving the legs at the default
  joint pose, which contradicted the registered posture card;
- the wheel contact used `solref=(0.005, 1)` at `dt=0.005`; MuJoCo documents
  that positive time constants should be at least `2 * timestep` and clamps
  smaller values when `refsafe` is enabled;
- MuJoCo Warp does not yet implement cylinder/capsule multicontact for general
  cylinder-box pairs ([mujoco_warp#1555](https://github.com/google-deepmind/mujoco_warp/issues/1555)); the maintainer confirms the resulting single support contact can alternate and oscillate. Native MuJoCo did not reproduce the finite-box failure in the matched local cross-check.

R0 therefore now uses one finite 1 m-thick flat box for the zero cell, resets
both leg joints and root orientation to the registered posture card, and pins
its wheel contact to `solref=(0.020, 1)` and
`solimp=(0.90, 0.95, 0.001)`. R1 imports exactly the same roll-first contact
constants and the same byte-verified final-C1 controller/calibration/posture
artifacts (without changing historical Stage0--5/campaign physics), and resets
its legs to the registered posture-map target together with the root. It does
**not** relax the safety rule: the probe latches every 5 ms physics substep
from the post-reset settle through recorded success, and any bilateral
zero-force substep still fails the trial. The earlier local CPU qualification
covered both cards, `16 env x 3 repeats`, and recorded zero bilateral
unsupported physics substeps, zero termination, and zero non-wheel contact
during its 10 s drive windows, but it predated the settle-through-success scope
correction and does not qualify the settle interval. A later targeted CPU fault
injection confirmed that a settle substep failure invalidates success and clears
the success time. Both are diagnostic/local checks only; a formal CUDA R0 under
the complete scope remains required before any Croll or PPO claim.

A cell passes at `44/48` successes only if termination, non-wheel contact, and bilateral airborne counts are all zero. The result is an interval:

```text
Croll,classical in [Hpass, Hfail)
```

If `Hpass = 0`, stair PPO is forbidden. If the next tier is unsafe, training is forbidden. A safe bracket enables R1. Passing through 10 mm extends in 2.5 mm increments up to the paper cap of 30 mm.

## R1: StairRollAssist

Task ID:

```text
HopperTrex-Hybrid-v2-StairRollAssist
```

Frozen interface:

- actor: original Stage5 34-D proprioceptive prefix;
- critic: actor prefix plus height, riser distance, and independent left/right contact forces (38-D);
- action mask: `(False, False, True, True, True, True)`;
- four leg residual limits: `0.035 rad`;
- fresh actor output head is zero-initialized, so the initial deterministic mean is numerically the zero-residual classical path;
- no contact-trigger mode, leg-reference freeze, authored leg trajectory, drive feedforward, jump, or landing FSM.

The first 64 of 256 slots are flat Stage5 retention and the remaining 192 are stair slots. The RollAssist stair reset is explicitly aligned to the same first-riser geometry as R0 (0.25 m outside the face), uses the registered envelope-center posture `(0.3092089487 m, 0.016 rad)`, and writes its absolute leg-map targets and root pitch together. The R1 training reset is deterministic on stair slots; the two-card, seed-controlled R0 reset protocol remains the formal evaluation contract. Stage5 reset disturbances remain only on flat-retention slots. Every stair episode commands zero forward velocity for exactly 100 control steps (2 s at 50 Hz), then `0.07 m/s`; the observed command and controller command are the same during settle. Because MjLab increments `episode_length_buf` before reward/termination evaluation and updates the command afterward, progress and stable-success gates remain off at buffer value 100 and start at 101, after the actor/controller has observed the drive command.

Updates 0--24 use Hpass. The update-25 decision occurs only after common step 600 (`25 x 24` complete environment steps) and uses cumulative **completed stair episodes**, not live episode state. It switches exactly once to Hnext only when cumulative success is at least 0.80 and cumulative termination, non-wheel-contact, and bilateral-airborne episode counts are all zero. The state and its counters are checkpointed and restored.

Reward calibration measures inherited positive reward rate `B` in the final safe 3 s of a zero-residual Hnext stall and freezes:

```text
progress_weight = 2B / 0.07
success_weight  = 2B
```

Invalid or unsafe final windows prohibit training. R0, R1 training/evaluation, and reward-stall calibration all inspect every unchanged 5 ms physics substep; any bilateral zero-force support sample is latched through the 50 Hz control step, terminates the R1 episode, invalidates success, and makes the stall unsafe. There is no grace period or control-interval OR rule. Any termination or non-wheel contact anywhere in the settle-plus-drive stall rollout also fails closed.
The environment and every checkpoint bind both the exact reward-calibration file bytes and its canonical JSON self-hash. R0 consumption also requires the R0 Git SHA to equal the current checkout; the wrapper, reward measurement/calibration, training preflight, runner, and evaluator all fail closed on provenance drift.

## Checkpoint selection, extension, and evaluation

- Initial budget: 100 updates, save interval 25.
- K=3: rejection-only screens over the exact latest three actual RSL-RL saves, then newest passer; never score-rank. For the initial 100-update block these counts are `51, 76, 100` (`model_50`, `model_75`, final `model_99`).
  A canonical K=3 validator replays the exact save grid and newest-passer rule, verifies strict update ordering and three distinct checkpoint hashes, and byte-verifies all screen envelopes before packaging.
- `src/hoppertrex_mjlab/scripts/evaluate_roll_assist.py` creates byte/hash-bound checkpoint envelopes and runs:
  - the existing Stage5 robust retention suite with the RollAssist legs-only mask;
  - Hpass candidate trials;
  - paired Hnext candidate and zero-residual trials with identical reset seeds; their progress vectors begin at drive onset, not during the zero-command settle;
  - safety, wheel-mask, success, paired progress bootstrap, and final two-card gates.
- An extension authorization binds the selected checkpoint bytes and continuation evidence to exactly one additional 100-update block. Because selection may choose an intermediate save, the total target is exactly `selected_completed_updates + 100`, capped at 500; it is not assumed to be a round hundred.
- A formal Hnext pass stops training immediately. If a selected passer fails the formal continuation gate, the protocol archives `ROLL_ASSIST_NO_EXPANSION` immediately because further training is forbidden; if continuation remains authorized it proceeds in 100-update blocks and may not package before the next block, up to the 500-update cap. If all K=3 screens reject, no formal checkpoint is selected: the run stops immediately and packages the ordered, byte-bound three screen envelopes as `ROLL_ASSIST_NO_EXPANSION`. Action rights, reward weights, and height are never changed. Passer packages include the selected checkpoint, R0 verdict, reward calibration, and formal evaluation; no-passer packages include R0, reward calibration, selection, and all three screen envelopes.

Formal expansion requires both Hnext posture cards at `44/48`, all safety gates, and exactly zero applied wheel residual:

```text
Croll,leg >= Croll,classical + 2.5 mm
```

Recovery improvement is a separate paired claim. This implementation deliberately emits `recovery_claim.eligible=false` because paired per-reset recovery-time bootstrap vectors are not yet collected; therefore the only allowed positive claim is boundary expansion.

## Interpretation of existing dynamic result

The archived StairDynamic result remains negative evidence only for its fixed 1 cm default dynamic feedforward path. It is not overwritten and is not used to infer the R0 boundary. This mainline remains single-seed, simulation-only, and provisional until separately extended.

## Non-evidentiary R0 diagnostics

`hoppertrex_mjlab.scripts.diagnose_roll_boundary` is a diagnostic-only entrypoint for the first positive tier. It never emits RollBoundary evidence and must not be used for PPO promotion.

- `--mode events` continues a trial after the first force-defined support loss and stores bounded 5 ms windows with wheel/contact, body, LQR, leg target, leg state, and leg actuator fields.
- `--mode posture-grid` scans the registered posture-map height/pitch envelope at 0 and 2.5 mm with matched reset perturbations.
- `--mode schedule-grid` scans twelve position-indexed, two-pose classical posture schedules plus two static regression sentinels. Wheels remain on the classical LQR path; PPO, leg/wheel residuals, stair FSM, contact trigger, lift, and drive feedforward remain disabled.
- Schedule candidates use the registered low/negative-pitch start poses, registered positive-pitch climb poses, and completion distances 30/15/0 mm before the riser. Progress is monotone per environment and applied posture is rate-limited by the qualified height/pitch slew rates.
- All modes require an output outside the Git checkout and label `evidence_eligible=false`; formal R0 remains unchanged. A schedule screen cannot authorize PPO or replace the frozen R0 artifact.
