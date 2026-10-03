# HopperTrex roll-first stair mainline

Status: implemented protocol; no formal RollBoundary or RollAssist result is claimed by this document.

## Evidence and scope

The control allocation is deliberately narrower than the earlier StairCamp/StairDynamic route:

1. R0 measures direct wheel roll-over under the frozen final C1 classical stack.
2. R1, only after a positive safe R0 bracket, grants PPO the four leg residuals while the classical controller permanently owns both wheel channels.
3. Contact-triggered lifting, synchronized/alternating feedforward, a jumping/landing FSM, online height classification, multi-seed claims, and real-hardware claims remain Future Work.

The literature supports the existence of distinct nearby modes, not a universal height threshold:

- [Li et al., Balancing Control and Pose Optimization](https://arxiv.org/abs/2109.09934): the optimized critical poses are combined with a QP force-balance layer whose leg command is `tau_QP + Kp(q* - q) + Kd(qdot* - qdot)`; the paper is not evidence that pose references alone are sufficient.
- [Bjelonic et al., Rolling in the Deep](https://arxiv.org/abs/1909.07193): ZMP-aware online trajectory optimization and hierarchical whole-body contact-force control provide support-feasibility context; the paper does not prove that HopperTrex requires the same full architecture.
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
- All modes reserve an output outside both the project and MjLab Git checkouts before simulation, capture Git/worktree/source-hash provenance before execution, and reject provenance drift before the atomic write. Dirty CPU diagnostics may be run only with `--allow-dirty` and record the dirty fingerprints; non-CPU schedule screens cannot use that override.
- Schedule-grid output is fail-closed: all fourteen candidates must have complete, count-checked, uniform trial schemas and every policy/residual/feedforward authority metric must be finite and exactly zero.
- Every output labels `evidence_eligible=false`; formal R0 remains unchanged. A schedule screen cannot authorize PPO or replace the frozen R0 artifact.

### Schedule-grid result and R0c-SYNC decision (2026-08-15)

The clean CUDA schedule screen at project SHA
`0e1d39f23782888f0492d9c728827dd5473d602e` and MjLab SHA
`43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6` is diagnostic-only. Its
224 trials retained all five exact-zero authority checks. All 112 flat trials
were safe successes, while the twelve dynamic schedules at 2.5 mm produced
12 safe successes, 83 unsafe trials, and one safe stall. The best diagnostic
candidate, A-to-D ending 30 mm before the riser, was only 3/8 safe with five
unsafe trials. This rejects completion-distance and posture-grid refinement; it
does not update the formal `[0, 2.5 mm)` R0 bracket or authorize PPO.

The event audit also narrows what is and is not established:

- A-to-D pitch reaches its endpoint in about 42 control ticks while height needs
  about 152 ticks under the independent qualified slew limits. Two of the five
  first support losses occurred before height completion and three after it.
  This unisolated synchronization variable permits exactly one bounded
  ablation, not another parameter grid.
- Persistent wheel-torque saturation is actuator-headroom evidence, not a
  demonstrated root cause: flat and safe trials have equally high or higher
  saturation fractions.
- Four of five first-loss events have bilateral positive clearance and four of
  five have positive root vertical velocity. The supported conclusion is a
  reset-sensitive whole-support collapse before the riser, not a universal
  claim that every event is pure ballistic flight.

`--mode r0c-sync` implements the preregistered rejection-only ablation. It runs
exactly two controller modes over flat and 2.5 mm terrain with eight matched
resets per cell and one repeat (32 trials total):

1. `r0c_sync_c0_independent_sa_cd_d030mm` preserves the archived A-to-D,
   30 mm schedule with independent height/pitch slew.
2. `r0c_sync_c1_synchronized_sa_cd_d030mm` uses the same nominal schedule but
   rate-limits one shared applied alpha by the slower qualified channel. It
   changes no wheel LQR, posture map, actuator, residual, stair FSM, contact
   trigger, leg feedforward, or drive feedforward authority.

Each R0c-SYNC trial records a 50 Hz `control_trace` containing nominal/applied
alpha, separate height/pitch applied alpha, applied posture, body vertical/pitch
state, and left/right/total vertical normal load; summary rows also retain load
means and the minimum control-step total. First-support-loss windows add the same
load and alpha fields at the 5 ms physics cadence. Event samples distinguish
one-based `episode_control_step` from one-based `drive_control_step`, and every
raw support-loss trial must bind to exactly one complete 8+1+12 sample window.
Vertical load is computed only for found contact slots as
`abs(contact_frame_normal_force * global_normal_z)` and is diagnostic; it has no
control authority in R0c-SYNC. Artifact construction also cross-checks the raw
unsupported-substep count against success/safety booleans and compares every
C0/C1 root, velocity, orientation, and joint reset field exactly before claiming
matched perturbations.

Run only from a clean committed checkout, with output outside both Git trees:

```powershell
uv run python src/hoppertrex_mjlab/scripts/diagnose_roll_boundary.py `
  --mode r0c-sync `
  --device cuda:0 `
  --output D:\mjlab_workspace\r0c_sync_COMMIT_SHA\r0c_sync_screen.json
```

The CLI pins `device=cuda:0` and freezes `envs_per_height=8`, `repeats=1`, `settle_steps=100`,
`drive_steps=500`, and `stable_steps=25`; overrides are rejected. CUDA cannot
use `--allow-dirty`.

The output is invalid if either flat arm is not 8/8 safe or if the C0 2.5 mm
arm does not reproduce the registered 3-success/5-unsafe split within the frozen
one-trial aggregate tolerance (2--4 successes, no safe stalls). C1 passes this
screen only with zero unsafe trials and at least 7/8 safe successes. A pass
still requires a new clean-SHA formal 16-environment x 3-repeat replication. Any
C1 unsafe trial, or an all-safe result with fewer than seven successes, rejects
synchronized pose-only control
without a rate/threshold sweep and advances the next development step to the
predictive load/ZMP-constrained classical reference governor (R0c-LRG).


### R0c native same-substep contact replay and backend attribution (2026-08-15)

The synchronized-pose ablation reproduced the reviewed exact-reset result:
flat remained strict/geometric 8/8, while 2.5 mm remained strict 3/8 but
geometric 8/8.  The five unsafe resets contained fourteen isolated bilateral
zero-force samples in total; the maximum consecutive run was one 5 ms physics
substep.  This left two competing explanations: a real whole-support flight
requiring additional wheel/leg authority, or a contact-backend-specific sample.

`probe_r0c_native_contact_replay.py` now performs a rejection-only,
development diagnostic over every such sample.  For each MJWarp bilateral-zero
substep it captures the state at the beginning of the transition (`qpos`,
`qvel`, actuator state, warm start, applied forces, control, and time), then
loads that state into native MuJoCo using the same host `MjModel` and advances
exactly one 5 ms step.  The comparison is phase explicit:

- `during_step_contact` is read after `mj_step` and before another
  `mj_forward`.  It describes the constraint solve that generated the step and
  matches the lagged contact-sensor timing documented by MjLab's decimation
  loop.
- `integrated_endpoint_contact` is read only after forwarding the integrated
  endpoint and is retained as a secondary persistence/geometric check.
- wheel forces are restricted to wheel-collision-geom versus terrain-body
  pairs, exactly matching the RollBoundary contact sensors.
- twenty-three collision, inertial, joint, and actuator model arrays are
  checked before replay.  All sixteen MJWarp worlds were exactly identical;
  world 0 and the native host model differed by at most
  `1.9073486345888568e-07`, consistent with float32 bridge roundoff.

Development artifact:

```text
r0c_native_contact_replay_retry2_dev.json
SHA256 8aa9522c4ac15e35a91c107d08bc4eb12b0fad56c674085bc207ddde20366143
source r0c_sync_screen.json
source SHA256 dd8826dcb1abd77ddeb1c68d2bef1406d74b045365d49915d672379e74267c14
```

The result is unambiguous for the captured states:

| quantity | result |
|---|---:|
| MJWarp strict bilateral-zero samples captured | 14/14 |
| native same-solve-phase bilateral-zero samples | 0/14 |
| native integrated-endpoint bilateral-zero samples | 0/14 |
| minimum native same-phase single-wheel force | 7.746 N |
| minimum native same-phase bilateral force sum | 103.619 N |
| strict verdict modified | no |
| controller/reset modified | no |

Thus none of the fourteen strict failures is a near-threshold native
whole-support loss at the same state.  This directly supports attribution to
the current MJWarp cylinder-box contact path, consistent with
[mujoco_warp#1555](https://github.com/google-deepmind/mujoco_warp/issues/1555).
It does **not** by itself claim that a full native trajectory is 8/8: native
and MJWarp velocities diverge after the replayed solve, so a formal native
rollout must be separately preregistered if that backend becomes the R0
contract.

The exact-reset control ablations are therefore closed rather than tuned:

- a wheel rated-torque hard guard stalled before the riser (2.5 mm strict and
  geometric 0/8);
- integrating leg-position admittance degraded to strict 1/8 and geometric
  4/8;
- a non-integrating correction below 0.5 mrad still degraded to strict 0/8 and
  geometric 2/8;
- replacing the actuator with MjLab `BuiltinPdActuatorCfg` did not preserve the
  plant and terminated all trials.

No position-admittance gain sweep is authorized.  The verified per-side
support-transfer state machine remains shadow-only and has no wheel or leg
action authority.  The formal MJWarp R0 verdict remains unchanged at 3/8 for
2.5 mm.  Resolving the backend-specific verdict now requires an explicit
physics-contract decision: either preregister a native-MuJoCo RollBoundary
replication, or first validate an MJWarp-supported wheel collision proxy and
then treat that geometry as a new physical contract.  Neither change may be
silently promoted by this development artifact.


### Native full-rollout qualification result (2026-08-15)

After the same-state replay attributed all fourteen reviewed MJWarp zero-force
samples to the cylinder-box contact path, a separate development probe tested
whether native MuJoCo could replace MJWarp as the complete R0 dynamics backend.
`probe_r0c_native_full_rollout.py` keeps the existing MjLab command and
`HybridWheelLegAction` implementations as the closed-loop controller, mirrors
native state into them once per 20 ms control step, and applies their exact
`ctrl` output to four native 5 ms steps. It never calls `env.step` or
`mjwarp.step`; contact and safety verdicts come only from native MuJoCo.

The probe uses all sixteen reviewed C0 exact resets and the unchanged
100-settle/500-drive/25-stable protocol. The controller asserts exact-zero
residual/dynamic authority and the original pure-classical wheel slew path on
every step. The model-equivalence check remains identical to the one-step
replay.

Development artifact:

```text
r0c_native_full_rollout_retry1_dev.json
SHA256 fca4bef7803c21576bb8057a0d5143019f0fc9a0523681e85668173cc4917b31
source r0c_sync_screen.json
source SHA256 dd8826dcb1abd77ddeb1c68d2bef1406d74b045365d49915d672379e74267c14
```

The full native trajectory does not qualify as the formal replacement:

| cell | strict | geometric | native zero-force substeps | max consecutive |
|---|---:|---:|---:|---:|
| flat | 0/8 | 8/8 | 52 | 1 |
| 2.5 mm | 0/8 | 7/8 | 56 | 2 |

There were no non-wheel contacts or terminations. Unlike the original MJWarp
failure states, these new native events are real geometric micro-flight caused
by the backend-diverged closed-loop trajectory. For example, flat env 2 at
`t=0.750 s` had no native contact pair; at the solve state its wheel centers
were `0.1001779 m` and `0.1000764 m` high for a `0.1000000 m` wheel radius.
The root was descending from `vz=-0.02547 m/s` and reached
`vz=-0.07409 m/s` at the integrated endpoint before contact returned on the
next substep. The observed sequence is left support, one unsupported sample,
then right support: an actual lateral load-transfer/rocking event, not a force
threshold artifact.

Consequently the project must not silently switch full R0 dynamics to native
MuJoCo, and a hybrid contract in which MJWarp supplies dynamics while native
only overrides its force verdict is also rejected as physically inconsistent.
The current controller/calibration stack is backend-specific: making native
the full contract would require re-identifying the actuator/LQR/posture stack,
not merely changing the evaluator.

The installed MJWarp collision table gives a narrower next hypothesis than a
parameter sweep. Cylinder-box uses the generic convex path implicated by
`mujoco_warp#1555`; sphere-box is a one-contact primitive and capsule-box is a
two-contact primitive, but neither can preserve both the wheel's 0.1 m radius
and its much smaller axial half-width. Ellipsoid-box is supported by the convex
path and can preserve all three wheel radii exactly, with a physically
point-like rigid contact instead of pretending to recover cylinder
multicontact. A single ellipsoid-collision development ablation may therefore
be evaluated against both backends, but it is a new geometry contract and is
not authorized by either native artifact.


### Axis-preserving ellipsoid collision rejection (2026-08-15)

The one authorized collision-representation ablation replaced only
`robot/wheel_{left,right}_collision` from cylinder size
`[0.100, 0.018, 0]` to ellipsoid radii `[0.100, 0.100, 0.018]` in geom-local
coordinates. A strict compiled-model diff proved that all model options,
visuals, rigid-body mass/inertia, joints, actuators, friction,
`solref`/`solimp`, and sensors were byte-for-byte or array-exactly unchanged.
No shape parameter was exposed.

Development artifact:

```text
r0c_ellipsoid_collision_dual_backend_dev.json
SHA256 9e55a5885d4d066f1a3aaaabfd4abff0d4b4c1602050b3f4a35390faec9fd00e
source r0c_sync_screen.json
source SHA256 dd8826dcb1abd77ddeb1c68d2bef1406d74b045365d49915d672379e74267c14
```

| backend/cell | strict | geometric | zero-force substeps | disposition |
|---|---:|---:|---:|---|
| MJWarp flat | 8/8 | 8/8 | 0 | retention only |
| MJWarp 2.5 mm | 3/8 | 7/8 | 9 | reject |
| native flat | 8/8 | 8/8 | 0 | improved |
| native 2.5 mm | 0/8 | 0/8 | 0 | eight safe stalls |

The native step trials stopped `12.8--15.7 mm` before the riser with no
support loss, non-wheel contact, or termination. Thus the ellipsoid's
physically point-like contact removes the native flat lateral rocking, but it
also removes the narrow cylinder's edge interaction needed by the current
controller to climb the sharp 2.5 mm box. MJWarp retained its 3/8 strict result,
changed which resets passed, reduced but did not eliminate zero-force samples,
and lost one geometric completion. Trial classifications do not match across
backends.

The candidate fails every preregistered promotion condition and is closed with
no ellipsoid-size sweep. It must not enter the formal robot or RollBoundary
configuration. Together with the cylinder results, this establishes that a
collision-only substitution cannot simultaneously preserve traversal and the
5 ms support invariant for the current position/velocity actuator controller.
The next technically coherent direction is an actuation/control architecture
change that explicitly owns bilateral load and roll damping; it is not another
shape, position-offset, or verdict-threshold ablation.

### HAL v2 and explicit bilateral-load effort sanity result (2026-08-15)

The collision-only path having been closed, the next single causal candidate
made left/right support load an explicit control variable rather than another
leg-position offset.  Before changing the development actuator, the actual
motor protocols were checked:

- DaMiao MIT mode accepts `p_des`, `v_des`, `kp`, `kd`, and `t_ff`.  Its
  feedback frame carries mapped position, velocity, torque estimate, status,
  and MOS/rotor temperatures.  The 6248P mapping defaults are
  `PMAX=12.566 rad`, `VMAX=20 rad/s`, and `TMAX=120 N m`; those are configurable
  encoding ranges and are not the physical `30/97 N m` rated/peak envelope.
  Sources: [DaMiao communication protocol](https://damiao-motor.jia-xie.com/concept/communication-protocol),
  [DM-J6248P manual mirror](https://wiki.aifitlab.com/damiao-docs/dm-j6248p-2ec-motor-instruction-manual),
  and `python-damiao-driver` commit
  `3c6773aa303b5eb9d2613b83fc0b8a5274b79b94`.
- The official MyActuator Motor Motion Protocol V4.2 defines RMD command
  `0xA1` as signed int16 iq current at `0.01 A/LSB`; its reply contains measured
  iq, output speed, output angle, and temperature.  The L-9025 product manual
  gives `3.46 A / 2.79 N m` rated, `7.6 A / 5.8 N m` instantaneous, and a
  nominal `0.76 N m/A` torque constant.  The new HAL therefore keeps raw iq
  separate from a calibrated torque estimate and refuses effort mode when the
  torque mapping is not calibrated.

The additive HAL v2 retains the old position/velocity `MotorBus` unchanged and
adds optional synchronized effort telemetry, capability negotiation, a
separate effort command, and fail-closed safety forwarding.  Missing
capability/calibration, missing or stale effort telemetry, or non-finite data
latches `FAULT`; there is no silent fallback to the legacy command.  The
independent-review fixes additionally reject non-finite control/sensor times,
non-finite IMU or joint state, timestamps more than the bounded 40 ms future
skew, control-clock rollback, and rollback of previously accepted IMU, joint,
or effort timestamps.  Pitch and roll now have independent `0.35 rad` tilt
limits.  Effort feedback trips immediately above `97 N m` leg torque, `7.6 A`
wheel iq, or `5.8 N m` calibrated wheel torque.
Pure DM MIT and RMD `0xA1` codecs have golden-frame tests, but no USB-CAN
transport is claimed.

The development controller changed only the compiled actuator law to direct
`<motor>` effort.  An exact array audit verified unchanged geometry, contact,
inertia, joints, armature/friction, sensors, solver options, and timestep.  The
only changed compiled fields were:

```text
actuator_biastype
actuator_ctrllimited
actuator_gainprm
actuator_biasprm
actuator_ctrlrange
```

The 200 Hz controller used no contact force as an input.  At each causal 5 ms
sample it mirrored current state into the same native model, computed the mass
matrix, bias, and vertical Jacobians at the two wheel axle centres, allocated
at least 5 N per side with total target `164.5048209 N`, generated a critically
damped roll moment, mapped it to inverse-static joint effort, added
mass-derived DM-compatible leg impedance, and converted the unchanged outer
wheel velocity target through the preregistered
`clamp(200 * velocity_error, +/-5.8 N m)` rule.

The required two-reset sanity ran source env 0 flat and source env 8 at 2.5 mm
on MJWarp and native in sequence.  The earlier artifacts were superseded after
each review because safety and provenance code changed after they were written.
The same reset pair was therefore rerun after the final review fixes; because
that sanity still failed, the preregistered sixteen-reset run was not started.
The CLI now also rejects full mode unless it receives a SHA-bound, passing
schema-v2 sanity artifact whose source, Git HEAD, and causal manifests all match.

```text
r0c_effort_wbc_sanity_finalreview_dev.json
SHA256 ee5bbcb5f9e301d09763602d2fbcafbd56fae0caf99047a53aec6b9caad2838b
source r0c_sync_screen.json
source SHA256 dd8826dcb1abd77ddeb1c68d2bef1406d74b045365d49915d672379e74267c14
Git HEAD d1d12e4ab94bf2785c18c98bd033615083944103
```

The schema-v2 artifact records identical pre/post sixteen-file causal SHA256
manifests and `causal_provenance_gate_pass=true`; any mid-run drift now aborts
before artifact writing.  Every manifest entry also matched the working tree
immediately after the run.  Effort
summary metrics are explicitly a fixed `600 x 4` controller-sample window
(settle plus drive), include samples after a trial's classification becomes
inactive, and must not be interpreted as an active-window comparison.

| backend/cell | strict | geometric | bilateral-zero | single-wheel unload L/R | disposition |
|---|---:|---:|---:|---:|---|
| MJWarp flat | 1/1 | 1/1 | 0 | 0/0 control samples | sanity pass |
| MJWarp 2.5 mm | 0/1 | 0/1 | 0 | 3/1 control samples | safe stall |
| native flat | 1/1 | 1/1 | 0 | 0/0 physics substeps | sanity pass |
| native 2.5 mm | 0/1 | 0/1 | 0 | 6/9 physics substeps | safe stall |

Both backends had zero non-wheel contacts and zero terminations and produced the
same per-trial classification.  Across these four individual sanity cells no
sample had both wheels simultaneously at zero force.  This narrow observation
does **not** establish that left/right load realization is consistent or that
the 8-reset support problem is solved.  The native 2.5 mm trial still reached
`0 N` on each wheel separately, with 6 left-unloaded and 9 right-unloaded 5 ms
substeps; its minimum combined wheel force was `70.3907 N`.  MJWarp likewise
reported 3 left-unload and 1 right-unload control samples.  Although the QP
commands stayed near `82 N` per side and maximum computed leg effort was only
`8.234 N m`, target feasibility is not proof that contact geometry realized
those targets.  This candidate also changed actuator realization and leg
impedance, so the no-bilateral-zero observation cannot be uniquely attributed
to the allocator.

The combined candidate nevertheless fails geometric traversal.  Both 2.5 mm
trajectories stopped about `25.6--26.0 mm` before the riser and repeatedly hit
the `5.8 N m` wheel peak.  Adjacent 5 ms wheel commands reversed sign on about
`91--93%` of the fixed-window stair samples.  This strongly supports, and is
consistent with, a 5 ms bang-bang/phase mismatch from directly reusing the
legacy implicit velocity gain; it does not isolate that mechanism as the
unique root cause.  Flat trials also had high reversal rates while passing,
and no wheel-inner-loop-only counterfactual has yet been run.  Consequently
`wheel_peak_torque_saturation` remains a diagnostic first-failure category,
not a causal proof.

This is a rejection of the combined fixed candidate, not a validation or
rejection of explicit load allocation in isolation.  Per preregistration there
was no gain sweep, no second wheel law, no full sixteen-reset run, and no formal
actuator promotion.  A future experiment would have to preregister a physically
matched wheel torque/velocity inner loop while holding the allocator and leg
controller fixed solely to isolate the wheel-loop hypothesis.  The review-fix
verification completed with `74` targeted tests passing, full-repository
`1247 passed, 1 skipped, 384 subtests passed`, and Ruff clean on the touched
Python files.
