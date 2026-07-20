# HopperTrex Sim-to-Real Runbook

> Status: pre-hardware. Written against commits a8e5870 (tolerance probe),
> cdd2c1f (portable classical stack), 7cc2c91 (policy export), and this
> batch (deploy package). The tolerance-probe numbers section is a
> placeholder until the machine room runs `probe_hybrid_latency_noise`.

The ladder mirrors the simulation curriculum: classical first, residual
last, every rung gated by measurable acceptance criteria and an explicit
abort rule. Do not skip rungs. The PPO policy does not touch hardware
until R2 is stable.

## Rung overview

| Rung | Vehicle state | Goal | Payload |
|---|---|---|---|
| R0 | On stand, wheels free | Comms + safety proven | deploy.hal adapters, deploy.safety |
| R1 | On stand | Joint-level sanity + excitation data | deploy.loop (legs PD hold, wheel jogs) |
| R2 | On ground, spotter | Hardware LQR balance + calibrations | hybrid.classical_stack + re-identified artifacts |
| R3 | On ground | PPO residual on top of R2 | exported TorchScript + observation_builder |

## Hardware requirements (measured, tolerance probe 2026-07-19)

Source: `probe_hybrid_latency_noise` at 6e1f03c, 40 cells (delay 0-4
steps x 4 noise tiers x {zero residual, frozen Stage5 candidate
48a053f0}), 16 envs/cell, standing + 0.07 m/s tracking + 1x kick
recovery per cell. Full data:
`experiments\hybrid_latency_noise_6e1f03c\tolerance_seed1.json`.

Measured tolerance of the classical stack (zero residual):

- Zero falls at every noise tier through 60 ms loop delay (3 steps).
  Recovery degrades gracefully: 0.10 s (clean) -> 0.40 s at 60 ms.
- The cliff is at 80 ms (4 steps): recovery 0.92-3.58 s and the first
  falls appear (mems_imu 1, mems_imu_2x 2 terminated events).
- Consumer-MEMS IMU noise (pitch 0.005 rad, rate 0.02 rad/s) costs
  ~1.8x recovery at zero delay (0.18 s vs 0.10 s) with zero falls -
  acceptable. The 2x tier still survives to 60 ms but recovery
  triples; treat it as the noise ceiling.

Measured behavior of the PPO residual under latency (key deployment
finding): the candidate matches or slightly beats the classical stack
at 0-40 ms, but at 60-80 ms it is MORE fragile than zero residual
(falls: 2 at 60 ms clean, 6-22 at 80 ms across tiers; the classical
stack has at most 2). The residual was trained at zero latency and its
corrections become destabilizing when stale.

Derived requirements:

- End-to-end loop latency (sense -> compute -> actuate) MUST be
  <= 40 ms; design target 20 ms (one control period). Compute is
  negligible (~0.2 ms classical, small MLP for R3); the budget is
  spent on bus transactions and sensor latency, so measure the real
  loop with `ControlLoop` jitter statistics during R0.
- Any onboard computer that closes a CAN round-trip in a few ms is
  sufficient - including Raspberry Pi class. GPU not required.
- IMU: consumer MEMS class is acceptable if filtered pitch/rate arrive
  at 50 Hz; avoid anything noisier than the 2x tier.
- R3 gating rule (pre-registered): enable the residual only after the
  measured R0 loop latency is confirmed <= 40 ms; at higher measured
  latency the classical stack alone is the safer controller.

## Open hardware questions (answer before R0)

| Question | Affects | Notes |
|---|---|---|
| Onboard computer (Jetson / Pi / laptop tether)? | Loop latency budget, torch availability for R3 | classical stack needs only numpy (~0.2 ms/tick measured); TorchScript inference needs libtorch/CPU torch |
| Motor bus adapters (which USB-CAN, one bus or two)? | HAL adapter implementation, R0 schedule | Legs DM-J6248P, wheels RMD L-9025 35T, both CAN; vendor drivers/teammate code? |
| IMU model and mounting? | Pitch convention calibration, noise tier | R2 needs pitch + pitch rate at 50 Hz minimum |
| Session length / count this week? | How far up the ladder to plan | R0+R1 fit one short session; R2 needs identification time |

## R0 — Communications and safety (wheels off the ground)

Goal: prove we can talk to every motor and the IMU, and that the safety
chain kills torque.

1. Implement the `MotorBus`/`Imu` Protocols (deploy/hal.py) for the real
   bus. Keep the mock semantics: sorted-actuator joint order, SI units,
   monotonic timestamps from the same clock the loop uses.
2. Zero/sign calibration: move each joint by hand, confirm encoder signs
   match the simulation convention (wheel forward = +, leg signs per
   INIT_JOINT_POS mirror symmetry). Record the zero offsets — these
   become the HAL adapter's remap table, NOT edits to the controller.
3. IMU convention check: nose down = positive pitch (matches the sim's
   atan2(g_x, -g_z) construction). Fix in the adapter if inverted.
4. Safety drills, in order, ALL on the stand:
   - E-stop cuts motor power (hardware path, not software).
   - `SafetySupervisor.fault()` drops torque (software path).
   - Watchdog: pause the loop >40 ms -> FAULT latches, torque off.
   - Tilt: tilt the stand past 0.35 rad -> FAULT latches.

Accept when: all four drills pass twice in a row; joint states stream at
50 Hz with no gaps > 40 ms over a 60 s window.
Abort if: any motor fails to disable on either path — do not proceed to
torque-on work until fixed.

## R1 — Joint-level sanity (still on the stand)

Goal: closed-loop control of individual joints + excitation data for R2.

1. Legs: command the nominal standing posture via the supervisor
   (position targets from the posture map bias row). Verify steady hold,
   no oscillation, temperatures stable over 2 minutes.
2. Wheels: velocity jogs (±1, ±3, ±6 rad/s steps). Verify tracking and
   symmetric response; log with `ControlLoop` (log_path set).
3. Excitation session for identification: with the vehicle HELD on the
   stand in the balance posture, run wheel velocity chirps/steps while
   logging. `session_log_to_identification_arrays` converts the JSONL
   into the states/inputs/next_states contract of
   `identify_hybrid_controller.py`.

Accept when: leg hold error < 0.02 rad steady; wheel velocity tracking
error < 10% at 6 rad/s; a >= 60 s clean excitation log exists.
Abort if: any joint oscillates or overheats — tune the motor-side PD
(firmware) first; the artifact chain assumes stable joint servos.

## R2 — Classical stack on the ground (the first real "stands up")

Goal: hardware LQR balance + migrated calibrations. This is the same k.0
discipline as simulation: identify -> qualify -> only then close the loop.

1. Re-identify on hardware data: ground-truth balance data requires
   short assisted-balance sessions (spotter). Fit with
   `identify_hybrid_controller.py` (same Q/R as sim to start). The
   qualification bar is unchanged: rank 4, NRMSE <= 0.15, else the
   artifact is a labelled PD fallback and R2 stops for analysis.
2. vx estimator decision (pre-registered options):
   (a) wheel odometry (deploy.loop.WheelOdometryEstimator) — baseline,
   exact without slip; (b) odometry + IMU complementary filter if the
   probe/log shows slip corruption. Decide from R1 logs, record which.
3. Assemble `ClassicalStackArtifacts` from the hardware artifacts
   (`load_classical_stack_artifacts` refuses cross-bound mismatches).
4. First balance: spotter-assisted, envelope center posture, zero
   velocity command. Supervisor ACTIVE, tilt guard 0.35 rad.
5. Calibration transfers, in order, each a bounded session:
   velocity calibration sweep -> yaw feedforward probe -> (only if
   posture commands will be used) posture map + station calibration.
   Protocol identical to the simulation probes; artifacts carry the
   hardware git_sha and bind to the hardware controller hash.

Accept when: 60 s unassisted balance, |pitch| p95 < 0.05 rad, no
supervisor fault; velocity tracking at 0.05 m/s within the sim floor
x2 (0.012 m/s error budget).
Abort if: identification does not qualify (falls back to PD) — collect
more/better excitation data rather than hand-tuning gains.

## R3 — PPO residual (only after R2 is boring)

Goal: the trained Stage5 policy's residual on top of the hardware
classical stack, attribution-style.

1. Export the frozen candidate: `export_hybrid_policy
   --checkpoint-file <model_99> --stage 5 --output policy.ts`.
   The metadata JSON must match the DEPLOYED artifact hashes — the
   deployment runtime refuses a mismatched pairing. NOTE: the sim
   candidate was trained against the SIM controller hash; the honest
   R3 story is a re-trained or explicitly transferred policy, and the
   first hardware PPO session is an evaluation, not a demo.
2. Observation gap review (modeled, from observation_builder): the
   policy consumed privileged base_lin_vel (3 dims) in training;
   hardware supplies (vx_estimate, 0, 0). Lateral/vertical velocity
   are near zero in the qualification regime, but this is a listed
   sim-to-real gap — evaluate with the residual scaled down first.
3. Ramp protocol: residual scales at 25% -> 50% -> 100%, each a bounded
   session with the R2 acceptance metrics logged; any regression vs the
   pure classical baseline stops the ramp (mirror of the ablation
   attribution logic).

Accept when: with 100% residual, balance/tracking metrics are not worse
than R2, and the large-push recovery (hand pushes, spotter) is
subjectively no worse. Quantitative recovery comparison needs a
repeatable push rig — out of scope for the first sessions.

## Data discipline

Every session: JSONL logs via ControlLoop (they are identification-ready),
artifact JSONs with hashes, one line per session in experiment_log.md
(same recording standard as the simulation campaign). Machine-room rules
apply unchanged: hardware artifacts live outside git; only code and
documentation are committed.
