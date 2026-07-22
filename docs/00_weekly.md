# Weekly Progress Log

---

## Week Template

### Week N — YYYY-MM-DD

**Attended this week's meeting:** Yes / No

**Progress this week**
- _Summary of tasks completed_

**Challenges & blockers**
- _Summary of issues faced_

**Next steps**
- _Goals for the upcoming week_

**Hours spent:**

**Links:**

---

<!-- =================  YOUR ENTRIES BELOW  ================= -->

### Week 1 — 2026-06-15

**Attended this week's meeting:** Yes

**Progress this week**
- Initialized the research repository from the provided FURP template.
- Set up the `mjlab` simulation framework on my local Windows laptop. Since my computer only has an integrated Intel graphics card, I configured it in CPU-only mode and successfully built the `mujoco-warp` package.
- Set up a Python virtual environment in this repository using `uv` and linked the local `mjlab-main` framework as an editable dependency, which allows me to develop custom code under the `/src` folder.
- Cloned the senior's reference project repository to study the QP + RL whole-body control framework.
- Verified my local setup by running the environment list command and testing the Go1 robot flat terrain simulation with a random policy in the web viewer.

**Challenges & blockers**
- My local computer does not have an NVIDIA GPU (running on Intel Iris Xe integrated graphics), which prevents running Isaac Sim or Isaac Lab locally. I resolved this by using the CPU-compatible `mjlab` framework for local prototyping and debugging, planning to move to GPU servers for actual training.
- Addressed connection timeouts during package downloads by configuring the Tsinghua University registry mirror.

**Next steps**
- Obtain or build the URDF/XML description file of the target wheel-legged robot.
- Study the default velocity-tracking environment configurations in `mjlab` to design the observation, action, and reward terms for the wheel-legged balance task.

**Hours spent:** 10h

**Links:**
- [Senior's Reference Repository](https://github.com/ControlSystemLab-UNNC-UG/SEP-FURP-Mobile-Manipulator-2026)

---

### Week 2 — 2026-06-22

**Attended this week's meeting:** Yes

**Progress this week**
- Designed and implemented a hands-on RL practice project (continuous 2D maze car navigation) to build familiarity with OpenAI Gymnasium, Stable-Baselines3, and reward shaping.
- Iteratively designed and tested the environment through 4 stages:
  - **V1**: Static maze with Euclidean distance rewards and 5 Lidar rays.
  - **V2**: Dynamic procedurally generated mazes (DFS), 24-ray 360° Lidar, and strict sparse rewards. Discovered the "survival paradox" where the agent preferred to stall in place to delay crash penalties.
  - **V3**: Custom BFS path distance calculations, potential-based reward shaping, local waypoint coordinate observations, and maximum spawn separation. Created a premium dark-themed Pygame visualizer with glowing neon pathing.
  - **V4 Plus (Replicated)**: Replicated an advanced environment featuring soft-collision dynamics, observation/reward normalization (`VecNormalize`), frame stacking (`VecFrameStack(n_stack=3)`), and State Dependent Exploration (SDE).
- **Results**: The PPO model successfully achieved **100% evaluation success rate** on 5x5 mazes in under **200,000 steps** (~3.5 minutes on CPU). Zero-shot generalization tests showed a 50% success rate on completely unseen maze layouts.
- Launched curriculum learning (`curriculum_v4_plus.py`) to scale the training from simple 5x5 fixed mazes to complex 7x7 random mazes.
- Integrated the **HopperTrex two-wheeled legged robot simulation package** under `src/hoppertrex_mjlab`, which contains the Onshape-exported MuJoCo MJCF model, environment configs, and PPO training launchers using RSL-RL.
- Verified the integrated environment and RL pipeline by running a local CPU-only dry-run PPO training session for 2 iterations successfully.

**Challenges & blockers**
- **Headless GUI Hangs**: Running Pygame-based evaluation via background IDE tasks caused process freezing because Pygame requires an active interactive display session to initialize. Resolved by terminating the background process and instructing the user to run the visualization script directly in their interactive local terminal.
- **Actuator Exploration & Sparse Rewards**: The continuous action space combined with strict survival bounds caused exploration failure. Resolved by introducing potential-based reward shaping using topological path distances (BFS) and replacing terminal crashes with soft contact friction and local waypoint guides.

**Next steps**
- Complete the automated curriculum training for the 2D maze car and log final evaluation metrics.
- Synchronize the `hoppertrex_mjlab` package to the remote GPU server and launch the full-scale PPO training (4096 envs) for the 3D balance and velocity tracking task.
- Read and review the trained balance policy performance via `play.py` on the GPU server.

**Hours spent:** 12h

**Links:**
- [maze_car_env_v4_plus.py](file:///d:/mjlab_workspace/furp-2026-Zijie-Zhang-WheelLeggedRL/src/practice/maze_car/maze_car_env_v4_plus.py)
- [train_v4_plus.py](file:///d:/mjlab_workspace/furp-2026-Zijie-Zhang-WheelLeggedRL/src/practice/maze_car/train_v4_plus.py)
- [evaluate_v4_plus.py](file:///d:/mjlab_workspace/furp-2026-Zijie-Zhang-WheelLeggedRL/src/practice/maze_car/evaluate_v4_plus.py)
- [curriculum_v4_plus.py](file:///d:/mjlab_workspace/furp-2026-Zijie-Zhang-WheelLeggedRL/src/practice/maze_car/curriculum_v4_plus.py)

---

### Week 2 Update - 2026-06-26

**Attended this week's meeting:** No, extra technical progress update

**Progress this week**
- Completed the first reliable fixed-leg two-wheel balance stage for HopperTrex. The robot can now stand on the two main wheels without relying on thigh, calf, or chassis contact.
- Diagnosed the previous failure mode as a geometry and task-design problem, not simply a PPO or environment issue. The old fixed leg pose allowed non-wheel structures to become support points, so reward metrics could improve while the viewer still showed invalid low-posture support.
- Updated the clean balance task around a stricter definition of success: fixed legs, 1D coupled wheel action, clean wheel support reward, wheel contact checking, non-wheel contact penalty/termination, and viewer-based validation.
- Used `src/hoppertrex_mjlab/scripts/fixed_wheel_sweep.py` to check whether the main wheels are the lowest valid contact points and to identify non-wheel contact risks before training.
- Verified the clean balance behavior across multiple seeds. This changed the conclusion from "one lucky seed can balance" to "the clean task is now reproducible enough for the next stage."
- Improved the remote lab PC workflow with `setup_remote.ps1`, including repository synchronization, `uv` setup, smoke tests, GPU checks, and training command generation for new lab machines.
- Added the robust stationary balance task variant `Mjlab-HopperTrex-Balance-Robust-v0`. It keeps legs fixed and velocity command at zero, but adds small reset perturbations in roll/pitch, root x velocity, and roll/pitch angular velocity.
- Created a learning note document for the next study phase: `docs/rl_wheel_balance_learning_notes.md`.

**Challenges & blockers**
- The main technical blocker was non-wheel contact support. The robot could appear to survive, but it was not performing true two-wheel dynamic balance.
- Reward and TensorBoard metrics were sometimes misleading. Mean reward, alive reward, or flat orientation could look acceptable even when the viewer showed thigh/calf/chassis support.
- Several small parameter fixes did not solve the root issue, including pulse escape, higher teacher gains, stronger contact penalties, and hard contact termination. The real fix required changing the geometry assumption and the task structure.
- The Viser velocity command GUI had a zero-range slider issue when `lin_vel_y` was configured as `0.0 ~ 0.0`. This required a viewer-side patch in `mjlab-main`.
- New lab PC bootstrap had practical issues around missing local files, `uv` PATH setup, and deciding when to use Git synchronization instead of manually copying files.

**Next steps**
- Start robust stationary balance training by resuming from clean checkpoints instead of training from random initialization.
- Run robust multi-seed validation under small reset perturbations and check whether the policy actively recovers from tilt/velocity errors.
- If robust init succeeds, increase perturbations gradually to about `±5 deg` roll/pitch and larger root velocity/rate ranges.
- Do not add forward velocity commands, terrain, or leg control until stationary perturbation recovery is stable.

**Hours spent:** 18h

**Links:**
- `src/hoppertrex_mjlab/tasks/hoppertrex_balance_task.py`
- `src/hoppertrex_mjlab/scripts/fixed_wheel_sweep.py`
- `setup_remote.ps1`
- `docs/rl_wheel_balance_learning_notes.md`

---

### Week 3 - 2026-07-02

**Attended this week's meeting:** No

**Progress this week**
- Completed the fixed-leg robust two-wheel balance L2 stage. Multiple seeds passed with `+-5 deg` reset roll/pitch disturbance while maintaining clean support on the two main wheels.
- Completed Push Recovery L3. The policy can recover from periodic light push events while keeping thigh, calf, and chassis off the ground.
- Added and trained the SlowSpeed forward/backward movement task. The policy preserved clean two-wheel support, although velocity tracking quality still differs across seeds.
- Advanced the Turn L4 and SlowSpeedTurn tasks. Key issues diagnosed and addressed included incorrect yaw direction, a negatively biased yaw action head, yaw overshoot, and large wheel target jumps.
- Established the current best fixed-leg slow-turn task as `Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-SafeV2-YawScale2p5-Smooth-MidForward-Slew6-v0`.
- Identified the current best fixed-leg slow-turn checkpoint as `slow_speed_turn_sign_obs_scale_safe_v2_yawscale2p5_smooth_midforward_slew6_seed1/model_892.pt`. Model checkpoints remain archived locally and are not committed to GitHub.

**Challenges & blockers**
- TensorBoard and reward metrics were not sufficient by themselves. Viewer checks and custom diagnostic scripts were needed to verify whether the learned behavior was physically valid.
- Early Turn L4 runs showed several misleading local optima, including standing cleanly without turning, turning both command signs in the same direction, and a fixed negative yaw action bias.
- Fixed-leg wheel-only turning still has a structural limitation: the same wheels must handle yaw tracking, forward velocity, and pitch balance recovery, which can cause occasional backward recovery steps during turns.
- Further fixed-leg reward or action micro-tuning showed diminishing returns, so long training runs without a clear probe signal should be avoided.

**Next steps**
- Archive the current fixed-leg slow-turn best checkpoint as the wheel-only baseline.
- Do not continue `VarYawNoBack` or reward-only tuning as the main direction.
- Move to a limited leg assist or body lean assist stage, allowing small leg posture adjustments to reduce reliance on wheel-only pitch recovery during turns.
- Keep using short probe runs first: about 100 iterations, followed by `diagnose_turn_policy.py` and viewer validation before extending training.

**Hours spent:** 20h

**Links:**
- `docs/experiments/robust_balance_results.md`
- `src/hoppertrex_mjlab/tasks/hoppertrex_balance_task.py`
- `src/hoppertrex_mjlab/scripts/rsl_rl/diagnose_turn_policy.py`
- `D:\mjlab_workspace\handover.md`

---

### Week 4 - 2026-07-11

**Attended this week's meeting:** No

**Progress this week**
- Paused further legacy Stage2 training and conducted a focused design audit of the MDP, action space, rewards, curriculum structure, and promotion gates.
- Preserved the legacy Stage0-8 pure-PPO curriculum as a reproducible baseline and established a separate Hybrid v2 Stage0-5 route based on classical wheel-balance control plus residual PPO.
- Implemented an invariant six-dimensional residual action interface: wheel balance, wheel yaw, left/right thigh, and left/right knee. The robot has two physical legs with four actuated leg joints.
- Implemented local linear-model data collection, LQR/PD qualification, controller artifacts, two-leg posture sweeps, feasible-envelope filtering, and a local height/pitch-to-joint posture map.
- Added capability-driven Hybrid gates, Stage1 checkpoint bootstrap tools, training preflight checks, and an automated Windows machine-room workflow.
- Completed a 2,500-step identification run on the remote RTX 2080 SUPER. The resulting LQR satisfied rank-four controllability and the held-out one-step NRMSE qualification limit.
- Completed the Hybrid Stage0 controller gate for evaluation seeds 1, 2, and 3. All seeds had zero termination events and passed the pitch and pitch-rate safety thresholds. The repeated velocity results were approximately:
  - target -0.07 m/s: signed speed ratio 0.975;
  - target 0.00 m/s: forward drift +0.0135 m/s;
  - target +0.07 m/s: signed speed ratio 1.36.
- Concluded that Stage0 did not pass: the controller maintains stable upright balance, but its velocity channel has a deterministic positive bias.
- Implemented a separate velocity-calibration artifact, single-seed coarse/fine parameter sweep, Stage0Probe workflow, and controller/calibration/gate hash provenance checks.
- No Hybrid PPO training had started by the end of this reporting period.

**Challenges & blockers**
- The local laptop has no NVIDIA GPU, so local verification is limited to CPU unit and integration tests; real MuJoCo-Warp CUDA rollouts must run on the remote server.
- Initial remote evaluation exposed CPU/CUDA tensor-placement errors in termination masks. The shared fix now covers fixed linear velocity, fixed yaw, and integrated rollout paths.
- The first calibration attempt exposed a mismatch between the evaluator's serialized metrics envelope and the calibration parser. A subsequent non-GPU end-to-end audit also fixed artifact SHA provenance, resume-manifest validation, calibration-hash binding, and subprocess exit-code classification.
- The three Stage0 seeds showed that the main failure is neither random variation nor falling, but a systematic bias in the classical controller's velocity reference. Residual PPO should not be used to hide this lower-level error.
- Only one remote GPU server is currently available. Development sweeps and probe runs will therefore use seed 1 by default; three seeds are reserved for formal promotion and reported conclusions.

**Next steps**
- Complete the single-seed velocity-calibration coarse/fine sweep on the remote server.
- Run the full 3,000-step Stage0Probe with seed 1 using the selected calibration artifact.
- Run the formal Stage0 gate for seeds 1-3 and perform a Viser inspection only after the single-seed probe passes.
- Complete the formal two-leg posture sweep and posture map after Stage0 promotion.
- Launch a single-seed, 100-iteration Hybrid Stage1 residual-PPO probe. Stop on structural failure rather than extending training blindly.
- Retain the legacy Stage2 checkpoints as the pure-PPO historical baseline and avoid further unbounded reward micro-tuning.

**Hours spent:** 20h

**Links:**
- docs/hybrid_v2_remote_experiments.md
- src/hoppertrex_mjlab/hybrid/
- src/hoppertrex_mjlab/tasks/hoppertrex_hybrid_task.py
- src/hoppertrex_mjlab/scripts/calibrate_hybrid_velocity.py
- scripts/run_hybrid_v2_machine_room.ps1

---

### Week 5 - 2026-07-19

**Attended this week's meeting:** No

**Progress this week**
- Completed the Hybrid v2/v3 curriculum end to end, from Stage0 calibration to the Stage5 residual-PPO adjudication.
- Stage0/1: the velocity-calibration sweep fixed the deterministic velocity bias; Stage1-A/B residual PPO passed its formal retention gate. Both are frozen evidence.
- Stage2: root-caused the planar gate failure with a fixed-action transfer probe - the yaw channel is first-order and constant-gain trackable, so it was misallocated to PPO, and several gate thresholds had been authored below the measured noise floor. Corrected by folding a probe-fitted yaw feedforward into the classical baseline, recalibrating every threshold from measured floors, and pre-registering a single primary improvement metric (formal-only, minimum event count). The corrected formal run then honestly falsified residual value on the planar channels: 2.81% improvement versus the 10% pre-registered bar.
- Stage3: produced the first posture map, and the balance-qualification probe exposed a systematic station-keeping drift, affine in commanded pitch (v = -0.0136 - 0.751 p). Built a station-keeping feedforward from the measured law (worst drift 0.074 -> 0.0116 m/s, out-of-sample zero-crossing predicted -0.018, measured -0.0185). A transition probe showed step posture commands are violent (|vx| surge 10x the command domain), so posture references are now rate-limited; the pre-registered tier selection pinned the 2.0 s traverse (18x surge reduction).
- Kick-magnitude sweep: the classical stack never fell up to 8x the Stage1 impulse (0.32 m/s) at any posture - a strong robustness result on its own - with a degraded-but-recoverable band across 2x-8x. This fixed the pre-registered PPO battleground: center-posture recovery time at the 8x kick.
- Stage4: the yaw-transfer-across-posture probe measured at most 10.3% deviation versus the center posture (rule: 20%), so the global yaw calibration stands; the Viser check passed. With stages 3/4 classically closed, PPO consolidated into a single Stage5 campaign (Route A).
- Stage5 campaign: implemented the 2->3->4->5 migration chain with a codified no-harm-carrier policy, a large-kick recovery gate scenario (formal-only improvement check, >=128 kick events), an evaluation-time leg-residual ablation switch for attribution, and the full machine-room script. Result: the FIRST pre-registered PPO win of the project - recovery improved 12.42% over the zero-residual classical stack (0.887 s vs 1.013 s, 128/128 events, zero falls on both sides, 14x the measured run noise), and the leg ablation collapses the improvement to 3.82%, attributing about two thirds of the value to the leg residuals.
- Hardening: a bounded 500-iteration ceiling probe showed longer training is WORSE (8.0% improvement, a 22% small-kick regression, and the newest checkpoint failing the retention screen - the third falsification of newest-checkpoint selection), so the 100-iteration candidate is frozen. A kick-magnitude curve shows the improvement spans the whole 2x-8x band (+10% to +21%) with legs carrying 45-82% at every scale.
- Sim-to-real preparation (all hardware-independent, real robot access possible next week): a latency/noise tolerance probe (delay x sensor-noise sweep whose output is the hardware requirements sheet), a portable pure-numpy classical stack pinned to the simulation runtime at one float32 ULP (~0.2 ms/tick against the 20 ms budget), TorchScript policy export with a bit-exact observation builder, and a deployment package (HAL protocol + mocks, a latching safety state machine, a 50 Hz control loop whose session logs feed the re-identification tool directly) plus an R0-R3 hardware bring-up runbook.

**Challenges & blockers**
- Twice this week a gate failed because thresholds had been authored without measured noise floors (Stage2 planar, then the Stage5 robust screen where widened training pushes contaminated the calibrated tracking scenarios). Both fixes followed the same discipline: measure the floor first, then isolate channels - fine-grained tracking in a clean environment, robustness claims on controlled, measured disturbances only.
- Route A (skipping Stage3/4 training) left the Stage5 robust gate without its Stage4 reference envelope; resolved by measuring the reference from the zero-residual classical stack in-run.
- Checkpoint recency was falsified a third time (model_499 failed retention while model_400 passed), confirming the fixed K=3 screening rule.
- Operationally: Ctrl+C on the first Viser still kills the whole machine-room script, and a network drop mid-run forced manual continuation - both recoverable because gate JSONs are only written on completion, but the script defect stays on the backlog.
- The Viser viewer cannot show the Stage5 result (the deciding scenario is a 0.32 m/s kick that teleoperation never triggers, and the 0.126 s recovery delta is invisible by eye), which is exactly why the adjudication rests on the 128-event paired statistics; a push-button demo viewer was added for the video instead.

**Next steps**
- Answer the four open hardware questions (onboard computer, CAN adapters and existing drivers, IMU model, session length) and implement the real HAL adapters against the mocks.
- Real-hardware R0/R1 (communications, safety drills, joint-level sanity on a stand) once access is confirmed; R2 re-identification only after those pass.
- Optional by schedule: multi-seed reproduction of the Stage5 result, and the recovery-curve rerun on the frozen 100-iteration candidate.
- Start assembling the paper results skeleton from the experiment log (classical falsification chain + the pre-registered Stage5 win + attribution).

**Hours spent:** 25h

**Links:**
- D:\mjlab_workspace\experiment_log.md (sections 2.6-2.10, 3.5-3.14)
- docs/sim_to_real_runbook.md
- src/hoppertrex_mjlab/hybrid/classical_stack.py
- src/hoppertrex_mjlab/deploy/
- scripts/run_hybrid_stage5_seed1.ps1
- src/hoppertrex_mjlab/scripts/probe_hybrid_latency_noise.py
