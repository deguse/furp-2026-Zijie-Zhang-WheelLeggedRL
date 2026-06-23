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

