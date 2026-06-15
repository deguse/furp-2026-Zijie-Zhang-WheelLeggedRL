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
