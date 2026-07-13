# Reinforcement Learning Control for a Wheel-Leg Robot in Isaac Lab

Undergraduate Research Practice · Faculty of Science and Engineering · University of Nottingham Ningbo China

---

## Project Info

| Field | Your entry |
|---|---|
| Student name(s) | Zijie Zhang |
| Project title | Reinforcement Learning Control for a Wheel-Leg Robot in Isaac Lab |
| Project tag | WheelLeggedRL |
| Track | Research |
| Supervising faculty | _..._ |
| Project lead | _..._ |
| Team or individual | Individual |
| Cited paper being replicated | mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning (https://arxiv.org/abs/2502.13963) |

**One-line summary:** This project studies wheel-legged robot control in MjLab using a qualified identified LQR, or an explicitly labelled PD fallback, as the wheel-balance baseline with a fixed six-dimensional residual PPO policy for wheel and two-leg joint control.

The current Hybrid v2 baseline uses classical state feedback, residual PPO,
actuator limits, and capability-driven gates. QP/WBC is future work and is not
implemented in the current mainline. Early Stage1 GPU probes were run under the
superseded low-speed objective; they are retained as historical artifacts but
are not promotion evidence. The current route restarts Stage1 from a fresh
zero-residual bootstrap and has not authorized Stage2-5 or a large formal run.

---

## Repository Structure

```text
/docs
 ├── 00_weekly.md         ← Weekly progress log
 └── meeting_notes/       ← Key takeaways from team meetings
/src                      ← Source code, experiments, and materials
FURP_Showcase.pdf         ← Final poster / presentation PDF
```

- **`docs/00_weekly.md`** — Weekly progress log and research journal.
- **`docs/meeting_notes/`** — Meeting notes capturing decisions and action items.
- **`src/`** — Source code, simulation scripts, and experiment materials.
- **`FURP_Showcase.pdf`** — Final presentation poster.
