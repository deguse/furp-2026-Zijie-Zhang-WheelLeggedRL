#!/usr/bin/env python3
"""Deterministic fixed-wheel sweep diagnostic for HopperTrex.

Bypasses PPO, teacher, and the full mjlab env. Builds a minimal MuJoCo sim
with the same actuator parameters used in training (position actuators for
fixed legs, velocity actuators for wheels), then sweeps through fixed wheel
velocity commands and records tilt / pitch rate / root height / contact /
actual wheel velocities per step.

Usage:
    uv run python src/hoppertrex_mjlab/scripts/fixed_wheel_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

PROJECT_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_PATH))

from assets.HopperTrex_CFG import (
    ALL_JOINT_NAMES,
    INIT_JOINT_POS,
    LEG_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    HOPPERTREX_XML,
    HOPPERTREX_INIT_STATE,
)

NON_WHEEL_GROUND_GEOMS = (
    "thigh_left_collision",
    "thigh_right_collision",
    "calf_left_collision",
    "calf_right_collision",
    "chassis_base_collision",
)

WHEEL_GROUND_GEOMS = (
    "wheel_left_collision",
    "wheel_right_collision",
)

# Sweep cases: (case_name, left_wheel_target, right_wheel_target)  [rad/s]
SWEEP_CASES = [
    ("zero",     0.0,  0.0),
    ("L+2_R-2",  2.0, -2.0),
    ("L-2_R+2", -2.0,  2.0),
    ("L+4_R-4",  4.0, -4.0),
    ("L-4_R+4", -4.0,  4.0),
    ("L+6_R-6",  6.0, -6.0),
    ("L-6_R+6", -6.0,  6.0),
    ("L+8_R-8",  8.0, -8.0),
    ("L-8_R+8", -8.0,  8.0),
]

EFFORT_LIMITS = [5.8, 30.0, 50.0]
SWEEP_STEPS = 150


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=SWEEP_STEPS)
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV output path; prints to stdout if omitted.")
    return parser.parse_args()


def build_model(effort_limit: float) -> mujoco.MjModel:
    """Build MjModel with leg position + wheel velocity actuators."""
    spec = mujoco.MjSpec.from_file(str(HOPPERTREX_XML))

    ground = spec.worldbody.add_geom()
    ground.name = "ground"
    ground.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground.size[:] = np.array([5.0, 5.0, 0.1])
    ground.friction[:] = np.array([1.0, 0.005, 0.0001])

    # --- leg position actuators (fixed pose, same as training) ---
    for name in LEG_JOINT_NAMES:
        act = spec.add_actuator(name=name, target=name)
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.dyntype = mujoco.mjtDyn.mjDYN_NONE
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = 8000.0          # stiffness
        act.biasprm[1] = -8000.0          # -stiffness * qpos
        act.biasprm[2] = -80.0            # -damping * qvel
        act.forcelimited = True
        act.forcerange[:] = np.array([-97.0, 97.0])
        act.ctrllimited = False
        act.inheritrange = 0.0

    # --- wheel velocity actuators (same structure as training) ---
    for name in WHEEL_JOINT_NAMES:
        act = spec.add_actuator(name=name, target=name)
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.dyntype = mujoco.mjtDyn.mjDYN_NONE
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = 200.0           # damping gain
        act.biasprm[2] = -200.0           # -damping * qvel
        act.forcelimited = True
        act.forcerange[:] = np.array([-effort_limit, effort_limit])

    return spec.compile()


def get_joint_indices(model: mujoco.MjModel) -> dict[str, int]:
    """Map joint names to qpos/qvel indices."""
    out = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        addr = model.jnt_qposadr[i]
        dof = model.jnt_dofadr[i]
        out[name] = (addr, dof)
    return out


def get_actuator_indices(model: mujoco.MjModel) -> dict[str, int]:
    """Map actuator names to actuator indices."""
    out = {}
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        out[name] = i
    return out


@dataclass
class SweepRow:
    case: str
    effort: float
    step: int
    tilt_x: float
    pitch_rate_y: float
    lin_vel_x: float
    root_z: float
    wheel_contact: int
    non_wheel_contact: int
    left_cmd: float
    right_cmd: float
    actual_left_vel: float
    actual_right_vel: float
    terminated: bool = False

    def as_tuple(self) -> tuple:
        return (
            self.case, self.effort, self.step,
            self.tilt_x, self.pitch_rate_y, self.lin_vel_x,
            self.root_z, self.wheel_contact, self.non_wheel_contact,
            self.left_cmd, self.right_cmd,
            self.actual_left_vel, self.actual_right_vel,
            int(self.terminated),
        )


def run_sweep(args: argparse.Namespace) -> list[SweepRow]:
    rows: list[SweepRow] = []

    for effort_limit in EFFORT_LIMITS:
        model = build_model(effort_limit)
        data = mujoco.MjData(model)
        ji = get_joint_indices(model)
        ai = get_actuator_indices(model)
        ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        non_wheel_geom_ids = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in NON_WHEEL_GROUND_GEOMS
        }
        non_wheel_geom_ids.discard(-1)
        wheel_geom_ids = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in WHEEL_GROUND_GEOMS
        }
        wheel_geom_ids.discard(-1)

        for case_name, left_cmd, right_cmd in SWEEP_CASES:
            # Reset
            mujoco.mj_resetData(model, data)
            # Set init pose
            init_pos = HOPPERTREX_INIT_STATE.pos
            floating_qposadr = model.jnt_qposadr[0]  # freejoint is joint 0
            data.qpos[floating_qposadr:floating_qposadr + 3] = init_pos
            for jname, angle in INIT_JOINT_POS.items():
                addr, _ = ji[jname]
                data.qpos[addr] = angle

            # Leg targets = init pose
            leg_ctrl = {name: INIT_JOINT_POS[name] for name in LEG_JOINT_NAMES}
            # Wheel targets = fixed command
            wheel_ctrl = {"wheel_left": left_cmd, "wheel_right": right_cmd}

            terminated = False
            for step in range(args.steps):
                if not terminated:
                    # Set ctrl
                    for name in ALL_JOINT_NAMES:
                        if name in LEG_JOINT_NAMES:
                            data.ctrl[ai[name]] = leg_ctrl[name]
                        else:
                            data.ctrl[ai[name]] = (
                                left_cmd if name == "wheel_left" else right_cmd
                            )

                    mujoco.mj_step(model, data)

                    # Read physics state
                    root_z = data.qpos[floating_qposadr + 2]
                    # projected_gravity from sensor or from quat
                    quat = data.qpos[floating_qposadr + 3:floating_qposadr + 7]
                    gravity_body = np.zeros(3)
                    mujoco.mju_rotVecQuat(gravity_body, np.array([0, 0, -1]), quat)
                    tilt_x = gravity_body[0]
                    tilt_y = gravity_body[1]
                    lin_vel = data.qvel[floating_qposadr:floating_qposadr + 3]
                    ang_vel = data.qvel[floating_qposadr + 3:floating_qposadr + 6]
                    pitch_rate = ang_vel[1]

                    # Ground contact classification.
                    wheel_contact = 0
                    non_wheel_contact = 0
                    for ci in range(data.ncon):
                        contact = data.contact[ci]
                        g1 = contact.geom1
                        g2 = contact.geom2
                        if g1 != ground_id and g2 != ground_id:
                            continue
                        other = g2 if g1 == ground_id else g1
                        if other in wheel_geom_ids:
                            wheel_contact = 1
                        if other in non_wheel_geom_ids:
                            non_wheel_contact = 1

                    # Actual wheel velocities
                    _, wl_dof = ji["wheel_left"]
                    _, wr_dof = ji["wheel_right"]
                    actual_left = data.qvel[wl_dof]
                    actual_right = data.qvel[wr_dof]

                    # Termination checks (matching training)
                    if abs(tilt_x) > 1.0 or abs(tilt_y) > 1.0:
                        terminated = True
                    if root_z < 0.18:
                        terminated = True
                else:
                    actual_left = 0.0
                    actual_right = 0.0
                    wheel_contact = 0
                    non_wheel_contact = 0

                rows.append(SweepRow(
                    case=case_name,
                    effort=effort_limit,
                    step=step,
                    tilt_x=tilt_x,
                    pitch_rate_y=pitch_rate,
                    lin_vel_x=lin_vel[0],
                    root_z=root_z,
                    wheel_contact=wheel_contact,
                    non_wheel_contact=non_wheel_contact,
                    left_cmd=left_cmd,
                    right_cmd=right_cmd,
                    actual_left_vel=actual_left,
                    actual_right_vel=actual_right,
                    terminated=terminated,
                ))

    return rows


def print_summary(rows: list[SweepRow], steps: int) -> None:
    """Print per-case summary: when contact starts, max tilt, terminal state."""
    print(f"\n{'='*90}")
    print(f"Fixed-wheel sweep summary ({steps} steps per case)")
    print(f"{'='*90}")
    print(f"{'case':<12} {'effort':>7} {'wheel@':>8} {'nonwheel@':>10} {'max_tilt':>9} "
          f"{'term@':>6} {'last_root_z':>10} {'avg_wl':>8} {'avg_wr':>8}")
    print(
        f"{'-'*12} {'-'*7} {'-'*8} {'-'*10} {'-'*9} {'-'*6} "
        f"{'-'*10} {'-'*8} {'-'*8}"
    )

    import itertools
    for (case_name, effort), group in itertools.groupby(
        sorted(rows, key=lambda r: (r.case, r.effort)),
        key=lambda r: (r.case, r.effort),
    ):
        grp = list(group)
        first_wheel_contact = next(
            (r.step for r in grp if r.wheel_contact and not r.terminated), -1
        )
        first_non_wheel_contact = next(
            (r.step for r in grp if r.non_wheel_contact and not r.terminated), -1
        )
        max_tilt = max(abs(r.tilt_x) for r in grp)
        term_step = next((r.step for r in grp if r.terminated), steps)
        last = grp[-1]
        avg_wl = np.mean([r.actual_left_vel for r in grp if not r.terminated])
        avg_wr = np.mean([r.actual_right_vel for r in grp if not r.terminated])

        wheel_str = f"{first_wheel_contact:>3d}" if first_wheel_contact >= 0 else "  none"
        non_wheel_str = (
            f"{first_non_wheel_contact:>3d}"
            if first_non_wheel_contact >= 0
            else "  none"
        )
        print(
            f"{case_name:<12} {effort:>7.1f} {wheel_str:>8} {non_wheel_str:>10} "
            f"{max_tilt:>9.4f} "
            f"{term_step:>6d} {last.root_z:>10.4f} {avg_wl:>8.3f} {avg_wr:>8.3f}"
        )


def main() -> None:
    args = parse_args()
    print("[INFO] Building models and running sweep...")
    rows = run_sweep(args)
    print_summary(rows, args.steps)

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "case", "effort", "step", "tilt_x", "pitch_rate_y",
                "lin_vel_x", "root_z", "wheel_contact", "non_wheel_contact",
                "left_cmd", "right_cmd", "actual_left_vel",
                "actual_right_vel", "terminated",
            ])
            for r in rows:
                writer.writerow(r.as_tuple())
        print(f"\n[INFO] CSV written to {args.output}")


if __name__ == "__main__":
    main()
