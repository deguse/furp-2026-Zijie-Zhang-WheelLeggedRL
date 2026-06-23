#!/usr/bin/env python3
"""Load the HopperTrex MJCF model and run a short MuJoCo sanity check."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco

XML_PATH = Path(__file__).resolve().parent / "hoppertrex" / "robot.xml"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--steps", type=int, default=200)
  parser.add_argument("--xml", type=Path, default=XML_PATH)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  model = mujoco.MjModel.from_xml_path(str(args.xml))
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)

  print(f"[INFO] Loaded {args.xml}")
  print(f"[INFO] nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} ngeom={model.ngeom}")
  print("[INFO] joints:")
  for joint_id in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    print(f"  - {joint_id}: {name}")
  print("[INFO] actuators: none in raw XML; MjLab adds them from HopperTrex_CFG.py")

  for _ in range(args.steps):
    mujoco.mj_step(model, data)

  print(f"[INFO] stepped={args.steps} qpos={data.qpos.tolist()}")
  print(f"[INFO] contacts={data.ncon}")


if __name__ == "__main__":
  main()

