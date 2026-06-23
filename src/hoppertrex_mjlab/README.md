# HopperTrex MjLab Task

Standalone MjLab package for the HopperTrex two-wheeled legged robot. This folder does not depend on `hoppertrex_isaaclab` at runtime, so it can be distributed on its own after removing local credentials and runtime logs.

## Environment

MjLab is installed at:

```bash
/home/mocap/mjlab
```

Use its uv environment:

```bash
source /home/mocap/mjlab/.venv/bin/activate
```

The main task id is:

```text
Mjlab-HopperTrex-Balance-v0
```

The old alias is also registered:

```text
hoppertrex-balance-v0
```

## Robot Asset

The robot MJCF is generated from the HopperTrex Onshape assembly:

```text
assets/hoppertrex/robot.xml
assets/hoppertrex/meshes/
```

The original USD is kept only as a reference asset. The MjLab runtime uses MJCF/XML, not IsaacLab USD loading.

Robot config:

```text
assets/HopperTrex_CFG.py
```

Motor and pose settings:

- Leg motors: Damiao DM-J6248P
- Wheel motors: Lingkong/RMD-L-9025-35T
- Initial hip pose: left `+39 deg`, right `-39 deg`
- Initial knee pose: `0 rad`
- Wheel velocity action scale: `12.0 rad/s`

## Task

Task config:

```text
tasks/hoppertrex_balance_task.py
```

The legs are held at the fixed standing pose. The policy controls wheel velocity for balance and planar velocity tracking.

Command name:

```text
twist
```

Command ranges:

```text
lin_vel_x: [-1.0, 1.0] m/s
lin_vel_y: [0.0, 0.0] m/s
ang_vel_z: [-1.0, 1.0] rad/s
```

Action groups:

```text
leg_pos   : 4 dims, JointPositionActionCfg, scale = 0.0
wheel_vel : 2 dims, JointVelocityActionCfg, scale = 12.0
```

Observation shape:

```text
actor:  30
critic: 30
```

## Smoke Tests

From the repository root:

```bash
/home/mocap/mjlab/.venv/bin/python hoppertrex_mjlab/assets/spawn_robot.py --steps 5
/home/mocap/mjlab/.venv/bin/python hoppertrex_mjlab/scripts/zero_agent.py --device cpu --num_envs 1 --max_steps 5
/home/mocap/mjlab/.venv/bin/python hoppertrex_mjlab/scripts/random_agent.py --device cpu --num_envs 1 --max_steps 5
```

For normal GPU use, omit `--device cpu` or set `--device cuda:0`.

## Train

Short debug run:

```bash
/home/mocap/mjlab/.venv/bin/python hoppertrex_mjlab/scripts/rsl_rl/train.py \
  --env.scene.num-envs 64 \
  --agent.max-iterations 50
```

Typical training:

```bash
/home/mocap/mjlab/.venv/bin/python hoppertrex_mjlab/scripts/rsl_rl/train.py \
  --env.scene.num-envs 4096
```

Use CPU only for smoke tests:

```bash
/home/mocap/mjlab/.venv/bin/python hoppertrex_mjlab/scripts/rsl_rl/train.py \
  --gpu-ids None \
  --env.scene.num-envs 1 \
  --agent.max-iterations 1 \
  --agent.num-steps-per-env 2 \
  --agent.algorithm.num-mini-batches 1
```

Training logs are written under:

```text
hoppertrex_mjlab/logs/
```

## Play

Zero-action viewer:

```bash
/home/mocap/mjlab/.venv/bin/python hoppertrex_mjlab/scripts/rsl_rl/play.py \
  --agent zero \
  --num-envs 1 \
  --device cuda:0
```

Play a selected checkpoint:

```bash
/home/mocap/mjlab/.venv/bin/python hoppertrex_mjlab/scripts/rsl_rl/play.py \
  --agent trained \
  --checkpoint-file /path/to/model.pt \
  --num-envs 1 \
  --device cuda:0
```

## Regenerate MJCF From Onshape

The exporter is:

```text
tools/export_onshape_robot.py
```

It uses `onshape-urdf-exporter`, which currently needs Python 3.10 because of its `open3d==0.16.0` dependency. A temporary export environment can be created outside this package:

```bash
uv venv /tmp/hoppertrex_onshape_venv --python /usr/bin/python3.10
uv pip install --python /tmp/hoppertrex_onshape_venv/bin/python onshape-urdf-exporter trimesh mujoco
cp hoppertrex_mjlab/tools/onshape.key.example hoppertrex_mjlab/tools/onshape.key
# Edit hoppertrex_mjlab/tools/onshape.key with your own Onshape access key and secret.
/tmp/hoppertrex_onshape_venv/bin/python hoppertrex_mjlab/tools/export_onshape_robot.py \
  --key-file hoppertrex_mjlab/tools/onshape.key \
  --output hoppertrex_mjlab/assets/hoppertrex
```

Do not distribute `tools/onshape.key` to students.

## Git Policy

Do not commit runtime logs, generated output, or smoke-test logs:

```text
hoppertrex_mjlab/logs/
hoppertrex_mjlab/output/
hoppertrex_mjlab/outputs/
```

Only selected and validated policy weights should be copied into and committed from:

```text
hoppertrex_mjlab/policy/
```

Before committing a policy, verify that it can balance and track velocity commands reliably.
