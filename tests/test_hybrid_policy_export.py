import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv

from hoppertrex_mjlab.hybrid.observation_builder import (
  JOINT_ORDER,
  OBSERVATION_DIM,
  OBSERVATION_TERMS,
  ObservationInputs,
  build_observation,
)
from hoppertrex_mjlab.scripts.export_hybrid_policy import (
  build_actor_module,
  export_metadata,
  load_actor_weights,
  main as export_main,
)
from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
  make_hoppertrex_hybrid_env_cfg,
)


def _one_env_cfg(stage: int):
  # play=True: deployment mirrors inference semantics, where observation
  # corruption is disabled (enable_corruption = not play in the task cfg).
  cfg = make_hoppertrex_hybrid_env_cfg(stage=stage, play=True)
  cfg.scene.num_envs = 1
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.num_envs = 1
  return cfg


def _inputs_from_env(env: ManagerBasedRlEnv) -> ObservationInputs:
  robot = env.scene["robot"]
  term = env.action_manager.get_term("hybrid_wheel_leg")
  data = robot.data
  twist = env.command_manager.get_command("twist")[0]
  posture = env.command_manager.get_command("posture")[0]
  return ObservationInputs(
    base_lin_vel=tuple(data.root_link_lin_vel_b[0].tolist()),
    base_ang_vel=tuple(data.root_link_ang_vel_b[0].tolist()),
    projected_gravity=tuple(data.projected_gravity_b[0].tolist()),
    velocity_command=tuple(twist.tolist()),
    posture_command=tuple(posture.tolist()),
    joint_pos=tuple(data.joint_pos[0].tolist()),
    joint_vel=tuple(data.joint_vel[0].tolist()),
    controller_baseline=tuple(term.controller_baseline[0].tolist()),
    applied_residual=tuple(term.applied_residual[0].tolist()),
  )


class ObservationBuilderEquivalenceTest(unittest.TestCase):
  def test_builder_matches_live_actor_observations(self):
    """Core acceptance: builder output == env actor observation, per step."""

    env = ManagerBasedRlEnv(cfg=_one_env_cfg(stage=5), device="cpu")
    try:
      env.reset(seed=2026)
      robot = env.scene["robot"]
      self.assertEqual(tuple(robot.joint_names), JOINT_ORDER)
      defaults_pos = tuple(
        robot.data.default_joint_pos[0].cpu().tolist()
      )
      defaults_vel = tuple(
        robot.data.default_joint_vel[0].cpu().tolist()
      )
      schedule = (
        (0.6, 0.2, 0.1, -0.1, 0.3, -0.3),
        (-0.4, -0.2, 0.0, 0.2, -0.1, 0.1),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      )
      for row in schedule:
        observations, _r, _t, _tr, _e = env.step(
          torch.tensor([row], dtype=torch.float32)
        )
        built = build_observation(
          _inputs_from_env(env),
          default_joint_pos=defaults_pos,
          default_joint_vel=defaults_vel,
        )
        np.testing.assert_allclose(
          observations["actor"][0].cpu().numpy(),
          built,
          rtol=0.0,
          atol=1.0e-6,
        )
    finally:
      env.close()

  def test_layout_sums_to_34(self):
    self.assertEqual(sum(dim for _n, dim in OBSERVATION_TERMS), 34)
    self.assertEqual(OBSERVATION_DIM, 34)

  def test_builder_rejects_wrong_dims(self):
    inputs = ObservationInputs(
      base_lin_vel=(0.0, 0.0, 0.0),
      base_ang_vel=(0.0, 0.0, 0.0),
      projected_gravity=(0.0, 0.0, -1.0),
      velocity_command=(0.0, 0.0, 0.0),
      posture_command=(0.31, 0.0),
      joint_pos=(0.0,) * 5,
      joint_vel=(0.0,) * 6,
      controller_baseline=(0.0, 0.0),
      applied_residual=(0.0,) * 6,
    )
    with self.assertRaises(ValueError):
      build_observation(inputs, default_joint_pos=(0.0,) * 6)


class PolicyExportEquivalenceTest(unittest.TestCase):
  def _fake_checkpoint(self) -> dict:
    torch.manual_seed(11)
    module = build_actor_module()
    actor_state = {
      f"mlp.{key}": value.clone()
      for key, value in module.state_dict().items()
    }
    actor_state["std_param"] = torch.full((6,), 0.6)
    return {
      "actor_state_dict": actor_state,
      "critic_state_dict": {},
      "iter": 99,
      "infos": {
        "hybrid_training": {"git_sha": "f" * 40},
        "hybrid_stage1_bootstrap": {
          "controller_gain_hash": "c" * 64,
          "calibration_hash": "v" * 64,
        },
        "hybrid_stage_migration": {
          "yaw_calibration_hash": "y" * 64,
          "posture_map_hash": "p" * 64,
          "station_calibration_hash": "s" * 64,
        },
      },
    }

  def test_exported_module_matches_source_weights_exactly(self):
    checkpoint = self._fake_checkpoint()
    module = build_actor_module()
    load_actor_weights(module, checkpoint["actor_state_dict"])
    module.eval()

    reference = build_actor_module()
    reference.load_state_dict(
      {
        key.removeprefix("mlp."): value
        for key, value in checkpoint["actor_state_dict"].items()
        if key.startswith("mlp.")
      }
    )
    reference.eval()

    torch.manual_seed(3)
    batch = torch.randn(64, OBSERVATION_DIM)
    with torch.no_grad():
      torch.testing.assert_close(
        module(batch), reference(batch), rtol=0.0, atol=0.0
      )

  def test_load_rejects_architecture_drift(self):
    checkpoint = self._fake_checkpoint()
    state = dict(checkpoint["actor_state_dict"])
    state["mlp.6.weight"] = torch.zeros(6, 128)
    module = build_actor_module()
    with self.assertRaisesRegex(ValueError, "do not match"):
      load_actor_weights(module, state)

  def test_end_to_end_export_writes_torchscript_and_metadata(self):
    checkpoint = self._fake_checkpoint()
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      checkpoint_path = root / "model_99.pt"
      torch.save(checkpoint, checkpoint_path)
      output = root / "policy.ts"

      export_main(
        [
          "--checkpoint-file",
          str(checkpoint_path),
          "--stage",
          "5",
          "--output",
          str(output),
        ]
      )

      traced = torch.jit.load(str(output))
      module = build_actor_module()
      load_actor_weights(module, checkpoint["actor_state_dict"])
      module.eval()
      torch.manual_seed(5)
      batch = torch.randn(16, OBSERVATION_DIM)
      with torch.no_grad():
        torch.testing.assert_close(
          traced(batch), module(batch), rtol=0.0, atol=0.0
        )

      metadata = json.loads(
        output.with_suffix(".metadata.json").read_text(encoding="utf-8")
      )
      self.assertEqual(metadata["observation_dim"], 34)
      self.assertEqual(metadata["training_git_sha"], "f" * 40)
      self.assertEqual(
        metadata["artifact_hashes"]["posture_map_hash"], "p" * 64
      )
      self.assertEqual(len(metadata["action_mask"]), 6)
      self.assertEqual(len(metadata["checkpoint_file_sha256"]), 64)

  def test_metadata_records_stage_mask_and_terms(self):
    checkpoint = self._fake_checkpoint()
    with tempfile.TemporaryDirectory() as temp:
      path = Path(temp) / "model.pt"
      torch.save(checkpoint, path)
      metadata = export_metadata(
        checkpoint_path=path, checkpoint=checkpoint, stage=5
      )
    self.assertEqual(
      [term["name"] for term in metadata["observation_terms"]],
      [name for name, _dim in OBSERVATION_TERMS],
    )
    self.assertEqual(metadata["action_mask"], [True] * 6)


if __name__ == "__main__":
  unittest.main()
