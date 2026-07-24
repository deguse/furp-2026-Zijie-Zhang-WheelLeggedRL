from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from hoppertrex_mjlab.hybrid.controller_schedule import (
  SCHEDULE_STATE_DEFINITION,
  parse_controller_schedule,
)
from hoppertrex_mjlab.scripts.build_hybrid_controller_schedule import (
  build_schedule,
)
from tests.test_hybrid_controller_schedule import _payload


class BuildHybridControllerScheduleTest(unittest.TestCase):
  def _fixture(self, root: Path) -> dict[str, object]:
    template = _payload()
    selection_evidence = dict(template["selection"])
    selection_evidence.pop("evaluation_artifact_sha256")
    selection_path = root / "flat_gate_selection.json"
    selection_path.write_text(json.dumps(selection_evidence), encoding="utf-8")
    manifest: dict[str, object] = {
      "height_nodes": template["height_nodes"],
      "pitch_nodes": template["pitch_nodes"],
      "q_diag": template["q_diag"],
      "r_diag": template["r_diag"],
      "bindings": template["bindings"],
      "selection": {
        "evaluation_artifact_path": selection_path.name,
        "evaluation_artifact_sha256": hashlib.sha256(
          selection_path.read_bytes()
        ).hexdigest(),
      },
      "nodes": [],
    }
    for h_index, height in enumerate(template["height_nodes"]):
      for p_index, pitch in enumerate(template["pitch_nodes"]):
        stem = f"node_{h_index}_{p_index}"
        npz = root / f"{stem}.npz"
        npz.write_bytes(b"frozen-node-data")
        equilibrium = 0.01 * (h_index + p_index)
        metadata = {
          "num_envs": 32,
          "steps": 2500,
          "warmup_steps": 250,
          "hold_steps": 5,
          "heldout_fraction": 0.20,
          "height_command": height,
          "pitch_command": pitch,
          "equilibrium_pitch": equilibrium,
          "state_definition_version": SCHEDULE_STATE_DEFINITION,
          "controller": {
            "gain_hash": template["bindings"][
              "identification_controller_gain_hash"
            ]
          },
          "calibration_hash": template["bindings"][
            "identification_calibration_hash"
          ],
          "posture_artifact_hash": template["bindings"][
            "posture_artifact_hash"
          ],
        }
        npz.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
        node = template["nodes"][h_index][p_index]
        controller = {
          **node,
          "gain": [node["gain"]],
          "q_diag": template["q_diag"],
          "r_diag": template["r_diag"],
          "source_npz": str(npz),
          "source_npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
          "source_metadata_sha256": hashlib.sha256(
            npz.with_suffix(".json").read_bytes()
          ).hexdigest(),
          "state_construction": {
            "state_definition_version": SCHEDULE_STATE_DEFINITION,
            "wheel_radius": 0.1,
          },
        }
        controller_path = root / f"{stem}_controller.json"
        controller_path.write_text(json.dumps(controller), encoding="utf-8")
        manifest["nodes"].append(
          {
            "height_m": height,
            "pitch_rad": pitch,
            "equilibrium_pitch": equilibrium,
            "controller_path": controller_path.name,
          }
        )
    return manifest

  def test_builds_only_from_cross_bound_node_sidecars(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      manifest = self._fixture(root)
      schedule = build_schedule(manifest, root)
      parsed = parse_controller_schedule(schedule)
      self.assertEqual(parsed.height_nodes, (0.29, 0.31, 0.33))
      first = schedule["nodes"][0][0]
      self.assertEqual(len(first["source_npz_sha256"]), 64)
      self.assertEqual(len(first["source_metadata_sha256"]), 64)

      first_manifest = manifest["nodes"][0]
      source = root / "node_0_0.json"
      metadata = json.loads(source.read_text(encoding="utf-8"))
      metadata["pitch_command"] = 0.0123
      source.write_text(json.dumps(metadata), encoding="utf-8")
      controller_path = root / "node_0_0_controller.json"
      controller = json.loads(controller_path.read_text(encoding="utf-8"))
      controller["source_metadata_sha256"] = hashlib.sha256(
        source.read_bytes()
      ).hexdigest()
      controller_path.write_text(json.dumps(controller), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "pitch metadata mismatch"):
        build_schedule(manifest, root)
      self.assertEqual(first_manifest["pitch_rad"], -0.032)

  def test_rejects_node_source_changed_after_controller_fit(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      manifest = self._fixture(root)
      (root / "node_0_0.npz").write_bytes(b"replaced-after-fit")
      with self.assertRaisesRegex(ValueError, "changed after fitting"):
        build_schedule(manifest, root)

  def test_rejects_unbound_flat_gate_evidence(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      manifest = self._fixture(root)
      (root / "flat_gate_selection.json").write_text("{}", encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
        build_schedule(manifest, root)


if __name__ == "__main__":
  unittest.main()
