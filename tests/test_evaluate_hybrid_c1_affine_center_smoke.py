from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hoppertrex_mjlab.scripts import evaluate_hybrid_c1_affine_center_smoke as smoke
from hoppertrex_mjlab.scripts.evaluate_hybrid_c1_affine_center_smoke import (
    FORMAL_CENTER_HEIGHT,
    center_cells,
)


class HybridC1AffineCenterSmokeTest(unittest.TestCase):
    def test_collection_git_sha_must_be_shared_and_full(self) -> None:
        nodes = {
            stem: {"metadata": {"git_sha": "1" * 40}} for stem in smoke.NODE_STEMS
        }
        self.assertEqual(smoke._collection_git_sha(nodes), "1" * 40)
        nodes[smoke.NODE_STEMS[-1]]["metadata"]["git_sha"] = "2" * 40
        with self.assertRaisesRegex(ValueError, "share one collection Git SHA"):
            smoke._collection_git_sha(nodes)

    def test_cli_accepts_each_required_provenance_argument_once(self) -> None:
        args = smoke.parse_args(
            [
                "--nodes-dir",
                "nodes",
                "--output",
                "smoke.json",
                "--compensated-qualification",
                "qualification.json",
                "--git-sha",
                "1" * 40,
                "--mjlab-git-sha",
                "2" * 40,
            ]
        )
        self.assertEqual(args.git_sha, "1" * 40)
        self.assertEqual(args.mjlab_git_sha, "2" * 40)

    def test_center_smoke_is_exactly_three_registered_cells(self) -> None:
        self.assertEqual(
            center_cells(),
            [
                (FORMAL_CENTER_HEIGHT, 0.0, 0.0),
                (FORMAL_CENTER_HEIGHT, 0.0, 0.05),
                (FORMAL_CENTER_HEIGHT, 0.0, -0.05),
            ],
        )

    def test_loader_requires_affine_equilibrium_and_bound_delta_means(self) -> None:
        shapes = {
            "states": (2, 4),
            "inputs": (2, 1),
            "next_states": (2, 4),
            "heldout_states": (1, 4),
            "heldout_inputs": (1, 1),
            "heldout_next_states": (1, 4),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for stem in smoke.NODE_STEMS:
                arrays = {name: np.zeros(shape) for name, shape in shapes.items()}
                np.savez(root / f"{stem}.npz", **arrays)
                metadata = {
                    "state_definition_version": smoke.AFFINE_SCHEDULE_STATE_DEFINITION,
                    "equilibrium_state": [0.01, 0.0, 0.0, 0.0],
                    "equilibrium_input": [0.1],
                    "delta_state_mean": [0.0, 0.0, 0.0, 0.0],
                    "delta_input_mean": [0.0],
                    "controller": {"gain": [1.0, 2.0, 3.0, 4.0]},
                }
                (root / f"{stem}.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
            with patch.object(smoke, "EXPECTED_ARRAY_SHAPES", shapes):
                self.assertEqual(len(smoke.load_affine_nodes(root)), 9)
                metadata_path = root / "node_h0_p0.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["delta_input_mean"] = [0.5]
                metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "delta input mean"):
                    smoke.load_affine_nodes(root)


if __name__ == "__main__":
    unittest.main()
