from __future__ import annotations

import math
import unittest

import numpy as np

from hoppertrex_mjlab.hybrid.controller_schedule import (
    AFFINE_SCHEDULE_STATE_DEFINITION,
    REGISTERED_HEIGHT_NODES,
    SCHEDULE_ARTIFACT_TYPE,
    SCHEDULE_STATE_DEFINITION,
    canonical_hash,
    parse_controller_schedule,
    qr_candidate_grid,
    select_symmetric_pitch_nodes,
)
from hoppertrex_mjlab.hybrid.identification import (
    CONTROLLER_STATE_NAMES,
    NOMINAL_WHEEL_RADIUS_M,
)


def _payload() -> dict[str, object]:
    candidates = []
    selected_index = -1
    for index, item in enumerate(qr_candidate_grid()):
        if item["q_diag"] == [20.0, 2.0, 4.0, 0.5] and item["r_diag"] == [1.0]:
            selected_index = index
        candidates.append(
            {
                "q_diag": item["q_diag"],
                "r_diag": item["r_diag"],
                "anchor_alpha": 0.125,
                "flat_gate_passed": True,
                "worst_velocity_error": 0.01 + index * 1.0e-4,
                "p95_pitch": 0.02,
                "p99_pitch_rate": 0.2,
                "wheel_target_rate": 0.3,
            }
        )
    candidates[selected_index].update(
        {
            "worst_velocity_error": 0.001,
            "p95_pitch": 0.001,
            "p99_pitch_rate": 0.001,
            "wheel_target_rate": 0.001,
        }
    )
    nodes = []
    for h_index in range(3):
        row = []
        for p_index in range(3):
            row.append(
                {
                    "controller_type": "lqr",
                    "gain": [h_index, p_index, h_index + p_index, 1.0],
                    "raw_gain": [
                        8.0 + (h_index - 8.0) / 0.125,
                        1.0 + (p_index - 1.0) / 0.125,
                        3.0 + (h_index + p_index - 3.0) / 0.125,
                        0.2 + (1.0 - 0.2) / 0.125,
                    ],
                    "equilibrium_pitch": 0.01 * (h_index + p_index),
                    "equilibrium_state": [
                        0.01 * (h_index + p_index),
                        0.001 * h_index,
                        0.002 * p_index,
                        0.003 * (h_index + p_index),
                    ],
                    "equilibrium_input": 0.1 * (h_index + p_index),
                    "model": {
                        "a": np.eye(4).tolist(),
                        "b": np.ones((4, 1)).tolist(),
                    },
                    "controllability_rank": 4,
                    "heldout_one_step_nrmse": {"maximum": 0.05},
                    "fallback_reasons": [],
                    "source_npz": f"node_{h_index}_{p_index}.npz",
                    "controller_file_sha256": "d" * 64,
                    "source_npz_sha256": "e" * 64,
                    "source_metadata_sha256": "f" * 64,
                }
            )
        nodes.append(row)
    payload: dict[str, object] = {
        "schema_version": 2,
        "artifact_type": SCHEDULE_ARTIFACT_TYPE,
        "state_names": list(CONTROLLER_STATE_NAMES),
        "state_construction": {
            "state_definition_version": AFFINE_SCHEDULE_STATE_DEFINITION,
            "wheel_radius": NOMINAL_WHEEL_RADIUS_M,
        },
        "height_nodes": list(REGISTERED_HEIGHT_NODES),
        "pitch_nodes": [-0.032, 0.0, 0.032],
        "q_diag": [20.0, 2.0, 4.0, 0.5],
        "r_diag": [1.0],
        "anchor_alpha": 0.125,
        "incumbent_gain": [8.0, 1.0, 3.0, 0.2],
        "bindings": {
            "identification_controller_gain_hash": "a" * 64,
            "identification_calibration_hash": "b" * 64,
            "posture_artifact_hash": "c" * 64,
        },
        "collection_protocol": {
            "num_envs": 32,
            "steps": 2500,
            "warmup_steps": 250,
            "equilibrium_window_steps": 100,
            "hold_steps": 5,
            "heldout_fraction": 0.20,
        },
        "selection": {
            "status": "flat_gate_selected",
            "evaluation_artifact_sha256": "1" * 64,
            "git_sha": "2" * 40,
            "mjlab_git_sha": "3" * 40,
            "selected_candidate_index": selected_index,
            "evaluated_candidates": candidates,
            "ranking": [
                "worst_velocity_error",
                "p95_pitch",
                "p99_pitch_rate",
                "wheel_target_rate",
            ],
        },
        "nodes": nodes,
    }
    payload["schedule_hash"] = canonical_hash(payload, hash_field="schedule_hash")
    return payload


class HybridControllerScheduleTest(unittest.TestCase):
    def test_qr_grid_has_27_unique_candidates(self) -> None:
        candidates = qr_candidate_grid()
        self.assertEqual(len(candidates), 27)
        self.assertEqual(
            len(
                {(tuple(item["q_diag"]), tuple(item["r_diag"])) for item in candidates}
            ),
            27,
        )

    def test_symmetric_pitch_falls_back_to_widest_qualified_range(self) -> None:
        qualified = {
            -0.032: (False, True, True),
            0.032: (True, True, True),
            -0.024: (True, True, True),
            0.024: (True, True, True),
            0.0: (True, True, True),
        }
        self.assertEqual(select_symmetric_pitch_nodes(qualified), (-0.024, 0.0, 0.024))

    def test_bilinear_interpolation_and_clamp(self) -> None:
        schedule = parse_controller_schedule(_payload())
        midpoint = 0.5 * (REGISTERED_HEIGHT_NODES[0] + REGISTERED_HEIGHT_NODES[1])
        gain, equilibrium, clamped = schedule.interpolate(midpoint, -0.016)
        np.testing.assert_allclose(gain, [0.5, 0.5, 1.0, 1.0])
        self.assertAlmostEqual(equilibrium, 0.01)
        self.assertFalse(clamped)
        gain, state, control, clamped = schedule.interpolate_affine(
            midpoint, -0.016
        )
        np.testing.assert_allclose(gain, [0.5, 0.5, 1.0, 1.0])
        np.testing.assert_allclose(state, [0.01, 0.0005, 0.001, 0.003])
        self.assertAlmostEqual(control, 0.1)
        self.assertFalse(clamped)
        gain, _, clamped = schedule.interpolate(0.50, 0.20)
        np.testing.assert_allclose(gain, [2.0, 2.0, 4.0, 1.0])
        self.assertTrue(clamped)

    def test_schema_one_pitch_only_schedule_remains_compatible(self) -> None:
        payload = _payload()
        payload["schema_version"] = 1
        payload["state_construction"][  # type: ignore[index]
            "state_definition_version"
        ] = SCHEDULE_STATE_DEFINITION
        payload["collection_protocol"].pop(  # type: ignore[union-attr]
            "equilibrium_window_steps"
        )
        payload.pop("anchor_alpha")
        payload.pop("incumbent_gain")
        for candidate in payload["selection"]["evaluated_candidates"]:  # type: ignore[index]
            candidate.pop("anchor_alpha")
        for row in payload["nodes"]:  # type: ignore[union-attr]
            for node in row:
                node.pop("equilibrium_state")
                node.pop("equilibrium_input")
        payload["schedule_hash"] = canonical_hash(
            payload, hash_field="schedule_hash"
        )
        schedule = parse_controller_schedule(payload)
        np.testing.assert_allclose(schedule.equilibrium_state[:, :, 1:], 0.0)
        np.testing.assert_allclose(schedule.equilibrium_input, 0.0)

    def test_rejects_unqualified_node_and_hash_drift(self) -> None:
        payload = _payload()
        payload["nodes"][0][0]["controller_type"] = "pd"  # type: ignore[index]
        payload["schedule_hash"] = canonical_hash(payload, hash_field="schedule_hash")
        with self.assertRaisesRegex(ValueError, "qualified LQR"):
            parse_controller_schedule(payload)
        payload = _payload()
        payload["schedule_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash"):
            parse_controller_schedule(payload)

    def test_rejects_affine_gain_not_equal_to_recorded_anchor_blend(self) -> None:
        payload = _payload()
        payload["nodes"][0][0]["gain"][0] += 0.5  # type: ignore[index]
        payload["schedule_hash"] = canonical_hash(
            payload, hash_field="schedule_hash"
        )
        with self.assertRaisesRegex(ValueError, "anchor blend"):
            parse_controller_schedule(payload)

    def test_rejects_forged_qr_selection(self) -> None:
        payload = _payload()
        payload["selection"]["evaluated_candidates"] = payload["selection"][  # type: ignore[index]
            "evaluated_candidates"
        ][:-1]
        payload["schedule_hash"] = canonical_hash(
            payload, hash_field="schedule_hash"
        )
        with self.assertRaisesRegex(ValueError, "27 evaluated"):
            parse_controller_schedule(payload)

    def test_rejects_nonfinite_metrics_and_nonhex_provenance(self) -> None:
        payload = _payload()
        payload["nodes"][0][0]["heldout_one_step_nrmse"][  # type: ignore[index]
            "maximum"
        ] = math.nan
        payload["schedule_hash"] = canonical_hash(
            payload, hash_field="schedule_hash"
        )
        with self.assertRaisesRegex(ValueError, "NRMSE limit"):
            parse_controller_schedule(payload)

        payload = _payload()
        payload["selection"]["git_sha"] = "z" * 40  # type: ignore[index]
        payload["schedule_hash"] = canonical_hash(
            payload, hash_field="schedule_hash"
        )
        with self.assertRaisesRegex(ValueError, "hex digest"):
            parse_controller_schedule(payload)

    def test_rejects_unregistered_grid_and_boolean_index(self) -> None:
        payload = _payload()
        payload["height_nodes"] = [0.29, 0.31, 0.33]
        payload["schedule_hash"] = canonical_hash(
            payload, hash_field="schedule_hash"
        )
        with self.assertRaisesRegex(ValueError, "registered grid"):
            parse_controller_schedule(payload)

        payload = _payload()
        payload["selection"]["selected_candidate_index"] = True  # type: ignore[index]
        payload["schedule_hash"] = canonical_hash(
            payload, hash_field="schedule_hash"
        )
        with self.assertRaisesRegex(ValueError, "selected index"):
            parse_controller_schedule(payload)


if __name__ == "__main__":
    unittest.main()
