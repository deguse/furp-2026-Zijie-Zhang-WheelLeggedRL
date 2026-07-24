from __future__ import annotations

import unittest

import numpy as np

from hoppertrex_mjlab.hybrid.controller_schedule import (
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
    nodes = []
    for h_index in range(3):
        row = []
        for p_index in range(3):
            row.append(
                {
                    "controller_type": "lqr",
                    "gain": [h_index, p_index, h_index + p_index, 1.0],
                    "equilibrium_pitch": 0.01 * (h_index + p_index),
                    "controllability_rank": 4,
                    "heldout_one_step_nrmse": {"maximum": 0.05},
                    "fallback_reasons": [],
                }
            )
        nodes.append(row)
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": SCHEDULE_ARTIFACT_TYPE,
        "state_names": list(CONTROLLER_STATE_NAMES),
        "state_construction": {
            "state_definition_version": SCHEDULE_STATE_DEFINITION,
            "wheel_radius": NOMINAL_WHEEL_RADIUS_M,
        },
        "height_nodes": [0.29, 0.31, 0.33],
        "pitch_nodes": [-0.032, 0.0, 0.032],
        "q_diag": [20.0, 2.0, 4.0, 0.5],
        "r_diag": [1.0],
        "bindings": {"posture_artifact_hash": "a" * 64},
        "selection": {
            "status": "flat_gate_selected",
            "candidate_count": 27,
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
        gain, equilibrium, clamped = schedule.interpolate(0.30, -0.016)
        np.testing.assert_allclose(gain, [0.5, 0.5, 1.0, 1.0])
        self.assertAlmostEqual(equilibrium, 0.01)
        self.assertFalse(clamped)
        gain, _, clamped = schedule.interpolate(0.50, 0.20)
        np.testing.assert_allclose(gain, [2.0, 2.0, 4.0, 1.0])
        self.assertTrue(clamped)

    def test_rejects_unqualified_node_and_hash_drift(self) -> None:
        payload = _payload()
        payload["nodes"][0][0]["controller_type"] = "pd"  # type: ignore[index]
        payload["schedule_hash"] = canonical_hash(payload, hash_field="schedule_hash")
        with self.assertRaisesRegex(ValueError, "qualified LQR"):
            parse_controller_schedule(payload)
        payload = _payload()
        payload["height_nodes"] = [0.28, 0.31, 0.33]
        with self.assertRaisesRegex(ValueError, "hash"):
            parse_controller_schedule(payload)


if __name__ == "__main__":
    unittest.main()
