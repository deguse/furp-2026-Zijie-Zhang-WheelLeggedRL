from __future__ import annotations

import unittest

import numpy as np

from hoppertrex_mjlab.hybrid.stair_classical import StairPhase
from hoppertrex_mjlab.hybrid.stair_residual import (
    PRIVILEGED_FIELD_NAMES,
    StairCurriculumState,
    build_stair_residual_observations,
    residual_promotion_decision,
    update_stair_curriculum,
)


class HybridStairResidualTest(unittest.TestCase):
    def test_actor_excludes_privileged_values(self) -> None:
        common = dict(
            proprioception=[1.0, 2.0],
            phase=StairPhase.CLIMB,
            classical_wheel_baseline=[0.1, -0.1],
            nominal_leg_targets=[0.0] * 4,
            classical_errors=[0.2, 0.3],
            previous_residual=[0.0] * 6,
            stair_mode=True,
        )
        actor_a, critic_a = build_stair_residual_observations(
            **common,
            privileged={
                name: index + 1.0 for index, name in enumerate(PRIVILEGED_FIELD_NAMES)
            },
        )
        actor_b, critic_b = build_stair_residual_observations(
            **common,
            privileged={name: 100.0 for name in PRIVILEGED_FIELD_NAMES},
        )
        np.testing.assert_array_equal(actor_a, actor_b)
        self.assertFalse(np.array_equal(critic_a, critic_b))

    def test_curriculum_requires_three_consecutive_ready_evaluations(self) -> None:
        state = StairCurriculumState(0.01, 0.04)
        for _ in range(2):
            state = update_stair_curriculum(state, success_rate=0.80)
        self.assertEqual(state.upper_height_m, 0.04)
        state = update_stair_curriculum(state, success_rate=0.80)
        self.assertEqual(state.upper_height_m, 0.05)
        self.assertEqual(state.consecutive_ready_evaluations, 0)

    def test_promotion_requires_boundary_extension_and_all_gates(self) -> None:
        classical = [
            {
                "height_m": 0.00,
                "success_rate": 1.0,
                "terminations": 0,
                "non_wheel_contacts": 0,
            },
            {
                "height_m": 0.01,
                "success_rate": 1.0,
                "terminations": 0,
                "non_wheel_contacts": 0,
            },
            {
                "height_m": 0.02,
                "success_rate": 0.5,
                "terminations": 0,
                "non_wheel_contacts": 0,
            },
        ]
        residual = [
            {
                "height_m": 0.00,
                "success_rate": 1.0,
                "terminations": 0,
                "non_wheel_contacts": 0,
            },
            {
                "height_m": 0.01,
                "success_rate": 1.0,
                "terminations": 0,
                "non_wheel_contacts": 0,
            },
            {
                "height_m": 0.02,
                "success_rate": 0.95,
                "terminations": 0,
                "non_wheel_contacts": 0,
            },
        ]
        result = residual_promotion_decision(
            classical_rows=classical,
            residual_rows=residual,
            flat_gate_passed=True,
            standing_gate_passed=True,
            velocity_gate_passed=True,
            stage5_gate_passed=True,
            ablations_complete=True,
        )
        self.assertTrue(result["promotion_eligible"])
        result = residual_promotion_decision(
            classical_rows=classical,
            residual_rows=residual,
            flat_gate_passed=True,
            standing_gate_passed=True,
            velocity_gate_passed=True,
            stage5_gate_passed=False,
            ablations_complete=True,
        )
        self.assertFalse(result["promotion_eligible"])


if __name__ == "__main__":
    unittest.main()
