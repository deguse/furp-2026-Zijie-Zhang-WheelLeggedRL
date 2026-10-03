from __future__ import annotations

import unittest

import numpy as np

from hoppertrex_mjlab.hybrid.support_load_allocator import (
    SupportLoadAllocatorConfig,
    allocate_support_loads,
)


class SupportLoadAllocatorTest(unittest.TestCase):
    def setUp(self):
        self.config = SupportLoadAllocatorConfig(
            track_width_m=0.4,
            minimum_contact_load_n=20.0,
            maximum_contact_load_n=120.0,
            total_load_weight=1.0 / 160.0**2,
            roll_moment_weight=1.0 / 24.0**2,
        )

    def test_symmetric_and_roll_moment_exact_when_inside_bounds(self):
        symmetric = allocate_support_loads(
            self.config,
            desired_total_load_n=160.0,
            desired_roll_moment_nm=0.0,
            left_in_contact=True,
            right_in_contact=True,
        )
        self.assertAlmostEqual(symmetric.left_load_n, 80.0, places=5)
        self.assertAlmostEqual(symmetric.right_load_n, 80.0, places=5)
        self.assertTrue(symmetric.bilateral_minimum_satisfied)

        rolled = allocate_support_loads(
            self.config,
            desired_total_load_n=160.0,
            desired_roll_moment_nm=8.0,
            left_in_contact=True,
            right_in_contact=True,
        )
        self.assertAlmostEqual(rolled.left_load_n, 100.0, places=5)
        self.assertAlmostEqual(rolled.right_load_n, 60.0, places=5)
        self.assertAlmostEqual(rolled.achieved_roll_moment_nm, 8.0, places=5)

    def test_contact_modes_and_minimum_load_constraint(self):
        left = allocate_support_loads(
            self.config,
            desired_total_load_n=160.0,
            desired_roll_moment_nm=0.0,
            left_in_contact=True,
            right_in_contact=False,
        )
        self.assertEqual(left.contact_mode, "left_only")
        self.assertEqual(left.right_load_n, 0.0)
        self.assertGreaterEqual(left.left_load_n, 20.0)
        self.assertFalse(left.bilateral_minimum_satisfied)

        unsupported = allocate_support_loads(
            self.config,
            desired_total_load_n=160.0,
            desired_roll_moment_nm=0.0,
            left_in_contact=False,
            right_in_contact=False,
        )
        self.assertEqual(unsupported.contact_mode, "unsupported")
        self.assertEqual(unsupported.left_load_n, 0.0)
        self.assertEqual(unsupported.right_load_n, 0.0)

        preload = allocate_support_loads(
            self.config,
            desired_total_load_n=0.0,
            desired_roll_moment_nm=0.0,
            left_in_contact=True,
            right_in_contact=True,
        )
        self.assertEqual(preload.left_load_n, 20.0)
        self.assertEqual(preload.right_load_n, 20.0)

    def test_active_set_solution_beats_dense_grid(self):
        result = allocate_support_loads(
            self.config,
            desired_total_load_n=205.0,
            desired_roll_moment_nm=18.0,
            left_in_contact=True,
            right_in_contact=True,
        )
        values = np.linspace(20.0, 120.0, 1001)
        left, right = np.meshgrid(values, values, indexing="ij")
        objective = (
            self.config.total_load_weight * (left + right - 205.0) ** 2
            + self.config.roll_moment_weight
            * (0.2 * (left - right) - 18.0) ** 2
            + self.config.regularization_weight
            * ((left - 102.5) ** 2 + (right - 102.5) ** 2)
        )
        self.assertLessEqual(result.objective, float(objective.min()) + 1.0e-9)

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            SupportLoadAllocatorConfig(
                track_width_m=0.0,
                minimum_contact_load_n=1.0,
                maximum_contact_load_n=2.0,
            )
        with self.assertRaises(ValueError):
            allocate_support_loads(
                self.config,
                desired_total_load_n=-1.0,
                desired_roll_moment_nm=0.0,
                left_in_contact=True,
                right_in_contact=True,
            )
        with self.assertRaises(TypeError):
            allocate_support_loads(
                self.config,
                desired_total_load_n=1.0,
                desired_roll_moment_nm=0.0,
                left_in_contact=1,  # type: ignore[arg-type]
                right_in_contact=True,
            )


if __name__ == "__main__":
    unittest.main()
