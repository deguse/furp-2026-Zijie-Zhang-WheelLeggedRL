from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hoppertrex_mjlab.scripts.probe_r0c_support_load_feasibility import (
    _force_interval,
    parse_args,
)


class SupportLoadFeasibilityProbeTest(unittest.TestCase):
    def test_parser_is_cpu_only_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            output = Path(directory) / "output.json"
            source.write_text("{}", encoding="utf-8")
            self.assertEqual(
                parse_args(
                    ["--source-result", str(source), "--output", str(output)]
                ).device,
                "cpu",
            )
            output.write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit):
                parse_args(
                    ["--source-result", str(source), "--output", str(output)]
                )

    def test_force_interval_intersects_all_torque_limits(self):
        bias = np.asarray([5.0, -2.0, 0.0])
        jacobian = np.asarray([0.1, -0.2, 0.0])
        lower, upper = _force_interval(
            bias,
            jacobian,
            (0, 1, 2),
            (10.0, 10.0, 1.0),
        )
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 60.0)

    def test_force_interval_fails_when_zero_jacobian_bias_exceeds_limit(self):
        lower, upper = _force_interval(
            np.asarray([2.0]),
            np.asarray([0.0]),
            (0,),
            (1.0,),
        )
        self.assertTrue(math.isinf(lower))
        self.assertLess(upper, lower)


if __name__ == "__main__":
    unittest.main()

