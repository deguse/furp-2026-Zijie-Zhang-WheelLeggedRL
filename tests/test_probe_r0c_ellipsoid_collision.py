from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mujoco

from hoppertrex_mjlab.assets.HopperTrex_CFG import get_spec
from hoppertrex_mjlab.hybrid.wheel_collision_proxy import (
    WHEEL_COLLISION_GEOM_NAMES,
    make_ellipsoid_wheel_spec,
)
from hoppertrex_mjlab.scripts.probe_r0c_ellipsoid_collision import (
    _backend_qualifies,
    _compare_model_contract,
    _model_snapshot,
    parse_args,
)


class EllipsoidCollisionProbeTest(unittest.TestCase):
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

    def test_contract_accepts_only_the_two_expected_shape_changes(self):
        baseline = get_spec().compile()
        proxy = make_ellipsoid_wheel_spec().compile()
        snapshot = _model_snapshot(baseline, WHEEL_COLLISION_GEOM_NAMES)
        report = _compare_model_contract(
            snapshot, proxy, WHEEL_COLLISION_GEOM_NAMES
        )
        self.assertTrue(report["only_wheel_geom_type_and_size_changed"])
        self.assertEqual(len(report["changes"]), 2)

        geom_id = mujoco.mj_name2id(
            proxy,
            mujoco.mjtObj.mjOBJ_GEOM,
            WHEEL_COLLISION_GEOM_NAMES[0],
        )
        proxy.geom_friction[geom_id, 0] += 0.1
        with self.assertRaisesRegex(RuntimeError, "geom_friction"):
            _compare_model_contract(
                snapshot, proxy, WHEEL_COLLISION_GEOM_NAMES
            )

    def test_backend_qualification_requires_both_strict_cells(self):
        summaries = [
            {
                "stair_height_m": height,
                "trials": 8,
                "successes": 8,
                "geometric_successes_ignoring_support": 8,
                "bilateral_unsupported_physics_substeps": 0,
                "non_wheel_contact_trials": 0,
                "terminated_trials": 0,
            }
            for height in (0.0, 0.0025)
        ]
        self.assertTrue(_backend_qualifies(summaries))
        summaries[1]["successes"] = 7
        self.assertFalse(_backend_qualifies(summaries))


if __name__ == "__main__":
    unittest.main()
