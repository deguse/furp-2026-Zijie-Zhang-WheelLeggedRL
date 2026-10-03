from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mujoco

from hoppertrex_mjlab.scripts.probe_r0c_native_contact_replay import (
    _terrain_geom_ids,
    _validate_capture_counts,
    _wheel_geom_ids,
    parse_args,
)


class NativeContactReplayProbeTest(unittest.TestCase):
    def test_parser_is_cpu_only_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "source.json"
            out = Path(d) / "out.json"
            source.write_text("{}")
            self.assertEqual(
                parse_args(
                    ["--source-result", str(source), "--output", str(out)]
                ).device,
                "cpu",
            )
            out.write_text("{}")
            with self.assertRaises(SystemExit):
                parse_args(["--source-result", str(source), "--output", str(out)])

    def test_capture_counts_must_match_every_strict_zero_substep(self):
        captures = [[{"event": 0}, {"event": 1}], []]
        trials = {
            0: {"bilateral_unsupported_physics_substeps": 2},
            1: {"bilateral_unsupported_physics_substeps": 0},
        }
        _validate_capture_counts(captures, trials)

        captures[1].append({"event": 0})
        with self.assertRaisesRegex(RuntimeError, "capture-count mismatch"):
            _validate_capture_counts(captures, trials)

    def test_geom_resolution_matches_wheels_and_terrain_body_only(self):
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <worldbody>
                <body name="terrain">
                  <geom name="terrain_a" type="box" size="1 1 0.1"/>
                  <geom name="terrain_b" type="box" pos="3 0 0" size="1 1 0.1"/>
                </body>
                <body name="robot/left">
                  <geom name="robot/wheel_left_collision" type="sphere" size="0.1"/>
                </body>
                <body name="robot/right">
                  <geom name="robot/wheel_right_collision" type="sphere" size="0.1"/>
                </body>
                <body name="not_terrain">
                  <geom name="distractor" type="box" size="0.1 0.1 0.1"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        left, right = _wheel_geom_ids(model)
        self.assertEqual(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, left),
            "robot/wheel_left_collision",
        )
        self.assertEqual(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, right),
            "robot/wheel_right_collision",
        )
        terrain_names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            for geom_id in _terrain_geom_ids(model)
        }
        self.assertEqual(terrain_names, {"terrain_a", "terrain_b"})


if __name__ == "__main__":
    unittest.main()
