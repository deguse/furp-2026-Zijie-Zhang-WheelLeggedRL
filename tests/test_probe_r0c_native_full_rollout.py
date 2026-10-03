from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hoppertrex_mjlab.scripts.probe_r0c_native_full_rollout import (
    TrialAccumulator,
    _update_unsupported_run,
    parse_args,
)


class NativeFullRolloutProbeTest(unittest.TestCase):
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

    def test_unsupported_run_is_consecutive(self):
        current = maximum = 0
        observed = []
        for unsupported in (False, True, True, False, True):
            current, maximum = _update_unsupported_run(
                unsupported, current, maximum
            )
            observed.append((current, maximum))
        self.assertEqual(
            observed,
            [(0, 0), (1, 1), (2, 2), (0, 2), (1, 2)],
        )

    def test_accumulator_separates_physics_and_endpoint_samples(self):
        state = TrialAccumulator()
        state.observe_substep_support(
            left_force_n=0.0,
            right_force_n=0.0,
            phase="drive",
            control_step=7,
            physics_substep=2,
            time_s=0.14,
        )
        state.observe_endpoint_support(
            left_force_n=0.0,
            right_force_n=0.0,
            phase="drive",
            control_step=7,
            time_s=0.16,
        )
        self.assertTrue(state.support_failed)
        self.assertEqual(state.unsupported_physics_substeps, 1)
        self.assertEqual(state.unsupported_endpoint_samples, 1)
        self.assertEqual(state.unsupported_max_run, 1)
        self.assertEqual(
            [event["sample_kind"] for event in state.support_events],
            ["native_during_step", "native_integrated_endpoint"],
        )


if __name__ == "__main__":
    unittest.main()
