from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hoppertrex_mjlab.scripts.probe_r0c_support_force_filter import parse_args


class SupportForceFilterProbeTest(unittest.TestCase):
    def test_parser_is_cpu_only_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            output = Path(directory) / "output.json"
            source.write_text("{}", encoding="utf-8")
            self.assertEqual(
                parse_args(
                    [
                        "--source-result",
                        str(source),
                        "--output",
                        str(output),
                    ]
                ).device,
                "cpu",
            )
            output.write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--source-result",
                        str(source),
                        "--output",
                        str(output),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
