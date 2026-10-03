from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hoppertrex_mjlab.scripts.probe_r0c_support_transfer_shadow import parse_args


class SupportTransferShadowProbeTest(unittest.TestCase):
    def test_parser_is_cpu_only_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            output = Path(directory) / "output.json"
            source.write_text("{}", encoding="utf-8")
            args = parse_args(
                [
                    "--source-result",
                    str(source),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(args.device, "cpu")
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
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--source-result",
                        str(source),
                        "--output",
                        str(output) + ".new",
                        "--device",
                        "cuda:0",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
