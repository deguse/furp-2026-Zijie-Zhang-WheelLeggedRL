import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
WHEEL_NAME = "mujoco-3.8.1.dev913242127-cp311-cp311-win_amd64.whl"
WHEEL = ROOT / "vendor" / "wheels" / WHEEL_NAME
MANIFEST = ROOT / "vendor" / "wheels" / "manifest.json"
EXPECTED_SHA256 = (
  "bf3d17128e6b37c706c33bef43a86741fea4ccd7015d083dabd3005607b9f3ba"
)


class RuntimeWheelArtifactTest(unittest.TestCase):
  def test_manifest_matches_frozen_wheel_bytes(self):
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    self.assertEqual(payload["artifact"], WHEEL_NAME)
    self.assertEqual(payload["file_sha256"], EXPECTED_SHA256)
    self.assertEqual(payload["file_size_bytes"], WHEEL.stat().st_size)
    self.assertEqual(hashlib.sha256(WHEEL.read_bytes()).hexdigest(), EXPECTED_SHA256)

  def test_uv_configuration_uses_local_frozen_wheel(self):
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    self.assertIn(f'mujoco = {{ path = "vendor/wheels/{WHEEL_NAME}" }}', pyproject)
    self.assertIn('override-dependencies = ["mujoco==3.8.1.dev913242127"]', pyproject)
    self.assertIn('version = "3.8.1.dev913242127"', lock)
    self.assertIn(f'source = {{ path = "vendor/wheels/{WHEEL_NAME}" }}', lock)
    self.assertIn("vendor/wheels/*.whl -text -diff", attributes)


if __name__ == "__main__":
  unittest.main()
