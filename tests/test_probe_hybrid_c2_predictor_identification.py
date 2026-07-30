import unittest

import numpy as np

from hoppertrex_mjlab.scripts import probe_hybrid_c2_predictor_identification as probe


class ProbeHybridC2PredictorIdentificationTest(unittest.TestCase):
  def test_shaped_posture_validation_rejects_wrong_tick_or_node(self):
    valid = np.tile([0.3092089487, 0.0], (4, 1))
    probe.validate_shaped_posture(valid, height=0.3092089487, pitch=0.0)
    with self.assertRaisesRegex(RuntimeError, "shaped posture drifted"):
      probe.validate_shaped_posture(
        valid + np.array([0.0, 1.0e-4]), height=0.3092089487, pitch=0.0
      )

  def test_official_protocol_is_frozen_and_non_evidence(self):
    protocol = probe.protocol(False, "cuda:0")
    self.assertEqual(protocol["num_envs"], 32)
    self.assertEqual(protocol["fit_envs"], list(range(24)))
    self.assertEqual(protocol["heldout_envs"], list(range(24, 32)))
    self.assertEqual(protocol["warmup_steps"], 250)
    self.assertEqual(protocol["collection_steps"], 2500)
    self.assertEqual(protocol["prbs"]["collection_stream_ticks"], [250, 2749])
    self.assertFalse(protocol["evidence_eligible"])
    self.assertFalse(protocol["detector_fit_eligible"])

  def test_env_major_flattening_does_not_mix_split_rows(self):
    values = np.empty((3, 4, 2))
    for tick in range(3):
      for env in range(4):
        values[tick, env] = [100 * env + tick, env]
    flattened = probe._env_major(values, [1, 3])
    np.testing.assert_array_equal(
      flattened[:, 0], [100, 101, 102, 300, 301, 302]
    )


if __name__ == "__main__":
  unittest.main()
