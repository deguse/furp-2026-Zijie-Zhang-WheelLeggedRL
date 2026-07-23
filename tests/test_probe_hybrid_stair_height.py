import math
from types import SimpleNamespace
import unittest

import torch

from mjlab.envs import ManagerBasedRlEnv

from hoppertrex_mjlab.scripts.probe_hybrid_stair_height import (
  CLASSIFICATIONS,
  HEIGHTS_M,
  POSTURE_CARDS,
  RESET_PITCH_RATE_JITTER_RADPS,
  RESET_VX_JITTER_MPS,
  RESET_X_JITTER_M,
  RESET_Y_JITTER_M,
  STEP_WIDTH_M,
  aggregate_trials,
  approach_geometry,
  build_payload,
  classify_results,
  make_stair_env_cfg,
  merge_contact_observations,
  parse_args,
  protocol_for_mode,
  reset_perturbations,
  run_card_repeat,
  update_contact_history,
  update_valid_max_progress,
)
from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
  make_hoppertrex_hybrid_env_cfg,
)


def _trials(flags_by_card, *, repeats=3, envs=16):
  rows = []
  for card in POSTURE_CARDS:
    card_name = card["name"]
    for height in HEIGHTS_M:
      passed = flags_by_card[card_name][height]
      successes = envs if passed else 0
      for repeat in range(1, repeats + 1):
        for env_id in range(envs):
          rows.append({
            "posture_card": card_name,
            "stair_height_m": height,
            "repeat": repeat,
            "env_id": env_id,
            "success": env_id < successes,
            "terminated": False,
            "non_wheel_contact": False,
          })
  return rows


def _monotonic_flags(first_failure):
  return {
    card["name"]: {
      height: first_failure is None or height < first_failure
      for height in HEIGHTS_M
    }
    for card in POSTURE_CARDS
  }


class ProtocolTest(unittest.TestCase):
  def test_official_protocol_is_frozen(self):
    protocol = protocol_for_mode(False)
    self.assertEqual(HEIGHTS_M, tuple(index / 100 for index in range(11)))
    self.assertEqual(protocol["envs_per_height"], 16)
    self.assertEqual(protocol["repeats"], 3)
    self.assertEqual(protocol["settle_steps"], 100)
    self.assertEqual(protocol["drive_steps"], 500)
    self.assertEqual(protocol["stable_steps"], 25)
    self.assertTrue(protocol["evidence_eligible"])
    self.assertEqual(STEP_WIDTH_M, 0.30)
    self.assertEqual(make_stair_env_cfg((0.0,), 1).seed, 1)
    self.assertEqual(
      POSTURE_CARDS,
      (
        {
          "name": "envelope_center",
          "height_m": 0.3092089487,
          "pitch_rad": 0.016,
        },
        {
          "name": "high_zero_pitch",
          "height_m": 0.3276857266,
          "pitch_rad": 0.0,
        },
      ),
    )

  def test_cli_separates_smoke_from_official_evidence(self):
    official = parse_args(["--output", "result.json"])
    self.assertEqual(official.device, "cuda:0")
    self.assertFalse(official.smoke)
    smoke = parse_args([
      "--output", "smoke.json", "--device", "cpu", "--smoke"
    ])
    self.assertTrue(smoke.smoke)
    self.assertFalse(protocol_for_mode(True)["evidence_eligible"])
    with self.assertRaises(SystemExit):
      parse_args(["--output", "bad.json", "--device", "cpu"])

  def test_reset_geometry_starts_outside_and_crosses_inside(self):
    geometry = approach_geometry(12.0)
    self.assertLess(geometry["start_x"], geometry["outer_face_x"])
    self.assertGreater(geometry["cross_x"], geometry["outer_face_x"])
    self.assertAlmostEqual(
      geometry["cross_x"] - geometry["outer_face_x"], 0.15
    )

  def test_reset_perturbations_are_seeded_distinct_and_bounded(self):
    first = reset_perturbations(
      slots=16, card_name=POSTURE_CARDS[0]["name"], repeat=1
    )
    repeated = reset_perturbations(
      slots=16, card_name=POSTURE_CARDS[0]["name"], repeat=1
    )
    next_repeat = reset_perturbations(
      slots=16, card_name=POSTURE_CARDS[0]["name"], repeat=2
    )
    self.assertTrue(torch.equal(first, repeated))
    self.assertFalse(torch.equal(first, next_repeat))
    self.assertEqual(len(torch.unique(first, dim=0)), 16)
    limits = (
      RESET_X_JITTER_M,
      RESET_Y_JITTER_M,
      RESET_VX_JITTER_MPS,
      RESET_PITCH_RATE_JITTER_RADPS,
    )
    for index, limit in enumerate(limits):
      self.assertLessEqual(float(first[:, index].abs().max()), limit)

  def test_terminal_contact_and_progress_use_pre_reset_state(self):
    direct_after_reset = torch.tensor([False, False])
    terminal_contact = torch.tensor([True, False])
    self.assertEqual(
      merge_contact_observations(
        direct_after_reset, terminal_contact
      ).tolist(),
      [True, False],
    )
    self.assertEqual(
      update_contact_history(
        torch.tensor([False, False]),
        torch.tensor([True, True]),
        torch.tensor([True, False]),
      ).tolist(),
      [True, False],
    )
    maximum = torch.tensor([0.1, 0.2])
    post_step_progress = torch.tensor([3.0, 0.3])
    valid = torch.tensor([False, True])
    self.assertEqual(
      update_valid_max_progress(maximum, post_step_progress, valid).tolist(),
      torch.tensor([0.1, 0.3]).tolist(),
    )


class AggregationAndClassificationTest(unittest.TestCase):
  def _verdict(self, flags):
    cells, repeat_cells = aggregate_trials(_trials(flags), heights=HEIGHTS_M)
    return classify_results(cells, repeat_cells, heights=HEIGHTS_M)

  def test_classical_death_height_is_bracketed(self):
    verdict = self._verdict(_monotonic_flags(0.05))
    self.assertEqual(
      verdict["classification"], "CLASSICAL_DEATH_HEIGHT_BRACKETED"
    )
    self.assertEqual(verdict["p3_candidate_height_m"], 0.05)

  def test_extends_when_both_cards_pass_maximum(self):
    verdict = self._verdict(_monotonic_flags(None))
    self.assertEqual(
      verdict["classification"], "EXTEND_SWEEP_BEFORE_P3"
    )
    self.assertIsNone(verdict["p3_candidate_height_m"])

  def test_extends_when_only_one_card_reaches_sweep_ceiling(self):
    flags = _monotonic_flags(0.05)
    flags[POSTURE_CARDS[1]["name"]] = {
      height: True for height in HEIGHTS_M
    }
    verdict = self._verdict(flags)
    self.assertEqual(verdict["classification"], "EXTEND_SWEEP_BEFORE_P3")
    self.assertFalse(verdict["mixed_repeat"])
    self.assertFalse(verdict["non_monotonic"])
    self.assertIsNone(verdict["p3_candidate_height_m"])

  def test_non_monotonic_results_stop_for_analysis(self):
    flags = _monotonic_flags(0.05)
    flags[POSTURE_CARDS[0]["name"]][0.06] = True
    verdict = self._verdict(flags)
    self.assertEqual(
      verdict["classification"], "STOP_FOR_VARIANCE_ANALYSIS"
    )
    self.assertTrue(verdict["non_monotonic"])

  def test_mixed_repeats_stop_for_analysis(self):
    flags = _monotonic_flags(0.05)
    rows = _trials(flags)
    for row in rows:
      if (
        row["posture_card"] == POSTURE_CARDS[0]["name"]
        and math.isclose(row["stair_height_m"], 0.04)
        and row["repeat"] == 3
      ):
        row["success"] = False
    cells, repeat_cells = aggregate_trials(rows, heights=HEIGHTS_M)
    verdict = classify_results(cells, repeat_cells, heights=HEIGHTS_M)
    self.assertEqual(
      verdict["classification"], "STOP_FOR_VARIANCE_ANALYSIS"
    )
    self.assertTrue(verdict["mixed_repeat"])

  def test_invalid_flat_control_has_priority(self):
    flags = _monotonic_flags(0.05)
    flags[POSTURE_CARDS[1]["name"]][0.0] = False
    verdict = self._verdict(flags)
    self.assertEqual(
      verdict["classification"], "INVALID_FLAT_CONTROL_STOP"
    )

  def test_all_four_official_classifications_are_reachable(self):
    observed = {
      self._verdict(_monotonic_flags(0.05))["classification"],
      self._verdict(_monotonic_flags(None))["classification"],
    }
    flags = _monotonic_flags(0.05)
    flags[POSTURE_CARDS[0]["name"]][0.06] = True
    observed.add(self._verdict(flags)["classification"])
    flags = _monotonic_flags(0.05)
    flags[POSTURE_CARDS[0]["name"]][0.0] = False
    observed.add(self._verdict(flags)["classification"])
    self.assertEqual(observed, set(CLASSIFICATIONS))

  def test_official_aggregation_rejects_missing_or_duplicate_trials(self):
    rows = _trials(_monotonic_flags(0.05))
    with self.assertRaisesRegex(ValueError, "expected 48"):
      aggregate_trials(
        rows[:-1],
        heights=HEIGHTS_M,
        expected_repeats=3,
        expected_envs_per_height=16,
      )
    duplicate = list(rows)
    duplicate[-1] = {**duplicate[-1], "env_id": duplicate[-2]["env_id"]}
    with self.assertRaisesRegex(ValueError, "duplicate env_ids"):
      aggregate_trials(
        duplicate,
        heights=HEIGHTS_M,
        expected_repeats=3,
        expected_envs_per_height=16,
      )


class ConfigurationIsolationTest(unittest.TestCase):
  def _flat_fingerprint(self):
    return tuple(
      (
        stage,
        make_hoppertrex_hybrid_env_cfg(stage=stage, play=True)
        .scene.terrain.terrain_type,
        tuple(
          make_hoppertrex_hybrid_env_cfg(stage=stage, play=True)
          .actions["hybrid_wheel_leg"].action_scales
        ),
      )
      for stage in range(6)
    )

  def test_stair_cfg_does_not_mutate_registered_flat_configs(self):
    before = self._flat_fingerprint()
    stair = make_stair_env_cfg((0.0, 0.01), 1)
    after = self._flat_fingerprint()
    self.assertEqual(before, after)
    self.assertEqual(stair.scene.terrain.terrain_type, "generator")
    self.assertEqual(stair.scene.num_envs, 2)


class ProbeSmokeTest(unittest.TestCase):
  def test_short_cpu_generated_terrain_rollout_is_not_evidence(self):
    protocol = protocol_for_mode(True)
    heights = tuple(protocol["heights_m"])
    cfg = make_stair_env_cfg(heights, protocol["envs_per_height"])
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
      rows = run_card_repeat(
        env,
        heights=heights,
        card=POSTURE_CARDS[0],
        repeat=1,
        settle_steps=protocol["settle_steps"],
        drive_steps=protocol["drive_steps"],
        stable_steps=protocol["stable_steps"],
      )
    finally:
      env.close()
    self.assertFalse(protocol["evidence_eligible"])
    self.assertEqual(cfg.seed, 1)
    self.assertEqual(len(rows), 1)
    self.assertEqual(rows[0]["stair_height_m"], 0.0)
    self.assertTrue(math.isfinite(rows[0]["max_progress_past_face_m"]))
    self.assertIn("root_reset", rows[0])


class PayloadTest(unittest.TestCase):
  def test_payload_pins_zero_residual_and_full_provenance(self):
    action_cfg = SimpleNamespace(
      controller_gain_hash="controller",
      calibration_hash="calibration",
      yaw_calibration_hash=None,
      posture_map_hash="posture-map",
      posture_artifact_hash="posture-artifact",
      station_calibration_hash="station",
      action_scales=(0.5, 0.3, 0.035, 0.035, 0.035, 0.035),
    )
    protocol = protocol_for_mode(True)
    payload = build_payload(
      trials=[],
      cells=[],
      repeat_cells=[],
      verdict=None,
      action_cfg=action_cfg,
      protocol=protocol,
      device="cpu",
      runtime_metadata={
        "device": "cpu",
        "cuda_available": False,
        "gpu_name": None,
        "driver_version": None,
        "torch_version": "test",
        "cuda_version": None,
      },
    )
    self.assertFalse(payload["evidence_eligible"])
    self.assertFalse(payload["promotion_eligible"])
    self.assertFalse(payload["training_eligible"])
    self.assertIsNone(payload["checkpoint"])
    self.assertIsNone(payload["checkpoint_file_sha256"])
    self.assertIsNone(payload["yaw_calibration_hash"])
    self.assertEqual(payload["runtime"]["device"], "cpu")
    self.assertEqual(payload["posture_artifact_hash"], "posture-artifact")
    self.assertEqual(payload["protocol"]["policy_action"], [0.0] * 6)
    self.assertEqual(payload["protocol"]["environment_seed"], 1)
    self.assertEqual(payload["protocol"]["settle_duration_s"], 0.04)
    self.assertEqual(payload["protocol"]["maximum_drive_duration_s"], 0.1)
    self.assertEqual(payload["protocol"]["required_stable_duration_s"], 0.04)
    self.assertEqual(
      payload["protocol"]["root_reset"]["start_offset_outside_m"], 0.25
    )
    self.assertEqual(
      payload["protocol"]["root_reset"]["success_line_inside_m"], 0.15
    )
    self.assertEqual(
      payload["protocol"]["root_reset"]["x_jitter_abs_m"], 0.02
    )


if __name__ == "__main__":
  unittest.main()
