from __future__ import annotations

import inspect
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from mjlab.envs import ManagerBasedRlEnv

from hoppertrex_mjlab.scripts import probe_hybrid_c2_paired_capture_v1 as probe
from hoppertrex_mjlab.scripts import probe_hybrid_stall_causal_v2 as causal
from hoppertrex_mjlab.scripts import (
  fit_hybrid_stair_contact_detector as fitter,
)
from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
  make_hoppertrex_hybrid_env_cfg,
)


class CaptureProvenanceContractTest(unittest.TestCase):
  """Locks the payload keys the machine-room wrapper hard-fails without.

  Regression for the 2026-07-26 finding: the probe once spread
  ``hybrid_provenance_lines(env)`` (a ``list[str]``) with ``**`` into the
  payload, which would TypeError after the full GPU capture and leave the
  wrapper with none of the provenance keys it validates.
  """

  WRAPPER_CONSUMED_KEYS = (
    "git_sha",
    "mjlab_git_sha",
    "calibration_hash",
    "posture_artifact_hash",
    "station_calibration_hash",
  )

  def _fake_cfg(self) -> SimpleNamespace:
    action_cfg = SimpleNamespace(
      controller_gain_hash="a" * 64,
      calibration_hash="b" * 64,
      posture_artifact_hash="c" * 64,
      station_calibration_hash="d" * 64,
    )
    return SimpleNamespace(actions={"hybrid_wheel_leg": action_cfg})

  def test_provenance_is_mapping_with_wrapper_keys(self) -> None:
    with (
      mock.patch.object(probe.stair, "_git_sha", side_effect=["0" * 40, "1" * 40]),
      mock.patch.object(
        probe.stair, "_runtime_metadata", return_value={"python": "x"}
      ),
    ):
      provenance = probe._capture_provenance(self._fake_cfg(), "cpu")
    self.assertIsInstance(provenance, dict)
    for key in self.WRAPPER_CONSUMED_KEYS:
      self.assertIn(key, provenance)
    merged = {"schema_version": 1, **provenance}
    self.assertEqual(merged["git_sha"], "0" * 40)
    self.assertEqual(merged["mjlab_git_sha"], "1" * 40)
    self.assertEqual(merged["calibration_hash"], "b" * 64)
    self.assertEqual(merged["posture_artifact_hash"], "c" * 64)
    self.assertEqual(merged["station_calibration_hash"], "d" * 64)

  def test_provenance_lines_is_not_a_mapping(self) -> None:
    lines = probe.hybrid_provenance_lines(self._fake_cfg())
    self.assertIsInstance(lines, list)

  def test_classifications_match_wrapper_allowed_set(self) -> None:
    self.assertEqual(
      probe.CLASSIFICATIONS, ("ANALYSIS_READY", "INVALID_CAPTURE")
    )

  def test_task_identity_matches_wrapper_expectation(self) -> None:
    # The wrapper hard-fails unless payload task equals this registry id;
    # the payload value comes from the env cfg built via stair.TASK. The
    # 364e053 machine-room artifact proved the emitted value is Stage5.
    self.assertEqual(probe.stair.TASK, "HopperTrex-Hybrid-v2-Stage5")


class C2ScheduleStackBindingTest(unittest.TestCase):
  """Builds a stage cfg on the real frozen C2 artifact stack.

  Regression for the 2026-07-27 machine-room preflight crash: with a
  gain-scheduled controller artifact, companion-artifact binding checks
  compared against the schedule_hash instead of the schedule's registered
  identification_controller_gain_hash, so the registered C2 stack could
  never load.
  """

  ARTIFACTS = Path(__file__).resolve().parents[1] / (
    "docs/experiments/artifacts"
  )

  def test_stage5_cfg_builds_with_registered_c2_stack(self) -> None:
    cfg = make_hoppertrex_hybrid_env_cfg(
      stage=5,
      play=True,
      controller_path=self.ARTIFACTS
      / "c1_schedule_candidate24_1f54968_seed1/c1_schedule.json",
      calibration_path=self.ARTIFACTS
      / "hybrid_runtime_seed1/velocity_calibration_seed1.json",
      posture_map_path=self.ARTIFACTS
      / "c1_posture_requalification_seed1/posture_map_seed1_registered_p032.json",
      station_calibration_path=self.ARTIFACTS
      / "c1_posture_requalification_seed1/station_calibration_seed1.json",
    )
    action_cfg = cfg.actions["hybrid_wheel_leg"]
    self.assertIsNotNone(action_cfg.controller_schedule)
    self.assertEqual(
      action_cfg.controller_schedule.schedule_hash,
      "8fe8548bca85978c164bbd7de39d2d6463cdfd8d7ab91796cf57696b0f64e203",
    )
    self.assertEqual(
      action_cfg.calibration_hash,
      "f62648b57bd17a3503bcbdbf58f349f91fcd8de8ef0cf04551c200401233ed01",
    )
    self.assertEqual(
      action_cfg.posture_artifact_hash,
      "3b96fd3dae66ad781b5b875c74184db101c42da02c53dfcc40a5137a6b5de11a",
    )
    self.assertEqual(
      action_cfg.station_calibration_hash,
      "c00e859b3093b4812d54799253accdaeb99171a2cf4028b08bc39e68eaaa7d8a",
    )


class DetectorSeriesCaptureContractTest(unittest.TestCase):
  """Pins the capture/serialisation path repaired on 2026-07-29.

  The original code was unrunnable end to end: it sampled a non-existent
  ``robot.data.root_quat_w``, never recorded body_vx or wheel_speed at all
  (it read live post-rollout ``robot.data`` instead of the series), indexed
  the ``[step, env]`` sample stacks as ``[env, step]``, projected the wheel
  pair with a plain mean (which cancels the opposite-signed drive channel),
  and subscripted ``ContactData``. Each of those would have burned a full
  machine-room GPU session.
  """

  def _stacked(self, steps: int, envs: int) -> dict[str, torch.Tensor]:
    # value = step * 100 + env makes any axis transposition unmistakable.
    base = torch.arange(steps, dtype=torch.float32)[:, None] * 100.0
    base = base + torch.arange(envs, dtype=torch.float32)[None, :]
    return {
      field: base + offset
      for offset, field in enumerate(probe.DETECTOR_SERIES_FIELDS)
    }

  def test_series_fields_are_exactly_what_the_fitter_consumes(self) -> None:
    consumed = {
      "pitch_rad",
      "body_vx_mps",
      "wheel_speed_radps",
      "wheel_target_radps",
    }
    self.assertEqual(set(probe.DETECTOR_SERIES_FIELDS), consumed)
    # The fitter is the real consumer: feed it exactly our field set.
    flat = {field: [0.0, 0.0, 0.0] for field in probe.DETECTOR_SERIES_FIELDS}
    self.assertEqual(len(fitter._sequence(flat)), 3)

  def test_extract_series_slices_step_axis_not_env_axis(self) -> None:
    samples = self._stacked(steps=12, envs=4)
    series = probe.extract_detector_series(
      None, samples, env_id=3, start_index=5, count=4
    )
    self.assertEqual(sorted(series), sorted(probe.DETECTOR_SERIES_FIELDS))
    # env 3, steps 5..8 -> 503, 603, 703, 803 (+ per-field offset)
    for offset, field in enumerate(probe.DETECTOR_SERIES_FIELDS):
      self.assertEqual(
        series[field],
        [503.0 + offset, 603.0 + offset, 703.0 + offset, 803.0 + offset],
        msg=f"{field} sliced on the wrong axis",
      )

  def test_paired_capture_anchors_the_window_on_impact(self) -> None:
    samples = self._stacked(steps=40, envs=4)
    protocol = {"pre_impact_steps": 3, "post_impact_steps": 2}
    capture = probe.make_paired_capture(
      None,
      samples,
      slot=0,
      flat_env_id=1,
      stair_env_id=2,
      impact_step=10,
      protocol=protocol,
    )
    self.assertTrue(capture["valid"])
    self.assertIsNone(capture["invalid_reason"])
    self.assertEqual(capture["impact_step"], 10)
    flat = capture["aligned_series"]["flat"]["pitch_rad"]
    stair = capture["aligned_series"]["stair"]["pitch_rad"]
    self.assertEqual(len(flat), 6)
    # window starts at impact-3 = step 7, envs 1 and 2 respectively
    self.assertEqual(flat[0], 701.0)
    self.assertEqual(stair[0], 702.0)
    self.assertEqual(flat[3], 1001.0)

  def test_window_that_does_not_fit_is_marked_invalid_not_clamped(
    self,
  ) -> None:
    """A clamped or truncated window is worse than a lost capture.

    The fitter takes ``protocol.pre_impact_steps`` as the impact index for
    every capture, so silently shifting the window feeds post-impact samples
    in as the pre-impact baseline and measures latency from the wrong tick --
    and still satisfies every wrapper check.
    """

    samples = self._stacked(steps=40, envs=4)
    protocol = {"pre_impact_steps": 25, "post_impact_steps": 75}

    too_early = probe.make_paired_capture(
      None, samples, slot=0, flat_env_id=1, stair_env_id=2,
      impact_step=5, protocol=protocol,
    )
    self.assertFalse(too_early["valid"])
    self.assertEqual(
      too_early["invalid_reason"], "impact_lacks_pre_impact_history"
    )
    self.assertIsNone(too_early["aligned_series"])

    too_late = probe.make_paired_capture(
      None, samples, slot=0, flat_env_id=1, stair_env_id=2,
      impact_step=30, protocol={"pre_impact_steps": 3, "post_impact_steps": 20},
    )
    self.assertFalse(too_late["valid"])
    self.assertEqual(
      too_late["invalid_reason"], "impact_lacks_post_impact_history"
    )

  def test_non_finite_series_is_rejected_before_serialisation(self) -> None:
    samples = self._stacked(steps=40, envs=4)
    samples["body_vx_mps"][12, 2] = float("nan")
    capture = probe.make_paired_capture(
      None, samples, slot=0, flat_env_id=1, stair_env_id=2,
      impact_step=10, protocol={"pre_impact_steps": 3, "post_impact_steps": 2},
    )
    self.assertFalse(capture["valid"])
    self.assertEqual(capture["invalid_reason"], "non_finite_series")

  def test_riser_criterion_delegates_to_the_c0_implementation(self) -> None:
    sentinel = torch.zeros(1, 1, 2, dtype=torch.bool)
    with mock.patch.object(
      causal, "riser_contact_mask", return_value=sentinel
    ) as delegate:
      result = probe.riser_contact_mask(
        found=torch.zeros(1, 1, 2),
        force_contact_frame=torch.zeros(1, 1, 2, 3),
        pos_global=torch.zeros(1, 1, 2, 3),
        normal_global=torch.zeros(1, 1, 2, 3),
        outer_face_x=torch.zeros(1),
      )
    delegate.assert_called_once()
    self.assertIs(result, sentinel)
    for name in (
      "RISER_MIN_ABS_NORMAL_X",
      "RISER_FACE_X_TOLERANCE_M",
      "RISER_MIN_NORMAL_FORCE_N",
    ):
      self.assertEqual(
        getattr(probe, name), getattr(causal, name), msg=f"{name} drifted"
      )
    # float `found` (as the sensor reports it) must not blow up a bitwise and
    found = torch.tensor([[[1.0, 0.0]]])
    force = torch.zeros(1, 1, 2, 3)
    force[..., 0] = 5.0
    pos = torch.zeros(1, 1, 2, 3)
    normal = torch.zeros(1, 1, 2, 3)
    normal[..., 0] = 1.0
    mask = probe.riser_contact_mask(
      found=found,
      force_contact_frame=force,
      pos_global=pos,
      normal_global=normal,
      outer_face_x=torch.zeros(1),
    )
    self.assertEqual(mask.dtype, torch.bool)
    self.assertTrue(bool(mask[0, 0, 0]))
    self.assertFalse(bool(mask[0, 0, 1]))

  def test_first_impact_step_reads_step_major_history(self) -> None:
    steps, envs, slots = 6, 2, 1
    found = torch.zeros(steps, envs, slots)
    found[4, 1, 0] = 1.0  # stair env hits the riser at step 4
    force = torch.zeros(steps, envs, slots, 3)
    force[..., 0] = 5.0
    pos = torch.zeros(steps, envs, slots, 3)
    normal = torch.zeros(steps, envs, slots, 3)
    normal[..., 0] = 1.0
    history = {"found": found, "force": force, "pos": pos, "normal": normal}
    first = probe.first_riser_impact_step(
      history,
      stair_env_ids=torch.tensor([1]),
      outer_face_x=torch.zeros(envs),
    )
    self.assertEqual(first.tolist(), [4])

  def test_first_impact_step_with_many_stair_envs_and_per_env_faces(
    self,
  ) -> None:
    """Regression for the shape trap a single stair env hides.

    The C0 riser criterion is written for one instant (``[env, slot]`` with
    ``outer_face_x`` of shape ``(env,)``). Handing it a time axis broadcasts
    the face position against the step axis; with exactly one stair env that
    is silently harmless, so the official 16-envs-per-height run would have
    crashed after the full GPU rollout even though the CPU smoke passed.
    Steps and env count are deliberately different so a transposed or
    mis-broadcast axis cannot coincidentally line up.
    """

    steps, envs, slots = 7, 6, 2
    stair_env_ids = torch.tensor([1, 3, 5])
    impacts = {1: 2, 3: 5, 5: 4}
    # Each env sits at a different face position; only contacts near that
    # env's own face may qualify.
    outer_face_x = torch.tensor([0.0, 10.0, 0.0, 20.0, 0.0, 30.0])

    found = torch.zeros(steps, envs, slots)
    force = torch.zeros(steps, envs, slots, 3)
    force[..., 0] = 5.0
    normal = torch.zeros(steps, envs, slots, 3)
    normal[..., 0] = 1.0
    pos = torch.zeros(steps, envs, slots, 3)
    # Park every contact far from every face, then place the real impacts.
    pos[..., 0] = -500.0
    for env_id, step in impacts.items():
      found[step:, env_id, 0] = 1.0
      pos[step:, env_id, 0, 0] = float(outer_face_x[env_id])

    history = {"found": found, "force": force, "pos": pos, "normal": normal}
    first = probe.first_riser_impact_step(
      history,
      stair_env_ids=stair_env_ids,
      outer_face_x=outer_face_x,
    )
    self.assertEqual(first.tolist(), [2, 5, 4])

  def test_riser_mask_over_time_matches_per_step_application(self) -> None:
    steps, envs, slots = 5, 4, 3
    torch.manual_seed(0)
    found = (torch.rand(envs, steps, slots) > 0.5).float()
    force = torch.rand(envs, steps, slots, 3) * 10.0
    normal = torch.rand(envs, steps, slots, 3)
    outer_face_x = torch.linspace(-0.5, 0.5, envs)
    # Bias positions onto each env's OWN face so a mis-ordered face vector
    # changes the answer; a uniformly-random field would leave the mask all
    # False and the comparison vacuous.
    pos = torch.rand(envs, steps, slots, 3) * 0.01
    pos[..., 0] = pos[..., 0] + outer_face_x[:, None, None]

    batched = probe.riser_contact_mask_over_time(
      found=found,
      force_contact_frame=force,
      pos_global=pos,
      normal_global=normal,
      outer_face_x=outer_face_x,
    )
    self.assertGreater(
      int(batched.sum()), 0, "fixture produced no qualifying contacts"
    )
    self.assertLess(
      int(batched.sum()),
      batched.numel(),
      "fixture qualifies everything, so ordering errors would not show",
    )
    for step in range(steps):
      per_step = causal.riser_contact_mask(
        found=found[:, step],
        force_contact_frame=force[:, step],
        pos_global=pos[:, step],
        normal_global=normal[:, step],
        outer_face_x=outer_face_x,
      )
      self.assertTrue(
        torch.equal(batched[:, step], per_step),
        msg=f"batched riser mask diverges from the C0 criterion at {step}",
      )
    # A tiled face vector (env index varying fastest) must NOT agree: that is
    # the ordering mistake `repeat_interleave` exists to avoid.
    tiled = probe.riser_contact_mask(
      found=found.reshape(envs * steps, slots),
      force_contact_frame=force.reshape(envs * steps, slots, 3),
      pos_global=pos.reshape(envs * steps, slots, 3),
      normal_global=normal.reshape(envs * steps, slots, 3),
      outer_face_x=outer_face_x.repeat(steps),
    ).reshape(envs, steps, slots)
    self.assertFalse(
      torch.equal(batched, tiled),
      "fixture cannot distinguish repeat_interleave from repeat",
    )

  def test_task_identity_is_the_registered_stage5_id(self) -> None:
    """Pins the emitted payload key, not just the module constant.

    ``str(cfg.name)`` used to produce this field and raised AttributeError
    after the whole rollout; asserting only ``stair.TASK`` would have passed
    against that broken code.
    """

    source = inspect.getsource(probe.main)
    self.assertIn('"task": stair.TASK', source)
    self.assertNotIn("cfg.name", source)
    self.assertEqual(probe.stair.TASK, "HopperTrex-Hybrid-v2-Stage5")

  def test_settle_phase_holds_zero_command_velocity(self) -> None:
    """Settling at the command velocity strikes the riser before recording.

    200 settle steps x 0.07 m/s / 50 Hz = 0.28 m, past the 0.25 m start
    offset, so the impact lands in the unrecorded settle and the aligned
    window silently holds post-impact samples as its pre-impact baseline --
    while still passing every wrapper check. The registered C0 v2 producer
    settles at zero.
    """

    source = inspect.getsource(probe.run_cell)
    settle_call = re.search(
      r"range\(int\(protocol\[.settle_steps.\]\)\):\s*\n\s*_step\(([^,]+),",
      source,
    )
    self.assertIsNotNone(settle_call, "settle loop not found")
    self.assertEqual(settle_call.group(1).strip(), "0.0")
    causal_settle = re.search(
      r"range\(int\(protocol\[.settle_steps.\]\)\):\s*\n\s*_step\(([^,]+),",
      inspect.getsource(causal.run_cell),
    )
    self.assertIsNotNone(causal_settle, "C0 settle loop not found")
    self.assertEqual(
      settle_call.group(1).strip(),
      causal_settle.group(1).strip(),
      "C2 settle command drifted from the registered C0 v2 producer",
    )


class C2ProbePreflightRealTermTest(unittest.TestCase):
  """No-mock-term regression for the 2026-07-29 machine-room preflight crash.

  The probe read ``action_term.controller_schedule_hash``, an attribute the
  real HybridWheelLegAction term never exposes (the schedule lives on
  ``term.cfg.controller_schedule``). Mock-based tests cannot catch an
  invented attribute name, so this builds the REAL causal env on the frozen
  C2 artifact stack (CPU, 2 envs) and runs the probe's preflight resolver
  against the real term surface. Only the registry loader is substituted,
  because task registration happens at import time and would otherwise pin
  whatever artifacts the environment had when the module was first imported.
  """

  ARTIFACTS = Path(__file__).resolve().parents[1] / (
    "docs/experiments/artifacts"
  )

  def _registered_c2_stack_cfg(self, task_name: str, play: bool = False):
    del task_name, play
    return make_hoppertrex_hybrid_env_cfg(
      stage=5,
      play=True,
      controller_path=self.ARTIFACTS
      / "c1_schedule_candidate24_1f54968_seed1/c1_schedule.json",
      calibration_path=self.ARTIFACTS
      / "hybrid_runtime_seed1/velocity_calibration_seed1.json",
      posture_map_path=self.ARTIFACTS
      / "c1_posture_requalification_seed1/posture_map_seed1_registered_p032.json",
      station_calibration_path=self.ARTIFACTS
      / "c1_posture_requalification_seed1/station_calibration_seed1.json",
    )

  def test_preflight_resolves_schedule_hash_on_real_term(self) -> None:
    with mock.patch.object(
      probe.stair, "load_env_cfg", self._registered_c2_stack_cfg
    ):
      cfg = probe.make_causal_env_cfg(probe.DIAGNOSTIC_HEIGHTS_M, 1)
    cfg.seed = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
      action_term = env.action_manager.get_term("hybrid_wheel_leg")
      # The invented attribute must stay absent, otherwise this regression
      # would silently stop testing anything.
      self.assertFalse(hasattr(action_term, "controller_schedule_hash"))
      self.assertEqual(
        probe._require_schedule_hash(action_term),
        probe.C1_SCHEDULE_HASH,
      )
    finally:
      env.close()

  def test_preflight_rejects_a_fixed_gain_stack(self) -> None:
    fixed_gain_cfg = make_hoppertrex_hybrid_env_cfg(
      stage=5,
      play=True,
      controller_path=self.ARTIFACTS
      / "hybrid_runtime_seed1/controller_seed1.json",
      calibration_path=self.ARTIFACTS
      / "hybrid_runtime_seed1/velocity_calibration_seed1.json",
      posture_map_path=self.ARTIFACTS
      / "c1_posture_requalification_seed1/posture_map_seed1_registered_p032.json",
      station_calibration_path=self.ARTIFACTS
      / "c1_posture_requalification_seed1/station_calibration_seed1.json",
    )
    action_term = SimpleNamespace(
      cfg=fixed_gain_cfg.actions["hybrid_wheel_leg"]
    )
    with self.assertRaisesRegex(RuntimeError, "controller_schedule_hash"):
      probe._require_schedule_hash(action_term)


if __name__ == "__main__":
  unittest.main()
