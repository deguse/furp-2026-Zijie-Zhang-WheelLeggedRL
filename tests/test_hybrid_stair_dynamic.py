from __future__ import annotations

import unittest

from hoppertrex_mjlab.hybrid.stair_classical import PHASE_COUNT, StairPhase
from hoppertrex_mjlab.hybrid.stair_dynamic import (
  DYNAMIC_MANEUVER_REQUIRED_BINDINGS,
  DYNAMIC_STAIR_CEM_ITERATIONS,
  DYNAMIC_STAIR_CEM_POPULATION,
  DYNAMIC_STAIR_CEM_REPLICATES,
  DYNAMIC_STAIR_FEEDFORWARD_LIMIT_RAD,
  DynamicLiftMode,
  DynamicStairManeuver,
  DynamicStairSensors,
  DynamicStairState,
  LeadSide,
  StairTraversalMode,
  choose_lead_side,
  dynamic_leg_feedforward,
  dynamic_maneuver_payload,
  dynamic_stair_step,
  half_cosine_bump,
  parse_dynamic_maneuver,
)


def _maneuver(**overrides: object) -> DynamicStairManeuver:
  values: dict[str, object] = {
    "lift_mode": DynamicLiftMode.ALTERNATING,
    "split_amplitude_rad": 0.05,
    "lift_amplitude_rad": 0.07,
    "trailing_delay_s": 0.0,
    "drive_feedforward_radps": 1.0,
    "approach_duration_s": 0.02,
    "preload_duration_s": 0.02,
    "lift_duration_s": 0.04,
    "recover_duration_s": 0.04,
    "contact_timeout_s": 0.2,
    "trail_contact_timeout_s": 0.1,
    "cross_timeout_s": 0.1,
  }
  values.update(overrides)
  return DynamicStairManeuver(**values)  # type: ignore[arg-type]


class DynamicStairPrimitiveTest(unittest.TestCase):
  def test_phase_aliases_preserve_v2_numbers(self) -> None:
    self.assertEqual(StairPhase.LEAD_LIFT, StairPhase.CLIMB)
    self.assertEqual(StairPhase.TRAIL_LIFT, StairPhase.CREST)
    self.assertEqual(PHASE_COUNT, 9)

  def test_half_cosine_has_zero_endpoints_and_unit_midpoint(self) -> None:
    self.assertEqual(half_cosine_bump(0.0, 0.6), 0.0)
    self.assertAlmostEqual(half_cosine_bump(0.3, 0.6), 1.0)
    self.assertEqual(half_cosine_bump(0.6, 0.6), 0.0)

  def test_simultaneous_contact_uses_force_then_preferred_side(self) -> None:
    self.assertEqual(
      choose_lead_side(
        left_loaded=True,
        right_loaded=True,
        left_force_n=20.0,
        right_force_n=19.0,
        preferred_side=LeadSide.RIGHT,
      ),
      LeadSide.LEFT,
    )
    self.assertEqual(
      choose_lead_side(
        left_loaded=True,
        right_loaded=True,
        left_force_n=20.0,
        right_force_n=20.0,
        preferred_side=LeadSide.RIGHT,
      ),
      LeadSide.RIGHT,
    )

  def test_request_false_is_exact_idle_zero_feedforward(self) -> None:
    maneuver = _maneuver()
    state = DynamicStairState(
      phase=StairPhase.LEAD_LIFT,
      phase_elapsed_s=0.02,
      lead_side=LeadSide.LEFT,
    )
    target, reset = dynamic_stair_step(
      maneuver,
      state,
      DynamicStairSensors(0.01, 40.0, 0.0),
      stair_request=False,
    )
    self.assertEqual(reset.phase, StairPhase.IDLE)
    self.assertEqual(target.leg_feedforward, (0.0, 0.0, 0.0, 0.0))
    self.assertEqual(target.drive_feedforward_radps, 0.0)
    self.assertEqual(target.phase_one_hot(), (1.0,) + (0.0,) * 8)

  def test_alternating_contact_lift_recover_and_next_step(self) -> None:
    maneuver = _maneuver()
    state = DynamicStairState()
    quiet = DynamicStairSensors(0.0, 0.0, 0.0)
    _target, state = dynamic_stair_step(
      maneuver, state, quiet, stair_request=True
    )
    self.assertEqual(state.phase, StairPhase.APPROACH)

    for _ in range(3):
      target, state = dynamic_stair_step(
        maneuver,
        state,
        DynamicStairSensors(0.0, 20.0, 0.0),
        stair_request=True,
      )
    self.assertEqual(state.phase, StairPhase.LEAD_LIFT)
    self.assertEqual(state.lead_side, LeadSide.LEFT)
    self.assertEqual(state.traversal_mode, StairTraversalMode.DYNAMIC)
    self.assertEqual(target.drive_feedforward_radps, 1.0)

    # The trailing sensor remains active during LEAD_LIFT.
    for _ in range(3):
      target, state = dynamic_stair_step(
        maneuver,
        state,
        DynamicStairSensors(0.0, 0.0, 20.0),
        stair_request=True,
      )
    self.assertEqual(state.phase, StairPhase.TRAIL_LIFT)
    self.assertTrue(state.right_loaded_contact)

    for _ in range(2):
      target, state = dynamic_stair_step(
        maneuver,
        state,
        DynamicStairSensors(0.21, 0.0, 0.0),
        stair_request=True,
      )
    self.assertEqual(state.phase, StairPhase.RECOVER)

    for _ in range(25):
      target, state = dynamic_stair_step(
        maneuver,
        state,
        DynamicStairSensors(0.0, 0.0, 0.0, stable=True),
        stair_request=True,
      )
    self.assertEqual(state.phase, StairPhase.APPROACH)
    self.assertEqual(state.step_index, 1)
    self.assertEqual(state.preferred_side, LeadSide.RIGHT)
    self.assertEqual(state.lead_side, LeadSide.NONE)

  def test_crossing_without_contact_is_roll(self) -> None:
    maneuver = _maneuver()
    _target, state = dynamic_stair_step(
      maneuver,
      DynamicStairState(),
      DynamicStairSensors(0.0, 0.0, 0.0),
      stair_request=True,
    )
    _target, state = dynamic_stair_step(
      maneuver,
      state,
      DynamicStairSensors(0.41, 0.0, 0.0),
      stair_request=True,
    )
    self.assertEqual(state.phase, StairPhase.RECOVER)
    self.assertEqual(state.traversal_mode, StairTraversalMode.ROLL)

  def test_abort_is_fail_closed(self) -> None:
    maneuver = _maneuver()
    target, state = dynamic_stair_step(
      maneuver,
      DynamicStairState(),
      DynamicStairSensors(0.0, 0.0, 0.0, non_wheel_contact=True),
      stair_request=True,
    )
    self.assertEqual(state.phase, StairPhase.ABORT)
    self.assertEqual(state.abort_reason, "non_wheel_contact")
    self.assertEqual(target.vx, 0.0)
    self.assertEqual(target.leg_feedforward, (0.0, 0.0, 0.0, 0.0))

  def test_composed_feedforward_is_clamped_to_point_zero_seven(self) -> None:
    maneuver = _maneuver(split_amplitude_rad=0.07)
    state = DynamicStairState(
      phase=StairPhase.LEAD_LIFT,
      phase_elapsed_s=0.02,
      preferred_side=LeadSide.LEFT,
      lead_side=LeadSide.LEFT,
    )
    feedforward = dynamic_leg_feedforward(maneuver, state)
    self.assertLessEqual(
      max(abs(value) for value in feedforward),
      DYNAMIC_STAIR_FEEDFORWARD_LIMIT_RAD,
    )

  def test_artifact_roundtrip_binds_hash_and_fixed_cem_protocol(self) -> None:
    maneuver = _maneuver(
      approach_duration_s=0.2,
      preload_duration_s=0.4,
      lift_duration_s=0.6,
      recover_duration_s=0.5,
      contact_timeout_s=5.0,
      trail_contact_timeout_s=1.5,
      cross_timeout_s=1.5,
    )
    payload = dynamic_maneuver_payload(
      maneuver,
      bindings={
        name: ("b" * 40 if name == "git_sha" else "a" * 64)
        for name in DYNAMIC_MANEUVER_REQUIRED_BINDINGS
      },
    )
    parsed = parse_dynamic_maneuver(payload)
    self.assertEqual(parsed.lift_mode, DynamicLiftMode.ALTERNATING)
    self.assertEqual(parsed.maneuver_hash, payload["maneuver_hash"])
    self.assertEqual(DYNAMIC_STAIR_CEM_POPULATION, 32)
    self.assertEqual(DYNAMIC_STAIR_CEM_ITERATIONS, 5)
    self.assertEqual(DYNAMIC_STAIR_CEM_REPLICATES, 8)
    mutated = dict(payload)
    mutated["maneuver_hash"] = "0" * 64
    with self.assertRaises(ValueError):
      parse_dynamic_maneuver(mutated)


if __name__ == "__main__":
  unittest.main()
