from __future__ import annotations

import unittest

from hoppertrex_mjlab.hybrid.stair_dynamic import LeadSide
from hoppertrex_mjlab.hybrid.support_transfer import (
    SupportTransferConfig,
    SupportTransferPhase,
    SupportTransferSensors,
    SupportTransferState,
    support_transfer_step,
)


def _config(**overrides: object) -> SupportTransferConfig:
    values: dict[str, object] = {
        "control_dt_s": 0.005,
        "support_confirm_steps": 2,
        "both_confirm_steps": 3,
        "minimum_guard_travel_m": 0.01,
        "lead_timeout_s": 0.025,
        "trail_timeout_s": 0.025,
        "completion_timeout_s": 0.050,
    }
    values.update(overrides)
    return SupportTransferConfig(**values)  # type: ignore[arg-type]


def _sensors(
    *,
    left_riser: float = 0.0,
    right_riser: float = 0.0,
    left_support: float = 80.0,
    right_support: float = 80.0,
    forward_increment: float = 0.0,
    unsafe: bool = False,
) -> SupportTransferSensors:
    return SupportTransferSensors(
        left_riser_force_n=left_riser,
        right_riser_force_n=right_riser,
        left_vertical_force_n=left_support,
        right_vertical_force_n=right_support,
        forward_increment_m=forward_increment,
        unsafe=unsafe,
    )


class SupportTransferTest(unittest.TestCase):
    def test_request_false_is_exactly_idle(self) -> None:
        config = _config()
        state = SupportTransferState(
            phase=SupportTransferPhase.TRAIL_TRANSFER,
            lead_side=LeadSide.LEFT,
            left_riser_seen=True,
        )
        target, state = support_transfer_step(
            config,
            state,
            _sensors(),
            stair_request=False,
        )
        self.assertEqual(state.phase, SupportTransferPhase.IDLE)
        self.assertFalse(target.active)
        self.assertEqual(target.hold_support_side, LeadSide.NONE)
        self.assertEqual(target.compliant_side, LeadSide.NONE)

    def test_left_first_make_before_break_roles(self) -> None:
        config = _config()
        state = SupportTransferState()
        target, state = support_transfer_step(
            config,
            state,
            _sensors(left_riser=8.0),
            stair_request=True,
        )
        self.assertEqual(state.phase, SupportTransferPhase.LEAD_GUARD)
        self.assertEqual(state.lead_side, LeadSide.LEFT)
        self.assertEqual(target.hold_support_side, LeadSide.RIGHT)
        self.assertEqual(target.compliant_side, LeadSide.LEFT)

        # A loaded lead wheel is not promoted while the riser contact remains.
        for _ in range(3):
            target, state = support_transfer_step(
                config,
                state,
                _sensors(left_riser=4.0),
                stair_request=True,
            )
        self.assertEqual(state.phase, SupportTransferPhase.LEAD_GUARD)

        # Riser release plus persistent vertical support establishes the new side.
        for _ in range(2):
            target, state = support_transfer_step(
                config,
                state,
                _sensors(left_riser=0.0),
                stair_request=True,
            )
        self.assertEqual(state.phase, SupportTransferPhase.TRAIL_TRANSFER)
        self.assertEqual(target.hold_support_side, LeadSide.LEFT)
        self.assertEqual(target.compliant_side, LeadSide.RIGHT)

        # The trailing side must itself touch and re-establish vertical support.
        target, state = support_transfer_step(
            config,
            state,
            _sensors(right_riser=8.0),
            stair_request=True,
        )
        self.assertEqual(state.phase, SupportTransferPhase.TRAIL_TRANSFER)
        for _ in range(2):
            target, state = support_transfer_step(
                config,
                state,
                _sensors(right_riser=0.0),
                stair_request=True,
            )
        self.assertEqual(state.phase, SupportTransferPhase.BOTH_SUPPORTED)

        for _ in range(3):
            target, state = support_transfer_step(
                config,
                state,
                _sensors(forward_increment=0.005),
                stair_request=True,
            )
        self.assertEqual(state.phase, SupportTransferPhase.DONE)
        self.assertFalse(target.active)

    def test_done_latches_until_request_clears(self) -> None:
        config = _config()
        state = SupportTransferState(
            phase=SupportTransferPhase.DONE,
            lead_side=LeadSide.LEFT,
            left_riser_seen=True,
            right_riser_seen=True,
        )
        target, state = support_transfer_step(
            config,
            state,
            _sensors(left_support=0.0, right_support=0.0),
            stair_request=True,
        )
        self.assertEqual(state.phase, SupportTransferPhase.DONE)
        self.assertFalse(target.active)

    def test_both_support_does_not_release_before_minimum_travel(self) -> None:
        config = _config(minimum_guard_travel_m=0.02)
        state = SupportTransferState(
            phase=SupportTransferPhase.BOTH_SUPPORTED,
            lead_side=LeadSide.LEFT,
            left_riser_seen=True,
            right_riser_seen=True,
        )
        for _ in range(5):
            target, state = support_transfer_step(
                config,
                state,
                _sensors(forward_increment=0.001),
                stair_request=True,
            )
        self.assertEqual(state.phase, SupportTransferPhase.BOTH_SUPPORTED)
        self.assertTrue(target.active)
        self.assertAlmostEqual(state.travel_since_trigger_m, 0.005)

        for _ in range(3):
            target, state = support_transfer_step(
                config,
                state,
                _sensors(forward_increment=0.005),
                stair_request=True,
            )
        self.assertEqual(state.phase, SupportTransferPhase.DONE)
        self.assertFalse(target.active)

    def test_riser_hysteresis_rejects_flicker(self) -> None:
        config = _config()
        state = SupportTransferState()
        _target, state = support_transfer_step(
            config,
            state,
            _sensors(left_riser=5.0),
            stair_request=True,
        )
        self.assertTrue(state.left_riser_active)
        _target, state = support_transfer_step(
            config,
            state,
            _sensors(left_riser=3.0),
            stair_request=True,
        )
        self.assertTrue(state.left_riser_active)
        self.assertEqual(state.confirm_streak, 0)
        _target, state = support_transfer_step(
            config,
            state,
            _sensors(left_riser=2.5),
            stair_request=True,
        )
        self.assertFalse(state.left_riser_active)
        self.assertEqual(state.confirm_streak, 1)

    def test_simultaneous_contact_uses_force_then_preference(self) -> None:
        config = _config()
        _target, stronger = support_transfer_step(
            config,
            SupportTransferState(preferred_side=LeadSide.RIGHT),
            _sensors(left_riser=8.0, right_riser=7.0),
            stair_request=True,
        )
        self.assertEqual(stronger.lead_side, LeadSide.LEFT)
        _target, tied = support_transfer_step(
            config,
            SupportTransferState(preferred_side=LeadSide.RIGHT),
            _sensors(left_riser=8.0, right_riser=8.0),
            stair_request=True,
        )
        self.assertEqual(tied.lead_side, LeadSide.RIGHT)

    def test_unconfirmed_lead_aborts_fail_closed(self) -> None:
        config = _config(lead_timeout_s=0.010)
        state = SupportTransferState()
        _target, state = support_transfer_step(
            config,
            state,
            _sensors(left_riser=8.0),
            stair_request=True,
        )
        for _ in range(2):
            target, state = support_transfer_step(
                config,
                state,
                _sensors(left_riser=8.0, left_support=0.0),
                stair_request=True,
            )
        self.assertEqual(state.phase, SupportTransferPhase.ABORT)
        self.assertEqual(state.abort_reason, "lead_support_timeout")
        self.assertTrue(target.abort)

    def test_unsafe_aborts_immediately(self) -> None:
        config = _config()
        target, state = support_transfer_step(
            config,
            SupportTransferState(),
            _sensors(unsafe=True),
            stair_request=True,
        )
        self.assertEqual(state.phase, SupportTransferPhase.ABORT)
        self.assertEqual(state.abort_reason, "unsafe")
        self.assertTrue(target.abort)

    def test_configuration_requires_real_hysteresis_and_exact_timing(self) -> None:
        with self.assertRaisesRegex(ValueError, "below"):
            _config(riser_off_force_n=5.0)
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            _config(lead_timeout_s=0.011)
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            _sensors(left_support=-1.0).validate()


if __name__ == "__main__":
    unittest.main()
