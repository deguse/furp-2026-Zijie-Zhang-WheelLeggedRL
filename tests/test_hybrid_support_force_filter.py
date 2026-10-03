from __future__ import annotations

import unittest

from hoppertrex_mjlab.hybrid.roll_feedback import (
    ROLL_FEEDBACK_MAX_IDENTIFIED_AMPLITUDE_RAD,
)
from hoppertrex_mjlab.hybrid.stair_dynamic import (
    LEFT_LIFT_BASIS,
    RIGHT_LIFT_BASIS,
    LeadSide,
)
from hoppertrex_mjlab.hybrid.support_force_filter import (
    SupportForceFilterConfig,
    SupportForceFilterState,
    support_force_filter_step,
    support_force_leg_offsets,
)
from hoppertrex_mjlab.hybrid.support_transfer import (
    SupportTransferPhase,
    SupportTransferTargets,
)


def _roles(hold: LeadSide, compliant: LeadSide) -> SupportTransferTargets:
    return SupportTransferTargets(
        phase=SupportTransferPhase.LEAD_GUARD,
        lead_side=compliant,
        hold_support_side=hold,
        compliant_side=compliant,
        active=True,
    )


class SupportForceFilterTest(unittest.TestCase):
    def test_inactive_zero_state_is_exactly_inert(self) -> None:
        output, state = support_force_filter_step(
            SupportForceFilterConfig(),
            SupportForceFilterState(),
            SupportTransferTargets(
                phase=SupportTransferPhase.IDLE,
                lead_side=LeadSide.NONE,
            ),
            left_vertical_force_n=75.0,
            right_vertical_force_n=75.0,
        )
        self.assertEqual(output.leg_offsets_rad, (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(state.left_amplitude_rad, 0.0)
        self.assertEqual(state.right_amplitude_rad, 0.0)

    def test_hold_side_receives_larger_target_without_integrating(self) -> None:
        config = SupportForceFilterConfig()
        roles = _roles(LeadSide.RIGHT, LeadSide.LEFT)
        output, state = support_force_filter_step(
            config,
            SupportForceFilterState(),
            roles,
            left_vertical_force_n=75.0,
            right_vertical_force_n=75.0,
        )
        self.assertEqual(output.left_target_force_n, 60.0)
        self.assertEqual(output.right_target_force_n, 90.0)
        self.assertGreater(output.left_amplitude_rad, 0.0)
        self.assertLess(output.right_amplitude_rad, 0.0)
        repeated, _ = support_force_filter_step(
            config,
            state,
            roles,
            left_vertical_force_n=75.0,
            right_vertical_force_n=75.0,
        )
        self.assertEqual(repeated.left_amplitude_rad, output.left_amplitude_rad)
        self.assertEqual(repeated.right_amplitude_rad, output.right_amplitude_rad)

    def test_correction_is_hard_bounded_below_identified_authority(self) -> None:
        config = SupportForceFilterConfig(force_to_velocity_gain_rad_per_n_s=1.0)
        output, _ = support_force_filter_step(
            config,
            SupportForceFilterState(),
            _roles(LeadSide.LEFT, LeadSide.RIGHT),
            left_vertical_force_n=500.0,
            right_vertical_force_n=0.0,
        )
        self.assertEqual(output.left_amplitude_rad, config.max_amplitude_rad)
        self.assertEqual(output.right_amplitude_rad, -config.max_amplitude_rad)
        self.assertLess(
            config.max_amplitude_rad,
            ROLL_FEEDBACK_MAX_IDENTIFIED_AMPLITUDE_RAD,
        )

    def test_inactive_filter_releases_exactly_without_memory(self) -> None:
        config = SupportForceFilterConfig()
        active, state = support_force_filter_step(
            config,
            SupportForceFilterState(),
            _roles(LeadSide.LEFT, LeadSide.RIGHT),
            left_vertical_force_n=100.0,
            right_vertical_force_n=50.0,
        )
        self.assertNotEqual(active.leg_offsets_rad, (0.0, 0.0, 0.0, 0.0))
        inactive, state = support_force_filter_step(
            config,
            state,
            SupportTransferTargets(
                phase=SupportTransferPhase.DONE,
                lead_side=LeadSide.LEFT,
            ),
            left_vertical_force_n=100.0,
            right_vertical_force_n=50.0,
        )
        self.assertEqual(inactive.leg_offsets_rad, (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(state.left_amplitude_rad, 0.0)
        self.assertEqual(state.right_amplitude_rad, 0.0)

    def test_joint_offsets_follow_each_identified_lift_basis(self) -> None:
        offsets = support_force_leg_offsets(0.001, -0.002)
        self.assertEqual(
            offsets,
            (
                0.001 * LEFT_LIFT_BASIS[0],
                -0.002 * RIGHT_LIFT_BASIS[0],
                0.001 * LEFT_LIFT_BASIS[1],
                -0.002 * RIGHT_LIFT_BASIS[1],
            ),
        )

    def test_configuration_rejects_unidentified_or_degenerate_authority(self) -> None:
        with self.assertRaisesRegex(ValueError, "identified"):
            SupportForceFilterConfig(
                max_amplitude_rad=ROLL_FEEDBACK_MAX_IDENTIFIED_AMPLITUDE_RAD + 1e-6,
            )
        with self.assertRaisesRegex(ValueError, "hold-load"):
            SupportForceFilterConfig(hold_load_fraction=0.49)
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            support_force_filter_step(
                SupportForceFilterConfig(),
                SupportForceFilterState(),
                _roles(LeadSide.LEFT, LeadSide.RIGHT),
                left_vertical_force_n=-1.0,
                right_vertical_force_n=0.0,
            )


if __name__ == "__main__":
    unittest.main()
