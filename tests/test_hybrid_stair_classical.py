from __future__ import annotations

import unittest

import numpy as np

from hoppertrex_mjlab.hybrid.stair_classical import (
    CandidateScore,
    ContactDetectorCfg,
    ContactDetectorState,
    StairControllerState,
    StairManeuver,
    StairPhase,
    StairSensors,
    classical_plateau_decision,
    contact_detector_step,
    contact_detector_wheel_reference_radps,
    optimize_cem,
    qualify_contact_detector,
    stair_controller_step,
)


def _maneuver() -> StairManeuver:
    return StairManeuver(
        approach_vx=0.08,
        preload_trigger_m=0.01,
        preload_duration_s=0.04,
        preload_height_m=0.30,
        preload_pitch_rad=-0.02,
        contact_vx=0.06,
        climb_vx=0.08,
        drive_feedforward_radps=1.0,
        climb_height_m=0.32,
        climb_pitch_rad=-0.01,
        climb_timeout_s=1.0,
        crest_progress_m=0.40,
        recover_duration_s=0.04,
        detector=ContactDetectorCfg(0.1, 0.2, 1.0, 2),
        maneuver_hash="a" * 64,
        bindings={"controller_schedule_hash": "b" * 64},
    )


class HybridStairClassicalTest(unittest.TestCase):
    def test_detector_wheel_reference_matches_deployment_contract(self) -> None:
        self.assertAlmostEqual(
            contact_detector_wheel_reference_radps(
                command_vx=0.07,
                velocity_command_scale=0.86,
                velocity_command_bias=-0.012,
                wheel_radius=0.1,
            ),
            0.482,
        )
        self.assertAlmostEqual(
            contact_detector_wheel_reference_radps(
                command_vx=0.07,
                velocity_command_scale=0.86,
                velocity_command_bias=-0.012,
                station_drift_mps=-0.014,
                wheel_radius=0.1,
            ),
            0.622,
        )

    def test_detector_requires_two_votes_for_two_ticks(self) -> None:
        cfg = ContactDetectorCfg(0.1, 0.2, 1.0, 2)
        state = ContactDetectorState(previous_pitch_rate=0.0)
        detected, state, votes = contact_detector_step(
            cfg,
            state,
            pitch_rate=0.2,
            wheel_speed_error=0.3,
            body_deceleration=2.5,
        )
        self.assertEqual(votes, (True, True, True))
        self.assertFalse(detected)
        detected, _, _ = contact_detector_step(
            cfg,
            state,
            pitch_rate=0.4,
            wheel_speed_error=0.3,
            body_deceleration=2.5,
        )
        self.assertTrue(detected)

    def test_detector_first_sample_initializes_pitch_baseline(self) -> None:
        cfg = ContactDetectorCfg(0.1, 0.2, 1.0, 1)
        detected, state, votes = contact_detector_step(
            cfg,
            ContactDetectorState(),
            pitch_rate=0.4,
            wheel_speed_error=0.3,
            body_deceleration=0.0,
        )
        self.assertEqual(votes, (False, True, False))
        self.assertFalse(detected)
        self.assertEqual(state.previous_pitch_rate, 0.4)

    def test_approach_detection_is_latched_until_contact_wait(self) -> None:
        maneuver = _maneuver()
        state = StairControllerState()
        quiet = StairSensors(0.0, 0.0, 0.0, 3.0, 0.0)
        _target, state = stair_controller_step(
            maneuver,
            state,
            quiet,
            stair_mode=True,
            nominal_height=0.31,
            nominal_pitch=0.0,
        )
        for pitch_rate in (0.2, 0.4):
            target, state = stair_controller_step(
                maneuver,
                state,
                StairSensors(0.0, pitch_rate, 2.0, 3.0, 0.3),
                stair_mode=True,
                nominal_height=0.31,
                nominal_pitch=0.0,
            )
        self.assertTrue(target.contact_detected)
        for _ in range(4):
            target, state = stair_controller_step(
                maneuver,
                state,
                quiet,
                stair_mode=True,
                nominal_height=0.31,
                nominal_pitch=0.0,
            )
        self.assertEqual(target.phase, StairPhase.CLIMB)

    def test_detector_qualification_rejects_flat_false_positive(self) -> None:
        cfg = ContactDetectorCfg(0.1, 0.2, 1.0, 1)
        quiet = [(0.0, 0.0, 0.1)] * 5
        impact = [(0.0, 0.0, 0.1), (0.2, 0.3, 0.0)]
        result = qualify_contact_detector(
            cfg,
            flat_sequences=[quiet],
            stair_sequences=[impact] * 20,
            impact_indices=[1] * 20,
        )
        self.assertTrue(result["qualified"])
        noisy = [[(0.0, 0.0, 0.1), (0.2, 0.3, 0.0)]]
        result = qualify_contact_detector(
            cfg,
            flat_sequences=noisy,
            stair_sequences=[impact] * 20,
            impact_indices=[1] * 20,
        )
        self.assertFalse(result["qualified"])

    def test_detector_qualification_does_not_ignore_pre_impact_detection(self) -> None:
        cfg = ContactDetectorCfg(0.1, 0.2, 1.0, 1)
        early_then_late = [
            (0.0, 0.3, 2.0),
            (0.2, 0.3, 2.0),
            (0.2, 0.3, 2.0),
        ]
        result = qualify_contact_detector(
            cfg,
            flat_sequences=[[(0.0, 0.0, 0.0)] * 3],
            stair_sequences=[early_then_late] * 20,
            impact_indices=[2] * 20,
        )
        self.assertFalse(result["qualified"])
        self.assertEqual(result["timely_detection_count"], 0)
        self.assertEqual(result["stair_pre_impact_detection_sequences"], 20)

    def test_qualification_respects_per_tick_activation_mask(self) -> None:
        cfg = ContactDetectorCfg(0.1, 0.2, 1.0, 1)
        sequence = [(0.0, 0.3, 2.0), (0.2, 0.3, 2.0)]
        result = qualify_contact_detector(
            cfg,
            flat_sequences=[sequence],
            stair_sequences=[sequence] * 20,
            impact_indices=[1] * 20,
            flat_active_masks=[[False, False]],
            stair_active_masks=[[False, True]] * 20,
        )
        self.assertTrue(result["qualified"])

    def test_state_machine_reaches_climb_and_aborts_safely(self) -> None:
        maneuver = _maneuver()
        state = StairControllerState()
        sensor = StairSensors(0.0, 0.0, 0.08, 3.0, 0.0)
        target, state = stair_controller_step(
            maneuver,
            state,
            sensor,
            stair_mode=True,
            nominal_height=0.31,
            nominal_pitch=0.0,
        )
        self.assertEqual(target.phase, StairPhase.APPROACH)
        for _ in range(3):
            target, state = stair_controller_step(
                maneuver,
                state,
                sensor,
                stair_mode=True,
                nominal_height=0.31,
                nominal_pitch=0.0,
            )
        self.assertIn(target.phase, (StairPhase.PRELOAD, StairPhase.CONTACT_WAIT))
        for pitch_rate in (0.2, 0.4, 0.6):
            impact = StairSensors(0.0, pitch_rate, 0.0, 0.0, 0.3)
            target, state = stair_controller_step(
                maneuver,
                state,
                impact,
                stair_mode=True,
                nominal_height=0.31,
                nominal_pitch=0.0,
            )
        self.assertEqual(target.phase, StairPhase.CLIMB)
        self.assertEqual(target.drive_feedforward_radps, 1.0)
        target, state = stair_controller_step(
            maneuver,
            state,
            StairSensors(0.0, 0.0, 0.0, 0.0, 0.0, non_wheel_contact=True),
            stair_mode=True,
            nominal_height=0.31,
            nominal_pitch=0.0,
        )
        self.assertEqual(target.phase, StairPhase.ABORT)
        self.assertEqual(state.abort_reason, "non_wheel_contact")
        self.assertEqual(target.vx, 0.0)

    def test_cem_is_deterministic_and_respects_bounds(self) -> None:
        def evaluate(candidate: np.ndarray) -> CandidateScore:
            error = float(np.sum((candidate - 0.25) ** 2))
            return CandidateScore(1, -error, error, error, error)

        first = optimize_cem(
            evaluate,
            lower=np.array([-1.0, -1.0]),
            upper=np.array([1.0, 1.0]),
            population=32,
            iterations=5,
            seed=1,
        )
        second = optimize_cem(
            evaluate,
            lower=np.array([-1.0, -1.0]),
            upper=np.array([1.0, 1.0]),
            population=32,
            iterations=5,
            seed=1,
        )
        np.testing.assert_array_equal(first.parameters, second.parameters)
        self.assertTrue(np.all(first.parameters >= -1.0))
        self.assertTrue(np.all(first.parameters <= 1.0))

    def test_plateau_requires_two_consecutive_stalled_rounds(self) -> None:
        rounds = [
            {"highest_contiguous_pass_m": 0.03, "first_failure_success_rate": 0.40},
            {"highest_contiguous_pass_m": 0.03, "first_failure_success_rate": 0.43},
            {"highest_contiguous_pass_m": 0.03, "first_failure_success_rate": 0.47},
        ]
        result = classical_plateau_decision(rounds)
        self.assertTrue(result["freeze"])
        rounds[-1]["highest_contiguous_pass_m"] = 0.04
        self.assertFalse(classical_plateau_decision(rounds)["freeze"])


if __name__ == "__main__":
    unittest.main()
