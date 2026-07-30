"""Fit a proprioceptive stair-contact detector from paired v2 captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from hoppertrex_mjlab.hybrid.stair_classical import (
    CONTROL_DT_S,
    ContactDetectorCfg,
    qualify_contact_detector,
)

DETECTOR_SIGNAL_SCHEMA = "deployment_attempt_v2"
DETECTOR_SERIES_FIELDS = (
    "pitch_rate_radps",
    "wheel_speed_error_radps",
    "body_deceleration_mps2",
)
EXPECTED_CAPTURE_COUNT = 32
EXPECTED_CAPTURES_PER_CELL = 16
EXPECTED_CELL_COUNT = 2
EXPECTED_ENVS_PER_HEIGHT = 16
EXPECTED_SETTLE_STEPS = 200
EXPECTED_DRIVE_STEPS = 500
EXPECTED_PRE_IMPACT_STEPS = 25
EXPECTED_POST_IMPACT_STEPS = 75
EXPECTED_STABLE_STEPS = 25
EXPECTED_SERIES_SAMPLES = 500
FLAT_CONTROL_SUCCESS_RATE = 0.90
EXPECTED_HEIGHTS_M = (0.0, 0.01)
EXPECTED_COMMAND_CELLS = (
    {"name": "pitch_zero", "pitch_rad": 0.0, "vx_mps": 0.07},
    {"name": "fast_lean_0p032", "pitch_rad": -0.032, "vx_mps": 0.10},
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _sequence(
    series: dict[str, Any], *, expected_samples: int | None = None
) -> list[tuple[float, float, float]]:
    missing = [field for field in DETECTOR_SERIES_FIELDS if field not in series]
    if missing:
        raise ValueError(
            "Capture lacks direct deployment detector signals: " + ", ".join(missing)
        )
    pitch_rate = np.asarray(series["pitch_rate_radps"], dtype=np.float64)
    wheel_error = np.asarray(series["wheel_speed_error_radps"], dtype=np.float64)
    body_deceleration = np.asarray(
        series["body_deceleration_mps2"], dtype=np.float64
    )
    lengths = {len(pitch_rate), len(wheel_error), len(body_deceleration)}
    if len(lengths) != 1:
        raise ValueError("Deployment detector signal series must have equal lengths.")
    if expected_samples is not None and lengths != {expected_samples}:
        raise ValueError(
            f"Deployment detector signal series must contain {expected_samples} samples."
        )
    if not all(
        np.all(np.isfinite(values))
        for values in (pitch_rate, wheel_error, body_deceleration)
    ):
        raise ValueError("Deployment detector signal series must be finite.")
    return list(zip(pitch_rate, wheel_error, body_deceleration, strict=True))


def fit_detector(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("probe") != "hybrid_c2_paired_capture_v1":
        raise ValueError("Capture probe identity is not registered for C2 fitting.")
    if payload.get("classification") != "ANALYSIS_READY":
        raise ValueError("Capture classification must be ANALYSIS_READY.")
    if payload.get("evidence_eligible") is not True:
        raise ValueError("Capture must be marked evidence eligible.")
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise TypeError("Capture protocol must contain a JSON object.")
    if protocol.get("detector_signal_schema") != DETECTOR_SIGNAL_SCHEMA:
        raise ValueError(
            f"Capture detector_signal_schema must be {DETECTOR_SIGNAL_SCHEMA}."
        )
    if protocol.get("detector_activation") != "stair_attempt_start":
        raise ValueError("Capture detector activation must be stair_attempt_start.")
    if tuple(protocol.get("detector_series_fields", ())) != DETECTOR_SERIES_FIELDS:
        raise ValueError("Capture detector_series_fields do not match deployment replay.")
    expected_attempt_fields = DETECTOR_SERIES_FIELDS + ("detector_active",)
    if tuple(protocol.get("detector_attempt_fields", ())) != expected_attempt_fields:
        raise ValueError("Capture detector_attempt_fields do not match deployment replay.")
    if float(protocol.get("control_dt_s", float("nan"))) != CONTROL_DT_S:
        raise ValueError(f"Capture control_dt_s must be {CONTROL_DT_S}.")
    registered_protocol = {
        "envs_per_height": EXPECTED_ENVS_PER_HEIGHT,
        "settle_steps": EXPECTED_SETTLE_STEPS,
        "drive_steps": EXPECTED_DRIVE_STEPS,
        "pre_impact_steps": EXPECTED_PRE_IMPACT_STEPS,
        "post_impact_steps": EXPECTED_POST_IMPACT_STEPS,
        "stable_steps": EXPECTED_STABLE_STEPS,
        "detector_series_samples": EXPECTED_SERIES_SAMPLES,
        "expected_capture_count": EXPECTED_CAPTURE_COUNT,
    }
    for field, expected in registered_protocol.items():
        if int(protocol.get(field, -1)) != expected:
            raise ValueError(f"Capture protocol {field} must be {expected}.")
    if tuple(protocol.get("heights_m", ())) != EXPECTED_HEIGHTS_M:
        raise ValueError("Capture protocol heights_m do not match registration.")
    if tuple(protocol.get("command_cells", ())) != EXPECTED_COMMAND_CELLS:
        raise ValueError("Capture protocol command_cells do not match registration.")
    captures = payload.get("paired_captures")
    if not isinstance(captures, list):
        raise TypeError("Capture paired_captures must be a JSON array.")
    if len(captures) != EXPECTED_CAPTURE_COUNT:
        raise ValueError(f"Capture must contain {EXPECTED_CAPTURE_COUNT} pairs.")
    if int(payload.get("valid_capture_count", -1)) != EXPECTED_CAPTURE_COUNT:
        raise ValueError(f"Capture must declare {EXPECTED_CAPTURE_COUNT} valid pairs.")
    if int(payload.get("invalid_capture_count", -1)) != 0:
        raise ValueError("Capture must declare zero invalid pairs.")
    if payload.get("flat_control_passed") is not True:
        raise ValueError("Capture flat control must pass.")
    if any(not isinstance(item, dict) or item.get("valid") is not True for item in captures):
        raise ValueError("Every paired capture must be valid.")
    if any(
        not isinstance(item.get("attempt_series"), dict)
        or int(item.get("impact_step", -1)) < 0
        or int(item.get("impact_step", -1)) >= EXPECTED_DRIVE_STEPS
        for item in captures
    ):
        raise ValueError("Every capture requires a full attempt and valid impact step.")
    trials = payload.get("trials")
    if not isinstance(trials, list) or len(trials) != EXPECTED_CELL_COUNT:
        raise ValueError(f"Capture must contain {EXPECTED_CELL_COUNT} trial rows.")
    if any(
        int(trial.get("recorded_drive_steps", -1)) != EXPECTED_DRIVE_STEPS
        or int(trial.get("stair_terminated", -1)) != 0
        or int(trial.get("stair_envs_without_impact", -1)) != 0
        or int(trial.get("flat_terminated", -1)) != 0
        or int(trial.get("flat_non_wheel_contact", -1)) != 0
        or int(trial.get("paired_captures", -1)) != EXPECTED_CAPTURES_PER_CELL
        or int(trial.get("valid_paired_captures", -1))
        != EXPECTED_CAPTURES_PER_CELL
        or float(trial.get("flat_success_rate", -1.0))
        < FLAT_CONTROL_SUCCESS_RATE
        for trial in trials
    ):
        raise ValueError("Capture trial health counters do not qualify for fitting.")
    flat = [
        _sequence(
            item["attempt_series"]["flat"],
            expected_samples=EXPECTED_SERIES_SAMPLES,
        )
        for item in captures
    ]
    stair = [
        _sequence(
            item["attempt_series"]["stair"],
            expected_samples=EXPECTED_SERIES_SAMPLES,
        )
        for item in captures
    ]
    impact_indices = [int(item["impact_step"]) for item in captures]
    flat_active_masks = [
        [bool(value) for value in item["attempt_series"]["flat"]["detector_active"]]
        for item in captures
    ]
    stair_active_masks = [
        [bool(value) for value in item["attempt_series"]["stair"]["detector_active"]]
        for item in captures
    ]
    if any(
        len(mask) != EXPECTED_SERIES_SAMPLES
        for mask in flat_active_masks + stair_active_masks
    ):
        raise ValueError("Detector activation masks must cover every attempt tick.")
    candidates: list[dict[str, Any]] = []
    for pitch_threshold in (0.02, 0.04, 0.06, 0.08, 0.10):
        for wheel_threshold in (0.10, 0.20, 0.30, 0.50, 1.00):
            for decel_threshold in (0.5, 1.0, 2.0, 3.0, 5.0):
                cfg = ContactDetectorCfg(
                    pitch_rate_delta=pitch_threshold,
                    wheel_speed_error=wheel_threshold,
                    body_deceleration=decel_threshold,
                    consecutive_ticks=2,
                )
                qualification = qualify_contact_detector(
                    cfg,
                    flat_sequences=flat,
                    stair_sequences=stair,
                    impact_indices=impact_indices,
                    flat_active_masks=flat_active_masks,
                    stair_active_masks=stair_active_masks,
                )
                candidates.append(
                    {
                        "contact_detector": {
                            "pitch_rate_delta": pitch_threshold,
                            "wheel_speed_error": wheel_threshold,
                            "body_deceleration": decel_threshold,
                            "consecutive_ticks": 2,
                        },
                        "qualification": qualification,
                    }
                )
    qualified = [item for item in candidates if item["qualification"]["qualified"]]
    if not qualified:
        raise RuntimeError(
            "No proprioceptive detector achieved zero flat false positives "
            "and 95% timely detection."
        )
    qualified.sort(
        key=lambda item: (
            item["qualification"]["timely_detection_rate"],
            -np.mean(item["qualification"]["detection_delays_ticks"]),
            item["contact_detector"]["pitch_rate_delta"],
            item["contact_detector"]["wheel_speed_error"],
            item["contact_detector"]["body_deceleration"],
        ),
        reverse=True,
    )
    return {
        "schema_version": 1,
        "source_probe": payload["probe"],
        "source_git_sha": payload["git_sha"],
        "detector_signal_schema": DETECTOR_SIGNAL_SCHEMA,
        "capture_count": len(captures),
        "selected": qualified[0],
        "qualified_candidate_count": len(qualified),
        "candidate_count": len(candidates),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError("Capture input must contain a JSON object.")
    result = fit_detector(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
    )
    print(f"Wrote qualified contact detector: {args.output.resolve()}")


if __name__ == "__main__":
    main()
