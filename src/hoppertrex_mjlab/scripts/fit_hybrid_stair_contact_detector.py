"""Fit a proprioceptive stair-contact detector from paired v2 captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from hoppertrex_mjlab.hybrid.stair_classical import (
    ContactDetectorCfg,
    qualify_contact_detector,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _sequence(series: dict[str, Any]) -> list[tuple[float, float, float]]:
    pitch = np.asarray(series["pitch_rad"], dtype=np.float64)
    body_vx = np.asarray(series["body_vx_mps"], dtype=np.float64)
    wheel_speed = np.asarray(series["wheel_speed_radps"], dtype=np.float64)
    wheel_target = np.asarray(series["wheel_target_radps"], dtype=np.float64)
    pitch_rate = np.gradient(pitch, 0.02)
    wheel_error = wheel_speed - wheel_target
    return list(zip(pitch_rate, wheel_error, body_vx, strict=True))


def fit_detector(payload: dict[str, Any]) -> dict[str, Any]:
    captures = [item for item in payload["paired_captures"] if item["valid"]]
    flat = [_sequence(item["aligned_series"]["flat"]) for item in captures]
    stair = [_sequence(item["aligned_series"]["stair"]) for item in captures]
    impact_index = int(payload["protocol"]["pre_impact_steps"])
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
                    impact_indices=[impact_index] * len(stair),
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
