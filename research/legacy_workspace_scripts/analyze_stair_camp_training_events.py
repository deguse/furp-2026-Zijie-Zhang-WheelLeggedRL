#!/usr/bin/env python3
"""Read-only TensorBoard diagnostics for StairCamp training runs.

This is an operator-side analysis helper, not part of the promotion protocol.
(Codex: 2026-08-11)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

INHERITED_POSITIVE_TAGS = (
    "Episode_Reward/upright",
    "Episode_Reward/clean_wheel_support",
    "Episode_Reward/track_linear_velocity",
    "Episode_Reward/wheel_ground_contact",
    "Episode_Reward/alive",
    "Episode_Reward/track_angular_velocity",
)
STAIR_TAGS = (
    "Episode_Reward/stair_progress",
    "Episode_Reward/stair_climb_success",
)
CORE_TAGS = (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Policy/mean_std",
    "Curriculum/stair_height_band/evaluations",
    "Curriculum/stair_height_band/upper_height_m",
    "Curriculum/stair_height_band/consecutive_ready",
    "Curriculum/stair_height_band/mean_level",
    "Episode_Metrics/stair_camp_step",
    "Episode_Termination/bad_orientation",
    "Episode_Termination/root_too_low",
    "Episode_Termination/non_wheel_ground_contact",
    "Episode_Termination/nan_detection",
)
REQUIRED_TAGS = tuple(dict.fromkeys(INHERITED_POSITIVE_TAGS + STAIR_TAGS + CORE_TAGS))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: float, *, tag: str, step: int) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite scalar {tag} at step {step}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--steps", type=int, nargs="+", default=(100, 300, 999))
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.window < 1:
        raise ValueError("--window must be >= 1")
    run_directory = args.run_directory.resolve()
    events = sorted(run_directory.glob("events.out.tfevents.*"))
    if len(events) != 1:
        raise ValueError(f"expected exactly one event file, found {len(events)}")
    event = events[0]
    accumulator = EventAccumulator(str(event), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", ()))
    missing = sorted(set(REQUIRED_TAGS) - available)
    if missing:
        raise ValueError(f"missing required scalar tags: {missing}")
    series = {
        tag: {
            int(item.step): finite(item.value, tag=tag, step=int(item.step))
            for item in accumulator.Scalars(tag)
        }
        for tag in REQUIRED_TAGS
    }
    reference_steps = sorted(series["Train/mean_reward"])

    reports: list[dict[str, object]] = []
    for target in args.steps:
        if target not in series["Train/mean_reward"]:
            reports.append({"requested_step": target, "available": False})
            continue
        exact = {tag: values[target] for tag, values in series.items()}
        inherited = sum(max(0.0, exact[tag]) for tag in INHERITED_POSITIVE_TAGS)
        stair = sum(max(0.0, exact[tag]) for tag in STAIR_TAGS)
        denominator = inherited + stair
        window_steps = [
            step
            for step in reference_steps
            if target - args.window + 1 <= step <= target
        ]
        if not window_steps:
            raise ValueError(f"empty diagnostic window at step {target}")
        inherited_window = [
            sum(max(0.0, series[tag][step]) for tag in INHERITED_POSITIVE_TAGS)
            for step in window_steps
        ]
        stair_window = [
            sum(max(0.0, series[tag][step]) for tag in STAIR_TAGS)
            for step in window_steps
        ]
        window_denominator = sum(inherited_window) + sum(stair_window)
        reports.append(
            {
                "requested_step": target,
                "available": True,
                "exact": {
                    "mean_reward": exact["Train/mean_reward"],
                    "mean_episode_length": exact["Train/mean_episode_length"],
                    "mean_action_std": exact["Policy/mean_std"],
                    "stair_progress": exact[STAIR_TAGS[0]],
                    "stair_climb_success": exact[STAIR_TAGS[1]],
                    "inherited_positive_income": inherited,
                    "stair_positive_income": stair,
                    "stair_positive_share": stair / denominator if denominator else 0.0,
                    "curriculum_evaluations_event_value": exact[
                        "Curriculum/stair_height_band/evaluations"
                    ],
                    "upper_height_m_event_value": exact[
                        "Curriculum/stair_height_band/upper_height_m"
                    ],
                    "consecutive_ready_event_value": exact[
                        "Curriculum/stair_height_band/consecutive_ready"
                    ],
                    "mean_level_event_value": exact[
                        "Curriculum/stair_height_band/mean_level"
                    ],
                    "stair_camp_step": exact["Episode_Metrics/stair_camp_step"],
                    "terminations": {
                        name.rsplit("/", 1)[-1]: exact[name]
                        for name in CORE_TAGS
                        if name.startswith("Episode_Termination/")
                    },
                },
                "trailing_window": {
                    "requested_width": args.window,
                    "actual_samples": len(window_steps),
                    "first_step": window_steps[0],
                    "last_step": window_steps[-1],
                    "mean_stair_progress": fmean(series[STAIR_TAGS[0]][s] for s in window_steps),
                    "mean_stair_climb_success": fmean(
                        series[STAIR_TAGS[1]][s] for s in window_steps
                    ),
                    "nonzero_stair_climb_success_steps": sum(
                        series[STAIR_TAGS[1]][s] > 0.0 for s in window_steps
                    ),
                    "stair_positive_share_of_window_sums": (
                        sum(stair_window) / window_denominator
                        if window_denominator
                        else 0.0
                    ),
                },
            }
        )
    output = {
        "schema_version": 1,
        "kind": "stair_camp_tensorboard_diagnostic",
        "protocol_status": "NON_PROMOTABLE_DIAGNOSTIC_ONLY",
        "run_directory": str(run_directory),
        "event_file": str(event.resolve()),
        "event_file_sha256": sha256(event),
        "window": args.window,
        "reports": reports,
    }
    text = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
