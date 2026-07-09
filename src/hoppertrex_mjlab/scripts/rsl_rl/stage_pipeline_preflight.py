#!/usr/bin/env python3
"""Static preflight for the scratch Stage2-5 training/evaluation pipeline."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_PATH = Path(__file__).resolve().parents[2]
SRC_PATH = Path(__file__).resolve().parents[3]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

try:
  import hoppertrex_mjlab.tasks as hoppertrex_tasks
  from hoppertrex_mjlab.scripts.rsl_rl.evaluate_stage_gate import STAGE_TASKS
  from hoppertrex_mjlab.tasks.hoppertrex_balance_task import (
    WHEEL_JOINT_NAMES,
    joint_pos_rel_without_wheel_position,
  )
except ImportError:
  import tasks as hoppertrex_tasks
  from scripts.rsl_rl.evaluate_stage_gate import STAGE_TASKS
  from tasks.hoppertrex_balance_task import (
    WHEEL_JOINT_NAMES,
    joint_pos_rel_without_wheel_position,
  )
from mjlab.tasks.registry import load_env_cfg


SUSTAINED_EPISODE_S = 60.0
SUSTAINED_RESAMPLE_RANGE = (30.0, 60.0)


@dataclass(frozen=True)
class CheckResult:
  name: str
  passed: bool
  detail: str


def _check_equal(name: str, actual: Any, expected: Any) -> CheckResult:
  return CheckResult(
    name=name,
    passed=actual == expected,
    detail=f"{actual!r} == {expected!r}",
  )


def _check_is(name: str, actual: Any, expected: Any) -> CheckResult:
  return CheckResult(
    name=name,
    passed=actual is expected,
    detail=f"{getattr(actual, '__name__', actual)!r} is {getattr(expected, '__name__', expected)!r}",
  )


def _stage_task_expectations() -> dict[int, str]:
  return {
    2: hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE2_BIDIR_LIN_SMOOTH_SLEW6_REWARD_BALANCE_FF_TASK_ID,
    3: hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE3_YAW_ONLY_MEDIUM_ALIGNED_SMOOTH_TASK_ID,
    4: hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE4_SMALL_LIN_SMALL_YAW_TASK_ID,
    5: hoppertrex_tasks.HOPPERTREX_SCRATCH_STAGE5_FULL_LIN_FULL_YAW_TASK_ID,
  }


def run_preflight() -> list[CheckResult]:
  checks: list[CheckResult] = []
  expected_tasks = _stage_task_expectations()

  for stage, expected_task in expected_tasks.items():
    checks.append(_check_equal(f"stage{stage}_gate_default_task", STAGE_TASKS[stage], expected_task))

    cfg = load_env_cfg(expected_task)
    action = cfg.actions["wheel_balance"]
    twist = cfg.commands["twist"]
    actor_joint_pos = cfg.observations["actor"].terms["joint_pos"]
    critic_joint_pos = cfg.observations["critic"].terms["joint_pos"]

    checks.append(_check_equal(f"stage{stage}_episode_length_s", cfg.episode_length_s, SUSTAINED_EPISODE_S))
    checks.append(_check_equal(f"stage{stage}_resampling_time_range", twist.resampling_time_range, SUSTAINED_RESAMPLE_RANGE))
    checks.append(_check_is(f"stage{stage}_actor_zero_wheel_pos_obs", actor_joint_pos.func, joint_pos_rel_without_wheel_position))
    checks.append(_check_is(f"stage{stage}_critic_zero_wheel_pos_obs", critic_joint_pos.func, joint_pos_rel_without_wheel_position))
    checks.append(_check_equal(f"stage{stage}_actor_wheel_joint_names", actor_joint_pos.params["wheel_joint_names"], WHEEL_JOINT_NAMES))
    checks.append(_check_equal(f"stage{stage}_critic_wheel_joint_names", critic_joint_pos.params["wheel_joint_names"], WHEEL_JOINT_NAMES))

    if stage == 2:
      checks.append(_check_equal("stage2_balance_smoothing_alpha", getattr(action, "balance_smoothing_alpha", None), 0.65))
      checks.append(_check_equal("stage2_target_slew_limit", getattr(action, "target_slew_limit", None), 6.0))
      checks.append(_check_equal("stage2_action_dim_kind", type(action).__name__, "CommandFeedforwardCoupledWheelVelocityActionCfg"))
      checks.append(_check_equal("stage2_residual_scale", getattr(action, "residual_scale", None), 0.15))
      checks.append(_check_equal("stage2_command_gain", getattr(action, "command_gain", None), 2.0))
      checks.append(_check_equal("stage2_feedforward_clip", getattr(action, "feedforward_clip", None), 0.25))
    elif stage == 3:
      checks.append(_check_equal("stage3_yaw_smoothing_alpha", getattr(action, "yaw_smoothing_alpha", None), 0.50))
      checks.append(_check_equal("stage3_target_slew_limit", getattr(action, "target_slew_limit", None), 12.0))
      checks.append(_check_equal("stage3_action_dim_kind", type(action).__name__, "DifferentialWheelVelocityActionCfg"))
    else:
      checks.append(_check_equal(f"stage{stage}_balance_smoothing_alpha", getattr(action, "balance_smoothing_alpha", None), 0.65))
      checks.append(_check_equal(f"stage{stage}_yaw_smoothing_alpha", getattr(action, "yaw_smoothing_alpha", None), 0.50))
      checks.append(_check_equal(f"stage{stage}_target_slew_limit", getattr(action, "target_slew_limit", None), 6.0))
      checks.append(_check_equal(f"stage{stage}_action_dim_kind", type(action).__name__, "DifferentialWheelVelocityActionCfg"))

  return checks


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--json", action="store_true", help="Print JSON-style Python dicts.")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  checks = run_preflight()
  failed = [check for check in checks if not check.passed]

  if args.json:
    for check in checks:
      print({"name": check.name, "pass": check.passed, "detail": check.detail})
  else:
    print("Stage2-5 pipeline preflight")
    for check in checks:
      print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")

  if failed:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
