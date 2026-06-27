#!/usr/bin/env python3
"""Warm-start a 2D turning policy from a trained 1D balance checkpoint."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

PROJECT_PATH = Path(__file__).resolve().parents[2]
if str(PROJECT_PATH) not in sys.path:
  sys.path.insert(0, str(PROJECT_PATH))

import tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

DEFAULT_TARGET_TASK = "Mjlab-HopperTrex-Balance-Turn-L4-Track-v2"
DEFAULT_EXPERIMENT_NAME = "hoppertrex_balance"
DEFAULT_SOURCE_RUN_PATTERNS = (
  "push_l3_seed{seed}",
  "robust_l2_seed{seed}",
  "robust_init_seed{seed}",
  "clean_wheel_seed{seed}",
  "slow_speed_seed{seed}",
)


def _checkpoint_iteration(path: Path) -> int:
  match = re.fullmatch(r"model_(\d+)\.pt", path.name)
  return int(match.group(1)) if match else -1


def _find_latest_checkpoint(run_dir: Path) -> Path:
  checkpoints = sorted(
    run_dir.glob("model_*.pt"),
    key=lambda path: (_checkpoint_iteration(path), path.stat().st_mtime),
    reverse=True,
  )
  if not checkpoints:
    raise FileNotFoundError(f"No model_*.pt checkpoints found in: {run_dir}")
  return checkpoints[0]


def _find_source_checkpoint(
  log_dir: Path,
  seed: int,
  source_checkpoint: Path | None,
  source_run: str | None,
) -> tuple[Path, str]:
  if source_checkpoint is not None:
    if not source_checkpoint.exists():
      raise FileNotFoundError(f"Source checkpoint not found: {source_checkpoint}")
    return source_checkpoint, source_checkpoint.parent.name

  if not log_dir.exists():
    raise FileNotFoundError(f"Log directory not found: {log_dir}")

  patterns = [source_run] if source_run else list(DEFAULT_SOURCE_RUN_PATTERNS)
  checked: list[str] = []
  for pattern_template in patterns:
    assert pattern_template is not None
    pattern = pattern_template.format(seed=seed)
    checked.append(pattern)
    candidates = sorted(
      [path for path in log_dir.iterdir() if path.is_dir() and pattern in path.name],
      key=lambda path: path.stat().st_mtime,
      reverse=True,
    )
    if not candidates:
      continue
    checkpoint = _find_latest_checkpoint(candidates[0])
    return checkpoint, candidates[0].name

  checked_text = ", ".join(checked)
  raise FileNotFoundError(
    "Could not find a source checkpoint. Checked run-name patterns: "
    f"{checked_text}. Pass --source-checkpoint explicitly if the run has a "
    "different name."
  )


def _normalize_checkpoint_state_dicts(checkpoint: dict[str, Any]) -> None:
  """Normalize legacy/current MjLab checkpoint formats in-place."""
  if "model_state_dict" in checkpoint:
    model_state_dict = checkpoint.pop("model_state_dict")
    actor_state_dict: dict[str, torch.Tensor] = {}
    critic_state_dict: dict[str, torch.Tensor] = {}

    for key, value in model_state_dict.items():
      if key.startswith("actor."):
        actor_state_dict[key.replace("actor.", "mlp.")] = value
      elif key.startswith("actor_obs_normalizer."):
        actor_state_dict[key.replace("actor_obs_normalizer.", "obs_normalizer.")] = (
          value
        )
      elif key in ("std", "log_std"):
        actor_state_dict[key] = value

      if key.startswith("critic."):
        critic_state_dict[key.replace("critic.", "mlp.")] = value
      elif key.startswith("critic_obs_normalizer."):
        critic_state_dict[key.replace("critic_obs_normalizer.", "obs_normalizer.")] = (
          value
        )

    checkpoint["actor_state_dict"] = actor_state_dict
    checkpoint["critic_state_dict"] = critic_state_dict

  actor_sd = checkpoint.get("actor_state_dict", {})
  if "std" in actor_sd:
    actor_sd["distribution.std_param"] = actor_sd.pop("std")
  if "log_std" in actor_sd:
    actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")


def _copy_expand_rows_cols(
  source: torch.Tensor,
  target: torch.Tensor,
  zero_new_values: bool = False,
) -> tuple[torch.Tensor, str]:
  migrated = torch.zeros_like(target) if zero_new_values else target.clone()

  if source.ndim == 0 or target.ndim == 0:
    if source.shape == target.shape:
      return source.clone(), "exact"
    return migrated, "skipped-scalar-shape"

  slices = tuple(slice(0, min(src, dst)) for src, dst in zip(source.shape, target.shape))
  migrated[slices] = source[slices].to(device=target.device, dtype=target.dtype)
  return migrated, f"expanded {tuple(source.shape)} -> {tuple(target.shape)}"


def _migrate_state_dict(
  source_sd: dict[str, torch.Tensor],
  target_sd: dict[str, torch.Tensor],
  label: str,
) -> tuple[dict[str, torch.Tensor], list[str]]:
  migrated: dict[str, torch.Tensor] = {}
  report: list[str] = []

  for key, target_value in target_sd.items():
    source_value = source_sd.get(key)
    if source_value is None:
      migrated[key] = target_value.clone()
      report.append(f"{label}: keep target init for missing key {key}")
      continue

    if source_value.shape == target_value.shape:
      migrated[key] = source_value.to(
        device=target_value.device,
        dtype=target_value.dtype,
      ).clone()
      report.append(f"{label}: copied exact {key} {tuple(target_value.shape)}")
      continue

    zero_new_values = label == "actor" and key.startswith("mlp.4.")
    new_value, action = _copy_expand_rows_cols(
      source_value,
      target_value,
      zero_new_values=zero_new_values,
    )
    migrated[key] = new_value
    suffix = " with zero new actor outputs" if zero_new_values else ""
    report.append(f"{label}: {action}{suffix} for {key}")

  return migrated, report


def _validate_expected_1d_to_2d_shapes(
  source: dict[str, Any],
  target: dict[str, Any],
  allow_unexpected: bool,
) -> None:
  checks = (
    ("source actor obs", source["actor_state_dict"]["mlp.0.weight"].shape[1], 25),
    ("source critic obs", source["critic_state_dict"]["mlp.0.weight"].shape[1], 25),
    ("source actor action", source["actor_state_dict"]["mlp.4.weight"].shape[0], 1),
    ("target actor obs", target["actor_state_dict"]["mlp.0.weight"].shape[1], 26),
    ("target critic obs", target["critic_state_dict"]["mlp.0.weight"].shape[1], 26),
    ("target actor action", target["actor_state_dict"]["mlp.4.weight"].shape[0], 2),
  )
  failed = [
    f"{label}: got {actual}, expected {expected}"
    for label, actual, expected in checks
    if actual != expected
  ]
  if failed and not allow_unexpected:
    joined = "\n  - ".join(failed)
    raise ValueError(
      "Unexpected checkpoint shape for 1D-balance -> 2D-turn migration:\n"
      f"  - {joined}\n"
      "This usually means the source checkpoint is not from the current fixed-leg "
      "1D wheel-balance task. Pass --source-checkpoint for the correct model, or "
      "use --allow-unexpected-source-shapes only for debugging."
    )
  if failed:
    print("[WARN] Continuing despite unexpected source/target shapes:")
    for item in failed:
      print(f"       {item}")


def _create_target_checkpoint(
  target_task: str,
  log_dir: Path,
  output_run: str,
  device: str,
  seed: int,
) -> tuple[dict[str, Any], Path]:
  env_cfg = load_env_cfg(target_task)
  rl_cfg = load_rl_cfg(target_task)

  env_cfg.scene.num_envs = 1
  env_cfg.seed = seed
  rl_cfg.seed = seed
  rl_cfg.upload_model = False

  output_dir = log_dir / output_run
  output_dir.mkdir(parents=True, exist_ok=True)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
  runner_cls = load_runner_cls(target_task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped_env, asdict(rl_cfg), str(output_dir), device)
  checkpoint = runner.alg.save()
  checkpoint["iter"] = 0
  checkpoint["infos"] = {
    "migration": {
      "created_at": datetime.now().isoformat(timespec="seconds"),
      "target_task": target_task,
      "seed": seed,
    }
  }
  env.close()
  return checkpoint, output_dir


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Create a 2D Turn-L4 checkpoint by expanding a trained 1D balance policy."
    )
  )
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--target-task", default=DEFAULT_TARGET_TASK)
  parser.add_argument("--output-run", default=None)
  parser.add_argument("--device", default="cpu")
  parser.add_argument(
    "--log-dir",
    type=Path,
    default=PROJECT_PATH / "logs" / "rsl_rl" / DEFAULT_EXPERIMENT_NAME,
  )
  parser.add_argument(
    "--source-checkpoint",
    type=Path,
    default=None,
    help="Optional explicit 1D source checkpoint path.",
  )
  parser.add_argument(
    "--source-run",
    default=None,
    help=(
      "Optional source run-name substring. Supports {seed}, for example "
      "push_l3_seed{seed}."
    ),
  )
  parser.add_argument(
    "--force",
    action="store_true",
    help="Overwrite output run directory if it already exists.",
  )
  parser.add_argument(
    "--allow-unexpected-source-shapes",
    action="store_true",
    help="Bypass the default 25-observation/1-action source checkpoint check.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if args.output_run is None:
    args.output_run = f"migrated_turn_l4_track_v2_seed{args.seed}"
  log_dir: Path = args.log_dir.resolve()
  output_dir = log_dir / args.output_run
  if output_dir.exists():
    if not args.force:
      raise FileExistsError(
        f"Output run already exists: {output_dir}. Use --force to overwrite."
      )
    shutil.rmtree(output_dir)

  source_checkpoint, source_run = _find_source_checkpoint(
    log_dir=log_dir,
    seed=args.seed,
    source_checkpoint=args.source_checkpoint,
    source_run=args.source_run,
  )
  source_checkpoint = source_checkpoint.resolve()
  print(f"[INFO] Source run: {source_run}")
  print(f"[INFO] Source checkpoint: {source_checkpoint}")

  source = torch.load(source_checkpoint, map_location=args.device, weights_only=False)
  _normalize_checkpoint_state_dicts(source)

  target, output_dir = _create_target_checkpoint(
    target_task=args.target_task,
    log_dir=log_dir,
    output_run=args.output_run,
    device=args.device,
    seed=args.seed,
  )
  try:
    _validate_expected_1d_to_2d_shapes(
      source,
      target,
      allow_unexpected=args.allow_unexpected_source_shapes,
    )
  except Exception:
    if output_dir.exists():
      shutil.rmtree(output_dir)
    raise

  target["actor_state_dict"], actor_report = _migrate_state_dict(
    source["actor_state_dict"],
    target["actor_state_dict"],
    "actor",
  )
  target["critic_state_dict"], critic_report = _migrate_state_dict(
    source["critic_state_dict"],
    target["critic_state_dict"],
    "critic",
  )
  target["iter"] = 0
  target["infos"] = {
    "migration": {
      "created_at": datetime.now().isoformat(timespec="seconds"),
      "target_task": args.target_task,
      "source_run": source_run,
      "source_checkpoint": str(source_checkpoint),
      "seed": args.seed,
    }
  }

  output_checkpoint = output_dir / "model_0.pt"
  torch.save(target, output_checkpoint)

  report_path = output_dir / "migration_report.txt"
  report_path.write_text(
    "\n".join(
      [
        f"target_task={args.target_task}",
        f"source_run={source_run}",
        f"source_checkpoint={source_checkpoint}",
        f"output_checkpoint={output_checkpoint}",
        "",
        "[actor]",
        *actor_report,
        "",
        "[critic]",
        *critic_report,
        "",
      ]
    ),
    encoding="utf-8",
  )

  print(f"[OK] Wrote migrated checkpoint: {output_checkpoint}")
  print(f"[OK] Wrote migration report: {report_path}")
  print("[NEXT] Use --agent.resume True with:")
  print(f'       --agent.load-run ".*{args.output_run}.*"')
  print('       --agent.load-checkpoint "model_0.pt"')


if __name__ == "__main__":
  main()
