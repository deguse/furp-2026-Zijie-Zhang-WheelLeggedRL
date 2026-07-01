#!/usr/bin/env python3
"""Create a 2D turn checkpoint with the yaw output head reset to neutral."""

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

DEFAULT_EXPERIMENT_NAME = "hoppertrex_balance"
DEFAULT_TARGET_TASK = "Mjlab-HopperTrex-Balance-SlowSpeedTurn-Sign-ObsScale-v0"
DEFAULT_SOURCE_RUN_PATTERN = "slow_speed_turn_probe_seed{seed}"
DEFAULT_OUTPUT_RUN_PATTERN = "reset_yaw_head_sign_obs_scale_seed{seed}"
YAW_ACTION_INDEX = 1


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

  pattern = (source_run or DEFAULT_SOURCE_RUN_PATTERN).format(seed=seed)
  candidates = sorted(
    [path for path in log_dir.iterdir() if path.is_dir() and pattern in path.name],
    key=lambda path: path.stat().st_mtime,
    reverse=True,
  )
  if not candidates:
    raise FileNotFoundError(
      "Could not find source checkpoint. Checked run-name pattern: "
      f"{pattern}. Pass --source-checkpoint explicitly if needed."
    )
  checkpoint = _find_latest_checkpoint(candidates[0])
  return checkpoint, candidates[0].name


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


def _create_fresh_target_checkpoint(
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
  env.close()

  checkpoint["iter"] = 0
  checkpoint["infos"] = {}
  return checkpoint, output_dir


def _validate_actor_shapes(source_actor: dict[str, torch.Tensor], target_actor: dict[str, torch.Tensor]) -> None:
  expected = {
    "mlp.0.weight": (128, 26),
    "mlp.0.bias": (128,),
    "mlp.2.weight": (128, 128),
    "mlp.2.bias": (128,),
    "mlp.4.weight": (2, 128),
    "mlp.4.bias": (2,),
  }
  failures: list[str] = []
  for key, shape in expected.items():
    if key not in source_actor:
      failures.append(f"missing source actor key {key}")
    elif tuple(source_actor[key].shape) != shape:
      failures.append(
        f"source actor {key}: got {tuple(source_actor[key].shape)}, expected {shape}"
      )
    if key not in target_actor:
      failures.append(f"missing target actor key {key}")
    elif tuple(target_actor[key].shape) != shape:
      failures.append(
        f"target actor {key}: got {tuple(target_actor[key].shape)}, expected {shape}"
      )

  std_key = "distribution.std_param"
  if std_key in source_actor and tuple(source_actor[std_key].shape) != (2,):
    failures.append(
      f"source actor {std_key}: got {tuple(source_actor[std_key].shape)}, expected (2,)"
    )
  if std_key in target_actor and tuple(target_actor[std_key].shape) != (2,):
    failures.append(
      f"target actor {std_key}: got {tuple(target_actor[std_key].shape)}, expected (2,)"
    )

  if failures:
    raise ValueError(
      "Unexpected actor checkpoint shape for yaw-head reset:\n  - "
      + "\n  - ".join(failures)
    )


def _copy_actor_with_reset_yaw_head(
  source_actor: dict[str, torch.Tensor],
  target_actor: dict[str, torch.Tensor],
  yaw_std: float,
) -> tuple[dict[str, torch.Tensor], list[str]]:
  _validate_actor_shapes(source_actor, target_actor)

  actor = {key: value.clone() for key, value in target_actor.items()}
  report: list[str] = []

  for key in ("mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias"):
    actor[key] = source_actor[key].to(
      device=target_actor[key].device,
      dtype=target_actor[key].dtype,
    ).clone()
    report.append(f"copied hidden actor parameter {key}")

  actor["mlp.4.weight"] = target_actor["mlp.4.weight"].clone()
  actor["mlp.4.bias"] = target_actor["mlp.4.bias"].clone()
  actor["mlp.4.weight"][0, :] = source_actor["mlp.4.weight"][0, :].to(
    device=actor["mlp.4.weight"].device,
    dtype=actor["mlp.4.weight"].dtype,
  )
  actor["mlp.4.bias"][0] = source_actor["mlp.4.bias"][0].to(
    device=actor["mlp.4.bias"].device,
    dtype=actor["mlp.4.bias"].dtype,
  )
  actor["mlp.4.weight"][YAW_ACTION_INDEX, :] = 0.0
  actor["mlp.4.bias"][YAW_ACTION_INDEX] = 0.0
  report.append("copied action[0] output row and zeroed action[1] yaw output row")

  std_key = "distribution.std_param"
  if std_key in actor and std_key in source_actor:
    actor[std_key] = target_actor[std_key].clone()
    actor[std_key][0] = source_actor[std_key][0].to(
      device=actor[std_key].device,
      dtype=actor[std_key].dtype,
    )
    actor[std_key][YAW_ACTION_INDEX] = yaw_std
    report.append(
      f"copied action[0] std and set action[1] yaw std to {yaw_std:.3f}"
    )
  elif std_key in actor:
    actor[std_key][YAW_ACTION_INDEX] = yaw_std
    report.append(f"set target action[1] yaw std to {yaw_std:.3f}")
  else:
    report.append("no distribution.std_param found; skipped std reset")

  return actor, report


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--target-task", default=DEFAULT_TARGET_TASK)
  parser.add_argument("--source-run", default=None)
  parser.add_argument("--source-checkpoint", type=Path, default=None)
  parser.add_argument("--output-run", default=None)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--yaw-std", type=float, default=1.0)
  parser.add_argument(
    "--log-dir",
    type=Path,
    default=PROJECT_PATH / "logs" / "rsl_rl" / DEFAULT_EXPERIMENT_NAME,
  )
  parser.add_argument(
    "--force",
    action="store_true",
    help="Overwrite output run directory if it already exists.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  output_run = args.output_run or DEFAULT_OUTPUT_RUN_PATTERN.format(seed=args.seed)
  log_dir: Path = args.log_dir.resolve()
  output_dir = log_dir / output_run
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

  target, output_dir = _create_fresh_target_checkpoint(
    target_task=args.target_task,
    log_dir=log_dir,
    output_run=output_run,
    device=args.device,
    seed=args.seed,
  )

  target["actor_state_dict"], actor_report = _copy_actor_with_reset_yaw_head(
    source["actor_state_dict"],
    target["actor_state_dict"],
    yaw_std=args.yaw_std,
  )
  target["iter"] = 0
  target["infos"] = {
    "reset_yaw_head": {
      "created_at": datetime.now().isoformat(timespec="seconds"),
      "target_task": args.target_task,
      "source_run": source_run,
      "source_checkpoint": str(source_checkpoint),
      "seed": args.seed,
      "yaw_action_index": YAW_ACTION_INDEX,
      "yaw_std": args.yaw_std,
      "critic": "fresh_target",
      "optimizer": "fresh_target",
    }
  }

  output_checkpoint = output_dir / "model_0.pt"
  torch.save(target, output_checkpoint)

  report_path = output_dir / "reset_yaw_head_report.txt"
  report_path.write_text(
    "\n".join(
      [
        f"target_task={args.target_task}",
        f"source_run={source_run}",
        f"source_checkpoint={source_checkpoint}",
        f"output_checkpoint={output_checkpoint}",
        f"yaw_std={args.yaw_std}",
        "",
        "[actor]",
        *actor_report,
        "",
        "[critic]",
        "kept fresh target critic",
        "",
        "[optimizer]",
        "kept fresh target optimizer state",
        "",
      ]
    ),
    encoding="utf-8",
  )

  print(f"[OK] Wrote reset checkpoint: {output_checkpoint}")
  print(f"[OK] Wrote reset report: {report_path}")
  print("[NEXT] Use --agent.resume True with:")
  print(f'       --agent.load-run ".*{output_run}.*"')
  print('       --agent.load-checkpoint "model_0.pt"')


if __name__ == "__main__":
  main()
