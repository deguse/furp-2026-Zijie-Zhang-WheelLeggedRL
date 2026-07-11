#!/usr/bin/env python3
"""Create a fresh, zero-residual Hybrid v2 Stage1 training checkpoint."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch

PROJECT_PATH = Path(__file__).resolve().parents[2]
SRC_PATH = Path(__file__).resolve().parents[3]
REPOSITORY_PATH = Path(__file__).resolve().parents[4]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from hoppertrex_mjlab.hybrid.calibration import (  # noqa: E402
  parse_calibration_artifact,
)
from hoppertrex_mjlab.hybrid.config import (  # noqa: E402
  HYBRID_ACTION_NAMES,
  HYBRID_ACTION_STD,
)


STAGE1_TASK = "HopperTrex-Hybrid-v2-Stage1"
STAGE1_ACTION_STD = HYBRID_ACTION_STD
DEFAULT_EXPERIMENT_NAME = "hoppertrex_balance"


def _action_head_keys(actor_state: Mapping[str, torch.Tensor]) -> tuple[str, str]:
  candidates: list[tuple[str, str]] = []
  for key, value in actor_state.items():
    if not key.endswith(".weight") or value.ndim != 2 or value.shape[0] != 6:
      continue
    bias_key = f"{key[:-len('.weight')]}.bias"
    bias = actor_state.get(bias_key)
    if bias is not None and bias.shape == (6,):
      candidates.append((key, bias_key))
  if len(candidates) != 1:
    raise ValueError(
      "Expected exactly one six-row actor output head, "
      f"found {len(candidates)} candidates."
    )
  return candidates[0]


def qualified_controller_provenance(path: Path) -> dict[str, str]:
  """Validate an identified controller and return immutable provenance."""

  controller_path = path.expanduser().resolve()
  from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import _load_controller

  controller = _load_controller(controller_path)
  if not controller.qualified:
    raise ValueError(
      "Stage1 bootstrap requires a qualified controller artifact; "
      f"got {controller.controller_type.upper()} from {controller_path}."
    )
  if controller.gain_hash is None:
    raise ValueError("Qualified controller artifact has no gain hash.")
  return {
    "path": str(controller_path),
    "gain_hash": controller.gain_hash,
    "file_sha256": hashlib.sha256(controller_path.read_bytes()).hexdigest(),
    "controller_type": controller.controller_type,
  }


def calibration_provenance(
  path: Path,
  *,
  controller_gain_hash: str,
) -> dict[str, object]:
  '''Validate velocity calibration and bind it to the controller artifact.'''

  calibration_path = path.expanduser().resolve()
  payload = json.loads(calibration_path.read_text(encoding='utf-8'))
  if not isinstance(payload, Mapping):
    raise ValueError('Calibration artifact must contain a JSON object.')
  calibration = parse_calibration_artifact(
    payload,
    controller_gain_hash=controller_gain_hash,
  )
  return {
    'path': str(calibration_path),
    'calibration_hash': calibration.calibration_hash,
    'file_sha256': hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
    'controller_gain_hash': calibration.controller_gain_hash,
    'velocity_command_scale': calibration.scale,
    'velocity_command_bias': calibration.bias,
  }


def bootstrap_stage1_checkpoint(
  checkpoint: Mapping[str, Any],
  *,
  task: str,
  seed: int,
  controller_provenance: Mapping[str, str],
  calibration_provenance: Mapping[str, object],
  git_sha: str,
  created_at: str,
) -> dict[str, Any]:
  """Initialize all six actor outputs and optimizer state for Stage1."""

  if "actor_state_dict" not in checkpoint:
    raise ValueError("Fresh checkpoint has no actor_state_dict.")
  if "optimizer_state_dict" not in checkpoint:
    raise ValueError("Fresh checkpoint has no optimizer_state_dict.")
  result = deepcopy(dict(checkpoint))
  actor_state = result["actor_state_dict"]
  weight_key, bias_key = _action_head_keys(actor_state)
  actor_state[weight_key].zero_()
  actor_state[bias_key].zero_()

  std_key = "distribution.std_param"
  std = actor_state.get(std_key)
  if std is None or std.shape != (6,):
    raise ValueError(f"{std_key} must contain six action values.")
  actor_state[std_key] = torch.tensor(
    STAGE1_ACTION_STD,
    device=std.device,
    dtype=std.dtype,
  )

  optimizer = result["optimizer_state_dict"]
  if not isinstance(optimizer, dict) or "state" not in optimizer:
    raise ValueError("Fresh optimizer_state_dict has no state mapping.")
  optimizer["state"] = {}
  result["iter"] = 0
  result["infos"] = {
    "hybrid_stage1_bootstrap": {
      "created_at": created_at,
      "task": task,
      "stage": 1,
      "seed": seed,
      "git_sha": git_sha,
      "controller_path": controller_provenance["path"],
      "controller_gain_hash": controller_provenance["gain_hash"],
      "controller_file_sha256": controller_provenance["file_sha256"],
      "controller_type": controller_provenance["controller_type"],
      "action_std": list(STAGE1_ACTION_STD),
      'calibration_path': calibration_provenance['path'],
      'calibration_hash': calibration_provenance['calibration_hash'],
      'calibration_file_sha256': calibration_provenance['file_sha256'],
      'velocity_command_scale': calibration_provenance['velocity_command_scale'],
      'velocity_command_bias': calibration_provenance['velocity_command_bias'],
      'action_order': list(HYBRID_ACTION_NAMES),
    }
  }
  return result


def _git_sha() -> str:
  completed = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=REPOSITORY_PATH,
    check=True,
    capture_output=True,
    text=True,
  )
  return completed.stdout.strip()


def _fresh_runner_checkpoint(
  *,
  controller_path: Path,
  calibration_path: Path,
  output_dir: Path,
  device: str,
  seed: int,
) -> dict[str, Any]:
  import hoppertrex_mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
  from hoppertrex_mjlab.tasks.hoppertrex_hybrid_task import (
    make_hoppertrex_hybrid_env_cfg,
  )

  env_cfg = make_hoppertrex_hybrid_env_cfg(
    stage=1,
    play=False,
    controller_path=controller_path,
    calibration_path=calibration_path,
  )
  action_cfg = env_cfg.actions["hybrid_wheel_leg"]
  if not action_cfg.controller_qualified:
    raise ValueError("Stage1 environment did not load a qualified controller.")
  if action_cfg.calibration_hash is None:
    raise ValueError('Stage1 environment did not load velocity calibration.')
  env_cfg.scene.num_envs = 1
  if env_cfg.scene.terrain is not None:
    env_cfg.scene.terrain.num_envs = 1
  env_cfg.seed = seed

  rl_cfg = load_rl_cfg(STAGE1_TASK)
  rl_cfg.seed = seed
  rl_cfg.upload_model = False
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  try:
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
    runner_cls = load_runner_cls(STAGE1_TASK) or MjlabOnPolicyRunner
    runner = runner_cls(
      wrapped_env,
      asdict(rl_cfg),
      str(output_dir),
      device,
    )
    return runner.alg.save()
  finally:
    env.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--calibration-path', type=Path, required=True)
  parser.add_argument("--controller-path", type=Path, required=True)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--device", default="cpu")
  parser.add_argument(
    "--log-dir",
    type=Path,
    default=PROJECT_PATH / "logs" / "rsl_rl" / DEFAULT_EXPERIMENT_NAME,
  )
  parser.add_argument("--output-run", default=None)
  parser.add_argument(
    "--force",
    action="store_true",
    help="Overwrite model_0.pt and its provenance file in an existing run.",
  )
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  output_run = (
    args.output_run
    if args.output_run is not None
    else f"hybrid_v2_stage1_bootstrap_seed{args.seed}"
  )
  log_dir = args.log_dir.expanduser().resolve()
  output_dir = (log_dir / output_run).resolve()
  if output_dir.parent != log_dir:
    raise ValueError("--output-run must be one directory name inside --log-dir.")
  output_checkpoint = output_dir / "model_0.pt"
  provenance_path = output_dir / "bootstrap_provenance.json"
  existing = [path for path in (output_checkpoint, provenance_path) if path.exists()]
  if existing and not args.force:
    raise FileExistsError(
      f"Bootstrap output already exists: {existing[0]}. Use --force to overwrite."
    )

  controller = qualified_controller_provenance(args.controller_path)
  calibration = calibration_provenance(
    args.calibration_path,
    controller_gain_hash=str(controller['gain_hash']),
  )
  output_dir.mkdir(parents=True, exist_ok=True)
  fresh = _fresh_runner_checkpoint(
    calibration_path=Path(str(calibration['path'])),
    controller_path=Path(controller["path"]),
    output_dir=output_dir,
    device=args.device,
    seed=args.seed,
  )
  created_at = datetime.now().astimezone().isoformat(timespec="seconds")
  checkpoint = bootstrap_stage1_checkpoint(
    fresh,
    calibration_provenance=calibration,
    task=STAGE1_TASK,
    seed=args.seed,
    controller_provenance=controller,
    git_sha=_git_sha(),
    created_at=created_at,
  )
  torch.save(checkpoint, output_checkpoint)
  provenance = checkpoint["infos"]["hybrid_stage1_bootstrap"]
  provenance_path.write_text(
    json.dumps(provenance, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )

  print(f"[OK] Wrote Stage1 bootstrap: {output_checkpoint}")
  print(f"[OK] Wrote provenance: {provenance_path}")
  print("[NEXT] Resume Stage1 training with:")
  print("       --agent.resume True")
  print(f'       --agent.load-run ".*{output_run}.*"')
  print('       --agent.load-checkpoint "model_0.pt"')


if __name__ == "__main__":
  main()
