"""Export a trained hybrid checkpoint to TorchScript for deployment (R3).

Builds a plain ``nn.Sequential`` mirror of the actor MLP (34 -> 128 -> ELU
-> 128 -> ELU -> 6), loads the ``actor_state_dict`` MLP weights, verifies
the mirror against the checkpoint weights, and traces it to TorchScript.
Deterministic inference equals the raw MLP output (the runner's
``deterministic_output`` for a Gaussian is the mean; there is no
observation normalizer in this project), so the traced module reproduces
play/gate inference exactly — the equivalence test pins this.

The sidecar metadata JSON records everything the deployed runtime needs to
refuse a mismatched pairing: observation layout, action names/mask/scales,
checkpoint SHA-256, training git SHA, and the five classical artifact
hashes the checkpoint was trained against.

The mjlab ONNX exporter is NOT used: its metadata helper assumes a
JointPositionAction term and is incompatible with the hybrid action term.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import torch

PROJECT_PATH = Path(__file__).resolve().parents[1]
SRC_PATH = Path(__file__).resolve().parents[2]
for path in (PROJECT_PATH, SRC_PATH):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

try:
  from hoppertrex_mjlab.hybrid.config import (
    HYBRID_ACTION_NAMES,
    HYBRID_STAGES,
  )
  from hoppertrex_mjlab.hybrid.observation_builder import (
    OBSERVATION_DIM,
    OBSERVATION_TERMS,
  )
except ImportError:
  from hybrid.config import (  # type: ignore[no-redef]
    HYBRID_ACTION_NAMES,
    HYBRID_STAGES,
  )
  from hybrid.observation_builder import (  # type: ignore[no-redef]
    OBSERVATION_DIM,
    OBSERVATION_TERMS,
  )

ACTOR_HIDDEN_DIMS = (128, 128)
ACTION_DIM = 6


def build_actor_module(hidden_dims: tuple[int, int] = ACTOR_HIDDEN_DIMS):
  return torch.nn.Sequential(
    torch.nn.Linear(OBSERVATION_DIM, hidden_dims[0]),
    torch.nn.ELU(),
    torch.nn.Linear(hidden_dims[0], hidden_dims[1]),
    torch.nn.ELU(),
    torch.nn.Linear(hidden_dims[1], ACTION_DIM),
  )


def load_actor_weights(
  module: torch.nn.Sequential,
  actor_state_dict: Mapping[str, torch.Tensor],
) -> None:
  """Copy the checkpoint's actor MLP weights into the plain mirror.

  Checkpoint keys are ``mlp.<layer>.weight``/``mlp.<layer>.bias`` with the
  same Sequential indices (0, 2, 4). Every expected key must exist and
  match shapes; unexpected MLP keys are rejected so an architecture drift
  cannot silently export garbage.
  """

  expected = {
    f"mlp.{index}.{kind}"
    for index in (0, 2, 4)
    for kind in ("weight", "bias")
  }
  mlp_keys = {
    key for key in actor_state_dict if key.startswith("mlp.")
  }
  if mlp_keys != expected:
    raise ValueError(
      f"Actor MLP keys {sorted(mlp_keys)} do not match the expected "
      f"34->128->128->6 layout {sorted(expected)}."
    )
  state = {
    key.removeprefix("mlp."): value
    for key, value in actor_state_dict.items()
    if key.startswith("mlp.")
  }
  module.load_state_dict(state, strict=True)


def export_metadata(
  *,
  checkpoint_path: Path,
  checkpoint: Mapping[str, object],
  stage: int,
) -> dict[str, object]:
  infos = checkpoint.get("infos")
  training = (
    infos.get("hybrid_training") if isinstance(infos, Mapping) else None
  )
  migration = (
    infos.get("hybrid_stage_migration")
    if isinstance(infos, Mapping)
    else None
  )
  bootstrap = (
    infos.get("hybrid_stage1_bootstrap")
    if isinstance(infos, Mapping)
    else None
  )
  stage_cfg = HYBRID_STAGES[stage]

  def _record(source: object, key: str) -> object:
    return source.get(key) if isinstance(source, Mapping) else None

  return {
    "schema_version": 1,
    "export": "hybrid_policy_torchscript",
    "checkpoint": str(checkpoint_path),
    "checkpoint_file_sha256": hashlib.sha256(
      checkpoint_path.read_bytes()
    ).hexdigest(),
    "training_git_sha": _record(training, "git_sha"),
    "stage": stage,
    "action_names": list(HYBRID_ACTION_NAMES),
    "action_mask": list(stage_cfg.action_mask),
    "action_scales": list(stage_cfg.action_scales),
    "observation_dim": OBSERVATION_DIM,
    "observation_terms": [
      {"name": name, "dim": dim} for name, dim in OBSERVATION_TERMS
    ],
    "inference": "deterministic mean (raw MLP output, no normalizer)",
    "artifact_hashes": {
      "controller_gain_hash": _record(bootstrap, "controller_gain_hash"),
      "calibration_hash": _record(bootstrap, "calibration_hash"),
      "yaw_calibration_hash": _record(migration, "yaw_calibration_hash"),
      "posture_map_hash": _record(migration, "posture_map_hash"),
      "station_calibration_hash": _record(
        migration, "station_calibration_hash"
      ),
    },
  }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint-file", type=Path, required=True)
  parser.add_argument("--stage", type=int, default=5, choices=range(6))
  parser.add_argument(
    "--output",
    type=Path,
    required=True,
    help="TorchScript output path; metadata JSON lands next to it.",
  )
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  checkpoint_path = args.checkpoint_file.expanduser().resolve()
  if not checkpoint_path.is_file():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
  checkpoint = torch.load(
    checkpoint_path, map_location="cpu", weights_only=False
  )
  if not isinstance(checkpoint, Mapping):
    raise ValueError("Checkpoint must contain a mapping.")
  actor_state = checkpoint.get("actor_state_dict")
  if not isinstance(actor_state, Mapping):
    raise ValueError("Checkpoint is missing actor_state_dict.")

  module = build_actor_module()
  load_actor_weights(module, actor_state)
  module.eval()
  example = torch.zeros(1, OBSERVATION_DIM)
  traced = torch.jit.trace(module, example)

  args.output.parent.mkdir(parents=True, exist_ok=True)
  traced.save(str(args.output))
  metadata = export_metadata(
    checkpoint_path=checkpoint_path,
    checkpoint=checkpoint,
    stage=args.stage,
  )
  metadata_path = args.output.with_suffix(".metadata.json")
  metadata_path.write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  print(f"[export] TorchScript: {args.output}")
  print(f"[export] metadata:    {metadata_path}")
  print(
    "[export] sha256:      "
    f"{metadata['checkpoint_file_sha256']}"
  )


if __name__ == "__main__":
  main()
