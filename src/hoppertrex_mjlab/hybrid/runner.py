"""RSL-RL runner safeguards for the Hybrid v2 checkpoint chain."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any, Mapping

from mjlab.rl import MjlabOnPolicyRunner


REPOSITORY_PATH = Path(__file__).resolve().parents[3]


def repository_git_sha() -> str:
  """Return the code revision used by the current training process."""

  completed = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=REPOSITORY_PATH,
    check=True,
    capture_output=True,
    text=True,
  )
  return completed.stdout.strip()


def merge_hybrid_checkpoint_infos(
  loaded_infos: Mapping[str, Any] | None,
  save_infos: Mapping[str, Any] | None,
) -> dict[str, Any]:
  """Preserve bootstrap/migration provenance across later checkpoint saves."""

  return {**dict(loaded_infos or {}), **dict(save_infos or {})}


def hybrid_action_scales_from_env(env: object) -> list[float]:
  """Read the six applied scales from the real wrapped environment config."""

  unwrapped = getattr(env, "unwrapped", env)
  env_cfg = getattr(unwrapped, "cfg", None)
  actions = getattr(env_cfg, "actions", {})
  action = actions.get("hybrid_wheel_leg") if isinstance(actions, dict) else None
  scales = getattr(action, "action_scales", ())
  if len(scales) != 6:
    raise ValueError("Hybrid runner expected six environment action scales.")
  return [float(value) for value in scales]


class HybridOnPolicyRunner(MjlabOnPolicyRunner):
  """MjLab runner that retains Hybrid checkpoint provenance after resume."""

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    self._hybrid_loaded_infos: dict[str, Any] = {}
    self._hybrid_training_git_sha = repository_git_sha()
    env = args[0] if args else kwargs.get("env")
    self._hybrid_action_scales = hybrid_action_scales_from_env(env)
    super().__init__(*args, **kwargs)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    infos = super().load(path, load_cfg, strict, map_location)
    self._hybrid_loaded_infos = dict(infos or {})
    return infos

  def save(self, path: str, infos: dict | None = None) -> None:
    current_infos = {
      **dict(infos or {}),
      "hybrid_training": {
        "git_sha": self._hybrid_training_git_sha,
        "action_scales": self._hybrid_action_scales,
      },
    }
    merged = merge_hybrid_checkpoint_infos(
      self._hybrid_loaded_infos,
      current_infos,
    )
    super().save(path, merged)


__all__ = [
  "HybridOnPolicyRunner",
  "hybrid_action_scales_from_env",
  "merge_hybrid_checkpoint_infos",
  "repository_git_sha",
]
