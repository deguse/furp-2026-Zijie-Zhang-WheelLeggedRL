"""RSL-RL runner safeguards for the Hybrid v2 checkpoint chain."""

from __future__ import annotations

from typing import Any, Mapping

from mjlab.rl import MjlabOnPolicyRunner


def merge_hybrid_checkpoint_infos(
  loaded_infos: Mapping[str, Any] | None,
  save_infos: Mapping[str, Any] | None,
) -> dict[str, Any]:
  """Preserve bootstrap/migration provenance across later checkpoint saves."""

  return {**dict(loaded_infos or {}), **dict(save_infos or {})}


class HybridOnPolicyRunner(MjlabOnPolicyRunner):
  """MjLab runner that retains Hybrid checkpoint provenance after resume."""

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    self._hybrid_loaded_infos: dict[str, Any] = {}
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
    merged = merge_hybrid_checkpoint_infos(self._hybrid_loaded_infos, infos)
    super().save(path, merged)


__all__ = ["HybridOnPolicyRunner", "merge_hybrid_checkpoint_infos"]
