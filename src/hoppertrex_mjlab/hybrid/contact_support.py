"""Contact-support predicates shared by RollBoundary and RollAssist."""

from __future__ import annotations

import torch


def wheel_supported_during_control_interval(
  found: torch.Tensor | None,
  force_history: torch.Tensor | None,
) -> torch.Tensor:
  """Return per-env wheel support over one complete controller interval.

  MjLab exposes ``found`` only at the final physics substep of a 20 ms
  controller interval. A rolling wheel can therefore report ``found=False``
  at that single sample even though it carried force during another one of the
  four 5 ms physics substeps. Safety uses the complete substep force history
  rather than inventing an N-controller-step grace period.
  """

  if found is None:
    raise RuntimeError("Wheel contact sensor exposes no found field.")
  if force_history is None:
    raise RuntimeError("Wheel contact sensor exposes no substep force history.")
  if found.ndim < 2:
    raise ValueError("Wheel found data must have a batch and contact dimension.")
  if (
    force_history.ndim != 4
    or force_history.shape[0] != found.shape[0]
    or force_history.shape[-1] != 3
  ):
    raise ValueError("Wheel force history must have shape [B, N, H, 3].")
  current = torch.any(found.reshape(found.shape[0], -1) > 0, dim=-1)
  historical_force = torch.linalg.vector_norm(force_history, dim=-1) > 0.0
  historical = torch.any(
    historical_force.reshape(historical_force.shape[0], -1), dim=-1
  )
  return current | historical
