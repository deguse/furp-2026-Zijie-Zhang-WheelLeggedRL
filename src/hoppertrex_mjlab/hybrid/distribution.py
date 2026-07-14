"""Action-mask-aware policy distributions for Hybrid v2 training."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from rsl_rl.modules.distribution import GaussianDistribution
from torch.distributions import Normal


class MaskedGaussianDistribution(GaussianDistribution):
  """Exclude inactive Hybrid action heads from PPO probability statistics.

  Hybrid keeps a fixed six-action policy interface across every stage, while
  the environment masks actions that are not active yet.  A normal Gaussian
  still includes those masked heads in entropy, log probability, and KL,
  creating reward-independent gradients on outputs that cannot affect the
  environment.  This distribution samples all six outputs for checkpoint and
  interface compatibility, but PPO statistics include active heads only.
  """

  def __init__(
    self,
    output_dim: int,
    *,
    active_mask: Sequence[bool],
    **kwargs: object,
  ) -> None:
    super().__init__(output_dim, **kwargs)
    if len(active_mask) != output_dim:
      raise ValueError(
        "Hybrid distribution active_mask must match output_dim; "
        f"got {len(active_mask)} values for {output_dim} outputs."
      )
    mask = torch.tensor(tuple(active_mask), dtype=torch.bool)
    # The mask belongs to the task configuration, not learned checkpoint
    # state.  Keeping it non-persistent preserves strict compatibility with
    # existing Gaussian Stage1 bootstrap checkpoints.
    self.register_buffer("active_mask", mask, persistent=False)

  @property
  def entropy(self) -> torch.Tensor:
    return self._distribution.entropy()[..., self.active_mask].sum(dim=-1)  # type: ignore[union-attr]

  def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    values = self._distribution.log_prob(outputs)  # type: ignore[union-attr]
    return values[..., self.active_mask].sum(dim=-1)

  def kl_divergence(
    self,
    old_params: tuple[torch.Tensor, ...],
    new_params: tuple[torch.Tensor, ...],
  ) -> torch.Tensor:
    old_mean, old_std = old_params
    new_mean, new_std = new_params
    values = torch.distributions.kl_divergence(
      Normal(old_mean, old_std),
      Normal(new_mean, new_std),
    )
    return values[..., self.active_mask].sum(dim=-1)


__all__ = ["MaskedGaussianDistribution"]
