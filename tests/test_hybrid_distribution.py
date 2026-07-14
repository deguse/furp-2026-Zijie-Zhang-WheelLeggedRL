import torch
from rsl_rl.modules.distribution import GaussianDistribution

from hoppertrex_mjlab.hybrid.distribution import MaskedGaussianDistribution


def _distribution() -> MaskedGaussianDistribution:
  distribution = MaskedGaussianDistribution(
    3,
    active_mask=(True, False, True),
    init_std=0.5,
    std_type="scalar",
  )
  distribution.update(torch.zeros((2, 3)))
  return distribution


def test_inactive_outputs_do_not_change_log_probability_or_entropy():
  distribution = _distribution()
  first = torch.tensor([[0.1, -100.0, 0.2], [0.3, 100.0, -0.2]])
  second = first.clone()
  second[:, 1] *= -3.0

  torch.testing.assert_close(
    distribution.log_prob(first),
    distribution.log_prob(second),
  )
  expected_entropy = torch.distributions.Normal(0.0, 0.5).entropy() * 2
  torch.testing.assert_close(
    distribution.entropy,
    torch.full((2,), expected_entropy),
  )


def test_inactive_outputs_do_not_change_masked_kl():
  distribution = _distribution()
  old_mean = torch.zeros((2, 3))
  old_std = torch.full((2, 3), 0.5)
  inactive_mean_only = old_mean.clone()
  inactive_mean_only[:, 1] = 50.0

  torch.testing.assert_close(
    distribution.kl_divergence(
      (old_mean, old_std),
      (inactive_mean_only, old_std),
    ),
    torch.zeros(2),
  )


def test_existing_gaussian_state_dict_loads_strictly():
  existing = GaussianDistribution(3, init_std=0.5, std_type="scalar")
  masked = _distribution()

  masked.load_state_dict(existing.state_dict(), strict=True)
  assert tuple(masked.state_dict()) == ("std_param",)
