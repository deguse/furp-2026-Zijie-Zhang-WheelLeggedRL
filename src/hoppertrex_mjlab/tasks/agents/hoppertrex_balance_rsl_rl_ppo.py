"""RSL-RL configuration for HopperTrex balance."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


HYBRID_DISTRIBUTION_CLASS = (
  "hoppertrex_mjlab.hybrid.distribution.MaskedGaussianDistribution"
)


def hoppertrex_balance_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(128, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.6,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(128, 128),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="hoppertrex_balance",
    logger="tensorboard",
    upload_model=False,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=5000,
  )


def hoppertrex_hybrid_ppo_runner_cfg(
  active_mask: tuple[bool, ...],
) -> RslRlOnPolicyRunnerCfg:
  """Return PPO config whose statistics ignore inactive Hybrid actions."""

  cfg = hoppertrex_balance_ppo_runner_cfg()
  assert cfg.actor.distribution_cfg is not None
  cfg.actor.distribution_cfg["class_name"] = HYBRID_DISTRIBUTION_CLASS
  cfg.actor.distribution_cfg["active_mask"] = tuple(active_mask)
  return cfg
