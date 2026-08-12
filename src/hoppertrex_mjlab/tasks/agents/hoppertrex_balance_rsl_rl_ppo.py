"""RSL-RL configuration for HopperTrex balance."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from hoppertrex_mjlab.hybrid.config import STAIR_CAMP_ACTION_MASK
from hoppertrex_mjlab.hybrid.roll_assist import (
  ROLL_ASSIST_ACTION_MASK,
  ROLL_ASSIST_INITIAL_UPDATES,
  ROLL_ASSIST_SAVE_INTERVAL,
  ROLL_ASSIST_STEPS_PER_UPDATE,
)
from hoppertrex_mjlab.hybrid.stair_dynamic_contract import (
  DYNAMIC_STAIR_ACTION_MASK,
  DYNAMIC_STAIR_PROBE_UPDATES,
  DYNAMIC_STAIR_SAVE_INTERVAL,
  DYNAMIC_STAIR_STEPS_PER_ITERATION,
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


def hoppertrex_stair_camp_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Return the preregistered S5B runner defaults without CLI pinning."""

  cfg = hoppertrex_hybrid_ppo_runner_cfg(STAIR_CAMP_ACTION_MASK)
  cfg.experiment_name = "hoppertrex_stair_camp_s5b"
  cfg.num_steps_per_env = 24
  cfg.max_iterations = 1000
  cfg.save_interval = 100
  return cfg


def hoppertrex_stair_camp_lqr_alpha05_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Return the isolated seed-1 alpha=0.5 failure-rung runner defaults."""

  cfg = hoppertrex_stair_camp_ppo_runner_cfg()
  cfg.experiment_name = "hoppertrex_stair_camp_s5b_lqr_alpha05"
  return cfg


def hoppertrex_stair_dynamic_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Return Hybrid-v3 seed-1 probe defaults (100 total updates)."""

  cfg = hoppertrex_hybrid_ppo_runner_cfg(DYNAMIC_STAIR_ACTION_MASK)
  cfg.experiment_name = "hoppertrex_stair_dynamic_v3"
  cfg.seed = 1
  cfg.resume = True
  cfg.num_steps_per_env = DYNAMIC_STAIR_STEPS_PER_ITERATION
  cfg.max_iterations = DYNAMIC_STAIR_PROBE_UPDATES
  cfg.save_interval = DYNAMIC_STAIR_SAVE_INTERVAL
  return cfg

def hoppertrex_stair_roll_assist_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Return seed-1 RollAssist defaults for the initial 100-update block."""

  cfg = hoppertrex_hybrid_ppo_runner_cfg(ROLL_ASSIST_ACTION_MASK)
  cfg.experiment_name = "hoppertrex_stair_roll_assist"
  cfg.seed = 1
  cfg.resume = False
  cfg.num_steps_per_env = ROLL_ASSIST_STEPS_PER_UPDATE
  cfg.max_iterations = ROLL_ASSIST_INITIAL_UPDATES
  cfg.save_interval = ROLL_ASSIST_SAVE_INTERVAL
  return cfg
