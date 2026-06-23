import argparse
import os
from typing import Dict

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize

from maze_car_env_v4_plus import ContinuousMazeCarEnvV4Plus


MODEL_DIR = "./models"
LOG_DIR = "./logs"
VECNORM_PATH = os.path.join(MODEL_DIR, "vecnormalize_v4_plus.pkl")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_maze_car_v4_plus_final")
BEST_MODEL_DIR = os.path.join(MODEL_DIR, "best_v4_plus")


def linear_schedule(initial_value: float):
    """Linear learning-rate schedule used by SB3."""
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule


def make_single_env(config: Dict, monitor: bool = True):
    env = ContinuousMazeCarEnvV4Plus(**config)
    if monitor:
        env = Monitor(env)
    return env


def build_vec_env(config: Dict, n_envs: int, training: bool):
    vec_env = make_vec_env(lambda: make_single_env(config, monitor=True), n_envs=n_envs)

    vec_env = VecNormalize(
        vec_env,
        training=training,
        norm_obs=True,
        norm_reward=training,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.995,
    )

    if not training:
        vec_env.training = False
        vec_env.norm_reward = False

    # Put frame stacking after VecNormalize.
    vec_env = VecFrameStack(vec_env, n_stack=3)
    return vec_env


def find_vecnormalize(vec_env):
    """Find the VecNormalize wrapper even if VecFrameStack wraps it."""
    current = vec_env
    while current is not None:
        if isinstance(current, VecNormalize):
            return current
        current = getattr(current, "venv", None)
    raise RuntimeError("VecNormalize wrapper was not found.")


class SaveVecNormalizeCallback(BaseCallback):
    """Periodically save VecNormalize statistics."""

    def __init__(self, save_path: str, save_freq: int = 50_000, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = int(save_freq)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            find_vecnormalize(self.training_env).save(self.save_path)
            if self.verbose:
                print(f"Saved VecNormalize stats to {self.save_path}")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--random-maze", action="store_true")
    parser.add_argument("--start-mode", type=str, default="easy", choices=["easy", "medium", "far", "random"])
    parser.add_argument("--check-env", action="store_true")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    env_config = dict(
        render_mode=None,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        random_maze=args.random_maze,
        start_mode=args.start_mode,
        collision_ends_episode=False,
    )

    if args.check_env:
        print("Checking custom environment with SB3 check_env...")
        check_env(ContinuousMazeCarEnvV4Plus(**env_config), warn=True)
        print("Environment check complete.")

    train_env = build_vec_env(env_config, n_envs=args.n_envs, training=True)
    eval_env = build_vec_env(env_config, n_envs=1, training=False)

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=linear_schedule(3e-4),
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs=policy_kwargs,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 50_000 // args.n_envs),
        save_path=MODEL_DIR,
        name_prefix="ppo_maze_car_v4_plus",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=os.path.join(LOG_DIR, "eval_v4_plus"),
        eval_freq=max(1, 20_000 // args.n_envs),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    save_vecnorm_callback = SaveVecNormalizeCallback(
        VECNORM_PATH,
        save_freq=max(1, 50_000 // args.n_envs),
        verbose=1,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback, save_vecnorm_callback])

    print("Starting V4 Plus training")
    print(f"Config: {env_config}")
    print(f"Timesteps: {args.timesteps}, n_envs: {args.n_envs}")

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        tb_log_name="PPO_V4_PLUS",
    )

    model.save(FINAL_MODEL_PATH)
    find_vecnormalize(train_env).save(VECNORM_PATH)
    train_env.close()
    eval_env.close()

    print(f"Saved final model to: {FINAL_MODEL_PATH}.zip")
    print(f"Saved VecNormalize stats to: {VECNORM_PATH}")
    print(f"Best model directory: {BEST_MODEL_DIR}")


if __name__ == "__main__":
    main()
