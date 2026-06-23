import argparse
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from maze_car_env_v4_plus import ContinuousMazeCarEnvV4Plus


MODEL_DIR = "./models"
DEFAULT_BEST_MODEL = os.path.join(MODEL_DIR, "best_v4_plus", "best_model.zip")
DEFAULT_FINAL_MODEL = os.path.join(MODEL_DIR, "ppo_maze_car_v4_plus_final.zip")
DEFAULT_VECNORM = os.path.join(MODEL_DIR, "vecnormalize_v4_plus.pkl")


def make_eval_env(config):
    return Monitor(ContinuousMazeCarEnvV4Plus(**config))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--vecnorm", type=str, default=DEFAULT_VECNORM)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--random-maze", action="store_true")
    parser.add_argument("--start-mode", type=str, default="easy", choices=["easy", "medium", "far", "random"])
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.01)
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        model_path = DEFAULT_BEST_MODEL if os.path.exists(DEFAULT_BEST_MODEL) else DEFAULT_FINAL_MODEL

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run train_v4_plus.py first."
        )
    if not os.path.exists(args.vecnorm):
        raise FileNotFoundError(
            f"VecNormalize stats not found: {args.vecnorm}. Run train_v4_plus.py first."
        )

    env_config = dict(
        render_mode=None if args.no_render else "human",
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        random_maze=args.random_maze,
        start_mode=args.start_mode,
        collision_ends_episode=False,
    )

    base_env = DummyVecEnv([lambda: make_eval_env(env_config)])
    env = VecNormalize.load(args.vecnorm, base_env)
    env.training = False
    env.norm_reward = False
    env = VecFrameStack(env, n_stack=3)

    model = PPO.load(model_path, env=env)

    success_count = 0
    total_steps = 0
    total_collisions = 0
    total_rewards = []

    print(f"Evaluating model: {model_path}")
    print(f"Config: {env_config}")

    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        episode_collisions = 0
        success = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            reward = float(rewards[0])
            info = infos[0]
            done = bool(dones[0])

            episode_reward += reward
            episode_steps += 1
            episode_collisions += int(info.get("collided", False))

            if done:
                success = bool(info.get("is_success", False))

            if not args.no_render:
                time.sleep(args.sleep)

        success_count += int(success)
        total_steps += episode_steps
        total_collisions += episode_collisions
        total_rewards.append(episode_reward)

        print(
            f"Episode {episode + 1:02d}: "
            f"success={success}, steps={episode_steps}, "
            f"collisions={episode_collisions}, reward={episode_reward:.2f}"
        )

    print("-" * 60)
    print(f"Success rate: {success_count}/{args.episodes} = {success_count / args.episodes:.1%}")
    print(f"Average steps: {total_steps / args.episodes:.1f}")
    print(f"Average collisions: {total_collisions / args.episodes:.1f}")
    print(f"Average reward: {sum(total_rewards) / len(total_rewards):.2f}")

    env.close()


if __name__ == "__main__":
    main()
