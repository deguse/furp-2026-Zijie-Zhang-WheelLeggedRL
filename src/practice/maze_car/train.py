import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from maze_car_env import ContinuousMazeCarEnv

def main():
    # Setup paths
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create the environment
    # We use make_vec_env for vectorization (bumped to 16 for faster data collection)
    env = make_vec_env(lambda: ContinuousMazeCarEnv(render_mode=None), n_envs=16)
    
    # Initialize PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=5e-4, # Increased for faster learning
        n_steps=256,        # Reduced from 2048 to update policy 8x more frequently
        batch_size=128,     # Reduced from 256 to match smaller n_steps
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    
    # Save a checkpoint every ~50,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=3125, # 3125 * 16 = 50,000 steps of vec_env
        save_path="./models/",
        name_prefix="ppo_maze_car_v3"
    )
    
    print("Starting V3 training (dense BFS rewards + frequent updates)...")
    model.learn(total_timesteps=500000, callback=checkpoint_callback, tb_log_name="PPO_V3")
    
    # Save the final model
    model.save("./models/ppo_maze_car_v3_final")
    print("Training finished and model saved.")

if __name__ == "__main__":
    main()
