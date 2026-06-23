import time
from stable_baselines3 import PPO
from maze_car_env import ContinuousMazeCarEnv

def main():
    # Load the best or final model
    model_path = "./models/ppo_maze_car_v3_100000_steps.zip"
    
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Please run train.py first to generate a model.")
        return

    # Create the environment with rendering enabled
    env = ContinuousMazeCarEnv(render_mode="human")
    
    # Run a few episodes
    for episode in range(5):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step = 0
        
        print(f"--- Episode {episode + 1} ---")
        while not (done or truncated):
            # Get action from the model (deterministic for evaluation)
            action, _states = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            # Add a small delay so human can watch it
            time.sleep(0.01)
            
        print(f"Finished in {step} steps with reward: {total_reward:.2f}")
        
    env.close()

if __name__ == "__main__":
    main()
