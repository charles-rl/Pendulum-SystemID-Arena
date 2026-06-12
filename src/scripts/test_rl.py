import argparse
import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from sb3_contrib import RecurrentPPO, TQC  # For LSTM PPO
from stable_baselines3 import PPO
import yaml
from src.envs.pendulum import SinglePendulumEnv
import numpy as np
import time

def main():
    with open("./src/configs/rl_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # 2. Verify the model path exists
    if not os.path.exists(config["eval"]["load_path"]):
        raise FileNotFoundError(f"Model file not found at: {config['eval']['load_path']}")

    # 3. Create the environment based on your chosen mode
    print(f"Initializing environment in '{config['eval']['render_mode']}' mode...")
    env = SinglePendulumEnv(render_mode=config['eval']['render_mode'], print_info=True)

    # If in video-saving mode, wrap the environment
    if config['eval']['render_mode'] == "rgb_array":
        os.makedirs(config['eval']['output_dir'], exist_ok=True)
        print(f"Videos will be recorded and saved to: {config['eval']['output_dir']}")
        env = RecordVideo(
            env,
            video_folder=config['eval']['output_dir'],
            episode_trigger=lambda ep: True,  # Record all testing episodes
            disable_logger=False
        )

    # 4. Load the trained TQC model
    print(f"Loading trained model from: {config['eval']['load_path']}...")
    model = TQC.load(config['eval']['load_path'], env=env)

    total_episode_reward = 0
    # 5. Run evaluation loop
    for episode in range(config['eval']['episodes']):
        simulation_delay_speed = 2.0  # default is 1.0
        test_params = {
            "kp": 998.22,
            "kv": 2.731,
            "max_torque": 2.94,          # Set nominal (2.94) or peak voltage performance (3.35)
            "frictionloss": 0.052,
            "damping": 0.6,
            "armature": 0.028,
            "backlash_max_deg": 0.87,
            "backlash_armature": 0.01,
            "backlash_damping": 0.01,
        }
        options = {"params": test_params}
        obs, info = env.reset(options=options)
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        if episode == 0:
            time.sleep(2)  # Pause briefly to see the initial state before starting the episode
        
        print(f"\n--- Starting Episode {episode + 1} ---")
        while not (done or truncated):
            step_start = time.time()
            # Predict the action deterministically (no exploration noise)
            action, _states = model.predict(obs, deterministic=True)
            
            # Step the physics
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            episode_reward += reward
            steps += 1
            
            time_until_next_step = env.model.opt.timestep * env.FRAME_SKIP - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step * simulation_delay_speed)  # Adjust the speed of the simulation

        print(f"Episode {episode + 1} Finished | Steps: {steps} | Total Reward: {episode_reward:.2f}")
        total_episode_reward += episode_reward
    total_episode_reward /= config['eval']['episodes']  # Average reward per episode
    # 6. Clean up and close
    print("\nClosing environment and saving files...")
    env.close()
    print("Done!")
    print(f"Total Episode Reward: {total_episode_reward:.2f}")

if __name__ == "__main__":
    main()
