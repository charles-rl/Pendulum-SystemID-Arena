from envs.pendulum import SinglePendulumEnv
from sysid.randompolicy import WarmUpActionPolicy
import numpy as np
from multiprocessing import Pool, cpu_count
import os
from tqdm import tqdm

NUM_CORES = cpu_count() # Uses all available cores
SAVE_PATH = "./dataset/raw_pendulum_sysid_dataset.npz"
NUM_EPISODES = 50_000
# Absolute minimum and maximum
MINMAX_PARAMS = {
    "damping": [0.5, 5.0],
    "friction": [0.2, 2.0],
    "armature": [0.0, 1.0]
}

def collect_one_episode(episode_idx):
    """
    This function runs in its own process.
    It creates its own environment instance.
    """
    # Create env inside the worker (MuJoCo objects aren't picklable)
    env = SinglePendulumEnv(render_mode=None)
    
    policy = WarmUpActionPolicy(
        action_space=env.action_space, 
        total_timesteps=env.max_episode_steps, 
        dt=0.001, 
        frame_skip=env.FRAME_SKIP
    )
    
    # Randomly sample parameters for this trajectory
    params = [
        np.random.uniform(MINMAX_PARAMS["damping"][0], MINMAX_PARAMS["damping"][1]),
        np.random.uniform(MINMAX_PARAMS["friction"][0], MINMAX_PARAMS["friction"][1]),
        np.random.uniform(MINMAX_PARAMS["armature"][0], MINMAX_PARAMS["armature"][1])
    ]
    
    obs, info = env.reset(seed=episode_idx, options={"parameters": params})
    policy.reset()  # Reset the policy to choose a new signal type for this episode
    
    trajectory_obs = np.zeros((env.max_episode_steps, env.observation_space.shape[0]))
    trajectory_acts = np.zeros((env.max_episode_steps, env.action_space.shape[0]))
    
    done = False
    while not done:
        # Control logic
        current_timestep = env.timesteps
        action = policy.act(current_timestep)
        
        # Record everything at the CURRENT timestep
        trajectory_obs[current_timestep] = obs
        trajectory_acts[current_timestep] = action
        
        # Step the environment
        obs_, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        obs = obs_
    
    env.close()
    return trajectory_obs, trajectory_acts, params


if __name__ == "__main__":
    # 1. Ensure the directory exists
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    print(f"Starting data collection on {NUM_CORES} cores...")
    
    # 2. Use a Pool to run episodes in parallel
    with Pool(processes=NUM_CORES) as pool:
        results = list(tqdm(pool.imap(collect_one_episode, range(NUM_EPISODES)), 
                            total=NUM_EPISODES, 
                            desc="Collecting Trajectories"))

    print("\nProcessing and packing dataset slices...")
    # 3. Unpack results
    # results is a list of (trajectory_obs, trajectory_acts, params) tuples
    # NOTE: Bear in mind the dtype of the data here
    all_trajectories = np.array([r[0] for r in results])
    all_actions = np.array([r[1] for r in results])
    all_parameters = np.array([r[2] for r in results])

    # 4. Save to compressed NPZ
    print(f"Saving dataset to {SAVE_PATH}...")
    np.savez_compressed(SAVE_PATH, 
                        trajectories=all_trajectories,
                        actions=all_actions,
                        parameters=all_parameters)
    
    print("--- Collection Phase Complete ---")
    print("States Dataset Shape: ", all_trajectories.shape)
    print("Actions Dataset Shape:", all_actions.shape)
    print("Targets Dataset Shape:", all_parameters.shape)
