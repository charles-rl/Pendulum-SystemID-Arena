from src.envs.pendulum import SinglePendulumEnv
from src.sysid.randompolicy import WarmUpActionPolicy
import numpy as np
from multiprocessing import Pool, cpu_count
import os
from tqdm import tqdm
import yaml
from src.utils import generate_deterministic_dr_params

NUM_CORES = cpu_count() # Uses all available cores

def collect_one_episode(args):
    episode_idx, params_dict, params_flat = args
    # Create env inside the worker (MuJoCo objects aren't picklable)
    # Command shouldn't be part of the SysID
    env = SinglePendulumEnv(render_mode=None, track_targets=False)
    
    policy = WarmUpActionPolicy(
        action_space=env.action_space, 
        total_timesteps=env.MAX_EPISODE_STEPS, 
        dt=env.model.opt.timestep, 
        frame_skip=env.FRAME_SKIP
    )
    
    # Pass the pre-calculated Sobol parameters directly into the environment
    obs, info = env.reset(seed=episode_idx, options={"params": params_dict})
    
    # Compute deterministic index out of 10 variations and pass the seed
    variation_idx = episode_idx % 10
    policy.reset(variation_idx=variation_idx, seed=episode_idx)
    
    trajectory_obs = np.zeros((env.MAX_EPISODE_STEPS, env.observation_space.shape[0]))
    trajectory_acts = np.zeros((env.MAX_EPISODE_STEPS, env.action_space.shape[0]))
    
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
    return trajectory_obs, trajectory_acts, params_flat


if __name__ == "__main__":
    with open("./src/configs/sysid_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # 1. Ensure the directory exists
    os.makedirs(os.path.dirname(config["dataset"]["raw_path"]), exist_ok=True)
    
    # Compute the m value for sobol sequence based on desired total timesteps and episode length
    # I want to sample normal chirp 2 times and shifted chirp 4 times
    warmup_policy_num_signal_types = WarmUpActionPolicy.NUM_SIGNAL_TYPES - 2 # exclude the 2 chirp types
    warmup_policy_num_signal_types += 6 # add the 6 variations of the chirp types (4 shifted, 2 normal)
    number_of_timesteps_per_signal_type = SinglePendulumEnv.MAX_EPISODE_STEPS * warmup_policy_num_signal_types
    m = round(np.log2(config["dataset"]["desired_total_timesteps"] / number_of_timesteps_per_signal_type))
    actual_total_timesteps = number_of_timesteps_per_signal_type * (2 ** m)
    NUM_EPISODES = actual_total_timesteps // SinglePendulumEnv.MAX_EPISODE_STEPS
    print(f"Calculated m value for Sobol sequence: {m}")
    print(f"Actual total timesteps: {actual_total_timesteps}")
    print(f"Number of episodes to collect: {NUM_EPISODES}")

    # Extract bounds and generate the global DR matrix
    chosen_bounds = config["dataset"]["chosen_bounds"]
    dr_bounds = config[chosen_bounds]
    nominal_params = config["nominal_params"]
    keys = list(dr_bounds.keys())
    lower_bounds = [bounds[0] for bounds in dr_bounds.values()]
    upper_bounds = [bounds[1] for bounds in dr_bounds.values()]
    
    dr_matrix = generate_deterministic_dr_params(
        dr_lower_bound=lower_bounds,
        dr_upper_bound=upper_bounds,
        power_of_two_samples=m,
        seed=config["seed"]
    )
    
    print(f"Starting data collection on {NUM_CORES} cores...")
    worker_args = []
    for i in range(NUM_EPISODES):
        row = dr_matrix[i % len(dr_matrix)]
        params_dict = {keys[j]: float(row[j]) for j in range(len(keys))}
        
        # Inject default values for nominal parameters not present in dr_bounds
        for k, v in nominal_params.items():
            if k not in params_dict:
                params_dict[k] = float(v)
                
        worker_args.append((i, params_dict, row))
    
    # 2. Use a Pool to run episodes in parallel
    with Pool(processes=NUM_CORES) as pool:
        results = list(tqdm(pool.imap(collect_one_episode, worker_args), 
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
    print(f"Saving dataset to {config['dataset']['raw_path']}...")
    np.savez_compressed(config['dataset']['raw_path'], 
                        trajectories=all_trajectories,
                        actions=all_actions,
                        parameters=all_parameters)
    
    print("--- Collection Phase Complete ---")
    print("States Dataset Shape: ", all_trajectories.shape)
    print("Actions Dataset Shape:", all_actions.shape)
    print("Targets Dataset Shape:", all_parameters.shape)
