from src.envs.pendulum import SinglePendulumEnv
from src.sysid.randompolicy import WarmUpActionPolicy
import numpy as np
from multiprocessing import Pool, cpu_count
import os
from tqdm import tqdm
import yaml
from src.utils import generate_deterministic_dr_params

NUM_CORES = cpu_count() # Uses all available cores

# Helper mapping from configuration signal strings to valid WarmUpActionPolicy variation indices
SIGNAL_TO_VARIATION_IDX = {
    "noise": 0,
    "prbs": 1,
    "multisine": 2,
    "impulse": 3,
    "chirp_normal": 4,      # Maps to 4 or 5 inside reset()
    "chirp_shifted": 6      # Maps to 6, 7, 8, or 9 inside reset()
}

def collect_one_episode(args):
    episode_idx, params_dict, params_flat, variation_idx = args
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
    
    # Reset using the specific variation index resolved dynamically from the config list
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
    return trajectory_obs, trajectory_acts, params_flat, variation_idx


if __name__ == "__main__":
    with open("./src/configs/sysid_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # 1. Ensure the directory exists
    os.makedirs(os.path.dirname(config["dataset"]["raw_path"]), exist_ok=True)
    
    # Retrieve configured signal types list dynamically
    signal_types_list = config["dataset"]["collection_signal_types"]
    num_variations = len(signal_types_list)
    print(f"Configured Signal Types for Collection: {signal_types_list}")
    print(f"Number of signal variations: {num_variations}")

    # Compute the m value for sobol sequence based on desired total timesteps and episode length
    number_of_timesteps_per_signal_type = SinglePendulumEnv.MAX_EPISODE_STEPS * num_variations
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
        
        # Cyclically retrieve the signal type configured for this episode
        signal_name = signal_types_list[i % len(signal_types_list)]
        if signal_name not in SIGNAL_TO_VARIATION_IDX:
            raise ValueError(
                f"Unknown signal type '{signal_name}' found in config list. "
                f"Please choose from: {list(SIGNAL_TO_VARIATION_IDX.keys())}"
            )
        variation_idx = SIGNAL_TO_VARIATION_IDX[signal_name]
                
        worker_args.append((i, params_dict, row, variation_idx))
    
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
    all_signals = np.array([r[3] for r in results], dtype=np.int32)

    # 4. Save to compressed NPZ
    print(f"Saving dataset to {config['dataset']['raw_path']}...")
    np.savez_compressed(config['dataset']['raw_path'], 
                        trajectories=all_trajectories,
                        actions=all_actions,
                        parameters=all_parameters,
                        signal_types=all_signals)
    
    print("--- Collection Phase Complete ---")
    print("States Dataset Shape: ", all_trajectories.shape)
    print("Actions Dataset Shape:", all_actions.shape)
    print("Targets Dataset Shape:", all_parameters.shape)
    print("Signals Dataset Shape:", all_signals.shape)
