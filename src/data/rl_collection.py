import os
import yaml
import torch
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from src.sysid.agent import AsymmetricAgent
from src.sysid.cnnlstm import CNNLSTMModel
from src.envs.pendulum import SinglePendulumEnv
from src.envs.wrappers import DomainRandomizationWrapper, RealismWrapper, SysIDWrapper

NUM_CORES = int(cpu_count() * 0.8)  # Uses all available CPU cores
# NUM_CORES = cpu_count()  # For maximum parallelism during collection, can adjust if needed

# Worker global context references
global_agent = None
global_sysid_model = None
global_config = None
global_sysid_config = None


def init_worker(config_dict, sysid_config_dict):
    """Initializes the RL Agent and SysID Model once per CPU worker process to minimize overhead."""
    global global_agent, global_sysid_model, global_config, global_sysid_config
    global_config = config_dict
    global_sysid_config = sysid_config_dict

    # Enforce CPU execution inside worker subprocesses to avoid CUDA multiprocessing context forks
    device = "cuda" if torch.cuda.is_available() else "cpu"

    HPARAMS = global_sysid_config["hyperparameters"]
    CHKPT_PATH = global_sysid_config["model"]["chkpt_path"]
    CHKPT_PATH = global_sysid_config.get("model", {}).get("chkpt_path", "./models/best_sysid_model_sl.pth")
    N_PARAMS = len(global_sysid_config["sysid_bounds"].keys())

    # Load System ID Model
    global_sysid_model = CNNLSTMModel(config=HPARAMS, n_params=N_PARAMS, chkpt_file_pth=CHKPT_PATH, device=device)
    global_sysid_model.load_model(print_info=False)
    global_sysid_model.eval()

    # Load Active Inference RL Explorer Agent
    global_agent = AsymmetricAgent(global_config, device)
    checkpoint_path = "./models/best_rl_explorer.pth"
    global_agent.load_model(checkpoint_path, device)
    global_agent.eval()


def collect_one_rl_episode(episode_idx):
    """Executes a single episode rollout harvesting raw physical data matching preprocess.py."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    param_keys = list(global_sysid_config["sysid_bounds"].keys())

    # Reconstruct environment pipeline matching evaluation parameters
    env = SinglePendulumEnv(render_mode=None, track_targets=False)
    env = DomainRandomizationWrapper(
        env,
        seed=global_config["seed"],
        power_of_two_samples=global_config["power_of_two_samples"],
        dr_bounds=global_config["dr_bounds"],
        nominal_params=global_config["nominal_params"]
    )
    env = RealismWrapper(
        env,
        encoder_resolution=global_config["realism"]["encoder_resolution"],
    )
    env = SysIDWrapper(
        env,
        sysid_model=global_sysid_model,
        n_params=global_config["n_params"],
        param_keys=param_keys,
        scaler_path=global_config["scaler_path"],
        evaluate=True,
    )

    # CRITICAL FIX: Because the environment is instantiated fresh for every single episode 
    # in parallel, we must pass the unique 'episode_idx' as the seed. This forces your 
    # DomainRandomizationWrapper to pull the correct, unique row from your Sobol matrix [3].
    obs, info = env.reset(seed=episode_idx)

    # CRITICAL FIX: Extract UNSCALED true parameters because preprocess.py scales them itself
    unscaled_params = np.array([info["parameters"][k] for k in param_keys], dtype=np.float32)

    # Initialize agent history tracking
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    priv_obs = info["priv_obs"]
    priv_obs_t = torch.tensor(priv_obs, dtype=torch.float32).unsqueeze(0).to(device)
    global_agent.init_history(obs_t, priv_obs_t)

    MAX_EPISODE_STEPS = env.unwrapped.MAX_EPISODE_STEPS
    action_dims = env.action_space.shape[0]

    # Pre-allocate trajectory memory matching raw dataset requirements
    ep_trajectories = np.zeros((MAX_EPISODE_STEPS, 3), dtype=np.float32)  # [theta, omega, torque]
    ep_actions = np.zeros((MAX_EPISODE_STEPS, action_dims), dtype=np.float32)

    for step in range(MAX_EPISODE_STEPS):
        # CRITICAL FIX: Extract unscaled noisy channels & torque matching preprocess.py schema
        ep_trajectories[step, 0] = info["pole_angle_noisy"]
        ep_trajectories[step, 1] = info["pole_velocity_noisy"]
        ep_trajectories[step, 2] = obs[2]  # Unwrapped baseline torque channel index

        curr_history_obs = global_agent.buffer.actor_history.clone()
        with torch.no_grad():
            action_t = global_agent.select_action(obs_t, curr_history_obs, evaluate=True)

        action = action_t.squeeze(0).cpu().numpy()
        ep_actions[step] = action

        next_obs, reward, terminated, truncated, info = env.step(action)

        next_obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)
        next_priv_obs_t = torch.tensor(info["priv_obs"], dtype=torch.float32).unsqueeze(0).to(device)
        next_done_t = torch.tensor([terminated or truncated], dtype=torch.float32).to(device)

        global_agent.update_history(next_obs_t, next_priv_obs_t, next_done_t)

        obs = next_obs
        obs_t = next_obs_t

    env.close()
    return ep_trajectories, ep_actions, unscaled_params


if __name__ == "__main__":
    # 1. Load Configuration Files
    with open("./src/configs/final_rl_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config["power_of_two_samples"] = 12  # Kept in lockstep with evaluation setup
    # Force single-environment configuration for clean sequential testing
    config["env"]["num_envs"] = 1

    with open("./src/configs/sysid_config.yaml", "r") as f:
        sysid_config = yaml.safe_load(f)

    # Resolve dataset paths contextually to stay aligned with preprocess.py
    rl_processed_path = sysid_config["dataset"]["rl_raw_processed_path"]
    os.makedirs(os.path.dirname(rl_processed_path), exist_ok=True)

    # 2. Extract Shape Information via a temporary environment validation pass
    print("Verifying environment observation dimensions...")
    temp_sysid = CNNLSTMModel(
        config=sysid_config["hyperparameters"],
        n_params=len(sysid_config["sysid_bounds"].keys()),
        chkpt_file_pth=sysid_config.get("model", {}).get("chkpt_path", "./models/best_sysid_model_sl.pth"),
        device=torch.device("cpu")
    )
    dummy_env = SinglePendulumEnv(render_mode=None, track_targets=False)
    dummy_env = DomainRandomizationWrapper(dummy_env, seed=config["seed"], power_of_two_samples=config["power_of_two_samples"], dr_bounds=config["dr_bounds"], nominal_params=config["nominal_params"])
    dummy_env = RealismWrapper(dummy_env, encoder_resolution=config["realism"]["encoder_resolution"])
    dummy_env = SysIDWrapper(dummy_env, sysid_model=temp_sysid, n_params=config["n_params"], param_keys=list(sysid_config["sysid_bounds"].keys()), scaler_path=config["scaler_path"], evaluate=True)
    
    obs_dims = dummy_env.observation_space.shape[0]
    priv_obs_dims = obs_dims + config["n_params"]
    action_dims = dummy_env.action_space.shape[0]
    dummy_env.close()

    # Dynamically correct config constraints
    batch_size = int(config["env"]["num_envs"] * config["env"]["num_steps"])
    minibatch_size = int(batch_size // config["hyperparameters"]["num_minibatches"])
    num_iterations = int(config["env"]["total_timesteps"] // batch_size)
    config["hyperparameters"]["batch_size"] = batch_size
    config["hyperparameters"]["minibatch_size"] = minibatch_size
    config["hyperparameters"]["num_iterations"] = num_iterations
    config["critic"]["privileged_obs_dims"] = priv_obs_dims
    config["actor"]["obs_dims"] = obs_dims
    config["actor"]["action_dims"] = action_dims

    NUM_EPISODES = 2 ** config["power_of_two_samples"]
    print(f"Total episodes to collect via Sobol sequence: {NUM_EPISODES}")
    print(f"Starting parallelized RL active data collection on {NUM_CORES} CPU cores...")

    # 3. Use multiprocessing Pool with initializing tracking
    worker_args = list(range(NUM_EPISODES))
    with Pool(processes=NUM_CORES, initializer=init_worker, initargs=(config, sysid_config)) as pool:
        results = list(tqdm(pool.imap(collect_one_rl_episode, worker_args), 
                            total=NUM_EPISODES, 
                            desc="Harvesting RL Explorations"))

    print("\nProcessing and packing RL dataset matrices...")
    
    # 4. Unpack results explicitly matching the variable signatures expected by preprocess.py
    all_trajectories = np.array([r[0] for r in results], dtype=np.float32)
    all_actions = np.array([r[1] for r in results], dtype=np.float32)
    all_parameters = np.array([r[2] for r in results], dtype=np.float32)

    # 5. Save compressed NPZ archive to the specified path
    print(f"Saving compiled active collection to: {rl_processed_path}...")
    np.savez_compressed(
        rl_processed_path,
        trajectories=all_trajectories,
        actions=all_actions,
        parameters=all_parameters
    )

    print("--- RL Collection Phase Complete ---")
    print("Trajectories Space Shape (N, T, 3): ", all_trajectories.shape)
    print("Actions Space Shape (N, T, Act):    ", all_actions.shape)
    print("Parameters Space Shape (N, Param):  ", all_parameters.shape)