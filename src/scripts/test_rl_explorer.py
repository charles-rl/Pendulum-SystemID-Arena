import os
import yaml
import torch
import numpy as np
from tqdm import tqdm

from src.sysid.agent import AsymmetricAgent
from src.sysid.cnnlstm import CNNLSTMModel
from src.envs.pendulum import SinglePendulumEnv
from src.envs.wrappers import DomainRandomizationWrapper, RealismWrapper, SysIDWrapper

def make_env(config, sysid_config, sysid_model):
    def thunk():
        env = SinglePendulumEnv(render_mode=None, track_targets=False)
        env = DomainRandomizationWrapper(
            env,
            seed=config["seed"],
            power_of_two_samples=config["power_of_two_samples"],
            dr_bounds=config["dr_bounds"],
            nominal_params=config["nominal_params"]
        )
        env = RealismWrapper(
            env,
            encoder_resolution=config["realism"]["encoder_resolution"],
        )
        env = SysIDWrapper(
            env,
            sysid_model=sysid_model,
            n_params=config["n_params"],
            param_keys=list(sysid_config["sysid_bounds"].keys()),
            scaler_path=config["scaler_path"],
            evaluate=True,
        )
        return env
    return thunk

def main():
    # Load Configurations
    with open("./src/configs/final_rl_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config["power_of_two_samples"] = 6
        
    with open("./src/configs/sysid_config.yaml", "r") as f:
        sysid_config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Force single-environment configuration for clean sequential testing
    config["env"]["num_envs"] = 1
    
    HPARAMS = sysid_config["hyperparameters"]
    CHKPT_PATH = sysid_config.get("model", {}).get("chkpt_path", "./models/best_sysid_model_sl.pth")
    N_PARAMS = len(sysid_config["sysid_bounds"].keys())
    
    sysid_model = CNNLSTMModel(config=HPARAMS, n_params=N_PARAMS, chkpt_file_pth=CHKPT_PATH, device=device)
    sysid_model.load_model()
    sysid_model.eval()

    # Determine environment space dimensions
    dummy_env = make_env(config, sysid_config, sysid_model)()
    obs_dims = dummy_env.observation_space.shape[0]
    priv_obs_dims = obs_dims + config["n_params"]
    action_dims = dummy_env.action_space.shape[0]
    dummy_env.close()

    # Math details
    batch_size = int(config["env"]["num_envs"] * config["env"]["num_steps"])
    minibatch_size = int(batch_size // config["hyperparameters"]["num_minibatches"])
    num_iterations = int(config["env"]["total_timesteps"] // batch_size)
    config["hyperparameters"]["batch_size"] = batch_size
    config["hyperparameters"]["minibatch_size"] = minibatch_size
    config["hyperparameters"]["num_iterations"] = num_iterations
    config["critic"]["privileged_obs_dims"] = priv_obs_dims
    config["actor"]["obs_dims"] = obs_dims
    config["actor"]["action_dims"] = action_dims
    
    # Load the agent and best explorer model
    agent = AsymmetricAgent(config, device)
    checkpoint_path = "./models/best_rl_explorer.pth"
    print(f"Loading checkpoint from: {checkpoint_path}")
    agent.load_model(checkpoint_path, device)
    agent.eval()

    # Create evaluation environment
    env = make_env(config, sysid_config, sysid_model)()
    
    NUM_EPISODES = 2 ** config["power_of_two_samples"]
    MAX_EPISODE_STEPS = env.unwrapped.MAX_EPISODE_STEPS
    print(f"Testing explorer model over {NUM_EPISODES} deterministic Sobol randomized episodes...")

    # Set up memory buffers for rollouts
    all_states = np.zeros((NUM_EPISODES, MAX_EPISODE_STEPS, obs_dims), dtype=np.float32)
    all_actions = np.zeros((NUM_EPISODES, MAX_EPISODE_STEPS, action_dims), dtype=np.float32)
    all_true_parameters = np.zeros((NUM_EPISODES, N_PARAMS), dtype=np.float32)
    all_estimated_mu = np.zeros((NUM_EPISODES, MAX_EPISODE_STEPS, N_PARAMS), dtype=np.float32)
    all_estimated_sigma = np.zeros((NUM_EPISODES, MAX_EPISODE_STEPS, N_PARAMS), dtype=np.float32)
    all_rewards = np.zeros((NUM_EPISODES, MAX_EPISODE_STEPS), dtype=np.float32)

    for ep in tqdm(range(NUM_EPISODES), desc="Testing Explorer Model"):
        # Cycle through Sobol randomized parameters sequentially
        if ep == 0:
            obs, info = env.reset(seed=config["seed"])
        else:
            obs, info = env.reset() # Seed=None allows sequential Sobol incrementation
        
        # Scale parameters are stored at the end of the priv_obs array
        priv_obs = info["priv_obs"]
        true_params_scaled = priv_obs[-N_PARAMS:]
        all_true_parameters[ep] = true_params_scaled

        # Track history initialization
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        priv_obs_t = torch.tensor(priv_obs, dtype=torch.float32).unsqueeze(0).to(device)
        agent.init_history(obs_t, priv_obs_t)

        for step in range(MAX_EPISODE_STEPS):
            all_states[ep, step] = obs
            
            # --- FIX: Extract dynamic active estimates from privileged observation instead of static actor obs ---
            all_estimated_mu[ep, step] = priv_obs[-(3 * N_PARAMS):-(2 * N_PARAMS)]
            all_estimated_sigma[ep, step] = priv_obs[-(2 * N_PARAMS):-N_PARAMS]
            
            curr_history_obs = agent.buffer.actor_history.clone()
            with torch.no_grad():
                action_t = agent.select_action(obs_t, curr_history_obs, evaluate=True)
                
            action = action_t.squeeze(0).cpu().numpy()
            all_actions[ep, step] = action
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            all_rewards[ep, step] = reward
            
            # Cast tensors to float32 explicitly to prevent Double precision warnings
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)
            next_priv_obs_t = torch.tensor(info["priv_obs"], dtype=torch.float32).unsqueeze(0).to(device)
            next_done_t = torch.tensor([terminated or truncated], dtype=torch.float32).to(device)
            
            agent.update_history(next_obs_t, next_priv_obs_t, next_done_t)
            
            obs = next_obs
            obs_t = next_obs_t
            # --- FIX: Dynamically update local privileged observation reference each step ---
            priv_obs = info["priv_obs"]

    env.close()

    # Save to a compressed archive
    save_dir = "./data"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "rl_explorer_test_rollouts.npz")
    print(f"Saving trajectory logs to {save_path}...")
    np.savez_compressed(
        save_path,
        states=all_states,
        actions=all_actions,
        true_parameters=all_true_parameters,
        estimated_mu=all_estimated_mu,
        estimated_sigma=all_estimated_sigma,
        rewards=all_rewards
    )
    print("Testing pipeline completed successfully.")

if __name__ == "__main__":
    main()
