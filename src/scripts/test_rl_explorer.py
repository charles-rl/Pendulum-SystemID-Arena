import os
import yaml
import torch
import numpy as np
from tqdm import tqdm

from src.sysid.agent import AsymmetricAgent
from src.sysid.cnnlstm import CNNLSTMModel
from src.envs.pendulum import SinglePendulumEnv
from src.envs.wrappers import *

def create_env(config, sysid_config, sysid_model, power_of_two_samples, evaluate=False):
    env = SinglePendulumEnv(render_mode=None, track_targets=False)
    env = DomainRandomizationWrapper(
        env,
        seed=config["seed"],
        power_of_two_samples=power_of_two_samples,
        dr_bounds=config["dr_bounds"],
        nominal_params=config["nominal_params"]  # Pass nominal params from the yaml config
    )
    env = RealismWrapper(
        env,
        encoder_resolution=sysid_config["dataset"]["encoder_resolution"],
        noise_ticks=sysid_config["dataset"]["noise_ticks"],
    )
    env = SysIDWrapper(
        env,
        sysid_model=sysid_model,
        n_params=config["n_params"],
        param_keys=list(sysid_config["sysid_bounds"].keys()),  # Pass the parameter keys for consistent ordering
        scaler_path=config["scaler_path"],
        evaluate=evaluate
    )
    env = ScalingWrapper(env)
    return env

def make_env(config, sysid_config, sysid_model, power_of_two_samples, evaluate):
    def thunk():
        env = create_env(config, sysid_config, sysid_model, power_of_two_samples=power_of_two_samples, evaluate=evaluate)
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
    dummy_env = make_env(config, sysid_config, sysid_model, config["power_of_two_samples"], evaluate=True)()
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
    env = make_env(config, sysid_config, sysid_model, config["power_of_two_samples"], evaluate=True)()
    
    NUM_EPISODES = 2 ** config["power_of_two_samples"]
    MAX_EPISODE_STEPS = env.unwrapped.MAX_EPISODE_STEPS
    print(f"Testing explorer model over {NUM_EPISODES} deterministic Sobol randomized episodes...")
    NUM_TURNS = 5

    # Set up memory buffers for rollouts
    all_states = np.zeros((NUM_EPISODES, NUM_TURNS, MAX_EPISODE_STEPS, obs_dims), dtype=np.float32)
    all_actions = np.zeros((NUM_EPISODES, NUM_TURNS, MAX_EPISODE_STEPS, action_dims), dtype=np.float32)
    all_true_parameters = np.zeros((NUM_EPISODES, N_PARAMS), dtype=np.float32)
    all_estimated_mu = np.zeros((NUM_EPISODES, NUM_TURNS, MAX_EPISODE_STEPS, N_PARAMS), dtype=np.float32)
    all_estimated_sigma = np.zeros((NUM_EPISODES, NUM_TURNS, MAX_EPISODE_STEPS, N_PARAMS), dtype=np.float32)
    all_rewards = np.zeros((NUM_EPISODES, NUM_TURNS, MAX_EPISODE_STEPS), dtype=np.float32)
    all_final_estimated_mu = np.zeros((NUM_EPISODES, N_PARAMS), dtype=np.float32)
    all_final_estimated_sigma = np.zeros((NUM_EPISODES, N_PARAMS), dtype=np.float32)

    for ep in tqdm(range(NUM_EPISODES), desc="Testing Explorer Model"):
        best_estimated_mu = np.zeros((N_PARAMS,), dtype=np.float32)
        best_estimated_sigma = np.zeros((N_PARAMS,), dtype=np.float32)
        for turn in tqdm(range(NUM_TURNS), desc=f"Episode {ep+1}/{NUM_EPISODES}"):
            options = None
            if turn > 0:
                options = {
                    "predicted_mu": best_estimated_mu,
                    "predicted_sigma": best_estimated_sigma
                }
            
            # Cycle through Sobol randomized parameters sequentially
            if ep == 0:
                obs, info = env.reset(seed=config["seed"], options=options) # Seed=None allows sequential Sobol incrementation
            else:
                obs, info = env.reset(options=options) # Seed=None allows sequential Sobol incrementation
            
            # Scale parameters are stored at the end of the priv_obs array
            priv_obs = info["priv_obs"]
            true_params_scaled = priv_obs[-N_PARAMS:]
            all_true_parameters[ep] = true_params_scaled

            # Track history initialization
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            priv_obs_t = torch.tensor(priv_obs, dtype=torch.float32).unsqueeze(0).to(device)
            agent.init_history(obs_t, priv_obs_t)

            for step in range(MAX_EPISODE_STEPS):
                all_states[ep, turn, step] = obs
                
                all_estimated_mu[ep, turn, step] = priv_obs[-(3 * N_PARAMS):-(2 * N_PARAMS)]
                all_estimated_sigma[ep, turn, step] = priv_obs[-(2 * N_PARAMS):-N_PARAMS]
                
                curr_history_obs = agent.buffer.actor_history.clone()
                with torch.no_grad():
                    action_t = agent.select_action(obs_t, curr_history_obs, evaluate=True)
                    
                action = action_t.squeeze(0).cpu().numpy()
                all_actions[ep, turn, step] = action
                
                next_obs, reward, terminated, truncated, info = env.step(action)
                all_rewards[ep, turn, step] = reward
                
                # Cast tensors to float32 explicitly to prevent Double precision warnings
                next_obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)
                next_priv_obs_t = torch.tensor(info["priv_obs"], dtype=torch.float32).unsqueeze(0).to(device)
                next_done_t = torch.tensor([terminated or truncated], dtype=torch.float32).to(device)
                
                agent.update_history(next_obs_t, next_priv_obs_t, next_done_t)
                
                obs = next_obs
                obs_t = next_obs_t
                priv_obs = info["priv_obs"]
                
            # --- FIX: Overwrite the last slot with the true terminal inference of the turn ---
            all_estimated_mu[ep, turn, step] = priv_obs[-(3 * N_PARAMS):-(2 * N_PARAMS)]
            all_estimated_sigma[ep, turn, step] = priv_obs[-(2 * N_PARAMS):-N_PARAMS]
                
            # --- FIX: Extract final 1D vectors from the current turn's last step ---
            final_mu_this_turn = priv_obs[-(3 * N_PARAMS):-(2 * N_PARAMS)]
            final_sigma_this_turn = priv_obs[-(2 * N_PARAMS):-N_PARAMS]
            
            # Replace the best estimated mu and sigma if this has low sigma
            if turn == 0:
                # For the first turn, initialize with the current turn's final step values
                best_estimated_mu = final_mu_this_turn.copy()
                best_estimated_sigma = final_sigma_this_turn.copy()
            else:
                # For subsequent turns, replace selectively per-parameter if uncertainty dropped
                for param_idx in range(N_PARAMS):
                    # If damping or armature then skip
                    # TODO: Convert param_idx to actual word and call it confident system parameters
                    if param_idx == 1 or param_idx == 2:
                        pass
                    else:
                        if final_sigma_this_turn[param_idx] < best_estimated_sigma[param_idx]:
                            best_estimated_mu[param_idx] = final_mu_this_turn[param_idx]
                            best_estimated_sigma[param_idx] = final_sigma_this_turn[param_idx]
        
        all_final_estimated_mu[ep] = best_estimated_mu
        all_final_estimated_sigma[ep] = best_estimated_sigma

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
        final_estimated_mu=all_final_estimated_mu,
        final_estimated_sigma=all_final_estimated_sigma,
        rewards=all_rewards
    )
    print("Testing pipeline completed successfully.")

if __name__ == "__main__":
    main()
