import random
import time
import yaml
import wandb

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.sysid.agent import AsymmetricAgent
from src.envs.pendulum import SinglePendulumEnv
from src.envs.wrappers import *
from src.sysid.cnnlstm import CNNLSTMModel

def make_env(config, sysid_config, sysid_model):
    def thunk():
        env = SinglePendulumEnv(render_mode=None, track_targets=False)
        env = DomainRandomizationWrapper(
            env,
            seed=config["seed"],
            power_of_two_samples=config["power_of_two_samples"],
            dr_bounds=config["dr_bounds"],
            nominal_params=config["nominal_params"]  # Pass nominal params from the yaml config
        )
        env = RealismWrapper(
            env,
            encoder_resolution=config["realism"]["encoder_resolution"],
        )
        env = SysIDWrapper(
            env,
            sysid_model=sysid_model,
            n_params=config["n_params"],
            param_keys=list(sysid_config["sysid_bounds"].keys()),  # Pass the parameter keys for consistent ordering
            scaler_path=config["scaler_path"],
        )
        return env
    return thunk

def main():
    with open("./src/configs/final_rl_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    with open("./src/configs/sysid_config.yaml", "r") as f:
        sysid_config = yaml.safe_load(f)
        
    # Seeding
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.backends.cudnn.deterministic = True
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{config['wandb']['run_name']}__{config['seed']}"

    HPARAMS = sysid_config["hyperparameters"]
    CHKPT_PATH = sysid_config["model"]["chkpt_path"] if not sysid_config["dataset"]["use_sl_with_rl_model"] else sysid_config["model"]["rl_chkpt_path"]
    N_PARAMS = len(sysid_config["sysid_bounds"].keys())
    
    # Weights & Biases
    if config["track"]:
        wandb.init(
            project=config["wandb"]["project"],
            config=config,
            sync_tensorboard=True,
            name=run_name,
            save_code=True,
        )
        
    writer = SummaryWriter(f"runs/{run_name}")
    
    sysid_model = CNNLSTMModel(config=HPARAMS, n_params=N_PARAMS, chkpt_file_pth=CHKPT_PATH, device=device)
    sysid_model.load_model()
    sysid_model.eval()
    envs = gym.vector.SyncVectorEnv(
        [make_env(config, sysid_config, sysid_model) for i in range(config["env"]["num_envs"])]
    )
    
    obs_dims = np.array(envs.single_observation_space.shape).prod()
    priv_obs_dims = obs_dims + config["n_params"]
    action_dims = np.array(envs.single_action_space.shape).prod()
    
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
    
    # Instantiate Agent
    agent = AsymmetricAgent(config, device)
    
    # Clean manual trackers (completely ignores wrapper dependency)
    num_envs = config["env"]["num_envs"]
    episodic_returns = np.zeros(num_envs)
    episodic_lengths = np.zeros(num_envs)
    
    # Initial states
    global_step = 0
    best_avg_return = -float("inf") # Add this line to track performance
    start_time = time.time()
    next_obs, info = envs.reset(seed=config["seed"])
    next_obs = torch.Tensor(next_obs).to(device)
    next_priv_obs = torch.tensor(info["priv_obs"], dtype=torch.float32).to(device)
    next_done = torch.zeros(num_envs).to(device)
    
    # Initialize agent's Buffer sliding tracker
    agent.init_history(next_obs, next_priv_obs)
    
    # Training loop
    for iteration in range(1, num_iterations + 1):
        completed_returns = []
        if config["hyperparameters"]["anneal_lr"]:
            agent.anneal_lr(iteration)
        
        for step in range(0, config["env"]["num_steps"]):
            global_step += num_envs

            # Pre-action observation & tracking snapshot
            curr_history_obs = agent.buffer.actor_history.clone()
            curr_history_priv_obs = agent.buffer.critic_history.clone()
            obs_t, priv_obs_t, done_t = next_obs, next_priv_obs, next_done
            
            # Get action from actor
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    obs_t, curr_history_obs, priv_obs_t, curr_history_priv_obs
                )
            
            # Step the environments
            next_obs_np, reward, terminations, truncations, info = envs.step(action.cpu().numpy())
            next_done = torch.Tensor(np.logical_or(terminations, truncations)).to(device)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_priv_obs = torch.tensor(info["priv_obs"], dtype=torch.float32).to(device)
            
            # Process episodic metrics manually (completely bypasses Gymnasium wrappers)
            reward_np = np.array(reward)
            episodic_returns += reward_np
            episodic_lengths += 1

            # Log rewards if an episode finishes (autoresets happen instantly in background)
            for i in range(num_envs):
                if next_done[i].item():
                    final_return = episodic_returns[i]
                    final_length = episodic_lengths[i]
                    # print(f"global_step={global_step} | env={i} | return={final_return:.2f} | len={final_length}")
                    completed_returns.append(final_return)
                    
                    writer.add_scalar("charts/episodic_return", final_return, global_step)
                    writer.add_scalar("charts/episodic_length", final_length, global_step)
                    
                    # Reset trackers for this parallel environment index
                    episodic_returns[i] = 0.0
                    episodic_lengths[i] = 0
                    
            # Store the transition
            agent.remember(
                step=step, obs=obs_t, priv_obs=priv_obs_t, done=done_t,
                action=action, logprob=logprob, reward=torch.tensor(reward, dtype=torch.float32).to(device).view(-1), value=value.flatten()
            )

            # Update historical sequence structures
            agent.update_history(next_obs, next_priv_obs, next_done)

        # Bootstrapped value tracking
        agent.compute_gae(next_priv_obs, agent.buffer.critic_history.clone(), next_done)

        # Optimize network and read loss tracking metrics
        metrics = agent.learn()
        
        # Log losses
        writer.add_scalar("charts/learning_rate", agent.optimizer.param_groups[0]["lr"], global_step)
        for key, val in metrics.items():
            writer.add_scalar(f"losses/{key}", val, global_step)

        sps = int(global_step / (time.time() - start_time))
        if len(completed_returns) > 0:
            avg_return = np.mean(completed_returns)
            print(f"Iter {iteration}/{num_iterations} | Step: {global_step} | Avg Return: {avg_return:.2f} | SPS: {sps}")
            
            # Save checkpoint if average return improves
            if avg_return > best_avg_return:
                best_avg_return = avg_return
                checkpoint_path = "./models/best_rl_explorer.pth"
                agent.save_model(checkpoint_path)
                print(f"--> Saved new best explorer model (Avg Return: {avg_return:.2f}) to {checkpoint_path}")
        else:
            print(f"Iter {iteration}/{num_iterations} | Step: {global_step} | SPS: {sps}")
        writer.add_scalar("charts/SPS", sps, global_step)

    envs.close()
    writer.close()
    if config["track"]:
        wandb.finish()


if __name__ == "__main__":
    main()