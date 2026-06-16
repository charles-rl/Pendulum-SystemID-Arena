import os
import gymnasium as gym
import wandb
import numpy as np
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.env_checker import check_env
from src.envs.pendulum import SinglePendulumEnv
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib import RecurrentPPO, TQC  # For LSTM PPO
from stable_baselines3 import PPO  # For standard PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
import yaml
from src.utils import generate_deterministic_dr_params

    
class DomainRandomizationWrapper(gym.Wrapper):
    def __init__(self, env, seed, power_of_two_samples, dr_bounds):
        super().__init__(env)
        self.dr_bounds = dr_bounds
        self.keys = list(dr_bounds.keys())
        
        # FIX A: Correctly parse the low and high values out of your YAML pairs
        lower_bounds = [bounds[0] for bounds in dr_bounds.values()]
        upper_bounds = [bounds[1] for bounds in dr_bounds.values()]
        
        self.dr_matrix = generate_deterministic_dr_params(
            dr_lower_bound=lower_bounds,
            dr_upper_bound=upper_bounds,
            power_of_two_samples=power_of_two_samples,
            seed=seed
        )
        self.num_samples = len(self.dr_matrix)
        self.sample_idx = 0
        self.initialized_offset = False

    def reset(self, seed=None, options=None):
        # TRAP FIX: SB3 gives each parallel worker a unique seed on the first reset loop.
        # We catch that worker seed here to offset our matrix entry point.
        # This prevents all 4 parallel environments from running identical parameters simultaneously!
        if not self.initialized_offset and seed is not None:
            self.sample_idx = seed % self.num_samples
            self.initialized_offset = True

        if options is None:
            # FIX B: Pull the pre-calculated row out of your Sobol matrix
            current_sample = self.dr_matrix[self.sample_idx]
            
            # Map the flat row array back to your parameter keys dictionary structure
            randomized_params = {self.keys[i]: float(current_sample[i]) for i in range(len(self.keys))}
            options = {"params": randomized_params}
            
            # Cycle sequentially to the next row vector for the next reset call
            self.sample_idx = (self.sample_idx + 1) % self.num_samples
            
        return self.env.reset(seed=seed, options=options)


def main():
    with open("./src/configs/rl_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    set_random_seed(config["seed"])
    dr_bounds = config["dr_bounds"]
    config_rl = config["rl_hyperparameters"]
    
    local_test_env = SinglePendulumEnv(render_mode="rgb_array")
    # 2. Check the environment structure
    # This ensures your Observation and Action spaces are mathematically correct
    print("Checking Gymnasium environment compliance...")
    check_env(local_test_env, warn=True)
    print("Environment check passed! Preparing for training...")
    
    # 1. Initialize Parallel Environments
    num_envs = config_rl["num_envs"]
    wrapper_kwargs = {
        "seed": config["seed"], 
        "power_of_two_samples": config["train"]["sobol_power_of_two_samples"], 
        "dr_bounds": dr_bounds
    }
    env = make_vec_env(
        SinglePendulumEnv, 
        n_envs=num_envs, 
        seed=config["seed"], 
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"render_mode": "rgb_array"},
        wrapper_class=DomainRandomizationWrapper,  # <-- Injects wrapper logic
        wrapper_kwargs=wrapper_kwargs
    )
    env = VecMonitor(env)

    # 3. Initialize Evaluation Environment (Also randomizes parameters to check robust tracking performance)
    eval_env = make_vec_env(
        SinglePendulumEnv, 
        n_envs=1, 
        env_kwargs={"render_mode": "rgb_array"},
        wrapper_class=DomainRandomizationWrapper,
        wrapper_kwargs=wrapper_kwargs
    )
    eval_env = VecMonitor(eval_env)

    # 4. Initialize Weights & Biases (W&B)
    run = wandb.init(
        project=config["wandb"]["project"],
        config=config,
        sync_tensorboard=True,
        save_code=True,
        name=f"{config["wandb"]["run_name"]}"
    )

    # 5. Initialize or Load the Chosen Model
    model = TQC(
        policy=config_rl["policy_type"],
        env=env,
        learning_rate=config_rl["learning_rate"],
        buffer_size=config_rl["buffer_size"],
        batch_size=config_rl["batch_size"],
        tau=config_rl["tau"],
        gamma=config_rl["gamma"],
        train_freq=config_rl["train_freq"],
        gradient_steps=num_envs // 2 if num_envs > 1 else 1,  # More envs -> more frequent updates
        tensorboard_log=f"runs/{run.id}",
        verbose=1,
    )
    # 6. Set up the W&B Callback
    # This automatically tracks rewards, episode length, and actor/critic losses, 
    # and saves the best model weights during the training process.
    save_path = config["train"]["save_path"]
    os.makedirs(f"{save_path}/{run.id}", exist_ok=True)
    wandb_callback = WandbCallback(
        model_save_path=f"{save_path}/{run.id}",
        verbose=2,
    )
    
    # ADD: EvalCallback to monitor and save the absolute best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}/{run.id}/best_model",
        log_path=f"{save_path}/{run.id}/logs",
        eval_freq=config["train"]["eval_freq"],  # Adjust eval frequency based on number of envs
        n_eval_episodes=config["train"]["eval_episodes"],
        deterministic=True,    # Test the policy without exploration noise
        render=False
    )

    # 7. Start the training process
    print(f"Starting TQC training for {config_rl['total_timesteps']} steps...")
    model.learn(
        total_timesteps=config_rl['total_timesteps'],
        callback=[wandb_callback, eval_callback],  # Pass both callbacks to the learning process
        tb_log_name="TQC"
    )

    # 8. Clean up and close the run
    run.finish()
    print("Training finished successfully!")

if __name__ == "__main__":
    main()
