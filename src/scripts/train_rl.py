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

    
class DomainRandomizationWrapper(gym.Wrapper):
    def __init__(self, env, dr_bounds):
        super().__init__(env)
        self.dr_bounds = dr_bounds

    def reset(self, seed=None, options=None):
        # When SB3 calls reset internally, options will be None.
        # We intercept it here and inject our randomized parameters!
        if options is None:
            randomized_params = {
                "kp": float(np.random.uniform(*self.dr_bounds["kp"])),
                "kv": float(np.random.uniform(*self.dr_bounds["kv"])),
                "max_torque": float(np.random.uniform(*self.dr_bounds["max_torque"])),
                "frictionloss": float(np.random.uniform(*self.dr_bounds["frictionloss"])),
                "damping": float(np.random.uniform(*self.dr_bounds["damping"])),
                "armature": float(np.random.uniform(*self.dr_bounds["armature"])),
                "backlash_max_deg": float(np.random.uniform(*self.dr_bounds["backlash_max_deg"])),
                "backlash_armature": float(np.random.uniform(*self.dr_bounds["backlash_armature"])),
                "backlash_damping": float(np.random.uniform(*self.dr_bounds["backlash_damping"]))
            }
            options = {"params": randomized_params}
            
        return self.env.reset(seed=seed, options=options)

set_random_seed(42)

def main():
    with open("./src/configs/rl_config.yaml", "r") as f:
        config = yaml.safe_load(f)
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
    env = make_vec_env(
        SinglePendulumEnv, 
        n_envs=num_envs, 
        seed=0, 
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"render_mode": "rgb_array"},
        wrapper_class=DomainRandomizationWrapper,  # <-- Injects wrapper logic
        wrapper_kwargs={"dr_bounds": dr_bounds}    # <-- Passes config parameters
    )
    env = VecMonitor(env)

    # 3. Initialize Evaluation Environment (Also randomizes parameters to check robust tracking performance)
    eval_env = make_vec_env(
        SinglePendulumEnv, 
        n_envs=1, 
        env_kwargs={"render_mode": "rgb_array"},
        wrapper_class=DomainRandomizationWrapper,
        wrapper_kwargs={"dr_bounds": dr_bounds}
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
    # model = TQC.load("models/3g0olyca/best_model/best_model.zip", env=env)

    # 6. Set up the W&B Callback
    # This automatically tracks rewards, episode length, and actor/critic losses, 
    # and saves the best model weights during the training process.
    save_path = config["save_path"]
    os.makedirs(f"{save_path}/{run.id}", exist_ok=True)
    wandb_callback = WandbCallback(
        model_save_path=f"{save_path}/{run.id}",
        verbose=2,
    )
    
    # ADD: EvalCallback to monitor and save the absolute best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/{run.id}/best_model",
        log_path=f"models/{run.id}/logs",
        eval_freq=10_000,  # Adjust eval frequency based on number of envs
        deterministic=True,    # Test the policy without exploration noise
        render=False
    )

    # 7. Start the training process
    print(f"Starting TQC training on 'Bipedal-Robot-Test' for {config_rl['total_timesteps']} steps...")
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
