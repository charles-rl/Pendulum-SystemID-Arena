import gymnasium as gym
import torch
import numpy as np
import pickle
import yaml
import os
import uuid
from src.utils import generate_deterministic_dr_params

# =====================================================================
# Phase 1: Domain Randomization Wrapper
# =====================================================================
class DomainRandomizationWrapper(gym.Wrapper):
    def __init__(self, env, seed, power_of_two_samples, dr_bounds, nominal_params):
        super().__init__(env)
        self.dr_bounds = dr_bounds
        self.nominal_params = nominal_params  # Save nominal parameters dictionary
        self.keys = list(dr_bounds.keys())
        
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
        if not self.initialized_offset and seed is not None:
            self.sample_idx = seed % self.num_samples
            self.initialized_offset = True
            
        if options is None:
            # Pull the pre-calculated row out of your Sobol matrix
            current_sample = self.dr_matrix[self.sample_idx]
            
            # 1. Start with a full copy of the nominal parameters (kp, kv, etc.)
            randomized_params = self.nominal_params.copy()
            
            # 2. Overwrite only the parameters configured for domain randomization
            for i, key in enumerate(self.keys):
                randomized_params[key] = float(current_sample[i])
                
            options = {"params": randomized_params}
            
            # Cycle sequentially to the next row vector for the next reset call
            self.sample_idx = (self.sample_idx + 1) % self.num_samples
        else:
            # Robustness fallback: if options are passed manually but lack nominal params, merge them
            if "params" in options:
                merged = self.nominal_params.copy()
                merged.update(options["params"])
                options["params"] = merged
            else:
                options["params"] = self.nominal_params.copy()
            
        return self.env.reset(seed=seed, options=options)


# =====================================================================
# Phase 2: Realism Wrapper (Quantization & Jitter)
# =====================================================================
class RealismWrapper(gym.Wrapper):
    def __init__(self, env, encoder_resolution=4096, noise_ticks=3):
        super().__init__(env)
        self.encoder_res = encoder_resolution
        self.noise_ticks = noise_ticks
        
    def _apply_realism(self, obs, info):
        # Extract raw, true continuous mechanical positions from info
        raw_theta = info["pole_angle"]
        raw_omega = info["pole_velocity"]
        
        # 1. Apply 12-bit encoder quantization and tick jitter
        theta_ticks = np.round(raw_theta * (self.encoder_res / (2 * np.pi)))
        theta_ticks += np.round(np.random.normal(0, self.noise_ticks))
        theta_noisy = theta_ticks * ((2 * np.pi) / self.encoder_res)
        
        # 2. Derive physical velocity noise from position jitter boundary limits
        dt = self.env.unwrapped.model.opt.timestep * self.env.unwrapped.FRAME_SKIP
        sigma_omega = (self.noise_ticks * 2 * np.pi) / (self.encoder_res * dt)
        omega_noisy = raw_omega + np.random.normal(0, sigma_omega)
        
        # Rebuild the noisy observation vector matching the pendulum.py layout
        obs_noisy = obs.copy()
        obs_noisy[0] = (theta_noisy + np.pi) / (2 * np.pi) * 2 - 1  # scaled_pole_angle
        
        max_velocity = (45 / 60) * 2 * np.pi
        obs_noisy[1] = omega_noisy / max_velocity  # scaled_pole_velocity
        
        # Cache raw noisy signals into info for the downstream SysIDWrapper to read cleanly
        info["pole_angle_noisy"] = theta_noisy
        info["pole_velocity_noisy"] = omega_noisy
        
        return obs_noisy, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        priv_obs = obs.copy()  # Clean, completely non-noisy scaled observation
        obs_noisy, info = self._apply_realism(obs, info)
        
        info["priv_obs"] = priv_obs  # Saved to info for automatic parallel environment support
        return obs_noisy, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        priv_obs = obs.copy()  # Clean, completely non-noisy scaled observation
        obs_noisy, info = self._apply_realism(obs, info)
        
        info["priv_obs"] = priv_obs
        return obs_noisy, reward, terminated, truncated, info


# =====================================================================
# Phase 3: Active Inference Wrapper (The "Doctor")
# =====================================================================
class SysIDWrapper(gym.Wrapper):
    def __init__(self, env, sysid_model, n_params, param_keys, scaler_path, evaluate=False): # Update path if needed
        super().__init__(env)
        self.env = env
        self.sysid = sysid_model
        self.n_params = n_params
        self.param_keys = param_keys
        self.evaluate = evaluate
        self.reset_mu_value = 0.0  
        self.reset_sigma_value = 1.0  
        
        old_shape = env.observation_space.shape[0]
        new_shape = old_shape + 2 * n_params
        low = np.concatenate([env.observation_space.low, np.full(2 * n_params, -np.inf)])
        high = np.concatenate([env.observation_space.high, np.full(2 * n_params, np.inf)])
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(new_shape,), dtype=np.float32)
        
        self.recorded_states = np.zeros((old_shape + 2, env.unwrapped.MAX_EPISODE_STEPS + 1))
        self.recorded_actions = np.zeros((env.action_space.shape[0], env.unwrapped.MAX_EPISODE_STEPS))
        self.previous_mae, self.previous_sigma = None, None

        # --- Load the parameter scaler ---
        # Load the pre-trained scalers
        with open(scaler_path, "rb") as f:
            scalers = pickle.load(f)
            self.x_scaler = scalers["x_scaler"]
            self.action_scaler = scalers["action_scaler"]
            self.y_scaler = scalers["y_scaler"]
        
        self.base_obs_shape = old_shape
        self.current_base_obs = None
        self.episode_obs = None
        self.episode_acts = None
        self.episode_scenario_idx = 0
            
    def _get_scaled_true_params(self, info):
        """Extracts true parameters from info dictionary and scales them to [-1, 1] range."""
        # Sort/extract values in the exact same key order used by DomainRandomizationWrapper
        # Assuming info["params"] is a dict mapping keys to float values
        parameters = info["parameters"]
        true_params = np.array([parameters[k] for k in self.param_keys]).reshape(1, -1)
        scaled_params = self.y_scaler.transform(true_params).flatten()
        return scaled_params
        
    def get_cos_sin(self, obs):
        theta = obs[0]
        return np.cos(theta), np.sin(theta)
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        priv_obs = info["priv_obs"]
        
        dummy_obs = obs.copy()
        cosine_obs, sine_obs = self.get_cos_sin(dummy_obs)
        self.recorded_states[:, self.env.unwrapped.timesteps] = np.concatenate([dummy_obs, [cosine_obs, sine_obs]])
        
        # Extract and scale true parameters from info
        true_params_scaled = self._get_scaled_true_params(info)
        
        # --- Path B: Faking the Turns ---
        # Default state is completely ignorant
        mu_obs = np.zeros(self.n_params)
        sigma_obs = np.ones(self.n_params)
        
        dice = np.random.rand()
        if dice < 0.20:
            # Scenario A: Turn 1 Wake-up (~20%) -> Leave all as ignorant
            self.episode_scenario_idx = 0  # Scenario A
            pass
        elif dice < 0.80:
            # Scenario B: Turn 2 Realistic Mid-game (~60%)
            self.episode_scenario_idx = 1  # Scenario B
            # damping and armature are solved; frictionloss, max_torque, and backlash are ignorant
            for idx, key in enumerate(self.param_keys):
                if key == "damping" or key == "armature":
                    sigma_random = np.random.uniform(0.004, 0.11)
                    mu_random = np.random.uniform(-0.09, 0.09)
                    mu_obs[idx] = true_params_scaled[idx] + mu_random
                    sigma_obs[idx] = sigma_random
        else:
            # Scenario C: Turn 3 Sniper Mode (~20%)
            # All parameters solved except one randomly chosen parameter
            self.episode_scenario_idx = 2  # Scenario C
            mu_obs = true_params_scaled.copy()
            sigma_random = np.random.uniform(0.004, 0.11)
            sigma_obs = np.full(self.n_params, sigma_random)
            ignorant_idx = np.random.randint(self.n_params)
            mu_obs[ignorant_idx] = 0.0
            sigma_obs[ignorant_idx] = 1.0
            
        # Store these on the instance so the step function can refer to them during cold start
        self.current_mu = mu_obs.copy()
        self.current_sigma = sigma_obs.copy()
        # --------------------------------
        current_mae = np.sum(np.abs(mu_obs - true_params_scaled))
        self.previous_sigma = sigma_obs.copy()
        self.previous_mae = current_mae.copy()
    
        # Actor only sees public obs + estimations (mu, sigma)
        obs = np.concatenate([obs, mu_obs, sigma_obs])
        # Critic (Privileged) sees un-noisy/noisy states + estimations + TRUE underlying parameters
        priv_obs = np.concatenate([priv_obs, mu_obs, sigma_obs, true_params_scaled])
        
        info["priv_obs"] = priv_obs.copy()
        return obs, info
    
    def sysid_inference(self):
        # Reconstruct the precise 5 channels used during training: [theta, omega, torque, cos, sin]
            theta_hist = self.recorded_states[0, :self.env.unwrapped.timesteps]
            omega_hist = self.recorded_states[1, :self.env.unwrapped.timesteps]
            torque_hist = self.recorded_states[2, :self.env.unwrapped.timesteps]
            
            states_raw = np.stack([
                theta_hist,
                omega_hist,
                torque_hist,
                np.cos(theta_hist),
                np.sin(theta_hist)
            ], axis=1)  # Shape: (Timesteps, 5)
            
            # Align action history to match the exact same timestep length
            actions_raw = self.recorded_actions[:, :self.env.unwrapped.timesteps].T  # Shape: (Timesteps, Action_Dim)
            
            # Apply Preprocessing Scalers
            states_scaled = self.x_scaler.transform(states_raw)
            actions_scaled = self.action_scaler.transform(actions_raw)
            
            # Convert to PyTorch Tensors & Move to Model Device
            device = self.sysid.device
            
            # Convert and permute shapes from (T, C) -> (Batch=1, Channels, Timesteps) to match Conv1D expectations
            states_t = torch.tensor(states_scaled, dtype=torch.float32, device=device).T.unsqueeze(0)
            actions_t = torch.tensor(actions_scaled, dtype=torch.float32, device=device).T.unsqueeze(0)
            
            # Run inference without tracking gradients
            with torch.no_grad():
                mu_t, sigma_t = self.sysid.forward(states_t, actions_t)
            
            # Convert outputs back to numpy vectors
            mu_critic = mu_t.squeeze(0).cpu().numpy()
            sigma_critic = sigma_t.squeeze(0).cpu().numpy()
            
            return mu_critic, sigma_critic
        
    def step(self, action):
        # 1. Advance the environment
        obs, reward_safety, terminated, truncated, info = self.env.step(action)
        priv_obs = info["priv_obs"]
        
        # 2. Record raw states and actions 
        # (Assuming recorded_states[0] is theta and recorded_states[1] is omega)
        self.recorded_states[:, self.env.unwrapped.timesteps] = np.concatenate([obs.copy(), self.get_cos_sin(obs.copy())]) # Adjust if your base obs differs
        self.recorded_actions[:, self.env.unwrapped.timesteps - 1] = action.copy()
        
        # Extract and scale true parameters from info
        true_params_scaled = self._get_scaled_true_params(info)
        
        # 3. Cold-start Guard: Only run SysID after gathering enough context steps (e.g., 5 steps)
        # This prevents passing empty/short sequences to the CNN-LSTM
        # Establish base priors using the faked scenario rolled during reset
        mu_actor = self.current_mu.copy()
        sigma_actor = self.current_sigma.copy()
        if self.env.unwrapped.timesteps >= 50 and not self.evaluate:
            mu_critic, sigma_critic = self.sysid_inference()
        elif self.env.unwrapped.timesteps >= self.unwrapped.MAX_EPISODE_STEPS - 5 and self.evaluate:
            # At the last few steps we run sysid inference even in evaluation mode to see how well it converges
            mu_critic, sigma_critic = self.sysid_inference()
        else:
            mu_critic = self.current_mu.copy()
            sigma_critic = self.current_sigma.copy()

        # 4. Append SysID estimates back to observations
        obs = np.concatenate([obs, mu_actor, sigma_actor])
        priv_obs = np.concatenate([priv_obs, mu_critic, sigma_critic, true_params_scaled])
        
        current_mae = np.sum(np.abs(mu_critic - true_params_scaled))
        
        # Keep your existing reward and tracking logic below...
        reward_safety = 2.0 * reward_safety
        reward_sigma = 10.0 * (self.previous_sigma - sigma_critic)
        reward_mu = 0.0 * (self.previous_mae - current_mae)
        # print(f"Reward Breakdown -> Safety: {reward_safety:.3f}, Sigma: {np.sum(reward_sigma):.3f}, Mu: {np.sum(reward_mu):.3f}, Total: {reward_safety + np.sum(reward_sigma) + np.sum(reward_mu):.3f}")

        # Update historical trackers for the next step
        self.previous_sigma = sigma_critic.copy()
        self.previous_mae = current_mae.copy()  # you need the true params to scale this correctly
        reward = reward_safety + np.sum(reward_sigma) + np.sum(reward_mu)
        
        info["priv_obs"] = priv_obs.copy()
        return obs, reward, terminated, truncated, info