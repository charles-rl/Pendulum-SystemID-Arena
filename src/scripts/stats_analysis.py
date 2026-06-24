import os
import yaml
import numpy as np
import pandas as pd
import definitive_screening_design as dsd
import statsmodels.api as sm
from sb3_contrib import TQC
from src.envs.pendulum import SinglePendulumEnv

def main():
    # TODO: To add more data, just use more seeds to test against.
    # 1. Load Configurations
    with open("./src/configs/rl_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    dr_bounds = config["dr_absolute_bounds"]
    param_names = list(dr_bounds.keys())
    num_params = len(param_names)
    
    # 2. Generate the Coded DSD Matrix (-1, 0, 1)
    dsd_df = dsd.generate(n_num=num_params)
    design_np = dsd_df.to_numpy() if hasattr(dsd_df, 'to_numpy') else np.array(dsd_df)
    
    print(f"Generated DSD Matrix with {len(design_np)} design points for {num_params} parameters.")

    # 3. Scale Coded Values to Physical Environments Bounds
    scaled_runs = []
    for row in design_np:
        run_params = {}
        for idx, name in enumerate(param_names):
            low, high = dr_bounds[name]
            mid = (low + high) / 2.0
            
            # Map coded level to physical scale
            if row[idx] == -1:
                run_params[name] = low
            elif row[idx] == 1:
                run_params[name] = high
            else:
                run_params[name] = mid
        scaled_runs.append(run_params)

    # 4. Initialize Environment and Load Trained RL Model
    env = SinglePendulumEnv(render_mode="rgb_array", print_info=False)
    model_path = config["eval"]["load_path"]
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"To run screening, you need a trained baseline model at: {model_path}")
        
    print(f"Loading baseline model from {model_path} for policy sensitivity testing...")
    model = TQC.load(model_path, env=env)

    # 5. Execute Simulation Loop across all Design Points
    performance_results = []
    n_eval_episodes = 50  # Average over multiple runs to smooth out stochastic initial states
    
    print("\n--- Starting DSD Simulation Rollouts ---")
    for i, params in enumerate(scaled_runs):
        run_rewards = []
        for ep in range(n_eval_episodes):
            obs, info = env.reset(seed=config["seed"], options={"params": params})
            done, truncated = False, False
            episode_reward = 0
            
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                
            run_rewards.append(episode_reward)
            
        mean_reward = np.mean(run_rewards)
        performance_results.append(mean_reward)
        print(f"Design Point {i+1:02d}/{len(scaled_runs)} | Mean Reward: {mean_reward:.2f}")
    
    env.close()

    # 6. Statistical Analysis using Coded Values
    print("\n--- Organizing Data for Linear Regression ---")
    analysis_df = pd.DataFrame(design_np, columns=param_names)
    analysis_df['Reward'] = performance_results

    # Build design matrix with main effects only (frees up 9 degrees of freedom to prevent overfitting)
    X_total = analysis_df[param_names]
    X_total = sm.add_constant(X_total)
    y = analysis_df['Reward']

    # Fit Ordinary Least Squares (OLS) model
    ols_model = sm.OLS(y, X_total).fit()
    
    print("\n==============================================================================")
    print("                        DSD PARAMETER SCREENING REPORT                        ")
    print("==============================================================================")
    print(ols_model.summary())
    
    # Extract p-values to help filter active parameters
    p_val_threshold = 0.77
    print(f"\nSuggested Parameter Selection (Threshold: p-value < {p_val_threshold}):")
    p_values = ols_model.pvalues.drop('const')
    for param, p_val in p_values.items():
        status = "🔥 ACTIVE (Highly Sensitive)" if p_val < p_val_threshold else "❄️ INSENSITIVE (Safe to make static)"
        print(f" - {param:<25}: p-value = {p_val:.4f} -> {status}")

if __name__ == "__main__":
    main()
