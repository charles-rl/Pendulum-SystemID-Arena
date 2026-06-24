import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from src.envs.pendulum import SinglePendulumEnv
import yaml

# TODO: Action needs some noise and discretization to reflect the real world, maybe some delays too but only a little bit
# TODO: Do the noise here such that it will be universal and taken from utils.

with open("./src/configs/sysid_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- PATHS ---
RAW_TRAIN_PATH = "./dataset/raw_pendulum_sysid_dataset.npz"
RAW_ABS_PATH = "./dataset/raw_ood_pendulum_sysid_dataset.npz"

PROCESSED_DATA_PATH = config["dataset"]["processed_path"] if not config["dataset"]["include_rl_dataset"] else config["dataset"]["rl_processed_path"]
SCALER_PATH = config["dataset"]["scalers_path"] if not config["dataset"]["include_rl_dataset"] else config["dataset"]["rl_scalers_path"]
ENCODER_RESOLUTION = config["dataset"]["encoder_resolution"]


def preprocess_raw_data(X_states_raw, ENCODER_RESOLUTION, DT):
    """
    Modular preprocessing helper expanded to support unscaled motor torque channels.
    Assumes X_states_raw has shape (N, T, 3) where index 2 is motor torque.
    """
    N, timesteps, _ = X_states_raw.shape
    
    # 1. Apply 12-bit encoder quantization and operational tick jitter to angle
    theta_noisy = np.round(X_states_raw[:, :, 0] * (ENCODER_RESOLUTION / (2 * np.pi)))
    noise_ticks = 3
    theta_noisy += np.round(np.random.normal(0, noise_ticks, theta_noisy.shape))
    theta_noisy = theta_noisy * ((2 * np.pi) / ENCODER_RESOLUTION)
    
    # Apply velocity noise derived from position jitter
    sigma_omega = (noise_ticks * 2 * np.pi) / (ENCODER_RESOLUTION * DT)
    omega_noisy = X_states_raw[:, :, 1] + np.random.normal(0, sigma_omega, X_states_raw[:, :, 1].shape)
    
    # Extract clean torque signal directly from channel index 2
    torque = X_states_raw[:, :, 2]

    # 2. Re-map trigonometric configurations across 5 channels
    X_engineered = np.zeros((N, timesteps, 5))
    X_engineered[:, :, 0] = theta_noisy              # theta
    X_engineered[:, :, 1] = omega_noisy              # omega
    X_engineered[:, :, 2] = torque                   # ADDED: torque
    X_engineered[:, :, 3] = np.cos(theta_noisy)      # cos(theta)
    X_engineered[:, :, 4] = np.sin(theta_noisy)      # sin(theta)
    
    return X_engineered


def main():
    dummy_env = SinglePendulumEnv(render_mode=None)
    DT = dummy_env.model.opt.timestep * dummy_env.FRAME_SKIP
    
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    
    # =========================================================
    # 1. LOAD DATASETS
    # =========================================================
    print(f"Loading primary training raw dataset: {RAW_TRAIN_PATH}")
    train_data = np.load(RAW_TRAIN_PATH)
    X_train_raw = train_data['trajectories']  # Shape: (N, T, 2)
    actions_train_raw = train_data['actions']  # Shape: (N, T, 1)
    Y_train_raw = train_data['parameters']    # Shape: (N, 6)
    
    num_actions = actions_train_raw.shape[2]
    
    print(f"Loading absolute raw dataset for True OOD: {RAW_ABS_PATH}")
    if os.path.exists(RAW_ABS_PATH):
        abs_data = np.load(RAW_ABS_PATH)
        X_abs_raw = abs_data['trajectories']
        actions_abs_raw = abs_data['actions']
        Y_abs_raw = abs_data['parameters']
    else:
        raise FileNotFoundError(
            f"Absolute bounds dataset not found at {RAW_ABS_PATH}. "
            "Please generate this file to secure your True OOD evaluation set [3]."
        )

    # =========================================================
    # 2. RUN MODULAR FEATURE EXTRACTION
    # =========================================================
    print("Applying encoder noise and extracting Sin/Cos features...")
    X_train_eng = preprocess_raw_data(X_train_raw, ENCODER_RESOLUTION, DT)
    X_abs_eng = preprocess_raw_data(X_abs_raw, ENCODER_RESOLUTION, DT)

    # =========================================================
    # 3. SPLITTING LOGIC (ID and True OOD only)
    # =========================================================
    print("Processing split boundaries...")
    sysid_bounds = config["sysid_bounds"]
    
    # A. 100% of the training dataset is processed as In-Distribution (ID) [3]
    X_in = X_train_eng
    actions_in = actions_train_raw
    Y_in = Y_train_raw

    # B. Inject Consolidated RL Explorer Trajectories smoothly into the ID training pool
    if config["dataset"]["include_rl_dataset"]:
        rl_data_path = config["dataset"]["rl_raw_processed_path"]
        if os.path.exists(rl_data_path):
            print(f"Loading and appending RL collected dataset: {rl_data_path}")
            rl_data = np.load(rl_data_path)
            X_rl_raw = rl_data['trajectories']  # Shape: (N, T, 3) -> [theta, omega, torque]
            actions_rl = rl_data['actions']     # Shape: (N, T, Action_Dim)
            Y_rl = rl_data['parameters']        # Shape: (N, Param_Dim)
            
            # Perform channel expansion (3 -> 5) WITHOUT adding extra quantization noise or tick jitter
            X_rl_eng = np.zeros((X_rl_raw.shape[0], X_rl_raw.shape[1], 5))
            X_rl_eng[:, :, 0] = X_rl_raw[:, :, 0]             # theta (already noisy from wrapper)
            X_rl_eng[:, :, 1] = X_rl_raw[:, :, 1]             # omega (already noisy from wrapper)
            X_rl_eng[:, :, 2] = X_rl_raw[:, :, 2]             # torque
            X_rl_eng[:, :, 3] = np.cos(X_rl_raw[:, :, 0])     # cos(theta)
            X_rl_eng[:, :, 4] = np.sin(X_rl_raw[:, :, 0])     # sin(theta)
            
            # Concatenate right into the In-Distribution matrix space before the data splitting pass
            X_in = np.concatenate([X_in, X_rl_eng], axis=0)
            actions_in = np.concatenate([actions_in, actions_rl], axis=0)
            Y_in = np.concatenate([Y_in, Y_rl], axis=0)
            print(f"Successfully merged {X_rl_raw.shape[0]} active RL trajectories into ID dataset.")
        else:
            print(f"Warning: 'include_rl_dataset' is True but file was not found at {rl_data_path}. Proceeding with base data.")

    # C. Extract True OOD from the ABSOLUTE dataset [3]
    # Any trajectory that has at least one parameter outside the sysid_bounds is True OOD [3]
    inside_bounds_masks = []
    for i, param in enumerate(sysid_bounds.keys()):
        p_min, p_max = sysid_bounds[param]
        inside_mask = (Y_abs_raw[:, i] >= p_min) & (Y_abs_raw[:, i] <= p_max)
        inside_bounds_masks.append(inside_mask)
        
    inside_bounds_masks = np.stack(inside_bounds_masks, axis=0)
    is_inside_bounds = np.all(inside_bounds_masks, axis=0)
    is_true_ood = ~is_inside_bounds

    X_true_ood = X_abs_eng[is_true_ood]
    actions_true_ood = actions_abs_raw[is_true_ood]
    Y_true_ood = Y_abs_raw[is_true_ood]

    print(f"Total Active Training Space (ID): {len(X_train_eng)}")
    print(f"Total True OOD (Absolute Holdout): {len(X_true_ood)}")

    # Split ID data (80% Train, 10% Val, 10% Test) [3]
    X_temp, X_test, actions_temp, actions_test, Y_temp, Y_test = train_test_split(
        X_in, actions_in, Y_in, test_size=0.1, random_state=42
    )
    X_train, X_val, actions_train, actions_val, Y_train, Y_val = train_test_split(
        X_temp, actions_temp, Y_temp, test_size=1/9, random_state=42
    )

    # =========================================================
    # 4. NORMALIZATION & SCALING
    # =========================================================
    print("Fitting scalers...")

    # State Scaling (Fit on ID Training data only) [3]
    # Change from .reshape(-1, 4) to .reshape(-1, 5) to account for the new torque column
    X_train_flat = X_train.reshape(-1, 5)
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    
    X_val_scaled = x_scaler.transform(X_val.reshape(-1, 5)).reshape(X_val.shape)
    X_test_scaled = x_scaler.transform(X_test.reshape(-1, 5)).reshape(X_test.shape)
    X_true_ood_scaled = x_scaler.transform(X_true_ood.reshape(-1, 5)).reshape(X_true_ood.shape)
    
    # Action Scaling (Fit on ID Action Training only) [3]
    actions_train_flat = actions_train.reshape(-1, num_actions)
    action_scaler = StandardScaler()
    actions_train_scaled = action_scaler.fit_transform(actions_train_flat).reshape(actions_train.shape)
    
    actions_val_scaled = action_scaler.transform(actions_val.reshape(-1, num_actions)).reshape(actions_val.shape)
    actions_test_scaled = action_scaler.transform(actions_test.reshape(-1, num_actions)).reshape(actions_test.shape)
    actions_true_ood_scaled = action_scaler.transform(actions_true_ood.reshape(-1, num_actions)).reshape(actions_true_ood.shape)

    # Y Scaling - Fit strictly to the training bounds to simulate un-modeled OOD limits [3]
    abs_min = [sysid_bounds[k][0] for k in sysid_bounds.keys()]
    abs_max = [sysid_bounds[k][1] for k in sysid_bounds.keys()]
    y_limits = np.array([abs_min, abs_max])

    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler.fit(y_limits) 

    Y_train_scaled = y_scaler.transform(Y_train)
    Y_val_scaled = y_scaler.transform(Y_val)
    Y_test_scaled = y_scaler.transform(Y_test)
    Y_true_ood_scaled = y_scaler.transform(Y_true_ood)  # Scales beyond [-1.0, 1.0] [3]

    # =========================================================
    # 5. SAVE PROCESSED DATA AND SCALERS
    # =========================================================
    print("Saving processed data and scalers...")
    np.savez_compressed(
        PROCESSED_DATA_PATH,
        # ID Datasets (NLL Targets) [3]
        states_train=X_train_scaled, actions_train=actions_train_scaled, Y_train=Y_train_scaled,
        states_val=X_val_scaled,     actions_val=actions_val_scaled,     Y_val=Y_val_scaled,
        states_test=X_test_scaled,   actions_test=actions_test_scaled,   Y_test=Y_test_scaled,
        
        # True OOD Dataset (Absolute Holdout harvested from the absolute run) [3]
        states_true_ood=X_true_ood_scaled, actions_true_ood=actions_true_ood_scaled, Y_true_ood=Y_true_ood_scaled
    )
    
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump({'x_scaler': x_scaler, 'action_scaler': action_scaler, 'y_scaler': y_scaler}, f)

    print("Preprocessing Complete!")
    print(f"Train ID Shape: {X_train_scaled.shape} / {actions_train_scaled.shape}")
    print(f"True OOD Shape: {X_true_ood_scaled.shape} / {actions_true_ood_scaled.shape}")


if __name__ == "__main__":
    main()