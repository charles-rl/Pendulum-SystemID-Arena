import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from data_collection import MINMAX_PARAMS

# --- PATHS ---
RAW_DATA_PATH = "../data/raw_actuator_sysid_dataset.npz"
PROCESSED_DATA_PATH = "../data/processed_actuator_sysid_dataset.npz"
# If you want to skip the noise/quantization step and just use the raw data with engineered features, use this path instead:
# PROCESSED_DATA_PATH = "../data/processed_clean_actuator_sysid_dataset.npz"
SCALER_PATH = "../models/scalers.pkl"

def main():
    print("Loading raw dataset...")
    data = np.load(RAW_DATA_PATH)
    X_raw = data['trajectories']  # Shape: (N, 600, 2) -> [theta, omega]
    Y_raw = data['parameters']    # Shape: (N, 3) -> [damping, friction, armature]
    
    N, timesteps, num_features = X_raw.shape

    # ==========================================
    # 1. QUANTIZATION & NOISE (User Implementation)
    # ==========================================
    print("Applying encoder quantization and noise...")

    # Quantize X_raw based on a 12-bit encoder which has 4096 resolution
    # This simulates quantization error
    encoder_resolution = 4096.0
    theta_noisy = np.round(X_raw[:, :, 0] * (encoder_resolution / (2 * np.pi)))
    # An encoder usually has 3 encoder pulses of noise when operating
    # This is about 3 ticks so (2pi*3)/4096 = 0.0046 radians
    noise_ticks = 3
    theta_noisy += np.round(np.random.normal(0, noise_ticks, theta_noisy.shape))
    # Turn it back to radians
    theta_noisy = theta_noisy * ((2 * np.pi) / encoder_resolution)
    
    dt = 0.02
    sigma_omega = (noise_ticks * 2 * np.pi) / (4096.0 * dt)
    # Add noise to omega (angular velocity)
    omega_noisy = X_raw[:, :, 1] + np.random.normal(0, sigma_omega, X_raw[:, :, 1].shape)
    
    X_noisy = np.zeros(X_raw.shape)
    X_noisy[:, :, 0] = theta_noisy
    X_noisy[:, :, 1] = omega_noisy
    # For a clean baseline, comment out the noise/quantization and just use the raw data
    # X_noisy[:, :, 0] = X_raw[:, :, 0]
    # X_noisy[:, :, 1] = X_raw[:, :, 1]

    # ==========================================
    # 2. FEATURE ENGINEERING (Cos/Sin)
    # ==========================================
    print("Extracting Sin/Cos features...")
    # We want X_engineered to be shape (N, 600, 4) -> [theta, omega, cos(theta), sin(theta)]
    X_engineered = np.zeros((N, timesteps, 4))
    
    X_engineered[:, :, 0] = X_noisy[:, :, 0] # noisy theta
    X_engineered[:, :, 1] = X_noisy[:, :, 1] # noisy omega
    X_engineered[:, :, 2] = np.cos(X_noisy[:, :, 0])      # cos(noisy theta)
    X_engineered[:, :, 3] = np.sin(X_noisy[:, :, 0])      # sin(noisy theta)

    # ==========================================
    # 3. TRAIN / VAL / TEST SPLIT (Including OOD Logic)
    # ==========================================
    print("Splitting dataset into ID (In-Distribution) and OOD (Out-Of-Distribution)...")
    
    # Cut off 2% of the bottom and top of each parameter's physical range
    MARGIN = 0.02
    id_masks =[]
    for i, param in enumerate(["damping", "friction", "armature"]):
        p_min, p_max = MINMAX_PARAMS[param]
        p_range = p_max - p_min
        
        # Calculate the safe inner bounds
        lower_bound = p_min + (MARGIN * p_range)
        upper_bound = p_max - (MARGIN * p_range)
        
        # Create a boolean mask for this specific parameter
        param_mask = (Y_raw[:, i] >= lower_bound) & (Y_raw[:, i] <= upper_bound)
        id_masks.append(param_mask)
    # A sample is ONLY In-Distribution if ALL THREE parameters are inside the safe bounds
    is_id = id_masks[0] & id_masks[1] & id_masks[2]
    is_ood = ~is_id

    X_in = X_engineered[is_id]
    Y_in = Y_raw[is_id]

    X_ood = X_engineered[is_ood]
    Y_ood = Y_raw[is_ood]

    print(f"Total episodes: {len(X_engineered)}")
    print(f"ID episodes (Train/Val/Test): {len(X_in)} ({len(X_in)/len(X_engineered)*100:.1f}%)")
    print(f"OOD episodes (Holdout): {len(X_ood)} ({len(X_ood)/len(X_engineered)*100:.1f}%)")

    # Standard split on the ID data
    # 80% for training, 10% for validation, 10% for testing
    X_temp, X_test, Y_temp, Y_test = train_test_split(X_in, Y_in, test_size=0.1, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=1/9, random_state=42)

    # ==========================================
    # 4. NORMALIZATION & SCALING
    # ==========================================
    print("Fitting scalers...")

    # X Scaling (Z-Score on Training data only)
    X_train_flat = X_train.reshape(-1, 4)
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = x_scaler.transform(X_val.reshape(-1, 4)).reshape(X_val.shape)
    X_test_scaled = x_scaler.transform(X_test.reshape(-1, 4)).reshape(X_test.shape)
    X_ood_scaled = x_scaler.transform(X_ood.reshape(-1, 4)).reshape(X_ood.shape)

    # Y Scaling (ABSOLUTE MIN/MAX to [-1, 1])
    # We create a dummy array representing the absolute physical limits
    y_absolute_limits = np.array([
        [MINMAX_PARAMS["damping"][0], MINMAX_PARAMS["friction"][0], MINMAX_PARAMS["armature"][0]],[MINMAX_PARAMS["damping"][1], MINMAX_PARAMS["friction"][1], MINMAX_PARAMS["armature"][1]]
    ])

    # Use feature_range=(-1, 1) so it pairs perfectly with tanh!
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler.fit(y_absolute_limits) # Fit on the theoretical limits, NOT the training data!

    Y_train_scaled = y_scaler.transform(Y_train)
    Y_val_scaled = y_scaler.transform(Y_val)
    Y_test_scaled = y_scaler.transform(Y_test)
    Y_ood_scaled = y_scaler.transform(Y_ood) # This will scale to values < -0.9 and > 0.9

    # ==========================================
    # 5. SAVE PROCESSED DATA AND SCALERS
    # ==========================================
    print("Saving processed data and scalers...")
    np.savez_compressed(PROCESSED_DATA_PATH,
                        X_train=X_train_scaled, Y_train=Y_train_scaled,
                        X_val=X_val_scaled,     Y_val=Y_val_scaled,
                        X_test=X_test_scaled,   Y_test=Y_test_scaled,
                        X_ood=X_ood_scaled,     Y_ood=Y_ood_scaled)
    
    # Save the scalers so we can inverse_transform predictions during evaluation
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, f)

    print("Preprocessing Complete!")
    print(f"Train shape: {X_train_scaled.shape}")
    print(f"Val shape:   {X_val_scaled.shape}")
    print(f"Test shape:  {X_test_scaled.shape}")

if __name__ == "__main__":
    main()
