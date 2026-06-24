import os
import pickle
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm

from src.sysid.cnnlstm import CNNLSTMModel
from src.data.preprocess import preprocess_raw_data

# --- LOAD CONFIGURATION & SCALERS ---
with open("./src/configs/sysid_config.yaml", "r") as f:
    config = yaml.safe_load(f)

HPARAMS = config["hyperparameters"]
RAW_TRAIN_PATH = "./dataset/raw_pendulum_sysid_dataset.npz"
SCALER_PATH = config["dataset"]["scalers_path"] if not config["dataset"]["include_rl_dataset"] else config["dataset"]["rl_scalers_path"]
FIGURES_PATH = config["dataset"]["figures_path"]
CHKPT_PATH = config["model"]["chkpt_path"] if not config["dataset"]["include_rl_dataset"] else config["model"]["rl_chkpt_path"]
ENCODER_RESOLUTION = config["dataset"]["encoder_resolution"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_PARAMS = len(config["sysid_bounds"].keys())

# Mapping index values back to clean display names [3]
SIGNAL_NAMES = {
    0: "Noise",
    1: "PRBS",
    2: "Multisine",
    3: "Impulse",
    4: "Chirp (Normal)",
    5: "Chirp (Normal)",
    6: "Chirp (Shifted)",
    7: "Chirp (Shifted)",
    8: "Chirp (Shifted)",
    9: "Chirp (Shifted)",
    10: "RL Explorer"
}

def get_signal_name(idx):
    return SIGNAL_NAMES.get(int(idx), f"RL/Unknown (ID: {idx})")

def main():
    print(f"Device: {DEVICE}")
    
    # 1. Load fitted scalers [3]
    with open(SCALER_PATH, 'rb') as f:
        scalers = pickle.load(f)
    x_scaler = scalers['x_scaler']
    action_scaler = scalers['action_scaler']
    y_scaler = scalers['y_scaler']

    # 2. Load raw training data to reconstruct test split with signal types [3]
    print("Loading raw training dataset...")
    raw_data = np.load(RAW_TRAIN_PATH)
    X_raw = raw_data['trajectories']
    actions_raw = raw_data['actions']
    Y_raw = raw_data['parameters']
    signals_raw = raw_data['signal_types']
    
    # Compute environment step length
    from src.envs.pendulum import SinglePendulumEnv
    dummy_env = SinglePendulumEnv(render_mode=None)
    DT = dummy_env.model.opt.timestep * dummy_env.FRAME_SKIP
    dummy_env.close()

    # Rerun trig feature mapping on raw data [3]
    X_engineered = preprocess_raw_data(X_raw, ENCODER_RESOLUTION, DT)

    # Reconstruct ID masks strictly matching preprocess.py [3]
    # sysid_bounds = config["sysid_bounds"]
    # param_names = list(sysid_bounds.keys())
    
    # # Track physical bounds ranges for normalization
    # param_ranges = np.array([sysid_bounds[k][1] - sysid_bounds[k][0] for k in param_names])
    
    # MARGIN = 0.02
    # id_masks = []
    # for i, param in enumerate(param_names):
    #     p_min, p_max = sysid_bounds[param]
    #     p_range = p_max - p_min
    #     lower_bound = p_min + (MARGIN * p_range)
    #     upper_bound = p_max - (MARGIN * p_range)
    #     id_mask = (Y_raw[:, i] >= lower_bound) & (Y_raw[:, i] <= upper_bound)
    #     id_masks.append(id_mask)
        
    # is_id = np.all(np.stack(id_masks, axis=0), axis=0)

    # # Filter down to ID pool
    # X_in = X_engineered[is_id]
    # actions_in = actions_raw[is_id]
    # Y_in = Y_raw[is_id]
    # signals_in = signals_raw[is_id]
    
    # Reconstruct ID masks strictly matching preprocess.py [3]
    sysid_bounds = config["sysid_bounds"]
    param_names = list(sysid_bounds.keys())
    
    # Track physical bounds ranges for normalization
    param_ranges = np.array([sysid_bounds[k][1] - sysid_bounds[k][0] for k in param_names])

    # DYNAMIC COLUMN ALIGNMENT: 
    # It attempts to read 'raw_parameter_order' from your config. If it doesn't find it,
    # it safely falls back to the original 6-parameter order to align indices [3].
    RAW_PARAM_ORDER = config["dataset"].get(
        "raw_parameter_order", 
        ["frictionloss", "damping", "armature", "backlash_armature", "backlash_damping"]
    )
    active_indices = [RAW_PARAM_ORDER.index(name) for name in param_names]
    
    MARGIN = 0.02
    id_masks = []
    for idx, param in zip(active_indices, param_names):
        p_min, p_max = sysid_bounds[param]
        p_range = p_max - p_min
        lower_bound = p_min + (MARGIN * p_range)
        upper_bound = p_max - (MARGIN * p_range)
        
        # Correctly index Y_raw dynamically based on the NPZ raw column order [3]
        id_mask = (Y_raw[:, idx] >= lower_bound) & (Y_raw[:, idx] <= upper_bound)
        id_masks.append(id_mask)
        
    is_id = np.all(np.stack(id_masks, axis=0), axis=0)

    # Filter down to ID pool and slice columns strictly to match your active config parameters [3]
    X_in = X_engineered[is_id]
    actions_in = actions_raw[is_id]
    Y_in = Y_raw[is_id][:, active_indices]  # Slices Y to 5 active parameters in the correct order
    signals_in = signals_raw[is_id]

    # =========================================================================
    # NEW: Inject Consolidated RL Explorer Trajectories into the ID evaluation pool
    # =========================================================================
    if config["dataset"]["include_rl_dataset"]:
        rl_data_path = config["dataset"]["rl_raw_processed_path"]
        if os.path.exists(rl_data_path):
            print(f"Loading and appending RL dataset to matching split space: {rl_data_path}")
            rl_data = np.load(rl_data_path)
            X_rl_raw = rl_data['trajectories']
            actions_rl = rl_data['actions']
            Y_rl = rl_data['parameters']
            
            # SIMPLIFIED STEP: Completely ignore internal 0, 1, 2 indices and force index 10
            signals_rl = np.full(X_rl_raw.shape[0], 10, dtype=np.int32)
            
            # Map channel dimensions smoothly matching the wrapper's noise profiles
            X_rl_eng = np.zeros((X_rl_raw.shape[0], X_rl_raw.shape[1], 5))
            X_rl_eng[:, :, 0] = X_rl_raw[:, :, 0]
            X_rl_eng[:, :, 1] = X_rl_raw[:, :, 1]
            X_rl_eng[:, :, 2] = X_rl_raw[:, :, 2]
            X_rl_eng[:, :, 3] = np.cos(X_rl_raw[:, :, 0])
            X_rl_eng[:, :, 4] = np.sin(X_rl_raw[:, :, 0])
            
            # Concat into evaluation baseline space before train_test_split
            X_in = np.concatenate([X_in, X_rl_eng], axis=0)
            actions_in = np.concatenate([actions_in, actions_rl], axis=0)
            Y_in = np.concatenate([Y_in, Y_rl], axis=0)
            signals_in = np.concatenate([signals_in, signals_rl], axis=0)

    # Reconstruct the exact test split (random_state=42) [3]
    _, X_test, _, actions_test, _, Y_test, _, signals_test = train_test_split(
        X_in, actions_in, Y_in, signals_in, test_size=0.1, random_state=42
    )

    print(f"Synchronized unseen ID Test Set size: {len(X_test)} episodes")

    # 3. Apply ID Scalers strictly [3]
    X_test_scaled = x_scaler.transform(X_test.reshape(-1, 5)).reshape(X_test.shape)
    actions_test_scaled = action_scaler.transform(actions_test.reshape(-1, num_actions := actions_test.shape[2])).reshape(actions_test.shape)
    
    # 4. Load Model
    model = CNNLSTMModel(config=HPARAMS, n_params=N_PARAMS, chkpt_file_pth=CHKPT_PATH, device=DEVICE)
    model.load_model()
    model.eval()
    print("Model loaded successfully.")

    # 5. Run Inference
    # Batch processing to prevent GPU memory spikes
    batch_size = 256
    n_batches = int(np.ceil(len(X_test_scaled) / batch_size))
    
    all_mu_scaled = []
    all_sigma_scaled = []

    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Evaluating model"):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(X_test_scaled))
            
            # Format shape to channels-first (N, C, T) for Conv1D [3]
            s_batch = torch.tensor(X_test_scaled[start_idx:end_idx], dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
            a_batch = torch.tensor(actions_test_scaled[start_idx:end_idx], dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
            
            mu, sigma = model.forward(s_batch, a_batch)
            all_mu_scaled.append(mu.cpu().numpy())
            all_sigma_scaled.append(sigma.cpu().numpy())

    mu_scaled = np.concatenate(all_mu_scaled, axis=0)
    sigma_scaled = np.concatenate(all_sigma_scaled, axis=0)

    # 6. Physical Inverse-Scaling Math [3]
    # Predictions (mu) inverse-transform smoothly
    mu_physical = y_scaler.inverse_transform(mu_scaled)
    
    # Standard deviations (sigma) scale linearly with the range of the MinMaxScaler:
    # Since MinMaxScaler maps MinMax to [-1, 1], the scaling factor is range / 2 [3]
    sigma_physical = sigma_scaled * (param_ranges / 2.0)

    # =========================================================================
    # 7. Goal 2: Global Evaluation (Fair Parameter Evaluation across all signal types) [3]
    # =========================================================================
    print("\n" + "="*80)
    print("GLOBAL PHYSICAL PARAMETER PREDICTION ACCURACY")
    print("="*80)
    print(f"{'Parameter':<22} | {'Global MAE (Units)':<20} | {'Global R^2 Score':<18} | {'Avg Pred σ':<15}")
    print("-"*80)
    for i, p_name in enumerate(param_names):
        mae = np.mean(np.abs(mu_physical[:, i] - Y_test[:, i]))
        r2 = r2_score(Y_test[:, i], mu_physical[:, i])
        avg_sigma = np.mean(sigma_physical[:, i])
        print(f"{p_name:<22} | {mae:<20.6f} | {r2:<18.4f} | {avg_sigma:<15.6f}")
    print("="*80)

    # DYNAMIC GRID CALCULATION: Scales perfectly to any odd/even parameter count [3]
    n_params = len(param_names)
    n_cols = 3
    n_rows = int(np.ceil(n_params / n_cols))
    
    # squeeze=False is critical to prevent axes indexing bugs if n_rows = 1 [3]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.2 * n_rows), squeeze=False)
    fig.suptitle("Ground Truth vs. Predicted Parameters (Colored by Signal Type)", fontsize=14, y=0.98)
    
    # Map index to legible signal name strings
    signals_named = np.array([SIGNAL_NAMES.get(idx, f"Other ({idx})") for idx in signals_test])
    unique_signals = np.unique(signals_named)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_signals)))
    color_map = dict(zip(unique_signals, colors))

    for i, p_name in enumerate(param_names):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        # Plot points grouped by signal type to keep legend entries unique
        for s_type in unique_signals:
            mask = (signals_named == s_type)
            ax.scatter(
                Y_test[mask, i], 
                mu_physical[mask, i], 
                color=color_map[s_type], 
                s=4.0, 
                alpha=0.6, 
                label=s_type if i == 0 else "" # Only add to legend on first plot
            )
            
        # Draw ideal y=x reference line
        p_min, p_max = sysid_bounds[p_name]
        ax.plot([p_min, p_max], [p_min, p_max], color="black", linestyle="--", alpha=0.5, linewidth=1.2)
        
        ax.set_title(p_name, fontsize=11)
        ax.set_xlabel("Ground Truth", fontsize=9)
        ax.set_ylabel("Predicted Mean", fontsize=9)
        ax.grid(alpha=0.2)
        ax.set_xlim(p_min, p_max)
        ax.set_ylim(p_min, p_max)

    # Dynamically turn off and hide any unused axes panels in the grid [3]
    for i in range(n_params, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis("off")

    # Responsive spacing for suptitle and bottom legend [3]
    fig.tight_layout()
    fig.subplots_adjust(top=0.92 - (0.02 * n_rows), bottom=0.15 if n_rows > 1 else 0.24, hspace=0.35, wspace=0.3)
    fig.legend(
        loc="lower center", 
        ncol=min(len(unique_signals), 5), 
        bbox_to_anchor=(0.5, 0.01), 
        fontsize=9
    )
    
    output_scatter_path = os.path.join(FIGURES_PATH, "ground_truth_vs_predictions.png")
    fig.savefig(output_scatter_path, dpi=300, bbox_inches="tight")
    print(f"Saved Ground Truth vs. Prediction scatter plots to: {output_scatter_path}")

    # 8. Goal 1: Signal-Type Contribution Analysis [3]
    # We normalize error and sigma by parameter range to allow a clean, scale-invariant heatmap [3]
    signal_mae_matrix = np.zeros((n_params := len(param_names), len(unique_signals)))
    signal_sigma_matrix = np.zeros((n_params, len(unique_signals)))

    for col_idx, s_type in enumerate(unique_signals):
        mask = (signals_named == s_type)
        for row_idx, p_name in enumerate(param_names):
            # Compute MAE normalized by parameter's range (error as % of physical span) [3]
            raw_mae = np.mean(np.abs(mu_physical[mask, row_idx] - Y_test[mask, row_idx]))
            normalized_mae = raw_mae / param_ranges[row_idx]
            signal_mae_matrix[row_idx, col_idx] = normalized_mae * 100 # Convert to percentage
            
            # Compute Average uncertainty normalized by range (% of physical span) [3]
            raw_sigma = np.mean(sigma_physical[mask, row_idx])
            normalized_sigma = raw_sigma / param_ranges[row_idx]
            signal_sigma_matrix[row_idx, col_idx] = normalized_sigma * 100

    # =========================================================================
    # --- INSERT THIS PRINT BLOCK HERE TO PRINT COPY-PASTABLE TABLES ---
    # =========================================================================
    print("\n" + "="*120)
    print("SIGNAL CONTRIBUTION: NORMALIZED MAE (% of Parameter Range) BY SIGNAL TYPE")
    print("="*120)
    header = f"{'Parameter':<18} | " + " | ".join([f"{s_type:<15}" for s_type in unique_signals])
    print(header)
    print("-" * len(header))
    for r in range(n_params):
        row_str = f"{param_names[r]:<18} | " + " | ".join([f"{signal_mae_matrix[r, c]:.2f}%".ljust(15) for c in range(len(unique_signals))])
        print(row_str)
    print("="*120)

    print("\n" + "="*120)
    print("SIGNAL CONTRIBUTION: AVERAGE PREDICTED UNCERTAINTY (σ as % of Parameter Range) BY SIGNAL TYPE")
    print("="*120)
    print(header)
    print("-" * len(header))
    for r in range(n_params):
        row_str = f"{param_names[r]:<18} | " + " | ".join([f"{signal_sigma_matrix[r, c]:.2f}%".ljust(15) for c in range(len(unique_signals))])
        print(row_str)
    print("="*120)
    # =========================================================================
    
    # Plot double-panel Heatmap
    fig_heat, (ax_heat_mae, ax_heat_sig) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Heatmap A: Normalized MAE %
    im_mae = ax_heat_mae.imshow(signal_mae_matrix, cmap="YlOrRd", aspect="auto")
    ax_heat_mae.set_title("Normalized Mean Absolute Error (% of Parameter Range)", fontsize=12)
    ax_heat_mae.set_xticks(np.arange(len(unique_signals)))
    ax_heat_mae.set_xticklabels(unique_signals, rotation=30, ha="right", fontsize=9)
    ax_heat_mae.set_yticks(np.arange(n_params))
    ax_heat_mae.set_yticklabels(param_names, fontsize=9)
    fig_heat.colorbar(im_mae, ax=ax_heat_mae, label="MAE (%)")
    
    # Annotate cells with values
    for r in range(n_params):
        for c in range(len(unique_signals)):
            ax_heat_mae.text(c, r, f"{signal_mae_matrix[r, c]:.2f}%", ha="center", va="center", 
                             color="black" if signal_mae_matrix[r, c] < 8.0 else "white", fontsize=9)

    # Heatmap B: Normalized predicted Uncertainty % [3]
    im_sig = ax_heat_sig.imshow(signal_sigma_matrix, cmap="Purples", aspect="auto")
    ax_heat_sig.set_title("Average Predicted Uncertainty (σ as % of Parameter Range)", fontsize=12)
    ax_heat_sig.set_xticks(np.arange(len(unique_signals)))
    ax_heat_sig.set_xticklabels(unique_signals, rotation=30, ha="right", fontsize=9)
    ax_heat_sig.set_yticks(np.arange(n_params))
    ax_heat_sig.set_yticklabels(param_names, fontsize=9)
    fig_heat.colorbar(im_sig, ax=ax_heat_sig, label="Uncertainty σ (%)")
    
    for r in range(n_params):
        for c in range(len(unique_signals)):
            ax_heat_sig.text(c, r, f"{signal_sigma_matrix[r, c]:.2f}%", ha="center", va="center", 
                             color="black" if signal_sigma_matrix[r, c] < 10.0 else "white", fontsize=9)

    fig_heat.tight_layout()
    output_heat_path = os.path.join(FIGURES_PATH, "signal_contribution_analysis.png")
    fig_heat.savefig(output_heat_path, dpi=300, bbox_inches="tight")
    print(f"Saved Signal Contribution heatmaps to: {output_heat_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
    