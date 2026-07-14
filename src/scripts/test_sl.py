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

def plot_sparsification_curve(Y_test, mu_physical, sigma_physical, param_names, figures_path):
    """
    Plots the Error-Rejection (Sparsification) curve.
    X-axis: % of highest-uncertainty predictions rejected.
    Y-axis: Mean Absolute Error (MAE) of the remaining dataset.
    """
    n_params = len(param_names)
    n_cols = 3
    n_rows = int(np.ceil(n_params / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.2 * n_rows), squeeze=False)
    fig.suptitle("Uncertainty Sparsification (Error-Rejection) Curves", fontsize=14, y=0.98)
    
    # We evaluate dropping from 0% up to 95% of the data (can't drop 100% or MAE is undefined)
    drop_fractions = np.linspace(0, 0.95, 50) 
    
    for i, p_name in enumerate(param_names):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        abs_err = np.abs(mu_physical[:, i] - Y_test[:, i])
        sig = sigma_physical[:, i]
        N = len(abs_err)
        
        # Sort indices ascending: lowest sigma (most confident) first
        sort_idx_sig = np.argsort(sig)
        
        # Oracle sort: lowest actual error first (perfect self-awareness)
        sort_idx_err = np.argsort(abs_err)
        
        mae_by_sigma = []
        mae_by_oracle = []
        
        for f in drop_fractions:
            n_keep = max(1, int(N * (1 - f))) 
            
            # Keep only the most confident `n_keep` samples
            kept_errs_by_sig = abs_err[sort_idx_sig[:n_keep]]
            kept_errs_by_oracle = abs_err[sort_idx_err[:n_keep]]
            
            mae_by_sigma.append(np.mean(kept_errs_by_sig))
            mae_by_oracle.append(np.mean(kept_errs_by_oracle))
            
        # Plot curves
        ax.plot(drop_fractions * 100, mae_by_sigma, label="Model Uncertainty (σ)", color="blue", linewidth=2)
        ax.plot(drop_fractions * 100, mae_by_oracle, label="Oracle (True Error)", color="black", linestyle="--", alpha=0.7)
        
        ax.set_title(p_name, fontsize=11)
        ax.set_xlabel("% of Data Rejected (Highest σ)", fontsize=9)
        ax.set_ylabel("MAE of Remaining Data", fontsize=9)
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(fontsize=9)
            
    # Clean up unused grid panels
    for i in range(n_params, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis("off")
        
    fig.tight_layout()
    fig.subplots_adjust(top=0.92 - (0.02 * n_rows), hspace=0.35, wspace=0.3)
    
    output_path = os.path.join(figures_path, "sparsification_curves.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved Sparsification Curves to: {output_path}")


def plot_dual_sparsification_curve(Y_test, mu_physical, sigma_physical, param_names, figures_path):
    """
    Plots a Dual-Sided Error-Rejection (Sparsification) curve with respective Oracles.
    Splits the test data into Left (Low) and Right (High) halves based on the ground truth 
    median to check if the uncertainty metric holds up equally across both sides of the data.
    """
    n_params = len(param_names)
    n_cols = 3
    n_rows = int(np.ceil(n_params / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.2 * n_rows), squeeze=False)
    fig.suptitle("Dual-Sided Uncertainty Sparsification (Left vs. Right with Oracles)", fontsize=14, y=0.98)
    
    drop_fractions = np.linspace(0, 0.95, 50) 
    
    for i, p_name in enumerate(param_names):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        abs_err = np.abs(mu_physical[:, i] - Y_test[:, i])
        sig = sigma_physical[:, i]
        gt = Y_test[:, i]
        
        # Split the data into Left (lower half) and Right (upper half) based on median ground truth
        median_val = np.median(gt)
        left_mask = gt <= median_val
        right_mask = gt > median_val
        
        # Helper function to compute both Sigma and Oracle curves for a subset
        def compute_subset_curves(mask):
            sub_err = abs_err[mask]
            sub_sig = sig[mask]
            sub_N = len(sub_err)
            
            if sub_N == 0:
                blank = [0.0] * len(drop_fractions)
                return blank, blank
                
            # Sort by predicted uncertainty (sigma)
            sort_idx_sig = np.argsort(sub_sig)
            # Sort by true error (oracle)
            sort_idx_err = np.argsort(sub_err)
            
            mae_sigma_line = []
            mae_oracle_line = []
            
            for f in drop_fractions:
                n_keep = max(1, int(sub_N * (1 - f)))
                
                mae_sigma_line.append(np.mean(sub_err[sort_idx_sig[:n_keep]]))
                mae_oracle_line.append(np.mean(sub_err[sort_idx_err[:n_keep]]))
                
            return mae_sigma_line, mae_oracle_line

        # Compute curves
        left_sigma, left_oracle = compute_subset_curves(left_mask)
        right_sigma, right_oracle = compute_subset_curves(right_mask)
        
        # Plot Model Curves (Solid Lines)
        ax.plot(drop_fractions * 100, left_sigma, label="Left Half (Model σ)", color="teal", linewidth=2)
        ax.plot(drop_fractions * 100, right_sigma, label="Right Half (Model σ)", color="darkorange", linewidth=2)
        
        # Plot Oracle Curves (Dashed Lines)
        ax.plot(drop_fractions * 100, left_oracle, label="Left Oracle", color="teal", linestyle="--", alpha=0.6)
        ax.plot(drop_fractions * 100, right_oracle, label="Right Oracle", color="darkorange", linestyle="--", alpha=0.6)
        
        ax.set_title(f"{p_name} (Median: {median_val:.3f})", fontsize=11)
        ax.set_xlabel("% of Data Rejected (Highest σ)", fontsize=9)
        ax.set_ylabel("MAE of Remaining Data", fontsize=9)
        ax.grid(alpha=0.2)
        
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
            
    for i in range(n_params, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis("off")
        
    fig.tight_layout()
    fig.subplots_adjust(top=0.92 - (0.02 * n_rows), hspace=0.35, wspace=0.3)
    
    output_path = os.path.join(figures_path, "sparsification_curves_left_right_oracle.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved Dual-Sided Sparsification Curves with Oracles to: {output_path}")


# --- MODULAR METRICS & PLOTTING FUNCTION ---
def compute_and_plot_metrics(Y_test, mu_physical, sigma_physical, param_names, param_ranges, sysid_bounds, signals_test):
    """
    Consolidated metric calculation and visualization engine.
    Add any new evaluation metrics or custom charts directly inside this function.
    """
    # 1. Global Performance Printout
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

    # 2. Scatter Plotting Engine
    n_params = len(param_names)
    n_cols = 3
    n_rows = int(np.ceil(n_params / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.2 * n_rows), squeeze=False)
    fig.suptitle("Ground Truth vs. Predicted Parameters (Colored by Signal Type)", fontsize=14, y=0.98)
    
    signals_named = np.array([SIGNAL_NAMES.get(idx, f"Other ({idx})") for idx in signals_test])
    unique_signals = np.unique(signals_named)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_signals)))
    color_map = dict(zip(unique_signals, colors))

    for i, p_name in enumerate(param_names):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        for s_type in unique_signals:
            mask = (signals_named == s_type)
            ax.scatter(
                Y_test[mask, i], 
                mu_physical[mask, i], 
                color=color_map[s_type], 
                s=4.0, 
                alpha=0.6, 
                label=s_type if i == 0 else ""
            )
            
        p_min, p_max = sysid_bounds[p_name]
        ax.plot([p_min, p_max], [p_min, p_max], color="black", linestyle="--", alpha=0.5, linewidth=1.2)
        ax.set_title(p_name, fontsize=11)
        ax.set_xlabel("Ground Truth", fontsize=9)
        ax.set_ylabel("Predicted Mean", fontsize=9)
        ax.grid(alpha=0.2)
        ax.set_xlim(p_min, p_max)
        ax.set_ylim(p_min, p_max)

    for i in range(n_params, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis("off")

    fig.tight_layout()
    fig.subplots_adjust(top=0.92 - (0.02 * n_rows), bottom=0.15 if n_rows > 1 else 0.24, hspace=0.35, wspace=0.3)
    fig.legend(loc="lower center", ncol=min(len(unique_signals), 5), bbox_to_anchor=(0.5, 0.01), fontsize=9)
    
    output_scatter_path = os.path.join(FIGURES_PATH, "ground_truth_vs_predictions.png")
    fig.savefig(output_scatter_path, dpi=300, bbox_inches="tight")
    print(f"Saved Ground Truth vs. Prediction scatter plots to: {output_scatter_path}")

    # 3. Signal Contribution & Heatmap Matrices
    signal_mae_matrix = np.zeros((n_params, len(unique_signals)))
    signal_sigma_matrix = np.zeros((n_params, len(unique_signals)))

    for col_idx, s_type in enumerate(unique_signals):
        mask = (signals_named == s_type)
        for row_idx, p_name in enumerate(param_names):
            raw_mae = np.mean(np.abs(mu_physical[mask, row_idx] - Y_test[mask, row_idx]))
            signal_mae_matrix[row_idx, col_idx] = (raw_mae / param_ranges[row_idx]) * 100 
            
            raw_sigma = np.mean(sigma_physical[mask, row_idx])
            signal_sigma_matrix[row_idx, col_idx] = (raw_sigma / param_ranges[row_idx]) * 100

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
    
    fig_heat, (ax_heat_mae, ax_heat_sig) = plt.subplots(1, 2, figsize=(15, 6))
    
    im_mae = ax_heat_mae.imshow(signal_mae_matrix, cmap="YlOrRd", aspect="auto")
    ax_heat_mae.set_title("Normalized Mean Absolute Error (% of Parameter Range)", fontsize=12)
    ax_heat_mae.set_xticks(np.arange(len(unique_signals)))
    ax_heat_mae.set_xticklabels(unique_signals, rotation=30, ha="right", fontsize=9)
    ax_heat_mae.set_yticks(np.arange(n_params))
    ax_heat_mae.set_yticklabels(param_names, fontsize=9)
    fig_heat.colorbar(im_mae, ax=ax_heat_mae, label="MAE (%)")
    
    for r in range(n_params):
        for c in range(len(unique_signals)):
            ax_heat_mae.text(c, r, f"{signal_mae_matrix[r, c]:.2f}%", ha="center", va="center", 
                             color="black" if signal_mae_matrix[r, c] < 8.0 else "white", fontsize=9)

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
    
    # 4. Generate the Sparsification (Error-Rejection) Curves
    plot_sparsification_curve(
        Y_test=Y_test, 
        mu_physical=mu_physical, 
        sigma_physical=sigma_physical, 
        param_names=param_names, 
        figures_path=FIGURES_PATH
    )
    
    # 5. Generate the Dual Sparsification (Error-Rejection) Curves
    plot_dual_sparsification_curve(
        Y_test=Y_test, 
        mu_physical=mu_physical, 
        sigma_physical=sigma_physical, 
        param_names=param_names, 
        figures_path=FIGURES_PATH
    )


# --- MAIN PIPELINE ---
def main():
    print(f"Device: {DEVICE}")
    
    with open(SCALER_PATH, 'rb') as f:
        scalers = pickle.load(f)
    x_scaler = scalers['x_scaler']
    action_scaler = scalers['action_scaler']
    y_scaler = scalers['y_scaler']

    print("Loading raw training dataset...")
    raw_data = np.load(RAW_TRAIN_PATH)
    X_raw = raw_data['trajectories']
    actions_raw = raw_data['actions']
    Y_raw = raw_data['parameters']
    signals_raw = raw_data['signal_types']
    
    from src.envs.pendulum import SinglePendulumEnv
    dummy_env = SinglePendulumEnv(render_mode=None)
    DT = dummy_env.model.opt.timestep * dummy_env.FRAME_SKIP
    dummy_env.close()

    X_engineered = preprocess_raw_data(X_raw)
    
    sysid_bounds = config["sysid_bounds"]
    param_names = list(sysid_bounds.keys())
    param_ranges = np.array([sysid_bounds[k][1] - sysid_bounds[k][0] for k in param_names])

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
        
        id_mask = (Y_raw[:, idx] >= lower_bound) & (Y_raw[:, idx] <= upper_bound)
        id_masks.append(id_mask)
        
    is_id = np.all(np.stack(id_masks, axis=0), axis=0)

    X_in = X_engineered[is_id]
    actions_in = actions_raw[is_id]
    Y_in = Y_raw[is_id][:, active_indices]  
    signals_in = signals_raw[is_id]

    if config["dataset"]["include_rl_dataset"]:
        rl_data_path = config["dataset"]["rl_raw_processed_path"]
        if os.path.exists(rl_data_path):
            print(f"Loading and appending RL dataset to matching split space: {rl_data_path}")
            rl_data = np.load(rl_data_path)
            X_rl_raw = rl_data['trajectories']
            actions_rl = rl_data['actions']
            Y_rl = rl_data['parameters']
            
            signals_rl = np.full(X_rl_raw.shape[0], 10, dtype=np.int32)
            
            X_rl_eng = preprocess_raw_data(X_rl_raw)
            
            X_in = np.concatenate([X_in, X_rl_eng], axis=0)
            actions_in = np.concatenate([actions_in, actions_rl], axis=0)
            Y_in = np.concatenate([Y_in, Y_rl], axis=0)
            signals_in = np.concatenate([signals_in, signals_rl], axis=0)

    _, X_test, _, actions_test, _, Y_test, _, signals_test = train_test_split(
        X_in, actions_in, Y_in, signals_in, test_size=0.1, random_state=42
    )

    print(f"Synchronized unseen ID Test Set size: {len(X_test)} episodes")

    X_test_scaled = x_scaler.transform(X_test.reshape(-1, 5)).reshape(X_test.shape)
    num_actions = actions_test.shape[2]
    actions_test_scaled = action_scaler.transform(actions_test.reshape(-1, num_actions)).reshape(actions_test.shape)
    
    model = CNNLSTMModel(config=HPARAMS, n_params=N_PARAMS, chkpt_file_pth=CHKPT_PATH, device=DEVICE)
    model.load_model()
    model.eval()
    print("Model loaded successfully.")

    batch_size = 256
    n_batches = int(np.ceil(len(X_test_scaled) / batch_size))
    
    all_mu = []
    all_sigma = []

    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Evaluating model"):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(X_test_scaled))
            
            s_batch = torch.tensor(X_test_scaled[start_idx:end_idx], dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
            a_batch = torch.tensor(actions_test_scaled[start_idx:end_idx], dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
            
            mu, sigma = model.forward(s_batch, a_batch)
            all_mu.append(mu.cpu().numpy())
            all_sigma.append(sigma.cpu().numpy())

    mu_log_space = np.concatenate(all_mu, axis=0)
    sigma_log_space = np.concatenate(all_sigma, axis=0)

    # --- CORRECT LOG-NORMAL PHYSICAL TRANSLATION MATH ---
    # 1. Bring mu from log-space [-1, 1] back into the exponential scaled space [e^-1, e^1]
    mu_scaled_domain = np.exp(mu_log_space)
    mu_physical = y_scaler.inverse_transform(mu_scaled_domain)
    
    # 2. Extract standard deviation in the scaled domain using log-normal distribution variance tracking
    sigma_scaled_domain = np.sqrt(
        (np.exp(sigma_log_space**2) - 1.0) * np.exp(2.0 * mu_log_space + sigma_log_space**2)
    )
    
    # 3. Apply standard deviation scaler translation factor based on the true spread of MinMaxScaler (e^1 - e^-1)
    scaler_range_factor = np.exp(1) - np.exp(-1)
    sigma_physical = sigma_scaled_domain * (param_ranges / scaler_range_factor)

    # --- INVOKE MODULAR FUNCTION ---
    compute_and_plot_metrics(
        Y_test=Y_test,
        mu_physical=mu_physical,
        sigma_physical=sigma_physical,
        param_names=param_names,
        param_ranges=param_ranges,
        sysid_bounds=sysid_bounds,
        signals_test=signals_test
    )
    
    plt.show()

if __name__ == "__main__":
    main()