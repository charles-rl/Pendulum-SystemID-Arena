import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Graceful fallback in case scienceplots is not installed in the testing environment
try:
    import scienceplots
    plt.style.use('science')
except ImportError:
    print("scienceplots package not found. Falling back to matplotlib default styling.")
    plt.style.use('default')

# Import your corrected model classes
from sysid.cnnlstm import CNNLSTMEncoderDecoderModel, CNNLSTMModel

# --- CONFIGURATION ---
DATA_PATH = "./dataset/processed_pendulum_sysid_dataset.npz"
SCALER_PATH = "./models/scalers.pkl"
CHKPT_PATH = "./models/best_sysid_model_sl_without_pool.pth"
# CHKPT_PATH = "./models/best_sysid_model_sl.pth"
# Toggle which model to evaluate
# Options: "encoder-decoder" (Pre-trained SSL downstream target) or "standard" (Pure SL)
EVAL_MODEL = "encoder-decoder"
# EVAL_MODEL = "standard"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "state_dims": 4,
    "action_dims": 1,
    "in_channels": 5,      # state_dims + action_dims (used for standard model)
    "learning_rate": 1e-4,
    "cnn1_dims": 128,
    "cnn2_dims": 64,
    "lstm_dims": 64,
    "weight_decay": 1e-3,
    "clip_value": 5.0,
}

# --- DATASET CLASS ---
class SysIDTestDataset(Dataset):
    def __init__(self, states_data, actions_data, targets_data):
        # Convert shapes from (N, T, C) -> (N, C, T) for PyTorch Conv1D layers
        self.states = torch.tensor(states_data, dtype=torch.float32).permute(0, 2, 1)
        self.actions = torch.tensor(actions_data, dtype=torch.float32).permute(0, 2, 1)
        self.targets = torch.tensor(targets_data, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.targets[idx]

# --- EVALUATION FUNCTION ---
def evaluate_set(model, loader, model_type, name):
    model.eval()
    all_mus = []
    all_sigmas = []
    all_targets = []
    
    with torch.no_grad():
        for states, actions, targets in tqdm(loader, desc=f"Evaluating {name} Set"):
            states = states.to(DEVICE)
            actions = actions.to(DEVICE)
            targets = targets.to(DEVICE)
            
            if model_type == "encoder-decoder":
                B, _, T = states.size()
                # Feed a fully unmasked trajectory during downstream evaluation
                mask = torch.zeros((B, 1, T), dtype=torch.bool, device=DEVICE)
                mu, sigma = model.forward(mask, states, actions)
            else:
                # Concatenate states and actions along channel dimension for baseline
                x = torch.cat([states, actions], dim=1)
                mu, sigma = model.forward(x)
                
            all_mus.append(mu.cpu().numpy())
            all_sigmas.append(sigma.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    return np.vstack(all_mus), np.vstack(all_sigmas), np.vstack(all_targets)

def main():
    # 1. Load Data and Scalers
    print("Loading data and scalers...")
    data = np.load(DATA_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scalers = pickle.load(f)
    y_scaler = scalers['y_scaler']
    x_scaler = scalers['x_scaler']

    # Unpack keys safely to avoid KeyError discrepancies
    X_test       = data['X_test'] if 'X_test' in data else data['states_test']
    X_ood        = data['X_ood'] if 'X_ood' in data else data['states_ood']
    actions_test = data['actions_test']
    actions_ood  = data['actions_ood']
    Y_test       = data['Y_test'] if 'Y_test' in data else data['targets_test']
    Y_ood        = data['Y_ood'] if 'Y_ood' in data else data['targets_ood']

    # 2. Setup DataLoaders
    test_loader = DataLoader(SysIDTestDataset(X_test, actions_test, Y_test), batch_size=256, shuffle=False)
    ood_loader  = DataLoader(SysIDTestDataset(X_ood, actions_ood, Y_ood), batch_size=256, shuffle=False)

    # 3. Load Models and Run Evaluation
    if EVAL_MODEL == "encoder-decoder":
        print("Loading CNN-LSTM Encoder-Decoder Target Model...")
        model = CNNLSTMEncoderDecoderModel(config=CONFIG, n_params=3, chkpt_file_pth=CHKPT_PATH, device=DEVICE)
        model.mode = "target"  # Freeze state / downstream prediction head mode
        model.load_model()
        print("Model loaded successfully.")
        
        print("Evaluating Test Set (ID)...")
        mu_id, sigma_id, y_id = evaluate_set(model, test_loader, "encoder-decoder", "ID")
        
        print("Evaluating OOD Set...")
        mu_ood, sigma_ood, y_ood = evaluate_set(model, ood_loader, "encoder-decoder", "OOD")
        
    elif EVAL_MODEL == "standard":
        print("Loading Standard CNN-LSTM Model...")
        model = CNNLSTMModel(config=CONFIG, n_params=3, chkpt_file_pth=CHKPT_PATH, device=DEVICE)
        model.load_model()
        print("Model loaded successfully.")
        
        print("Evaluating Test Set (ID)...")
        mu_id, sigma_id, y_id = evaluate_set(model, test_loader, "standard", "ID")
        
        print("Evaluating OOD Set...")
        mu_ood, sigma_ood, y_ood = evaluate_set(model, ood_loader, "standard", "OOD")
    else:
        raise ValueError(f"Unknown EVAL_MODEL switch: {EVAL_MODEL}")

    # 4. Inverse Transform (Back to physical units: Damping, Friction, Armature)
    y_id_phys = y_scaler.inverse_transform(y_id)
    mu_id_phys = y_scaler.inverse_transform(mu_id)
    
    y_ood_phys = y_scaler.inverse_transform(y_ood)
    mu_ood_phys = y_scaler.inverse_transform(mu_ood)

    # 5. CALCULATE FINAL METRICS
    mse_id = np.mean((y_id - mu_id)**2)
    mse_ood = np.mean((y_ood - mu_ood)**2)
    
    avg_sigma_id = np.mean(sigma_id)
    avg_sigma_ood = np.mean(sigma_ood)

    print("\n" + "="*50)
    print(f"RESULTS: IN-DISTRIBUTION (TEST) [{EVAL_MODEL.upper()}]")
    print(f"  MSE (Scaled): {mse_id:.6f}")
    print(f"  Avg Uncertainty (Sigma): {avg_sigma_id:.6f}")
    
    print("-" * 50)
    print(f"RESULTS: OUT-OF-DISTRIBUTION (OOD) [{EVAL_MODEL.upper()}]")
    print(f"  MSE (Scaled): {mse_ood:.6f} ({mse_ood/mse_id:.1f}x higher error)")
    print(f"  Avg Uncertainty (Sigma): {avg_sigma_ood:.6f} ({avg_sigma_ood/avg_sigma_id:.1f}x higher uncertainty)")
    print("="*50)

    # 6. VISUALIZATION
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Graph 1: Ground Truth vs Prediction (Parity Plot)
    # Focus on the first physical param (e.g., Damping)
    ax1.scatter(y_id_phys[:, 0], mu_id_phys[:, 0], alpha=0.3, label='Test (ID)', color='blue')
    ax1.scatter(y_ood_phys[:, 0], mu_ood_phys[:, 0], alpha=0.3, label='OOD', color='red')
    
    # Calculate perfect prediction line based on limits dynamically
    min_val = min(y_id_phys[:, 0].min(), y_ood_phys[:, 0].min())
    max_val = max(y_id_phys[:, 0].max(), y_ood_phys[:, 0].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
    
    ax1.set_xlabel("Ground Truth Parameter [0]")
    ax1.set_ylabel(r"Predicted Mean ($\mu$)")
    ax1.set_title(f"Regression Accuracy: ID vs OOD ({EVAL_MODEL})")
    ax1.legend()

    # Graph 2: Uncertainty Distribution (Histogram)
    ax2.hist(sigma_id.flatten(), bins=50, alpha=0.5, label='ID Uncertainty', color='blue', density=True)
    ax2.hist(sigma_ood.flatten(), bins=50, alpha=0.5, label='OOD Uncertainty', color='red', density=True)
    ax2.set_xlabel(r"Predicted Standard Deviation ($\sigma$)")
    ax2.set_ylabel("Density (Log Scale)")
    ax2.set_yscale('log')
    ax2.set_title("Bayesian: Uncertainty Shift")
    ax2.legend()

    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig(f"../figures/test_results_comparison_{EVAL_MODEL}.png")
    plt.show()

if __name__ == "__main__":
    main()