import os
import torch
import numpy as np
import wandb
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml

from src.sysid.cnnlstm import CNNLSTMModel

# --- LOAD CONFIGURATION ---
with open("./src/configs/sysid_config.yaml", "r") as f:
    config = yaml.safe_load(f)

HPARAMS = config["hyperparameters"]
DATA_PATH = config["dataset"]["processed_path"]
CHKPT_PATH = config.get("model", {}).get("chkpt_path", "./models/best_sysid_model_sl.pth")

# Pure In-Distribution (ID) Training Configuration [3]
BATCH_SIZE = HPARAMS["batch_size"]

EPOCHS = HPARAMS["epochs"]
MIN_STEPS = HPARAMS["min_steps"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_PARAMS = len(config["sysid_bounds"].keys())

# --- DATASET CLASS ---
class SysIDSupervisedDataset(Dataset):
    def __init__(self, states, actions, targets):
        # Convert shapes from (N, T, C) -> (N, C, T) for PyTorch Conv1D layers [3]
        self.states = torch.tensor(states, dtype=torch.float32).permute(0, 2, 1)
        self.actions = torch.tensor(actions, dtype=torch.float32).permute(0, 2, 1)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.targets[idx]


# --- TRAINING FUNCTION ---
def train():
    # 1. Initialize WandB
    wandb.init(
        project=config["wandb"]["project"],
        config=HPARAMS,
        name=config["wandb"]["run_name"],
    )
    
    os.makedirs(os.path.dirname(CHKPT_PATH), exist_ok=True)
    
    print(f"Device: {DEVICE}")
    print(f"Running Pure ID Training -> Batch Size: {BATCH_SIZE}")
    
    # Load processed partitions (Y_train represents 100% of the training bounds dataset) [3]
    data = np.load(DATA_PATH)
    
    states_train     = data['states_train']
    states_val       = data['states_val']
    actions_train    = data['actions_train']
    actions_val      = data['actions_val']
    targets_train    = data['Y_train']
    targets_val      = data['Y_val']

    # Unified loader utilizing the full batch size for ID training [3]
    train_id_loader = DataLoader(
        SysIDSupervisedDataset(states_train, actions_train, targets_train), 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        SysIDSupervisedDataset(states_val, actions_val, targets_val), 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )
    
    # 2. Model Initialization
    model = CNNLSTMModel(config=HPARAMS, n_params=N_PARAMS, chkpt_file_pth=CHKPT_PATH, device=DEVICE)
    print("  --> Initialized Standard Supervised Learning Model")
        
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer, 'min', patience=10, factor=0.5
    )
    
    best_val_mse = float('inf')

    for epoch in range(EPOCHS):
        # -- TRAIN --
        model.train()
        
        # Accumulate metrics across standard regression profiles only
        epoch_metrics = {"loss": 0.0, "nll_loss": 0.0}
        
        for batch_states_id, batch_actions_id, batch_targets_id in tqdm(train_id_loader, desc=f"Epoch {epoch+1} [Train]"):
            
            # Dynamic Batch-Level Truncation (The RL Simulation Trick) [3]
            max_seq_len = batch_states_id.size(2)
            t = torch.randint(low=MIN_STEPS, high=max_seq_len + 1, size=(1,)).item()
            
            batch_states_id = batch_states_id[:, :, :t]
            batch_actions_id = batch_actions_id[:, :, :t]
            
            # Move ID batches to GPU/CPU
            batch_states_id = batch_states_id.to(DEVICE)
            batch_actions_id = batch_actions_id.to(DEVICE)
            batch_targets_id = batch_targets_id.to(DEVICE)

            # Optimize using the clean, simplified 3-argument learn interface [3]
            step_metrics = model.learn(
                states=batch_states_id,
                actions=batch_actions_id,
                target=batch_targets_id,
            )
            
            # Weighted average calculation
            for k in epoch_metrics.keys():
                if k in step_metrics:
                    epoch_metrics[k] += step_metrics[k] * batch_states_id.size(0)
        
        total_samples = len(train_id_loader.dataset)
        avg_train_loss = epoch_metrics["loss"] / total_samples
        avg_train_nll = epoch_metrics["nll_loss"] / total_samples

        # -- VALIDATION --
        model.eval()
        val_loss = 0.0
        val_mse = 0.0

        with torch.no_grad():
            for batch_states, batch_actions, batch_targets in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                B, _, T = batch_states.size()
                batch_states = batch_states.to(DEVICE)
                batch_actions = batch_actions.to(DEVICE)
                batch_targets = batch_targets.to(DEVICE)

                mu, sigma = model.forward(batch_states, batch_actions)
                v_loss = model.nll_loss(mu, batch_targets, sigma.pow(2))
                
                val_loss += v_loss.item() * B
                
                # Mean Squared Error on parameters [3]
                mse = F.mse_loss(mu, batch_targets)
                val_mse += mse.item() * B
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_mse = val_mse / len(val_loader.dataset)
        
        # 1. Update Learning Rate based on validation accuracy
        scheduler.step(avg_val_mse)

        # 2. Log Metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_total_loss": avg_train_loss,
            "train_nll_loss": avg_train_nll,
            "val_nll_loss": avg_val_loss,
            "val_mse": avg_val_mse,
            "learning_rate": model.optimizer.param_groups[0]['lr']
        })

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {avg_train_loss:.4f} (NLL: {avg_train_nll:.4f}) | "
            f"Val NLL: {avg_val_loss:.4f} | "
            f"Val MSE: {avg_val_mse:.6f}"
        )

        # 3. Checkpointing
        if avg_val_mse < best_val_mse:
            best_val_mse = avg_val_mse
            model.save_model()
            wandb.run.summary["best_val_nll"] = avg_val_loss
            wandb.run.summary["best_val_mse"] = best_val_mse
            print(f"  --> Saved Best Model (Val MSE: {best_val_mse:.6f})")

    wandb.finish()

if __name__ == "__main__":
    train()