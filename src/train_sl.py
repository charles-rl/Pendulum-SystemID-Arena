import os
import torch
import numpy as np
import wandb
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sysid.cnnlstm import CNNLSTMEncoderDecoderModel, CNNLSTMModel

# --- CONFIGURATION SWITCHES ---
MODEL_TYPE = "encoder_decoder"  # "encoder_decoder" or "standard"
LOAD_SSL_PRETRAINED = True      # Load weights from SSL pre-training (Only applies to encoder_decoder)
FINE_TUNE = False               # False = Linear Probing (frozen encoder), True = Fine-tuning (train all layers)

# --- HYPERPARAMETERS ---
EPOCHS = 200 if LOAD_SSL_PRETRAINED else 500
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./dataset/processed_pendulum_sysid_dataset.npz"
CHKPT_PATH = "./models/best_sysid_model_sl.pth"
SSL_PRETRAINED_PATH = "./models/best_ssl_encoder_decoder.pth"
N_PARAMS = 3  # mass, inertia, damping, etc.

CONFIG = {
    "state_dims": 4,
    "action_dims": 1,
    "in_channels": 5,      # state_dims + action_dims (used for standard model)
    "learning_rate": 1e-4,
    "cnn1_dims": 128,
    "cnn2_dims": 64,
    "lstm_dims": 64,
    "weight_decay": 1e-4,
    "clip_value": 5.0,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    
    # Meta tags for logging
    "model_type": MODEL_TYPE,
    "pretrained": LOAD_SSL_PRETRAINED if MODEL_TYPE == "encoder_decoder" else False,
    "fine_tune": FINE_TUNE if MODEL_TYPE == "encoder_decoder" else True
}

# --- DATASET CLASS ---
class SysIDSupervisedDataset(Dataset):
    def __init__(self, states, actions, targets):
        # Convert shapes from (N, T, C) -> (N, C, T) for PyTorch Conv1D layers
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
    run_name = f"{MODEL_TYPE.upper()}"
    if MODEL_TYPE == "encoder_decoder":
        run_name += "-FineTune" if FINE_TUNE else "-LinearProbe"
    else:
        run_name += "-SL-Baseline"

    wandb.init(
        project="Pendulum-SystemID-Arena",
        config=CONFIG,
        name=run_name
    )
    
    os.makedirs(os.path.dirname(CHKPT_PATH), exist_ok=True)
    
    print(f"Device: {DEVICE}")
    data = np.load(DATA_PATH)
    
    states_train  = data['states_train']
    states_val    = data['states_val']
    actions_train = data['actions_train']
    actions_val   = data['actions_val']
    targets_train = data['Y_train']
    targets_val   = data['Y_val']

    train_loader = DataLoader(
        SysIDSupervisedDataset(states_train, actions_train, targets_train), 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    val_loader = DataLoader(
        SysIDSupervisedDataset(states_val, actions_val, targets_val), 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )
    
    # 2. Model Selection and Initialization
    if MODEL_TYPE == "encoder_decoder":
        model = CNNLSTMEncoderDecoderModel(config=CONFIG, n_params=N_PARAMS, chkpt_file_pth=CHKPT_PATH, device=DEVICE)
        model.mode = "target"  # Set downstream target mode
        
        if LOAD_SSL_PRETRAINED:
            # Temporarily point model loader to pre-trained SSL weights
            model.chkpt_file_pth = SSL_PRETRAINED_PATH
            model.load_model()
            # Reset save path to the final model destination
            model.chkpt_file_pth = CHKPT_PATH
            
            if FINE_TUNE:
                # Re-enable gradients on encoder and decoder for fine-tuning
                for param in model.encoder.parameters():
                    param.requires_grad = True
                for param in model.decoder.parameters():
                    param.requires_grad = True
                model.encoder.train()
                model.decoder.train()
                print("  --> Loaded Pre-trained SSL Model (Fine-Tuning Mode)")
            else:
                print("  --> Loaded Pre-trained SSL Model (Linear Probing Mode / Frozen)")
    else:
        model = CNNLSTMModel(config=CONFIG, n_params=N_PARAMS, chkpt_file_pth=CHKPT_PATH, device=DEVICE)
        print("  --> Initialized Standard Supervised Learning Model")
        
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer, 'min', patience=10, factor=0.5
    )
    
    best_val_mse = float('inf')

    for epoch in range(EPOCHS):
        # -- TRAIN --
        model.train()
        train_loss = 0.0
        for batch_states, batch_actions, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            B, _, T = batch_states.size()
            batch_states = batch_states.to(DEVICE)
            batch_actions = batch_actions.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)

            if MODEL_TYPE == "encoder_decoder":
                # During downstream tasks, use a zero-mask (all False) so the model sees the entire trajectory
                mask = torch.zeros((B, 1, T), dtype=torch.bool, device=DEVICE)
                loss = model.learn_target(mask, batch_states, batch_actions, batch_targets)
            else:
                # For standard model, concatenate states and actions along the channel dimension (dim=1)
                batch_x = torch.cat([batch_states, batch_actions], dim=1)
                loss = model.learn(batch_x, batch_targets)
                
            train_loss += loss * B
        
        avg_train_loss = train_loss / len(train_loader.dataset)

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

                if MODEL_TYPE == "encoder_decoder":
                    mask = torch.zeros((B, 1, T), dtype=torch.bool, device=DEVICE)
                    mu, sigma = model.forward(mask, batch_states, batch_actions)
                    v_loss = model.target_loss(mu, batch_targets, sigma.pow(2))
                else:
                    batch_x = torch.cat([batch_states, batch_actions], dim=1)
                    mu, sigma = model.forward(batch_x)
                    v_loss = model.loss(mu, batch_targets, sigma.pow(2))
                
                val_loss += v_loss.item() * B
                
                # Standard MSE of predicted parameters (ground truth vs predictions)
                mse = F.mse_loss(mu, batch_targets)
                val_mse += mse.item() * B
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_mse = val_mse / len(val_loader.dataset)
        
        # 1. Update Learning Rate (using true validation MSE of physical parameters)
        scheduler.step(avg_val_mse)

        # 2. Log Metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_nll_loss": avg_train_loss,
            "val_nll_loss": avg_val_loss,
            "val_mse": avg_val_mse,
            "learning_rate": model.optimizer.param_groups[0]['lr']
        })

        print(f"Epoch {epoch+1:02d} | Train NLL: {avg_train_loss:.4f} | Val NLL: {avg_val_loss:.4f} | Val MSE: {avg_val_mse:.6f}")

        # 3. Checkpointing (tracks prediction accuracy on system params)
        if avg_val_mse < best_val_mse:
            best_val_mse = avg_val_mse
            model.save_model()
            wandb.run.summary["best_val_nll"] = avg_val_loss
            wandb.run.summary["best_val_mse"] = best_val_mse
            print(f"  --> Saved Best Model (Val MSE: {best_val_mse:.6f})")

    wandb.finish()

if __name__ == "__main__":
    train()