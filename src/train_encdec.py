import os
import torch
import numpy as np
import wandb
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sysid.cnnlstm import CNNLSTMEncoderDecoderModel
SYSIDModel = CNNLSTMEncoderDecoderModel

# --- HYPERPARAMETERS ---
EPOCHS = 300
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./dataset/processed_pendulum_sysid_dataset.npz"
CHKPT_PATH = "./models/best_ssl_encoder_decoder.pth"
WANDB_EXTRA_TAG = ""

CONFIG = {
    "state_dims": 4,
    "action_dims": 1,
    "learning_rate": 1e-4,
    "cnn1_dims": 128,
    "cnn2_dims": 64,
    "lstm_dims": 64,
    "weight_decay": 1e-4,
    "clip_value": 5.0,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    
    # --- Masked Autoencoder Specific Configs ---
    "num_patches": 10,     
    "mask_ratio": 0.20
}

# --- SSL DATASET CLASS ---
class SysIDSSLDataset(Dataset):
    def __init__(self, x_data, action_data):
        # Convert to channels-first (N, C, T) for PyTorch Conv1D layers
        self.x = torch.tensor(x_data, dtype=torch.float32).permute(0, 2, 1)
        self.actions = torch.tensor(action_data, dtype=torch.float32).permute(0, 2, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.actions[idx]

# --- TRAINING FUNCTION ---
def train():
    # 1. Initialize WandB
    wandb.init(
        project="Pendulum-SystemID-Arena",
        config=CONFIG,
        name=f"MAE-SSL-Pretraining{WANDB_EXTRA_TAG}",
    )
    
    os.makedirs(os.path.dirname(CHKPT_PATH), exist_ok=True)
    
    print(f"Device: {DEVICE}")
    data = np.load(DATA_PATH)
    
    states_train, actions_train = data['states_train'], data['actions_train']
    states_val, actions_val     = data['states_val'], data['actions_val']

    train_loader = DataLoader(SysIDSSLDataset(states_train, actions_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(SysIDSSLDataset(states_val, actions_val), batch_size=BATCH_SIZE, shuffle=False)
    
    model = CNNLSTMEncoderDecoderModel(config=CONFIG, n_params=3, chkpt_file_pth=CHKPT_PATH, device=DEVICE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer, 'min', patience=10, factor=0.5
    )
    
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # -- TRAIN --
        model.train()
        train_loss = 0.0
        for batch_states, batch_actions in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train SSL]"):
            B = batch_states.size(0)
            T = batch_states.size(2)  # sequence_length (e.g., 600)
            batch_states, batch_actions = batch_states.to(DEVICE), batch_actions.to(DEVICE)

            # === YOUR EXACT MASK GENERATION LOGIC ===
            num_patches = CONFIG["num_patches"]
            assert T % num_patches == 0, f"sequence_length must be divisible by {num_patches}"
            patch_size = T // num_patches
            num_masked = int(num_patches * CONFIG["mask_ratio"])

            # Create random mask on device directly to save performance overhead
            noise = torch.rand(B, num_patches, device=DEVICE)
            patch_mask = (torch.argsort(noise, dim=1) < num_masked).float()

            # Expand Patch Mask to full sequence length
            mask_float = patch_mask.unsqueeze(-1).repeat(1, 1, patch_size).view(B, T)
            mask = mask_float.to(torch.bool).unsqueeze(1) # 1 is masked, 0 is unmasked
            # ========================================
            
            # Native function execution call
            loss = model.learn_encdec(mask, batch_states, batch_actions)
            train_loss += loss * B
        
        avg_train_loss = train_loss / len(train_loader.dataset)

        # -- VALIDATION --
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_states, batch_actions in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val SSL]"):
                B = batch_states.size(0)
                T = batch_states.size(2)
                batch_states, batch_actions = batch_states.to(DEVICE), batch_actions.to(DEVICE)

                # Mirror the exact same mask setup for the evaluation slice
                num_patches = CONFIG["num_patches"]
                patch_size = T // num_patches
                num_masked = int(num_patches * CONFIG["mask_ratio"])

                noise = torch.rand(B, num_patches, device=DEVICE)
                patch_mask = (torch.argsort(noise, dim=1) < num_masked).float()

                mask_float = patch_mask.unsqueeze(-1).repeat(1, 1, patch_size).view(B, T)
                mask = mask_float.to(torch.bool).unsqueeze(1)
                
                # Validation forward mirroring the learn_encdec logic without updating gradients
                reconstructed_states = model.forward_encdec(mask, batch_states, batch_actions)
                
                if mask.dim() == 2:
                    mask = mask.unsqueeze(1)
                    
                mask_expand = mask.expand_as(batch_states)
                reconstructed_states_masked_only = reconstructed_states[mask_expand]
                original_states_masked_only = batch_states[mask_expand]

                v_loss = model.encdec_loss(reconstructed_states_masked_only, original_states_masked_only)
                val_loss += v_loss.item() * B
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        # 1. Update Learning Rate
        scheduler.step(avg_val_loss)

        # 2. Log Metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_ssl_loss": avg_train_loss,
            "val_ssl_loss": avg_val_loss,
            "learning_rate": model.optimizer.param_groups[0]['lr']
        })

        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # 3. Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_model()
            wandb.run.summary["best_val_ssl_loss"] = best_val_loss
            print("  --> Saved Best SSL Model Weights")

    wandb.finish()

if __name__ == "__main__":
    train()