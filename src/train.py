import os
import torch
import numpy as np
import wandb
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sysid_models import CNNLSTMModel
SYSIDModel = CNNLSTMModel

# --- HYPERPARAMETERS ---
EPOCHS = 500
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "../data/processed_actuator_sysid_dataset.npz"
CHKPT_PATH = "../models/best_sysid_model.pth"

CONFIG = {
    "in_channels": 4,      
    "learning_rate": 1e-4,
    "cnn1_dims": 128,
    "cnn2_dims": 64,
    "lstm_dims": 64,
    "weight_decay": 1e-4,
    "clip_value": 5.0,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS
}

# --- DATASET CLASS ---
class SysIDDataset(Dataset):
    def __init__(self, x_data, y_data):
        # Transpose (N, 600, 4) -> (N, 4, 600) for Conv1d
        self.x = torch.tensor(x_data, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y_data, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# --- TRAINING FUNCTION ---
def train():
    # 1. Initialize WandB
    wandb.init(
        project="NYCU-AI-Capstone-SysID",
        config=CONFIG,
        name="CNN-LSTM-NormFlows"
    )
    
    os.makedirs(os.path.dirname(CHKPT_PATH), exist_ok=True)
    
    print(f"Device: {DEVICE}")
    data = np.load(DATA_PATH)
    
    X_train, Y_train = data['X_train'], data['Y_train']
    # Subsample 10% for experiment
    X_train, Y_train = X_train[:len(X_train)//10], Y_train[:len(Y_train)//10]

    train_loader = DataLoader(SysIDDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(SysIDDataset(data['X_val'],   data['Y_val']),   batch_size=BATCH_SIZE, shuffle=False)
    
    model = CNNLSTMModel(config=CONFIG, n_params=3, chkpt_file_pth=CHKPT_PATH, device=DEVICE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer, 'min', patience=10, factor=0.5
    )
    
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # -- TRAIN --
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            loss = model.learn(batch_x, batch_y)
            train_loss += loss * batch_x.size(0)
        
        avg_train_loss = train_loss / len(train_loader.dataset)

        # -- VALIDATION --
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                mu, _ = model.forward_high_accuracy(batch_x)
                
                # NLL Loss
                # v_loss = model.loss(mu, batch_y, sigma.pow(2))
                # val_loss += v_loss.item() * batch_x.size(0)
                h = model.get_context(batch_x)
                loss = model.flow_model.forward_kld(batch_y, context=h)
                constant = 0.5 * model.n_params * torch.log(2 * torch.pi)
                val_loss = loss.item() - constant
                
                # Accuracy Metric (MSE)
                mse = F.mse_loss(mu, batch_y)
                val_mse += mse.item() * batch_x.size(0)
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_mse = val_mse / len(val_loader.dataset)
        
        # 1. Update Learning Rate
        scheduler.step(avg_val_mse)

        # 2. Log Metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_nll_loss": avg_train_loss,
            "val_nll_loss": avg_val_loss,
            "val_mse": avg_val_mse,
            "learning_rate": CONFIG["learning_rate"]
        })

        print(f"Epoch {epoch+1:02d} | Train NLL: {avg_train_loss:.4f} | Val NLL: {avg_val_loss:.4f} | Val MSE: {avg_val_mse:.6f}")

        # 3. Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_model()
            # Log best metrics as summary
            wandb.run.summary["best_val_nll"] = best_val_loss
            wandb.run.summary["best_val_mse"] = avg_val_mse
            print("  --> Saved Best Model")

    wandb.finish()

if __name__ == "__main__":
    train()