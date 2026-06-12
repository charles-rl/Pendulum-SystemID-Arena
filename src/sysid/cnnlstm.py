from .base import BaseModel, BaseEncoderDecoder, LinearHead
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNNLSTMModel(BaseModel):
    def __init__(self, config: dict, n_params, chkpt_file_pth, device):
        super(CNNLSTMModel, self).__init__(chkpt_file_pth, device)
        
        in_channels = config["in_channels"]
        lr = float(config["learning_rate"])
        cnn1_dims = config["cnn1_dims"]
        cnn2_dims = config["cnn2_dims"]
        lstm_dims = config["lstm_dims"]
        weight_decay = float(config["weight_decay"])
        self.clip_value = config["clip_value"]
        
        self.cnn1_low_freq = nn.Conv1d(
            in_channels=in_channels, out_channels=cnn1_dims, kernel_size=7, dilation=16, padding="same"
        )
        self.cnn1_mid_freq = nn.Conv1d(
            in_channels=in_channels, out_channels=cnn1_dims, kernel_size=5, dilation=8, padding="same"
        )
        self.cnn1_high_freq = nn.Conv1d(
            in_channels=in_channels, out_channels=cnn1_dims, kernel_size=3, padding="same"
        )
        self.bn1_low_freq = nn.BatchNorm1d(cnn1_dims)
        self.bn1_mid_freq = nn.BatchNorm1d(cnn1_dims)
        self.bn1_high_freq = nn.BatchNorm1d(cnn1_dims)
        
        # Concatenating the outputs of the 3 CNNs so cnn1_dims * 3
        self.cnn2 = nn.Conv1d(
            in_channels=cnn1_dims * 3, out_channels=cnn2_dims, kernel_size=3, padding="same"
        )
        self.bn2 = nn.BatchNorm1d(cnn2_dims)
        
        # Downsampling to about half the sequence length so that the LSTM sees about 300 timesteps instead of 600 timesteps
        # Which reduces the vanishing gradient
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.lstm = nn.LSTM(
            input_size=cnn2_dims,
            hidden_size=lstm_dims,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.ln_lstm = nn.LayerNorm(lstm_dims * 2)
        
        self.fc = nn.Linear(lstm_dims * 2, lstm_dims)
        self.ln_fc = nn.LayerNorm(lstm_dims)
        
        self.mu_fc = nn.Linear(lstm_dims, n_params)
        self.sigma_fc = nn.Linear(lstm_dims, n_params)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.to(self.device)
        
    def forward(self, x: torch.tensor):
        y_low = self.bn1_low_freq(F.mish(self.cnn1_low_freq(x)))
        y_mid = self.bn1_mid_freq(F.mish(self.cnn1_mid_freq(x)))
        y_high = self.bn1_high_freq(F.mish(self.cnn1_high_freq(x)))
        
        # We want it to be dim=1 because the shape becomes (batch_size, cnn1_dims * 3, timesteps)
        # if dim=-1 then shape becomes (batch_size, cnn1_dims, timesteps * 3)
        y = torch.cat([y_low, y_mid, y_high], dim=1)
        y = self.bn2(F.mish(self.cnn2(y)))
        
        y = self.pool(y)
        
        # swap in_channels and sequence_length
        y = y.permute(0, 2, 1)
        
        output_lstm, (h_n, c_n) = self.lstm(y)
        
        # we only want the h_n
        # flatten
        last_layer_forward_h = h_n[-2, :, :]  # Shape: (batch_size, lstm_dim)
        last_layer_backward_h = h_n[-1, :, :]  # Shape: (batch_size, lstm_dim)
        y = torch.cat((last_layer_forward_h, last_layer_backward_h), dim=1)
        y = self.ln_lstm(F.mish(y))
        
        y = self.ln_fc(F.mish(self.fc(y)))
        
        mu = torch.tanh(self.mu_fc(y)) * 1.2  # Scale the tanh to allow for OOD predictions
        sigma = torch.exp(self.sigma_fc(y)) + 1e-6  # epsilon added to help with the stability when the sigma is near 0
        # sigma = standard deviation
        # sigma^2 = variance = var
        # upon testing exponential and sigma works better
        return mu, sigma
    
    
class CNNLSTMEncoderModel(nn.Module):
    def __init__(
        self,
        state_dims,
        action_dims,
        cnn1_dims,
        cnn2_dims,
        lstm_dims,
    ):
        super(CNNLSTMEncoderModel, self).__init__()
        
        # We only mask the states and not the actions because the actions are known to us and we want to use them as a strong signal for the system identification
        # Reshape token from (S) -> (1, S, 1) so it broadcasts across Batch and Length
        self.mask_state_token = nn.Parameter(torch.zeros(1, state_dims, 1))
        
        # We combine state and action
        in_channels = state_dims + action_dims
        self.cnn1_low_freq = nn.Conv1d(
            in_channels=in_channels, out_channels=cnn1_dims, kernel_size=7, dilation=16, padding="same"
        )
        self.cnn1_mid_freq = nn.Conv1d(
            in_channels=in_channels, out_channels=cnn1_dims, kernel_size=5, dilation=8, padding="same"
        )
        self.cnn1_high_freq = nn.Conv1d(
            in_channels=in_channels, out_channels=cnn1_dims, kernel_size=3, padding="same"
        )
        self.bn1_low_freq = nn.BatchNorm1d(cnn1_dims)
        self.bn1_mid_freq = nn.BatchNorm1d(cnn1_dims)
        self.bn1_high_freq = nn.BatchNorm1d(cnn1_dims)
        
        # Concatenating the outputs of the 3 CNNs so cnn1_dims * 3
        self.cnn2 = nn.Conv1d(
            in_channels=cnn1_dims * 3, out_channels=cnn2_dims, kernel_size=3, padding="same"
        )
        self.bn2 = nn.BatchNorm1d(cnn2_dims)
        
        # Downsampling to about half the sequence length so that the LSTM sees about 300 timesteps instead of 600 timesteps
        # Which reduces the vanishing gradient
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.lstm = nn.LSTM(
            input_size=cnn2_dims,
            hidden_size=lstm_dims,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        # TODO: Unsure if I should keep the layer norm here
        self.ln_lstm = nn.LayerNorm(lstm_dims * 2)
        
    def forward(self, mask, states, actions):
        # REMOVE 'with torch.no_grad():' so the mask token receives gradients!
        masked_states = torch.where(mask, self.mask_state_token, states)
        x = torch.cat([masked_states, actions], dim=1)
        
        y_low = self.bn1_low_freq(F.mish(self.cnn1_low_freq(x)))
        y_mid = self.bn1_mid_freq(F.mish(self.cnn1_mid_freq(x)))
        y_high = self.bn1_high_freq(F.mish(self.cnn1_high_freq(x)))
        
        # We want it to be dim=1 because the shape becomes (batch_size, cnn1_dims * 3, timesteps)
        # if dim=-1 then shape becomes (batch_size, cnn1_dims, timesteps * 3)
        y = torch.cat([y_low, y_mid, y_high], dim=1)
        y = self.bn2(F.mish(self.cnn2(y)))
        
        y = self.pool(y)
        
        # swap in_channels and sequence_length
        y = y.permute(0, 2, 1)
        
        output_lstm, (h_n, c_n) = self.lstm(y)
        
        # Global Average Pooling (Mean)
        mean_latent = output_lstm.mean(dim=1) # Shape: (batch, 2 * lstm_dims)
        
        # Global Max Pooling (Max)
        # Note: we unpack the tuple to only get the max values, discarding indices
        max_latent, _ = output_lstm.max(dim=1) # Shape: (batch, 2 * lstm_dims)
        
        # Concatenate along the feature dimension
        latent = torch.cat([mean_latent, max_latent], dim=1) # Shape: (batch, 4 * lstm_dims)
        
        return latent, output_lstm
    
    
class CNNLSTMDecoderModel(nn.Module):
    def __init__(self, lstm_dims, cnn2_dims, cnn1_dims, state_dims):
        super(CNNLSTMDecoderModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=2 * lstm_dims,
            hidden_size=cnn2_dims // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        # TODO: Confused if this should be 2 * cnn2_dims or the other way around like this
        self.ln_lstm = nn.LayerNorm(cnn2_dims)
        
        # Instead of pool, add an upsample layer to restore the sequence length
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Concatenating the outputs of the 3 CNNs so cnn1_dims * 3
        self.cnn2 = nn.Conv1d(
            in_channels=cnn2_dims, out_channels=cnn1_dims * 3, kernel_size=3, padding="same"
        )
        self.bn2 = nn.BatchNorm1d(cnn1_dims * 3)
        
        # Only reconstruct the states and not the actions because the actions are known to us
        self.cnn1_dims = cnn1_dims
        self.cnn1_low_freq = nn.Conv1d(
            in_channels=cnn1_dims, out_channels=state_dims, kernel_size=7, dilation=16, padding="same"
        )
        self.cnn1_mid_freq = nn.Conv1d(
            in_channels=cnn1_dims, out_channels=state_dims, kernel_size=5, dilation=8, padding="same"
        )
        self.cnn1_high_freq = nn.Conv1d(
            in_channels=cnn1_dims, out_channels=state_dims, kernel_size=3, padding="same"
        )
        # TODO: Unsure if I should keep the batch norm here in the decoder
        self.bn1_low_freq = nn.BatchNorm1d(state_dims)
        self.bn1_mid_freq = nn.BatchNorm1d(state_dims)
        self.bn1_high_freq = nn.BatchNorm1d(state_dims)
        
        self.merge_conv = nn.Conv1d(state_dims * 3, state_dims, kernel_size=1)
        
    def forward(self, output_lstm):
        # Input must preserve order from the encoder, not use the hidden latent state
        output_lstm, (h_n, c_n) = self.lstm(output_lstm)
        output_lstm = self.ln_lstm(F.mish(output_lstm))
        
        # swap in_channels and sequence_length
        y = output_lstm.permute(0, 2, 1)
        
        y = self.upsample(y)
        
        y = self.bn2(F.mish(self.cnn2(y)))
        
        y_low, y_mid, y_high = torch.split(y, self.cnn1_dims, dim=1)
        
        out_low = self.bn1_low_freq(F.mish(self.cnn1_low_freq(y_low)))
        out_mid = self.bn1_mid_freq(F.mish(self.cnn1_mid_freq(y_mid)))
        out_high = self.bn1_high_freq(F.mish(self.cnn1_high_freq(y_high)))
        
        y_combined = torch.cat([out_low, out_mid, out_high], dim=1)
        recon_states = self.merge_conv(y_combined)
        return recon_states
        
    
class CNNLSTMEncoderDecoderModel(BaseEncoderDecoder):
    def __init__(self, config: dict, n_params, chkpt_file_pth, device):
        super(CNNLSTMEncoderDecoderModel, self).__init__(chkpt_file_pth, device)
        
        self.state_dims = config["state_dims"]
        action_dims = config["action_dims"]
        cnn1_dims = config["cnn1_dims"]
        cnn2_dims = config["cnn2_dims"]
        lstm_dims = config["lstm_dims"]
        lr = float(config["learning_rate"])
        weight_decay = float(config["weight_decay"])
        self.clip_value = config["clip_value"]
        
        self.encoder = CNNLSTMEncoderModel(
            state_dims=self.state_dims,
            action_dims=action_dims,
            cnn1_dims=cnn1_dims,
            cnn2_dims=cnn2_dims,
            lstm_dims=lstm_dims,
        ).to(self.device)
        
        self.decoder = CNNLSTMDecoderModel(
            lstm_dims=lstm_dims,
            cnn2_dims=cnn2_dims,
            cnn1_dims=cnn1_dims,
            state_dims=self.state_dims,
        ).to(self.device)
        
        self.mu_sigma = LinearHead(lstm_dims, n_params).to(self.device)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward_encdec(self, mask, states, actions):
        # TODO: you may need to add to device here
        _, output_lstm = self.encoder(mask, states, actions)
        reconstructed_states = self.decoder(output_lstm)
        return reconstructed_states
    
    def forward(self, mask, states, actions):
        with torch.no_grad():
            # Just to be safe and to make sure encoder is frozen
            latent, _ = self.encoder(mask, states, actions)
        mu, sigma = self.mu_sigma(latent)
        return mu, sigma
