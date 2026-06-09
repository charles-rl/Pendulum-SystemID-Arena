from .base import BaseEncoderDecoder, LinearHead
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNNTokenizer(nn.Module):
    def __init__(self, in_channels, cnn_dims, cnn_type="action"):
        """_summary_

        Args:
            in_channels (_type_): _description_
            cnn_dims (_type_): _description_
            cnn_type (str, optional): _description_. Defaults to "action".
            cnn_type can be action or state.
            Action CNN sees the entire sequence so higher dilation.
            State CNN sees only the current patch so lower dilation.
        """
        super(CNNTokenizer, self).__init__()
        dilation_low = 16 if cnn_type == "action" else 3
        dilation_mid = 8 if cnn_type == "action" else 2
        self.cnn1_low_freq = nn.Conv1d(
            in_channels=in_channels, out_channels=cnn_dims, kernel_size=7, dilation=dilation_low, padding="same"
        )
        self.cnn1_mid_freq = nn.Conv1d(
            in_channels=in_channels, out_channels=cnn_dims, kernel_size=5, dilation=dilation_mid, padding="same"
        )
        self.cnn1_high_freq = nn.Conv1d(
            in_channels=in_channels, out_channels=cnn_dims, kernel_size=3, padding="same"
        )
        self.bn1_low_freq = nn.BatchNorm1d(cnn_dims)
        self.bn1_mid_freq = nn.BatchNorm1d(cnn_dims)
        self.bn1_high_freq = nn.BatchNorm1d(cnn_dims)
        
    def forward(self, x):
        # x shape: (batch_size * num_patches, state_size, patch_size)
        y_low = self.bn1_low_freq(F.mish(self.cnn1_low_freq(x)))
        y_mid = self.bn1_mid_freq(F.mish(self.cnn1_mid_freq(x)))
        y_high = self.bn1_high_freq(F.mish(self.cnn1_high_freq(x)))
        
        # We want it to be dim=1 because the shape becomes (batch_size, cnn1_dims * 3, timesteps)
        # if dim=-1 then shape becomes (batch_size, cnn1_dims, timesteps * 3)
        y = torch.cat([y_low, y_mid, y_high], dim=1)
        return y
    
class RotaryContinuous(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, t):
        # x shape: (batch, num_heads, seq_len, head_dim)
        # t shape: (batch, seq_len)  <--- Now includes batch!
        t = t.type_as(self.inv_freq)

        # 'b' = batch, 'i' = sequence, 'j' = frequencies
        # This multiplies each timestamp by the frequencies per batch
        freqs = torch.einsum("bi,j->bij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

        # Reshape to (batch, 1, seq_len, head_dim) to broadcast across heads
        # Notice we changed [None, None, :, :] to [:, None, :, :]
        cos = emb.cos()[:, None, :, :]
        sin = emb.sin()[:, None, :, :]
        return cos, sin

# rotary pos emb helpers:
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, base=10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # RoPE operates on the individual head dimension
        self.rotary_pos_emb = RotaryContinuous(self.head_dim, base=base)

        # Projections output the full dimension (we split it into heads later)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Standard practice: A final projection layer to mix the heads back together
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, timesteps):
        # x is your CNN output: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, _ = x.shape

        # 1. Linear Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape for Multi-Head: (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE to Q and K (after projection and reshaping)
        cos, sin = self.rotary_pos_emb(q, timesteps)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 4. Scaled Dot-Product Attention (handles the math and softmax for you)
        output = F.scaled_dot_product_attention(q, k, v)

        # 5. Reshape back into flat sequence: (batch, seq_len, embed_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # 6. Final mix
        return self.out_proj(output)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_multiplier=4, dropout=0.0, base=10000):
        """
        Chose to do post-norm here because we are using a shallow network,
        so gradients aren't a big problem unlike SOTA LLMs.
        """
        super().__init__()
        self.attention = Attention(embed_dim, num_heads, base=base)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim * hidden_multiplier)
        self.fc2 = nn.Linear(embed_dim * hidden_multiplier, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, timesteps):
        # Attention
        attn_output = self.attention(x, timesteps)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # FFN Block, apply mish only to first layer
        x_ = self.fc1(x)
        x_ = F.mish(x_)
        x_ = self.dropout(x_)

        x_ = self.fc2(x_)
        x_ = self.dropout(x_)
        x = self.norm2(x + x_)
        return x


class CNNTransformerEncoder(nn.Module):
    def __init__(
        self,
        batch_size,
        num_patches,
        num_masked,
        patch_size,
        cnn_dims, 
        state_in_channels,
        action_in_channels,
        num_heads,
        hidden_multiplier,
        dropout,
        base=10000
    ):
        super(CNNTransformerEncoder, self).__init__()
        self.batch_size = batch_size
        self.num_patches = num_patches
        self.num_masked = num_masked
        self.patch_size = patch_size
        self.cnn_dims = cnn_dims
        self.state_in_channels = state_in_channels
        self.layernorm = nn.LayerNorm(cnn_dims * 3)
        self.masked_state_token = nn.Parameter(torch.zeros(state_in_channels, patch_size))
        self.state_cnn = CNNTokenizer(in_channels=state_in_channels, cnn_dims=cnn_dims, cnn_type="state")
        self.action_cnn = CNNTokenizer(in_channels=action_in_channels, cnn_dims=cnn_dims, cnn_type="action")
        self.transformer = TransformerBlock(embed_dim=cnn_dims * 3, num_heads=num_heads, hidden_multiplier=hidden_multiplier, dropout=dropout, base=base)

    def forward(self, patch_masked, states, actions, timesteps_batch):
        # --- 1. Action CNN Tokenization ---
        action_features = self.action_cnn(actions)
        action_patches = action_features.view(
            self.batch_size, self.cnn_dims * 3, self.num_patches, self.patch_size
        ).transpose(1, 2)
        
        # --- 2. State CNN Tokenization and Masking ---
        state_patches = states.view(
            self.batch_size, self.state_in_channels, self.num_patches, self.patch_size
        ).transpose(1, 2)
        # Create a batch index to match the shape of patch_masked for advanced indexing
        batch_idx = torch.arange(self.batch_size).unsqueeze(1).expand(-1, self.num_masked)
        state_patches[batch_idx, patch_masked] = self.masked_state_token
        states_flat = state_patches.contiguous().view(
            self.batch_size * self.num_patches, self.state_in_channels, self.patch_size
        )
        state_features = self.state_cnn(states_flat)
        state_patches = state_features.view(self.batch_size, self.num_patches, self.cnn_dims * 3, self.patch_size)
        
        # --- 3. Fusion ---
        # Add the state patch features to the global action features
        # Masked steps will just be (Action + 0.0), Unmasked will be (Action + State)
        combined = action_patches + state_patches
        # Reorganize dimensions: (batch, num_patches, embed_dim, patch_size) -> (batch, num_patches, patch_size, embed_dim)
        combined = combined.permute(0, 1, 3, 2)
        combined = self.layernorm(combined)

        # --- 4. Final Formatting for Transformer ---
        # Flatten the patch structure back into a single continuous sequence dimension
        # We use sequence_length (which is num_patches * patch_size)
        transformer_input = combined.reshape(self.batch_size, self.num_patches * self.patch_size, self.cnn_dims * 3)
        
        # --- 5. Transformer Block ---
        output = self.transformer(transformer_input, timesteps_batch)

        # Mean and max pooling
        mean_pool = output.mean(dim=1)
        max_pool = output.max(dim=1)[0]
        latent = torch.cat([mean_pool, max_pool], dim=-1) # Shape: (batch, embed_dim * 2)
        return latent, output
    

class CNNDetokenizer(nn.Module):
    def __init__(self, in_channels, cnn_dims):
        """_summary_
        Will always be state because we are decoding back into the complete state
        """
        super(CNNDetokenizer, self).__init__()
        self.cnn1_low_freq = nn.Conv1d(
            in_channels=in_channels, out_channels=cnn_dims, kernel_size=7, dilation=4, padding="same"
        )
        self.cnn1_mid_freq = nn.Conv1d(
            in_channels=in_channels, out_channels=cnn_dims, kernel_size=5, dilation=2, padding="same"
        )
        self.cnn1_high_freq = nn.Conv1d(
            in_channels=in_channels, out_channels=cnn_dims, kernel_size=3, padding="same"
        )
        self.bn1_low_freq = nn.BatchNorm1d(cnn_dims)
        self.bn1_mid_freq = nn.BatchNorm1d(cnn_dims)
        self.bn1_high_freq = nn.BatchNorm1d(cnn_dims)
        
    def forward(self, x):
        # x shape: (batch, cnn_dims * 3, sequence_length)
        # Split the channels back into the three frequency streams
        cnn_dims = x.shape[1] // 3
        y_low, y_mid, y_high = torch.split(x, cnn_dims, dim=1)
        
        # Apply the inverse filters
        out_low = self.cnn1_low_freq(y_low)
        out_mid = self.cnn1_mid_freq(y_mid)
        out_high = self.cnn1_high_freq(y_high)
        
        # Combine (average) the frequency reconstructions
        return (out_low + out_mid + out_high) / 3
    
class CNNTransformerDecoder(nn.Module):
    def __init__(self, state_in_channels, cnn_dims, sequence_length, num_heads=6, hidden_multiplier=4, dropout=0.1, base=10000):
        super(CNNTransformerDecoder, self).__init__()
        embed_dim = cnn_dims * 3
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.state_cnn_detokenizer = CNNDetokenizer(in_channels=cnn_dims, cnn_dims=state_in_channels)
        self.transformer = TransformerBlock(embed_dim=cnn_dims * 3, num_heads=num_heads, hidden_multiplier=hidden_multiplier, dropout=dropout, base=base)

    def forward(self, transformer_output, timesteps_batch):
        # --- Step 1: Transformer (Order Fliped: Now before CNN) ---
        # The transformer uses RoPE to organize the expanded latent in time
        x = self.transformer(transformer_output, timesteps_batch)
        
        # --- Step 2: CNN Detokenizer ---
        # Switch to (batch, channels, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        reconstructed_states = self.state_cnn_detokenizer(x)
        
        return reconstructed_states # Shape: (batch, 4, 600)


class CNNTransformerModel(BaseEncoderDecoder):
    def __init__(self, config: dict, n_params, chkpt_file_pth, device):
        super(CNNTransformerModel, self).__init__(chkpt_file_pth, device)
        lr = float(config["learning_rate"])
        weight_decay = float(config["weight_decay"])
        mask_ratio = config["mask_ratio"]
        sequence_length = config["sequence_length"]
        self.num_patches = config["num_patches"]
        assert sequence_length % self.num_patches == 0, "Sequence length must be divisible by number of patches"
        self.patch_size = sequence_length // self.num_patches
        self.num_masked = int(self.num_patches * mask_ratio)
        self.clip_value = config["clip_value"]
        
        self.encoder = CNNTransformerEncoder(
            batch_size=config["batch_size"],
            num_patches=self.num_patches,
            num_masked=self.num_masked,
            patch_size=self.patch_size,
            cnn_dims=config["cnn_dims"],
            state_in_channels=config["state_size"],
            action_in_channels=config["action_size"],
            num_heads=config["num_heads"],
            hidden_multiplier=config["hidden_multiplier"],
            dropout=config["dropout_encoder"],
            base=config["base"]
        )
        
        self.decoder = CNNTransformerDecoder(
            state_in_channels=config["state_size"],
            cnn_dims=config["cnn_dims"],
            sequence_length=sequence_length,
            num_heads=config["num_heads"],
            hidden_multiplier=config["hidden_multiplier"],
            dropout=config["dropout_decoder"],
            base=config["base"]
        )
        # embed_dim = config["cnn_dims"] * 3
        # latent is mean pooled and max pooled so times 2
        self.mu_sigma = LinearHead(in_dim=config["cnn_dims"] * 3 * 2, out_dim=config["state_size"])
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
    def forward(self, patch_masked, states_full, actions_full, timesteps_batch):
        _, transformer_output = self.encoder(patch_masked, states_full, actions_full, timesteps_batch)
        reconstructed_states = self.decoder(transformer_output, timesteps_batch)
        return reconstructed_states
    
    def forward_target(self, patch_masked, states_full, actions_full, timesteps_batch):
        with torch.no_grad():
            latent, _ = self.encoder(patch_masked, states_full, actions_full, timesteps_batch)
        mu, sigma = self.mu_sigma(latent)
        return mu, sigma
