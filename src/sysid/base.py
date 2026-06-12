import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, chkpt_file_pth, device):
        super(BaseModel, self).__init__()

        self.chkpt_file_pth = chkpt_file_pth
        self.loss = nn.GaussianNLLLoss(full=False)

        self.device = device

    def forward(self, data):
        return NotImplementedError
    
    def learn(self, data, target):
        data = data.to(self.device)
        target = target.to(self.device)
        mu, sigma = self.forward(data)
        # if sigma then square here else if var then do not square
        loss = self.loss(mu, target, sigma.pow(2))
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.clip_value)
        self.optimizer.step()
        return loss.item()

    def save_model(self):
        print("...saving checkpoint...")
        torch.save({"model": self.state_dict()}, self.chkpt_file_pth)

    def load_model(self):
        print("...loading checkpoint...")
        checkpoint = torch.load(self.chkpt_file_pth, map_location=self.device)
        self.load_state_dict(checkpoint["model"])


class BaseEncoderDecoder(nn.Module):
    def __init__(self, chkpt_file_pth, device):
        super(BaseEncoderDecoder, self).__init__()

        self.chkpt_file_pth = chkpt_file_pth
        self.target_loss = nn.GaussianNLLLoss(full=False)
        # self.encdec_loss = nn.L1Loss()
        # OR MSE
        self.encdec_loss = nn.MSELoss()
        
        self.mode = "encdec" # or "target"
        
        self.encoder = None
        self.decoder = None        
        self.mu_sigma = None

        self.device = device
        
    def inputs_to_device(self, mask, states, actions):
        mask = mask.to(self.device)
        states = states.to(self.device)
        actions = actions.to(self.device)
        return mask, states, actions
    
    def learn_target(self, mask, states, actions, target):
        mask, states, actions = self.inputs_to_device(mask, states, actions)
        
        mu, sigma = self.forward(mask, states, actions)
       
        # if sigma then square here else if var then do not square
        loss = self.target_loss(mu, target, sigma.pow(2))
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.clip_value)
        self.optimizer.step()
        return loss.item()
    
    def learn_encdec(self, mask, states, actions):
        mask, states, actions = self.inputs_to_device(mask, states, actions)
        reconstructed_states = self.forward_encdec(mask, states, actions)
        
        # Ensure mask is 3D: (batch, 1, seq_len)
        if mask.dim() == 2:
            mask = mask.unsqueeze(1)
            
        # Broadcast mask to match states: (batch, state_dims, seq_len)
        mask_expand = mask.expand_as(states)

        # Boolean indexing flattens everything into 1D arrays of masked elements
        reconstructed_states_masked_only = reconstructed_states[mask_expand]
        original_states_masked_only = states[mask_expand]

        # Calculate the loss on just the unknown data
        loss = self.encdec_loss(reconstructed_states_masked_only, original_states_masked_only)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.clip_value)
        self.optimizer.step()
        return loss.item()

    def save_model(self):
        print("...saving checkpoint...")
        torch.save({"encoder": self.encoder.state_dict(), "decoder": self.decoder.state_dict(), "mu_sigma": self.mu_sigma.state_dict()}, self.chkpt_file_pth)

    def load_model(self):
        print("...loading checkpoint...")
        checkpoint = torch.load(self.chkpt_file_pth, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.mu_sigma.load_state_dict(checkpoint["mu_sigma"])
        if self.mode == "target":
            # Freeze both encoder and decoder
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
            self.decoder.eval()
        elif self.mode == "encdec":
            # Freeze only the mu_sigma head
            for param in self.mu_sigma.parameters():
                param.requires_grad = False
            self.mu_sigma.eval()
            
    def forward_encdec(self):
        return NotImplementedError
        
    def forward(self):
        return NotImplementedError


class LinearHead(nn.Module):
    def __init__(self, lstm_dims, out_dim):
        super(LinearHead, self).__init__()
        self.fc = nn.Linear(lstm_dims * 4, lstm_dims * 2)
        self.ln_fc = nn.LayerNorm(lstm_dims * 2)
        self.mu_head = nn.Linear(lstm_dims * 2, out_dim)
        self.sigma_head = nn.Linear(lstm_dims * 2, out_dim)

    def forward(self, x):
        x = self.ln_fc(F.mish(self.fc(x)))
        mu = torch.tanh(self.mu_head(x)) * 1.2  # Scale the tanh to allow for OOD predictions
        sigma = torch.exp(self.sigma_head(x)) + 1e-6  # epsilon added to help with the stability when the sigma is near 0
        # sigma = standard deviation
        # sigma^2 = variance = var
        # upon testing exponential and sigma works better
        return mu, sigma
