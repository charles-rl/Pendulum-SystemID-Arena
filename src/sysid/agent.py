import os
import random
import time
from dataclasses import dataclass

import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# TODO: Verify the logic and the sizes and everything about the historical observations

class Buffer:
    def __init__(self, num_steps, num_envs, obs_dims, priv_obs_dims, actor_history_len, critic_history_len, action_dims, device):
        self.obs = torch.zeros((num_steps, num_envs, obs_dims), device=device)
        self.priv_obs = torch.zeros((num_steps, num_envs, priv_obs_dims), device=device)
        self.history_obs = torch.zeros((num_steps, num_envs, obs_dims, actor_history_len), device=device)
        self.history_priv_obs = torch.zeros((num_steps, num_envs, priv_obs_dims, critic_history_len), device=device)
        self.actions = torch.zeros((num_steps, num_envs, action_dims), device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.advantages = torch.zeros((num_steps, num_envs), device=device)
        self.returns = torch.zeros((num_steps, num_envs), device=device)
        
        self.actor_history = torch.zeros((num_envs, obs_dims, actor_history_len), device=device)
        self.critic_history = torch.zeros((num_envs, priv_obs_dims, critic_history_len), device=device)
    
    @torch.no_grad()
    def push_history(self, obs, priv_obs):
        """
        Append the new observation as the most-recent entry (index -1),
        dropping the oldest entry (index 0).
        """
        self.actor_history = torch.cat([self.actor_history[:, :, 1:], obs.unsqueeze(-1)], dim=-1)
        self.critic_history = torch.cat([self.critic_history[:, :, 1:], priv_obs.unsqueeze(-1)], dim=-1)

    @torch.no_grad()
    def reset_history(self, env_idx, obs, priv_obs):
        """
        Zero out the history for a single env that just terminated,
        then re-seed with the new initial observation at index -1.
        """
        self.actor_history[env_idx] = 0.0
        self.critic_history[env_idx] = 0.0
        self.actor_history[env_idx, :, -1] = obs[env_idx]
        self.critic_history[env_idx, :, -1] = priv_obs[env_idx]

    @torch.no_grad()
    def insert(self, step, obs, priv_obs, history_obs, history_priv_obs, actions, logprobs, rewards, dones, values):
        self.obs[step] = obs
        self.priv_obs[step] = priv_obs
        self.history_obs[step] = history_obs
        self.history_priv_obs[step] = history_priv_obs
        self.actions[step] = actions
        self.logprobs[step] = logprobs
        self.rewards[step] = rewards
        self.dones[step] = dones
        self.values[step] = values
        

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Critic(nn.Module):
    def __init__(self, privileged_obs_dims, cnn1_output_dims, cnn2_output_dims, fc1_dims, fc2_dims, fc3_dims, privileged_history_sequence_length, num_obs_flatten, device):
        super().__init__()
        self.num_obs_flatten = num_obs_flatten  # Number of observations to flatten versus the total history_sequence_length
        # Formula based on L_out here: https://docs.pytorch.org/docs/2.12/generated/torch.nn.Conv1d.html
        self.cnn_sequence_length = privileged_history_sequence_length - num_obs_flatten
        
        # Conv1d layers with kernel=3, padding=0, stride=1 shrink the sequence length by 2 per layer
        sequence_len_after_cnn = (self.cnn_sequence_length - 2 - 2) * cnn2_output_dims
        mlp_input_dims = privileged_obs_dims + sequence_len_after_cnn + (privileged_obs_dims * num_obs_flatten)
        assert num_obs_flatten > 0, "num_obs_flatten must be greater than 0"
        
        self.cnn1 = nn.Conv1d(in_channels=privileged_obs_dims, out_channels=cnn1_output_dims, kernel_size=3, stride=1, padding=0)
        self.cnn2 = nn.Conv1d(in_channels=cnn1_output_dims, out_channels=cnn2_output_dims, kernel_size=3, stride=1, padding=0)
        self.gn1 = nn.GroupNorm(1, cnn1_output_dims)
        self.gn2 = nn.GroupNorm(1, cnn2_output_dims)
        
        self.fc1 = layer_init(nn.Linear(mlp_input_dims, fc1_dims))
        self.ln_fc1 = nn.LayerNorm(fc1_dims)
        
        self.fc2 = layer_init(nn.Linear(fc1_dims, fc2_dims))
        self.ln_fc2 = nn.LayerNorm(fc2_dims)
        
        self.fc3 = layer_init(nn.Linear(fc2_dims, fc3_dims))
        self.ln_fc3 = nn.LayerNorm(fc3_dims)
        
        self.fc4 = layer_init(nn.Linear(fc3_dims, 1), std=1.0)
        self.to(device)

    def forward(self, obs, history_obs):
        # EXPECTED SHAPES:
        # obs:         (batch, obs_dims) -> Current step
        # history_obs: (batch, obs_dims, history_sequence_length) 
        #              Ordered chronologically: index 0 is oldest, index -1 is newest.
        
        # We only take the last `history_sequence_length - num_obs_flatten` observations for the CNN and flatten the rest
        short_history = history_obs[:, :, -self.num_obs_flatten:]
        flattened_history = short_history.reshape(short_history.size(0), -1) # Batch-safe flattening
        
        # Slice from the beginning up to the remaining sequence length
        long_history = history_obs[:, :, :self.cnn_sequence_length]
        condensed_history = self.gn1(F.mish(self.cnn1(long_history)))
        condensed_history = self.gn2(F.mish(self.cnn2(condensed_history)))
        condensed_history = condensed_history.reshape(condensed_history.size(0), -1) # Batch-safe flattening
        
        # Combine everything along the feature dimension (dim=1)
        all_features = torch.cat([obs, flattened_history, condensed_history], dim=1)
        
        x = self.ln_fc1(F.mish(self.fc1(all_features)))
        x = self.ln_fc2(F.mish(self.fc2(x)))
        x = self.ln_fc3(F.mish(self.fc3(x)))
        return self.fc4(x)

class Actor(nn.Module):
    def __init__(self, obs_dims, cnn_output_dims, fc1_dims, fc2_dims, action_dims, history_sequence_length, num_obs_flatten, device):
        super().__init__()
        self.num_obs_flatten = num_obs_flatten  # Number of observations to flatten versus the total history_sequence_length
        # Formula based on L_out here: https://docs.pytorch.org/docs/2.12/generated/torch.nn.Conv1d.html
        self.cnn_sequence_length = history_sequence_length - num_obs_flatten
        
        sequence_len_after_cnn = (self.cnn_sequence_length - 2) * cnn_output_dims
        mlp_input_dims = obs_dims + sequence_len_after_cnn + (obs_dims * num_obs_flatten)
        assert num_obs_flatten > 0, "num_obs_flatten must be greater than 0"
        
        self.cnn = nn.Conv1d(in_channels=obs_dims, out_channels=cnn_output_dims, kernel_size=3, stride=1, padding=0)
        
        self.fc1 = layer_init(nn.Linear(mlp_input_dims, fc1_dims))
        self.ln_fc1 = nn.LayerNorm(fc1_dims)
        
        self.fc2 = layer_init(nn.Linear(fc1_dims, fc2_dims))
        self.ln_fc2 = nn.LayerNorm(fc2_dims)
        
        self.fc3 = layer_init(nn.Linear(fc2_dims, action_dims), std=0.01)
        self.logstd = nn.Parameter(torch.zeros(1, action_dims))
        
        self.to(device)
        
    def forward(self, obs, history_obs):
        # EXPECTED SHAPES:
        # obs:         (batch, obs_dims) -> Current step
        # history_obs: (batch, obs_dims, history_sequence_length) 
        #              Ordered chronologically: index 0 is oldest, index -1 is newest.
        
        # We only take the last `history_sequence_length - num_obs_flatten` observations for the CNN and flatten the rest
        short_history = history_obs[:, :, -self.num_obs_flatten:]
        flattened_history = short_history.reshape(short_history.size(0), -1) # Batch-safe flattening
        
        # Slice from the beginning up to the remaining sequence length
        long_history = history_obs[:, :, :self.cnn_sequence_length]
        # TODO: Do I need batch normalization for small CNN input and for actor network? Maybe add LN after flattening here
        condensed_history = F.relu(self.cnn(long_history))
        condensed_history = condensed_history.reshape(condensed_history.size(0), -1) # Batch-safe flattening
        
        # Combine everything along the feature dimension (dim=1)
        all_features = torch.cat([obs, flattened_history, condensed_history], dim=1)
        
        y = self.ln_fc1(F.relu(self.fc1(all_features)))
        y = self.ln_fc2(F.relu(self.fc2(y)))
        # We get raw mu for training stability and no tanh
        # During learning we will use raw but during inference we put in tanh again
        mu = self.fc3(y)
        logstd = self.logstd.expand_as(mu)
        sigma = torch.exp(logstd)
        # sigma = standard deviation
        # sigma^2 = var
        return mu, sigma


class AsymmetricAgent(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.batch_size = config["hyperparameters"]["batch_size"]
        self.minibatch_size = config["hyperparameters"]["minibatch_size"]
        self.update_epochs = config["hyperparameters"]["update_epochs"]
        self.clip_coef = config["hyperparameters"]["clip_coef"]
        self.norm_adv = config["hyperparameters"]["norm_adv"]
        self.ent_coef = config["hyperparameters"]["ent_coef"]
        self.vf_coef = config["hyperparameters"]["vf_coef"]
        self.max_grad_norm = config["hyperparameters"]["max_grad_norm"]
        self.target_kl = config["hyperparameters"]["target_kl"]
        self.num_steps = config["env"]["num_steps"]
        self.clip_vloss = config["hyperparameters"]["clip_vloss"]
        self.num_iterations = config["hyperparameters"]["num_iterations"]
        self.learning_rate = config["hyperparameters"]["learning_rate"]
        self.gamma = config["hyperparameters"]["gamma"]
        self.gae_lambda = config["hyperparameters"]["gae_lambda"]
        self.rpo_alpha = config["hyperparameters"]["rpo_alpha"]

        critic_config = config["critic"]
        actor_config = config["actor"]
        self.critic = Critic(
            privileged_obs_dims=critic_config["privileged_obs_dims"],
            cnn1_output_dims=critic_config["cnn1_output_dims"],
            cnn2_output_dims=critic_config["cnn2_output_dims"],
            fc1_dims=critic_config["fc1_dims"],
            fc2_dims=critic_config["fc2_dims"],
            fc3_dims=critic_config["fc3_dims"],
            privileged_history_sequence_length=critic_config["privileged_history_sequence_length"],
            num_obs_flatten=critic_config["num_obs_flatten"],
            device=device
        )
        self.actor = Actor(
            obs_dims=actor_config["obs_dims"],
            cnn_output_dims=actor_config["cnn_output_dims"],
            fc1_dims=actor_config["fc1_dims"],
            fc2_dims=actor_config["fc2_dims"],
            action_dims=actor_config["action_dims"],
            history_sequence_length=actor_config["history_sequence_length"],
            num_obs_flatten=actor_config["num_obs_flatten"],
            device=device
        )
        self.buffer = Buffer(
            num_steps=config["env"]["num_steps"],
            num_envs=config["env"]["num_envs"],
            obs_dims=actor_config["obs_dims"],
            priv_obs_dims=critic_config["privileged_obs_dims"],
            actor_history_len=actor_config["history_sequence_length"],
            critic_history_len=critic_config["privileged_history_sequence_length"],
            action_dims=actor_config["action_dims"],
            device=device
        )

        self.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5)
        
    def select_action(self, obs, history_obs, evaluate=False):
        mu, sigma = self.actor(obs, history_obs)
        probs = Normal(mu, sigma)
        if evaluate:
            # Deterministic action for evaluation
            return mu
        else:
            action = probs.sample()
            return action
        
    def get_action_and_value(self, obs, history_obs, priv_obs, history_priv_obs, action=None):
        # TODO: For goal oriented, consider separating a goal vector that passes through the actor network instantly instead of going through history.
        # TODO: Ah but I guess the goal can change in history in the future.
        mu, sigma = self.actor(obs, history_obs)
        probs = Normal(mu, sigma)
        
        if action is None:
            action = probs.sample()
        else:  # RPO Implementation
            # sample again to add stochasticity to the policy
            z = torch.FloatTensor(mu.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(self.device)
            mu = mu + z
            probs = Normal(mu, sigma)
        # NOTE: Make sure this is clipped before it goes to environment
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(priv_obs, history_priv_obs)
    
    def init_history(self, obs, priv_obs):
        """Initializes the rolling history at step 0."""
        self.buffer.actor_history.zero_()
        self.buffer.critic_history.zero_()
        self.buffer.actor_history[:, :, -1] = obs
        self.buffer.critic_history[:, :, -1] = priv_obs
        
    def update_history(self, next_obs, next_priv_obs, dones):
        """Updates the rolling history tracker with the newest observation."""
        self.buffer.push_history(next_obs, next_priv_obs)
        # Safely extract indices of environment terminations/truncations
        done_idx = dones.nonzero(as_tuple=True)[0]
        if len(done_idx) > 0:
            self.buffer.reset_history(done_idx, next_obs, next_priv_obs)
    
    def remember(self, step, obs, priv_obs, done, action, logprob, reward, value):
        """Call this to save what was used to CHOOSE the action"""
        self.buffer.insert(
            step=step,
            obs=obs,
            priv_obs=priv_obs,
            # We save a snapshot of the history currently in the buffer tracker
            history_obs=self.buffer.actor_history.clone(),
            history_priv_obs=self.buffer.critic_history.clone(),
            dones=done,
            actions=action,
            logprobs=logprob,
            rewards=reward,
            values=value,
        )
        
    def compute_gae(self, next_priv_obs, history_next_priv_obs, next_done):
        with torch.no_grad():
            next_value = self.critic(next_priv_obs, history_next_priv_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.buffer.rewards)
            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.buffer.dones[t + 1]
                    nextvalues = self.buffer.values[t + 1]
                delta = self.buffer.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.buffer.values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.buffer.values
        self.buffer.advantages = advantages
        self.buffer.returns = returns
    
    def anneal_lr(self, iteration):
        frac = 1.0 - (iteration - 1.0) / self.num_iterations
        lrnow = frac * self.learning_rate
        self.optimizer.param_groups[0]["lr"] = lrnow

    def learn(self):
        obs = self.buffer.obs.reshape((-1, self.buffer.obs.size(-1)))
        priv_obs = self.buffer.priv_obs.reshape((-1, self.buffer.priv_obs.size(-1)))
        history_obs = self.buffer.history_obs.reshape(-1, self.buffer.history_obs.size(-2), self.buffer.history_obs.size(-1))
        history_priv_obs = self.buffer.history_priv_obs.reshape(-1, self.buffer.history_priv_obs.size(-2), self.buffer.history_priv_obs.size(-1))
        logprobs = self.buffer.logprobs.reshape(-1)
        actions = self.buffer.actions.reshape((-1, self.buffer.actions.size(-1)))
        values = self.buffer.values.reshape(-1)
        advantages = self.buffer.advantages.reshape(-1)
        returns = self.buffer.returns.reshape(-1)
        
        indices = np.arange(self.batch_size)
        clipfracs = []
        for epoch in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                minibatch_indices = indices[start:end]

                _, newlogprob, entropy, newvalue = self.get_action_and_value(
                    obs[minibatch_indices], 
                    history_obs=history_obs[minibatch_indices],
                    priv_obs=priv_obs[minibatch_indices],
                    history_priv_obs=history_priv_obs[minibatch_indices],
                    action=actions[minibatch_indices]
                )
                logratio = newlogprob - logprobs[minibatch_indices]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]
                
                minibatch_advantages = advantages[minibatch_indices]
                if self.norm_adv:
                    minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -minibatch_advantages * ratio
                pg_loss2 = -minibatch_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - returns[minibatch_indices]) ** 2
                    v_clipped = values[minibatch_indices] + torch.clamp(
                        newvalue - values[minibatch_indices],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - returns[minibatch_indices]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - returns[minibatch_indices]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None and approx_kl > self.target_kl:
                break
            
        y_pred = values.cpu().numpy()
        y_true = returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        loss_metrics = {
            "value_loss": v_loss.item(),
            "policy_loss": pg_loss.item(),
            "entropy": entropy_loss.item(),
            "old_approx_kl": old_approx_kl.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": np.mean(clipfracs),
            "explained_variance": explained_var
        }
        return loss_metrics
    
    def save_model(self, filepath):
        """Saves the actor, critic, and optimizer state dictionaries."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "actor_state": self.actor.state_dict(),
            "critic_state": self.critic.state_dict(),
            "optimizer_state": self.optimizer.state_dict()
        }
        torch.save(checkpoint, filepath)
        
    def load_model(self, filepath, device):
        """Loads the actor, critic, and optimizer state dictionaries."""
        checkpoint = torch.load(filepath, map_location=device)
        self.actor.load_state_dict(checkpoint["actor_state"])
        self.critic.load_state_dict(checkpoint["critic_state"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
