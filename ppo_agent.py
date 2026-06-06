
"""
Standalone PPO agent for continuous control.
Policy: Gaussian with tanh-squash to [0, 1] action space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0.0)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims=None):
        if hidden_dims is None:
            hidden_dims = [256, 256]
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        self.mean_head = nn.Linear(in_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))
        self.value_head = nn.Linear(in_dim, 1)

        self.apply(init_weights)
        nn.init.constant_(self.log_std, -0.5)

    def forward(self, obs):
        features = self.trunk(obs)
        mean = self.mean_head(features)
        value = self.value_head(features).squeeze(-1)
        return mean, self.log_std.expand_as(mean), value

    def act(self, obs, deterministic=False):
        mean, log_std, value = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        if deterministic:
            raw_action = mean
        else:
            raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        action_scaled = (action + 1.0) / 2.0
        log_prob = dist.log_prob(raw_action)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return action_scaled, log_prob, value

    def evaluate(self, obs, actions):
        raw_actions = actions * 2.0 - 1.0
        raw_actions_clamped = raw_actions.clamp(-0.999, 0.999)
        raw_actions_unscaled = torch.atanh(raw_actions_clamped)

        mean, log_std, value = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)

        log_prob = dist.log_prob(raw_actions_unscaled)
        log_prob = log_prob - torch.log(1 - raw_actions_clamped.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy, value


class RolloutBuffer:
    def __init__(self, num_envs, num_steps, obs_dim, act_dim, device):
        self.device = device
        self.num_envs = num_envs
        self.num_steps = num_steps

        self.obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs, act_dim), device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.step = 0

    def insert(self, obs, actions, log_probs, rewards, dones, values):
        self.obs[self.step].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.log_probs[self.step].copy_(log_probs)
        self.rewards[self.step].copy_(rewards)
        self.dones[self.step].copy_(dones)
        self.values[self.step].copy_(values)
        self.step += 1

    def compute_returns(self, next_values, gamma, lam):
        advantages = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        returns = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        gae = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(self.num_steps)):
            next_val = next_values if t == self.num_steps - 1 else self.values[t + 1]
            next_done = 1.0 - self.dones[t].float()
            delta = self.rewards[t] + gamma * next_val * next_done - self.values[t]
            gae = delta + gamma * lam * next_done * gae
            advantages[t] = gae
            returns[t] = gae + self.values[t]

        return advantages, returns

    def get_training_data(self):
        return (
            self.obs.view(-1, self.obs.shape[-1]),
            self.actions.view(-1, self.actions.shape[-1]),
            self.log_probs.view(-1),
            self.advantages.view(-1),
            self.returns.view(-1),
        )


class PPO:
    def __init__(self, obs_dim, act_dim, config, device):
        self.device = device
        self.config = config

        self.model = ActorCritic(obs_dim, act_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])

        self.gamma = config["gamma"]
        self.lam = config["lam"]
        self.clip_param = config["clip_param"]
        self.value_coef = config["value_coef"]
        self.entropy_coef = config["entropy_coef"]
        self.max_grad_norm = config["max_grad_norm"]
        self.num_epochs = config["num_epochs"]
        self.batch_size = config["batch_size"]

    def update(self, buffer, next_values):
        advantages, returns = buffer.compute_returns(next_values, self.gamma, self.lam)
        buffer.advantages = advantages
        buffer.returns = returns

        obs, actions, old_log_probs, advs, rets = buffer.get_training_data()

        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        n_samples = obs.shape[0]
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for epoch in range(self.num_epochs):
            perm = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, self.batch_size):
                batch_idx = perm[start:start + self.batch_size]

                new_log_probs, entropy, values = self.model.evaluate(
                    obs[batch_idx], actions[batch_idx]
                )

                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])

                surr1 = ratio * advs[batch_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advs[batch_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                old_values = buffer.values.view(-1)[batch_idx]
                values_clipped = old_values + (values - old_values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss_unclipped = (values - rets[batch_idx]).pow(2)
                value_loss_clipped = (values_clipped - rets[batch_idx]).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "approx_kl": (old_log_probs - self.model.evaluate(obs, actions)[0].detach()).mean().abs().item(),
        }

    def save(self, path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
