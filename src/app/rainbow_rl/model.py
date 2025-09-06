# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size, device=self.weight_mu.device)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)


class DQNMLP(nn.Module):
    """
    Dueling C51 MLP for vector observations.
    Input:  x shape [B, H, F]  (H = history_length, F = feature_dim, e.g. F=2 for [x, dx])
    Output: [B, action_space, atoms] with softmax over atoms (dim=2), matching your trainer.
    """
    def __init__(self, args, action_space, feature_dim: int = 2):
        super().__init__()
        self.atoms = args.atoms
        self.action_space = action_space
        self.history_length = args.history_length
        in_dim = self.history_length * feature_dim

        # small hidden width (configurable; default ~64)
        hidden = getattr(args, "hidden_size_small", None) or max(32, min(128, getattr(args, "hidden_size", 512)//8))
        Noisy = NoisyLinear if not getattr(args, "no_noisy", False) else nn.Linear
        noisy_std = getattr(args, "noisy_std", 0.1)

        self.encoder = nn.Identity()  # we just flatten
        self.fc1 = Noisy(in_dim, hidden, std_init=noisy_std)
        self.fc_h_v = Noisy(hidden, hidden, std_init=noisy_std)
        self.fc_h_a = Noisy(hidden, hidden, std_init=noisy_std)
        self.fc_z_v = Noisy(hidden, self.atoms, std_init=noisy_std)
        self.fc_z_a = Noisy(hidden, self.action_space * self.atoms, std_init=noisy_std)

    def forward(self, x, log=False):
        # x: [B, H, F]  -> flatten
        if x.dim() == 2:  # allow [B, in_dim] too
            z = x
        else:
            z = x.view(x.size(0), -1)  # [B, H*F]

        h = F.relu(self.fc1(self.encoder(z)))
        v = self.fc_z_v(F.relu(self.fc_h_v(h)))         # [B, atoms]
        a = self.fc_z_a(F.relu(self.fc_h_a(h)))         # [B, action_space * atoms]
        v = v.view(-1, 1, self.atoms)                   # [B, 1, atoms]
        a = a.view(-1, self.action_space, self.atoms)   # [B, action_space, atoms]
        q = v + a - a.mean(1, keepdim=True)             # dueling combine
        return F.log_softmax(q, dim=2) if log else F.softmax(q, dim=2)

    def reset_noise(self):
        for m in (self.fc1, self.fc_h_v, self.fc_h_a, self.fc_z_v, self.fc_z_a):
            if hasattr(m, "reset_noise"):
                m.reset_noise()
                
class DQN(nn.Module):
    """
    Tiny dueling C51 head with a lightweight encoder.
    Output: [B, action_space, atoms] with (log_)softmax over atoms (dim=2),
    matching your current training code.
    """
    def __init__(self, args, action_space):
        super().__init__()
        self.atoms = args.atoms
        self.action_space = action_space

        # ---- Lightweight encoder (fused downsampling via strides) ----
        # Input is [B, history_length, 84, 84] with history_length=4
        ch_in = args.history_length
        self.encoder = nn.Sequential(
            nn.Conv2d(ch_in, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(inplace=True),  # 84->42
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),     nn.ReLU(inplace=True),  # 42->21
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),     nn.ReLU(inplace=True),  # 21->11
            nn.AdaptiveAvgPool2d(1),  # -> [B, 32, 1, 1]
            nn.Flatten(),             # -> [B, 32]
        )
        enc_out = 32

        # Tiny hidden size (≪ 512). Feel free to hardcode 64.
        hidden = getattr(args, "hidden_size_small", None) or max(32, min(128, getattr(args, "hidden_size", 512)//8))

        # Optional: swap to nn.Linear to remove noise & save params.
        Noisy = NoisyLinear if not getattr(args, "no_noisy", False) else nn.Linear

        # ---- Dueling distributional heads (C51-compatible) ----
        self.fc_h_v = Noisy(enc_out, hidden, std_init=getattr(args, "noisy_std", 0.1))
        self.fc_h_a = Noisy(enc_out, hidden, std_init=getattr(args, "noisy_std", 0.1))
        self.fc_z_v = Noisy(hidden, self.atoms, std_init=getattr(args, "noisy_std", 0.1))
        self.fc_z_a = Noisy(hidden, action_space * self.atoms, std_init=getattr(args, "noisy_std", 0.1))

    def forward(self, x, log=False):
        z = self.encoder(x)                         # [B, 32]
        v = self.fc_z_v(F.relu(self.fc_h_v(z)))     # [B, atoms]
        a = self.fc_z_a(F.relu(self.fc_h_a(z)))     # [B, action_space * atoms]
        v = v.view(-1, 1, self.atoms)               # [B, 1, atoms]
        a = a.view(-1, self.action_space, self.atoms)
        q = v + a - a.mean(1, keepdim=True)         # [B, action_space, atoms]
        return F.log_softmax(q, dim=2) if log else F.softmax(q, dim=2)

    def reset_noise(self):
        # Safe when using nn.Linear: just skip
        for m in (self.fc_h_v, self.fc_h_a, self.fc_z_v, self.fc_z_a):
            if hasattr(m, "reset_noise"):
                m.reset_noise()
