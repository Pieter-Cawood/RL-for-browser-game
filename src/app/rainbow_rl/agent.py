# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

# ⬇️ import both heads; the CNN DQN stays default
from .model import DQN
try:
  # If you placed DQNMLP in the same file as DQN, this import will work.
  # Otherwise, change to: from .mlp_model import DQNMLP
  from .model import DQNMLP
except Exception:
  DQNMLP = None


class RainbowRLAgent():
  """
  Rainbow agent (distributional DQN with NoisyNets, double DQN, PER).
  Cleaned-up learn() that logs aggregate stats (no per-sample printing),
  corrects indexing math, and returns a small diagnostics dict.
  """
  def __init__(self, args, env):
    self.action_space = env.action_space()
    self.atoms = args.atoms
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.device = args.device

    # --- Choose model class based on env output type ---
    # If env exposes vector obs, we prefer the tiny MLP head.
    self._use_vector_obs = bool(getattr(env, "use_vector_obs", False))
    self._feature_dim = int(getattr(env, "feature_dim", len(getattr(env, "feature_keys", ()))) or 0)

    if self._use_vector_obs:
      if DQNMLP is None:
        raise RuntimeError("Env is vector-based but DQNMLP is not importable. Ensure it's defined and imported.")
      model_cls = DQNMLP
      model_kwargs = {"feature_dim": max(1, self._feature_dim)}
    else:
      model_cls = DQN
      model_kwargs = {}

    # Fixed C51 support z
    self.support = torch.linspace(args.V_min, args.V_max, self.atoms, device=self.device)  # [atoms]
    self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)

    # Hyperparams
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.norm_clip = args.norm_clip

    # Networks
    self.online_net = model_cls(args, self.action_space, **model_kwargs).to(device=self.device)
    if args.model:
      if os.path.isfile(args.model):
        print("Loading pretrained model:", args.model)
        state = torch.load(args.model, map_location="cpu")  # load on CPU first
        self.online_net.load_state_dict(state, strict=False)
      else:
        raise FileNotFoundError(args.model)
    self.online_net.train()

    self.target_net = model_cls(args, self.action_space, **model_kwargs).to(device=self.device)
    self.update_target_net()
    self.target_net.train()
    for p in self.target_net.parameters():
      p.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

  # Resets noisy weights in all linear layers (of online net only)
  def reset_noise(self):
    self.online_net.reset_noise()

  # Acts based on single state (no batch)
  def act(self, state):
    with torch.no_grad():
      # state can be [C,H,W] (CNN) or [H,F] (MLP)
      q_atoms = self.online_net(state.unsqueeze(0))      # [1, A, atoms] (probs)
      q = (q_atoms * self.support).sum(2)                # [1, A]
      return q.argmax(1).item()

  # Acts with an ε-greedy policy (typically for eval or debugging)
  def act_e_greedy(self, state, epsilon=0.33):
    return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

  def learn(self, mem):
    """
    One PER update step.
    Returns a small dict of aggregate stats (no per-sample printing).
    """
    # ---- Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

    # ---- Online log-probs for selected actions
    log_ps = self.online_net(states, log=True)                 # [B, A, atoms] (log-probs)
    log_ps_a = log_ps[range(self.batch_size), actions]         # [B, atoms]

    with torch.no_grad():
      # Double DQN: action selection on online, evaluation on target
      pns_online = self.online_net(next_states)                # [B, A, atoms] (probs)
      dns = self.support.expand_as(pns_online) * pns_online    # [B, A, atoms]
      argmax_indices_ns = dns.sum(2).argmax(1)                 # [B]

      # Target distribution
      self.target_net.reset_noise()
      pns_tgt = self.target_net(next_states)                   # [B, A, atoms] (probs)
      pns_a = pns_tgt[range(self.batch_size), argmax_indices_ns]  # [B, atoms]

      # Bellman update on support (before clamp for diagnostics)
      Tz_raw = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # [B, atoms]
      clamp_low_rate  = (Tz_raw < self.Vmin).float().mean().item()
      clamp_high_rate = (Tz_raw > self.Vmax).float().mean().item()

      Tz = Tz_raw.clamp(min=self.Vmin, max=self.Vmax)

      # Projection onto fixed support
      b = (Tz - self.Vmin) / self.delta_z                      # [B, atoms]
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (self.atoms - 1)) * (l == u)] += 1

      m = states.new_zeros(self.batch_size, self.atoms)        # [B, atoms]
      offset = (torch.arange(self.batch_size, device=actions.device) * self.atoms).unsqueeze(1).expand(self.batch_size, self.atoms)
      m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))
      m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))

    # ---- Cross-entropy per-sample (used for loss and PER priorities)
    ce_per_sample = -torch.sum(m * log_ps_a, dim=1)            # [B]

    # ---- Weighted mean loss for backprop
    loss_vec = weights * ce_per_sample
    loss_mean = loss_vec.mean()

    self.online_net.zero_grad(set_to_none=True)
    loss_mean.backward()
    clip_grad_norm_(self.online_net.parameters(), self.norm_clip)
    self.optimiser.step()

    # ---- Update PER priorities using the *unweighted* per-sample CE
    mem.update_priorities(idxs, ce_per_sample.detach().cpu().numpy())

    # ---- Diagnostics
    def _grad_global_norm(model):
      sq = 0.0
      for p in model.parameters():
        if p.grad is not None:
          g = p.grad.data
          sq += float(g.pow(2).sum().item())
      return sq ** 0.5

    stats = {
      "loss":        float(loss_mean.item()),
      "grad_gn":     float(_grad_global_norm(self.online_net)),
      "c51_clip_lo": float(clamp_low_rate),
      "c51_clip_hi": float(clamp_high_rate),
      "td_ce_mean":  float(ce_per_sample.mean().item()),
      "td_ce_std":   float(ce_per_sample.std().item()),
      "replay_len":  int(len(mem)),
    }
    return stats

  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  def save(self, path, name='model.pth'):
    torch.save(self.online_net.state_dict(), os.path.join(path, name))

  def evaluate_q(self, state):
    with torch.no_grad():
      q_atoms = self.online_net(state.unsqueeze(0))  # [1, A, atoms]
      q = (q_atoms * self.support).sum(2)            # [1, A]
      return q.max(1)[0].item()

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()
