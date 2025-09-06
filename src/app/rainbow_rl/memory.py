# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import torch


# ---------------------------------------------------------------------
# Helpers: support BOTH image frames and vector observations in replay
# ---------------------------------------------------------------------

def make_transition_dtype(vector_obs: bool, feature_dim: int = 0):
  """
  Build a structured dtype for a single transition element and its blank value.
  - vector_obs=False: stores a single 84x84 uint8 frame (as in the original code)
  - vector_obs=True : stores a single float32 feature vector of length `feature_dim`
  Returns: (np.dtype, blank_tuple)
  """
  if vector_obs:
    if feature_dim <= 0:
      raise ValueError("feature_dim must be > 0 when vector_obs=True.")
    dtype = np.dtype([
      ('timestep',   np.int32),
      ('state',      np.float32, (feature_dim,)),  # last feature vector (no /255)
      ('action',     np.int32),
      ('reward',     np.float32),
      ('nonterminal', np.bool_),
    ])
    blank = (0, np.zeros((feature_dim,), dtype=np.float32), 0, 0.0, False)
  else:
    dtype = np.dtype([
      ('timestep',   np.int32),
      ('state',      np.uint8, (84, 84)),          # last frame (uint8, will be /255 on sample)
      ('action',     np.int32),
      ('reward',     np.float32),
      ('nonterminal', np.bool_),
    ])
    blank = (0, np.zeros((84, 84), dtype=np.uint8), 0, 0.0, False)

  return dtype, blank


# ---------------------------------------------------------------------
# Segment Tree (sum-tree) for Prioritized Replay
# ---------------------------------------------------------------------

class SegmentTree():
  """
  Sum-tree that stores transition tuples in `self.data` and per-leaf priorities
  in `self.sum_tree`. Tree layout matches the common "array heap" pattern.
  """
  def __init__(self, size, transition_dtype, blank_trans):
    self.index = 0
    self.size = size
    self.full = False  # track whether we've wrapped at least once

    # Put the leaves on the last level >= size (power-of-two aligned)
    self.tree_start = 2 ** (size - 1).bit_length() - 1
    self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)

    # Structured array holds all transition fields compactly
    self.data = np.array([blank_trans] * size, dtype=transition_dtype)

    # Initial max priority (to ensure new samples are sampled at least once)
    self.max = 1.0

  # Recompute parents from child nodes for a vector of indices
  def _update_nodes(self, indices):
    children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
    self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

  # Propagate changes upward for a vector of tree indices
  def _propagate(self, indices):
    parents = (indices - 1) // 2
    unique_parents = np.unique(parents)
    self._update_nodes(unique_parents)
    if parents[0] != 0:
      self._propagate(parents)

  # Propagate a single index (slightly faster when updating one leaf)
  def _propagate_index(self, index):
    parent = (index - 1) // 2
    left, right = 2 * parent + 1, 2 * parent + 2
    self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
    if parent != 0:
      self._propagate_index(parent)

  # Vector update of priorities at exact tree indices
  def update(self, indices, values):
    self.sum_tree[indices] = values
    self._propagate(indices)
    current_max_value = float(np.max(values))
    self.max = max(current_max_value, self.max)

  # Single-index priority update
  def _update_index(self, index, value):
    self.sum_tree[index] = value
    self._propagate_index(index)
    self.max = max(float(value), self.max)

  # Append a transition with a given priority
  def append(self, data, value):
    # store at cyclic data index
    self.data[self.index] = data
    # leaf index in the tree
    leaf = self.index + self.tree_start
    self._update_index(leaf, value)
    # advance cyclic index
    self.index = (self.index + 1) % self.size
    self.full = self.full or self.index == 0
    self.max = max(float(value), self.max)

  # Recursively descend to find leaf indices matching cumulative values
  def _retrieve(self, indices, values):
    children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1))
    # If at/after the last internal level, we've reached leaves
    if children_indices[0, 0] >= self.sum_tree.shape[0]:
      return indices
    # Clamp in case totals slightly overshoot at the last level
    elif children_indices[0, 0] >= self.tree_start:
      children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)

    left_children_values = self.sum_tree[children_indices[0]]
    successor_choices = (values > left_children_values).astype(np.int32)
    successor_indices = children_indices[successor_choices, np.arange(indices.size)]
    successor_values = values - successor_choices * left_children_values
    return self._retrieve(successor_indices, successor_values)

  # Sample leaves by their cumulative priority values; return (values, data_idx, tree_idx)
  def find(self, values):
    indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
    data_index = indices - self.tree_start
    return (self.sum_tree[indices], data_index, indices)

  # Vectorized getter: accepts scalars or arrays
  def get(self, data_index):
    return self.data[data_index % self.size]

  def total(self):
    return float(self.sum_tree[0])


# ---------------------------------------------------------------------
# Prioritized Replay Memory (PER) supporting both observation types
# ---------------------------------------------------------------------

class ReplayMemory():
  """
  Works with:
    - Image observations: states are [H,84,84] float in [0,1]; stores only the last frame (uint8).
    - Vector observations: states are [H,F] float (already scaled); stores only the last vector (float32).
  Sampling reconstructs stacked states and n-step targets exactly like the original.
  """
  def __init__(self, args, capacity, env=None):
    self.device = args.device
    self.capacity = capacity
    self.history = args.history_length
    self.discount = args.discount
    self.n = args.multi_step
    self.priority_weight = args.priority_weight
    self.priority_exponent = args.priority_exponent

    # episode timestep counter
    self.t = 0

    # n-step discount vector
    self.n_step_scaling = torch.tensor(
      [self.discount ** i for i in range(self.n)],
      dtype=torch.float32, device=self.device
    )

    # Detect vector vs image mode (prefer args, fall back to env)
    self.vector_obs = bool(getattr(args, "vector_obs", getattr(env, "use_vector_obs", False)))
    self.feature_dim = int(getattr(args, "feature_dim",
                           getattr(env, "feature_dim", len(getattr(env, "feature_keys", ()))) or 0))

    transition_dtype, blank_trans = make_transition_dtype(self.vector_obs, self.feature_dim)
    self._blank_trans = blank_trans
    self.transitions = SegmentTree(capacity, transition_dtype, blank_trans)

  # Store transition at time t:
  #   - state is stacked history (image: [H,84,84], vector: [H,F])
  #   - we store only the last element of the stack at this timestep
  def append(self, state, action, reward, terminal):
    if self.vector_obs:
      # store last vector as float32 (no scaling)
      last_vec = state[-1].to(dtype=torch.float32, device=torch.device('cpu')).numpy()
      self.transitions.append((self.t, last_vec, action, reward, not terminal), self.transitions.max)
    else:
      # original behavior: store last frame as uint8 (×255)
      last_u8 = state[-1].mul(255).to(dtype=torch.uint8, device=torch.device('cpu')).numpy()
      self.transitions.append((self.t, last_u8, action, reward, not terminal), self.transitions.max)

    self.t = 0 if terminal else self.t + 1

  # Gather a window of transitions around indices and blank appropriately at episode boundaries
  def _get_transitions(self, idxs):
    # indices from t-h+1 .. t+n
    transition_idxs = np.arange(-self.history + 1, self.n + 1) + np.expand_dims(idxs, axis=1)
    transitions = self.transitions.get(transition_idxs)

    transitions_firsts = transitions['timestep'] == 0
    blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)

    # Blank past frames if a later frame is the first of an episode
    for t in range(self.history - 2, -1, -1):  # e.g., 2 1 0
      blank_mask[:, t] = np.logical_or(blank_mask[:, t + 1], transitions_firsts[:, t + 1])

    # Blank future (n-step) frames if current/past is first of episode
    for t in range(self.history, self.history + self.n):  # e.g., 4 5 6
      blank_mask[:, t] = np.logical_or(blank_mask[:, t - 1], transitions_firsts[:, t])

    # Apply blanks
    transitions[blank_mask] = self._blank_trans
    return transitions

  # PER segment sampling (unchanged logic)
  def _get_samples_from_segments(self, batch_size, p_total):
    segment_length = p_total / batch_size
    segment_starts = np.arange(batch_size) * segment_length

    valid = False
    while not valid:
      samples = np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts
      probs, idxs, tree_idxs = self.transitions.find(samples)
      # Validity checks to avoid sampling across the circular boundary within the n/history window
      if np.all((self.transitions.index - idxs) % self.capacity > self.n) and \
         np.all((idxs - self.transitions.index) % self.capacity >= self.history) and \
         np.all(probs != 0):
        valid = True

    # Collect windows and rebuild stacked states
    transitions = self._get_transitions(idxs)
    all_states = transitions['state']  # shapes:
                                       # - vector: [B, H+n, F]
                                       # - image : [B, H+n, 84, 84]
    all_states = np.ascontiguousarray(all_states) 

    # shapes:
    # - vector: [B, H+n, F]
    # - image : [B, H+n, 84, 84]
    if self.vector_obs:
        states = torch.tensor(all_states[:, :self.history, :],
                              device=self.device, dtype=torch.float32)
        next_states = torch.tensor(all_states[:, self.n:self.n + self.history, :],
                                  device=self.device, dtype=torch.float32)
    else:
        states = torch.tensor(all_states[:, :self.history],
                              device=self.device, dtype=torch.float32).div_(255)
        next_states = torch.tensor(all_states[:, self.n:self.n + self.history],
                                  device=self.device, dtype=torch.float32).div_(255)

    # Actions at time t (the last of the history block)
    actions = torch.tensor(np.copy(transitions['action'][:, self.history - 1]),
                           dtype=torch.int64, device=self.device)

    # n-step returns (sum of n rewards with discount)
    rewards = torch.tensor(np.copy(transitions['reward'][:, self.history - 1:-1]),
                           dtype=torch.float32, device=self.device)  # [B,n]
    R = torch.matmul(rewards, self.n_step_scaling)  # [B]

    # Mask for non-terminal nth-next states
    nonterminals = torch.tensor(
      np.expand_dims(transitions['nonterminal'][:, self.history + self.n - 1], axis=1),
      dtype=torch.float32, device=self.device
    )  # [B,1]

    return probs, idxs, tree_idxs, states, actions, R, next_states, nonterminals

  def sample(self, batch_size):
    p_total = self.transitions.total()
    probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = \
      self._get_samples_from_segments(batch_size, p_total)

    probs = probs / p_total
    capacity = self.capacity if self.transitions.full else self.transitions.index
    weights = (capacity * probs) ** -self.priority_weight
    weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)
    return tree_idxs, states, actions, returns, next_states, nonterminals, weights

  def update_priorities(self, idxs, priorities):
    priorities = np.power(priorities, self.priority_exponent)
    self.transitions.update(idxs, priorities)

  def __len__(self) -> int:
    return len(self.transitions.data)

  # Optional: iterator yielding validation states (shape matches env mode)
  def __iter__(self):
    self.current_idx = 0
    return self

  def __next__(self):
    if self.current_idx == self.capacity:
      raise StopIteration
    transitions = self.transitions.data[np.arange(self.current_idx - self.history + 1,
                                                  self.current_idx + 1)]
    transitions_firsts = transitions['timestep'] == 0
    blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
    for t in reversed(range(self.history - 1)):
      blank_mask[t] = np.logical_or(blank_mask[t + 1], transitions_firsts[t + 1])
    transitions[blank_mask] = self._blank_trans

    st = transitions['state']  # vector: [H,F], image: [H,84,84]
    st = np.ascontiguousarray(st)
    if self.vector_obs:
      state = torch.tensor(st, dtype=torch.float32, device=self.device)
    else:
      state = torch.tensor(st, dtype=torch.float32, device=self.device).div_(255)

    self.current_idx += 1
    return state

  # Python 2 alias (kept for parity with original)
  next = __next__
