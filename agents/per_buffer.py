"""
agents/per_buffer.py
--------------------
Prioritized Experience Replay (PER) buffer with n-step return support.

Combines two Rainbow components:
  1. Prioritized Experience Replay (Schaul et al., 2015)
     - Samples transitions proportional to their TD error magnitude
     - High-error transitions are replayed more frequently
     - Importance sampling (IS) weights correct the induced bias
  2. N-step returns (Sutton, 1988; used in Rainbow, Hessel et al., 2017)
     - Accumulates n consecutive rewards before storing a transition
     - Target becomes: r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n Q(s_{t+n})
     - Directly addresses Solaris's long-horizon credit assignment problem

Implementation notes:
  - Sum-tree for O(log N) priority updates and O(log N) stratified sampling
  - Observations stored as uint8 (4x memory saving vs float32)
  - N-step buffer is a fixed-size deque; transitions are only committed once
    n steps have been collected (or episode ends)

References:
  - Schaul et al. (2015) "Prioritized Experience Replay". ICLR 2016.
  - Hessel et al. (2017) "Rainbow: Combining Improvements in Deep RL". AAAI 2018.
  - Sutton & Barto (2018) "Reinforcement Learning: An Introduction", Ch. 7.
"""

import numpy as np
from collections import deque
from typing import Tuple, NamedTuple, Optional


# ---------------------------------------------------------------------------
# Batch type
# ---------------------------------------------------------------------------

class PERBatch(NamedTuple):
    """A prioritized minibatch of transitions."""
    observations:      np.ndarray   # (batch, 4, 84, 84) uint8
    actions:           np.ndarray   # (batch,)           int64
    rewards:           np.ndarray   # (batch,)           float32  — n-step return
    next_observations: np.ndarray   # (batch, 4, 84, 84) uint8   — s_{t+n}
    dones:             np.ndarray   # (batch,)           float32
    weights:           np.ndarray   # (batch,)           float32  — IS weights
    indices:           np.ndarray   # (batch,)           int64    — for priority update


# ---------------------------------------------------------------------------
# Sum-tree
# ---------------------------------------------------------------------------

class SumTree:
    """
    Binary sum-tree for O(log N) priority sampling and updates.

    Leaf nodes store individual transition priorities p_i^alpha.
    Internal nodes store the sum of their children.
    The root stores the total priority sum, used for proportional sampling.

    Tree layout (capacity=4):
        Index:   0
                / \\
               1   2
              / \\ / \\
             3  4 5  6   ← leaves (indices 3..6 = data indices 0..3)

    Args:
        capacity: Maximum number of transitions (must be a power of 2
                  for clean indexing, but we handle arbitrary sizes).
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        # Full binary tree has 2*capacity - 1 nodes
        self._tree   = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._write  = 0   # Next write position in leaf space

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _propagate(self, idx: int, delta: float) -> None:
        """Propagate a priority change up to the root."""
        parent = (idx - 1) // 2
        self._tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, value: float) -> int:
        """Walk down the tree to find the leaf whose prefix sum >= value."""
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self._tree):
            return idx   # leaf
        if value <= self._tree[left]:
            return self._retrieve(left, value)
        return self._retrieve(right, value - self._tree[left])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def total(self) -> float:
        """Total priority sum (root node)."""
        return float(self._tree[0])

    def add(self, priority: float) -> int:
        """
        Add a new priority at the current write position.

        Returns:
            The leaf index (used to update priority later).
        """
        leaf_idx = self._write + self.capacity - 1
        self.update(leaf_idx, priority)
        self._write = (self._write + 1) % self.capacity
        return leaf_idx

    def update(self, leaf_idx: int, priority: float) -> None:
        """Update the priority of an existing leaf."""
        delta = priority - self._tree[leaf_idx]
        self._tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def sample(self, value: float) -> Tuple[int, float]:
        """
        Sample a leaf proportional to priority.

        Args:
            value: A uniform random value in [0, total].

        Returns:
            (leaf_idx, priority) — leaf_idx is the tree index,
            data_idx = leaf_idx - (capacity - 1) is the buffer index.
        """
        leaf_idx = self._retrieve(0, value)
        return leaf_idx, self._tree[leaf_idx]

    def get_priority(self, leaf_idx: int) -> float:
        return float(self._tree[leaf_idx])


# ---------------------------------------------------------------------------
# N-step buffer
# ---------------------------------------------------------------------------

class NStepBuffer:
    """
    Accumulates n consecutive transitions and computes the n-step return.

    Holds a sliding window of (obs, action, reward, next_obs, done) tuples.
    When the buffer has n entries, it computes:

        R_n = r_0 + γ*r_1 + γ²*r_2 + ... + γ^{n-1}*r_{n-1}

    and returns a transition (obs_0, action_0, R_n, obs_n, done_n).

    On episode termination (done=True), the buffer is flushed — each
    remaining transition is committed with its truncated n-step return.

    Args:
        n:     Number of steps to accumulate.
        gamma: Discount factor.
    """

    def __init__(self, n: int, gamma: float) -> None:
        self.n     = n
        self.gamma = gamma
        self._buf: deque = deque(maxlen=n)

    def add(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> list:
        """
        Add a transition and return any completed n-step transitions.

        Returns a list of (obs, action, n_step_reward, next_obs_n, done_n)
        tuples ready to store in the replay buffer. Usually returns 0 or 1
        items; returns up to n items when an episode terminates.
        """
        self._buf.append((obs, action, reward, next_obs, done))
        ready = []

        if done:
            # Flush entire buffer with truncated n-step returns
            while self._buf:
                ready.append(self._make_transition(0))
                self._buf.popleft()
        elif len(self._buf) == self.n:
            ready.append(self._make_transition(0))

        return ready

    def _make_transition(self, start: int) -> tuple:
        """Compute the n-step transition starting at position `start` in buf."""
        buf = list(self._buf)[start:]
        obs_0, action_0, _, _, _ = buf[0]

        # Compute discounted n-step return
        R = 0.0
        for i, (_, _, r, _, d) in enumerate(buf):
            R += (self.gamma ** i) * r
            if d:
                # Episode ended before n steps — final state is terminal
                _, _, _, next_obs_n, done_n = buf[i]
                return obs_0, action_0, R, next_obs_n, True

        # Full n-step: bootstrap from s_{t+n}
        _, _, _, next_obs_n, done_n = buf[-1]
        return obs_0, action_0, R, next_obs_n, done_n

    def reset(self) -> None:
        """Clear the buffer at the start of a new episode."""
        self._buf.clear()


# ---------------------------------------------------------------------------
# PER Buffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer with n-step returns.

    Stores transitions with priorities, samples proportionally to priority^alpha,
    and returns importance-sampling weights to correct the induced bias.

    Args:
        capacity:   Maximum number of transitions.
        obs_shape:  Shape of a single observation (e.g. (4, 84, 84)).
        alpha:      Priority exponent. 0 = uniform, 1 = fully prioritized.
                    Recommended: 0.6 (Schaul et al., 2015).
        beta_start: Initial IS weight exponent. Annealed to 1.0 over training.
                    Recommended: 0.4.
        beta_steps: Steps over which beta is annealed from beta_start to 1.0.
        epsilon:    Small constant added to all priorities to ensure non-zero
                    sampling probability. Recommended: 1e-6.
        n_step:     Number of steps for n-step returns. 1 = standard 1-step.
                    Recommended: 3 for Solaris (credit assignment benefit).
        gamma:      Discount factor (used for n-step return computation).

    Example:
        >>> buf = PrioritizedReplayBuffer(capacity=200_000, obs_shape=(4,84,84))
        >>> transitions = buf.add(obs, action, reward, next_obs, done)
        >>> batch = buf.sample(batch_size=32, beta=0.5)
        >>> buf.update_priorities(batch.indices, new_td_errors)
    """

    def __init__(
        self,
        capacity:   int,
        obs_shape:  Tuple[int, ...] = (4, 84, 84),
        alpha:      float = 0.6,
        beta_start: float = 0.4,
        beta_steps: int   = 2_000_000,
        epsilon:    float = 1e-6,
        n_step:     int   = 3,
        gamma:      float = 0.99,
    ) -> None:
        self.capacity   = capacity
        self.alpha      = alpha
        self.beta_start = beta_start
        self.beta_steps = beta_steps
        self.epsilon    = epsilon
        self.n_step     = n_step

        self._tree = SumTree(capacity)
        self._size = 0

        # Pre-allocated storage (uint8 for memory efficiency)
        self._observations      = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self._next_observations = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self._actions           = np.zeros((capacity,),            dtype=np.int64)
        self._rewards           = np.zeros((capacity,),            dtype=np.float32)
        self._dones             = np.zeros((capacity,),            dtype=np.float32)

        # Max priority seen so far — new transitions get this priority
        # so they are guaranteed to be sampled at least once before
        # their TD error is known.
        self._max_priority = 1.0

        # N-step accumulator
        self._n_step_buf = NStepBuffer(n=n_step, gamma=gamma)

        # Write pointer tracks the current storage slot
        self._write = 0

    # ------------------------------------------------------------------
    # Beta annealing
    # ------------------------------------------------------------------

    def beta(self, t: int) -> float:
        """
        Linearly anneal beta from beta_start to 1.0 over beta_steps.

        At t=0 beta=beta_start (less correction, more bias).
        At t=beta_steps beta=1.0 (full IS correction, unbiased updates).
        """
        progress = min(t / self.beta_steps, 1.0)
        return self.beta_start + progress * (1.0 - self.beta_start)

    # ------------------------------------------------------------------
    # Add transitions
    # ------------------------------------------------------------------

    def add(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        """
        Add a raw transition. The n-step buffer accumulates internally;
        complete n-step transitions are committed to storage automatically.

        Args:
            obs:      Current observation.
            action:   Action taken.
            reward:   Clipped reward received.
            next_obs: Next observation.
            done:     Whether the episode ended.
        """
        ready = self._n_step_buf.add(obs, action, reward, next_obs, done)
        for transition in ready:
            self._store(*transition)

    def _store(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        """Store a completed (n-step) transition in the circular buffer."""
        idx = self._write

        self._observations[idx]      = obs
        self._next_observations[idx] = next_obs
        self._actions[idx]           = action
        self._rewards[idx]           = reward
        self._dones[idx]             = float(done)

        # New transitions get max priority so they're sampled at least once
        priority = self._max_priority ** self.alpha
        self._tree.add(priority)

        self._write = (self._write + 1) % self.capacity
        self._size  = min(self._size + 1, self.capacity)

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------

    def sample(self, batch_size: int, t: int) -> PERBatch:
        """
        Stratified sampling: divide the priority range into batch_size
        equal segments and sample one transition from each segment.
        This reduces variance compared to pure random sampling.

        Args:
            batch_size: Number of transitions to sample.
            t:          Current training step (for beta annealing).

        Returns:
            PERBatch with IS weights and tree indices for priority updates.
        """
        assert self._size >= batch_size, (
            f"Buffer has only {self._size} transitions; "
            f"cannot sample batch of {batch_size}."
        )

        beta = self.beta(t)
        segment = self._tree.total / batch_size

        leaf_indices = np.zeros(batch_size, dtype=np.int64)
        priorities   = np.zeros(batch_size, dtype=np.float64)

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = np.random.uniform(lo, hi)
            leaf_idx, priority = self._tree.sample(value)
            leaf_indices[i] = leaf_idx
            priorities[i]   = priority

        # Data indices: leaf_idx - (capacity - 1)
        data_indices = leaf_indices - (self.capacity - 1)
        data_indices = np.clip(data_indices, 0, self._size - 1)

        # Importance-sampling weights
        # w_i = (N * P(i))^{-beta} / max_j(w_j)
        # where P(i) = p_i / sum(p_j)
        sampling_probs = priorities / self._tree.total
        weights = (self._size * sampling_probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        return PERBatch(
            observations=      self._observations[data_indices],
            actions=           self._actions[data_indices],
            rewards=           self._rewards[data_indices],
            next_observations= self._next_observations[data_indices],
            dones=             self._dones[data_indices],
            weights=           weights,
            indices=           leaf_indices,
        )

    # ------------------------------------------------------------------
    # Priority update
    # ------------------------------------------------------------------

    def update_priorities(
        self,
        leaf_indices: np.ndarray,
        td_errors:    np.ndarray,
    ) -> None:
        """
        Update priorities after computing new TD errors.

        Called after each gradient update with the fresh TD errors
        for the sampled batch.

        Args:
            leaf_indices: Tree leaf indices from the batch (batch.indices).
            td_errors:    Absolute TD errors |δ_i| for each transition.
        """
        for idx, error in zip(leaf_indices, td_errors):
            priority = (abs(float(error)) + self.epsilon) ** self.alpha
            self._tree.update(int(idx), priority)
            self._max_priority = max(self._max_priority, priority)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    def memory_usage_gb(self) -> float:
        total_bytes = (
            self._observations.nbytes
            + self._next_observations.nbytes
            + self._actions.nbytes
            + self._rewards.nbytes
            + self._dones.nbytes
            + self._tree._tree.nbytes
        )
        return total_bytes / (1024 ** 3)
