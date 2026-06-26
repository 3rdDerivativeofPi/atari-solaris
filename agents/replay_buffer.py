"""
agents/replay_buffer.py
-----------------------
Memory-efficient experience replay buffer for Atari DQN.

Design decisions:
  - Stores observations as uint8 (not float32) -> 4x memory saving.
    A 500K buffer of (4, 84, 84) uint8 frames uses ~14 GB if stored as
    float32, but only ~3.5 GB as uint8. Normalisation happens in the network.
  - Uses pre-allocated NumPy arrays (no Python list overhead).
  - Supports uniform random sampling (baseline DQN).
  - Structured for easy extension to Prioritized Experience Replay (Week 2).

Reference: Mnih et al. (2015), Section Methods -- Experience Replay.
"""

import numpy as np
from typing import Tuple, NamedTuple


class Batch(NamedTuple):
    """A sampled minibatch of transitions."""
    observations:      np.ndarray   # (batch, 4, 84, 84) uint8
    actions:           np.ndarray   # (batch,)           int64
    rewards:           np.ndarray   # (batch,)           float32
    next_observations: np.ndarray   # (batch, 4, 84, 84) uint8
    dones:             np.ndarray   # (batch,)           float32  (1.0 = terminal)


class ReplayBuffer:
    """
    Circular replay buffer with uniform random sampling.

    Memory layout uses pre-allocated arrays for efficiency.
    Observations are stored as uint8 to minimise memory footprint.

    Args:
        capacity:     Maximum number of transitions to store.
        obs_shape:    Shape of a single observation (e.g. (4, 84, 84)).
        obs_dtype:    Dtype for observations (default uint8).

    Example:
        >>> buf = ReplayBuffer(capacity=500_000, obs_shape=(4, 84, 84))
        >>> buf.add(obs, action, reward, next_obs, done)
        >>> batch = buf.sample(batch_size=32)
        >>> batch.observations.shape  # (32, 4, 84, 84)
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...] = (4, 84, 84),
        obs_dtype: np.dtype = np.uint8,
    ) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape
        self._ptr = 0       # Write pointer (circular)
        self._size = 0      # Current number of stored transitions

        # Pre-allocate storage arrays
        self._observations      = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self._next_observations = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self._actions           = np.zeros((capacity,), dtype=np.int64)
        self._rewards           = np.zeros((capacity,), dtype=np.float32)
        self._dones             = np.zeros((capacity,), dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
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
        Store a single transition in the buffer.

        Args:
            obs:      Current observation (4, 84, 84) uint8.
            action:   Action taken.
            reward:   Reward received (clipped during training).
            next_obs: Next observation (4, 84, 84) uint8.
            done:     Whether the episode ended (True/False).
        """
        self._observations[self._ptr]      = obs
        self._next_observations[self._ptr] = next_obs
        self._actions[self._ptr]           = action
        self._rewards[self._ptr]           = reward
        self._dones[self._ptr]             = float(done)

        # Advance circular pointer
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        """
        Sample a random minibatch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Batch namedtuple with numpy arrays.

        Raises:
            ValueError: If the buffer contains fewer transitions than batch_size.
        """
        if self._size < batch_size:
            raise ValueError(
                f"Buffer has only {self._size} transitions; "
                f"cannot sample batch of {batch_size}."
            )

        indices = np.random.randint(0, self._size, size=batch_size)

        return Batch(
            observations=      self._observations[indices],
            actions=           self._actions[indices],
            rewards=           self._rewards[indices],
            next_observations= self._next_observations[indices],
            dones=             self._dones[indices],
        )

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        """True if the buffer has at least one transition stored."""
        return self._size > 0

    def memory_usage_gb(self) -> float:
        """Estimate current memory usage of the buffer in GB."""
        total_bytes = (
            self._observations.nbytes
            + self._next_observations.nbytes
            + self._actions.nbytes
            + self._rewards.nbytes
            + self._dones.nbytes
        )
        return total_bytes / (1024 ** 3)
