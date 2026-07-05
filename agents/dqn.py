"""
agents/dqn.py
-------------
Deep Q-Network (DQN) agent for Atari Solaris.

Implements the core DQN algorithm from Mnih et al. (2015) with:
  - Experience replay buffer (uniform or prioritized)
  - Target network (hard update every C steps)
  - Epsilon-greedy exploration with linear annealing
  - Gradient clipping (norm 10) for training stability
  - Optional Double DQN action selection (Hasselt et al., 2016)
  - Optional Prioritized Experience Replay (Schaul et al., 2015)
  - Optional N-step returns (Sutton, 1988; Rainbow, Hessel et al., 2017)

References:
  - Mnih et al. (2015) "Human-level control through deep reinforcement learning"
  - Hasselt et al. (2016) "Deep Reinforcement Learning with Double Q-learning"
  - Schaul et al. (2015) "Prioritized Experience Replay"
  - Hessel et al. (2017) "Rainbow: Combining Improvements in Deep RL"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple

from agents.network import build_network
from agents.replay_buffer import ReplayBuffer, Batch
from agents.per_buffer import PrioritizedReplayBuffer, PERBatch


class DQNAgent:
    """
    DQN agent with target network and epsilon-greedy exploration.

    Args:
        n_actions:          Size of the discrete action space.
        obs_shape:          Shape of a single observation (C, H, W).
        device:             PyTorch device ('cuda' or 'cpu').
        lr:                 Learning rate for Adam optimiser.
        gamma:              Discount factor.
        buffer_capacity:    Maximum replay buffer size.
        batch_size:         Minibatch size for each gradient update.
        learning_starts:    Steps before training begins.
        target_update_freq: Steps between target network hard updates.
        eps_start:          Initial epsilon for eps-greedy exploration.
        eps_end:            Final epsilon after annealing.
        eps_decay_steps:    Number of steps to anneal epsilon over.
        grad_clip_norm:     Max gradient norm for clipping (None = disabled).
        double_dqn:         Use Double DQN action selection if True.
        dueling:            Use Dueling network architecture if True.
        use_per:            Use Prioritized Experience Replay if True.
        per_alpha:          PER priority exponent (0=uniform, 1=fully prioritized).
        per_beta_start:     Initial IS weight exponent (annealed to 1.0).
        per_beta_steps:     Steps to anneal beta from beta_start to 1.0.
        per_epsilon:        Small constant added to priorities (non-zero sampling).
        n_step:             N-step return length (1 = standard 1-step TD).
    """

    def __init__(
        self,
        n_actions:          int,
        obs_shape:          Tuple[int, ...] = (4, 84, 84),
        device:             str = "cuda",
        lr:                 float = 1e-4,
        gamma:              float = 0.99,
        buffer_capacity:    int = 500_000,
        batch_size:         int = 32,
        learning_starts:    int = 20_000,
        target_update_freq: int = 10_000,
        eps_start:          float = 1.0,
        eps_end:            float = 0.01,
        eps_decay_steps:    int = 500_000,
        grad_clip_norm:     Optional[float] = 10.0,
        double_dqn:         bool = False,
        dueling:            bool = False,
        use_per:            bool = False,
        per_alpha:          float = 0.6,
        per_beta_start:     float = 0.4,
        per_beta_steps:     int = 2_000_000,
        per_epsilon:        float = 1e-6,
        n_step:             int = 1,
    ) -> None:

        self.n_actions          = n_actions
        self.device             = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gamma              = gamma
        self.batch_size         = batch_size
        self.learning_starts    = learning_starts
        self.target_update_freq = target_update_freq
        self.eps_start          = eps_start
        self.eps_end            = eps_end
        self.eps_decay_steps    = eps_decay_steps
        self.grad_clip_norm     = grad_clip_norm
        self.double_dqn         = double_dqn
        self.use_per            = use_per
        self.n_step             = n_step

        # Step counters
        self.t               = 0   # Total environment steps
        self.updates         = 0   # Total gradient updates
        self.last_mean_q     = 0.0 # Mean Q-value from most recent update (overestimation diagnostic)

        # ----------------------------------------------------------------
        # Networks
        # ----------------------------------------------------------------
        self.online_net = build_network(
            n_actions=n_actions,
            dueling=dueling,
            in_channels=obs_shape[0],
        ).to(self.device)

        self.target_net = build_network(
            n_actions=n_actions,
            dueling=dueling,
            in_channels=obs_shape[0],
        ).to(self.device)

        # Initialise target network with same weights as online network
        self._sync_target_network()
        self.target_net.eval()  # Target network is never trained directly

        # ----------------------------------------------------------------
        # Optimiser
        # ----------------------------------------------------------------
        # Adam with eps=1.5e-4 as recommended in Rainbow (Hessel et al., 2017)
        # Mnih et al. (2015) used RMSProp; Adam is generally more stable
        self.optimiser = optim.Adam(
            self.online_net.parameters(),
            lr=lr,
            eps=1.5e-4,
        )

        # ----------------------------------------------------------------
        # Replay buffer — uniform or prioritized
        # ----------------------------------------------------------------
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_capacity,
                obs_shape=obs_shape,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_steps=per_beta_steps,
                epsilon=per_epsilon,
                n_step=n_step,
                gamma=gamma,
            )
        else:
            self.replay_buffer = ReplayBuffer(
                capacity=buffer_capacity,
                obs_shape=obs_shape,
            )

        arch_str = f"{'Dueling ' if dueling else ''}{'Double ' if double_dqn else ''}DQN"
        per_str  = f" + PER(α={per_alpha}, β={per_beta_start}→1.0)" if use_per else ""
        nstep_str = f" + {n_step}-step returns" if n_step > 1 else ""
        print(f"\n{'='*55}")
        print(f"  DQN Agent Initialised")
        print(f"{'='*55}")
        print(f"  Device          : {self.device}")
        print(f"  Architecture    : {arch_str}{per_str}{nstep_str}")
        print(f"  Parameters      : {sum(p.numel() for p in self.online_net.parameters()):,}")
        print(f"  Buffer capacity : {buffer_capacity:,}")
        print(f"  Buffer memory   : {self.replay_buffer.memory_usage_gb():.2f} GB (allocated)")
        print(f"  Learning starts : {learning_starts:,} steps")
        print(f"  eps: {eps_start} -> {eps_end} over {eps_decay_steps:,} steps")
        print(f"{'='*55}\n")

    # ------------------------------------------------------------------
    # Epsilon-greedy action selection
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        """Current exploration rate (linearly annealed)."""
        progress = min(self.t / self.eps_decay_steps, 1.0)
        return self.eps_start + progress * (self.eps_end - self.eps_start)

    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        """
        Select an action using eps-greedy policy.

        During evaluation (eval_mode=True), uses a fixed small epsilon (0.05)
        as recommended by Mnih et al. to account for stochasticity.

        Args:
            obs:       Current observation (4, 84, 84) uint8 numpy array.
            eval_mode: If True, use fixed small epsilon for evaluation.

        Returns:
            Selected action index.
        """
        eps = 0.05 if eval_mode else self.epsilon

        if np.random.random() < eps:
            return np.random.randint(self.n_actions)

        # Greedy action from online network
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.uint8, device=self.device)
            obs_t = obs_t.unsqueeze(0)          # Add batch dim: (1, 4, 84, 84)
            q_values = self.online_net(obs_t)   # (1, n_actions)
            return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def step(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> Optional[float]:
        """
        Store transition and perform a training update if ready.

        Call this once per environment step.

        Args:
            obs:      Current observation.
            action:   Action taken.
            reward:   Reward received (clipped).
            next_obs: Next observation.
            done:     Whether episode terminated.

        Returns:
            TD loss value if a training update was performed, else None.
        """
        # Store transition in replay buffer
        self.replay_buffer.add(obs, action, reward, next_obs, done)
        self.t += 1

        # Hard update target network periodically
        if self.t % self.target_update_freq == 0:
            self._sync_target_network()

        # Don't train until we have enough data
        if self.t < self.learning_starts:
            return None

        # Perform one gradient update
        if self.use_per:
            batch = self.replay_buffer.sample(self.batch_size, t=self.t)
        else:
            batch = self.replay_buffer.sample(self.batch_size)
        loss = self._compute_loss_and_update(batch)
        self.updates += 1

        return loss

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_loss_and_update(self, batch) -> float:
        """
        Compute TD loss and perform a single gradient update.

        Handles both uniform replay (standard Huber loss) and PER
        (IS-weighted Huber loss with priority update after the step).

        Standard DQN target:
            y = r + γ^n * max_a' Q_target(s', a')   [n-step, not done]
            y = r                                     [done]

        Double DQN (if self.double_dqn):
            y = r + γ^n * Q_target(s', argmax_a' Q_online(s', a'))

        PER weighting:
            loss = mean(w_i * huber(Q(s,a) - y_i))
            where w_i = (N * P(i))^{-beta} / max_j(w_j)

        Returns:
            Scalar loss value (float).
        """
        is_per = isinstance(batch, PERBatch)

        obs_t      = torch.as_tensor(batch.observations,      device=self.device)
        next_obs_t = torch.as_tensor(batch.next_observations, device=self.device)
        actions_t  = torch.as_tensor(batch.actions,           device=self.device)
        rewards_t  = torch.as_tensor(batch.rewards,           device=self.device)
        dones_t    = torch.as_tensor(batch.dones,             device=self.device)

        # IS weights (ones for uniform replay, actual weights for PER)
        if is_per:
            weights_t = torch.as_tensor(batch.weights, device=self.device)
        else:
            weights_t = torch.ones(self.batch_size, device=self.device)

        # Current Q-values for taken actions
        current_q = self.online_net(obs_t).gather(
            1, actions_t.unsqueeze(1)
        ).squeeze(1)

        # Target Q-values
        # Note: for n-step returns, gamma is already folded into the
        # stored reward (r + γr' + γ²r'' + ...), so we use γ^n here.
        # The PERBatch.rewards already contain the full n-step return.
        gamma_n = self.gamma ** self.n_step
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.online_net(next_obs_t).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_obs_t).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_obs_t).max(dim=1).values

            target_q = rewards_t + gamma_n * next_q * (1.0 - dones_t)

        # Element-wise Huber loss
        td_errors = current_q - target_q
        huber     = nn.functional.huber_loss(current_q, target_q, reduction="none")

        # IS-weighted mean loss
        loss = (weights_t * huber).mean()

        self.optimiser.zero_grad()
        loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip_norm)
        self.optimiser.step()

        # Update priorities in PER with fresh TD errors
        if is_per:
            self.replay_buffer.update_priorities(
                batch.indices,
                td_errors.detach().abs().cpu().numpy(),
            )

        self.last_mean_q = current_q.mean().item()
        return loss.item()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _sync_target_network(self) -> None:
        """Hard-copy online network weights to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save_checkpoint(self, path: str, hp: dict = None) -> None:
        """
        Save agent state to disk.

        Saves: network weights, optimiser state, step counters.
        Optionally saves the full hyperparameter dict (hp) so the
        exact training config can be reconstructed at eval/recording time.
        """
        payload = {
            "t":                  self.t,
            "updates":            self.updates,
            "online_net":         self.online_net.state_dict(),
            "target_net":         self.target_net.state_dict(),
            "optimiser":          self.optimiser.state_dict(),
        }
        if hp is not None:
            payload["hp"] = hp
        torch.save(payload, path)
        print(f"  [disk] Checkpoint saved -> {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load agent state from a checkpoint file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.t       = checkpoint["t"]
        self.updates = checkpoint["updates"]
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimiser.load_state_dict(checkpoint["optimiser"])
        print(f"  [disk] Checkpoint loaded <- {path} (step {self.t:,})")