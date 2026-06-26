"""
agents/network.py
-----------------
Neural network architectures for Atari DQN.

The CNN architecture follows Mnih et al. (2015) exactly:
  - Conv1: 32 filters, 8x8 kernel, stride 4
  - Conv2: 64 filters, 4x4 kernel, stride 2
  - Conv3: 64 filters, 3x3 kernel, stride 1
  - FC1:   512 units
  - FC2:   n_actions units (output)

Input:  (batch, 4, 84, 84) uint8 tensor -- stacked grayscale frames
Output: (batch, n_actions) float32 Q-values

Reference: Mnih et al. (2015), Extended Data Table 1.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class AtariCNN(nn.Module):
    """
    Standard Atari DQN convolutional network (Mnih et al., 2015).

    Accepts uint8 observations and normalises to [0, 1] internally,
    so the replay buffer can store memory-efficient uint8 frames.

    Args:
        n_actions:    Number of discrete actions in the environment.
        in_channels:  Number of stacked frames (default 4).
    """

    def __init__(self, n_actions: int, in_channels: int = 4) -> None:
        super().__init__()

        # Convolutional feature extractor
        # Input: (batch, 4, 84, 84)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),  # -> (batch, 32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           # -> (batch, 64,  9,  9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),           # -> (batch, 64,  7,  7)
            nn.ReLU(),
        )

        # Compute flattened conv output size dynamically (avoids hardcoding)
        conv_out_size = self._get_conv_output_size(in_channels)

        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def _get_conv_output_size(self, in_channels: int) -> int:
        """Pass a dummy tensor through conv layers to compute output size."""
        dummy = torch.zeros(1, in_channels, 84, 84)
        with torch.no_grad():
            out = self.conv(dummy)
        return int(np.prod(out.shape[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: uint8 or float32 tensor of shape (batch, 4, 84, 84).
               If uint8, normalised to [0, 1] automatically.

        Returns:
            Q-values of shape (batch, n_actions).
        """
        # Normalise uint8 -> float32 [0, 1] inside the network
        # This keeps the replay buffer in uint8 (4x memory saving)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


class DuelingAtariCNN(nn.Module):
    """
    Dueling Network architecture (Wang et al., 2016) -- for Week 2 / Rainbow.

    Splits the FC head into two streams:
      - Value stream V(s):        scalar state value
      - Advantage stream A(s, a): per-action advantage

    Q(s, a) = V(s) + (A(s, a) - mean_a(A(s, a)))

    The mean subtraction (rather than max) improves stability by keeping
    advantage estimates centred around zero.

    Reference: Wang et al. (2016) "Dueling Network Architectures for
    Deep Reinforcement Learning". ICML.

    Args:
        n_actions:   Number of discrete actions.
        in_channels: Number of stacked frames (default 4).
    """

    def __init__(self, n_actions: int, in_channels: int = 4) -> None:
        super().__init__()
        self.n_actions = n_actions

        # Shared convolutional backbone (identical to AtariCNN)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_output_size(in_channels)

        # Value stream: s -> V(s) (scalar)
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # Advantage stream: (s, a) -> A(s, a) (vector, one per action)
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def _get_conv_output_size(self, in_channels: int) -> int:
        dummy = torch.zeros(1, in_channels, 84, 84)
        with torch.no_grad():
            out = self.conv(dummy)
        return int(np.prod(out.shape[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        features = self.conv(x).flatten(start_dim=1)
        value = self.value_stream(features)               # (batch, 1)
        advantage = self.advantage_stream(features)       # (batch, n_actions)

        # Combine: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


def build_network(
    n_actions: int,
    dueling: bool = False,
    in_channels: int = 4,
) -> nn.Module:
    """
    Factory function to select between standard and dueling architectures.

    Args:
        n_actions:   Size of the action space.
        dueling:     If True, use DuelingAtariCNN; else use AtariCNN.
        in_channels: Number of stacked input frames.

    Returns:
        Initialised nn.Module.
    """
    if dueling:
        return DuelingAtariCNN(n_actions=n_actions, in_channels=in_channels)
    return AtariCNN(n_actions=n_actions, in_channels=in_channels)
