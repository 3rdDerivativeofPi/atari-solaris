"""
env/wrappers.py
---------------
Standard Atari pre-processing wrappers for the Solaris environment.

Based on the preprocessing pipeline from:
  - Mnih et al. (2015) "Human-level control through deep reinforcement learning"
  - Machado et al. (2018) "Revisiting the Arcade Learning Environment"

Key design decisions documented inline.
"""

import numpy as np
import ale_py
import gymnasium as gym
from gymnasium import spaces
import cv2
from collections import deque
from typing import Optional, Tuple, Any

# Register ALE environments so 'ALE/Solaris-v5' is discoverable
gym.register_envs(ale_py)


# ---------------------------------------------------------------------------
# 1. No-op reset: randomises the start state to reduce overfitting
# ---------------------------------------------------------------------------
class NoopResetEnv(gym.Wrapper):
    """
    Execute a random number of no-op actions at the start of each episode.

    This stochastically shifts the starting state, which reduces the risk of
    the agent memorising a fixed trajectory from a deterministic start.

    Reference: Mnih et al. (2015), Section Methods.

    Args:
        env:       The wrapped environment.
        noop_max:  Maximum number of no-ops to sample uniformly from [1, noop_max].
    """

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0  # Action 0 is NOOP in all ALE games
        assert env.unwrapped.get_action_meanings()[0] == "NOOP", (
            "Action 0 must be NOOP for this wrapper to function correctly."
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        n_noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        for _ in range(n_noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


# ---------------------------------------------------------------------------
# 2. Fire reset: some Atari games require pressing FIRE to start an episode
# ---------------------------------------------------------------------------
class FireResetEnv(gym.Wrapper):
    """
    Press FIRE at the start of each episode for environments that require it.

    Solaris does not strictly require this, but it is included for
    completeness and compatibility with the standard wrapper stack.
    Skips silently if FIRE is not in the action set.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._fire_required = (
            len(env.unwrapped.get_action_meanings()) > 1
            and env.unwrapped.get_action_meanings()[1] == "FIRE"
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        if not self._fire_required:
            return obs, info
        obs, _, terminated, truncated, _ = self.env.step(1)  # FIRE
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)  # Stabilise
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


# ---------------------------------------------------------------------------
# 3. Max-pooling over two consecutive frames (removes flickering artifacts)
# ---------------------------------------------------------------------------
class MaxAndSkipEnv(gym.Wrapper):
    """
    Return the pixel-wise maximum over the last two frames and repeat each
    action for `skip` steps (frame skipping).

    Frame skipping (default 4) is critical for training speed: the agent only
    sees and decides every 4 ALE frames, which reduces temporal redundancy and
    dramatically speeds up training without hurting performance.

    Max-pooling addresses the ALE sprite flickering problem: some objects are
    only rendered on alternating frames due to hardware limitations of the
    original Atari 2600.

    Reference: Mnih et al. (2015), Extended Data Table 1.

    Args:
        env:   The wrapped environment.
        skip:  Number of frames to repeat each action.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self._skip = skip
        # Pre-allocate buffer for the last two raw frames
        obs_shape = env.observation_space.shape
        self._obs_buffer = np.zeros((2, *obs_shape), dtype=np.uint8)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        # Pixel-wise max over last two frames to eliminate flickering
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# 4. Grayscale + resize to 84x84
# ---------------------------------------------------------------------------
class WarpFrame(gym.ObservationWrapper):
    """
    Convert RGB frames to grayscale and resize to (84, 84).

    This reduces the observation dimensionality by ~3x (color -> grayscale)
    and standardises the spatial resolution used by the CNN.

    Uses INTER_AREA interpolation which is preferred for downscaling
    (preserves average intensity, avoids aliasing).

    Reference: Mnih et al. (2015); standard in virtually all Atari RL work.

    Args:
        env:    The wrapped environment.
        width:  Target frame width (default 84).
        height: Target frame height (default 84).
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84) -> None:
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(height, width, 1),
            dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, np.newaxis]  # Add channel dim: (84, 84, 1)


# ---------------------------------------------------------------------------
# 5. Frame stacking: gives the agent a sense of motion/velocity
# ---------------------------------------------------------------------------
class FrameStack(gym.Wrapper):
    """
    Stack the last `k` frames along the channel dimension.

    A single frame is a static image with no velocity information. Stacking
    4 consecutive frames lets the CNN infer object direction and speed, which
    is critical for games like Solaris where ship and enemy motion matters.

    Returns observations of shape (k, H, W) -- channels-first for PyTorch.

    Reference: Mnih et al. (2015), standard for all Atari CNN agents.

    Args:
        env: The wrapped environment.
        k:   Number of frames to stack (default 4).
    """

    def __init__(self, env: gym.Env, k: int = 4) -> None:
        super().__init__(env)
        self.k = k
        self._frames: deque = deque(maxlen=k)
        old_space = env.observation_space
        # Output: (k, H, W), uint8, channels-first for PyTorch
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(k, old_space.shape[0], old_space.shape[1]),
            dtype=np.uint8,
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        # Fill the stack by repeating the first frame k times
        for _ in range(self.k):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        assert len(self._frames) == self.k
        # Stack along axis 0: (k, H, W)
        return np.concatenate(list(self._frames), axis=2).transpose(2, 0, 1)


# ---------------------------------------------------------------------------
# 6. Reward clipping: stabilises training across games with different scales
# ---------------------------------------------------------------------------
class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip rewards to {-1, 0, +1} by taking the sign.

    This ensures gradient magnitudes from the TD error are consistent
    regardless of the true reward scale, which is important for applying
    the same hyperparameters across different Atari games.

    IMPORTANT NOTE for Solaris: Reward clipping is used for the baseline
    DQN agent for comparability with published results. However, it
    discards information about score magnitude. In Week 2, we may want to
    experiment with soft clipping (e.g., tanh or log-scaling) to retain
    more signal.

    Reference: Mnih et al. (2015).
    """

    def reward(self, reward: float) -> float:
        return float(np.sign(reward))


# ---------------------------------------------------------------------------
# 7. Transpose wrapper: ensures (C, H, W) tensor format for PyTorch
# ---------------------------------------------------------------------------
class TransposeObs(gym.ObservationWrapper):
    """
    Transpose observations from (H, W, C) to (C, H, W) for PyTorch.

    Only applied if FrameStack is NOT used (FrameStack handles this itself).
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        old_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[2], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs.transpose(2, 0, 1)


# ---------------------------------------------------------------------------
# 8. Episodic life: treat each life loss as a terminal signal during training
# ---------------------------------------------------------------------------
class EpisodicLifeEnv(gym.Wrapper):
    """
    Treat losing a life as a terminal state during training (but not eval).

    This provides a much denser terminal signal: without it, the agent only
    gets a terminal at the very end of the game (after all lives are lost),
    making credit assignment over long episodes extremely difficult.

    During evaluation this wrapper should NOT be used, as we want to measure
    full-game performance.

    Reference: Mnih et al. (2015); OpenAI Baselines.

    Args:
        env: The wrapped environment.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        # Signal terminal if a life was lost (but don't actually reset)
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # Step with no-op to advance past life-loss screen
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


# ---------------------------------------------------------------------------
# 9. Scaled float observations: convert uint8 [0,255] -> float32 [0,1]
# ---------------------------------------------------------------------------
class ScaledFloatFrame(gym.ObservationWrapper):
    """
    Normalise pixel values from [0, 255] uint8 to [0.0, 1.0] float32.

    This is applied lazily (at the neural network input) rather than stored
    in the replay buffer, as storing float32 frames uses 4x more memory
    than uint8. See LazyFrames / the replay buffer implementation.

    NOTE: This wrapper stores float32 in the observation space -- only attach
    it directly before the network, not before the replay buffer.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return np.array(obs, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Factory function: compose the full wrapper stack
# ---------------------------------------------------------------------------
def make_solaris_env(
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
    clip_rewards: bool = True,
    episodic_life: bool = True,
    frame_stack: int = 4,
    frame_skip: int = 4,
    noop_max: int = 30,
    # Machado et al. (2018) recommend sticky actions (repeat_action_probability=0.25)
    # for more robust evaluation. Set to 0.0 for the deterministic variant.
    sticky_action_prob: float = 0.25,
) -> gym.Env:
    """
    Construct the fully-wrapped Solaris environment.

    Applies the standard Atari preprocessing pipeline used in DQN and Rainbow:
      1. NoopResetEnv        -- randomised start states
      2. MaxAndSkipEnv       -- frame skipping + flicker elimination
      3. EpisodicLifeEnv     -- dense terminal signal (training only)
      4. FireResetEnv        -- press FIRE if needed
      5. WarpFrame           -- grayscale + resize to 84x84
      6. FrameStack          -- stack 4 frames, (4, 84, 84), channels-first

    Reward clipping and observation scaling are handled separately
    (clipping here; scaling inside the network or training loop).

    Args:
        render_mode:        "human" to display, "rgb_array" for recording, None for training.
        seed:               RNG seed for reproducibility.
        clip_rewards:       Clip rewards to {-1, 0, +1}. Disable for evaluation.
        episodic_life:      Treat life loss as terminal. Disable for evaluation.
        frame_stack:        Number of frames to stack (default 4).
        frame_skip:         Number of frames to repeat each action (default 4).
        noop_max:           Max no-ops at episode start (default 30).
        sticky_action_prob: Probability of repeating last action (0.25 recommended).

    Returns:
        A fully wrapped gym.Env ready for training or evaluation.

    Example:
        >>> env = make_solaris_env(seed=42)
        >>> obs, info = env.reset()
        >>> obs.shape  # (4, 84, 84)
    """
    env = gym.make(
        "ALE/Solaris-v5",
        render_mode=render_mode,
        repeat_action_probability=sticky_action_prob,
        frameskip=1,       # We handle frame skipping ourselves in MaxAndSkipEnv
        full_action_space=False,  # Use the minimal action set (18 actions max)
    )

    if seed is not None:
        env.reset(seed=seed)

    env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=frame_skip)

    if episodic_life:
        env = EpisodicLifeEnv(env)

    env = FireResetEnv(env)
    env = WarpFrame(env)

    if clip_rewards:
        env = ClipRewardEnv(env)

    if frame_stack > 1:
        env = FrameStack(env, k=frame_stack)

    return env


def make_eval_env(
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
) -> gym.Env:
    """
    Evaluation environment: no reward clipping, no episodic life.

    Evaluation should reflect true game performance, not the modified
    reward signal used during training. Sticky actions are kept at the
    standard 0.25 as recommended by Machado et al. (2018).
    """
    return make_solaris_env(
        render_mode=render_mode,
        seed=seed,
        clip_rewards=False,
        episodic_life=False,
    )
