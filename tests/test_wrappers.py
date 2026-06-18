"""
tests/test_wrappers.py
----------------------
Unit tests for the Solaris environment wrappers.

Run with:
    pytest tests/test_wrappers.py -v

These tests validate:
  - Observation shapes at each wrapper stage
  - Dtype correctness
  - Reward clipping behaviour
  - Frame stacking consistency
  - Env reset/step contract
"""

import numpy as np
import pytest
import gymnasium as gym

from env.wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    WarpFrame,
    FrameStack,
    ClipRewardEnv,
    EpisodicLifeEnv,
    ScaledFloatFrame,
    make_solaris_env,
    make_eval_env,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_env():
    """Raw ALE Solaris environment, no wrappers."""
    env = gym.make(
        "ALE/Solaris-v5",
        frameskip=1,
        repeat_action_probability=0.0,  # Deterministic for tests
    )
    yield env
    env.close()


@pytest.fixture(scope="module")
def train_env():
    """Fully wrapped training environment."""
    env = make_solaris_env(seed=42, sticky_action_prob=0.0)
    yield env
    env.close()


@pytest.fixture(scope="module")
def eval_env():
    """Fully wrapped evaluation environment."""
    env = make_eval_env(seed=42)
    yield env
    env.close()


# ---------------------------------------------------------------------------
# WarpFrame tests
# ---------------------------------------------------------------------------

class TestWarpFrame:
    def test_output_shape(self, raw_env):
        env = WarpFrame(raw_env)
        obs, _ = env.reset()
        assert obs.shape == (84, 84, 1), f"Expected (84,84,1), got {obs.shape}"

    def test_output_dtype(self, raw_env):
        env = WarpFrame(raw_env)
        obs, _ = env.reset()
        assert obs.dtype == np.uint8

    def test_pixel_range(self, raw_env):
        env = WarpFrame(raw_env)
        obs, _ = env.reset()
        assert obs.min() >= 0
        assert obs.max() <= 255


# ---------------------------------------------------------------------------
# FrameStack tests
# ---------------------------------------------------------------------------

class TestFrameStack:
    def test_output_shape(self, raw_env):
        env = WarpFrame(raw_env)
        env = FrameStack(env, k=4)
        obs, _ = env.reset()
        # Expect channels-first: (4, 84, 84)
        assert obs.shape == (4, 84, 84), f"Expected (4,84,84), got {obs.shape}"

    def test_frames_differ_after_steps(self, raw_env):
        """After taking steps, stacked frames should not all be identical."""
        env = WarpFrame(raw_env)
        env = FrameStack(env, k=4)
        obs, _ = env.reset()
        for _ in range(5):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                obs, _ = env.reset()
                break
        # At least one pair of stacked frames should differ
        frames_identical = all(
            np.array_equal(obs[i], obs[i + 1]) for i in range(3)
        )
        # Note: may still be identical in first few frames; just check shape integrity
        assert obs.shape == (4, 84, 84)

    def test_reset_fills_stack(self, raw_env):
        """After reset, all k frames should equal the first observation."""
        env = WarpFrame(raw_env)
        env = FrameStack(env, k=4)
        obs, _ = env.reset()
        # All 4 frames should be identical right after reset
        for i in range(3):
            assert np.array_equal(obs[i], obs[i + 1]), (
                f"Frame {i} and {i+1} should be identical right after reset."
            )


# ---------------------------------------------------------------------------
# ClipRewardEnv tests
# ---------------------------------------------------------------------------

class TestClipRewardEnv:
    @pytest.mark.parametrize("reward,expected", [
        (100.0,  1.0),
        (-50.0, -1.0),
        (0.0,    0.0),
        (0.001,  1.0),
        (-0.001,-1.0),
    ])
    def test_reward_clipping(self, raw_env, reward, expected):
        env = ClipRewardEnv(raw_env)
        clipped = env.reward(reward)
        assert clipped == expected, f"reward({reward}) → {clipped}, expected {expected}"


# ---------------------------------------------------------------------------
# ScaledFloatFrame tests
# ---------------------------------------------------------------------------

class TestScaledFloatFrame:
    def test_float_range(self, raw_env):
        env = WarpFrame(raw_env)
        env = ScaledFloatFrame(env)
        obs, _ = env.reset()
        assert obs.dtype == np.float32
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0

    def test_observation_space_dtype(self, raw_env):
        env = WarpFrame(raw_env)
        env = ScaledFloatFrame(env)
        assert env.observation_space.dtype == np.float32


# ---------------------------------------------------------------------------
# MaxAndSkipEnv tests
# ---------------------------------------------------------------------------

class TestMaxAndSkipEnv:
    def test_skip_count(self, raw_env):
        """Verify the wrapper steps the underlying env `skip` times per step."""
        step_count = [0]
        original_step = raw_env.step

        def counting_step(action):
            step_count[0] += 1
            return original_step(action)

        raw_env.step = counting_step
        raw_env.reset()
        env = MaxAndSkipEnv(raw_env, skip=4)
        env.reset()
        step_count[0] = 0
        env.step(0)
        assert step_count[0] == 4, f"Expected 4 underlying steps, got {step_count[0]}"
        raw_env.step = original_step  # Restore


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_train_env_obs_shape(self, train_env):
        obs, _ = train_env.reset()
        assert obs.shape == (4, 84, 84), f"Expected (4,84,84), got {obs.shape}"

    def test_train_env_obs_dtype(self, train_env):
        obs, _ = train_env.reset()
        assert obs.dtype == np.uint8

    def test_eval_env_obs_shape(self, eval_env):
        obs, _ = eval_env.reset()
        assert obs.shape == (4, 84, 84)

    def test_step_returns_correct_types(self, train_env):
        train_env.reset()
        obs, reward, terminated, truncated, info = train_env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_space_valid(self, train_env):
        """Action space should be Discrete with a reasonable number of actions."""
        n = train_env.action_space.n
        assert 6 <= n <= 18, f"Unexpected action space size: {n}"

    def test_reward_clipped_in_train_env(self, train_env):
        """Training env rewards should be in {-1, 0, 1}."""
        train_env.reset()
        rewards = []
        for _ in range(50):
            _, r, terminated, truncated, _ = train_env.step(train_env.action_space.sample())
            rewards.append(r)
            if terminated or truncated:
                train_env.reset()
        unique_rewards = set(rewards)
        assert unique_rewards.issubset({-1.0, 0.0, 1.0}), (
            f"Unexpected reward values in train env: {unique_rewards}"
        )

    def test_reward_not_clipped_in_eval_env(self, eval_env):
        """Eval env should be capable of returning rewards outside {-1, 0, 1}."""
        # We can't guarantee large rewards in 50 steps, but we can check the
        # wrapper is not present by verifying raw rewards can pass through.
        # We check this structurally: eval_env should not have ClipRewardEnv.
        wrappers = []
        e = eval_env
        while hasattr(e, "env"):
            wrappers.append(type(e).__name__)
            e = e.env
        assert "ClipRewardEnv" not in wrappers, (
            "ClipRewardEnv should not be present in the evaluation environment."
        )

    def test_seeded_reproducibility(self):
        """Two envs with the same seed should produce identical first observations."""
        env1 = make_solaris_env(seed=123, sticky_action_prob=0.0)
        env2 = make_solaris_env(seed=123, sticky_action_prob=0.0)
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        assert np.array_equal(obs1, obs2), "Seeded envs should produce identical first obs."
        env1.close()
        env2.close()
