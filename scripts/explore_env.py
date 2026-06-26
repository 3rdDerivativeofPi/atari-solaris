import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

"""
scripts/explore_env.py
----------------------
Day 1-2: Environment exploration with a random agent.

Profiles the Solaris environment to understand:
  - Reward distribution (sparsity, scale, frequency)
  - Episode length distribution
  - Action space
  - Observation statistics

Run:
    python scripts/explore_env.py --episodes 50 --seed 42

Outputs a summary to stdout and saves plots to logs/exploration/.
"""

import argparse
import os
import time
from collections import defaultdict
from typing import List, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; safe for headless runs
import matplotlib.pyplot as plt
import gymnasium as gym

from env.wrappers import make_solaris_env, make_eval_env


# ---------------------------------------------------------------------------
# Random agent profiler
# ---------------------------------------------------------------------------

def run_random_agent(
    n_episodes: int = 50,
    seed: int = 42,
    max_steps_per_episode: int = 10_000,
) -> Dict:
    """
    Run a uniformly-random agent for `n_episodes` and collect statistics.

    Args:
        n_episodes:            Number of episodes to run.
        seed:                  RNG seed.
        max_steps_per_episode: Safety cap on episode length.

    Returns:
        Dictionary of collected metrics.
    """
    # Use eval env (no reward clipping) to see true reward scale
    env = make_eval_env(seed=seed)
    rng = np.random.default_rng(seed)

    stats = defaultdict(list)

    print(f"\n{'='*60}")
    print(f"  Solaris Environment Exploration -- {n_episodes} episodes")
    print(f"{'='*60}")
    print(f"  Observation space : {env.observation_space}")
    print(f"  Action space      : {env.action_space}")
    print(f"  Action meanings   : {env.unwrapped.get_action_meanings()}")
    print(f"{'='*60}\n")

    t_start = time.time()

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_steps = 0
        ep_nonzero_rewards = 0
        ep_reward_timestamps = []   # Step at which each nonzero reward occurred

        terminated = truncated = False

        while not (terminated or truncated) and ep_steps < max_steps_per_episode:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1

            if reward != 0.0:
                ep_nonzero_rewards += 1
                ep_reward_timestamps.append(ep_steps)

        stats["episode_rewards"].append(ep_reward)
        stats["episode_lengths"].append(ep_steps)
        stats["nonzero_reward_count"].append(ep_nonzero_rewards)
        stats["reward_timestamps"].extend(ep_reward_timestamps)
        stats["terminated"].append(int(terminated))

        if (ep + 1) % 10 == 0:
            elapsed = time.time() - t_start
            mean_r = np.mean(stats["episode_rewards"])
            mean_l = np.mean(stats["episode_lengths"])
            print(
                f"  Episode {ep+1:>3}/{n_episodes} | "
                f"Mean reward: {mean_r:>8.1f} | "
                f"Mean length: {mean_l:>7.1f} steps | "
                f"Elapsed: {elapsed:.1f}s"
            )

    env.close()
    return dict(stats)


# ---------------------------------------------------------------------------
# Statistics summary
# ---------------------------------------------------------------------------

def print_summary(stats: Dict) -> None:
    rewards = np.array(stats["episode_rewards"])
    lengths = np.array(stats["episode_lengths"])
    nonzero = np.array(stats["nonzero_reward_count"])

    print(f"\n{'='*60}")
    print("  SUMMARY STATISTICS")
    print(f"{'='*60}")

    print("\n  Episode Rewards (true, unclipped):")
    print(f"    Mean   : {rewards.mean():.2f}")
    print(f"    Std    : {rewards.std():.2f}")
    print(f"    Min    : {rewards.min():.2f}")
    print(f"    Max    : {rewards.max():.2f}")
    print(f"    Median : {np.median(rewards):.2f}")

    print("\n  Episode Lengths (frames after skip, i.e. agent steps):")
    print(f"    Mean   : {lengths.mean():.1f}")
    print(f"    Std    : {lengths.std():.1f}")
    print(f"    Min    : {lengths.min()}")
    print(f"    Max    : {lengths.max()}")

    print("\n  Reward Sparsity (nonzero reward events per episode):")
    print(f"    Mean   : {nonzero.mean():.2f}")
    print(f"    Std    : {nonzero.std():.2f}")
    print(f"    Min    : {nonzero.min()}")
    print(f"    Max    : {nonzero.max()}")
    total_steps = lengths.sum()
    reward_density = nonzero.sum() / total_steps if total_steps > 0 else 0
    print(f"    Density: {reward_density:.4f} (nonzero rewards per agent step)")

    print(f"\n  Reward density interpretation:")
    if reward_density < 0.01:
        print("    !  VERY SPARSE -- intrinsic exploration bonuses strongly recommended")
    elif reward_density < 0.05:
        print("    !  SPARSE -- standard DQN will struggle; consider RND in Week 2")
    else:
        print("    [v]  MODERATE -- standard DQN may work, but exploration help is still beneficial")

    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_plots(stats: Dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    rewards = np.array(stats["episode_rewards"])
    lengths = np.array(stats["episode_lengths"])
    nonzero = np.array(stats["nonzero_reward_count"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Solaris -- Random Agent Environment Exploration", fontsize=14, fontweight="bold")

    # 1. Episode reward distribution
    ax = axes[0, 0]
    ax.hist(rewards, bins=20, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(rewards.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {rewards.mean():.1f}")
    ax.set_xlabel("Episode Return")
    ax.set_ylabel("Count")
    ax.set_title("Episode Return Distribution")
    ax.legend()

    # 2. Episode length distribution
    ax = axes[0, 1]
    ax.hist(lengths, bins=20, color="darkorange", edgecolor="white", linewidth=0.5)
    ax.axvline(lengths.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {lengths.mean():.0f}")
    ax.set_xlabel("Episode Length (agent steps)")
    ax.set_ylabel("Count")
    ax.set_title("Episode Length Distribution")
    ax.legend()

    # 3. Reward per episode over time (learning curve baseline)
    ax = axes[1, 0]
    episodes = np.arange(1, len(rewards) + 1)
    ax.plot(episodes, rewards, color="steelblue", alpha=0.5, linewidth=0.8, label="Per episode")
    # Rolling mean (window=10)
    if len(rewards) >= 10:
        rolling = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        ax.plot(np.arange(10, len(rewards) + 1), rolling, color="red", linewidth=2, label="10-ep mean")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("Return Over Episodes (Random Agent)")
    ax.legend()

    # 4. Reward sparsity: nonzero reward events per episode
    ax = axes[1, 1]
    ax.bar(episodes, nonzero, color="mediumpurple", width=0.8, alpha=0.8)
    ax.axhline(nonzero.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {nonzero.mean():.1f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Nonzero Reward Events")
    ax.set_title("Reward Sparsity (nonzero rewards per episode)")
    ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "random_agent_exploration.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [chart] Plots saved to: {plot_path}")


def save_sample_frames(output_dir: str, seed: int = 42) -> None:
    """Save a grid of sample observations to visualise the preprocessing."""
    os.makedirs(output_dir, exist_ok=True)

    env = make_solaris_env(seed=seed, sticky_action_prob=0.0)
    obs, _ = env.reset(seed=seed)

    # Collect a few observations after random steps
    frames = [obs]
    for _ in range(15):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        frames.append(obs)
        if terminated or truncated:
            break
    env.close()

    # Plot the 4 stacked channels of the last observation
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Solaris -- Stacked Frames (most recent obs, channels 0-3)", fontsize=12)
    for i, ax in enumerate(axes):
        ax.imshow(frames[-1][i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Frame t-{3-i}" if i < 3 else "Frame t")
        ax.axis("off")

    plt.tight_layout()
    frame_path = os.path.join(output_dir, "sample_frames.png")
    plt.savefig(frame_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [image]  Sample frames saved to: {frame_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Solaris environment exploration script.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="logs/exploration", help="Output directory for plots.")
    args = parser.parse_args()

    stats = run_random_agent(n_episodes=args.episodes, seed=args.seed)
    print_summary(stats)
    save_plots(stats, args.output_dir)
    save_sample_frames(args.output_dir, seed=args.seed)
    print("  [OK] Exploration complete.\n")


if __name__ == "__main__":
    main()
