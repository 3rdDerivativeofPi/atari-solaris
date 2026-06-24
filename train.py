"""
train.py
--------
Main training script for DQN on Solaris.

Usage:
    python train.py --config configs/dqn_baseline.yaml
    python train.py --config configs/dqn_baseline.yaml --seed 123
    python train.py --config configs/dqn_baseline.yaml --no-wandb

Logs metrics to Weights & Biases (or stdout if --no-wandb is set).
Saves checkpoints to checkpoints/<experiment_name>/.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import argparse
import time
import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import yaml

from env.wrappers import make_solaris_env, make_eval_env
from agents.dqn import DQNAgent


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_global_seeds(seed: int) -> None:
    """Fix all RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic convolutions (slight speed cost, ensures reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the most recently-saved step checkpoint in a directory.

    Looks for files named `step_NNNNNNNNN.pt` and returns the one with the
    highest step count. Ignores `final.pt` (only written on full completion,
    so its presence means there's nothing to resume).

    Args:
        checkpoint_dir: Directory to search.

    Returns:
        Path to the latest checkpoint, or None if no checkpoints exist.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    step_ckpts = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("step_") and f.endswith(".pt")
    ]
    if not step_ckpts:
        return None

    # Sort by the numeric step embedded in the filename
    step_ckpts.sort(key=lambda f: int(f[len("step_"):-len(".pt")]))
    return os.path.join(checkpoint_dir, step_ckpts[-1])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    agent:     DQNAgent,
    n_episodes: int = 10,
    seed:      int = 0,
) -> dict:
    """
    Evaluate the agent's current policy over n_episodes.

    Uses the eval environment (no reward clipping, no episodic life),
    with a fixed small epsilon (0.05) for action selection.

    Returns:
        Dictionary with mean, std, min, max episode returns.
    """
    env = make_eval_env(seed=seed)
    returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.select_action(obs, eval_mode=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward

        returns.append(ep_return)

    env.close()

    return {
        "eval/mean_return": float(np.mean(returns)),
        "eval/std_return":  float(np.std(returns)),
        "eval/min_return":  float(np.min(returns)),
        "eval/max_return":  float(np.max(returns)),
        "eval/n_episodes":  n_episodes,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: dict, seed: int, use_wandb: bool, resume: bool) -> None:
    """
    Main training loop.

    Args:
        config:    Parsed YAML config dictionary.
        seed:      Random seed (overrides config if provided).
        use_wandb: Whether to log to Weights & Biases.
        resume:    If True, automatically resume from the latest checkpoint
                   in this experiment's checkpoint directory, if one exists.
                   Designed for chunked training on session-limited platforms
                   (Colab, Kaggle) where a single run can't finish in one sitting.
    """
    exp_cfg = config["experiment"]
    env_cfg = config["environment"]
    agent_cfg = config["agent"]
    log_cfg = config["logging"]

    # Apply seed
    seed = seed if seed is not None else exp_cfg["seed"]
    set_global_seeds(seed)

    # Experiment name
    exp_name = f"{exp_cfg['name']}_seed{seed}"
    checkpoint_dir = os.path.join("checkpoints", exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Experiment : {exp_name}")
    print(f"  Seed       : {seed}")
    print(f"  Device     : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"{'='*55}\n")

    # ----------------------------------------------------------------
    # W&B initialisation
    # ----------------------------------------------------------------
    wandb_run = None
    if use_wandb and log_cfg.get("use_wandb", False):
        try:
            import wandb
            # Use a stable run id derived from the experiment name so that
            # resumed sessions append to the SAME W&B run instead of creating
            # a new one each time — critical for a continuous learning curve
            # when training is chunked across multiple Colab/Kaggle sessions.
            wandb_run = wandb.init(
                project=log_cfg["wandb_project"],
                entity=log_cfg.get("wandb_entity"),
                name=exp_name,
                id=exp_name,
                resume="allow",
                config={**config, "seed": seed},
            )
            print(f"  📡 W&B run: {wandb_run.url}\n")
        except ImportError:
            print("  ⚠️  wandb not installed; logging to stdout only.\n")

    # ----------------------------------------------------------------
    # Environment
    # ----------------------------------------------------------------
    env = make_solaris_env(
        seed=seed,
        clip_rewards=env_cfg["clip_rewards"],
        episodic_life=env_cfg["episodic_life"],
        frame_stack=env_cfg["frame_stack"],
        frame_skip=env_cfg["frame_skip"],
        noop_max=env_cfg["noop_max"],
        sticky_action_prob=env_cfg["sticky_action_prob"],
    )

    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape  # (4, 84, 84)

    # ----------------------------------------------------------------
    # Agent
    # ----------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = DQNAgent(
        n_actions=          n_actions,
        obs_shape=          obs_shape,
        device=             device,
        lr=                 agent_cfg["lr"],
        gamma=              agent_cfg["gamma"],
        buffer_capacity=    agent_cfg["buffer_capacity"],
        batch_size=         agent_cfg["batch_size"],
        learning_starts=    agent_cfg["learning_starts"],
        target_update_freq= agent_cfg["target_update_freq"],
        eps_start=          agent_cfg["eps_start"],
        eps_end=            agent_cfg["eps_end"],
        eps_decay_steps=    agent_cfg["eps_decay_steps"],
        grad_clip_norm=     agent_cfg.get("grad_clip_norm", 10.0),
        double_dqn=         agent_cfg.get("double_dqn", False),
        dueling=            agent_cfg.get("architecture", "standard") == "dueling",
    )

    # ----------------------------------------------------------------
    # Resume from checkpoint (chunked-session training)
    # ----------------------------------------------------------------
    # NOTE: the replay buffer is NOT restored — it's too large to checkpoint
    # cheaply (10-26GB). Resuming starts with an empty buffer that refills
    # over the first `learning_starts` steps. Network weights, optimiser
    # state, and step counters (t, epsilon schedule, target update timing)
    # ARE restored, so the learned policy and training schedule continue
    # correctly; only the buffer's diversity takes a short time to rebuild.
    start_step = 0
    if resume:
        latest_ckpt = find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt is not None:
            agent.load_checkpoint(latest_ckpt)
            start_step = agent.t
            print(f"  ▶️  Resuming from step {start_step:,} ({latest_ckpt})\n")
        else:
            print(f"  ℹ️  --resume set but no checkpoint found in "
                  f"{checkpoint_dir}; starting fresh.\n")

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    total_steps    = exp_cfg["total_steps"]
    eval_freq      = exp_cfg["eval_freq"]
    checkpoint_freq = exp_cfg["checkpoint_freq"]
    log_freq       = exp_cfg["log_freq"]
    eval_episodes  = exp_cfg["eval_episodes"]

    # Running stats
    obs, _              = env.reset(seed=seed)
    ep_return           = 0.0
    ep_steps            = 0
    ep_count            = 0
    recent_returns      = deque(maxlen=100)   # Last 100 episode returns
    recent_losses       = deque(maxlen=1000)  # Last 1000 TD losses
    t_start             = time.time()
    last_log_t          = 0

    print(f"  🚀 Training started — {total_steps:,} total steps "
          f"(resuming from {start_step:,})\n" if start_step else
          f"  🚀 Training started — {total_steps:,} total steps\n")

    if start_step >= total_steps:
        print(f"  ✅ Checkpoint already at/past total_steps ({start_step:,} "
              f">= {total_steps:,}); nothing left to train.\n")
        env.close()
        if wandb_run:
            wandb_run.finish()
        return

    for t in range(start_step + 1, total_steps + 1):
        # Select and execute action
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition and train
        loss = agent.step(obs, action, reward, next_obs, terminated)

        if loss is not None:
            recent_losses.append(loss)

        ep_return += reward
        ep_steps  += 1
        obs        = next_obs

        # Episode end
        if done:
            ep_count += 1
            recent_returns.append(ep_return)
            obs, _ = env.reset()
            ep_return = 0.0
            ep_steps  = 0

        # ── Logging ──────────────────────────────────────────────────
        if t % log_freq == 0:
            elapsed   = time.time() - t_start
            # SPS counts only steps taken THIS session (not total t),
            # so it doesn't deflate on resumed runs where t starts high.
            session_steps = t - start_step
            sps       = session_steps / elapsed if elapsed > 0 else 0
            mean_ret  = np.mean(recent_returns) if recent_returns else 0.0
            mean_loss = np.mean(recent_losses) if recent_losses else 0.0

            log_data = {
                "train/step":           t,
                "train/episodes":       ep_count,
                "train/mean_return_100":mean_ret,
                "train/mean_loss":      mean_loss,
                "train/mean_q_value":   agent.last_mean_q,
                "train/epsilon":        agent.epsilon,
                "train/steps_per_sec":  sps,
                "train/buffer_size":    len(agent.replay_buffer),
                "train/updates":        agent.updates,
            }

            if wandb_run:
                # Always log with the true global step t so that resumed
                # sessions append monotonically to the existing W&B run,
                # rather than restarting from 1 and triggering step warnings.
                wandb_run.log(log_data, step=t)

            if t % (log_freq * 10) == 0:
                print(
                    f"  Step {t:>9,} | "
                    f"Episodes: {ep_count:>5} | "
                    f"Mean return (100ep): {mean_ret:>8.1f} | "
                    f"Loss: {mean_loss:.4f} | "
                    f"Q: {agent.last_mean_q:>7.2f} | "
                    f"ε: {agent.epsilon:.3f} | "
                    f"SPS: {sps:.0f}"
                )

        # ── Evaluation ───────────────────────────────────────────────
        if t % eval_freq == 0:
            print(f"\n  📊 Evaluating at step {t:,}...")
            eval_metrics = evaluate(agent, n_episodes=eval_episodes, seed=seed + 1000)

            print(
                f"     Eval return: {eval_metrics['eval/mean_return']:.1f} "
                f"± {eval_metrics['eval/std_return']:.1f} "
                f"(min={eval_metrics['eval/min_return']:.1f}, "
                f"max={eval_metrics['eval/max_return']:.1f})"
            )

            if wandb_run:
                wandb_run.log(eval_metrics, step=t)  # t is always the global step

        # ── Checkpointing ────────────────────────────────────────────
        if t % checkpoint_freq == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"step_{t:09d}.pt")
            agent.save_checkpoint(ckpt_path)

    # ── Final checkpoint & cleanup ───────────────────────────────────
    final_path = os.path.join(checkpoint_dir, "final.pt")
    agent.save_checkpoint(final_path)
    env.close()

    if wandb_run:
        wandb_run.finish()

    total_time = time.time() - t_start
    print(f"\n  ✅ Training complete in {total_time/3600:.2f} hours.")
    print(f"  Final checkpoint: {final_path}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train DQN on Solaris.")
    parser.add_argument(
        "--config", type=str, default="configs/dqn_baseline.yaml",
        help="Path to YAML config file."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides config)."
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable Weights & Biases logging."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the latest checkpoint in this experiment's "
             "checkpoint directory, if one exists. Use this for chunked "
             "training across multiple Colab/Kaggle sessions."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config=config, seed=args.seed, use_wandb=not args.no_wandb, resume=args.resume)


if __name__ == "__main__":
    main()