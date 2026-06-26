"""
scripts/record_video.py
-----------------------
Record MP4 videos of a trained agent (or random baseline) playing Solaris.

Uses imageio directly (gymnasiums RecordVideo crashes on Kaggle with
fps=None). The recorded env preserves full RGB frames at native Atari
resolution (210x160) so the video is human-watchable.

For trained agents, the checkpoints saved hp dict is used to rebuild
the exact training env config (frame_skip, sticky actions, etc.).

Usage:
# Trained agent
python scripts/record_video.py \
--checkpoint checkpoints/double_dqn_baseline_seed616/final.pt \
--episodes 1 --output videos/trained.mp4

# Random baseline
python scripts/record_video.py \
--random --episodes 1 --output videos/random.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

# Make repo root importable regardless of CWD
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
	sys.path.insert(0, str(_REPO))

from agents.dqn import DQNAgent
from agents.network import build_network
from env.wrappers import make_eval_env


def make_video_env(seed: int):
	"""Build a minimal env that gives us RGB frames at native Atari
	resolution (210x160). No grayscale, no resize, no frame stack.
	"""
	import gymnasium as gym
	import ale_py
	gym.register_envs(ale_py)
	env = gym.make(
		"ALE/Solaris-v5",
		render_mode="rgb_array",
		frameskip=1,
		repeat_action_probability=0.25,
		full_action_space=False,
	)
	env.reset(seed=seed)
	return env


def record_random(out_path, episodes, seed, fps):
	"""Record episodes of a uniform random policy. Returns mean reward."""
	env = make_video_env(seed=seed)
	returns = []
	for ep in range(episodes):
		ep_path = out_path if episodes == 1 else out_path.with_name(
			f"{out_path.stem}-ep{ep+1}{out_path.suffix}"
		)
		print(f" Recording random episode {ep+1} -> {ep_path}")
		obs, _ = env.reset(seed=seed + ep)
		total, steps, done = 0.0, 0, False
		with imageio.get_writer(ep_path, fps=fps) as writer:
			while not done:
				writer.append_data(env.render())
				action = env.action_space.sample()
				obs, r, term, trunc, _ = env.step(action)
				total += float(r)
				steps += 1
				done = term or trunc
		print(f" Episode {ep+1}: {steps} steps, reward={total:.1f}")
		returns.append(total)
	env.close()
	return float(np.mean(returns))


def record_trained(out_path, checkpoint_path, episodes, seed, fps):
	"""Record episodes of the trained agent. Returns mean episode reward."""
	ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
	hp = ckpt.get("hp", {})
	agent_cfg = hp.get("agent", {})
	env_cfg = hp.get("environment", {})
	
	n_actions = int(agent_cfg.get("n_actions", 18))
	obs_shape = tuple(agent_cfg.get("obs_shape", (4, 84, 84)))
	online_net = build_network(n_actions=n_actions, dueling=bool(agent_cfg.get("dueling", False)))
	agent = DQNAgent(
		n_actions=n_actions,
		obs_shape=obs_shape,
		device="cpu",
		lr=float(agent_cfg.get("lr", 1e-4)),
		gamma=float(agent_cfg.get("gamma", 0.99)),
		buffer_capacity=int(agent_cfg.get("buffer_capacity", 200_000)),
		batch_size=int(agent_cfg.get("batch_size", 32)),
		learning_starts=int(agent_cfg.get("learning_starts", 20_000)),
		target_update_freq=int(agent_cfg.get("target_update_freq", 10_000)),
		eps_start=float(agent_cfg.get("eps_start", 1.0)),
		eps_end=float(agent_cfg.get("eps_end", 0.03)),
		eps_decay_steps=int(agent_cfg.get("eps_decay_steps", 750_000)),
		grad_clip_norm=float(agent_cfg.get("grad_clip_norm", 10.0)),
		double_dqn=bool(agent_cfg.get("double_dqn", False)),
		dueling=bool(agent_cfg.get("dueling", False)),
	)
	agent.online_net.load_state_dict(ckpt["online_net"])
	agent.online_net.eval()
	
	env = make_video_env(seed=seed)
	agent_env = make_eval_env(
		seed=seed,
		clip_rewards=bool(env_cfg.get("clip_rewards", False)),
		episodic_life=bool(env_cfg.get("episodic_life", False)),
		frame_stack=int(env_cfg.get("frame_stack", 4)),
		frame_skip=int(env_cfg.get("frame_skip", 4)),
		noop_max=int(env_cfg.get("noop_max", 30)),
		sticky_action_prob=float(env_cfg.get("sticky_action_prob", 0.25)),
	)
	
	returns = []
	for ep in range(episodes):
		ep_path = out_path if episodes == 1 else out_path.with_name(
			f"{out_path.stem}-ep{ep+1}{out_path.suffix}"
		)
		print(f" Recording trained episode {ep+1} -> {ep_path}")
		obs_rgb, _ = env.reset(seed=seed + ep)
		obs_agent, _ = agent_env.reset(seed=seed + ep)
		total, steps, done = 0.0, 0, False
		with imageio.get_writer(ep_path, fps=fps) as writer:
			while not done:
				writer.append_data(obs_rgb)
				action = agent.select_action(obs_agent, eval_mode=True)
				obs_rgb, r, term, trunc, _ = env.step(action)
				obs_agent, _, _, _, _ = agent_env.step(action)
				total += float(r)
				steps += 1
				done = term or trunc
		print(f" Episode {ep+1}: {steps} steps, reward={total:.1f}")
		returns.append(total)
	env.close()
	agent_env.close()
	return float(np.mean(returns))


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--checkpoint", type=str, default=None)
	p.add_argument("--episodes", type=int, default=1)
	p.add_argument("--output", type=str, required=True)
	p.add_argument("--seed", type=int, default=42)
	p.add_argument("--fps", type=int, default=30)
	args = p.parse_args()
	
	out_path = Path(args.output)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	
	if args.checkpoint is None:
		print(f" Recording random baseline -> {out_path}")
		mean_r = record_random(out_path, args.episodes, args.seed, args.fps)
	else:
		print(f" Recording trained agent ({args.checkpoint}) -> {out_path}")
		mean_r = record_trained(
			out_path, Path(args.checkpoint), args.episodes, args.seed, args.fps,
		)
	
	print(f"\n[done] mean reward over {args.episodes} episode(s): {mean_r:.1f}")
	return 0


if __name__ == "__main__":
	sys.exit(main())