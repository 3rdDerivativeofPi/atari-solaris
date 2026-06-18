# Solaris RL — Atari Reinforcement Learning Research Project

Reproducible RL research on the Solaris Atari 2600 game using ALE + PyTorch.

---

## Hardware Target

| Component | Spec |
|-----------|------|
| CPU | Intel Core i9-14900HX |
| GPU | NVIDIA RTX 4080 Laptop (CUDA) |
| RAM | 32 GB |

---

## Project Structure

```
solaris_rl/
├── env/
│   ├── __init__.py
│   └── wrappers.py          # All Atari preprocessing wrappers
├── agents/                  # DQN, Rainbow (added in Week 1-2)
├── configs/                 # YAML hyperparameter configs
├── scripts/
│   └── explore_env.py       # Day 1-2: random agent + env profiling
├── tests/
│   └── test_wrappers.py     # Unit tests for wrappers
├── logs/                    # Training logs, W&B runs
├── checkpoints/             # Model checkpoints
├── notebooks/               # Jupyter exploration notebooks
└── requirements.txt
```

---

## Installation

### 1. Create a Conda environment (recommended)

```bash
conda create -n solaris_rl python=3.11 -y
conda activate solaris_rl
```

### 2. Install PyTorch with CUDA 12.x (for RTX 4080)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install project dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the Atari ROMs

```bash
AutoROM --accept-license
```

### 5. Verify the environment

```bash
python -c "
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
env = gym.make('ALE/Solaris-v5', frameskip=1)
obs, _ = env.reset()
print('✅ Solaris env OK — obs shape:', obs.shape)
env.close()
"
```

---

## Usage

### Run environment exploration (Days 1-2)

```bash
python scripts/explore_env.py --episodes 50 --seed 42
```

This runs a random agent and saves reward/length distribution plots to `logs/exploration/`.

### Run unit tests

```bash
pytest tests/test_wrappers.py -v
```

---

## Key Design Decisions

### Wrapper Stack (in order)
1. **NoopResetEnv** — Random 1–30 no-ops at episode start (reduces start-state overfitting)
2. **MaxAndSkipEnv** — Frame skip=4 + pixel-wise max over last 2 frames (eliminates flicker)
3. **EpisodicLifeEnv** — Life loss = terminal signal during training (denser credit assignment)
4. **FireResetEnv** — Press FIRE if required at episode start
5. **WarpFrame** — Grayscale + resize to 84×84 (INTER_AREA interpolation)
6. **ClipRewardEnv** — Clip to {-1, 0, +1} during training only
7. **FrameStack(4)** — Stack 4 frames → (4, 84, 84), channels-first for PyTorch

### Sticky Actions
- Training and eval use `repeat_action_probability=0.25` (Machado et al., 2018)
- This is the current ALE standard for more robust, less overfitted evaluation
- Set `sticky_action_prob=0.0` for the deterministic variant

---

## References

- Mnih et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
- Hessel et al. (2017). *Rainbow: Combining Improvements in Deep Reinforcement Learning*. AAAI.
- Machado et al. (2018). *Revisiting the Arcade Learning Environment*. JAIR.
- Farama Foundation. [Gymnasium ALE documentation](https://ale.farama.org/)
