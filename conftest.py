# tests/conftest.py
import sys
import os

# Add the project root to sys.path so pytest can find the `env` module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Register ALE environments with Gymnasium before any test runs
import ale_py
import gymnasium as gym
gym.register_envs(ale_py)