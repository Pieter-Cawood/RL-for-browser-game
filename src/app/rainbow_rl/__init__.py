"""
Rainbow RL components for browser-based training:
- RainbowRLAgent: Rainbow DQN agent (distributional, dueling, noisy nets, n-step, PER, double Q)
- ReplayMemory: Prioritized experience replay buffer
- RainbowRLConfig: load .env config
- BrowserGameEnv: Environment wrapper that streams frames from a browser canvas via Playwright

"""

from .agent import RainbowRLAgent
from .memory import ReplayMemory
from .config import RainbowRLConfig
from .env import BrowserGameEnv

__all__ = [
    "RainbowRLAgent",
    "ReplayMemory",
    "RainbowRLConfig",
    "BrowserGameEnv",
]
