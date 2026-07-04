"""DDPG agent for mobile-robot navigation in the MobileRobotEnv environment.

The public API: import from the package root, not from submodules.
"""

from __future__ import annotations

from mobile_robot_navigation.config_schema import Config
from mobile_robot_navigation.lib import train, TrainResult
from mobile_robot_navigation.environment import MobileRobotEnv
from mobile_robot_navigation.agent import Actor, Critic, OUNoise

__version__ = "0.1.0"

__all__ = [
    "train",
    "TrainResult",
    "Config",
    "Actor",
    "Critic",
    "OUNoise",
    "MobileRobotEnv",
]
