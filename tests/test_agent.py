"""End-to-end smoke test of the DDPG training loop (tiny, CPU, stage 1).

Runs `train_policy` for two short episodes with miniature networks against a
randomized environment: the loop must survive replay-buffer warm-up, network
updates and episode bookkeeping, and hand back a usable actor.
"""

from __future__ import annotations

import random

import torch
import numpy as np

from mobile_robot_navigation.agent import train_policy
from mobile_robot_navigation.environment import ChopperScape

CPU = torch.device("cpu")


def test_train_policy_smoke() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # ── Two tiny episodes: enough to exercise updates end-to-end ──
    env = ChopperScape(seed=0)
    actor, rewards = train_policy(
        env,
        num_episodes=2,
        num_noise_episodes=2,
        max_steps=15,
        gamma=0.99,
        tau=0.05,
        buffer_size=100,
        batch_size=8,
        hidden1=16,
        hidden2=16,
        device=CPU,
    )

    # ── Reward history matches the episode count ──
    assert len(rewards) == 2
    assert all(isinstance(r, float) for r in rewards)

    # ── The returned actor maps a 10-d state to a bounded 2-d action ──
    with torch.no_grad():
        action = actor(torch.zeros(1, 10))
    assert action.shape == (1, 2)
    assert bool((action.abs() <= 1.0).all())
