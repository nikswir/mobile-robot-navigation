"""DDPG training-loop tests (tiny, CPU, stage 1).

A smoke test drives `train_policy` end-to-end; the unit tests below pin the
pieces the smoke test only executes — the Polyak soft update, the FIFO replay
buffer, and the `train()` config→actor wiring.
"""

from __future__ import annotations

import random

import torch
import numpy as np

from mobile_robot_navigation import train, Config
from mobile_robot_navigation.environment import MobileRobotEnv

from mobile_robot_navigation.agent import (
    Actor,
    train_policy,
    add_to_replay_buffer,
    update_target_network,
)

CPU = torch.device("cpu")

########################################
#              Smoke test              #
########################################


def test_train_policy_smoke() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # ── Two tiny episodes: enough to exercise updates end-to-end ──
    env = MobileRobotEnv(seed=0)
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


########################################
#          Update primitives           #
########################################


def test_update_target_network_polyak() -> None:
    """tau=1 hard-copies the source; tau=0.5 lands on the midpoint."""
    torch.manual_seed(0)
    source = Actor(4, 2, hidden1=8, hidden2=8)
    target = Actor(4, 2, hidden1=8, hidden2=8)

    # ── tau=1: target becomes an exact copy of the source ──
    update_target_network(target, source, 1.0)
    pairs = zip(target.parameters(), source.parameters(), strict=True)
    for t, s in pairs:
        assert torch.equal(t, s)

    # ── tau=0.5: each target param is the midpoint of old-self and source ──
    torch.manual_seed(1)
    source = Actor(4, 2, hidden1=8, hidden2=8)
    target = Actor(4, 2, hidden1=8, hidden2=8)
    before = [p.clone() for p in target.parameters()]
    update_target_network(target, source, 0.5)
    triples = zip(
        target.parameters(),
        before,
        source.parameters(),
        strict=True,
    )
    for t, b, s in triples:
        assert torch.allclose(t, 0.5 * b + 0.5 * s)


def test_replay_buffer_evicts_oldest_past_capacity() -> None:
    """The buffer is FIFO-capped: adding past capacity drops the oldest."""
    buffer: list[list[torch.Tensor]] = []
    for i in range(5):
        add_to_replay_buffer(
            buffer,
            3,
            torch.tensor([float(i)]),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
        )

    # ── Capped at 3, holding the three most recent states (2, 3, 4) ──
    assert len(buffer) == 3
    states = [entry[0].item() for entry in buffer]
    assert states == [2.0, 3.0, 4.0]


def test_train_wires_config_into_the_actor() -> None:
    """`train` threads the agent config through to the built actor."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # ── Tiny run that still fires a gradient update (batch fills in-episode)
    #    so the whole train() -> train_policy() path is exercised ──
    cfg = Config()
    cfg.training.num_episodes = 1
    cfg.training.num_noise_episodes = 1
    cfg.training.max_steps = 4
    cfg.training.batch_size = 2
    cfg.training.buffer_size = 50
    cfg.agent.hidden1 = 8
    cfg.agent.hidden2 = 12

    result = train(cfg, device=CPU)

    # ── One episode of history, and the actor's shape reflects the config
    #    (state_size -> hidden1 -> hidden2 -> action_size), so a swapped or
    #    dropped kwarg between train() and train_policy() fails here ──
    assert len(result.episode_rewards) == 1
    assert result.actor.linear1.in_features == cfg.agent.state_size
    assert result.actor.linear1.out_features == cfg.agent.hidden1
    assert result.actor.linear2.out_features == cfg.agent.hidden2
    assert result.actor.linear3.out_features == cfg.agent.action_size
