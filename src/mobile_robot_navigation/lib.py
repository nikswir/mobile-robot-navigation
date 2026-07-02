"""Library API — the package's real work and its result type.

Pure entry point: turns a composed `Config` into a trained actor and the
per-episode reward history. Independent of Hydra and the CLI and free of file
I/O — `run.py` unpacks the config, calls `train`, and saves the checkpoint.
`__init__` re-exports this surface as the public API.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass

from mobile_robot_navigation.config_schema import Config
from mobile_robot_navigation.environment import ChopperScape
from mobile_robot_navigation.agent import Actor, train_policy

########################################
#           Public contract            #
########################################


@dataclass
class TrainResult:
    """What a training run produces — the public result type."""

    actor: Actor
    episode_rewards: list[float]


########################################
#            Training entry            #
########################################


def train(
    cfg: Config,
    *,
    device: torch.device | None = None,
) -> TrainResult:
    """Build the env and agent from `cfg`, train, and report the rewards."""
    # ── Build the environment from the environment group ──
    env = ChopperScape(
        target_threshold=cfg.environment.target_threshold,
        obstacle_threshold=cfg.environment.obstacle_threshold,
        poi_threshold=cfg.environment.poi_threshold,
        min_obstacles=cfg.environment.min_obstacles,
        max_obstacles=cfg.environment.max_obstacles,
        min_obstacle_size=cfg.environment.min_obstacle_size,
        max_obstacle_size=cfg.environment.max_obstacle_size,
        min_start_distance=cfg.environment.min_start_distance,
        observation_height=cfg.environment.observation_height,
        observation_width=cfg.environment.observation_width,
    )

    # ── Run the DDPG loop with the agent / training / noise groups ──
    actor, episode_rewards = train_policy(
        env,
        num_episodes=cfg.training.num_episodes,
        num_noise_episodes=cfg.training.num_noise_episodes,
        max_steps=cfg.training.max_steps,
        gamma=cfg.training.gamma,
        tau=cfg.training.tau,
        buffer_size=cfg.training.buffer_size,
        batch_size=cfg.training.batch_size,
        actor_lr=cfg.agent.actor_lr,
        critic_lr=cfg.agent.critic_lr,
        state_size=cfg.agent.state_size,
        action_size=cfg.agent.action_size,
        critic_input_size=cfg.agent.critic_input_size,
        hidden1=cfg.agent.hidden1,
        hidden2=cfg.agent.hidden2,
        end_factor=cfg.training.end_factor,
        noise_mu=cfg.noise.mu,
        noise_theta=cfg.noise.theta,
        noise_sigma=cfg.noise.sigma,
        device=device,
    )

    return TrainResult(actor=actor, episode_rewards=episode_rewards)
