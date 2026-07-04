"""Structured-config schema — the run's typed contract.

Hydra validates `configs/` against these dataclasses: each field has a type and
a literal default (overridden by the matching config-group option file).
Registering the root makes Hydra reject wrong types / unknown fields at startup,
instead of crashing deep inside the run. One dataclass per config group.
"""

from __future__ import annotations

from dataclasses import field, dataclass
from hydra.core.config_store import ConfigStore

########################################
#            Group schemas             #
########################################


@dataclass
class EnvironmentConfig:
    """The MobileRobotEnv environment knobs (layouts randomize per episode)."""

    poi_threshold: float = 5
    min_obstacles: int = 2
    max_obstacles: int = 5
    target_threshold: float = 10
    obstacle_threshold: float = 1
    min_obstacle_size: int = 60
    max_obstacle_size: int = 220
    observation_width: int = 800
    observation_height: int = 600
    min_start_distance: float = 300


@dataclass
class AgentConfig:
    """Network sizes and per-network learning rates."""

    actor_lr: float = 1e-4
    critic_lr: float = 5e-4
    hidden1: int = 128
    hidden2: int = 256
    state_size: int = 10
    action_size: int = 2
    critic_input_size: int = 12


@dataclass
class TrainingConfig:
    """DDPG training-loop hyper-parameters."""

    gamma: float = 0.99
    tau: float = 0.005
    max_steps: int = 500
    batch_size: int = 256
    buffer_size: int = 100000
    end_factor: float = 0.05
    num_episodes: int = 3000
    num_noise_episodes: int = 2400


@dataclass
class NoiseConfig:
    """Ornstein-Uhlenbeck exploration-noise parameters."""

    mu: float = 0.0
    theta: float = 0.15
    sigma: float = 0.2


########################################
#           Root & registry            #
########################################


@dataclass
class Config:
    """The composed run config — one field per group."""

    noise: NoiseConfig = field(default_factory=NoiseConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = field(
        default_factory=EnvironmentConfig,
    )


def register() -> None:
    # ── Expose the schema as `config_schema` for config.yaml's defaults ──
    ConfigStore.instance().store(name="config_schema", node=Config)
