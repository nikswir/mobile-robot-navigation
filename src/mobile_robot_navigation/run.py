"""Hydra entry point for mobile-robot-navigation.

A run is composed from `configs/` (environment / agent / training / noise) and
written into Hydra's per-run output directory, so repeated runs and `--multirun`
sweeps never collide. Things to try::

    python -m mobile_robot_navigation.run --cfg job    # print the config
    python -m mobile_robot_navigation.run training.num_episodes=5
"""

from __future__ import annotations

import torch
import hydra

from typing import cast
from pathlib import Path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from mobile_robot_navigation import config_schema
from mobile_robot_navigation.config_schema import Config
from mobile_robot_navigation.lib import train, TrainResult

# Register the structured-config schema so Hydra type-checks the composed YAML.
config_schema.register()

########################################
#               Core run               #
########################################


def run(cfg: Config, out_dir: Path) -> TrainResult:
    """Train from the composed config and save the actor checkpoint."""
    # ── Train, then persist the actor state dict into the run dir ──
    result = train(cfg)
    checkpoint = {"model": result.actor.state_dict()}
    torch.save(checkpoint, out_dir / "my-DDPG-ckpt.pt")
    return result


########################################
#             Entry point              #
########################################


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    # ── Hydra gives each run (and each --multirun job) its own output dir ──
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    result = run(cast(Config, cfg), out_dir)
    print(f"trained {len(result.episode_rewards)} episodes -> {out_dir}")


if __name__ == "__main__":
    main()
