"""Train the DDPG agent and dump artifacts for the report.

Runs a full training session (MPS when available), records the per-episode
reward curve, saves the trained actor, and rolls out one greedy episode to
capture a trajectory. Outputs land in report/assets/.

    uv run python report/scripts/run_experiment.py
"""

from __future__ import annotations

import os
import json
import torch
import random
import contextlib
import numpy as np

from pathlib import Path

from mobile_robot_navigation.agent import Actor, train_policy
from mobile_robot_navigation.environment import MobileRobotEnv

SEED = 7
EPISODES = 3000
NOISE_EPISODES = 2400
MAX_STEPS = 500

ASSETS = Path(__file__).parents[1] / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

########################################
#                Device                #
########################################


def select_device() -> torch.device:
    # MRN_DEVICE overrides autodetection (e.g. MRN_DEVICE=cpu to dodge the
    # deterministic MPS command-buffer hang seen on long runs).
    forced = os.environ.get("MRN_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


########################################
#               Rollout                #
########################################


def greedy_trajectory(actor: Actor, device: torch.device) -> dict:
    """Roll out one episode with the deterministic (no-noise) policy."""
    env = MobileRobotEnv(seed=SEED + 1)
    state = env.reset()
    xs, ys, alphas = [env.robot.x], [env.robot.y], [env.robot.alpha]
    outcome = "timeout"
    for _ in range(MAX_STEPS):
        s = torch.FloatTensor(state).reshape(1, -1).to(device)
        with torch.no_grad():
            action = actor(s).cpu().numpy().reshape(-1)
        state, _reward, done, arrived = env.step(action)
        xs.append(env.robot.x)
        ys.append(env.robot.y)
        alphas.append(env.robot.alpha)
        if arrived:
            outcome = "arrived"
            break
        if done:
            # Distinguish a real collision from leaving the field.
            outcome = (
                "out_of_bounds"
                if env.out_of_boundary(env.robot)
                else "collision"
            )
            break
    obstacles = [
        {"x_max": o[0], "x_min": o[1], "y_max": o[2], "y_min": o[3]}
        for o in env.obstacles_cord[4:]
    ]
    return {
        "xs": xs,
        "ys": ys,
        "alphas": alphas,
        "outcome": outcome,
        "target": [env.target_x, env.target_y],
        "obstacles": obstacles,
        "width": env.observation_shape[1],
        "height": env.observation_shape[0],
        "target_threshold": env.target_threshold,
    }


########################################
#             Entry point              #
########################################


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = select_device()
    print(f"device = {device}")

    env = MobileRobotEnv(seed=SEED)
    actor, rewards = train_policy(
        env,
        num_episodes=EPISODES,
        num_noise_episodes=NOISE_EPISODES,
        max_steps=MAX_STEPS,
        gamma=0.99,
        tau=0.005,
        buffer_size=100000,
        batch_size=256,
        device=device,
    )

    torch.save({"model": actor.state_dict()}, ASSETS / "actor.pt")
    (ASSETS / "rewards.json").write_text(json.dumps(rewards))
    print(f"saved actor + {len(rewards)} episode rewards")

    traj = greedy_trajectory(actor, device)
    (ASSETS / "trajectory.json").write_text(json.dumps(traj))
    print(f"greedy rollout: {traj['outcome']}, {len(traj['xs'])} steps")


if __name__ == "__main__":
    # Silence the noisy gym/cv2 banners on stderr; keep our prints.
    with contextlib.redirect_stderr(open("/dev/null", "w")):
        main()
