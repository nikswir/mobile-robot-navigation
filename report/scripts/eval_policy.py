"""Evaluate the trained actor across random layouts.

Loads report/assets/actor.pt and rolls out the deterministic policy on many
freshly sampled layouts to estimate the arrival rate. Saves the fastest
successful trajectory (and one failure, when observed) with their layout
geometry for the report figures.

    uv run python report/scripts/eval_policy.py
"""

from __future__ import annotations

import os
import json
import torch
import random
import contextlib
import numpy as np

from pathlib import Path

from mobile_robot_navigation.agent import Actor
from mobile_robot_navigation.environment import MobileRobotEnv

SEED = 11
N_ROLLOUTS = 300
MAX_STEPS = 500

ASSETS = Path(__file__).parents[1] / "assets"


def rollout(actor, device, env):
    state = env.reset()
    xs, ys, alphas = [env.robot.x], [env.robot.y], [env.robot.alpha]
    outcome = "timeout"
    for _ in range(MAX_STEPS):
        s = torch.FloatTensor(state).reshape(1, -1).to(device)
        with torch.no_grad():
            a = actor(s).cpu().numpy().reshape(-1)
        state, _r, done, arrived = env.step(a)
        xs.append(env.robot.x)
        ys.append(env.robot.y)
        alphas.append(env.robot.alpha)
        if arrived:
            outcome = "arrived"
            break
        if done:
            outcome = "collision"
            break
    meta = {
        "target": [env.target_x, env.target_y],
        "obstacles": [
            {"x_max": o[0], "x_min": o[1], "y_max": o[2], "y_min": o[3]}
            for o in env.obstacles_cord[4:]
        ],
        "width": env.observation_shape[1],
        "height": env.observation_shape[0],
        "target_threshold": env.target_threshold,
    }
    return xs, ys, alphas, outcome, meta


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        os.environ.get(
            "MRN_DEVICE",
            "mps" if torch.backends.mps.is_available() else "cpu",
        ),
    )
    actor = Actor(10, 2)
    ckpt = torch.load(ASSETS / "actor.pt", map_location="cpu")
    actor.load_state_dict(ckpt["model"])
    actor.to(device).eval()

    env = MobileRobotEnv(seed=SEED)
    arrived = 0
    best = None
    worst = None
    for _ in range(N_ROLLOUTS):
        xs, ys, alphas, outcome, meta = rollout(actor, device, env)
        if outcome == "arrived":
            arrived += 1
            if best is None or len(xs) < len(best[0]):
                best = (xs, ys, alphas, meta)
        elif worst is None and outcome == "collision":
            worst = (xs, ys, alphas, meta)

    rate = arrived / N_ROLLOUTS
    print(f"arrival rate: {arrived}/{N_ROLLOUTS} = {rate:.1%}")

    stats = {"n": N_ROLLOUTS, "arrived": arrived, "rate": rate}
    (ASSETS / "eval_stats.json").write_text(json.dumps(stats))

    if best is not None:
        xs, ys, alphas, meta = best
        traj = {
            "xs": xs,
            "ys": ys,
            "alphas": alphas,
            "outcome": "arrived",
            **meta,
        }
        (ASSETS / "trajectory_success.json").write_text(json.dumps(traj))
        print(f"saved successful trajectory ({len(xs)} steps)")
    else:
        print("no successful rollout captured")

    if worst is not None:
        xs, ys, alphas, meta = worst
        traj = {
            "xs": xs,
            "ys": ys,
            "alphas": alphas,
            "outcome": "collision",
            **meta,
        }
        (ASSETS / "trajectory.json").write_text(json.dumps(traj))
        print(f"saved failure trajectory ({len(xs)} steps)")


if __name__ == "__main__":
    with contextlib.redirect_stderr(open("/dev/null", "w")):
        main()
