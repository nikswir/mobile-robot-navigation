"""Animate trained-policy rollouts on random layouts into a README GIF.

Loads report/assets/actor.pt, rolls out the deterministic policy on freshly
sampled layouts until it collects a few successful episodes, then re-renders
each recorded episode frame-by-frame with the vector renderer (`viz`) and
writes an animated GIF preview to docs/assets/demo.gif.

    uv run python report/make_gif.py
"""

from __future__ import annotations

import io
import random
import contextlib
from pathlib import Path

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from mobile_robot_navigation import viz
from mobile_robot_navigation.agent import Actor
from mobile_robot_navigation.environment import ChopperScape

SEED = 29
N_EPISODES = 10
MAX_STEPS = 500
MAX_TRIES = 200
FRAME_STRIDE = 3
HOLD_LAST = 8
FRAME_MS = 90

OUT = Path(__file__).parent.parent / "docs" / "assets" / "demo.gif"
ASSETS = Path(__file__).parent / "assets"


def record_episode(actor, device, env):
    """Roll out one deterministic episode and record poses + scans."""
    state = env.reset()
    frames = [(env.chopper.x, env.chopper.y, env.chopper.alpha, env.scan())]
    outcome = "timeout"
    for _ in range(MAX_STEPS):
        s = torch.FloatTensor(state).reshape(1, -1).to(device)
        with torch.no_grad():
            a = actor(s).cpu().numpy().reshape(-1)
        state, _r, done, arrived = env.step(a)
        frames.append(
            (env.chopper.x, env.chopper.y, env.chopper.alpha, env.scan()),
        )
        if arrived:
            outcome = "arrived"
            break
        if done:
            outcome = "collision"
            break
    meta = {
        "target": (env.target_x, env.target_y),
        "obstacles": [
            {"x_max": o[0], "x_min": o[1], "y_max": o[2], "y_min": o[3]}
            for o in env.obstacles_cord[4:]
        ],
        "width": env.observation_shape[1],
        "height": env.observation_shape[0],
        "target_threshold": env.target_threshold,
        "scan_range": list(env.chopper.scan_range),
        "max_linear": env.max_linear,
    }
    return frames, outcome, meta


def render_frame(fig, ax, meta, trail, pose, episode, total):
    """Draw one animation frame and rasterize it to a PIL image."""
    ax.clear()
    viz.draw_field(ax, meta["width"], meta["height"])
    viz.draw_obstacles(ax, meta["obstacles"])
    viz.draw_target(ax, meta["target"], meta["target_threshold"])
    if len(trail) > 1:
        xs, ys = zip(*trail)
        ax.plot(
            xs, ys, color=viz.TRAIL_COLOR, linewidth=2.0,
            alpha=0.75, solid_capstyle="round", zorder=7,
        )
    x, y, alpha, scans = pose
    viz.draw_lidar(
        ax, x, y, alpha,
        meta["scan_range"], scans, meta["max_linear"],
    )
    viz.draw_rover(ax, x, y, alpha, scale=0.9)
    ax.set_title(
        f"DDPG policy on a random layout  ({episode}/{total})",
        fontsize=10,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu",
    )
    actor = Actor(10, 2)
    ckpt = torch.load(ASSETS / "actor.pt", map_location="cpu")
    actor.load_state_dict(ckpt["model"])
    actor.to(device).eval()

    env = ChopperScape(seed=SEED)
    episodes = []
    for _ in range(MAX_TRIES):
        frames, outcome, meta = record_episode(actor, device, env)
        if outcome == "arrived":
            episodes.append((frames, meta))
            print(f"episode {len(episodes)}: arrived in {len(frames)} steps")
        if len(episodes) == N_EPISODES:
            break
    if not episodes:
        raise SystemExit("no successful episodes to animate")

    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.02)
    images = []
    for idx, (frames, meta) in enumerate(episodes, start=1):
        picks = list(range(0, len(frames), FRAME_STRIDE))
        if picks[-1] != len(frames) - 1:
            picks.append(len(frames) - 1)
        for k in picks:
            trail = [(f[0], f[1]) for f in frames[: k + 1]]
            img = render_frame(
                fig, ax, meta, trail, frames[k], idx, len(episodes),
            )
            images.append(img)
        images.extend([images[-1]] * HOLD_LAST)
    plt.close(fig)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        OUT,
        save_all=True,
        append_images=images[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
    )
    size_mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT} ({len(images)} frames, {size_mb:.1f} MB)")


if __name__ == "__main__":
    with contextlib.redirect_stderr(open("/dev/null", "w")):
        main()
