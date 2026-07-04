"""Animate trained-policy rollouts on random layouts into a README GIF.

Loads report/assets/actor.pt, rolls out the deterministic policy on freshly
sampled layouts until it collects a few successful episodes, then re-renders
each recorded episode frame-by-frame with the same vector renderer (`viz`)
that draws every report figure, and writes an animated GIF preview to
docs/assets/demo.gif.

    uv run python report/make_gif.py
"""

from __future__ import annotations

import io
import os
import random
import contextlib
from pathlib import Path

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Circle

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
DPI = 110

OUT = Path(__file__).parent.parent / "docs" / "assets" / "demo.gif"
ASSETS = Path(__file__).parent / "assets"

INK = "#27324a"
MUTED = "#6b7896"


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


def render_frame(fig, ax, meta, trail, pose, episode, total, step):
    """One frame: the field fills the whole image, HUD text sits on it.

    Full-bleed with square corners, so no page background ever peeks out —
    the GIF looks right on GitHub's dark and light themes alike.
    """
    w, h = meta["width"], meta["height"]
    ax.clear()

    # ── Full-bleed field with a flat border ──
    ax.set_facecolor(viz.FIELD_FACE)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    viz.draw_obstacles(ax, meta["obstacles"])
    viz.draw_target(ax, meta["target"], meta["target_threshold"])

    # ── Path so far, with the start marker used in the report figures ──
    if len(trail) > 1:
        xs, ys = zip(*trail)
        ax.plot(
            xs, ys, color=viz.TRAIL_COLOR, linewidth=2.2,
            alpha=0.85, solid_capstyle="round", zorder=7,
        )
        ax.add_patch(
            Circle(
                (xs[0], ys[0]), 7.0, facecolor="white",
                edgecolor=viz.TRAIL_COLOR, linewidth=2.0, zorder=7,
            ),
        )

    # ── Rover and its lidar fan, both anchored at the body centre ──
    x, y, alpha, scans = pose
    cx, cy = x + 16, y + 16
    viz.draw_lidar(
        ax, cx, cy, alpha,
        meta["scan_range"], scans, meta["max_linear"],
    )
    viz.draw_rover(ax, cx, cy, alpha)

    # ── HUD on the field itself ──
    ax.text(
        16, 14, "DDPG policy on unseen random layouts",
        color=INK, fontsize=11, fontweight="bold", va="top", zorder=9,
    )
    ax.text(
        w - 16, 14, f"episode {episode:02d}/{total:02d} · step {step:03d}",
        color=MUTED, fontsize=10, family="monospace",
        ha="right", va="top", zorder=9,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=viz.FIELD_FACE)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE)


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

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    images = []
    for idx, (frames, meta) in enumerate(episodes, start=1):
        picks = list(range(0, len(frames), FRAME_STRIDE))
        if picks[-1] != len(frames) - 1:
            picks.append(len(frames) - 1)
        for k in picks:
            trail = [(f[0] + 16, f[1] + 16) for f in frames[: k + 1]]
            img = render_frame(
                fig, ax, meta, trail, frames[k],
                idx, len(episodes), k,
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
