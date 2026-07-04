"""Generate every figure used in the report.

Reads training artifacts from report/assets/ and the live environment geometry,
then writes vector PDFs (and PNG previews) to report/figures/.

    uv run python report/scripts/make_figures.py
"""

from __future__ import annotations

import os

# Pin the headless backend before pyplot is imported anywhere below.
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import math
import contextlib
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from mobile_robot_navigation import viz
from mobile_robot_navigation.environment import POI, MobileRobotEnv

HERE = Path(__file__).parents[1]
FIG = HERE / "figures"
ASSETS = HERE / "assets"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "savefig.bbox": "tight",
        "figure.dpi": 130,
    },
)

INK = "#27324a"
ACCENT = "#2f6df0"
GOOD = "#1f9d6b"
BAD = "#d6453d"


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print(f"wrote {name}")


########################################
#       1. Environment schematic       #
########################################


def fig_environment():
    env = MobileRobotEnv(seed=13)
    env.reset()
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    viz.render_env(env, ax=ax)
    sx, sy = env.robot.x, env.robot.y
    ax.annotate(
        "start",
        (sx, sy),
        (sx, sy + 60),
        color=INK,
        fontsize=10,
        ha="center",
        arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.8},
    )
    ax.annotate(
        "goal",
        (env.target_x, env.target_y),
        (env.target_x - 70, env.target_y),
        color=GOOD,
        fontsize=10,
        ha="right",
        va="center",
    )
    first = env.obstacles[0]
    ax.text(
        first.x + first.icon_w / 2,
        first.y + first.icon_h / 2,
        "obstacle",
        color="white",
        fontsize=10,
        ha="center",
        va="center",
        fontweight="bold",
    )
    ax.set_title(
        "MobileRobotEnv navigation environment (one sampled layout)",
    )
    save(fig, "environment")


########################################
#      1b. Random layouts montage      #
########################################


def fig_layouts():
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.2))
    for ax, seed in zip(axes, (21, 22, 23), strict=False):
        env = MobileRobotEnv(seed=seed)
        env.reset()
        viz.render_env(env, ax=ax, show_lidar=False)
        ax.set_title(f"sampled layout (seed {seed})", fontsize=10)
    fig.suptitle(
        "Per-episode layout randomization: obstacle count, size, position "
        "and start pose all vary",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "layouts")


########################################
#     2. Actor-Critic architecture     #
########################################


def _layer(ax, x, y, w, h, text, face):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=face,
            edgecolor=INK,
            linewidth=1.3,
        ),
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        color=INK,
    )


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=14,
            color=INK,
            linewidth=1.2,
        ),
    )


def fig_architecture():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))

    nets = [
        (
            "Actor  $\\mu(s)$",
            ["state\n(10)", "FC 128\nReLU", "FC 256\nReLU", "action (2)\ntanh"],
            ["#eef3ff", "#dbe6ff", "#c6d8ff", "#9bbcff"],
        ),
        (
            "Critic  $Q(s,a)$",
            [
                "state+action\n(12)",
                "FC 128\nReLU",
                "FC 256\nReLU",
                "Q-value\n(1)",
            ],
            ["#eafaf2", "#cdeedd", "#b5e6cd", "#8fd9b6"],
        ),
    ]
    for ax, (title, labels, faces) in zip(axes, nets, strict=False):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4)
        ax.axis("off")
        ax.set_title(title)
        n = len(labels)
        w, gap = 1.9, 0.45
        total = n * w + (n - 1) * gap
        x = (10 - total) / 2
        for i, (lab, face) in enumerate(zip(labels, faces, strict=False)):
            _layer(ax, x, 1.3, w, 1.4, lab, face)
            if i < n - 1:
                _arrow(ax, x + w, 2.0, x + w + gap, 2.0)
            x += w + gap

    fig.suptitle(
        "DDPG networks: deterministic actor and action-value critic",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "architecture")


########################################
#        3. POI heuristic field        #
########################################


def fig_poi_heatmap():
    env = MobileRobotEnv(seed=13)
    env.reset()
    rover = (env.robot.x, env.robot.y)

    width = env.observation_shape[1]
    height = env.observation_shape[0]
    step = 10
    xs = np.arange(step / 2, width, step)
    ys = np.arange(step / 2, height, step)
    grid = np.full((len(ys), len(xs)), np.nan)

    def in_obstacle(px, py):
        # Shrink by one grid step so coloured cells reach under the patch
        # edges — no light halo between the heatmap and the obstacles.
        for x_max, x_min, y_max, y_min in env.obstacles_cord[4:]:
            if (
                x_min + step <= px <= x_max - step
                and y_min + step <= py <= y_max - step
            ):
                return True
        return False

    probe = POI(0, 0)
    for j, py in enumerate(ys):
        for i, px in enumerate(xs):
            if in_obstacle(px, py):
                continue
            probe.x, probe.y = int(px), int(py)
            dist_o = math.hypot(rover[0] - px, rover[1] - py)
            dist_g = math.hypot(env.target_x - px, env.target_y - py)
            h = probe.calculate_heuristic_score(
                dist_o,
                dist_g,
                5,
                20,
                probe.kernel_size,
                env.exploration_field,
            )
            grid[j, i] = h

    fig, ax = plt.subplots(figsize=(7.4, 5.6))

    # NaN cells sit under the obstacle patches: make them transparent and
    # paint the axes in the obstacle colour so no white gaps appear.
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad(alpha=0)
    ax.set_facecolor(viz.OBSTACLE_FACE)

    extent = (0, width, height, 0)
    im = ax.imshow(
        grid,
        extent=extent,
        origin="upper",
        cmap=cmap,
        aspect="equal",
    )
    obstacles = [
        {"x_max": o[0], "x_min": o[1], "y_max": o[2], "y_min": o[3]}
        for o in env.obstacles_cord[4:]
    ]
    viz.draw_obstacles(ax, obstacles)
    viz.draw_target(ax, (env.target_x, env.target_y), env.target_threshold)
    viz.draw_rover(ax, rover[0], rover[1], 0.4)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("heuristic score $h$ (lower = preferred POI)")
    ax.set_title("Exploration heuristic over candidate frontier points")
    save(fig, "poi_heatmap")


########################################
#   4. Learning curve + arrival rate   #
########################################


def _moving_avg(a, k=25):
    if len(a) < k:
        return a
    kernel = np.ones(k) / k
    return np.convolve(a, kernel, mode="valid")


def fig_learning_curve():
    rewards = np.array(json.loads((ASSETS / "rewards.json").read_text()))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))

    ax1.plot(rewards, color=ACCENT, alpha=0.25, linewidth=0.8, label="episode")
    ma = _moving_avg(rewards, 25)
    ax1.plot(
        np.arange(len(ma)) + 12,
        ma,
        color=ACCENT,
        linewidth=2.2,
        label="25-ep moving avg",
    )
    ax1.axhline(0, color=INK, linewidth=0.6, linestyle=":")
    ax1.set_xlabel("episode")
    ax1.set_ylabel("episode return")
    ax1.set_title("Training return")
    ax1.legend(frameon=False, fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    win = 100
    buckets = np.arange(0, len(rewards), win)
    rate = [(rewards[b : b + win] > 50).mean() * 100 for b in buckets]
    ax2.bar(buckets + win / 2, rate, color=GOOD, alpha=0.85, width=win * 0.8)
    ax2.set_ylabel("arrival rate (%)")
    ax2.set_xlabel("episode")
    ax2.set_title(f"Goal-arrival rate ({win}-episode windows)")
    ax2.set_ylim(0, 100)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    save(fig, "learning_curve")


########################################
#            5. Trajectory             #
########################################


def _draw_rollout(ax, t, color, label, dashed=False):
    """One rollout in its own layout, both taken from the trajectory JSON."""
    viz.draw_field(ax, t["width"], t["height"])
    viz.draw_obstacles(ax, t["obstacles"])
    viz.draw_target(ax, tuple(t["target"]), t["target_threshold"])
    if dashed:
        ax.plot(
            t["xs"],
            t["ys"],
            color=color,
            linewidth=2.0,
            alpha=0.7,
            linestyle="--",
            zorder=6,
            label=label,
        )
        ax.scatter(
            t["xs"][-1],
            t["ys"][-1],
            color=color,
            marker="x",
            s=80,
            zorder=8,
        )
    else:
        viz.draw_trajectory(ax, t["xs"], t["ys"], color=color, label=label)
    alpha0 = t["alphas"][0] if t.get("alphas") else 0.0
    viz.draw_rover(ax, t["xs"][0], t["ys"][0], alpha0)
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="lower left",
        bbox_to_anchor=(0.03, 0.04),
    )


def fig_trajectory():
    success_path = ASSETS / "trajectory_success.json"
    fail_path = ASSETS / "trajectory.json"
    panels = []
    if success_path.exists():
        t = json.loads(success_path.read_text())
        panels.append((t, GOOD, "successful rollout (arrived)", False))
    if fail_path.exists():
        t = json.loads(fail_path.read_text())
        if t.get("outcome") == "collision":
            panels.append((t, BAD, "failed rollout (collision)", True))
    if not panels:
        print("skip trajectory (no rollout assets)")
        return

    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5.6 * len(panels), 4.4),
    )
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, (t, color, label, dashed) in zip(axes, panels, strict=False):
        _draw_rollout(ax, t, color, label, dashed)
    fig.suptitle(
        "Deterministic policy on unseen random layouts",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "trajectory")


def main():
    fig_environment()
    fig_layouts()
    fig_architecture()
    fig_poi_heatmap()
    if (ASSETS / "rewards.json").exists():
        fig_learning_curve()
    fig_trajectory()


if __name__ == "__main__":
    with contextlib.redirect_stderr(open("/dev/null", "w")):
        main()
