"""Top-down vector visualization of the MobileRobotEnv navigation scene.

Draws the field, obstacles, target and a stylized mobile rover (heading and
lidar rays) as clean matplotlib vector graphics. Used by both
`MobileRobotEnv.render` and the report figure scripts. Matplotlib is imported at
module load, so this module is only pulled in when something actually renders --
the training path (`agent`, `lib`, `run`) never imports it.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from typing import TYPE_CHECKING
from matplotlib.axes import Axes
from matplotlib.transforms import Affine2D
from matplotlib.patches import Circle, Polygon, FancyBboxPatch

if TYPE_CHECKING:
    from mobile_robot_navigation.environment import MobileRobotEnv

########################################
#               Palette                #
########################################

# Flat, muted palette so the scene reads as a schematic, not a screenshot.
FIELD_FACE = "#f4f6fb"
FIELD_EDGE = "#c2c9d6"
OBSTACLE_FACE = "#9aa7bd"
OBSTACLE_EDGE = "#6b7896"
TARGET_COLOR = "#1f9d6b"
ROVER_BODY = "#2f6df0"
ROVER_CABIN = "#cfe0ff"
ROVER_WHEEL = "#27324a"
LIDAR_COLOR = "#f0922f"
TRAIL_COLOR = "#2f6df0"


########################################
#             Static scene             #
########################################


def draw_field(
    ax: Axes,
    width: int,
    height: int,
) -> None:
    """Paint the bounded field and fix the image-style coordinate frame."""

    # ── Background panel ──────────────────────
    ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            width,
            height,
            boxstyle="round,pad=0,rounding_size=12",
            facecolor=FIELD_FACE,
            edgecolor=FIELD_EDGE,
            linewidth=1.5,
        ),
    )

    # ── Image-style frame: origin top-left ────
    ax.set_xlim(-20, width + 20)
    ax.set_ylim(-20, height + 20)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")


def draw_obstacles(
    ax: Axes,
    obstacles: list[dict[str, float]],
) -> None:
    """Draw each rectangular obstacle as a soft rounded block."""

    for obs in obstacles:
        x_min = obs["x_min"]
        y_min = obs["y_min"]
        w = obs["x_max"] - x_min
        h = obs["y_max"] - y_min
        ax.add_patch(
            FancyBboxPatch(
                (x_min, y_min),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=10",
                facecolor=OBSTACLE_FACE,
                edgecolor=OBSTACLE_EDGE,
                linewidth=1.5,
                alpha=0.95,
            ),
        )


def draw_target(
    ax: Axes,
    target: tuple[float, float],
    threshold: float,
) -> None:
    """Draw the goal: a tolerance halo plus a small flag."""

    tx, ty = target

    # ── Arrival-tolerance halo ────────────────
    ax.add_patch(
        Circle(
            (tx, ty),
            max(threshold, 14.0),
            facecolor=TARGET_COLOR,
            edgecolor="none",
            alpha=0.18,
        ),
    )
    ax.add_patch(
        Circle(
            (tx, ty),
            5.0,
            facecolor=TARGET_COLOR,
            edgecolor="white",
            linewidth=1.2,
            zorder=5,
        ),
    )

    # ── Flag pole + pennant ───────────────────
    ax.plot(
        [tx, tx],
        [ty, ty - 34],
        color=TARGET_COLOR,
        linewidth=2.0,
        zorder=5,
    )
    ax.add_patch(
        Polygon(
            [(tx, ty - 34), (tx + 22, ty - 28), (tx, ty - 22)],
            facecolor=TARGET_COLOR,
            edgecolor="none",
            zorder=5,
        ),
    )


########################################
#                Rover                 #
########################################


def draw_rover(
    ax: Axes,
    x: float,
    y: float,
    alpha: float,
    scale: float = 1.0,
) -> None:
    """Draw a top-down rover at (x, y) pointing along heading `alpha`."""

    rot = Affine2D().rotate_around(x, y, alpha) + ax.transData

    # ── 1. Wheels (drawn first, under the body) ──
    half_l = 17 * scale
    half_w = 11 * scale
    wheel_l = 9 * scale
    wheel_w = 4 * scale
    corners = [
        (x - half_l + 3, y - half_w - 1),
        (x - half_l + 3, y + half_w - wheel_w + 1),
        (x + half_l - wheel_l - 3, y - half_w - 1),
        (x + half_l - wheel_l - 3, y + half_w - wheel_w + 1),
    ]
    for wx, wy in corners:
        ax.add_patch(
            FancyBboxPatch(
                (wx, wy),
                wheel_l,
                wheel_w,
                boxstyle="round,pad=0,rounding_size=2",
                facecolor=ROVER_WHEEL,
                edgecolor="none",
                transform=rot,
                zorder=8,
            ),
        )

    # ── 2. Chassis ────────────────────────────
    ax.add_patch(
        FancyBboxPatch(
            (x - half_l, y - half_w),
            2 * half_l,
            2 * half_w,
            boxstyle="round,pad=0,rounding_size=6",
            facecolor=ROVER_BODY,
            edgecolor="white",
            linewidth=1.4,
            transform=rot,
            zorder=9,
        ),
    )

    # ── 3. Cabin / sensor window ──────────────
    ax.add_patch(
        FancyBboxPatch(
            (x - 4 * scale, y - 7 * scale),
            10 * scale,
            14 * scale,
            boxstyle="round,pad=0,rounding_size=4",
            facecolor=ROVER_CABIN,
            edgecolor="none",
            transform=rot,
            zorder=10,
        ),
    )

    # ── 4. Heading beak ───────────────────────
    nose = [
        (x + half_l, y),
        (x + half_l - 8 * scale, y - 7 * scale),
        (x + half_l - 8 * scale, y + 7 * scale),
    ]
    ax.add_patch(
        Polygon(
            nose,
            closed=True,
            facecolor=ROVER_BODY,
            edgecolor="white",
            linewidth=1.2,
            transform=rot,
            zorder=9,
        ),
    )


def draw_lidar(
    ax: Axes,
    x: float,
    y: float,
    alpha: float,
    scan_range: list[float],
    distances: list[float],
    max_linear: float,
) -> None:
    """Draw lidar rays from the rover, scaled by normalized hit distances."""

    for detector_alpha, dist in zip(scan_range, distances, strict=False):
        ray_alpha = alpha + detector_alpha
        reach = dist * max_linear
        ex = x + reach * np.cos(ray_alpha)
        ey = y + reach * np.sin(ray_alpha)
        ax.plot(
            [x, ex],
            [y, ey],
            color=LIDAR_COLOR,
            linewidth=1.0,
            alpha=0.55,
            zorder=6,
        )
        ax.add_patch(
            Circle(
                (ex, ey),
                3.0,
                facecolor=LIDAR_COLOR,
                edgecolor="none",
                alpha=0.8,
                zorder=6,
            ),
        )


def draw_trajectory(
    ax: Axes,
    xs: list[float],
    ys: list[float],
    color: str = TRAIL_COLOR,
    label: str | None = None,
) -> None:
    """Overlay a travelled path with a faded start marker."""

    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=2.4,
        alpha=0.9,
        solid_capstyle="round",
        zorder=7,
        label=label,
    )
    ax.add_patch(
        Circle(
            (xs[0], ys[0]),
            7.0,
            facecolor="white",
            edgecolor=color,
            linewidth=2.0,
            zorder=7,
        ),
    )


########################################
#             Convenience              #
########################################


def render_env(
    env: MobileRobotEnv,
    ax: Axes | None = None,
    show_lidar: bool = True,
) -> Axes:
    """Render the live state of a `MobileRobotEnv` instance.

    Pulls geometry straight off the environment, so it always matches the
    current robot pose, obstacles and target.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    width = int(env.observation_shape[1])
    height = int(env.observation_shape[0])
    obstacles = [
        {
            "x_max": o[0],
            "x_min": o[1],
            "y_max": o[2],
            "y_min": o[3],
        }
        for o in env.obstacles_cord[4:]
    ]

    draw_field(ax, width, height)
    draw_obstacles(ax, obstacles)
    draw_target(ax, (env.target_x, env.target_y), env.target_threshold)
    if show_lidar:
        draw_lidar(
            ax,
            env.robot.x,
            env.robot.y,
            env.robot.alpha,
            env.robot.scan_range,
            env.scan(),
            env.max_linear,
        )
    draw_rover(
        ax,
        env.robot.x,
        env.robot.y,
        env.robot.alpha,
    )
    return ax
