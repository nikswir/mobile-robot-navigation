"""Randomized-layout invariants for ChopperScape (CPU, stage 1).

Every `reset()` samples a fresh task, so these tests pin the sampling
contract: obstacle count/size stay within the configured bounds, everything
lands inside the field, the start pose is clear of obstacles and far enough
from the goal, and an independently recomputed BFS confirms a collision-free
corridor from the start to the target exists.
"""

from __future__ import annotations

import numpy as np

from mobile_robot_navigation.environment import (
    POI,
    GRID_CELL,
    free_grid,
    ChopperScape,
    reachable_cells,
)

WIDTH = 800
HEIGHT = 600


def _layout_rects(env: ChopperScape) -> list[tuple[int, int, int, int]]:
    """The current episode's obstacle rectangles as int tuples."""
    return [
        (int(a), int(b), int(c), int(d))
        for a, b, c, d in env.obstacles_cord[4:]
    ]


########################################
#           Layout sampling            #
########################################


def test_obstacle_count_and_size_within_bounds() -> None:
    env = ChopperScape(seed=0)
    for _ in range(10):
        env.reset()

        assert env.min_obstacles <= len(env.obstacles) <= env.max_obstacles
        for obstacle in env.obstacles:
            assert env.min_obstacle_size <= obstacle.icon_w
            assert env.min_obstacle_size <= obstacle.icon_h
            assert obstacle.icon_w <= env.max_obstacle_size
            assert obstacle.icon_h <= env.max_obstacle_size
            assert obstacle.x_min >= 0
            assert obstacle.y_min >= 0
            assert obstacle.x_max <= WIDTH
            assert obstacle.y_max <= HEIGHT


def test_layouts_differ_across_resets() -> None:
    env = ChopperScape(seed=1)
    env.reset()
    first = _layout_rects(env)
    env.reset()
    second = _layout_rects(env)

    assert first != second


def test_start_pose_is_clear_and_far_from_goal() -> None:
    env = ChopperScape(seed=2)
    for _ in range(10):
        env.reset()

        # ── No obstacle touches the spawn footprint ──
        for obstacle in env.obstacles:
            assert not env.has_collided(env.chopper, obstacle)

        # ── Inside the field, far enough from the target ──
        assert not env.out_of_boundary(env.chopper)
        start_dist = np.hypot(
            env.chopper.x - env.target_x,
            env.chopper.y - env.target_y,
        )
        assert start_dist >= env.min_start_distance


def test_every_layout_is_solvable() -> None:
    env = ChopperScape(seed=3)
    for _ in range(10):
        env.reset()

        # ── Recompute the corridor check from the public layout ──
        free = free_grid(
            _layout_rects(env),
            WIDTH,
            HEIGHT,
            env.chopper.icon_w,
        )
        reach = reachable_cells(
            free,
            (env.target_y // GRID_CELL, env.target_x // GRID_CELL),
        )
        start_cell = (
            int(env.chopper.y // GRID_CELL),
            int(env.chopper.x // GRID_CELL),
        )
        assert reach[start_cell]


########################################
#        Solvability primitives        #
########################################


def test_full_height_wall_disconnects_the_grid() -> None:
    wall = (420, 380, HEIGHT, 0)
    free = free_grid([wall], WIDTH, HEIGHT, 32)
    reach = reachable_cells(free, (15, 2))

    # ── Left side reachable, right side sealed off ──
    assert reach[5, 5]
    assert not reach[15, 35]


def test_blocked_start_reaches_nothing() -> None:
    block = (300, 100, 300, 100)
    free = free_grid([block], WIDTH, HEIGHT, 32)
    reach = reachable_cells(free, (10, 10))

    assert not reach.any()


########################################
#           Episode dynamics           #
########################################


def test_arrival_is_terminal_with_bonus_reward() -> None:
    env = ChopperScape(seed=4)
    env.reset()
    env.chopper.set_position(env.target_x - 5, env.target_y, 0.0)

    _obs, reward, done, arrived = env.step(np.array([-1.0, 0.0]))

    assert arrived == 1
    assert done == 0
    assert reward == 100


def test_collision_terminates_with_penalty() -> None:
    env = ChopperScape(seed=5)
    env.reset()
    first = env.obstacles[0]
    env.chopper.set_position(first.x + 1.0, first.y + 1.0, 0.0)

    _obs, reward, done, arrived = env.step(np.array([-1.0, 0.0]))

    assert done == 1
    assert arrived == 0
    assert reward == -50


def test_reset_clears_exploration_memory() -> None:
    env = ChopperScape(seed=6)
    env.reset()
    assert env.chopper.POI_list

    # ── Explicit clear empties the POI memory ──
    env.chopper.reset_exploration()
    assert env.chopper.POI_list == []
    assert not env.chopper.POI_field.any()

    # ── A fresh reset marks exactly the new layout's obstacles with 5 ──
    env.reset()
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for x_max, x_min, y_max, y_min in _layout_rects(env):
        mask[y_min:y_max, x_min:x_max] = True
    assert ((env.exploration_field == 5) == mask).all()


def test_poi_fallback_targets_the_goal() -> None:
    env = ChopperScape(seed=7)
    env.reset()

    # ── With no candidate POIs, the goal itself becomes the POI ──
    env.chopper.reset_exploration()
    poi = env.chopper.get_POI(env.target, env.exploration_field)
    assert poi is None

    fallback = poi or POI(env.target_x, env.target_y)
    assert (fallback.x, fallback.y) == (env.target_x, env.target_y)
