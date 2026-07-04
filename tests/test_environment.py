"""Randomized-layout invariants for MobileRobotEnv (CPU, stage 1).

Every `reset()` samples a fresh task, so these tests pin the sampling
contract: obstacle count/size stay within the configured bounds, everything
lands inside the field, the start pose is clear of obstacles and far enough
from the goal, and an independently recomputed BFS confirms a collision-free
corridor from the start to the target exists.
"""

from __future__ import annotations

import math
import pytest

import numpy as np

from mobile_robot_navigation.environment import (
    POI,
    GRID_CELL,
    free_grid,
    MobileRobotEnv,
    reachable_cells,
)

WIDTH = 800
HEIGHT = 600


def _layout_rects(env: MobileRobotEnv) -> list[tuple[int, int, int, int]]:
    """The current episode's obstacle rectangles as int tuples."""
    return [
        (int(a), int(b), int(c), int(d))
        for a, b, c, d in env.obstacles_cord[4:]
    ]


########################################
#           Layout sampling            #
########################################


def test_obstacle_count_and_size_within_bounds() -> None:
    env = MobileRobotEnv(seed=0)
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
    env = MobileRobotEnv(seed=1)
    env.reset()
    first = _layout_rects(env)
    env.reset()
    second = _layout_rects(env)

    assert first != second


def test_start_pose_is_clear_and_far_from_goal() -> None:
    env = MobileRobotEnv(seed=2)
    for _ in range(10):
        env.reset()

        # ── No obstacle touches the spawn footprint ──
        for obstacle in env.obstacles:
            assert not env.has_collided(env.robot, obstacle)

        # ── Inside the field, far enough from the target ──
        assert not env.out_of_boundary(env.robot)
        start_dist = np.hypot(
            env.robot.x - env.target_x,
            env.robot.y - env.target_y,
        )
        assert start_dist >= env.min_start_distance


def test_every_layout_is_solvable() -> None:
    env = MobileRobotEnv(seed=3)
    for _ in range(10):
        env.reset()

        # ── Recompute the corridor check from the public layout ──
        free = free_grid(
            _layout_rects(env),
            WIDTH,
            HEIGHT,
            env.robot.icon_w,
        )
        reach = reachable_cells(
            free,
            (env.target_y // GRID_CELL, env.target_x // GRID_CELL),
        )
        start_cell = (
            int(env.robot.y // GRID_CELL),
            int(env.robot.x // GRID_CELL),
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
    env = MobileRobotEnv(seed=4)
    env.reset()
    env.robot.set_position(env.target_x - 5, env.target_y, 0.0)

    _obs, reward, done, arrived = env.step(np.array([-1.0, 0.0]))

    assert arrived == 1
    assert done == 0
    assert reward == 100


def test_collision_terminates_with_penalty() -> None:
    env = MobileRobotEnv(seed=5)
    env.reset()
    first = env.obstacles[0]
    env.robot.set_position(first.x + 1.0, first.y + 1.0, 0.0)

    _obs, reward, done, arrived = env.step(np.array([-1.0, 0.0]))

    assert done == 1
    assert arrived == 0
    assert reward == -50


def test_reset_clears_exploration_memory() -> None:
    env = MobileRobotEnv(seed=6)
    env.reset()
    assert env.robot.POI_list

    # ── Explicit clear empties the POI memory ──
    env.robot.reset_exploration()
    assert env.robot.POI_list == []
    assert not env.robot.POI_field.any()

    # ── A fresh reset marks exactly the new layout's obstacles with 5 ──
    env.reset()
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for x_max, x_min, y_max, y_min in _layout_rects(env):
        mask[y_min:y_max, x_min:x_max] = True
    assert ((env.exploration_field == 5) == mask).all()


def test_poi_fallback_targets_the_goal() -> None:
    env = MobileRobotEnv(seed=7)
    env.reset()

    # ── With no candidate POIs, the goal itself becomes the POI ──
    env.robot.reset_exploration()
    poi = env.robot.get_POI(env.target, env.exploration_field)
    assert poi is None

    fallback = poi or POI(env.target_x, env.target_y)
    assert (fallback.x, fallback.y) == (env.target_x, env.target_y)


########################################
#           Reward & sensing           #
########################################


def test_reward_is_shaping_off_a_terminal() -> None:
    """Off a terminal, reward = throttle - |steer| + 0.1*progress - 1."""
    env = MobileRobotEnv(seed=8)
    env.reset()

    # ── Clear the layout so the step is neither a collision nor OOB ──
    env.obstacles = []
    env.obstacles_cord = env._border_rects()
    env.robot.set_position(400.0, 300.0, 0.0)
    prev = math.hypot(400.0 - env.target_x, 300.0 - env.target_y)
    env.prev_target_dist = prev

    action = np.array([0.4, -0.2])
    _obs, reward, done, arrived = env.step(action)

    # ── Recompute the expected reward from the same kinematics ──
    alpha = action[1] % (2 * np.pi)
    speed = 10 * ((action[0] + 1) / 2)
    new_x = 400.0 + speed * np.cos(alpha)
    new_y = 300.0 + speed * np.sin(alpha)
    new_dist = math.hypot(new_x - env.target_x, new_y - env.target_y)
    expected = action[0] - abs(action[1]) + 0.1 * (prev - new_dist) - 1

    assert done == 0
    assert arrived == 0
    assert reward == pytest.approx(expected)


def test_scan_measures_distance_to_obstacle_ahead() -> None:
    """The forward beam returns the normalized range to an obstacle ahead."""
    env = MobileRobotEnv(seed=9)
    env.reset()

    # ── One obstacle straight ahead of a centred, east-facing rover ──
    env.robot.set_position(100.0, 100.0, 0.0)
    obstacle = (400.0, 300.0, 200.0, 50.0)  # (x_max, x_min, y_max, y_min)
    env.obstacles_cord = env._border_rects() + [obstacle]

    scans = env.scan()

    # ── Beam 3 (detector angle 0) casts from the body centre and hits the
    #    obstacle's near face at x_min = 300; beam 6 (+90°) clears it and
    #    reads the far bottom wall instead ──
    x_0 = 100.0 + env.robot.icon_w // 2
    assert scans[3] == pytest.approx((300.0 - x_0) / env.max_linear)
    assert scans[6] == pytest.approx((HEIGHT - x_0) / env.max_linear)


def test_out_of_boundary_true_past_each_wall() -> None:
    """The centre is inside; a pose hard against any wall is out of bounds."""
    env = MobileRobotEnv(seed=10)
    env.reset()

    env.robot.set_position(400.0, 300.0, 0.0)
    assert not env.out_of_boundary(env.robot)

    for x, y in ((0.0, 300.0), (WIDTH, 300.0), (400.0, 0.0), (400.0, HEIGHT)):
        env.robot.set_position(float(x), float(y), 0.0)
        assert env.out_of_boundary(env.robot)


def test_step_out_of_bounds_terminates_with_penalty() -> None:
    """Driving off the field ends the episode with the -50 penalty."""
    env = MobileRobotEnv(seed=11)
    env.reset()
    env.obstacles = []
    env.obstacles_cord = env._border_rects()

    # ── Just inside the left wall, heading west, full throttle ──
    env.robot.set_position(2.0, 300.0, np.pi)
    _obs, reward, done, arrived = env.step(np.array([1.0, 0.0]))

    assert done == 1
    assert arrived == 0
    assert reward == -50
