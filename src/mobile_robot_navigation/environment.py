"""MobileRobotEnv environment for mobile-robot navigation.

Every `reset()` samples a fresh task: a random number of axis-aligned
rectangular obstacles with random sizes and positions, plus a random start
pose (position and heading). A breadth-first search over an inflated
occupancy grid guarantees each sampled layout is solvable — a collision-free
corridor from the start to the target always exists. The agent observes
seven normalised lidar ranges plus three POI-relative features and steers
with a continuous throttle / steering pair.
"""

from __future__ import annotations

import math
import numpy as np

from math import exp
from gym import Env, spaces
from collections import deque

########################################
#          Geometry primitive          #
########################################


class Point:
    """A named element on the field with an axis-aligned footprint."""

    # ── Footprint set by subclasses (used by AABB collision checks) ──
    icon_w: int
    icon_h: int

    def __init__(
        self,
        name: str,
        x_max: int,
        x_min: int,
        y_max: int,
        y_min: int,
    ) -> None:
        self.x: float = 0.0
        self.y: float = 0.0
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.name = name

    def set_position(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def get_position(self) -> tuple[float, float]:
        return (self.x, self.y)


########################################
#                Robot                 #
########################################


class Robot(Point):
    def __init__(
        self,
        name: str,
        x_max: int,
        x_min: int,
        y_max: int,
        y_min: int,
    ) -> None:
        super().__init__(name, x_max, x_min, y_max, y_min)
        self.alpha: float = 0.0
        self.icon_w = 32
        self.icon_h = 32
        self.max_linear = math.hypot(x_max - x_min, y_max - y_min)
        self.POI_field = np.zeros(
            (y_max - y_min, x_max - x_min),
            dtype=np.int8,
        )
        self.POI_area_width = 100
        self.POI_area_height = 100
        self.POI_list: list[POI] = []
        self.scan_range = [i * np.pi / 6 for i in range(-3, 4)]

    def set_position(self, x: float, y: float, alpha: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.alpha = alpha

    def reset_exploration(self) -> None:
        """Forget the previous episode's POIs (each reset is a new layout)."""
        self.POI_field.fill(0)
        self.POI_list = []

    def get_POI(
        self,
        target: Point,
        field: np.ndarray,
        current_poi: POI | None = None,
    ) -> POI | None:
        min_h = 10e10
        best_poi: POI | None = None
        for point in self.POI_list:
            dist_o = math.hypot(self.x - point.x, self.y - point.y)
            dist_g = math.hypot(target.x - point.x, target.y - point.y)
            h_i = point.calculate_heuristic_score(
                dist_o,
                dist_g,
                5,
                20,
                point.kernel_size,
                field,
            )
            if h_i < min_h and point != current_poi:
                min_h = h_i
                best_poi = point

        return best_poi

    def set_POI(self, scan: list[float], field: np.ndarray) -> None:
        height, width = field.shape

        for distance, detector_angle in zip(
            scan,
            self.scan_range,
            strict=False,
        ):
            try_dist = 0.1
            angle = detector_angle + self.alpha

            # ── Cast from the body centre, the same origin scan() measures
            #    from, so a POI never overshoots the free span into a wall ──
            origin_x = self.x + self.icon_w // 2
            origin_y = self.y + self.icon_h // 2
            while try_dist + 0.05 < distance:
                reach = self.max_linear * try_dist
                x_poi = int(origin_x + reach * math.cos(angle))
                y_poi = int(origin_y + reach * math.sin(angle))

                top = max(0, y_poi - self.POI_area_height // 2 + 1)
                bottom = min(height, y_poi + self.POI_area_height // 2 - 1)
                left = max(0, x_poi - self.POI_area_width // 2 + 1)
                right = min(width, x_poi + self.POI_area_width // 2 - 1)

                if (self.POI_field[top:bottom, left:right] == 0).all():
                    new_poi = POI(x_poi, y_poi)

                    self.POI_field[top:bottom, left:right] = 1
                    self.POI_list.append(new_poi)

                try_dist += 0.1


########################################
#          Points of interest          #
########################################


class POI:
    def __init__(self, x: int, y: int) -> None:
        self.icon_w = 98
        self.icon_h = 98
        self.x = x
        self.y = y
        self.kernel_size = 100

    def count_information(
        self,
        field: np.ndarray,
        kernel_size: int,
    ) -> float:
        left = max(0, self.x - int(kernel_size / 2))
        right = min(len(field[0]), self.x + int(kernel_size / 2))
        top = max(0, self.y - int(kernel_size / 2))
        bottom = min(len(field), self.y + int(kernel_size / 2))
        point_information = np.sum(field[top:bottom, left:right])
        count = (right - left) * (bottom - top)
        information = exp(point_information / count)
        return information

    def calculate_heuristic_score(
        self,
        dist_o: float,
        dist_g: float,
        dist_l1: float,
        dist_l2: float,
        kernel_size: int,
        field: np.ndarray,
    ) -> float:
        dist_o = dist_o / 20
        dist_g = dist_g / 20
        d1 = (
            np.tanh(
                np.exp((dist_o / dist_l1) ** 2) / exp((dist_l2 / dist_l1) ** 2),
            )
            * dist_l2
        )
        d2 = dist_g
        information = self.count_information(field, kernel_size)
        return float(d1 + d2 + information)


########################################
#         Obstacles & targets          #
########################################


class Obstacle(Point):
    def __init__(
        self,
        name: str,
        x_max: int,
        x_min: int,
        y_max: int,
        y_min: int,
    ) -> None:
        super().__init__(name, x_max, x_min, y_max, y_min)
        self.icon_w = x_max - x_min
        self.icon_h = y_max - y_min
        self.x = x_min
        self.y = y_min


class Target(Point):
    def __init__(
        self,
        name: str,
        x_max: int,
        x_min: int,
        y_max: int,
        y_min: int,
    ) -> None:
        super().__init__(name, x_max, x_min, y_max, y_min)
        self.icon_w = 16
        self.icon_h = 16


########################################
#       Layout solvability grid        #
########################################

# A rectangle in the legacy (x_max, x_min, y_max, y_min) order used across
# the environment (`obstacles_cord`, `scan`, the report scripts).
Rect = tuple[int, int, int, int]

# Occupancy-grid resolution and the clearance kept around obstacles when
# carving free-space corridors for the reachability check.
GRID_CELL = 20
SAFETY_GAP = 8
MAX_LAYOUT_TRIES = 200


def free_grid(
    obstacles: list[Rect],
    width: int,
    height: int,
    footprint: int,
) -> np.ndarray:
    """Boolean grid of cells where the robot footprint fits collision-free.

    A cell is free when a robot whose top-left corner sits at the cell
    centre cannot touch any obstacle: each rectangle is inflated by the
    robot footprint (towards the top-left, matching the AABB collision
    convention) plus a safety gap.
    """
    cols = width // GRID_CELL
    rows = height // GRID_CELL
    free = np.ones((rows, cols), dtype=bool)
    centers_x = (np.arange(cols) + 0.5) * GRID_CELL
    centers_y = (np.arange(rows) + 0.5) * GRID_CELL

    # ── Inflate every rectangle and knock out the cells it covers ──
    for x_max, x_min, y_max, y_min in obstacles:
        bad_x = (centers_x >= x_min - footprint - SAFETY_GAP) & (
            centers_x <= x_max + SAFETY_GAP
        )
        bad_y = (centers_y >= y_min - footprint - SAFETY_GAP) & (
            centers_y <= y_max + SAFETY_GAP
        )
        free[np.ix_(bad_y, bad_x)] = False

    return free


def reachable_cells(
    free: np.ndarray,
    start: tuple[int, int],
) -> np.ndarray:
    """Flood-fill (BFS) of the free grid from `start` = (row, col)."""
    rows, cols = free.shape
    seen = np.zeros_like(free)
    row0, col0 = start
    if not free[row0, col0]:
        return seen

    # ── 4-connected breadth-first search ──
    seen[row0, col0] = True
    queue = deque([(row0, col0)])
    while queue:
        row, col = queue.popleft()
        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n_row, n_col = row + d_row, col + d_col
            inside = 0 <= n_row < rows and 0 <= n_col < cols
            if inside and free[n_row, n_col] and not seen[n_row, n_col]:
                seen[n_row, n_col] = True
                queue.append((n_row, n_col))

    return seen


def _overlaps(a: Rect, b: Rect, gap: int = 20) -> bool:
    """True when rectangles `a` and `b` come closer than `gap` pixels."""
    ax_max, ax_min, ay_max, ay_min = a
    bx_max, bx_min, by_max, by_min = b
    x_apart = ax_max + gap <= bx_min or bx_max + gap <= ax_min
    y_apart = ay_max + gap <= by_min or by_max + gap <= ay_min
    return not (x_apart or y_apart)


########################################
#             Environment              #
########################################


class MobileRobotEnv(Env):
    def __init__(
        self,
        *,
        target_threshold: float = 10,
        obstacle_threshold: float = 1,
        poi_threshold: float = 5,
        min_obstacles: int = 2,
        max_obstacles: int = 5,
        min_obstacle_size: int = 60,
        max_obstacle_size: int = 220,
        min_start_distance: float = 300,
        observation_height: int = 600,
        observation_width: int = 800,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        # ── Field geometry and the true 10-d observation space ──
        self.observation_shape = (observation_height, observation_width, 3)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(10,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.y_min = 0
        self.x_min = 0
        self.y_max = observation_height
        self.x_max = observation_width

        # ── Thresholds and layout-randomization knobs ──
        self.obstacle_threshold = obstacle_threshold
        self.target_threshold = target_threshold
        self.POI_threshold = poi_threshold
        self.min_obstacles = min_obstacles
        self.max_obstacles = max_obstacles
        self.min_obstacle_size = min_obstacle_size
        self.max_obstacle_size = max_obstacle_size
        self.min_start_distance = min_start_distance
        self.rng = np.random.default_rng(seed)

        # ── Fixed goal near the bottom-right corner ──
        self.target_x = observation_width - 80
        self.target_y = observation_height - 80
        self.max_linear = math.hypot(observation_height, observation_width)

        # ── Elements; the layout itself is sampled on every reset() ──
        self.robot = Robot(
            "robot",
            self.x_max,
            self.x_min,
            self.y_max,
            self.y_min,
        )
        self.target = Target(
            "target",
            self.x_max,
            self.x_min,
            self.y_max,
            self.y_min,
        )
        self.obstacles: list[Obstacle] = []
        self.obstacles_cord: list[tuple[float, float, float, float]] = (
            self._border_rects()
        )
        self.elements: list[Point] = [self.robot]
        self.exploration_field = np.zeros(
            (observation_height, observation_width),
        )
        self.POI = POI(self.target_x, self.target_y)
        self.prev_target_dist = self.max_linear

    ########################################
    #           Layout sampling            #
    ########################################

    def _border_rects(self) -> list[tuple[float, float, float, float]]:
        """The four field walls in (x_max, x_min, y_max, y_min) order."""
        width = float(self.observation_shape[1])
        height = float(self.observation_shape[0])
        return [
            (0.0, -np.inf, np.inf, -np.inf),
            (np.inf, width, np.inf, -np.inf),
            (np.inf, -np.inf, 0.0, -np.inf),
            (np.inf, -np.inf, np.inf, height),
        ]

    def _sample_obstacles(self) -> list[Rect] | None:
        """One candidate layout, or None when placement got stuck."""
        width = self.observation_shape[1]
        height = self.observation_shape[0]
        count = int(
            self.rng.integers(self.min_obstacles, self.max_obstacles + 1),
        )

        # ── Sizes are clamped so a rectangle always fits the field ──
        w_hi = min(self.max_obstacle_size, width - 3 * GRID_CELL)
        h_hi = min(self.max_obstacle_size, height - 3 * GRID_CELL)

        rects: list[Rect] = []
        tries = 0
        while len(rects) < count and tries < 50 * count:
            tries += 1
            w = int(self.rng.integers(self.min_obstacle_size, w_hi + 1))
            h = int(self.rng.integers(self.min_obstacle_size, h_hi + 1))
            x_lo = int(self.rng.integers(GRID_CELL, width - w - GRID_CELL + 1))
            y_lo = int(
                self.rng.integers(GRID_CELL, height - h - GRID_CELL + 1),
            )
            rect: Rect = (x_lo + w, x_lo, y_lo + h, y_lo)

            # ── Keep the goal clear and the rectangles disjoint ──
            if self._covers_target(rect) or any(
                _overlaps(rect, other) for other in rects
            ):
                continue
            rects.append(rect)

        return rects if len(rects) == count else None

    def _covers_target(self, rect: Rect) -> bool:
        """True when `rect` (with margin) blocks the goal point."""
        x_max, x_min, y_max, y_min = rect
        margin = 60
        return (
            x_min - margin <= self.target_x <= x_max + margin
            and y_min - margin <= self.target_y <= y_max + margin
        )

    def _start_candidates(
        self,
        reach: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Grid cells that are valid, goal-connected start positions."""
        rows, cols = reach.shape
        out: list[tuple[int, int]] = []
        for row in range(2, rows - 2):
            for col in range(2, cols - 2):
                if not reach[row, col]:
                    continue
                x = (col + 0.5) * GRID_CELL
                y = (row + 0.5) * GRID_CELL
                far = math.hypot(x - self.target_x, y - self.target_y)
                if far >= self.min_start_distance:
                    out.append((row, col))
        return out

    def _sample_task(self) -> tuple[list[Rect], float, float, float]:
        """Sample obstacles plus a start pose with a guaranteed corridor.

        BFS runs from the target over the inflated occupancy grid, so any
        chosen start cell is connected to the goal by construction.
        """
        target_cell = (
            int(self.target_y // GRID_CELL),
            int(self.target_x // GRID_CELL),
        )

        for _ in range(MAX_LAYOUT_TRIES):
            rects = self._sample_obstacles()
            if rects is None:
                continue

            # ── Reachability: flood-fill the free grid from the goal ──
            free = free_grid(
                rects,
                self.observation_shape[1],
                self.observation_shape[0],
                self.robot.icon_w,
            )
            reach = reachable_cells(free, target_cell)
            starts = self._start_candidates(reach)
            if not starts:
                continue

            # ── Random start cell + heading among the valid ones ──
            row, col = starts[int(self.rng.integers(len(starts)))]
            x = (col + 0.5) * GRID_CELL
            y = (row + 0.5) * GRID_CELL
            alpha = float(self.rng.uniform(0, 2 * np.pi))
            return rects, x, y, alpha

        raise RuntimeError(
            "could not sample a solvable layout: relax the obstacle "
            "count/size or the start-distance constraints",
        )

    ########################################
    #            Gym lifecycle             #
    ########################################

    def reset(self) -> list[float]:  # type: ignore[override]  # legacy gym API

        # ── 1. Sample a fresh solvable task ──
        rects, x, y, alpha = self._sample_task()
        self.obstacles = [
            Obstacle(f"obs_{i + 1}", *rect) for i, rect in enumerate(rects)
        ]
        self.obstacles_cord = self._border_rects() + [
            (float(a), float(b), float(c), float(d)) for a, b, c, d in rects
        ]

        # ── 2. Rebuild the exploration memory for the new layout ──
        self.exploration_field = np.zeros(
            (self.observation_shape[0], self.observation_shape[1]),
        )
        for x_hi, x_lo, y_hi, y_lo in rects:
            self.exploration_field[y_lo:y_hi, x_lo:x_hi] = 5
        self.robot.reset_exploration()

        # ── 3. Place the robot and the target ──
        self.robot.set_position(x, y, alpha)
        self.target.set_position(self.target_x, self.target_y)
        self.elements = [self.robot, *self.obstacles]
        self.prev_target_dist = math.hypot(
            x - self.target_x,
            y - self.target_y,
        )

        # ── 4. Seed the first POIs from the initial scan ──
        scan = self.scan()
        self.exploration_field[
            int(self.robot.y) : int(self.robot.y + self.robot.icon_h),
            int(self.robot.x) : int(self.robot.x + self.robot.icon_w),
        ] = 1
        self.robot.set_POI(scan, self.exploration_field)
        self.POI = self.robot.get_POI(
            self.target,
            self.exploration_field,
        ) or POI(self.target_x, self.target_y)

        return self._observe(self.scan())

    def render(self, show_poi: bool = False) -> None:
        # `show_poi` kept for API compatibility; the vector renderer in
        # `viz` draws the rover, lidar, obstacles and target instead of the
        # old raster canvas.
        import matplotlib.pyplot as plt

        from mobile_robot_navigation import viz

        viz.render_env(self, show_lidar=True)
        plt.show()

    ########################################
    #           Sensing & checks           #
    ########################################

    def scan(self) -> list[float]:
        scan_result = []
        x_0 = self.robot.x + self.robot.icon_w // 2
        y_0 = self.robot.y + self.robot.icon_h // 2
        for detector_alpha in self.robot.scan_range:
            min_dist = self.max_linear
            scan_alpha = (self.robot.alpha + detector_alpha) % (2 * np.pi)
            for x_max, x_min, y_max, y_min in self.obstacles_cord:
                if np.pi / 2 < scan_alpha < 3 * np.pi / 2:
                    if x_0 > x_max:
                        y_intersect = math.tan(scan_alpha) * (x_max - x_0) + y_0
                        if y_min < y_intersect < y_max:
                            min_dist = min(
                                min_dist,
                                math.hypot(x_0 - x_max, y_0 - y_intersect),
                            )
                elif (
                    scan_alpha != np.pi / 2
                    and scan_alpha != 3 * np.pi / 2
                    and x_0 < x_min
                ):
                    y_intersect = math.tan(scan_alpha) * (x_min - x_0) + y_0
                    if y_min < y_intersect < y_max:
                        min_dist = min(
                            min_dist,
                            math.hypot(x_0 - x_min, y_0 - y_intersect),
                        )

                if np.pi < scan_alpha < 2 * np.pi:
                    if y_0 > y_max:
                        x_intersect = (1 / math.tan(scan_alpha)) * (
                            y_max - y_0
                        ) + x_0
                        if x_min < x_intersect < x_max:
                            min_dist = min(
                                min_dist,
                                math.hypot(x_0 - x_intersect, y_0 - y_max),
                            )
                elif (
                    scan_alpha != 0
                    and scan_alpha != np.pi
                    and scan_alpha != 2 * np.pi
                    and y_0 < y_min
                ):
                    x_intersect = (1 / math.tan(scan_alpha)) * (
                        y_min - y_0
                    ) + x_0
                    if x_min < x_intersect < x_max:
                        min_dist = min(
                            min_dist,
                            math.hypot(x_0 - x_intersect, y_0 - y_min),
                        )

            scan_result.append(min_dist / self.max_linear)

        return scan_result

    def has_collided(self, elem1: Point, elem2: Point) -> bool:
        x_col = False
        y_col = False

        elem1_x, elem1_y = elem1.get_position()
        elem2_x, elem2_y = elem2.get_position()

        if (elem1_x >= elem2_x and elem1_x - elem2_x <= elem2.icon_w) or (
            elem1_x < elem2_x and elem2_x - elem1_x <= elem1.icon_w
        ):
            x_col = True

        if (elem1_y >= elem2_y and elem1_y - elem2_y <= elem2.icon_h) or (
            elem1_y < elem2_y and elem2_y - elem1_y <= elem1.icon_h
        ):
            y_col = True

        return bool(x_col and y_col)

    def out_of_boundary(self, elem: Point) -> bool:

        elem_x, elem_y = elem.get_position()

        return (
            elem_x - self.x_min < self.obstacle_threshold
            or self.x_max - elem_x - self.robot.icon_w < self.obstacle_threshold
            or elem_y - self.y_min < self.obstacle_threshold
            or self.y_max - elem_y - self.robot.icon_h < self.obstacle_threshold
        )

    ########################################
    #             Observation              #
    ########################################

    def _poi_features(self) -> tuple[float, float, float]:
        """Distance, bearing and signed heading error to the active POI.

        The heading error is signed and wrapped to [-pi, pi): its sign tells
        the policy which way to steer, which an absolute-value error cannot.
        """
        poi = self.POI
        dist_to_poi = math.hypot(
            self.robot.x - poi.x,
            self.robot.y - poi.y,
        )
        rel_theta = math.atan2(
            poi.y - self.robot.y,
            poi.x - self.robot.x,
        ) % (2 * np.pi)
        diff_angle = (rel_theta - self.robot.alpha + np.pi) % (
            2 * np.pi
        ) - np.pi
        return dist_to_poi, rel_theta, diff_angle

    def _observe(self, scans: list[float]) -> list[float]:
        """Assemble the 10-d observation: 7 scans + 3 POI features."""
        dist_to_poi, rel_theta, diff_angle = self._poi_features()
        return (
            scans
            + [rel_theta / (2 * np.pi), (diff_angle + np.pi) / (2 * np.pi)]
            + [dist_to_poi / self.max_linear]
        )

    ########################################
    #                 Step                 #
    ########################################

    def step(  # type: ignore[override]  # legacy gym <0.26 API
        self,
        action: np.ndarray,
    ) -> tuple[list[float], float, int, int]:
        done = 0
        arrived = 0

        # ── 1. Advance the robot with the chosen action ──
        self.robot.alpha += action[1]
        self.robot.alpha = self.robot.alpha % (2 * np.pi)
        self.robot.x += 10 * ((action[0] + 1) / 2) * np.cos(self.robot.alpha)
        self.robot.y += 10 * ((action[0] + 1) / 2) * np.sin(self.robot.alpha)

        # ── 2. Reward: arrival bonus, or shaping plus goal progress ──
        dist_to_target = math.hypot(
            self.robot.x - self.target.x,
            self.robot.y - self.target.y,
        )
        if dist_to_target < self.target_threshold:
            reward = 101
            arrived = 1
        else:
            # Potential-based progress term: dense credit for closing the
            # distance to the goal (preserves the optimal policy).
            progress = self.prev_target_dist - dist_to_target
            reward = action[0] - abs(action[1]) + 0.1 * progress
        reward -= 1
        self.prev_target_dist = dist_to_target

        # ── 3. Terminal checks: leaving the field or hitting a block ──
        if self.out_of_boundary(self.robot):
            done = 1
            reward = -50
        for obstacle in self.obstacles:
            if self.has_collided(self.robot, obstacle):
                done = 1
                reward = -50

        # ── 4. Expand the explored field and seed new POIs ──
        robot_patch = self.exploration_field[
            int(self.robot.y) : int(self.robot.y + self.robot.icon_h),
            int(self.robot.x) : int(self.robot.x + self.robot.icon_w),
        ]
        if not done and (robot_patch == 0).all():
            scan = self.scan()
            self.exploration_field[
                int(self.robot.y) : int(self.robot.y + self.robot.icon_h),
                int(self.robot.x) : int(self.robot.x + self.robot.icon_w),
            ] = 1

            self.robot.set_POI(scan, self.exploration_field)

        # ── 5. Switch to a fresh POI when the current one is reached ──
        dist_to_poi, _, _ = self._poi_features()
        if dist_to_poi < self.POI_threshold:
            self.POI = self.robot.get_POI(
                self.target,
                self.exploration_field,
                self.POI,
            ) or POI(self.target_x, self.target_y)

        # ── 6. Aim straight at the target once it is close ──
        if dist_to_target < 64:
            self.POI = POI(int(self.target.x), int(self.target.y))

        return self._observe(self.scan()), reward, done, arrived
