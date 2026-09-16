import numpy as np


class DroneController:
    """
    Local onboard controller for one drone.

    Responsibilities:
        - read this drone's local consensus grid;
        - choose exploration or boundary tracking;
        - compute this drone's motion command.
    """

    def __init__(
        self,
        sim_map,
        communication_radius,
        fully_connected=False,
        occupancy_threshold=0.5,
        resolution=0.1,
    ):
        self.sim_map = sim_map

        self.communication_radius = float(
            communication_radius
        )

        self.fully_connected = bool(
            fully_connected
        )

        self.occupancy_threshold = float(
            occupancy_threshold
        )

        self.resolution = float(
            resolution
        )

        # Motion parameters.
        self.max_speed = 0.12
        self.exploration_speed = 0.08

        self.k_t = 1.0
        self.k_n = 1.5

        self.boundary_lock_gain = 3.0

    # ==================================================================
    # VECTOR UTILITIES
    # ==================================================================

    @staticmethod
    def _normalize(vector):
        vector = np.asarray(
            vector,
            dtype=float,
        )

        norm = float(
            np.linalg.norm(vector)
        )

        if norm <= 1e-12:
            return None

        return vector / norm

    def _clip_action(
        self,
        action,
        max_speed=None,
    ):
        if max_speed is None:
            max_speed = self.max_speed

        action = np.asarray(
            action,
            dtype=float,
        )

        speed = float(
            np.linalg.norm(action)
        )

        if speed <= 1e-12:
            return np.zeros(
                2,
                dtype=float,
            )

        if speed > max_speed:
            action = (
                action
                * max_speed
                / speed
            )

        return action

    @staticmethod
    def _random_direction():
        angle = np.random.uniform(
            0.0,
            2.0 * np.pi,
        )

        return np.array(
            [
                np.cos(angle),
                np.sin(angle),
            ],
            dtype=float,
        )

    # ==================================================================
    # FIELD INTERPOLATION
    # ==================================================================

    def _interpolate_field(
        self,
        world_field,
        position,
        x_coords,
        y_coords,
    ):
        """Bilinearly interpolate the world field."""

        position = np.asarray(
            position,
            dtype=float,
        )

        x = float(
            np.clip(
                position[0],
                x_coords[0],
                x_coords[-1],
            )
        )

        y = float(
            np.clip(
                position[1],
                y_coords[0],
                y_coords[-1],
            )
        )

        i1 = int(
            np.searchsorted(
                x_coords,
                x,
                side="right",
            )
        )

        j1 = int(
            np.searchsorted(
                y_coords,
                y,
                side="right",
            )
        )

        i0 = max(
            0,
            min(
                i1 - 1,
                len(x_coords) - 1,
            ),
        )

        j0 = max(
            0,
            min(
                j1 - 1,
                len(y_coords) - 1,
            ),
        )

        i1 = max(
            0,
            min(
                i1,
                len(x_coords) - 1,
            ),
        )

        j1 = max(
            0,
            min(
                j1,
                len(y_coords) - 1,
            ),
        )

        if i0 == i1:
            wx = 0.0
        else:
            wx = (
                x - x_coords[i0]
            ) / (
                x_coords[i1]
                - x_coords[i0]
            )

        if j0 == j1:
            wy = 0.0
        else:
            wy = (
                y - y_coords[j0]
            ) / (
                y_coords[j1]
                - y_coords[j0]
            )

        q00 = world_field[i0, j0]
        q10 = world_field[i1, j0]
        q01 = world_field[i0, j1]
        q11 = world_field[i1, j1]

        return float(
            (1 - wx) * (1 - wy) * q00
            + wx * (1 - wy) * q10
            + (1 - wx) * wy * q01
            + wx * wy * q11
        )

    def _gradient(
        self,
        world_field,
        position,
        x_coords,
        y_coords,
    ):
        """Estimate the local field gradient."""

        dx = max(
            abs(
                float(
                    x_coords[1]
                    - x_coords[0]
                )
            ),
            self.resolution,
        )

        dy = max(
            abs(
                float(
                    y_coords[1]
                    - y_coords[0]
                )
            ),
            self.resolution,
        )

        position = np.asarray(
            position,
            dtype=float,
        )

        x_plus = self._interpolate_field(
            world_field,
            position + [dx, 0.0],
            x_coords,
            y_coords,
        )

        x_minus = self._interpolate_field(
            world_field,
            position - [dx, 0.0],
            x_coords,
            y_coords,
        )

        y_plus = self._interpolate_field(
            world_field,
            position + [0.0, dy],
            x_coords,
            y_coords,
        )

        y_minus = self._interpolate_field(
            world_field,
            position - [0.0, dy],
            x_coords,
            y_coords,
        )

        return np.array(
            [
                (x_plus - x_minus)
                / (2.0 * dx),

                (y_plus - y_minus)
                / (2.0 * dy),
            ],
            dtype=float,
        )

    # ==================================================================
    # MOTION
    # ==================================================================

    def _exploration_action(self, drone):
        """Random exploration with boundary bouncing."""

        direction = getattr(
            drone,
            "exploration_direction",
            None,
        )

        if direction is None:
            direction = self._random_direction()

        direction = self._normalize(
            direction
        )

        if direction is None:
            direction = self._random_direction()

        next_x = (
            drone.x
            + direction[0]
            * self.exploration_speed
        )

        next_y = (
            drone.y
            + direction[1]
            * self.exploration_speed
        )

        if (
            next_x < self.sim_map.xlim[0]
            or next_x > self.sim_map.xlim[1]
        ):
            direction[0] *= -1.0

        if (
            next_y < self.sim_map.ylim[0]
            or next_y > self.sim_map.ylim[1]
        ):
            direction[1] *= -1.0

        norm_dir = self._normalize(direction)
        if norm_dir is None:
            norm_dir = self._random_direction()
        drone.exploration_direction = norm_dir

        return (
            drone.exploration_direction
            * self.exploration_speed
        )

    def _boundary_tracking_action(
        self,
        drone,
        world_field,
        x_coords,
        y_coords,
    ):
        """Follow the concentration contour locally."""

        position = np.array(
            [drone.x, drone.y],
            dtype=float,
        )

        concentration = self._interpolate_field(
            world_field,
            position,
            x_coords,
            y_coords,
        )

        gradient = self._gradient(
            world_field,
            position,
            x_coords,
            y_coords,
        )

        normal = self._normalize(
            gradient
        )

        if normal is None:
            return None

        tangent = np.array(
            [
                -normal[1],
                normal[0],
            ],
            dtype=float,
        )

        tangent = self._normalize(
            tangent
        )

        if tangent is None:
            return None

        error = (
            concentration
            - self.occupancy_threshold
        )

        error = float(
            np.clip(
                error,
                -1.0,
                1.0,
            )
        )

        normal_gain = (
            self.k_n * error
            + self.boundary_lock_gain * error
        )

        action = (
            self.k_t * tangent
            - normal_gain * normal
        )

        return self._clip_action(
            action
        )

    def _grid_target(self, drone):
        """Return the nearest occupied cell in the consensus grid in world coordinates."""
        grid = np.asarray(
            getattr(drone, "grid", np.zeros((1, 1), dtype=float)),
            dtype=float,
        )

        if grid.size == 0:
            return None

        occupied = np.argwhere(grid > self.occupancy_threshold)
        if occupied.size == 0:
            return None

        grid_bounds = getattr(drone, "grid_bounds", None)
        if grid_bounds is None:
            return None

        x_min, x_max, y_min, y_max = grid_bounds
        if x_max <= x_min or y_max <= y_min:
            return None

        occupied_points = []
        for ix, iy in occupied:
            x = x_min + (ix + 0.5) * self.resolution
            y = y_min + (iy + 0.5) * self.resolution
            occupied_points.append(np.array([x, y], dtype=float))

        if not occupied_points:
            return None

        occupied_points = np.asarray(occupied_points, dtype=float)
        deltas = occupied_points - np.asarray([drone.x, drone.y], dtype=float)
        distances = np.linalg.norm(deltas, axis=1)
        best_idx = int(np.argmin(distances))

        return occupied_points[best_idx]

    def compute_action(
        self,
        drone,
        world_field,
        x_coords,
        y_coords,
    ):
        """
        Compute this drone's local action.

        The decision is driven by the local consensus occupancy grid: if the
        consensus map contains occupied boundary cells, the drone stays in
        boundary_tracking and heads toward the closest occupied target. This
        applies even when the local sensor flag is false but the map received
        from neighbors contains the boundary information.

        Returns
        -------
        np.ndarray
            [vx, vy] command.
        """

        target = self._grid_target(drone)
        should_track_boundary = target is not None

        if should_track_boundary:
            drone.last_control_mode = "boundary_tracking"

            action = self._boundary_tracking_action(
                drone,
                world_field,
                x_coords,
                y_coords,
            )

            if action is None:
                direction = self._normalize(
                    np.asarray(target, dtype=float)
                    - np.array([drone.x, drone.y], dtype=float)
                )
                if direction is not None:
                    action = direction * self.exploration_speed
                else:
                    action = self._exploration_action(drone)
                    drone.last_control_mode = "explore"

        else:
            action = self._exploration_action(drone)
            drone.last_control_mode = "explore"

        return self._clip_action(action)
