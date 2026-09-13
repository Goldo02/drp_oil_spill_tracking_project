import numpy as np


class Controller:
    """
    Distributed controller for multi-drone exploration, oil-spill boundary tracking,
    and decentralized equidistant spacing along closed boundaries.

    Responsibilities:
        - determine communication neighbours;
        - perform distributed consensus (grids and drone positions);
        - detect boundary closure via DFS flood-fill;
        - determine local 1D ring ordering relative to boundary center of mass;
        - compute dynamic tangential spacing gain ($k_{t,i}$);
        - generate motion actions for each drone across operational states
          ("explore", "boundary_tracking", "equi_distant").
    """

    def __init__(
        self,
        sim_map,
        communication_radius,
        fully_connected=False,
        occupancy_threshold=0.5,
        resolution=0.1,
        k_spacing=1.5,
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
        self.k_spacing = float(k_spacing)

        self.boundary_lock_gain = 3.0

    # ==================================================================
    # COMMUNICATION
    # ==================================================================

    def get_neighbors(self, drone, drones):
        """Return drones that can communicate with `drone`."""

        if self.fully_connected:
            return [
                other
                for other in drones
                if other is not drone
            ]

        neighbours = []

        for other in drones:

            if other is drone:
                continue

            distance = float(
                np.hypot(
                    drone.x - other.x,
                    drone.y - other.y,
                )
            )

            if distance <= self.communication_radius:
                neighbours.append(other)

        return neighbours

    # ==================================================================
    # CONSENSUS & MULTI-HOP SHARING
    # ==================================================================

    def consensus_step(self, drones):
        """
        Perform one synchronous consensus iteration.

        Each drone replaces its grid with the average of its own grid
        and the grids of its current communication neighbours.
        Simultaneously, drones share and merge known positions of all drones.
        """

        # Ensure each drone's own entry in its known_positions is current
        for drone in drones:
            if not hasattr(drone, "known_positions") or not isinstance(
                drone.known_positions, dict
            ):
                drone.known_positions = {}
            drone.known_positions[drone.drone_id] = np.array(
                [drone.x, drone.y], dtype=float
            )

        grid_snapshot = {
            drone.drone_id: np.asarray(
                drone.grid,
                dtype=float,
            ).copy()
            for drone in drones
        }

        pos_snapshot = {
            drone.drone_id: {
                k: np.asarray(v, dtype=float).copy()
                for k, v in drone.known_positions.items()
            }
            for drone in drones
        }

        updated_grids = {}
        updated_positions = {}

        for drone in drones:
            neighbours = self.get_neighbors(
                drone,
                drones,
            )

            grids = [grid_snapshot[drone.drone_id]]
            grids.extend(
                grid_snapshot[neighbour.drone_id] for neighbour in neighbours
            )

            updated_grids[drone.drone_id] = np.mean(
                grids,
                axis=0,
            )

            # Multi-hop position sharing: merge own dictionary with neighbors' dictionaries
            merged_positions = dict(pos_snapshot[drone.drone_id])
            for neighbour in neighbours:
                for k, pos in pos_snapshot[neighbour.drone_id].items():
                    merged_positions[k] = np.asarray(pos, dtype=float).copy()

            updated_positions[drone.drone_id] = merged_positions

        for drone in drones:
            drone.grid = updated_grids[drone.drone_id]
            drone.known_positions = updated_positions[drone.drone_id]

    # ==================================================================
    # POLYGON CLOSURE DETECTION (DFS FLOOD-FILL)
    # ==================================================================

    def is_polygon_closed(self, grid):
        """
        Determine if the occupancy grid contains a closed polygon contour.

        Performs a Depth-First Search (DFS) flood-fill starting from all outer
        grid boundaries on free space cells (`grid < occupancy_threshold`).
        If at least one free cell remains unvisited (enclosed/trapped inside
        the boundary), the polygon is considered closed.

        Parameters
        ----------
        grid : np.ndarray
            2D occupancy grid.

        Returns
        -------
        bool
            True if the polygon forms a closed loop enclosing free space.
        """
        grid = np.asarray(grid, dtype=float)
        if grid.ndim != 2 or grid.size == 0:
            return False

        # If there are no occupied cells, there cannot be an enclosed boundary
        if np.sum(grid >= self.occupancy_threshold) == 0:
            return False

        nx, ny = grid.shape
        visited = np.zeros((nx, ny), dtype=bool)
        stack = []

        # Seed the DFS with all free boundary cells along the four outer edges
        for x in range(nx):
            # Top and bottom edges
            for y in (0, ny - 1):
                if grid[x, y] < self.occupancy_threshold and not visited[x, y]:
                    visited[x, y] = True
                    stack.append((x, y))

        for y in range(ny):
            # Left and right edges
            for x in (0, nx - 1):
                if grid[x, y] < self.occupancy_threshold and not visited[x, y]:
                    visited[x, y] = True
                    stack.append((x, y))

        # Perform DFS flood-fill across 4-connected free space
        while stack:
            cx, cy = stack.pop()
            for nx_idx, ny_idx in (
                (cx + 1, cy),
                (cx - 1, cy),
                (cx, cy + 1),
                (cx, cy - 1),
            ):
                if 0 <= nx_idx < nx and 0 <= ny_idx < ny:
                    if (
                        not visited[nx_idx, ny_idx]
                        and grid[nx_idx, ny_idx] < self.occupancy_threshold
                    ):
                        visited[nx_idx, ny_idx] = True
                        stack.append((nx_idx, ny_idx))

        # Check if any free cells were trapped/enclosed (unvisited)
        free_mask = grid < self.occupancy_threshold
        unvisited_free = free_mask & (~visited)

        return bool(np.any(unvisited_free))

    # ==================================================================
    # RING ORDERING & CONSENSUS PROJECTION
    # ==================================================================

    def compute_ring_ordering(self, drone, drones=None):
        """
        Project known drone positions onto the local consensus grid boundary,
        compute their angles relative to the boundary's center of mass, and
        sort them to establish a 1D ring sequence with predecessor and successor.

        Parameters
        ----------
        drone : Drone
            The reference drone executing the local decision.
        drones : list of Drone, optional
            Active drones in simulation (used as fallback for known positions).

        Returns
        -------
        dict or None
            Ring information including sorted ring sequence, center of mass,
            predecessor, successor, and total known drone count.
        """
        grid = np.asarray(
            getattr(drone, "grid", np.zeros((1, 1), dtype=float)),
            dtype=float,
        )
        if grid.size == 0:
            return None

        occupied_indices = np.argwhere(grid >= self.occupancy_threshold)
        if occupied_indices.size == 0:
            return None

        grid_bounds = getattr(drone, "grid_bounds", None)
        if grid_bounds is None:
            return None

        x_min, x_max, y_min, y_max = grid_bounds
        if x_max <= x_min or y_max <= y_min:
            return None

        # Convert occupied cell grid coordinates to world coordinates
        occupied_points = []
        for ix, iy in occupied_indices:
            x = x_min + (ix + 0.5) * self.resolution
            y = y_min + (iy + 0.5) * self.resolution
            occupied_points.append([x, y])
        occupied_points = np.asarray(occupied_points, dtype=float)

        # Compute local center of mass of the boundary
        center_of_mass = np.mean(occupied_points, axis=0)

        # Gather known positions
        known_pos_dict = getattr(drone, "known_positions", None)
        if not known_pos_dict:
            if drones is not None:
                known_pos_dict = {
                    d.drone_id: np.array([d.x, d.y], dtype=float)
                    for d in drones
                }
            else:
                known_pos_dict = {
                    drone.drone_id: np.array([drone.x, drone.y], dtype=float)
                }

        # Project each known drone position onto closest boundary point and find angle
        ring_entries = []
        for d_id, d_pos in known_pos_dict.items():
            pos = np.asarray(d_pos, dtype=float)
            deltas = occupied_points - pos
            distances = np.linalg.norm(deltas, axis=1)
            closest_idx = int(np.argmin(distances))
            proj_point = occupied_points[closest_idx]

            # Angle relative to center of mass in [-pi, pi]
            angle = float(
                np.arctan2(
                    proj_point[1] - center_of_mass[1],
                    proj_point[0] - center_of_mass[0],
                )
            )

            ring_entries.append(
                {
                    "drone_id": d_id,
                    "angle": angle,
                    "projected_pos": proj_point,
                    "pos": pos,
                }
            )

        # Sort entries by angle ascending (counter-clockwise ring order)
        ring_entries.sort(key=lambda item: item["angle"])

        n_drones = len(ring_entries)
        drone_ids = [entry["drone_id"] for entry in ring_entries]

        if drone.drone_id not in drone_ids:
            # Fallback if own id missing
            return None

        curr_idx = drone_ids.index(drone.drone_id)
        pred_idx = (curr_idx - 1) % n_drones
        succ_idx = (curr_idx + 1) % n_drones

        return {
            "ring": ring_entries,
            "center_of_mass": center_of_mass,
            "current_idx": curr_idx,
            "current": ring_entries[curr_idx],
            "pred": ring_entries[pred_idx],
            "succ": ring_entries[succ_idx],
            "N": n_drones,
        }

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
    # MOTION ACTIONS
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
        k_t=None,
    ):
        """Follow the concentration contour locally with a given tangential gain."""

        if k_t is None:
            k_t = self.k_t

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
            float(k_t) * tangent
            - normal_gain * normal
        )

        return self._clip_action(
            action
        )

    def _equidistant_action(
        self,
        drone,
        ring_info,
        world_field,
        x_coords,
        y_coords,
    ):
        """
        Compute motion action in the "equi_distant" state with dynamic tangential gain.

        Computes the angular spacing error relative to ideal spacing
        (delta_theta_ideal = 2*pi / N) and dynamically determines k_{t,i}.
        k_{t,i} can be negative to invert movement direction along the contour
        when overtakes or tight gaps occur.
        """
        if ring_info is None or ring_info.get("N", 0) < 2:
            return self._boundary_tracking_action(
                drone,
                world_field,
                x_coords,
                y_coords,
                k_t=self.k_t,
            )

        n = ring_info["N"]
        theta_ideal = 2.0 * np.pi / float(n)

        theta_curr = ring_info["current"]["angle"]
        theta_succ = ring_info["succ"]["angle"]
        theta_pred = ring_info["pred"]["angle"]

        # Angular distance to successor (counter-clockwise forward)
        delta_succ = (theta_succ - theta_curr) % (2.0 * np.pi)
        if delta_succ <= 1e-12:
            delta_succ = 2.0 * np.pi

        # Angular distance to predecessor (counter-clockwise backward)
        delta_pred = (theta_curr - theta_pred) % (2.0 * np.pi)
        if delta_pred <= 1e-12:
            delta_pred = 2.0 * np.pi

        # Spacing error: if delta_succ > delta_pred, gap ahead is larger than behind -> speed up.
        # If delta_succ < delta_pred, too close to successor -> slow down or reverse.
        spacing_error = (delta_succ - delta_pred) / theta_ideal

        # Dynamic tangential gain k_{t,i}
        k_t_i = self.k_t + self.k_spacing * spacing_error
        drone.last_kt = float(k_t_i)

        # Compute contour action with dynamic k_{t,i}
        action = self._boundary_tracking_action(
            drone,
            world_field,
            x_coords,
            y_coords,
            k_t=k_t_i,
        )

        if action is None:
            # Fallback towards projected point on boundary
            target = ring_info["current"]["projected_pos"]
            direction = self._normalize(
                np.asarray(target, dtype=float)
                - np.array([drone.x, drone.y], dtype=float)
            )
            if direction is not None:
                action = direction * self.exploration_speed
            else:
                action = self._exploration_action(drone)

        return action

    def _grid_target(self, drone):
        """Return the nearest occupied cell in the consensus grid in world coordinates."""
        grid = np.asarray(
            getattr(drone, "grid", np.zeros((1, 1), dtype=float)),
            dtype=float,
        )

        if grid.size == 0:
            return None

        occupied = np.argwhere(grid >= self.occupancy_threshold)
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

    # ==================================================================
    # STATE MACHINE & ACTION COMPUTATION
    # ==================================================================

    def compute_actions(
        self,
        drones,
        world_field,
        x_coords,
        y_coords,
    ):
        """
        Compute motion actions for all drones using a state-based architecture:
        - "explore": no occupied cells discovered yet in local grid.
        - "boundary_tracking": occupied cells present, but polygon is not yet closed.
        - "equi_distant": local occupancy grid confirms polygon is closed via DFS.

        Returns
        -------
        dict
            {drone_id: np.ndarray([vx, vy])}
        """

        actions = {}

        for drone in drones:
            target = self._grid_target(drone)
            has_boundary_info = target is not None

            if has_boundary_info:
                # Check polygon closure via DFS flood-fill on local occupancy grid
                is_closed = self.is_polygon_closed(drone.grid)

                if is_closed:
                    drone.last_control_mode = "equi_distant"
                    ring_info = self.compute_ring_ordering(drone, drones)
                    action = self._equidistant_action(
                        drone,
                        ring_info,
                        world_field,
                        x_coords,
                        y_coords,
                    )
                else:
                    drone.last_control_mode = "boundary_tracking"
                    action = self._boundary_tracking_action(
                        drone,
                        world_field,
                        x_coords,
                        y_coords,
                        k_t=self.k_t,
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

            actions[drone.drone_id] = self._clip_action(action)

        # ==============================================================
        # DEBUG PRINTS FOR OVERLAP & EQUIDISTANT CONTROL
        # ==============================================================
        print("\n--- [DEBUG] Control Loop Spacing & Projection Info ---")
        for drone in drones:
            target = self._grid_target(drone)
            is_closed = self.is_polygon_closed(drone.grid) if target is not None else False
            mode = getattr(drone, "last_control_mode", "unknown")
            real_pos = f"({drone.x:.3f}, {drone.y:.3f})"
            proj_pos = f"({target[0]:.3f}, {target[1]:.3f})" if target is not None else "None"

            print(f"Drone {drone.drone_id} | Mode: {mode} | Real Pos: {real_pos} | Proj Pos: {proj_pos}")

            if is_closed:
                ring_info = self.compute_ring_ordering(drone, drones)
                if ring_info is not None:
                    com = ring_info["center_of_mass"]
                    angles_str = ", ".join(
                        f"{entry['drone_id']}: {entry['angle']:.3f} rad ({np.degrees(entry['angle']):.1f}°)"
                        for entry in ring_info["ring"]
                    )
                    pred_id = ring_info["pred"]["drone_id"]
                    succ_id = ring_info["succ"]["drone_id"]
                    kt_val = getattr(drone, "last_kt", self.k_t)

                    print(f"  CoM: ({com[0]:.3f}, {com[1]:.3f}) | Sorted Angles: [{angles_str}]")
                    print(f"  Predecessor: {pred_id} | Successor: {succ_id} | Tangential Gain (kt): {kt_val:.3f}")
                else:
                    print("  Ring info: None (no ring ordering computed)")
            else:
                print(f"  Boundary closed: False | Tangential Gain (kt): {self.k_t:.3f}")
        print("-------------------------------------------------------\n")

        return actions
