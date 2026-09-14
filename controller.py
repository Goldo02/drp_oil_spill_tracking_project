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
        settling_steps=20,
        d_safe=0.5,
        repulsion_gain=0.3,
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
        self.settling_steps = int(settling_steps)
        self.known_boundary_closed = False
        self.known_boundary_points = np.empty((0, 2), dtype=float)
        self.known_boundary_initialized = False

        self.boundary_lock_gain = 3.0

        # Collision avoidance / safety bubble parameters.
        self.d_safe = float(d_safe)                    # minimum safe inter-drone distance [m]
        self.repulsion_gain = float(repulsion_gain)    # repulsion force scale

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
        """Temporarily keep consensus disabled: each drone only tracks its own local state."""

        for drone in drones:
            if not hasattr(drone, "known_positions") or not isinstance(
                drone.known_positions, dict
            ):
                drone.known_positions = {}

            drone.known_positions = {
                drone.drone_id: np.asarray([drone.x, drone.y], dtype=float).copy()
            }

    # ==================================================================
    # KNOWN BOUNDARY INITIALIZATION
    # ==================================================================

    def extract_boundary_contour(self, world_field, x_coords=None, y_coords=None):
        """Return valid outer-boundary contour points for a closed oil-spill mask."""
        field = np.asarray(world_field, dtype=float)
        if field.ndim != 2 or field.size == 0:
            return np.empty((0, 2), dtype=float)

        occupied = field >= self.occupancy_threshold
        if not np.any(occupied):
            return np.empty((0, 2), dtype=float)

        boundary = np.zeros_like(occupied, dtype=bool)
        nx, ny = field.shape

        for ix in range(nx):
            for iy in range(ny):
                if not occupied[ix, iy]:
                    continue
                has_free_neighbor = False
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        x2 = ix + dx
                        y2 = iy + dy
                        if 0 <= x2 < nx and 0 <= y2 < ny:
                            if not occupied[x2, y2]:
                                has_free_neighbor = True
                                break
                    if has_free_neighbor:
                        break
                if has_free_neighbor:
                    boundary[ix, iy] = True

        if not np.any(boundary):
            return np.empty((0, 2), dtype=float)

        points = []
        for ix, iy in np.argwhere(boundary):
            if x_coords is not None and len(x_coords) > ix:
                x = float(x_coords[ix])
            else:
                x = float(ix)
            if y_coords is not None and len(y_coords) > iy:
                y = float(y_coords[iy])
            else:
                y = float(iy)
            points.append((x, y))

        return np.asarray(points, dtype=float)

    def build_boundary_grid(self, world_field, x_coords=None, y_coords=None):
        """Create a binary contour-grid from a known closed boundary."""
        contour = self.extract_boundary_contour(world_field, x_coords, y_coords)
        if contour.size == 0:
            return np.zeros_like(np.asarray(world_field, dtype=float), dtype=float)

        field = np.asarray(world_field, dtype=float)
        contour_grid = np.zeros_like(field, dtype=float)
        for px, py in contour:
            if x_coords is not None and y_coords is not None:
                ix = int(np.argmin(np.abs(np.asarray(x_coords) - px)))
                iy = int(np.argmin(np.abs(np.asarray(y_coords) - py)))
                if 0 <= ix < field.shape[0] and 0 <= iy < field.shape[1]:
                    contour_grid[ix, iy] = 1.0
            else:
                ix = int(round(px))
                iy = int(round(py))
                if 0 <= ix < field.shape[0] and 0 <= iy < field.shape[1]:
                    contour_grid[ix, iy] = 1.0

        return contour_grid

    def initialize_known_boundary(self, world_field, x_coords=None, y_coords=None, force_closed=True):
        """Mark the boundary as pre-mapped and closed so the equidistant controller can start immediately."""
        boundary_points = self.extract_boundary_contour(world_field, x_coords, y_coords)
        self.known_boundary_points = np.asarray(boundary_points, dtype=float)
        self.known_boundary_initialized = True
        self.known_boundary_closed = bool(force_closed and self.known_boundary_points.size > 0)
        return self.known_boundary_points.copy()

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
        if getattr(self, "known_boundary_closed", False) and getattr(self, "known_boundary_points", None) is not None:
            if self.known_boundary_points.size > 0:
                return True

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

    def _grid_to_world_points(self, grid, grid_bounds):
        """Map occupied grid cells to continuous world coordinates."""
        grid = np.asarray(grid, dtype=float)
        if grid.size == 0:
            return np.empty((0, 2), dtype=float)

        x_min, x_max, y_min, y_max = grid_bounds
        if x_max <= x_min or y_max <= y_min:
            return np.empty((0, 2), dtype=float)

        occupied_indices = np.argwhere(grid >= self.occupancy_threshold)
        if occupied_indices.size == 0:
            return np.empty((0, 2), dtype=float)

        nx, ny = grid.shape
        dx_cell = (x_max - x_min) / float(nx) if nx > 0 else self.resolution
        dy_cell = (y_max - y_min) / float(ny) if ny > 0 else self.resolution

        x_centers = x_min + (np.arange(nx, dtype=float) + 0.5) * dx_cell
        y_centers = y_min + (np.arange(ny, dtype=float) + 0.5) * dy_cell

        # The occupancy grid is indexed as [x_index, y_index], i.e. the first axis
        # is the world x-coordinate and the second is the world y-coordinate.
        return np.column_stack([
            x_centers[occupied_indices[:, 0]],
            y_centers[occupied_indices[:, 1]],
        ])

    def _compute_consensus_center_of_mass(self, drone, occupied_points, drones=None):
        """Temporarily disable inter-drone consensus: use the local boundary CoM only."""
        local_com = np.mean(occupied_points, axis=0)

        prev_com = getattr(drone, "center_of_mass_estimate", None)
        if prev_com is not None:
            prev_com = np.asarray(prev_com, dtype=float)
            delta = local_com - prev_com
            norm_delta = float(np.linalg.norm(delta))
            max_step = 0.30
            if norm_delta > max_step:
                delta = delta / max(norm_delta, 1e-9) * max_step
                local_com = prev_com + delta
            alpha = 0.45
            local_com = alpha * local_com + (1.0 - alpha) * prev_com

        drone.center_of_mass_estimate = np.asarray(local_com, dtype=float).copy()
        return drone.center_of_mass_estimate.copy()

    def compute_ring_ordering(self, drone, drones=None):
        """
        Order the boundary by walking the closed contour chain instead of using
        polar angles around the center of mass. This keeps the 1D ring stable even
        for non-convex closed shapes with deep indentations.
        """
        grid = np.asarray(
            getattr(drone, "grid", np.zeros((1, 1), dtype=float)),
            dtype=float,
        )
        if grid.size == 0:
            return None

        grid_bounds = getattr(drone, "grid_bounds", None)
        if grid_bounds is None:
            return None

        occupied_points = self._grid_to_world_points(grid, grid_bounds)
        if occupied_points.size == 0:
            return None

        # ------------------------------------------------------------------
        # Build a sequential closed contour chain from the occupied boundary
        # points. This is a nearest-neighbor walk along the contour, not a
        # polar-angle sort around the CoM.
        # ------------------------------------------------------------------
        ordered_points = occupied_points.copy()
        if ordered_points.shape[0] > 1:
            remaining = list(range(ordered_points.shape[0]))
            start_idx = int(np.argmin(np.linalg.norm(ordered_points - np.mean(ordered_points, axis=0), axis=1)))
            ordered = [start_idx]
            remaining.remove(start_idx)

            current_idx = start_idx
            while remaining:
                current_point = ordered_points[current_idx]
                dists = np.linalg.norm(ordered_points[remaining] - current_point, axis=1)
                next_local = int(np.argmin(dists))
                next_idx = remaining.pop(next_local)
                ordered.append(next_idx)
                current_idx = next_idx

            ordered_points = ordered_points[np.asarray(ordered, dtype=int)]
        else:
            ordered_points = occupied_points.copy()

        # Close the loop explicitly so the chain has a proper circular topology.
        n_chain = ordered_points.shape[0]
        if n_chain > 1:
            boundary_chain = np.concatenate([ordered_points, ordered_points[:1]], axis=0)
        else:
            boundary_chain = ordered_points.copy()

        center_of_mass = np.mean(ordered_points, axis=0)

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

        # ------------------------------------------------------------------
        # Each drone is assigned its closest contour index along the ordered loop.
        # This replaces arctan2 radial ordering with chain-index ordering.
        # ------------------------------------------------------------------
        ring_entries = []
        for d_id, d_pos in known_pos_dict.items():
            pos = np.asarray(d_pos, dtype=float)
            deltas = boundary_chain[:-1] - pos[None, :]
            distances = np.linalg.norm(deltas, axis=1)
            closest_idx = int(np.argmin(distances))
            proj_point = boundary_chain[closest_idx].copy()
            chain_index = float(closest_idx)
            theta = 2.0 * np.pi * chain_index / float(max(n_chain, 1))

            ring_entries.append(
                {
                    "drone_id": d_id,
                    "chain_index": chain_index,
                    "angle": theta,
                    "projected_pos": proj_point,
                    "pos": pos,
                }
            )

        ring_entries.sort(key=lambda item: item["chain_index"])

        # Choose the contour orientation consistently around the geometric center of
        # the boundary, without using arctan2. This preserves a clockwise/counter-
        # clockwise ordering that is stable under non-convex shapes.
        signed_area = 0.0
        for i, entry in enumerate(ring_entries):
            next_entry = ring_entries[(i + 1) % len(ring_entries)]
            a = np.asarray(entry["pos"], dtype=float) - center_of_mass
            b = np.asarray(next_entry["pos"], dtype=float) - center_of_mass
            signed_area += a[0] * b[1] - a[1] * b[0]

        if signed_area < 0.0:
            ring_entries = list(reversed(ring_entries))

        n_drones = len(ring_entries)
        drone_ids = [entry["drone_id"] for entry in ring_entries]

        if drone.drone_id not in drone_ids:
            return None

        current_raw_idx = drone_ids.index(drone.drone_id)
        ring_entries = ring_entries[current_raw_idx:] + ring_entries[:current_raw_idx]
        drone_ids = [entry["drone_id"] for entry in ring_entries]
        curr_idx = 0
        if n_drones == 1:
            pred_idx = 0
            succ_idx = 0
        else:
            pred_idx = -1
            succ_idx = 1

        # ------------------------------------------------------------------
        # 1D Voronoi partition on the closed contour chain. For each boundary pixel,
        # assign it to the drone whose chain index is closest along the wrapped ring.
        # ------------------------------------------------------------------
        drone_chain_indices = np.asarray([entry["chain_index"] for entry in ring_entries], dtype=float)
        boundary_chain_indices = np.arange(n_chain, dtype=float)

        if n_drones > 0:
            wrapped_delta = (boundary_chain_indices[:, None] - drone_chain_indices[None, :]) % n_chain
            circular_dist = np.minimum(wrapped_delta, n_chain - wrapped_delta)
            assigned_drone_indices = np.argmin(circular_dist, axis=1)
        else:
            assigned_drone_indices = np.zeros(n_chain, dtype=int)

        # ------------------------------------------------------------------
        # Target centroid computed as the midpoint of each assigned chain segment,
        # projected back to the contour point list.
        # ------------------------------------------------------------------
        prev_target = getattr(drone, "target_centroid", None)
        if prev_target is not None:
            prev_target = np.asarray(prev_target, dtype=float).copy()

        alpha = 0.35
        max_step = 0.35

        for j, entry in enumerate(ring_entries):
            cell_mask = assigned_drone_indices == j
            cell_indices = np.where(cell_mask)[0]

            if cell_indices.size > 0:
                cell_points = ordered_points[cell_indices]
                arc_midpoint = np.mean(cell_points, axis=0)
                local_dists = np.linalg.norm(cell_points - arc_midpoint, axis=1)
                best_local_idx = int(np.argmin(local_dists))
                target_point = cell_points[best_local_idx].copy()
                target_chain_idx = float(cell_indices[best_local_idx])
                cell_size = int(cell_indices.size)
            else:
                target_point = entry["projected_pos"].copy()
                target_chain_idx = float(entry["chain_index"])
                cell_size = 0

            if prev_target is not None and entry["drone_id"] == drone.drone_id:
                candidate = target_point.copy()
                delta = candidate - prev_target
                norm_delta = float(np.linalg.norm(delta))
                if norm_delta > max_step:
                    delta = delta / max(norm_delta, 1e-9) * max_step
                    candidate = prev_target + delta
                target_point = alpha * candidate + (1.0 - alpha) * prev_target

            entry["target_centroid"] = target_point
            entry["voronoi_cell_size"] = cell_size
            entry["target_chain_index"] = target_chain_idx

        drone.target_centroid = ring_entries[curr_idx]["target_centroid"].copy()
        drone.target_centroid_prev = drone.target_centroid.copy()

        return {
            "ring": ring_entries,
            "center_of_mass": center_of_mass,
            "current_idx": curr_idx,
            "current": ring_entries[curr_idx],
            "pred": ring_entries[pred_idx],
            "succ": ring_entries[succ_idx],
            "N": n_drones,
            "occupied_points": ordered_points,
            "assigned_drone_indices": assigned_drone_indices,
            "chain_length": n_chain,
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

        # Use a single, consistent convention for the boundary normal in the
        # continuous world frame: the inward normal points from the high-value
        # side of the field toward the contour. This keeps the ring-angle
        # convention and the contour-following tangent direction aligned.
        normal = self._normalize(
            -gradient
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
            + normal_gain * normal
        )

        return self._clip_action(
            action
        )

    # ------------------------------------------------------------------
    # COLLISION AVOIDANCE
    # ------------------------------------------------------------------

    def _compute_repulsion(self, drone):
        """
        Compute a short-range repulsion vector to avoid drone collisions.

        For every known drone closer than ``self.d_safe`` metres, adds a
        repulsion contribution pointing *away* from that drone, with magnitude
        proportional to how deep inside the safety bubble it is:

            f_rep = (d_safe - d) / d_safe  *  (pos_self - pos_other) / d

        Returns
        -------
        np.ndarray, shape (2,)
            Summed repulsion vector (zero if no neighbour is inside d_safe).
        """
        pos_self = np.array([drone.x, drone.y], dtype=float)
        repulsion = np.zeros(2, dtype=float)

        known = getattr(drone, "known_positions", {})
        for other_id, pos_other in known.items():
            if other_id == drone.drone_id:
                continue
            pos_other = np.asarray(pos_other, dtype=float)
            diff = pos_self - pos_other
            dist = float(np.linalg.norm(diff))
            if 0.0 < dist < self.d_safe:
                # Soft linear ramp: full strength at contact, zero at d_safe
                magnitude = (self.d_safe - dist) / self.d_safe
                repulsion += magnitude * (diff / dist)

        return repulsion

    def _equidistant_action(
        self,
        drone,
        ring_info,
        world_field,
        x_coords,
        y_coords,
    ):
        """Hybrid soft-constraint control law on the known boundary contour.

        The behavior is:
        1. direct 2D centroid attraction toward the Voronoi target,
        2. dominant tangential/spacing motion along the local curve,
        3. strong normal anchoring to the nearest boundary point,
        4. short-range collision repulsion,
        5. final clipping to account for max-speed limitations.
        """
        pos = np.asarray([drone.x, drone.y], dtype=float)
        action = np.zeros(2, dtype=float)

        # ---- Step 1: read the Voronoi centroid target ----
        target_centroid = None
        if ring_info is not None:
            current = ring_info.get("current", {})
            if isinstance(current, dict):
                target_centroid = current.get("target_centroid")

        if target_centroid is None:
            target_centroid = pos.copy()
        target_centroid = np.asarray(target_centroid, dtype=float)

        error_2d = target_centroid - pos

        # ---- Boundary-aware local tangent and nearest contour point ----
        boundary = np.asarray(self.known_boundary_points, dtype=float)
        if boundary.size > 0:
            diff = boundary - pos[None, :]
            dists = np.linalg.norm(diff, axis=1)
            closest_idx = int(np.argmin(dists))
            closest_point = boundary[closest_idx]

            n = boundary.shape[0]
            prev_idx = (closest_idx - 1) % n
            next_idx = (closest_idx + 1) % n
            tangent = boundary[next_idx] - boundary[prev_idx]
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm <= 1e-12:
                tangent = np.array([1.0, 0.0], dtype=float)
            else:
                tangent = tangent / tangent_norm

            # Hybrid blending: keep the dominant tangential spacing term, but allow a
            # small direct 2D component so the drone can cut through difficult contour
            # geometries instead of stalling when the tangent becomes locally vertical.
            alpha = 0.8
            tangential_component = float(error_2d.dot(tangent)) * tangent
            direct_component = error_2d
            desired_movement = alpha * tangential_component + (1.0 - alpha) * direct_component
            action += desired_movement

            # Strict boundary enforcement: always keep the drone anchored to the contour.
            normal_vec = closest_point - pos
            normal_norm = float(np.linalg.norm(normal_vec))
            if normal_norm > 1e-12:
                action += self.k_n * normal_vec

            # Collapse guard: if the ring is compressed and the target is already reached,
            # keep a small tangential drift so the drone does not freeze in place.
            if ring_info is not None and ring_info.get("N", 0) >= 2:
                theta_curr = float(ring_info.get("current", {}).get("angle", 0.0))
                theta_pred = float(ring_info.get("pred", {}).get("angle", theta_curr))
                theta_succ = float(ring_info.get("succ", {}).get("angle", theta_curr))
                delta_pred = (theta_curr - theta_pred) % (2.0 * np.pi)
                delta_succ = (theta_succ - theta_curr) % (2.0 * np.pi)
                theta_ideal = 2.0 * np.pi / float(max(ring_info.get("N", 2), 2))
                if min(delta_pred, delta_succ) < 0.85 * theta_ideal:
                    guard_dir = 1.0 if delta_succ <= delta_pred else -1.0
                    action += 0.15 * self.max_speed * guard_dir * tangent

        else:
            # Fallback when the contour has not been initialized yet: direct attraction
            # plus a small tangential drift to keep the drone moving.
            error_norm = float(np.linalg.norm(error_2d))
            if error_norm > 1e-12:
                action += 0.5 * (error_2d / error_norm) * self.max_speed

            if ring_info is not None and ring_info.get("N", 0) >= 2:
                com = np.asarray(ring_info.get("center_of_mass", np.zeros(2, dtype=float)), dtype=float)
                radial = pos - com
                radial_norm = float(np.linalg.norm(radial))
                if radial_norm > 1e-12:
                    radial_dir = radial / radial_norm
                else:
                    radial_dir = np.array([1.0, 0.0], dtype=float)

                tangent_dir = np.array([-radial_dir[1], radial_dir[0]], dtype=float)
                tangent_dir = self._normalize(tangent_dir)
                if tangent_dir is None:
                    tangent_dir = np.array([0.0, 1.0], dtype=float)

                theta_curr = float(ring_info.get("current", {}).get("angle", 0.0))
                theta_pred = float(ring_info.get("pred", {}).get("angle", theta_curr))
                theta_succ = float(ring_info.get("succ", {}).get("angle", theta_curr))
                delta_pred = (theta_curr - theta_pred) % (2.0 * np.pi)
                delta_succ = (theta_succ - theta_curr) % (2.0 * np.pi)
                theta_ideal = 2.0 * np.pi / float(max(ring_info.get("N", 2), 2))
                if min(delta_pred, delta_succ) < 0.85 * theta_ideal:
                    guard_dir = 1.0 if delta_succ <= delta_pred else -1.0
                    action += 0.20 * self.max_speed * guard_dir * tangent_dir
                elif error_norm > 1e-12:
                    action += 0.2 * self.max_speed * float(error_2d.dot(tangent_dir)) * tangent_dir

        # ---- safety and clipping ----
        action += self.repulsion_gain * self._compute_repulsion(drone)
        return self._clip_action(action)

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
        """Compute the ring-spacing control action for drones that are already on the boundary."""

        actions = {}

        for drone in drones:
            ring_info = self.compute_ring_ordering(drone, drones)

            if ring_info is None:
                drone.last_control_mode = "equi_distant"
                action = np.zeros(2, dtype=float)
            else:
                drone.last_control_mode = "equi_distant"
                drone.last_ring_info = ring_info
                action = self._equidistant_action(
                    drone,
                    ring_info,
                    world_field,
                    x_coords,
                    y_coords,
                )

            actions[drone.drone_id] = self._clip_action(action)

        return actions
