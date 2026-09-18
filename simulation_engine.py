import numpy as np

from drone import Drone
from controller import Controller, DroneController

try:
    import contourpy
except ImportError:  # pragma: no cover
    contourpy = None


class SimulationEngine:
    """
    Main coordinator of the multi-drone simulation.

    Responsibilities:
        - update the environment;
        - trigger drone sensing;
        - update local occupancy grids;
        - execute distributed consensus;
        - compute diagnostics;
        - request actions from the controller;
        - apply actions to drones;
        - expose state to the visualizer.
    """

    def __init__(
        self,
        sim_map,
        oil_spill,
        x_min=-10.0,
        x_max=10.0,
        y_min=-10.0,
        y_max=10.0,
        resolution=0.1,
        sensor_size=100,
        measure_every=3,
        communication_radius_cells=205,
        fully_connected=False,
        occupancy_threshold=0.5,
        temporal_alpha=0.05,
        consensus_rounds=10,
        dt=1.0,
        verbose=True,
        closure_min_enclosed_false_cells=250,
        mapping_point_radius_cells=1,
    ):
        self.sim_map = sim_map
        self.oil_spill = oil_spill

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)

        self.resolution = float(resolution)
        self.dt = float(dt)

        self.sensor_size = int(sensor_size)
        self.measure_every = max(1, int(measure_every))
        self.occupancy_threshold = float(occupancy_threshold)
        self.temporal_alpha = None
        self.consensus_rounds = max(1, int(consensus_rounds))
        self.verbose = bool(verbose)
        self.fully_connected = bool(fully_connected)

        self.Nx = int(round((self.x_max - self.x_min) / self.resolution))
        self.Ny = int(round((self.y_max - self.y_min) / self.resolution))
        self.grid_shape = (self.Nx, self.Ny)
        self.grid_bounds = (self.x_min, self.x_max, self.y_min, self.y_max)

        self.world_field = self._get_world_field()

        dx = self.sim_map.dx if self.sim_map.dx > 0 else self.resolution
        dy = self.sim_map.dy if self.sim_map.dy > 0 else self.resolution
        self.communication_radius_cells = int(communication_radius_cells)
        self.communication_radius = (
            self.communication_radius_cells * 0.5 * (abs(dx) + abs(dy))
        )

        self.drones = []
        self.frame = 0
        self.error_history = []
        self.mean_grid_history = []
        self.latest_mean_grid = np.zeros(self.grid_shape, dtype=float)
        self.measurement_consensus_history = []
        self._current_measurement_trace = None
        self.control_state = "mapping"
        self.transition_frame = None
        self.closed_boundary_points = np.empty((0, 2), dtype=float)
        self.boundary_controller = Controller(
            sim_map=self.sim_map,
            communication_radius=self.communication_radius,
            occupancy_threshold=self.occupancy_threshold,
        )
        self.closure_min_boundary_cells = 24
        self.closure_min_enclosed_false_cells = int(closure_min_enclosed_false_cells)
        self.mapping_point_radius_cells = max(0, int(mapping_point_radius_cells))
        self.last_closure_enclosed_false_cells = 0

    def _get_world_field(self):
        """Return the current environment field."""
        return np.asarray(self.oil_spill.get_field(self.sim_map.X, self.sim_map.Y), dtype=float)

    def _update_environment(self):
        """Advance the environment by one simulation timestep."""
        self.oil_spill.update(self.dt)
        self.world_field = self._get_world_field()

    def add_drone(
        self,
        drone_id,
        x,
        y,
        gps_noise=0.1,
        camera_noise=0.1,
    ):
        """Create and register a drone."""

        drone = Drone(
            drone_id=drone_id,
            x=x,
            y=y,
            grid_shape=self.grid_shape,
            grid_bounds=self.grid_bounds,
            sensor_size=self.sensor_size,
            gps_noise=gps_noise,
            camera_noise=camera_noise,
            controller=DroneController(
                sim_map=self.sim_map,
                communication_radius=self.communication_radius,
                fully_connected=self.fully_connected,
                occupancy_threshold=self.occupancy_threshold,
                resolution=self.resolution,
            ),
        )

        self.drones.append(drone)

        return drone

    def _perform_measurement(self):
        """Perform sensing and local grid updates."""

        for drone in self.drones:

            edge_points = drone.sense(
                self.world_field,
                self.sim_map.x_coords,
                self.sim_map.y_coords,
            )

            drone.update_grid(
                edge_points=edge_points,
                x_min=self.x_min,
                y_min=self.y_min,
                resolution=self.resolution,
                alpha=None,
                point_radius_cells=self.mapping_point_radius_cells,
            )

    def _get_neighbors(self, drone):
        if self.fully_connected:
            return [other for other in self.drones if other is not drone]

        return [
            other
            for other in self.drones
            if other is not drone
            and float(np.linalg.norm(drone.position - other.position))
            <= self.communication_radius
        ]

    def _exchange_consensus_messages(self):
        """Deliver one synchronous round of local map messages."""
        messages = {drone.drone_id: drone.create_consensus_message() for drone in self.drones}
        delivered = {
            drone.drone_id: [
                messages[neighbor.drone_id] for neighbor in self._get_neighbors(drone)
            ]
            for drone in self.drones
        }

        for drone in self.drones:
            drone.consensus_step(
                delivered[drone.drone_id],
                own_grid=messages[drone.drone_id]["grid"],
            )

    def _perform_consensus(self):
        """Run the configured number of consensus iterations."""
        for _ in range(self.consensus_rounds):
            self._exchange_consensus_messages()

    def compute_mean_grid(self):
        """Return the mean occupancy grid."""

        if not self.drones:
            return np.zeros(self.grid_shape, dtype=float)

        return np.mean(
            [np.asarray(drone.grid, dtype=float) for drone in self.drones],
            axis=0,
        )

    def compute_disagreement_error(self):
        """Return mean L2 disagreement from the global mean."""

        if not self.drones:
            return 0.0, np.zeros(self.grid_shape, dtype=float)

        mean_grid = self.compute_mean_grid()
        errors = [
            np.linalg.norm(np.asarray(drone.grid, dtype=float) - mean_grid)
            for drone in self.drones
        ]
        return float(np.mean(errors)), mean_grid

    def _drone_error_snapshot(self):
        mean_grid = self.compute_mean_grid()

        return {
            drone.drone_id: float(np.linalg.norm(np.asarray(drone.grid, dtype=float) - mean_grid))
            for drone in self.drones
        }

    def _print_error_snapshot(self, header):
        if not self.verbose:
            return

        snapshot = self._drone_error_snapshot()

        values = list(snapshot.values())
        mean_error = float(np.mean(values)) if values else 0.0
        max_error = float(np.max(values)) if values else 0.0

        ordered = ", ".join(
            f"{drone_id}={value:.6f}" for drone_id, value in snapshot.items()
        )

        print(f"{header} | mean_error={mean_error:.6f} | max_error={max_error:.6f}")
        print(f"    per-drone: {ordered}")

    def _start_new_measurement_trace(self):

        if self._current_measurement_trace is not None:

            self.measurement_consensus_history.append(
                {
                    drone_id: list(values)
                    for drone_id, values in self._current_measurement_trace.items()
                }
            )

        self._current_measurement_trace = {drone.drone_id: [] for drone in self.drones}

    def _record_measurement_trace(self):

        if self._current_measurement_trace is None:
            self._current_measurement_trace = {
                drone.drone_id: [] for drone in self.drones
            }

        snapshot = self._drone_error_snapshot()

        for drone_id, value in snapshot.items():
            self._current_measurement_trace[drone_id].append(value)

    def _apply_mapping_actions(self):
        """Compute mapping/orbiting actions and apply them."""

        for drone in self.drones:

            action = drone.compute_action(
                self.world_field,
                self.sim_map.x_coords,
                self.sim_map.y_coords,
            )

            drone.action(
                action,
                bounds=(
                    self.sim_map.xlim,
                    self.sim_map.ylim,
                ),
            )

    @staticmethod
    def _sensed_position(drone):
        if hasattr(drone, "get_gps_pos"):
            return np.asarray(drone.get_gps_pos(), dtype=float)
        return np.asarray(drone.position, dtype=float)

    def _exchange_positions_multihop(self):
        sensed_positions = {
            drone.drone_id: self._sensed_position(drone)
            for drone in self.drones
        }
        sensed_arcs = {
            drone.drone_id: getattr(drone, "boundary_s", None)
            for drone in self.drones
        }

        for drone in self.drones:
            drone.known_positions = {
                drone.drone_id: sensed_positions[drone.drone_id].copy(),
            }
            drone.known_boundary_arcs = {}
            if sensed_arcs[drone.drone_id] is not None:
                drone.known_boundary_arcs[drone.drone_id] = float(
                    sensed_arcs[drone.drone_id]
                )

        hop_count = max(1, len(self.drones))
        for _ in range(hop_count):
            pending_updates = [{} for _ in self.drones]
            pending_arc_updates = [{} for _ in self.drones]

            for i, drone_i in enumerate(self.drones):
                for j, drone_j in enumerate(self.drones):
                    if i == j:
                        continue
                    in_range = getattr(self, "fully_connected", False) or (
                        float(np.linalg.norm(drone_i.position - drone_j.position))
                        <= self.communication_radius
                    )
                    if in_range:
                        pending_updates[i].update(drone_j.known_positions)
                        pending_arc_updates[i].update(
                            getattr(drone_j, "known_boundary_arcs", {})
                        )

            for i, drone in enumerate(self.drones):
                drone_id = drone.drone_id
                for known_id, position in pending_updates[i].items():
                    if known_id != drone_id:
                        drone.known_positions[known_id] = np.asarray(position, dtype=float)
                for known_id, boundary_s in pending_arc_updates[i].items():
                    if boundary_s is not None:
                        drone.known_boundary_arcs[known_id] = float(boundary_s)
                drone.known_positions[drone_id] = sensed_positions[drone_id].copy()
                if sensed_arcs[drone_id] is not None:
                    drone.known_boundary_arcs[drone_id] = float(sensed_arcs[drone_id])

    def _apply_lloyd_actions(self):
        """Compute decentralized 1D Voronoi/Lloyd actions and apply them."""
        self._exchange_positions_multihop()

        actions = {
            drone.drone_id: drone.compute_action()
            for drone in self.drones
        }

        for drone in self.drones:
            action = actions.get(drone.drone_id, np.zeros(2, dtype=float))
            drone.action(
                action,
                bounds=(
                    self.sim_map.xlim,
                    self.sim_map.ylim,
                ),
            )
            drone.project_to_boundary()

    def _apply_actions(self):
        if self.control_state == "lloyd":
            self._apply_lloyd_actions()
        else:
            self._apply_mapping_actions()

    def _occupied_boundary_cells(self, grid):
        occupied = np.asarray(grid, dtype=float) > self.occupancy_threshold
        cells = [tuple(cell) for cell in np.argwhere(occupied)]
        return set(cells)

    @staticmethod
    def _cell_neighbors(cell):
        ix, iy = cell
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                yield ix + dx, iy + dy

    def _dfs_polygon_closure_check(self, grid):
        """Use DFS over occupied boundary cells to detect one closed loop."""
        cells = self._occupied_boundary_cells(grid)
        if len(cells) < self.closure_min_boundary_cells:
            self.last_closure_enclosed_false_cells = 0
            return False, np.empty((0, 2), dtype=float)

        adjacency = {
            cell: [neighbor for neighbor in self._cell_neighbors(cell) if neighbor in cells]
            for cell in cells
        }
        usable = {cell for cell, neighbors in adjacency.items() if len(neighbors) >= 2}
        if len(usable) < self.closure_min_boundary_cells:
            self.last_closure_enclosed_false_cells = 0
            return False, np.empty((0, 2), dtype=float)

        start = next(iter(usable))
        stack = [start]
        visited = set()
        while stack:
            cell = stack.pop()
            if cell in visited or cell not in usable:
                continue
            visited.add(cell)
            stack.extend(neighbor for neighbor in adjacency[cell] if neighbor in usable)

        if len(visited) != len(usable):
            self.last_closure_enclosed_false_cells = 0
            return False, np.empty((0, 2), dtype=float)

        has_open_endpoint = any(
            sum(1 for neighbor in adjacency[cell] if neighbor in usable) < 2
            for cell in usable
        )
        if has_open_endpoint:
            self.last_closure_enclosed_false_cells = 0
            return False, np.empty((0, 2), dtype=float)

        enclosed_false_cells = self._count_enclosed_false_cells(grid)
        self.last_closure_enclosed_false_cells = enclosed_false_cells
        if enclosed_false_cells < getattr(self, "closure_min_enclosed_false_cells", 0):
            return False, np.empty((0, 2), dtype=float)

        enclosed_contour = self._ordered_enclosed_contour_points_from_grid(grid)
        if enclosed_contour.shape[0] >= self.closure_min_boundary_cells:
            return True, enclosed_contour

        ordered_cells = self._order_loop_cells(visited, adjacency)
        neighbor_counts = np.asarray(
            [
                sum(1 for neighbor in adjacency[cell] if neighbor in usable)
                for cell in usable
            ],
            dtype=float,
        )
        if neighbor_counts.size and float(np.percentile(neighbor_counts, 90.0)) <= 4.0:
            return True, self._cells_to_world_points(ordered_cells)

        centerline_points = self._ordered_centerline_points_from_grid(grid)
        if (
            centerline_points.shape[0] >= self.closure_min_boundary_cells
            and self._is_boundary_trace_continuous(centerline_points)
        ):
            return True, centerline_points

        return True, self._cells_to_world_points(ordered_cells)

    def _enclosed_free_mask(self, grid):
        occupied = np.asarray(grid, dtype=float) > self.occupancy_threshold
        free = ~occupied
        if free.size == 0:
            return np.zeros_like(free, dtype=bool)

        nx, ny = free.shape
        visited = np.zeros_like(free, dtype=bool)
        stack = []

        for ix in range(nx):
            for iy in (0, ny - 1):
                if free[ix, iy] and not visited[ix, iy]:
                    visited[ix, iy] = True
                    stack.append((ix, iy))

        for iy in range(ny):
            for ix in (0, nx - 1):
                if free[ix, iy] and not visited[ix, iy]:
                    visited[ix, iy] = True
                    stack.append((ix, iy))

        while stack:
            ix, iy = stack.pop()
            for x2, y2 in (
                (ix - 1, iy),
                (ix + 1, iy),
                (ix, iy - 1),
                (ix, iy + 1),
            ):
                if (
                    0 <= x2 < nx
                    and 0 <= y2 < ny
                    and free[x2, y2]
                    and not visited[x2, y2]
                ):
                    visited[x2, y2] = True
                    stack.append((x2, y2))

        enclosed = free & ~visited
        return enclosed

    def _count_enclosed_false_cells(self, grid):
        enclosed = self._enclosed_free_mask(grid)
        return int(np.count_nonzero(enclosed))

    def _ordered_enclosed_contour_points_from_grid(self, grid):
        if contourpy is None:
            return np.empty((0, 2), dtype=float)

        enclosed = self._enclosed_free_mask(grid)
        if np.count_nonzero(enclosed) < 1:
            return np.empty((0, 2), dtype=float)

        x_coords = self.x_min + (np.arange(enclosed.shape[0]) + 0.5) * self.resolution
        y_coords = self.y_min + (np.arange(enclosed.shape[1]) + 0.5) * self.resolution

        generator = contourpy.contour_generator(
            x=x_coords,
            y=y_coords,
            z=enclosed.astype(float).T,
            name="serial",
        )
        lines = generator.lines(0.5)
        if not lines:
            return np.empty((0, 2), dtype=float)

        def path_length(line):
            if len(line) < 2:
                return 0.0
            return float(np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1)))

        contour = np.asarray(max(lines, key=path_length), dtype=float)
        if contour.ndim != 2 or contour.shape[1] != 2:
            return np.empty((0, 2), dtype=float)
        if len(contour) > 1 and np.linalg.norm(contour[0] - contour[-1]) < 1e-9:
            contour = contour[:-1]

        if not self._is_boundary_trace_continuous(contour):
            return np.empty((0, 2), dtype=float)

        return contour.copy()

    def _ordered_centerline_points_from_grid(self, grid):
        """Return a single ordered boundary centerline from occupied map cells."""
        occupied_cells = np.argwhere(np.asarray(grid, dtype=float) > self.occupancy_threshold)
        if occupied_cells.shape[0] < self.closure_min_boundary_cells:
            return np.empty((0, 2), dtype=float)

        points = self._cells_to_world_points([tuple(cell) for cell in occupied_cells])
        center = np.mean(points, axis=0)
        deltas = points - center.reshape(1, 2)
        radii = np.linalg.norm(deltas, axis=1)
        valid = radii > 1e-12
        if np.count_nonzero(valid) < self.closure_min_boundary_cells:
            return np.empty((0, 2), dtype=float)

        points = points[valid]
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        normalized = (angles + 2.0 * np.pi) % (2.0 * np.pi)

        # Several sensor hits can occupy a small radial band at the same angle.
        # Averaging per angular bin collapses that band to one centerline sample,
        # avoiding the inner/outer double-contour produced by contour extraction.
        bin_count = int(
            np.clip(
                occupied_cells.shape[0] // 2,
                self.closure_min_boundary_cells,
                720,
            )
        )
        bin_ids = np.floor(normalized / (2.0 * np.pi) * bin_count).astype(int)
        bin_ids = np.clip(bin_ids, 0, bin_count - 1)

        centerline = []
        centerline_angles = []
        for bin_id in range(bin_count):
            mask = bin_ids == bin_id
            if not np.any(mask):
                continue
            centerline.append(np.mean(points[mask], axis=0))
            centerline_angles.append(float(np.mean(normalized[mask])))

        if len(centerline) < self.closure_min_boundary_cells:
            return np.empty((0, 2), dtype=float)

        order = np.argsort(centerline_angles)
        return np.asarray(centerline, dtype=float)[order].copy()

    def _is_boundary_trace_continuous(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[0] < 2:
            return False

        segment_lengths = np.linalg.norm(
            np.roll(points, -1, axis=0) - points,
            axis=1,
        )
        max_expected_step = 3.0 * float(self.resolution)
        return bool(np.max(segment_lengths) <= max_expected_step)

    def _order_loop_cells(self, cells, adjacency):
        cells = set(cells)
        if len(cells) <= 2:
            return list(cells)

        start = min(cells)
        order = [start]
        visited = {start}
        stack = [start]

        while len(visited) < len(cells) and stack:
            current = stack[-1]
            previous = stack[-2] if len(stack) > 1 else None
            candidates = [
                neighbor
                for neighbor in adjacency.get(current, [])
                if neighbor in cells and neighbor not in visited
            ]

            if candidates:
                next_cell = self._choose_next_loop_cell(previous, current, candidates)
                stack.append(next_cell)
                order.append(next_cell)
                visited.add(next_cell)
                continue

            stack.pop()
            if stack:
                order.append(stack[-1])

        while len(stack) > 1:
            stack.pop()
            order.append(stack[-1])

        return order

    @staticmethod
    def _choose_next_loop_cell(previous, current, candidates):
        if previous is None or len(candidates) == 1:
            return min(candidates)

        incoming = np.asarray(current, dtype=float) - np.asarray(previous, dtype=float)
        incoming_norm = float(np.linalg.norm(incoming))
        if incoming_norm <= 1e-12:
            return min(candidates)
        incoming = incoming / incoming_norm

        def turn_cost(candidate):
            outgoing = np.asarray(candidate, dtype=float) - np.asarray(current, dtype=float)
            outgoing_norm = float(np.linalg.norm(outgoing))
            if outgoing_norm <= 1e-12:
                return 2.0
            outgoing = outgoing / outgoing_norm
            return 1.0 - float(np.dot(incoming, outgoing))

        return min(candidates, key=turn_cost)

    def _cells_to_world_points(self, cells):
        points = np.asarray(
            [
                (
                    self.x_min + (ix + 0.5) * self.resolution,
                    self.y_min + (iy + 0.5) * self.resolution,
                )
                for ix, iy in cells
            ],
            dtype=float,
        )
        return points.copy()

    def is_mapped_polygon_closed(self, mean_grid=None):
        if mean_grid is None:
            mean_grid = self.compute_mean_grid()
        return self._dfs_polygon_closure_check(mean_grid)

    def _transition_to_lloyd_state(self, boundary_points):
        boundary_points = np.asarray(boundary_points, dtype=float).reshape(-1, 2)
        if self.control_state == "lloyd" or boundary_points.shape[0] < 3:
            return

        self.closed_boundary_points = boundary_points.copy()
        self.boundary_controller.known_boundary_points = boundary_points.copy()
        self.boundary_controller.known_boundary_closed = True
        self.boundary_controller.known_boundary_ordered = True
        self.control_state = "lloyd"
        self.transition_frame = self.frame

        known_boundary_arcs = {}
        for drone in self.drones:
            drone.control_state = "lloyd"
            drone.set_known_boundary(
                boundary_points,
                known_boundary_closed=True,
                already_ordered=True,
            )
            drone.pending_boundary_s = None
            drone.pending_boundary_point = None
            drone.project_to_boundary()
            known_boundary_arcs[drone.drone_id] = float(drone.boundary_s)

        known_positions = {
            drone.drone_id: np.array([drone.x, drone.y], dtype=float)
            for drone in self.drones
        }

        for drone in self.drones:
            drone.known_positions = {
                drone_id: position.copy()
                for drone_id, position in known_positions.items()
            }
            drone.known_boundary_arcs = dict(known_boundary_arcs)

        if self.verbose:
            print(
                "  State transition: mapping -> lloyd "
                f"(DFS closed loop with {boundary_points.shape[0]} boundary cells)"
            )

    def get_visualization_data(self):
        """Return state required by the visualizer."""

        error, mean_grid = self.compute_disagreement_error()

        return {
            "frame": self.frame,
            "world_field": self.world_field.copy(),
            "mean_grid": mean_grid.copy(),
            "disagreement_error": error,
            "drones": self.drones,
            "communication_radius": self.communication_radius,
            "control_state": self.control_state,
            "transition_frame": self.transition_frame,
            "closed_boundary_points": self.closed_boundary_points.copy(),
            "error_history": list(self.error_history),
            "closure_enclosed_false_cells": self.last_closure_enclosed_false_cells,
            "closure_min_enclosed_false_cells": self.closure_min_enclosed_false_cells,
        }

    def _print_sensor_status(self):

        for drone in self.drones:

            if drone.edge_detected and drone.last_edge_point is not None:

                print(
                    f"    {drone.drone_id}: "
                    f"edge_points={drone.last_edge_count}, "
                    f"nearest_edge=("
                    f"{drone.last_edge_point[0]:.4f}, "
                    f"{drone.last_edge_point[1]:.4f})"
                )

            else:

                print(f"    {drone.drone_id}: " f"no edge detected")

    def step(self):
        """
        Execute one complete simulation timestep.

        Order:
            1. update environment;
            2. sensing;
            3. consensus;
            4. diagnostics;
            5. distributed control;
            6. drone motion.
        """

        self.frame += 1

        measurement_frame = (
            self.control_state == "mapping"
            and (self.frame - 1) % self.measure_every == 0
        )

        if self.verbose:

            if self.control_state == "lloyd":
                frame_type = "lloyd"
            else:
                frame_type = "measurement" if measurement_frame else "consensus"

            print(f"\nFrame {self.frame} " f"[{frame_type}]")
        # Environment

        self._update_environment()
        # Measurement

        if measurement_frame:

            self._start_new_measurement_trace()

            self._perform_measurement()

            self._record_measurement_trace()

            self._print_error_snapshot("  After sensing")

            if self.verbose:
                self._print_sensor_status()
        # Consensus

        if self.control_state == "mapping":
            for round_idx in range(self.consensus_rounds):

                self._exchange_consensus_messages()

                self._record_measurement_trace()

                self._print_error_snapshot(
                    f"  Consensus iteration " f"{round_idx + 1}/" f"{self.consensus_rounds}"
                )
        # Diagnostics

        error, mean_grid = self.compute_disagreement_error()

        self.error_history.append(error)

        self.mean_grid_history.append(mean_grid.copy())

        self.latest_mean_grid = mean_grid
        if self.control_state == "mapping":
            closed, boundary_points = self.is_mapped_polygon_closed(mean_grid)
            if closed:
                self._transition_to_lloyd_state(boundary_points)
        # Control

        self._apply_actions()

        if self.verbose:

            mode_summary = ", ".join(
                f"{drone.drone_id}:" f"{getattr(drone, 'last_control_mode', 'unknown')}"
                for drone in self.drones
            )

            print(
                f"  Frame summary: "
                f"global_disagreement="
                f"{error:.6f} | "
                f"modes: {mode_summary}"
            )

        return error

    def run(
        self,
        iterations,
        render_callback=None,
    ):
        """Run the simulation."""

        for _ in range(int(iterations)):

            self.step()

            if render_callback is not None:
                render_callback(self.get_visualization_data())

        self.finalize_histories()

    def finalize_histories(self):

        if self._current_measurement_trace is not None and any(
            len(values) > 0 for values in self._current_measurement_trace.values()
        ):

            self.measurement_consensus_history.append(
                {
                    drone_id: list(values)
                    for drone_id, values in self._current_measurement_trace.items()
                }
            )

        self._current_measurement_trace = None
