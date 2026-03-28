import numpy as np

from drone import Drone
from boundary_regularization import lloyd_regularize_boundary_points


class SimulationEngine:
    """Run distributed occupancy grid mapping with consensus across drones."""

    def __init__(
        self,
        sim_map,
        oil_spill,
        x_min=-10,
        x_max=10,
        y_min=-10,
        y_max=10,
        resolution=0.1,
        sensor_size=100,
        measure_every=3,
        communication_radius_cells=205,
        fully_connected=False,
        occupancy_threshold=0.5,
        temporal_alpha=0.05,
        consensus_rounds=10,
        consensus_epsilon=1e-3,
        boundary_tracking=True,
        boundary_threshold=float("nan"),
        boundary_regularization_max_points=40,
        boundary_regularization_iterations=30,
        boundary_regularization_epsilon=1e-3,
        verbose=True,
    ):
        self.sim_map = sim_map
        self.oil_spill = oil_spill
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.resolution = float(resolution)
        self.occupancy_threshold = float(occupancy_threshold)
        self.temporal_alpha = float(temporal_alpha) if temporal_alpha is not None else None
        self.sensor_size = int(sensor_size)
        self.measure_every = max(1, int(measure_every))
        self.fully_connected = bool(fully_connected)
        self.consensus_rounds = max(10, int(consensus_rounds))
        self.consensus_epsilon = float(consensus_epsilon)
        self.boundary_tracking = bool(boundary_tracking)
        self.boundary_threshold = float(boundary_threshold)
        self.boundary_regularization_max_points = int(boundary_regularization_max_points)
        self.boundary_regularization_iterations = int(boundary_regularization_iterations)
        self.boundary_regularization_epsilon = float(boundary_regularization_epsilon)
        self.verbose = bool(verbose)

        self.Nx = int((self.x_max - self.x_min) / self.resolution)
        self.Ny = int((self.y_max - self.y_min) / self.resolution)
        self.grid_shape = (self.Nx, self.Ny)
        self.grid_bounds = (self.x_min, self.x_max, self.y_min, self.y_max)

        # World field sampled on the high-resolution simulation map.
        self.world_field = oil_spill.field(sim_map.X, sim_map.Y)

        dx = sim_map.x_coords[1] - sim_map.x_coords[0] if len(sim_map.x_coords) > 1 else 1.0
        dy = sim_map.y_coords[1] - sim_map.y_coords[0] if len(sim_map.y_coords) > 1 else 1.0
        self.communication_radius_cells = int(communication_radius_cells)
        self.communication_radius = float(self.communication_radius_cells * 0.5 * (dx + dy))

        self.drones = []
        self.frame = 0
        self.error_history = []
        self.measurement_consensus_history = []
        self._current_measurement_trace = None
        self.mean_grid_history = []
        self.latest_mean_grid = np.zeros(self.grid_shape, dtype=float)
        self.latest_boundary_result = None
        self.consensus_converged = False
        self.consensus_converged_frame = None

    def add_drone(self, drone_id, x, y, gps_noise=0.1, camera_noise=0.1):
        drone = Drone(
            drone_id=drone_id,
            x=x,
            y=y,
            grid_shape=self.grid_shape,
            grid_bounds=self.grid_bounds,
            sensor_size=self.sensor_size,
            gps_noise=gps_noise,
            camera_noise=camera_noise,
        )
        # Give every drone a lightweight motion anchor around the spill so the
        # visualization shows real movement instead of static symbols.
        target_radius = float(np.hypot(x - self.oil_spill.x0, y - self.oil_spill.y0))
        drone.configure_motion(
            anchor_x=self.oil_spill.x0,
            anchor_y=self.oil_spill.y0,
            target_radius=target_radius,
        )
        self.drones.append(drone)

    def _communication_neighbors(self, drone, candidates):
        """Return drones within the communication radius or everyone if fully connected."""
        if self.fully_connected:
            return [other for other in candidates if other is not drone]

        neighbors = []
        for other in candidates:
            if other is drone:
                continue
            distance = float(np.hypot(drone.x - other.x, drone.y - other.y))
            if distance <= self.communication_radius:
                neighbors.append(other)
        return neighbors

    def _apply_synchronous_consensus(self):
        """Average each drone's grid with the grids of its current neighbors."""
        snapshot = {drone.drone_id: drone.grid.copy() for drone in self.drones}
        updated_grids = {}

        for drone in self.drones:
            neighbors = self._communication_neighbors(drone, self.drones)
            grids = [snapshot[drone.drone_id]]
            grids.extend(snapshot[neighbor.drone_id] for neighbor in neighbors)
            updated_grids[drone.drone_id] = np.mean(grids, axis=0)

        for drone in self.drones:
            drone.grid = updated_grids[drone.drone_id]

    def _sort_boundary_points(self, points):
        points = np.asarray(points, dtype=float)
        if len(points) == 0:
            return points
        center = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        order = np.argsort(angles)
        return points[order]

    def _assign_boundary_targets(self, boundary_points):
        points = self._sort_boundary_points(boundary_points)
        if len(points) == 0:
            for drone in self.drones:
                drone.clear_boundary_target()
            return []

        num_drones = len(self.drones)
        if num_drones == 0:
            return []

        indices = np.linspace(0, len(points), num=num_drones, endpoint=False, dtype=int)
        targets = []
        for drone, idx in zip(self.drones, indices):
            target = points[int(idx) % len(points)]
            drone.set_boundary_target(target[0], target[1])
            targets.append((drone.drone_id, float(target[0]), float(target[1])))
        return targets

    def update_boundary_targets(self, threshold=None):
        """Run Voronoi/Lloyd regularization on the mean grid and assign targets."""
        if not self.boundary_tracking or not self.drones:
            self.latest_boundary_result = None
            return None

        mean_grid = self.compute_mean_grid()
        result = lloyd_regularize_boundary_points(
            mean_grid,
            x_min=self.x_min,
            y_min=self.y_min,
            resolution=self.resolution,
            threshold=self.boundary_threshold if threshold is None else threshold,
            max_points=self.boundary_regularization_max_points,
            max_iterations=self.boundary_regularization_iterations,
            epsilon=self.boundary_regularization_epsilon,
            bounding_box=(self.x_min, self.x_max, self.y_min, self.y_max),
        )
        self.latest_boundary_result = result
        self._assign_boundary_targets(result["final_points"])
        return result

    def compute_mean_grid(self):
        if not self.drones:
            return np.zeros(self.grid_shape, dtype=float)
        return sum(drone.grid for drone in self.drones) / float(len(self.drones))

    def compute_disagreement_error(self):
        """Compute the mandatory consensus disagreement metric."""
        if not self.drones:
            return 0.0, np.zeros(self.grid_shape, dtype=float)

        mean_grid = self.compute_mean_grid()
        error = sum(np.linalg.norm(drone.grid - mean_grid) for drone in self.drones) / float(len(self.drones))
        return float(error), mean_grid

    def _drone_error_snapshot(self):
        """Return the disagreement error of each drone against the current mean grid."""
        mean_grid = self.compute_mean_grid()
        return {
            drone.drone_id: float(np.linalg.norm(drone.grid - mean_grid))
            for drone in self.drones
        }

    def _print_error_snapshot(self, header):
        if not self.verbose:
            return

        snapshot = self._drone_error_snapshot()
        mean_error = float(np.mean(list(snapshot.values()))) if snapshot else 0.0
        max_error = float(np.max(list(snapshot.values()))) if snapshot else 0.0
        ordered = ", ".join(f"{drone_id}={value:.6f}" for drone_id, value in snapshot.items())
        print(f"{header} | mean_error={mean_error:.6f} | max_error={max_error:.6f}")
        print(f"    per-drone: {ordered}")

    def _start_new_measurement_trace(self):
        if self._current_measurement_trace is not None:
            self.measurement_consensus_history.append(
                {drone_id: list(values) for drone_id, values in self._current_measurement_trace.items()}
            )

        self._current_measurement_trace = {drone.drone_id: [] for drone in self.drones}

    def _record_measurement_trace(self):
        if self._current_measurement_trace is None:
            self._current_measurement_trace = {drone.drone_id: [] for drone in self.drones}

        snapshot = self._drone_error_snapshot()
        for drone_id, value in snapshot.items():
            self._current_measurement_trace[drone_id].append(value)

    def step(self):
        """
        One full iteration:
        1. Sense edge points in global coordinates.
        2. Update each local occupancy grid.
        3. Exchange full grids and run consensus.
        4. Store disagreement error.
        """
        self.frame += 1
        measurement_frame = ((self.frame - 1) % self.measure_every) == 0

        if self.verbose:
            frame_kind = "measurement" if measurement_frame else "consensus"
            print(f"\nFrame {self.frame} [{frame_kind}]")

        if measurement_frame:
            # A fresh sensor update can change the occupancy grids, so any
            # previously reached consensus is no longer guaranteed to hold.
            self.consensus_converged = False
            self.consensus_converged_frame = None

        # Move the drones a little before sensing so the animation and the
        # measured edge points update over time.
        for drone in self.drones:
            drone.move(
                x_min=self.x_min,
                x_max=self.x_max,
                y_min=self.y_min,
                y_max=self.y_max,
            )

        # Only every `measure_every` frames do the drones refresh their sensor
        # evidence. In between, they keep exchanging the current full grids.
        if measurement_frame:
            self._start_new_measurement_trace()
            for drone in self.drones:
                edge_points = drone.sense(
                    self.world_field,
                    self.sim_map.x_coords,
                    self.sim_map.y_coords,
                    occupancy_threshold=self.occupancy_threshold,
                )
                drone.update_grid(
                    edge_points,
                    self.x_min,
                    self.y_min,
                    self.resolution,
                    alpha=self.temporal_alpha,
                )

            # Capture the disagreement right after the new measurement has been
            # fused locally.
            self._record_measurement_trace()
            self._print_error_snapshot("  After sensing")

            if self.verbose:
                for drone in self.drones:
                    if drone.edge_detected and drone.last_edge_point is not None:
                        print(
                            f"    {drone.drone_id}: edge_points={drone.last_edge_count}, "
                            f"nearest_edge=({drone.last_edge_point[0]:.4f}, {drone.last_edge_point[1]:.4f})"
                        )
                    else:
                        print(f"    {drone.drone_id}: no edge detected")

        # Distributed consensus over the full occupancy grids.
        if self.consensus_converged and not measurement_frame:
            if self.verbose:
                print(
                    "  Consensus already converged in a previous iteration; "
                    "skipping consensus rounds for this frame."
                )
        else:
            for round_idx in range(self.consensus_rounds):
                self._apply_synchronous_consensus()
                self._record_measurement_trace()
                error, _ = self.compute_disagreement_error()
                self._print_error_snapshot(f"  Consensus iteration {round_idx + 1}/{self.consensus_rounds}")

                if error <= self.consensus_epsilon:
                    self.consensus_converged = True
                    self.consensus_converged_frame = self.frame
                    if self.verbose:
                        print(
                            "  Consensus threshold reached "
                            f"(error={error:.6f} <= epsilon={self.consensus_epsilon:.6f}); "
                            "skipping remaining rounds."
                        )
                    break
            else:
                self.consensus_converged = False
                self.consensus_converged_frame = None

        # Mandatory convergence metric.
        error, mean_grid = self.compute_disagreement_error()
        self.error_history.append(error)
        self.mean_grid_history.append(mean_grid.copy())
        self.latest_mean_grid = mean_grid
        if self.boundary_tracking:
            self.update_boundary_targets()

        if self.verbose:
            print(f"  Frame summary: global_disagreement={error:.6f}")

        return error

    def run(self, iterations, render_callback=None):
        """Run several iterations and optionally render after each step."""
        for _ in range(int(iterations)):
            self.step()
            if render_callback is not None:
                render_callback(self.drones)

        self.finalize_histories()

    def finalize_histories(self):
        """Flush the last measurement cycle into the saved history."""
        if self._current_measurement_trace is not None and any(
            len(values) > 0 for values in self._current_measurement_trace.values()
        ):
            self.measurement_consensus_history.append(
                {drone_id: list(values) for drone_id, values in self._current_measurement_trace.items()}
            )
        self._current_measurement_trace = None
