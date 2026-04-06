import numpy as np

from drone import Drone


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

        # Motion-control parameters for the hybrid boundary-following policy.
        self.max_speed = 0.12
        self.exploration_speed = 0.08
        self.k_t = 1.0
        self.k_n = 1.5
        self.boundary_tracking_oil_fraction = 0.5
        self.boundary_lock_distance = max(0.15, 2.0 * self.resolution)
        self.boundary_lock_gain = 3.0
        self.boundary_error_cap = max(0.15, 3.0 * self.resolution)

        self.drones = []
        self.frame = 0
        self.error_history = []
        self.measurement_consensus_history = []
        self._current_measurement_trace = None
        self.mean_grid_history = []
        self.latest_mean_grid = np.zeros(self.grid_shape, dtype=float)

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

    @staticmethod
    def _normalize_vector(vector):
        vec = np.asarray(vector, dtype=float)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            return None, norm
        return vec / norm, norm

    def _clip_command(self, command, max_speed):
        vec = np.asarray(command, dtype=float)
        speed = float(np.linalg.norm(vec))
        if speed <= 1e-12:
            return np.zeros(2, dtype=float)
        if speed > max_speed:
            vec = vec * (max_speed / speed)
        return vec

    @staticmethod
    def _random_unit_direction():
        angle = np.random.uniform(0.0, 2.0 * np.pi)
        return np.array([np.cos(angle), np.sin(angle)], dtype=float)

    @staticmethod
    def _principal_tangent(points):
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 3:
            return None, None, None

        centroid = np.mean(pts, axis=0)
        centered = pts - centroid
        cov = np.cov(centered.T)
        if not np.all(np.isfinite(cov)):
            return None, None, None

        vals, vecs = np.linalg.eigh(cov)
        tangent = vecs[:, int(np.argmax(vals))]
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-12:
            return None, None, None
        tangent = tangent / tangent_norm
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-12:
            return None, None, None
        normal = normal / normal_norm
        return centroid, tangent, normal

    def _bounce_exploration_command(self, drone):
        direction = np.asarray(getattr(drone, "exploration_direction", None), dtype=float)
        if direction.size != 2:
            direction = self._random_unit_direction()
        else:
            direction, norm = self._normalize_vector(direction)
            if direction is None:
                direction = self._random_unit_direction()

        command = direction * self.exploration_speed
        next_x = drone.x + command[0]
        next_y = drone.y + command[1]
        bounced = False

        if next_x < self.sim_map.xlim[0] or next_x > self.sim_map.xlim[1]:
            direction[0] *= -1.0
            bounced = True
        if next_y < self.sim_map.ylim[0] or next_y > self.sim_map.ylim[1]:
            direction[1] *= -1.0
            bounced = True

        if bounced:
            direction, _ = self._normalize_vector(direction)
            if direction is None:
                direction = self._random_unit_direction()
            command = direction * self.exploration_speed
        drone.exploration_direction = direction
        return command

    def _boundary_tracking_command(self, drone):
        points = getattr(drone, "last_nls_points", None)
        centroid = None
        tangent = None
        normal = None

        if points is not None:
            centroid, tangent, normal = self._principal_tangent(points)

        if centroid is None or tangent is None or normal is None:
            target_point = getattr(drone, "last_boundary_anchor_point", None)
            if target_point is None:
                target_point = drone.last_edge_point
            if target_point is None:
                return None

            edge_point = np.asarray(target_point, dtype=float)
            position = np.array([drone.x, drone.y], dtype=float)
            offset = position - edge_point
            normal, distance = self._normalize_vector(offset)
            if normal is None:
                return None

            tangent = np.array([normal[1], -normal[0]], dtype=float)
            oil_fraction = drone.last_oil_fraction
            if oil_fraction is None or not np.isfinite(oil_fraction):
                oil_fraction = self.boundary_tracking_oil_fraction

            distance_error = float(np.clip(distance - self.boundary_lock_distance, -self.boundary_error_cap, self.boundary_error_cap))
            tangent_scale = np.clip(self.boundary_lock_distance / max(distance, self.boundary_lock_distance), 0.35, 1.0)
            normal_gain = self.k_n * (float(oil_fraction) - self.boundary_tracking_oil_fraction)
            normal_gain += self.boundary_lock_gain * distance_error

            command = (self.k_t * tangent_scale) * tangent - normal_gain * normal
            return self._clip_command(command, self.max_speed)

        position = np.array([drone.x, drone.y], dtype=float)
        rel = position - centroid
        signed_distance = float(np.dot(rel, normal))
        if signed_distance < 0.0:
            normal = -normal
            signed_distance = -signed_distance

        # Keep a persistent tangent orientation by matching the previous command
        # when possible. This avoids frame-to-frame flipping along the boundary.
        prev = np.asarray(getattr(drone, "last_control_vector", np.zeros(2, dtype=float)), dtype=float)
        if float(np.linalg.norm(prev)) > 1e-12 and float(np.dot(prev, tangent)) < 0.0:
            tangent = -tangent

        oil_fraction = drone.last_oil_fraction
        if oil_fraction is None or not np.isfinite(oil_fraction):
            oil_fraction = self.boundary_tracking_oil_fraction

        distance_error = float(np.clip(signed_distance - self.boundary_lock_distance, -self.boundary_error_cap, self.boundary_error_cap))
        tangent_scale = np.clip(self.boundary_lock_distance / max(signed_distance, self.boundary_lock_distance), 0.35, 1.0)
        normal_gain = self.k_n * (float(oil_fraction) - self.boundary_tracking_oil_fraction)
        normal_gain += self.boundary_lock_gain * distance_error

        command = (self.k_t * tangent_scale) * tangent - normal_gain * normal
        return self._clip_command(command, self.max_speed)

    def _reacquire_command(self, drone):
        target_point = getattr(drone, "last_boundary_anchor_point", None)
        if target_point is None:
            target_point = drone.last_edge_point
        if target_point is None:
            return None

        position = np.array([drone.x, drone.y], dtype=float)
        direction, norm = self._normalize_vector(np.asarray(target_point, dtype=float) - position)
        if direction is None:
            return None

        command = direction * self.exploration_speed
        return self._clip_command(command, self.max_speed)

    def _select_motion_command(self, drone):
        if drone.edge_detected and drone.last_edge_point is not None:
            command = self._boundary_tracking_command(drone)
            if command is not None:
                drone.last_control_mode = "boundary_tracking"
                return command

        if not drone.edge_detected and drone.last_edge_point is not None:
            command = self._reacquire_command(drone)
            if command is not None:
                drone.last_control_mode = "reacquire"
                return command

        drone.last_control_mode = "explore"
        return self._bounce_exploration_command(drone)

    def _apply_motion(self):
        for drone in self.drones:
            command = self._select_motion_command(drone)
            command = self._clip_command(command, self.max_speed)
            drone.last_control_vector = np.asarray(command, dtype=float)
            drone.x = float(np.clip(drone.x + command[0], self.sim_map.xlim[0], self.sim_map.xlim[1]))
            drone.y = float(np.clip(drone.y + command[1], self.sim_map.ylim[0], self.sim_map.ylim[1]))

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
        for round_idx in range(self.consensus_rounds):
            self._apply_synchronous_consensus()
            self._record_measurement_trace()
            self._print_error_snapshot(f"  Consensus iteration {round_idx + 1}/{self.consensus_rounds}")

        # Mandatory convergence metric.
        error, mean_grid = self.compute_disagreement_error()
        self.error_history.append(error)
        self.mean_grid_history.append(mean_grid.copy())
        self.latest_mean_grid = mean_grid

        if self.verbose:
            mode_summary = ", ".join(f"{d.drone_id}:{d.last_control_mode}" for d in self.drones)
            print(f"  Frame summary: global_disagreement={error:.6f} | modes: {mode_summary}")

        self._apply_motion()

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
