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

    def _interpolate_world_value(self, position):
        pos = np.asarray(position, dtype=float)
        if pos.shape != (2,) or not np.all(np.isfinite(pos)):
            return None

        x_coords = np.asarray(self.sim_map.x_coords, dtype=float)
        y_coords = np.asarray(self.sim_map.y_coords, dtype=float)
        field = np.asarray(self.world_field, dtype=float)
        if field.ndim != 2 or field.shape != (x_coords.size, y_coords.size):
            return None

        x = float(np.clip(pos[0], x_coords[0], x_coords[-1]))
        y = float(np.clip(pos[1], y_coords[0], y_coords[-1]))

        i1 = int(np.searchsorted(x_coords, x, side="right"))
        j1 = int(np.searchsorted(y_coords, y, side="right"))
        i0 = max(0, min(i1 - 1, x_coords.size - 1))
        j0 = max(0, min(j1 - 1, y_coords.size - 1))
        i1 = max(0, min(i1, x_coords.size - 1))
        j1 = max(0, min(j1, y_coords.size - 1))

        if i0 == i1 and x_coords.size > 1:
            i0 = max(0, i1 - 1)
        if j0 == j1 and y_coords.size > 1:
            j0 = max(0, j1 - 1)

        x0 = float(x_coords[i0])
        x1 = float(x_coords[i1])
        y0 = float(y_coords[j0])
        y1 = float(y_coords[j1])

        q00 = float(field[i0, j0])
        q10 = float(field[i1, j0])
        q01 = float(field[i0, j1])
        q11 = float(field[i1, j1])
        if not np.all(np.isfinite([q00, q10, q01, q11])):
            return None

        if abs(x1 - x0) <= 1e-12 and abs(y1 - y0) <= 1e-12:
            return q00
        if abs(x1 - x0) <= 1e-12:
            wy = (y - y0) / max(y1 - y0, 1e-12)
            return (1.0 - wy) * q00 + wy * q01
        if abs(y1 - y0) <= 1e-12:
            wx = (x - x0) / max(x1 - x0, 1e-12)
            return (1.0 - wx) * q00 + wx * q10

        wx = (x - x0) / (x1 - x0)
        wy = (y - y0) / (y1 - y0)
        return (
            (1.0 - wx) * (1.0 - wy) * q00
            + wx * (1.0 - wy) * q10
            + (1.0 - wx) * wy * q01
            + wx * wy * q11
        )

    def _estimate_local_gradient(self, position):
        pos = np.asarray(position, dtype=float)
        if pos.shape != (2,) or not np.all(np.isfinite(pos)):
            return None

        dx = self.sim_map.x_coords[1] - self.sim_map.x_coords[0] if len(self.sim_map.x_coords) > 1 else self.resolution
        dy = self.sim_map.y_coords[1] - self.sim_map.y_coords[0] if len(self.sim_map.y_coords) > 1 else self.resolution
        hx = max(abs(float(dx)), self.resolution, 1e-3)
        hy = max(abs(float(dy)), self.resolution, 1e-3)

        # Use short orthogonal averaging strips around the query point to smooth
        # the finite-difference estimate and reduce chattering near noisy edges.
        y_offsets = (-0.5 * hy, 0.0, 0.5 * hy)
        x_offsets = (-0.5 * hx, 0.0, 0.5 * hx)

        dcdx_samples = []
        for y_offset in y_offsets:
            c_plus = self._interpolate_world_value(pos + np.array([hx, y_offset], dtype=float))
            c_minus = self._interpolate_world_value(pos - np.array([hx, -y_offset], dtype=float))
            if c_plus is None or c_minus is None:
                return None
            dcdx_samples.append((c_plus - c_minus) / (2.0 * hx))

        dcdy_samples = []
        for x_offset in x_offsets:
            c_plus = self._interpolate_world_value(pos + np.array([x_offset, hy], dtype=float))
            c_minus = self._interpolate_world_value(pos - np.array([-x_offset, hy], dtype=float))
            if c_plus is None or c_minus is None:
                return None
            dcdy_samples.append((c_plus - c_minus) / (2.0 * hy))

        gradient = np.array([np.mean(dcdx_samples), np.mean(dcdy_samples)], dtype=float)
        if not np.all(np.isfinite(gradient)):
            return None
        return gradient

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
        position = np.array([drone.x, drone.y], dtype=float)
        local_concentration = self._interpolate_world_value(position)
        if local_concentration is None:
            return None

        gradient = self._estimate_local_gradient(position)
        normal, gradient_norm = self._normalize_vector(gradient)
        if normal is None or gradient_norm <= 1e-8:
            return None

        # The previous controller was an orbit-following law: it built the
        # normal from `position - center` and regulated a radial distance to an
        # implicit circle. That works only when the spill boundary is well
        # approximated by a center/radius geometry.
        #
        # This controller is contour-following instead. The local field
        # gradient points in the direction of steepest concentration increase,
        # so its normalized version is the contour normal. Rotating that normal
        # by 90 degrees produces a tangent direction that moves along the
        # `c(x, y) = occupancy_threshold` level set for arbitrary spill shapes.
        tangent = np.array([-normal[1], normal[0]], dtype=float)
        tangent, tangent_norm = self._normalize_vector(tangent)
        if tangent is None or tangent_norm <= 1e-12:
            return None

        concentration_error = float(local_concentration - self.occupancy_threshold)
        # Concentration error lives in field-value units, not meters, so clamp
        # it independently from the legacy radial-distance tuning constants.
        bounded_error = float(np.clip(concentration_error, -1.0, 1.0))
        tangent_scale = 1.0
        if abs(bounded_error) > 0.1:
            tangent_scale = 0.35

        normal_gain = self.k_n * bounded_error
        if bounded_error > 0.0:
            normal_gain += 0.5 * self.boundary_lock_gain * bounded_error
        else:
            normal_gain += self.boundary_lock_gain * bounded_error

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
