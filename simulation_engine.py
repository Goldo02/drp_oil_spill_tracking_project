import numpy as np
import matplotlib.pyplot as plt
from drone import Drone
from edge_detection import detect_edges, extract_edge_points


class SimulationEngine:
    """Orchestrates radius estimation, Voronoi partitioning, and control."""

    def __init__(
        self,
        sim_map,
        oil_spill,
        dt=0.1,
        sigma_gps=0.1,
        sigma_cam=0.1,
        measure_every=3,
        consensus_iters=10,
        communication_radius_cells=205,
        fully_connected=False,
        control_gain=1.8,
        max_speed=0.6,
        voronoi_grid_step=8,
        boundary_tangent_band=0.35,
    ):
        self.sim_map = sim_map
        self.oil_spill = oil_spill
        self.dt = dt
        self.sigma_gps = sigma_gps
        self.sigma_cam = sigma_cam
        self.drones = []
        self.frame = 0
        # Center is known in this experiment.
        self.true_x0 = oil_spill.x0
        self.true_y0 = oil_spill.y0

        # Pre-computed oil field sampled on the simulation grid.
        self.world_field = oil_spill.field(sim_map.X, sim_map.Y)

        # Consensus parameters for distributed averaging.
        self.alpha = 0.4  # Consensus gain (neighbor weight)
        self.gamma = 0.2  # Innovation gain (local measure weight)
        self.measure_every = max(1, int(measure_every))
        self.consensus_iters = max(1, int(consensus_iters))
        self.canny_threshold1 = 20
        self.canny_threshold2 = 60
        self.consensus_oil_fraction_threshold = 0.10
        self.oil_cell_threshold = 0.5
        self.fully_connected = bool(fully_connected)
        self.control_gain = float(control_gain)
        self.max_speed = float(max_speed)
        self.voronoi_grid_step = max(1, int(voronoi_grid_step))
        self.boundary_tangent_band = float(boundary_tangent_band)
        self.boundary_detection_oil_fraction_min = 0.05
        self.exploration_oil_fraction_threshold = 0.95

        dx = self.sim_map.x_coords[1] - self.sim_map.x_coords[0] if len(self.sim_map.x_coords) > 1 else 1.0
        dy = self.sim_map.y_coords[1] - self.sim_map.y_coords[0] if len(self.sim_map.y_coords) > 1 else 1.0
        self.communication_radius_cells = int(communication_radius_cells)
        self.communication_radius = float(self.communication_radius_cells * 0.5 * (dx + dy))

        self.control_x_coords = self._sample_axis(self.sim_map.x_coords, self.voronoi_grid_step)
        self.control_y_coords = self._sample_axis(self.sim_map.y_coords, self.voronoi_grid_step)
        self.control_X, self.control_Y = np.meshgrid(self.control_x_coords, self.control_y_coords)
        self.control_points = np.column_stack((self.control_X.ravel(), self.control_Y.ravel()))
        # No ground-truth radius is used in control; this is kept as a neutral fallback.
        self.control_density = np.zeros_like(self.control_X, dtype=float).ravel()

        # History for plotting local measurements and consensus convergence.
        self.estimates_history = {}
        self.measurement_consensus_history = []
        self._current_measure_trace = None

    def _sample_axis(self, coords, step):
        sampled = np.asarray(coords[::step], dtype=float)
        if sampled.size == 0 or sampled[-1] != coords[-1]:
            sampled = np.append(sampled, float(coords[-1]))
        return sampled

    def _boundary_density(self, X, Y, x0=None, y0=None, r0=None):
        """Weight the Voronoi centroids toward the spill boundary."""
        if x0 is None:
            x0 = self.oil_spill.x0
        if y0 is None:
            y0 = self.oil_spill.y0

        if r0 is None:
            return np.zeros_like(X, dtype=float)

        dist = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
        sigma = float(getattr(self.oil_spill, "sigma", 0.5))
        field = np.where(
            dist <= r0,
            1.0,
            np.exp(-((dist - r0) ** 2) / (2 * sigma**2)),
        )
        return np.clip(4.0 * field * (1.0 - field), 0.0, None)

    def add_drone(self, drone_id, x, y):
        drone = Drone(
            drone_id,
            x,
            y,
            map_bounds=(*self.sim_map.xlim, *self.sim_map.ylim),
            gps_noise=self.sigma_gps,
            camera_noise=self.sigma_cam,
            true_x0=self.true_x0,
            true_y0=self.true_y0,
            max_speed=self.max_speed,
        )
        # Ensure exploration has significant X and Y components (angle not too close to axes)
        while True:
            angle = float(np.random.uniform(0.0, 2.0 * np.pi))
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
            if abs(direction[0]) > 0.3 and abs(direction[1]) > 0.3:
                break
        drone.exploration_direction = direction
        drone.exploration_speed = float(0.8 * self.max_speed)
        self.drones.append(drone)
        self.estimates_history[drone_id] = {
            "x": [],
            "y": [],
            "x0": [],
            "y0": [],
            "edge_x": [],
            "edge_y": [],
            "edge_detected": [],
            "gradient_magnitude": [],
            "gradient_peak": [],
            "oil_fraction": [],
            "r0_local": [],
            "r0_measure_start": [],
            "r0_measure_end": [],
            "r0_consensus": [],
            "r0_post": [],
            "u_x": [],
            "u_y": [],
            "voronoi_cx": [],
            "voronoi_cy": [],
            "exploration_x": [],
            "exploration_y": [],
        }

    def _bounce_exploration_command(self, drone):
        """Return a persistent exploration command reflected at map bounds."""
        direction = getattr(drone, "exploration_direction", None)
        speed = float(getattr(drone, "exploration_speed", 0.0))
        if direction is None or speed <= 0.0:
            while True:
                angle = float(np.random.uniform(0.0, 2.0 * np.pi))
                direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
                if abs(direction[0]) > 0.3 and abs(direction[1]) > 0.3:
                    break
            drone.exploration_direction = direction
            drone.exploration_speed = float(0.8 * self.max_speed)
            speed = float(drone.exploration_speed)

        direction = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            direction = np.array([1.0, 0.0], dtype=float)
            norm = 1.0
        direction = direction / norm

        command = direction * speed
        xmin, xmax, ymin, ymax = drone.map_bounds
        next_x = float(drone.x + command[0] * self.dt)
        next_y = float(drone.y + command[1] * self.dt)

        bounced = False
        if next_x < xmin or next_x > xmax:
            direction[0] *= -1.0
            command[0] *= -1.0
            bounced = True
        if next_y < ymin or next_y > ymax:
            direction[1] *= -1.0
            command[1] *= -1.0
            bounced = True

        if bounced:
            drone.exploration_direction = direction / max(float(np.linalg.norm(direction)), 1e-12)

        return command

    def _communication_neighbors(self, drone, candidates):
        """Return candidate drones within the communication radius or everyone in fully-connected mode."""
        if self.fully_connected:
            return list(candidates)

        neighbors = []
        for other in candidates:
            distance = float(np.hypot(drone.x - other.x, drone.y - other.y))
            if distance <= self.communication_radius:
                neighbors.append(other)
        return neighbors

    def _estimate_local_radius(self, drone):
        """Estimate the spill radius only when a meaningful edge is detected."""
        camera_view = drone.get_camera_view(
            self.world_field,
            self.sim_map.x_coords,
            self.sim_map.y_coords,
        )
        if camera_view.size == 0:
            drone.edge_detected = False
            drone.last_gradient_peak = None
            drone.last_oil_fraction = None
            drone.last_r0_local = None
            return {
                "edge_x": np.nan,
                "edge_y": np.nan,
                "edge_detected": False,
                "gradient_magnitude": np.nan,
                "gradient_peak": np.nan,
                "oil_fraction": np.nan,
                "r0_local_raw": np.nan,
                "r0_local": np.nan,
            }

        oil_fraction = float(np.mean(camera_view >= self.oil_cell_threshold))
        if (
            oil_fraction <= self.boundary_detection_oil_fraction_min
            or oil_fraction >= self.exploration_oil_fraction_threshold
        ):
            drone.edge_detected = False
            drone.last_gradient_peak = 0.0
            drone.last_oil_fraction = oil_fraction
            drone.last_r0_local = None
            return {
                "edge_x": np.nan,
                "edge_y": np.nan,
                "edge_detected": False,
                "gradient_magnitude": 0.0,
                "gradient_peak": 0.0,
                "oil_fraction": oil_fraction,
                "r0_local_raw": np.nan,
                "r0_local": np.nan,
            }

        # Rebuild the local world-coordinate frame around the drone position.
        dx = self.sim_map.x_coords[1] - self.sim_map.x_coords[0] if len(self.sim_map.x_coords) > 1 else 1.0
        dy = self.sim_map.y_coords[1] - self.sim_map.y_coords[0] if len(self.sim_map.y_coords) > 1 else 1.0
        i_center = int((drone.x - self.sim_map.x_coords[0]) / dx)
        j_center = int((drone.y - self.sim_map.y_coords[0]) / dy)
        half = camera_view.shape[0] // 2

        i_min = max(0, i_center - half)
        i_max = min(len(self.sim_map.x_coords), i_center + half + 1)
        j_min = max(0, j_center - half)
        j_max = min(len(self.sim_map.y_coords), j_center + half + 1)

        local_x_coords = self.sim_map.x_coords[i_min:i_max]
        local_y_coords = self.sim_map.y_coords[j_min:j_max]

        # If the camera footprint is clipped by the map boundary, zero padding
        # can create artificial edges. In that case we skip edge estimation and
        # keep the drone in exploration mode.
        if (
            i_min == 0
            or j_min == 0
            or i_max == len(self.sim_map.x_coords)
            or j_max == len(self.sim_map.y_coords)
        ):
            drone.edge_detected = False
            drone.last_gradient_peak = 0.0
            drone.last_oil_fraction = oil_fraction
            drone.last_r0_local = None
            return {
                "edge_x": np.nan,
                "edge_y": np.nan,
                "edge_detected": False,
                "gradient_magnitude": 0.0,
                "gradient_peak": 0.0,
                "oil_fraction": oil_fraction,
                "r0_local_raw": np.nan,
                "r0_local": np.nan,
            }

        # Smooth the drone's sensing matrix first, then apply Canny.
        edges = detect_edges(
            camera_view,
            threshold1=self.canny_threshold1,
            threshold2=self.canny_threshold2,
            blur_kernel=(5, 5),
            blur_sigma=1.2,
        )
        edge_pixels = extract_edge_points(edges)

        if edge_pixels.size == 0:
            drone.edge_detected = False
            drone.last_gradient_peak = None
            drone.last_oil_fraction = None
            drone.last_r0_local = None
            return {
                "edge_x": np.nan,
                "edge_y": np.nan,
                "edge_detected": False,
                "gradient_magnitude": np.nan,
                "gradient_peak": np.nan,
                "oil_fraction": np.nan,
                "r0_local_raw": np.nan,
                "r0_local": np.nan,
            }

        edge_rows = edge_pixels[:, 0]
        edge_cols = edge_pixels[:, 1]

        # The camera matrix is stored with the first axis aligned to x and the
        # second axis aligned to y, so rows map to x and cols map to y here.
        world_x = local_x_coords[np.clip(edge_rows, 0, len(local_x_coords) - 1)]
        world_y = local_y_coords[np.clip(edge_cols, 0, len(local_y_coords) - 1)]

        # Keep the centroid for estimation, but store the closest detected edge
        # point to the drone so the visualizer can highlight it directly.
        distances = np.hypot(world_x - drone.x, world_y - drone.y)
        nearest_idx = int(np.argmin(distances))
        nearest_edge_point = (float(world_x[nearest_idx]), float(world_y[nearest_idx]))

        edge_x = float(np.mean(world_x))
        edge_y = float(np.mean(world_y))
        center_x = float(drone.estimate_x0)
        center_y = float(drone.estimate_y0)
        raw_r0_local = float(np.mean(np.hypot(world_x - center_x, world_y - center_y)))
        r0_local = raw_r0_local

        drone.edge_detected = True
        drone.has_radius_estimate = True
        drone.last_edge_point = nearest_edge_point
        drone.last_gradient_peak = float(np.count_nonzero(edges))
        drone.last_oil_fraction = oil_fraction
        drone.last_r0_local = r0_local
        drone.estimate_r0 = r0_local

        # Sync with consensus required attributes
        drone.has_measure = True
        drone.local_measure = r0_local

        return {
            "edge_x": edge_x,
            "edge_y": edge_y,
            "edge_detected": True,
            "gradient_magnitude": float(np.count_nonzero(edges)),
            "gradient_peak": float(np.count_nonzero(edges)),
            "oil_fraction": oil_fraction,
            "r0_local_raw": raw_r0_local,
            "r0_local": r0_local,
        }

    def _compute_voronoi_targets(self):
        """Compute one weighted Voronoi centroid per drone on a sampled grid.

        Each drone uses its own post-consensus radius estimate to weight the
        centroid of its Voronoi cell.
        """
        if not self.drones:
            return {}

        positions = np.array([[d.x, d.y] for d in self.drones], dtype=float)
        points = self.control_points
        diff = points[:, None, :] - positions[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        assignment = np.argmin(dist2, axis=1)

        targets = {}
        for idx, drone in enumerate(self.drones):
            mask = assignment == idx
            if not np.any(mask):
                targets[drone.drone_id] = np.array([drone.x, drone.y], dtype=float)
                continue

            cell_points = points[mask]
            r0_est = getattr(drone, "estimate_r0", None)
            if r0_est is None:
                r0_est = 1.0
            r0_est = float(r0_est)
            if not np.isfinite(r0_est) or r0_est <= 0.0:
                r0_est = 1.0

            cell_weights = self._boundary_density(
                self.control_X,
                self.control_Y,
                x0=float(getattr(drone, "estimate_x0", self.true_x0)),
                y0=float(getattr(drone, "estimate_y0", self.true_y0)),
                r0=r0_est,
            ).ravel()[mask]
            weight_sum = float(np.sum(cell_weights))
            if weight_sum <= 1e-12:
                centroid = np.mean(cell_points, axis=0)
            else:
                centroid = np.sum(cell_points * cell_weights[:, None], axis=0) / weight_sum
            targets[drone.drone_id] = centroid

        return targets

    def _tangential_boundary_command(self, command, drone, gps_pos):
        """Remove the radial component when the drone is close to the boundary.

        The boundary normal is approximated from the spill center to the drone
        position, which keeps the motion tangent to a circular spill boundary.
        """
        if not getattr(drone, "edge_detected", False):
            return command, False

        edge_point = getattr(drone, "last_edge_point", None)
        if edge_point is None:
            return command, False

        edge_point = np.asarray(edge_point, dtype=float)
        if not np.all(np.isfinite(edge_point)):
            return command, False

        boundary_error = edge_point - gps_pos
        distance_to_edge = float(np.linalg.norm(boundary_error))
        if distance_to_edge > self.boundary_tangent_band:
            return command, False

        center = np.array([float(getattr(drone, "estimate_x0", self.true_x0)), float(getattr(drone, "estimate_y0", self.true_y0))], dtype=float)
        normal = gps_pos - center
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-12:
            normal = edge_point - center
            normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-12:
            return command, False

        normal_unit = normal / normal_norm
        tangential_command = command - np.dot(command, normal_unit) * normal_unit
        tangential_speed = float(np.linalg.norm(tangential_command))
        if tangential_speed <= 1e-12:
            return np.zeros_like(command), True
        return tangential_command, True

    def _apply_voronoi_control(self):
        """Single-integrator motion toward the weighted Voronoi centroid.

        Each drone uses its own estimated radius to build the weighted Voronoi
        target. When the drone is near the detected boundary, the command is
        projected onto the tangent direction so it moves along the contour
        instead of cutting radially through it.
        """
        targets = self._compute_voronoi_targets()
        for drone in self.drones:
            if getattr(drone, "has_radius_estimate", False):
                target = np.asarray(targets.get(drone.drone_id, (drone.x, drone.y)), dtype=float)
                gps_pos = np.asarray(drone.get_gps_pos(), dtype=float)
                command = self.control_gain * (target - gps_pos)
                command, tangentialized = self._tangential_boundary_command(command, drone, gps_pos)
                drone.last_boundary_tangential = bool(tangentialized)
                drone.last_control_mode = "voronoi"
                drone.last_voronoi_target = (float(target[0]), float(target[1]))
                drone.last_exploration_target = None
                speed = float(np.linalg.norm(command))
                if self.max_speed > 0 and speed > self.max_speed:
                    command = command / speed * self.max_speed
            else:
                command = self._bounce_exploration_command(drone)
                drone.last_exploration_target = (float(drone.x + command[0]), float(drone.y + command[1]))
                drone.last_voronoi_target = None
                drone.last_boundary_tangential = False
                drone.last_control_mode = "explore"

            drone.set_control(float(command[0]), float(command[1]))
            drone.update_position(self.dt, map_bounds=drone.map_bounds, max_speed=self.max_speed)

            history = self.estimates_history[drone.drone_id]
            history["u_x"].append(float(drone.u_x))
            history["u_y"].append(float(drone.u_y))
            if getattr(drone, "has_radius_estimate", False):
                history["voronoi_cx"].append(float(drone.last_voronoi_target[0]))
                history["voronoi_cy"].append(float(drone.last_voronoi_target[1]))
            else:
                history["voronoi_cx"].append(np.nan)
                history["voronoi_cy"].append(np.nan)
            history["exploration_x"].append(
                float(drone.last_exploration_target[0]) if drone.last_exploration_target is not None else np.nan
            )
            history["exploration_y"].append(
                float(drone.last_exploration_target[1]) if drone.last_exploration_target is not None else np.nan
            )
            history["x"].append(float(drone.x))
            history["y"].append(float(drone.y))


    def _run_consensus(self):
        """
        Unified consensus step: all drones update estimates from neighbors.
        Drones with local measures also add an innovation term.
        """
        # 1. First, save current estimates to avoid semi-sequential updates in the same loop
        current_estimates = {d.drone_id: d.estimate_r0 for d in self.drones}
        
        new_estimates = {}
        max_diff = 0.0

        for drone in self.drones:
            # Calculate media_vicini (neighbor average)
            if not drone.neighbors:
                media_vicini = drone.estimate_r0
            else:
                neighbor_vals = [current_estimates[n.drone_id] for n in drone.neighbors]
                media_vicini = np.mean(neighbor_vals)
            
            # Formula: x_i = x_i + alpha * (media_vicini - x_i) + gamma * (local_measure - x_i)
            # The gamma term only if the drone has measure.
            
            innovation_term = 0.0
            if drone.has_measure:
                innovation_term = self.gamma * (drone.local_measure - drone.estimate_r0)
            
            consensus_term = self.alpha * (media_vicini - drone.estimate_r0)
            
            new_estimate = drone.estimate_r0 + consensus_term + innovation_term
            new_estimates[drone.drone_id] = new_estimate
            
            diff = abs(new_estimate - drone.estimate_r0)
            if diff > max_diff:
                max_diff = diff

        # 2. Update all drones
        for drone in self.drones:
            drone.estimate_r0 = new_estimates[drone.drone_id]

        return 1, max_diff

    def _update_communication_topology(self):
        """Update the neighbors list for each drone based on current positions."""
        for drone in self.drones:
            drone.neighbors = self._communication_neighbors(drone, [d for d in self.drones if d != drone])

    def step(self):
        """Sense the spill edge, estimate radii, run consensus, then move the drones."""
        self.frame += 1

        measurement_frame = (self.frame % self.measure_every == 0)
        if measurement_frame:
            print(f"Frame {self.frame} - MEASUREMENT FRAME:")
        else:
            print(f"Frame {self.frame} - CONSENSUS ONLY FRAME:")

        # 1. Local edge detection and radius estimation, performed only on
        # measurement frames.
        for drone in self.drones:
            # Reset measure flag for this step
            drone.has_measure = False
            
            history = self.estimates_history[drone.drone_id]
            history["x0"].append(drone.estimate_x0)
            history["y0"].append(drone.estimate_y0)

            if measurement_frame:
                local_estimate = self._estimate_local_radius(drone)
                history["edge_x"].append(local_estimate["edge_x"])
                history["edge_y"].append(local_estimate["edge_y"])
                history["edge_detected"].append(local_estimate["edge_detected"])
                history["gradient_magnitude"].append(local_estimate["gradient_magnitude"])
                history["gradient_peak"].append(local_estimate["gradient_peak"])
                history["oil_fraction"].append(local_estimate["oil_fraction"])
                history["r0_local"].append(local_estimate["r0_local"])
                history["r0_measure_start"].append(float(drone.estimate_r0))

                if local_estimate["edge_detected"]:
                    print(
                        f"{drone.drone_id}: edge detected, "
                        f"grad={local_estimate['gradient_magnitude']:.6f}, "
                        f"oil={local_estimate['oil_fraction']:.3f}, "
                        f"raw r0 = {local_estimate['r0_local_raw']:.6f}"
                    )
                else:
                    print(
                        f"{drone.drone_id}: no edge detected, "
                        f"grad={local_estimate['gradient_magnitude']:.6f}, "
                        f"oil={local_estimate['oil_fraction']:.3f}"
                    )
            else:
                history["edge_x"].append(history["edge_x"][-1] if history["edge_x"] else np.nan)
                history["edge_y"].append(history["edge_y"][-1] if history["edge_y"] else np.nan)
                history["edge_detected"].append(history["edge_detected"][-1] if history["edge_detected"] else False)
                history["gradient_magnitude"].append(
                    history["gradient_magnitude"][-1] if history["gradient_magnitude"] else np.nan
                )
                history["gradient_peak"].append(history["gradient_peak"][-1] if history["gradient_peak"] else np.nan)
                history["oil_fraction"].append(history["oil_fraction"][-1] if history["oil_fraction"] else np.nan)
                history["r0_local"].append(history["r0_local"][-1] if history["r0_local"] else np.nan)
                print(f"{drone.drone_id}: sensing skipped, carrying previous measurement")

        if measurement_frame:
            self._current_measure_trace = {
                drone.drone_id: [float(drone.estimate_r0)] for drone in self.drones
            }
            start_snapshot = ", ".join(f"{d.drone_id}={d.estimate_r0:.6f}" for d in self.drones)
            print(f"Measurement start radii: {start_snapshot}")

        # 2. Consensus over the local estimates.
        self._update_communication_topology()
        
        max_diff = 0.0
        for iter_idx in range(self.consensus_iters):
            _, diff = self._run_consensus()
            max_diff = max(max_diff, diff)

        # Record the state after the active consensus step.
        for drone in self.drones:
            self.estimates_history[drone.drone_id]["r0_consensus"].append(drone.estimate_r0)

        print(f"AFTER CONSENSUS ({self.consensus_iters} iters/frame, max_diff={max_diff:.2e}):")
        for drone in self.drones:
            self.estimates_history[drone.drone_id]["r0_post"].append(drone.estimate_r0)
            if measurement_frame:
                self.estimates_history[drone.drone_id]["r0_measure_end"].append(float(drone.estimate_r0))
            print(f"{drone.drone_id}: agreed r0 = {drone.estimate_r0:.6f}")

        print("FRAME RADIUS ESTIMATES:")
        for drone in self.drones:
            print(f"{drone.drone_id}: r0 = {drone.estimate_r0:.6f}")

        self._apply_voronoi_control()
        print("CONTROL:")
        for drone in self.drones:
            mode = getattr(drone, "last_control_mode", None)
            if mode is None:
                mode = "voronoi" if getattr(drone, "has_radius_estimate", False) else "explore"
            target = drone.last_voronoi_target if mode == "voronoi" else drone.last_exploration_target
            tangential = "tangent" if getattr(drone, "last_boundary_tangential", False) else "none"
            print(
                f"{drone.drone_id}: pos=({drone.x:.3f}, {drone.y:.3f}), "
                f"u=({drone.u_x:.3f}, {drone.u_y:.3f}), "
                f"mode={mode}, target={target}, boundary_mode={tangential}"
            )

        if measurement_frame:
            if self._current_measure_trace is not None:
                self.measurement_consensus_history.append(
                    {drone_id: list(values) for drone_id, values in self._current_measure_trace.items()}
                )
                self._current_measure_trace = None
            end_snapshot = ", ".join(f"{d.drone_id}={d.estimate_r0:.6f}" for d in self.drones)
            print(f"Measurement end radii: {end_snapshot}")
