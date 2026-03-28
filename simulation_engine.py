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
        # Measurements and consensus now run at different cadences.
        self.consensus_gain = 0.5
        self.neighbor_gain = 0.35
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
        self.boundary_detection_oil_fraction_min = 0.02
        self.exploration_oil_fraction_threshold = 0.98

        dx = self.sim_map.x_coords[1] - self.sim_map.x_coords[0] if len(self.sim_map.x_coords) > 1 else 1.0
        dy = self.sim_map.y_coords[1] - self.sim_map.y_coords[0] if len(self.sim_map.y_coords) > 1 else 1.0
        self.communication_radius_cells = int(communication_radius_cells)
        self.communication_radius = float(self.communication_radius_cells * 0.5 * (dx + dy))

        self.control_x_coords = self._sample_axis(self.sim_map.x_coords, self.voronoi_grid_step)
        self.control_y_coords = self._sample_axis(self.sim_map.y_coords, self.voronoi_grid_step)
        self.control_X, self.control_Y = np.meshgrid(self.control_x_coords, self.control_y_coords)
        self.control_points = np.column_stack((self.control_X.ravel(), self.control_Y.ravel()))
        self.control_density = self._boundary_density(self.control_X, self.control_Y).ravel()

        # History for plotting local measurements and consensus convergence.
        self.estimates_history = {}
        self.measurement_consensus_history = []
        self._current_measure_trace = None

    def _sample_axis(self, coords, step):
        sampled = np.asarray(coords[::step], dtype=float)
        if sampled.size == 0 or sampled[-1] != coords[-1]:
            sampled = np.append(sampled, float(coords[-1]))
        return sampled

    def _boundary_density(self, X, Y):
        """Weight the Voronoi centroids toward the spill boundary."""
        field = self.oil_spill.field(X, Y)
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
        angle = float(np.random.uniform(0.0, 2.0 * np.pi))
        drone.exploration_direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
        drone.exploration_speed = float(0.5 * self.max_speed)
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
            angle = float(np.random.uniform(0.0, 2.0 * np.pi))
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
            drone.exploration_direction = direction
            drone.exploration_speed = float(0.5 * self.max_speed)
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
            drone.last_edge_point = None
            drone.last_gradient_peak = None
            drone.last_oil_fraction = None
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
            drone.last_edge_point = None
            drone.last_gradient_peak = 0.0
            drone.last_oil_fraction = oil_fraction
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
            drone.last_edge_point = None
            drone.last_gradient_peak = 0.0
            drone.last_oil_fraction = oil_fraction
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
            drone.last_edge_point = None
            drone.last_gradient_peak = None
            drone.last_oil_fraction = None
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
        drone.estimate_r0 = r0_local

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
        """Compute one weighted Voronoi centroid per drone on a sampled grid."""
        if not self.drones:
            return {}

        positions = np.array([[d.x, d.y] for d in self.drones], dtype=float)
        points = self.control_points
        diff = points[:, None, :] - positions[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        assignment = np.argmin(dist2, axis=1)

        targets = {}
        density = self.control_density
        for idx, drone in enumerate(self.drones):
            mask = assignment == idx
            if not np.any(mask):
                targets[drone.drone_id] = np.array([drone.x, drone.y], dtype=float)
                continue

            cell_points = points[mask]
            cell_weights = density[mask]
            weight_sum = float(np.sum(cell_weights))
            if weight_sum <= 1e-12:
                centroid = np.mean(cell_points, axis=0)
            else:
                centroid = np.sum(cell_points * cell_weights[:, None], axis=0) / weight_sum
            targets[drone.drone_id] = centroid

        return targets

    def _apply_voronoi_control(self):
        """Single-integrator motion toward the weighted Voronoi centroid."""
        targets = self._compute_voronoi_targets()
        for drone in self.drones:
            if getattr(drone, "has_radius_estimate", False):
                target = np.asarray(targets.get(drone.drone_id, (drone.x, drone.y)), dtype=float)
                error = target - np.array([drone.x, drone.y], dtype=float)
                command = self.control_gain * error
                speed = float(np.linalg.norm(command))
                if self.max_speed > 0 and speed > self.max_speed:
                    command = command / speed * self.max_speed

                drone.last_voronoi_target = (float(target[0]), float(target[1]))
                drone.last_exploration_target = None
            else:
                command = self._bounce_exploration_command(drone)
                drone.last_exploration_target = (float(drone.x + command[0]), float(drone.y + command[1]))
                drone.last_voronoi_target = None

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

    def _select_consensus_drones(self):
        """Keep only drones that see enough oil to be considered near the edge."""
        return [
            d
            for d in self.drones
            if getattr(d, "edge_detected", False)
            and getattr(d, "last_oil_fraction", 0.0) >= self.consensus_oil_fraction_threshold
        ]

    def _run_consensus(self, iteration_index=None, iteration_total=None):
        """
        One distributed averaging consensus step over drones near the edge.
        A single simulation frame corresponds to a single consensus iteration.
        """
        valid_drones = self._select_consensus_drones()
        if not valid_drones:
            if iteration_index is not None and iteration_total is not None:
                print(
                    f"  Consensus iter {iteration_index}/{iteration_total}: "
                    f"no eligible drones"
                )
            if self._current_measure_trace is not None:
                for drone in self.drones:
                    self._current_measure_trace[drone.drone_id].append(float(drone.estimate_r0))
            return 0, 0.0

        current = {d.drone_id: float(d.estimate_r0) for d in valid_drones}
        new_values = {}
        neighbor_info = {}
        max_diff = 0.0

        for drone in valid_drones:
            neighbors = self._communication_neighbors(drone, valid_drones)
            if not neighbors:
                neighbors = [drone]

            neighbor_estimates = np.array([current[n.drone_id] for n in neighbors], dtype=float)
            mean_estimate = float(np.mean(neighbor_estimates))
            new_value = current[drone.drone_id] + self.consensus_gain * (mean_estimate - current[drone.drone_id])
            new_values[drone.drone_id] = float(new_value)
            max_diff = max(max_diff, abs(new_value - current[drone.drone_id]))
            neighbor_info[drone.drone_id] = {
                "neighbors": [n.drone_id for n in neighbors],
                "mean": mean_estimate,
                "old": current[drone.drone_id],
                "new": float(new_value),
            }

        for drone in valid_drones:
            drone.estimate_r0 = new_values[drone.drone_id]

        if iteration_index is not None and iteration_total is not None:
            ordered_ids = [d.drone_id for d in valid_drones]
            details = []
            for drone_id in ordered_ids:
                info = neighbor_info[drone_id]
                neighbor_str = ",".join(info["neighbors"])
                details.append(
                    f"{drone_id}:{info['old']:.6f}->{info['new']:.6f} "
                    f"(mean={info['mean']:.6f}, n=[{neighbor_str}])"
                )
            print(
                f"  Consensus iter {iteration_index}/{iteration_total} | "
                f"max_diff={max_diff:.2e} | " + " ; ".join(details)
            )

        if self._current_measure_trace is not None:
            for drone in self.drones:
                self._current_measure_trace[drone.drone_id].append(float(drone.estimate_r0))

        # The final estimate after the frame is the agreed consensus state for
        # this single iteration. It will be used as the starting point on the
        # next frame.
        return 1, max_diff

    def _update_neighbor_drones(self, consensus_drones):
        """
        Let non-participating drones move toward nearby drones that already
        hold a radius estimate, propagating the estimate across multiple hops.
        """
        if not consensus_drones:
            return []

        source_ids = {drone.drone_id for drone in consensus_drones}
        source_estimates = {drone.drone_id: float(drone.estimate_r0) for drone in consensus_drones}
        source_paths = {drone.drone_id: [drone.drone_id] for drone in consensus_drones}
        updated_drones = []
        max_hops = max(1, len(self.drones) - 1)

        for hop_idx in range(max_hops):
            round_updates = []
            round_source_ids = set(source_ids)
            hop_number = hop_idx + 1

            for drone in self.drones:
                if drone.drone_id in round_source_ids:
                    continue

                nearby_sources = [
                    other
                    for other in self.drones
                    if other.drone_id in round_source_ids
                    and float(np.hypot(drone.x - other.x, drone.y - other.y)) <= self.communication_radius
                ]
                if not nearby_sources:
                    continue

                nearby_distances = np.array(
                    [np.hypot(drone.x - other.x, drone.y - other.y) for other in nearby_sources],
                    dtype=float,
                )
                nearby_estimates = np.array([source_estimates[other.drone_id] for other in nearby_sources], dtype=float)
                weights = 1.0 / np.maximum(nearby_distances, 1e-6)
                neighbor_estimate = float(np.average(nearby_estimates, weights=weights))
                parent_source = nearby_sources[int(np.argmin(nearby_distances))]

                new_r0 = float(drone.estimate_r0 + self.neighbor_gain * (neighbor_estimate - drone.estimate_r0))
                round_updates.append((drone, neighbor_estimate, new_r0, parent_source.drone_id))

            if not round_updates:
                break

            hop_messages = []
            for drone, neighbor_estimate, new_r0, parent_id in round_updates:
                drone.estimate_r0 = new_r0
                drone.has_radius_estimate = True
                source_ids.add(drone.drone_id)
                source_estimates[drone.drone_id] = new_r0
                parent_chain = list(source_paths.get(parent_id, [parent_id]))
                source_paths[drone.drone_id] = parent_chain + [drone.drone_id]
                updated_drones.append((drone.drone_id, neighbor_estimate, new_r0))
                chain_str = "<-".join(reversed(source_paths[drone.drone_id]))
                hop_messages.append(
                    f"{chain_str} (new={new_r0:.6f})"
                )

            print(
                f"  Neighbor hop {hop_number}: "
                + " ; ".join(hop_messages)
            )

        return updated_drones

    def step(self):
        """Sense the spill edge, estimate radii, run consensus, then move the drones."""
        self.frame += 1

        measurement_frame = (self.frame % self.measure_every == 0)
        if measurement_frame:
            print(f"Frame {self.frame} - MEASUREMENT FRAME:")
        else:
            print(f"Frame {self.frame} - CONSENSUS ONLY FRAME:")

        # 1. Local edge detection and radius estimation, performed only on
        # measurement frames. On skipped frames we carry forward the last
        # recorded measurement history without touching the live drone state.
        for drone in self.drones:
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
        consensus_drones = self._select_consensus_drones()
        consensus_ids = ", ".join(d.drone_id for d in consensus_drones) if consensus_drones else "none"
        print(
            f"Consensus participants (oil_fraction >= {self.consensus_oil_fraction_threshold:.2f}): "
            f"{consensus_ids}"
        )
        max_diff = 0.0
        for iter_idx in range(self.consensus_iters):
            _, diff = self._run_consensus(iteration_index=iter_idx + 1, iteration_total=self.consensus_iters)
            max_diff = max(max_diff, diff)

        # Record the state after the active consensus step.
        for drone in self.drones:
            self.estimates_history[drone.drone_id]["r0_consensus"].append(drone.estimate_r0)

        updated_neighbors = self._update_neighbor_drones(consensus_drones)
        if updated_neighbors:
            for drone_id, neighbor_estimate, new_r0 in updated_neighbors:
                print(
                    f"{drone_id}: neighbor update from {neighbor_estimate:.6f}, "
                    f"new r0 = {new_r0:.6f}"
                )
        else:
            print("Neighbor update: no non-participating drones had a nearby consensus source.")

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
            mode = "voronoi" if getattr(drone, "has_radius_estimate", False) else "explore"
            target = drone.last_voronoi_target if mode == "voronoi" else drone.last_exploration_target
            print(
                f"{drone.drone_id}: pos=({drone.x:.3f}, {drone.y:.3f}), "
                f"u=({drone.u_x:.3f}, {drone.u_y:.3f}), "
                f"mode={mode}, target={target}"
            )

        if measurement_frame:
            if self._current_measure_trace is not None:
                self.measurement_consensus_history.append(
                    {drone_id: list(values) for drone_id, values in self._current_measure_trace.items()}
                )
                self._current_measure_trace = None
            end_snapshot = ", ".join(f"{d.drone_id}={d.estimate_r0:.6f}" for d in self.drones)
            print(f"Measurement end radii: {end_snapshot}")
