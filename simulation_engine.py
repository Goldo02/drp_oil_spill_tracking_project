import numpy as np
import matplotlib.pyplot as plt
from drone import Drone
from edge_detection import detect_edges, extract_edge_points


class SimulationEngine:
    """Orchestrates static multi-drone radius estimation and consensus."""

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

        dx = self.sim_map.x_coords[1] - self.sim_map.x_coords[0] if len(self.sim_map.x_coords) > 1 else 1.0
        dy = self.sim_map.y_coords[1] - self.sim_map.y_coords[0] if len(self.sim_map.y_coords) > 1 else 1.0
        self.communication_radius_cells = int(communication_radius_cells)
        self.communication_radius = float(self.communication_radius_cells * 0.5 * (dx + dy))

        # History for plotting local measurements and consensus convergence.
        self.estimates_history = {}
        self.measurement_consensus_history = []
        self._current_measure_trace = None

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
        )
        self.drones.append(drone)
        self.estimates_history[drone_id] = {
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
        }

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
        drone.last_edge_point = nearest_edge_point
        drone.last_gradient_peak = float(np.count_nonzero(edges))
        drone.last_oil_fraction = float(np.mean(camera_view >= self.oil_cell_threshold))
        drone.estimate_r0 = r0_local

        return {
            "edge_x": edge_x,
            "edge_y": edge_y,
            "edge_detected": True,
            "gradient_magnitude": float(np.count_nonzero(edges)),
            "gradient_peak": float(np.count_nonzero(edges)),
            "oil_fraction": float(np.mean(camera_view >= self.oil_cell_threshold)),
            "r0_local_raw": raw_r0_local,
            "r0_local": r0_local,
        }

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
        Let non-participating drones move toward nearby consensus participants.
        """
        if not consensus_drones:
            return []

        updated_drones = []
        for drone in self.drones:
            if drone in consensus_drones:
                continue

            nearby_drones = self._communication_neighbors(drone, consensus_drones)
            if not nearby_drones:
                continue

            nearby_distances = np.array(
                [np.hypot(drone.x - other.x, drone.y - other.y) for other in nearby_drones],
                dtype=float,
            )
            nearby_estimates = np.array([other.estimate_r0 for other in nearby_drones], dtype=float)
            weights = 1.0 / np.maximum(nearby_distances, 1e-6)
            neighbor_estimate = float(np.average(nearby_estimates, weights=weights))

            drone.estimate_r0 = float(
                drone.estimate_r0 + self.neighbor_gain * (neighbor_estimate - drone.estimate_r0)
            )
            updated_drones.append((drone.drone_id, neighbor_estimate, drone.estimate_r0))

        return updated_drones

    def step(self):
        """Sense the spill edge, estimate radii, then run distributed consensus."""
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

        if measurement_frame:
            if self._current_measure_trace is not None:
                self.measurement_consensus_history.append(
                    {drone_id: list(values) for drone_id, values in self._current_measure_trace.items()}
                )
                self._current_measure_trace = None
            end_snapshot = ", ".join(f"{d.drone_id}={d.estimate_r0:.6f}" for d in self.drones)
            print(f"Measurement end radii: {end_snapshot}")
