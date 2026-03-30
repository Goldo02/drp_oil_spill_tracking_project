import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
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
        lambda_reg=0.2,
        lambda_smooth=5.0,
        lambda_center=0.05,
        min_total_angle=np.pi * 2 / 3,
        w_gps=5.0,
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

        # Estimation parameters
        self.measure_every = max(1, int(measure_every))
        self.wls_iters = max(1, int(consensus_iters))  # Reusing census_iters for WLS
        self.lambda_reg = float(lambda_reg)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_center = float(lambda_center)
        self.min_total_angle = float(min_total_angle)
        self.w_gps = float(w_gps)

        # Optimization & detection thresholds
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

    def _taubin_fit(self, points):
        """Taubin circle fit (algebraic). returns [cx, cy, r]."""
        N = points.shape[0]
        if N < 3: return None
        X, Y = points[:, 0], points[:, 1]
        A_lin = np.column_stack([2*X, 2*Y, np.ones(N)])
        B_lin = X**2 + Y**2
        try:
            sol, _, _, _ = np.linalg.lstsq(A_lin, B_lin, rcond=None)
            cx, cy, C = sol
            r2 = C + cx**2 + cy**2
            if r2 <= 0: return None
            return np.array([cx, cy, np.sqrt(r2)])
        except: return None

    def _geometric_circle_fit(self, points, theta_init, theta_prev=None, lambda_reg=0.0, center_prior=None, lambda_center=0.0):
        """Geometric circle fit with temporal and center regularization."""
        def residuals(theta):
            cx, cy, r = theta
            dists = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
            res = dists - r
            
            aug_res = [res]
            if theta_prev is not None and lambda_reg > 0:
                reg_res = np.sqrt(lambda_reg) * (theta - theta_prev)
                aug_res.append(reg_res)
            
            if center_prior is not None and lambda_center > 0:
                center_res = np.sqrt(lambda_center) * (theta[:2] - center_prior)
                aug_res.append(center_res)
                
            if len(aug_res) > 1:
                return np.concatenate(aug_res)
            return res
        xmin, xmax, ymin, ymax = self.sim_map.xlim[0], self.sim_map.xlim[1], self.sim_map.ylim[0], self.sim_map.ylim[1]
        lower_bounds = [xmin, ymin, 0.01]
        upper_bounds = [xmax, ymax, 10.0]
        try:
            res = least_squares(residuals, theta_init, bounds=(lower_bounds, upper_bounds), method='trf')
            return res.x
        except Exception:
            return theta_init

    def _compute_angular_span(self, points, center):
        """Compute angular span of points relative to center, handling wrap-around."""
        if points.shape[0] < 2: return 0.0
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        angles = np.sort(angles)
        gaps = np.diff(angles)
        wrap_gap = (2 * np.pi - angles[-1] + angles[0])
        max_gap = max(np.max(gaps), wrap_gap)
        return 2 * np.pi - max_gap

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
        # Initialize with random guesses for center and radius instead of ground truth.
        # Maps are typically [-5, 5], so guesses in [-2, 2] for center 
        # and [0.5, 2.5] for radius are reasonable deviations.
        init_cx = float(np.random.uniform(-2.0, 2.0))
        init_cy = float(np.random.uniform(-2.0, 2.0))
        init_r = float(np.random.uniform(0.5, 2.5))

        drone = Drone(
            drone_id,
            x,
            y,
            map_bounds=(*self.sim_map.xlim, *self.sim_map.ylim),
            gps_noise=self.sigma_gps,
            camera_noise=self.sigma_cam,
            initial_x0=init_cx,
            initial_y0=init_cy,
            initial_r0=init_r,
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
            "x": [], "y": [], "cx": [], "cy": [], "r": [],
            "cx_meas": [], "cy_meas": [], "r_meas": [],
            "cx_fused": [], "cy_fused": [], "r_fused": [],
            "edge_x": [], "edge_y": [], "edge_detected": [],
            "gradient_magnitude": [], "gradient_peak": [], "oil_fraction": [],
            "u_x": [], "u_y": [], "voronoi_cx": [], "voronoi_cy": [],
            "exploration_x": [], "exploration_y": [], "weight": [],
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

    def _detect_local_edges(self, drone):
        """Sense camera and extract edge points in world coordinates."""
        camera_view = drone.get_camera_view(self.world_field, self.sim_map.x_coords, self.sim_map.y_coords)
        if camera_view.size == 0:
            drone.edge_detected = False
            return None, 0.0

        oil_fraction = float(np.mean(camera_view >= self.oil_cell_threshold))
        if oil_fraction <= self.boundary_detection_oil_fraction_min or oil_fraction >= self.exploration_oil_fraction_threshold:
            drone.edge_detected = False
            drone.last_nls_points = None
            return None, oil_fraction

        dx = self.sim_map.x_coords[1] - self.sim_map.x_coords[0]
        dy = self.sim_map.y_coords[1] - self.sim_map.y_coords[0]
        i_center = int((drone.x - self.sim_map.x_coords[0]) / dx)
        j_center = int((drone.y - self.sim_map.y_coords[0]) / dy)
        half = camera_view.shape[0] // 2
        
        i_min, i_max = max(0, i_center - half), min(len(self.sim_map.x_coords), i_center + half + 1)
        j_min, j_max = max(0, j_center - half), min(len(self.sim_map.y_coords), j_center + half + 1)
        
        if i_min == 0 or j_min == 0 or i_max == len(self.sim_map.x_coords) or j_max == len(self.sim_map.y_coords):
            drone.edge_detected = False
            return None, 0.0

        edges = detect_edges(camera_view, threshold1=self.canny_threshold1, threshold2=self.canny_threshold2)
        edge_pixels = extract_edge_points(edges)
        if edge_pixels.size < 5:
            drone.edge_detected = False
            drone.last_nls_points = None
            return None, 0.0

        local_x, local_y = self.sim_map.x_coords[i_min:i_max], self.sim_map.y_coords[j_min:j_max]
        world_y = local_y[np.clip(edge_pixels[:, 0], 0, len(local_y) - 1)]
        world_x = local_x[np.clip(edge_pixels[:, 1], 0, len(local_x) - 1)]
        points = np.column_stack([world_x, world_y])
        
        # Cache for visualization
        dist_to_edges = np.hypot(world_x - drone.x, world_y - drone.y)
        drone.last_edge_point = (float(world_x[np.argmin(dist_to_edges)]), float(world_y[np.argmin(dist_to_edges)]))
        drone.last_gradient_peak = float(np.count_nonzero(edges))
        drone.last_oil_fraction = oil_fraction
        drone.last_nls_points = points.copy()
        drone.edge_detected = True
        return points, oil_fraction

    def _fit_local_circle(self, drone, points, center_prior=None, lambda_center=0.0):
        """Perform NLS fit with fallback for theta_measured."""
        theta_init = drone.theta.copy()
        lambda_r = self.lambda_reg if drone.theta_prev is not None else 0.0
        
        theta_measured = self._geometric_circle_fit(
            points, theta_init, theta_prev=drone.theta_prev, lambda_reg=lambda_r,
            center_prior=center_prior, lambda_center=lambda_center
        )
        
        # Robustness check: if NLS produce something completely unrealistic or hits bounds, use Taubin/LLS
        if np.linalg.norm(theta_measured - theta_init) > 5.0 or theta_measured[2] < 0.1:
            taubin = self._taubin_fit(points)
            if taubin is not None:
                theta_measured = taubin
        
        if drone.theta_prev is None: drone.theta_prev = theta_measured.copy()
        drone.theta_measured = theta_measured.copy()
        
        span = self._compute_angular_span(points, theta_measured[:2])
        weight = float(len(points) * (span / np.pi))
        drone.confidence_weight = weight
        drone.has_measure = True
        drone.has_radius_estimate = True
        return {"theta_measured": theta_measured, "weight": weight, "span": span}

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

        center = drone.theta[:2]
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
        """Single-integrator motion toward the weighted Voronoi centroid."""
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
            history["exploration_x"].append(float(drone.last_exploration_target[0]) if drone.last_exploration_target is not None else np.nan)
            history["exploration_y"].append(float(drone.last_exploration_target[1]) if drone.last_exploration_target is not None else np.nan)
            history["x"].append(float(drone.x))
            history["y"].append(float(drone.y))

    def _run_wls_fusion(self, gps_centroid=None, use_gps_prior=False):
        """Iterative Normalized WLS fusion with optional GPS center prior."""
        for drone in self.drones:
            drone.theta_fused = drone.theta_measured.copy()

        # Trace for visualization of consensus convergence
        trace = {d.drone_id: [d.theta_fused[2]] for d in self.drones}

        for k in range(self.wls_iters):
            new_fused = {}
            for drone in self.drones:
                weight_sum = drone.confidence_weight
                weighted_theta_sum = drone.confidence_weight * drone.theta_fused
                
                for neighbor in drone.neighbors:
                    weight_sum += neighbor.confidence_weight
                    weighted_theta_sum += neighbor.confidence_weight * neighbor.theta_fused
                
                if weight_sum > 1e-12:
                    if use_gps_prior and gps_centroid is not None:
                        # Apply w_gps only to cx, cy components
                        cx_fused = (weighted_theta_sum[0] + self.w_gps * gps_centroid[0]) / (weight_sum + self.w_gps)
                        cy_fused = (weighted_theta_sum[1] + self.w_gps * gps_centroid[1]) / (weight_sum + self.w_gps)
                        r_fused = weighted_theta_sum[2] / weight_sum
                        new_fused[drone.drone_id] = np.array([cx_fused, cy_fused, r_fused])
                    else:
                        new_fused[drone.drone_id] = weighted_theta_sum / weight_sum
                else:
                    new_fused[drone.drone_id] = drone.theta_fused
            
            for drone in self.drones:
                drone.theta_fused = new_fused[drone.drone_id]
                trace[drone.drone_id].append(drone.theta_fused[2])

        self._current_measure_trace = trace

        max_diff = 0.0
        for drone in self.drones:
            # Temporal tracking: theta = alpha * theta_fused + (1 - alpha) * theta_prev
            alpha = drone.confidence_weight / (drone.confidence_weight + self.lambda_smooth)
            if not drone.edge_detected and drone.confidence_weight < 1e-6:
                # If no current measurement, alpha is 0, keeping previous theta
                alpha = 0.0
                
            new_theta = alpha * drone.theta_fused + (1.0 - alpha) * drone.theta
            diff = np.linalg.norm(new_theta - drone.theta)
            drone.theta = new_theta
            drone.theta_prev = new_theta.copy()
            max_diff = max(max_diff, diff)
            
        return max_diff

    def _update_communication_topology(self):
        """Update the neighbors list for each drone based on current positions."""
        for drone in self.drones:
            drone.neighbors = self._communication_neighbors(drone, [d for d in self.drones if d != drone])

    def step(self):
        """Sense, estimate, fuse, and move."""
        self.frame += 1
        measurement_frame = (self.frame % self.measure_every == 0)

        # 1. Edge detection results for all (points, edge_detected)
        detection_results = {}
        all_on_border = True
        for drone in self.drones:
            points, oil_fraction = self._detect_local_edges(drone)
            detection_results[drone.drone_id] = (points, oil_fraction)
            if not getattr(drone, "edge_detected", False):
                all_on_border = False

        # 2. GPS Centroid of all drones
        px = [d.x for d in self.drones]
        py = [d.y for d in self.drones]
        cx_gps, cy_gps = np.mean(px), np.mean(py)
        gps_centroid = np.array([cx_gps, cy_gps])

        # 3. Fitting for all drones
        mean_theta = np.mean([d.theta for d in self.drones], axis=0)
        total_span = 0.0
        for d in self.drones:
            pts, _ = detection_results[d.drone_id]
            if d.edge_detected and pts is not None:
                total_span += self._compute_angular_span(pts, mean_theta[:2])
        
        use_gps_prior = (all_on_border and total_span >= self.min_total_angle)

        for drone in self.drones:
            drone.has_measure = False
            history = self.estimates_history[drone.drone_id]
            history["cx"].append(float(drone.theta[0]))
            history["cy"].append(float(drone.theta[1]))
            history["r"].append(float(drone.theta[2]))

            if measurement_frame:
                points, oil_fraction = detection_results[drone.drone_id]
                if drone.edge_detected and points is not None:
                    res = self._fit_local_circle(
                        drone, points, 
                        center_prior=gps_centroid if use_gps_prior else None,
                        lambda_center=self.lambda_center if use_gps_prior else 0.0
                    )
                    
                    history["edge_x"].append(float(np.mean(points[:, 0])))
                    history["edge_y"].append(float(np.mean(points[:, 1])))
                    history["edge_detected"].append(True)
                    history["gradient_magnitude"].append(drone.last_gradient_peak)
                    history["gradient_peak"].append(drone.last_gradient_peak)
                    history["oil_fraction"].append(oil_fraction)
                    history["cx_meas"].append(float(drone.theta_measured[0]))
                    history["cy_meas"].append(float(drone.theta_measured[1]))
                    history["r_meas"].append(float(drone.theta_measured[2]))
                    history["weight"].append(drone.confidence_weight)
                else:
                    history["edge_x"].append(np.nan)
                    history["edge_y"].append(np.nan)
                    history["edge_detected"].append(False)
                    history["gradient_magnitude"].append(0.0)
                    history["gradient_peak"].append(0.0)
                    history["oil_fraction"].append(oil_fraction if isinstance(oil_fraction, float) else np.nan)
                    history["cx_meas"].append(float(drone.theta_measured[0]))
                    history["cy_meas"].append(float(drone.theta_measured[1]))
                    history["r_meas"].append(float(drone.theta_measured[2]))
                    history["weight"].append(0.0)
            else:
                for k in ["edge_x", "edge_y", "edge_detected", "gradient_magnitude", "gradient_peak", "oil_fraction", "cx_meas", "cy_meas", "r_meas", "weight"]:
                    history[k].append(history[k][-1] if history[k] else np.nan)

        # Consensus/Fusion
        self._update_communication_topology()
        max_diff = self._run_wls_fusion(gps_centroid=gps_centroid, use_gps_prior=use_gps_prior)

        if measurement_frame and self._current_measure_trace:
            self.measurement_consensus_history.append(self._current_measure_trace)

        for drone in self.drones:
            history = self.estimates_history[drone.drone_id]
            history["cx_fused"].append(float(drone.theta[0]))
            history["cy_fused"].append(float(drone.theta[1]))
            history["r_fused"].append(float(drone.theta[2]))

        self._apply_voronoi_control()
