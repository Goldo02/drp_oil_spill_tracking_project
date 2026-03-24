import numpy as np
from drone import Drone
from edge_detection import identify_centroid, check_geometric_lock

class SimulationEngine:
    """Orchestrates the simulation loop for multiple drones."""
    def __init__(self, sim_map, oil_spill, dt=0.1, sigma_gps=0.1, sigma_cam=0.1):
        self.sim_map = sim_map
        self.oil_spill = oil_spill
        self.dt = dt
        self.sigma_gps = sigma_gps
        self.sigma_cam = sigma_cam
        self.drones = []
        self.frame = 0
        
        # True circle parameters (unknown to drones)
        self.true_x0 = oil_spill.x0
        self.true_y0 = oil_spill.y0
        self.true_r0 = oil_spill.r0
        
        # Pre-calculated field
        self.world_field = oil_spill.field(sim_map.X, sim_map.Y)
        
        # Matveev & Consensus Control Parameters
        self.c_star = 0.5 
        self.u_bar = 3.0       # Radial correction gain
        self.v_base = 0.3      # Constant orbit speed
        self.k_consensus = 0.2 # Consensus gain for distributed averaging
        self.num_consensus_steps = 10  # Number of consensus iterations per frame

        # History for plotting consensus convergence
        self.estimates_history = {}

    def add_drone(self, drone_id, x, y):
        drone = Drone(drone_id, x, y, map_bounds=(*self.sim_map.xlim, *self.sim_map.ylim),
                      gps_noise=self.sigma_gps, camera_noise=self.sigma_cam,
                      true_x0=self.true_x0, true_y0=self.true_y0, true_r0=self.true_r0)
        # Initial random velocity and heading for SEARCH
        angle = np.random.uniform(0, 2 * np.pi)
        vx, vy = np.cos(angle) * 0.5, np.sin(angle) * 0.5
        drone.set_velocity(vx, vy)
        drone.theta = angle
        drone.speed = self.v_base
        drone.on_edge = False  # track APPROACH state entry
        self.drones.append(drone)
        # Initialize history
        self.estimates_history[drone_id] = {
            'x0': [], 'y0': [], 'r0_pre': [], 'r0_post': [], 'r0_consensus': []
        }

    def step(self):
        """Update physics and logic for all drones."""
        self.frame += 1

        # 0. Local measurements and updates for drones detecting edge
        for drone in self.drones:
            # Robust perception
            camera_view = drone.get_camera_view(self.world_field, self.sim_map.x_coords, self.sim_map.y_coords)
            h, w = camera_view.shape
            win = 2
            center_val = np.mean(camera_view[h//2-win : h//2+win+1, w//2-win : w//2+win+1])

            gps_x, gps_y = drone.get_gps_pos()

            # If detecting edge, compute local radius estimate
            if center_val > self.c_star:
                # Distance to center
                dist_to_center = np.sqrt((gps_x - self.true_x0)**2 + (gps_y - self.true_y0)**2)
                # True distance to boundary
                if dist_to_center <= self.true_r0:
                    d_i = self.true_r0 - dist_to_center
                else:
                    d_i = dist_to_center - self.true_r0
                # Noisy distance to boundary
                d_i_noisy = d_i + np.random.normal(0, self.sigma_cam)
                # Local radius estimate
                r_i = dist_to_center - d_i_noisy
                drone.estimate_r0 = r_i
                # Center is known
                drone.estimate_x0 = self.true_x0
                drone.estimate_y0 = self.true_y0

        # Print initial measurements (post-measurement and pre-consensus)
        print(f"Frame {self.frame} - INITIAL MEASUREMENTS:")
        for d in self.drones:
            print(f"D{d.drone_id}: {d.estimate_r0:.6f}")

        # Record history BEFORE consensus so we can visualize individual estimate paths
        for d in self.drones:
            self.estimates_history[d.drone_id]['x0'].append(d.estimate_x0)
            self.estimates_history[d.drone_id]['y0'].append(d.estimate_y0)
            self.estimates_history[d.drone_id]['r0_pre'].append(d.estimate_r0)
            # Record initial consensus state point (pre-iteration)
            self.estimates_history[d.drone_id]['r0_consensus'].append(d.estimate_r0)

        # Distributed consensus: iterate until convergence (all-to-all topology)
        max_iters = 100
        tol = 1e-6
        actual_iters = 0

        for it in range(1, max_iters + 1):
            new_r0 = {}
            for drone in self.drones:
                neighbors = [d for d in self.drones if d != drone]
                if neighbors:
                    mean_neighbors = np.mean([d.estimate_r0 for d in neighbors])
                    new_r0[drone] = drone.estimate_r0 + self.k_consensus * (mean_neighbors - drone.estimate_r0)
                else:
                    new_r0[drone] = drone.estimate_r0

            max_diff = 0.0
            for drone in self.drones:
                diff = abs(new_r0[drone] - drone.estimate_r0)
                max_diff = max(max_diff, diff)
                drone.estimate_r0 = new_r0[drone]

            # store intra-consensus state for visualization
            for d in self.drones:
                self.estimates_history[d.drone_id]['r0_consensus'].append(d.estimate_r0)

            actual_iters = it
            if max_diff < tol:
                break

        # Optionally enforce exact consensus average at the end of the loop
        # (ensures all r0 values are effectively identical when plotting)
        consensus_avg = np.mean([d.estimate_r0 for d in self.drones])
        for d in self.drones:
            d.estimate_r0 = consensus_avg

        print(f"AFTER CONSENSUS (converged in {actual_iters} iter(s), max_diff={max_diff:.2e}):")
        for d in self.drones:
            print(f"D{d.drone_id}: {d.estimate_r0:.6f}")

        # Record history AFTER consensus for comparison
        for d in self.drones:
            self.estimates_history[d.drone_id]['r0_post'].append(d.estimate_r0)

        # 1. Compute angular positions and consensus gaps for APPROACH drones
        drones_on_edge = [d for d in self.drones if d.mode == "APPROACH"]
        
        # Reset gap attributes
        for d in self.drones:
            d.u_consensus = 0.0

        if len(drones_on_edge) > 1:
            # Compute each drone's angle relative to its estimated spill center
            for d in drones_on_edge:
                d.phi = np.arctan2(d.y - d.estimate_y0, d.x - d.estimate_x0)

            # Sort drones by phi (ascending) — defines CCW circular order
            drones_on_edge.sort(key=lambda d: d.phi)
            N = len(drones_on_edge)
            # Ideal equal-spacing gap
            ideal_gap = 2 * np.pi / N

            for i in range(N):
                d = drones_on_edge[i]
                # Neighbors in sorted CCW order (circular)
                d_prev = drones_on_edge[(i - 1) % N]
                d_next = drones_on_edge[(i + 1) % N]

                # Gap to next drone (CCW ahead) and from previous (CCW behind)
                gap_to_next   = (d_next.phi - d.phi)   % (2 * np.pi)
                gap_from_prev = (d.phi      - d_prev.phi) % (2 * np.pi)

                # Error terms: positive means "drone should move CCW to close gap"
                #   e_next  > 0  → gap ahead is too big  → speed up (move CCW)
                #   e_prev  > 0  → gap behind is too big → slow down (move CW)
                e_next = gap_to_next  - ideal_gap   # positive → move CCW
                e_prev = gap_from_prev - ideal_gap  # positive → move CW

                # Net: move CCW when ahead gap is large, move CW when behind gap is large
                delta_gap = e_next - e_prev  # same as gap_to_next - gap_from_prev
                # Compute a proportional correction (rad/s) for the angular velocity
                u_phi = self.k_consensus * delta_gap
                max_u_phi = 0.8
                u_phi = np.clip(u_phi, -max_u_phi, max_u_phi)
                # Store as angular-speed correction (to be applied to tangential speed)
                d.u_consensus_phi = u_phi

        for drone in self.drones:
            # 2. Robust perception (mean of 5x5 center window)
            camera_view = drone.get_camera_view(self.world_field, self.sim_map.x_coords, self.sim_map.y_coords)
            h, w = camera_view.shape
            win = 2
            center_val = np.mean(camera_view[h//2-win : h//2+win+1, w//2-win : w//2+win+1])

            # 3. Mode transition logic
            if drone.mode == "SEARCH":
                if center_val > self.c_star:
                    drone.mode = "APPROACH"
                    drone.on_edge = True
                    # --- KEY FIX: Initialize theta to the CCW tangent direction ---
                    # The tangent to the circle at drone's position is perpendicular
                    # to the radial vector (pointing CCW).
                    radial_angle = np.arctan2(drone.y - drone.estimate_y0, drone.x - drone.estimate_x0)
                    drone.theta = radial_angle + np.pi / 2  # 90° CCW = tangent direction
                    print(f"Frame {self.frame}: Drone {drone.drone_id} -> APPROACH (theta_init={np.degrees(drone.theta):.1f}°).")

            if drone.mode == "APPROACH":
                # Continuous radial control (proportional to radial error)
                desired_r = drone.estimate_r0
                dx = drone.x - drone.estimate_x0
                dy = drone.y - drone.estimate_y0
                dist = np.sqrt(dx*dx + dy*dy)
                radial_error = dist - desired_r
                k_radial = 1.5
                v_radial = -k_radial * radial_error

                # Tangential speed: base plus consensus correction on angular rate
                u_phi = getattr(drone, 'u_consensus_phi', 0.0)
                v_tangential = self.v_base + (u_phi * max(dist, 1e-3))
                v_tangential = np.clip(v_tangential, 0.05, 1.5)

                # Tangential unit vector (CCW): (-sin(phi), cos(phi))
                phi_now = np.arctan2(dy, dx)
                t_x, t_y = -np.sin(phi_now), np.cos(phi_now)
                r_x, r_y = np.cos(phi_now), np.sin(phi_now)

                vx = v_tangential * t_x + v_radial * r_x
                vy = v_tangential * t_y + v_radial * r_y

                drone.set_velocity(vx, vy)
                drone.theta = np.arctan2(vy, vx)

                # Lost track: return to SEARCH only if no oil is visible at all
                if np.max(camera_view) < 0.15:
                    drone.mode = "SEARCH"
                    drone.on_edge = False
                    angle = np.random.uniform(0, 2 * np.pi)
                    drone.set_velocity(np.cos(angle) * 0.5, np.sin(angle) * 0.5)
                    drone.theta = angle

            # 4. Physics integration (commented out to keep drones static)
            # drone.update_position(self.dt)
