import numpy as np
from drone import Drone


class SimulationEngine:
    """Orchestrates multi-drone consensus initialization and closed-loop control."""

    def __init__(self, sim_map, oil_spill, dt=0.1, sigma_gps=0.1, sigma_cam=0.1):
        self.sim_map = sim_map
        self.oil_spill = oil_spill
        self.dt = dt
        self.sigma_gps = sigma_gps
        self.sigma_cam = sigma_cam
        self.drones = []
        self.frame = 0

        # Static spill parameters (center known, radius unknown to drones)
        self.true_x0 = oil_spill.x0
        self.true_y0 = oil_spill.y0
        self.true_r0 = oil_spill.r0

        # Precomputed field map for camera sensing
        self.world_field = oil_spill.field(sim_map.X, sim_map.Y)

        # Parameters
        self.c_star = 0.5
        self.k_consensus = 0.2
        self.k_r = 1.2
        self.k_phi = 0.8

        # Shared estimate after one-time consensus
        self.estimate_r0 = None
        self.consensus_done = False
        self.initial_consensus_iterations = 0

        # History (optional plotting)
        self.estimates_history = {}

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
            true_r0=self.true_r0,
        )

        # Random initial search velocity
        angle = np.random.uniform(0, 2 * np.pi)
        vx, vy = 0.5 * np.cos(angle), 0.5 * np.sin(angle)
        drone.set_velocity(vx, vy)

        self.drones.append(drone)
        self.estimates_history[drone_id] = {
            "x0": [],
            "y0": [],
            "r0_pre": [],
            "r0_post": [],
            "r0_consensus": [],
        }

    @staticmethod
    def _wrap_to_pi(angle):
        """Wrap angle to [-pi, pi)."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def initial_consensus(self, max_iter=100, tolerance=1e-6):
        """
        One-time distributed averaging on radius estimate.
        After convergence, all drones share one fixed radius estimate.
        """
        if self.consensus_done:
            return

        print("Performing initial radius consensus...")

        # 1) Local noisy measurements
        for drone in self.drones:
            camera_view = drone.get_camera_view(
                self.world_field, self.sim_map.x_coords, self.sim_map.y_coords
            )
            h, w = camera_view.shape
            win = 2
            center_val = np.mean(camera_view[h // 2 - win : h // 2 + win + 1, w // 2 - win : w // 2 + win + 1])

            gps_x, gps_y = drone.get_gps_pos()

            if center_val > self.c_star:
                dist_to_center = np.hypot(gps_x - self.true_x0, gps_y - self.true_y0)
                d_i = abs(dist_to_center - self.true_r0)
                d_i_noisy = d_i + np.random.normal(0.0, self.sigma_cam)
                drone.estimate_r0 = dist_to_center - d_i_noisy

            # Center is known
            drone.estimate_x0 = self.true_x0
            drone.estimate_y0 = self.true_y0

        print("Initial measurements:")
        for d in self.drones:
            print(f"{d.drone_id}: r0 = {d.estimate_r0:.6f}")

        # Save initial point for consensus convergence plot
        for d in self.drones:
            self.estimates_history[d.drone_id]["r0_consensus"].append(d.estimate_r0)

        # 2) Distributed average consensus (all-to-all)
        converged = False
        it = 0
        while it < max_iter and not converged:
            it += 1
            new_r0 = {}

            for drone in self.drones:
                neighbors = [d for d in self.drones if d is not drone]
                if neighbors:
                    mean_neighbors = np.mean([n.estimate_r0 for n in neighbors])
                    new_r0[drone] = drone.estimate_r0 + self.k_consensus * (mean_neighbors - drone.estimate_r0)
                else:
                    new_r0[drone] = drone.estimate_r0

            for drone in self.drones:
                drone.estimate_r0 = new_r0[drone]

            for d in self.drones:
                self.estimates_history[d.drone_id]["r0_consensus"].append(d.estimate_r0)

            std_dev = np.std([d.estimate_r0 for d in self.drones])
            converged = std_dev < tolerance

        # 3) Force exact agreement
        r_shared = float(np.mean([d.estimate_r0 for d in self.drones]))
        for d in self.drones:
            d.estimate_r0 = r_shared

        self.estimate_r0 = r_shared
        self.initial_consensus_iterations = it
        self.consensus_done = True

        print(f"Consensus converged in {it} iterations")
        print(f"Final shared radius: {self.estimate_r0:.6f}")

    def step(self):
        """
        Closed-loop control step after one-time consensus.
        No further radius estimation or consensus updates are performed.
        """
        self.frame += 1

        if not self.consensus_done:
            raise RuntimeError("initial_consensus() must be called before step().")

        # Keep center estimate fixed (known) and radius fixed (shared consensus)
        for d in self.drones:
            d.estimate_x0 = self.true_x0
            d.estimate_y0 = self.true_y0
            d.estimate_r0 = self.estimate_r0

            self.estimates_history[d.drone_id]["x0"].append(d.estimate_x0)
            self.estimates_history[d.drone_id]["y0"].append(d.estimate_y0)
            self.estimates_history[d.drone_id]["r0_pre"].append(d.estimate_r0)

        if len(self.drones) >= 2:
            # Angular positions on circle
            for d in self.drones:
                d.phi = np.arctan2(d.y - self.true_y0, d.x - self.true_x0) % (2 * np.pi)

            drones_ccw = sorted(self.drones, key=lambda dr: dr.phi)
            n = len(drones_ccw)

            for i, d in enumerate(drones_ccw):
                prev_d = drones_ccw[(i - 1) % n]
                next_d = drones_ccw[(i + 1) % n]

                gap_prev = (d.phi - prev_d.phi) % (2 * np.pi)
                gap_next = (next_d.phi - d.phi) % (2 * np.pi)

                # Voronoi boundaries (angular midpoints with neighbors)
                boundary_prev = (d.phi - 0.5 * gap_prev) % (2 * np.pi)
                boundary_next = (d.phi + 0.5 * gap_next) % (2 * np.pi)

                owned_sweep = (boundary_next - boundary_prev) % (2 * np.pi)
                phi_target = (boundary_prev + 0.5 * owned_sweep) % (2 * np.pi)

                # Continuous control law
                dx = d.x - self.true_x0
                dy = d.y - self.true_y0
                dist = np.hypot(dx, dy)
                if dist < 1e-6:
                    dist = 1e-6

                radial = np.array([dx / dist, dy / dist])
                tangential = np.array([-dy / dist, dx / dist])

                radial_error = dist - self.estimate_r0
                angular_error = self._wrap_to_pi(phi_target - d.phi)

                v_radial = -self.k_r * radial_error
                v_tangential = self.k_phi * angular_error * self.estimate_r0

                v = v_radial * radial + v_tangential * tangential
                d.set_velocity(float(v[0]), float(v[1]))

        # Integrate dynamics
        for d in self.drones:
            d.update_position(self.dt)
            self.estimates_history[d.drone_id]["r0_post"].append(d.estimate_r0)
