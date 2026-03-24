import numpy as np
import matplotlib.pyplot as plt
from drone import Drone


class SimulationEngine:
    """Orchestrates multi-drone consensus and closed-loop control."""

    def __init__(self, sim_map, oil_spill, dt=0.1, sigma_gps=0.1, sigma_cam=0.1, visualize_voronoi=False):
        self.sim_map = sim_map
        self.oil_spill = oil_spill
        self.dt = dt
        self.sigma_gps = sigma_gps
        self.sigma_cam = sigma_cam
        self.drones = []
        self.frame = 0

        # Static spill parameters (center known, radius unknown)
        self.true_x0 = oil_spill.x0
        self.true_y0 = oil_spill.y0
        self.true_r0 = oil_spill.r0

        # Precomputed field for camera sensing
        self.world_field = oil_spill.field(sim_map.X, sim_map.Y)

        # Parameters
        self.c_star = 0.5
        self.k_consensus = 0.2
        self.k_r = 1.2
        self.k_phi = 0.8

        # Shared / evolving radius estimate
        self.estimate_r0 = None

        # Voronoi live visualization toggle
        self.visualize_voronoi = visualize_voronoi
        self._voronoi_fig = None
        self._voronoi_ax = None

        # History (for plotting)
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
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def initial_consensus(self, max_iter=100, tolerance=1e-6):
        """Kept for compatibility. Not required when using frame-by-frame consensus."""
        print("Performing initial radius consensus...")

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
                if dist_to_center <= self.true_r0:
                    r_i = dist_to_center + d_i_noisy
                else:
                    r_i = dist_to_center - d_i_noisy
                drone.estimate_r0 = max(0.05, float(r_i))

            drone.estimate_x0 = self.true_x0
            drone.estimate_y0 = self.true_y0

        print("Initial measurements:")
        for d in self.drones:
            print(f"{d.drone_id}: r0 = {d.estimate_r0:.6f}")

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
            converged = np.std([d.estimate_r0 for d in self.drones]) < tolerance

        self.estimate_r0 = float(np.mean([d.estimate_r0 for d in self.drones]))
        print(f"Consensus converged in {it} iterations")
        print(f"Final shared radius: {self.estimate_r0:.6f}")

    def _draw_voronoi(self, ax, voronoi_arcs, phi_targets):
        # Estimated common circle
        theta = np.linspace(0, 2 * np.pi, 400)
        cx = self.true_x0 + self.estimate_r0 * np.cos(theta)
        cy = self.true_y0 + self.estimate_r0 * np.sin(theta)
        ax.plot(cx, cy, "k--", linewidth=1.5)

        # Stable color map by drone id
        sorted_ids = sorted([d.drone_id for d in self.drones])
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(sorted_ids))))
        color_map = {did: colors[i % len(colors)] for i, did in enumerate(sorted_ids)}

        # Voronoi arcs
        for start_angle, sweep, did in voronoi_arcs:
            arc_angles = (start_angle + np.linspace(0.0, sweep, 120)) % (2 * np.pi)
            x_arc = self.true_x0 + self.estimate_r0 * np.cos(arc_angles)
            y_arc = self.true_y0 + self.estimate_r0 * np.sin(arc_angles)
            ax.plot(x_arc, y_arc, color=color_map[did], linewidth=2.2)

        # Drone positions
        for d in self.drones:
            ax.plot(d.x, d.y, "ro", markersize=5)

        # Target markers
        for phi_target, did in phi_targets:
            tx = self.true_x0 + self.estimate_r0 * np.cos(phi_target)
            ty = self.true_y0 + self.estimate_r0 * np.sin(phi_target)
            ax.plot(tx, ty, marker="x", color=color_map[did], markersize=8, markeredgewidth=1.8)
            ax.plot([self.true_x0, tx], [self.true_y0, ty], color=color_map[did], alpha=0.25, linewidth=1.0)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(self.sim_map.xlim)
        ax.set_ylim(self.sim_map.ylim)
        ax.grid(True, alpha=0.25)

    def save_voronoi_snapshot(self, out_path="final_voronoi_partition.png"):
        """Save final Voronoi partition image."""
        if not self.drones:
            return

        if self.estimate_r0 is None:
            self.estimate_r0 = float(np.mean([d.estimate_r0 for d in self.drones]))

        # Recompute arcs/targets from current positions
        voronoi_arcs = []
        phi_targets = []
        if len(self.drones) >= 2:
            for d in self.drones:
                d.phi = np.arctan2(d.y - self.true_y0, d.x - self.true_x0) % (2 * np.pi)

            drones_ccw = sorted(self.drones, key=lambda dr: dr.phi)
            n = len(drones_ccw)
            for i, d in enumerate(drones_ccw):
                prev_d = drones_ccw[(i - 1) % n]
                next_d = drones_ccw[(i + 1) % n]

                gap_prev = (d.phi - prev_d.phi) % (2 * np.pi)
                gap_next = (next_d.phi - d.phi) % (2 * np.pi)
                boundary_prev = (d.phi - 0.5 * gap_prev) % (2 * np.pi)
                boundary_next = (d.phi + 0.5 * gap_next) % (2 * np.pi)
                owned_sweep = (boundary_next - boundary_prev) % (2 * np.pi)
                phi_target = (boundary_prev + 0.5 * owned_sweep) % (2 * np.pi)

                voronoi_arcs.append((boundary_prev, owned_sweep, d.drone_id))
                phi_targets.append((phi_target, d.drone_id))

        fig, ax = plt.subplots(figsize=(7, 7))
        self._draw_voronoi(ax, voronoi_arcs, phi_targets)
        ax.set_title(f"Final Voronoi Arc Partition - Frame {self.frame}")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    def step(self):
        """
        Frame-by-frame control step.

        Includes distributed averaging consensus on radius estimate at each frame,
        then Voronoi + radial/tangential control law.
        """
        self.frame += 1

        # 0) Local measurement update (keep existing measurement logic)
        for d in self.drones:
            camera_view = d.get_camera_view(self.world_field, self.sim_map.x_coords, self.sim_map.y_coords)
            h, w = camera_view.shape
            win = 2
            center_val = np.mean(camera_view[h // 2 - win : h // 2 + win + 1, w // 2 - win : w // 2 + win + 1])
            gps_x, gps_y = d.get_gps_pos()

            if center_val > self.c_star:
                dist_to_center = np.hypot(gps_x - self.true_x0, gps_y - self.true_y0)
                d_i = abs(dist_to_center - self.true_r0)
                d_i_noisy = d_i + np.random.normal(0.0, self.sigma_cam)
                if dist_to_center <= self.true_r0:
                    r_i = dist_to_center + d_i_noisy
                else:
                    r_i = dist_to_center - d_i_noisy
                d.estimate_r0 = max(0.05, float(r_i))

            d.estimate_x0 = self.true_x0
            d.estimate_y0 = self.true_y0

            self.estimates_history[d.drone_id]["x0"].append(d.estimate_x0)
            self.estimates_history[d.drone_id]["y0"].append(d.estimate_y0)
            self.estimates_history[d.drone_id]["r0_pre"].append(d.estimate_r0)

        # 1) Frame-by-frame distributed consensus (small gain for gradual convergence)
        if len(self.drones) > 1:
            alpha = min(self.k_consensus * self.dt, self.k_consensus)
            new_r0 = {}
            for d in self.drones:
                neighbors = [n for n in self.drones if n is not d]
                mean_neighbors = np.mean([n.estimate_r0 for n in neighbors])
                new_r0[d] = d.estimate_r0 + alpha * (mean_neighbors - d.estimate_r0)
            for d in self.drones:
                d.estimate_r0 = float(new_r0[d])

        # Shared radius used by control law this frame
        self.estimate_r0 = float(np.mean([d.estimate_r0 for d in self.drones]))

        # Save post-consensus value for plotting
        for d in self.drones:
            self.estimates_history[d.drone_id]["r0_consensus"].append(d.estimate_r0)

        # 2) Voronoi targets + control law (unchanged in spirit)
        voronoi_arcs = []
        phi_targets = []

        if len(self.drones) >= 2:
            for d in self.drones:
                d.phi = np.arctan2(d.y - self.true_y0, d.x - self.true_x0) % (2 * np.pi)

            drones_ccw = sorted(self.drones, key=lambda dr: dr.phi)
            n = len(drones_ccw)

            for i, d in enumerate(drones_ccw):
                prev_d = drones_ccw[(i - 1) % n]
                next_d = drones_ccw[(i + 1) % n]

                gap_prev = (d.phi - prev_d.phi) % (2 * np.pi)
                gap_next = (next_d.phi - d.phi) % (2 * np.pi)

                boundary_prev = (d.phi - 0.5 * gap_prev) % (2 * np.pi)
                boundary_next = (d.phi + 0.5 * gap_next) % (2 * np.pi)
                owned_sweep = (boundary_next - boundary_prev) % (2 * np.pi)
                phi_target = (boundary_prev + 0.5 * owned_sweep) % (2 * np.pi)

                voronoi_arcs.append((boundary_prev, owned_sweep, d.drone_id))
                phi_targets.append((phi_target, d.drone_id))

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

        # 3) Optional real-time Voronoi plot in a separate figure
        should_plot_voronoi = self.visualize_voronoi and plt.get_backend().lower() != "agg"
        if should_plot_voronoi:
            if self._voronoi_fig is None:
                plt.ion()
                self._voronoi_fig, self._voronoi_ax = plt.subplots(figsize=(7, 7))

            self._voronoi_ax.clear()
            self._draw_voronoi(self._voronoi_ax, voronoi_arcs, phi_targets)
            self._voronoi_ax.set_title(f"Voronoi Arc Partition - Frame {self.frame}")
            self._voronoi_fig.canvas.draw_idle()
            self._voronoi_fig.canvas.flush_events()
            plt.pause(0.01)

        # 4) Integrate dynamics
        for d in self.drones:
            d.update_position(self.dt)
            self.estimates_history[d.drone_id]["r0_post"].append(d.estimate_r0)
