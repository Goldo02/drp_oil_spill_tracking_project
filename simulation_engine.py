import numpy as np
from drone import Drone


class SimulationEngine:
    """Coordinator for the static multi-drone boundary simulation."""

    def __init__(
        self,
        sim_map,
        oil_spill,
        controller=None,
        x_min=-10.0,
        x_max=10.0,
        y_min=-10.0,
        y_max=10.0,
        resolution=0.1,
        communication_radius_cells=205,
        verbose=True,
    ):
        self.sim_map = sim_map
        self.oil_spill = oil_spill
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.y_min, self.y_max = float(y_min), float(y_max)

        if hasattr(self.sim_map, "xlim"):
            self.x_min, self.x_max = map(float, self.sim_map.xlim)
        if hasattr(self.sim_map, "ylim"):
            self.y_min, self.y_max = map(float, self.sim_map.ylim)

        self.resolution = float(resolution)
        if hasattr(self.sim_map, "dx") and float(self.sim_map.dx) > 0.0:
            self.resolution = float(abs(self.sim_map.dx))

        self.verbose = bool(verbose)

        self.Nx = int(round((self.x_max - self.x_min) / self.resolution))
        self.Ny = int(round((self.y_max - self.y_min) / self.resolution))
        self.grid_shape = (self.Nx, self.Ny)
        self.grid_bounds = (self.x_min, self.x_max, self.y_min, self.y_max)

        self.world_field = self._get_world_field()

        dx = self.sim_map.dx if self.sim_map.dx > 0 else self.resolution
        dy = self.sim_map.dy if self.sim_map.dy > 0 else self.resolution
        self.communication_radius_cells = int(communication_radius_cells)
        self.communication_radius = self.communication_radius_cells * 0.5 * (abs(dx) + abs(dy))

        self.controller = controller
        self.drones = []
        self.frame = 0

    def _get_world_field(self):
        field = self.oil_spill.get_field(self.sim_map.X, self.sim_map.Y)
        return np.asarray(field, dtype=float)

    def add_drone(self, drone_id, x, y):
        drone = Drone(
            drone_id=drone_id,
            x=x,
            y=y,
            grid_shape=self.grid_shape,
            grid_bounds=self.grid_bounds,
        )
        self.drones.append(drone)
        return drone

    def initialize_world_boundary(self):
        x_coords = getattr(self.sim_map, "x_coords", None)
        y_coords = getattr(self.sim_map, "y_coords", None)
        if x_coords is None:
            x_coords = np.linspace(self.x_min, self.x_max, self.Nx)
        if y_coords is None:
            y_coords = np.linspace(self.y_min, self.y_max, self.Ny)

        self.world_boundary_points = self.controller.initialize_known_boundary(
            self.world_field,
            x_coords=x_coords,
            y_coords=y_coords,
            force_closed=True,
        )
        return self.world_boundary_points.copy()

    def spawn_drones_on_boundary(self, num_drones, rng=None):
        if num_drones <= 0:
            return []

        boundary_points = np.asarray(self.world_boundary_points, dtype=float)
        if boundary_points.size == 0:
            boundary_points = self.initialize_world_boundary()
        if boundary_points.size == 0:
            return []

        if rng is None:
            rng = np.random.default_rng()

        idx = rng.choice(len(boundary_points), size=num_drones, replace=(len(boundary_points) < num_drones))

        self.drones = []
        source_x = np.asarray(getattr(self.sim_map, "x_coords", np.linspace(self.x_min, self.x_max, self.world_field.shape[0])), dtype=float)
        source_y = np.asarray(getattr(self.sim_map, "y_coords", np.linspace(self.y_min, self.y_max, self.world_field.shape[1])), dtype=float)
        contour_grid = self.controller.build_boundary_grid(self.world_field, source_x, source_y)

        target_x = np.linspace(self.x_min, self.x_max, self.Nx)
        target_y = np.linspace(self.y_min, self.y_max, self.Ny)
        resampled_contour = np.zeros(self.grid_shape, dtype=float)
        for ix, x in enumerate(target_x):
            xi = int(np.argmin(np.abs(source_x - x)))
            for iy, y in enumerate(target_y):
                yi = int(np.argmin(np.abs(source_y - y)))
                resampled_contour[ix, iy] = contour_grid[xi, yi]

        for drone_idx, point_idx in enumerate(idx):
            x, y = boundary_points[int(point_idx)]
            drone = self.add_drone(
                drone_id=f"D{drone_idx}",
                x=float(x),
                y=float(y),
            )
            drone.grid = resampled_contour.copy()
            drone.settling_counter = 0
            drone.last_control_mode = "equi_distant"

        for drone in self.drones:
            drone.known_boundary_points = self.controller.known_boundary_points.copy()
            drone.known_positions = {
                other.drone_id: np.array([other.x, other.y], dtype=float)
                for other in self.drones
            }

        return list(self.drones)

    def compute_mean_grid(self):
        if not self.drones:
            return np.zeros(self.grid_shape, dtype=float)
        return np.mean([np.asarray(drone.grid, dtype=float) for drone in self.drones], axis=0)

    def _print_control_snapshot(self, header):
        if not self.verbose or not self.drones:
            return

        print(f"{header}")
        for drone in self.drones:
            pos = np.asarray([drone.x, drone.y], dtype=float)
            target = getattr(drone, "target_centroid", None)
            target_str = f"target=({target[0]:.3f}, {target[1]:.3f})" if target is not None else "target=None"

            action = np.asarray(getattr(drone, "last_control_vector", np.zeros(2, dtype=float)), dtype=float)
            speed = float(np.linalg.norm(action))

            ring_info = getattr(drone, "last_ring_info", None)
            if ring_info is not None:
                current = ring_info.get("current", {})
                ring_str = f"theta={current.get('angle', np.nan):.3f}, cell_len={current.get('cell_arc_length', np.nan):.3f}"
            else:
                ring_str = "theta=NA, cell_len=NA"

            mode = getattr(drone, "last_control_mode", "unknown")
            print(f"    {drone.drone_id}: pos=({pos[0]:.3f}, {pos[1]:.3f}), mode={mode}, {target_str}, {ring_str}, action=({action[0]:.3f}, {action[1]:.3f}), speed={speed:.3f}")

    def _apply_actions(self):
        # Il controller calcola già autonomamente multi-hop e target di Voronoi
        actions = self.controller.compute_actions(
            self.drones,
            world_field=self.world_field,
            x_coords=self.sim_map.x_coords,
            y_coords=self.sim_map.y_coords,
        )

        # Muove i droni e li corregge sul bordo
        for drone in self.drones:
            action = actions.get(drone.drone_id, np.zeros(2, dtype=float))
            drone.action(action, bounds=(self.sim_map.xlim, self.sim_map.ylim))
            if hasattr(self.controller, "project_drone_to_boundary"):
                self.controller.project_drone_to_boundary(drone)

    def get_visualization_data(self):
        return {
            "frame": self.frame,
            "world_field": self.world_field.copy(),
            "mean_grid": self.compute_mean_grid().copy(),
            "drones": self.drones,
            "communication_radius": self.communication_radius,
        }

    def step(self):
        self.frame += 1

        if self.verbose:
            print(f"\nFrame {self.frame}")

        self._apply_actions()

        if self.verbose:
            mode_summary = ", ".join(f"{d.drone_id}:{getattr(d, 'last_control_mode', 'unknown')}" for d in self.drones)
            print(f"  Frame summary: modes: {mode_summary}")
            self._print_control_snapshot("  Control snapshot")

    def run(self, iterations, render_callback=None):
        for _ in range(int(iterations)):
            self.step()
            if render_callback is not None:
                render_callback(self.get_visualization_data())
