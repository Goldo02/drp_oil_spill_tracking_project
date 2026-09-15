import numpy as np

from drone import Drone
from controller import Controller


class SimulationEngine:
    """
    Main coordinator of the multi-drone simulation.

    Responsibilities:
        - update the environment;
        - trigger drone sensing;
        - update local occupancy grids;
        - execute distributed consensus;
        - compute diagnostics;
        - request actions from the controller;
        - apply actions to drones;
        - expose state to the visualizer.
    """

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
        sensor_size=100,
        measure_every=3,
        communication_radius_cells=205,
        fully_connected=False,
        occupancy_threshold=0.5,
        temporal_alpha=0.05,
        consensus_rounds=0,
        dt=1.0,
        verbose=True,
    ):
        self.sim_map = sim_map
        self.oil_spill = oil_spill

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)

        # The robot occupancy grid must share the same physical domain as the
        # underlying simulation map. Otherwise the indices of the grid and the
        # continuous world coordinates refer to different coordinate systems.
        if hasattr(self.sim_map, "xlim"):
            self.x_min, self.x_max = map(float, self.sim_map.xlim)
        if hasattr(self.sim_map, "ylim"):
            self.y_min, self.y_max = map(float, self.sim_map.ylim)

        self.resolution = float(resolution)
        if hasattr(self.sim_map, "dx") and float(self.sim_map.dx) > 0.0:
            self.resolution = float(abs(self.sim_map.dx))
        self.dt = float(dt)

        self.sensor_size = int(sensor_size)
        self.measure_every = max(
            1,
            int(measure_every),
        )

        self.occupancy_threshold = float(
            occupancy_threshold
        )

        self.temporal_alpha = None

        self.consensus_rounds = max(
            0,
            int(consensus_rounds),
        )

        self.verbose = bool(verbose)

        # --------------------------------------------------------------
        # Occupancy grid
        # --------------------------------------------------------------

        self.Nx = int(
            round(
                (self.x_max - self.x_min)
                / self.resolution
            )
        )

        self.Ny = int(
            round(
                (self.y_max - self.y_min)
                / self.resolution
            )
        )

        self.grid_shape = (
            self.Nx,
            self.Ny,
        )

        self.grid_bounds = (
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
        )

        # --------------------------------------------------------------
        # Environment
        # --------------------------------------------------------------

        self.world_field = self._get_world_field()

        # --------------------------------------------------------------
        # Communication
        # --------------------------------------------------------------

        dx = (
            self.sim_map.dx
            if self.sim_map.dx > 0
            else self.resolution
        )

        dy = (
            self.sim_map.dy
            if self.sim_map.dy > 0
            else self.resolution
        )

        self.communication_radius_cells = int(
            communication_radius_cells
        )

        self.communication_radius = (
            self.communication_radius_cells
            * 0.5
            * (abs(dx) + abs(dy))
        )

        # --------------------------------------------------------------
        # Controller
        # --------------------------------------------------------------

        if controller is None:
            self.controller = Controller(
                sim_map=self.sim_map,
                communication_radius=self.communication_radius,
                fully_connected=fully_connected,
                occupancy_threshold=self.occupancy_threshold,
                resolution=self.resolution,
            )
        else:
            self.controller = controller

        # --------------------------------------------------------------
        # Simulation state
        # --------------------------------------------------------------

        self.drones = []
        self.frame = 0

        self.error_history = []
        self.mean_grid_history = []

        self.latest_mean_grid = np.zeros(
            self.grid_shape,
            dtype=float,
        )

        self.measurement_consensus_history = []
        self._current_measurement_trace = None

    # ==================================================================
    # ENVIRONMENT
    # ==================================================================

    def _get_world_field(self):
        """Return the current environment field."""
        field = self.oil_spill.get_field(
            self.sim_map.X,
            self.sim_map.Y,
        )

        return np.asarray(
            field,
            dtype=float,
        )

    def _update_environment(self):
        """Advance the environment by one simulation timestep."""
        self.oil_spill.update(self.dt)
        self.world_field = self._get_world_field()

    # ==================================================================
    # DRONES
    # ==================================================================

    def add_drone(
        self,
        drone_id,
        x,
        y,
        gps_noise=0.1,
        camera_noise=0.1,
    ):
        """Create and register a drone."""

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

        return drone

    def initialize_world_boundary(self):
        """Precompute the closed boundary contour and force the controller to use it immediately."""
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
        """Spawn drones directly on random boundary contour points, not in open space."""
        if num_drones <= 0:
            return []

        boundary_points = np.asarray(self.world_boundary_points, dtype=float)
        if boundary_points.size == 0:
            boundary_points = self.initialize_world_boundary()

        if boundary_points.size == 0:
            return []

        if rng is None:
            rng = np.random.default_rng()

        if len(boundary_points) >= num_drones:
            idx = rng.choice(len(boundary_points), size=num_drones, replace=False)
        else:
            idx = rng.choice(len(boundary_points), size=num_drones, replace=True)

        self.drones = []
        source_x = np.asarray(getattr(self.sim_map, "x_coords", np.linspace(self.x_min, self.x_max, self.world_field.shape[0])), dtype=float)
        source_y = np.asarray(getattr(self.sim_map, "y_coords", np.linspace(self.y_min, self.y_max, self.world_field.shape[1])), dtype=float)
        contour_grid = self.controller.build_boundary_grid(
            self.world_field,
            source_x,
            source_y,
        )

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
                gps_noise=0.03,
                camera_noise=0.03,
            )
            drone.grid = resampled_contour.copy()
            drone.settling_counter = 0
            drone.last_control_mode = "equi_distant"

        for drone in self.drones:
            drone.known_positions = {
                other.drone_id: np.array([other.x, other.y], dtype=float)
                for other in self.drones
            }

        return list(self.drones)

    # ==================================================================
    # SENSING
    # ==================================================================

    def _perform_measurement(self):
        """No sensing updates in this no-consensus baseline; local occupancy grids remain frozen."""

        for drone in self.drones:
            # Sensor-driven occupancy updates are intentionally disabled.
            # Each robot keeps its local grid fixed to the initial contour state.
            continue

    # ==================================================================
    # CONSENSUS
    # ==================================================================

    def _perform_consensus(self):
        """Consensus is intentionally disabled for this baseline."""

        return None

    # ==================================================================
    # DIAGNOSTICS
    # ==================================================================

    def compute_mean_grid(self):
        """Return the mean occupancy grid."""

        if not self.drones:
            return np.zeros(
                self.grid_shape,
                dtype=float,
            )

        return np.mean(
            [
                np.asarray(
                    drone.grid,
                    dtype=float,
                )
                for drone in self.drones
            ],
            axis=0,
        )

    def compute_disagreement_error(self):
        """Return mean L2 disagreement from the global mean."""

        if not self.drones:
            return (
                0.0,
                np.zeros(
                    self.grid_shape,
                    dtype=float,
                ),
            )

        mean_grid = self.compute_mean_grid()

        errors = [
            np.linalg.norm(
                np.asarray(
                    drone.grid,
                    dtype=float,
                )
                - mean_grid
            )
            for drone in self.drones
        ]

        return float(np.mean(errors)), mean_grid

    def _drone_error_snapshot(self):
        mean_grid = self.compute_mean_grid()

        return {
            drone.drone_id: float(
                np.linalg.norm(
                    np.asarray(
                        drone.grid,
                        dtype=float,
                    )
                    - mean_grid
                )
            )
            for drone in self.drones
        }

    def _print_error_snapshot(self, header):
        if not self.verbose:
            return

        snapshot = self._drone_error_snapshot()

        if snapshot:
            mean_error = float(
                np.mean(
                    list(snapshot.values())
                )
            )

            max_error = float(
                np.max(
                    list(snapshot.values())
                )
            )
        else:
            mean_error = 0.0
            max_error = 0.0

        ordered = ", ".join(
            f"{drone_id}={value:.6f}"
            for drone_id, value in snapshot.items()
        )

        print(
            f"{header} | "
            f"mean_error={mean_error:.6f} | "
            f"max_error={max_error:.6f}"
        )

        print(
            f"    per-drone: {ordered}"
        )

    def _print_control_snapshot(self, header):
        if not self.verbose:
            return

        if not self.drones:
            return

        print(f"{header}")
        for drone in self.drones:
            pos = np.asarray([drone.x, drone.y], dtype=float)
            target = getattr(drone, "target_centroid", None)
            if target is not None:
                target = np.asarray(target, dtype=float)
                target_str = f"target=({target[0]:.3f}, {target[1]:.3f})"
            else:
                target_str = "target=None"

            action = np.asarray(
                getattr(drone, "last_control_vector", np.zeros(2, dtype=float)),
                dtype=float,
            )
            action_str = f"action=({action[0]:.3f}, {action[1]:.3f})"
            speed = float(np.linalg.norm(action))

            ring_info = getattr(drone, "last_ring_info", None)
            if ring_info is not None:
                current = ring_info.get("current", {})
                theta = current.get("angle", np.nan)
                cell_len = current.get("cell_arc_length", np.nan)
                ring_str = f"theta={theta:.3f}, cell_len={cell_len:.3f}"
            else:
                ring_str = "theta=NA, cell_len=NA"

            mode = getattr(drone, "last_control_mode", "unknown")
            print(
                f"    {drone.drone_id}: pos=({pos[0]:.3f}, {pos[1]:.3f}), "
                f"mode={mode}, {target_str}, {ring_str}, "
                f"{action_str}, speed={speed:.3f}"
            )

    # ==================================================================
    # MEASUREMENT HISTORY
    # ==================================================================

    def _start_new_measurement_trace(self):

        if self._current_measurement_trace is not None:

            self.measurement_consensus_history.append(
                {
                    drone_id: list(values)
                    for drone_id, values
                    in self._current_measurement_trace.items()
                }
            )

        self._current_measurement_trace = {
            drone.drone_id: []
            for drone in self.drones
        }

    def _record_measurement_trace(self):

        if self._current_measurement_trace is None:
            self._current_measurement_trace = {
                drone.drone_id: []
                for drone in self.drones
            }

        snapshot = self._drone_error_snapshot()

        for drone_id, value in snapshot.items():
            self._current_measurement_trace[
                drone_id
            ].append(value)

    # ==================================================================
    # CONTROL
    # ==================================================================

    def _apply_actions(self):
        """Compute distributed actions and apply them to drones."""

        actions = self.controller.compute_actions(
            self.drones,
            world_field=self.world_field,
            x_coords=self.sim_map.x_coords,
            y_coords=self.sim_map.y_coords,
        )

        for drone in self.drones:

            action = actions.get(
                drone.drone_id,
                np.zeros(2, dtype=float),
            )

            drone.action(
                action,
                bounds=(
                    self.sim_map.xlim,
                    self.sim_map.ylim,
                ),
            )
            if hasattr(self.controller, "project_drone_to_boundary"):
                self.controller.project_drone_to_boundary(drone)

        if hasattr(self.controller, "_update_multihop_positions"):
            self.controller._update_multihop_positions(self.drones)
        if hasattr(self.controller, "_compute_voronoi_target"):
            self.controller._compute_voronoi_target(self.drones)

    # ==================================================================
    # VISUALIZATION
    # ==================================================================

    def get_visualization_data(self):
        """Return state required by the visualizer."""

        error, mean_grid = (
            self.compute_disagreement_error()
        )

        return {
            "frame": self.frame,
            "world_field": self.world_field.copy(),
            "mean_grid": mean_grid.copy(),
            "disagreement_error": error,
            "drones": self.drones,
            "communication_radius": self.communication_radius,
        }

    # ==================================================================
    # SENSOR DEBUGGING
    # ==================================================================

    def _print_sensor_status(self):

        for drone in self.drones:

            if (
                drone.edge_detected
                and drone.last_edge_point is not None
            ):

                print(
                    f"    {drone.drone_id}: "
                    f"edge_points={drone.last_edge_count}, "
                    f"nearest_edge=("
                    f"{drone.last_edge_point[0]:.4f}, "
                    f"{drone.last_edge_point[1]:.4f})"
                )

            else:

                print(
                    f"    {drone.drone_id}: "
                    f"no edge detected"
                )

    # ==================================================================
    # SIMULATION STEP
    # ==================================================================

    def step(self):
        """
        Execute one complete simulation timestep.

        Order:
            1. update environment;
            2. sensing;
            3. diagnostics;
            4. local control;
            5. drone motion.
        """
        
        self.frame += 1

        measurement_frame = (
            (self.frame - 1)
            % self.measure_every
            == 0
        )

        if self.verbose:

            frame_type = (
                "measurement"
                if measurement_frame
                else "cycle"
            )

            print(
                f"\nFrame {self.frame} "
                f"[{frame_type}]"
            )

        # --------------------------------------------------------------
        # Environment
        # --------------------------------------------------------------

        self._update_environment()

        # --------------------------------------------------------------
        # Measurement
        # --------------------------------------------------------------

        if measurement_frame:

            self._start_new_measurement_trace()

            self._perform_measurement()

            self._record_measurement_trace()

            self._print_error_snapshot(
                "  After sensing"
            )

            if self.verbose:
                self._print_sensor_status()

        # --------------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------------

        error, mean_grid = (
            self.compute_disagreement_error()
        )

        self.error_history.append(error)

        self.mean_grid_history.append(
            mean_grid.copy()
        )

        self.latest_mean_grid = mean_grid

        # --------------------------------------------------------------
        # Control
        # --------------------------------------------------------------

        self._apply_actions()

        if self.verbose:

            mode_summary = ", ".join(
                f"{drone.drone_id}:"
                f"{getattr(drone, 'last_control_mode', 'unknown')}"
                for drone in self.drones
            )

            print(
                f"  Frame summary: "
                f"global_disagreement="
                f"{error:.6f} | "
                f"modes: {mode_summary}"
            )

            self._print_control_snapshot(
                "  Control snapshot"
            )

        return error

    # ==================================================================
    # RUN
    # ==================================================================

    def run(
        self,
        iterations,
        render_callback=None,
    ):
        """Run the simulation."""

        for _ in range(int(iterations)):

            self.step()

            if render_callback is not None:
                render_callback(
                    self.get_visualization_data()
                )

        self.finalize_histories()

    # ==================================================================
    # HISTORY
    # ==================================================================

    def finalize_histories(self):

        if (
            self._current_measurement_trace is not None
            and any(
                len(values) > 0
                for values
                in self._current_measurement_trace.values()
            )
        ):

            self.measurement_consensus_history.append(
                {
                    drone_id: list(values)
                    for drone_id, values
                    in self._current_measurement_trace.items()
                }
            )

        self._current_measurement_trace = None
