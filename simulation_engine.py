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
        consensus_rounds=10,
        dt=1.0,
        verbose=True,
    ):
        self.sim_map = sim_map
        self.oil_spill = oil_spill

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)

        self.resolution = float(resolution)
        self.dt = float(dt)

        self.sensor_size = int(sensor_size)
        self.measure_every = max(
            1,
            int(measure_every),
        )

        self.occupancy_threshold = float(
            occupancy_threshold
        )

        self.temporal_alpha = (
            float(temporal_alpha)
            if temporal_alpha is not None
            else None
        )

        self.consensus_rounds = max(
            1,
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

    # ==================================================================
    # SENSING
    # ==================================================================

    def _perform_measurement(self):
        """Perform sensing and local grid updates."""

        for drone in self.drones:

            edge_points = drone.sense(
                self.world_field,
                self.sim_map.x_coords,
                self.sim_map.y_coords,
            )

            drone.update_grid(
                edge_points=edge_points,
                x_min=self.x_min,
                y_min=self.y_min,
                resolution=self.resolution,
                alpha=self.temporal_alpha,
            )

    # ==================================================================
    # CONSENSUS
    # ==================================================================

    def _perform_consensus(self):
        """Run the configured number of consensus iterations."""

        for _ in range(self.consensus_rounds):
            self.controller.consensus_step(
                self.drones
            )

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
            3. consensus;
            4. diagnostics;
            5. distributed control;
            6. drone motion.
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
                else "consensus"
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
        # Consensus
        # --------------------------------------------------------------

        for round_idx in range(
            self.consensus_rounds
        ):

            self.controller.consensus_step(
                self.drones
            )

            self._record_measurement_trace()

            self._print_error_snapshot(
                f"  Consensus iteration "
                f"{round_idx + 1}/"
                f"{self.consensus_rounds}"
            )

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
