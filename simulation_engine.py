import numpy as np

from drone import Drone
from controller import Controller, DroneController

class SimulationEngine:
    """
    Physics simulator, local-message router, and diagnostics collector.

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
        x_min=-10.0,
        x_max=10.0,
        y_min=-10.0,
        y_max=10.0,
        resolution=0.1,
        sensor_size=100,
        measure_every=3,
        communication_radius_cells=205,
        fully_connected=False,
        occupancy_threshold=0.6,
        temporal_alpha=0.05,
        consensus_rounds=3,
        dt=1.0,
        verbose=True,
        closure_min_enclosed_false_cells=250,
        mapping_point_radius_cells=1,
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
        self.measure_every = max(1, int(measure_every))
        self.occupancy_threshold = float(occupancy_threshold)
        self.temporal_alpha = None
        self.consensus_rounds = max(1, int(consensus_rounds))
        self.verbose = bool(verbose)
        self.fully_connected = bool(fully_connected)

        self.Nx = int(round((self.x_max - self.x_min) / self.resolution))
        self.Ny = int(round((self.y_max - self.y_min) / self.resolution))
        self.grid_shape = (self.Nx, self.Ny)
        self.grid_bounds = (self.x_min, self.x_max, self.y_min, self.y_max)

        self.world_field = self._get_world_field()

        dx = self.sim_map.dx if self.sim_map.dx > 0 else self.resolution
        dy = self.sim_map.dy if self.sim_map.dy > 0 else self.resolution
        self.communication_radius_cells = int(communication_radius_cells)
        self.communication_radius = (
            self.communication_radius_cells * 0.5 * (abs(dx) + abs(dy))
        )

        self.drones = []
        self.frame = 0
        self.error_history = []
        self.mean_grid_history = []
        self.latest_mean_grid = np.zeros(self.grid_shape, dtype=float)
        self.measurement_consensus_history = []
        self._current_measurement_trace = None
        self.control_state = "mapping"
        self.transition_frame = None
        self.closed_boundary_points = np.empty((0, 2), dtype=float)
        self.patrol_forgetting_alpha = 0.995
        self.patrol_start_frame = None
        self.closure_min_enclosed_false_cells = int(closure_min_enclosed_false_cells)
        self.mapping_point_radius_cells = max(0, int(mapping_point_radius_cells))
        self.last_closure_enclosed_false_cells = 0

    def _get_world_field(self):
        """Return the current environment field."""
        return np.asarray(self.oil_spill.get_field(self.sim_map.X, self.sim_map.Y), dtype=float)

    def _update_environment(self):
        """Advance the environment by one simulation timestep."""
        self.oil_spill.update(self.dt)
        self.world_field = self._get_world_field()

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
            occupancy_threshold=self.occupancy_threshold,
            controller=DroneController(
                sim_map=self.sim_map,
                communication_radius=self.communication_radius,
                fully_connected=self.fully_connected,
                occupancy_threshold=self.occupancy_threshold,
                resolution=self.resolution,
            ),
        )

        self.drones.append(drone)
        drone.closure_min_enclosed_false_cells = int(self.closure_min_enclosed_false_cells)

        return drone

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
                alpha=None,
                point_radius_cells=self.mapping_point_radius_cells,
                occupancy_threshold=self.occupancy_threshold,
                x_coords=self.sim_map.x_coords,
                y_coords=self.sim_map.y_coords,
            )

    def _apply_forgetting_if_needed(self):
        for drone in self.drones:
            if getattr(drone, "control_state", "mapping") != "boundary_patrol":
                continue
            drone.apply_forgetting(
                self.patrol_forgetting_alpha,
                occupancy_threshold=self.occupancy_threshold,
            )

    def _refresh_position_estimates(self):
        """Refresh each drone's GPS-based position estimate."""
        for drone in self.drones:
            if hasattr(drone, "update_position_estimate"):
                drone.update_position_estimate()

    def _get_neighbors(self, drone):
        if self.fully_connected:
            return [other for other in self.drones if other is not drone]

        drone_position = Controller._estimated_position(drone)
        return [
            other
            for other in self.drones
            if other is not drone
            and float(np.linalg.norm(drone_position - Controller._estimated_position(other)))
            <= self.communication_radius
        ]

    def _exchange_consensus_messages(self):
        """Deliver one synchronous round of local map messages."""
        messages = {drone.drone_id: drone.create_consensus_message() for drone in self.drones}
        delivered = {
            drone.drone_id: [
                messages[neighbor.drone_id] for neighbor in self._get_neighbors(drone)
            ]
            for drone in self.drones
        }

        for drone in self.drones:
            drone.consensus_step(
                delivered[drone.drone_id],
                own_grid=messages[drone.drone_id]["grid"],
                own_signal_grid=messages[drone.drone_id]["signal_grid"],
                own_information_grid=messages[drone.drone_id]["information_grid"],
                occupancy_threshold=self.occupancy_threshold,
                frame=getattr(self, "frame", None),
            )

    def _perform_consensus(self):
        """Run the configured number of consensus iterations."""
        for _ in range(self.consensus_rounds):
            self._exchange_consensus_messages()

    def compute_mean_grid(self):
        """Return the information-weighted mean occupancy probability grid."""

        if not self.drones:
            return np.full(self.grid_shape, 0.5, dtype=float)

        mean_signal = np.mean(
            [
                np.asarray(drone.occupancy_signal_grid, dtype=float)
                for drone in self.drones
            ],
            axis=0,
        )
        mean_information = self.compute_mean_information_grid()
        return Drone.probability_from_signal(mean_signal, mean_information)

    def compute_mean_information_grid(self):
        """Return the mean amount of available map information per cell."""

        if not self.drones:
            return np.zeros(self.grid_shape, dtype=float)

        return np.mean(
            [np.asarray(drone.information_grid, dtype=float) for drone in self.drones],
            axis=0,
        )

    def compute_mean_boundary_grid(self):
        """Return the consensus boundary induced by the probability map."""

        mean_grid = self.compute_mean_grid()
        mean_information = self.compute_mean_information_grid()
        return Drone.boundary_mask_from_probability(
            mean_grid,
            mean_information,
            occupancy_threshold=self.occupancy_threshold,
        ).astype(float)

    def compute_disagreement_error(self):
        """Return mean L2 disagreement from the global mean."""

        if not self.drones:
            return 0.0, np.zeros(self.grid_shape, dtype=float)

        mean_grid = self.compute_mean_grid()
        errors = [
            np.linalg.norm(np.asarray(drone.grid, dtype=float) - mean_grid)
            for drone in self.drones
        ]
        return float(np.mean(errors)), mean_grid

    def _drone_error_snapshot(self):
        mean_grid = self.compute_mean_grid()

        return {
            drone.drone_id: float(np.linalg.norm(np.asarray(drone.grid, dtype=float) - mean_grid))
            for drone in self.drones
        }

    def _print_error_snapshot(self, header):
        if not self.verbose:
            return

        snapshot = self._drone_error_snapshot()

        values = list(snapshot.values())
        mean_error = float(np.mean(values)) if values else 0.0
        max_error = float(np.max(values)) if values else 0.0

        ordered = ", ".join(
            f"{drone_id}={value:.6f}" for drone_id, value in snapshot.items()
        )

        print(f"{header} | mean_error={mean_error:.6f} | max_error={max_error:.6f}")
        print(f"    per-drone: {ordered}")

    def _start_new_measurement_trace(self):

        if self._current_measurement_trace is not None:

            self.measurement_consensus_history.append(
                {
                    drone_id: list(values)
                    for drone_id, values in self._current_measurement_trace.items()
                }
            )

        self._current_measurement_trace = {drone.drone_id: [] for drone in self.drones}

    def _record_measurement_trace(self):

        if self._current_measurement_trace is None:
            self._current_measurement_trace = {
                drone.drone_id: [] for drone in self.drones
            }

        snapshot = self._drone_error_snapshot()

        for drone_id, value in snapshot.items():
            self._current_measurement_trace[drone_id].append(value)

    @staticmethod
    def _sensed_position(drone):
        estimate = getattr(drone, "last_gps_position", None)
        if estimate is not None:
            estimate = np.asarray(estimate, dtype=float)
            if estimate.shape == (2,) and np.all(np.isfinite(estimate)):
                return estimate.copy()
        if hasattr(drone, "update_position_estimate"):
            return np.asarray(drone.update_position_estimate(), dtype=float)
        return Controller._estimated_position(drone)

    def _exchange_positions_multihop(self):
        sensed_positions = {
            drone.drone_id: self._sensed_position(drone)
            for drone in self.drones
        }
        sensed_arcs = {
            drone.drone_id: getattr(drone, "boundary_s", None)
            for drone in self.drones
        }

        for drone in self.drones:
            drone.known_positions = {
                drone.drone_id: sensed_positions[drone.drone_id].copy(),
            }
            drone.known_boundary_arcs = {}
            if sensed_arcs[drone.drone_id] is not None:
                drone.known_boundary_arcs[drone.drone_id] = float(
                    sensed_arcs[drone.drone_id]
                )

        hop_count = max(1, len(self.drones))
        for _ in range(hop_count):
            pending_updates = [{} for _ in self.drones]
            pending_arc_updates = [{} for _ in self.drones]

            for i, drone_i in enumerate(self.drones):
                for j, drone_j in enumerate(self.drones):
                    if i == j:
                        continue
                    in_range = getattr(self, "fully_connected", False) or (
                        float(
                            np.linalg.norm(
                                sensed_positions[drone_i.drone_id]
                                - sensed_positions[drone_j.drone_id]
                            )
                        )
                        <= self.communication_radius
                    )
                    if in_range:
                        pending_updates[i].update(drone_j.known_positions)
                        pending_arc_updates[i].update(
                            getattr(drone_j, "known_boundary_arcs", {})
                        )

            for i, drone in enumerate(self.drones):
                drone_id = drone.drone_id
                for known_id, position in pending_updates[i].items():
                    if known_id != drone_id:
                        drone.known_positions[known_id] = np.asarray(position, dtype=float)
                for known_id, boundary_s in pending_arc_updates[i].items():
                    if boundary_s is not None:
                        drone.known_boundary_arcs[known_id] = float(boundary_s)
                drone.known_positions[drone_id] = sensed_positions[drone_id].copy()
                if sensed_arcs[drone_id] is not None:
                    drone.known_boundary_arcs[drone_id] = float(sensed_arcs[drone_id])

    def _apply_actions(self):
        """Apply actions selected from each drone's local mode and knowledge."""
        for drone in self.drones:
            if hasattr(drone, "update_position_estimate"):
                drone.update_position_estimate()
            if getattr(drone, "control_state", "mapping") != "mapping":
                drone.update_boundary_projection()
        if any(getattr(d, "control_state", "mapping") != "mapping" for d in self.drones):
            self._exchange_positions_multihop()
        actions = {}
        for drone in self.drones:
            if getattr(drone, "control_state", "mapping") == "mapping":
                actions[drone.drone_id] = drone.compute_action(
                    None, self.sim_map.x_coords, self.sim_map.y_coords, dt=self.dt
                )
            else:
                actions[drone.drone_id] = drone.compute_action(dt=self.dt)
        for drone in self.drones:
            drone.action(actions[drone.drone_id], dt=self.dt,
                         bounds=(self.sim_map.xlim, self.sim_map.ylim))
            if getattr(drone, "control_state", "mapping") != "mapping":
                if hasattr(drone, "update_position_estimate"):
                    drone.update_position_estimate()
                drone.update_boundary_projection()

    def _observe_local_states(self):
        """Derive engine compatibility fields passively for logging and output."""
        states = [getattr(drone, "control_state", "mapping") for drone in self.drones]
        if not states:
            self.control_state = "mapping"
            return
        counts = {state: states.count(state) for state in set(states)}
        self.control_state = max(counts, key=lambda state: (counts[state], state))
        if self.transition_frame is None and any(state in ("dynamic_lloyd", "boundary_patrol") for state in states):
            self.transition_frame = self.frame
        if self.patrol_start_frame is None and any(state == "boundary_patrol" for state in states):
            self.patrol_start_frame = self.frame
        contours = [np.asarray(d.contour_points, dtype=float) for d in self.drones
                    if getattr(d, "contour_points", None) is not None]
        if contours:
            self.closed_boundary_points = max(contours, key=lambda points: points.shape[0]).copy()
        enclosed = [getattr(drone, "last_closure_enclosed_false_cells", 0) for drone in self.drones]
        self.last_closure_enclosed_false_cells = max(enclosed, default=0)

    def get_visualization_data(self):
        """Return state required by the visualizer."""

        error, mean_grid = self.compute_disagreement_error()
        mean_information_grid = self.compute_mean_information_grid()
        mean_boundary_grid = Drone.boundary_mask_from_probability(
            mean_grid,
            mean_information_grid,
            occupancy_threshold=self.occupancy_threshold,
        ).astype(float)

        return {
            "frame": self.frame,
            "world_field": self.world_field.copy(),
            "mean_grid": mean_grid.copy(),
            "mean_information_grid": mean_information_grid.copy(),
            "mean_boundary_grid": mean_boundary_grid.copy(),
            "disagreement_error": error,
            "drones": self.drones,
            "communication_radius": self.communication_radius,
            "control_state": self.control_state,
            "transition_frame": self.transition_frame,
            "closed_boundary_points": self.closed_boundary_points.copy(),
            "error_history": list(self.error_history),
            "closure_enclosed_false_cells": self.last_closure_enclosed_false_cells,
            "closure_min_enclosed_false_cells": self.closure_min_enclosed_false_cells,
            "grid_bounds": self.grid_bounds,
            "grid_resolution": self.resolution,
            "occupancy_threshold": self.occupancy_threshold,
        }

    def _print_sensor_status(self):

        for drone in self.drones:

            if drone.edge_detected and drone.last_edge_point is not None:

                print(
                    f"    {drone.drone_id}: "
                    f"edge_points={drone.last_edge_count}, "
                    f"nearest_edge=("
                    f"{drone.last_edge_point[0]:.4f}, "
                    f"{drone.last_edge_point[1]:.4f})"
                )

            else:

                print(f"    {drone.drone_id}: " f"no edge detected")

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

        local_states = [getattr(drone, "control_state", "mapping") for drone in self.drones]
        mapping_phase = all(state == "mapping" for state in local_states)
        active_mapping_phase = bool(local_states)
        measurement_frame = (
            ((self.frame - 1) % self.measure_every == 0)
            if mapping_phase
            else active_mapping_phase
        )

        if self.verbose:
            if self.control_state == "mapping":
                frame_type = "measurement" if measurement_frame else "consensus"
            else:
                frame_type = self.control_state

            print(f"\nFrame {self.frame} " f"[{frame_type}]")
        # Environment

        self._update_environment()
        self._refresh_position_estimates()
        self._apply_forgetting_if_needed()
        # Measurement

        if measurement_frame:

            self._start_new_measurement_trace()

            self._perform_measurement()

            self._record_measurement_trace()

            self._print_error_snapshot("  After sensing")

            if self.verbose:
                self._print_sensor_status()
        # Consensus

        if active_mapping_phase:
            for round_idx in range(self.consensus_rounds):

                self._exchange_consensus_messages()

                self._record_measurement_trace()

                self._print_error_snapshot(
                    f"  Consensus iteration " f"{round_idx + 1}/" f"{self.consensus_rounds}"
                )
        # Diagnostics

        error, mean_grid = self.compute_disagreement_error()

        self.error_history.append(error)

        self.mean_grid_history.append(mean_grid.copy())

        self.latest_mean_grid = mean_grid
        self._observe_local_states()
        # Control

        self._apply_actions()
        self._observe_local_states()

        if self.verbose:

            mode_summary = ", ".join(
                f"{drone.drone_id}:" f"{getattr(drone, 'last_control_mode', 'unknown')}"
                for drone in self.drones
            )

            print(
                f"  Frame summary: "
                f"global_disagreement="
                f"{error:.6f} | "
                f"modes: {mode_summary}"
            )

        return error

    def run(
        self,
        iterations,
        render_callback=None,
    ):
        """Run the simulation."""

        for _ in range(int(iterations)):

            self.step()

            if render_callback is not None:
                render_callback(self.get_visualization_data())

        self.finalize_histories()

    def finalize_histories(self):

        if self._current_measurement_trace is not None and any(
            len(values) > 0 for values in self._current_measurement_trace.values()
        ):

            self.measurement_consensus_history.append(
                {
                    drone_id: list(values)
                    for drone_id, values in self._current_measurement_trace.items()
                }
            )

        self._current_measurement_trace = None
