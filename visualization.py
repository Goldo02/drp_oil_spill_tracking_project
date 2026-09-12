import os

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle, RegularPolygon, Rectangle


class Visualizer:
    """
    Handles all matplotlib visualization of the simulation.

    The visualizer is deliberately separated from the simulation logic.
    It receives simulation state and only takes care of displaying it.
    """

    def __init__(
        self,
        sim_map,
        oil_spill=None,
        communication_radius=None,
        show_communication_radius=False,
        show_nls_points=False,
    ):
        plt.ion()

        self.sim_map = sim_map
        self.oil_spill = oil_spill

        self.communication_radius = communication_radius
        self.show_communication_radius = (
            show_communication_radius
            and communication_radius is not None
        )

        self.show_nls_points = show_nls_points

        # ------------------------------------------------------------------
        # Figure
        # ------------------------------------------------------------------

        self.fig, self.ax = plt.subplots(
            figsize=(10, 8)
        )

        self.ax.set_xlim(sim_map.xlim)
        self.ax.set_ylim(sim_map.ylim)
        self.ax.set_aspect("equal")

        if self.show_communication_radius:
            self.ax.set_title(
                "Distributed Occupancy Grid Mapping - "
                f"Communication radius Rc={communication_radius:.2f}"
            )
        else:
            self.ax.set_title(
                "Distributed Occupancy Grid Mapping - "
                "Fully Connected"
            )

        # ------------------------------------------------------------------
        # Initial environment
        # ------------------------------------------------------------------

        self.img = None
        self.contour = None

        if self.oil_spill is not None:
            self._draw_environment()

        # ------------------------------------------------------------------
        # Dynamic artists
        # ------------------------------------------------------------------

        self.drone_patches = {}
        self.texts = {}
        self.edge_markers = {}
        self.nls_markers = {}
        self.control_arrows = {}

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # ======================================================================
    # ENVIRONMENT
    # ======================================================================

    def _draw_environment(self):
        """
        Draw the current oil-spill field.
        """
        field_data = self.oil_spill.field(
            self.sim_map.X,
            self.sim_map.Y,
        )

        self.img = self.ax.imshow(
            field_data.T,
            extent=[
                *self.sim_map.xlim,
                *self.sim_map.ylim,
            ],
            origin="lower",
            cmap="Greys",
            alpha=0.8,
            vmin=0.0,
            vmax=1.0,
        )

        self.contour = self.ax.contour(
            self.sim_map.X,
            self.sim_map.Y,
            field_data,
            levels=[0.1, 0.5, 0.9],
            colors="black",
            alpha=0.5,
            linewidths=0.5,
        )

    def update_environment(self, world_field):
        """
        Update the displayed oil-spill field.

        Parameters
        ----------
        world_field : np.ndarray
            Current environment field.
        """
        if world_field is None:
            return

        world_field = np.asarray(
            world_field,
            dtype=float,
        )

        if self.img is None:
            self.img = self.ax.imshow(
                world_field.T,
                extent=[
                    *self.sim_map.xlim,
                    *self.sim_map.ylim,
                ],
                origin="lower",
                cmap="Greys",
                alpha=0.8,
                vmin=0.0,
                vmax=1.0,
            )
            return

        self.img.set_data(world_field.T)

    # ======================================================================
    # DRONES
    # ======================================================================

    def _remove_drone_artists(self, drone_id):
        """
        Remove all previously drawn artists belonging to one drone.
        """

        if drone_id in self.drone_patches:
            for patch in self.drone_patches[drone_id]:
                if patch is not None:
                    patch.remove()

            del self.drone_patches[drone_id]

        if drone_id in self.texts:
            if self.texts[drone_id] is not None:
                self.texts[drone_id].remove()

            del self.texts[drone_id]

        if drone_id in self.edge_markers:
            for artist in self.edge_markers[drone_id]:
                if artist is not None:
                    artist.remove()

            del self.edge_markers[drone_id]

        if drone_id in self.nls_markers:
            if self.nls_markers[drone_id] is not None:
                self.nls_markers[drone_id].remove()

            del self.nls_markers[drone_id]

        if drone_id in self.control_arrows:
            if self.control_arrows[drone_id] is not None:
                self.control_arrows[drone_id].remove()

            del self.control_arrows[drone_id]

    def update_drone(self, drone):
        """
        Update the visualization of a single drone.
        """

        drone_id = drone.drone_id

        self._remove_drone_artists(drone_id)

        patches = []

        # ------------------------------------------------------------------
        # Communication radius
        # ------------------------------------------------------------------

        if self.show_communication_radius:
            comm_circle = Circle(
                (drone.x, drone.y),
                radius=self.communication_radius,
                fill=False,
                edgecolor="darkorange",
                linewidth=1.2,
                linestyle="--",
                alpha=0.35,
                zorder=1,
            )

            self.ax.add_patch(comm_circle)
            patches.append(comm_circle)

        # ------------------------------------------------------------------
        # Drone body
        # ------------------------------------------------------------------

        body = RegularPolygon(
            (drone.x, drone.y),
            numVertices=6,
            radius=0.15,
            color="royalblue",
            zorder=5,
        )

        self.ax.add_patch(body)
        patches.append(body)

        # ------------------------------------------------------------------
        # Camera footprint
        # ------------------------------------------------------------------

        dx = self.sim_map.dx
        dy = self.sim_map.dy

        sensor_size = getattr(
            drone.camera,
            "size",
            1,
        )

        sensor_width = sensor_size * dx
        sensor_height = sensor_size * dy

        sensor_box = Rectangle(
            (
                drone.x - sensor_width / 2.0,
                drone.y - sensor_height / 2.0,
            ),
            sensor_width,
            sensor_height,
            edgecolor="blue",
            facecolor="none",
            alpha=0.3,
            linestyle="--",
            zorder=3,
        )

        self.ax.add_patch(sensor_box)
        patches.append(sensor_box)

        # ------------------------------------------------------------------
        # Label
        # ------------------------------------------------------------------

        label = self.ax.text(
            drone.x + 0.2,
            drone.y + 0.2,
            f"Drone {drone_id}",
            fontsize=8,
            zorder=9,
        )

        self.drone_patches[drone_id] = patches
        self.texts[drone_id] = label

        # ------------------------------------------------------------------
        # Edge detection
        # ------------------------------------------------------------------

        edge_marker = None
        edge_label = None

        if (
            getattr(drone, "edge_detected", False)
            and getattr(drone, "last_edge_point", None) is not None
        ):
            edge_point = np.asarray(
                drone.last_edge_point,
                dtype=float,
            )

            edge_marker = self.ax.scatter(
                [edge_point[0]],
                [edge_point[1]],
                s=70,
                c="limegreen",
                marker="X",
                edgecolors="black",
                linewidths=0.8,
                zorder=6,
            )

            oil_fraction = getattr(
                drone,
                "last_oil_fraction",
                None,
            )

            edge_count = getattr(
                drone,
                "last_edge_count",
                0,
            )

            if oil_fraction is None:
                annotation = f"n={edge_count}"
            else:
                annotation = (
                    f"n={edge_count}\n"
                    f"{100.0 * oil_fraction:.1f}% oil"
                )

            edge_label = self.ax.text(
                edge_point[0] + 0.12,
                edge_point[1] + 0.12,
                annotation,
                fontsize=7,
                color="limegreen",
                zorder=7,
            )

        self.edge_markers[drone_id] = [
            edge_marker,
            edge_label,
        ]

        # ------------------------------------------------------------------
        # NLS points
        # ------------------------------------------------------------------

        nls_marker = None

        if self.show_nls_points:
            points = getattr(
                drone,
                "last_nls_points",
                None,
            )

            if points is not None:
                points = np.asarray(
                    points,
                    dtype=float,
                )

                if points.ndim == 2 and points.shape[0] > 0:
                    nls_marker = self.ax.scatter(
                        points[:, 0],
                        points[:, 1],
                        s=2,
                        c="red",
                        alpha=0.4,
                        zorder=4,
                    )

        self.nls_markers[drone_id] = nls_marker

        # ------------------------------------------------------------------
        # Control vector
        # ------------------------------------------------------------------

        control_vec = np.asarray(
            getattr(
                drone,
                "last_control_vector",
                np.zeros(2, dtype=float),
            ),
            dtype=float,
        )

        if control_vec.shape != (2,):
            control_vec = np.zeros(
                2,
                dtype=float,
            )

        control_norm = float(
            np.linalg.norm(control_vec)
        )

        control_arrow = None

        if control_norm > 1e-12:

            display_vec = control_vec.copy()

            if control_norm > 0.12:
                display_vec *= (
                    0.12 / control_norm
                )

            control_arrow = self.ax.quiver(
                drone.x,
                drone.y,
                display_vec[0],
                display_vec[1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="crimson",
                width=0.0045,
                alpha=0.9,
                zorder=8,
            )

        self.control_arrows[drone_id] = (
            control_arrow
        )

    def _ensure_output_dir(self, directory="./tmp_output"):
        """Create a target directory for PNG exports and return its absolute path."""
        output_dir = os.path.abspath(directory)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _output_path(self, filename, directory="./tmp_output"):
        """Build an output path under the requested directory."""
        return os.path.join(self._ensure_output_dir(directory), filename)

    def save_final_state(self, filename="final_simulation_state.png", directory="./tmp_output"):
        """Salva lo stato visivo finale della simulazione."""
        output_path = self._output_path(filename, directory)
        self.fig.savefig(output_path, bbox_inches="tight")
        print(f"Final simulation state saved to {output_path}.")

    def save_final_occupancy_grid_per_robot(
        self,
        drones,
        directory="./tmp_output",
        filename_prefix="final_occupancy_grid_robot",
        alpha=1.0,
    ):
        """Save the final local occupancy grid of every drone to PNG files."""
        if drones is None:
            return

        output_dir = self._ensure_output_dir(directory)

        for drone in drones:
            grid = np.asarray(getattr(drone, "grid", np.zeros((1, 1))), dtype=float)
            if grid.size == 0:
                continue

            binary_grid = (grid >= 0.5).astype(float)
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(
                binary_grid.T,
                origin="lower",
                cmap="Greys",
                alpha=float(alpha),
                vmin=0.0,
                vmax=1.0,
            )

            ax.set_title(f"Final Occupancy Grid - Drone {drone.drone_id}")
            ax.set_xlabel("Grid X")
            ax.set_ylabel("Grid Y")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()

            output_path = os.path.join(
                output_dir,
                f"{filename_prefix}_{drone.drone_id}.png",
            )
            fig.savefig(
                output_path,
                bbox_inches="tight",
                facecolor="white",
                transparent=False,
            )
            plt.close(fig)

        print(f"Per-drone final occupancy grids saved to {output_dir}.")

    def plot_consensus_convergence(self, engine, filename="consensus_convergence.png", directory="./tmp_output"):
        """Genera e salva il grafico della convergenza del consenso."""
        error_history = np.asarray(engine.error_history, dtype=float)
        measurement_history = engine.measurement_consensus_history

        fig, ax = plt.subplots(figsize=(12, 5))

        if measurement_history:
            color_cycle = plt.cm.tab10(
                np.linspace(0, 1, max(1, len(engine.drones)))
            )

            for measure_idx, cycle_trace in enumerate(measurement_history, start=1):
                cycle_length = len(next(iter(cycle_trace.values())))
                x_values = np.linspace(measure_idx - 1.0, measure_idx, cycle_length)

                for drone_idx, drone in enumerate(engine.drones):
                    drone_id = drone.drone_id
                    y_values = np.asarray(cycle_trace[drone_id], dtype=float)

                    ax.plot(
                        x_values,
                        y_values,
                        color=color_cycle[drone_idx % len(color_cycle)],
                        linewidth=1.8,
                        marker="o",
                        markersize=3,
                        alpha=0.9,
                        label=drone_id if measure_idx == 1 else None,
                    )

            measurement_count = len(measurement_history)
            ax.set_xlim(0, measurement_count)
            ax.set_xticks(np.arange(0, measurement_count + 1, 1))
            ax.set_title("Consensus Convergence Between Measurements")
            ax.set_xlabel("Number of measurements + 1")
            ax.set_ylabel("Grid disagreement error")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        else:
            ax.plot(
                np.arange(1, len(error_history) + 1),
                error_history,
                linewidth=2.0,
            )
            ax.set_title("Consensus Disagreement Error")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Error")
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        output_path = self._output_path(filename, directory)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Consensus convergence plot saved to {output_path}.")

    def plot_final_occupancy_grid(
        self,
        final_grid,
        filename="final_occupancy_grid.png",
        directory="./tmp_output",
        alpha=1.0,
    ):
        """Genera e salva la griglia di occupazione finale unificata."""
        binary_grid = (np.asarray(final_grid, dtype=float) >= 0.5).astype(float)

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(
            binary_grid.T,
            origin="lower",
            cmap="Greys",
            alpha=float(alpha),
            vmin=0.0,
            vmax=1.0,
        )

        ax.set_title("Final Occupancy Grid")
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
        fig.colorbar(im, ax=ax)

        fig.tight_layout()
        output_path = self._output_path(filename, directory)
        fig.savefig(output_path, bbox_inches="tight", facecolor="white", transparent=False)
        plt.close(fig)
        print(f"Final occupancy grid saved to {output_path}.")

    # ======================================================================
    # RENDER
    # ======================================================================

    def render(self, simulation_data, pause=None):
        """
        Render a complete simulation state.

        Parameters
        ----------
        simulation_data : dict
            Data returned by SimulationEngine.get_visualization_data().
        pause : float or bool, optional
            Ignored or used for compatibility.
        """

        if simulation_data is None:
            return

        world_field = simulation_data.get(
            "world_field"
        )

        drones = simulation_data.get(
            "drones",
            [],
        )

        # Update environment first.
        if world_field is not None:
            self.update_environment(
                world_field
            )

        # Update drones.
        for drone in drones:
            self.update_drone(drone)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # Se viene passato un valore numerico per la pausa, usalo, altrimenti usa il default
        pause_time = pause if isinstance(pause, (int, float)) else 0.001
        if pause is not False:
            plt.pause(pause_time)

    def close(self):
        """
        Close the visualization window.
        """
        plt.close(self.fig)