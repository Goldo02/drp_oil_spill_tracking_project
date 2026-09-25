import os

import matplotlib.pyplot as plt
import numpy as np

from controller import Controller
from matplotlib.collections import LineCollection
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
        show_2d_voronoi=False,
    ):
        plt.ion()

        self.sim_map = sim_map
        self.oil_spill = oil_spill

        self.communication_radius = communication_radius
        self.show_communication_radius = (
            show_communication_radius and communication_radius is not None
        )

        self.show_nls_points = show_nls_points
        self.show_2d_voronoi = bool(show_2d_voronoi)

        self.fig, (self.ax, self.side_ax) = plt.subplots(
            1,
            2,
            figsize=(16, 7),
            gridspec_kw={"width_ratios": [1.25, 1.0]},
        )
        self.ax.set_xlim(sim_map.xlim)
        self.ax.set_ylim(sim_map.ylim)
        self.ax.set_aspect("equal")
        self.side_mode = None

        if self.show_communication_radius:
            self.ax.set_title(
                "Distributed Occupancy Grid Mapping - "
                f"Communication radius Rc={communication_radius:.2f}"
            )
        else:
            self.ax.set_title("Distributed Occupancy Grid Mapping - " "Fully Connected")
        # Initial environment

        self.img = None
        self.contour = None

        if self.oil_spill is not None:
            self._draw_environment()

        self.drone_patches = {}
        self.texts = {}
        self.edge_markers = {}
        self.nls_markers = {}
        self.control_arrows = {}
        self.centroid_markers = {}
        self.voronoi_map_artists = []
        self.occupancy_boundary_artists = []
        self.ring_color_map = {}
        self.animation_frames = []
        self.animation_recording_enabled = False
        self.animation_frame_stride = 1
        self.animation_render_count = 0

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

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

        world_field = np.asarray(world_field, dtype=float)

        if self.img is None:
            self.img = self.ax.imshow(
                world_field.T,
                extent=[*self.sim_map.xlim, *self.sim_map.ylim],
                origin="lower",
                cmap="Greys",
                alpha=0.8,
                vmin=0.0,
                vmax=1.0,
            )
            return

        self.img.set_data(world_field.T)

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

        if drone_id in self.centroid_markers:
            for artist in self.centroid_markers[drone_id]:
                if artist is not None:
                    artist.remove()

            del self.centroid_markers[drone_id]

    def _drone_color(self, drone_id):
        if drone_id not in self.ring_color_map:
            palette = plt.get_cmap("tab10")
            self.ring_color_map[drone_id] = palette(len(self.ring_color_map) % 10)
        return self.ring_color_map[drone_id]

    def _clear_voronoi_map_artists(self):
        """Remove the 2D Voronoi boundary overlay from the main map."""
        for artist in self.voronoi_map_artists:
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass
        self.voronoi_map_artists = []

    def _clear_occupancy_boundary_artists(self):
        """Remove the probabilistic occupancy-boundary overlay."""
        for artist in self.occupancy_boundary_artists:
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass
        self.occupancy_boundary_artists = []

    @staticmethod
    def _boundary_points_from_grid(boundary_grid, grid_bounds, grid_resolution):
        boundary = np.asarray(boundary_grid, dtype=float) >= 0.5
        cells = np.argwhere(boundary)
        if cells.size == 0:
            return np.empty((0, 2), dtype=float)

        x_min, _, y_min, _ = grid_bounds
        resolution = float(grid_resolution)
        return np.column_stack(
            (
                float(x_min) + (cells[:, 0] + 0.5) * resolution,
                float(y_min) + (cells[:, 1] + 0.5) * resolution,
            )
        ).astype(float)

    def update_occupancy_boundary_overlay(self, simulation_data):
        """Draw the boundary extracted from the average-consensus occupancy map."""
        self._clear_occupancy_boundary_artists()

        boundary_grid = simulation_data.get("mean_boundary_grid")
        grid_bounds = simulation_data.get("grid_bounds")
        grid_resolution = simulation_data.get("grid_resolution")
        if boundary_grid is None or grid_bounds is None or grid_resolution is None:
            return

        points = self._boundary_points_from_grid(
            boundary_grid,
            grid_bounds,
            grid_resolution,
        )
        if points.size == 0:
            return

        scatter = self.ax.scatter(
            points[:, 0],
            points[:, 1],
            s=12,
            c="#18a8ff",
            marker="s",
            edgecolors="none",
            alpha=0.9,
            zorder=4.5,
        )
        self.occupancy_boundary_artists.append(scatter)

    @staticmethod
    def _ring_data_from_drones(drones):
        for drone in drones:
            ring_data = getattr(drone, "last_ring_info", None)
            if isinstance(ring_data, dict) and "occupied_points" in ring_data:
                occupied = np.asarray(ring_data["occupied_points"], dtype=float)
                if occupied.ndim == 2 and occupied.shape[0] > 0:
                    return ring_data
        return None

    def update_voronoi_map_overlay(self, drones):
        """Color each 2D boundary segment by its Voronoi owner drone."""
        self._clear_voronoi_map_artists()

        if not self.show_2d_voronoi:
            return

        ring_data = self._ring_data_from_drones(drones)
        if ring_data is None:
            return

        points = np.asarray(ring_data.get("occupied_points", []), dtype=float)
        assignment = np.asarray(
            ring_data.get(
                "assigned_drone_indices",
                np.empty(points.shape[0], dtype=object),
            ),
            dtype=object,
        )
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or points.shape[0] < 2
            or assignment.shape[0] != points.shape[0]
        ):
            return

        is_closed = bool(ring_data.get("is_closed", True))
        segment_count = points.shape[0] if is_closed else points.shape[0] - 1
        segments_by_owner = {}

        for idx in range(segment_count):
            next_idx = (idx + 1) % points.shape[0]
            owner_left = assignment[idx]
            owner_right = assignment[next_idx]
            p_left = points[idx]
            p_right = points[next_idx]

            if owner_left is None and owner_right is None:
                continue

            if owner_left == owner_right:
                if owner_left is not None:
                    segments_by_owner.setdefault(owner_left, []).append(
                        np.vstack((p_left, p_right))
                    )
                continue

            midpoint = 0.5 * (p_left + p_right)
            if owner_left is not None:
                segments_by_owner.setdefault(owner_left, []).append(
                    np.vstack((p_left, midpoint))
                )
            if owner_right is not None:
                segments_by_owner.setdefault(owner_right, []).append(
                    np.vstack((midpoint, p_right))
                )

        for owner, segments in segments_by_owner.items():
            if not segments:
                continue
            collection = LineCollection(
                segments,
                colors=[self._drone_color(owner)],
                linewidths=4.0,
                alpha=0.95,
                zorder=4,
                capstyle="round",
                joinstyle="round",
            )
            self.ax.add_collection(collection)
            self.voronoi_map_artists.append(collection)

        assigned_mask = np.array([owner is not None for owner in assignment], dtype=bool)
        if np.any(assigned_mask):
            scatter = self.ax.scatter(
                points[assigned_mask, 0],
                points[assigned_mask, 1],
                s=9,
                c=[self._drone_color(owner) for owner in assignment[assigned_mask]],
                edgecolors="none",
                alpha=0.9,
                zorder=4.2,
            )
            self.voronoi_map_artists.append(scatter)

    def update_drone(self, drone):
        """
        Update the visualization of a single drone.
        """

        drone_id = drone.drone_id

        self._remove_drone_artists(drone_id)

        patches = []
        drone_color = self._drone_color(drone_id)

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
        body = RegularPolygon(
            (drone.x, drone.y),
            numVertices=6,
            radius=0.15,
            color=drone_color,
            zorder=5,
        )

        self.ax.add_patch(body)
        patches.append(body)

        if getattr(drone, "control_state", "mapping") == "mapping" and hasattr(drone, "camera"):
            dx = self.sim_map.dx
            dy = self.sim_map.dy
            sensor_size = getattr(drone.camera, "size", 1)
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
        label = self.ax.text(
            drone.x + 0.2,
            drone.y + 0.2,
            f"Drone {drone_id}",
            fontsize=8,
            zorder=9,
        )

        self.drone_patches[drone_id] = patches
        self.texts[drone_id] = label

        edge_marker = None
        edge_label = None

        if (
            getattr(drone, "control_state", "mapping") == "mapping"
            and self.show_nls_points
            and
            getattr(drone, "edge_detected", False)
            and getattr(drone, "last_edge_point", None) is not None
        ):
            edge_point = np.asarray(drone.last_edge_point, dtype=float)

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

            oil_fraction = getattr(drone, "last_oil_fraction", None)
            edge_count = getattr(drone, "last_edge_count", 0)

            if oil_fraction is None:
                annotation = f"n={edge_count}"
            else:
                annotation = f"n={edge_count}\n{100.0 * oil_fraction:.1f}% oil"

            edge_label = self.ax.text(
                edge_point[0] + 0.12,
                edge_point[1] + 0.12,
                annotation,
                fontsize=7,
                color="limegreen",
                zorder=7,
            )

        self.edge_markers[drone_id] = [edge_marker, edge_label]

        nls_marker = None

        if self.show_nls_points and getattr(drone, "control_state", "mapping") == "mapping":
            points = getattr(drone, "last_nls_points", None)

            if points is not None:
                points = np.asarray(points, dtype=float)

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

        control_vec = np.asarray(
            getattr(
                drone,
                "last_control_vector",
                np.zeros(2, dtype=float),
            ),
            dtype=float,
        )

        if control_vec.shape != (2,):
            control_vec = np.zeros(2, dtype=float)

        control_norm = float(np.linalg.norm(control_vec))

        control_arrow = None

        if control_norm > 1e-12:

            display_vec = control_vec.copy()

            if control_norm > 0.12:
                display_vec *= 0.12 / control_norm

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

        self.control_arrows[drone_id] = control_arrow

        centroid_artists = []
        target_centroid = getattr(drone, "target_centroid", None)
        if getattr(drone, "control_state", "mapping") == "lloyd" and target_centroid is not None:
            tc = np.asarray(target_centroid, dtype=float)
            if tc.shape == (2,) and np.all(np.isfinite(tc)):
                marker = self.ax.scatter(
                    [tc[0]],
                    [tc[1]],
                    s=90,
                    c=[drone_color],
                    marker="*",
                    edgecolors="black",
                    linewidths=0.9,
                    zorder=7,
                )
                label = self.ax.text(
                    tc[0] + 0.12,
                    tc[1] + 0.12,
                    f"C_{drone_id}",
                    fontsize=7,
                    color=drone_color,
                    fontweight="bold",
                    zorder=8,
                )
                centroid_artists.extend([marker, label])
        self.centroid_markers[drone_id] = centroid_artists

    def _prepare_side_axis(self, mode):
        if self.side_mode != mode:
            self.side_ax.clear()
            self.side_mode = mode
        else:
            self.side_ax.clear()

    def _render_mean_occupancy_panel(self, simulation_data):
        self._prepare_side_axis("mapping")
        mean_grid = simulation_data.get("mean_grid")
        if mean_grid is None:
            mean_grid = np.zeros((1, 1), dtype=float)

        mean_grid = np.asarray(mean_grid, dtype=float)
        self.side_ax.imshow(
            mean_grid.T,
            origin="lower",
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="auto",
        )
        boundary_grid = simulation_data.get("mean_boundary_grid")
        if boundary_grid is not None:
            cells = np.argwhere(np.asarray(boundary_grid, dtype=float) >= 0.5)
            if cells.size > 0:
                self.side_ax.scatter(
                    cells[:, 0],
                    cells[:, 1],
                    s=5,
                    c="#18a8ff",
                    marker="s",
                    edgecolors="none",
                    alpha=0.95,
                )
        self.side_ax.set_title("Mean Occupancy Grid")
        self.side_ax.set_xlabel("Grid X")
        self.side_ax.set_ylabel("Grid Y")

    def update_ring_partition(self, drones):
        self._prepare_side_axis("lloyd")
        self.side_ax.set_title("1D Voronoi Boundary Partition")
        self.side_ax.set_xlabel("Boundary arc length [m]")
        self.side_ax.set_ylabel("Cell")
        self.side_ax.set_yticks([])
        self.side_ax.grid(True, axis="x", alpha=0.25)

        if not drones:
            return

        ring_data = self._ring_data_from_drones(drones)
        if ring_data is None:
            boundary = None
            for drone in drones:
                points = getattr(drone, "known_boundary_points", None)
                if points is not None and len(points) > 0:
                    boundary = np.asarray(points, dtype=float)
                    break
            if boundary is None:
                return
            arc_lengths, total_length = Controller._boundary_arc_lengths(boundary, is_closed=True)
            seeds = []
            for drone in drones:
                seed_s, seed_point, seed_idx = Controller._arc_length_at_position(
                    boundary,
                    arc_lengths,
                    np.array([drone.x, drone.y], dtype=float),
                    total_length,
                    True,
                )
                seeds.append(
                    {
                        "robot_id": drone.drone_id,
                        "index": seed_idx,
                        "arc_length": seed_s,
                        "position_on_boundary": seed_point,
                    }
                )
            targets = Controller._lloyd_targets_from_seed_arcs(seeds, total_length, True)
            ring_entries = []
            for seed in seeds:
                target = targets.get(seed["robot_id"], {})
                ring_entries.append(
                    {
                        "drone_id": seed["robot_id"],
                        "seed_arc_length": seed["arc_length"],
                        "target_arc_length": target.get("target_arc_length", seed["arc_length"]),
                        "cell_start_arc_length": target.get("cell_start_arc_length", seed["arc_length"]),
                        "cell_end_arc_length": target.get("cell_end_arc_length", seed["arc_length"]),
                        "cell_arc_length": target.get("cell_arc_length", 0.0),
                    }
                )
        else:
            total_length = float(ring_data.get("total_boundary_length", 1.0))
            ring_entries = ring_data.get("ring", [])

        if not ring_entries or total_length <= 1e-12:
            return

        margin = max(0.25, 0.03 * total_length)
        self.side_ax.set_xlim(-margin, total_length + margin)
        self.side_ax.set_ylim(-0.55, 0.55)
        self.side_ax.hlines(0.0, 0.0, total_length, colors="gray", linewidths=1.0, alpha=0.35)

        for entry in ring_entries:
            drone_id = entry.get("drone_id")
            if drone_id is None:
                continue
            color = self._drone_color(drone_id)
            start = float(entry.get("cell_start_arc_length", 0.0)) % total_length
            end = float(entry.get("cell_end_arc_length", start)) % total_length
            cell_len = float(entry.get("cell_arc_length", 0.0))
            if cell_len >= total_length - 1e-9:
                segments = [(0.0, total_length)]
            elif start <= end:
                segments = [(start, end)]
            else:
                segments = [(start, total_length), (0.0, end)]

            for left, right in segments:
                if right > left:
                    self.side_ax.hlines(
                        0.0,
                        left,
                        right,
                        colors=[color],
                        linewidths=8.0,
                        alpha=0.95,
                        zorder=2,
                    )

            seed_arc = float(entry.get("seed_arc_length", start)) % total_length
            target_arc = float(entry.get("target_arc_length", seed_arc)) % total_length
            self.side_ax.scatter(
                [seed_arc],
                [0.22],
                s=115,
                c=[color],
                marker="o",
                edgecolors="black",
                linewidths=0.8,
                zorder=4,
            )
            self.side_ax.scatter(
                [target_arc],
                [0.0],
                s=180,
                c=[color],
                marker="*",
                edgecolors="black",
                linewidths=1.1,
                zorder=5,
            )
            self.side_ax.text(
                seed_arc,
                0.33,
                str(drone_id),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                zorder=6,
            )

    def _ensure_output_dir(self, directory="./tmp_output"):
        """Create a target directory for PNG exports and return its absolute path."""
        output_dir = os.path.abspath(directory)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _output_path(self, filename, directory="./tmp_output"):
        """Build an output path under the requested directory."""
        return os.path.join(self._ensure_output_dir(directory), filename)

    def start_animation_recording(self, frame_stride=1):
        """
        Enable frame capture for a later GIF export.

        Parameters
        ----------
        frame_stride : int, optional
            Capture one rendered frame every ``frame_stride`` calls to
            :meth:`render`. A value of 1 records every rendered state.
        """
        self.animation_frames = []
        self.animation_recording_enabled = True
        self.animation_frame_stride = max(1, int(frame_stride))
        self.animation_render_count = 0

    def stop_animation_recording(self):
        """Disable frame capture without discarding already captured frames."""
        self.animation_recording_enabled = False

    def clear_animation_frames(self):
        """Discard all frames currently buffered for GIF export."""
        self.animation_frames = []
        self.animation_render_count = 0

    def _capture_animation_frame(self):
        """Store the current Matplotlib canvas as an RGB frame."""
        if not self.animation_recording_enabled:
            return

        if self.animation_render_count % self.animation_frame_stride != 0:
            self.animation_render_count += 1
            return

        self.fig.canvas.draw()
        rgba = np.asarray(self.fig.canvas.buffer_rgba())
        rgb = np.asarray(rgba[:, :, :3], dtype=np.uint8).copy()
        self.animation_frames.append(rgb)
        self.animation_render_count += 1

    def save_animation_as_gif(
        self,
        filename="simulation_animation.gif",
        directory="./tmp_output",
        fps=10,
        loop=0,
    ):
        """
        Save the rendered simulation history as a GIF.

        Frames are captured from calls to :meth:`render` after
        :meth:`start_animation_recording` has been enabled. This keeps the
        animation identical to the normal Matplotlib visualization, including
        the map, drones, communication radii, occupancy side panel, and Voronoi
        overlays.

        Parameters
        ----------
        filename : str, optional
            GIF file name.
        directory : str, optional
            Output directory.
        fps : int or float, optional
            Playback frame rate.
        loop : int, optional
            GIF loop count. ``0`` means loop forever.

        Returns
        -------
        str
            Absolute output path of the saved GIF.
        """
        if not self.animation_frames:
            raise RuntimeError(
                "No animation frames were recorded. Call "
                "start_animation_recording() before rendering simulation frames."
            )

        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Saving GIF animations requires Pillow. Install it with "
                "`python3 -m pip install pillow`."
            ) from exc

        output_path = self._output_path(filename, directory)
        duration_ms = int(round(1000.0 / max(float(fps), 1e-9)))
        images = [Image.fromarray(frame) for frame in self.animation_frames]
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=int(loop),
            optimize=False,
        )
        print(
            f"Animation GIF saved to {output_path} "
            f"({len(images)} frames, {float(fps):.2f} fps)."
        )
        return output_path

    def _save_grid_plot(self, grid, title, output_path, alpha=1.0, threshold=0.6):
        binary_grid = (np.asarray(grid, dtype=float) >= float(threshold)).astype(float)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(
            binary_grid.T,
            origin="lower",
            cmap="Greys",
            alpha=float(alpha),
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(title)
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight", facecolor="white", transparent=False)
        plt.close(fig)

    def save_final_state(
        self, filename="final_simulation_state.png", directory="./tmp_output"
    ):
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
        threshold=0.6,
    ):
        """Save the final local occupancy grid of every drone to PNG files."""
        if drones is None:
            return

        output_dir = self._ensure_output_dir(directory)

        for drone in drones:
            grid = np.asarray(getattr(drone, "grid", np.zeros((1, 1))), dtype=float)
            if grid.size == 0:
                continue

            output_path = os.path.join(
                output_dir,
                f"{filename_prefix}_{drone.drone_id}.png",
            )
            self._save_grid_plot(
                grid,
                f"Final Occupancy Grid - Drone {drone.drone_id}",
                output_path,
                alpha=alpha,
                threshold=threshold,
            )

        print(f"Per-drone final occupancy grids saved to {output_dir}.")

    def plot_consensus_convergence(
        self, engine, filename="consensus_convergence.png", directory="./tmp_output"
    ):
        """Genera e salva il grafico della convergenza del consenso."""
        error_history = np.asarray(engine.error_history, dtype=float)
        measurement_history = engine.measurement_consensus_history

        fig, ax = plt.subplots(figsize=(12, 5))

        if measurement_history:
            color_cycle = plt.cm.tab10(np.linspace(0, 1, max(1, len(engine.drones))))

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
        threshold=0.6,
    ):
        """Genera e salva la griglia di occupazione finale unificata."""
        output_path = self._output_path(filename, directory)
        self._save_grid_plot(
            final_grid,
            "Final Occupancy Grid",
            output_path,
            alpha=alpha,
            threshold=threshold,
        )
        print(f"Final occupancy grid saved to {output_path}.")

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

        world_field = simulation_data.get("world_field")

        drones = simulation_data.get("drones", [])

        if world_field is not None:
            self.update_environment(world_field)

        self.update_occupancy_boundary_overlay(simulation_data)
        self.update_voronoi_map_overlay(drones)

        for drone in drones:
            self.update_drone(drone)

        control_state = simulation_data.get("control_state", "mapping")
        if control_state == "lloyd":
            self.update_ring_partition(drones)
        else:
            self._render_mean_occupancy_panel(simulation_data)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self._capture_animation_frame()

        pause_time = pause if isinstance(pause, (int, float)) else 0.001
        if pause is not False:
            plt.pause(pause_time)

    def close(self):
        """
        Close the visualization window.
        """
        plt.close(self.fig)
