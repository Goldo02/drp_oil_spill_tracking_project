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

        self.fig, (self.ax, self.ring_ax) = plt.subplots(
            1,
            2,
            figsize=(16, 7),
        )

        self.ax.set_xlim(sim_map.xlim)
        self.ax.set_ylim(sim_map.ylim)
        self.ax.set_aspect("equal")

        self.ring_ax.set_aspect("equal")
        self.ring_ax.set_xlabel("Boundary x [m]")
        self.ring_ax.set_ylabel("Boundary y [m]")
        self.ring_ax.set_title("1D Voronoi boundary partition")

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
        self.centroid_markers = {}
        self.voronoi_ring_artists = []
        self.ring_color_map = {}

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

        if drone_id in self.centroid_markers:
            for artist in self.centroid_markers[drone_id]:
                if artist is not None:
                    artist.remove()

            del self.centroid_markers[drone_id]

    def _drone_color(self, drone_id):
        """Return a deterministic color for a drone ID."""
        if drone_id not in self.ring_color_map:
            palette = plt.get_cmap("tab10")
            ids = sorted(self.ring_color_map.keys())
            idx = len(ids) % 10
            self.ring_color_map[drone_id] = palette(idx)
        return self.ring_color_map[drone_id]

    def update_ring_partition(self, drones):
        """Draw the 1D Voronoi partition as a line with one colored cell per drone."""
        for artist in self.voronoi_ring_artists:
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass
        self.voronoi_ring_artists = []

        self.ring_ax.clear()
        self.ring_ax.set_xlim(-0.5, 1.0)
        self.ring_ax.set_ylim(-0.8, 0.8)
        self.ring_ax.set_aspect("auto")
        self.ring_ax.set_xlabel("Boundary arc index")
        self.ring_ax.set_ylabel("Cell")
        self.ring_ax.set_title("1D Voronoi boundary partition")
        self.ring_ax.set_yticks([])

        if drones is None:
            return

        canonical_assignment = None
        canonical_points = None
        canonical_ring = None

        for drone in drones:
            ring_data = getattr(drone, "last_ring_info", None)
            if ring_data is None or "occupied_points" not in ring_data:
                continue

            occupied = np.asarray(ring_data["occupied_points"], dtype=float)
            if occupied.size == 0:
                continue

            assigned = np.asarray(
                ring_data.get("assigned_drone_indices", np.zeros(len(occupied), dtype=object)),
                dtype=object,
            )
            if assigned.size == 0:
                continue

            if canonical_assignment is None:
                canonical_assignment = assigned
                canonical_points = occupied
                canonical_ring = ring_data.get("ring", [])
                break

        if canonical_assignment is None:
            # Fallback: try to build a canonical ring assignment from any
            # drone's `known_boundary_points` if available. This allows the
            # 1D partition plot to show something even when `last_ring_info`
            # is not provided by the simulation.
            boundary_pts = None
            for drone in drones:
                kb = getattr(drone, "known_boundary_points", None)
                if kb is not None and len(kb) > 0:
                    boundary_pts = np.asarray(kb, dtype=float)
                    break

            if boundary_pts is None or boundary_pts.size == 0:
                return

            n_points = len(boundary_pts)
            # compute each drone's nearest boundary index
            drone_indices = []
            drone_ids = []
            for drone in drones:
                pos = np.array([drone.x, drone.y], dtype=float)
                dists = np.linalg.norm(boundary_pts - pos, axis=1)
                idx = int(np.argmin(dists))
                drone_indices.append(idx)
                drone_ids.append(drone.drone_id)

            # Build canonical_assignment using circular distance to nearest drone index
            canonical_assignment = np.zeros(n_points, dtype=object)
            for j in range(n_points):
                # circular distances
                diffs = np.abs((np.array(drone_indices, dtype=float) - float(j) + n_points / 2.0) % n_points - n_points / 2.0)
                nearest = int(np.argmin(diffs))
                canonical_assignment[j] = drone_ids[nearest]

            canonical_points = boundary_pts
            canonical_ring = []

        n_points = len(canonical_assignment)
        if n_points == 0:
            return

        x_margin = max(2.0, 0.03 * float(n_points))
        self.ring_ax.set_xlim(-x_margin, n_points - 1 + x_margin)

        cell_colors = []
        cell_indices = []
        # For each drone, prefer per-drone `last_ring_info`; otherwise fall back
        # to the canonical assignment determined above so drones are visible
        # on the 1D ring even when they don't carry individual ring metadata.
        for drone in drones:
            # determine assignment array and occupied points to use
            ring_data = getattr(drone, "last_ring_info", None)
            if ring_data is not None and "occupied_points" in ring_data:
                occupied = np.asarray(ring_data["occupied_points"], dtype=float)
                assigned = np.asarray(
                    ring_data.get("assigned_drone_indices", np.zeros(len(occupied), dtype=object)),
                    dtype=object,
                )
            else:
                # fallback to canonical arrays
                occupied = canonical_points
                assigned = canonical_assignment

            if occupied is None or occupied.size == 0:
                continue

            cell_mask = np.array([a == drone.drone_id for a in assigned], dtype=bool)
            if not np.any(cell_mask):
                continue

            drone_color = self._drone_color(drone.drone_id)
            points_x = np.flatnonzero(cell_mask)
            if points_x.size == 0:
                continue

            contiguous_runs = np.split(
                points_x,
                np.where(np.diff(points_x) > 1)[0] + 1,
            )
            for run in contiguous_runs:
                if run.size == 0:
                    continue
                segment = self.ring_ax.hlines(
                    0.0,
                    float(run[0]) - 0.5,
                    float(run[-1]) + 0.5,
                    colors=[drone_color],
                    linewidths=8.0,
                    alpha=0.95,
                    zorder=2,
                )
                self.voronoi_ring_artists.append(segment)
            cell_colors.append(drone_color)
            cell_indices.append(points_x)

            # target centroid: prefer canonical_ring entry then per-drone target_centroid
            target_idx = None
            seed_idx = None
            if canonical_ring:
                for entry in canonical_ring:
                    if entry.get("drone_id") == drone.drone_id:
                        seed_idx = entry.get("seed_index")
                        # some ring generators expose a target_chain_index
                        target_idx = entry.get("target_chain_index")
                        if target_idx is None and "target_centroid" in entry and canonical_points is not None:
                            # find nearest canonical point to the target centroid
                            tc = np.asarray(entry["target_centroid"], dtype=float)
                            dists = np.linalg.norm(canonical_points - tc.reshape(1,2), axis=1)
                            target_idx = float(np.argmin(dists))
                        break

            if target_idx is None:
                # try per-drone attribute
                target_centroid = getattr(drone, "target_centroid", None)
                if target_centroid is not None and canonical_points is not None:
                    tc = np.asarray(target_centroid, dtype=float)
                    dists = np.linalg.norm(canonical_points - tc.reshape(1,2), axis=1)
                    target_idx = float(np.argmin(dists)) if dists.size else float(np.median(points_x))

            if target_idx is None:
                target_idx = float(np.median(points_x))

            # Larger star for the target centroid
            sc_tc = self.ring_ax.scatter(
                [float(target_idx)],
                [0.0],
                s=180,
                c=[drone_color],
                marker="*",
                edgecolors="black",
                linewidths=1.2,
                zorder=6,
            )
            self.voronoi_ring_artists.append(sc_tc)

            if seed_idx is None:
                seed_idx = target_idx

            # Small marker above the axis representing the drone projection on the 1D ring
            drone_marker_y = 0.22
            dm = self.ring_ax.scatter(
                [float(seed_idx)],
                [drone_marker_y],
                s=120,
                c=[drone_color],
                marker="o",
                edgecolors="black",
                linewidths=0.8,
                zorder=7,
            )
            self.voronoi_ring_artists.append(dm)

            # Label with white background for readability
            txt = self.ring_ax.text(
                float(seed_idx),
                drone_marker_y + 0.08,
                f"{drone.drone_id}",
                color="black",
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="bottom",
                zorder=8,
            )
            txt.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.6))
            self.voronoi_ring_artists.append(txt)

        if canonical_assignment.size > 1:
            # canonical_assignment may contain non-numeric IDs (strings). Compute
            # separator positions by scanning for value changes to avoid numpy.diff
            separator_positions = [i + 0.5 for i in range(canonical_assignment.size - 1) if canonical_assignment[i] != canonical_assignment[i + 1]]
            if len(separator_positions):
                self.ring_ax.vlines(
                    separator_positions,
                    -0.55,
                    0.55,
                    colors="black",
                    linewidths=0.9,
                    alpha=0.75,
                    zorder=1,
                )

        self.ring_ax.hlines(0.0, -0.5, max(n_points - 0.5, 0.5), colors="gray", linewidths=0.8, alpha=0.35, zorder=0)

    def update_drone(self, drone):
        """
        Update the visualization of a single drone.
        """

        drone_id = drone.drone_id

        self._remove_drone_artists(drone_id)

        patches = []
        drone_color = self._drone_color(drone_id)

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
            color=drone_color,
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

        # ------------------------------------------------------------------
        # Target centroid marker (1D Boundary Voronoi)
        # ------------------------------------------------------------------
        centroid_artists = []
        target_centroid = getattr(drone, "target_centroid", None)
        if target_centroid is not None:
            tc = np.asarray(target_centroid, dtype=float)
            if tc.shape == (2,) and np.all(np.isfinite(tc)):
                c_marker = self.ax.scatter(
                    [tc[0]],
                    [tc[1]],
                    s=80,
                    c=[drone_color],
                    marker="*",
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=7,
                )
                c_label = self.ax.text(
                    tc[0] + 0.12,
                    tc[1] + 0.12,
                    f"C_{drone_id}",
                    fontsize=7,
                    color=drone_color,
                    fontweight="bold",
                    zorder=8,
                )
                centroid_artists.extend([c_marker, c_label])

        self.centroid_markers[drone_id] = centroid_artists

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
        """Disabled: inter-robot consensus is intentionally not used in this baseline."""
        return None

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

    def get_ring_partition_info(self, drones):
        """
        Extract canonical ring partition information used by the 1D plot.

        Returns a dict with keys:
        - n_points: number of ring points
        - canonical_assignment: array of assigned drone indices per ring point
        - canonical_points: coordinates of ring points
        - canonical_ring: optional ring metadata
        - per_drone: mapping drone_id -> {indices: np.ndarray, target_idx: float|None}
        """
        if drones is None:
            return None

        canonical_assignment = None
        canonical_points = None
        canonical_ring = None

        for drone in drones:
            ring_data = getattr(drone, "last_ring_info", None)
            if ring_data is None or "occupied_points" not in ring_data:
                continue

            occupied = np.asarray(ring_data["occupied_points"], dtype=float)
            if occupied.size == 0:
                continue

            assigned = np.asarray(
                ring_data.get("assigned_drone_indices", np.zeros(len(occupied), dtype=object)),
                dtype=object,
            )
            if assigned.size == 0:
                continue

            canonical_assignment = assigned
            canonical_points = occupied
            canonical_ring = ring_data.get("ring", [])
            break

        if canonical_assignment is None:
            return None

        info = {
            "n_points": int(canonical_assignment.size),
            "canonical_assignment": canonical_assignment,
            "canonical_points": canonical_points,
            "canonical_ring": canonical_ring,
            "per_drone": {},
        }

        for drone in drones:
            ring_data = getattr(drone, "last_ring_info", None)
            if ring_data is None or "occupied_points" not in ring_data:
                continue

            occupied = np.asarray(ring_data["occupied_points"], dtype=float)
            if occupied.size == 0:
                continue

            assigned = np.asarray(
                ring_data.get("assigned_drone_indices", np.zeros(len(occupied), dtype=object)),
                dtype=object,
            )
            if assigned.size == 0:
                continue

            cell_mask = np.array([owner == drone.drone_id for owner in assigned], dtype=bool)
            indices = np.flatnonzero(cell_mask)
            if indices.size == 0:
                continue

            # find target index from canonical_ring metadata if available
            target_idx = None
            if canonical_ring:
                for entry in canonical_ring:
                    if entry.get("drone_id") == drone.drone_id:
                        target_idx = entry.get("target_chain_index")
                        break

            if target_idx is None:
                if indices.size:
                    target_idx = float(np.median(indices))
                else:
                    target_idx = None

            info["per_drone"][drone.drone_id] = {
                "indices": indices,
                "target_idx": target_idx,
            }

        return info

    def print_voronoi_targets(self, drones):
        """Stampa su console i target Voronoi calcolati per ogni drone (1D/2D)."""
        if drones is None:
            print("No drones provided.")
            return

        info = self.get_ring_partition_info(drones)

        for drone in drones:
            tid = getattr(drone, "drone_id", None)
            tc = getattr(drone, "target_centroid", None)
            if tc is None:
                tc_str = "None"
            else:
                tc = np.asarray(tc, dtype=float)
                tc_str = np.array2string(tc, precision=3, separator=", ")

            per = None
            if info is not None:
                per = info["per_drone"].get(tid, None)

            idxs = per["indices"] if per is not None else None
            t_idx = per["target_idx"] if per is not None else None

            print(f"Drone {tid}: target_centroid={tc_str}, target_idx={t_idx}, assigned_points_count={None if idxs is None else len(idxs)}")

    def save_ring_partition_image(self, drones, filename="voronoi_ring.png", directory="./tmp_output", dpi=150):
        """Salva il grafico 1D (ring partition) su file PNG."""
        # Update the ring partition plot to ensure axes artists are current
        self.update_ring_partition(drones)

        output_path = self._output_path(filename, directory)
        try:
            # Save the full figure; ring_ax is part of self.fig
            self.fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
            print(f"Ring partition image saved to {output_path}.")
        except Exception as e:
            print(f"Failed to save ring partition image: {e}")

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

        self.update_ring_partition(drones)

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
