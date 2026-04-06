import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, RegularPolygon, Rectangle

class Visualizer:
    """Handles all plotting and animation for the simulation."""
    def __init__(
        self,
        sim_map,
        oil_spill,
        communication_radius=None,
        show_communication_radius=False,
        show_nls_points=False,
    ):
        plt.ion()
        self.sim_map = sim_map
        self.oil_spill = oil_spill
        self.communication_radius = communication_radius
        self.show_communication_radius = show_communication_radius and communication_radius is not None
        self.show_nls_points = show_nls_points
        
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(sim_map.xlim)
        self.ax.set_ylim(sim_map.ylim)
        if self.show_communication_radius:
            self.ax.set_title(
                f"Distributed Occupancy Grid Mapping - Communication radius Rc={communication_radius:.2f}"
            )
        else:
            self.ax.set_title("Distributed Occupancy Grid Mapping - Fully Connected")
        
        # Initial draw of the field
        field_data = oil_spill.field(sim_map.X, sim_map.Y)
        # Use 'Greys' colormap: 0 is white, higher values are darker
        self.img = self.ax.imshow(field_data.T, extent=[*sim_map.xlim, *sim_map.ylim], 
                                  origin='lower', cmap='Greys', alpha=0.8, vmin=0, vmax=1.0)
        # Draw multiple contour levels for a more continuous look
        self.contour = self.ax.contour(sim_map.X, sim_map.Y, field_data, levels=[0.1, 0.5, 0.9], 
                                       colors='black', alpha=0.5, linewidths=0.5)

        self.drone_patches = {} # {drone_id: [patch_comm, patch_body, patch_sensor]}
        self.texts = {} # {drone_id: text_label}
        self.edge_markers = {} # {drone_id: [scatter, annotation]}
        self.nls_markers = {} # {drone_id: scatter}
        self.control_arrows = {} # {drone_id: quiver}
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_drone(self, drone):
        # Remove old patches
        if drone.drone_id in self.drone_patches:
            for p in self.drone_patches[drone.drone_id]: p.remove()
            self.texts[drone.drone_id].remove()

        if drone.drone_id in self.edge_markers:
            for artist in self.edge_markers[drone.drone_id]:
                if artist is not None:
                    artist.remove()

        if drone.drone_id in self.nls_markers:
            if self.nls_markers[drone.drone_id] is not None:
                self.nls_markers[drone.drone_id].remove()

        if drone.drone_id in self.control_arrows:
            if self.control_arrows[drone.drone_id] is not None:
                self.control_arrows[drone.drone_id].remove()

        patches = []

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

        # Draw new body
        body = RegularPolygon((drone.x, drone.y), numVertices=6, radius=0.15, color='royalblue')
        self.ax.add_patch(body)
        patches.append(body)
        
        # Draw sensor area (based on real position for visualization)
        # Assuming grid spacing is constant
        dx = self.sim_map.x_coords[1] - self.sim_map.x_coords[0]
        dy = self.sim_map.y_coords[1] - self.sim_map.y_coords[0]
        s_w, s_h = drone.camera.size * dx, drone.camera.size * dy
        sensor_box = Rectangle((drone.x - s_w/2, drone.y - s_h/2), s_w, s_h, 
                               edgecolor='blue', facecolor='none', alpha=0.3, linestyle='--')
        self.ax.add_patch(sensor_box)
        patches.append(sensor_box)
        
        # Add label
        label = self.ax.text(drone.x + 0.2, drone.y + 0.2, f"Drone {drone.drone_id}", fontsize=8)

        self.drone_patches[drone.drone_id] = patches
        self.texts[drone.drone_id] = label

        edge_marker = None
        edge_label = None
        if drone.edge_detected and drone.last_edge_point is not None:
            edge_color = "limegreen"
            edge_marker = self.ax.scatter(
                [drone.last_edge_point[0]],
                [drone.last_edge_point[1]],
                s=70,
                c=edge_color,
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
                drone.last_edge_point[0] + 0.12,
                drone.last_edge_point[1] + 0.12,
                annotation,
                fontsize=7,
                color=edge_color,
                zorder=7,
            )

        self.edge_markers[drone.drone_id] = [edge_marker, edge_label]

        nls_marker = None
        if self.show_nls_points:
            pts = getattr(drone, "last_nls_points", None)
            if pts is not None:
                pts = np.asarray(pts, dtype=float)
                if pts.size > 0:
                    nls_marker = self.ax.scatter(
                        pts[:, 0],
                        pts[:, 1],
                        s=2,
                        c="red",
                        alpha=0.4,
                        zorder=4,
                    )

        self.nls_markers[drone.drone_id] = nls_marker

        control_arrow = None
        control_vec = np.asarray(getattr(drone, "last_control_vector", np.zeros(2, dtype=float)), dtype=float)
        control_norm = float(np.linalg.norm(control_vec))
        if control_norm > 1e-12:
            if control_norm > 0.12:
                control_vec = control_vec * (0.12 / control_norm)
            control_arrow = self.ax.quiver(
                drone.x,
                drone.y,
                control_vec[0],
                control_vec[1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="crimson",
                width=0.0045,
                alpha=0.9,
                zorder=8,
            )

        self.control_arrows[drone.drone_id] = control_arrow

    def render(self, drones, pause=True):
        for drone in drones:
            self.update_drone(drone)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        if pause:
            plt.pause(0.001)
