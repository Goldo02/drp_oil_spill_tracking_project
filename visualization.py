import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon, Rectangle

class Visualizer:
    """Handles all plotting and animation for the simulation."""
    def __init__(self, sim_map, oil_spill, communication_radius=None, show_communication_radius=False):
        self.sim_map = sim_map
        self.oil_spill = oil_spill
        self.communication_radius = communication_radius
        self.show_communication_radius = show_communication_radius and communication_radius is not None
        
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(sim_map.xlim)
        self.ax.set_ylim(sim_map.ylim)
        if self.show_communication_radius:
            self.ax.set_title(
                f"Voronoi Control + Radius Estimation - Communication radius Rc={communication_radius:.2f}"
            )
        else:
            self.ax.set_title("Voronoi Control + Radius Estimation - Fully Connected")
        
        # Initial draw of the field
        field_data = oil_spill.field(sim_map.X, sim_map.Y)
        # Use 'Greys' colormap: 0 is white, higher values are darker
        self.img = self.ax.imshow(field_data, extent=[*sim_map.xlim, *sim_map.ylim], 
                                  origin='lower', cmap='Greys', alpha=0.8, vmin=0, vmax=1.0)
        # Draw multiple contour levels for a more continuous look
        self.contour = self.ax.contour(sim_map.X, sim_map.Y, field_data, levels=[0.1, 0.5, 0.9], 
                                       colors='black', alpha=0.5, linewidths=0.5)

        self.drone_patches = {} # {drone_id: [patch_comm, patch_body, patch_sensor]}
        self.texts = {} # {drone_id: text_label}
        self.edge_markers = {} # {drone_id: [scatter, annotation]}
        self.target_markers = {} # {drone_id: [scatter, annotation]}

    def update_drone(self, drone):
        # Remove old patches
        if drone.drone_id in self.drone_patches:
            for p in self.drone_patches[drone.drone_id]: p.remove()
            self.texts[drone.drone_id].remove()

        if drone.drone_id in self.edge_markers:
            for artist in self.edge_markers[drone.drone_id]:
                if artist is not None:
                    artist.remove()

        if drone.drone_id in self.target_markers:
            for artist in self.target_markers[drone.drone_id]:
                if artist is not None:
                    artist.remove()

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
            if drone.last_gradient_peak is not None:
                oil_fraction = getattr(drone, "last_oil_fraction", None)
                if oil_fraction is None:
                    annotation = f"g={drone.last_gradient_peak:.3f}"
                else:
                    annotation = f"g={drone.last_gradient_peak:.3f}\n{100.0 * oil_fraction:.1f}% oil"
                edge_label = self.ax.text(
                    drone.last_edge_point[0] + 0.12,
                    drone.last_edge_point[1] + 0.12,
                    annotation,
                    fontsize=7,
                    color=edge_color,
                    zorder=7,
                )

        self.edge_markers[drone.drone_id] = [edge_marker, edge_label]

        target_marker = None
        target_label = None
        if getattr(drone, "last_voronoi_target", None) is not None:
            target_marker = self.ax.scatter(
                [drone.last_voronoi_target[0]],
                [drone.last_voronoi_target[1]],
                s=35,
                c="darkorange",
                marker="o",
                edgecolors="black",
                linewidths=0.6,
                alpha=0.8,
                zorder=5,
            )
            target_label = self.ax.text(
                drone.last_voronoi_target[0] + 0.1,
                drone.last_voronoi_target[1] + 0.1,
                "V",
                fontsize=7,
                color="darkorange",
                zorder=6,
            )

        self.target_markers[drone.drone_id] = [target_marker, target_label]

    def render(self, drones, pause=True):
        for drone in drones:
            self.update_drone(drone)
        plt.draw()
        if pause:
            plt.pause(0.001)
