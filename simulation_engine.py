import numpy as np
from drone import Drone
from edge_detection import identify_centroid, check_geometric_lock

class SimulationEngine:
    """Orchestrates the simulation loop for multiple drones."""
    def __init__(self, sim_map, oil_spill, dt=0.1):
        self.sim_map = sim_map
        self.oil_spill = oil_spill
        self.dt = dt
        self.drones = []
        self.frame = 0
        
        # Pre-calculated field
        self.world_field = oil_spill.field(sim_map.X, sim_map.Y)
        
        # Matveev Control Parameters
        self.c_star = 0.5 
        self.u_bar = 2.0 # Angular velocity gain
        self.tracking_speed = 0.4

    def add_drone(self, drone_id, x, y):
        drone = Drone(drone_id, x, y, map_bounds=(*self.sim_map.xlim, *self.sim_map.ylim))
        # Initial random velocity and heading for SEARCH
        vx, vy = np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)
        drone.set_velocity(vx, vy)
        drone.theta = np.arctan2(vy, vx)
        self.drones.append(drone)

    def step(self):
        """Update physics and logic for all drones."""
        self.frame += 1
        for drone in self.drones:
            # 1. Perception (Noisy)
            camera_view = drone.get_camera_view(self.world_field, self.sim_map.x_coords, self.sim_map.y_coords)
            print(f"Frame {self.frame}, Drone {drone.drone_id}")
            print("Camera sum:", np.sum(camera_view))
            print("Camera view (central 5x5):")
            print(camera_view)

            # 2. Logic (Matveev Boundary Tracking)
            half = camera_view.shape[0] // 2
            center_val = camera_view[half, half]

            if drone.mode == "SEARCH":
                if center_val > 0.5:
                    drone.mode = "APPROACH" # Transition to tracking
                    print(f"Frame {self.frame}: Drone {drone.drone_id} -> APPROACH (Matveev start).")
                
            if drone.mode == "APPROACH":
                # Matveev Control Law: u = u_bar * sgn(c - c_star)
                # If we are inside (center_val > 0.5), we want to turn out.
                # If we are outside (center_val < 0.5), we want to turn in.
                # Let's use: u = u_bar * (0.5 - center_val)
                # Since center_val is binary (0 or 1), this gives u = \pm 0.5*u_bar
                # Actually, let's just use the sgn logic directly:
                u = self.u_bar if center_val < 0.5 else -self.u_bar
                
                drone.theta += u * self.dt
                drone.set_velocity(self.tracking_speed * np.cos(drone.theta), 
                                   self.tracking_speed * np.sin(drone.theta))
                
                # Check for "locking" (optional, but keep it for visual feedback)
                if check_geometric_lock(camera_view):
                    # We can keep APPROACH mode or switch to LOCKED
                    # The user said "seguirlo", so we should keep moving.
                    # I'll use a color change in the visualizer instead of stopping.
                    pass
                
                # If completely lost (e.g. noise), return to search
                if np.all(camera_view < 0.1):
                    drone.mode = "SEARCH"
                    drone.set_velocity(np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5))
                    drone.theta = np.arctan2(drone.vy, drone.vx)

            # 3. Physics Integration
            drone.update_position(self.dt)
