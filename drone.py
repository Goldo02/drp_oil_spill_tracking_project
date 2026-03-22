import numpy as np
from sensors import GPSSensor, CameraSensor

class Drone:
    """
    Modular Drone class.
    Handles movement, sensor polling, and internal state.
    """
    def __init__(self, drone_id, x, y, map_bounds, sensor_size=25, 
                 gps_noise=0.00, camera_noise=0.00):
        self.drone_id = drone_id
        self.x = x
        self.y = y
        self.map_bounds = map_bounds # (xmin, xmax, ymin, ymax)
        
        # Internal Velocity
        self.vx = 0.0
        self.vy = 0.0
        
        # Sensors
        self.gps = GPSSensor(noise_std=gps_noise)
        self.camera = CameraSensor(size=sensor_size, noise_std=camera_noise)
        
        # State Machine
        self.mode = "SEARCH" # SEARCH, APPROACH, LOCKED
        self.last_dir = np.array([0.0, 0.0])

    def get_gps_pos(self):
        """Returns noisy (x, y) coordinates."""
        return self.gps.sense((self.x, self.y))

    def get_camera_view(self, world_field, x_coords, y_coords):
        """Returns noisy 25x25 local matrix."""
        return self.camera.sense(world_field, self.x, self.y, x_coords, y_coords)

    def set_velocity(self, vx, vy):
        self.vx = vx
        self.vy = vy

    def update_position(self, dt):
        """Integrated movement within map boundaries."""
        if self.mode == "LOCKED":
            self.vx, self.vy = 0.0, 0.0
            
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Bounce off boundaries
        xmin, xmax, ymin, ymax = self.map_bounds
        if self.x <= xmin or self.x >= xmax: self.vx *= -1
        if self.y <= ymin or self.y >= ymax: self.vy *= -1
        
        # Clamp to bounds
        self.x = np.clip(self.x, xmin, xmax)
        self.y = np.clip(self.y, ymin, ymax)
