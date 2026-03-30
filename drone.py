import numpy as np
from sensors import GPSSensor, CameraSensor

class Drone:
    """
    Simple single-integrator drone model.
    The drone senses the spill and is moved by an external control input.
    """
    def __init__(self, drone_id, x, y, map_bounds, sensor_size=100, 
                 gps_noise=0.03, camera_noise=0.03, initial_x0=0.0, initial_y0=0.0,
                 initial_r0=1.0, max_speed=0.6):
        self.drone_id = drone_id
        self.x = x
        self.y = y
        self.map_bounds = map_bounds # (xmin, xmax, ymin, ymax)
        self.max_speed = float(max_speed)

        # Sensors
        self.gps = GPSSensor(noise_std=gps_noise)
        self.camera = CameraSensor(size=sensor_size, noise_std=camera_noise)

        # Current local estimate of the spill: theta = [cx, cy, r]
        # Initialize with provided guess.
        self.theta = np.array([float(initial_x0), float(initial_y0), float(initial_r0)], dtype=float)
        self.theta_prev = None  # To be set after first valid estimation
        self.theta_measured = np.array([float(initial_x0), float(initial_y0), float(initial_r0)], dtype=float)
        self.theta_fused = np.array([float(initial_x0), float(initial_y0), float(initial_r0)], dtype=float)

        self.has_radius_estimate = False
        self.edge_detected = False
        self.last_gradient_peak = None
        self.last_edge_point = None
        self.last_oil_fraction = None
        self.last_boundary_tangential = False
        self.last_control_mode = None
        self.u_x = 0.0
        self.u_y = 0.0
        self.last_voronoi_target = None
        self.last_exploration_target = None
        self.exploration_direction = None
        self.exploration_speed = 0.0
        
        # Points used for local NLS fit (for visualization)
        self.last_nls_points = None
        
        # Consolidation for fusion requirements
        self.has_measure = False
        self.confidence_weight = 0.0
        self.neighbors = []

    @property
    def estimate_x0(self):
        return float(self.theta[0])

    @property
    def estimate_y0(self):
        return float(self.theta[1])

    @property
    def estimate_r0(self):
        return float(self.theta[2])

    def get_gps_pos(self):
        """Returns noisy (x, y) coordinates."""
        return self.gps.sense((self.x, self.y))

    def get_camera_view(self, world_field, x_coords, y_coords):
        """Returns noisy local matrix."""
        return self.camera.sense(world_field, self.x, self.y, x_coords, y_coords)

    def set_control(self, u_x, u_y):
        """Store the control command computed by the simulator."""
        self.u_x = float(u_x)
        self.u_y = float(u_y)

    def update_position(self, dt, map_bounds=None, max_speed=None):
        """Integrate a single-integrator command and keep the drone inside bounds."""
        command = np.array([self.u_x, self.u_y], dtype=float)
        speed = float(np.linalg.norm(command))
        limit = self.max_speed if max_speed is None else float(max_speed)

        if limit > 0 and speed > limit:
            command = command / speed * limit

        self.x += float(command[0] * dt)
        self.y += float(command[1] * dt)

        bounds = self.map_bounds if map_bounds is None else map_bounds
        if bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            self.x = float(np.clip(self.x, xmin, xmax))
            self.y = float(np.clip(self.y, ymin, ymax))

        self.u_x = float(command[0])
        self.u_y = float(command[1])
        return command
