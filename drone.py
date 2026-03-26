import numpy as np
from sensors import GPSSensor, CameraSensor

class Drone:
    """
    Static drone model used for radius estimation.
    The drone does not move during the simulation; it only senses.
    """
    def __init__(self, drone_id, x, y, map_bounds, sensor_size=100, 
                 gps_noise=0.03, camera_noise=0.03, true_x0=0.0, true_y0=0.0):
        self.drone_id = drone_id
        self.x = x
        self.y = y
        self.map_bounds = map_bounds # (xmin, xmax, ymin, ymax)

        # Sensors
        self.gps = GPSSensor(noise_std=gps_noise)
        self.camera = CameraSensor(size=sensor_size, noise_std=camera_noise)

        # Current local estimate of the spill center and radius.
        # The center is known in this simulation, but the field is structured
        # so the rest of the pipeline still works if it is later estimated.
        self.estimate_x0 = true_x0
        self.estimate_y0 = true_y0
        self.estimate_r0 = 1.0
        self.edge_detected = False
        self.last_gradient_peak = None
        self.last_edge_point = None
        self.last_oil_fraction = None

    def get_gps_pos(self):
        """Returns noisy (x, y) coordinates."""
        return self.gps.sense((self.x, self.y))

    def get_camera_view(self, world_field, x_coords, y_coords):
        """Returns noisy 25x25 local matrix."""
        return self.camera.sense(world_field, self.x, self.y, x_coords, y_coords)
