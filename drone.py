import numpy as np

from sensors import GPSSensor


class Drone:
    """Drone state used by the boundary controller."""

    def __init__(
        self,
        drone_id,
        x,
        y,
        grid_shape,
        grid_bounds,
        gps_noise=0.03,
        max_speed=0.12,
    ):
        self.drone_id = drone_id
        self.x = float(x)
        self.y = float(y)
        self.max_speed = float(max_speed)

        self.grid_shape = tuple(grid_shape)
        self.grid_bounds = tuple(grid_bounds)
        self.x_min, self.x_max, self.y_min, self.y_max = self.grid_bounds
        self.Nx, self.Ny = self.grid_shape
        self.grid = np.zeros(self.grid_shape, dtype=float)
        self.gps = GPSSensor(noise_std=gps_noise)

        self.last_control_mode = "idle"
        self.last_control_vector = np.zeros(2, dtype=float)
        self.known_positions = {
            self.drone_id: np.array([self.x, self.y], dtype=float),
        }
        self.target_centroid = None
        self.last_ring_info = None

    @property
    def position(self):
        return np.array([self.x, self.y], dtype=float)

    def get_gps_pos(self):
        return self.gps.sense(self.position)

    def action(self, command, dt=1.0, bounds=None):
        command = self._clip_command(command, self.max_speed)
        self.last_control_vector = command.copy()

        self.x += command[0] * float(dt)
        self.y += command[1] * float(dt)

        x_bounds, y_bounds = bounds if bounds is not None else (
            (self.x_min, self.x_max),
            (self.y_min, self.y_max),
        )
        self.x = float(np.clip(self.x, x_bounds[0], x_bounds[1]))
        self.y = float(np.clip(self.y, y_bounds[0], y_bounds[1]))
        self.known_positions[self.drone_id] = self.position
        return command

    def set_control_mode(self, mode):
        self.last_control_mode = str(mode)

    @staticmethod
    def _clip_command(command, max_speed):
        vec = np.asarray(command, dtype=float)
        if vec.shape != (2,) or not np.all(np.isfinite(vec)):
            return np.zeros(2, dtype=float)

        speed = float(np.linalg.norm(vec))
        if speed <= 1e-12:
            return np.zeros(2, dtype=float)
        if speed > max_speed:
            vec = vec * (max_speed / speed)
        return vec
