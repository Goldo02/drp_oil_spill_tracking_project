import numpy as np

class Sensor:
    """Base class for simple robot sensors."""

    def __init__(self, noise_std=0.0):
        self.noise_std = float(noise_std)

    def add_noise(self, value):
        value = np.asarray(value, dtype=float)
        if self.noise_std <= 0.0:
            return value
        return value + np.random.normal(0.0, self.noise_std, size=value.shape)


class GPSSensor(Sensor):
    """GPS sensor con un rumore predefinito (hardcodato a 0.03)."""

    def __init__(self, noise_std=0.03): 
        super().__init__(noise_std=noise_std)

    def sense(self, real_position):
        position = np.asarray(real_position, dtype=float)
        if position.shape != (2,):
            raise ValueError("GPS position must have shape (2,)")
        return self.add_noise(position)
