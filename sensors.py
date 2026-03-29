import numpy as np
from scipy.ndimage import gaussian_filter

class Sensor:
    """Base class for all sensors."""
    def __init__(self, noise_std=0.0):
        self.noise_std = noise_std

    def add_noise(self, value):
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=value.shape if hasattr(value, 'shape') else None)
            return value + noise
        return value

class GPSSensor(Sensor):
    """Proprioceptive GPS sensor: returns (x, y) + noise."""
    def sense(self, real_pos):
        real_pos = np.array(real_pos)
        return self.add_noise(real_pos)

class CameraSensor(Sensor):
    """Exteroceptive camera sensor: returns a local continuous matrix + noise."""
    def __init__(self, size=100, noise_std=0.1, apply_blur=False, blur_sigma=1.0):
        super().__init__(noise_std)
        self.size = size
        self.apply_blur = apply_blur
        self.blur_sigma = blur_sigma

    def sense(self, world_field, x, y, x_coords, y_coords):
        # Compute grid indices for the center
        # rows (j) map to y, cols (i) map to x
        dx = x_coords[1] - x_coords[0]
        dy = y_coords[1] - y_coords[0]
        
        # Grid indices (i=col, j=row)
        i_center = int((x - x_coords[0]) / dx)
        j_center = int((y - y_coords[0]) / dy)
        
        half = self.size // 2
        
        # Rows = Y, Cols = X
        j_min = max(0, j_center - half)
        j_max = min(world_field.shape[0], j_center + half + 1)
        i_min = max(0, i_center - half)
        i_max = min(world_field.shape[1], i_center + half + 1)
        
        # world_field[row, col] -> world_field[y, x]
        local_matrix = world_field[j_min:j_max, i_min:i_max].astype(float)
        
        # Add Gaussian noise
        noisy_matrix = self.add_noise(local_matrix)
        
        # Optional: apply Gaussian blur to smooth the field
        if self.apply_blur:
            noisy_matrix = gaussian_filter(noisy_matrix, sigma=self.blur_sigma)
        
        # Re-pad if at boundary
        if noisy_matrix.shape != (self.size, self.size):
            pad_before_j = max(0, half - j_center)
            pad_after_j = max(0, (j_center + half + 1) - world_field.shape[0])
            pad_before_i = max(0, half - i_center)
            pad_after_i = max(0, (i_center + half + 1) - world_field.shape[1])
            
            noisy_matrix = np.pad(noisy_matrix, 
                                  ((pad_before_j, pad_after_j), (pad_before_i, pad_after_i)), 
                                  'constant', constant_values=0)
        
        return noisy_matrix
