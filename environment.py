import numpy as np

class OilSpill:
    """Base class for oil spill models."""
    def field(self, X, Y):
        raise NotImplementedError

class CircleOilSpill(OilSpill):
    """Circular oil spill at (x0, y0) with radius r0, now softened with Gaussian."""
    def __init__(self, x0=0, y0=0, r0=2, sigma=0.5):
        self.x0 = x0
        self.y0 = y0
        self.r0 = r0
        self.sigma = sigma

    def field(self, X, Y):
        dist = np.sqrt((X - self.x0)**2 + (Y - self.y0)**2)
        # Softened circle: 1.0 inside, exponential decay outside r0
        return np.where(dist <= self.r0, 1.0, np.exp(-(dist - self.r0)**2 / (2 * self.sigma**2)))

class GaussianOilSpill(OilSpill):
    """Gaussian oil spill model for a more continuous field."""
    def __init__(self, x0=0, y0=0, sigma=1.5, amplitude=1.0):
        self.x0 = x0
        self.y0 = y0
        self.sigma = sigma
        self.amplitude = amplitude

    def field(self, X, Y):
        dist_sq = (X - self.x0)**2 + (Y - self.y0)**2
        return self.amplitude * np.exp(-dist_sq / (2 * self.sigma**2))

class SimulationMap:
    """Encapsulates world dimensions and grid coordinates."""
    def __init__(self, xlim=(-5, 5), ylim=(-5, 5), grid_size=500):
        self.xlim = xlim
        self.ylim = ylim
        self.grid_size = grid_size
        
        self.x_coords = np.linspace(xlim[0], xlim[1], grid_size)
        self.y_coords = np.linspace(ylim[0], ylim[1], grid_size)
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords)

    def is_inside(self, x, y):
        return (self.xlim[0] <= x <= self.xlim[1] and 
                self.ylim[0] <= y <= self.ylim[1])
