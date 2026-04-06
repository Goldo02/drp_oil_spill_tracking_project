import numpy as np

from matplotlib.path import Path

try:
    from scipy.interpolate import splprep, splev
except ImportError:  # pragma: no cover - SciPy is available in this workspace, but keep a fallback.
    splprep = None
    splev = None

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None


def _largest_connected_component(binary_mask):
    """Keep only the largest 4-connected component in a binary mask."""
    binary = np.asarray(binary_mask, dtype=bool)
    if binary.size == 0:
        return binary

    visited = np.zeros_like(binary, dtype=bool)
    best_component = []
    best_size = 0
    rows, cols = binary.shape

    for r in range(rows):
        for c in range(cols):
            if not binary[r, c] or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            component = []

            while stack:
                cr, cc = stack.pop()
                component.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and binary[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            if len(component) > best_size:
                best_size = len(component)
                best_component = component

    filtered = np.zeros_like(binary, dtype=bool)
    for r, c in best_component:
        filtered[r, c] = True
    return filtered

class OilSpill:
    """Base class for oil spill models."""
    def field(self, X, Y):
        raise NotImplementedError

class SmoothedPolygonOilSpill(OilSpill):
    """Generate a single compact irregular spill from a closed smoothed polygon."""

    def __init__(
        self,
        X,
        Y,
        n_vertices=36,
        r0=2.0,
        smoothness=0.2,
        x0=None,
        y0=None,
        seed=None,
        continuous=False,
        boundary_samples=None,
    ):
        self.X = np.asarray(X, dtype=float)
        self.Y = np.asarray(Y, dtype=float)
        if self.X.shape != self.Y.shape:
            raise ValueError("X and Y must have the same shape")

        self._rng = np.random.default_rng(seed)
        self.n_vertices = max(3, int(n_vertices))
        self.r0 = float(max(r0, 1e-6))
        self.smoothness = float(np.clip(smoothness, 0.0, 1.0))
        self.continuous = bool(continuous)
        self.boundary_samples = int(boundary_samples) if boundary_samples is not None else 500

        x_min = float(np.min(self.X))
        x_max = float(np.max(self.X))
        y_min = float(np.min(self.Y))
        y_max = float(np.max(self.Y))
        self.x0 = float((x_min + x_max) / 2.0 if x0 is None else x0)
        self.y0 = float((y_min + y_max) / 2.0 if y0 is None else y0)

        self.vertices = self._generate_vertices()
        self.boundary = self._build_smooth_boundary(self.vertices)
        self._path = Path(self.boundary, closed=True)
        self._field = self._evaluate_field(self.X, self.Y)
        self._mask = self._field >= 0.5
        self.radius = float(np.max(np.sqrt((self.boundary[:, 0] - self.x0) ** 2 + (self.boundary[:, 1] - self.y0) ** 2)))

    def _generate_vertices(self):
        """Sample random vertices around the center while keeping the polygon compact."""
        angles = np.sort(self._rng.uniform(0.0, 2.0 * np.pi, size=self.n_vertices))

        # Larger smoothness means less radial jitter and therefore a rounder shape.
        irregularity = 0.45 * (1.0 - self.smoothness)
        radial_noise = self._rng.normal(0.0, 1.0, size=self.n_vertices)
        phase_1 = self._rng.uniform(0.0, 2.0 * np.pi)
        phase_2 = self._rng.uniform(0.0, 2.0 * np.pi)
        harmonic = 0.55 * np.sin(3.0 * angles + phase_1) + 0.25 * np.sin(5.0 * angles + phase_2)
        radii = self.r0 * (1.0 + irregularity * (0.7 * radial_noise + harmonic))
        radii = np.clip(radii, 0.35 * self.r0, 1.85 * self.r0)

        x = self.x0 + radii * np.cos(angles)
        y = self.y0 + radii * np.sin(angles)
        return np.column_stack([x, y])

    def _build_smooth_boundary(self, vertices):
        """Create a closed spline boundary from the raw polygon vertices."""
        samples = max(int(self.boundary_samples), 24)
        if splprep is not None and splev is not None and len(vertices) >= 4:
            try:
                # s controls how much the spline smooths the random polygon.
                spline_s = float(self.smoothness * len(vertices) * 0.75)
                tck, _ = splprep([vertices[:, 0], vertices[:, 1]], s=spline_s, per=True)
                u_new = np.linspace(0.0, 1.0, samples, endpoint=False)
                x_new, y_new = splev(u_new, tck)
                return np.column_stack([x_new, y_new])
            except Exception:
                pass

        # Fallback: periodic Catmull-Rom spline.
        return self._catmull_rom_closed(vertices, samples)

    @staticmethod
    def _catmull_rom_closed(points, samples):
        pts = np.asarray(points, dtype=float)
        n = len(pts)
        if n < 3:
            return pts.copy()

        samples_per_segment = max(4, int(np.ceil(samples / n)))
        padded = np.vstack([pts[-1], pts, pts[0], pts[1]])
        boundary = []
        for i in range(n):
            p0, p1, p2, p3 = padded[i : i + 4]
            for t in np.linspace(0.0, 1.0, samples_per_segment, endpoint=False):
                t2 = t * t
                t3 = t2 * t
                point = 0.5 * (
                    (2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                )
                boundary.append(point)
        return np.asarray(boundary, dtype=float)

    def _evaluate_field(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        points = np.column_stack([X.ravel(), Y.ravel()])
        target_shape = X.shape
        inside = self._path.contains_points(points, radius=1e-9).reshape(target_shape)
        inside = _largest_connected_component(inside)
        if cKDTree is None:
            return inside.astype(float)

        tree = cKDTree(self.boundary)
        distance, _ = tree.query(points, k=1)
        distance = distance.reshape(target_shape)
        # A wider width gives a softer transition around the polygon edge.
        softness = 0.10 + 0.22 * self.smoothness
        if not self.continuous:
            softness *= 0.8
        boundary_width = max(self.r0 * softness, 1e-6)
        signed_distance = np.where(inside, -distance, distance)
        return 1.0 / (1.0 + np.exp(signed_distance / boundary_width))

    def field(self, X=None, Y=None):
        """Return the oil field on the stored grid or on a supplied grid."""
        if X is None or Y is None:
            return self._field.copy()

        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if X.shape != Y.shape:
            raise ValueError("X and Y must have the same shape")
        return self._evaluate_field(X, Y)

    def get_mask(self):
        return self._mask.copy()

    def get_field(self):
        return self._field.copy()

class CircleOilSpill(OilSpill):
    """Circular oil spill at (x0, y0) with a softened boundary."""
    def __init__(self, x0=0, y0=0, radius=2, sigma=0.5):
        self.x0 = x0
        self.y0 = y0
        self.radius = radius
        self.sigma = sigma

    def field(self, X, Y):
        dist = np.sqrt((X - self.x0)**2 + (Y - self.y0)**2)
        # Softened circle: 1.0 inside, exponential decay outside the radius.
        return np.where(dist <= self.radius, 1.0, np.exp(-(dist - self.radius)**2 / (2 * self.sigma**2)))

class SimulationMap:
    """Encapsulates world dimensions and grid coordinates."""
    def __init__(self, xlim=(-5, 5), ylim=(-5, 5), grid_size=500):
        self.xlim = xlim
        self.ylim = ylim
        self.grid_size = grid_size
        
        self.x_coords = np.linspace(xlim[0], xlim[1], grid_size)
        self.y_coords = np.linspace(ylim[0], ylim[1], grid_size)
        # Use matrix indexing so axis 0 corresponds to x and axis 1 corresponds to y.
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords, indexing="ij")

    def is_inside(self, x, y):
        return (self.xlim[0] <= x <= self.xlim[1] and 
                self.ylim[0] <= y <= self.ylim[1])
