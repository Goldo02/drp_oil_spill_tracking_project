import numpy as np

from matplotlib.path import Path

try:
    from scipy.interpolate import splprep, splev
except ImportError:  # pragma: no cover
    splprep = None
    splev = None

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None


# ============================================================================
# UTILITIES
# ============================================================================

def _largest_connected_component(binary_mask):
    """Keep only the largest 4-connected component."""
    binary = np.asarray(binary_mask, dtype=bool)

    if binary.size == 0:
        return binary

    visited = np.zeros_like(binary, dtype=bool)

    best_component = []
    best_size = 0

    rows, cols = binary.shape

    for row in range(rows):
        for col in range(cols):

            if not binary[row, col] or visited[row, col]:
                continue

            stack = [(row, col)]
            visited[row, col] = True
            component = []

            while stack:
                current_row, current_col = stack.pop()
                component.append((current_row, current_col))

                neighbours = (
                    (current_row - 1, current_col),
                    (current_row + 1, current_col),
                    (current_row, current_col - 1),
                    (current_row, current_col + 1),
                )

                for next_row, next_col in neighbours:
                    if not (
                        0 <= next_row < rows
                        and 0 <= next_col < cols
                    ):
                        continue

                    if (
                        binary[next_row, next_col]
                        and not visited[next_row, next_col]
                    ):
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))

            if len(component) > best_size:
                best_size = len(component)
                best_component = component

    filtered = np.zeros_like(binary, dtype=bool)

    for row, col in best_component:
        filtered[row, col] = True

    return filtered


# ============================================================================
# SIMULATION MAP
# ============================================================================

class SimulationMap:
    """Physical simulation domain and spatial discretization."""

    def __init__(
        self,
        xlim=(-5.0, 5.0),
        ylim=(-5.0, 5.0),
        grid_size=500,
    ):
        self.xlim = tuple(xlim)
        self.ylim = tuple(ylim)
        self.grid_size = int(grid_size)

        if self.grid_size < 2:
            raise ValueError("grid_size must be at least 2")

        if self.xlim[0] >= self.xlim[1]:
            raise ValueError("xlim must satisfy x_min < x_max")

        if self.ylim[0] >= self.ylim[1]:
            raise ValueError("ylim must satisfy y_min < y_max")

        self.x_coords = np.linspace(
            self.xlim[0],
            self.xlim[1],
            self.grid_size,
        )

        self.y_coords = np.linspace(
            self.ylim[0],
            self.ylim[1],
            self.grid_size,
        )

        self.X, self.Y = np.meshgrid(
            self.x_coords,
            self.y_coords,
            indexing="ij",
        )

    @property
    def dx(self):
        if len(self.x_coords) < 2:
            return 0.0

        return float(self.x_coords[1] - self.x_coords[0])

    @property
    def dy(self):
        if len(self.y_coords) < 2:
            return 0.0

        return float(self.y_coords[1] - self.y_coords[0])

    @property
    def shape(self):
        return self.X.shape

    def is_inside(self, x, y):
        return (
            self.xlim[0] <= x <= self.xlim[1]
            and self.ylim[0] <= y <= self.ylim[1]
        )

    def clip_position(self, x, y):
        return (
            float(np.clip(x, self.xlim[0], self.xlim[1])),
            float(np.clip(y, self.ylim[0], self.ylim[1])),
        )


# ============================================================================
# FIELD
# ============================================================================

class Field:
    """Abstract spatial scalar field."""

    def field(self, X, Y):
        raise NotImplementedError


# ============================================================================
# OIL SPILL
# ============================================================================

class OilSpill(Field):
    """
    Base class for oil-spill models.

    The spill owns its physical state and exposes:
        - update(dt)
        - field(X, Y)
        - get_field()
        - get_mask()
    """

    def update(self, dt):
        """Advance spill dynamics by dt."""
        del dt

    def field(self, X, Y):
        raise NotImplementedError

    def get_field(self, X=None, Y=None):
        """
        Return the current field.

        If X and Y are provided, evaluate the field there.
        Otherwise return the cached field.
        """
        if X is not None and Y is not None:
            return self.field(X, Y)

        if hasattr(self, "_field") and self._field is not None:
            return self._field.copy()

        raise ValueError(
            "No cached field is available. Provide X and Y."
        )

    def get_mask(self, threshold=0.5):
        """Return the current binary occupancy mask."""
        return self.get_field() >= float(threshold)


# ============================================================================
# CIRCLE OIL SPILL
# ============================================================================

class CircleOilSpill(OilSpill):
    """Circular oil spill with a softened boundary."""

    def __init__(
        self,
        x0=0.0,
        y0=0.0,
        radius=2.0,
        sigma=0.5,
    ):
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.radius = float(radius)
        self.sigma = float(sigma)

        if self.radius <= 0:
            raise ValueError("radius must be positive")

        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

        self._field = None

    def field(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        if X.shape != Y.shape:
            raise ValueError("X and Y must have the same shape")

        distance = np.sqrt(
            (X - self.x0) ** 2
            + (Y - self.y0) ** 2
        )

        return np.where(
            distance <= self.radius,
            1.0,
            np.exp(
                -((distance - self.radius) ** 2)
                / (2.0 * self.sigma ** 2)
            ),
        )

    def update(self, dt):
        """Static spill; extension point for future dynamics."""
        del dt


# ============================================================================
# SMOOTHED POLYGON OIL SPILL
# ============================================================================

class SmoothedPolygonOilSpill(OilSpill):
    """Irregular compact spill generated from a smoothed closed polygon."""

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
        self.smoothness = float(
            np.clip(smoothness, 0.0, 1.0)
        )

        self.continuous = bool(continuous)

        self.boundary_samples = (
            int(boundary_samples)
            if boundary_samples is not None
            else 500
        )

        x_min = float(np.min(self.X))
        x_max = float(np.max(self.X))
        y_min = float(np.min(self.Y))
        y_max = float(np.max(self.Y))

        self.x0 = float(
            (x_min + x_max) / 2.0
            if x0 is None
            else x0
        )

        self.y0 = float(
            (y_min + y_max) / 2.0
            if y0 is None
            else y0
        )

        self.vertices = self._generate_vertices()
        self.boundary = self._build_smooth_boundary(self.vertices)

        self._path = Path(
            self.boundary,
            closed=True,
        )

        self.radius = float(
            np.max(
                np.sqrt(
                    (self.boundary[:, 0] - self.x0) ** 2
                    + (self.boundary[:, 1] - self.y0) ** 2
                )
            )
        )

        self._field = self._evaluate_field(
            self.X,
            self.Y,
        )

    def _generate_vertices(self):
        angles = np.sort(
            self._rng.uniform(
                0.0,
                2.0 * np.pi,
                size=self.n_vertices,
            )
        )

        irregularity = 0.45 * (
            1.0 - self.smoothness
        )

        radial_noise = self._rng.normal(
            0.0,
            1.0,
            size=self.n_vertices,
        )

        phase_1 = self._rng.uniform(
            0.0,
            2.0 * np.pi,
        )

        phase_2 = self._rng.uniform(
            0.0,
            2.0 * np.pi,
        )

        harmonic = (
            0.55 * np.sin(
                3.0 * angles + phase_1
            )
            + 0.25 * np.sin(
                5.0 * angles + phase_2
            )
        )

        radii = self.r0 * (
            1.0
            + irregularity
            * (
                0.7 * radial_noise
                + harmonic
            )
        )

        radii = np.clip(
            radii,
            0.35 * self.r0,
            1.85 * self.r0,
        )

        return np.column_stack(
            [
                self.x0 + radii * np.cos(angles),
                self.y0 + radii * np.sin(angles),
            ]
        )

    def _build_smooth_boundary(self, vertices):
        samples = max(
            int(self.boundary_samples),
            24,
        )

        if (
            splprep is not None
            and splev is not None
            and len(vertices) >= 4
        ):
            try:
                spline_s = float(
                    self.smoothness
                    * len(vertices)
                    * 0.75
                )

                tck, _ = splprep(
                    [
                        vertices[:, 0],
                        vertices[:, 1],
                    ],
                    s=spline_s,
                    per=True,
                )

                u_new = np.linspace(
                    0.0,
                    1.0,
                    samples,
                    endpoint=False,
                )

                x_new, y_new = splev(
                    u_new,
                    tck,
                )

                return np.column_stack(
                    [x_new, y_new]
                )

            except Exception:
                pass

        return self._catmull_rom_closed(
            vertices,
            samples,
        )

    @staticmethod
    def _catmull_rom_closed(points, samples):
        points = np.asarray(
            points,
            dtype=float,
        )

        n = len(points)

        if n < 3:
            return points.copy()

        samples_per_segment = max(
            4,
            int(np.ceil(samples / n)),
        )

        padded = np.vstack(
            [
                points[-1],
                points,
                points[0],
                points[1],
            ]
        )

        boundary = []

        for i in range(n):
            p0, p1, p2, p3 = padded[i:i + 4]

            for t in np.linspace(
                0.0,
                1.0,
                samples_per_segment,
                endpoint=False,
            ):
                t2 = t * t
                t3 = t2 * t

                point = 0.5 * (
                    2.0 * p1
                    + (-p0 + p2) * t
                    + (
                        2.0 * p0
                        - 5.0 * p1
                        + 4.0 * p2
                        - p3
                    ) * t2
                    + (
                        -p0
                        + 3.0 * p1
                        - 3.0 * p2
                        + p3
                    ) * t3
                )

                boundary.append(point)

        return np.asarray(
            boundary,
            dtype=float,
        )

    def _evaluate_field(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        if X.shape != Y.shape:
            raise ValueError("X and Y must have the same shape")

        points = np.column_stack(
            [
                X.ravel(),
                Y.ravel(),
            ]
        )

        target_shape = X.shape

        inside = self._path.contains_points(
            points,
            radius=1e-9,
        ).reshape(target_shape)

        inside = _largest_connected_component(inside)

        if cKDTree is None:
            return inside.astype(float)

        tree = cKDTree(self.boundary)

        distance, _ = tree.query(
            points,
            k=1,
        )

        distance = distance.reshape(
            target_shape
        )

        softness = (
            0.10
            + 0.22 * self.smoothness
        )

        if not self.continuous:
            softness *= 0.8

        boundary_width = max(
            self.r0 * softness,
            1e-6,
        )

        signed_distance = np.where(
            inside,
            -distance,
            distance,
        )

        return 1.0 / (
            1.0
            + np.exp(
                signed_distance
                / boundary_width
            )
        )

    def field(self, X=None, Y=None):
        if X is None or Y is None:
            return self._field.copy()

        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        if X.shape != Y.shape:
            raise ValueError("X and Y must have the same shape")

        return self._evaluate_field(X, Y)

    def update(self, dt):
        """Static spill; extension point for future dynamics."""
        del dt
