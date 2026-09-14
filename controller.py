import numpy as np


class Controller:
    """
    Minimal Controller stubbed for user-driven reimplementation.

    This file intentionally removes all autonomous motion and control logic.
    Control methods are left as clear TODO stubs for the user to implement.

    The class preserves the minimal data structures and helpers required by
    other modules in the workspace, and provides a small helper to place
    drones randomly on a known boundary.
    """

    def __init__(
        self,
        sim_map,
        communication_radius,
        fully_connected=False,
        occupancy_threshold=0.5,
        resolution=0.1,
        **kwargs,
    ):
        # Basic environment/configuration state (kept for compatibility)
        self.sim_map = sim_map
        self.communication_radius = float(communication_radius)
        self.fully_connected = bool(fully_connected)
        self.occupancy_threshold = float(occupancy_threshold)
        self.resolution = float(resolution)

        # Known boundary representation: ordered list of (x,y) points.
        # The user requested that each drone "already knows" the shape; this
        # controller will keep a shared canonical copy and helpers to assign it.
        self.known_boundary_points = np.empty((0, 2), dtype=float)
        self.known_boundary_initialized = False

    # ------------------------------------------------------------------
    # BOUNDARY / INITIALIZATION HELPERS
    # ------------------------------------------------------------------

    def initialize_known_boundary(self, world_field_or_points, x_coords=None, y_coords=None, force_closed=True):
        """Initialize known boundary from either a list of (x,y) points or a grid.

        Accepts either:
        - an iterable of (x,y) pairs (boundary points), or
        - a 2D occupancy `world_field` with optional `x_coords`, `y_coords`.

        Returns the stored boundary points as an (N,2) ndarray.
        """
        arr = np.asarray(world_field_or_points)

        # Case A: already a list of boundary points
        if arr.ndim == 2 and arr.shape[1] == 2:
            pts = arr.copy()
            self.known_boundary_points = pts
            self.known_boundary_initialized = True
            return self.known_boundary_points.copy()

        # Case B: given an occupancy grid -> extract contour points
        if arr.ndim != 2:
            # not a grid nor a list of points
            raise ValueError("initialize_known_boundary expects a 2D grid or an (N,2) array")

        field = arr.astype(float)
        if x_coords is None:
            x_coords = np.arange(field.shape[0], dtype=float)
        else:
            x_coords = np.asarray(x_coords, dtype=float)
        if y_coords is None:
            y_coords = np.arange(field.shape[1], dtype=float)
        else:
            y_coords = np.asarray(y_coords, dtype=float)

        occupied = field >= self.occupancy_threshold
        if not np.any(occupied):
            self.known_boundary_points = np.empty((0, 2), dtype=float)
            self.known_boundary_initialized = False
            return self.known_boundary_points.copy()

        nx, ny = field.shape
        boundary_mask = np.zeros_like(occupied, dtype=bool)
        for ix in range(nx):
            for iy in range(ny):
                if not occupied[ix, iy]:
                    continue
                # check 8-neighbors for a free pixel
                has_free = False
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        x2 = ix + dx
                        y2 = iy + dy
                        if 0 <= x2 < nx and 0 <= y2 < ny:
                            if not occupied[x2, y2]:
                                has_free = True
                                break
                    if has_free:
                        break
                if has_free:
                    boundary_mask[ix, iy] = True

        pts = []
        for ix, iy in np.argwhere(boundary_mask):
            x = float(x_coords[ix]) if len(x_coords) > ix else float(ix)
            y = float(y_coords[iy]) if len(y_coords) > iy else float(iy)
            pts.append((x, y))

        pts = np.asarray(pts, dtype=float)
        self.known_boundary_points = pts
        self.known_boundary_initialized = pts.size > 0
        return self.known_boundary_points.copy()

    def build_boundary_grid(self, world_field, x_coords=None, y_coords=None):
        """Create a binary grid with contour pixels marked as 1.0.

        This maps known boundary points (extracted from `world_field` or
        previously-initialized `known_boundary_points`) back to a grid with
        the same indexing convention as the input `world_field`.
        """
        field = np.asarray(world_field, dtype=float)
        if field.size == 0:
            return np.zeros_like(field, dtype=float)

        # Ensure we have boundary points available for mapping.
        if self.known_boundary_points is None or self.known_boundary_points.size == 0:
            # try to initialize from the provided field
            try:
                self.initialize_known_boundary(field, x_coords=x_coords, y_coords=y_coords)
            except Exception:
                return np.zeros_like(field, dtype=float)

        contour_grid = np.zeros_like(field, dtype=float)
        points = np.asarray(self.known_boundary_points, dtype=float)
        if points.size == 0:
            return contour_grid

        if x_coords is None:
            x_coords = np.arange(field.shape[0], dtype=float)
        else:
            x_coords = np.asarray(x_coords, dtype=float)
        if y_coords is None:
            y_coords = np.arange(field.shape[1], dtype=float)
        else:
            y_coords = np.asarray(y_coords, dtype=float)

        for px, py in points:
            # find nearest grid indices
            ix = int(np.argmin(np.abs(x_coords - float(px)))) if x_coords.size > 0 else int(round(px))
            iy = int(np.argmin(np.abs(y_coords - float(py)))) if y_coords.size > 0 else int(round(py))
            if 0 <= ix < contour_grid.shape[0] and 0 <= iy < contour_grid.shape[1]:
                contour_grid[ix, iy] = 1.0

        return contour_grid

    def place_drones_on_boundary_random(self, drones, rng=None):
        """Place provided drone objects randomly along the known boundary.

        This routine mutates each drone by setting `x`, `y`, and a copy of the
        known boundary under `known_boundary_points` and a simple `known_positions` map.
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.known_boundary_points.size == 0:
            raise RuntimeError("known_boundary_points must be initialized first")

        n = self.known_boundary_points.shape[0]
        for i, drone in enumerate(drones):
            idx = int(rng.integers(0, n))
            x, y = self.known_boundary_points[idx]
            drone.x = float(x)
            drone.y = float(y)
            # Each drone receives the shared copy of the boundary (per user request)
            drone.known_boundary_points = self.known_boundary_points.copy()
            # Known positions map: each drone only knows itself by default.
            drone.known_positions = {getattr(drone, 'drone_id', i): np.array([drone.x, drone.y], dtype=float)}

    # ------------------------------------------------------------------
    # CONTROL STUBS (intentionally left for user re-implementation)
    # ------------------------------------------------------------------

    def _boundary_tracking_action(self, *args, **kwargs):
        """TODO: Implement boundary tracking control.

        This method intentionally contains no control logic. Implement the
        desired behavior in 2D here.
        """
        # TODO: implement boundary following control
        return np.zeros(2, dtype=float)

    def _equidistant_action(self, *args, **kwargs):
        """TODO: Implement equidistant (1D arc-index) control.

        The user requested that all movement/control logic be removed. Use this
        method to reintroduce the arc-index-based law when ready.
        """
        # TODO: implement arc-index equidistant control
        return np.zeros(2, dtype=float)

    # ------------------------------------------------------------------
    # PUBLIC INTERFACE: simplified compute_actions that only sets mode stubs
    # ------------------------------------------------------------------

    def compute_actions(self, drones, world_field=None, x_coords=None, y_coords=None):
        """Return a dictionary of zeroed actions and set drone modes to 'idle'.

        This keeps the rest of the system functional while control is implemented
        by the user.
        """
        actions = {}
        for drone in drones:
            # reflect the requested initial state: drones know the shape
            if not hasattr(drone, 'known_boundary_points'):
                drone.known_boundary_points = self.known_boundary_points.copy()
            if not hasattr(drone, 'known_positions'):
                drone.known_positions = {getattr(drone, 'drone_id', 0): np.array([drone.x, drone.y], dtype=float)}

            # set a clear mode indicating control is intentionally empty
            drone.last_control_mode = 'idle'
            actions[getattr(drone, 'drone_id', None)] = np.zeros(2, dtype=float)

        return actions

    # ------------------------------------------------------------------
    # Small utility helpers kept for compatibility
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(v):
        v = np.asarray(v, dtype=float)
        n = float(np.linalg.norm(v))
        if n <= 1e-12:
            return None
        return v / n

    def _clip_action(self, action, max_speed=0.12):
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            return np.zeros(2, dtype=float)
        s = float(np.linalg.norm(action))
        if s <= 1e-12:
            return np.zeros(2, dtype=float)
        if s > max_speed:
            return action * (max_speed / s)
        return action
