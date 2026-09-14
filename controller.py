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
        k_t=1.0,
        k_spacing=1.0,
        settling_steps=0,
        d_safe=0.5,
        repulsion_gain=0.0,
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
        # whether the known boundary is closed (found as a loop)
        self.known_boundary_closed = False

        # Equidistant controller parameters
        self.k_t = float(k_t)
        self.k_spacing = float(k_spacing)
        self.settling_steps = int(settling_steps)
        # collision avoidance params
        self.d_safe = float(d_safe)
        self.repulsion_gain = float(repulsion_gain)

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
        # If initialization came from an occupancy grid, attempt to mark closedness
        try:
            # If user passed a 2D grid, we estimated pts from it above; set closed flag
            if hasattr(world_field_or_points, 'ndim') and getattr(world_field_or_points, 'ndim', 1) == 2:
                self.known_boundary_closed = self.is_polygon_closed(np.asarray(world_field_or_points, dtype=float))
            else:
                # if points are provided explicitly, consider closed if first and last are near
                if pts.size and np.linalg.norm(pts[0] - pts[-1]) < 1e-6:
                    self.known_boundary_closed = True
                else:
                    self.known_boundary_closed = False
        except Exception:
            self.known_boundary_closed = False
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

    # ------------------------------------------------------------------
    # EXTRA PUBLIC HELPERS / CONSENSUS
    # ------------------------------------------------------------------

    def consensus_step(self, drones):
        """Optional consensus step. Default baseline: do not merge occupancy grids.

        For compatibility with tests, this method intentionally does not
        combine per-robot occupancy grids when `fully_connected` is False.
        It will ensure each drone has a `known_positions` mapping.
        """
        for drone in drones:
            if not hasattr(drone, 'known_positions') or drone.known_positions is None:
                drone.known_positions = {getattr(drone, 'drone_id', 0): np.array([drone.x, drone.y], dtype=float)}
        # no grid merging in this baseline
        return

    def is_polygon_closed(self, grid):
        """Return True if the binary `grid` contains a closed contour enclosing area.

        Approach: treat occupied cells as walls, flood-fill from the grid border
        over free cells; if any free cell is not reachable from the border then
        the wall encloses an interior -> closed polygon.
        """
        grid = np.asarray(grid, dtype=float)
        if grid.size == 0:
            return False

        occupied = grid >= self.occupancy_threshold
        if not np.any(occupied):
            return False

        nx, ny = occupied.shape
        visited = np.zeros_like(occupied, dtype=bool)

        from collections import deque

        q = deque()
        # enqueue all boundary free cells
        for ix in range(nx):
            for iy in (0, ny - 1):
                if not occupied[ix, iy] and not visited[ix, iy]:
                    visited[ix, iy] = True
                    q.append((ix, iy))
        for iy in range(ny):
            for ix in (0, nx - 1):
                if not occupied[ix, iy] and not visited[ix, iy]:
                    visited[ix, iy] = True
                    q.append((ix, iy))

        while q:
            x, y = q.popleft()
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx2 = x + dx
                ny2 = y + dy
                if 0 <= nx2 < nx and 0 <= ny2 < ny:
                    if not occupied[nx2, ny2] and not visited[nx2, ny2]:
                        visited[nx2, ny2] = True
                        q.append((nx2, ny2))

        # If there exists any free cell not visited, it's an interior -> closed
        interior_exists = np.any(~visited & ~occupied)
        return bool(interior_exists)

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

    def _update_multihop_positions(self, drones):
        """
        Private helper to simulate multi-hop communication between drones.
        
        It checks the Euclidean distance between drones against `communication_radius`.
        If two drones are within range, they share and merge their `known_positions` 
        dictionaries, propagating information across the network (multi-hop).
        """
        # 1. Ensure every drone registers its own current real position first
        for drone in drones:
            drone_id = getattr(drone, 'drone_id', 0)
            if not hasattr(drone, 'known_positions'):
                drone.known_positions = {}
            drone.known_positions[drone_id] = np.array([drone.x, drone.y], dtype=float)

        # 2. Multi-hop propagation loop (e.g., 2 iterations to let information travel across neighbors)
        for _ in range(5):
            # Create a list of dictionaries to accumulate updates for each drone in this round
            pending_updates = [{} for _ in drones]

            for i, drone_i in enumerate(drones):
                for j, drone_j in enumerate(drones):
                    if i == j:
                        continue
                    
                    # Compute distance between drone i and drone j
                    dist = np.linalg.norm(drone_i.position - drone_j.position)
                    
                    if dist <= self.communication_radius:
                        # If within communication range, drone_i learns what drone_j knows
                        pending_updates[i].update(drone_j.known_positions)

            # Apply the accumulated knowledge updates to each drone
            for i, drone in enumerate(drones):
                if pending_updates[i]:
                    drone.known_positions.update(pending_updates[i])

    def _compute_voronoi_target(self, drones):
        """
        Compute the 1D Arc-Index Voronoi target centroid for each drone 
        along the closed boundary, based on its `known_positions`.
        """
        if self.known_boundary_points is None or len(self.known_boundary_points) == 0:
            return

        n_boundary = len(self.known_boundary_points)
        boundary_pts = np.asarray(self.known_boundary_points, dtype=float)

        for drone in drones:
            # 1. Get the positions known by this specific drone (from multi-hop)
            known_pos = drone.known_positions

            # Step 1: Find the closest boundary index (s_i) for each known drone
            drone_indices = {}
            for d_id, pos in known_pos.items():
                distances = np.linalg.norm(boundary_pts - pos, axis=1)
                drone_indices[d_id] = int(np.argmin(distances))
            
            # Step 2: Sort drones along the ring based on their s index
            sorted_drones = sorted(drone_indices.items(), key=lambda item: item[1])
            drone_ids_order = [d_id for d_id, s in sorted_drones]

            my_id = getattr(drone, 'drone_id', 0)
            if my_id not in drone_ids_order:
                drone.target_centroid = np.array([drone.x, drone.y], dtype=float)
                continue

            my_rank = drone_ids_order.index(my_id)
            M = len(sorted_drones)

            # Edge case: if only one drone is known/active
            if M == 1:
                drone.target_centroid = boundary_pts[drone_indices[my_id]]
                continue

            # Step 3: Define Voronoi cell boundaries (midpoints between adjacent drones on the ring)
            prev_rank = (my_rank - 1) % M
            next_rank = (my_rank + 1) % M

            s_prev = sorted_drones[prev_rank][1]
            s_curr = sorted_drones[my_rank][1]
            s_next = sorted_drones[next_rank][1]

            # Handle wrap-around for the closed loop
            if s_next < s_prev:
                s_next += n_boundary
            if s_curr < s_prev:
                s_curr += n_boundary

            cell_start = (s_prev + s_curr) / 2.0
            cell_end = (s_curr + s_next) / 2.0

            # Step 4: Extract boundary points belonging to this cell and compute geometric centroid
            cell_indices = []
            s_idx = int(np.floor(cell_start))
            s_end_idx = int(np.ceil(cell_end))

            for idx in range(s_idx, s_end_idx + 1):
                cell_indices.append(idx % n_boundary)

            cell_points = boundary_pts[cell_indices]
            
            if len(cell_points) > 0:
                drone.target_centroid = np.mean(cell_points, axis=0)
            else:
                drone.target_centroid = boundary_pts[drone_indices[my_id]]

    def compute_ring_ordering(self, current_drone, drones):
        """Compute simple angular ordering around center-of-mass from drone.known_positions.

        Returns a dict compatible with tests: keys `N`, `current`, `succ`, `pred`, `center_of_mass`.
        """
        # Prefer to extract the ring from the current drone's local grid if available
        grid = getattr(current_drone, 'grid', None)
        occupied_points = []
        if grid is not None and np.asarray(grid).size > 0:
            g = np.asarray(grid, dtype=float)
            occ_idx = np.argwhere(g >= self.occupancy_threshold)
            if occ_idx.size > 0:
                # map indices to coordinates using current_drone.grid_bounds
                x_min, x_max, y_min, y_max = current_drone.grid_bounds
                Nx, Ny = current_drone.grid_shape
                x_coords = np.linspace(x_min, x_max, Nx)
                y_coords = np.linspace(y_min, y_max, Ny)
                pts = np.column_stack((x_coords[occ_idx[:, 0]], y_coords[occ_idx[:, 1]]))
                if pts.size > 0:
                    # order by angle around center-of-mass
                    com = np.mean(pts, axis=0)
                    angles = np.arctan2(pts[:, 1] - com[1], pts[:, 0] - com[0])
                    order = np.argsort(angles)
                    occupied_points = pts[order]
                else:
                    occupied_points = np.empty((0, 2), dtype=float)

        # fallback: if no occupied_points, try to use known_boundary_points
        if len(occupied_points) == 0 and hasattr(self, 'known_boundary_points') and self.known_boundary_points is not None and len(self.known_boundary_points) > 0:
            occupied_points = np.asarray(self.known_boundary_points, dtype=float)

        if len(occupied_points) == 0:
            return None

        # Build canonical assignment: find nearest ring index for each drone
        known = getattr(current_drone, 'known_positions', None)
        if known is None or len(known) == 0:
            known = {d.drone_id: np.array([d.x, d.y], dtype=float) for d in drones}

        drone_ids = list(known.keys())
        drone_pos = [np.asarray(known[i], dtype=float) for i in drone_ids]

        # For each drone, find nearest occupied_points index
        n_pts = len(occupied_points)
        drone_indices = []
        for p in drone_pos:
            dists = np.linalg.norm(occupied_points - p.reshape(1, 2), axis=1)
            drone_indices.append(int(np.argmin(dists)))

        # sort drones by their index along the ring to form ordering
        sorted_idx = np.argsort(drone_indices)
        ordered_ids = [drone_ids[i] for i in sorted_idx]

        # build ring entries with cell assignment per drone
        ring = []
        assigned = np.empty(n_pts, dtype=object)
        for j in range(n_pts):
            # pick nearest drone by circular distance
            diffs = np.abs((np.array(drone_indices, dtype=float) - float(j) + n_pts / 2.0) % n_pts - n_pts / 2.0)
            nearest = int(np.argmin(diffs))
            assigned[j] = drone_ids[nearest]

        # For each ordered drone, compute its cell indices and centroid
        for did in ordered_ids:
            mask = np.array([a == did for a in assigned])
            indices = np.flatnonzero(mask)
            cell_pts = occupied_points[indices] if indices.size else np.empty((0,2), dtype=float)
            vor_size = int(indices.size)
            if vor_size > 0:
                target_centroid = np.mean(cell_pts, axis=0)
            else:
                # fallback: nearest boundary point
                idx_nearest = drone_indices[drone_ids.index(did)]
                target_centroid = occupied_points[idx_nearest]

            ring.append({
                'drone_id': did,
                'target_centroid': target_centroid,
                'voronoi_cell_size': vor_size,
            })

        # Set target_centroid on drone objects where appropriate
        for entry in ring:
            for d in drones:
                if getattr(d, 'drone_id', None) == entry['drone_id']:
                    d.target_centroid = np.asarray(entry['target_centroid'], dtype=float)

        # Identify current/pred/succ for current_drone
        cur_id = getattr(current_drone, 'drone_id', None)
        if cur_id not in ordered_ids:
            # try to insert current into ordering based on its nearest index
            ordered_ids.append(cur_id)

        idx_in_order = ordered_ids.index(cur_id)
        pred_idx = (idx_in_order - 1) % len(ordered_ids)
        succ_idx = (idx_in_order + 1) % len(ordered_ids)

        res = {
            'N': len(ordered_ids),
            'occupied_points': occupied_points,
            'assigned_drone_indices': assigned,
            'ring': ring,
            'current': {'drone_id': cur_id, 'angle': float(np.arctan2(current_drone.y - np.mean(occupied_points[:,1]), current_drone.x - np.mean(occupied_points[:,0])))},
            'pred': {'drone_id': ordered_ids[pred_idx]},
            'succ': {'drone_id': ordered_ids[succ_idx]},
            'center_of_mass': np.mean(occupied_points, axis=0),
        }

        return res


    def compute_actions(self, drones, world_field=None, x_coords=None, y_coords=None):
        """Return a dictionary of zeroed actions and set drone modes to 'idle',
        after updating the multi-hop network and computing Voronoi targets.
        """
        actions = {}
        
        # Ensure initial attributes exist on all drones
        for drone in drones:
            if not hasattr(drone, 'known_boundary_points'):
                drone.known_boundary_points = self.known_boundary_points.copy()
            if not hasattr(drone, 'known_positions'):
                drone.known_positions = {getattr(drone, 'drone_id', 0): np.array([drone.x, drone.y], dtype=float)}

        # 1. Update communication network and propagate positions via multi-hop
        self._update_multihop_positions(drones)

        # 2. Compute the 1D Voronoi target centroid for each drone
        self._compute_voronoi_target(drones)

        # 3. Determine control mode per-drone and compute actions
        for drone in drones:
            drone_id = getattr(drone, 'drone_id', None)

            # ensure known_positions exists
            if not hasattr(drone, 'known_positions') or drone.known_positions is None:
                drone.known_positions = {drone_id: np.array([drone.x, drone.y], dtype=float)}


            # If the local drone grid contains a closed polygon -> equidistant mode
            grid = getattr(drone, 'grid', None)
            if grid is not None and np.asarray(grid).size > 0 and self.is_polygon_closed(grid):
                drone.last_control_mode = 'equi_distant'
                ring_info = self.compute_ring_ordering(drone, drones) or {}
                action = self._equidistant_action(drone, ring_info, world_field, x_coords, y_coords)
                actions[drone_id] = self._clip_action(action)
                continue

            # If the global world_field contains a closed polygon -> equidistant mode
            if world_field is not None and self.is_polygon_closed(world_field):
                # switching to equidistant
                drone.last_control_mode = 'equi_distant'
                ring_info = self.compute_ring_ordering(drone, drones) or {}
                action = self._equidistant_action(drone, ring_info, world_field, x_coords, y_coords)
                actions[drone_id] = self._clip_action(action)
                continue

            # If the local drone grid contains an occupied cell -> boundary tracking
            grid = getattr(drone, 'grid', None)
            if grid is not None and np.any(np.asarray(grid, dtype=float) >= self.occupancy_threshold):
                drone.last_control_mode = 'boundary_tracking'
                # compute centroid of occupied cells in drone.grid mapped to x_coords/y_coords
                try:
                    g = np.asarray(grid, dtype=float)
                    occupied_idx = np.argwhere(g >= self.occupancy_threshold)
                    if occupied_idx.size:
                        # use provided coords if available
                        if x_coords is None or y_coords is None:
                            # fallback to index space
                            target = np.mean(occupied_idx, axis=0)
                            tx, ty = float(target[0]), float(target[1])
                        else:
                            # occupied_idx are (ix,iy) mapping to x_coords[ix], y_coords[iy]
                            xs = np.asarray(x_coords, dtype=float)[occupied_idx[:, 0]]
                            ys = np.asarray(y_coords, dtype=float)[occupied_idx[:, 1]]
                            tx, ty = float(np.mean(xs)), float(np.mean(ys))
                        desired = np.array([tx, ty], dtype=float)
                        action = desired - np.array([drone.x, drone.y], dtype=float)
                        # if computed action is zero (e.g., drone already at centroid)
                        # provide a small exploratory/tangential push so robots don't freeze
                        if float(np.linalg.norm(action)) <= 1e-6:
                            explor = getattr(drone, 'exploration_direction', None)
                            if explor is None:
                                action = np.array([0.02, 0.0], dtype=float)
                            else:
                                explor = np.asarray(explor, dtype=float)
                                ne = explor / (np.linalg.norm(explor) + 1e-12)
                                action = 0.02 * ne

                        actions[drone_id] = self._clip_action(action)
                    else:
                        actions[drone_id] = np.zeros(2, dtype=float)
                except Exception:
                    actions[drone_id] = np.zeros(2, dtype=float)
                continue

            # Default idle
            drone.last_control_mode = 'idle'
            actions[drone_id] = np.zeros(2, dtype=float)

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

    def _boundary_tracking_action(self, *args, **kwargs):
        """Basic boundary tracking: move toward provided target centroid if present."""
        # legacy compatibility wrapper
        if len(args) >= 1:
            drone = args[0]
            tc = getattr(drone, 'target_centroid', None)
            if tc is not None:
                action = np.asarray(tc, dtype=float) - np.array([drone.x, drone.y], dtype=float)
                return self._clip_action(action)
        return np.zeros(2, dtype=float)

    def _equidistant_action(self, drone, ring_info, world_field, x_coords, y_coords):
        """Simple equidistant controller that moves toward the assigned target centroid
        and adds a small tangential component to avoid stalling.
        """
        # prefer ring_info current target_centroid if present
        tc = None
        if ring_info and isinstance(ring_info, dict):
            tc = ring_info.get('current', {}).get('target_centroid', None)
        if tc is None:
            tc = getattr(drone, 'target_centroid', None)
        if tc is None:
            return np.zeros(2, dtype=float)

        pos = np.array([drone.x, drone.y], dtype=float)
        desired = np.asarray(tc, dtype=float)
        primary = desired - pos

        # tangential push based on ring center if available
        center = np.array(ring_info.get('center_of_mass', [0.0, 0.0]), dtype=float) if ring_info else np.zeros(2, dtype=float)
        radial = pos - center
        tang = np.array([-radial[1], radial[0]], dtype=float)
        tang_norm = self._normalize(tang)

        if tang_norm is None:
            action = primary
        else:
            action = primary + 0.08 * tang_norm

        return self._clip_action(action)

    def _compute_repulsion(self, drone):
        """Compute simple inter-drone repulsion based on `d_safe` and `repulsion_gain`.

        Returns a 2D vector (possibly zero) pointing away from neighbors that are
        closer than `d_safe`.
        """
        kp = float(self.repulsion_gain)
        d_safe = float(getattr(self, 'd_safe', 0.5))
        if kp == 0.0:
            return np.zeros(2, dtype=float)

        known = getattr(drone, 'known_positions', None)
        if known is None:
            return np.zeros(2, dtype=float)

        mypos = np.array([drone.x, drone.y], dtype=float)
        total = np.zeros(2, dtype=float)
        for other_id, pos in known.items():
            if other_id == getattr(drone, 'drone_id', None):
                continue
            pos = np.asarray(pos, dtype=float)
            diff = mypos - pos
            dist = float(np.linalg.norm(diff))
            if dist <= 1e-12:
                # if overlapping, push in arbitrary direction
                total += kp * np.array([1.0, 0.0])
            elif dist < d_safe:
                # linear repulsion magnitude
                mag = kp * (d_safe - dist)
                total += (diff / dist) * mag

        return total
