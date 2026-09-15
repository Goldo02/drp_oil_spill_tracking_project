import heapq

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
    # 1D VORONOI / MULTI-SOURCE SHORTEST PATH
    # ------------------------------------------------------------------

    @staticmethod
    def multi_source_shortest_path_voronoi(points, seeds, is_closed):
        """Assign every boundary point to the closest seed along the boundary.

        Parameters
        ----------
        points : array-like, shape (N, 2)
            Ordered boundary coordinates.
        seeds : iterable of dict
            Each seed must expose ``robot_id`` and ``index``.
        is_closed : bool
            If True, connect the last boundary point back to the first one.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``owner`` stores the owning robot ID for every boundary index.
            ``distances`` stores the arc distance from the owning seed.
        """
        boundary = np.asarray(points, dtype=float)
        if boundary.ndim != 2 or boundary.shape[1] != 2:
            raise ValueError("points must be an (N, 2) array")

        n_points = int(boundary.shape[0])
        distances = np.full(n_points, np.inf, dtype=float)
        owner = np.empty(n_points, dtype=object)
        owner[:] = None

        if n_points == 0:
            return owner, distances

        pq = []
        for order, seed in enumerate(seeds):
            if seed is None:
                continue

            robot_id = seed.get("robot_id") if isinstance(seed, dict) else getattr(seed, "robot_id", None)
            index = seed.get("index") if isinstance(seed, dict) else getattr(seed, "index", None)
            if robot_id is None or index is None:
                continue

            index = int(index)
            if not 0 <= index < n_points:
                continue

            if 0.0 < distances[index]:
                distances[index] = 0.0
                owner[index] = robot_id
                heapq.heappush(pq, (0.0, index, order, robot_id))

        while pq:
            current_dist, u, seed_order, robot_id = heapq.heappop(pq)
            if current_dist > distances[u]:
                continue

            if is_closed:
                neighbors = ((u - 1) % n_points, (u + 1) % n_points)
            elif u == 0:
                neighbors = (1,) if n_points > 1 else ()
            elif u == n_points - 1:
                neighbors = (u - 1,)
            else:
                neighbors = (u - 1, u + 1)

            for v in neighbors:
                weight = float(np.linalg.norm(boundary[u] - boundary[v]))
                new_dist = current_dist + weight
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    owner[v] = robot_id
                    heapq.heappush(pq, (new_dist, v, seed_order, robot_id))

        return owner, distances

    @staticmethod
    def _nearest_boundary_index(boundary_points, position):
        boundary = np.asarray(boundary_points, dtype=float)
        pos = np.asarray(position, dtype=float).reshape(1, 2)
        dists = np.linalg.norm(boundary - pos, axis=1)
        return int(np.argmin(dists))

    @staticmethod
    def _cell_target_index(boundary_points, indices, seed_index, is_closed):
        """Return a boundary index near the 1D arc midpoint of a Voronoi cell."""
        indices = np.asarray(indices, dtype=int)
        if indices.size == 0:
            return int(seed_index)

        boundary = np.asarray(boundary_points, dtype=float)
        if indices.size == 1:
            return int(indices[0])

        if is_closed:
            n_points = int(boundary.shape[0])
            offsets = (
                (indices - int(seed_index) + n_points / 2.0)
                % n_points
                - n_points / 2.0
            )
            ordered = indices[np.argsort(offsets)]
        else:
            ordered = np.sort(indices)

        cumulative = np.zeros(ordered.size, dtype=float)
        for idx in range(1, ordered.size):
            previous_idx = int(ordered[idx - 1])
            current_idx = int(ordered[idx])
            cumulative[idx] = cumulative[idx - 1] + float(
                np.linalg.norm(boundary[current_idx] - boundary[previous_idx])
            )

        midpoint = 0.5 * cumulative[-1]
        target_pos = int(np.searchsorted(cumulative, midpoint, side="left"))
        return int(ordered[min(target_pos, ordered.size - 1)])

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
            self.known_boundary_closed = bool(force_closed) or (
                pts.shape[0] > 1 and np.linalg.norm(pts[0] - pts[-1]) < 1e-6
            )
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
    # BOUNDARY ORDERING / CIRCULAR HELPERS
    # ------------------------------------------------------------------

    def _ensure_ordered_closed_boundary(self):
        """Ensure `self.known_boundary_points` is an ordered closed loop.

        This performs a greedy nearest-neighbour trace to order boundary
        points so that indices 0..N-1 follow the contour sequentially and
        the last point is adjacent to the first. If ordering cannot be
        improved (e.g. few points), the array is left as-is.
        """
        pts = np.asarray(self.known_boundary_points, dtype=float)
        n = pts.shape[0]
        if n <= 2:
            return

        # Compute nearest-neighbour distances to estimate typical spacing
        # (exclude self-distance)
        dists = np.full((n,), np.inf, dtype=float)
        for i in range(n):
            diff = pts - pts[i:i+1]
            dist2 = np.einsum('ij,ij->i', diff, diff)
            dist2[i] = np.inf
            dists[i] = float(np.sqrt(np.min(dist2)))

        median_nn = float(np.median(dists)) if np.isfinite(dists).all() else 0.0
        if median_nn <= 0:
            return

        # Greedy nearest-neighbour ordering
        visited = np.zeros(n, dtype=bool)
        order = [0]
        visited[0] = True
        for _ in range(1, n):
            cur = order[-1]
            # distances to unvisited
            unvisited_idx = np.nonzero(~visited)[0]
            diffs = pts[unvisited_idx] - pts[cur:cur+1]
            dd = np.einsum('ij,ij->i', diffs, diffs)
            nearest_pos = int(np.argmin(dd))
            nearest = unvisited_idx[nearest_pos]
            order.append(nearest)
            visited[nearest] = True

        ordered = pts[order]
        # Check closedness (last neighboring first) using threshold relative to median
        last_to_first = float(np.linalg.norm(ordered[0] - ordered[-1]))
        if last_to_first <= 3.0 * median_nn:
            self.known_boundary_points = ordered.copy()
            self.known_boundary_closed = True
        else:
            # if not closed, keep original but mark closedness conservatively
            self.known_boundary_points = pts.copy()
            self.known_boundary_closed = False

    def _mod_index(self, idx, n):
        return int(idx % n)

    def _circular_signed_distance(self, a, b, n):
        """Return signed shortest distance from a to b on a circular index set of length n.

        Value in range (-n/2, n/2].
        """
        diff = (b - a) % n
        if diff > n / 2.0:
            diff -= n
        return diff

    def _circular_midpoint(self, a, b, n):
        """Return the midpoint index (float) between indices a and b along the shortest arc."""
        sd = self._circular_signed_distance(a, b, n)
        return (a + sd / 2.0) % n

    def _indices_in_arc(self, start, end, n):
        """Return integer indices j in 0..n-1 that lie in the closed arc [start, end]

        Arc is taken in the positive modular direction from `start` to `end`.
        Both `start` and `end` may be floats; inclusion is decided by modular
        arithmetic comparing (j - start) % n to arc_length=(end - start) % n.
        """
        arc_len = (end - start) % n
        if arc_len == 0:
            # full circle -> all indices
            return np.arange(n, dtype=int)

        indices = []
        for j in range(n):
            rel = (j - start) % n
            if rel <= arc_len + 1e-9:
                indices.append(j)
        return np.array(indices, dtype=int)

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
        Compute MSSP 1D Voronoi target centroids for all drones.
        """
        for drone in drones:
            self.compute_ring_ordering(drone, drones)

    def compute_ring_ordering(self, current_drone, drones):
        """Compute 1D Voronoi partitioning with Multi-Source Dijkstra.

        Returns a dict compatible with tests: keys `N`, `current`, `succ`, `pred`, `center_of_mass`, `occupied_points`, `assigned_drone_indices`, `ring`.
        """
        if self.known_boundary_points is None or len(self.known_boundary_points) == 0:
            grid = getattr(current_drone, 'grid', None)
            if grid is not None and np.asarray(grid).size > 0:
                try:
                    x_min, x_max, y_min, y_max = current_drone.grid_bounds
                    Nx, Ny = current_drone.grid_shape
                    x_coords = np.linspace(x_min, x_max, Nx)
                    y_coords = np.linspace(y_min, y_max, Ny)
                    self.initialize_known_boundary(grid, x_coords=x_coords, y_coords=y_coords)
                except Exception:
                    pass

        if self.known_boundary_points is None or len(self.known_boundary_points) == 0:
            return None

        # Ensure strict sequential ordering
        self._ensure_ordered_closed_boundary()
        occupied_points = np.asarray(self.known_boundary_points, dtype=float)
        n_pts = int(occupied_points.shape[0])
        if n_pts == 0:
            return None

        known = getattr(current_drone, 'known_positions', None)
        if known is None or len(known) == 0:
            known = {getattr(d, 'drone_id', 0): np.array([d.x, d.y], dtype=float) for d in drones}

        seeds = []
        for drone_id, position in known.items():
            seed_idx = self._nearest_boundary_index(occupied_points, position)
            seeds.append({'robot_id': drone_id, 'index': seed_idx})

        seeds.sort(key=lambda seed: int(seed['index']))
        ordered_ids = [seed['robot_id'] for seed in seeds]

        M = len(ordered_ids)
        if M == 0:
            return None

        is_closed = bool(self.known_boundary_closed)
        assigned, distances = self.multi_source_shortest_path_voronoi(
            occupied_points,
            seeds,
            is_closed=is_closed,
        )

        ring = []
        seed_by_id = {seed['robot_id']: int(seed['index']) for seed in seeds}
        for did in ordered_ids:
            mask = np.array([a == did for a in assigned])
            indices = np.flatnonzero(mask)
            cell_pts = occupied_points[indices] if indices.size else np.empty((0, 2), dtype=float)
            vor_size = int(indices.size)
            if vor_size > 0:
                target_chain_index = self._cell_target_index(
                    occupied_points,
                    indices,
                    seed_by_id[did],
                    is_closed,
                )
                target_centroid = occupied_points[target_chain_index]
            else:
                target_centroid = occupied_points[seed_by_id[did]]
                target_chain_index = seed_by_id[did]

            ring.append({
                'drone_id': did,
                'seed_index': seed_by_id[did],
                'target_centroid': target_centroid,
                'target_chain_index': target_chain_index,
                'indices': indices,
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
            ordered_ids.append(cur_id)

        idx_in_order = ordered_ids.index(cur_id)
        pred_idx = (idx_in_order - 1) % len(ordered_ids)
        succ_idx = (idx_in_order + 1) % len(ordered_ids)

        # Compute center of mass for reference/logging purposes only (does not affect ordering)
        com = np.mean(occupied_points, axis=0)
        cur_pos = np.array([current_drone.x, current_drone.y], dtype=float)
        cur_angle = float(np.arctan2(cur_pos[1] - com[1], cur_pos[0] - com[0]))

        res = {
            'N': len(ordered_ids),
            'occupied_points': occupied_points,
            'assigned_drone_indices': assigned,
            'distances': distances,
            'seeds': seeds,
            'is_closed': is_closed,
            'ring': ring,
            'current': {'drone_id': cur_id, 'angle': cur_angle},
            'pred': {'drone_id': ordered_ids[pred_idx]},
            'succ': {'drone_id': ordered_ids[succ_idx]},
            'center_of_mass': com,
        }

        for entry in ring:
            if entry['drone_id'] == cur_id:
                res['current'].update({
                    'seed_index': entry['seed_index'],
                    'target_centroid': entry['target_centroid'],
                    'target_chain_index': entry['target_chain_index'],
                    'voronoi_cell_size': entry['voronoi_cell_size'],
                })
                res['current_idx'] = entry['drone_id']
                break

        current_drone.last_ring_info = res

        return res


    def compute_actions(self, drones, world_field=None, x_coords=None, y_coords=None):
        """Update communication/Voronoi diagnostics while keeping robots fixed."""
        actions = {}
        
        # Ensure initial attributes exist on all drones
        for drone in drones:
            drone.known_boundary_points = self.known_boundary_points.copy()
            if not hasattr(drone, 'known_positions'):
                drone.known_positions = {getattr(drone, 'drone_id', 0): np.array([drone.x, drone.y], dtype=float)}

        # 1. Update communication network and propagate positions via multi-hop
        self._update_multihop_positions(drones)

        # 2. Compute the 1D Voronoi target centroid for each drone
        self._compute_voronoi_target(drones)
        for drone in drones:
            drone.known_boundary_points = self.known_boundary_points.copy()

        # 3. Keep every robot static. The mode records what is being inspected,
        # while the zero action guarantees no physical movement is applied.
        for drone in drones:
            drone_id = getattr(drone, 'drone_id', None)

            if not hasattr(drone, 'known_positions') or drone.known_positions is None:
                drone.known_positions = {drone_id: np.array([drone.x, drone.y], dtype=float)}

            drone.last_control_mode = 'voronoi_static'
            drone.last_control_vector = np.zeros(2, dtype=float)
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
        """Temporarily disabled: robots must remain static."""
        return np.zeros(2, dtype=float)

    def _equidistant_action(self, drone, ring_info, world_field, x_coords, y_coords):
        """Temporarily disabled: robots must remain static."""
        return np.zeros(2, dtype=float)

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
