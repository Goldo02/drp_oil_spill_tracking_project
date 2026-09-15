import heapq

import numpy as np


class Controller:
    """
    Controller for distributed 1D Voronoi/Lloyd coverage on a known boundary.

    The controller keeps the existing multi-hop position exchange, partitions
    the ordered boundary with Multi-Source Dijkstra, and moves each robot
    toward the Lloyd target of its current 1D Voronoi cell.
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
        self.constrain_to_boundary = bool(kwargs.get("constrain_to_boundary", True))

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
    def _boundary_arc_lengths(boundary_points, is_closed=False):
        """Return cumulative arc-length coordinates for ordered boundary points."""
        boundary = np.asarray(boundary_points, dtype=float)
        n_points = int(boundary.shape[0])
        arc_lengths = np.zeros(n_points, dtype=float)
        if n_points <= 1:
            return arc_lengths, 0.0

        segment_lengths = np.linalg.norm(np.diff(boundary, axis=0), axis=1)
        arc_lengths[1:] = np.cumsum(segment_lengths)
        total_length = float(arc_lengths[-1])
        if is_closed:
            total_length += float(np.linalg.norm(boundary[0] - boundary[-1]))
        return arc_lengths, total_length

    @staticmethod
    def _owned_arc_length(boundary_points, owner, robot_id, is_closed):
        """Approximate physical length of a robot's Voronoi cell on the boundary."""
        boundary = np.asarray(boundary_points, dtype=float)
        owner = np.asarray(owner, dtype=object)
        n_points = int(boundary.shape[0])
        if n_points <= 1:
            return 0.0

        total = 0.0
        for idx in range(n_points - 1):
            if owner[idx] == robot_id and owner[idx + 1] == robot_id:
                total += float(np.linalg.norm(boundary[idx + 1] - boundary[idx]))

        if is_closed and owner[0] == robot_id and owner[-1] == robot_id:
            total += float(np.linalg.norm(boundary[0] - boundary[-1]))

        return total

    @staticmethod
    def _point_at_arc_length(boundary_points, arc_lengths, arc_length, total_length, is_closed):
        """Interpolate a point on the ordered boundary at a given arc-length coordinate."""
        boundary = np.asarray(boundary_points, dtype=float)
        arc_lengths = np.asarray(arc_lengths, dtype=float)
        n_points = int(boundary.shape[0])
        if n_points == 0:
            return np.zeros(2, dtype=float)
        if n_points == 1 or total_length <= 1e-12:
            return boundary[0].copy()

        s = float(arc_length)
        if is_closed:
            s = s % float(total_length)
        else:
            s = float(np.clip(s, 0.0, float(total_length)))

        if s <= arc_lengths[0]:
            return boundary[0].copy()

        if s >= arc_lengths[-1]:
            if is_closed:
                closing_len = float(np.linalg.norm(boundary[0] - boundary[-1]))
                if closing_len <= 1e-12:
                    return boundary[0].copy()
                t = float(np.clip((s - arc_lengths[-1]) / closing_len, 0.0, 1.0))
                return (1.0 - t) * boundary[-1] + t * boundary[0]
            return boundary[-1].copy()

        left_idx = int(np.searchsorted(arc_lengths, s, side="right") - 1)
        right_idx = min(left_idx + 1, n_points - 1)
        seg_len = float(arc_lengths[right_idx] - arc_lengths[left_idx])
        if seg_len <= 1e-12:
            return boundary[left_idx].copy()

        t = float((s - arc_lengths[left_idx]) / seg_len)
        return (1.0 - t) * boundary[left_idx] + t * boundary[right_idx]

    @staticmethod
    def _arc_length_at_position(boundary_points, arc_lengths, position, total_length, is_closed):
        """Project a point onto the boundary and return its arc coordinate and position."""
        boundary = np.asarray(boundary_points, dtype=float)
        arc_lengths = np.asarray(arc_lengths, dtype=float)
        pos = np.asarray(position, dtype=float)
        n_points = int(boundary.shape[0])
        if n_points == 0:
            return 0.0, np.zeros(2, dtype=float), 0
        if n_points == 1:
            return 0.0, boundary[0].copy(), 0

        best_dist2 = np.inf
        best_s = 0.0
        best_point = boundary[0].copy()
        best_index = 0
        segment_count = n_points if is_closed else n_points - 1

        for idx in range(segment_count):
            next_idx = (idx + 1) % n_points
            start = boundary[idx]
            end = boundary[next_idx]
            vec = end - start
            seg_len2 = float(np.dot(vec, vec))
            if seg_len2 <= 1e-12:
                t = 0.0
            else:
                t = float(np.clip(np.dot(pos - start, vec) / seg_len2, 0.0, 1.0))
            projected = start + t * vec
            dist2 = float(np.dot(pos - projected, pos - projected))
            if dist2 < best_dist2:
                seg_len = float(np.sqrt(seg_len2))
                s = float(arc_lengths[idx] + t * seg_len)
                if is_closed and total_length > 1e-12:
                    s = s % float(total_length)
                best_dist2 = dist2
                best_s = s
                best_point = projected
                best_index = idx if t < 0.5 else next_idx

        return best_s, best_point, int(best_index)

    @staticmethod
    def _lloyd_targets_from_seed_arcs(seeds, total_length, is_closed):
        """Return Lloyd cell centers in arc-length coordinates for a 1D boundary."""
        if not seeds or total_length <= 1e-12:
            return {}

        ordered = sorted(seeds, key=lambda seed: float(seed["arc_length"]))
        targets = {}
        if len(ordered) == 1:
            only = ordered[0]
            targets[only["robot_id"]] = {
                "target_arc_length": float(only["arc_length"]),
                "cell_arc_length": float(total_length),
                "cell_start_arc_length": 0.0,
                "cell_end_arc_length": float(total_length),
            }
            return targets

        if is_closed:
            length = float(total_length)
            count = len(ordered)
            for idx, seed in enumerate(ordered):
                prev_s = float(ordered[idx - 1]["arc_length"])
                curr_s = float(seed["arc_length"])
                next_s = float(ordered[(idx + 1) % count]["arc_length"])
                left_gap = (curr_s - prev_s) % length
                right_gap = (next_s - curr_s) % length
                cell_len = 0.5 * (left_gap + right_gap)
                cell_start = (curr_s - 0.5 * left_gap) % length
                cell_end = (curr_s + 0.5 * right_gap) % length
                target_s = (cell_start + 0.5 * cell_len) % length
                targets[seed["robot_id"]] = {
                    "target_arc_length": float(target_s),
                    "cell_arc_length": float(cell_len),
                    "cell_start_arc_length": float(cell_start),
                    "cell_end_arc_length": float(cell_end),
                }
            return targets

        boundaries = [0.0]
        for left, right in zip(ordered[:-1], ordered[1:]):
            boundaries.append(0.5 * (float(left["arc_length"]) + float(right["arc_length"])))
        boundaries.append(float(total_length))

        for idx, seed in enumerate(ordered):
            left = boundaries[idx]
            right = boundaries[idx + 1]
            targets[seed["robot_id"]] = {
                "target_arc_length": 0.5 * (left + right),
                "cell_arc_length": float(max(0.0, right - left)),
                "cell_start_arc_length": float(left),
                "cell_end_arc_length": float(right),
            }
        return targets

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

    @staticmethod
    def _arc_indices_between(start_index, target_index, n_points, is_closed):
        """Return boundary indices from start to target along the shortest arc."""
        start_index = int(start_index)
        target_index = int(target_index)
        if start_index == target_index:
            return np.array([start_index], dtype=int)

        if not is_closed:
            step = 1 if target_index > start_index else -1
            return np.arange(start_index, target_index + step, step, dtype=int)

        forward_steps = (target_index - start_index) % n_points
        backward_steps = (start_index - target_index) % n_points
        if forward_steps <= backward_steps:
            return np.array(
                [(start_index + offset) % n_points for offset in range(forward_steps + 1)],
                dtype=int,
            )
        return np.array(
            [(start_index - offset) % n_points for offset in range(backward_steps + 1)],
            dtype=int,
        )

    @classmethod
    def _next_boundary_index_toward(cls, boundary_points, start_index, target_index, is_closed, max_step):
        """Choose the farthest boundary index reachable within one control step."""
        boundary = np.asarray(boundary_points, dtype=float)
        path = cls._arc_indices_between(
            start_index,
            target_index,
            int(boundary.shape[0]),
            is_closed,
        )
        if path.size <= 1:
            return int(path[0])

        traveled = 0.0
        chosen = int(path[1])
        for idx in range(1, path.size):
            previous_idx = int(path[idx - 1])
            current_idx = int(path[idx])
            edge_length = float(np.linalg.norm(boundary[current_idx] - boundary[previous_idx]))
            if traveled + edge_length > max_step:
                break
            traveled += edge_length
            chosen = current_idx

        return int(chosen)

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
    # PUBLIC INTERFACE
    # ------------------------------------------------------------------

    def _update_multihop_positions(self, drones):
        """
        Private helper to simulate multi-hop communication between drones.
        
        It checks the Euclidean distance between drones against `communication_radius`.
        If two drones are within range, they share and merge their `known_positions` 
        dictionaries, propagating information across the network (multi-hop).
        """
        # 1. Start each communication cycle from current measurements only.
        # Without timestamps, keeping old dictionaries can reintroduce stale
        # positions through neighbors and corrupt the 1D geodesic partition.
        current_positions = {
            getattr(drone, 'drone_id', 0): np.array([drone.x, drone.y], dtype=float)
            for drone in drones
        }
        for drone in drones:
            drone_id = getattr(drone, 'drone_id', 0)
            drone.known_positions = {
                drone_id: current_positions[drone_id].copy(),
            }

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
                    drone_id = getattr(drone, 'drone_id', 0)
                    for known_id, position in pending_updates[i].items():
                        if known_id == drone_id:
                            continue
                        drone.known_positions[known_id] = np.asarray(
                            position,
                            dtype=float,
                        )
                    drone.known_positions[drone_id] = np.array(
                        [drone.x, drone.y],
                        dtype=float,
                    )

    def _compute_voronoi_target(self, drones):
        """
        Compute MSSP 1D Voronoi target centroids for all drones.
        """
        for drone in drones:
            self.compute_ring_ordering(drone, drones)

    def project_drone_to_boundary(self, drone):
        """Snap a drone state to the nearest point of the known boundary."""
        if (
            not self.constrain_to_boundary
            or self.known_boundary_points is None
            or len(self.known_boundary_points) == 0
        ):
            return

        boundary = np.asarray(self.known_boundary_points, dtype=float)
        self._ensure_ordered_closed_boundary()
        boundary = np.asarray(self.known_boundary_points, dtype=float)
        is_closed = bool(self.known_boundary_closed)
        arc_lengths, total_length = self._boundary_arc_lengths(
            boundary,
            is_closed=is_closed,
        )
        boundary_s, projected, nearest_idx = self._arc_length_at_position(
            boundary,
            arc_lengths,
            np.array([drone.x, drone.y], dtype=float),
            total_length,
            is_closed,
        )
        drone.x = float(projected[0])
        drone.y = float(projected[1])
        drone.known_positions[drone.drone_id] = np.array(
            [drone.x, drone.y],
            dtype=float,
        )
        drone.boundary_index = int(nearest_idx)
        drone.boundary_s = float(boundary_s)

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

        is_closed = bool(self.known_boundary_closed)
        arc_lengths, total_boundary_length = self._boundary_arc_lengths(
            occupied_points,
            is_closed=is_closed,
        )

        seeds = []
        for drone_id, position in known.items():
            seed_s, projected, seed_idx = self._arc_length_at_position(
                occupied_points,
                arc_lengths,
                position,
                total_boundary_length,
                is_closed,
            )
            seeds.append({
                'robot_id': drone_id,
                'index': seed_idx,
                'arc_length': seed_s,
                'position_on_boundary': projected,
            })

        seeds.sort(key=lambda seed: float(seed['arc_length']))
        ordered_ids = [seed['robot_id'] for seed in seeds]

        M = len(ordered_ids)
        if M == 0:
            return None

        assigned, distances = self.multi_source_shortest_path_voronoi(
            occupied_points,
            seeds,
            is_closed=is_closed,
        )
        lloyd_targets = self._lloyd_targets_from_seed_arcs(
            seeds,
            total_boundary_length,
            is_closed,
        )

        ring = []
        seed_by_id = {seed['robot_id']: int(seed['index']) for seed in seeds}
        seed_arc_by_id = {seed['robot_id']: float(seed['arc_length']) for seed in seeds}
        for did in ordered_ids:
            mask = np.array([a == did for a in assigned])
            indices = np.flatnonzero(mask)
            vor_size = int(indices.size)
            target_data = lloyd_targets.get(did, {})
            target_arc_length = float(target_data.get('target_arc_length', seed_arc_by_id[did]))
            cell_start_arc_length = float(
                target_data.get('cell_start_arc_length', target_arc_length)
            )
            cell_end_arc_length = float(
                target_data.get('cell_end_arc_length', target_arc_length)
            )
            target_centroid = self._point_at_arc_length(
                occupied_points,
                arc_lengths,
                target_arc_length,
                total_boundary_length,
                is_closed,
            )
            target_chain_index = self._nearest_boundary_index(occupied_points, target_centroid)
            cell_arc_length = float(
                target_data.get(
                    'cell_arc_length',
                    self._owned_arc_length(occupied_points, assigned, did, is_closed),
                )
            )

            ring.append({
                'drone_id': did,
                'seed_index': seed_by_id[did],
                'seed_arc_length': seed_arc_by_id[did],
                'target_centroid': target_centroid,
                'target_chain_index': target_chain_index,
                'target_arc_length': target_arc_length,
                'cell_start_arc_length': cell_start_arc_length,
                'cell_end_arc_length': cell_end_arc_length,
                'indices': indices,
                'voronoi_cell_size': vor_size,
                'cell_arc_length': cell_arc_length,
            })

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
            'arc_lengths': arc_lengths,
            'total_boundary_length': total_boundary_length,
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
                    'seed_arc_length': entry['seed_arc_length'],
                    'target_centroid': entry['target_centroid'],
                    'target_chain_index': entry['target_chain_index'],
                    'target_arc_length': entry['target_arc_length'],
                    'cell_start_arc_length': entry['cell_start_arc_length'],
                    'cell_end_arc_length': entry['cell_end_arc_length'],
                    'voronoi_cell_size': entry['voronoi_cell_size'],
                    'cell_arc_length': entry['cell_arc_length'],
                })
                res['current_idx'] = entry['drone_id']
                current_drone.target_centroid = np.asarray(
                    entry['target_centroid'],
                    dtype=float,
                )
                current_drone.boundary_s = float(entry['seed_arc_length'])
                current_drone.boundary_index = int(entry['seed_index'])
                break

        current_drone.last_ring_info = res

        return res


    def compute_actions(self, drones, world_field=None, x_coords=None, y_coords=None):
        """Run one distributed Lloyd iteration on the 1D boundary."""
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

        # 3. Move each robot toward its current Lloyd target.
        for drone in drones:
            drone_id = getattr(drone, 'drone_id', None)

            if not hasattr(drone, 'known_positions') or drone.known_positions is None:
                drone.known_positions = {drone_id: np.array([drone.x, drone.y], dtype=float)}

            drone.last_control_mode = 'lloyd'
            action = self._equidistant_action(
                drone,
                getattr(drone, 'last_ring_info', None),
                world_field,
                x_coords,
                y_coords,
            )
            actions[drone_id] = self._clip_action(
                action,
                max_speed=getattr(drone, 'max_speed', 0.12),
            )

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
        """Boundary tracking is not used while testing 1D Lloyd coverage."""
        return np.zeros(2, dtype=float)

    def _equidistant_action(self, drone, ring_info, world_field, x_coords, y_coords):
        """Move the drone toward the current Lloyd target along the boundary arc."""
        target = None
        target_arc = None
        boundary = None
        arc_lengths = None
        total_length = None
        is_closed = None
        if isinstance(ring_info, dict):
            current = ring_info.get('current', {})
            target = current.get('target_centroid')
            target_arc = current.get('target_arc_length')
            boundary = ring_info.get('occupied_points')
            arc_lengths = ring_info.get('arc_lengths')
            total_length = ring_info.get('total_boundary_length')
            is_closed = bool(ring_info.get('is_closed', self.known_boundary_closed))
        if target is None:
            target = getattr(drone, 'target_centroid', None)
        if target is None:
            return np.zeros(2, dtype=float)

        max_speed = float(getattr(drone, 'max_speed', 0.12))
        current_pos = np.array([drone.x, drone.y], dtype=float)

        if (
            target_arc is not None
            and boundary is not None
            and arc_lengths is not None
            and total_length is not None
            and float(total_length) > 1e-12
        ):
            boundary = np.asarray(boundary, dtype=float)
            arc_lengths = np.asarray(arc_lengths, dtype=float)
            current_arc = current.get('seed_arc_length')
            if current_arc is None:
                current_arc = getattr(drone, 'boundary_s', None)
            if current_arc is None:
                current_arc, _, _ = self._arc_length_at_position(
                    boundary,
                    arc_lengths,
                    current_pos,
                    float(total_length),
                    bool(is_closed),
                )

            if bool(is_closed):
                signed_error = (
                    (float(target_arc) - float(current_arc) + 0.5 * float(total_length))
                    % float(total_length)
                    - 0.5 * float(total_length)
                )
            else:
                signed_error = float(target_arc) - float(current_arc)

            step = float(np.clip(
                float(self.k_t) * signed_error,
                -max_speed,
                max_speed,
            ))
            if abs(step) <= 1e-12:
                return np.zeros(2, dtype=float)

            next_arc = float(current_arc) + step
            next_point = self._point_at_arc_length(
                boundary,
                arc_lengths,
                next_arc,
                float(total_length),
                bool(is_closed),
            )
            return self._clip_action(next_point - current_pos, max_speed=max_speed)

        action = float(self.k_t) * (np.asarray(target, dtype=float) - current_pos)
        return self._clip_action(action, max_speed=max_speed)

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
