import heapq

import numpy as np

try:
    import contourpy
except ImportError:  # pragma: no cover
    contourpy = None


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
        occupancy_threshold=0.5,
        k_t=1.0,
        **kwargs,
    ):
        self.sim_map = sim_map
        self.communication_radius = float(communication_radius)
        self.occupancy_threshold = float(occupancy_threshold)

        self.known_boundary_points = np.empty((0, 2), dtype=float)
        self.known_boundary_closed = False
        self.known_boundary_ordered = False

        self.k_t = float(k_t)
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
    def _estimated_position(drone):
        estimate = getattr(drone, "last_gps_position", None)
        if estimate is not None:
            estimate = np.asarray(estimate, dtype=float)
            if estimate.shape == (2,) and np.all(np.isfinite(estimate)):
                return estimate.copy()

        if hasattr(drone, "update_position_estimate"):
            estimate = np.asarray(drone.update_position_estimate(), dtype=float)
            if estimate.shape == (2,) and np.all(np.isfinite(estimate)):
                return estimate.copy()

        if hasattr(drone, "gps"):
            raise ValueError(
                "Instrumented drones must provide a finite GPS position estimate"
            )

        position = getattr(drone, "position", None)
        if position is not None:
            position = np.asarray(position, dtype=float)
            if position.shape == (2,) and np.all(np.isfinite(position)):
                return position.copy()

        return np.array([drone.x, drone.y], dtype=float)

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
            self.known_boundary_closed = bool(force_closed) or (
                pts.shape[0] > 1 and np.linalg.norm(pts[0] - pts[-1]) < 1e-6
            )
            self.known_boundary_ordered = False
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
            self.known_boundary_ordered = False
            return self.known_boundary_points.copy()

        pts = self._extract_ordered_contour(field, x_coords, y_coords)
        is_ordered_contour = pts.size > 0
        if pts.size == 0:
            pts = self._extract_boundary_mask_points(occupied, x_coords, y_coords)

        self.known_boundary_points = pts
        self.known_boundary_closed = bool(force_closed)
        self.known_boundary_ordered = is_ordered_contour
        return self.known_boundary_points.copy()

    def _extract_ordered_contour(self, field, x_coords, y_coords):
        if contourpy is None:
            return np.empty((0, 2), dtype=float)

        generator = contourpy.contour_generator(
            x=np.asarray(x_coords, dtype=float),
            y=np.asarray(y_coords, dtype=float),
            z=np.asarray(field, dtype=float).T,
            name="serial",
        )
        lines = generator.lines(float(self.occupancy_threshold))
        if not lines:
            return np.empty((0, 2), dtype=float)

        def path_length(line):
            if len(line) < 2:
                return 0.0
            return float(np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1)))

        contour = np.asarray(max(lines, key=path_length), dtype=float)
        if contour.ndim != 2 or contour.shape[1] != 2:
            return np.empty((0, 2), dtype=float)
        if len(contour) > 1 and np.linalg.norm(contour[0] - contour[-1]) < 1e-9:
            contour = contour[:-1]
        return contour

    @staticmethod
    def _extract_boundary_mask_points(occupied, x_coords, y_coords):
        nx, ny = occupied.shape
        boundary_mask = np.zeros_like(occupied, dtype=bool)
        for ix in range(nx):
            for iy in range(ny):
                if not occupied[ix, iy]:
                    continue
                has_free = False
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        x2 = ix + dx
                        y2 = iy + dy
                        if 0 <= x2 < nx and 0 <= y2 < ny and not occupied[x2, y2]:
                            has_free = True
                            break
                    if has_free:
                        break
                if has_free:
                    boundary_mask[ix, iy] = True

        return np.asarray(
            [
                (
                    float(x_coords[ix]) if len(x_coords) > ix else float(ix),
                    float(y_coords[iy]) if len(y_coords) > iy else float(iy),
                )
                for ix, iy in np.argwhere(boundary_mask)
            ],
            dtype=float,
        )

    def build_boundary_grid(self, world_field, x_coords=None, y_coords=None):
        """Create a binary grid with contour pixels marked as 1.0.

        This maps known boundary points (extracted from `world_field` or
        previously-initialized `known_boundary_points`) back to a grid with
        the same indexing convention as the input `world_field`.
        """
        field = np.asarray(world_field, dtype=float)
        if field.size == 0:
            return np.zeros_like(field, dtype=float)

        if self.known_boundary_points is None or self.known_boundary_points.size == 0:
            self.initialize_known_boundary(field, x_coords=x_coords, y_coords=y_coords)

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
    # BOUNDARY ORDERING / CIRCULAR HELPERS
    # ------------------------------------------------------------------

    def _ensure_ordered_closed_boundary(self):
        """Ensure `self.known_boundary_points` is an ordered closed loop.

        This performs a greedy nearest-neighbour trace to order boundary
        points so that indices 0..N-1 follow the contour sequentially and
        the last point is adjacent to the first. If ordering cannot be
        improved (e.g. few points), the array is left as-is.
        """
        if self.known_boundary_ordered:
            return

        pts = np.asarray(self.known_boundary_points, dtype=float)
        n = pts.shape[0]
        if n <= 2:
            self.known_boundary_ordered = True
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
            self.known_boundary_ordered = True
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
        self.known_boundary_ordered = True

    def _drone_boundary_projection(self, drone, use_pending=False):
        if self.known_boundary_points is None or len(self.known_boundary_points) == 0:
            return

        self._ensure_ordered_closed_boundary()
        boundary = np.asarray(self.known_boundary_points, dtype=float)
        is_closed = bool(self.known_boundary_closed)
        arc_lengths, total_length = self._boundary_arc_lengths(
            boundary,
            is_closed=is_closed,
        )
        pending_s = getattr(drone, "pending_boundary_s", None)
        pending_point = getattr(drone, "pending_boundary_point", None)
        if use_pending and pending_s is not None and pending_point is not None:
            boundary_s = float(pending_s)
            if is_closed and total_length > 1e-12:
                boundary_s = boundary_s % float(total_length)
            else:
                boundary_s = float(np.clip(boundary_s, 0.0, float(total_length)))
            projected = np.asarray(pending_point, dtype=float)
            nearest_idx = self._nearest_boundary_index(boundary, projected)
            drone.pending_boundary_s = None
            drone.pending_boundary_point = None
        else:
            boundary_s, projected, nearest_idx = self._arc_length_at_position(
                boundary,
                arc_lengths,
                self._estimated_position(drone),
                total_length,
                is_closed,
            )

        return float(boundary_s), np.asarray(projected, dtype=float), int(nearest_idx)

    def update_boundary_projection(self, drone):
        """Update the drone arc coordinate without changing its physical state."""
        projection = self._drone_boundary_projection(drone, use_pending=False)
        if projection is None:
            return

        boundary_s, _, nearest_idx = projection
        drone.boundary_index = int(nearest_idx)
        drone.boundary_s = float(boundary_s)
        if not hasattr(drone, "known_boundary_arcs"):
            drone.known_boundary_arcs = {}
        drone.known_boundary_arcs[drone.drone_id] = float(boundary_s)

    def project_drone_to_boundary(self, drone):
        """Snap a drone state to the nearest point of the known boundary."""
        if not self.constrain_to_boundary:
            return

        projection = self._drone_boundary_projection(drone, use_pending=True)
        if projection is None:
            return

        boundary_s, projected, nearest_idx = projection
        drone.x = float(projected[0])
        drone.y = float(projected[1])
        if hasattr(drone, "update_position_estimate"):
            drone.update_position_estimate()
        drone.known_positions[drone.drone_id] = self._estimated_position(drone)
        drone.boundary_index = int(nearest_idx)
        drone.boundary_s = float(boundary_s)
        if not hasattr(drone, "known_boundary_arcs"):
            drone.known_boundary_arcs = {}
        drone.known_boundary_arcs[drone.drone_id] = float(boundary_s)

    def compute_ring_ordering(self, current_drone, drones):
        """Compute 1D Voronoi partitioning with Multi-Source Dijkstra.

        Returns a dict compatible with tests: keys `N`, `current`, `succ`, `pred`, `center_of_mass`, `occupied_points`, `assigned_drone_indices`, `ring`.
        """

        if self.known_boundary_points is None or len(self.known_boundary_points) == 0:
            return None

        # Ensure strict sequential ordering
        self._ensure_ordered_closed_boundary()
        occupied_points = np.asarray(self.known_boundary_points, dtype=float)
        n_pts = int(occupied_points.shape[0])
        if n_pts == 0:
            return None

        known = current_drone.known_positions
        known_arcs = getattr(current_drone, "known_boundary_arcs", {})

        is_closed = bool(self.known_boundary_closed)
        arc_lengths, total_boundary_length = self._boundary_arc_lengths(
            occupied_points,
            is_closed=is_closed,
        )

        seeds = []
        for drone_id, position in known.items():
            if drone_id in known_arcs:
                seed_s = float(known_arcs[drone_id])
                if is_closed and total_boundary_length > 1e-12:
                    seed_s = seed_s % float(total_boundary_length)
                else:
                    seed_s = float(np.clip(seed_s, 0.0, float(total_boundary_length)))
                projected = self._point_at_arc_length(
                    occupied_points,
                    arc_lengths,
                    seed_s,
                    total_boundary_length,
                    is_closed,
                )
                seed_idx = self._nearest_boundary_index(occupied_points, projected)
            else:
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
        cur_pos = self._estimated_position(current_drone)
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

    def _equidistant_action(self, drone, ring_info, world_field, x_coords, y_coords):
        """Move the drone toward the current Lloyd target on the boundary."""
        target = None
        if isinstance(ring_info, dict):
            current = ring_info.get('current', {})
            target = current.get('target_centroid')
        if target is None:
            target = getattr(drone, 'target_centroid', None)
        if target is None:
            return np.zeros(2, dtype=float)

        max_speed = float(getattr(drone, 'max_speed', 0.12))
        current_pos = self._estimated_position(drone)

        action = float(self.k_t) * (np.asarray(target, dtype=float) - current_pos)
        return self._clip_action(action, max_speed=max_speed)


class DroneController(Controller):
    """Local onboard controller for mapping first, then 1D Lloyd coverage."""

    def __init__(
        self,
        sim_map=None,
        communication_radius=0.0,
        fully_connected=False,
        occupancy_threshold=0.5,
        resolution=0.1,
        known_boundary_points=None,
        known_boundary_closed=True,
        k_t=1.0,
        constrain_to_boundary=True,
    ):
        super().__init__(
            sim_map=sim_map,
            communication_radius=communication_radius,
            occupancy_threshold=occupancy_threshold,
            k_t=k_t,
            constrain_to_boundary=constrain_to_boundary,
        )
        self.fully_connected = bool(fully_connected)
        self.resolution = float(resolution)

        self.max_speed = 0.12
        self.exploration_speed = 0.08
        self.k_n = 0.9
        self.boundary_lock_gain = 0.8
        self.edge_tracking_gain = 0.85
        self.edge_tangent_neighbors = 16
        self.mapping_tangent_smoothing = 0.75
        self.edge_lookahead_distance = 0.45
        self.edge_lookahead_gain = 0.30
        self.max_edge_normal_correction = 0.07

        if known_boundary_points is not None:
            self.set_known_boundary(
                known_boundary_points,
                known_boundary_closed=known_boundary_closed,
                already_ordered=True,
            )

    @staticmethod
    def _normalize(vector):
        vector = np.asarray(vector, dtype=float)
        if vector.ndim != 1 or not np.all(np.isfinite(vector)):
            return None
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            return None
        return vector / norm

    @staticmethod
    def _random_direction():
        angle = np.random.uniform(0.0, 2.0 * np.pi)
        return np.array([np.cos(angle), np.sin(angle)], dtype=float)

    def set_known_boundary(
        self,
        boundary_points,
        known_boundary_closed=True,
        already_ordered=True,
    ):
        self.known_boundary_points = np.asarray(boundary_points, dtype=float).copy()
        self.known_boundary_closed = bool(known_boundary_closed)
        self.known_boundary_ordered = bool(already_ordered)

    def _camera_field_estimate(self, drone):
        image = getattr(drone, "last_camera_image", None)
        if image is None:
            return None

        image = np.asarray(image, dtype=float)
        if image.ndim != 2 or min(image.shape) < 3:
            return None

        spacing = getattr(drone, "last_camera_spacing", None)
        if spacing is None:
            dx = dy = self.resolution
        else:
            dx, dy = spacing
            dx = max(abs(float(dx)), 1e-12)
            dy = max(abs(float(dy)), 1e-12)

        row = image.shape[0] // 2
        col = image.shape[1] // 2
        row = int(np.clip(row, 1, image.shape[0] - 2))
        col = int(np.clip(col, 1, image.shape[1] - 2))

        concentration = float(image[row, col])
        patch = image[row - 1 : row + 2, col - 1 : col + 2]
        sobel_x = np.array(
            [
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        sobel_y = np.array(
            [
                [-1.0, -2.0, -1.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 1.0],
            ],
            dtype=float,
        )
        grad_x = float(np.sum(sobel_y * patch) / (8.0 * dx))
        grad_y = float(np.sum(sobel_x * patch) / (8.0 * dy))
        gradient = np.array([grad_x, grad_y], dtype=float)

        if not np.all(np.isfinite(gradient)):
            return None

        return concentration, gradient

    def _exploration_action(self, drone):
        direction = getattr(drone, "exploration_direction", None)
        if direction is None:
            direction = self._random_direction()

        direction = self._normalize(direction)
        if direction is None:
            direction = self._random_direction()

        norm_dir = self._normalize(direction)
        if norm_dir is None:
            norm_dir = self._random_direction()
        drone.exploration_direction = norm_dir
        return drone.exploration_direction * self.exploration_speed

    def _point_cloud_tracking_action(self, drone, points):
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return None

        points = points.reshape(-1, 2)
        finite_mask = np.all(np.isfinite(points), axis=1)
        if not np.any(finite_mask):
            return None
        points = points[finite_mask]

        estimated_pos = self._estimated_position(drone)
        distances = np.linalg.norm(points - estimated_pos, axis=1)
        if distances.size == 0 or not np.all(np.isfinite(distances)):
            return None

        nearest_order = np.argsort(distances)
        nearest_point = points[int(nearest_order[0])]
        edge_error = nearest_point - estimated_pos

        neighbor_count = min(
            int(points.shape[0]),
            max(2, int(self.edge_tangent_neighbors)),
        )
        local_points = points[nearest_order[:neighbor_count]]

        tangent = None
        if local_points.shape[0] >= 2:
            centered = local_points - np.mean(local_points, axis=0)
            try:
                _, singular_values, vh = np.linalg.svd(
                    centered,
                    full_matrices=False,
                )
                if singular_values.size > 0 and singular_values[0] > 1e-9:
                    tangent = vh[0]
            except np.linalg.LinAlgError:
                tangent = None

        if tangent is None:
            normal = self._normalize(estimated_pos - nearest_point)
            if normal is not None:
                tangent = np.array([-normal[1], normal[0]], dtype=float)

        tangent = self._normalize(tangent)
        if tangent is None:
            return None

        previous_tangent = getattr(drone, "last_mapping_tangent", None)
        previous_tangent = self._normalize(previous_tangent) if previous_tangent is not None else None
        if previous_tangent is not None and float(np.dot(tangent, previous_tangent)) < 0.0:
            tangent = -tangent
        elif previous_tangent is None:
            reference = self._normalize(getattr(drone, "last_control_vector", None))
            if reference is None:
                reference = self._normalize(getattr(drone, "exploration_direction", None))
            if reference is not None and float(np.dot(tangent, reference)) < 0.0:
                tangent = -tangent

        if previous_tangent is not None:
            alpha = float(np.clip(self.mapping_tangent_smoothing, 0.0, 1.0))
            smoothed = self._normalize(alpha * previous_tangent + (1.0 - alpha) * tangent)
            if smoothed is not None:
                tangent = smoothed

        drone.last_mapping_tangent = tangent.copy()

        tangential_error = float(np.dot(edge_error, tangent)) * tangent
        normal_error = edge_error - tangential_error
        normal_norm = float(np.linalg.norm(normal_error))
        max_normal = float(max(0.0, self.max_edge_normal_correction))
        if normal_norm > max_normal > 0.0:
            normal_error = normal_error * (max_normal / normal_norm)

        lookahead_error = self.edge_lookahead_distance * tangent
        relative_points = points - nearest_point
        forward_distance = relative_points @ tangent
        lateral_vectors = relative_points - np.outer(forward_distance, tangent)
        lateral_distance = np.linalg.norm(lateral_vectors, axis=1)
        forward_mask = (
            (forward_distance > 0.1)
            & (forward_distance < max(0.75, 2.5 * self.edge_lookahead_distance))
            & (lateral_distance < 0.55)
        )
        if np.any(forward_mask):
            candidate_indices = np.flatnonzero(forward_mask)
            target_idx = candidate_indices[
                int(
                    np.argmin(
                        np.abs(
                            forward_distance[candidate_indices]
                            - self.edge_lookahead_distance
                        )
                    )
                )
            ]
            lookahead_error = points[int(target_idx)] - estimated_pos

        action = (
            self.exploration_speed * tangent
            + self.edge_tracking_gain * normal_error
            + self.edge_lookahead_gain * lookahead_error
        )
        return self._clip_action(action, max_speed=self.max_speed)

    def _edge_tracking_action(self, drone):
        edge_points = getattr(drone, "last_edge_points", None)
        if edge_points is None:
            return None
        return self._point_cloud_tracking_action(drone, edge_points)

    def _has_local_edge_points(self, drone):
        edge_points = getattr(drone, "last_edge_points", None)
        if edge_points is None:
            return False

        edge_points = np.asarray(edge_points, dtype=float)
        if edge_points.size == 0:
            return False

        edge_points = edge_points.reshape(-1, 2)
        return bool(np.any(np.all(np.isfinite(edge_points), axis=1)))

    def _boundary_tracking_action(
        self,
        drone,
        world_field,
        x_coords,
        y_coords,
    ):
        del world_field, x_coords, y_coords
        edge_action = self._edge_tracking_action(drone)
        if edge_action is not None:
            return edge_action

        camera_estimate = self._camera_field_estimate(drone)
        if camera_estimate is None:
            return None

        concentration, gradient = camera_estimate
        normal = self._normalize(gradient)
        if normal is None:
            return None

        tangent = self._normalize(np.array([-normal[1], normal[0]], dtype=float))
        if tangent is None:
            return None

        error = float(np.clip(concentration - self.occupancy_threshold, -1.0, 1.0))
        normal_gain = self.k_n * error + self.boundary_lock_gain * error
        return self._clip_action(
            self.k_t * tangent - normal_gain * normal,
            max_speed=self.max_speed,
        )

    def _grid_points(self, drone):
        grid = np.asarray(getattr(drone, "grid", np.zeros((1, 1), dtype=float)), dtype=float)
        if grid.size == 0:
            return None

        occupied = np.argwhere(grid > self.occupancy_threshold)
        if occupied.size == 0:
            return None

        grid_bounds = getattr(drone, "grid_bounds", None)
        if grid_bounds is None:
            return None

        x_min, x_max, y_min, y_max = grid_bounds
        if x_max <= x_min or y_max <= y_min:
            return None

        occupied_points = np.column_stack(
            (
                x_min + (occupied[:, 0] + 0.5) * self.resolution,
                y_min + (occupied[:, 1] + 0.5) * self.resolution,
            )
        ).astype(float)
        return occupied_points

    def _grid_target(self, drone):
        occupied_points = self._grid_points(drone)
        if occupied_points is None:
            return None

        estimated_pos = self._estimated_position(drone)
        deltas = occupied_points - estimated_pos
        distances = np.linalg.norm(deltas, axis=1)
        return occupied_points[int(np.argmin(distances))]

    def _mapping_action(
        self,
        drone,
        world_field,
        x_coords,
        y_coords,
    ):
        target = self._grid_target(drone)
        if target is not None:
            if self._has_local_edge_points(drone):
                drone.last_control_mode = "boundary_tracking"
                action = self._boundary_tracking_action(drone, world_field, x_coords, y_coords)
            else:
                action = None

            if action is None:
                direction = self._normalize(
                    np.asarray(target, dtype=float) - self._estimated_position(drone)
                )
                if direction is not None:
                    drone.exploration_direction = direction.copy()
                    drone.last_mapping_tangent = None
                    drone.last_control_mode = "consensus_seek"
                    action = direction * self.exploration_speed
                else:
                    action = self._exploration_action(drone)
                    drone.last_control_mode = "explore"
        else:
            action = self._exploration_action(drone)
            drone.last_control_mode = "explore"

        return self._clip_action(action, max_speed=self.max_speed)

    def _lloyd_action(self, drone):
        ring_info = self.compute_ring_ordering(drone, None)
        drone.last_control_mode = "lloyd"
        action = self._equidistant_action(
            drone,
            ring_info,
            world_field=None,
            x_coords=None,
            y_coords=None,
        )
        return self._clip_action(
            action,
            max_speed=getattr(drone, "max_speed", self.max_speed),
        )

    def compute_action(
        self,
        drone,
        world_field=None,
        x_coords=None,
        y_coords=None,
    ):
        if getattr(drone, "control_state", "mapping") == "lloyd":
            return self._lloyd_action(drone)

        if world_field is None or x_coords is None or y_coords is None:
            if self.known_boundary_points is not None and len(self.known_boundary_points) > 0:
                return self._lloyd_action(drone)
            return self._clip_action(self._exploration_action(drone), max_speed=self.max_speed)

        return self._mapping_action(drone, world_field, x_coords, y_coords)

    def project_to_boundary(self, drone):
        self.project_drone_to_boundary(drone)
