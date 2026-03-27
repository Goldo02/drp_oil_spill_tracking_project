import numpy as np

from sensors import CameraSensor, GPSSensor


class Drone:
    """Static drone agent for distributed occupancy grid mapping."""

    def __init__(
        self,
        drone_id,
        x,
        y,
        grid_shape,
        grid_bounds,
        sensor_size=100,
        gps_noise=0.03,
        camera_noise=0.03,
    ):
        self.drone_id = drone_id
        self.x = float(x)
        self.y = float(y)
        self.grid_shape = tuple(grid_shape)
        self.grid_bounds = tuple(grid_bounds)
        self.x_min, self.x_max, self.y_min, self.y_max = self.grid_bounds
        self.Nx, self.Ny = self.grid_shape

        self.gps = GPSSensor(noise_std=gps_noise)
        self.camera = CameraSensor(size=sensor_size, noise_std=camera_noise)

        # Each drone keeps a full copy of the global occupancy grid.
        self.grid = np.zeros(self.grid_shape, dtype=float)

        # State used for visualization and debugging.
        self.edge_detected = False
        self.last_edge_point = None
        self.last_edge_points = np.empty((0, 2), dtype=float)
        self.last_edge_count = 0
        self.last_oil_fraction = None

    def get_gps_pos(self):
        """Return noisy GPS coordinates."""
        return self.gps.sense((self.x, self.y))

    def _camera_window(self, world_field, x_coords, y_coords):
        """Extract the local sensing window in global coordinates."""
        dx = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1.0
        dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 1.0

        i_center = int((self.x - x_coords[0]) / dx)
        j_center = int((self.y - y_coords[0]) / dy)
        half = self.camera.size // 2

        i_min = max(0, i_center - half)
        i_max = min(world_field.shape[0], i_center + half + 1)
        j_min = max(0, j_center - half)
        j_max = min(world_field.shape[1], j_center + half + 1)

        return (
            world_field[i_min:i_max, j_min:j_max].astype(float),
            x_coords[i_min:i_max],
            y_coords[j_min:j_max],
        )

    @staticmethod
    def _boundary_mask(binary_window):
        """Mark occupied cells that touch free space in the local window."""
        if binary_window.size == 0:
            return np.zeros_like(binary_window, dtype=bool)

        padded = np.pad(binary_window.astype(bool), 1, mode="constant", constant_values=False)
        center = padded[1:-1, 1:-1]
        north = padded[:-2, 1:-1]
        south = padded[2:, 1:-1]
        west = padded[1:-1, :-2]
        east = padded[1:-1, 2:]
        return center & (~north | ~south | ~west | ~east)

    @staticmethod
    def _mean_filter(array, kernel_size=5):
        """
        Smooth a 2D array with a simple uniform filter.

        This suppresses isolated noisy pixels that would otherwise create
        spurious interior edge fragments after thresholding.
        """
        if kernel_size <= 1:
            return np.asarray(array, dtype=float)

        arr = np.asarray(array, dtype=float)
        pad = kernel_size // 2
        padded = np.pad(arr, pad_width=pad, mode="edge")
        smoothed = np.zeros_like(arr, dtype=float)

        for di in range(kernel_size):
            for dj in range(kernel_size):
                smoothed += padded[di:di + arr.shape[0], dj:dj + arr.shape[1]]

        return smoothed / float(kernel_size * kernel_size)

    @staticmethod
    def _largest_connected_component(binary_window):
        """
        Keep only the largest 4-connected occupied component.

        This removes small noisy islands and holes that otherwise turn into
        internal boundary segments after thresholding.
        """
        binary = np.asarray(binary_window, dtype=bool)
        if binary.size == 0:
            return binary

        visited = np.zeros_like(binary, dtype=bool)
        best_component = []
        best_size = 0
        rows, cols = binary.shape

        for start_r in range(rows):
            for start_c in range(cols):
                if not binary[start_r, start_c] or visited[start_r, start_c]:
                    continue

                stack = [(start_r, start_c)]
                visited[start_r, start_c] = True
                component = []

                while stack:
                    r, c = stack.pop()
                    component.append((r, c))
                    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
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

    def sense(self, world_field, x_coords, y_coords, occupancy_threshold=0.5):
        """
        Detect boundary pixels in the local camera window and return them in
        global coordinates.
        """
        # The camera introduces measurement noise, so the boundary estimate
        # can move slightly from one sensing cycle to the next.
        local_field = self.camera.sense(world_field, self.x, self.y, x_coords, y_coords)
        _, local_x_coords, local_y_coords = self._camera_window(
            world_field,
            x_coords,
            y_coords,
        )

        if local_field.size == 0:
            self.edge_detected = False
            self.last_edge_points = np.empty((0, 2), dtype=float)
            self.last_edge_point = None
            self.last_edge_count = 0
            self.last_oil_fraction = None
            return self.last_edge_points

        # Smooth the noisy camera field before thresholding so isolated noisy
        # pixels do not create fake interior segments.
        local_field = self._mean_filter(local_field, kernel_size=5)

        # The drone already works in the global frame, so boundary points can
        # be mapped directly into the shared occupancy grid.
        binary_window = local_field >= occupancy_threshold
        binary_window = self._largest_connected_component(binary_window)
        boundary = self._boundary_mask(binary_window)
        rows, cols = np.where(boundary)

        if rows.size == 0:
            self.edge_detected = False
            self.last_edge_points = np.empty((0, 2), dtype=float)
            self.last_edge_point = None
            self.last_edge_count = 0
            self.last_oil_fraction = float(np.mean(binary_window))
            return self.last_edge_points

        edge_x = local_x_coords[rows]
        edge_y = local_y_coords[cols]
        edge_points = np.column_stack((edge_x, edge_y)).astype(float)

        # Drop detections that land on the border of the local crop. Those are
        # usually artifacts from the square camera window, not true spill edge
        # evidence, and they create the straight internal segments in the final
        # occupancy map.
        margin = max(3, min(binary_window.shape) // 20)
        interior_mask = (
            (rows >= margin)
            & (rows < binary_window.shape[0] - margin)
            & (cols >= margin)
            & (cols < binary_window.shape[1] - margin)
        )
        filtered_points = edge_points[interior_mask]
        if filtered_points.size == 0:
            filtered_points = edge_points
        filtered_distances = np.hypot(filtered_points[:, 0] - self.x, filtered_points[:, 1] - self.y)

        self.edge_detected = True
        self.last_edge_points = filtered_points
        self.last_edge_count = int(filtered_points.shape[0])
        self.last_oil_fraction = float(np.mean(binary_window))

        # Use the boundary point closest to the drone for visualization.
        nearest_idx = int(np.argmin(filtered_distances))
        self.last_edge_point = (float(filtered_points[nearest_idx, 0]), float(filtered_points[nearest_idx, 1]))
        return filtered_points

    def update_grid(self, edge_points, x_min, y_min, resolution, alpha=None):
        """
        Convert global edge points into the shared occupancy grid.

        If alpha is None, a direct count update is applied.
        If alpha is provided, exponential smoothing is used instead.
        """
        if edge_points is None or len(edge_points) == 0:
            return 0

        measurement_grid = np.zeros_like(self.grid)
        valid_updates = 0

        for x, y in np.asarray(edge_points, dtype=float):
            ix = int((x - x_min) / resolution)
            iy = int((y - y_min) / resolution)
            if 0 <= ix < self.Nx and 0 <= iy < self.Ny:
                measurement_grid[ix, iy] += 1.0
                valid_updates += 1

        if valid_updates == 0:
            return 0

        if alpha is None:
            self.grid += measurement_grid
        else:
            alpha = float(alpha)
            self.grid = (1.0 - alpha) * self.grid + alpha * measurement_grid

        return valid_updates

    def consensus_update(self, neighbors, own_grid=None):
        """
        Average this drone's full grid with the full grids received from its
        neighbors.
        """
        grids = [np.array(own_grid if own_grid is not None else self.grid, dtype=float)]
        for neighbor in neighbors:
            grids.append(np.array(neighbor.grid, dtype=float))

        if len(grids) == 1:
            return self.grid

        self.grid = np.mean(grids, axis=0)
        return self.grid
