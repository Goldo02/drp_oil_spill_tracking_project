import numpy as np

from edge_detection import detect_edges, extract_edge_points


class Sensor:
    """Base class for robot sensors."""

    def __init__(self, noise_std=0.0):
        self.noise_std = float(noise_std)

    def add_noise(self, value):
        """Add zero-mean Gaussian noise to a measurement."""
        value = np.asarray(value, dtype=float)

        if self.noise_std <= 0.0:
            return value

        noise = np.random.normal(
            loc=0.0,
            scale=self.noise_std,
            size=value.shape,
        )

        return value + noise


class GPSSensor(Sensor):
    """
    Proprioceptive GPS sensor.

    Returns the robot position with additive Gaussian noise.
    """

    def sense(self, real_position):
        """Measure the robot position."""
        position = np.asarray(real_position, dtype=float)

        if position.shape != (2,):
            raise ValueError(
                "GPS position must have shape (2,)"
            )

        return self.add_noise(position)


class CameraMeasurement:
    """
    Result of a camera sensing operation.

    Attributes
    ----------
    image : np.ndarray
        Noisy local camera image.

    edge_points : np.ndarray
        Detected boundary points in global (x, y) coordinates.

    oil_fraction : float
        Fraction of pixels classified as occupied/oil.
    """

    def __init__(
        self,
        image,
        edge_points,
        oil_fraction,
    ):
        self.image = np.asarray(
            image,
            dtype=float,
        )

        self.edge_points = np.asarray(
            edge_points,
            dtype=float,
        )

        self.oil_fraction = float(
            oil_fraction
        )


class CameraSensor(Sensor):
    """
    Exteroceptive camera sensor.

    Responsibilities:
        1. Extract a local field of view.
        2. Add measurement noise.
        3. Optionally smooth the image.
        4. Detect oil-spill edges.
        5. Convert detected points to world coordinates.
        6. Estimate local oil occupancy.

    The camera does not modify the drone state, occupancy grid, or control.
    """

    def __init__(
        self,
        size=100,
        noise_std=0.1,
        apply_blur=False,
        blur_sigma=1.0,
        edge_threshold1=30,
        edge_threshold2=90,
        occupancy_threshold=0.5,
    ):
        super().__init__(noise_std)

        self.size = int(size)

        if self.size <= 0:
            raise ValueError(
                "Camera size must be positive"
            )

        self.apply_blur = bool(
            apply_blur
        )

        self.blur_sigma = float(
            blur_sigma
        )

        self.edge_threshold1 = float(
            edge_threshold1
        )

        self.edge_threshold2 = float(
            edge_threshold2
        )

        self.occupancy_threshold = float(
            occupancy_threshold
        )

    # ======================================================================
    # PUBLIC API
    # ======================================================================

    def sense(
        self,
        world_field,
        x,
        y,
        x_coords,
        y_coords,
        occupancy_threshold=None,
    ):
        """
        Acquire and process a local camera measurement.

        Parameters
        ----------
        world_field : np.ndarray
            Global environment field.

        x, y : float
            Current drone position.

        x_coords, y_coords : np.ndarray
            Coordinates corresponding to the environment field.

        occupancy_threshold : float, optional
            Threshold used for estimating oil occupancy.

        Returns
        -------
        CameraMeasurement
            Processed camera measurement.
        """

        field = np.asarray(
            world_field,
            dtype=float,
        )

        x_coords = np.asarray(
            x_coords,
            dtype=float,
        )

        y_coords = np.asarray(
            y_coords,
            dtype=float,
        )

        self._validate_inputs(
            field,
            x_coords,
            y_coords,
        )

        threshold = (
            self.occupancy_threshold
            if occupancy_threshold is None
            else float(occupancy_threshold)
        )

        # --------------------------------------------------------------
        # 1. Local field of view
        # --------------------------------------------------------------

        local_matrix = self._extract_local_window(
            world_field=field,
            x=x,
            y=y,
            x_coords=x_coords,
            y_coords=y_coords,
        )

        # --------------------------------------------------------------
        # 2. Measurement noise
        # --------------------------------------------------------------

        noisy_matrix = self.add_noise(
            local_matrix
        )

        # Camera measurements should remain within the physical
        # field range.
        noisy_matrix = np.clip(
            noisy_matrix,
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # 3. Optional smoothing
        # --------------------------------------------------------------

        if self.apply_blur:
            noisy_matrix = self._gaussian_blur(
                noisy_matrix,
                self.blur_sigma,
            )

        # --------------------------------------------------------------
        # 4. Oil occupancy
        # --------------------------------------------------------------

        binary_window = (
            noisy_matrix >= threshold
        )

        if binary_window.size > 0:
            oil_fraction = float(
                np.mean(binary_window)
            )
        else:
            oil_fraction = 0.0

        # --------------------------------------------------------------
        # 5. Edge detection
        # --------------------------------------------------------------

        edges = self._detect_edges(
            noisy_matrix
        )

        edge_points = extract_edge_points(
            edges
        )

        # --------------------------------------------------------------
        # 6. Local -> world coordinates
        # --------------------------------------------------------------

        edge_points = (
            self._local_to_world_coordinates(
                edge_points=edge_points,
                x=x,
                y=y,
                x_coords=x_coords,
                y_coords=y_coords,
                image_shape=noisy_matrix.shape,
            )
        )

        return CameraMeasurement(
            image=noisy_matrix,
            edge_points=edge_points,
            oil_fraction=oil_fraction,
        )

    # ======================================================================
    # VALIDATION
    # ======================================================================

    @staticmethod
    def _validate_inputs(
        field,
        x_coords,
        y_coords,
    ):
        """Validate environment data supplied to the camera."""

        if field.ndim != 2:
            raise ValueError(
                "world_field must be a 2D array"
            )

        if field.shape != (
            len(x_coords),
            len(y_coords),
        ):
            raise ValueError(
                "world_field shape must match "
                "x_coords and y_coords"
            )

        if len(x_coords) < 2:
            raise ValueError(
                "At least two x coordinates are required"
            )

        if len(y_coords) < 2:
            raise ValueError(
                "At least two y coordinates are required"
            )

    # ======================================================================
    # LOCAL FIELD OF VIEW
    # ======================================================================

    def _extract_local_window(
        self,
        world_field,
        x,
        y,
        x_coords,
        y_coords,
    ):
        """
        Extract a square camera window centered on the drone.

        The output always has shape `(size, size)`.
        """

        dx = float(
            x_coords[1] - x_coords[0]
        )

        dy = float(
            y_coords[1] - y_coords[0]
        )

        if abs(dx) <= 1e-12:
            raise ValueError(
                "x coordinate spacing must be non-zero"
            )

        if abs(dy) <= 1e-12:
            raise ValueError(
                "y coordinate spacing must be non-zero"
            )

        i_center = int(
            round(
                (float(x) - x_coords[0])
                / dx
            )
        )

        j_center = int(
            round(
                (float(y) - y_coords[0])
                / dy
            )
        )

        half = self.size // 2

        i_min = max(
            0,
            i_center - half,
        )

        i_max = min(
            world_field.shape[0],
            i_center + half + 1,
        )

        j_min = max(
            0,
            j_center - half,
        )

        j_max = min(
            world_field.shape[1],
            j_center + half + 1,
        )

        local_matrix = world_field[
            i_min:i_max,
            j_min:j_max,
        ].astype(float)

        # --------------------------------------------------------------
        # Padding outside simulation domain
        # --------------------------------------------------------------

        pad_before_i = max(
            0,
            half - i_center,
        )

        pad_after_i = max(
            0,
            (i_center + half + 1)
            - world_field.shape[0],
        )

        pad_before_j = max(
            0,
            half - j_center,
        )

        pad_after_j = max(
            0,
            (j_center + half + 1)
            - world_field.shape[1],
        )

        local_matrix = np.pad(
            local_matrix,
            (
                (
                    pad_before_i,
                    pad_after_i,
                ),
                (
                    pad_before_j,
                    pad_after_j,
                ),
            ),
            mode="constant",
            constant_values=0.0,
        )

        # --------------------------------------------------------------
        # Enforce exact output size
        # --------------------------------------------------------------

        target_shape = (
            self.size,
            self.size,
        )

        local_matrix = local_matrix[
            : self.size,
            : self.size,
        ]

        if local_matrix.shape != target_shape:
            padded = np.zeros(
                target_shape,
                dtype=float,
            )

            rows = min(
                local_matrix.shape[0],
                self.size,
            )

            cols = min(
                local_matrix.shape[1],
                self.size,
            )

            padded[
                :rows,
                :cols,
            ] = local_matrix[
                :rows,
                :cols,
            ]

            local_matrix = padded

        return local_matrix

    # ======================================================================
    # EDGE DETECTION
    # ======================================================================

    def _detect_edges(self, image):
        """
        Detect edges using the dedicated edge_detection module.

        The sensor owns the edge-detector configuration while the actual
        image-processing algorithm remains in edge_detection.py.
        """

        return detect_edges(
            image,
            threshold=self.occupancy_threshold,
            sigma=self.blur_sigma if self.apply_blur else 1.0,
        )

    # ======================================================================
    # IMAGE PROCESSING
    # ======================================================================

    @staticmethod
    def _gaussian_blur(
        image,
        sigma,
    ):
        """Apply Gaussian smoothing using scipy."""

        if sigma <= 0.0:
            return np.asarray(
                image,
                dtype=float,
            )

        try:
            from scipy.ndimage import gaussian_filter
        except ImportError as exc:
            raise ImportError(
                "Gaussian blur requires scipy"
            ) from exc

        return gaussian_filter(
            np.asarray(
                image,
                dtype=float,
            ),
            sigma=float(sigma),
        )

    # ======================================================================
    # COORDINATE TRANSFORMATION
    # ======================================================================

    @staticmethod
    def _local_to_world_coordinates(
        edge_points,
        x,
        y,
        x_coords,
        y_coords,
        image_shape,
    ):
        """
        Convert local image coordinates to global world coordinates.

        Input edge points are expected in `(row, column)` format.
        """

        points = np.asarray(
            edge_points,
            dtype=float,
        )

        if points.size == 0:
            return np.empty(
                (0, 2),
                dtype=float,
            )

        points = points.reshape(
            -1,
            2,
        )

        dx = (
            float(
                x_coords[1] - x_coords[0]
            )
            if len(x_coords) > 1
            else 1.0
        )

        dy = (
            float(
                y_coords[1] - y_coords[0]
            )
            if len(y_coords) > 1
            else 1.0
        )

        height, width = image_shape

        # Determine the exact grid cell that was used to extract the
        # local window. The extractor rounds the drone position to the
        # nearest grid index; using the grid coordinate here ensures the
        # forward and inverse mappings are consistent and avoids a
        # constant translation when the drone is not exactly on a grid
        # node.
        i_center = int(
            round((float(x) - x_coords[0]) / dx)
        )

        j_center = int(
            round((float(y) - y_coords[0]) / dy)
        )

        # Clip to valid indices in case the center was near the domain
        # boundary and the extractor padded the window.
        i_center = int(np.clip(i_center, 0, len(x_coords) - 1))
        j_center = int(np.clip(j_center, 0, len(y_coords) - 1))

        center_world_x = float(x_coords[i_center])
        center_world_y = float(y_coords[j_center])

        center_row = (height - 1) / 2.0
        center_col = (width - 1) / 2.0

        world_x = center_world_x + (points[:, 0] - center_row) * dx
        world_y = center_world_y + (points[:, 1] - center_col) * dy

        return np.column_stack(
            (
                world_x,
                world_y,
            )
        ).astype(float)