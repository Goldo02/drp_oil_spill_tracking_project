import numpy as np

try:
    from scipy.ndimage import gaussian_filter, binary_erosion
except ImportError:
    gaussian_filter = None
    binary_erosion = None


def detect_edges(
    image,
    threshold=0.5,
    sigma=1.0,
    **kwargs,
):
    """
    Detect edges (oil spill boundary contour) in a local sensor measurement.

    Directly uses the gradient/isoline of the smoothed concentration field
    without Canny:
    1. Smooths the noisy measurement field with a light Gaussian filter.
    2. Finds the single-pixel boundary where concentration crosses the threshold.
    3. Rejects empty regions (no false edges in clean water or solid oil).

    Parameters
    ----------
    image : np.ndarray
        2D local camera measurement array (values in [0, 1]).
    threshold : float, optional
        Occupancy/boundary threshold (default 0.5).
    sigma : float, optional
        Gaussian smoothing scale to suppress sensor noise (default 1.0).

    Returns
    -------
    np.ndarray
        Boolean 2D array of detected edge pixels (same shape as `image`).
    """
    image = np.asarray(image, dtype=float)

    if image.ndim != 2 or image.size == 0:
        return np.zeros_like(image, dtype=bool)

    # Return empty if the window has no significant variation (e.g. all empty water or all solid oil)
    val_min = float(np.min(image))
    val_max = float(np.max(image))
    if val_max - val_min < 0.15:
        return np.zeros_like(image, dtype=bool)

    # Handle legacy threshold arguments if provided
    if "threshold1" in kwargs:
        t1 = kwargs["threshold1"]
        if 0.0 < float(t1) < 1.0:
            threshold = float(t1)

    threshold = float(threshold)

    # 1. Smooth the measurement to eliminate high-frequency sensor noise
    if gaussian_filter is not None and sigma > 0.0:
        smoothed = gaussian_filter(image, sigma=float(sigma))
    else:
        smoothed = image

    # 2. Binary mask of the detected spill region
    mask = smoothed >= threshold

    # 3. Extract the single-pixel boundary contour
    if binary_erosion is not None:
        eroded = binary_erosion(mask, border_value=True)
        edges = mask ^ eroded
    else:
        padded = np.pad(mask, 1, mode="edge")
        eroded = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
        edges = mask & ~eroded

    return edges


def extract_edge_points(edges):
    """
    Convert a binary edge image into point coordinates.

    Parameters
    ----------
    edges : np.ndarray
        Binary edge image.

    Returns
    -------
    np.ndarray
        Array with shape `(N, 2)` where each point is `[row, column]`.
    """
    edges = np.asarray(edges, dtype=bool)

    if edges.ndim != 2 or edges.size == 0:
        return np.empty((0, 2), dtype=float)

    rows, cols = np.nonzero(edges)

    if len(rows) == 0:
        return np.empty((0, 2), dtype=float)

    return np.column_stack((rows, cols)).astype(float)