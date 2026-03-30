import numpy as np
import cv2


def _to_uint8_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to a single-channel uint8 array for Canny."""
    img = np.asarray(image)

    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        if img.size > 0 and float(np.nanmax(img)) <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def extract_outer_boundary(points: np.ndarray, num_bins: int = 180) -> np.ndarray:
    """
    Extract the outermost boundary points by converting to polar coordinates
    and keeping the point with the maximum radius for each angular bin.
    """
    if points.size == 0 or len(points) < 5:
        return points

    # points are (row, col) coordinates => y, x
    rows = points[:, 0]
    cols = points[:, 1]

    # A. Estimate a provisional center
    # cx = mean(x_i), cy = mean(y_i)
    cy = np.mean(rows)
    cx = np.mean(cols)

    # B. Convert points to polar coordinates
    # r_i = sqrt((x_i - cx)^2 + (y_i - cy)^2)
    # theta_i = atan2(y_i - cy, x_i - cx)
    dy = rows - cy
    dx = cols - cx
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)  # theta in [-pi, pi]

    # C. Discretize angles into fixed bins
    bins = ((theta + np.pi) / (2 * np.pi) * num_bins).astype(int)
    bins = np.clip(bins, 0, num_bins - 1)

    # D. Radial MAX selection: For each bin, keep ONLY the point with maximum radius
    # To handle multiple points in a bin efficiently:
    # Sort indices by bin (primary) and radius (secondary, descending)
    idx_sorted = np.lexsort((-r, bins))
    bins_sorted = bins[idx_sorted]
    
    # Keep only the first occurrence of each bin (which is the one with max radius)
    _, first_occurrences = np.unique(bins_sorted, return_index=True)
    filtered_indices = idx_sorted[first_occurrences]
    
    boundary_points = points[filtered_indices]

    # E. Robustness: Outlier rejection using Median Absolute Deviation (MAD)
    if len(boundary_points) > 10:
        # Recompute radius from center for these boundary points
        br = np.sqrt((boundary_points[:, 0] - cy)**2 + (boundary_points[:, 1] - cx)**2)
        med = np.median(br)
        mad = np.median(np.abs(br - med))
        if mad > 0:
            # Keep points within 3.0 MAD from the median radius
            mask = np.abs(br - med) < 3.0 * mad
            boundary_points = boundary_points[mask]

    return boundary_points


def detect_edges(
    image: np.ndarray,
    threshold1: int = 20,
    threshold2: int = 60,
    blur_kernel: tuple[int, int] = (5, 5),
    blur_sigma: float = 1.2,
    apply_morphology: bool = False,
) -> np.ndarray:
    """Detect edges with Gaussian smoothing followed by Canny."""
    gray = _to_uint8_grayscale(image)
    # Normalize contrast so Canny can work reliably on soft camera matrices.
    gray = cv2.normalize(gray.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    smoothed = cv2.GaussianBlur(gray, blur_kernel, blur_sigma)

    edges = cv2.Canny(smoothed, threshold1, threshold2)

    if apply_morphology:
        kernel = np.ones((3, 3), dtype=np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return edges.astype(np.uint8)


def extract_edge_points(edges: np.ndarray, debug: bool = False) -> np.ndarray:
    """Return edge pixel coordinates as (row, col) pairs, filtered for outer boundary."""
    # Wherever edge points are currently returned:
    raw_points = np.column_stack(np.where(edges > 0))
    
    if raw_points.size == 0:
        return raw_points

    # Transform the edge detection into an OUTER-BOUNDARY extractor.
    filtered_points = extract_outer_boundary(raw_points)
    
    if debug:
        import matplotlib.pyplot as plt
        plt.figure("Outer Boundary Detection Debug", figsize=(6, 6))
        # Swap x (col) and y (row) for plotting
        plt.scatter(raw_points[:, 1], raw_points[:, 0], c='red', s=1, alpha=0.4, label='Raw Points')
        plt.scatter(filtered_points[:, 1], filtered_points[:, 0], c='green', s=10, label='Outer Boundary')
        plt.gca().invert_yaxis()  # Invert for image coordinates
        plt.axis('equal')
        plt.legend()
        plt.title(f"Outer Boundary Estimation (n={len(filtered_points)})")
        plt.show()

    return filtered_points
