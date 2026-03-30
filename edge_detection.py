import numpy as np
import cv2


def _to_uint8_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to a single-channel uint8 array for processing."""
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
    Refine a set of points to a 1-pixel wide exterior shell using polar discretization.
    """
    if points.size == 0 or len(points) < 5:
        return points

    # points are (row, col) coordinates => y, x
    rows = points[:, 0]
    cols = points[:, 1]

    # center is mean of points
    cy = np.mean(rows)
    cx = np.mean(cols)

    # Polar coordinates
    dy = rows - cy
    dx = cols - cx
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    bins = ((theta + np.pi) / (2 * np.pi) * num_bins).astype(int)
    bins = np.clip(bins, 0, num_bins - 1)

    idx_sorted = np.lexsort((-r, bins))
    bins_sorted = bins[idx_sorted]
    
    _, first_occurrences = np.unique(bins_sorted, return_index=True)
    filtered_indices = idx_sorted[first_occurrences]
    
    boundary_points = points[filtered_indices]

    # Outlier rejection
    if len(boundary_points) > 10:
        br = np.sqrt((boundary_points[:, 0] - cy)**2 + (boundary_points[:, 1] - cx)**2)
        med = np.median(br)
        mad = np.median(np.abs(br - med))
        if mad > 0:
            mask = np.abs(br - med) < 3.0 * mad
            boundary_points = boundary_points[mask]

    return boundary_points


def detect_edges(
    image: np.ndarray,
    threshold1: int = 30,
    threshold2: int = 90,
    blur_kernel: tuple[int, int] = (5, 5),
    blur_sigma: float = 1.2,
    apply_morphology: bool = True,
) -> np.ndarray:
    """
    Isolates the oil spill boundary by thresholding and topological analysis.
    This replaces standard Canny detection to avoid rectangular FOV artifacts.
    """
    # 1) Robust Smoothing
    # We use a significant blur to suppress sensor noise before thresholding
    smoothed = cv2.GaussianBlur(image.astype(np.float32), (11, 11), 1.5)
    
    # 2) Binary Segmentation (Thresholding Oil vs Water)
    # The oil is modeled as 1.0, water as 0.0. 0.5 is a robust threshold.
    _, binary = cv2.threshold(smoothed, 0.5, 1.0, cv2.THRESH_BINARY)
    binary = (binary * 255).astype(np.uint8)
    
    # 3) Extract Boundary using Canny on the smoothed binary image
    # This guarantees a single, clean transition line.
    edges = cv2.Canny(binary, 50, 150)
    
    # Store images for extract_edge_points to leverage for FOV-clipping
    detect_edges._last_gray = binary 
    
    return edges.astype(np.uint8)


def extract_edge_points(edges: np.ndarray, debug: bool = False) -> np.ndarray:
    """Return coordinates of the primary oil-water arc, ignoring FOV-boundary clipped edges."""
    
    h, w = edges.shape
    
    # 1) Find all external contours in the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return np.empty((0, 2))
    
    # Combined Point Cloud from all external contours
    all_contour_points = []
    for cnt in contours:
        # contour is (N, 1, 2) [x, y]
        pts = cnt.reshape(-1, 2)
        
        # 2) FOV CLIP FILTER: Remove points that are exactly on the image border
        # False edges from sensor limits always occur at the perimeter (row 0, row h-1, etc.)
        # We use a 3-pixel margin for safety.
        mask = (pts[:, 1] > 2) & (pts[:, 1] < h - 3) & \
               (pts[:, 0] > 2) & (pts[:, 0] < w - 3)
        
        clipped_pts = pts[mask]
        if clipped_pts.size > 0:
            all_contour_points.append(clipped_pts)

    if not all_contour_points:
        return np.empty((0, 2))
    
    # Merge valid arcs
    points_merged = np.vstack(all_contour_points)
    
    # Swap (x, y) -> (row, col)
    res_points = np.column_stack([points_merged[:, 1], points_merged[:, 0]])
    
    # 3) Final refinement: extract the outermost shell of the detected arc
    # This helps if small internal 'bubbles' survived.
    final_points = extract_outer_boundary(res_points)
    
    if debug:
        import matplotlib.pyplot as plt
        plt.figure("Contour Refinement Debug", figsize=(6,6))
        plt.scatter(res_points[:, 1], res_points[:, 0], c='red', s=5, label='Filtered Arcs')
        plt.scatter(final_points[:, 1], final_points[:, 0], c='green', s=10, label='Final Boundary')
        plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.show()

    return final_points
