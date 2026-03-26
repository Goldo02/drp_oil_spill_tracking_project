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


def extract_edge_points(edges: np.ndarray) -> np.ndarray:
    """Return edge pixel coordinates as (row, col) pairs."""
    return np.column_stack(np.where(edges > 0))
