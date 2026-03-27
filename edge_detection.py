import numpy as np


def detect_edges(
    image: np.ndarray,
    threshold1: float = 0.5,
    threshold2: float = 0.5,
    blur_kernel: tuple[int, int] = (5, 5),
    blur_sigma: float = 1.2,
    apply_morphology: bool = False,
) -> np.ndarray:
    """
    Lightweight numpy-only boundary detector.

    This function is kept for compatibility, but the simulation now performs
    occupancy boundary extraction directly inside the drone sensing pipeline.
    """
    del threshold1, threshold2, blur_kernel, blur_sigma, apply_morphology

    arr = np.asarray(image, dtype=float)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    if np.nanmax(arr) <= 1.0:
        arr = arr * 255.0

    binary = arr >= 128.0
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    north = padded[:-2, 1:-1]
    south = padded[2:, 1:-1]
    west = padded[1:-1, :-2]
    east = padded[1:-1, 2:]

    edges = center & (~north | ~south | ~west | ~east)
    return edges.astype(np.uint8) * 255


def extract_edge_points(edges: np.ndarray) -> np.ndarray:
    """Return edge pixel coordinates as (row, col) pairs."""
    return np.column_stack(np.where(np.asarray(edges) > 0))
