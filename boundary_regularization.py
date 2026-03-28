"""Boundary point regularization using Voronoi / Lloyd-style updates.

The module starts from a 2D occupancy grid, extracts edge points, builds a
Voronoi diagram, computes region centroids, and projects every updated point
back onto the detected boundary so the points remain on the 1D curve embedded
in 2D.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import QhullError, Voronoi


def grid_to_world(indices, x_min, y_min, resolution):
    """Convert grid indices to world coordinates using cell centers."""
    indices = np.asarray(indices, dtype=float)
    if indices.size == 0:
        return np.zeros((0, 2), dtype=float)
    xs = x_min + (indices[:, 0] + 0.5) * resolution
    ys = y_min + (indices[:, 1] + 0.5) * resolution
    return np.column_stack((xs, ys))


def extract_boundary_points(final_grid, threshold, x_min, y_min, resolution):
    """Extract occupied cells and convert them into world-space boundary points."""
    mask = np.asarray(final_grid) > float(threshold)
    indices = np.argwhere(mask)
    return grid_to_world(indices, x_min, y_min, resolution)


def order_points_by_angle(points):
    """Sort a point cloud around its centroid for more stable circular processing."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return points
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    order = np.argsort(angles)
    return points[order]


def subsample_points(points, max_points=None, step=None):
    """Subsample points while preserving a roughly uniform angular spread."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return points

    if max_points is not None and max_points > 0 and len(points) > max_points:
        step = max(1, int(math.ceil(len(points) / float(max_points))))

    if step is None or step <= 1:
        return points.copy()

    return points[:: int(step)].copy()


def project_to_boundary(point, contour_points):
    """Project a point to the closest point on the boundary polyline.

    The contour is treated as a closed polyline built from the ordered samples,
    which gives a much smoother projection than snapping to the nearest pixel.
    """
    contour_points = np.asarray(contour_points, dtype=float)
    if len(contour_points) == 0:
        return np.asarray(point, dtype=float)
    if len(contour_points) == 1:
        return contour_points[0]
    if len(contour_points) == 2:
        a, b = contour_points
        ab = b - a
        denom = float(np.dot(ab, ab)) + 1e-15
        t = float(np.dot(np.asarray(point, dtype=float) - a, ab) / denom)
        t = float(np.clip(t, 0.0, 1.0))
        return a + t * ab

    p = np.asarray(point, dtype=float)
    best_point = contour_points[0]
    best_dist2 = np.inf
    for idx in range(len(contour_points)):
        a = contour_points[idx]
        b = contour_points[(idx + 1) % len(contour_points)]
        ab = b - a
        denom = float(np.dot(ab, ab)) + 1e-15
        t = float(np.dot(p - a, ab) / denom)
        t = float(np.clip(t, 0.0, 1.0))
        candidate = a + t * ab
        dist2 = float(np.sum((p - candidate) ** 2))
        if dist2 < best_dist2:
            best_dist2 = dist2
            best_point = candidate
    return best_point


def polygon_centroid(polygon):
    """Compute the centroid of a polygon with a robust fallback for degenerate cases."""
    polygon = np.asarray(polygon, dtype=float)
    if len(polygon) == 0:
        return np.zeros(2, dtype=float)
    if len(polygon) == 1:
        return polygon[0]
    if len(polygon) == 2:
        return np.mean(polygon, axis=0)

    x = polygon[:, 0]
    y = polygon[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = 0.5 * np.sum(cross)
    if abs(area) < 1e-12:
        return np.mean(polygon, axis=0)

    cx = np.sum((x + x_next) * cross) / (6.0 * area)
    cy = np.sum((y + y_next) * cross) / (6.0 * area)
    return np.array([cx, cy], dtype=float)


def _clip_polygon_to_box(polygon, x_min, x_max, y_min, y_max):
    """Clip a polygon against an axis-aligned bounding box."""

    def clip_against_edge(vertices, inside_fn, intersect_fn):
        if len(vertices) == 0:
            return vertices
        output = []
        prev = vertices[-1]
        prev_inside = inside_fn(prev)
        for curr in vertices:
            curr_inside = inside_fn(curr)
            if curr_inside:
                if not prev_inside:
                    output.append(intersect_fn(prev, curr))
                output.append(curr)
            elif prev_inside:
                output.append(intersect_fn(prev, curr))
            prev = curr
            prev_inside = curr_inside
        return np.asarray(output, dtype=float)

    poly = np.asarray(polygon, dtype=float)
    if len(poly) == 0:
        return poly

    poly = clip_against_edge(
        poly,
        lambda p: p[0] >= x_min,
        lambda p1, p2: np.array([x_min, p1[1] + (p2[1] - p1[1]) * (x_min - p1[0]) / (p2[0] - p1[0] + 1e-15)]),
    )
    poly = clip_against_edge(
        poly,
        lambda p: p[0] <= x_max,
        lambda p1, p2: np.array([x_max, p1[1] + (p2[1] - p1[1]) * (x_max - p1[0]) / (p2[0] - p1[0] + 1e-15)]),
    )
    poly = clip_against_edge(
        poly,
        lambda p: p[1] >= y_min,
        lambda p1, p2: np.array([p1[0] + (p2[0] - p1[0]) * (y_min - p1[1]) / (p2[1] - p1[1] + 1e-15), y_min]),
    )
    poly = clip_against_edge(
        poly,
        lambda p: p[1] <= y_max,
        lambda p1, p2: np.array([p1[0] + (p2[0] - p1[0]) * (y_max - p1[1]) / (p2[1] - p1[1] + 1e-15), y_max]),
    )
    return np.asarray(poly, dtype=float)


def _voronoi_finite_polygons_2d(vor, radius=None):
    """Reconstruct infinite Voronoi regions into finite polygons.

    Adapted from the SciPy Voronoi documentation recipe.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2.0

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges.get(p1, [])
        new_region = [v for v in vertices if v >= 0]

        if not ridges:
            new_regions.append(new_region)
            continue

        for p2, v1, v2 in ridges:
            if v1 >= 0 and v2 >= 0:
                continue
            v = v1 if v1 >= 0 else v2
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t) + 1e-15
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v] + direction * radius
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        # Sort region vertices counter-clockwise.
        vs = np.asarray([new_vertices[v] for v in new_region])
        region_center = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - region_center[1], vs[:, 0] - region_center[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices, dtype=float)


def _safe_voronoi(points):
    """Compute a Voronoi diagram with a small jitter fallback for degenerate cases."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        raise ValueError("Need at least 3 points for Voronoi tessellation")
    try:
        return Voronoi(points)
    except QhullError:
        jitter = 1e-9 * np.random.default_rng(0).standard_normal(points.shape)
        return Voronoi(points + jitter)


def lloyd_regularize_boundary_points(
    final_grid,
    x_min,
    y_min,
    resolution,
    threshold=None,
    max_points=200,
    max_iterations=30,
    epsilon=1e-3,
    subsample_step=None,
    bounding_box=None,
):
    """Regularize boundary samples with a projected Lloyd-style Voronoi iteration.

    Returns a dictionary with:
    - initial_points
    - final_points
    - contour_points
    - movement_history
    - spacing_variance_before
    - spacing_variance_after
    """

    grid = np.asarray(final_grid, dtype=float)
    nonzero = grid[grid > 0]
    if threshold is None or not np.isfinite(threshold):
        threshold = float(np.percentile(nonzero, 75)) if nonzero.size else 0.0

    contour_points = extract_boundary_points(grid, threshold, x_min, y_min, resolution)
    contour_points = order_points_by_angle(contour_points)
    contour_points = subsample_points(contour_points, max_points=max_points, step=subsample_step)

    if len(contour_points) == 0:
        return {
            "threshold": float(threshold),
            "initial_points": np.zeros((0, 2), dtype=float),
            "final_points": np.zeros((0, 2), dtype=float),
            "contour_points": np.zeros((0, 2), dtype=float),
            "movement_history": [],
            "spacing_variance_before": 0.0,
            "spacing_variance_after": 0.0,
            "voronoi": None,
        }

    if len(contour_points) < 3:
        final_points = contour_points.copy()
        return {
            "threshold": float(threshold),
            "initial_points": contour_points.copy(),
            "final_points": final_points,
            "contour_points": contour_points.copy(),
            "movement_history": [0.0],
            "spacing_variance_before": 0.0,
            "spacing_variance_after": 0.0,
            "voronoi": None,
        }

    initial_points = contour_points.copy()
    points = contour_points.copy()

    if bounding_box is None:
        x_max = x_min + grid.shape[0] * resolution
        y_max = y_min + grid.shape[1] * resolution
        bounding_box = (x_min, x_max, y_min, y_max)

    x_min_box, x_max_box, y_min_box, y_max_box = bounding_box
    movement_history = []
    vor = None

    for _ in range(int(max_iterations)):
        vor = _safe_voronoi(points)
        regions, vertices = _voronoi_finite_polygons_2d(vor, radius=2.0 * max(grid.shape) * resolution)
        new_points = points.copy()

        for idx, region in enumerate(regions):
            if len(region) < 3:
                new_points[idx] = project_to_boundary(points[idx], contour_points)
                continue

            polygon = vertices[region]
            polygon = _clip_polygon_to_box(polygon, x_min_box, x_max_box, y_min_box, y_max_box)
            if len(polygon) < 3:
                centroid = np.mean(points, axis=0)
            else:
                centroid = polygon_centroid(polygon)
            new_points[idx] = project_to_boundary(centroid, contour_points)

        movement = float(np.mean(np.linalg.norm(new_points - points, axis=1)))
        movement_history.append(movement)
        points = new_points
        if movement < float(epsilon):
            break

    # Equalize the spacing along the closed boundary without leaving it.
    points = _uniform_resample_closed_polyline(points, len(points))

    spacing_before = _spacing_variance_along_boundary(initial_points)
    spacing_after = _spacing_variance_along_boundary(points)

    return {
        "threshold": float(threshold),
        "initial_points": initial_points,
        "final_points": points,
        "contour_points": contour_points,
        "movement_history": movement_history,
        "spacing_variance_before": spacing_before,
        "spacing_variance_after": spacing_after,
        "voronoi": vor,
    }


def _spacing_variance_along_boundary(points):
    """Measure boundary spacing variability using consecutive arclength gaps."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return 0.0
    points = order_points_by_angle(points)
    closed = np.vstack([points, points[0]])
    seg_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    return float(np.var(seg_lengths))


def _uniform_resample_closed_polyline(points, num_points):
    """Resample a closed polyline at uniformly spaced arclength positions."""
    points = order_points_by_angle(points)
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return points
    if len(points) == 1 or num_points <= 1:
        return points[:1].copy()

    closed = np.vstack([points, points[0]])
    seg_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cumulative[-1]
    if total_length <= 1e-12:
        return np.repeat(points[:1], num_points, axis=0)

    targets = np.linspace(0.0, total_length, num_points, endpoint=False)
    resampled = []
    for target in targets:
        seg_idx = int(np.searchsorted(cumulative, target, side="right") - 1)
        seg_idx = min(max(seg_idx, 0), len(points) - 1)
        start = closed[seg_idx]
        end = closed[seg_idx + 1]
        seg_length = seg_lengths[seg_idx]
        if seg_length <= 1e-15:
            resampled.append(start)
            continue
        t = float((target - cumulative[seg_idx]) / seg_length)
        resampled.append(start + t * (end - start))
    return np.asarray(resampled, dtype=float)


def save_regularization_plots(result, output_prefix="boundary_regularization"):
    """Save initial/final scatter and convergence plots."""
    import matplotlib.pyplot as plt

    initial_points = np.asarray(result["initial_points"], dtype=float)
    final_points = np.asarray(result["final_points"], dtype=float)
    contour_points = np.asarray(result["contour_points"], dtype=float)
    movement_history = np.asarray(result["movement_history"], dtype=float)

    if len(contour_points):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(contour_points[:, 0], contour_points[:, 1], s=12, alpha=0.25, label="contour samples")
        if len(initial_points):
            ax.scatter(initial_points[:, 0], initial_points[:, 1], s=28, label="initial")
        if len(final_points):
            ax.scatter(final_points[:, 0], final_points[:, 1], s=28, label="regularized")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Boundary Point Regularization")
        ax.legend()
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_points.png", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    if len(movement_history):
        ax.plot(np.arange(1, len(movement_history) + 1), movement_history, marker="o")
    ax.set_title("Lloyd / CVT Convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean point movement")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_convergence.png", bbox_inches="tight")
    plt.close(fig)
