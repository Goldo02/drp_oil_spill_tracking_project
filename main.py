import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib

from environment import CircleOilSpill, SimulationMap, SmoothedPolygonOilSpill
from simulation_engine import SimulationEngine
from visualization import Visualizer


OUTPUT_DIR = "./tmp_output"
OIL_MAPPING_DATA_PATH = os.path.join(OUTPUT_DIR, "oil_mapping_data.npy")
OIL_MAPPING_METADATA_FILENAME = "oil_mapping_metadata.json"


def _load_oil_mapping(path, sim_map):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Oil mapping file not found: {path}. "
            "Run the mapping branch first or pass --oil-mapping-input."
        )

    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        try:
            points = data["points"]
        finally:
            data.close()
    else:
        points = data

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"Oil mapping must have shape (N, 2); got {points.shape} from {path}"
        )
    if points.shape[0] < 3:
        raise ValueError("Oil mapping must contain at least 3 boundary points")
    if not np.all(np.isfinite(points)):
        raise ValueError("Oil mapping contains non-finite coordinates")

    metadata = {}
    metadata_path = path.with_name(OIL_MAPPING_METADATA_FILENAME)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("coordinate_frame") not in (None, "world"):
            raise ValueError(
                "Oil mapping metadata is not in world coordinates: "
                f"{metadata.get('coordinate_frame')!r}"
            )
        if metadata.get("shape") is not None and list(points.shape) != metadata["shape"]:
            raise ValueError(
                f"Oil mapping metadata shape {metadata['shape']} does not match "
                f"loaded data shape {list(points.shape)}"
            )

    x_min, x_max = map(float, sim_map.xlim)
    y_min, y_max = map(float, sim_map.ylim)
    tolerance = 0.5 * max(abs(float(sim_map.dx)), abs(float(sim_map.dy)), 1e-12)
    outside = (
        (points[:, 0] < x_min - tolerance)
        | (points[:, 0] > x_max + tolerance)
        | (points[:, 1] < y_min - tolerance)
        | (points[:, 1] > y_max + tolerance)
    )
    if np.any(outside):
        bad = points[int(np.flatnonzero(outside)[0])]
        raise ValueError(
            "Oil mapping appears to use a different coordinate frame or scale; "
            f"point ({bad[0]:.3f}, {bad[1]:.3f}) is outside sim_map "
            f"xlim={sim_map.xlim}, ylim={sim_map.ylim}."
        )

    print(f"Loaded oil mapping: {path} ({points.shape[0]} world-coordinate boundary points)")
    return points, metadata


def _assert_loaded_mapping_is_active(engine, loaded_points):
    loaded_points = np.asarray(loaded_points, dtype=float)
    checks = [
        ("engine.world_boundary_points", getattr(engine, "world_boundary_points", None)),
        (
            "engine.controller.known_boundary_points",
            getattr(engine.controller, "known_boundary_points", None),
        ),
    ]

    for name, points in checks:
        if points is None or not np.array_equal(np.asarray(points, dtype=float), loaded_points):
            raise RuntimeError(f"{name} is not using the loaded oil mapping boundary.")

    for drone in engine.drones:
        drone_points = getattr(drone, "known_boundary_points", None)
        controller_points = getattr(drone.controller, "known_boundary_points", None)
        if drone_points is None or not np.array_equal(np.asarray(drone_points, dtype=float), loaded_points):
            raise RuntimeError(f"{drone.drone_id} is not using the loaded oil mapping boundary.")
        if controller_points is None or not np.array_equal(np.asarray(controller_points, dtype=float), loaded_points):
            raise RuntimeError(
                f"{drone.drone_id} controller is not using the loaded oil mapping boundary."
            )


def run_simulation(
    visualize=False,
    max_frames=500,
    seed=42,
    num_drones=5,
    oil_shape="smoothed_polygon",
    fully_connected=False,
    communication_radius_cells=205,
    polygon_vertices=36,
    polygon_r0=2.5,
    polygon_smoothness=0.2,
    polygon_x0=None,
    polygon_y0=None,
    polygon_continuous=False,
    load_oil_mapping=False,
    oil_mapping_input=OIL_MAPPING_DATA_PATH,
):
    # Random seed & environment setup
    np.random.seed(seed)
    random.seed(seed)
    print(f"Random seed: {seed}")

    try:
        matplotlib.use("TkAgg" if visualize else "Agg")
    except ImportError as exc:
        if not visualize:
            raise
        print(f"Interactive visualization unavailable ({exc}); using Agg backend.")
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sim_map = SimulationMap(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0), grid_size=500)

    if oil_shape == "circle":
        spill = CircleOilSpill(x0=0.0, y0=0.0, radius=2.0)
    else:
        spill = SmoothedPolygonOilSpill(
            sim_map.X, sim_map.Y,
            n_vertices=polygon_vertices, r0=polygon_r0,
            smoothness=polygon_smoothness, x0=polygon_x0,
            y0=polygon_y0, seed=seed, continuous=polygon_continuous,
        )

    dx, dy = sim_map.dx, sim_map.dy
    communication_radius = communication_radius_cells * 0.5 * (abs(dx) + abs(dy))

    engine = SimulationEngine(
        sim_map=sim_map,
        oil_spill=spill,
        x_min=-10.0, x_max=10.0,
        y_min=-10.0, y_max=10.0,
        resolution=0.1,
        communication_radius_cells=communication_radius_cells,
        verbose=True,
    )

    mapping_metadata = {}
    loaded_boundary_points = None
    if load_oil_mapping:
        boundary_points, mapping_metadata = _load_oil_mapping(oil_mapping_input, sim_map)
        loaded_boundary_points = boundary_points.copy()
        engine.initialize_loaded_boundary(
            boundary_points,
            known_boundary_closed=bool(mapping_metadata.get("closed_boundary", True)),
            already_ordered=bool(mapping_metadata.get("ordered_boundary", True)),
        )
    else:
        engine.initialize_world_boundary()

    engine.spawn_drones_on_boundary(num_drones, rng=np.random.default_rng(seed))
    if load_oil_mapping:
        _assert_loaded_mapping_is_active(engine, loaded_boundary_points)

    visualizer = Visualizer(
        sim_map=sim_map,
        oil_spill=spill,
        communication_radius=None if fully_connected else communication_radius,
        show_communication_radius=not fully_connected,
    )

    if visualize:
        plt.show(block=False)
    else:
        print("Visualization disabled. Headless mode (Agg backend).")

    # Simulation Info & Log
    print(f"Starting distributed occupancy grid simulation ({max_frames} frames)...")
    print(f"Oil shape: {oil_shape}")
    if oil_shape != "circle":
        print(f"Polygon parameters: vertices={polygon_vertices}, r0={polygon_r0}, smoothness={polygon_smoothness}, center=({spill.x0:.2f}, {spill.y0:.2f}), continuous={polygon_continuous}")
    if load_oil_mapping:
        print(f"Mode: loaded oil mapping boundary control (Rc={communication_radius:.2f})")
    else:
        print(f"Mode: static boundary control (Rc={communication_radius:.2f})")

    # Main Simulation Loop
    try:
        for frame in range(max_frames):
            engine.step()
            if visualize:
                visualizer.render(engine.get_visualization_data())
            if frame % 50 == 0:
                print(f"Frame {frame}/{max_frames}")
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")

    print("Simulation finished.")

    # Finalization & Saving
    output_dir = "./tmp_output"
    visualizer.render(engine.get_visualization_data(), pause=False)
    visualizer.save_final_state("final_simulation_state.png", directory=output_dir)

    final_grid = engine.compute_mean_grid()
    visualizer.plot_final_occupancy_grid(final_grid, "final_occupancy_grid.png", directory=output_dir, alpha=1.0)
    visualizer.save_final_occupancy_grid_per_robot(engine.drones, directory=output_dir, alpha=1.0)

    if visualize:
        print("Closing the window to exit.")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed oil spill occupancy grid mapping simulation")
    
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-drones", type=int, default=5)
    parser.add_argument("--oil-shape", choices=("circle", "smoothed_polygon"), default="smoothed_polygon")
    parser.add_argument("--polygon-vertices", type=int, default=36)
    parser.add_argument("--polygon-r0", type=float, default=2.5)
    parser.add_argument("--polygon-smoothness", type=float, default=0.2)
    parser.add_argument("--polygon-x0", type=float, default=None)
    parser.add_argument("--polygon-y0", type=float, default=None)
    parser.add_argument("--polygon-continuous", action="store_true")
    parser.add_argument("--fully-connected", action="store_true")
    parser.add_argument("--range-based", action="store_true")
    parser.add_argument("--communication-radius-cells", type=int, default=250)
    parser.add_argument(
        "--load-oil-mapping",
        action="store_true",
        help="Load precomputed oil boundary points from --oil-mapping-input.",
    )
    parser.add_argument(
        "--oil-mapping-input",
        default=OIL_MAPPING_DATA_PATH,
        help="Path to an exported (N, 2) oil boundary .npy file.",
    )

    args = parser.parse_args()

    run_simulation(
        visualize=args.visualize,
        max_frames=args.frames,
        seed=args.seed,
        num_drones=args.num_drones,
        oil_shape=args.oil_shape,
        fully_connected=(args.fully_connected and not args.range_based),
        communication_radius_cells=args.communication_radius_cells,
        polygon_vertices=args.polygon_vertices,
        polygon_r0=args.polygon_r0,
        polygon_smoothness=args.polygon_smoothness,
        polygon_x0=args.polygon_x0,
        polygon_y0=args.polygon_y0,
        polygon_continuous=args.polygon_continuous,
        load_oil_mapping=args.load_oil_mapping,
        oil_mapping_input=args.oil_mapping_input,
    )
