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


def _set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    print(f"Random seed: {seed}")


def _configure_matplotlib(visualize):
    try:
        matplotlib.use("TkAgg" if visualize else "Agg")
    except ImportError as exc:
        if not visualize:
            raise
        print(f"Interactive visualization unavailable ({exc}); using Agg backend.")
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _build_environment(
    oil_shape,
    seed,
    polygon_vertices,
    polygon_r0,
    polygon_smoothness,
    polygon_x0,
    polygon_y0,
    polygon_continuous,
):
    sim_map = SimulationMap(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0), grid_size=500)

    if oil_shape == "circle":
        spill = CircleOilSpill(x0=0.0, y0=0.0, radius=2.0)
    else:
        spill = SmoothedPolygonOilSpill(
            sim_map.X,
            sim_map.Y,
            n_vertices=polygon_vertices,
            r0=polygon_r0,
            smoothness=polygon_smoothness,
            x0=polygon_x0,
            y0=polygon_y0,
            seed=seed,
            continuous=polygon_continuous,
        )

    return sim_map, spill


def _communication_radius(sim_map, communication_radius_cells):
    return communication_radius_cells * 0.5 * (abs(sim_map.dx) + abs(sim_map.dy))


def _build_engine(
    sim_map,
    spill,
    communication_radius_cells,
    measure_every,
    fully_connected,
    dt,
):
    return SimulationEngine(
        sim_map=sim_map,
        oil_spill=spill,
        x_min=-10.0,
        x_max=10.0,
        y_min=-10.0,
        y_max=10.0,
        resolution=0.1,
        sensor_size=120,
        measure_every=measure_every,
        communication_radius_cells=communication_radius_cells,
        fully_connected=fully_connected,
        occupancy_threshold=0.5,
        temporal_alpha=0.05,
        consensus_rounds=10,
        dt=dt,
        verbose=True,
    )


def _add_drones(engine, sim_map, num_drones):
    for i in range(num_drones):
        engine.add_drone(
            drone_id=f"D{i}",
            x=np.random.uniform(*sim_map.xlim),
            y=np.random.uniform(*sim_map.ylim),
            gps_noise=0.03,
            camera_noise=0.03,
        )


def _build_visualizer(sim_map, spill, communication_radius, fully_connected, show_nls_points):
    return Visualizer(
        sim_map=sim_map,
        oil_spill=spill,
        communication_radius=None if fully_connected else communication_radius,
        show_communication_radius=not fully_connected,
        show_nls_points=show_nls_points,
    )


def _print_run_header(
    max_frames,
    oil_shape,
    spill,
    polygon_vertices,
    polygon_r0,
    polygon_smoothness,
    polygon_continuous,
    measure_every,
    consensus_rounds,
    fully_connected,
    communication_radius,
):
    print(f"Starting distributed occupancy grid simulation ({max_frames} frames)...")
    print(f"Oil shape: {oil_shape}")

    if oil_shape != "circle":
        print(
            "Polygon parameters: "
            f"vertices={polygon_vertices}, "
            f"r0={polygon_r0}, "
            f"smoothness={polygon_smoothness}, "
            f"center=({spill.x0:.2f}, {spill.y0:.2f}), "
            f"continuous={polygon_continuous}"
        )

    print(f"Measurement interval: every {measure_every} frames")
    print(f"Consensus iterations per measurement: {consensus_rounds}")
    print(
        "Mode: fully connected consensus"
        if fully_connected
        else f"Mode: range-based communication (Rc={communication_radius:.2f})"
    )


def _run_frames(engine, visualizer, visualize, max_frames):
    try:
        for frame in range(max_frames):
            error = engine.step()

            if visualize:
                visualizer.render(engine.get_visualization_data())

            if frame % 50 == 0:
                print(f"Frame {frame}/{max_frames} | disagreement error={error:.6f}")
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")


def _save_outputs(engine, visualizer):
    visualizer.render(engine.get_visualization_data(), pause=False)
    visualizer.save_final_state("final_simulation_state.png", directory=OUTPUT_DIR)
    visualizer.plot_consensus_convergence(
        engine,
        "consensus_convergence.png",
        directory=OUTPUT_DIR,
    )
    visualizer.plot_final_occupancy_grid(
        engine.compute_mean_grid(),
        "final_occupancy_grid.png",
        directory=OUTPUT_DIR,
        alpha=1.0,
    )
    visualizer.save_final_occupancy_grid_per_robot(
        engine.drones,
        directory=OUTPUT_DIR,
        alpha=1.0,
    )


def _ordered_points_by_angle(points):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) <= 2:
        return points.copy()

    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    return points[np.argsort(angles)].copy()


def _mapped_boundary_points_from_consensus(engine):
    mean_grid = np.asarray(engine.compute_mean_grid(), dtype=float)
    occupied = np.argwhere(mean_grid > float(engine.occupancy_threshold))
    if occupied.size == 0:
        return np.empty((0, 2), dtype=float)

    points = np.column_stack(
        (
            engine.x_min + (occupied[:, 0] + 0.5) * engine.resolution,
            engine.y_min + (occupied[:, 1] + 0.5) * engine.resolution,
        )
    )
    return _ordered_points_by_angle(points)


def _model_boundary_points(spill):
    if hasattr(spill, "boundary"):
        points = np.asarray(spill.boundary, dtype=float)
        if points.ndim == 2 and points.shape[1] == 2:
            return points.copy()

    if all(hasattr(spill, attr) for attr in ("x0", "y0", "radius")):
        theta = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
        return np.column_stack(
            (
                float(spill.x0) + float(spill.radius) * np.cos(theta),
                float(spill.y0) + float(spill.radius) * np.sin(theta),
            )
        )

    return np.empty((0, 2), dtype=float)


def _save_oil_mapping(engine, sim_map, spill, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = _mapped_boundary_points_from_consensus(engine)
    source = "consensus_mean_grid"

    if points.shape[0] < 3:
        points = _model_boundary_points(spill)
        source = "oil_spill_model_boundary"

    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if points.shape[0] < 3:
        raise RuntimeError("Unable to export oil mapping: fewer than 3 boundary points.")

    np.save(output_path, points)

    metadata = {
        "data_file": output_path.name,
        "source": source,
        "coordinate_frame": "world",
        "units": "simulation_world_units",
        "shape": list(points.shape),
        "dtype": str(points.dtype),
        "closed_boundary": True,
        "ordered_boundary": True,
        "occupancy_threshold": float(engine.occupancy_threshold),
        "sim_map": {
            "xlim": list(map(float, sim_map.xlim)),
            "ylim": list(map(float, sim_map.ylim)),
            "grid_size": int(sim_map.grid_size),
            "dx": float(sim_map.dx),
            "dy": float(sim_map.dy),
        },
        "mapping_grid": {
            "bounds": list(map(float, engine.grid_bounds)),
            "resolution": float(engine.resolution),
            "shape": list(map(int, engine.grid_shape)),
        },
    }

    metadata_path = output_path.with_name(OIL_MAPPING_METADATA_FILENAME)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        "Saved oil mapping: "
        f"{output_path} ({points.shape[0]} world-coordinate boundary points, "
        f"source={source})"
    )


def _print_final_diagnostics(engine):
    error_history = np.asarray(engine.error_history, dtype=float)
    if not error_history.size:
        return

    print("\n=== FINAL CONSENSUS RESULTS ===")
    print(f"Initial error: {error_history[0]:.6f}")
    print(f"Final error: {error_history[-1]:.6f}")
    print(f"Minimum error: {float(np.min(error_history)):.6f}")


def run_simulation(
    visualize=False,
    max_frames=500,
    seed=42,
    num_drones=5,
    oil_shape="smoothed_polygon",
    fully_connected=False,
    communication_radius_cells=205,
    measure_every=3,
    show_nls_points=False,
    polygon_vertices=36,
    polygon_r0=2.5,
    polygon_smoothness=0.2,
    polygon_x0=None,
    polygon_y0=None,
    polygon_continuous=False,
    dt=1.0,
    oil_mapping_output=OIL_MAPPING_DATA_PATH,
):
    _set_random_seed(seed)
    plt = _configure_matplotlib(visualize)

    sim_map, spill = _build_environment(
        oil_shape,
        seed,
        polygon_vertices,
        polygon_r0,
        polygon_smoothness,
        polygon_x0,
        polygon_y0,
        polygon_continuous,
    )
    communication_radius = _communication_radius(sim_map, communication_radius_cells)
    engine = _build_engine(
        sim_map,
        spill,
        communication_radius_cells,
        measure_every,
        fully_connected,
        dt,
    )
    _add_drones(engine, sim_map, num_drones)

    visualizer = _build_visualizer(
        sim_map,
        spill,
        communication_radius,
        fully_connected,
        show_nls_points,
    )

    if visualize:
        plt.show(block=False)
    else:
        print("Visualization disabled. Headless mode (Agg backend).")

    _print_run_header(
        max_frames,
        oil_shape,
        spill,
        polygon_vertices,
        polygon_r0,
        polygon_smoothness,
        polygon_continuous,
        measure_every,
        engine.consensus_rounds,
        fully_connected,
        communication_radius,
    )
    _run_frames(engine, visualizer, visualize, max_frames)

    engine.finalize_histories()
    print("Simulation finished.")

    _save_outputs(engine, visualizer)
    _save_oil_mapping(engine, sim_map, spill, oil_mapping_output)
    _print_final_diagnostics(engine)

    if visualize:
        print("Closing the window to exit.")
        plt.show()


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Distributed oil spill occupancy grid mapping simulation"
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-drones", type=int, default=5)
    parser.add_argument(
        "--oil-shape",
        choices=("circle", "smoothed_polygon"),
        default="smoothed_polygon",
    )
    parser.add_argument("--polygon-vertices", type=int, default=36)
    parser.add_argument("--polygon-r0", type=float, default=2.5)
    parser.add_argument("--polygon-smoothness", type=float, default=0.2)
    parser.add_argument("--polygon-x0", type=float, default=None)
    parser.add_argument("--polygon-y0", type=float, default=None)
    parser.add_argument("--polygon-continuous", action="store_true")
    parser.add_argument("--fully-connected", action="store_true")
    parser.add_argument("--range-based", action="store_true")
    parser.add_argument("--communication-radius-cells", type=int, default=250)
    parser.add_argument("--measure-every", type=int, default=3)
    parser.add_argument("--show-nls-points", action="store_true")
    parser.add_argument("--dt", type=float, default=1.0, help="Simulation timestep.")
    parser.add_argument(
        "--oil-mapping-output",
        default=OIL_MAPPING_DATA_PATH,
        help="Path for exported (N, 2) oil boundary points.",
    )
    return parser


def main():
    args = _build_parser().parse_args()
    run_simulation(
        visualize=args.visualize,
        max_frames=args.frames,
        seed=args.seed,
        num_drones=args.num_drones,
        oil_shape=args.oil_shape,
        fully_connected=args.fully_connected and not args.range_based,
        communication_radius_cells=args.communication_radius_cells,
        measure_every=args.measure_every,
        show_nls_points=args.show_nls_points,
        polygon_vertices=args.polygon_vertices,
        polygon_r0=args.polygon_r0,
        polygon_smoothness=args.polygon_smoothness,
        polygon_x0=args.polygon_x0,
        polygon_y0=args.polygon_y0,
        polygon_continuous=args.polygon_continuous,
        dt=args.dt,
        oil_mapping_output=args.oil_mapping_output,
    )


if __name__ == "__main__":
    main()
