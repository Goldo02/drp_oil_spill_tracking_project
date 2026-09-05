import argparse
import os
import random

import numpy as np

if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib


from environment import (
    CircleOilSpill,
    SimulationMap,
    SmoothedPolygonOilSpill,
)

from controller import Controller
from simulation_engine import SimulationEngine
from visualization import Visualizer


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
):
    # ==================================================================
    # RANDOM SEED
    # ==================================================================

    np.random.seed(seed)
    random.seed(seed)

    print(f"Random seed: {seed}")

    # ==================================================================
    # MATPLOTLIB
    # ==================================================================

    if visualize:
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    # ==================================================================
    # ENVIRONMENT
    # ==================================================================

    sim_map = SimulationMap(
        xlim=(-5.0, 5.0),
        ylim=(-5.0, 5.0),
        grid_size=500,
    )

    if oil_shape == "circle":

        spill = CircleOilSpill(
            x0=0.0,
            y0=0.0,
            radius=2.0,
        )

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

    # ==================================================================
    # CONTROLLER
    # ==================================================================

    dx = sim_map.dx
    dy = sim_map.dy

    communication_radius = (
        communication_radius_cells
        * 0.5
        * (abs(dx) + abs(dy))
    )

    controller = Controller(
        sim_map=sim_map,
        communication_radius=communication_radius,
        fully_connected=fully_connected,
        occupancy_threshold=0.5,
        resolution=0.1,
    )

    # ==================================================================
    # SIMULATION ENGINE
    # ==================================================================

    engine = SimulationEngine(
        sim_map=sim_map,
        oil_spill=spill,
        controller=controller,

        x_min=-10.0,
        x_max=10.0,
        y_min=-10.0,
        y_max=10.0,

        resolution=0.1,

        sensor_size=120,
        measure_every=measure_every,

        communication_radius_cells=(
            communication_radius_cells
        ),

        fully_connected=fully_connected,

        occupancy_threshold=0.5,
        temporal_alpha=0.05,
        consensus_rounds=10,

        dt=dt,

        verbose=True,
    )

    # ==================================================================
    # DRONE INITIALIZATION
    # ==================================================================

    initial_radius = spill.radius

    for i in range(num_drones):

        start_x = np.random.uniform(
            sim_map.xlim[0],
            sim_map.xlim[1],
        )

        start_y = np.random.uniform(
            sim_map.ylim[0],
            sim_map.ylim[1],
        )

        engine.add_drone(
            drone_id=f"D{i}",
            x=start_x,
            y=start_y,
            gps_noise=0.03,
            camera_noise=0.03,
        )

    # ==================================================================
    # VISUALIZATION
    # ==================================================================

    visualizer = Visualizer(
        sim_map=sim_map,
        oil_spill=spill,
        communication_radius=(
            None
            if fully_connected
            else communication_radius
        ),
        show_communication_radius=(
            not fully_connected
        ),
        show_nls_points=show_nls_points,
    )

    if visualize:
        plt.show(block=False)
    else:
        print(
            "Visualization disabled. "
            "Headless mode (Agg backend)."
        )

    # ==================================================================
    # SIMULATION
    # ==================================================================

    print(
        f"Starting distributed occupancy "
        f"grid simulation "
        f"({max_frames} frames)..."
    )

    print(
        f"Oil shape: {oil_shape}"
    )

    if oil_shape != "circle":

        print(
            "Polygon parameters: "
            f"vertices={polygon_vertices}, "
            f"r0={polygon_r0}, "
            f"smoothness={polygon_smoothness}, "
            f"center=({spill.x0:.2f}, "
            f"{spill.y0:.2f}), "
            f"continuous={polygon_continuous}"
        )

    print(
        f"Measurement interval: "
        f"every {measure_every} frames"
    )

    print(
        f"Consensus iterations per measurement: "
        f"{engine.consensus_rounds}"
    )

    if fully_connected:
        print(
            "Mode: fully connected consensus"
        )
    else:
        print(
            "Mode: range-based communication "
            f"(Rc={communication_radius:.2f})"
        )

    # ==================================================================
    # RUN
    # ==================================================================

    try:

        for frame in range(max_frames):

            error = engine.step()

            if visualize:

                visualizer.render(
                    engine.get_visualization_data()
                )

            if frame % 50 == 0:

                print(
                    f"Frame {frame}/"
                    f"{max_frames} | "
                    f"disagreement error="
                    f"{error:.6f}"
                )

    except KeyboardInterrupt:

        print(
            "Simulation interrupted by user."
        )

    engine.finalize_histories()

    print(
        "Simulation finished."
    )

    # ==================================================================
    # FINALIZATION & SAVING (Delegato al Visualizer)
    # ==================================================================

    # 1. Render e salvataggio dello stato finale dello scenario
    final_data = engine.get_visualization_data()
    visualizer.render(final_data, pause=False)
    visualizer.save_final_state("final_simulation_state.png")

    # 2. Generazione e salvataggio del grafico di convergenza del consenso
    visualizer.plot_consensus_convergence(engine, "consensus_convergence.png")

    # 3. Generazione e salvataggio della griglia di occupazione finale
    final_grid = engine.compute_mean_grid()
    visualizer.plot_final_occupancy_grid(final_grid, "final_occupancy_grid.png")

    # ==================================================================
    # FINAL DIAGNOSTICS
    # ==================================================================

    error_history = np.asarray(
        engine.error_history,
        dtype=float,
    )

    if error_history.size:

        print(
            "\n=== FINAL CONSENSUS RESULTS ==="
        )

        print(
            f"Initial error: "
            f"{error_history[0]:.6f}"
        )

        print(
            f"Final error: "
            f"{error_history[-1]:.6f}"
        )

        print(
            f"Minimum error: "
            f"{float(np.min(error_history)):.6f}"
        )

    if visualize:

        print(
            "Closing the window to exit."
        )

        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Distributed oil spill "
            "occupancy grid mapping simulation"
        )
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num-drones",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--oil-shape",
        choices=(
            "circle",
            "smoothed_polygon",
        ),
        default="smoothed_polygon",
    )

    parser.add_argument(
        "--polygon-vertices",
        type=int,
        default=36,
    )

    parser.add_argument(
        "--polygon-r0",
        type=float,
        default=2.5,
    )

    parser.add_argument(
        "--polygon-smoothness",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--polygon-x0",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--polygon-y0",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--polygon-continuous",
        action="store_true",
    )

    parser.add_argument(
        "--fully-connected",
        action="store_true",
    )

    parser.add_argument(
        "--range-based",
        action="store_true",
    )

    parser.add_argument(
        "--communication-radius-cells",
        type=int,
        default=205,
    )

    parser.add_argument(
        "--measure-every",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--show-nls-points",
        action="store_true",
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Simulation timestep.",
    )

    args = parser.parse_args()

    run_simulation(
        visualize=args.visualize,
        max_frames=args.frames,
        seed=args.seed,
        num_drones=args.num_drones,
        oil_shape=args.oil_shape,

        fully_connected=(
            args.fully_connected
            and not args.range_based
        ),

        communication_radius_cells=(
            args.communication_radius_cells
        ),

        measure_every=args.measure_every,

        show_nls_points=(
            args.show_nls_points
        ),

        polygon_vertices=(
            args.polygon_vertices
        ),

        polygon_r0=args.polygon_r0,

        polygon_smoothness=(
            args.polygon_smoothness
        ),

        polygon_x0=args.polygon_x0,
        polygon_y0=args.polygon_y0,

        polygon_continuous=(
            args.polygon_continuous
        ),

        dt=args.dt,
    )
