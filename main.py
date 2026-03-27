import argparse
import os
import random

import numpy as np

if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib

from environment import CircleOilSpill, SimulationMap
from simulation_engine import SimulationEngine


def run_multi_drone_simulation(
    visualize=False,
    max_frames=500,
    seed=42,
    fully_connected=False,
    communication_radius_cells=205,
    measure_every=3,
):
    if visualize:
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    np.random.seed(seed)
    random.seed(seed)
    print(f"Random seed: {seed}")

    # Global simulation domain used by all drones.
    sim_map = SimulationMap(xlim=(-5, 5), ylim=(-5, 5), grid_size=500)
    spill = CircleOilSpill(x0=0.0, y0=0.0, radius=2.0)

    engine = SimulationEngine(
        sim_map,
        spill,
        x_min=-10,
        x_max=10,
        y_min=-10,
        y_max=10,
        resolution=0.1,
        sensor_size=120,
        measure_every=measure_every,
        communication_radius_cells=communication_radius_cells,
        fully_connected=fully_connected,
        occupancy_threshold=0.5,
        temporal_alpha=0.05,
        consensus_rounds=10,
        verbose=True,
    )

    # Place the drones near the spill with random angles and a wider spread of
    # distances from the boundary. This makes the initial geometry less
    # regular and is useful for robustness tests.
    initial_radius = spill.radius
    num_drones = 5
    for i in range(num_drones):
        angle = np.random.uniform(0.0, 2.0 * np.pi)
        radial_offset = np.random.uniform(-0.5, 1.5)
        dist = max(0.1, initial_radius + radial_offset)
        start_x = spill.x0 + dist * np.cos(angle)
        start_y = spill.y0 + dist * np.sin(angle)
        start_x = np.clip(start_x, sim_map.xlim[0], sim_map.xlim[1])
        start_y = np.clip(start_y, sim_map.ylim[0], sim_map.ylim[1])
        engine.add_drone(drone_id=f"D{i}", x=start_x, y=start_y, gps_noise=0.03, camera_noise=0.03)

    viz = None
    if visualize:
        from visualization import Visualizer

        viz = Visualizer(
            sim_map,
            spill,
            communication_radius=None if fully_connected else engine.communication_radius,
            show_communication_radius=not fully_connected,
        )
        plt.show(block=False)
    else:
        print("Visualization disabled. Headless mode (Agg backend).")

    print(f"Starting distributed occupancy grid simulation ({max_frames} frames)...")
    print(f"Measurement interval: every {measure_every} frames")
    print(f"Consensus iterations per measurement: {engine.consensus_rounds}")
    if fully_connected:
        print("Mode: full-grid consensus on a fully connected communication graph")
    else:
        print(
            "Mode: full-grid consensus with range-based communication "
            f"(Rc={communication_radius_cells} cells, ~{engine.communication_radius:.2f} world units)"
        )

    try:
        for frame in range(max_frames):
            error = engine.step()

            if visualize:
                viz.render(engine.drones)

            if frame % 50 == 0:
                print(f"Frame {frame}/{max_frames} | disagreement error={error:.6f}")
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")

    engine.finalize_histories()

    print("Simulation finished.")

    if visualize:
        print("Saving the final scene to 'final_simulation_state.png'...")
        plt.savefig("final_simulation_state.png", bbox_inches="tight")

    error_history = np.asarray(engine.error_history, dtype=float)
    fig, ax = plt.subplots(figsize=(12, 5))

    measurement_history = engine.measurement_consensus_history
    color_cycle = plt.cm.tab10(np.linspace(0, 1, max(1, len(engine.drones))))

    if measurement_history:
        for measure_idx, cycle_trace in enumerate(measurement_history, start=1):
            # Each cycle is plotted on [measure_idx - 1, measure_idx] so the
            # x-axis reads as "number of measurements + 1".
            cycle_length = len(next(iter(cycle_trace.values())))
            x_values = np.linspace(measure_idx - 1.0, measure_idx, cycle_length)
            for drone_idx, drone in enumerate(engine.drones):
                drone_id = drone.drone_id
                y_values = np.asarray(cycle_trace[drone_id], dtype=float)
                ax.plot(
                    x_values,
                    y_values,
                    color=color_cycle[drone_idx % len(color_cycle)],
                    linewidth=1.8,
                    marker="o",
                    markersize=3,
                    alpha=0.9,
                    label=drone_id if measure_idx == 1 else None,
                )

        measurement_count = len(measurement_history)
        ax.set_xlim(0, measurement_count)
        ax.set_xticks(np.arange(0, measurement_count + 1, 1))
        ax.set_title("Consensus Convergence Between Measurements")
        ax.set_xlabel("Number of measurements + 1")
        ax.set_ylabel("Grid disagreement error")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    else:
        ax.plot(np.arange(1, len(error_history) + 1), error_history, linewidth=2.0, color="black")
        ax.set_title("Consensus Disagreement Error")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Error")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("consensus_convergence.png", bbox_inches="tight")
    print("Consensus convergence plot saved to 'consensus_convergence.png'")

    final_grid = engine.compute_mean_grid()
    max_value = float(np.max(final_grid)) if final_grid.size else 1.0
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        final_grid.T,
        origin="lower",
        cmap="Greys",
        vmin=0.0,
        vmax=max(1.0, max_value),
    )
    ax.set_title("Final Occupancy Grid")
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("final_occupancy_grid.png", bbox_inches="tight")
    print("Final occupancy grid saved to 'final_occupancy_grid.png'")

    if error_history.size:
        print("\n=== FINAL CONSENSUS RESULTS ===")
        print(f"Initial error: {error_history[0]:.6f}")
        print(f"Final error: {error_history[-1]:.6f}")
        print(f"Minimum error: {float(np.min(error_history)):.6f}")

    if visualize:
        print("Closing the window to exit.")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed oil spill occupancy grid mapping simulation")
    parser.add_argument("--visualize", action="store_true", help="Enable real-time visualization")
    parser.add_argument("--frames", type=int, default=500, help="Total number of simulation iterations")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducible initialization")
    parser.add_argument(
        "--fully-connected",
        action="store_true",
        help="Disable range-based communication and use a fully connected consensus graph",
    )
    parser.add_argument(
        "--range-based",
        action="store_true",
        help="Use range-based communication instead of the default fully connected consensus",
    )
    parser.add_argument(
        "--communication-radius-cells",
        type=int,
        default=205,
        help="Communication radius in grid cells when range-based communication is enabled",
    )
    parser.add_argument(
        "--measure-every",
        type=int,
        default=3,
        help="Perform a new sensing update every N frames",
    )
    args = parser.parse_args()

    run_multi_drone_simulation(
        visualize=args.visualize,
        max_frames=args.frames,
        seed=args.seed,
        fully_connected=(args.fully_connected and not args.range_based),
        communication_radius_cells=args.communication_radius_cells,
        measure_every=args.measure_every,
    )
