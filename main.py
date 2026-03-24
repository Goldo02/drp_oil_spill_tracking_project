import argparse
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from environment import CircleOilSpill, SimulationMap
from simulation_engine import SimulationEngine
from visualization import Visualizer


def run_multi_drone_simulation(visualize=False, max_frames=5000, seed=42, show_voronoi=False):
    np.random.seed(seed)
    random.seed(seed)
    print(f"Random seed: {seed}")

    sim_map = SimulationMap(xlim=(-5, 5), ylim=(-5, 5), grid_size=500)
    spill = CircleOilSpill(x0=0.0, y0=0.0)

    engine = SimulationEngine(
        sim_map,
        spill,
        dt=0.05,
        sigma_gps=0.1,
        sigma_cam=0.1,
        visualize_voronoi=show_voronoi,
    )

    # Random initialization outside circle
    for i in range(5):
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(spill.r0 + 0.1, spill.r0 + 2.5)
        x = np.clip(spill.x0 + dist * np.cos(angle), sim_map.xlim[0], sim_map.xlim[1])
        y = np.clip(spill.y0 + dist * np.sin(angle), sim_map.ylim[0], sim_map.ylim[1])
        engine.add_drone(drone_id=f"D{i}", x=x, y=y)

    print("Initial drone radius estimates:")
    for d in engine.drones:
        print(f"{d.drone_id}: r0 = {d.estimate_r0:.6f}")

    # NOTE: no one-shot consensus here; consensus runs frame-by-frame in step().

    viz = Visualizer(sim_map, spill) if visualize else None
    if not visualize:
        print("Visualization disabled. Headless mode (Agg backend).")

    print(f"Starting Multi-Drone Simulation ({max_frames} frames)...")
    print("Mode: frame-by-frame consensus + Voronoi + continuous control law")

    try:
        for frame in range(max_frames):
            engine.step()
            if visualize:
                viz.render(engine.drones)
            if frame % 500 == 0:
                print(f"Frame {frame}/{max_frames}...")
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")

    print("Simulation finished. Saving final state to 'final_simulation_state.png'...")
    if not visualize:
        viz = Visualizer(sim_map, spill)
    viz.render(engine.drones)
    plt.savefig("final_simulation_state.png")

    # Save final Voronoi partition image
    engine.save_voronoi_snapshot("final_voronoi_partition.png")
    print("Final Voronoi partition saved to 'final_voronoi_partition.png'")

    # Consensus convergence plot over frames
    fig, ax = plt.subplots(figsize=(12, 6))
    frames = list(range(len(engine.estimates_history[next(iter(engine.estimates_history))]["r0_consensus"])))

    for drone_id, history in engine.estimates_history.items():
        ax.plot(frames, history["r0_consensus"], label=drone_id, linewidth=2, alpha=0.75)

    ax.axhline(y=spill.r0, color="black", linestyle="--", linewidth=2.5, label="True r0")

    final_r0 = [engine.estimates_history[f"D{i}"]["r0_post"][-1] for i in range(len(engine.drones))]
    mean_r0 = np.mean(final_r0)
    std_r0 = np.std(final_r0)

    print("\n=== FINAL CONSENSUS RESULTS ===")
    for i, value in enumerate(final_r0):
        print(f"D{i} final r0: {value:.6f}")
    print(f"Mean r0: {mean_r0:.6f}")
    print(f"Std Dev: {std_r0:.6f}")
    print(f"True r0: {spill.r0:.6f}")
    print(f"Absolute error: {abs(mean_r0 - spill.r0):.6f}")

    ax.set_title("Frame-by-frame Distributed Radius Consensus")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Radius estimate")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig("consensus_convergence.png")
    print("Consensus convergence plot saved to 'consensus_convergence.png'")

    if visualize:
        print("Closing the window to exit.")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oil Spill Tracking Drone Simulation")
    parser.add_argument("--visualize", action="store_true", help="Enable real-time visualization")
    parser.add_argument("--frames", type=int, default=5000, help="Total number of simulation frames")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--show-voronoi", action="store_true", help="Show real-time Voronoi arc plot")
    args = parser.parse_args()

    if not args.visualize:
        matplotlib.use("Agg")

    run_multi_drone_simulation(
        visualize=args.visualize,
        max_frames=args.frames,
        seed=args.seed,
        show_voronoi=args.show_voronoi,
    )
