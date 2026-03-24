import argparse
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from environment import CircleOilSpill, SimulationMap
from simulation_engine import SimulationEngine
from visualization import Visualizer


def run_multi_drone_simulation(visualize=False, max_frames=5000, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    print(f"Random seed: {seed}")

    # 1) Environment
    sim_map = SimulationMap(xlim=(-5, 5), ylim=(-5, 5), grid_size=500)
    spill = CircleOilSpill(x0=0.0, y0=0.0)

    # 2) Engine
    engine = SimulationEngine(sim_map, spill, dt=0.05, sigma_gps=0.1, sigma_cam=0.1)

    # 3) Drone initialization (outside true circle)
    r0 = spill.r0
    for i in range(5):
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(r0 + 0.1, r0 + 2.5)
        x = spill.x0 + dist * np.cos(angle)
        y = spill.y0 + dist * np.sin(angle)
        x = np.clip(x, sim_map.xlim[0], sim_map.xlim[1])
        y = np.clip(y, sim_map.ylim[0], sim_map.ylim[1])
        engine.add_drone(drone_id=f"D{i}", x=x, y=y)

    # 4) One-time distributed consensus on radius
    engine.initial_consensus()

    # 5) Visualization setup
    viz = Visualizer(sim_map, spill) if visualize else None
    if not visualize:
        print("Visualization disabled. Headless mode (Agg backend).")

    print(f"Starting Multi-Drone Simulation ({max_frames} frames)...")
    print("Mode: one-time consensus -> Voronoi + continuous control law")

    # 6) Main loop
    try:
        for frame in range(max_frames):
            engine.step()
            if visualize:
                viz.render(engine.drones)
            if frame % 500 == 0:
                print(f"Frame {frame}/{max_frames}...")
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")

    # 7) Save final state
    print("Simulation finished. Saving final state to 'final_simulation_state.png'...")
    if not visualize:
        viz = Visualizer(sim_map, spill)
    viz.render(engine.drones)
    plt.savefig("final_simulation_state.png")

    # 8) Plot one-time consensus convergence history
    fig, ax = plt.subplots(figsize=(12, 6))
    history_len = len(engine.estimates_history[next(iter(engine.estimates_history))]["r0_consensus"])
    iterations = list(range(history_len))

    for drone_id, history in engine.estimates_history.items():
        ax.plot(iterations, history["r0_consensus"], label=drone_id, linewidth=2, alpha=0.75)

    ax.axhline(y=spill.r0, color="black", linestyle="--", linewidth=2.5, label="True r0")

    final_r0 = [engine.estimates_history[f"D{i}"]["r0_post"][-1] for i in range(len(engine.drones))]
    mean_r0 = np.mean(final_r0)
    std_r0 = np.std(final_r0)

    print("\n=== FINAL CONSENSUS RESULTS ===")
    print(f"Consensus iterations: {engine.initial_consensus_iterations}")
    for i, value in enumerate(final_r0):
        print(f"D{i} final r0: {value:.6f}")
    print(f"Shared r0 estimate: {engine.estimate_r0:.6f}")
    print(f"True r0: {spill.r0:.6f}")
    print(f"Std Dev across drones: {std_r0:.6f}")
    print(f"Absolute error: {abs(mean_r0 - spill.r0):.6f}")

    ax.set_title("Initial Distributed Radius Consensus")
    ax.set_xlabel("Consensus iteration")
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
    args = parser.parse_args()

    if not args.visualize:
        matplotlib.use("Agg")

    run_multi_drone_simulation(visualize=args.visualize, max_frames=args.frames, seed=args.seed)
