import os
import numpy as np
import argparse
import random

if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib
from environment import SimulationMap, GaussianOilSpill, CircleOilSpill
from simulation_engine import SimulationEngine

def run_multi_drone_simulation(
    visualize=False,
    max_frames=5000,
    seed=42,
    fully_connected=False,
    communication_radius_cells=205,
):
    if visualize:
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")

    # Seed both numpy and python random for full reproducibility
    np.random.seed(seed)
    random.seed(seed)
    print(f"Random seed: {seed}")
    # 1. Setup Environment
    sim_map = SimulationMap(xlim=(-5, 5), ylim=(-5, 5), grid_size=500)
    spill = CircleOilSpill(x0=0.0, y0=0.0)
    # spill = GaussianOilSpill(x0=0.0, y0=0.0, sigma=1.5, amplitude=1.0)
    
    # 2. Setup Engine
    engine = SimulationEngine(
        sim_map,
        spill,
        dt=0.05,
        sigma_gps=0.1,
        sigma_cam=0.1,
        communication_radius_cells=communication_radius_cells,
        fully_connected=fully_connected,
    )
    
    # 3. Add Drones with random start positions outside the circle, not too far
    r0 = spill.r0
    for i in range(5):
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(r0 + 0.1, r0 + 2.5)
        start_x = spill.x0 + dist * np.cos(angle)
        start_y = spill.y0 + dist * np.sin(angle)
        # Ensure within map bounds
        start_x = np.clip(start_x, sim_map.xlim[0], sim_map.xlim[1])
        start_y = np.clip(start_y, sim_map.ylim[0], sim_map.ylim[1])
        engine.add_drone(drone_id=f"D{i}", x=start_x, y=start_y)
    
    # 4. Setup Visualization (only if enabled)
    viz = None
    if visualize:
        from visualization import Visualizer
        viz = Visualizer(
            sim_map,
            spill,
            communication_radius=None if fully_connected else engine.communication_radius,
            show_communication_radius=not fully_connected,
        )
    else:
        print("Visualization disabled. Headless mode (Agg backend).")
    
    print(f"Starting Multi-Drone Simulation ({max_frames} frames)...")
    if fully_connected:
        print("Mode: Static radius estimation + fully connected consensus")
    else:
        print(
            "Mode: Static radius estimation + range-based communication "
            f"(Rc={communication_radius_cells} cells, ~{engine.communication_radius:.2f} world units)"
        )
    
    # 5. Main Loop
    try:
        for frame in range(max_frames):
            # Update Logic
            engine.step()
            
            # Update Visualization
            if visualize:
                viz.render(engine.drones)
            
            # Periodic status
            if frame % 500 == 0:
                print(f"Frame {frame}/{max_frames}...")
                
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    
    print("Simulation finished. Saving final state to 'final_simulation_state.png'...")
    # Create a final plot even if visualization was disabled
    if not visualize:
        from visualization import Visualizer
        # Late setup of Visualizer for static final frame
        viz = Visualizer(
            sim_map,
            spill,
            communication_radius=None if fully_connected else engine.communication_radius,
            show_communication_radius=not fully_connected,
        )
    
    viz.render(engine.drones, pause=False)
    import matplotlib.pyplot as plt
    plt.savefig('final_simulation_state.png')
    
    # Plot per-frame local and post-consensus estimates.
    frames = list(range(len(engine.estimates_history[next(iter(engine.estimates_history))]['r0_post'])))
    fig, ax = plt.subplots(figsize=(12, 6))

    for drone_id, history in engine.estimates_history.items():
        ax.plot(frames, history['r0_local'], label=f'{drone_id} local', linewidth=1.5, alpha=0.4)
        ax.plot(frames, history['r0_post'], label=f'{drone_id} consensus', linewidth=2.0, alpha=0.9)

    # Add true radius line
    ax.axhline(y=spill.r0, color='black', linestyle='--', linewidth=2.5, label='True r0')
    
    # Calculate final statistics
    final_r0_post = [engine.estimates_history[f'D{i}']['r0_post'][-1] for i in range(len(engine.drones))]
    initial_r0_local = [engine.estimates_history[f'D{i}']['r0_local'][0] for i in range(len(engine.drones))]
    initial_r0_post = [engine.estimates_history[f'D{i}']['r0_post'][0] for i in range(len(engine.drones))]
    mean_r0 = np.mean(final_r0_post)
    std_r0 = np.std(final_r0_post)
    
    # Print to console
    print("\n=== FINAL CONSENSUS RESULTS ===")
    for i in range(len(engine.drones)):
        print(
            f"D{i}: initial r0_local={initial_r0_local[i]:.6f}, "
            f"initial r0_post={initial_r0_post[i]:.6f}, "
            f"final r0_post={final_r0_post[i]:.6f}"
        )
    print(f"Mean r0: {mean_r0:.6f}")
    print(f"Std Dev: {std_r0:.6f}")
    print(f"True r0: {spill.r0:.6f}")
    print(f"Error from true: {abs(mean_r0 - spill.r0):.6f}")
    
    # Add text to plot
    textstr = f'Mean: {mean_r0:.4f}\nStd: {std_r0:.6f}\nTrue: {spill.r0:.4f}'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title('Radius Estimates: Local Measurements and Consensus Agreement')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Radius r0')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('consensus_convergence.png')
    print("Consensus convergence plot saved to 'consensus_convergence.png'")
    
    if visualize:
        print("Closing the window to exit.")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oil Spill Tracking Drone Simulation")
    parser.add_argument("--visualize", action="store_true", help="Enable real-time visualization (animation)")
    parser.add_argument("--frames", type=int, default=5000, help="Total number of simulation frames")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducible initialization")
    parser.add_argument(
        "--fully-connected",
        action="store_true",
        help="Disable range-based communication and use a fully connected consensus graph",
    )
    parser.add_argument(
        "--communication-radius-cells",
        type=int,
        default=201,
        help="Communication radius in grid cells when range-based communication is enabled",
    )
    args = parser.parse_args()

    run_multi_drone_simulation(
        visualize=args.visualize,
        max_frames=args.frames,
        seed=args.seed,
        fully_connected=args.fully_connected,
        communication_radius_cells=args.communication_radius_cells,
    )
