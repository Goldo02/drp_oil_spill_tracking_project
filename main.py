import numpy as np
import time
from environment import SimulationMap, GaussianOilSpill, CircleOilSpill
from simulation_engine import SimulationEngine
from visualization import Visualizer

def run_multi_drone_simulation():
    # 1. Setup Environment
    sim_map = SimulationMap(xlim=(-5, 5), ylim=(-5, 5), grid_size=500)
    spill = CircleOilSpill(x0=0.0, y0=0.0)
    # spill = GaussianOilSpill(x0=0.0, y0=0.0, sigma=1.5, amplitude=1.0)
    
    # 2. Setup Engine
    engine = SimulationEngine(sim_map, spill, dt=0.1)
    
    # 3. Add Drones with random start positions
    for i in range(3): # Test with 2 drones
        start_x = np.random.uniform(-4, 4)
        start_y = np.random.uniform(-4, 4)
        engine.add_drone(drone_id=f"D{i}", x=start_x, y=start_y)
    
    # 4. Setup Visualization
    viz = Visualizer(sim_map, spill)
    
    print("Starting Multi-Drone Simulation...")
    print("Mode: SEARCH -> APPROACH -> LOCKED (Targeting the geometric edge)")
    
    # 5. Main Loop
    try:
        for frame in range(1000):
            # Update Logic
            engine.step()
            
            # Update Visualization
            viz.render(engine.drones)
            
            # The simulation runs for 1000 frames or until interrupted
            pass
                
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    
    print("Simulation finished. Close the window to exit.")
    import matplotlib.pyplot as plt
    plt.show()

if __name__ == "__main__":
    run_multi_drone_simulation()
