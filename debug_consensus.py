import numpy as np
from environment import SimulationMap, CircleOilSpill
from simulation_engine import SimulationEngine

np.random.seed(2)
map_ = SimulationMap(xlim=(-5,5), ylim=(-5,5), grid_size=200)
spill = CircleOilSpill(x0=0.0, y0=0.0)
engine = SimulationEngine(map_, spill, dt=0.05)
for i in range(4):
    start_x = np.random.uniform(map_.xlim[0], map_.xlim[1])
    start_y = np.random.uniform(map_.ylim[0], map_.ylim[1])
    engine.add_drone(drone_id=f"D{i}", x=start_x, y=start_y)

for _ in range(2000):
    engine.step()

print("Final drone states:")
for d in engine.drones:
    phi = np.arctan2(d.y - spill.y0, d.x - spill.x0)
    print(f"{d.drone_id}: x={d.x:.3f}, y={d.y:.3f}, phi={phi:.3f} rad, phi_deg={np.degrees(phi):.1f}°")
