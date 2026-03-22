# Oil Spill Multi-Drone Simulation

A modular and scalable Python simulation for autonomous drones detecting and tracking oil spills using noisy sensors.

## Architecture

- **`main.py`**: Entry point. Initializes the environment and multi-drone simulation.
- **`simulation_engine.py`**: The core simulation loop. Manages agent states and sensor polling.
- **`drone.py`**: Modular drone class with proprioceptive (GPS) and exteroceptive (Camera) support.
- **`sensors.py`**: Implementation of noisy sensors (Gaussian noise on GPS and Camera data).
- **`environment.py`**: Defines the simulation map and oil spill mathematical models.
- **`edge_detection.py`**: Geometric algorithms for identifying oil spill boundaries from local camera data.
- **`visualization.py`**: Decoupled visualization module using Matplotlib.

## Features

- **Multi-Drone Support**: Simulates multiple agents independently searching the map.
- **Sensor Noise**: Real-world-like noise modeled for GPS and local camera perception.
- **Geometric Locking**: Drones identify the exact "straddling" edge of the oil spill.
- **Modern Viz**: Clear feedback for drone states (Searching: RED, Locked: GREEN).

## How to Run

Install dependencies:
```bash
pip install numpy matplotlib
```

Run simulation:
```bash
python3 main.py
```
