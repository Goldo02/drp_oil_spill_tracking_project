# Static Oil-Boundary Voronoi Control

This branch, `static-generic-oil-shape-voronoi-control-on-border`, implements a static-boundary, multi-agent coverage controller for drones deployed on the perimeter of an oil spill. The controller treats the spill boundary as a one-dimensional geometric manifold embedded in the plane, partitions that manifold among multiple agents with a multi-source shortest-path Voronoi method, and drives each agent toward the Lloyd centroid of its current one-dimensional cell.

The design goal is decentralized boundary coverage: each drone carries an onboard `DroneController`, maintains a local table of known peer positions, computes its own Voronoi/Lloyd target, applies a bounded velocity command, and is finally projected back to the oil-spill boundary. The simulation engine coordinates the static world, communication exchange, spawning, stepping, and visualization, but the control decision is made at the drone level.

## 1. System Architecture & Overview

### Repository Structure

The branch is organized around the following runtime modules:

| File | Responsibility |
| --- | --- |
| `controller.py` | Core 1D boundary Voronoi, arc-length, Lloyd-target, projection, and actuation logic. Defines `Controller` and `DroneController`. |
| `drone.py` | Drone state container and onboard control integration. Defines `Drone`. |
| `simulation_engine.py` | Simulation coordinator for boundary initialization, multi-hop communication, drone spawning, stepping, and action application. |
| `environment.py` | Simulation map and static oil-spill field models, including circular and smoothed polygon spills. |
| `visualization.py` | Visualization of the physical scene and the 1D Voronoi boundary partition. |
| `main.py` | Command-line entry point for running headless or visual simulations. |
| `tests/test_voronoi_mssp_static.py` | Regression tests for multi-source shortest-path assignment, Lloyd targets, action generation, and communication updates. |

### Decentralized / Distributed Multi-Agent Design

The system models a team of drones constrained to move along the oil-spill boundary. The boundary is represented by an ordered list of world-coordinate points:

```text
B = [p_0, p_1, ..., p_{N-1}],    p_i in R^2
```

Each drone is assigned a unique `drone_id` and keeps a local dictionary:

```python
drone.known_positions: dict[str, np.ndarray]
```

This dictionary is the drone's local belief over the positions of all drones it currently knows about. In the simulation, `SimulationEngine._exchange_positions_multihop()` refreshes this table through repeated local communication exchanges constrained by `communication_radius`. The controller itself does not require a global simulation object when deployed onboard; it only needs:

- the known boundary points,
- the closed/open boundary flag,
- the drone's current state,
- the drone's local `known_positions` table.

The resulting architecture is distributed in the control-theoretic sense: each drone independently recomputes a boundary partition using its own local information and then computes its own motion command. If all drones have the same position table, they compute a consistent global partition. If communication is partial, each drone computes a partition over the subset of agents it knows.

### `Controller` Versus `DroneController`

`controller.py` defines two related classes:

```python
class Controller:
    ...

class DroneController(Controller):
    ...
```

`Controller` is the general implementation of the boundary coverage algorithms. It contains:

- multi-source shortest-path Voronoi partitioning,
- arc-length parameterization and interpolation,
- Lloyd target computation,
- boundary initialization from either point arrays or occupancy grids,
- boundary ordering and projection helpers,
- action generation and velocity clipping.

`SimulationEngine` owns one global `Controller` instance. That global controller is used mainly for initializing or storing the world boundary:

```python
self.controller = Controller(
    sim_map=self.sim_map,
    communication_radius=self.communication_radius,
    occupancy_threshold=0.5,
)
```

`DroneController` is the onboard controller owned by each `Drone`:

```python
self.controller = DroneController()
```

It is intentionally lightweight. It calls the inherited algorithms but exposes the smaller API expected by an individual robot:

```python
DroneController(
    known_boundary_points=None,
    known_boundary_closed=True,
    k_t=1.0,
    constrain_to_boundary=True,
)

set_known_boundary(boundary_points, known_boundary_closed=True, already_ordered=True)
compute_action(drone)
project_to_boundary(drone)
```

During a control step, `Drone.compute_action()` delegates to `DroneController.compute_action()`. The onboard controller then:

1. computes the current 1D boundary Voronoi/Lloyd structure from `drone.known_positions`,
2. stores diagnostics in `drone.last_ring_info`,
3. stores the current target in `drone.target_centroid`,
4. computes a bounded action toward the target.

`Drone.action()` applies the returned velocity-like command to the drone state and clips the physical position to the simulation map bounds. Immediately afterward, `Drone.project_to_boundary()` snaps the drone back to the closest boundary segment, enforcing boundary-constrained motion.

### Simulation Step Flow

The main step sequence is implemented in `SimulationEngine.step()` and `_apply_actions()`:

```text
SimulationEngine.step()
    frame += 1
    _apply_actions()

SimulationEngine._apply_actions()
    _exchange_positions_multihop()
    actions = {drone_id: drone.compute_action() for each drone}
    for each drone:
        drone.action(action, bounds=map_bounds)
        drone.project_to_boundary()
```

This ordering is important. All drones first exchange position information, then actions are computed from the same frame's local beliefs, then actions are applied, and finally boundary projection restores the exact geometric constraint.

## 2. Mathematical Foundations & Algorithms

### Boundary as a 1D Embedded Manifold

The oil-spill boundary is represented as a discrete ordered chain:

```text
B = {p_i}_{i=0}^{N-1},    p_i = (x_i, y_i)
```

The chain may be open or closed:

- open boundary: edges exist only between `p_i` and `p_{i+1}` for `i = 0, ..., N-2`;
- closed boundary: the open-chain edges are augmented by a wraparound edge between `p_{N-1}` and `p_0`.

The controller does not use Euclidean distance through the interior of the spill to measure coverage along the boundary. Instead, it uses geodesic distance along the boundary polyline. This is the correct metric for a robot constrained to remain on the perimeter.

### Multi-Source Dijkstra Voronoi

The method `Controller.multi_source_shortest_path_voronoi(points, seeds, is_closed)` assigns every boundary sample to the nearest seed agent under shortest-path distance along the boundary graph.

#### Inputs

```python
points: array-like, shape (N, 2)
seeds: iterable of dicts with keys:
    "robot_id": hashable/string robot identifier
    "index": integer boundary sample index
is_closed: bool
```

Each seed corresponds to the nearest boundary index of a drone. The method returns:

```python
owner: np.ndarray, shape (N,), dtype=object
distances: np.ndarray, shape (N,), dtype=float
```

`owner[i]` is the robot ID that owns boundary point `p_i`; `distances[i]` is the shortest boundary distance from `p_i` to that owner's seed.

#### Graph Model

The boundary is treated as a weighted graph:

```text
G = (V, E)
V = {0, 1, ..., N-1}
```

For an open chain:

```text
E = {(i, i+1) | i = 0, ..., N-2}
```

For a closed loop:

```text
E = {(i, i+1) | i = 0, ..., N-2} union {(N-1, 0)}
```

Each edge has Euclidean chord length:

```text
w(i, j) = ||p_i - p_j||_2
```

The Voronoi ownership rule is:

```text
owner(i) = arg min_k d_G(i, s_k)
```

where `s_k` is the seed index for robot `k`, and `d_G` is the graph shortest-path distance along the boundary.

#### Priority Queue Propagation

The implementation initializes a priority queue with all valid seeds at distance zero:

```python
distances[index] = 0.0
owner[index] = robot_id
heapq.heappush(pq, (0.0, index, order, robot_id))
```

It then performs Dijkstra expansion from all sources simultaneously. At every pop, the algorithm visits the predecessor and successor boundary indices, using modulo wraparound only when `is_closed=True`.

For each neighbor `v` of current node `u`:

```text
new_dist = distances[u] + ||p_u - p_v||_2
```

If `new_dist` improves the stored distance, the neighbor receives the same owner as `u` and is pushed back into the queue. The result is a discrete geodesic Voronoi partition of the boundary.

For a single closed contour with `N` samples and `M` drones, runtime is:

```text
O((N + |E|) log N) = O(N log N)
```

because the boundary graph has linear edge count.

#### Role in the Controller

The Dijkstra Voronoi assignment is computed inside `Controller.compute_ring_ordering()`. It provides:

- `assigned_drone_indices`: ownership for each boundary sample,
- `distances`: shortest distance from the owning seed,
- `indices`: the sample indices owned by each robot,
- `voronoi_cell_size`: number of discrete samples owned by a robot.

The controller also computes analytic Lloyd targets from seed arc coordinates. Thus, Dijkstra assignment provides the discrete ownership field and visualization-compatible partition, while the Lloyd step computes target centroids in continuous arc-length coordinates.

### Arc-Length Parameterization

The controller converts the two-dimensional boundary polyline into a scalar coordinate system:

```text
s in [0, L]
```

where `L` is the total perimeter length for a closed boundary or the total chain length for an open boundary.

#### Cumulative Arc Lengths

`Controller._boundary_arc_lengths(boundary_points, is_closed=False)` returns:

```python
arc_lengths: np.ndarray, shape (N,)
total_length: float
```

For the ordered points `p_i`, the cumulative coordinate is:

```text
s_0 = 0
s_i = sum_{j=0}^{i-1} ||p_{j+1} - p_j||_2
```

The open-chain length is:

```text
L_open = s_{N-1}
```

If the chain is closed, the closing segment is added:

```text
L_closed = s_{N-1} + ||p_0 - p_{N-1}||_2
```

The returned `arc_lengths` array stores the cumulative coordinates of the explicit samples only. For closed loops, the final wraparound segment is represented through `total_length`, not by duplicating `p_0` at the end.

#### Curve Interpolation: `_point_at_arc_length`

`Controller._point_at_arc_length(boundary_points, arc_lengths, arc_length, total_length, is_closed)` maps a scalar arc coordinate `s` back to a two-dimensional point on the boundary.

For closed boundaries, the coordinate is wrapped:

```text
s <- s mod L
```

For open boundaries, it is clamped:

```text
s <- clip(s, 0, L)
```

The method then locates the segment containing `s`. If:

```text
s_i <= s <= s_{i+1}
```

the interpolation fraction is:

```text
t = (s - s_i) / (s_{i+1} - s_i)
```

and the returned point is:

```text
p(s) = (1 - t) p_i + t p_{i+1}
```

If `s` lies on the closing segment of a closed loop:

```text
s_{N-1} <= s < L
```

the interpolation is performed between `p_{N-1}` and `p_0`.

Degenerate cases are handled explicitly:

- zero points return `[0.0, 0.0]`,
- one point returns that point,
- near-zero total length returns the first point,
- near-zero segment length returns the left endpoint.

#### Position Projection: `_arc_length_at_position`

`Controller._arc_length_at_position(boundary_points, arc_lengths, position, total_length, is_closed)` projects an arbitrary 2D position onto the closest point of the boundary polyline and returns:

```python
boundary_s: float
projected: np.ndarray, shape (2,)
nearest_idx: int
```

For each segment from `a = p_i` to `b = p_j`, the method computes the Euclidean projection:

```text
v = b - a
t = clip(((x - a) dot v) / (v dot v), 0, 1)
q = a + t v
```

It selects the segment projection `q` minimizing:

```text
||x - q||_2^2
```

The corresponding arc coordinate is:

```text
s = s_i + t ||p_j - p_i||_2
```

For a closed boundary, the search includes the closing segment and wraps the result modulo `L`. The returned `nearest_idx` is chosen as the segment endpoint nearest to the projection parameter: `i` if `t < 0.5`, otherwise `j`.

This projection is used in three critical places:

1. converting drone positions into boundary seeds,
2. snapping drones back to the boundary after actuation,
3. recovering the current arc coordinate for control if no stored coordinate is available.

### Lloyd's Algorithm for 1D Coverage

The method `Controller._lloyd_targets_from_seed_arcs(seeds, total_length, is_closed)` computes one-dimensional Lloyd targets for the agents. It assumes that each seed has already been projected to the boundary and assigned an arc coordinate:

```python
{
    "robot_id": drone_id,
    "index": seed_idx,
    "arc_length": seed_s,
    "position_on_boundary": projected_point,
}
```

The seeds are sorted by arc coordinate. Lloyd coverage in one dimension uses Voronoi cell centroids. With uniform density along the boundary, the centroid of a 1D interval is its midpoint in arc length.

#### Closed Boundary Lloyd Cells

For a closed loop, seed order is circular. For seed `i` with coordinate `s_i`, let:

```text
s_{i-1}: previous seed coordinate
s_{i+1}: next seed coordinate
L: total loop length
```

The circular gaps are:

```text
left_gap  = (s_i - s_{i-1}) mod L
right_gap = (s_{i+1} - s_i) mod L
```

The cell starts halfway between the previous seed and current seed:

```text
cell_start = (s_i - 0.5 left_gap) mod L
```

The cell ends halfway between the current seed and next seed:

```text
cell_end = (s_i + 0.5 right_gap) mod L
```

The cell length is:

```text
cell_len = 0.5 (left_gap + right_gap)
```

The Lloyd centroid in arc coordinates is:

```text
target_s = (cell_start + 0.5 cell_len) mod L
```

The modulo operation is essential. It allows cells to cross the `s = 0` branch cut without discontinuity.

#### Open Boundary Lloyd Cells

For an open boundary, the first and last cells are clipped by the physical endpoints. The cell boundaries are:

```text
b_0 = 0
b_i = 0.5 (s_{i-1} + s_i),    i = 1, ..., M-1
b_M = L
```

For seed `i`, the interval is:

```text
[b_i, b_{i+1}]
```

and the target is:

```text
target_s = 0.5 (b_i + b_{i+1})
cell_len = b_{i+1} - b_i
```

This gives correct endpoint handling for an open chain: outer drones cover from the physical boundary endpoint to the midpoint between themselves and their nearest neighbor.

#### Single-Agent Case

If only one seed is known, the controller assigns the entire boundary length to that robot:

```text
cell_start = 0
cell_end = L
cell_len = L
target_s = current seed arc
```

The target remains the current seed arc rather than the global midpoint. This prevents arbitrary motion when only one drone is present or known.

#### From Arc Centroid to 2D Target

After `target_s` is computed, `compute_ring_ordering()` converts it to a world-coordinate target:

```python
target_centroid = _point_at_arc_length(
    occupied_points,
    arc_lengths,
    target_arc_length,
    total_boundary_length,
    is_closed,
)
```

The target is stored in:

```python
drone.target_centroid
drone.last_ring_info["current"]["target_centroid"]
```

The corresponding nearest sampled boundary index is also computed for diagnostics and visualization:

```python
target_chain_index = _nearest_boundary_index(occupied_points, target_centroid)
```

## 3. Boundary Handling & Geometric Preprocessing

### Boundary Initialization from Point Arrays

`Controller.initialize_known_boundary(world_field_or_points, x_coords=None, y_coords=None, force_closed=True)` accepts either an explicit `(N, 2)` point array or a 2D occupancy field.

If the input is already an `(N, 2)` array, it is copied directly:

```python
self.known_boundary_points = pts
self.known_boundary_closed = bool(force_closed) or (
    pts.shape[0] > 1 and np.linalg.norm(pts[0] - pts[-1]) < 1e-6
)
self.known_boundary_ordered = False
```

Explicit point arrays are marked unordered by default because the method cannot assume that arbitrary input points follow the contour. Ordering is later enforced lazily by `_ensure_ordered_closed_boundary()`.

For a loaded oil-mapping boundary, `SimulationEngine.initialize_loaded_boundary()` performs stricter validation:

- array must have shape `(N, 2)`,
- `N >= 3`,
- all coordinates must be finite,
- caller supplies whether the points are already ordered and closed.

`main.py` also validates loaded mapping metadata and verifies that coordinates lie inside the simulation map within a grid-based tolerance.

### Boundary Initialization from Occupancy Grids

When the input is a 2D scalar field, the controller thresholds it:

```python
occupied = field >= self.occupancy_threshold
```

The default `occupancy_threshold` is `0.5`. The controller then tries to extract a smooth ordered contour with `contourpy`:

```python
pts = self._extract_ordered_contour(field, x_coords, y_coords)
```

The contour extraction:

1. creates a `contourpy.contour_generator`,
2. extracts isolines at the occupancy threshold,
3. selects the longest contour by polyline length,
4. removes a duplicated closing point if the first and last samples coincide.

If `contourpy` is unavailable or no contour can be extracted, the controller falls back to mask-boundary extraction:

```python
pts = self._extract_boundary_mask_points(occupied, x_coords, y_coords)
```

The mask fallback scans occupied cells and marks an occupied cell as a boundary point when at least one of its 8-neighbors is free. The returned points are grid-coordinate samples mapped through `x_coords` and `y_coords`.

### Boundary Ordering

The controller requires boundary samples to be sequential along the contour. The helper:

```python
Controller._ensure_ordered_closed_boundary()
```

is called before projection, partitioning, and action computation. If `known_boundary_ordered` is already true, the method returns immediately.

For unordered point sets, it performs a greedy nearest-neighbor trace:

1. estimate typical point spacing using each point's nearest-neighbor distance,
2. start from point index `0`,
3. repeatedly append the nearest unvisited point,
4. test whether the ordered chain appears closed.

The closedness test compares the final point back to the first point:

```text
||ordered[0] - ordered[-1]|| <= 3 * median_nearest_neighbor_spacing
```

If the condition passes, the greedy ordering is accepted and `known_boundary_closed=True`. If the condition fails, the original point ordering is retained and `known_boundary_closed=False`. In both cases, the boundary is marked as ordered so the expensive ordering attempt is not repeated every frame.

This is a practical fallback rather than a full contour-reconstruction algorithm. For high-quality operation on complex geometries, pass an already ordered contour and set `already_ordered=True` through `DroneController.set_known_boundary()` or `SimulationEngine.initialize_loaded_boundary()`.

### Closed-Loop Integrity

Closed-loop integrity is handled without requiring duplicated first/last samples. A boundary can be closed even if:

```text
p_0 != p_{N-1}
```

The wraparound edge is implicit whenever `known_boundary_closed=True`. This avoids creating a zero-length duplicate segment at the seam and keeps sample indexing simple.

For contour extraction from grids, duplicated closing samples are explicitly removed:

```python
if len(contour) > 1 and np.linalg.norm(contour[0] - contour[-1]) < 1e-9:
    contour = contour[:-1]
```

For arc-length computation, the closing length is added only to `total_length`, not to `arc_lengths`.

### Geometric Safety Mechanisms

The code includes several safety mechanisms for degenerate or noisy geometry:

- Empty boundary input returns an empty array and causes `compute_ring_ordering()` to return `None`.
- Empty interpolation returns a zero vector rather than throwing inside the control loop.
- One-point and near-zero-length boundaries return the only available point.
- Non-finite or incorrectly shaped actions are converted to zero commands.
- Segment projections handle zero-length segments by setting projection fraction `t = 0`.
- Open-boundary arc coordinates are clamped to `[0, L]`.
- Closed-boundary arc coordinates are wrapped modulo `L`.
- Drone positions are clipped to map bounds in `Drone.action()`.
- Drone states are projected back to the boundary after every action when `constrain_to_boundary=True`.

These mechanisms allow the simulation to continue safely under imperfect input, while still surfacing hard validation errors for loaded mapping data with invalid shape, non-finite values, or incompatible coordinate frames.

## 4. Control Law & Actuation

### Lloyd Target Tracking

The actuation law is implemented in:

```python
Controller._equidistant_action(drone, ring_info, world_field, x_coords, y_coords)
```

Despite the method name, the active controller is a Lloyd target tracker along the boundary. It drives each drone toward the centroid of its current 1D Voronoi cell. The high-level proportional law is:

```text
u = k_t (x_target - x_current)
```

However, when arc-length metadata is available, the controller uses a boundary-coordinate version that respects the one-dimensional geometry more directly.

### Arc-Domain Proportional Control

If `ring_info` contains:

- `target_arc_length`,
- `seed_arc_length`,
- `occupied_points`,
- `arc_lengths`,
- `total_boundary_length`,
- `is_closed`,

then the controller computes the tracking error in arc coordinates.

For a closed loop, it uses the shortest signed circular error:

```text
e_s = ((s_target - s_current + 0.5 L) mod L) - 0.5 L
```

For an open boundary:

```text
e_s = s_target - s_current
```

The arc step is proportional but speed-limited:

```text
delta_s = clip(k_t e_s, -v_max, v_max)
```

The controller then computes the next desired point on the boundary:

```text
s_next = s_current + delta_s
x_next = p(s_next)
```

and returns:

```text
u = x_next - x_current
```

This action is clipped again by `_clip_action()`. Because the next point is chosen on the boundary and the drone is projected back to the boundary after actuation, the closed-loop behavior approximates constrained motion on the boundary curve.

### Cartesian Fallback Control

If arc metadata is unavailable but a target point is known, the method falls back to a direct Cartesian proportional command:

```text
u = k_t (x_target - x_current)
```

The same velocity clipping is applied. If no target is available, the controller returns zero.

### Velocity Clipping: `_clip_action`

`Controller._clip_action(action, max_speed=0.12)` enforces command validity and a maximum speed:

1. Convert the action to a floating NumPy vector.
2. If shape is not `(2,)`, or any element is non-finite, return `[0, 0]`.
3. If norm is near zero, return `[0, 0]`.
4. If norm exceeds `max_speed`, scale the vector:

```text
u_clipped = u * max_speed / ||u||
```

5. Otherwise return the original vector.

The default maximum speed is `0.12`, but the controller reads `drone.max_speed` when available.

`Drone.action()` also clips commands through `Drone._clip_command()`, so command limiting is applied both at controller output and at actuator integration. This duplication is deliberate defensive programming: the drone state update remains safe even if a command originates outside the controller.

### Boundary Constraint: `constrain_to_boundary`

Boundary projection is implemented in:

```python
Controller.project_drone_to_boundary(drone)
DroneController.project_to_boundary(drone)
Drone.project_to_boundary()
```

When `constrain_to_boundary=True`, the controller:

1. ensures the boundary is ordered,
2. computes arc-length metadata,
3. projects the drone position to the closest boundary segment,
4. overwrites `drone.x` and `drone.y` with the projected point,
5. updates `drone.known_positions[drone.drone_id]`,
6. stores `drone.boundary_index`,
7. stores `drone.boundary_s`.

Projection is skipped if:

- `constrain_to_boundary=False`,
- there are no known boundary points,
- the boundary is empty.

In the simulation loop, projection occurs after `Drone.action()`. Thus, even if the velocity command is a chord between two nearby boundary points rather than an exact curve-following trajectory, the final discrete state remains on the boundary.

## 5. API Usage & Integration Guide

### Running the Simulation

Run a headless simulation with the default smoothed-polygon spill:

```bash
python main.py --frames 500 --num-drones 5
```

Run with visualization:

```bash
python main.py --visualize --frames 500 --num-drones 5
```

Run a circular spill:

```bash
python main.py --oil-shape circle --frames 300 --num-drones 4
```

Run with a loaded oil-mapping boundary:

```bash
python main.py \
  --load-oil-mapping \
  --oil-mapping-input ./tmp_output/oil_mapping_data.npy \
  --frames 500 \
  --num-drones 5
```

Important command-line options:

| Option | Meaning |
| --- | --- |
| `--frames` | Number of simulation frames. |
| `--num-drones` | Number of drones spawned on the boundary. |
| `--oil-shape {circle,smoothed_polygon}` | Static spill geometry source. |
| `--communication-radius-cells` | Communication radius expressed in grid-cell units. |
| `--load-oil-mapping` | Use a precomputed boundary point file instead of extracting from the current spill field. |
| `--oil-mapping-input` | Path to `.npy` or `.npz` oil boundary data with shape `(N, 2)`. |
| `--visualize` | Enable interactive Matplotlib visualization. |

### Creating a Map and Spill

```python
from environment import SimulationMap, SmoothedPolygonOilSpill

sim_map = SimulationMap(
    xlim=(-5.0, 5.0),
    ylim=(-5.0, 5.0),
    grid_size=500,
)

spill = SmoothedPolygonOilSpill(
    sim_map.X,
    sim_map.Y,
    n_vertices=36,
    r0=2.5,
    smoothness=0.2,
    seed=42,
)
```

For a circular spill:

```python
from environment import CircleOilSpill

spill = CircleOilSpill(x0=0.0, y0=0.0, radius=2.0)
```

### Initializing the Simulation Engine

```python
from simulation_engine import SimulationEngine

engine = SimulationEngine(
    sim_map=sim_map,
    oil_spill=spill,
    communication_radius_cells=250,
    verbose=True,
)
```

The engine derives world bounds and grid spacing from `sim_map` when available. It builds the static world field by evaluating:

```python
engine.world_field = spill.get_field(sim_map.X, sim_map.Y)
```

### Initializing a Boundary from the World Field

```python
boundary = engine.initialize_world_boundary()
```

Internally this calls:

```python
Controller.initialize_known_boundary(
    world_field,
    x_coords=sim_map.x_coords,
    y_coords=sim_map.y_coords,
    force_closed=True,
)
```

The returned `boundary` is an `(N, 2)` array of world-coordinate points.

### Initializing a Boundary from Explicit Points

If an ordered boundary is already available:

```python
import numpy as np

boundary_points = np.load("./tmp_output/oil_mapping_data.npy")

engine.initialize_loaded_boundary(
    boundary_points,
    known_boundary_closed=True,
    already_ordered=True,
)
```

For a standalone controller:

```python
from controller import DroneController

controller = DroneController(
    known_boundary_points=boundary_points,
    known_boundary_closed=True,
    k_t=1.0,
    constrain_to_boundary=True,
)
```

or:

```python
controller = DroneController()
controller.set_known_boundary(
    boundary_points,
    known_boundary_closed=True,
    already_ordered=True,
)
```

### Spawning Drones on the Boundary

```python
drones = engine.spawn_drones_on_boundary(
    num_drones=5,
    rng=np.random.default_rng(42),
)
```

This method:

1. samples boundary indices,
2. creates `Drone` instances at those boundary points,
3. copies the contour grid into each drone's local grid,
4. assigns the same known boundary to every onboard controller,
5. initializes each drone's `known_positions` table with all drone positions.

### Executing a Simulation Loop

The simplest usage is:

```python
for _ in range(500):
    engine.step()
```

A manual loop with visualization data:

```python
for _ in range(500):
    engine.step()
    data = engine.get_visualization_data()
    # pass data to a renderer or logger
```

The higher-level helper:

```python
engine.run(iterations=500, render_callback=visualizer.render)
```

calls `step()` repeatedly and invokes an optional callback after each frame.

### Direct Onboard Controller Usage

The onboard usage pattern is:

```python
from drone import Drone

drone = Drone(
    drone_id="D0",
    x=boundary_points[0, 0],
    y=boundary_points[0, 1],
    grid_shape=(500, 500),
    grid_bounds=(-5.0, 5.0, -5.0, 5.0),
    max_speed=0.12,
)

drone.set_known_boundary(
    boundary_points,
    known_boundary_closed=True,
    already_ordered=True,
)

drone.known_positions = {
    "D0": drone.position,
    "D1": np.array([1.0, 2.0]),
    "D2": np.array([-1.0, 2.0]),
}

command = drone.compute_action()
drone.action(command, dt=1.0, bounds=((-5.0, 5.0), (-5.0, 5.0)))
drone.project_to_boundary()
```

After `compute_action()`, diagnostics are available:

```python
drone.target_centroid
drone.last_ring_info
drone.last_control_mode
```

`drone.last_ring_info` contains the current Voronoi/Lloyd state:

```python
{
    "N": number_of_known_agents,
    "occupied_points": boundary_points,
    "assigned_drone_indices": owner_array,
    "distances": geodesic_distance_array,
    "arc_lengths": cumulative_arc_lengths,
    "total_boundary_length": total_length,
    "seeds": projected_seed_records,
    "is_closed": closed_flag,
    "ring": per_drone_cell_records,
    "current": current_drone_record,
    "pred": predecessor_record,
    "succ": successor_record,
    "center_of_mass": boundary_point_mean,
}
```

### Key Method Signatures

```python
Controller(
    sim_map,
    communication_radius,
    occupancy_threshold=0.5,
    k_t=1.0,
    **kwargs,
)
```

```python
Controller.initialize_known_boundary(
    world_field_or_points,
    x_coords=None,
    y_coords=None,
    force_closed=True,
)
```

```python
Controller.multi_source_shortest_path_voronoi(
    points,
    seeds,
    is_closed,
)
```

```python
Controller.compute_ring_ordering(
    current_drone,
    drones,
)
```

```python
Controller.project_drone_to_boundary(
    drone,
)
```

```python
DroneController(
    known_boundary_points=None,
    known_boundary_closed=True,
    k_t=1.0,
    constrain_to_boundary=True,
)
```

```python
DroneController.set_known_boundary(
    boundary_points,
    known_boundary_closed=True,
    already_ordered=True,
)
```

```python
DroneController.compute_action(drone)
```

```python
Drone.action(command, dt=1.0, bounds=None)
```

```python
SimulationEngine.initialize_world_boundary()
SimulationEngine.initialize_loaded_boundary(boundary_points, known_boundary_closed=True, already_ordered=True)
SimulationEngine.spawn_drones_on_boundary(num_drones, rng=None)
SimulationEngine.step()
SimulationEngine.run(iterations, render_callback=None)
```

### Minimal End-to-End Example

```python
import numpy as np

from environment import SimulationMap, SmoothedPolygonOilSpill
from simulation_engine import SimulationEngine

seed = 42

sim_map = SimulationMap(
    xlim=(-5.0, 5.0),
    ylim=(-5.0, 5.0),
    grid_size=500,
)

spill = SmoothedPolygonOilSpill(
    sim_map.X,
    sim_map.Y,
    n_vertices=36,
    r0=2.5,
    smoothness=0.2,
    seed=seed,
)

engine = SimulationEngine(
    sim_map=sim_map,
    oil_spill=spill,
    communication_radius_cells=250,
    verbose=False,
)

engine.initialize_world_boundary()
engine.spawn_drones_on_boundary(
    num_drones=5,
    rng=np.random.default_rng(seed),
)

for _ in range(500):
    engine.step()

for drone in engine.drones:
    print(drone.drone_id, drone.position, drone.target_centroid)
```

## Design Notes and Expected Behavior

At convergence, drones should become approximately equidistant in arc length, not necessarily in Euclidean distance. This distinction matters for irregular oil-spill shapes: two drones may be close in Cartesian space across a concavity while still being far apart along the boundary. The controller intentionally uses boundary geodesic distance and arc-length centroids so that coverage is uniform over the perimeter itself.

The controller assumes the boundary is static during the simulation. Dynamic spill evolution would require updating `known_boundary_points`, refreshing each drone's onboard boundary, and handling continuity between old and new arc coordinates. The current branch is therefore best understood as a static perimeter coverage controller for known or precomputed oil-spill boundaries.
