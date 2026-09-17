import numpy as np

from controller import Controller
from drone import Drone
from simulation_engine import SimulationEngine


def make_engine_shell():
    engine = object.__new__(SimulationEngine)
    engine.x_min = -1.0
    engine.y_min = -1.0
    engine.resolution = 0.1
    engine.occupancy_threshold = 0.5
    engine.grid_shape = (12, 12)
    engine.boundary_controller = Controller(
        sim_map=None,
        communication_radius=0.0,
        occupancy_threshold=engine.occupancy_threshold,
    )
    engine.closure_min_boundary_cells = 8
    engine.closure_min_enclosed_false_cells = 8
    engine.last_closure_enclosed_false_cells = 0
    return engine


def test_dfs_polygon_closure_detects_closed_loop_not_open_chain():
    engine = make_engine_shell()
    closed_grid = np.zeros((12, 12), dtype=float)
    closed_grid[3:9, 3] = 1.0
    closed_grid[3:9, 8] = 1.0
    closed_grid[3, 3:9] = 1.0
    closed_grid[8, 3:9] = 1.0

    is_closed, points = engine.is_mapped_polygon_closed(closed_grid)

    assert is_closed
    assert points.shape[0] >= engine.closure_min_boundary_cells

    open_grid = closed_grid.copy()
    open_grid[3:6, 8] = 0.0

    is_closed, points = engine.is_mapped_polygon_closed(open_grid)

    assert not is_closed
    assert points.shape == (0, 2)


def test_dfs_polygon_closure_requires_minimum_enclosed_false_area():
    engine = make_engine_shell()
    engine.closure_min_enclosed_false_cells = 5

    small_loop = np.zeros((8, 8), dtype=float)
    small_loop[2:5, 2] = 1.0
    small_loop[2:5, 4] = 1.0
    small_loop[2, 2:5] = 1.0
    small_loop[4, 2:5] = 1.0

    is_closed, points = engine.is_mapped_polygon_closed(small_loop)

    assert not is_closed
    assert points.shape == (0, 2)
    assert engine.last_closure_enclosed_false_cells == 1


def test_closed_boundary_points_are_ordered_by_neighbor_trace():
    engine = make_engine_shell()
    closed_grid = np.zeros((12, 12), dtype=float)
    closed_grid[3:9, 3] = 1.0
    closed_grid[3:9, 8] = 1.0
    closed_grid[3, 3:9] = 1.0
    closed_grid[8, 3:9] = 1.0

    is_closed, points = engine.is_mapped_polygon_closed(closed_grid)

    assert is_closed
    step_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert np.max(step_lengths) <= np.sqrt(2.0) * engine.resolution + 1e-12


def test_transition_to_lloyd_state_loads_boundary_into_each_drone():
    engine = make_engine_shell()
    engine.control_state = "mapping"
    engine.closed_boundary_points = np.empty((0, 2), dtype=float)
    engine.boundary_controller = Controller(
        sim_map=None,
        communication_radius=10.0,
    )
    engine.frame = 7
    engine.verbose = False

    boundary = np.array(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ],
        dtype=float,
    )
    engine.drones = [
        Drone("D0", 0.0, -0.5, (4, 4), (-1.0, 1.0, -1.0, 1.0)),
        Drone("D1", 0.5, 0.0, (4, 4), (-1.0, 1.0, -1.0, 1.0)),
    ]

    engine._transition_to_lloyd_state(boundary)

    assert engine.control_state == "lloyd"
    assert engine.transition_frame == 7
    np.testing.assert_allclose(engine.closed_boundary_points, boundary)
    for drone in engine.drones:
        assert drone.control_state == "lloyd"
        np.testing.assert_allclose(drone.known_boundary_points, boundary)
        assert set(drone.known_positions) == {"D0", "D1"}
