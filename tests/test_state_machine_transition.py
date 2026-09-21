import numpy as np
import pytest

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
    engine.transition_frame = None
    engine._pending_lloyd_transition_frames = {}
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


def test_thick_occupied_boundary_collapses_to_single_centerline():
    engine = make_engine_shell()
    engine.x_min = 0.0
    engine.y_min = 0.0
    engine.grid_shape = (80, 80)
    engine.closure_min_boundary_cells = 24

    grid_x, grid_y = np.indices(engine.grid_shape)
    center_cell = np.array([40.0, 40.0])
    radius_cells = np.sqrt(
        (grid_x - center_cell[0]) ** 2 + (grid_y - center_cell[1]) ** 2
    )
    thick_ring = ((radius_cells >= 18.0) & (radius_cells <= 22.0)).astype(float)

    centerline = engine._ordered_centerline_points_from_grid(thick_ring)

    assert centerline.shape[0] >= engine.closure_min_boundary_cells

    center_world = (center_cell + 0.5) * engine.resolution
    radii = np.linalg.norm(centerline - center_world.reshape(1, 2), axis=1)

    assert np.mean(radii) == pytest.approx(2.0, abs=0.15)
    assert np.std(radii) < 0.2
    assert centerline.shape[0] < np.count_nonzero(thick_ring)


def test_enclosed_free_region_contour_has_no_long_jumps():
    engine = make_engine_shell()
    engine.x_min = 0.0
    engine.y_min = 0.0
    engine.grid_shape = (40, 40)
    engine.closure_min_boundary_cells = 16

    grid = np.zeros(engine.grid_shape, dtype=float)
    grid[10:30, 10] = 1.0
    grid[10:30, 29] = 1.0
    grid[10, 10:30] = 1.0
    grid[29, 10:30] = 1.0

    contour = engine._ordered_enclosed_contour_points_from_grid(grid)

    if contour.size == 0:
        pytest.skip("contourpy is not available")

    step_lengths = np.linalg.norm(np.roll(contour, -1, axis=0) - contour, axis=1)
    assert contour.shape[0] >= engine.closure_min_boundary_cells
    assert np.max(step_lengths) <= engine.resolution + 1e-12


def test_loop_cell_ordering_backtracks_without_geometric_jumps():
    engine = make_engine_shell()
    cells = {
        (1, 1),
        (2, 1),
        (3, 1),
        (3, 2),
        (3, 3),
        (2, 3),
        (1, 3),
        (1, 2),
        (2, 2),
    }
    adjacency = {
        cell: [neighbor for neighbor in engine._cell_neighbors(cell) if neighbor in cells]
        for cell in cells
    }

    ordered = engine._order_loop_cells(cells, adjacency)
    points = engine._cells_to_world_points(ordered)
    step_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)

    assert set(ordered) == cells
    assert np.max(step_lengths) <= np.sqrt(2.0) * engine.resolution + 1e-12


def test_decentralized_transition_loads_boundary_only_into_triggering_drone():
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
    initial_positions = {
        drone.drone_id: drone.position.copy()
        for drone in engine.drones
    }

    engine._transition_drone_to_lloyd_state(engine.drones[0], boundary)

    assert engine.control_state == "mapping"
    assert engine.transition_frame == 7
    np.testing.assert_allclose(engine.closed_boundary_points, boundary)

    switched = engine.drones[0]
    waiting = engine.drones[1]

    assert switched.control_state == "lloyd"
    np.testing.assert_allclose(switched.known_boundary_points, boundary)
    np.testing.assert_allclose(switched.position, initial_positions[switched.drone_id])
    assert switched.boundary_s is not None
    assert set(switched.known_positions) == {"D0"}
    assert set(switched.known_boundary_arcs) == {"D0"}

    assert waiting.control_state == "mapping"
    assert waiting.known_boundary_points.shape == (0, 2)
    np.testing.assert_allclose(waiting.position, initial_positions[waiting.drone_id])


def test_decentralized_transition_waits_for_consensus_after_closure_detection():
    engine = make_engine_shell()
    engine.control_state = "mapping"
    engine.closed_boundary_points = np.empty((0, 2), dtype=float)
    engine.frame = 7
    engine.verbose = False
    engine.drones = [
        Drone("D0", 0.0, -0.5, (12, 12), (-1.0, 1.0, -1.0, 1.0)),
    ]
    closed_grid = np.zeros((12, 12), dtype=float)
    closed_grid[3:9, 3] = 1.0
    closed_grid[3:9, 8] = 1.0
    closed_grid[3, 3:9] = 1.0
    closed_grid[8, 3:9] = 1.0
    engine.drones[0].grid = closed_grid

    engine._check_decentralized_mapping_transitions()

    assert engine.drones[0].control_state == "mapping"
    assert engine._pending_lloyd_transition_frames == {"D0": 7}

    engine.frame = 8
    engine._check_decentralized_mapping_transitions()

    assert engine.drones[0].control_state == "lloyd"
    assert engine._pending_lloyd_transition_frames == {}
