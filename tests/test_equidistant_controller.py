import matplotlib
matplotlib.use("Agg")

import numpy as np
import unittest

from controller import Controller
from drone import Drone
from visualization import Visualizer


class DummyMap:
    xlim = (-5.0, 5.0)
    ylim = (-5.0, 5.0)
    dx = 0.1
    dy = 0.1


class TestEquidistantController(unittest.TestCase):
    def setUp(self):
        self.sim_map = DummyMap()
        self.controller = Controller(
            sim_map=self.sim_map,
            communication_radius=3.0,
            fully_connected=False,
            occupancy_threshold=0.5,
            resolution=0.1,
            k_spacing=1.5,
        )

    # ==================================================================
    # 1. DFS POLYGON CLOSURE TESTS
    # ==================================================================

    def test_dfs_empty_grid_not_closed(self):
        grid = np.zeros((50, 50), dtype=float)
        self.assertFalse(self.controller.is_polygon_closed(grid))

    def test_dfs_open_line_not_closed(self):
        grid = np.zeros((50, 50), dtype=float)
        # Vertical wall from y=10 to 40 at x=25 (open at top and bottom)
        grid[25, 10:40] = 1.0
        self.assertFalse(self.controller.is_polygon_closed(grid))

    def test_dfs_open_c_shape_not_closed(self):
        grid = np.zeros((50, 50), dtype=float)
        # C-shape: open on the right side
        grid[15:35, 15] = 1.0  # left vertical
        grid[15, 15:35] = 1.0  # top horizontal
        grid[35, 15:35] = 1.0  # bottom horizontal
        self.assertFalse(self.controller.is_polygon_closed(grid))

    def test_dfs_closed_box_is_closed(self):
        grid = np.zeros((50, 50), dtype=float)
        # Closed rectangle: left, right, top, bottom
        grid[15:36, 15] = 1.0
        grid[15:36, 35] = 1.0
        grid[15, 15:36] = 1.0
        grid[35, 15:36] = 1.0
        self.assertTrue(self.controller.is_polygon_closed(grid))

    def test_dfs_closed_circle_is_closed(self):
        grid = np.zeros((60, 60), dtype=float)
        # Draw a discretized circle ring of radius 15 at center (30, 30)
        center = np.array([30, 30])
        for x in range(60):
            for y in range(60):
                dist = np.hypot(x - center[0], y - center[1])
                if 14.0 <= dist <= 16.0:
                    grid[x, y] = 1.0

        self.assertTrue(self.controller.is_polygon_closed(grid))

    # ==================================================================
    # 2. MULTI-HOP POSITION CONSENSUS TESTS
    # ==================================================================

    def test_consensus_is_disabled_keeps_occupancy_grid_static_and_keeps_positions_private(self):
        d0 = Drone("D0", 0.0, 0.0, (20, 20), (-2.0, 2.0, -2.0, 2.0))
        d1 = Drone("D1", 1.0, 0.0, (20, 20), (-2.0, 2.0, -2.0, 2.0))
        d2 = Drone("D2", 2.0, 0.0, (20, 20), (-2.0, 2.0, -2.0, 2.0))

        d0.grid.fill(0.0)
        d1.grid.fill(0.0)
        d2.grid.fill(0.0)
        d0.grid[10, 10] = 1.0
        d1.grid[11, 11] = 1.0
        d2.grid[12, 12] = 1.0

        d0_grid_before = d0.grid.copy()
        d1_grid_before = d1.grid.copy()
        d2_grid_before = d2.grid.copy()

        drones = [d0, d1, d2]
        controller = Controller(
            sim_map=self.sim_map,
            communication_radius=2.5,
            fully_connected=False,
        )

        controller.consensus_step(drones)

        np.testing.assert_array_equal(d0.grid, d0_grid_before)
        np.testing.assert_array_equal(d1.grid, d1_grid_before)
        np.testing.assert_array_equal(d2.grid, d2_grid_before)
        self.assertEqual(list(d0.known_positions.keys()), ["D0"])
        self.assertEqual(list(d1.known_positions.keys()), ["D1"])
        self.assertEqual(list(d2.known_positions.keys()), ["D2"])

    def test_consensus_is_disabled_no_multi_hop_position_sharing(self):
        # Create 3 drones in a line: D0 at (0, 0), D1 at (2, 0), D2 at (4, 0)
        # The controller intentionally performs no robot-to-robot consensus.
        d0 = Drone("D0", 0.0, 0.0, (50, 50), (-5.0, 5.0, -5.0, 5.0))
        d1 = Drone("D1", 2.0, 0.0, (50, 50), (-5.0, 5.0, -5.0, 5.0))
        d2 = Drone("D2", 4.0, 0.0, (50, 50), (-5.0, 5.0, -5.0, 5.0))
        drones = [d0, d1, d2]

        controller = Controller(
            sim_map=self.sim_map,
            communication_radius=2.5,
            fully_connected=False,
        )

        self.assertEqual(list(d0.known_positions.keys()), ["D0"])
        self.assertNotIn("D2", d0.known_positions)

        controller.consensus_step(drones)
        self.assertEqual(list(d0.known_positions.keys()), ["D0"])
        self.assertEqual(list(d1.known_positions.keys()), ["D1"])
        self.assertEqual(list(d2.known_positions.keys()), ["D2"])

        controller.consensus_step(drones)
        self.assertEqual(list(d0.known_positions.keys()), ["D0"])
        self.assertEqual(list(d2.known_positions.keys()), ["D2"])
        self.assertNotIn("D2", d0.known_positions)
        self.assertNotIn("D0", d2.known_positions)

    # ==================================================================
    # 3. RING ORDERING & CENTER OF MASS TESTS
    # ==================================================================

    def test_ring_ordering_and_neighbor_resolution(self):
        # Create a closed circle on a grid with radius 2.0 centered at (0.0, 0.0)
        grid_shape = (100, 100)
        grid_bounds = (-5.0, 5.0, -5.0, 5.0)
        grid = np.zeros(grid_shape, dtype=float)

        for ix in range(100):
            for iy in range(100):
                x = -5.0 + (ix + 0.5) * 0.1
                y = -5.0 + (iy + 0.5) * 0.1
                r = np.hypot(x, y)
                if 1.9 <= r <= 2.1:
                    grid[ix, iy] = 1.0

        # Create 4 drones positioned at angles 0, pi/2, pi, -pi/2 (3*pi/2)
        d0 = Drone("D0", 2.0, 0.0, grid_shape, grid_bounds)      # theta ~ 0
        d1 = Drone("D1", 0.0, 2.0, grid_shape, grid_bounds)      # theta ~ pi/2
        d2 = Drone("D2", -2.0, 0.0, grid_shape, grid_bounds)     # theta ~ pi
        d3 = Drone("D3", 0.0, -2.0, grid_shape, grid_bounds)     # theta ~ -pi/2
        drones = [d0, d1, d2, d3]

        for d in drones:
            d.grid = grid.copy()
            d.known_positions = {
                other.drone_id: np.array([other.x, other.y], dtype=float)
                for other in drones
            }

        ring_d0 = self.controller.compute_ring_ordering(d0, drones)
        self.assertIsNotNone(ring_d0)
        self.assertEqual(ring_d0["N"], 4)
        self.assertEqual(ring_d0["current"]["drone_id"], "D0")
        # In counter-clockwise angle order: D3 (-pi/2) -> D0 (0) -> D1 (pi/2) -> D2 (pi)
        self.assertEqual(ring_d0["succ"]["drone_id"], "D1")
        self.assertEqual(ring_d0["pred"]["drone_id"], "D3")

        # Check center of mass is near (0, 0)
        np.testing.assert_allclose(ring_d0["center_of_mass"], [0.0, 0.0], atol=0.1)

    # ==================================================================
    # 4. DYNAMIC TANGENTIAL GAIN (k_{t,i}) TESTS
    # ==================================================================

    def test_dynamic_tangential_gain_positive_and_negative(self):
        # Test spacing error calculation and negative gain for overtake
        # Case A: Equidistant spacing: 4 drones evenly spaced at 0, pi/2, pi, 3pi/2
        ring_info_balanced = {
            "N": 4,
            "current": {"angle": 0.0},
            "succ": {"angle": np.pi / 2.0},
            "pred": {"angle": -np.pi / 2.0},
        }
        theta_ideal = 2.0 * np.pi / 4.0  # pi/2
        delta_succ = (ring_info_balanced["succ"]["angle"] - ring_info_balanced["current"]["angle"]) % (2 * np.pi)
        delta_pred = (ring_info_balanced["current"]["angle"] - ring_info_balanced["pred"]["angle"]) % (2 * np.pi)
        spacing_error = (delta_succ - delta_pred) / theta_ideal
        k_t_i = self.controller.k_t + self.controller.k_spacing * spacing_error
        self.assertAlmostEqual(k_t_i, self.controller.k_t, places=5)

        # Case B: Drone is very close to successor (gap ahead small, gap behind large)
        # e.g. delta_succ = 0.1, delta_pred = 2.0
        ring_info_tight_ahead = {
            "N": 4,
            "current": {"angle": 1.4},
            "succ": {"angle": 1.5},   # delta_succ = 0.1
            "pred": {"angle": -0.5},  # delta_pred = 1.9
        }
        delta_succ = (ring_info_tight_ahead["succ"]["angle"] - ring_info_tight_ahead["current"]["angle"]) % (2 * np.pi)
        delta_pred = (ring_info_tight_ahead["current"]["angle"] - ring_info_tight_ahead["pred"]["angle"]) % (2 * np.pi)
        spacing_error = (delta_succ - delta_pred) / theta_ideal
        k_t_negative = self.controller.k_t + self.controller.k_spacing * spacing_error
        # Because delta_succ (0.1) < delta_pred (1.9), spacing_error is negative:
        # (0.1 - 1.9) / (pi/2) = -1.8 / 1.57 = -1.1459
        # k_t = 1.0 + 1.5 * (-1.1459) = -0.7188 -> strictly negative!
        self.assertLess(k_t_negative, 0.0)

        # Case C: Drone has huge gap ahead and is close to predecessor behind
        ring_info_gap_ahead = {
            "N": 4,
            "current": {"angle": 0.1},
            "succ": {"angle": 2.0},
            "pred": {"angle": 0.0},
        }
        delta_succ = (ring_info_gap_ahead["succ"]["angle"] - ring_info_gap_ahead["current"]["angle"]) % (2 * np.pi)
        delta_pred = (ring_info_gap_ahead["current"]["angle"] - ring_info_gap_ahead["pred"]["angle"]) % (2 * np.pi)
        spacing_error = (delta_succ - delta_pred) / theta_ideal
        k_t_accelerate = self.controller.k_t + self.controller.k_spacing * spacing_error
        self.assertGreater(k_t_accelerate, self.controller.k_t)

    def test_equidistant_action_uses_lloyd_coverage_target(self):
        """The 1D Lloyd centroid of the assigned Voronoi segment must directly influence motion."""
        drone = Drone("D0", 0.0, 0.0, (101, 101), (-5.0, 5.0, -5.0, 5.0))
        drone.known_positions = {"D0": np.array([0.0, 0.0])}
        drone.target_centroid = np.array([2.0, 0.0], dtype=float)

        ring_info = {
            "N": 4,
            "current": {"angle": 0.0, "target_centroid": np.array([2.0, 0.0], dtype=float)},
            "pred": {"angle": -0.2},
            "succ": {"angle": 0.2},
            "center_of_mass": np.array([0.0, 0.0], dtype=float),
        }

        x_coords = np.linspace(-5.0, 5.0, 101)
        y_coords = np.linspace(-5.0, 5.0, 101)
        X, Y = np.meshgrid(x_coords, y_coords, indexing="xy")
        world_field = np.zeros_like(X, dtype=float)

        original = self.controller._boundary_tracking_action
        self.controller._boundary_tracking_action = lambda *args, **kwargs: np.zeros(2, dtype=float)
        try:
            action = self.controller._equidistant_action(drone, ring_info, world_field, x_coords, y_coords)
        finally:
            self.controller._boundary_tracking_action = original

        self.assertIsNotNone(action)
        self.assertGreater(np.dot(action, np.array([2.0, 0.0], dtype=float)), 0.0)

    def test_equidistant_action_keeps_moving_when_ring_spacing_collapses(self):
        """If the ring compresses, a small tangential push must keep the drone moving instead of freezing."""
        drone = Drone("D0", 1.0, 0.0, (101, 101), (-5.0, 5.0, -5.0, 5.0))
        drone.known_positions = {"D0": np.array([1.0, 0.0])}
        drone.target_centroid = np.array([1.0, 0.0], dtype=float)

        ring_info = {
            "N": 4,
            "current": {"angle": 0.0, "target_centroid": np.array([1.0, 0.0], dtype=float)},
            "pred": {"angle": -0.04},
            "succ": {"angle": 0.04},
            "center_of_mass": np.array([0.0, 0.0], dtype=float),
        }

        x_coords = np.linspace(-5.0, 5.0, 101)
        y_coords = np.linspace(-5.0, 5.0, 101)
        X, Y = np.meshgrid(x_coords, y_coords, indexing="xy")
        world_field = np.exp(-((X ** 2 + Y ** 2) - 1.0) ** 2 / 0.05)

        original = self.controller._boundary_tracking_action
        self.controller._boundary_tracking_action = lambda *args, **kwargs: np.zeros(2, dtype=float)
        try:
            action = self.controller._equidistant_action(drone, ring_info, world_field, x_coords, y_coords)
        finally:
            self.controller._boundary_tracking_action = original

        self.assertIsNotNone(action)
        self.assertGreater(np.linalg.norm(action), 1e-6)

    # ==================================================================
    # 5. STATE MACHINE ACTIONS & SETTLING PHASE IN COMPUTE_ACTIONS
    # ==================================================================

    def test_state_transitions_in_compute_actions(self):
        grid_shape = (50, 50)
        grid_bounds = (-2.5, 2.5, -2.5, 2.5)

        # Boundary-only control model: all drones are assumed to already lie on the boundary.
        drone_on_ring = Drone("D_ring", 0.0, 1.5, grid_shape, grid_bounds)
        for ix in range(50):
            for iy in range(50):
                x = -2.5 + (ix + 0.5) * 0.1
                y = -2.5 + (iy + 0.5) * 0.1
                if 1.4 <= np.hypot(x, y) <= 1.6:
                    drone_on_ring.grid[ix, iy] = 1.0

        world_field = np.zeros((50, 50), dtype=float)
        x_coords = np.linspace(-2.5, 2.5, 50)
        y_coords = np.linspace(-2.5, 2.5, 50)

        self.controller.compute_actions(
            [drone_on_ring],
            world_field,
            x_coords,
            y_coords,
        )

        self.assertEqual(drone_on_ring.last_control_mode, "equi_distant")
        self.assertEqual(getattr(drone_on_ring, "settling_counter", 0), 0)

    def test_settling_synchronization_phase(self):
        controller = Controller(
            sim_map=self.sim_map,
            communication_radius=3.0,
            settling_steps=15,
        )
        grid_shape = (50, 50)
        grid_bounds = (-2.5, 2.5, -2.5, 2.5)

        drone = Drone("D_sync", 0.0, 1.5, grid_shape, grid_bounds)
        for ix in range(50):
            for iy in range(50):
                x = -2.5 + (ix + 0.5) * 0.1
                y = -2.5 + (iy + 0.5) * 0.1
                if 1.4 <= np.hypot(x, y) <= 1.6:
                    drone.grid[ix, iy] = 1.0

        world_field = np.zeros((50, 50), dtype=float)
        x_coords = np.linspace(-2.5, 2.5, 50)
        y_coords = np.linspace(-2.5, 2.5, 50)

        for _ in range(5):
            controller.compute_actions([drone], world_field, x_coords, y_coords)

        self.assertEqual(drone.last_control_mode, "equi_distant")
        self.assertEqual(getattr(drone, "settling_counter", 0), 0)
        self.assertIsNotNone(drone.target_centroid)

    def test_known_boundary_is_closed_and_drives_equidistant_mode_from_frame_zero(self):
        grid_shape = (80, 80)
        grid_bounds = (-4.0, 4.0, -4.0, 4.0)
        x_coords = np.linspace(grid_bounds[0], grid_bounds[1], grid_shape[0])
        y_coords = np.linspace(grid_bounds[2], grid_bounds[3], grid_shape[1])
        grid = np.zeros(grid_shape, dtype=float)

        for ix in range(grid_shape[0]):
            for iy in range(grid_shape[1]):
                x = x_coords[ix]
                y = y_coords[iy]
                r = np.hypot(x, y)
                if 2.7 <= r <= 3.1:
                    grid[ix, iy] = 1.0

        self.controller.initialize_known_boundary(grid, x_coords, y_coords)
        self.assertTrue(self.controller.known_boundary_closed)
        self.assertGreater(len(self.controller.known_boundary_points), 0)

        boundary_points = self.controller.known_boundary_points.copy()
        rng = np.random.default_rng(7)
        selected = boundary_points[rng.choice(len(boundary_points), size=3, replace=False)]

        drones = []
        for i, (x, y) in enumerate(selected):
            drone = Drone(f"D{i}", x, y, grid_shape, grid_bounds)
            drone.grid = np.zeros_like(grid)
            drone.grid[grid > 0.5] = 0.0
            drones.append(drone)

        for drone in drones:
            drone.grid = grid.copy()
            drone.known_positions = {
                other.drone_id: np.array([other.x, other.y], dtype=float)
                for other in drones
            }

        actions = self.controller.compute_actions(drones, grid, x_coords, y_coords)

        for drone in drones:
            self.assertEqual(drone.last_control_mode, "equi_distant")
            self.assertIn(drone.drone_id, actions)
            self.assertGreater(np.linalg.norm(actions[drone.drone_id]), 0.0)

    def test_equidistant_action_uses_known_boundary_anchor(self):
        ctrl = Controller(
            sim_map=self.sim_map,
            communication_radius=10.0,
            fully_connected=True,
            d_safe=0.5,
            repulsion_gain=0.3,
        )

        boundary = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ], dtype=float)
        ctrl.known_boundary_points = boundary.copy()

        drone = Drone("D0", 0.2, 0.0, (20, 20), (-5.0, 5.0, -5.0, 5.0))
        drone.known_positions = {
            "D0": np.array([0.2, 0.0], dtype=float),
            "D1": np.array([0.8, 0.0], dtype=float),
        }

        ring_info = {
            "N": 2,
            "current": {"angle": 0.0, "target_centroid": np.array([1.0, 0.0], dtype=float)},
            "pred": {"angle": 3.0 * np.pi / 2.0},
            "succ": {"angle": np.pi / 2.0},
            "center_of_mass": np.array([0.0, 0.0], dtype=float),
        }

        def _must_not_call_boundary_tracking(*args, **kwargs):
            raise AssertionError("_equidistant_action must not depend on field-gradient tracking")

        ctrl._boundary_tracking_action = _must_not_call_boundary_tracking
        action = ctrl._equidistant_action(drone, ring_info, None, None, None)

        self.assertTrue(np.all(np.isfinite(action)))
        self.assertGreater(np.linalg.norm(action), 0.0)

    def test_ring_partition_colors_each_assigned_segment(self):
        vis = Visualizer(self.sim_map)

        occupied = np.array([
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [1.5, 0.0],
            [2.0, 0.0],
        ], dtype=float)

        d0 = Drone("D0", 0.25, 0.0, (20, 20), (-5.0, 5.0, -5.0, 5.0))
        d1 = Drone("D1", 1.75, 0.0, (20, 20), (-5.0, 5.0, -5.0, 5.0))

        d0.last_ring_info = {
            "occupied_points": occupied,
            "assigned_drone_indices": np.array([0, 0, 1, 1, 1], dtype=int),
            "current_idx": 0,
        }
        d1.last_ring_info = {
            "occupied_points": occupied,
            "assigned_drone_indices": np.array([0, 0, 1, 1, 1], dtype=int),
            "current_idx": 1,
        }

        vis.update_ring_partition([d0, d1])

        ring_artists = vis.voronoi_ring_artists
        self.assertGreater(len(ring_artists), 1)
        self.assertTrue(
            any(
                artist is not None and len(artist.get_facecolors()) > 1
                for artist in ring_artists
            )
        )

    def test_clustered_drones_target_global_ring_midpoints(self):
        grid_shape = (100, 100)
        grid = np.zeros(grid_shape, dtype=float)
        for ix in range(100):
            for iy in range(100):
                x = -5.0 + (ix + 0.5) * 0.1
                y = -5.0 + (iy + 0.5) * 0.1
                r = np.hypot(x, y)
                if 1.9 <= r <= 2.1:
                    grid[ix, iy] = 1.0

        drones = [
            Drone("D0", -2.0, 0.0, grid_shape, (-5.0, 5.0, -5.0, 5.0)),
            Drone("D1", -1.8, 0.8, grid_shape, (-5.0, 5.0, -5.0, 5.0)),
            Drone("D2", -2.4, 1.0, grid_shape, (-5.0, 5.0, -5.0, 5.0)),
            Drone("D3", -2.2, -1.1, grid_shape, (-5.0, 5.0, -5.0, 5.0)),
            Drone("D4", -1.5, -0.7, grid_shape, (-5.0, 5.0, -5.0, 5.0)),
        ]
        for d in drones:
            d.grid = grid.copy()
            d.known_positions = {other.drone_id: np.array([other.x, other.y], dtype=float) for other in drones}

        for d in drones:
            self.controller.compute_ring_ordering(d, drones)

        x_targets = np.array([d.target_centroid[0] for d in drones])
        self.assertGreater(np.sum(x_targets > 0.0), 0)

    def test_boundary_voronoi_partition_and_target_centroids(self):
        grid_shape = (100, 100)
        grid_bounds = (-5.0, 5.0, -5.0, 5.0)
        grid = np.zeros(grid_shape, dtype=float)

        for ix in range(100):
            for iy in range(100):
                x = -5.0 + (ix + 0.5) * 0.1
                y = -5.0 + (iy + 0.5) * 0.1
                r = np.hypot(x, y)
                if 1.9 <= r <= 2.1:
                    grid[ix, iy] = 1.0

        # Create 4 drones positioned at angles 0, pi/2, pi, -pi/2
        d0 = Drone("D0", 2.0, 0.0, grid_shape, grid_bounds)
        d1 = Drone("D1", 0.0, 2.0, grid_shape, grid_bounds)
        d2 = Drone("D2", -2.0, 0.0, grid_shape, grid_bounds)
        d3 = Drone("D3", 0.0, -2.0, grid_shape, grid_bounds)
        drones = [d0, d1, d2, d3]

        for d in drones:
            d.grid = grid.copy()
            d.known_positions = {
                other.drone_id: np.array([other.x, other.y], dtype=float)
                for other in drones
            }

        ring_d0 = self.controller.compute_ring_ordering(d0, drones)
        self.assertIsNotNone(ring_d0)

        # Verify each drone has a target_centroid and voronoi_cell_size
        total_cells = 0
        for entry in ring_d0["ring"]:
            self.assertIn("target_centroid", entry)
            self.assertIn("voronoi_cell_size", entry)
            self.assertGreater(entry["voronoi_cell_size"], 0)
            total_cells += entry["voronoi_cell_size"]

        # Total points in Voronoi cells must equal total occupied points
        self.assertEqual(total_cells, len(ring_d0["occupied_points"]))

        # For D0 placed symmetrically at (2.0, 0.0), its target centroid should be near (2.0, 0.0)
        np.testing.assert_allclose(d0.target_centroid, [2.0, 0.0], atol=0.2)

    # ==================================================================
    # 12. INTER-DRONE REPULSION (COLLISION AVOIDANCE)
    # ==================================================================

    def test_repulsion_nonzero_when_drones_overlap(self):
        """Two drones within d_safe must receive a non-zero repulsion pointing away from each other."""
        ctrl = Controller(
            sim_map=self.sim_map,
            communication_radius=10.0,
            fully_connected=True,
            d_safe=0.5,
            repulsion_gain=0.3,
        )

        # Place drone_a at origin and drone_b at (0.2, 0.0) — within d_safe=0.5
        drone_a = Drone.__new__(Drone)
        drone_a.drone_id = "Da"
        drone_a.x = 0.0
        drone_a.y = 0.0
        drone_a.known_positions = {
            "Da": np.array([0.0, 0.0]),
            "Db": np.array([0.2, 0.0]),
        }

        drone_b = Drone.__new__(Drone)
        drone_b.drone_id = "Db"
        drone_b.x = 0.2
        drone_b.y = 0.0
        drone_b.known_positions = drone_a.known_positions

        rep_a = ctrl._compute_repulsion(drone_a)
        rep_b = ctrl._compute_repulsion(drone_b)

        # Repulsion vectors must be non-zero
        self.assertGreater(np.linalg.norm(rep_a), 0.0, "drone_a should feel repulsion")
        self.assertGreater(np.linalg.norm(rep_b), 0.0, "drone_b should feel repulsion")

        # rep_a should point in the -x direction (away from drone_b at +x)
        self.assertLess(rep_a[0], 0.0, "drone_a repulsion x-component should be negative (push left)")
        # rep_b should point in the +x direction (away from drone_a at -x)
        self.assertGreater(rep_b[0], 0.0, "drone_b repulsion x-component should be positive (push right)")

        # Newton's 3rd law: equal and opposite magnitudes
        np.testing.assert_allclose(np.linalg.norm(rep_a), np.linalg.norm(rep_b), atol=1e-10)

    def test_repulsion_zero_when_drones_far_apart(self):
        """Drones outside d_safe must produce zero repulsion."""
        ctrl = Controller(
            sim_map=self.sim_map,
            communication_radius=10.0,
            d_safe=0.5,
            repulsion_gain=0.3,
        )

        drone_a = Drone.__new__(Drone)
        drone_a.drone_id = "Da"
        drone_a.x = 0.0
        drone_a.y = 0.0
        drone_a.known_positions = {
            "Da": np.array([0.0, 0.0]),
            "Db": np.array([2.0, 0.0]),   # 2.0 m apart >> d_safe
        }

        rep = ctrl._compute_repulsion(drone_a)
        np.testing.assert_array_equal(rep, np.zeros(2), "no repulsion expected when drones are far apart")


if __name__ == "__main__":
    unittest.main()
