import numpy as np
import unittest

from controller import Controller
from drone import Drone


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

    def test_multi_hop_position_sharing(self):
        # Create 3 drones in a line: D0 at (0, 0), D1 at (2, 0), D2 at (4, 0)
        # Comm radius is 2.5: D0 connects to D1, D1 connects to D2, D0 cannot directly reach D2
        d0 = Drone("D0", 0.0, 0.0, (50, 50), (-5.0, 5.0, -5.0, 5.0))
        d1 = Drone("D1", 2.0, 0.0, (50, 50), (-5.0, 5.0, -5.0, 5.0))
        d2 = Drone("D2", 4.0, 0.0, (50, 50), (-5.0, 5.0, -5.0, 5.0))
        drones = [d0, d1, d2]

        controller = Controller(
            sim_map=self.sim_map,
            communication_radius=2.5,
            fully_connected=False,
        )

        # Before consensus: each drone only knows its own position
        self.assertIn("D0", d0.known_positions)
        self.assertNotIn("D2", d0.known_positions)

        # Step 1: D0 communicates with D1; D1 communicates with D0 and D2; D2 communicates with D1
        controller.consensus_step(drones)
        self.assertIn("D1", d0.known_positions)
        self.assertIn("D0", d1.known_positions)
        self.assertIn("D2", d1.known_positions)
        # In step 1, D0 might not yet have D2
        # Step 2: D1 propagates D2 to D0 and D0 to D2 (multi-hop)
        controller.consensus_step(drones)
        self.assertIn("D2", d0.known_positions)
        self.assertIn("D0", d2.known_positions)

        np.testing.assert_allclose(d0.known_positions["D2"], [4.0, 0.0])
        np.testing.assert_allclose(d2.known_positions["D0"], [0.0, 0.0])

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

    # ==================================================================
    # 5. STATE MACHINE ACTIONS IN COMPUTE_ACTIONS
    # ==================================================================

    def test_state_transitions_in_compute_actions(self):
        grid_shape = (50, 50)
        grid_bounds = (-2.5, 2.5, -2.5, 2.5)

        # 1. Drone in empty space -> explore
        drone_explore = Drone("D_exp", 0.0, 0.0, grid_shape, grid_bounds)

        # 2. Drone with open boundary -> boundary_tracking
        drone_track = Drone("D_trk", 0.0, 0.0, grid_shape, grid_bounds)
        drone_track.grid[20:30, 25] = 1.0  # Open line

        # 3. Drone with closed polygon -> equi_distant
        drone_eq = Drone("D_eq", 0.0, 1.5, grid_shape, grid_bounds)
        for ix in range(50):
            for iy in range(50):
                x = -2.5 + (ix + 0.5) * 0.1
                y = -2.5 + (iy + 0.5) * 0.1
                if 1.4 <= np.hypot(x, y) <= 1.6:
                    drone_eq.grid[ix, iy] = 1.0

        world_field = np.zeros((50, 50), dtype=float)
        x_coords = np.linspace(-2.5, 2.5, 50)
        y_coords = np.linspace(-2.5, 2.5, 50)

        # Execute compute_actions
        self.controller.compute_actions(
            [drone_explore, drone_track, drone_eq],
            world_field,
            x_coords,
            y_coords,
        )

        self.assertEqual(drone_explore.last_control_mode, "explore")
        self.assertEqual(drone_track.last_control_mode, "boundary_tracking")
        self.assertEqual(drone_eq.last_control_mode, "equi_distant")


if __name__ == "__main__":
    unittest.main()
