import numpy as np

from controller import Controller


class StaticDrone:
    def __init__(self, drone_id, x, y):
        self.drone_id = drone_id
        self.x = float(x)
        self.y = float(y)
        self.known_positions = {
            self.drone_id: np.array([self.x, self.y], dtype=float),
        }
        self.grid = np.zeros((1, 1), dtype=float)
        self.grid_shape = self.grid.shape
        self.grid_bounds = (0.0, 1.0, 0.0, 1.0)
        self.last_control_vector = np.array([9.0, 9.0], dtype=float)

    @property
    def position(self):
        return np.array([self.x, self.y], dtype=float)


def test_mssp_assigns_open_boundary_by_arc_distance():
    points = np.array([[float(i), 0.0] for i in range(5)], dtype=float)
    seeds = [
        {"robot_id": "A", "index": 0},
        {"robot_id": "B", "index": 3},
    ]

    owner, distances = Controller.multi_source_shortest_path_voronoi(
        points,
        seeds,
        is_closed=False,
    )

    assert owner.tolist() == ["A", "A", "B", "B", "B"]
    np.testing.assert_allclose(distances, [0.0, 1.0, 1.0, 0.0, 1.0])


def test_mssp_assigns_closed_boundary_across_wraparound_edge():
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    seeds = [
        {"robot_id": "A", "index": 0},
        {"robot_id": "B", "index": 3},
    ]

    owner, distances = Controller.multi_source_shortest_path_voronoi(
        points,
        seeds,
        is_closed=True,
    )

    assert owner.tolist() == ["A", "A", "B", "B", "B", "A"]
    np.testing.assert_allclose(distances[[0, 3]], [0.0, 0.0])


def test_compute_actions_updates_voronoi_and_moves_toward_lloyd_target():
    controller = Controller(
        sim_map=None,
        communication_radius=100.0,
    )
    controller.initialize_known_boundary(
        np.array([[float(i), 0.0] for i in range(5)], dtype=float),
        force_closed=False,
    )

    drones = [
        StaticDrone("A", 0.0, 0.0),
        StaticDrone("B", 3.0, 0.0),
    ]

    actions = controller.compute_actions(drones)

    assert set(actions) == {"A", "B"}
    for drone in drones:
        assert drone.last_control_mode == "lloyd"
        assert drone.last_ring_info["assigned_drone_indices"].tolist() == [
            "A",
            "A",
            "B",
            "B",
            "B",
        ]

    np.testing.assert_allclose(drones[0].target_centroid, [0.75, 0.0])
    np.testing.assert_allclose(drones[1].target_centroid, [2.75, 0.0])

    np.testing.assert_allclose(actions["A"], [0.12, 0.0])
    np.testing.assert_allclose(actions["B"], [-0.12, 0.0])
    np.testing.assert_allclose(
        drones[0].last_ring_info["current"]["target_arc_length"],
        0.75,
    )


def test_closed_lloyd_targets_split_boundary_by_arc_midpoints():
    points = np.array([[float(i), 0.0] for i in range(8)], dtype=float)
    arc_lengths, total_length = Controller._boundary_arc_lengths(points, is_closed=True)
    seeds = [
        {"robot_id": "A", "index": 0, "arc_length": arc_lengths[0]},
        {"robot_id": "B", "index": 2, "arc_length": arc_lengths[2]},
        {"robot_id": "C", "index": 5, "arc_length": arc_lengths[5]},
    ]

    targets = Controller._lloyd_targets_from_seed_arcs(
        seeds,
        total_length,
        is_closed=True,
    )

    assert set(targets) == {"A", "B", "C"}
    np.testing.assert_allclose(
        sum(target["cell_arc_length"] for target in targets.values()),
        total_length,
    )
    np.testing.assert_allclose(targets["B"]["target_arc_length"], 2.25)
    np.testing.assert_allclose(targets["B"]["cell_start_arc_length"], 1.0)
    np.testing.assert_allclose(targets["B"]["cell_end_arc_length"], 3.5)


def test_closed_geodesic_cell_boundaries_are_midpoints_between_adjacent_seeds():
    controller = Controller(
        sim_map=None,
        communication_radius=100.0,
    )
    points = np.array([[float(i), 0.0] for i in range(10)], dtype=float)
    arc_lengths, total_length = Controller._boundary_arc_lengths(points, is_closed=True)
    seeds = [
        {"robot_id": "A", "index": 0, "arc_length": arc_lengths[0]},
        {"robot_id": "B", "index": 3, "arc_length": arc_lengths[3]},
        {"robot_id": "C", "index": 7, "arc_length": arc_lengths[7]},
    ]

    targets = controller._lloyd_targets_from_seed_arcs(
        seeds,
        total_length,
        is_closed=True,
    )

    ordered = sorted(seeds, key=lambda seed: seed["arc_length"])
    for left_seed, right_seed in zip(ordered, ordered[1:] + ordered[:1]):
        left_s = float(left_seed["arc_length"])
        right_s = float(right_seed["arc_length"])
        gap = (right_s - left_s) % total_length
        boundary = float(targets[right_seed["robot_id"]]["cell_start_arc_length"])
        dist_from_left = (boundary - left_s) % total_length
        dist_from_right = (right_s - boundary) % total_length
        np.testing.assert_allclose(dist_from_left, 0.5 * gap)
        np.testing.assert_allclose(dist_from_right, 0.5 * gap)


def test_compute_ring_ordering_does_not_overwrite_other_drone_targets():
    controller = Controller(
        sim_map=None,
        communication_radius=100.0,
    )
    controller.initialize_known_boundary(
        np.array([[float(i), 0.0] for i in range(5)], dtype=float),
        force_closed=False,
    )

    drone_a = StaticDrone("A", 0.0, 0.0)
    drone_b = StaticDrone("B", 4.0, 0.0)
    drone_a.known_positions = {
        "A": np.array([0.0, 0.0], dtype=float),
        "B": np.array([2.0, 0.0], dtype=float),
    }
    drone_b.known_positions = {
        "A": np.array([0.0, 0.0], dtype=float),
        "B": np.array([4.0, 0.0], dtype=float),
    }

    controller.compute_ring_ordering(drone_a, [drone_a, drone_b])
    np.testing.assert_allclose(drone_a.target_centroid, [0.5, 0.0])

    controller.compute_ring_ordering(drone_b, [drone_a, drone_b])

    np.testing.assert_allclose(drone_a.target_centroid, [0.5, 0.0])
    np.testing.assert_allclose(drone_b.target_centroid, [3.0, 0.0])


def test_equidistant_action_uses_current_seed_arc_from_ring_info():
    controller = Controller(
        sim_map=None,
        communication_radius=100.0,
    )
    points = np.array([[float(i), 0.0] for i in range(6)], dtype=float)
    controller.initialize_known_boundary(points, force_closed=False)
    arc_lengths, total_length = controller._boundary_arc_lengths(points, is_closed=False)

    drone = StaticDrone("A", 2.0, 0.0)
    drone.max_speed = 0.12
    drone.boundary_s = 0.0
    ring_info = {
        "occupied_points": points,
        "arc_lengths": arc_lengths,
        "total_boundary_length": total_length,
        "is_closed": False,
        "current": {
            "seed_arc_length": 2.0,
            "target_arc_length": 1.0,
            "target_centroid": np.array([1.0, 0.0], dtype=float),
        },
    }

    action = controller._equidistant_action(
        drone,
        ring_info,
        world_field=None,
        x_coords=None,
        y_coords=None,
    )

    np.testing.assert_allclose(action, [-0.12, 0.0])


def test_multihop_does_not_overwrite_a_drone_own_current_position():
    controller = Controller(
        sim_map=None,
        communication_radius=100.0,
    )
    drone_a = StaticDrone("A", 1.0, 0.0)
    drone_b = StaticDrone("B", 2.0, 0.0)
    drone_b.known_positions["A"] = np.array([-9.0, 0.0], dtype=float)

    controller._update_multihop_positions([drone_a, drone_b])

    np.testing.assert_allclose(drone_a.known_positions["A"], [1.0, 0.0])


def test_multihop_refreshes_stale_neighbor_positions_with_current_values():
    controller = Controller(
        sim_map=None,
        communication_radius=100.0,
    )
    drone_a = StaticDrone("A", 1.0, 0.0)
    drone_b = StaticDrone("B", 2.0, 0.0)
    drone_c = StaticDrone("C", 3.0, 0.0)
    drone_a.known_positions["C"] = np.array([-9.0, 0.0], dtype=float)

    controller._update_multihop_positions([drone_a, drone_b, drone_c])

    np.testing.assert_allclose(drone_a.known_positions["C"], [3.0, 0.0])
