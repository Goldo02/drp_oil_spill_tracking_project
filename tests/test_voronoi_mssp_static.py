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
