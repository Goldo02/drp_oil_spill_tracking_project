import numpy as np

from controller import Controller


class DummyMap:
    xlim = (-2.0, 2.0)
    ylim = (-2.0, 2.0)


class DummyDrone:
    def __init__(self, x=0.0, y=0.0, occupied_cell=None):
        self.drone_id = 0
        self.x = float(x)
        self.y = float(y)
        self.grid = np.zeros((41, 41), dtype=float)
        self.grid_bounds = (-2.0, 2.0, -2.0, 2.0)

        if occupied_cell is not None:
            ix, iy = occupied_cell
            self.grid[ix, iy] = 1.0

        self.exploration_direction = np.array([1.0, 0.0], dtype=float)
        self.last_control_mode = "idle"
        self.last_edge_point = None
        self.edge_detected = False


def test_compute_actions_uses_local_grid_not_edge_flag():
    controller = Controller(
        sim_map=DummyMap(),
        communication_radius=1.0,
        occupancy_threshold=0.5,
        resolution=0.1,
    )
    drone = DummyDrone(occupied_cell=(20, 20))

    world_field = np.zeros((41, 41), dtype=float)
    x_coords = np.linspace(-2.0, 2.0, 41)
    y_coords = np.linspace(-2.0, 2.0, 41)

    controller.compute_actions(
        [drone],
        world_field,
        x_coords,
        y_coords,
    )

    assert drone.last_control_mode == "boundary_tracking"
    assert np.linalg.norm(controller.compute_actions([drone], world_field, x_coords, y_coords)[drone.drone_id]) > 0.0


def test_compute_actions_tracks_occupied_target_in_consensus_grid():
    controller = Controller(
        sim_map=DummyMap(),
        communication_radius=1.0,
        occupancy_threshold=0.5,
        resolution=0.1,
    )
    drone = DummyDrone(x=0.0, y=0.0, occupied_cell=(35, 20))

    world_field = np.zeros((41, 41), dtype=float)
    x_coords = np.linspace(-2.0, 2.0, 41)
    y_coords = np.linspace(-2.0, 2.0, 41)

    action = controller.compute_actions([drone], world_field, x_coords, y_coords)[drone.drone_id]

    assert drone.last_control_mode == "boundary_tracking"
    assert np.linalg.norm(action) > 0.0
    assert action[0] > 0.0
