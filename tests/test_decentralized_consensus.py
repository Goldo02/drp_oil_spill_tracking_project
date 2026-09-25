import numpy as np

from drone import Drone


def make_drone(drone_id):
    return Drone(
        drone_id=drone_id,
        x=0.0,
        y=0.0,
        grid_shape=(4, 4),
        grid_bounds=(-1.0, 1.0, -1.0, 1.0),
        sensor_size=4,
    )


def test_consensus_step_uses_received_messages_not_live_neighbors():
    receiver = make_drone("D0")
    sender = make_drone("D1")

    receiver.occupancy_signal_grid[0, 0] = 1.0
    receiver.information_grid[0, 0] = 1.0
    receiver._refresh_probability_grid()
    sender.occupancy_signal_grid[1, 1] = 1.0
    sender.information_grid[1, 1] = 1.0
    sender._refresh_probability_grid()

    message = sender.create_consensus_message()
    sender.occupancy_signal_grid[2, 2] = 1.0
    sender.information_grid[2, 2] = 1.0
    sender._refresh_probability_grid()

    receiver.consensus_step([message])

    assert receiver.grid[0, 0] == 1.0
    assert receiver.grid[1, 1] == 1.0
    assert receiver.grid[2, 2] == 0.5
    assert receiver.information_grid[0, 0] == 0.5
    assert receiver.information_grid[1, 1] == 0.5
    assert receiver.information_grid[2, 2] == 0.0


def test_consensus_step_averages_conflicting_observations():
    receiver = make_drone("D0")
    sender = make_drone("D1")

    receiver.occupancy_signal_grid[0, 0] = 1.0
    receiver.information_grid[0, 0] = 1.0
    receiver._refresh_probability_grid()
    sender.occupancy_signal_grid[0, 0] = 0.0
    sender.information_grid[0, 0] = 1.0
    sender._refresh_probability_grid()

    receiver.consensus_step([sender.create_consensus_message()])

    assert receiver.information_grid[0, 0] == 1.0
    assert receiver.grid[0, 0] == 0.5
