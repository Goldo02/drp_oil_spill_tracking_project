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

    receiver.grid[0, 0] = 1.0
    sender.grid[1, 1] = 1.0

    message = sender.create_consensus_message()
    sender.grid[2, 2] = 1.0

    receiver.consensus_step([message])

    assert receiver.grid[0, 0] == 1.0
    assert receiver.grid[1, 1] == 1.0
    assert receiver.grid[2, 2] == 0.0
    assert np.count_nonzero(receiver.grid) == 2
