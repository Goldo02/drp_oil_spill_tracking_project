import numpy as np
import pytest

from controller import DroneController
from drone import Drone
from simulation_engine import SimulationEngine


def make_drone(drone_id, x=0.0):
    controller = DroneController(communication_radius=2.0, resolution=0.1)
    drone = Drone(drone_id, x, 0.0, (32, 32), (-1.6, 1.6, -1.6, 1.6),
                  gps_noise=0.0, controller=controller)
    drone.closure_min_enclosed_false_cells = 8
    return drone


def square_boundary(drone):
    mask = np.zeros(drone.grid_shape, dtype=bool)
    mask[8:24, 8] = mask[8:24, 23] = True
    mask[8, 8:24] = mask[23, 8:24] = True
    drone.boundary_grid = mask.astype(float)
    drone.information_grid[:] = 1.0
    drone.occupancy_signal_grid[:] = 0.0
    drone.occupancy_signal_grid[mask] = 1.0
    drone.grid[:] = drone.occupancy_signal_grid


def send_round(group, frame):
    messages = {drone.drone_id: drone.create_consensus_message() for drone in group}
    for drone in group:
        drone.consensus_step(
            [messages[other.drone_id] for other in group if other is not drone],
            frame=frame,
        )


def test_local_closure_is_shared_by_max_consensus_and_transitions_without_engine():
    d0, d1 = make_drone("D0"), make_drone("D1", 0.2)
    square_boundary(d0)

    send_round([d0, d1], frame=1)
    assert d0.closure_detected and not d1.closure_detected

    send_round([d0, d1], frame=2)
    assert d0.closure_detected and d1.closure_detected
    assert d1.contour_leader_id == "D0"
    np.testing.assert_allclose(d1.contour_points, d0.contour_points)
    assert d0.control_state == "dynamic_lloyd"
    assert d1.control_state == "mapping"

    send_round([d0, d1], frame=3)
    assert d0.control_state == d1.control_state == "dynamic_lloyd"
    assert d0.consecutive_closure_frames >= 2
    assert d1.consecutive_closure_frames == 2


def test_disconnected_components_evolve_independently_then_sync_when_connected():
    a, b, c = (make_drone("A", 0.0), make_drone("B", 0.2), make_drone("C", 5.0))
    for drone, leader, quality, epoch in ((a, "A", 0.2, 1), (b, "B", 0.9, 2), (c, "C", 0.5, 4)):
        drone.closure_detected = True
        drone.contour_points = np.array([[0., 0.], [1., 0.], [1., 1.]]) + ord(leader) * 0.001
        drone.contour_leader_id = leader
        drone.contour_quality = quality
        drone.contour_epoch = epoch

    engine = object.__new__(SimulationEngine)
    engine.drones = [a, b, c]
    engine.fully_connected = False
    engine.communication_radius = 1.0
    engine.occupancy_threshold = 0.6
    engine.frame = 1
    engine._exchange_consensus_messages()
    assert a.contour_leader_id == b.contour_leader_id == "B"
    assert c.contour_leader_id == "C"
    c.x = 0.4
    c.last_gps_position = c.position.copy()
    engine.frame = 2
    engine._exchange_consensus_messages()
    assert a.contour_leader_id == b.contour_leader_id == c.contour_leader_id == "C"
    np.testing.assert_allclose(a.contour_points, c.contour_points)


def test_local_platooning_feedback_reduces_ring_gap_imbalance():
    length = 20.0
    positions = np.array([0.0, 3.0, 11.0])
    identifiers = ["A", "B", "C"]
    controller = DroneController()

    def gap_spread(values):
        ordered = np.sort(values)
        gaps = np.diff(np.r_[ordered, ordered[0] + length])
        return float(np.std(gaps))

    initial_spread = gap_spread(positions)
    for _ in range(300):
        arcs = {did: float(s) for did, s in zip(identifiers, positions)}
        speeds = np.array([
            controller._platooning_speed(positions[i], arcs, identifiers[i], length)
            for i in range(len(positions))
        ])
        positions = (positions + speeds * 1.0) % length
    assert gap_spread(positions) < initial_spread


def test_isolated_patrol_drone_moves_at_nominal_speed():
    drone = make_drone("solo")
    drone.control_state = "boundary_patrol"
    drone.set_known_boundary(np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.]]))
    drone.boundary_s = 0.0
    drone.known_boundary_arcs = {"solo": 0.0}
    drone.last_gps_position = np.array([0., 0.])
    drone.controller._patrol_action(drone, dt=2.0)
    assert drone.patrol_target_s == pytest.approx(0.1)
