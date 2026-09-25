import numpy as np
import pytest

from controller import Controller
from drone import Drone
from main import _save_oil_mapping


class RecordingCamera:
    def __init__(self):
        self.calls = []

    def sense(self, **kwargs):
        self.calls.append(kwargs)
        return None


class BrokenGPSDrone:
    drone_id = "D0"
    gps = object()
    last_gps_position = np.array([np.nan, 0.0], dtype=float)

    @property
    def position(self):
        return np.array([10.0, 20.0], dtype=float)


class EmptyConsensusEngine:
    occupancy_threshold = 0.5
    x_min = -1.0
    y_min = -1.0
    resolution = 0.1
    grid_bounds = (-1.0, 1.0, -1.0, 1.0)
    grid_shape = (4, 4)

    def compute_mean_grid(self):
        return np.zeros(self.grid_shape, dtype=float)


class DummyMap:
    xlim = (-1.0, 1.0)
    ylim = (-1.0, 1.0)
    grid_size = 4
    dx = 0.1
    dy = 0.1


def test_drone_camera_request_is_centered_on_gps_estimate():
    drone = Drone(
        drone_id="D0",
        x=0.0,
        y=0.0,
        grid_shape=(4, 4),
        grid_bounds=(-1.0, 1.0, -1.0, 1.0),
        gps_noise=0.0,
    )
    drone.last_gps_position = np.array([0.75, -0.25], dtype=float)
    drone.update_position_estimate = lambda: drone.last_gps_position.copy()
    camera = RecordingCamera()
    drone.camera = camera

    drone.sense(
        np.zeros((4, 4), dtype=float),
        np.linspace(-1.0, 1.0, 4),
        np.linspace(-1.0, 1.0, 4),
    )

    assert camera.calls
    assert camera.calls[0]["x"] == pytest.approx(0.75)
    assert camera.calls[0]["y"] == pytest.approx(-0.25)
    np.testing.assert_allclose(
        camera.calls[0]["position_estimate"],
        [0.75, -0.25],
    )


def test_instrumented_drone_position_does_not_fall_back_to_truth_without_gps():
    with pytest.raises(ValueError, match="GPS position estimate"):
        Controller._estimated_position(BrokenGPSDrone())


def test_oil_mapping_export_does_not_fall_back_to_spill_model(tmp_path):
    class SpillWithGroundTruthBoundary:
        boundary = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        )

    with pytest.raises(RuntimeError, match="sensor-derived consensus"):
        _save_oil_mapping(
            EmptyConsensusEngine(),
            DummyMap(),
            SpillWithGroundTruthBoundary(),
            tmp_path / "oil_mapping.npy",
        )
