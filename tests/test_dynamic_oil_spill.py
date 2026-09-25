import numpy as np

from environment import SimulationMap, SmoothedPolygonOilSpill
from main import _dynamic_drift_velocity


def test_smoothed_polygon_drift_translates_geometry_and_field():
    sim_map = SimulationMap(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0), grid_size=80)
    spill = SmoothedPolygonOilSpill(
        sim_map.X,
        sim_map.Y,
        seed=7,
        drift_velocity=(0.1, -0.05),
    )

    initial_center = np.array([spill.x0, spill.y0], dtype=float)
    initial_vertices = spill.vertices.copy()
    initial_boundary = spill.boundary.copy()
    initial_field = spill.get_field()

    spill.update(2.0)

    displacement = np.array([0.2, -0.1], dtype=float)
    np.testing.assert_allclose([spill.x0, spill.y0], initial_center + displacement)
    np.testing.assert_allclose(spill.vertices, initial_vertices + displacement)
    np.testing.assert_allclose(spill.boundary, initial_boundary + displacement)
    assert not np.allclose(spill.get_field(), initial_field)


def test_dynamic_drift_velocity_is_seeded_and_map_scaled():
    sim_map = SimulationMap(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0), grid_size=20)

    velocity_a = _dynamic_drift_velocity(sim_map, seed=42, speed=0.005)
    velocity_b = _dynamic_drift_velocity(sim_map, seed=42, speed=0.005)
    velocity_c = _dynamic_drift_velocity(sim_map, seed=43, speed=0.005)
    fast_velocity = _dynamic_drift_velocity(sim_map, seed=42, speed=0.02)

    np.testing.assert_allclose(velocity_a, velocity_b)
    assert not np.allclose(velocity_a, velocity_c)
    assert np.linalg.norm(velocity_a) == np.float64(0.005)
    assert np.linalg.norm(fast_velocity) == np.float64(0.02)
    np.testing.assert_allclose(
        fast_velocity / np.linalg.norm(fast_velocity),
        velocity_a / np.linalg.norm(velocity_a),
    )
