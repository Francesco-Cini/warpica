import numpy as np
import pytest

from warpica.analyser import do_frame_transfer, get_energy_conditions
from warpica.metrics import metric_get_alcubierre
from warpica.solver import get_energy_tensor
from warpica.ui.core.analysis import (
    CONDITIONS,
    SCALARS,
    condition_key,
    energy_condition_map,
    eulerian_energy_density,
    scalar_fields,
    scalar_key,
    summarise_condition,
)

SHAPE = (1, 7, 7, 7)


def solved(velocity):
    """Return a small Alcubierre metric and its solved energy tensor."""
    grid = np.array(SHAPE)
    scale = np.array([1.0, 0.5, 0.5, 0.5])
    metric = metric_get_alcubierre(
        grid, (grid + 1) / 2 * scale, velocity, 1.0, 1.0, scale
    )
    return metric, get_energy_tensor(metric, "fourth")


@pytest.fixture(scope="module")
def moving():
    return solved(0.4)


def test_the_result_keys_name_what_they_hold():
    assert condition_key("Weak") == "Weak energy condition"
    assert scalar_key("Shear") == "Shear scalar"


@pytest.mark.parametrize("condition", CONDITIONS)
def test_an_energy_condition_map_has_one_value_per_cell(moving, condition):
    metric, tensor = moving

    result = energy_condition_map(tensor, metric, condition, 20, 5)

    assert result.shape == SHAPE
    assert np.isfinite(result).all()


def test_the_energy_condition_map_matches_the_analyser(moving):
    metric, tensor = moving

    result = energy_condition_map(tensor, metric, "Weak", 20, 5)
    expected = get_energy_conditions(tensor, metric, "Weak", 20, 5)

    np.testing.assert_allclose(result, expected)


def test_an_unknown_condition_is_rejected(moving):
    metric, tensor = moving

    with pytest.raises(ValueError, match="energy condition"):
        energy_condition_map(tensor, metric, "Imaginary", 20, 5)


def test_the_scalars_are_the_three_kinematic_scalars(moving):
    metric, _ = moving

    result = scalar_fields(metric, "fourth")

    assert tuple(result) == SCALARS
    for values in result.values():
        assert values.shape == SHAPE


def test_the_energy_density_is_the_eulerian_time_time_component(moving):
    metric, tensor = moving

    result = eulerian_energy_density(tensor, metric)
    expected = do_frame_transfer(metric, tensor, "Eulerian")["tensor"][0][0]

    assert result.shape == SHAPE
    np.testing.assert_allclose(result, expected)


def test_a_bubble_that_does_not_move_needs_no_energy():
    metric, tensor = solved(0.0)

    result = eulerian_energy_density(tensor, metric)

    assert np.all(np.abs(result) < 1e-6)


def test_a_moving_bubble_needs_energy(moving):
    metric, tensor = moving

    result = eulerian_energy_density(tensor, metric)

    assert np.nanmax(np.abs(result)) > 1e10


def test_the_summary_counts_violated_cells_and_finds_the_worst_value():
    values = np.array([[-2.0, 3.0], [-1.0, 0.0]])

    summary = summarise_condition(values)

    assert summary.worst == -2.0
    assert summary.violated == 2
    assert summary.total == 4


def test_cells_without_a_value_are_not_counted_as_violated():
    values = np.array([-1.0, np.nan, 2.0])

    summary = summarise_condition(values)

    assert summary.worst == -1.0
    assert summary.violated == 1
    assert summary.total == 3


def test_a_map_with_no_finite_value_has_no_worst_value():
    summary = summarise_condition(np.full((2, 2), np.nan))

    assert np.isnan(summary.worst)
    assert summary.violated == 0
    assert summary.total == 4
