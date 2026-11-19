import numpy as np
import pytest

from warpica.ui.core.registry import METRICS


@pytest.mark.parametrize("spec", METRICS.values(), ids=list(METRICS))
def test_every_default_lies_inside_its_range(spec):
    for parameter in spec.parameters:
        assert parameter.minimum <= parameter.default <= parameter.maximum


def test_defaults_are_keyed_by_parameter_name():
    spec = METRICS["alcubierre"]

    assert spec.defaults() == {"v": 0.4, "R": 2.0, "sigma": 1.0}


def test_alcubierre_builder_returns_a_metric():
    spec = METRICS["alcubierre"]
    grid_size = np.array([1, 5, 5, 5])
    grid_scale = np.ones(4)

    metric = spec.build(grid_size, (grid_size + 1) / 2, grid_scale, **spec.defaults())

    assert metric["name"] == "Alcubierre"
    assert metric["params"]["velocity"] == 0.4


@pytest.mark.parametrize("spec", METRICS.values(), ids=list(METRICS))
def test_every_metric_builds_with_its_default_values(spec):
    grid_size = np.array([1, 5, 5, 5])
    grid_scale = np.array([1.0, 0.5, 0.5, 0.5])

    metric = spec.build(
        grid_size, (grid_size + 1) / 2 * grid_scale, grid_scale, **spec.defaults()
    )

    assert metric["type"] == "metric"
    assert np.shape(metric["tensor"][0][0]) == (1, 5, 5, 5)


def test_flat_space_has_no_parameters():
    assert METRICS["minkowski"].parameters == ()
