import copy

import numpy as np
import pytest

from warpica.analyser.get_energy_conditions import get_energy_conditions
from warpica.analyser.utils.generate_uniform_field import generate_uniform_field
from warpica.metrics.minkowski.metric_get_minkowski import metric_get_minkowski


@pytest.mark.parametrize("condition", ["Null", "Weak", "Strong", "Dominant"])
@pytest.mark.parametrize("density", [2.0, -2.0])
def test_dust_sign_and_sample_return(condition, density):
    metric = metric_get_minkowski([1, 2, 2, 2])
    before = metric["tensor"].copy()
    energy = copy.deepcopy(metric)
    energy.update(type="Stress-Energy", index="contravariant")
    energy["tensor"][:] = 0
    energy["tensor"][0, 0] = density
    values, samples, vectors = get_energy_conditions(
        energy, metric, condition, 12, 3, True
    )
    assert np.isfinite(values).all()
    assert np.all(values * density > 0)
    assert samples.shape == (1, 2, 2, 2, 12) + (
        () if condition in ("Null", "Dominant") else (3,)
    )
    np.testing.assert_allclose(values, samples.min(axis=tuple(range(4, samples.ndim))))
    np.testing.assert_array_equal(metric["tensor"], before)
    assert energy["index"] == "contravariant"


def test_timelike_samples_are_strictly_timelike():
    field = generate_uniform_field("timelike", 12, 4)
    norm = -(field[0] ** 2) + np.sum(field[1:] ** 2, axis=0)
    assert np.all(norm < 0)


def test_strong_condition_perfect_fluid_at_rest():
    metric = metric_get_minkowski([1, 2, 2, 2])
    energy = copy.deepcopy(metric)
    energy.update(type="Stress-Energy", index="contravariant")
    energy["tensor"][:] = 0
    energy["tensor"][0, 0] = 2
    for i in range(1, 4):
        energy["tensor"][i, i] = 3
    result = get_energy_conditions(energy, metric, "Strong", 8, 1, False)
    np.testing.assert_allclose(result, 0.5 * (2 + 3 * 3))
