import copy
import itertools

import numpy as np
import pytest

from warpica.analyser.change_tensor_index import change_tensor_index
from warpica.analyser.do_frame_transfer import do_frame_transfer
from warpica.analyser.utils.get_eulerian_transformation_matrix import (
    get_eulerian_transformation_matrix,
)
from warpica.analyser.utils.get_inner_product import get_inner_product
from warpica.analyser.utils.get_trace import get_trace
from warpica.metrics.minkowski.metric_get_minkowski import metric_get_minkowski
from warpica.metrics.three_plus_one_builder import three_plus_one_builder


def curved_metric():
    metric = metric_get_minkowski([1, 2, 2, 2])
    shape = (1, 2, 2, 2)
    gamma = np.broadcast_to(
        np.array([[2, 0.2, 0.1], [0.2, 1.5, 0.3], [0.1, 0.3, 1]])[
            :, :, None, None, None, None
        ],
        (3, 3, *shape),
    )
    beta = np.broadcast_to(
        np.array([0.2, -0.3, 0.1])[:, None, None, None, None], (3, *shape)
    )
    metric["tensor"] = three_plus_one_builder(np.full(shape, 1.3), beta, gamma)
    return metric


@pytest.mark.parametrize(
    "source,target",
    list(
        itertools.product(
            ["covariant", "contravariant", "mixedupdown", "mixeddownup"], repeat=2
        )
    ),
)
def test_tensor_index_round_trips_without_mutation(source, target):
    metric = curved_metric()
    original = np.asarray(metric["tensor"]).copy()
    tensor = copy.deepcopy(metric)
    tensor.update(type="Stress-Energy", index=source)
    tensor["tensor"] = np.random.default_rng(2).normal(size=(4, 4, 1, 2, 2, 2))
    before = tensor["tensor"].copy()
    changed = change_tensor_index(tensor, target, metric)
    restored = change_tensor_index(changed, source, metric)
    np.testing.assert_allclose(restored["tensor"], before, atol=1e-13)
    np.testing.assert_array_equal(metric["tensor"], original)
    np.testing.assert_array_equal(tensor["tensor"], before)
    assert tensor["index"] == source


def test_metric_inverse_and_trace_are_nonmutating():
    metric = curved_metric()
    original = np.asarray(metric["tensor"]).copy()
    inverse = change_tensor_index(metric, "contravariant")
    np.testing.assert_allclose(get_trace(metric, metric), 4)
    np.testing.assert_allclose(
        change_tensor_index(inverse, "covariant")["tensor"], original
    )
    np.testing.assert_array_equal(metric["tensor"], original)


def test_mixed_vector_contraction():
    metric = curved_metric()
    a = {"field": np.arange(4) + 1, "index": "covariant"}
    b = {"field": np.arange(4) + 2, "index": "contravariant"}
    assert get_inner_product(a, b, metric) == 40


def test_eulerian_basis_and_time_space_signs():
    metric = curved_metric()
    g = np.moveaxis(np.asarray(metric["tensor"]), (0, 1), (-2, -1))
    basis = get_eulerian_transformation_matrix(g)
    np.testing.assert_allclose(
        basis.swapaxes(-1, -2) @ g @ basis,
        np.broadcast_to(np.diag([-1, 1, 1, 1]), g.shape),
        atol=1e-13,
    )
    flat = metric_get_minkowski([1, 2, 2, 2])
    energy = copy.deepcopy(flat)
    energy.update(type="Stress-Energy", index="contravariant")
    energy["tensor"] = np.random.default_rng(2).normal(size=(4, 4, 1, 2, 2, 2))
    result = do_frame_transfer(flat, energy, "Eulerian")
    np.testing.assert_allclose(result["tensor"], energy["tensor"], atol=1e-14)
    np.testing.assert_allclose(
        do_frame_transfer(flat, result, "Eulerian")["tensor"], result["tensor"]
    )
