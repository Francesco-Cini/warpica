"""Index algebra checked against independent einsum contractions and identities."""

import copy
import itertools

import numpy as np
import pytest

from warpica.analyser import change_tensor_index
from warpica.analyser.utils.get_eulerian_transformation_matrix import (
    get_eulerian_transformation_matrix,
)
from warpica.analyser.utils.get_inner_product import get_inner_product
from warpica.analyser.utils.get_trace import get_trace

ETA = np.diag([-1.0, 1, 1, 1])
SHAPE = (1, 2, 2, 2)
CONVENTIONS = ["covariant", "contravariant", "mixedupdown", "mixeddownup"]


def random_lorentzian(shape=SHAPE, seed=0):
    """Return a random metric g_ab(x) of shape (4, 4, *shape) built from ADM pieces.

    Lapse alpha, shift beta^i and a positive-definite spatial metric gamma_ij
    give g_00 = -alpha^2 + beta_i beta^i, g_0i = beta_i, g_ij = gamma_ij. Unlike a
    random L^T eta L, this always has spacelike t = const surfaces, so Eulerian
    observers exist at every point.
    """
    rng = np.random.default_rng(seed)
    g = np.empty((4, 4, *shape))
    for cell in np.ndindex(*shape):
        a = 0.15 * rng.standard_normal((3, 3))
        gamma = np.eye(3) + a @ a.T
        beta_up = 0.3 * rng.standard_normal(3)
        beta_down = gamma @ beta_up
        alpha = 0.7 + 0.3 * rng.random()
        block = np.zeros((4, 4))
        block[0, 0] = -(alpha**2) + beta_up @ beta_down
        block[0, 1:] = block[1:, 0] = beta_down
        block[1:, 1:] = gamma
        g[(slice(None), slice(None), *cell)] = block
    return g


def as_cells(array):
    return [[np.array(array[i, j]) for j in range(4)] for i in range(4)]


def tensor_dict(array, index, kind="Stress-Energy"):
    return {
        "type": kind,
        "coords": "cartesian",
        "name": "test",
        "scaling": np.ones(4),
        "index": index,
        "tensor": as_cells(array),
    }


def as_array(tensor):
    return np.array(tensor["tensor"])


@pytest.fixture(scope="module")
def representations():
    """One physical tensor written in all four index conventions."""
    g = random_lorentzian()
    g_inv = np.linalg.inv(np.moveaxis(g, (0, 1), (-2, -1)))
    g_inv = np.moveaxis(g_inv, (-2, -1), (0, 1))
    upper = np.random.default_rng(1).standard_normal((4, 4, *SHAPE))
    return {
        "g": g,
        "g_inv": g_inv,
        "contravariant": upper,
        "covariant": np.einsum("ac...,bd...,cd...->ab...", g, g, upper),
        "mixedupdown": np.einsum("ac...,cb...->ab...", upper, g),
        "mixeddownup": np.einsum("ac...,cb...->ab...", g, upper),
    }


@pytest.mark.parametrize("metric_index", ["covariant", "contravariant"])
@pytest.mark.parametrize(
    ("source", "target"), list(itertools.product(CONVENTIONS, repeat=2))
)
def test_raising_and_lowering_matches_einsum(
    representations, metric_index, source, target
):
    metric = tensor_dict(representations["g"], "covariant", "Metric")
    if metric_index == "contravariant":
        metric = tensor_dict(representations["g_inv"], "contravariant", "Metric")
    tensor = tensor_dict(representations[source], source)
    result = change_tensor_index(tensor, target, metric)
    assert result["index"] == target
    np.testing.assert_allclose(
        as_array(result), representations[target], rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize(("source", "target"), [("covariant", "contravariant")])
def test_metric_inverse_matches_numpy(representations, source, target):
    metric = tensor_dict(representations["g"], source, "Metric")
    result = change_tensor_index(metric, target)
    np.testing.assert_allclose(
        as_array(result), representations["g_inv"], rtol=1e-12, atol=1e-12
    )
    back = change_tensor_index(result, source)
    np.testing.assert_allclose(
        as_array(back), representations["g"], rtol=1e-12, atol=1e-12
    )


def test_convention_names_are_case_insensitive(representations):
    tensor = tensor_dict(representations["contravariant"], "CONTRAVARIANT")
    metric = tensor_dict(representations["g"], "Covariant", "METRIC")
    result = change_tensor_index(tensor, "CoVaRiAnT", metric)
    assert result["index"] == "covariant"
    np.testing.assert_allclose(
        as_array(result), representations["covariant"], rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("target", CONVENTIONS)
def test_conversion_returns_independent_arrays_and_keeps_metadata(
    representations, target
):
    metric = tensor_dict(representations["g"], "covariant", "Metric")
    tensor = tensor_dict(representations["contravariant"], "contravariant")
    tensor["frame"] = "Eulerian"
    before = copy.deepcopy(tensor)
    result = change_tensor_index(tensor, target, metric)
    for key in ("type", "coords", "name", "frame"):
        assert result[key] == tensor[key]
    np.testing.assert_array_equal(as_array(tensor), as_array(before))
    assert tensor["index"] == "contravariant"
    result["tensor"][0][0][...] = 123.0
    np.testing.assert_array_equal(as_array(tensor), as_array(before))


def test_same_index_returns_a_copy_not_the_input(representations):
    metric = tensor_dict(representations["g"], "covariant", "Metric")
    result = change_tensor_index(metric, "covariant")
    assert result["tensor"][0][0] is not metric["tensor"][0][0]
    np.testing.assert_array_equal(as_array(result), as_array(metric))


def test_array_form_tensors_are_accepted(representations):
    metric = tensor_dict(representations["g"], "covariant", "Metric")
    metric["tensor"] = representations["g"]
    result = change_tensor_index(metric, "contravariant")
    np.testing.assert_allclose(
        as_array(result), representations["g_inv"], rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize(
    ("target", "source"),
    [("sideways", "covariant"), ("covariant", "sideways")],
)
def test_unknown_conventions_are_rejected(representations, target, source):
    tensor = tensor_dict(representations["covariant"], source, "Metric")
    with pytest.raises(ValueError, match="Unknown tensor index"):
        change_tensor_index(tensor, target)


@pytest.mark.parametrize(
    ("source", "target"), [("covariant", "mixedupdown"), ("mixeddownup", "covariant")]
)
def test_metrics_cannot_be_mixed(representations, source, target):
    metric = tensor_dict(representations["g"], source, "Metric")
    with pytest.raises(ValueError, match="cannot use mixed"):
        change_tensor_index(metric, target)


def test_non_metric_tensors_need_a_reference_metric(representations):
    tensor = tensor_dict(representations["contravariant"], "contravariant")
    with pytest.raises(ValueError, match="metric_tensor is required"):
        change_tensor_index(tensor, "covariant")


def test_the_reference_metric_cannot_be_mixed(representations):
    tensor = tensor_dict(representations["contravariant"], "contravariant")
    metric = tensor_dict(representations["g"], "mixedupdown", "Metric")
    with pytest.raises(ValueError, match="cannot use mixed"):
        change_tensor_index(tensor, "covariant", metric)


# ---- traces and inner products ------------------------------------------------


@pytest.mark.parametrize("metric_index", ["covariant", "contravariant"])
@pytest.mark.parametrize("index", CONVENTIONS)
def test_trace_is_the_full_contraction(representations, metric_index, index):
    g, g_inv = representations["g"], representations["g_inv"]
    metric = tensor_dict(
        g if metric_index == "covariant" else g_inv, metric_index, "Metric"
    )
    tensor = tensor_dict(representations[index], index)
    expected = np.einsum("aa...->...", representations["mixedupdown"])
    np.testing.assert_allclose(
        get_trace(tensor, metric), expected, rtol=1e-12, atol=1e-12
    )


def test_trace_does_not_modify_its_inputs(representations):
    metric = tensor_dict(representations["g"], "covariant", "Metric")
    tensor = tensor_dict(representations["contravariant"], "contravariant")
    before = (copy.deepcopy(metric), copy.deepcopy(tensor))
    get_trace(tensor, metric)
    np.testing.assert_array_equal(as_array(metric), as_array(before[0]))
    np.testing.assert_array_equal(as_array(tensor), as_array(before[1]))


def vector(field, index):
    return {"field": [np.array(field[i]) for i in range(4)], "index": index}


@pytest.fixture(scope="module")
def vectors(representations):
    rng = np.random.default_rng(2)
    g = representations["g"]
    up_a, up_b = rng.standard_normal((2, 4, *SHAPE))
    return {
        "up": (up_a, up_b),
        "down": (
            np.einsum("ab...,b...->a...", g, up_a),
            np.einsum("ab...,b...->a...", g, up_b),
        ),
        "scalar": np.einsum("ab...,a...,b...->...", g, up_a, up_b),
    }


@pytest.mark.parametrize("metric_index", ["covariant", "contravariant"])
@pytest.mark.parametrize(
    ("a_index", "b_index"), list(itertools.product(["up", "down"], repeat=2))
)
def test_inner_product_is_the_invariant_pairing(
    representations, vectors, metric_index, a_index, b_index
):
    """Whatever the index positions, the pairing of two vectors is g_ab v^a w^b."""
    g, g_inv = representations["g"], representations["g_inv"]
    metric = tensor_dict(
        g if metric_index == "covariant" else g_inv, metric_index, "Metric"
    )
    names = {"up": "contravariant", "down": "covariant"}
    first = vector(vectors[a_index][0], names[a_index])
    second = vector(vectors[b_index][1], names[b_index])
    np.testing.assert_allclose(
        get_inner_product(first, second, metric),
        vectors["scalar"],
        rtol=1e-12,
        atol=1e-12,
    )


# ---- Eulerian transformation matrix ---------------------------------------------


def stacked(g):
    """Move the (a, b) axes of a (4, 4, *grid) array to the end, as the matrix wants."""
    return np.moveaxis(g, (0, 1), (-2, -1))


@pytest.mark.parametrize("seed", range(5))
def test_eulerian_matrix_maps_the_metric_to_minkowski(seed):
    g = stacked(random_lorentzian(shape=(1,), seed=seed))[0]
    matrix = get_eulerian_transformation_matrix(g, "cartesian")
    np.testing.assert_allclose(matrix.T @ g @ matrix, ETA, atol=1e-12)
    np.testing.assert_allclose(matrix @ ETA @ matrix.T, np.linalg.inv(g), atol=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_eulerian_matrix_first_column_is_the_eulerian_observer(seed):
    """The timelike leg must be u^mu = -alpha g^{mu 0}, with alpha^2 = -1/g^{00}."""
    g = stacked(random_lorentzian(shape=(1,), seed=seed))[0]
    inverse = np.linalg.inv(g)
    alpha = 1 / np.sqrt(-inverse[0, 0])
    matrix = get_eulerian_transformation_matrix(g, "cartesian")
    np.testing.assert_allclose(matrix[:, 0], -alpha * inverse[:, 0], atol=1e-12)


def test_eulerian_matrix_of_flat_space_is_the_identity():
    np.testing.assert_allclose(
        get_eulerian_transformation_matrix(ETA, "cartesian"), np.eye(4), atol=1e-14
    )


def test_gridded_matrix_agrees_with_the_single_point_matrix():
    grid = stacked(random_lorentzian(shape=(2, 2, 2, 2), seed=7))
    matrix = get_eulerian_transformation_matrix(grid, "cartesian")
    assert matrix.shape == grid.shape
    for cell in np.ndindex(2, 2, 2, 2):
        np.testing.assert_allclose(
            matrix[cell],
            get_eulerian_transformation_matrix(grid[cell], "cartesian"),
            atol=1e-14,
        )


def test_the_matrix_does_not_modify_the_metric():
    g = stacked(random_lorentzian(shape=(1,), seed=3))[0]
    before = g.copy()
    get_eulerian_transformation_matrix(g, "cartesian")
    np.testing.assert_array_equal(g, before)


def test_a_non_symmetric_metric_is_rejected():
    with pytest.raises(ValueError, match="not symmetric"):
        get_eulerian_transformation_matrix(np.triu(np.ones((4, 4))), "cartesian")


def test_a_singular_metric_has_no_eulerian_matrix():
    with pytest.raises(ValueError, match="not finite"):
        get_eulerian_transformation_matrix(np.diag([-1.0, 1, 1, 0]), "cartesian")


def test_a_metric_with_timelike_slices_has_no_eulerian_matrix():
    """Where t = const is not spacelike no observer is at rest: it used to be
    reported as "imaginary" (a MATLAB result) although NumPy returns NaN."""
    with pytest.raises(ValueError, match="not finite"):
        get_eulerian_transformation_matrix(np.diag([1.0, 1, 1, 1]), "cartesian")


@pytest.mark.parametrize("shape", [(4,), (4, 4, 4), (3, 3), (2, 2, 2, 2, 3, 3)])
def test_unrecognised_shapes_are_rejected(shape):
    with pytest.raises(ValueError, match="Unrecognised matrix size"):
        get_eulerian_transformation_matrix(np.zeros(shape), "cartesian")
