"""Validation, defaults and analytic checks for the solver building blocks."""

import copy

import numpy as np
import pytest

from warplab.analyser import change_tensor_index
from warplab.metrics import metric_get_alcubierre, metric_get_minkowski
from warplab.solver import get_energy_tensor
from warplab.solver.utils.c3_inv import c3_inv
from warplab.solver.utils.c4_inv import c4_inv
from warplab.solver.utils.cov_div import cov_div
from warplab.solver.utils.met_2_den import met_2_den
from warplab.solver.utils.trilinear_interp import trilinear_interp
from warplab.units import c


def bubble():
    return metric_get_alcubierre(
        np.array([1, 5, 5, 5]),
        np.array([1, 2.8, 2.7, 2.6]),
        0.2,
        1,
        1,
        np.ones(4),
    )


def as_array(tensor):
    return np.array(tensor["tensor"])


@pytest.fixture(scope="module")
def reference():
    metric = bubble()
    return metric, get_energy_tensor(metric, "fourth")


# ---- get_energy_tensor ------------------------------------------------------


@pytest.mark.parametrize("gpu", [None, "numpy"])
def test_an_unknown_diff_order_is_rejected_on_both_code_paths(gpu):
    with pytest.raises(ValueError, match="fourth"):
        get_energy_tensor(bubble(), "sixth", gpu=gpu)


@pytest.mark.parametrize("order", ["Fourth", "FOURTH", "Second", "SECOND"])
def test_diff_order_is_case_insensitive(reference, order):
    metric, fourth = reference
    result = get_energy_tensor(metric, order)
    expected = (
        fourth if order.lower() == "fourth" else get_energy_tensor(metric, "second")
    )
    np.testing.assert_array_equal(as_array(result), as_array(expected))


def test_diff_order_defaults_to_fourth(reference):
    metric, fourth = reference
    np.testing.assert_array_equal(
        as_array(get_energy_tensor(metric, None)), as_array(fourth)
    )
    np.testing.assert_array_equal(as_array(get_energy_tensor(metric)), as_array(fourth))
    assert fourth["order"] == "fourth"


@pytest.mark.parametrize("order", ["fourth", "second"])
def test_the_numpy_device_branch_matches_the_direct_path(reference, order):
    """gpu='numpy' runs the device code path on the CPU, so it can be checked here."""
    metric, _ = reference
    direct = get_energy_tensor(metric, order)
    through_branch = get_energy_tensor(metric, order, gpu="numpy")
    np.testing.assert_allclose(
        as_array(through_branch), as_array(direct), rtol=1e-12, atol=0
    )
    assert through_branch["order"] == order


def test_a_contravariant_metric_gives_the_same_result(reference):
    metric, fourth = reference
    contravariant = change_tensor_index(metric, "contravariant")
    result = get_energy_tensor(contravariant)
    np.testing.assert_allclose(
        as_array(result),
        as_array(fourth),
        rtol=1e-9,
        atol=1e-9 * np.abs(as_array(fourth)).max(),
    )


def test_an_unverified_metric_is_rejected():
    with pytest.warns(UserWarning, match="type does not exist"):
        with pytest.raises(ValueError, match="not verified"):
            get_energy_tensor({})


def test_the_metric_is_not_modified_and_metadata_is_reported(reference):
    metric, energy = reference
    before = copy.deepcopy(metric)
    get_energy_tensor(metric, "second")
    np.testing.assert_array_equal(as_array(metric), as_array(before))
    assert metric["index"] == "covariant"
    assert energy["type"] == "Stress-Energy"
    assert energy["index"] == "contravariant"
    assert energy["coords"] == metric["coords"]
    assert energy["name"] == metric["name"]


def test_flat_space_has_no_stress_energy():
    energy = get_energy_tensor(metric_get_minkowski([1, 5, 5, 5]))
    assert not np.any(as_array(energy))


def test_met_2_den_defaults_to_unit_spacing():
    tensor = bubble()["tensor"]
    np.testing.assert_array_equal(
        np.array(met_2_den(tensor, None)), np.array(met_2_den(tensor, np.ones(4)))
    )


# ---- inverses ---------------------------------------------------------------


def random_cells(size, seed=0):
    rng = np.random.default_rng(seed)
    identity = np.eye(size)[:, :, None, None, None]
    base = identity + 0.2 * rng.standard_normal((size, size, 2, 2, 2))
    return [[base[i, j].copy() for j in range(size)] for i in range(size)]


@pytest.mark.parametrize(("size", "inverse"), [(3, c3_inv), (4, c4_inv)])
def test_inverse_cell_arrays_invert_every_cell(size, inverse):
    cells = random_cells(size)
    result = inverse(cells)
    for cell in np.ndindex(2, 2, 2):
        forward = np.array(
            [[cells[i][j][cell] for j in range(size)] for i in range(size)]
        )
        backward = np.array(
            [[result[i][j][cell] for j in range(size)] for i in range(size)]
        )
        np.testing.assert_allclose(forward @ backward, np.eye(size), atol=1e-12)


@pytest.mark.parametrize(
    ("inverse", "message"), [(c3_inv, "not 3x3"), (c4_inv, "not 4x4")]
)
@pytest.mark.parametrize("wrong", [2, 5])
def test_inverse_cell_arrays_reject_the_wrong_size(inverse, message, wrong):
    with pytest.raises(ValueError, match=message):
        inverse(random_cells(wrong))


@pytest.mark.parametrize(("size", "inverse"), [(3, c3_inv), (4, c4_inv)])
def test_inverse_cell_arrays_reject_ragged_rows(size, inverse):
    cells = random_cells(size)
    cells[-1] = cells[-1][:-1]
    with pytest.raises(ValueError):
        inverse(cells)


# ---- covariant divergence ---------------------------------------------------

HUBBLE = 0.2
TIME_STEP = 0.1


def expanding_universe(samples=9):
    """FLRW with scale factor a(t) = 1 + H t, in the (t, x, y, z) cell layout."""
    t = np.arange(samples).reshape(samples, 1, 1, 1) * TIME_STEP
    scale = (1 + HUBBLE * t) ** 2
    zero = np.zeros_like(scale)
    metric = [[zero.copy() for _ in range(4)] for _ in range(4)]
    metric[0][0] = -np.ones_like(scale)
    for i in range(1, 4):
        metric[i][i] = scale.copy()
    return metric, t, np.array([TIME_STEP, 1, 1, 1])


def observer_divergence(order, branch):
    """Contract the covariant derivative of the comoving observer with g^{mu nu}."""
    metric, t, delta = expanding_universe()
    inverse = c4_inv(metric)
    ones, zero = np.ones_like(t), np.zeros_like(t)
    up = [ones, zero, zero, zero]
    down = [-ones, zero, zero, zero]
    total = 0
    for mu in range(4):
        if branch == "contravariant":
            total = total + cov_div(metric, inverse, up, down, mu, mu, delta, 1, order)
        else:
            for nu in range(4):
                total = total + inverse[mu][nu] * cov_div(
                    metric, inverse, up, down, mu, nu, delta, 0, order
                )
    return total, t


@pytest.mark.parametrize("branch", ["contravariant", "covariant"])
@pytest.mark.parametrize(("order", "trim"), [("second", 1), ("fourth", 2)])
def test_the_divergence_of_the_comoving_observer_is_three_hubble(order, trim, branch):
    divergence, t = observer_divergence(order, branch)
    expected = 3 * HUBBLE / (1 + HUBBLE * t) / c()
    np.testing.assert_allclose(divergence[trim:-trim], expected[trim:-trim], rtol=1e-11)


def test_the_variance_selector_must_be_zero_or_one():
    metric, t, delta = expanding_universe()
    vector = [np.ones_like(t)] * 4
    with pytest.raises(ValueError, match="Invalid variance"):
        cov_div(metric, c4_inv(metric), vector, vector, 0, 0, delta, 2)


def test_cov_div_rejects_an_unknown_order_and_ignores_case():
    metric, t, delta = expanding_universe()
    inverse = c4_inv(metric)
    vector = [np.ones_like(t), *[np.zeros_like(t)] * 3]
    with pytest.raises(ValueError, match="second or fourth"):
        cov_div(metric, inverse, vector, vector, 0, 0, delta, 1, "sixth")
    for spelling in ("Fourth", "FOURTH"):
        np.testing.assert_array_equal(
            cov_div(metric, inverse, vector, vector, 0, 0, delta, 1, spelling),
            cov_div(metric, inverse, vector, vector, 0, 0, delta, 1, "fourth"),
        )


# ---- trilinear interpolation ------------------------------------------------


@pytest.mark.parametrize(
    ("field", "point"),
    [
        (np.zeros((4, 4)), [2, 2, 2]),
        (np.zeros((4, 4, 4, 4)), [2, 2, 2]),
        (np.zeros((4, 4, 4)), [2, 2]),
        (np.zeros((4, 4, 4)), [[2, 2, 2]]),
    ],
    ids=["2D field", "4D field", "two coordinates", "nested point"],
)
def test_interpolation_needs_a_3d_field_and_three_coordinates(field, point):
    with pytest.raises(ValueError, match="Expected a 3D field"):
        trilinear_interp(field, point)


@pytest.mark.parametrize(
    "point",
    [[0.5, 2, 2], [2, 4.5, 2], [2, 2, 0], [np.nan, 2, 2], [np.inf, 2, 2]],
)
def test_interpolation_rejects_points_outside_the_field(point):
    with pytest.raises(ValueError, match="outside the field"):
        trilinear_interp(np.zeros((4, 4, 4)), point)


@pytest.mark.parametrize("point", [[1, 1, 1], [4, 4, 4], [1, 4, 2.5]])
def test_interpolation_accepts_the_grid_boundary(point):
    field = np.arange(64.0).reshape(4, 4, 4)
    index = tuple(int(round(p)) - 1 for p in point)
    if all(float(p).is_integer() for p in point):
        assert trilinear_interp(field, point) == field[index]
    else:
        assert np.isfinite(trilinear_interp(field, point))
