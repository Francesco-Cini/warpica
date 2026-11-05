"""Validation and analytic checks for the second-order helper functions."""

import numpy as np
import pytest

from warplab.metrics import metric_get_alcubierre
from warplab.solver.utils.second_order.c4_inv_2 import c4_inv_2
from warplab.solver.utils.second_order.c_det_2 import c_det_2
from warplab.solver.utils.second_order.met_2_den_2 import met_2_den_2

CELLS = (2, 2, 2)


def random_cells(size, seed=0):
    rng = np.random.default_rng(seed)
    identity = np.eye(size)[:, :, None, None, None]
    base = identity + 0.2 * rng.standard_normal((size, size, *CELLS))
    return [[base[i, j].copy() for j in range(size)] for i in range(size)]


def per_cell(cells, cell):
    return np.array([[row[j][cell] for j in range(len(cells))] for row in cells])


@pytest.mark.parametrize("size", [1, 2, 3, 4])
def test_determinant_matches_numpy_in_every_cell(size):
    cells = random_cells(size)
    result = c_det_2(cells)
    for cell in np.ndindex(*CELLS):
        np.testing.assert_allclose(
            result[cell], np.linalg.det(per_cell(cells, cell)), rtol=1e-12
        )


def test_a_one_by_one_determinant_is_its_only_entry():
    entry = np.arange(8.0).reshape(CELLS)
    assert c_det_2([[entry]]) is entry


@pytest.mark.parametrize(
    "cells",
    [[], [[np.ones(CELLS)] * 2], [[np.ones(CELLS)] * 2] * 3, [[np.ones(CELLS)], []]],
    ids=["empty", "one row of two", "two columns by three rows", "ragged"],
)
def test_determinant_needs_a_nonempty_square_array(cells):
    with pytest.raises(ValueError, match="nonempty and square"):
        c_det_2(cells)


def test_the_four_by_four_inverse_inverts_every_cell():
    cells = random_cells(4)
    inverse = c4_inv_2(cells)
    for cell in np.ndindex(*CELLS):
        np.testing.assert_allclose(
            per_cell(cells, cell) @ per_cell(inverse, cell), np.eye(4), atol=1e-12
        )


@pytest.mark.parametrize("size", [2, 3, 5])
def test_the_four_by_four_inverse_rejects_other_sizes(size):
    with pytest.raises(ValueError, match="not 4x4"):
        c4_inv_2(random_cells(size))


def test_second_order_stress_energy_defaults_to_unit_spacing():
    tensor = metric_get_alcubierre(
        np.array([1, 5, 5, 5]),
        np.array([1, 2.8, 2.7, 2.6]),
        0.2,
        1,
        1,
        np.ones(4),
    )["tensor"]
    np.testing.assert_array_equal(
        np.array(met_2_den_2(tensor, None)), np.array(met_2_den_2(tensor, np.ones(4)))
    )
