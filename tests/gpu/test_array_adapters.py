"""The CPU side of the array adapters: conversion, defaults and errors."""

import numpy as np
import pytest

from warplab.gpu import asarray, asnumpy


@pytest.mark.parametrize("library", [None, "numpy"])
def test_asarray_returns_a_numpy_array_for_the_cpu_backends(library):
    result = asarray([[1, 2], [3, 4]], library=library)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [[1, 2], [3, 4]])


def test_asarray_does_not_copy_an_existing_array():
    data = np.arange(4.0)
    assert asarray(data) is data


@pytest.mark.parametrize("library", ["jax", "torch", "CUPY", ""])
def test_asarray_rejects_an_unsupported_library(library):
    with pytest.raises(ValueError, match="Unsupported array library"):
        asarray([1.0], library=library)


def test_asnumpy_passes_host_data_through():
    data = np.arange(6.0).reshape(2, 3)
    result = asnumpy(data)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, data)
    np.testing.assert_array_equal(asnumpy([1, 2, 3]), [1, 2, 3])
