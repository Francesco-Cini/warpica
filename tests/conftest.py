import numpy as np
import pytest

@pytest.fixture
def small_grid():
    return np.array([2, 3, 3, 3])

@pytest.fixture
def unit_scaling():
    return np.ones(4)

@pytest.fixture
def world_centre(small_grid, unit_scaling):
    return (small_grid - 1) * unit_scaling / 2