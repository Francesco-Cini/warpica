"""The warp shell interpolates its radial profiles linearly, not by Legendre.

Upstream WarpFactory uses Legendre interpolation. These tests bound how much
the difference can matter by measuring the interpolation error directly.
"""

import numpy as np
import pytest

from warplab.metrics import metric_get_warp_shell_comoving

SHELL_ARGS = [
    pytest.param([1e25, 0.5, 1.5, 0, 1, 1, 0, False], id="sharp"),
    pytest.param([1e25, 0.5, 1.5, 0.2, 2, 1, 0, False], id="buffered"),
]


@pytest.mark.parametrize("args", SHELL_ARGS)
@pytest.mark.parametrize("profile", ["A", "B"])
def test_linear_interpolation_error_is_negligible(args, profile):
    """Coarser sampling of the same profile must change the metric very little.

    Linear interpolation error scales with the square of the sample spacing.
    Using every 10th sample (10x spacing) must already agree to 1e-8, so the
    100,000-sample production grid is accurate to roughly 1e-10 or better.
    """
    metric = metric_get_warp_shell_comoving(
        np.array([1, 5, 5, 5]), np.array([1, 3, 3, 3]), *args, np.ones(4)
    )
    radii = metric["params"]["rVec"]
    values = metric["params"][profile]
    axis = (np.arange(5) + 1) - 3.0
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    grid_radius = np.sqrt(x * x + y * y + z * z)
    fine = np.interp(grid_radius, radii, values)
    coarse = np.interp(grid_radius, radii[::10], values[::10])
    error = np.max(np.abs(fine - coarse)) / np.max(np.abs(values))
    assert error < 1e-8
