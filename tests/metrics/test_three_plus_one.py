import numpy as np

from warpica.metrics.minkowski.metric_get_minkowski import metric_get_minkowski
from warpica.metrics.three_plus_one_builder import three_plus_one_builder
from warpica.metrics.three_plus_one_decomposer import three_plus_one_decomposer


def test_nondiagonal_spatial_metric_round_trip():
    shape = (1, 2, 3, 4)
    alpha = np.full(shape, 1.3)
    beta = np.broadcast_to(
        np.array([0.2, -0.1, 0.3])[:, None, None, None, None], (3, *shape)
    ).copy()
    matrix = np.array([[2, 0.2, 0.1], [0.2, 3, 0.4], [0.1, 0.4, 1]])
    gamma = np.broadcast_to(matrix[:, :, None, None, None, None], (3, 3, *shape)).copy()
    metric = metric_get_minkowski(shape)
    metric["tensor"] = three_plus_one_builder(alpha, beta, gamma)
    a, b, g, bu, gu = three_plus_one_decomposer(metric)
    np.testing.assert_allclose(a, alpha)
    np.testing.assert_allclose(b, beta)
    np.testing.assert_allclose(g, gamma)
    np.testing.assert_allclose(np.asarray(gu)[:, :, 0, 0, 0, 0], np.linalg.inv(matrix))
    np.testing.assert_allclose(
        np.asarray(bu)[:, 0, 0, 0, 0], np.linalg.solve(matrix, [0.2, -0.1, 0.3])
    )
