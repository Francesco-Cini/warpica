import numpy as np

from warplab.metrics.minkowski.metric_get_minkowski import metric_get_minkowski


def test_minkowski_signature_and_scaling():
    metric = metric_get_minkowski([1, 3, 4, 5], [0.1, 2, 3, 4])
    np.testing.assert_array_equal(metric["scaling"], [0.1, 2, 3, 4])
    tensor = np.asarray(metric["tensor"])
    expected = np.broadcast_to(
        np.diag([-1, 1, 1, 1])[:, :, None, None, None, None], tensor.shape
    )
    np.testing.assert_array_equal(tensor, expected)
