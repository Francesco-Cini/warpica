import numpy as np
import pytest

from warplab.metrics.minkowski.metric_get_minkowski import metric_get_minkowski


@pytest.fixture
def small_grid_size() -> np.ndarray:
    """A grid large enough for metric tests without making them expensive."""
    return np.array([5, 7, 7, 7])


@pytest.fixture
def unit_grid_scaling() -> np.ndarray:
    return np.ones(4)


@pytest.fixture
def small_world_centre(small_grid_size: np.ndarray) -> np.ndarray:
    return (small_grid_size + 1) / 2


@pytest.fixture
def numerical_tolerances() -> dict[str, float]:
    return {
        "atol": 1e-12,
        "rtol": 1e-10,
    }


@pytest.fixture
def minkowski_metric(
    small_grid_size: np.ndarray, unit_grid_scaling: np.ndarray
) -> dict:
    return metric_get_minkowski(small_grid_size, unit_grid_scaling)


def pytest_addoption(parser):
    parser.addoption(
        "--gpu-backend",
        default=None,
        choices=["cupy", "torch-directml"],
        help="Opt into real hardware tests; a missing requested backend is a failure",
    )
    parser.addoption(
        "--run-performance",
        action="store_true",
        default=False,
        help="Run the advisory performance guard in tests/performance",
    )


@pytest.fixture(scope="session")
def performance_enabled(request):
    if not request.config.getoption("--run-performance"):
        pytest.skip("Pass --run-performance to run the advisory performance guard")


@pytest.fixture(scope="session")
def gpu_backend(request):
    backend = request.config.getoption("--gpu-backend")
    if backend is None:
        pytest.skip(
            "Select --gpu-backend=cupy or --gpu-backend=torch-directml to test hardware"
        )
    from warplab.gpu import asarray, asnumpy

    try:
        value = asarray(np.array([1.0, 2.0]), library=backend)
        np.testing.assert_allclose(asnumpy(value * value), [1, 4])
    except Exception as exc:
        pytest.fail(f"Requested GPU backend {backend} is unavailable: {exc}")
    return backend
