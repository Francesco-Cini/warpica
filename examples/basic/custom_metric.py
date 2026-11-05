"""Build Minkowski space from user-supplied lapse, shift and spatial metric."""

import numpy as np

from warplab.metrics import custom_metric
from warplab.solver import get_energy_tensor


def main():
    grid = np.array([1, 5, 5, 5])
    metric = custom_metric(
        grid,
        (grid + 1) / 2,
        np.ones(4),
        alpha_function=lambda t, x, y, z: 1.0,
        beta_function=lambda t, x, y, z: np.zeros(3),
        gamma_function=lambda t, x, y, z: np.eye(3),
    )
    energy = get_energy_tensor(metric, "second")
    print(
        f"Custom flat metric: maximum absolute stress-energy = {np.max(np.abs(energy['tensor'])):.3g}"
    )


if __name__ == "__main__":
    main()
