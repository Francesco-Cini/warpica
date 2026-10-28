"""Run from the repository root: python -m examples.basic.minkowski"""

import numpy as np

from warplab.metrics.minkowski.metric_get_minkowski import metric_get_minkowski
from warplab.solver.get_energy_tensor import get_energy_tensor


def main():
    metric = metric_get_minkowski(np.array([1, 7, 7, 7]))
    for order in ("second", "fourth"):
        energy = get_energy_tensor(metric, order)
        print(
            f"{order}: maximum absolute stress-energy = {np.max(np.abs(energy['tensor'])):.3g}"
        )


if __name__ == "__main__":
    main()
