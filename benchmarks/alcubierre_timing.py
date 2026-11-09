import time

import numpy as np

from warpica.metrics.alcubierre.metric_get_alcubierre import metric_get_alcubierre
from warpica.solver.get_energy_tensor import get_energy_tensor


def main():
    grid_size = np.array([1, 9, 9, 9])
    world_centre = (grid_size + 1) / 2
    velocity = 0.5
    R = 2
    sigma = 0.5

    start_time = time.time()

    metric = metric_get_alcubierre(grid_size, world_centre, velocity, R, sigma)
    get_energy_tensor(metric, "fourth")

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.6f} s")


if __name__ == "__main__":
    main()
