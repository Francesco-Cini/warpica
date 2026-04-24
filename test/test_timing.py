import numpy as np
import time

from warp_factory_py.metrics.alcubierre.metric_get_alcubierre import metric_get_alcubierre
from warp_factory_py.solver.get_energy_tensor import get_energy_tensor

grid_size = np.array([5, 20, 20, 20])
world_centre = (grid_size + 1) / 2
velocity = 0.5
R = 5
sigma = 0.5

start_time = time.time()

metric = metric_get_alcubierre(grid_size, world_centre, velocity, R, sigma)
energy_tensor = get_energy_tensor(metric, "fourth")

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.6f} s")