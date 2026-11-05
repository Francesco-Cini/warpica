# Quickstart

This page takes you from an install to an energy-condition result in five
steps. Each snippet runs as written. For the meaning of the conventions used
here (axis order, units, one- versus zero-based indices) see
[concepts](concepts.md).

## 1. Install

Python 3.11 or newer is required. `warplab` is not on PyPI yet, so install from
a checkout:

```bash
git clone https://github.com/Francesco-Cini/pywarp
cd pywarp
python -m venv .venv
.venv/bin/python -m pip install -e ".[plot]"
```

On Windows use `.venv\Scripts\python.exe`. The `plot` extra adds Matplotlib;
leave it out if you do not need figures.

## 2. Make a grid and a metric

A grid is `(time, x, y, z)` sample counts. The spacing is in seconds for time
and metres for space.

```python
import numpy as np
from warplab.metrics import metric_get_alcubierre

grid = np.array([1, 9, 9, 9])
spacing = np.array([1.0, 0.5, 0.5, 0.5])
centre = (grid + 1) / 2 * spacing  # puts the bubble in the middle

metric = metric_get_alcubierre(grid, centre, v=0.2, R=1, sigma=2, grid_scale=spacing)
```

`v` is the bubble speed as a fraction of c, `R` its radius in metres and `sigma`
the wall sharpness.

## 3. Compute the stress-energy tensor

```python
from warplab.solver import get_energy_tensor

energy = get_energy_tensor(metric, diff_order="fourth")
rho = energy["tensor"][0][0]  # shape (1, 9, 9, 9), SI units
```

`diff_order` is `"second"` or `"fourth"`. Fourth order needs at least 5 samples
along every axis that has more than one sample.

## 4. Check the energy conditions

```python
from warplab.analyser import eval_metric

result = eval_metric(metric, num_angular_vec=24, num_time_vec=4)
for name in ("null", "weak", "strong", "dominant"):
    print(name, result[name].min())
```

A negative value means the condition is violated for at least one of the sampled
directions. More directions (`num_angular_vec`, `num_time_vec`) give a finer
test and a longer run. The result also holds `expansion`, `shear`, `vorticity`,
`metric`, `energy_tensor` and `energy_tensor_eulerian`.

## 5. Plot

```python
from warplab.visualiser import plot_tensor

fig, axes = plot_tensor(metric)
fig.savefig("metric.png", dpi=150)
```

Plot functions return the figure and never open a window; call
`matplotlib.pyplot.show()` yourself if you want one.

## Where next

- [Runnable examples](../examples/README.md), including a custom metric.
- [Moving from MATLAB WarpFactory](matlab-migration.md).
- [Public API](api.md).
