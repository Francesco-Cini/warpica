# Public API and conventions

## What is public

The supported API is the set of names in `__all__` of these packages. They
are what the version number promises to keep stable.

| Package | Public names |
| --- | --- |
| `warplab.metrics` | `metric_get_minkowski`, `metric_get_alcubierre`, `metric_get_lentz`, `metric_get_modified_time`, `metric_get_van_den_broeck`, `metric_get_schwarzschild`, the `*_comoving` variants, `metric_get_warp_shell_comoving`, `custom_metric`, `three_plus_one_builder`, `three_plus_one_decomposer` |
| `warplab.solver` | `get_energy_tensor` |
| `warplab.analyser` | `eval_metric`, `get_energy_conditions`, `get_scalars`, `get_momentum_flow_lines`, `change_tensor_index`, `do_frame_transfer` |
| `warplab.units` | `c`, `G` |
| `warplab.visualiser` | `plot_tensor`, `plot_three_plus_one` (needs the `plot` extra) |

```python
from warplab.metrics import metric_get_alcubierre
from warplab.solver import get_energy_tensor
from warplab.analyser import eval_metric
```

The metrics, solver and analyser names are also available from the top-level
`warplab` package, which loads them on first use. `import warplab` itself
imports neither Matplotlib nor any GPU library.

Everything else is internal and may change without notice: the helper
modules under every `utils` directory, `warplab.solver.utils.second_order`,
`warplab.gpu`, and the `Metric` dataclass in `warplab.metrics.metrics`. Deep
paths such as `warplab.solver.get_energy_tensor` keep working, but only the
names listed above are covered by the compatibility promise.

## Metrics and tensors

A metric is a dictionary containing `type`, `name`, `coords`, `index`, `date`,
`scaling`, and `tensor`. Components use `tensor[mu][nu]`, each shaped
`(time, x, y, z)`. A `(4,4,t,x,y,z)` array is also accepted. Grid spacing is
in seconds and metres; velocities are fractions of c. Python tensor indices
are zero-based. Generator positions retain WarpFactory's `(index+1)*spacing
- world_centre` convention.

`metric_get_minkowski(grid_size, grid_scaling=None)` creates flat space.
`metric_get_alcubierre(grid_size, world_centre, v, R, sigma, grid_scale=None)`
creates a moving bubble. Comoving generators require one time slice.
`custom_metric(grid_size, world_centre, grid_scaling, alpha_function,
beta_function, gamma_function)` accepts callables of `(t,x,y,z)` returning
a scalar lapse, three covariant shift components, and a 3x3 spatial metric.
See the examples and metric modules for family-specific parameters.

## Solve and analyse

```python
energy = get_energy_tensor(metric, diff_order="fourth", gpu=None)
result = eval_metric(
    metric,
    keep_positive=True,
    num_angular_vec=100,
    num_time_vec=10,
    gpu=None,
    diff_order="fourth",
)
```

Import from `warplab.solver` and
`warplab.analyser`. Both second and fourth orders are supported.
The solver returns contravariant stress-energy in SI units, with the same
spacetime grid. `eval_metric` returns `metric`, `energy_tensor`,
`energy_tensor_eulerian`, `null`, `weak`, `strong`, `dominant`, `expansion`,
`shear`, and `vorticity`. `keep_positive=False` clips positive condition-map
entries to zero. Energy conditions use finite direction sampling.

`change_tensor_index(tensor, index, metric_tensor=None)` returns a new tensor
in `covariant`, `contravariant`, `mixedupdown`, or `mixeddownup` form.
`do_frame_transfer(metric, energy, "Eulerian")` returns orthonormal-frame
components. `get_scalars(metric, diff_order="fourth")` returns expansion,
shear and vorticity. These operations preserve their input arrays.

The `gpu` option selects solver curvature acceleration only: `None` or
`"numpy"` uses CPU, `"cupy"` uses CUDA float64, and `"torch-directml"` uses
DirectML float32. SI scaling/index contraction and downstream analysis run
on CPU float64. All public outputs are NumPy arrays or lists of NumPy arrays.
See [GPU validation](gpu.md) before choosing a device backend.

## Plotting

Install `.[plot]`, then:

```python
from warplab.visualiser import plot_tensor, plot_three_plus_one

fig, axes = plot_tensor(metric, sliced_planes=[1, 4], slice_locations=[1, 5])
fig.savefig("metric.png", dpi=150)
fig, axes = plot_three_plus_one(metric)
```

Plots return Matplotlib figures and axes; they never call `show()`.
Call `matplotlib.pyplot.show()` yourself for interactive display.
`plot_tensor` returns a 4x4 component grid; `plot_three_plus_one` returns a
2x5 grid containing lapse, covariant shift, and six spatial components.
The older `plotThreePlusOne` spelling remains available.

For compatibility with the original plotting interface, fixed axis numbers
and slice locations are **one-based**: 1=t, 2=x, 3=y, 4=z. Default slicing
fixes time and z at their middle cells. The first remaining array dimension
is plotted horizontally; the second vertically. Physical axes use grid
spacing and the generator center if available. Without spacing, axes show
one-based grid indices. Spatial aspect ratios are equal. Components each
have their own symmetric zero-centered colour scale; compare colourbar
values, not colours alone. `alpha` controls opacity, with a default of 1.

Boundary conventions and minimum stencil sizes are documented in
[testing.md](testing.md).
