# Examples

Examples are small programs you run to learn the API. Run these commands
from the repository root after installing the project (`pip install -e .`).
They do not require copying files into `warplab/` or editing `sys.path`.

| Command | Purpose |
| --- | --- |
| `python -m examples.basic.minkowski` | Generate flat space and compute its zero stress-energy with both orders. |
| `python -m examples.basic.custom_metric` | Supply your own lapse, shift and spatial metric functions. |
| `python -m examples.analysis.alcubierre` | Run a complete CPU analysis on a small curved metric. |
| `python -m examples.analysis.compare_orders` | See second- and fourth-order errors shrink as the grid is refined. |
| `python -m examples.visualisation.tensor_slices --output-dir figures` | Save the public tensor and 3+1 plots as PNG files. |
| `python -m examples.visualisation.alcubierre_plots` | Run the existing multi-figure plotting demonstration; requires `pip install -e ".[plot]"`. |
| `python -m benchmarks.alcubierre_timing` | Measure a small solver run on your machine. |
| `python -m benchmarks.backend_scaling --backend torch-directml` | Measure CPU/GPU scaling across grid sizes; requires the selected GPU backend. |

Every script has a `main()` function and a `__main__` guard. Importing a
script therefore does not start a simulation or open a plot. The four
non-graphical learning examples are also exercised by integration tests.

All grids use `(t, x, y, z)` order. Components use `tensor[mu][nu]`.
Generator sample coordinates follow WarpFactory's `(index + 1)*spacing -
world_centre` convention, with Python array indices starting at zero.
For a centered grid use `(grid_size + 1)/2 * spacing`.
Time spacing is in seconds; the solver converts derivatives to the `ct`
basis. A differentiated axis needs at least 3 samples for second order and 5 for
fourth order; an axis of 2 to 4 samples (fourth order) or 2 (second order) raises
`ValueError`, and a single sample is a snapshot with no derivatives along it.
A one-time-slice example is a spatial snapshot and cannot resolve time
variation of a moving bubble.

These small grids and direction counts keep examples quick; they are not
converged research configurations. See `docs/testing.md` before interpreting
numerical results or choosing tolerances.

`examples/analysis/compare_orders.py` imports the finite-difference kernels
directly because it measures them; those are internal helpers, not public API.
