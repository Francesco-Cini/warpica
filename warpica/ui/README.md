# warpica desktop interface

A desktop window for the `warpica` pipeline: build a metric, solve for its
stress-energy tensor and analyse the result, without writing any code. It is a
thin layer over the library. Every number it shows comes from the same
functions you would call from Python.

The interface is **optional and outside the stable API**. `import warpica`
never loads it, and it may change between releases.

## Install and run

Requires Python 3.11 or newer. The `ui` extra adds PySide6 (the Qt bindings)
and Matplotlib. Install from a checkout:

```bash
git clone https://github.com/Francesco-Cini/warpica
cd warpica
python -m venv .venv
.venv/bin/python -m pip install -e ".[ui]"     # Windows: .venv\Scripts\python.exe
```

Then start it:

```bash
python -m warpica.ui
```

## How it works

The sidebar on the left lists the three stages in the order you do them. A
stage is greyed out until the one before it has produced something, and the
line under each name tells you where it stands.

| Stage | Needs | Produces |
|---|---|---|
| **Metric** | nothing | a metric on a grid |
| **Solver** | a metric | the stress-energy tensor |
| **Results** | a solved tensor | energy density, energy conditions, scalars, flow lines |

Changing an earlier stage throws away everything that depended on it. Build a
new metric and the solved tensor and every result disappear. Solve again and
the results disappear. This way a plot can never show numbers that belong to
settings you have since changed.

Nothing is saved automatically. Each plot has a toolbar with zoom, pan and a
save button for the picture on screen.

### Metric

Choose a metric, set its parameters and the grid, and press **Build metric**.

- **Metrics:** Alcubierre and Minkowski (flat space). The other metrics in the
  library are available from Python and are not offered here yet.
- **Alcubierre parameters:** speed `v` from 0 to 2 (in units of c), radius `R`
  from 0.5 to 3 m and wall steepness `sigma` from 0.1 to 5 per metre.
- **Grid:** the number of points and the spacing along t, x, y and z. The
  default is 1 × 17 × 17 × 9 points, with 1 s between time steps and 0.5 m
  between space points. Space takes 3 to 41 points per axis and time 1 to 16,
  because the metric is built on the window's own thread. The bubble is always
  placed in the middle of the grid.
- **Preview:** either the 16 metric components or the lapse, shift and spatial
  metric (the 3+1 fields), on a slice through the middle of the grid.

### Solver

Choose how to compute the stress-energy tensor and press **Solve**.

- **Derivative order:** fourth order (the default, more accurate) or second
  order (faster). Fourth order needs at least 5 points along every axis that
  has more than one point, and second order needs 3. A grid that is too small
  gives a clear error and changes nothing.
- **Backend:** the CPU (NumPy), which is the only stable one. The experimental
  CUDA and DirectML backends appear in the list only when CuPy or
  `torch-directml` is installed. See [GPU validation](../../docs/gpu.md).

The solve runs in a separate process, so the window stays usable. The solver
cannot report how far it has got, so you see a busy bar and the elapsed time
rather than a percentage. **Cancel** really stops the work. Closing the window
stops it too. Starting a process costs about a second, which is noticeable on
small grids and negligible on large ones.

When it finishes, the 16 components of the tensor are drawn, in SI units.

### Results

Four tabs, each reading from and writing to the same stored results.

- **Energy density:** the energy density measured by observers at rest in the
  grid, in J/m³. It is computed automatically when the tensor is solved.
- **Energy conditions:** the null, weak, strong or dominant condition, sampled
  over a number of directions (and, for weak and strong, speeds). Each cell
  shows the worst value found, and a negative value means the condition is
  violated there. A summary lists the worst value and how many cells are
  violated. Sampling finitely many observers cannot prove that a condition
  holds.
- **Scalars:** the expansion, shear and vorticity of the observers at rest in
  the grid, computed with the derivative order used for the tensor.
- **Flow lines:** lines that follow the momentum density from a square lattice
  of seed points on a plane you choose. You set the plane and where it sits, the
  seeds per side, the step size in grid cells and the most steps per line. A
  step moves a line by the step size where the momentum is strongest, and
  lines stop at the edge of the grid. Large numbers of seeds take longer.

The energy conditions, scalars and flow lines run in the background and can be
cancelled. The field plots share a slice viewer: pick the field, the plane
(x-y, x-z or y-z), the position along the remaining axis and, on a grid with
several time steps, the time step. The colour scale is centred on zero, so
negative values are blue and positive values are red.

On a grid with several time steps, the Alcubierre generator places the bubble
at the middle time step only. The viewer and the flow lines start there, and
the other time steps are flat, so they have no flow to trace.

## Layout of the code

```
warpica/ui/
├── __main__.py        # python -m warpica.ui
├── core/              # no Qt imports, tested without a screen
│   ├── registry.py    # the metrics on offer and their parameters
│   ├── state.py       # Session: what is built, solved and analysed
│   ├── solving.py     # solver choices and the function the solve runs
│   ├── analysis.py    # the analysis functions and result summaries
│   ├── slicing.py     # slices and coordinates of the grid
│   ├── flow_plot.py   # the flow line figure
│   ├── jobs.py        # run a function in a separate process, with cancel
│   └── worker.py      # the entry point of that process
├── windows/           # the main window and its sidebar
├── pages/             # one page for each stage
└── widgets/           # forms, the slice viewer, the Run/Cancel bar
```

The rule that shapes it: `core/` never imports Qt, and the Qt code only
collects input, calls `core/` and draws the result. The interface adds no
physics of its own.

### Adding a metric

The input fields are built from `core/registry.py`, so a new metric needs
one entry there and no window code. Describe its parameters and give a function
that builds it:

```python
def _build_example(grid_size, world_centre, grid_scale, radius):
    return metric_get_example(grid_size, world_centre, radius, grid_scale)


METRICS["example"] = MetricSpec(
    label="Example",
    parameters=(Parameter("radius", "Radius R (m)", 1.0, 0.5, 3.0, 0.1),),
    build=_build_example,
)
```

A `Parameter` is `(name, label, default, minimum, maximum, step)`, and the
builder receives the grid size, centre and spacing followed by one keyword per
parameter. Choose defaults and ranges that make physical sense on the default
grid, which is only a few metres across.

## Tests

The Qt-free tests need nothing extra. The window tests need PySide6 and
pytest-qt and are skipped when either is missing:

```bash
python -m pip install -e ".[ui,test,ui-test]"
python -m pytest tests/ui
```

On a machine without a display, tell Qt not to open windows by setting
`QT_QPA_PLATFORM=offscreen` first (PowerShell: `$env:QT_QPA_PLATFORM = "offscreen"`).
Hosted CI installs only the `test` and `plot` extras, so it runs the Qt-free
tests and skips the window tests.

## Limits

- Only the Alcubierre and Minkowski metrics are offered.
- The solver and the analyses cannot show a percentage, only elapsed time.
- Metrics are built on the window's thread, so the grid size is capped.
- The grid is always centred on the origin, and the flow line plots assume it.
- It is not packaged as a standalone application. A background job starts with
  `python -m`, which needs an installed `warpica`.
