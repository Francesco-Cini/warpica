# GPU support and validation

## Stability: CPU is stable, GPU backends are experimental

From 1.0, the compatibility promise covers the **CPU (NumPy) path**: its
results and the `gpu=None` / `gpu="numpy"` behaviour. The `"cupy"` (CUDA) and
`"torch-directml"` backends are **experimental**. They are tested as
described below, but they may change in a 1.x release: precision, tolerances,
supported versions and the extras that install them are not frozen, and a
backend can be fixed, redesigned or removed without a major version bump.
CUDA has not been run on real hardware at all. Passing the CPU test suite
says nothing about either device backend.

The public `gpu` argument accelerates **curvature calculation**. Final SI
scaling/index contraction and analysis (frames, energy conditions and scalars)
run on CPU float64. Outputs are NumPy arrays. GPU calculations are opt-in;
the default remains CPU. There is no silent fallback when a requested device
backend is unavailable.

| Backend | Status | Precision | Validation status |
| --- | --- | --- | --- |
| NumPy | **Stable** | float64 | CPU analytic, integration and MATLAB reference checks |
| torch-directml | Experimental | float32 curvature, float64 host energy conversion | Tested on Radeon RX 580, Windows, Python 3.12, torch 2.4.1, torch-directml 0.2.5.dev240914 |
| CuPy | Experimental | float64 | Shared hardware tests provided; not executed on real hardware because no NVIDIA GPU has been available |

DirectML float64 reciprocal failed on this device, and strided slice writes
produced wrong derivatives without raising errors. The DirectML stencil path
now permutes each working axis into contiguous storage and builds boundaries
with concatenation. All four first-derivative axes and all 16 second-derivative
axis combinations are checked against NumPy for both orders. SI energy
prefactors exceed float32 range, so that conversion is deliberately done on
the host. CPU postprocessing does not restore precision lost in float32
curvature; use CPU/CUDA float64 for precision-sensitive investigations.

## Install and run hardware tests

DirectML currently needs an older supported Python; the tested environment
uses Python 3.12. Use a separate environment from Python 3.14:

```powershell
py -3.12 -m venv .venv-directml
.\.venv-directml\Scripts\python.exe -m pip install -e ".[test,directml]"
.\.venv-directml\Scripts\python.exe -m pytest tests/gpu --gpu-backend=torch-directml -q
.\.venv-directml\Scripts\python.exe -m benchmarks.backend_comparison --backend torch-directml
```

For an NVIDIA machine with a compatible CUDA 12 driver:

```powershell
python -m pip install -e ".[test,cuda]"
python -m pytest tests/gpu --gpu-backend=cupy -q
python -m benchmarks.backend_comparison --backend cupy
```

Without `--gpu-backend`, hardware tests explicitly skip. If you select a
backend that cannot initialize, tests fail rather than skip. Hosted CPU CI
excludes the hardware marker; it does not certify CUDA or DirectML.

Full GPU tests compare both solver orders on static and time-dependent
Alcubierre metrics and all analysis outputs. DirectML uses a scale-aware
2e-5 tolerance; CuPy uses 1e-10. Near-zero components use a tolerance scaled
to the largest reference component. These are selected regression cases,
not universal error bounds for arbitrary grids or near-singular metrics.

## Performance

`directml-results.json` records the machine, backend, grid, repeats, median
CPU/device elapsed times and maximum normalized error. Timings include data
transfers and float64 host conversion; the benchmark gathers outputs before
stopping the timer. GPU initialization and one full solver warmup per backend
and order occur before measurements (`--warmups 0` disables solver warmup). The small
5x5x5 spatial grid is slower on this DirectML device than CPU because this
implementation dispatches many small tensor operations. This small-grid result
does not predict performance on larger grids; see the scaling benchmark below.
The report does not measure native GPU memory usage.

The Ricci solvers cache inverse-metric contractions instead of recalculating
them inside every component loop. Fourth order also computes only the ten
independent symmetric components and their derivatives. On the RX 580, a
5x5x5 comparison with three repeats and no solver warmup measured:

| Order | CPU before / after | DirectML before / after |
| --- | --- | --- |
| Second | 0.0527 / 0.0148 s | 9.697 / 2.406 s |
| Fourth | 0.0585 / 0.0138 s | 13.000 / 2.120 s |

These are end-to-end medians on this machine, not general speed guarantees.
Maximum normalized CPU/device errors remained below 9e-7 in this comparison.
CPU is still much faster for this small workload: many separate device
operations remain, and host conversion is included. The contraction caches
retain 192 additional grid arrays (768 bytes per grid point in float32,
1536 in float64), so larger grids trade memory for less repeated work.

## Measuring the crossover

The [RX 580 scaling report](gpu-scaling-rx580.md) measures a speed crossover
between 50³ and 51³ points and a roughly 4× GPU advantage at 80³. Those faster
DirectML cases exceed the benchmark's strict 2e-5 normalized-error threshold;
the report separates timing wins from accuracy and memory limits.

Run a grid sweep with fresh worker processes for each grid/order:

```powershell
.\.venv-directml\Scripts\python.exe -m benchmarks.backend_scaling --backend torch-directml --sizes 5 8 16 24 32 48 64 80 96 --output backend-scaling.json
python -m benchmarks.plot_scaling backend-scaling.json --output backend-scaling.png
```

Plotting needs the optional `plot` dependencies. The sweep records every timed
sample, medians, normalized CPU/device error and an accuracy flag. It checkpoints
after each case and skips grids whose estimated host working set exceeds 75%
of currently available physical RAM. This guard is a conservative estimate,
not a measurement of peak RAM or GPU memory. Each worker also has a configurable
timeout (`--timeout`, default 600 seconds).

The benchmark refines a fixed eight-unit spatial domain with one time slice,
Alcubierre velocity 0.2, radius 1 and sigma 1. Metric construction and backend
initialization are excluded; input/output transfers, synchronization, float64
host conversion and final index contraction are included. One full solver
warmup precedes three timed calls by default. CPU uses float64; DirectML uses
float32 curvature and float64 host postprocessing. The error flag compares
`max(abs(device - cpu)) / max(abs(cpu))` against 2e-5 for DirectML (1e-10 for
CuPy), rather than estimating discretization error against an analytic solution.

For this implementation, the number of tensor components and stencil operations
is fixed: arithmetic work and stored grid arrays grow in proportion to the total
point count, `N = nt * nx * ny * nz`. A spatial side-length doubling therefore
multiplies the nominal work and memory by eight. Measured runtime need not follow
that factor: dispatch overhead dominates small GPU cases, while memory/cache
effects and allocation costs can change CPU throughput as arrays grow. A static
`n³` crossover should not be treated as an exact rule for time-dependent grids,
different hardware, CUDA float64 or the full analysis pipeline.

The RX 580 sweep completed through 80³ points. At 96³, a second-order DirectML
worker failed during a timed call with an insufficient-video-memory allocation
error; larger grids were not attempted. The warmup and earlier work in that
worker had already run, so this is a failure of the repeated-run protocol, not
proof that a single 96³ solve can never fit. Host RAM estimates do not bound
DirectML's temporary allocations or allocator/driver memory overhead.
