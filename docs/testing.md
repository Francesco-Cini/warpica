# Testing and examples: a beginner's guide

A **test** asks the computer to check a specific claim and fail if it is
false. An **example** teaches a human how to use the library. A
**benchmark** measures speed or memory. These serve different purposes:
a useful-looking plot is not evidence that its underlying mathematics is
correct, and a fast calculation may still be wrong.

## Folder structure

```text
warpica/                         Library implementation
  metrics/                      Metric generators and 3+1 components
  solver/                       Derivatives, curvature, stress-energy
  analyser/                     Frames, energy conditions, scalar quantities
  visualiser/                   Plotting helpers

tests/                          Automated correctness checks
  conftest.py                   Reusable pytest fixtures (small input data)
  metrics/                      Generators, limits, signature, 3+1 reconstruction
  solver/                       Derivative stencils, inverses, second-order pipeline
  analyser/                     Index changes, frames, conditions, scalars, flow
  numerical/                    Analytic curvature and convergence checks
  integration/                  Full workflows and executable examples
  reference/                    Frozen MATLAB outputs and parity tests
  visualiser/                   Slices, labels, figure returns and saved images
  gpu/                          Opt-in real device tests

examples/                       Programs to read, run, and adapt
  basic/
    minkowski.py                Your first metric and energy tensor
    custom_metric.py            Supply lapse, shift and spatial functions
  analysis/
    alcubierre.py               Complete analysis with both physical outputs and units
    compare_orders.py           Observe how numerical errors decrease
  visualisation/
    alcubierre_plots.py          Existing plotting demonstration, moved out of tests

benchmarks/
  alcubierre_timing.py           Existing timing script, moved out of tests

docs/
  testing.md                    This guide
```

The domain folders (`metrics`, `solver`, `analyser`) mostly contain **unit
tests**: small checks of one function or closely related functions.
The second-order test file also contains numerical and public-API checks;
folders help navigation but do not enforce a strict classification.

An **integration test** connects several modules. For example, constructing
Alcubierre space, calculating stress-energy, changing frame, and evaluating
energy conditions checks that all their input/output conventions agree.

A **numerical test** compares against mathematics we already know. Examples:
Minkowski curvature is zero; the derivative of sin(x) is cos(x); and a known
curved metric has an analytic Ricci tensor. A convergence test halves the
grid spacing and checks that the error falls at the expected rate. For the
interior stencils here, second order gives about 4 times smaller error and
fourth order gives about 16 times smaller error for a smooth function.

A **regression test** is any test added to prevent a discovered bug from
returning. The constant-field second-derivative test is a regression test:
it catches the old incorrect signs that produced curvature in flat space.

## Install and run

Run these from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test,plot]"
.\.venv\Scripts\python.exe -m pytest -q
```

`-e` installs the local project in editable mode: source changes are used
without reinstalling. `[test]` adds pytest and its coverage plugin; `[plot]` adds Matplotlib.
`-q` means quieter output. A dot means a passed case, `F` a failed assertion,
and `E` an error that prevented a case from running. A nonzero exit status
means the run failed. Read the first failure and its traceback before fixing
later failures that may have the same cause.

The environment used during this repair is `.venv-second-order`. To run
with that existing environment, replace `.venv` in the commands above with
`.venv-second-order`. Installing another environment is not necessary.

Useful commands (replace `python` with your environment's executable):

```powershell
python -m pytest tests/solver -q
python -m pytest tests/analyser/test_energy_conditions.py -q
python -m pytest -k "flrw" -q
python -m pytest -m numerical -q
python -m pytest -m integration -q
python -m pytest -x
python -m pytest --cov=warpica --cov-report=term-missing
python -m examples.basic.minkowski
python -m examples.analysis.alcubierre
python -m benchmarks.alcubierre_timing
```

`-k` selects test names, `-m` selects explicit markers, and `-x` stops on the
first failure. The normal test run includes numerical and integration tests;
none of the implemented CPU checks are hidden behind slow or GPU markers.
Hardware tests skip unless you select `--gpu-backend`; CI uses `-m "not gpu"`.
Coverage reports show which lines executed, not whether the numerical
answers were correct. A high percentage is useful but not a correctness
certificate.

## How to read and write a test

Pytest discovers files named `test_*.py` and functions named `test_*`.
This example follows the usual arrange, act, assert pattern:

```python
import numpy as np
from warpica.solver.utils.take_finite_difference_2 import take_finite_difference_2


def test_constant_field_has_zero_second_derivative():
    field = np.ones((1, 7, 7, 7))  # Arrange an input with a known answer.
    actual = take_finite_difference_2(field, 1, 1, np.ones(4))  # Act.
    np.testing.assert_allclose(actual, 0, atol=1e-12)  # Assert.
```

`assert` expresses a claim. `assert_allclose` permits small floating-point
rounding errors. It checks approximately `abs(actual - expected) <= atol +
rtol * abs(expected)`. Use `atol` for a known zero, and choose `rtol` based on
the expected scale and discretization error. Do not increase tolerances just
to make a failing test green: first determine whether the expected result,
implementation, boundary convention, or grid resolution is wrong.

A **fixture** supplies shared inputs. For example, `minkowski_metric` in
`conftest.py` constructs small flat-space data; placing its name in a test's
arguments asks pytest to supply it. Keep fixture arrays small so tests run
quickly. `@pytest.mark.parametrize` runs the same assertion for several axes,
metrics, or solver orders without duplicating the test body.

When adding a feature:

1. Choose a small example with a known answer or invariant.
2. Write a test that demonstrates the missing behavior or bug.
3. Implement the change and run that test while working.
4. Run the full suite before committing.
5. Add an example only if users need to learn a new workflow.

For a metric, test shape, symmetry, finite values, one negative eigenvalue,
a simple limiting case, and 3+1 reconstruction. For a derivative, test
constants, polynomials, all relevant axes, boundary handling, and convergence.
For an analysis operation, test a known physical result and ensure the
operation does not silently modify its inputs.

## Numerical conventions and current limits

- Tensor arrays have component axes first: `(4, 4, t, x, y, z)`. A nested
  4x4 list of `(t,x,y,z)` arrays is also accepted by the solver.
- Generator positions are `(Python index + 1)*spacing - world_centre`,
  retaining the original WarpFactory coordinate convention. Supply the center
  in physical units, not unscaled cell indices.
- Time spacing is seconds; curvature and scalar derivatives use the `ct`
  basis. A single time slice intentionally has zero time derivatives.
- Second order needs 3 samples along a differentiated axis; fourth order
  needs 5. An axis may also have exactly one sample (nothing varies along it).
  `get_energy_tensor`, `get_scalars` and `eval_metric` raise `ValueError` for an
  axis of two or more samples that is too short. The low-level derivative
  functions still return zeros there, as upstream does, so call them through
  the public entry points.
- Pure derivatives copy the nearest interior value to their boundary cells.
  Mixed derivatives leave the boundary band zero. Convergence tests exclude
  these bands; copied edges are not high-order boundary conditions.
- Energy-condition maps are minima over sampled directions (dominant uses
  a causal/future-directed flux score). Negative means a sampled violation.
  Finite sampling cannot establish a condition in every possible direction.
  Timelike samples exclude exactly null velocities. Map magnitudes depend
  on vector normalization and should not be compared as invariant scalars.
- Comoving Modified Time and Van Den Broeck use their comoving shift in
  `g00`, correcting an upstream inconsistency so their unmodified limits
  agree with comoving Alcubierre. Modified Time also uses the correct spatial
  coordinate axes rather than upstream's time/spatial indexing typo.
- Warp shell now builds density, mass, pressure, lapse, radial metric and
  optional shift profiles. It follows the upstream constant-density pressure
  prescription, not a new general TOV solver. It uses 100,000 radial samples,
  centered moving-average smoothing, and **linear radial interpolation**
  rather than upstream's Legendre interpolation. Flat and Schwarzschild
  exterior limits are tested; this interpolation choice intentionally differs
  from MATLAB. Core Ricci/energy comparisons use separate Alcubierre fixtures.
- Schwarzschild rejects grids containing its origin or coordinate horizon.
  Eulerian 3+1 analysis requires a positive-definite spatial metric and a
  real lapse, so use the exterior region for that workflow.
- MATLAB fixtures and DirectML hardware checks now exist. See
  `reference-validation.md` and `gpu.md` for coverage, intentional differences
  and precision limits. CUDA hardware validation remains external.
- Both public plotting functions now return figures and have automated slicing
  and render checks. The plotting demonstration remains a user-facing example.

## Changes covered by this repair

Fourth-order pure second-derivative signs, time-axis conversion and double-time
scaling; non-mutating tensor index changes and contractions; Eulerian
conversion calls and all time-space signs; finite energy-condition reductions,
correct strong-condition contraction and dominant future direction; scalar
order/time handling; final momentum-flow points; and the metric generators'
indexing, metadata, coordinates, callable constants and limiting behavior.

The original plotting and timing scripts were preserved under `examples/`
and `benchmarks/` with `main()` guards. They no longer execute during pytest
collection. The four small non-graphical examples also run in integration
tests so API changes cannot silently break the teaching material.
