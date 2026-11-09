# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and from 1.0.0 the
project follows [semantic versioning](https://semver.org); see
[docs/policy.md](docs/policy.md).

## Unreleased

Work towards 1.0.0.

### Changed

- Rename the package from `pywarp` to `warpica`.
- Define the public API as the names in `__all__` of `warpica.metrics`,
  `solver`, `analyser`, `units` and `visualiser`, with lazy top-level re-exports.
  Everything else is internal.
- Document the CPU path as stable and the CuPy and DirectML backends as
  experimental.
- Require Python 3.11 or newer. Python 3.10 reached its scheduled end of life in
  October 2026, and current Matplotlib and NumPy no longer support it. The test
  matrix is now 3.11, 3.12, 3.13 and 3.14; the code needed no change.
- Declare dependency floors: `numpy>=1.26`, `array-api-compat>=1.5` and
  `matplotlib>=3.8`. Each is the oldest version the full test suite passes on,
  checked by a `minimum` CI job that installs exactly those versions.
- Annotate the public API and ship a `py.typed` marker. Every public function
  has full annotations (enforced by a test), metrics and tensors are
  `dict[str, Any]`, `diff_order` and the backend are `Literal` types, and
  `get_energy_conditions` is overloaded on `return_vec`. CI runs mypy and a test
  that type-checks small user programs.
- Raise `ValueError` everywhere the package used a bare `Exception`, and for the
  one `Warning` raised for an unknown frame. `ValueError` is a subclass of
  `Exception`, so `except Exception` code keeps working. A test rejects any bare
  `Exception`, `BaseException` or `Warning` raise in the package. The messages of
  the comoving generators, Schwarzschild and the warp shell for a single time
  slice are unified and report the size received.
- `get_energy_tensor`, `get_scalars` and `eval_metric` raise `ValueError` for an
  axis too short for the finite-difference stencil (2 to 4 samples for fourth
  order, 2 for second order). Upstream silently returns zero derivatives, which
  drops the time dependence of a moving metric. A single sample is still
  accepted.
- `verify_tensor` fails for non-Cartesian `coords` (upstream only warns), so the
  solver and analysis functions reject such metrics instead of silently applying
  Cartesian formulas. `get_energy_conditions` no longer raises a bare `Warning`
  for this case or fails with `KeyError` when `coords` is missing.
- `plot_three_plus_one` verifies its metric like `plot_tensor`, so a malformed
  or non-Cartesian metric raises `ValueError("Invalid tensor")` instead of a
  `KeyError`. `do_frame_transfer` now verifies the metric and stress-energy
  tensor first, so a malformed metric reports "not verified".
- The second-order stress-energy uses G and c from `warpica.units`, like the fourth-order
  path, instead of the rounded constants of upstream's second-order code. The two orders now
  agree, and second-order results are about 3.9e-5 smaller than MATLAB's. See
  `docs/reference-validation.md`.
- Accept `diff_order` case-insensitively in `get_scalars` and `cov_div`, as
  `get_energy_tensor` already did.
- Rename the internal `surf_q_modified.sqm` surface-plot helper to
  `plot_themed_surface.plot_themed_surface`, and reject a wrong number of arrays
  or a non-2D height array with a clear `ValueError` before creating any figure.
- `get_eulerian_transformation_matrix` raises `ValueError` for a non-symmetric
  metric (previously an `assert`, which disappears under `python -O`), for an
  unsupported shape (a 3 by 3 array used to fail with an `IndexError`), and for
  a metric with no Eulerian observers.
- Add explanatory comments across the package: the solver and analyser pipeline, every
  metric generator, the finite-difference stencils and their edge handling, the plotting
  code and the GPU helpers, and docstrings for the remaining helpers. Comments and
  docstrings only: the parsed syntax tree of every touched file is identical to before
  once docstrings are removed.
- Examples import from the public API, and `examples/README.md` states the
  grid-size limits correctly.
- Remove the stray `backend-results.json`; `benchmarks/backend_comparison.py` now
  writes to the gitignored `benchmarks/results/backend-results.json` by default.

### Added

- Validation against MATLAB R2026a WarpFactory on ten reference cases, with the
  deliberate divergences documented in `docs/reference-validation.md` and pinned
  by tests.
- An advisory performance guard (`tests/performance`, run with
  `--run-performance`) that fails on super-linear cost growth or a slowdown of
  about three times against a fixed NumPy kernel. It runs as its own CI job that
  never blocks a merge.
- A 98% line-coverage floor in CI. The device-only CUDA and DirectML branches
  are the only exclusions. CI also gains a `lint` job (`ruff check` and
  `ruff format --check`) and Python 3.13.
- Documentation: README with WarpFactory attribution and citation, quickstart,
  concepts, a "how it works" guide with a symbol glossary, MATLAB migration guide and a versioning and support policy.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`, issue
  and pull request templates, and Dependabot for GitHub Actions.
- Document why the `directml` extra stays pinned to `torch-directml`
  0.2.5.dev240914: it is the only installable release line and has no wheels
  for Python 3.13 or newer.

### Fixed

- `AttributeError: module 'numpy' has no attribute 'acos'` on NumPy 1.x:
  `np.acos` was added in NumPy 2.0, so every energy-condition calculation failed
  there. Use `np.arccos`.
- `eval_metric(diff_order=None)` raised `ValueError`; it now means "fourth".
- `metric_get_lentz` and `metric_get_lentz_comoving` failed with `TypeError` when
  `scale` was omitted and `grid_size` was a list.
- `verify_tensor` raised `TypeError` for input without a `type` field (a
  mistyped `warnings(...)` call); it now warns and returns `False`, as upstream
  does. Its `index` message is corrected.

### Removed

- The `plotThreePlusOne` alias; use `plot_three_plus_one` with `sliced_planes`
  and `slice_locations`.
- The `phi_phi_flag` argument of `take_finite_difference_1` and `_2`. It was
  always 0, and its branches set boundary values to hard-coded constants.
- Unused internal code: `flip_index`, the `Metric` dataclass, `red_blue`, the
  superseded `surf_q` helper, dead assignments and an unreachable branch in
  `get_energy_conditions`, and an "imaginary" check in the Eulerian
  transformation that could never trigger (NumPy returns NaN where MATLAB
  returns a complex value).
- The unused `count_lines.py` script.

## 0.1.1, 0.1.2 and 0.2.0

Early development tags published under the earlier project names WarpFactoryPy
and PyWarp (`v0.1.1` and `v0.1.2` in May 2026, `v0.2.0` in July 2026, the last
one being the rename to PyWarp). They have no detailed notes. All behaviour described above supersedes them.

## 0.1.0 - release candidate

- Complete and validate second- and fourth-order CPU curvature/energy solvers.
- Repair tensor indexing, frame conversion, energy-condition maps and scalars.
- Correct built-in metric generators and complete the spherical warp shell.
- Add component and 3+1 slice plotting with returned Matplotlib figures.
- Add pinned MATLAB fixtures, analytic/convergence tests and hardware tests.
- Fix DirectML precision and strided stencil writes; keep SI scaling on CPU.
- Add CI, distribution checks, examples, API documentation and release guide.

Compatibility notes: plots use one-based slice arguments; computation uses
zero-based Python array indices. Plot functions no longer show windows
implicitly. DirectML curvature uses float32. Analysis remains on CPU.
See `docs/reference-validation.md` for intentional upstream differences.
