# Changelog

## Unreleased

- Rename the package from `pywarp` to `warplab`.
- Document the CPU path as stable and the CuPy and DirectML backends as
  experimental.
- Rename the internal `surf_q_modified.sqm` surface-plot helper to
  `plot_themed_surface.plot_themed_surface`, reject a wrong number of arrays or
  a non-2D height array with a clear `ValueError` before creating any figure,
  and add tests, including a smoke test of the example that uses it.
- `do_frame_transfer` now verifies the metric and stress-energy tensor before
  transforming them, so a malformed metric reports "not verified" instead of a
  `KeyError`.
- Accept `diff_order` case-insensitively in `get_scalars` and `cov_div`, as
  `get_energy_tensor` already did, so `eval_metric(diff_order="Fourth")` works.
- `get_eulerian_transformation_matrix` raises `ValueError` for a non-symmetric
  metric (previously an `assert`, which disappears under `python -O`), for an
  unsupported shape (a 3 by 3 array used to fail with an `IndexError`), and for
  a metric with no Eulerian observers. Remove the "imaginary" check that could
  never trigger, because NumPy returns NaN where MATLAB returns a complex value.
- Remove an unreachable branch from `get_energy_conditions`.
- Raise `ValueError` from `get_energy_tensor`, `get_scalars` and `eval_metric`
  for an axis too short for the finite-difference stencil (2 to 4 samples for
  fourth order, 2 for second order). Upstream silently returns zero derivatives,
  which drops the time dependence of a moving metric. A single sample is still
  accepted.
- Make `verify_tensor` fail for non-Cartesian `coords` (upstream only warns), so
  the solver and analysis functions reject such metrics instead of silently
  applying Cartesian formulas. `get_energy_conditions` no longer raises a bare
  `Warning` for this case or fails with `KeyError` when `coords` is missing.
- Fix `verify_tensor` raising `TypeError` for input without a `type` field (a
  mistyped `warnings(...)` call); it now warns and returns `False`, as upstream
  does. Remove its unreachable branch and fix the `index` message.
- Remove the `phi_phi_flag` argument of `take_finite_difference_1` and `_2`. It
  was always 0, and its branches set boundary values to hard-coded constants.
- Remove unused internal code: `flip_index`, the `Metric` dataclass,
  `red_blue` and the superseded `surf_q` helper.
- Remove the `plotThreePlusOne` alias; use `plot_three_plus_one` with
  `sliced_planes` and `slice_locations`.

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
No PyPI publication has been performed by this repository setup.
