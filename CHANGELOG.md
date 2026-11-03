# Changelog

## Unreleased

- Rename the package from `pywarp` to `warplab`.
- Document the CPU path as stable and the CuPy and DirectML backends as
  experimental.
- Rename the internal `surf_q_modified.sqm` surface-plot helper to
  `plot_themed_surface.plot_themed_surface`, reject a wrong number of arrays or
  a non-2D height array with a clear `ValueError` before creating any figure,
  and add tests, including a smoke test of the example that uses it.
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
