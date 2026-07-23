# WarpFactoryPy v0.2.0

Version 0.2.0 expands WarpFactoryPy from its initial metric and solver
implementation into a broader spacetime-analysis toolkit. This release adds
energy-condition evaluation, scalar analysis, frame and tensor transformations,
additional metrics, solver utilities, and more complete examples.

> WarpFactoryPy remains experimental. APIs may change, and numerical results
> should be independently verified before being used in research.

## Highlights

- Added evaluation of the null, weak, strong, and dominant energy conditions.
- Added expansion, shear, and vorticity scalar calculations.
- Added momentum-flow-line generation and supporting interpolation utilities.
- Added frame-transfer and tensor-index transformation tools.
- Added Schwarzschild and Van Den Broeck metric implementations.
- Restored a Minkowski reference metric for analysis routines.
- Added metric decomposition into 3+1 form.
- Expanded solver utilities for covariant divergence, Christoffel symbols,
  tensor conversion, and trilinear interpolation.
- Added a complete Alcubierre analysis example and a timing example.
- Expanded the README with installation, usage, attribution, and project-status
  guidance.

## Breaking changes

- The import package has been renamed from `warp_factory_py` to `pywarp`.
  Applications must update their imports:

  ```python
  # Before
  from warp_factory_py.metrics.alcubierre.metric_get_alcubierre import (
      metric_get_alcubierre,
  )

  # Version 0.2.0
  from pywarp.metrics.alcubierre.metric_get_alcubierre import (
      metric_get_alcubierre,
  )
  ```

- The separate Alcubierre, Lentz, and Modified Time `*_comoving` modules have
  been removed. Use the corresponding primary metric constructors instead.
- The project version has moved from the initial `0.1.x` series to `0.2.0` to
  reflect the expanded API and package rename.

## Attribution and license

WarpFactoryPy is an independent Python adaptation of the MIT-licensed
[WarpFactory MATLAB toolkit](https://github.com/NerdsWithAttitudes/WarpFactory)
created by Christopher Helmerich and Jared Fuchs. It is not an official release
of WarpFactory and is not affiliated with or endorsed by the original authors.

The project is distributed under the MIT License.

**Full changelog:** https://github.com/Francesco-Cini/pywarp/compare/v0.1.2...v0.2.0

## Pre-publication checklist

- [ ] Complete the package rename in internal imports and package discovery.
- [ ] Confirm that the wheel contains the `pywarp` package and required
      dependency metadata.
- [ ] Run metric, solver, analyser, and clean-install smoke tests.
- [ ] Add or confirm continuous-integration checks for the release commit.
- [ ] Create tag `v0.2.0` from the validated release commit.
- [ ] Attach the final wheel and source distribution, if distributing through
      GitHub Releases.
