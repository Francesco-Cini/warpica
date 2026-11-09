# warpica

[![CPU tests and package](https://github.com/Francesco-Cini/warpica/actions/workflows/ci.yml/badge.svg)](https://github.com/Francesco-Cini/warpica/actions/workflows/ci.yml)

Numerical spacetime metrics, stress-energy tensors and sampled energy-condition
analysis in Python. `warpica` is a port of
[WarpFactory](https://github.com/NerdsWithAttitudes/WarpFactory), the MATLAB
toolkit for analysing warp-drive geometries.

> **Status.** The project is preparing a 1.0 release.
> It is not on PyPI yet; install from a checkout.

## Credit to WarpFactory

All of the physics, the algorithms and the structure of this package come from
WarpFactory, created by **Christopher Helmerich and Jared Fuchs**, with
contributions from Alexey Bobrick, Luke Sellers, Brandon Melcher, Justin Feng and
Gianni Martire. This is an independent port, not an official WarpFactory
release, and the WarpFactory authors have not reviewed it. WarpFactory is MIT
licensed, as is this project.

If you use this code in published work, please cite the WarpFactory paper, as
its authors ask:

> C. Helmerich, J. Fuchs, A. Bobrick, L. Sellers, B. Melcher and G. Martire,
> "Analyzing warp drive spacetimes with Warp Factory",
> *Classical and Quantum Gravity* (2024),
> [doi:10.1088/1361-6382/ad2e42](https://doi.org/10.1088/1361-6382/ad2e42),
> [arXiv:2404.03095](https://arxiv.org/abs/2404.03095).

The authors also describe the solver, analyser and optimiser modules in
[arXiv:2404.10855](https://arxiv.org/abs/2404.10855). To cite this port itself,
see [CITATION.cff](CITATION.cff).

## What it does

- **Metrics:** Minkowski, Alcubierre, Lentz, Modified Time, Van Den Broeck,
  Schwarzschild and a spherical warp shell (plus comoving variants), a
  `custom_metric` builder from lapse, shift and spatial metric, and 3+1
  decomposition helpers.
- **Solver:** the stress-energy tensor of any sampled metric, with second- or
  fourth-order finite differences, in SI units.
- **Analyser:** null, weak, strong and dominant energy conditions sampled over
  directions, expansion, shear and vorticity scalars, momentum flow lines,
  index changes and frame transfers.
- **Visualiser:** Matplotlib slices of metric and tensor components.

## Quick start

Requires Python 3.11 or newer.

```bash
git clone https://github.com/Francesco-Cini/warpica
cd warpica
python -m venv .venv
.venv/bin/python -m pip install -e ".[plot]"     # Windows: .venv\Scripts\python.exe
```

```python
import numpy as np
from warpica.metrics import metric_get_alcubierre
from warpica.analyser import eval_metric

grid = np.array([1, 9, 9, 9])  # (time, x, y, z) samples
spacing = np.array([1.0, 0.5, 0.5, 0.5])  # seconds and metres
centre = (grid + 1) / 2 * spacing
metric = metric_get_alcubierre(grid, centre, v=0.2, R=1, sigma=2, grid_scale=spacing)

result = eval_metric(metric, num_angular_vec=24, num_time_vec=4)
print(result["weak"].min())  # negative: the condition is violated
```

The [quickstart](docs/quickstart.md) walks through this step by step.

## Stability

The **CPU path is stable** and is what the 1.x compatibility promise covers.
The CUDA (CuPy) and DirectML backends are **experimental**: DirectML has been
tested on one AMD RX 580, and CUDA has never been run on NVIDIA hardware. See
[GPU validation](docs/gpu.md).

Results are checked against MATLAB WarpFactory on ten reference cases, and the
places where this port deliberately differs are listed and justified in
[MATLAB comparisons](docs/reference-validation.md). Read that page before
comparing numbers with published WarpFactory results.

## Documentation

- Hosted site: <https://francesco-cini.github.io/warpica/> (once published)
- [Quickstart](docs/quickstart.md) and [concepts](docs/concepts.md) (tensor layout, units, indexing)
- [How it works](docs/how-it-works.md): the pipeline and a glossary of the symbols
- [Moving from MATLAB WarpFactory](docs/matlab-migration.md)
- [Public API and plotting](docs/api.md)
- [Runnable examples](examples/README.md)
- [Versioning and support policy](docs/policy.md)
- [MATLAB comparisons and intentional differences](docs/reference-validation.md)
- [GPU precision, hardware tests and benchmark](docs/gpu.md)
- [Testing guide](docs/testing.md), [development checks](docs/development.md)
- [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md)

## Licence

MIT. See [LICENSE](LICENSE).
