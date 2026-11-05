# warplab

Python numerical tools for spacetime metrics, stress-energy tensors and
sampled energy-condition analysis, ported from WarpFactory.

## Start here

From the repository root, with Python 3.11 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test,plot]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m examples.basic.minkowski
.\.venv\Scripts\python.exe -m examples.analysis.alcubierre
```

Install `".[test,plot]"` if you also want plotting examples. On Linux/macOS,
use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.

- [Testing and examples: a beginner's guide](docs/testing.md)
- [Runnable examples](examples/README.md)
- [Second-order source conventions](warplab/solver/utils/second_order/README.md)

CPU checks cover both finite-difference orders and all built-in metric
families. Independent MATLAB fixtures cover representative flat/static/dynamic
cases. The CPU path is the stable, supported one. GPU curvature backends
are experimental: DirectML has been tested on an AMD RX 580, and CUDA has not
yet been run on NVIDIA hardware. Analysis postprocessing runs on CPU.

- [Public API and plotting](docs/api.md)
- [MATLAB comparisons and intentional differences](docs/reference-validation.md)
- [GPU precision, hardware tests and benchmark](docs/gpu.md)
- [Build and release checks](docs/releasing.md)
- [Changelog](CHANGELOG.md)
- [Code style and development checks](docs/development.md)
