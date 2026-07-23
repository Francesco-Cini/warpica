# PyWarp

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-experimental-orange.svg)]()

PyWarp is an experimental Python library for constructing and analysing
warp-drive spacetimes using Einstein's theory of general relativity. It is a
Python adaptation of the original WarpFactory MATLAB toolkit.

> [!WARNING]
> This project is under active development. Its API may change between releases,
> and results should be independently verified before being used in research.

## Features

- Construct several warp-drive and reference metrics.
- Calculate the Einstein tensor and corresponding energy tensor.
- Evaluate null, weak, strong, and dominant energy conditions.
- Calculate expansion, shear, and vorticity scalars.
- Visualise tensor components and momentum-flow lines.

## Requirements

- Python 3.10 or newer
- NumPy
- Matplotlib for visualisation

## Installation

Until a package is published to PyPI, install the latest development version
directly from GitHub:

```bash
python -m pip install "git+https://github.com/Francesco-Cini/pywarp.git"
```

For local development:

```bash
git clone https://github.com/Francesco-Cini/pywarp.git
cd pywarp
python -m pip install -e .
```

## Quick start

```python
import numpy as np

from pywarp.metrics.alcubierre.metric_get_alcubierre import (
    metric_get_alcubierre,
)
from pywarp.solver.get_energy_tensor import get_energy_tensor

grid_size = np.array([5, 10, 10, 10])
world_centre = (grid_size + 1) / 2

metric = metric_get_alcubierre(
    grid_size=grid_size,
    world_centre=world_centre,
    v=0.1,
    R=2,
    sigma=0.5,
    grid_scale=None,
)

energy_tensor = get_energy_tensor(metric, "fourth")
print(energy_tensor["tensor"].shape)
```

More complete examples are available in the [`examples`](examples) directory.

## Attribution

PyWarp is an independent adaptation of the MIT-licensed
[WarpFactory MATLAB toolkit](https://github.com/NerdsWithAttitudes/WarpFactory)
created by Christopher Helmerich and Jared Fuchs.

This project is not an official release of WarpFactory and is not affiliated
with or endorsed by the original authors.

## License

PyWarp is distributed under the [MIT License](LICENSE). The license
retains attribution for the original WarpFactory authors and identifies the
copyright holder of this Python adaptation.
