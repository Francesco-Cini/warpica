"""Type aliases shared by the public signatures.

Metrics and stress-energy tensors are plain dictionaries whose layout is
documented in the API guide. They are typed as ``dict[str, Any]`` on purpose, so
the type system does not freeze a key layout that the package may still extend.
"""

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

# A NumPy array of any shape and dtype.
Array: TypeAlias = NDArray[Any]
# A short numeric vector such as a grid size or spacing: a sequence or 1-D array.
Vector: TypeAlias = Sequence[float] | NDArray[Any]
# A grid size such as (nt, nx, ny, nz): a sequence of ints or an integer array.
GridSize: TypeAlias = Sequence[int] | NDArray[np.integer[Any]]
# A metric, stress-energy tensor or other tensor dictionary.
TensorDict: TypeAlias = dict[str, Any]
# Finite-difference accuracy order.
DiffOrder: TypeAlias = Literal["second", "fourth"]
# Array backends accepted by the solver.
Backend: TypeAlias = Literal["numpy", "cupy", "torch-directml"]
