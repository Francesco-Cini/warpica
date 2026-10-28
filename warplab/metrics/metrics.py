from dataclasses import dataclass, field
from datetime import date as Date
from datetime import datetime

import numpy as np


@dataclass
class Metric:
    """Store a metric and its coordinate, index, and construction metadata.

    Attributes
    ----------
    name : str
        Human-readable metric name.
    scaling : ndarray
        Grid spacings in (t, x, y, z) order.
    coords : str
        Coordinate system, normally cartesian.
    index : str
        Covariant or contravariant component convention.
    date : date, datetime, or str
        Construction date.
    tensor : ndarray or list of lists of ndarrays
        Components indexed [mu][nu], each with shape (nt, nx, ny, nz).
    type : str
        Tensor category; defaults to metric.
    params : dict
        Construction parameters, independently allocated for each instance.
    """

    name: str
    scaling: np.ndarray
    coords: str
    index: str
    date: Date | datetime | str

    tensor: np.ndarray | list[list[np.ndarray]]

    type: str = "metric"
    params: dict[str, object] = field(default_factory=dict)
