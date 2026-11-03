"""Numerical spacetime metrics, stress-energy and energy-condition analysis.

The supported public API is the set of names in ``__all__`` of this package
and of ``warplab.metrics``, ``warplab.solver``, ``warplab.analyser``,
``warplab.units`` and ``warplab.visualiser``. Everything else, including the
helper modules under ``utils`` directories, is internal and may change
without notice.

Top-level names are loaded on first use, so ``import warplab`` is cheap and
never imports the optional plotting or GPU dependencies.
"""

import importlib

__version__ = "0.3.0.dev0"

_SUBPACKAGES = ("analyser", "gpu", "metrics", "solver", "units", "visualiser")

# Top-level convenience names and the subpackage that defines each one.
# tests/test_public_api.py keeps this in step with the subpackage __all__.
_EXPORTS = {
    **dict.fromkeys(
        (
            "custom_metric",
            "metric_get_alcubierre",
            "metric_get_alcubierre_comoving",
            "metric_get_lentz",
            "metric_get_lentz_comoving",
            "metric_get_minkowski",
            "metric_get_modified_time",
            "metric_get_modified_time_comoving",
            "metric_get_schwarzschild",
            "metric_get_van_den_broeck",
            "metric_get_van_den_broeck_comoving",
            "metric_get_warp_shell_comoving",
            "three_plus_one_builder",
            "three_plus_one_decomposer",
        ),
        "metrics",
    ),
    "get_energy_tensor": "solver",
    **dict.fromkeys(
        (
            "change_tensor_index",
            "do_frame_transfer",
            "eval_metric",
            "get_energy_conditions",
            "get_momentum_flow_lines",
            "get_scalars",
        ),
        "analyser",
    ),
}

__all__ = ["__version__", *sorted(_EXPORTS)]


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    elif name in _SUBPACKAGES:
        value = importlib.import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted({*globals(), *_EXPORTS, *_SUBPACKAGES})
