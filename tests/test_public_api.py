import importlib
import subprocess
import sys

import pytest

import warplab

CORE_PACKAGES = ["metrics", "solver", "analyser", "units"]
ALL_PACKAGES = [*CORE_PACKAGES, "visualiser"]

# Deep module paths used by existing user code; they must keep working.
LEGACY_PATHS = [
    ("warplab.metrics.minkowski.metric_get_minkowski", "metric_get_minkowski"),
    ("warplab.metrics.alcubierre.metric_get_alcubierre", "metric_get_alcubierre"),
    ("warplab.metrics.custom_metric", "custom_metric"),
    ("warplab.solver.get_energy_tensor", "get_energy_tensor"),
    ("warplab.analyser.eval_metric", "eval_metric"),
    ("warplab.analyser.change_tensor_index", "change_tensor_index"),
    ("warplab.units.universal_constants.c", "c"),
    ("warplab.units.universal_constants.G", "G"),
]


def run_python(code):
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )


@pytest.mark.parametrize("package", ALL_PACKAGES)
def test_all_names_resolve(package):
    module = importlib.import_module(f"warplab.{package}")
    assert module.__all__ == sorted(module.__all__) or package == "visualiser"
    for name in module.__all__:
        assert callable(getattr(module, name)), f"{package}.{name}"


@pytest.mark.parametrize("package", ALL_PACKAGES)
def test_package_imports_in_a_fresh_interpreter(package):
    # Guards against circular imports between subpackages, which depend on
    # which package happens to be imported first.
    result = run_python(f"import warplab.{package}")
    assert result.returncode == 0, result.stderr


def test_top_level_names_match_subpackages():
    expected = {}
    for package in ["metrics", "solver", "analyser"]:
        for name in importlib.import_module(f"warplab.{package}").__all__:
            expected[name] = package
    assert warplab._EXPORTS == expected
    assert set(warplab.__all__) == {"__version__", *expected}


def test_top_level_names_are_the_subpackage_objects():
    for name, package in warplab._EXPORTS.items():
        assert getattr(warplab, name) is getattr(
            importlib.import_module(f"warplab.{package}"), name
        )


def test_unknown_top_level_name_raises():
    with pytest.raises(AttributeError):
        warplab.no_such_function


def test_importing_warplab_is_lightweight():
    result = run_python(
        "import sys, warplab\n"
        "heavy = {'numpy', 'matplotlib', 'cupy', 'torch'} & set(sys.modules)\n"
        "print(sorted(heavy))"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


@pytest.mark.parametrize(("module", "name"), LEGACY_PATHS)
def test_legacy_deep_paths_still_work(module, name):
    assert callable(getattr(importlib.import_module(module), name))


def test_version_matches_installed_metadata():
    from importlib.metadata import version

    assert warplab.__version__ == version("warplab")


def test_constants():
    from warplab.units import G, c

    assert c() == 2.99792458e8
    assert G() == 6.6743e-11
