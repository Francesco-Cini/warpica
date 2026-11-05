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


@pytest.mark.parametrize("name", ["analyser", "gpu", "metrics", "solver", "units"])
def test_subpackages_are_loaded_on_first_access(name):
    result = run_python(
        "import sys, warplab\n"
        f"assert 'warplab.{name}' not in sys.modules\n"
        f"module = warplab.{name}\n"
        f"assert module is sys.modules['warplab.{name}']\n"
        f"assert warplab.{name} is module\n"
    )
    assert result.returncode == 0, result.stderr


def test_dir_lists_the_public_names_before_they_are_loaded():
    result = run_python(
        "import warplab\n"
        "names = dir(warplab)\n"
        "assert 'metric_get_alcubierre' in names and 'get_energy_tensor' in names\n"
        "assert 'units' in names and 'visualiser' in names and '__version__' in names\n"
        "assert names == sorted(names)\n"
    )
    assert result.returncode == 0, result.stderr


def test_a_loaded_name_is_cached_on_the_package():
    result = run_python(
        "import warplab\n"
        "first = warplab.eval_metric\n"
        "assert 'eval_metric' in vars(warplab)\n"
        "assert warplab.eval_metric is first\n"
    )
    assert result.returncode == 0, result.stderr


def test_a_subpackage_is_resolved_lazily_when_it_is_not_cached(monkeypatch):
    """Drop the cached attribute so module-level __getattr__ has to import it."""
    monkeypatch.delattr(warplab, "units", raising=False)
    assert "units" not in vars(warplab)
    module = warplab.units
    assert module.__name__ == "warplab.units"
    assert vars(warplab)["units"] is module


def test_dir_includes_the_lazy_names():
    names = dir(warplab)
    assert names == sorted(names)
    for name in (*warplab._EXPORTS, *warplab._SUBPACKAGES, "__version__"):
        assert name in names
