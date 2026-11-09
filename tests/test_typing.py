"""The public API is annotated, ships its types, and checks correctly for users."""

import importlib
import inspect
import os
import subprocess
import sys
import typing
from pathlib import Path

import pytest

import warpica

PUBLIC_PACKAGES = ["metrics", "solver", "analyser", "units", "visualiser", "gpu"]
PACKAGE_DIRECTORY = Path(warpica.__file__).parent


def public_functions():
    for package in PUBLIC_PACKAGES:
        if package == "visualiser":
            pytest.importorskip("matplotlib")
        module = importlib.import_module(f"warpica.{package}")
        for name in module.__all__:
            yield pytest.param(getattr(module, name), id=f"{package}.{name}")


@pytest.mark.parametrize("function", list(public_functions()))
def test_every_public_function_is_fully_annotated(function):
    signature = inspect.signature(function)
    missing = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.annotation is inspect.Parameter.empty
    ]
    assert not missing, f"unannotated parameters: {missing}"
    assert signature.return_annotation is not inspect.Signature.empty


@pytest.mark.parametrize("function", list(public_functions()))
def test_annotations_resolve_to_real_types(function):
    """A typo or a missing import in an annotation would fail here at runtime."""
    hints = typing.get_type_hints(function)
    assert "return" in hints


def test_the_package_ships_the_typed_marker():
    assert (PACKAGE_DIRECTORY / "py.typed").is_file()


GOOD = """
import numpy as np
from warpica.analyser import eval_metric, get_energy_conditions, get_scalars
from warpica.metrics import metric_get_alcubierre, metric_get_minkowski
from warpica.solver import get_energy_tensor

metric = metric_get_minkowski([1, 5, 5, 5])
bubble = metric_get_alcubierre(
    np.array([1, 5, 5, 5]), [1.0, 3.0, 3.0, 3.0], 0.2, 1.0, 1.0, None
)
energy = get_energy_tensor(metric, "second", gpu="numpy")
expansion, shear, vorticity = get_scalars(metric, "fourth")
result = eval_metric(metric, diff_order="second")
conditions = get_energy_conditions(energy, metric, "null", 4, 2)
with_samples = get_energy_conditions(energy, metric, "Null", 4, 2, True)
values, samples, vectors = with_samples
reveal_type(conditions)
reveal_type(with_samples)
reveal_type(energy)
"""

BAD = """
from warpica.metrics import metric_get_minkowski
from warpica.solver import get_energy_tensor

metric = metric_get_minkowski([1, 5, 5, 5])
get_energy_tensor(metric, "sixth")
get_energy_tensor(metric, gpu="opencl")
get_energy_tensor("not a tensor")
metric_get_minkowski("5 by 5")
"""


def run_mypy(tmp_path, **sources):
    files = []
    for name, text in sources.items():
        path = tmp_path / f"{name}.py"
        path.write_text(text)
        files.append(str(path))
    repository = PACKAGE_DIRECTORY.parent
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(repository / "pyproject.toml"),
            "--cache-dir",
            str(tmp_path / "cache"),
            *files,
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "MYPYPATH": str(repository)},
    )
    return result.stdout


def test_user_code_is_checked_against_the_signatures(tmp_path):
    pytest.importorskip("mypy")
    output = run_mypy(tmp_path, good=GOOD, bad=BAD)
    good = [line for line in output.splitlines() if line.startswith("good.py")]
    bad = [line for line in output.splitlines() if line.startswith("bad.py")]
    # Correct code type-checks; only the reveal_type notes appear for it.
    assert all("note: Revealed type" in line for line in good), good
    # The overloads give each form of get_energy_conditions its own return type.
    revealed = [line.split('"')[1] for line in good]
    assert revealed[0].startswith("numpy.ndarray"), revealed
    assert revealed[1].startswith("tuple[numpy.ndarray"), revealed
    assert revealed[2] == "dict[str, Any]", revealed
    # Each mistake is reported on its own line.
    reported = {int(line.split(":")[1]) for line in bad if "error:" in line}
    assert reported == {6, 7, 8, 9}, bad
