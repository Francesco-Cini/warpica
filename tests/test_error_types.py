"""The package raises specific exception types, never bare ``Exception``."""

import ast
from pathlib import Path

import pytest

import warplab

PACKAGE = Path(warplab.__file__).parent
FORBIDDEN = {"Exception", "BaseException", "Warning"}


def raised_names(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        if isinstance(target, ast.Name):
            yield target.id, node.lineno


@pytest.mark.parametrize(
    "path",
    sorted(PACKAGE.rglob("*.py")),
    ids=lambda p: p.relative_to(PACKAGE).as_posix(),
)
def test_no_module_raises_a_generic_exception(path):
    offending = [
        f"line {line}: raise {name}"
        for name, line in raised_names(path)
        if name in FORBIDDEN
    ]
    assert not offending, offending


def test_the_checker_finds_a_generic_raise(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        "def f():\n    raise Exception('x')\n\ndef g():\n    raise ValueError\n"
    )
    assert list(raised_names(sample)) == [("Exception", 2), ("ValueError", 5)]
