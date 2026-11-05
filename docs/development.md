# Code style and development checks

Install the development tools alongside the test and plotting dependencies:

```powershell
python -m pip install -e ".[dev,test,plot]"
python -m ruff check .
python -m ruff format --check .
python -m pytest -q -W error -m "not gpu"
```

## Coverage

```powershell
python -m pytest -q -W error -m "not gpu" --cov=warplab --cov-report=term-missing
```

CI fails when line coverage drops below 98% (`fail_under` in `pyproject.toml`);
the suite currently measures 100%. Only the CUDA and DirectML adapters are
excluded, because they need a physical device that hosted runners do not have:
`warplab/gpu/utils/` is omitted and the device-only `if` branches are listed in
`exclude_also`. Add tests for new code instead of widening the exclusions, and
raise the floor when coverage rises. Run the tests with the `plot` extra
installed, or the plotting modules count as uncovered.

Run `python -m ruff check . --fix` and `python -m ruff format .` to apply
import ordering and formatting fixes. Review the diff before committing.
Ruff's configuration lives in `pyproject.toml`: four-space indentation,
an 88-character formatting target, grouped imports, and NumPy-style docstrings.
Public mathematical names and compatibility aliases retain their existing
spelling so that keyword arguments and imports remain compatible.

Document public functions with a short imperative summary, then `Parameters`
and `Returns` sections when the interface needs explanation. Include array
shapes, axis order, units, index conventions, defaults, and meaningful
limitations. Use `Notes` for numerical conventions and backend differences.
Small, self-explanatory helpers can use a one-line docstring.

Comments should explain a reason that the code cannot express directly, such
as a boundary convention, symmetry assumption, or conversion between time and
length coordinates. Avoid narrating assignments or duplicating docstrings.

Tensor components are indexed `[mu][nu]`, and each field has axes
`(t, x, y, z)`. Solver differentiation axes are zero-based. Plot slicing and
flow-line coordinates preserve the upstream one-based public conventions;
document that distinction when extending those interfaces.

For readability changes, preserve the order of floating-point operations and
run the existing numerical regression tests. GPU tests require an explicitly
selected hardware backend; see [GPU testing](gpu.md).
