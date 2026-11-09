# Code style and development checks

Install the development tools alongside the test and plotting dependencies:

```powershell
python -m pip install -e ".[dev,test,plot]"
python -m ruff check .
python -m ruff format --check .
python -m pytest -q -W error -m "not gpu"
```

## Type checking

```powershell
python -m pip install -e ".[dev,test,plot]"
python -m mypy
python -m pytest -q tests/test_typing.py
```

`mypy` reads its settings from `pyproject.toml` and checks the functions that
carry annotations, which includes the whole public API. `tests/test_typing.py`
fails if a public function loses an annotation, if an annotation names a type
that does not exist, if `py.typed` goes missing, and it runs mypy over small
user programs to confirm that correct calls pass and wrong ones are reported.
Run `mypy --check-untyped-defs` to see the roughly 120 errors still hiding in
unannotated internals before deciding how far to extend the annotations.

## Coverage

```powershell
python -m pytest -q -W error -m "not gpu" --cov=warpica --cov-report=term-missing
```

CI fails when line coverage drops below 98% (`fail_under` in `pyproject.toml`);
the suite currently measures 100%. Only the CUDA and DirectML adapters are
excluded, because they need a physical device that hosted runners do not have:
`warpica/gpu/utils/` is omitted and the device-only `if` branches are listed in
`exclude_also`. Add tests for new code instead of widening the exclusions, and
raise the floor when coverage rises. Run the tests with the `plot` extra
installed, or the plotting modules count as uncovered.

## Performance guard (advisory)

```powershell
python -m pytest tests/performance --run-performance -m performance -q
```

The guard times the stress-energy solver, `get_energy_conditions` and
`get_scalars` on two grid sizes and fails on either of two machine-independent
signals, so it works on any CPU:

- **Scaling.** The cost may grow no faster than `cells**1.15`. Healthy code
  measures 0.6 to 0.9, because fixed Python overhead dominates the small grid.
- **Calibrated ratio.** The cost relative to a fixed NumPy kernel timed in the
  same process may not exceed three times the value in
  `tests/performance/baseline.json`.

It is deliberately coarse. Injecting known slowdowns showed it flags a per-cell
loop that makes an operation about 3.1 times slower overall, and quadratic work
that equals the operation's own runtime at the largest grid. A 20% regression,
or a smaller quadratic term, goes unnoticed. It does not measure GPU speed;
use the scripts in `benchmarks/` on the hardware you care about.

Normal test runs skip the timing tests, but they do run `test_guard_machinery.py`,
which checks the verdict logic, the baseline file and that every guarded
operation still executes. In CI the guard is its own job. It is **advisory**: it
never blocks a merge, it reports in the job summary and as a warning
annotation, and it must not be added to the required status checks. Rerun it
before treating one failure as a real regression.

The baseline was recorded on Windows with Python 3.14, and hosted runners differ.
After a few CI runs, compare the "Ratio to reference" column in the job summary
with `baseline.json`, and refresh it with
`python -m tests.performance.timing --write-baseline` (or edit the ratios by
hand) so the tolerance is centred on the runner.

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

## Documentation site

The site is built with MkDocs (Material theme) and mkdocstrings, which turns the
docstrings into the API reference. Preview it locally:

```bash
python -m pip install -e ".[docs,plot]"
mkdocs serve
```

CI runs `mkdocs build --strict`, so a broken link or a docstring that does not
resolve fails the build. Pages outside `docs/` (the changelog, `CONTRIBUTING.md`)
are linked by their GitHub URL, because MkDocs only serves files in `docs/`.
New public functions need a line in `docs/reference.md`.
