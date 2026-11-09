# Contributing

Thanks for your interest. Bug reports, physics corrections and documentation
fixes are all welcome.

## Before you start

- This is a port of MATLAB [WarpFactory](https://github.com/NerdsWithAttitudes/WarpFactory).
  A difference from upstream is either a bug here or a deliberate, documented
  divergence. Read [docs/reference-validation.md](docs/reference-validation.md)
  before reporting a numerical mismatch.
- Open an issue before a large change, so the direction can be agreed first.
- Questions about the physics of WarpFactory belong with the upstream project.

## Reporting a bug

Include the smallest script that shows it, the full traceback or the wrong
numbers, your `warpica`, Python and NumPy versions, and your operating system.
For a numerical problem, say what you compared against and why you expect it.

## Setting up

```bash
git clone https://github.com/Francesco-Cini/warpica
cd warpica
python -m venv .venv
.venv/bin/python -m pip install -e ".[test,plot,dev]"
```

On Windows use `.venv\Scripts\python.exe`.

## Checks to run before opening a pull request

```bash
python -m pytest -q -W error -m "not gpu" --cov=warpica
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

All four must pass. Coverage must stay at or above the 98% floor. See
[docs/development.md](docs/development.md) and [docs/testing.md](docs/testing.md).

## Conventions

- One public function per file, MATLAB-derived names in snake_case.
- NumPy-style docstrings; public functions are fully type annotated.
- Raise `ValueError` for invalid input, never a bare `Exception` or `Warning`.
- Add or update tests with every change, and check that a new test fails when the
  behaviour it protects is broken.
- Record user-visible changes in [CHANGELOG.md](CHANGELOG.md).
- Public names are listed in [docs/api.md](docs/api.md); changes to them follow
  the [policy](docs/policy.md).
- The CUDA and DirectML backends are experimental. Hardware test reports are
  especially welcome, see [docs/gpu.md](docs/gpu.md).

## Pull requests

Branch from `develop` and target `develop`. Keep a pull request to one topic, and
describe what changed and how you checked it. By contributing you agree that your
work is released under the project's MIT licence.

Please follow the [code of conduct](CODE_OF_CONDUCT.md).
