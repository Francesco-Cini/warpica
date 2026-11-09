## What and why

<!-- What changes, and why. Link the issue if there is one. -->

## How I checked it

- [ ] `python -m pytest -q -W error -m "not gpu" --cov=warpica`
- [ ] `python -m ruff check .` and `python -m ruff format --check .`
- [ ] `python -m mypy`
- [ ] New or changed behaviour has a test that fails without the change
- [ ] `CHANGELOG.md` updated for user-visible changes
- [ ] Docs updated if public behaviour changed
