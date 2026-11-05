# Release procedure

The repository has a release candidate; these commands build and validate it.
They do not publish anything or create a Git tag automatically.

```powershell
python -m pip install -e ".[test,plot,release]"
python -m pytest -q -W error -m "not gpu"
python -m build --outdir .release-check/dist
python -m twine check .release-check/dist/*
python tools/check_wheel.py --dist .release-check/dist
```

Use a new/empty output directory for each version. `check_wheel.py` requires
exactly one wheel, installs it into a temporary clean environment, changes
outside the checkout, and checks that imports really come from site-packages.
It exercises both solvers, the analysis pipeline and plotting. The editable
checkout cannot conceal missing packaged modules.

`.github/workflows/ci.yml` runs CPU tests and committed MATLAB fixture
comparisons on Windows and Linux with Python 3.10, 3.12 and 3.14. It also
builds wheel/sdist, checks metadata and runs the installed-wheel smoke test.
These jobs will first execute remotely when the changes are pushed; adding
the workflow does not mean those remote runs have already passed.

The advisory performance guard (see [development checks](development.md)) runs
as a separate CI job and does not gate a release; look at its summary for an
unexpected slowdown before tagging.

Before publication, review the changelog, choose the version, inspect CI
results, and confirm package metadata. GPU backends are documented as
experimental, so they do not block a release; run real hardware checks
before promoting a backend to stable. Publishing and tagging are separate explicit actions.
The optional `cuda` extra targets CUDA 12 wheels; the `directml` extra pins
the tested backend and currently requires a supported Python (tested 3.12).
Install plotting separately with `.[plot]`; the core solver needs only NumPy
and array-api-compat. Source distributions also contain tests/examples/tools;
the wheel contains only the installable `warplab` library and license metadata.
