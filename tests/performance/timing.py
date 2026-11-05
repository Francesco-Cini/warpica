"""Timing helpers for the advisory performance guard.

Absolute times depend on the machine, so the guard only uses quantities that
cancel its speed: how the cost grows with the grid, and the cost relative to a
fixed NumPy workload timed in the same process. Run this module to refresh the
recorded baseline or to render a report::

    python -m tests.performance.timing --write-baseline
    python -m tests.performance.timing --summary performance-report.json
"""

import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

from warplab.analyser import do_frame_transfer, get_energy_conditions, get_scalars
from warplab.metrics import metric_get_alcubierre
from warplab.solver import get_energy_tensor

BASELINE_FILE = Path(__file__).with_name("baseline.json")
REPEATS = 5

# The cost may grow no faster than cells**MAX_EXPONENT. Healthy exponents are
# 0.6 to 0.9, below 1 because fixed Python overhead dominates the smaller grid.
# Injecting extra quadratic work showed that 1.15 flags it once it equals the
# solver's own largest-grid runtime (exponent 1.19), while 1.3 missed that case.
MAX_EXPONENT = 1.15
# An operation may cost this many times its recorded ratio to the reference.
# Injected per-cell work was caught from about a 3.1x overall slowdown.
TOLERANCE = 3.0


def best_time(function, repeats=REPEATS):
    """Return the fastest of several runs after one warm-up call.

    The first call pays for imports, caches and allocator growth, and the
    fastest run is the least disturbed by other work on the machine.
    """
    function()
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def reference_workload():
    """A fixed NumPy kernel shaped like the solver: many slicing stencils."""
    field = np.linspace(0.0, 1.0, 19**3).reshape(19, 19, 19)
    out = np.zeros_like(field)
    for _ in range(400):
        out[2:-2] = (-(field[4:] - field[:-4]) + 8 * (field[3:-1] - field[1:-3])) / 12.0
    return out


def bubble(n):
    """An Alcubierre bubble on an n by n by n grid with one time slice."""
    centre = np.array([1, n / 2 + 0.3, n / 2 + 0.2, n / 2 + 0.1])
    return metric_get_alcubierre(
        np.array([1, n, n, n]), centre, 0.2, n / 6, 1, np.ones(4)
    )


def prepare_energy_tensor(n):
    metric = bubble(n)
    return lambda: get_energy_tensor(metric, "fourth")


def prepare_scalars(n):
    metric = bubble(n)
    return lambda: get_scalars(metric)


def prepare_energy_conditions(n):
    metric = bubble(n)
    energy = do_frame_transfer(metric, get_energy_tensor(metric), "Eulerian")
    return lambda: get_energy_conditions(energy, metric, "Weak", 20, 4)


# name: (prepare function, small grid size, large grid size)
# The sizes are large enough that the cost per cell, not fixed Python overhead,
# starts to show: measured exponents are about 0.6 to 0.9.
OPERATIONS = {
    "energy_tensor": (prepare_energy_tensor, 19, 35),
    "scalars": (prepare_scalars, 13, 27),
    "energy_conditions": (prepare_energy_conditions, 19, 35),
}


def exponent_of(small_seconds, large_seconds, small_grid, large_grid):
    """Return p in time ~ cells**p between two cubic grids."""
    return math.log(large_seconds / small_seconds) / math.log(
        large_grid**3 / small_grid**3
    )


def problems(result, baseline_ratio):
    """List the ways a measurement breaks the guard; empty means it passes."""
    found = []
    name = result["operation"]
    if result["exponent"] > MAX_EXPONENT:
        found.append(
            f"{name}: cost grows as cells**{result['exponent']:.2f} "
            f"(limit {MAX_EXPONENT}); {result['small_seconds'] * 1000:.0f} ms at "
            f"{result['small_grid']}^3 but {result['large_seconds'] * 1000:.0f} ms "
            f"at {result['large_grid']}^3"
        )
    limit = TOLERANCE * baseline_ratio
    if result["ratio"] > limit:
        found.append(
            f"{name}: costs {result['ratio']:.1f}x the reference kernel, above "
            f"{limit:.1f} ({TOLERANCE:g}x the recorded {baseline_ratio:.1f})"
        )
    return found


def measure(name):
    """Time one operation on two grids and relative to the reference kernel."""
    prepare, small, large = OPERATIONS[name]
    small_seconds = best_time(prepare(small))
    large_seconds = best_time(prepare(large))
    reference_seconds = best_time(reference_workload)
    exponent = exponent_of(small_seconds, large_seconds, small, large)
    return {
        "operation": name,
        "small_grid": small,
        "large_grid": large,
        "small_seconds": small_seconds,
        "large_seconds": large_seconds,
        "reference_seconds": reference_seconds,
        "exponent": exponent,
        "ratio": large_seconds / reference_seconds,
    }


def load_baseline():
    return json.loads(BASELINE_FILE.read_text())["operations"]


def write_baseline():
    results = {name: measure(name) for name in OPERATIONS}
    baseline = {
        "note": (
            "Cost of each operation on its large grid relative to "
            "reference_workload, measured on the machine below. Refresh with "
            "python -m tests.performance.timing --write-baseline"
        ),
        "machine": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "operations": {
            name: {"ratio": round(result["ratio"], 4)}
            for name, result in results.items()
        },
    }
    BASELINE_FILE.write_text(json.dumps(baseline, indent=2) + "\n")
    return results


def summary(report_file):
    """Render a recorded report as a Markdown table."""
    report = json.loads(Path(report_file).read_text())
    try:
        baseline = load_baseline()
    except FileNotFoundError:
        baseline = {}
    lines = [
        "### Performance guard (advisory)",
        "",
        "| Operation | Grids | Small / large (ms) | Exponent (limit "
        f"{MAX_EXPONENT}) | Ratio to reference | Baseline x {TOLERANCE:g} |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, result in report.items():
        limit = baseline.get(name, {}).get("ratio")
        lines.append(
            f"| {name} | {result['small_grid']}^3 / {result['large_grid']}^3 "
            f"| {result['small_seconds'] * 1000:.1f} / "
            f"{result['large_seconds'] * 1000:.1f} "
            f"| {result['exponent']:.2f} | {result['ratio']:.2f} "
            f"| {'-' if limit is None else f'{limit * TOLERANCE:.2f}'} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    if "--write-baseline" in sys.argv:
        for name, result in write_baseline().items():
            print(
                f"{name}: exponent {result['exponent']:.2f}, "
                f"ratio {result['ratio']:.3f}"
            )
    elif "--summary" in sys.argv:
        print(summary(sys.argv[sys.argv.index("--summary") + 1]))
    else:
        sys.exit(__doc__)
