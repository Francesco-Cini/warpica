"""Advisory performance guard: opt in with ``--run-performance``.

These tests time real work, so they are skipped in normal runs and run in their
own CI job that reports problems without blocking a merge. They compare only
machine-independent quantities (see ``timing``). Set ``PERFORMANCE_REPORT`` to a
file name to keep the measurements as JSON.
"""

import json
import os
from pathlib import Path

import pytest

from .timing import OPERATIONS, load_baseline, measure, problems

pytestmark = pytest.mark.performance


@pytest.fixture(scope="module")
def measurements(performance_enabled):
    results = {}
    yield results
    destination = os.environ.get("PERFORMANCE_REPORT")
    if destination and results:
        Path(destination).write_text(json.dumps(results, indent=2) + "\n")


def result_for(measurements, name):
    if name not in measurements:
        measurements[name] = measure(name)
    return measurements[name]


@pytest.mark.parametrize("name", list(OPERATIONS))
def test_cost_does_not_outgrow_the_grid(measurements, name):
    result = result_for(measurements, name)
    found = [text for text in problems(result, float("inf")) if "grows" in text]
    assert not found, found


@pytest.mark.parametrize("name", list(OPERATIONS))
def test_cost_stays_near_the_recorded_baseline(measurements, name):
    result = result_for(measurements, name)
    found = [
        text
        for text in problems(result, load_baseline()[name]["ratio"])
        if "reference kernel" in text
    ]
    assert not found, found
