"""Measure CPU/device solver time and error, including transfers; no speedup assumed."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from warplab.gpu import asarray, asnumpy
from warplab.metrics.alcubierre.metric_get_alcubierre import metric_get_alcubierre
from warplab.solver.get_energy_tensor import get_energy_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cupy", "torch-directml"], required=True)
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="untimed full solver runs per backend and order",
    )
    parser.add_argument("--output", type=Path, default=Path("backend-results.json"))
    args = parser.parse_args()
    if args.size < 5 or args.repeats < 1:
        parser.error("size >= 5 and repeats >= 1 required")
    if args.warmups < 0:
        parser.error("warmups >= 0 required")
    # Force initialization before timing. Gathering synchronizes each measured run.
    asnumpy(asarray(np.ones(1), library=args.backend))
    grid = np.array([1, args.size, args.size, args.size])
    metric = metric_get_alcubierre(grid, (grid + 1) / 2, 0.2, 1, 1)
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "backend": args.backend,
        "grid": grid.tolist(),
        "repeats": args.repeats,
        "warmups": args.warmups,
        "results": [],
    }
    if args.backend == "torch-directml":
        import torch_directml

        report["device"] = torch_directml.device_name(0).rstrip("\x00")
    else:
        import cupy

        report["device"] = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    for order in ("second", "fourth"):
        outputs = {}
        timings = {}
        for backend in (None, args.backend):
            for _ in range(args.warmups):
                get_energy_tensor(metric, order, gpu=backend)
            durations = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                outputs[backend] = np.asarray(
                    get_energy_tensor(metric, order, gpu=backend)["tensor"]
                )
                durations.append(time.perf_counter() - start)
            timings[str(backend)] = float(np.median(durations))
        reference = outputs[None]
        result = outputs[args.backend]
        report["results"].append(
            {
                "order": order,
                "cpu_seconds": timings["None"],
                "device_seconds": timings[args.backend],
                "cpu_over_device_speed_ratio": timings["None"] / timings[args.backend],
                "relative_max_error": float(
                    np.max(abs(result - reference)) / np.max(abs(reference))
                ),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
