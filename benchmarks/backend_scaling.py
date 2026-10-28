"""Measure solver scaling with synchronized, warmed CPU/device runs.

Each grid/order uses a fresh subprocess to release backend allocations between
cases. Results are checkpointed after each case. Metric construction is not
timed; the physical domain and bubble parameters stay fixed as grids refine.
"""

import argparse
import ctypes
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def available_memory():
    """Available physical host RAM, when the OS exposes it without dependencies."""
    if sys.platform == "win32":

        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in (
                    "total",
                    "available",
                    "page_total",
                    "page_available",
                    "virtual_total",
                    "virtual_available",
                    "extended",
                )
            ]

        status = Memory()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.available
    elif hasattr(os, "sysconf"):
        try:
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError):
            pass
    return None


def run_case(size, order, backend, repeats, warmups):
    from warplab.gpu import asarray, asnumpy
    from warplab.metrics.alcubierre.metric_get_alcubierre import metric_get_alcubierre
    from warplab.solver.get_energy_tensor import get_energy_tensor

    asnumpy(asarray(np.ones(1), library=backend))
    if backend == "torch-directml":
        import torch_directml

        device = torch_directml.device_name(0).rstrip("\x00")
    else:
        import cupy

        device = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    grid = np.array([1, size, size, size])
    spacing = np.array([1, *([8 / (size - 1)] * 3)])
    metric = metric_get_alcubierre(grid, (grid + 1) / 2 * spacing, 0.2, 1, 1, spacing)
    samples, outputs = {}, {}
    for name, library in (("cpu", None), ("device", backend)):
        for _ in range(warmups):
            get_energy_tensor(metric, order, gpu=library)
        samples[name] = []
        for repeat in range(repeats):
            start = time.perf_counter()
            output = np.asarray(get_energy_tensor(metric, order, gpu=library)["tensor"])
            samples[name].append(time.perf_counter() - start)
            if repeat == repeats - 1:
                outputs[name] = output
            del output
    reference, result = outputs["cpu"], outputs["device"]
    # Avoid allocating full-tensor temporaries on the largest grids.
    scale = max(float(np.max(np.abs(v))) for v in reference)
    error = max(float(np.max(np.abs(a - b))) for a, b in zip(reference, result))
    finite = bool(np.isfinite(reference).all() and np.isfinite(result).all())
    relative_error = error / scale if scale else error
    tolerance = 2e-5 if backend == "torch-directml" else 1e-10
    cpu = float(np.median(samples["cpu"]))
    gpu = float(np.median(samples["device"]))
    return {
        "size": size,
        "grid": grid.tolist(),
        "points": size**3,
        "order": order,
        "device": device,
        "cpu_seconds": cpu,
        "device_seconds": gpu,
        "cpu_over_device_speed_ratio": cpu / gpu,
        "samples_seconds": samples,
        "relative_max_error": relative_error,
        "error_tolerance": tolerance,
        "accuracy_passed": finite and relative_error <= tolerance,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cupy", "torch-directml"], required=True)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[5, 8, 16, 24, 32, 48, 64, 96]
    )
    parser.add_argument(
        "--orders",
        nargs="+",
        choices=["second", "fourth"],
        default=["second", "fourth"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--timeout", type=float, default=600, help="seconds allowed per grid/order"
    )
    parser.add_argument("--output", type=Path, default=Path("backend-scaling.json"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.sizes) < 5 or args.repeats < 1 or args.warmups < 0 or args.timeout <= 0:
        parser.error("sizes >= 5, repeats >= 1, warmups >= 0, timeout > 0 required")
    if args.worker:
        print(
            json.dumps(
                run_case(
                    args.sizes[0],
                    args.orders[0],
                    args.backend,
                    args.repeats,
                    args.warmups,
                )
            )
        )
        return
    versions = {}
    for name in (
        "numpy",
        "array-api-compat",
        "torch",
        "torch-directml",
        "cupy-cuda12x",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": versions,
        "backend": args.backend,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "domain": {
            "spatial_extent": 8,
            "time_slices": 1,
            "velocity": 0.2,
            "R": 1,
            "sigma": 1,
        },
        "timing": "median end-to-end get_energy_tensor including transfers and host conversion; excludes metric construction",
        "results": [],
        "skipped": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for size in args.sizes:
        # Conservative host working-set estimate, NOT a measured peak or VRAM bound.
        estimate = 512 * 8 * size**3 + 512 * 2**20
        free = available_memory()
        if free is not None and estimate > 0.75 * free:
            report["skipped"].append(
                {
                    "size": size,
                    "reason": "host memory guard",
                    "estimated_bytes": estimate,
                    "available_bytes": free,
                }
            )
            print(
                f"Skipping {size}^3: estimated host memory exceeds 75% of available RAM",
                flush=True,
            )
            save()
            continue
        for order in args.orders:
            print(f"Measuring {size}^3 {order}...", flush=True)
            command = [
                sys.executable,
                "-m",
                "benchmarks.backend_scaling",
                "--worker",
                "--backend",
                args.backend,
                "--sizes",
                str(size),
                "--orders",
                order,
                "--repeats",
                str(args.repeats),
                "--warmups",
                str(args.warmups),
            ]
            try:
                completed = subprocess.run(
                    command, capture_output=True, text=True, timeout=args.timeout
                )
                if completed.returncode:
                    raise RuntimeError(completed.stderr[-2000:])
                result = json.loads(completed.stdout)
                if completed.stderr.strip():
                    result["worker_stderr"] = completed.stderr.strip()
            except (
                subprocess.TimeoutExpired,
                RuntimeError,
                json.JSONDecodeError,
            ) as exc:
                report["skipped"].append(
                    {"size": size, "order": order, "reason": str(exc)}
                )
                save()
                print(f"Stopped after failed case: {exc}", flush=True)
                raise SystemExit(1)
            report["results"].append(result)
            save()
            print(
                f"  CPU {result['cpu_seconds']:.4f}s; GPU {result['device_seconds']:.4f}s; "
                f"CPU/GPU {result['cpu_over_device_speed_ratio']:.3f}; "
                f"error {result['relative_max_error']:.3g}; accuracy {result['accuracy_passed']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
