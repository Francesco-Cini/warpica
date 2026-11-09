"""Plot a backend_scaling report; requires the optional matplotlib dependency."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("backend-scaling.png"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if not report["results"]:
        parser.error("report has no completed cases to plot")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), layout="constrained")
    colors = {"cpu": "#2463ad", "device": "#c05621"}
    for ax, order in zip(axes, ("second", "fourth")):
        rows = sorted(
            (r for r in report["results"] if r["order"] == order),
            key=lambda r: r["size"],
        )
        sizes = [r["size"] for r in rows]
        gpu_label = (
            "GPU · float32 curvature"
            if report["backend"] == "torch-directml"
            else "GPU · float64"
        )
        for backend, label, marker in (
            ("cpu", "CPU · float64", "o"),
            ("device", gpu_label, "s"),
        ):
            medians = np.array([r[backend + "_seconds"] for r in rows])
            lows = np.array([min(r["samples_seconds"][backend]) for r in rows])
            highs = np.array([max(r["samples_seconds"][backend]) for r in rows])
            ax.errorbar(
                sizes,
                medians,
                yerr=[medians - lows, highs - medians],
                color=colors[backend],
                marker=marker,
                label=label,
                capsize=3,
                lw=2,
            )
        failures = [r for r in rows if not r["accuracy_passed"]]
        if failures:
            ax.scatter(
                [r["size"] for r in failures],
                [r["device_seconds"] for r in failures],
                marker="x",
                s=110,
                linewidths=2,
                color="#a31621",
                zorder=5,
                label="GPU error exceeds 2e−5"
                if report["backend"] == "torch-directml"
                else "GPU error exceeds tolerance",
            )
        ax.set(
            xscale="log",
            yscale="log",
            title=f"{order.capitalize()} order",
            xlabel="Spatial side length n (total points = n³)",
            ylabel="End-to-end solver time (seconds)",
        )
        ticks = [
            n
            for n in (5, 8, 16, 32, 48, 64, 80, 96, 128)
            if sizes and min(sizes) <= n <= max(sizes)
        ]
        ax.set_xticks(ticks, [str(n) for n in ticks])
        ax.minorticks_off()
        ax.grid(True, which="major", alpha=0.2)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    device = report["results"][0]["device"] if report["results"] else report["backend"]
    fig.suptitle(f"warpica scaling · {device} / {report['backend']}", fontsize=16)
    counts = sorted({len(r["samples_seconds"]["cpu"]) for r in report["results"]})
    repetitions = (
        "–".join(map(str, (counts[0], counts[-1])))
        if len(counts) > 1
        else str(counts[0])
    )
    fig.supxlabel(
        f"Fixed physical domain; one time slice; {report['warmups']} warmup + "
        f"{repetitions} timed runs. Points: medians; bars: min–max. Transfers included.",
        fontsize=10,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
