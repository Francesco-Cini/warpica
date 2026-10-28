"""python -m examples.visualisation.tensor_slices --output-dir figures"""

import argparse
from pathlib import Path

import numpy as np

from warplab.metrics.alcubierre.metric_get_alcubierre import metric_get_alcubierre
from warplab.visualiser import plot_tensor, plot_three_plus_one


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid = np.array([1, 17, 17, 9])
    scale = np.array([1, 0.5, 0.5, 0.5])
    metric = metric_get_alcubierre(grid, (grid + 1) / 2 * scale, 0.4, 2, 1, scale)
    for name, plot in (
        ("metric", plot_tensor),
        ("three_plus_one", plot_three_plus_one),
    ):
        figure, _ = plot(metric, sliced_planes=[1, 4])
        path = args.output_dir / (name + ".png")
        figure.savefig(path, dpi=150)
        print(path.resolve())


if __name__ == "__main__":
    main()
