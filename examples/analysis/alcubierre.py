"""Run from the repository root: python -m examples.analysis.alcubierre"""

import numpy as np

from warplab.analyser.eval_metric import eval_metric
from warplab.metrics.alcubierre.metric_get_alcubierre import metric_get_alcubierre


def main():
    # Grid axes are (time, x, y, z). A single time sample has no time derivatives.
    grid = np.array([1, 9, 9, 9])
    spacing = np.array([1.0, 0.5, 0.5, 0.5])
    centre = (grid + 1) / 2 * spacing
    metric = metric_get_alcubierre(
        grid, centre, v=0.2, R=1, sigma=2, grid_scale=spacing
    )
    result = eval_metric(
        metric, num_angular_vec=24, num_time_vec=4, diff_order="fourth"
    )
    for name in (
        "null",
        "weak",
        "strong",
        "dominant",
        "expansion",
        "shear",
        "vorticity",
    ):
        values = result[name]
        print(f"{name}: min={values.min():.6g}, max={values.max():.6g}")
    print(
        "Negative energy-condition values indicate violations among the sampled directions."
    )


if __name__ == "__main__":
    main()
