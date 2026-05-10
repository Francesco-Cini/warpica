import numpy as np
import matplotlib.pyplot as plt

from warp_factory_py.metrics.alcubierre.metric_get_alcubierre import metric_get_alcubierre
from warp_factory_py.solver.get_energy_tensor import get_energy_tensor
from warp_factory_py.visualiser.utils.surf_q_modified import sqm

grid_size = np.array([5, 25, 25, 25])
world_centre = (grid_size + 1) / 2
velocity = 0.5
R = 5
sigma = 0.5

metric = metric_get_alcubierre(grid_size, world_centre, velocity, R, sigma)
energy_tensor = get_energy_tensor(metric, "fourth")

slice_idx = int(np.round(world_centre[3])) - 1

# ---------- Plot Metric ----------
fig, axs = plt.subplots(
    4, 4,
    figsize=(16, 14),
    subplot_kw={"projection": "3d"}
)
fig.patch.set_facecolor("#0e1117")
fig.suptitle(metric["name"], fontsize=18, y=0.97, color="white")

for i in range(4):
    for j in range(4):
        ax = axs[i, j]
        sqm(
            metric["tensor"][i][j][2, :, :, slice_idx],
            ax=ax,
            dark=True,
            cmap="magma",
            colorbar=False,
            title=f"T[{i+1},{j+1}]",
            xlabel="x",
            ylabel="y",
            zlabel="E",
            title_pad=20,
            xlabel_pad=2,
            ylabel_pad=2,
            zlabel_pad=4,
            title_fontsize=10,
            label_fontsize=8,
            tick_fontsize=7,
            tick_pad=1,
            box_aspect=(1, 1, 0.5),
        )

fig.subplots_adjust(wspace=0.18, hspace=0.28, top=0.90)


# ---------- Plot Energy Tensor ----------
fig, axs = plt.subplots(
    4, 4,
    figsize=(16, 14),
    subplot_kw={"projection": "3d"}
)
fig.suptitle(metric["name"] + " Energy Tensor", fontsize=18, y=0.97, color="white")

for i in range(4):
    for j in range(4):
        ax = axs[i, j]
        sqm(
            energy_tensor["tensor"][i][j][2, :, :, slice_idx],
            ax=ax,
            dark=True,
            cmap="magma",
            colorbar=False,
            title=f"T[{i+1},{j+1}]",
            xlabel="x",
            ylabel="y",
            zlabel="E",
            title_pad=20,
            xlabel_pad=2,
            ylabel_pad=2,
            zlabel_pad=4,
            title_fontsize=10,
            label_fontsize=8,
            tick_fontsize=7,
            tick_pad=1,
            box_aspect=(1, 1, 0.5),
        )

fig.subplots_adjust(wspace=0.18, hspace=0.28, top=0.90)
plt.show()