import numpy as np
import matplotlib.pyplot as plt

from warp_factory_py.metrics.alcubierre.metric_get_alcubierre import metric_get_alcubierre
from warp_factory_py.solver.get_energy_tensor import get_energy_tensor
from warp_factory_py.analyser.get_energy_conditions import get_energy_conditions
from warp_factory_py.analyser.get_scalars import get_scalars
from warp_factory_py.analyser.get_momentum_flow_lines import get_momentum_flow_lines
from warp_factory_py.analyser.change_tensor_index import change_tensor_index
from warp_factory_py.visualiser.utils.surf_q_modified import sqm


grid_size = np.array([5, 20, 20, 20])
world_centre = (grid_size + 1) / 2

velocity = 0.5
R = 5
sigma = 0.5

metric = metric_get_alcubierre(grid_size, world_centre, velocity, R, sigma, None)
energy_tensor = get_energy_tensor(metric, "fourth")

print(type(energy_tensor))
print(energy_tensor.keys())
print(type(energy_tensor["tensor"]))
print(np.shape(energy_tensor["tensor"]))
print(type(energy_tensor["tensor"][0][0]))
print(np.shape(energy_tensor["tensor"][0][0]))

null_energy_condition = get_energy_conditions(energy_tensor, metric, "Null", None, None, None)
weak_energy_condition = get_energy_conditions(energy_tensor, metric, "Weak", None, None, None)
strong_energy_condition = get_energy_conditions(energy_tensor, metric, "Strong", None, None, None)
dominant_energy_condition = get_energy_conditions(energy_tensor, metric, "Dominant", None, None, None)

expansion_scalar, shear_scalar, vorticity_scalar = get_scalars(metric)

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
fig.patch.set_facecolor("#0e1117")
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


# ---------- Plot Energy Conditions ----------
fig, axs = plt.subplots(
    2, 2,
    figsize=(12, 10),
    subplot_kw={"projection": "3d"}
)
fig.patch.set_facecolor("#0e1117")
fig.suptitle(metric["name"] + " Energy Conditions", fontsize=18, y=0.97, color="white")

energy_conditions = [
    (null_energy_condition, "Null Energy Condition"),
    (weak_energy_condition, "Weak Energy Condition"),
    (strong_energy_condition, "Strong Energy Condition"),
    (dominant_energy_condition, "Dominant Energy Condition"),
]

for ax, condition_data in zip(axs.flat, energy_conditions):
    condition = condition_data[0]
    condition_name = condition_data[1]

    sqm(
        condition[2, :, :, slice_idx],
        ax=ax,
        dark=True,
        cmap="magma",
        colorbar=False,
        title=condition_name,
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


# ---------- Plot Metric Scalars ----------
fig, axs = plt.subplots(
    1, 3,
    figsize=(16, 5),
    subplot_kw={"projection": "3d"}
)
fig.patch.set_facecolor("#0e1117")
fig.suptitle(metric["name"] + " Metric Scalars", fontsize=18, y=0.97, color="white")

metric_scalars = [
    (expansion_scalar, "Expansion Scalar"),
    (shear_scalar, "Shear Scalar"),
    (vorticity_scalar, "Vorticity Scalar"),
]

for ax, scalar_data in zip(axs.flat, metric_scalars):
    scalar = scalar_data[0]
    scalar_name = scalar_data[1]

    sqm(
        scalar[2, :, :, slice_idx],
        ax=ax,
        dark=True,
        cmap="magma",
        colorbar=False,
        title=scalar_name,
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

fig.subplots_adjust(wspace=0.18, hspace=0.28, top=0.85)


# ---------- Momentum Flow Lines ----------
ngridsteps = 4
flow_step_size = 0.75
flow_max_steps = 10000

energy_tensor_contravariant = change_tensor_index(energy_tensor.copy(), "contravariant", metric.copy())

for i in range(4):
    for j in range(4):
        energy_tensor_contravariant["tensor"][i][j] = energy_tensor_contravariant["tensor"][i][j][2, :, :, :]

flow_scale_factor = 1 / np.max(np.abs(energy_tensor_contravariant["tensor"][0][1]))

x_vals = np.arange(1, grid_size[1] + 1, ngridsteps)
y_vals = np.arange(1, grid_size[2] + 1, ngridsteps)
z_vals = np.arange(1, grid_size[3] + 1, ngridsteps)

X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing="xy")

start_points = [X, Y, Z]

paths = get_momentum_flow_lines(
    energy_tensor_contravariant,
    start_points,
    flow_step_size,
    flow_max_steps,
    flow_scale_factor
)

fig = plt.figure(figsize=(10, 8))
fig.patch.set_facecolor("#0e1117")

ax = fig.add_subplot(111, projection="3d")
ax.set_title(metric["name"] + " Momentum Flow Lines", color="white")

path_length = 30
start_idx = 10

for path in paths:
    if path is not None and path.shape[0] > start_idx + path_length:
        ax.plot(
            path[start_idx:start_idx + path_length, 0],
            path[start_idx:start_idx + path_length, 1],
            path[start_idx:start_idx + path_length, 2],
        )

ax.view_init(elev=45, azim=45)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.show()
