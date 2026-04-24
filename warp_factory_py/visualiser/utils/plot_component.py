import numpy as np
import matplotlib.pyplot as plt

def plot_component(array, title_text, x_label_text, y_label_text, alpha = 0.2):

    array = np.array(array)

    fig = plt.figure()
    fig.patch.set_facecolor("white")

    ax = fig.add_subplot(111, projection="3d")

    x = np.arange(1, array.shape[0] + 1)
    y = np.arange(1, array.shape[1] + 1)

    X, Y = np.meshgrid(x, y, indexing="ij")

    surf = ax.plot_surface(X, Y, array, alpha = 1.0, linewidth = 0, antialiased = True)

    surf.set_alpha(1)

    ax.set_title(title_text)
    ax.set_xlabel(x_label_text)
    ax.set_ylabel(y_label_text)

    plt.set_cmap("seismic")

    ax.view_init(elev=90, azim=-90)

    ax.set_xlim(1, array.shape[0])
    ax.set_ylim(1, array.shape[1])

    plt.show()

    return plot_component