import numpy as np


def trilinear_interp(f, x):

    x = np.asarray(x) + 10 ** (-8)

    x_d = (x[0] - np.floor(x[0])) / (np.ceil(x[0]) - np.floor(x[0]))
    y_d = (x[1] - np.floor(x[1])) / (np.ceil(x[1]) - np.floor(x[1]))
    z_d = (x[2] - np.floor(x[2])) / (np.ceil(x[2]) - np.floor(x[2]))

    c_00 = (
        f[int(np.floor(x[0])) - 1, int(np.floor(x[1])) - 1, int(np.floor(x[2])) - 1] * (1 - x_d)
        + f[int(np.ceil(x[0])) - 1, int(np.floor(x[1])) - 1, int(np.floor(x[2])) - 1] * x_d
    )

    c_01 = (
        f[int(np.floor(x[0])) - 1, int(np.floor(x[1])) - 1, int(np.ceil(x[2])) - 1] * (1 - x_d)
        + f[int(np.ceil(x[0])) - 1, int(np.floor(x[1])) - 1, int(np.ceil(x[2])) - 1] * x_d
    )

    c_10 = (
        f[int(np.floor(x[0])) - 1, int(np.ceil(x[1])) - 1, int(np.floor(x[2])) - 1] * (1 - x_d)
        + f[int(np.ceil(x[0])) - 1, int(np.ceil(x[1])) - 1, int(np.floor(x[2])) - 1] * x_d
    )

    c_11 = (
        f[int(np.floor(x[0])) - 1, int(np.ceil(x[1])) - 1, int(np.ceil(x[2])) - 1] * (1 - x_d)
        + f[int(np.ceil(x[0])) - 1, int(np.ceil(x[1])) - 1, int(np.ceil(x[2])) - 1] * x_d
    )

    c_0 = c_00 * (1 - y_d) + c_10 * y_d
    c_1 = c_01 * (1 - y_d) + c_11 * y_d
     
    c = c_0 * (1 - z_d) + c_1 * z_d

    return c