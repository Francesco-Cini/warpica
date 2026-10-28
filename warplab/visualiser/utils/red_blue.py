import numpy as np


def red_blue(value, gradient_num):
    """Retain the unfinished legacy red/blue colormap prototype.

    Parameters
    ----------
    value : array_like
        Values intended to determine the colour range.
    gradient_num : int or None
        Requested gradient size; None selects 1024.

    Notes
    -----
    This legacy helper is incomplete and does not implement a working
    colormap. Use plot_component, which uses Matplotlib's RdBu_r colormap.
    """
    if gradient_num is None:
        gradient_num = 1024

    min_value = np.min(value, [], "all")
    max_value = np.max(value, [], "all")

    if not (min_value <= 0 and max_value >= 0):
        if min_value > 0 and max_value > 0:
            returned_map = 1

    return returned_map
