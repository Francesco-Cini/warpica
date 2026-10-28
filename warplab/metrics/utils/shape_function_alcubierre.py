import numpy as np


def shape_function_alcubierre(r, R, sigma):
    """Evaluate the smooth radial Alcubierre bubble profile.

    Parameters
    ----------
    r : float or ndarray
        Distance from the bubble centre.
    R : float
        Bubble radius in spatial coordinate units.
    sigma : float
        Wall steepness, in inverse spatial coordinate units.

    Returns
    -------
    profile : float or ndarray
        Normalised hyperbolic-tangent profile evaluated at r.
    """
    f = (np.tanh(sigma * (R + r)) + np.tanh(sigma * (R - r))) / (2 * np.tanh(R * sigma))
    return f
