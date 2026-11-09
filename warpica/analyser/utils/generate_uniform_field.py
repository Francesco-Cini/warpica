import numpy as np

from warpica.analyser.utils.get_even_points_on_sphere import get_even_points_on_sphere
from warpica.solver.utils.strcmpi import strcmpi


def generate_uniform_field(type, num_angular_vec, num_time_vec, gpu=None):
    """Generate sampled null or timelike four-vectors.

    Parameters
    ----------
    type : {'nulllike', 'timelike'}
        Causal type of the sampled vectors.
    num_angular_vec : int
        Number of spatial directions.
    num_time_vec : int
        Number of speeds in [0, 1), used only for timelike vectors.
    gpu : str or None, optional
        Optional output backend; None keeps the field on the CPU.

    Returns
    -------
    vec_field : array
        Shape (4, num_angular_vec) for null vectors, or
        (4, num_angular_vec, num_time_vec) for timelike vectors.

    Notes
    -----
    Vectors have unit Euclidean norm, not unit Minkowski norm. Their
    causal type is defined using signature (-, +, +, +).
    """
    if not (strcmpi(type, "nulllike") or strcmpi(type, "timelike")):
        raise ValueError(
            "Vector field type not generated, use either: 'nulllike' or 'timelike'"
        )

    if strcmpi(type, "timelike"):
        # Speeds as fractions of c. endpoint=False excludes exactly 1, so every
        # sampled vector is strictly timelike (speed 1 would be light-like).
        bb = np.linspace(0, 1, num_time_vec, endpoint=False)

        # Each vector is (1, v n): time component 1 and a direction n scaled by
        # the speed v. It is then divided by its Euclidean norm, which fixes the
        # overall scale of the reported values but not their sign.
        vec_field = np.ones((4, num_angular_vec, num_time_vec))

        for jj in range(num_time_vec):
            vec_field[1:, :, jj] = get_even_points_on_sphere(bb[jj], num_angular_vec)
            vec_field[:, :, jj] = vec_field[:, :, jj] / (
                (
                    (vec_field[0, :, jj] ** 2)
                    + (vec_field[1, :, jj] ** 2)
                    + (vec_field[2, :, jj] ** 2)
                    + (vec_field[3, :, jj] ** 2)
                )
                ** 0.5
            )

    elif strcmpi(type, "nulllike"):
        vec_field = np.ones((4, num_angular_vec))
        vec_field[1:, :] = get_even_points_on_sphere(1, num_angular_vec)
        vec_field = vec_field / (
            (
                (vec_field[0, :] ** 2)
                + (vec_field[1, :] ** 2)
                + (vec_field[2, :] ** 2)
                + (vec_field[3, :] ** 2)
            )
            ** 0.5
        )

    if gpu is not None:
        from warpica.gpu import asarray as gpu_asarray

        vec_field = gpu_asarray(vec_field, library=gpu)

    return vec_field
