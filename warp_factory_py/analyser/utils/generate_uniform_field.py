import numpy as np

from warp_factory_py.solver.utils.strcmpi import strcmpi
from warp_factory_py.analyser.utils.get_even_points_on_sphere import get_even_points_on_sphere

def generate_uniform_field(type, num_angular_vec, num_time_vec):

    if not (strcmpi(type, "nulllike") or strcmpi(type, "timelike")):
        raise Exception("Vector field type not generated, used either: 'nulllike', 'timelike")

    if strcmpi(type, "timelike"):
        bb = np.linspace(0, 1, num_time_vec)

        vec_field = np.ones((4, num_angular_vec, num_time_vec))

        for jj in range(num_time_vec):
            vec_field[1:, :, jj] = get_even_points_on_sphere(1 - bb[jj], num_angular_vec)
            vec_field[:, :, jj] = vec_field[:, :, jj] / (((vec_field[0, :, jj] ** 2) + (vec_field[1, :, jj] ** 2) + (vec_field[2, :, jj] ** 2) + (vec_field[3, :, jj] ** 2)) ** 0.5)

    elif strcmpi(type, "nulllike"):
        vec_field = np.ones((4, num_angular_vec))
        vec_field[1:, :] = get_even_points_on_sphere(1, num_angular_vec)
        vec_field = vec_field / (((vec_field[0, :] ** 2) + (vec_field[1, :] ** 2) + (vec_field[2, :] ** 2) + (vec_field[3, :] ** 2)) ** 0.5)
        
    return vec_field
