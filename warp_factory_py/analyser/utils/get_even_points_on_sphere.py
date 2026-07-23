import numpy as np

def get_even_points_on_sphere(R, number_of_points):

    golden_ratio = (1 + (5 ** 0.5)) / 2
    vector = np.zeros((3, number_of_points))

    for i in range(number_of_points):
        theta = (2 * np.pi * i) / golden_ratio
        phi = np.acos(1 - 2 * (i + 0.5) / number_of_points)

        vector[0, i] = R * np.cos(theta) * np.sin(phi)
        vector[1, i] = R * np.sin(theta) * np.sin(phi)
        vector[2, i] = R * np.cos(phi)

    vector = np.real(vector)
    
    return vector 
