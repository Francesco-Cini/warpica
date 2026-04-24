import numpy as np

def red_blue(value, gradient_num):

    if gradient_num is None:
        gradient_num = 1024

    min_value = np.min(value, [], 'all')
    max_value = np.max(value, [], 'all')

    if not(min_value <= 0 and max_value >= 0):
        if min_value > 0 and max_value > 0:
            returned_map = 1

    return returned_map