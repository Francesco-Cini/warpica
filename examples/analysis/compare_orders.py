"""Compare interior derivative accuracy against a known analytic function."""

import numpy as np

from warplab.solver.utils.second_order.take_finite_difference_1_2 import (
    take_finite_difference_1_2,
)
from warplab.solver.utils.take_finite_difference_1 import take_finite_difference_1


def main():
    print("samples    second-order error    fourth-order error")
    for n in (17, 33, 65):
        x = np.linspace(0, 2 * np.pi, n)
        field = np.sin(x).reshape(1, n, 1, 1)
        spacing = [1, x[1] - x[0], 1, 1]
        errors = []
        for derivative in (take_finite_difference_1_2, take_finite_difference_1):
            numerical = derivative(field, 1, spacing)[0, 2:-2, 0, 0]
            errors.append(np.max(np.abs(numerical - np.cos(x[2:-2]))))
        print(f"{n:7d}    {errors[0]:18.6g}    {errors[1]:18.6g}")
    print("Halving grid spacing should reduce these errors by roughly 4 and 16.")


if __name__ == "__main__":
    main()
