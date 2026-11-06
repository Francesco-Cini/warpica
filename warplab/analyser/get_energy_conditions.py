from typing import Literal, overload

import numpy as np

from warplab._typing import Array, TensorDict
from warplab.analyser.change_tensor_index import change_tensor_index
from warplab.analyser.do_frame_transfer import do_frame_transfer
from warplab.analyser.utils.generate_uniform_field import generate_uniform_field
from warplab.analyser.utils.get_inner_product import get_inner_product
from warplab.analyser.utils.get_trace import get_trace
from warplab.metrics.minkowski.metric_get_minkowski import metric_get_minkowski
from warplab.solver.utils.strcmpi import strcmpi
from warplab.solver.verify_tensor import verify_tensor


@overload
def get_energy_conditions(
    energy_tensor: TensorDict,
    metric: TensorDict,
    condition: str,
    num_angular_vec: int | None = None,
    num_time_vec: int | None = None,
    return_vec: Literal[False] | None = None,
    gpu: str | None = None,
) -> Array: ...


@overload
def get_energy_conditions(
    energy_tensor: TensorDict,
    metric: TensorDict,
    condition: str,
    num_angular_vec: int | None,
    num_time_vec: int | None,
    return_vec: Literal[True],
    gpu: str | None = None,
) -> tuple[Array, Array, Array]: ...


def get_energy_conditions(
    energy_tensor: TensorDict,
    metric: TensorDict,
    condition: str,
    num_angular_vec: int | None = None,
    num_time_vec: int | None = None,
    return_vec: bool | None = None,
    gpu: str | None = None,
) -> Array | tuple[Array, Array, Array]:
    """Evaluate an energy condition over sampled observer directions.

    Parameters
    ----------
    energy_tensor : dict
        Stress-energy tensor; transformed to the Eulerian frame if needed.
    metric : dict
        Metric metadata and components indexed as tensor[mu][nu], each
        with shape (nt, nx, ny, nz).
    condition : {'Null', 'Weak', 'Strong', 'Dominant'}
        Energy condition, compared case-insensitively.
    num_angular_vec : int or None, optional
        Number of spatial directions; None selects 100.
    num_time_vec : int or None, optional
        Number of speed samples for timelike observers; None selects 10.
    return_vec : bool or None, optional
        If true, also return per-sample values and the sampled vector field.
    gpu : str or None, optional
        Retained for API compatibility; analysis runs on the CPU.

    Returns
    -------
    map_array : ndarray
        Worst sampled value at each grid cell. Negative values indicate
        a violation; shape is (nt, nx, ny, nz).
    vec : ndarray, optional
        Per-direction values, returned only when return_vec is true.
        Trailing axes are angular samples, then speed samples for timelike
        conditions.
    vector_field_out : ndarray, optional
        Sampled four-vectors, returned only when return_vec is true.

    Notes
    -----
    Finite directional sampling is not a proof that a condition holds for
    all observers. Neither input tensor is modified.
    """
    if num_angular_vec is None:
        num_angular_vec = 100

    if num_time_vec is None:
        num_time_vec = 10

    if return_vec is None:
        return_vec = False

    if not (
        strcmpi(condition, "Null")
        or strcmpi(condition, "Weak")
        or strcmpi(condition, "Dominant")
        or strcmpi(condition, "Strong")
    ):
        raise ValueError(
            'Incorrect energy condition input, use either: "Null", "Weak", "Dominant", "Strong"'
        )

    if not verify_tensor(metric, 1):
        raise ValueError(
            "Metric is not verified. Please verify metric using verify_tensor(metric)."
        )

    if not verify_tensor(energy_tensor, 1):
        raise ValueError(
            "Stress-energy is not verified. Please verify Stress-energy tensor using "
            "verify_tensor(energy_tensor)."
        )

    import copy

    energy_tensor = copy.deepcopy(energy_tensor)
    metric = copy.deepcopy(metric)
    if num_angular_vec < 1 or num_time_vec < 1:
        raise ValueError("Vector sample counts must be positive")

    # Analysis is CPU float64. The public solver gathers device results before
    # this stage; do not mix NumPy arrays with CuPy/DirectML tensors here.
    # Grid shape (t, x, y, z): a, b, c, d are the sample counts on each axis.
    a, b, c, d = metric["tensor"][0][0].shape

    # Express T in the orthonormal frame of the Eulerian observers (those at
    # rest relative to the constant-time slices), where the metric is exactly
    # Minkowski. The conditions are statements about what such observers see.
    energy_tensor = do_frame_transfer(metric, energy_tensor, "Eulerian", gpu)

    # Components are now in an orthonormal frame: use Minkowski to raise/lower.
    metric = metric_get_minkowski(np.array([a, b, c, d]))

    # Null and dominant conditions sample light-like vectors; weak and strong
    # sample timelike ones at several speeds.
    if strcmpi(condition, "Null") or strcmpi(condition, "Dominant"):
        type = "nulllike"

    elif strcmpi(condition, "Weak") or strcmpi(condition, "Strong"):
        type = "timelike"

    # vec_field[mu][ii] (light-like) or vec_field[mu][ii][jj] (timelike) is
    # component mu of sampled direction ii at speed index jj.
    vec_field = generate_uniform_field(type, num_angular_vec, num_time_vec)

    # map_array holds the worst value seen so far in every cell. It starts at
    # +inf because the conditions keep a running minimum; the dominant
    # condition keeps a maximum and starts at -inf (its sign is flipped at the
    # end so that negative always means violated).
    map_array = np.full(
        (a, b, c, d), -np.inf if strcmpi(condition, "Dominant") else np.inf
    )
    if return_vec == 1:
        vec = np.zeros(
            (a, b, c, d, num_angular_vec)
            + (() if type == "nulllike" else (num_time_vec,))
        )

    # Null energy condition
    # Requires T_mu_nu k^mu k^nu >= 0 for every light-like k. Each sample
    # contracts T with the vector, summed over mu and nu.
    if strcmpi(condition, "Null"):
        energy_tensor = change_tensor_index(energy_tensor, "covariant", metric)

        for ii in range(num_angular_vec):
            temp = np.zeros((a, b, c, d))

            for mu in range(4):
                for nu in range(4):
                    temp = (
                        temp
                        + energy_tensor["tensor"][mu][nu]
                        * vec_field[mu][ii]
                        * vec_field[nu][ii]
                    )

            map_array = np.minimum(map_array, temp)

            if return_vec == 1:
                vec[:, :, :, :, ii] = temp

    # Weak energy condition
    # Requires T_mu_nu u^mu u^nu >= 0 for every timelike u: the energy density
    # measured by an observer with that 4-velocity is never negative.
    elif strcmpi(condition, "Weak"):
        energy_tensor = change_tensor_index(energy_tensor, "covariant", metric)

        for jj in range(num_time_vec):
            for ii in range(num_angular_vec):
                temp = np.zeros((a, b, c, d))

                for mu in range(4):
                    for nu in range(4):
                        temp = (
                            temp
                            + energy_tensor["tensor"][mu][nu]
                            * vec_field[mu][ii][jj]
                            * vec_field[nu][ii][jj]
                        )

                map_array = np.minimum(map_array, temp)

                if return_vec == 1:
                    vec[:, :, :, :, ii, jj] = temp

    # Dominant energy condition
    # Requires the energy flux -T^mu_nu u^nu seen by each observer to be
    # timelike or null and future directed. The flux is built as a 4-vector
    # below and its signed Minkowski length is the tested quantity.
    elif strcmpi(condition, "Dominant"):
        metric_minkowski = metric_get_minkowski(np.array([a, b, c, d]))
        metric_minkowski = change_tensor_index(metric_minkowski, "covariant")

        energy_tensor = change_tensor_index(
            energy_tensor, "mixedupdown", metric_minkowski
        )

        for ii in range(num_angular_vec):
            temp = np.zeros((a, b, c, d, 4))

            for mu in range(4):
                for nu in range(4):
                    temp[:, :, :, :, mu] = (
                        temp[:, :, :, :, mu]
                        - energy_tensor["tensor"][mu][nu] * vec_field[nu, ii]
                    )

            vector: TensorDict = {}
            vector["field"] = np.array(
                [
                    temp[:, :, :, :, 0],
                    temp[:, :, :, :, 1],
                    temp[:, :, :, :, 2],
                    temp[:, :, :, :, 3],
                ]
            )
            vector["index"] = "contravariant"
            vector["type"] = "4-vector"

            # Find inner product to determine if timelike or null
            diff = get_inner_product(vector, vector, metric_minkowski)
            diff = np.sign(diff) * np.sqrt(np.abs(diff))
            # Causality alone also admits past-directed flux: enforce future direction.
            diff = np.maximum(diff, -temp[..., 0])

            map_array = np.maximum(map_array, diff)

            if return_vec == 1:
                vec[:, :, :, :, ii] = diff

        # Flip sign of the dominant energy condition to better align with evaluations of
        # other conditions (i.e. negative is violating)

        map_array = -map_array
        if return_vec == 1:
            vec = -vec

    # Strong energy condition
    # Requires (T_mu_nu - (1/2) T eta_mu_nu) u^mu u^nu >= 0 for every timelike
    # u, where T is the trace of the stress-energy (e_trace below).
    elif strcmpi(condition, "Strong"):
        metric_minkowski = metric_get_minkowski(np.array([a, b, c, d]))
        metric_minkowski = change_tensor_index(metric_minkowski, "covariant")

        energy_tensor = change_tensor_index(
            energy_tensor, "covariant", metric_minkowski
        )

        e_trace = get_trace(energy_tensor, metric_minkowski)

        for jj in range(num_time_vec):
            for ii in range(num_angular_vec):
                temp = np.zeros((a, b, c, d))

                for mu in range(4):
                    for nu in range(4):
                        temp = (
                            temp
                            + (
                                energy_tensor["tensor"][mu][nu]
                                - 0.5 * e_trace * metric_minkowski["tensor"][mu][nu]
                            )
                            * vec_field[mu][ii][jj]
                            * vec_field[nu][ii][jj]
                        )

                map_array = np.minimum(map_array, temp)
                if return_vec == 1:
                    vec[:, :, :, :, ii, jj] = temp

    if return_vec == 1:
        return map_array, vec, vec_field

    return map_array
