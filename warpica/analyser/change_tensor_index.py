import numpy as np

from warpica._typing import TensorDict
from warpica.solver.utils.c4_inv import c4_inv


def change_tensor_index(
    input_tensor: TensorDict,
    index: str,
    metric_tensor: TensorDict | None = None,
) -> TensorDict:
    """Return a tensor copy with the requested index convention.

    Parameters
    ----------
    input_tensor : dict
        Tensor metadata and component fields.
    index : str
        Target: 'covariant', 'contravariant', 'mixedupdown', or
        'mixeddownup'. Metric tensors cannot use mixed indices.
    metric_tensor : dict, optional
        Reference metric, required for non-metric tensors.

    Returns
    -------
    output : dict
        New dictionary with independent component arrays and updated index.
        Other metadata is shallow-copied.
    """
    # For each form, whether (first index, second index) is raised (contravariant):
    # covariant = (False, False), contravariant = (True, True), and the mixed forms
    # have one index up and one down.
    states = {
        "covariant": (False, False),
        "contravariant": (True, True),
        "mixedupdown": (True, False),
        "mixeddownup": (False, True),
    }
    index = index.lower()
    source = input_tensor["index"].lower()
    if index not in states or source not in states:
        raise ValueError("Unknown tensor index convention")
    output = dict(input_tensor)
    # A metric changes between covariant and contravariant by matrix inversion,
    # not by contracting with a metric; mixed forms make no sense for it.
    if input_tensor["type"].lower() == "metric":
        if index.startswith("mixed") or source.startswith("mixed"):
            raise ValueError("Metric tensors cannot use mixed indices")
        output["tensor"] = (
            c4_inv(input_tensor["tensor"])
            if source != index
            else [[v.copy() for v in row] for row in input_tensor["tensor"]]
        )
    else:
        if metric_tensor is None:
            raise ValueError("metric_tensor is required for non-metric tensors")
        metric_index = metric_tensor["index"].lower()
        if metric_index not in ("covariant", "contravariant"):
            raise ValueError("Metric tensors cannot use mixed indices")
        # Any other tensor: lower an index by contracting with the covariant metric,
        # raise it with the inverse metric. Pick whichever the given metric needs.
        lower = (
            metric_tensor["tensor"]
            if metric_index == "covariant"
            else c4_inv(metric_tensor["tensor"])
        )
        upper = (
            metric_tensor["tensor"]
            if metric_index == "contravariant"
            else c4_inv(metric_tensor["tensor"])
        )
        output["tensor"] = [[v.copy() for v in row] for row in input_tensor["tensor"]]
        # Convert one index at a time, and only the indices whose position differs.
        for axis in range(2):
            if states[source][axis] != states[index][axis]:
                factor = {"tensor": upper if states[index][axis] else lower}
                output["tensor"] = (mix_index_1 if axis == 0 else mix_index_2)(
                    output, factor
                )
    output["index"] = index
    return output


def mix_index_1(input_tensor, metric_tensor):
    """Raise or lower the first tensor index with the supplied metric.

    Parameters
    ----------
    input_tensor : dict
        Tensor containing a 4 by 4 table of component arrays.
    metric_tensor : dict
        Covariant metric to lower indices, or inverse metric to raise them.

    Returns
    -------
    tensor : list of lists of arrays
        A 4 by 4 component table; each component has the input grid shape.
    """
    temp_output_tensor = [[None for _ in range(4)] for _ in range(4)]

    for i in range(4):
        for j in range(4):
            temp_output_tensor[i][j] = np.zeros(input_tensor["tensor"][i][j].shape)

            # Contract the first index: out[i][j] = sum_a in[a][j] * M[a][i].
            for a in range(4):
                temp_output_tensor[i][j] = (
                    temp_output_tensor[i][j]
                    + input_tensor["tensor"][a][j] * metric_tensor["tensor"][a][i]
                )
    return temp_output_tensor


def mix_index_2(input_tensor, metric_tensor):
    """Raise or lower the second tensor index with the supplied metric.

    Parameters
    ----------
    input_tensor : dict
        Tensor containing a 4 by 4 table of component arrays.
    metric_tensor : dict
        Covariant metric to lower indices, or inverse metric to raise them.

    Returns
    -------
    tensor : list of lists of arrays
        A 4 by 4 component table; each component has the input grid shape.
    """
    temp_output_tensor = [[None for _ in range(4)] for _ in range(4)]

    for i in range(4):
        for j in range(4):
            temp_output_tensor[i][j] = np.zeros(input_tensor["tensor"][i][j].shape)

            # Contract the second index: out[i][j] = sum_a in[i][a] * M[a][j].
            for a in range(4):
                temp_output_tensor[i][j] = (
                    temp_output_tensor[i][j]
                    + input_tensor["tensor"][i][a] * metric_tensor["tensor"][a][j]
                )
    return temp_output_tensor
