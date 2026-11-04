import copy

import numpy as np
import pytest

from warplab.metrics import metric_get_minkowski
from warplab.solver.verify_tensor import verify_tensor


def valid_metric():
    return metric_get_minkowski([1, 3, 3, 3])


def check(tensor, **kwargs):
    return verify_tensor(tensor, 1, **kwargs)


def test_valid_metric_and_stress_energy_pass_without_warnings():
    metric = valid_metric()
    assert check(metric) is True
    energy = copy.deepcopy(metric)
    energy["type"] = "Stress-Energy"
    energy["index"] = "contravariant"
    assert check(energy) is True


def test_type_comparison_ignores_case():
    for name in ("metric", "METRIC", "stress-energy", "Stress-ENERGY"):
        tensor = valid_metric()
        tensor["type"] = name
        assert check(tensor) is True


@pytest.mark.parametrize(
    "index", ["contravariant", "covariant", "mixedupdown", "mixeddownup"]
)
def test_every_documented_index_is_accepted(index):
    tensor = valid_metric()
    tensor["index"] = index
    assert check(tensor) is True


@pytest.mark.parametrize("bad", [{}, [1, 2], None, "metric", 3])
def test_input_without_a_type_is_rejected_not_a_crash(bad):
    """Regression: the original port called the warnings module and raised."""
    with pytest.warns(UserWarning, match="type does not exist"):
        assert check(bad) is False


def test_unknown_type_is_rejected():
    tensor = valid_metric()
    tensor["type"] = "Banana"
    with pytest.warns(UserWarning, match="Unknown Type"):
        assert check(tensor) is False


def test_missing_tensor_is_rejected():
    tensor = valid_metric()
    del tensor["tensor"]
    with pytest.warns(UserWarning, match="tensor: Empty"):
        assert check(tensor) is False


@pytest.mark.parametrize(
    "bad_tensor",
    [
        "not a tensor",
        [[np.zeros((1, 2, 2, 2))] * 4] * 3,
        [[np.zeros((1, 2, 2, 2))] * 3] * 4,
        [[np.zeros((2, 2, 2))] * 4] * 4,
        np.zeros((3, 4, 1, 2, 2, 2)),
        np.zeros((4, 4, 2, 2, 2)),
    ],
    ids=["string", "3 rows", "3 columns", "3D components", "bad ndarray", "5D ndarray"],
)
def test_malformed_tensor_is_rejected(bad_tensor):
    tensor = valid_metric()
    tensor["tensor"] = bad_tensor
    with pytest.warns(UserWarning, match="not formatted correctly"):
        assert check(tensor) is False


def test_both_supported_tensor_layouts_are_accepted():
    tensor = valid_metric()
    array = np.asarray(tensor["tensor"])
    assert array.shape == (4, 4, 1, 3, 3, 3)
    tensor["tensor"] = [[array[i, j] for j in range(4)] for i in range(4)]
    assert check(tensor) is True
    tensor["tensor"] = array
    assert check(tensor) is True


def test_missing_coords_is_rejected():
    tensor = valid_metric()
    del tensor["coords"]
    with pytest.warns(UserWarning, match="coords: Empty"):
        assert check(tensor) is False


def test_non_cartesian_coords_are_flagged():
    tensor = valid_metric()
    tensor["coords"] = "spherical"
    with pytest.warns(UserWarning, match="Non-cartesian"):
        check(tensor)


def test_missing_index_is_rejected():
    tensor = valid_metric()
    del tensor["index"]
    with pytest.warns(UserWarning, match="index: Empty"):
        assert check(tensor) is False


def test_unknown_index_is_rejected():
    tensor = valid_metric()
    tensor["index"] = "sideways"
    with pytest.warns(UserWarning, match="Unknown index"):
        assert check(tensor) is False


def test_every_problem_is_reported_not_just_the_first():
    with pytest.warns(UserWarning) as record:
        assert check({"type": "Banana"}) is False
    messages = [str(item.message) for item in record]
    assert any("Unknown Type" in m for m in messages)
    assert any("tensor: Empty" in m for m in messages)
    assert any("coords: Empty" in m for m in messages)
    assert any("index: Empty" in m for m in messages)


def test_verification_does_not_modify_the_input():
    tensor = valid_metric()
    before = copy.deepcopy(tensor)
    check(tensor)
    assert tensor.keys() == before.keys()
    np.testing.assert_array_equal(tensor["tensor"], before["tensor"])
    assert tensor["index"] == before["index"]


def test_messages_are_printed_unless_suppressed(capsys):
    verify_tensor(valid_metric())
    printed = capsys.readouterr().out
    assert "Type: Metric" in printed
    assert "tensor: Verified" in printed
    assert "coords: cartesian" in printed
    assert "index: covariant" in printed

    verify_tensor(valid_metric(), 1)
    assert capsys.readouterr().out == ""
