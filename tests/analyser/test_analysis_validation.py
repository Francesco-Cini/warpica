"""Input validation, defaults and edge cases of the analysis entry points."""

import copy
import importlib

import numpy as np
import pytest

from warplab.analyser import (
    do_frame_transfer,
    eval_metric,
    get_energy_conditions,
    get_momentum_flow_lines,
)
from warplab.analyser.utils.generate_uniform_field import generate_uniform_field
from warplab.metrics import (
    metric_get_alcubierre,
    metric_get_minkowski,
    metric_get_warp_shell_comoving,
)
from warplab.solver import get_energy_tensor

CONDITIONS = ["Null", "Weak", "Strong", "Dominant"]


def bubble():
    return metric_get_alcubierre(
        np.array([1, 5, 5, 5]),
        np.array([1, 2.8, 2.7, 2.6]),
        0.2,
        1,
        1,
        np.ones(4),
    )


def shell():
    """A shell with positive energy density, so condition maps have positive cells."""
    return metric_get_warp_shell_comoving(
        np.array([1, 7, 7, 7]),
        np.array([1, 4.1, 4.2, 3.9]),
        2e26,
        0.5,
        1.5,
        0.1,
        1,
        1,
        0.2,
        True,
        np.ones(4),
    )


def tensors_equal(first, second):
    return np.array_equal(np.array(first["tensor"]), np.array(second["tensor"]))


# ---- do_frame_transfer -----------------------------------------------------


@pytest.fixture(scope="module")
def bubble_and_energy():
    metric = bubble()
    return metric, get_energy_tensor(metric)


def test_frame_transfer_rejects_an_unverified_metric(bubble_and_energy):
    _, energy = bubble_and_energy
    with pytest.warns(UserWarning, match="type does not exist"):
        with pytest.raises(ValueError, match="Metric is not verified"):
            do_frame_transfer({}, energy, "Eulerian")


def test_frame_transfer_reports_a_metric_missing_a_field_cleanly(bubble_and_energy):
    """It used to be a KeyError from change_tensor_index, before verification."""
    metric, energy = bubble_and_energy
    broken = {key: value for key, value in metric.items() if key != "index"}
    with pytest.warns(UserWarning, match="index: Empty"):
        with pytest.raises(ValueError, match="Metric is not verified"):
            do_frame_transfer(broken, energy, "Eulerian")


def test_frame_transfer_rejects_an_unverified_energy_tensor(bubble_and_energy):
    metric, energy = bubble_and_energy
    broken = {key: value for key, value in energy.items() if key != "coords"}
    with pytest.warns(UserWarning, match="coords: Empty"):
        with pytest.raises(ValueError, match="Stress-energy is not verified"):
            do_frame_transfer(metric, broken, "Eulerian")


def test_frame_transfer_rejects_an_unknown_frame(bubble_and_energy):
    metric, energy = bubble_and_energy
    with pytest.raises(ValueError, match="Frame not found"):
        do_frame_transfer(metric, energy, "Lagrangian")


def test_frame_names_are_case_insensitive(bubble_and_energy):
    metric, energy = bubble_and_energy
    reference = do_frame_transfer(metric, energy, "Eulerian")
    assert tensors_equal(do_frame_transfer(metric, energy, "eulerian"), reference)
    assert reference["frame"] == "Eulerian"
    assert reference["index"] == "contravariant"


def test_an_eulerian_tensor_is_returned_unchanged_as_a_copy(bubble_and_energy):
    metric, energy = bubble_and_energy
    eulerian = do_frame_transfer(metric, energy, "Eulerian")
    again = do_frame_transfer(metric, eulerian, "Eulerian")
    assert again is not eulerian
    assert tensors_equal(again, eulerian)
    again["tensor"][0][0][...] = 0
    assert not tensors_equal(again, eulerian)


def test_frame_transfer_does_not_modify_its_inputs(bubble_and_energy):
    metric, energy = bubble_and_energy
    before = copy.deepcopy((metric, energy))
    do_frame_transfer(metric, energy, "Eulerian")
    assert tensors_equal(metric, before[0])
    assert tensors_equal(energy, before[1])
    assert "frame" not in energy


# ---- get_energy_conditions ---------------------------------------------------


def test_unknown_condition_names_are_rejected(bubble_and_energy):
    metric, energy = bubble_and_energy
    with pytest.raises(ValueError, match="Incorrect energy condition"):
        get_energy_conditions(energy, metric, "Bogus")


@pytest.mark.parametrize("condition", CONDITIONS)
def test_condition_names_are_case_insensitive(bubble_and_energy, condition):
    metric, energy = bubble_and_energy
    eulerian = do_frame_transfer(metric, energy, "Eulerian")
    reference = get_energy_conditions(eulerian, metric, condition, 6, 2)
    for spelling in (condition.lower(), condition.upper()):
        np.testing.assert_array_equal(
            get_energy_conditions(eulerian, metric, spelling, 6, 2), reference
        )


@pytest.mark.parametrize(("angular", "timelike"), [(0, 3), (3, 0), (-1, 3)])
def test_sample_counts_must_be_positive(bubble_and_energy, angular, timelike):
    metric, energy = bubble_and_energy
    with pytest.raises(ValueError, match="must be positive"):
        get_energy_conditions(energy, metric, "Null", angular, timelike)


def test_unverified_inputs_are_rejected(bubble_and_energy):
    metric, energy = bubble_and_energy
    with pytest.warns(UserWarning, match="type does not exist"):
        with pytest.raises(ValueError, match="Metric is not verified"):
            get_energy_conditions(energy, {}, "Null")
    with pytest.warns(UserWarning, match="type does not exist"):
        with pytest.raises(ValueError, match="Stress-energy is not verified"):
            get_energy_conditions({}, metric, "Null")


def test_default_sample_counts_are_100_directions_and_10_speeds():
    metric = metric_get_minkowski([1, 2, 2, 2])
    energy = copy.deepcopy(metric)
    energy.update(type="Stress-Energy", index="contravariant")
    energy["tensor"][:] = 0
    energy["tensor"][0, 0] = 1
    null = get_energy_conditions(energy, metric, "Null", None, None, True)
    weak = get_energy_conditions(energy, metric, "Weak", None, None, True)
    assert null[1].shape[-1] == 100
    assert weak[1].shape[-2:] == (100, 10)


def test_samples_are_only_returned_on_request(bubble_and_energy):
    metric, energy = bubble_and_energy
    eulerian = do_frame_transfer(metric, energy, "Eulerian")
    assert isinstance(get_energy_conditions(eulerian, metric, "Null", 4, 2), np.ndarray)
    assert len(get_energy_conditions(eulerian, metric, "Null", 4, 2, True)) == 3


# ---- eval_metric --------------------------------------------------------------


def test_eval_metric_defaults_to_100_directions_and_10_speeds(monkeypatch):
    module = importlib.import_module("warplab.analyser.eval_metric")
    calls = []

    def spy(energy, metric, condition, angular, timelike, *rest):
        calls.append((condition, angular, timelike))
        return np.zeros(energy["tensor"][0][0].shape)

    monkeypatch.setattr(module, "get_energy_conditions", spy)
    eval_metric(bubble(), keep_positive=None, num_angular_vec=None, num_time_vec=None)
    assert calls == [(name, 100, 10) for name in CONDITIONS]


def test_keep_positive_false_clips_positive_cells_to_zero():
    metric = shell()
    kept = eval_metric(metric, num_angular_vec=6, num_time_vec=2)
    clipped = eval_metric(
        metric, keep_positive=False, num_angular_vec=6, num_time_vec=2
    )
    for name in ("null", "weak", "strong", "dominant"):
        assert (kept[name] > 0).any(), f"{name}: the test needs positive cells"
        np.testing.assert_array_equal(clipped[name], np.minimum(kept[name], 0))
    for name in ("expansion", "shear", "vorticity"):
        np.testing.assert_array_equal(clipped[name], kept[name])


def test_keep_positive_defaults_to_true():
    metric = shell()
    default = eval_metric(metric, num_angular_vec=6, num_time_vec=2)
    explicit = eval_metric(
        metric, keep_positive=True, num_angular_vec=6, num_time_vec=2
    )
    for name in ("null", "weak", "strong", "dominant"):
        np.testing.assert_array_equal(default[name], explicit[name])


def test_eval_metric_returns_every_documented_field():
    result = eval_metric(bubble(), num_angular_vec=4, num_time_vec=2)
    assert set(result) == {
        "metric",
        "energy_tensor",
        "energy_tensor_eulerian",
        "null",
        "weak",
        "strong",
        "dominant",
        "expansion",
        "shear",
        "vorticity",
    }
    assert result["energy_tensor_eulerian"]["frame"] == "Eulerian"


def test_eval_metric_accepts_the_numpy_backend():
    metric = bubble()
    plain = eval_metric(metric, num_angular_vec=4, num_time_vec=2)
    numpy_backend = eval_metric(metric, num_angular_vec=4, num_time_vec=2, gpu="numpy")
    for name in ("null", "weak", "strong", "dominant", "expansion", "shear"):
        np.testing.assert_allclose(numpy_backend[name], plain[name], rtol=1e-12)


# ---- get_momentum_flow_lines ----------------------------------------------------


def flow_energy(components=None, index="contravariant", shape=(8, 8, 8)):
    energy = {"index": index, "tensor": np.zeros((4, 4, *shape))}
    energy["tensor"][0, 1] = 1.0
    if components is not None:
        energy["tensor"] = components
    return energy


def test_flow_lines_need_a_contravariant_tensor():
    with pytest.raises(ValueError, match="should be contravariant"):
        get_momentum_flow_lines(
            flow_energy(index="covariant"), [[3], [3], [3]], 0.25, 4, 1
        )


@pytest.mark.parametrize(("step", "steps"), [(0, 4), (-0.5, 4), (0.25, -1)])
def test_flow_line_step_arguments_are_validated(step, steps):
    with pytest.raises(ValueError, match="max_steps must be nonnegative"):
        get_momentum_flow_lines(flow_energy(), [[3], [3], [3]], step, steps, 1)


def test_a_single_time_slice_is_accepted_and_several_are_not():
    reference = get_momentum_flow_lines(flow_energy(), [[3], [3], [3]], 0.25, 4, 1)
    one_slice = np.zeros((4, 4, 1, 8, 8, 8))
    one_slice[0, 1] = 1.0
    sliced = get_momentum_flow_lines(
        flow_energy(components=one_slice), [[3], [3], [3]], 0.25, 4, 1
    )
    np.testing.assert_array_equal(sliced[0], reference[0])
    two_slices = np.zeros((4, 4, 2, 8, 8, 8))
    with pytest.raises(ValueError, match="one time slice"):
        get_momentum_flow_lines(
            flow_energy(components=two_slices), [[3], [3], [3]], 0.25, 4, 1
        )


def test_zero_steps_returns_just_the_starting_point():
    paths = get_momentum_flow_lines(flow_energy(), [[3], [4], [5]], 0.25, 0, 1)
    np.testing.assert_array_equal(paths[0], [[3, 4, 5]])


def test_a_path_stops_where_the_interpolation_stencil_leaves_the_grid():
    paths = get_momentum_flow_lines(flow_energy(), [[6.5], [4], [4]], 1.0, 50, 1)
    np.testing.assert_allclose(paths[0], [[6.5, 4, 4], [7.5, 4, 4]])


def test_a_path_starting_outside_the_usable_grid_does_not_move():
    paths = get_momentum_flow_lines(flow_energy(), [[1], [4], [4]], 1.0, 5, 1)
    np.testing.assert_array_equal(paths[0], [[1, 4, 4]])


def test_each_seed_gets_its_own_path():
    paths = get_momentum_flow_lines(flow_energy(), [[3, 4], [3, 4], [3, 4]], 0.25, 2, 1)
    assert len(paths) == 2
    np.testing.assert_allclose(paths[1][0], [4, 4, 4])


def test_flow_lines_do_not_modify_the_tensor():
    energy = flow_energy()
    before = copy.deepcopy(energy)
    get_momentum_flow_lines(energy, [[3], [3], [3]], 0.25, 4, 1)
    np.testing.assert_array_equal(energy["tensor"], before["tensor"])


# ---- generate_uniform_field -------------------------------------------------------


def test_the_field_type_must_be_known():
    with pytest.raises(ValueError, match="Vector field type not generated"):
        generate_uniform_field("spacelike", 4, 2)


@pytest.mark.parametrize("kind", ["NullLike", "TIMELIKE"])
def test_field_types_are_case_insensitive(kind):
    assert generate_uniform_field(kind, 4, 2).shape[:2] == (4, 4)


@pytest.mark.parametrize("kind", ["nulllike", "timelike"])
def test_the_numpy_backend_returns_the_same_field(kind):
    np.testing.assert_array_equal(
        generate_uniform_field(kind, 6, 3, gpu="numpy"),
        generate_uniform_field(kind, 6, 3),
    )
