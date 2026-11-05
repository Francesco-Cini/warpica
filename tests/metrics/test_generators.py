import importlib

import numpy as np
import pytest

from warplab.metrics.custom_metric import custom_metric
from warplab.metrics.minkowski.metric_get_minkowski import metric_get_minkowski
from warplab.metrics.three_plus_one_builder import three_plus_one_builder
from warplab.metrics.three_plus_one_decomposer import three_plus_one_decomposer
from warplab.solver.verify_tensor import verify_tensor

CASES = [
    ("alcubierre", "alcubierre", [0.2, 1, 1]),
    ("alcubierre", "alcubierre_comoving", [0.2, 1, 1]),
    ("lentz", "lentz", [0.2, 1]),
    ("lentz", "lentz_comoving", [0.2, 1]),
    ("modified_time", "modified_time", [0.2, 1, 1, 2]),
    ("modified_time", "modified_time_comoving", [0.2, 1, 1, 2]),
    ("van_den_broeck", "van_den_broeck", [0.2, 1, 1, 2, 1, 0.1]),
    ("van_den_broeck", "van_den_broeck_comoving", [0.2, 1, 1, 2, 1, 0.1]),
    ("schwarzschild", "schwarzschild", [0.1]),
    ("warp_shell", "warp_shell_comoving", [1e20, 0.5, 1.5, 0, 1, 3, 0.1, True]),
]


def build(family, name, args):
    module = importlib.import_module(f"warplab.metrics.{family}.metric_get_{name}")
    fn = getattr(module, f"metric_get_{name}")
    return fn(np.array([1, 5, 5, 5]), np.array([1, 2.8, 2.7, 2.6]), *args, np.ones(4))


@pytest.mark.parametrize("family,name,args", CASES, ids=[c[1] for c in CASES])
def test_metric_shape_signature_and_reconstruction(family, name, args):
    metric = build(family, name, args)
    assert verify_tensor(metric, 1)
    assert "date" in metric
    tensor = np.asarray(metric["tensor"])
    assert tensor.shape == (4, 4, 1, 5, 5, 5)
    assert np.isfinite(tensor).all()
    np.testing.assert_array_equal(tensor, tensor.swapaxes(0, 1))
    eigenvalues = np.linalg.eigvalsh(np.moveaxis(tensor, (0, 1), (-2, -1)))
    assert np.all(np.sum(eigenvalues < 0, axis=-1) == 1)
    alpha, beta, gamma, _, _ = three_plus_one_decomposer(metric)
    np.testing.assert_allclose(
        three_plus_one_builder(alpha, beta, gamma), tensor, atol=1e-13
    )


@pytest.mark.parametrize("family,name,args", CASES[:8], ids=[c[1] for c in CASES[:8]])
def test_flat_limits(family, name, args):
    args = list(args)
    args[0] = 0
    if family == "modified_time":
        args[-1] = 1
    if family == "van_den_broeck":
        args[-1] = 0
    metric = build(family, name, args)
    np.testing.assert_allclose(
        metric["tensor"], metric_get_minkowski([1, 5, 5, 5])["tensor"], atol=1e-14
    )


def test_schwarzschild_zero_mass_and_formula():
    metric = build("schwarzschild", "schwarzschild", [0])
    np.testing.assert_array_equal(
        metric["tensor"], metric_get_minkowski([1, 5, 5, 5])["tensor"]
    )
    metric = build("schwarzschild", "schwarzschild", [0.1])
    radius = np.linalg.norm(np.array([1, 1, 1]) - np.array([2.8, 2.7, 2.6]))
    assert metric["tensor"][0, 0, 0, 0, 0, 0] == pytest.approx(-(1 - 0.1 / radius))


def test_shell_zero_mass_and_enclosed_mass():
    zero = build("warp_shell", "warp_shell_comoving", [0, 0.5, 1.5, 0, 1, 3, 0, False])
    np.testing.assert_allclose(
        zero["tensor"], metric_get_minkowski([1, 5, 5, 5])["tensor"], atol=1e-14
    )
    shell = build(
        "warp_shell", "warp_shell_comoving", [1e20, 0.5, 1.5, 0, 1, 3, 0, False]
    )
    assert shell["params"]["M"][-1] == pytest.approx(1e20, rel=1e-3)
    assert np.all(shell["params"]["P"] >= 0)


def test_custom_metric_coordinates_and_flat_limit():
    seen = []

    def lapse(t, x, y, z):
        seen.append((t, x, y, z))
        return 1

    metric = custom_metric(
        [1, 2, 2, 2],
        [0, 0, 0, 0],
        [0.1, 2, 3, 4],
        lapse,
        lambda *args: [0, 0, 0],
        lambda *args: np.eye(3),
    )
    assert seen[0] == (0.1, 2, 3, 4)
    np.testing.assert_array_equal(
        metric["tensor"], metric_get_minkowski([1, 2, 2, 2])["tensor"]
    )


def test_lentz_template_regions():
    from warplab.metrics.lentz.metric_get_lentz import (
        get_warp_factor_by_region as region,
    )

    assert region(1.5, 0, 1) == (-2, 0)
    assert region(1.5, 1, 1) == (-1, 1)
    assert region(1.5, -1, 1) == (-1, -1)


@pytest.mark.parametrize("comoving", [False, True])
def test_modified_metrics_reduce_to_alcubierre(comoving):
    suffix = "_comoving" if comoving else ""
    original = build("alcubierre", "alcubierre" + suffix, [0.2, 2, 1])
    modified = build("modified_time", "modified_time" + suffix, [0.2, 2, 1, 1])
    expanded = build("van_den_broeck", "van_den_broeck" + suffix, [0.2, 1, 1, 2, 1, 0])
    np.testing.assert_allclose(modified["tensor"], original["tensor"], atol=1e-14)
    np.testing.assert_allclose(expanded["tensor"], original["tensor"], atol=1e-14)


@pytest.mark.parametrize(
    "family,name,args", [case for case in CASES if "comoving" in case[1]]
)
def test_comoving_requires_one_time_slice(family, name, args):
    fn = getattr(
        importlib.import_module(f"warplab.metrics.{family}.metric_get_{name}"),
        f"metric_get_{name}",
    )
    with pytest.raises(ValueError, match="requires one time slice, got 2"):
        fn(np.array([2, 3, 3, 3]), np.array([1, 2, 2, 2]), *args, np.ones(4))


@pytest.mark.integration
@pytest.mark.parametrize("order", ["second", "fourth"])
@pytest.mark.parametrize("family,name,args", CASES, ids=[case[1] for case in CASES])
def test_each_generator_reaches_public_solver(family, name, args, order):
    from warplab.solver.get_energy_tensor import get_energy_tensor

    metric = build(family, name, args)
    before = np.asarray(metric["tensor"]).copy()
    result = get_energy_tensor(metric, order)
    values = np.asarray(result["tensor"])
    assert values.shape == before.shape
    assert np.isfinite(values).all()
    np.testing.assert_allclose(
        values, values.swapaxes(0, 1), rtol=1e-12, atol=1e-12 * np.max(abs(values))
    )
    np.testing.assert_array_equal(metric["tensor"], before)


def test_shell_exterior_matches_schwarzschild_profile():
    from warplab.units.universal_constants.c import c
    from warplab.units.universal_constants.G import G

    metric = build(
        "warp_shell", "warp_shell_comoving", [1e25, 0.5, 1.5, 0, 1, 1, 0, False]
    )
    params = metric["params"]
    exterior = params["rVec"] > 1.6
    expected = 1 - 2 * G() * params["M"][-1] / c() ** 2 / params["rVec"][exterior]
    np.testing.assert_allclose(params["A"][exterior], -expected, rtol=1e-9)
    np.testing.assert_allclose(params["B"][exterior], 1 / expected, rtol=1e-12)


@pytest.mark.integration
@pytest.mark.parametrize("family,name,args", CASES, ids=[case[1] for case in CASES])
def test_all_metrics_complete_analysis(family, name, args):
    from warplab.analyser.eval_metric import eval_metric

    metric = build(family, name, args)
    result = eval_metric(metric, num_angular_vec=6, num_time_vec=2, diff_order="second")
    for key in (
        "null",
        "weak",
        "strong",
        "dominant",
        "expansion",
        "shear",
        "vorticity",
    ):
        assert np.isfinite(result[key]).all()


def test_shell_is_finite_when_a_grid_point_is_the_centre():
    """Upstream returns NaN when a grid radius lands exactly on a radial sample.

    Its Legendre interpolation divides by zero there, and r == 0 is sample 1.
    """
    from warplab.solver import get_energy_tensor

    metric = build_at_origin_shell()
    assert np.isfinite(metric["tensor"]).all()
    assert np.isfinite(get_energy_tensor(metric)["tensor"]).all()


def build_at_origin_shell():
    from warplab.metrics import metric_get_warp_shell_comoving

    # World centre (3, 3, 3) with unit spacing puts grid point 2 at r == 0.
    return metric_get_warp_shell_comoving(
        np.array([1, 5, 5, 5]),
        np.array([1, 3, 3, 3]),
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


def test_schwarzschild_rejects_a_grid_on_the_origin_or_horizon():
    """Upstream silently adds 1e-10 to r instead; Python refuses the grid."""
    from warplab.metrics import metric_get_schwarzschild

    with pytest.raises(ValueError, match="origin or coordinate horizon"):
        metric_get_schwarzschild(
            np.array([1, 5, 5, 5]), np.array([1, 3, 3, 3]), 0.05, np.ones(4)
        )
